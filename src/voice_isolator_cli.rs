//! `oxinfer isolate IN [OUT]`: voice isolation with Inworld's Voice Isolator
//! ([`oxinfer::voice_isolator`]), as the reference's `voice-isolate` command: the voice to OUT
//! (default `IN_clean.wav`), and offline the removed background beside it
//! (`OUT_background.wav`), float32 WAVs at 48 kHz.

use anyhow::{Context, Result, bail};
use oxinfer::voice_isolator::{self as vi, IsolateOptions, IsolatorOptions, Mode, VoiceIsolator};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const USAGE: &str = "usage: oxinfer isolate IN [OUT] [options]\n\
\n\
Isolates the voice of IN (WAV, or any audio file oxideav decodes) with Inworld's Voice Isolator:\n\
the voice goes to OUT (default IN_clean.wav), offline the removed background to OUT_background.wav.\n\
\n\
  --model SPEC          the repo id (default inworld/voice-isolator), a folder or a .safetensors file\n\
  --mode M              offline (default: voice-isolator-v1, 8 s windows) or streaming\n\
                        (voice-isolator-v1-realtime, causal, 20 ms pieces as a live source)\n\
  --lookahead N         streaming look-ahead in 10 ms frames: 0, 2 (default) or 4\n\
  --block-frames N      streaming: frames computed together (default 4)\n\
  --cache-frames N      streaming: time attention past (default the checkpoint's window, 1200)\n\
  --window-frames N     streaming: attention windows (default 600; 0 = off, the plain stream)\n\
  --crossfade-frames N  streaming: crossfade before each window boundary (default 200)\n\
  --piece-ms MS         streaming: the pieces pushed (default 20)\n\
  --chunk-seconds S     offline window (default 8)\n\
  --overlap-seconds S   offline crossfade (default 2)\n\
  --passes N            offline passes over the output (default 1)\n\
  --output-sr HZ        output rate (default 48000; 0 keeps the input's)\n\
  --background PATH     where the background goes (also in streaming mode)\n\
  --no-background       write only the voice\n\
  --profile             time per phase (per 10 ms frame)\n\
  --pcm16               16-bit PCM output (peak-normalized above full scale) instead of float32\n\
  --bf16                bf16 matrices (default f32)\n\
  --device D            cpu, gpu, metal, ... (default auto)\n\
  --threads N           CPU threads\n\
  --cache-dir DIR       download cache";

fn stem_with(p: &Path, suffix: &str) -> PathBuf {
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "out".into());
    p.with_file_name(format!("{stem}{suffix}.wav"))
}

fn write(path: &Path, samples: &[f32], rate: u32, pcm16: bool) -> Result<()> {
    let bytes = if pcm16 {
        let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s: Vec<f32> = if peak > 1.0 {
            samples.iter().map(|v| v / peak).collect()
        } else {
            samples.to_vec()
        };
        oxinfer::tts::wav_bytes(&s, rate)
    } else {
        oxinfer::tts::wav_bytes_f32(samples, rate)
    };
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

/// Runs the arguments after `isolate`.
pub fn run(args: &[String]) -> Result<()> {
    let mut lo = IsolatorOptions::default();
    let mut o = IsolateOptions::default();
    let mut model = vi::REPO.to_string();
    let (mut output_sr, mut piece_ms) = (Some(48_000u32), 20usize);
    let (mut background, mut no_background, mut pcm16, mut profile) = (None::<PathBuf>, false, false, false);
    let mut pos = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (a.as_str(), None),
        };
        let mut val = || -> Result<String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it.next().cloned().with_context(|| format!("{flag} needs a value")),
            }
        };
        match flag {
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            "--model" | "-m" => model = val()?,
            "--mode" => lo.mode = Mode::parse(&val()?)?,
            "--lookahead" => o.set("lookahead_frames", &val()?)?,
            "--block-frames" => o.set("block_frames", &val()?)?,
            "--cache-frames" => o.set("cache_frames", &val()?)?,
            "--window-frames" => o.set("window_frames", &val()?)?,
            "--crossfade-frames" => o.set("crossfade_frames", &val()?)?,
            "--chunk-seconds" => o.set("chunk_seconds", &val()?)?,
            "--overlap-seconds" => o.set("overlap_seconds", &val()?)?,
            "--passes" => o.set("offline_passes", &val()?)?,
            "--piece-ms" => piece_ms = val()?.parse().context("--piece-ms")?,
            "--output-sr" => {
                let v: u32 = val()?.parse().context("--output-sr")?;
                output_sr = (v != 0).then_some(v);
            }
            "--background" => background = Some(val()?.into()),
            "--no-background" => no_background = true,
            "--pcm16" => pcm16 = true,
            "--profile" => profile = true,
            "--bf16" => lo.f32_weights = false,
            "--device" => lo.set("device", &val()?)?,
            "--threads" => lo.set("threads", &val()?)?,
            "--cache-dir" => lo.set("cache_dir", &val()?)?,
            s if s.starts_with('-') => bail!("unknown option {s}\n{USAGE}"),
            s => pos.push(PathBuf::from(s)),
        }
    }
    let (input, output) = match pos.as_slice() {
        [i] => (i.clone(), stem_with(i, "_clean")),
        [i, o] => (i.clone(), o.clone()),
        _ => bail!("{USAGE}"),
    };
    let t0 = Instant::now();
    let iso = VoiceIsolator::load(&model, &lo)?;
    let info = iso.info();
    eprintln!(
        "[loaded {} ({:.1}M parameters, {} weights) on {} in {:.2}s]",
        info.name,
        info.params as f64 / 1e6,
        if info.f32_weights { "f32" } else { "bf16" },
        info.device,
        t0.elapsed().as_secs_f64()
    );
    let (pcm, rate) = vi::load_audio(&input)?;
    let sr = iso.sample_rate();
    let x = if rate == sr {
        pcm.clone()
    } else {
        oxinfer::audio::resample_kaiser_best(&pcm, rate as usize, sr as usize)
    };
    let secs = x.len() as f64 / sr as f64;
    let t1 = Instant::now();
    let voice = if info.mode == Mode::Streaming {
        // As a live source: pieces of `piece_ms`.
        let mut s = iso.stream(&o)?;
        let piece = (sr as usize * piece_ms / 1000).max(1);
        let mut out = Vec::with_capacity(x.len());
        let mut times = Vec::with_capacity(x.len() / piece + 1);
        for c in x.chunks(piece) {
            let tp = Instant::now();
            out.extend(s.push(c));
            times.push(tp.elapsed().as_secs_f64());
        }
        out.extend(s.flush());
        out.truncate(x.len());
        // the pushes that took longer than the audio they carried
        let late = times.iter().filter(|&&t| t * 1000.0 > piece_ms as f64).count();
        let (at, worst) = times
            .iter()
            .copied()
            .enumerate()
            .fold((0, 0f64), |a, (i, t)| if t > a.1 { (i, t) } else { a });
        times.sort_by(f64::total_cmp);
        let p99 = times.get((times.len().saturating_sub(1) * 99) / 100).copied().unwrap_or(0.0);
        eprintln!(
            "[streamed {secs:.2}s in {piece_ms} ms pieces: latency {:.0} ms, slowest push {:.1} ms (at {:.2}s), p99 {:.1} ms, {late} of {} over {piece_ms} ms]",
            s.latency_samples() as f64 * 1000.0 / sr as f64,
            worst * 1000.0,
            (at * piece) as f64 / sr as f64,
            p99 * 1000.0,
            times.len()
        );
        out
    } else {
        iso.isolate(&x, sr, &o)?
    };
    let el = t1.elapsed().as_secs_f64();
    if profile {
        oxinfer::prof::report(el, (secs * 100.0) as usize);
    }
    eprintln!("[isolated {secs:.2}s of audio in {el:.2}s: {:.1}x real time]", secs / el.max(1e-9));
    let (out_rate, voice, input_at) = match output_sr {
        Some(r) if r == sr => (r, voice, x),
        Some(r) => (
            r,
            oxinfer::audio::resample_kaiser_best(&voice, sr as usize, r as usize),
            oxinfer::audio::resample_kaiser_best(&x, sr as usize, r as usize),
        ),
        None => {
            let mut v = oxinfer::audio::resample_kaiser_best(&voice, sr as usize, rate as usize);
            v.resize(pcm.len(), 0.0);
            (rate, v, pcm)
        }
    };
    write(&output, &voice, out_rate, pcm16)?;
    eprintln!("[voice: {}]", output.display());
    let bg = match (background, no_background, info.mode) {
        (_, true, _) => None,
        (Some(p), _, _) => Some(p),
        (None, _, Mode::Offline) => Some(stem_with(&output, "_background")),
        (None, _, Mode::Streaming) => None,
    };
    if let Some(bg) = bg {
        let residual: Vec<f32> = input_at.iter().zip(&voice).map(|(a, v)| a - v).collect();
        write(&bg, &residual, out_rate, pcm16)?;
        eprintln!("[background: {}]", bg.display());
    }
    Ok(())
}
