//! `oxinfer --model Edge0/Audio8-TTS-Realistic-Air --speak TEXT -o out.wav`: text to speech
//! through `oxinfer::arktts`, with the options of the other speech command lines where they apply
//! (`--voice`, `--ref-audio` with `--ref-text`, `--temp`, `--top-p`, `--max-tokens`, `--seed`,
//! `--speak-json` for the rest such as `top_k`, `--speak-file`, `--speak-split`,
//! `--speak-stream`, `--dump-pcm`, `--show-tokens`).

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::arktts::{Arktts, ArkttsOptions, ArkttsSpeak};
use oxinfer::engine::Flow;
use oxinfer::tts::{Audio, split_sentences};
use std::io::Read;
use std::time::Instant;

/// `path` with `-N` before its extension (`out.wav` -> `out-3.wav`).
fn numbered(path: &std::path::Path, n: usize) -> std::path::PathBuf {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let name = match path.extension() {
        Some(e) => format!("{stem}-{n}.{}", e.to_string_lossy()),
        None => format!("{stem}-{n}"),
    };
    path.with_file_name(name)
}

fn write_audio(out: &std::path::Path, dump_pcm: Option<&std::path::Path>, audio: &Audio) -> Result<()> {
    audio.write_wav(out)?;
    eprintln!(
        "[wrote {} ({:.2} s of audio at {} Hz)]",
        out.display(),
        audio.duration_secs(),
        audio.sample_rate
    );
    if let Some(p) = dump_pcm {
        let bytes: Vec<u8> = audio.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes)?;
        eprintln!("[wrote {} f32 samples to {}]", audio.samples.len(), p.display());
    }
    Ok(())
}

pub fn run(args: &Args) -> Result<()> {
    if args.decode_tokens.is_some() || args.snac_encode.is_some() || args.decode_codes.is_some() {
        bail!("--decode-tokens, --snac-encode and --decode-codes are for Orpheus and SNAC models");
    }
    let stdin = || -> Result<String> {
        let mut t = String::new();
        std::io::stdin().read_to_string(&mut t)?;
        Ok(t)
    };
    let mut texts: Vec<String> = Vec::new();
    match args.speak.as_deref() {
        Some("-") => texts.push(stdin()?),
        Some(t) => texts.push(t.to_string()),
        None => {}
    }
    texts.extend(args.speak_more.iter().cloned());
    if let Some(f) = &args.speak_file {
        let all = if f.as_os_str() == "-" {
            stdin()?
        } else {
            std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?
        };
        texts.extend(all.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string));
    }
    if texts.is_empty() {
        bail!("--speak TEXT or --speak-file FILE: nothing to say");
    }
    let batch = texts.len() > 1 || args.speak_file.is_some() || args.speak_split;
    let texts: Vec<String> = if args.speak_split {
        texts.iter().flat_map(|t| split_sentences(t, 240)).collect()
    } else {
        texts.iter().map(|t| t.trim().to_string()).collect()
    };
    let mut so = match &args.speak_json {
        Some(j) => ArkttsSpeak::from_json(j)?,
        None => ArkttsSpeak::default(),
    };
    if let Some(v) = &args.voice {
        so.voice = Some(v.clone());
    }
    match (args.ref_audio.as_slice(), args.ref_text.as_slice()) {
        ([], []) => {}
        ([], [t]) if so.voice.is_some() => so.voice_text = Some(t.clone()),
        ([clip], []) => so.voice = Some(clip.to_string_lossy().into_owned()),
        ([clip], [t]) => {
            so.voice = Some(clip.to_string_lossy().into_owned());
            so.voice_text = Some(t.clone());
        }
        _ => bail!("--ref-audio / --ref-text: Audio8 TTS takes one reference clip and its transcript"),
    }
    if let Some(t) = args.temperature_set {
        so.temperature = Some(t);
    }
    if let Some(p) = args.top_p_set {
        so.top_p = Some(p);
    }
    if let Some(n) = args.max_tokens_set {
        so.max_tokens = Some(n);
    }
    so.seed = args.seed;

    let mut lo = ArkttsOptions::default();
    lo.cache_dir = args.cache_dir.clone();
    let m = Arktts::load(&args.model.to_string_lossy(), &lo)?;
    let info = m.info();
    eprintln!(
        "[audio8 tts: {} + {} layers, hidden {}, {:.2} GB of weights from {}; codec at {} Hz, {:.1} frames/s; {} voices; cpu, {} threads; loaded in {:.2}s]",
        info.layers,
        info.fast_layers,
        info.hidden_size,
        info.weight_bytes as f64 / 1e9,
        info.dir.display(),
        info.sample_rate,
        info.frame_rate,
        info.voices.len(),
        info.threads,
        info.load_secs
    );
    let out = args.output.clone().unwrap_or_else(|| "speech.wav".into());
    let mut joined: Vec<f32> = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let t0 = Instant::now();
        let mut chunks = 0;
        let speech = if args.speak_stream {
            m.speak_stream(text, &so, &mut |c| {
                chunks += 1;
                if args.show_tokens {
                    eprintln!(
                        "[chunk {chunks}: {:.3}-{:.3} s at {:.3} s]",
                        c.start as f64 / c.sample_rate as f64,
                        (c.start + c.samples.len()) as f64 / c.sample_rate as f64,
                        t0.elapsed().as_secs_f64()
                    );
                }
                Flow::Continue
            })?
        } else {
            m.speak(text, &so)?
        };
        if args.show_tokens {
            eprintln!("[frames ({} x 10 codes): {:?}]", speech.stats.frames, speech.tokens);
        }
        let st = &speech.stats;
        eprintln!(
            "[{}{} prompt rows, prefill {:.2}s | {} frames in {:.2}s = {:.1} frames/s (real time needs {:.1}) | codec {:.2}s | {:.2}s audio in {:.2}s: RTF {:.2}{} | {:?}]",
            if batch { format!("{}: ", i + 1) } else { String::new() },
            st.prompt_tokens,
            st.prefill_secs,
            st.frames,
            st.generate_secs,
            st.tok_s(),
            info.frame_rate,
            st.snac_secs,
            st.audio_secs,
            st.total_secs,
            st.rtf(),
            st.first_chunk_secs
                .filter(|_| args.speak_stream)
                .map(|s| format!(", first audio after {s:.2}s"))
                .unwrap_or_default(),
            speech.finish
        );
        if args.speak_split {
            joined.extend_from_slice(&speech.audio.samples);
        } else if batch {
            write_audio(&numbered(&out, i + 1), None, &speech.audio)?;
        } else {
            write_audio(&out, args.dump_pcm.as_deref(), &speech.audio)?;
        }
    }
    if args.speak_split {
        let audio = Audio {
            samples: joined,
            sample_rate: info.sample_rate,
        };
        write_audio(&out, args.dump_pcm.as_deref(), &audio)?;
    }
    Ok(())
}
