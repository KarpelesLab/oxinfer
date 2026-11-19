//! `oxinfer --model MiniMaxAI/MiniMax-H3 --prompt "..." --frames 124 --size 1344x768 -o out.avi`:
//! text to video, and first / last frame to video with `--init-image` / `--last-image`, through
//! `oxinfer::minimax_h3::video`.
//!
//! The output follows the extension of `-o`: `.avi` is Motion JPEG in an AVI container and
//! `.webp` an animated WebP (both through oxideav, the `media` feature: no pure-Rust H.264 / VP9 /
//! AV1 encoder is available); anything else is a PNG sequence (`NAME_0000.png`, ...). The stereo
//! soundtrack (32 kHz, decoded when the model folder has `audio_vae/` or `--audio-vae` names the
//! audio VAE: a folder, or ComfyUI's `vae/minimax_h3_audio_vae_fp32.safetensors`) is a 16-bit PCM stream in
//! the AVI, interleaved frame by frame; animated WebP has no audio and a PNG sequence is only
//! frames, so both get a sibling `NAME.wav`.
//!
//! `--model` is the release folder in diffusers' layout, or a single-file transformer in ComfyUI's
//! layout (`Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`) with
//! `--vae FILE`; `--transformer`, `--vae`, `--audio-vae` and `--text-encoder` replace a part of a
//! folder too.
//! `--lora FILE[:SCALE]` applies LoRA files; a turbo LoRA sets the step count and the shifts its
//! name stands for (`--steps`, `--shift`, `--audio-shift` override them).
//!
//! `OXINFER_H3_PROMPT_EMBEDS=FILE` keeps the text encoder's output of a text-only prompt in FILE
//! (written when missing, read when present), so the 63 GB text encoder and the 66 GB transformer
//! need not be on disk at the same time; when the transformer is missing the run stops after
//! writing it.
//!
//! `OXINFER_H3_LATENTS=FILE` keeps the denoised video and audio rows the same way: a second run
//! with the same size and frame count decodes them again without the transformer's forwards.
//!
//! `OXINFER_H3_DUMP=DIR` keeps every forward's rows and velocities in DIR (and, with
//! `OXINFER_H3_DUMP_BLOCKS=i`, forward `i`'s sequence after every block), for
//! `tools/minimax_h3_video_ref.py` to compare with torch on the same inputs: see
//! `oxinfer::minimax_h3::dit`.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::minimax_h3::video::{Conditioning, FPS, Sources, Video, VideoParams, VideoPipeline};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn load_image(path: &Path) -> Result<oxinfer::sd::Image> {
    #[cfg(feature = "media")]
    {
        let im = oxinfer::media::Media::new()
            .load_image(path)
            .with_context(|| format!("load {}", path.display()))?;
        Ok(oxinfer::sd::Image {
            width: im.width,
            height: im.height,
            rgb: im.data,
        })
    }
    #[cfg(not(feature = "media"))]
    bail!("built without the `media` feature: {} cannot be decoded", path.display())
}

/// A part given as a local path, `org/name/path/in/repo.safetensors` (or `.gguf`), or a
/// repository id.
fn resolve_part(spec: &str, args: &Args) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    if spec.ends_with(".safetensors") || spec.ends_with(".gguf") {
        return oxinfer::engine::resolve_file(spec, args.cache_dir.as_deref());
    }
    oxinfer::engine::resolve_model(spec, args.cache_dir.as_deref())
}

/// Whether the transformer's weights are on disk (not only its configuration).
fn has_dit(dir: &Path) -> bool {
    if dir.is_file() {
        return true;
    }
    // diffusers' names, or the release's own (`FL2VA/transformer/model-*.safetensors`).
    ["diffusion_pytorch_model", "model"].iter().any(|stem| {
        dir.join(format!("{stem}.safetensors")).exists()
            || std::fs::read(dir.join(format!("{stem}.safetensors.index.json")))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .is_some_and(|v| {
                    v["weight_map"]
                        .as_object()
                        .is_some_and(|m| m.values().filter_map(|f| f.as_str()).all(|f| dir.join(f).exists()))
                })
    })
}

fn read_embeds(path: &Path) -> Result<Conditioning> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    let n = u32::from_le_bytes(b.get(..4).context("prompt embeddings: truncated")?.try_into()?) as usize;
    let rest = &b[4..];
    if rest.len() < n || !(rest.len() - n).is_multiple_of(4) {
        bail!("{}: not a prompt embedding file", path.display());
    }
    Ok(Conditioning {
        tags: rest[..n].to_vec(),
        text: rest[n..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
    })
}

fn write_embeds(path: &Path, c: &Conditioning) -> Result<()> {
    let mut b = (c.tags.len() as u32).to_le_bytes().to_vec();
    b.extend_from_slice(&c.tags);
    c.text.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
    std::fs::write(path, b).with_context(|| format!("write {}", path.display()))
}

/// Motion JPEG frames in an AVI (RIFF) container with an index; the soundtrack, when there is
/// one, as a second stream of 16-bit PCM, one chunk after each frame.
#[cfg(feature = "media")]
fn avi_mjpeg(v: &Video) -> Result<Vec<u8>> {
    let opts = oxideav_mjpeg::EncodeOptions::default();
    let jpegs: Vec<Vec<u8>> = v
        .frames
        .iter()
        .map(|f| oxideav_mjpeg::encode_rgb8(v.width as u32, v.height as u32, f, &opts).map_err(|e| anyhow::anyhow!("JPEG encode: {e}")))
        .collect::<Result<_>>()?;
    let u = |x: usize| (x as u32).to_le_bytes();
    let chunk = |id: &[u8; 4], data: &[u8]| {
        let mut c = id.to_vec();
        c.extend_from_slice(&u(data.len()));
        c.extend_from_slice(data);
        if data.len() % 2 == 1 {
            c.push(0);
        }
        c
    };
    let list = |kind: &[u8; 4], body: &[u8]| {
        let mut c = b"LIST".to_vec();
        c.extend_from_slice(&u(body.len() + 4));
        c.extend_from_slice(kind);
        c.extend_from_slice(body);
        c
    };
    let (n, max) = (jpegs.len(), jpegs.iter().map(Vec::len).max().unwrap_or(0));
    // The soundtrack: its rate, bytes per sample of every channel, and the samples of each frame
    // (the last frame's chunk takes what is left).
    let audio = v.audio.as_ref().filter(|a| !a.is_empty() && n > 0);
    let (rate, block) = audio.map_or((0, 0), |a| (a.sample_rate as usize, 2 * a.channels.len()));
    let span = |a: &oxinfer::minimax_h3::video::Soundtrack, i: usize| {
        let end = if i + 1 == n { a.len() } else { ((i + 1) * rate / FPS).min(a.len()) };
        (i * rate / FPS).min(end)..end
    };
    let max_audio = audio.map_or(0, |a| (0..n).map(|i| span(a, i).len() * block).max().unwrap_or(0));
    // avih: microseconds per frame, max bytes / s, padding, flags (has index), frames, initial
    // frames, streams, buffer size, width, height, reserved x4.
    let mut avih = Vec::new();
    let streams = 1 + audio.is_some() as usize;
    let rest = [0, 0x10, n, 0, streams, max.max(max_audio), v.width, v.height, 0, 0, 0, 0];
    for x in [1_000_000 / FPS, max * FPS + rate * block].into_iter().chain(rest) {
        avih.extend_from_slice(&u(x));
    }
    // strh: type, handler, flags, priority + language, initial frames, scale, rate, start, length,
    // buffer size, quality, sample size, frame rectangle.
    let mut strh = b"vidsMJPG".to_vec();
    for x in [0, 0, 0, 1, FPS, 0, n, max, 0xFFFF_FFFF, 0] {
        strh.extend_from_slice(&u(x));
    }
    for x in [0, 0, v.width, v.height] {
        strh.extend_from_slice(&(x as u16).to_le_bytes());
    }
    // strf: BITMAPINFOHEADER.
    let mut strf = Vec::new();
    strf.extend_from_slice(&u(40));
    strf.extend_from_slice(&u(v.width));
    strf.extend_from_slice(&u(v.height));
    strf.extend_from_slice(&1u16.to_le_bytes());
    strf.extend_from_slice(&24u16.to_le_bytes());
    strf.extend_from_slice(b"MJPG");
    for x in [v.width * v.height * 3, 0, 0, 0, 0] {
        strf.extend_from_slice(&u(x));
    }
    let mut strls = list(b"strl", &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat());
    if let Some(a) = audio {
        // strh as above: one unit of `scale / rate` seconds is a sample of every channel.
        let mut strh = b"auds\0\0\0\0".to_vec();
        for x in [0, 0, 0, block, rate * block, 0, a.len(), max_audio, 0xFFFF_FFFF, block] {
            strh.extend_from_slice(&u(x));
        }
        strh.extend_from_slice(&[0; 8]);
        // strf: WAVEFORMATEX, integer PCM.
        let mut strf = Vec::new();
        strf.extend_from_slice(&1u16.to_le_bytes());
        strf.extend_from_slice(&(a.channels.len() as u16).to_le_bytes());
        strf.extend_from_slice(&u(rate));
        strf.extend_from_slice(&u(rate * block));
        strf.extend_from_slice(&(block as u16).to_le_bytes());
        strf.extend_from_slice(&16u16.to_le_bytes());
        strf.extend_from_slice(&0u16.to_le_bytes());
        strls.extend(list(b"strl", &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat()));
    }
    let hdrl = list(b"hdrl", &[chunk(b"avih", &avih), strls].concat());
    let (mut movi, mut idx) = (Vec::new(), Vec::new());
    let mut add = |id: &[u8; 4], data: &[u8]| {
        idx.extend_from_slice(id);
        idx.extend_from_slice(&u(0x10));
        idx.extend_from_slice(&u(movi.len() + 4));
        idx.extend_from_slice(&u(data.len()));
        movi.extend(chunk(id, data));
    };
    for (i, j) in jpegs.iter().enumerate() {
        add(b"00dc", j);
        if let Some(a) = audio {
            add(b"01wb", &oxinfer::tts::pcm16_bytes(&a.interleaved(span(a, i))));
        }
    }
    let body = [hdrl, list(b"movi", &movi), chunk(b"idx1", &idx)].concat();
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&u(body.len() + 4));
    out.extend_from_slice(b"AVI ");
    out.extend_from_slice(&body);
    Ok(out)
}

#[cfg(feature = "media")]
fn animated_webp(v: &Video) -> Result<Vec<u8>> {
    use oxideav_webp::anim_encode::AnimFrame;
    let frames: Vec<AnimFrame> = v
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let rgba: Vec<u8> = f.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
            // 24 fps in whole milliseconds: 41.67 ms on average.
            let d = ((i + 1) * 1000 / FPS - i * 1000 / FPS) as u32;
            AnimFrame::new(v.width as u32, v.height as u32, rgba, d)
        })
        .collect();
    oxideav_webp::encode_animation_frames(&frames, &oxideav_webp::Metadata::default(), &oxideav_webp::EncodeOptions::new())
        .map_err(|e| anyhow::anyhow!("WebP encode: {e}"))
}

/// Writes the soundtrack of `v`, when it has one, as `stem.wav`; returns what to report.
fn write_wav(v: &Video, stem: &Path) -> Result<String> {
    let Some(a) = v.audio.as_ref().filter(|a| !a.is_empty()) else {
        return Ok(String::new());
    };
    let p = PathBuf::from(format!("{}.wav", stem.display()));
    std::fs::write(&p, a.to_wav()).with_context(|| format!("write {}", p.display()))?;
    Ok(format!(
        " and {} ({} Hz, {} channels)",
        p.display(),
        a.sample_rate,
        a.channels.len()
    ))
}

/// Writes `v` as `out` says (see the module docs); returns what was written.
fn write_video(v: &Video, out: &Path) -> Result<String> {
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        #[cfg(feature = "media")]
        "avi" => {
            std::fs::write(out, avi_mjpeg(v)?).with_context(|| format!("write {}", out.display()))?;
            let audio = match v.audio.as_ref().filter(|a| !a.is_empty()) {
                Some(a) => format!(", {} Hz 16-bit PCM, {} channels", a.sample_rate, a.channels.len()),
                None => String::new(),
            };
            Ok(format!("{} (Motion JPEG, {FPS} fps{audio})", out.display()))
        }
        #[cfg(feature = "media")]
        "webp" => {
            std::fs::write(out, animated_webp(v)?).with_context(|| format!("write {}", out.display()))?;
            Ok(format!(
                "{} (animated WebP){}",
                out.display(),
                write_wav(v, &out.with_extension(""))?
            ))
        }
        #[cfg(not(feature = "media"))]
        "avi" | "webp" => bail!("built without the `media` feature: .{ext} cannot be written (use a .png name for a PNG sequence)"),
        "mp4" | "mkv" | "webm" | "mov" | "gif" => {
            bail!(".{ext} output is not supported: use .avi (Motion JPEG), .webp (animated) or .png (a PNG sequence)")
        }
        _ => {
            let stem = out.with_extension("");
            for (i, f) in v.frames.iter().enumerate() {
                let p = PathBuf::from(format!("{}_{i:04}.png", stem.display()));
                std::fs::write(&p, oxinfer::png::encode_rgb(v.width, v.height, f)).with_context(|| format!("write {}", p.display()))?;
            }
            Ok(format!(
                "{}_0000.png ... _{:04}.png{}",
                stem.display(),
                v.frames.len().saturating_sub(1),
                write_wav(v, &stem)?
            ))
        }
    }
}

pub fn run(args: &Args) -> Result<()> {
    let t0 = Instant::now();
    let prompt = args.prompt.clone().context("--prompt is required")?;
    let mut p = VideoParams::new(&prompt);
    p.size = args.size;
    p.seed = args.seed;
    if let Some(f) = args.frames {
        p.frames = f;
    }
    if let Some(s) = args.steps {
        p.steps = s;
    }
    if let Some(i) = args.init_images.first() {
        p.first = Some(load_image(i)?);
    }
    if let Some(i) = &args.last_image {
        p.last = Some(load_image(i)?);
    }
    p.latents = std::env::var_os("OXINFER_H3_LATENTS").map(PathBuf::from);
    let cache = std::env::var_os("OXINFER_H3_PROMPT_EMBEDS").map(PathBuf::from);
    if cache.is_some() && (p.first.is_some() || p.last.is_some()) {
        bail!("OXINFER_H3_PROMPT_EMBEDS holds a text-only prompt: unset it with keyframes");
    }
    // The parts: a release folder, or a single-file transformer with the VAE named beside it.
    let single = args.model.is_file();
    let mut src = Sources {
        dir: (!single).then(|| args.model.clone()),
        transformer: match (&args.transformer, single) {
            (Some(t), _) => Some(resolve_part(&t.to_string_lossy(), args)?),
            (None, true) => Some(args.model.clone()),
            (None, false) => None,
        },
        vae: args.flux.vae.as_deref().map(|v| resolve_part(v, args)).transpose()?,
        audio_vae: args.audio_vae.as_deref().map(|v| resolve_part(v, args)).transpose()?,
        // The text encoder at Q4 is not offered (Qwen-Image's moves its hidden states by ~25%).
        text_quant: Some(match args.quant {
            oxinfer::model::Quant::Q4 => oxinfer::model::Quant::Q8,
            q => q,
        }),
        ..Default::default()
    };
    if single && src.vae.is_none() {
        bail!(
            "{} is a single-file transformer: name the video VAE with --vae (ComfyUI's minimax_h3_video_vae_fp16.safetensors or a diffusers vae folder)",
            args.model.display()
        );
    }
    let text_encoder = || -> Result<PathBuf> {
        match &args.flux.text_encoder {
            Some(t) => resolve_part(t, args),
            None => crate::minimax_h3_cli::text_encoder(&args.model),
        }
    };
    let tpath = src.transformer.clone().unwrap_or_else(|| args.model.join("transformer"));
    let have_dit = has_dit(&tpath);
    let cond = match &cache {
        Some(f) if f.exists() => Some(read_embeds(f)?),
        Some(f) => {
            let te = text_encoder()?;
            let c = oxinfer::minimax_h3::video::encode_prompt_quant(&args.model, &te, src.text_quant.unwrap(), &prompt, &[])?;
            write_embeds(f, &c)?;
            eprintln!(
                "[text encoder: {} rows in {:.1}s, wrote {}]",
                c.tags.len(),
                t0.elapsed().as_secs_f64(),
                f.display()
            );
            if !have_dit {
                eprintln!(
                    "[no transformer weights in {}: stopping after the text encoder]",
                    args.model.display()
                );
                return Ok(());
            }
            Some(c)
        }
        None => None,
    };
    src.text_encoder = if cond.is_some() { PathBuf::new() } else { text_encoder()? };
    let t1 = Instant::now();
    let mut pipe = VideoPipeline::load_from(&src)?;
    if pipe.audio_vae.is_none() {
        eprintln!("[no audio VAE (audio_vae/ in the model folder, or --audio-vae): the video will have no soundtrack]");
    }
    for line in pipe.apply_loras(&args.loras)? {
        eprintln!("{line}");
    }
    // A turbo LoRA is distilled for a step count and a pair of shifts (its name says which).
    if let Some(t) = args.loras.iter().find_map(|l| oxinfer::minimax_h3::lora::turbo_settings(&l.path)) {
        if args.steps.is_none() {
            p.steps = t.forwards + 1;
        }
        pipe.shifts = (t.video_shift, t.audio_shift);
    }
    if let Some(s) = args.shift {
        pipe.shifts.0 = s;
    }
    if let Some(s) = args.audio_shift {
        pipe.shifts.1 = s;
    }
    let (h, w, frames, g) = pipe.geometry(&p)?;
    let secs = frames as f64 / FPS as f64;
    if !(5.0..=15.0).contains(&secs) {
        eprintln!("[warning: {frames} frames is {secs:.1}s; MiniMax-H3 is released for 5 to 15 seconds (124 to 362 frames)]");
    }
    eprintln!(
        "[MiniMax-H3: {} blocks, {w}x{h}, {frames} frames at {FPS} fps ({} latent frames, {} video + {} audio rows), {} steps, shift {} (audio {}); loaded in {:.1}s]",
        pipe.dit.cfg.layers,
        g.lt,
        g.lt * (g.lh / 2) * (g.lw / 2),
        2 * g.n_audio,
        p.steps.saturating_sub(1),
        pipe.shifts.0,
        pipe.shifts.1,
        t1.elapsed().as_secs_f64()
    );
    let tp = Instant::now();
    let mut progress = |i: usize, n: usize| eprint!("\r[step {i}/{n}, {:.1}s]", tp.elapsed().as_secs_f64());
    let video = pipe.generate(&p, cond, &mut progress)?;
    let default = if cfg!(feature = "media") { "video.avi" } else { "video.png" };
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from(default));
    let what = write_video(&video, &out)?;
    eprintln!(
        "[{}x{}, {} frames | {:.1}s | wrote {what}{}]",
        video.width,
        video.height,
        video.frames.len(),
        t0.elapsed().as_secs_f64(),
        if video.audio.is_some() { "" } else { "; no audio track" }
    );
    if args.profile {
        eprint!("[time by operation]\n{}", oxinfer::sd::ops::timing::report());
    }
    Ok(())
}

#[cfg(all(test, feature = "media"))]
mod tests {
    use super::*;
    use oxideav_core::MediaType;
    use oxinfer::minimax_h3::video::Soundtrack;

    /// The AVI demuxed by oxideav: every frame and every audio sample comes back.
    #[test]
    fn avi_carries_the_soundtrack() {
        let (w, h, n) = (32usize, 16usize, 5usize);
        let frames: Vec<Vec<u8>> = (0..n).map(|f| (0..w * h * 3).map(|i| (i * 7 + f * 40) as u8).collect()).collect();
        // A little longer than the frames, as the model's soundtracks are.
        let len = n * 32000 / FPS + 123;
        let ramp = |k: usize| -> Vec<f32> { (0..len).map(|i| ((i * k) % 2000) as f32 / 1000.0 - 1.0).collect() };
        let audio = Soundtrack {
            sample_rate: 32000,
            channels: vec![ramp(3), ramp(7)],
        };
        let want = oxinfer::tts::pcm16_bytes(&audio.interleaved(0..len));
        let v = Video {
            width: w,
            height: h,
            frames,
            audio: Some(audio),
        };
        let avi = avi_mjpeg(&v).unwrap();
        let media = oxinfer::media::Media::new();
        let ctx = media.context();
        let mut input: Box<dyn oxideav_core::ReadSeek> = Box::new(std::io::Cursor::new(avi));
        let format = ctx.containers.probe_input(&mut *input, Some("avi")).unwrap();
        let mut demux = ctx.containers.open_demuxer(&format, input, &ctx.codecs).unwrap();
        let streams = demux.streams().to_vec();
        assert_eq!(streams.len(), 2);
        let a = streams.iter().find(|s| s.params.media_type == MediaType::Audio).unwrap();
        assert_eq!(a.params.sample_rate, Some(32000));
        assert_eq!(a.params.channels, Some(2));
        let (mut video_packets, mut pcm) = (0, Vec::new());
        while let Ok(p) = demux.next_packet() {
            if p.stream_index == a.index {
                pcm.extend_from_slice(&p.data);
            } else {
                video_packets += 1;
            }
        }
        assert_eq!(video_packets, n);
        assert_eq!(pcm, want);
    }
}
