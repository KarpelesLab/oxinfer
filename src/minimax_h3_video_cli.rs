//! `oxinfer --model MiniMaxAI/MiniMax-H3 --prompt "..." --frames 124 --size 1344x768 -o out.mp4`:
//! text to video, and first / last frame to video with `--init-image` / `--last-image`, through
//! `oxinfer::minimax_h3::video`; with the release's `Ref2VA/`, video from references
//! (`--ref FILE`, repeated in the order the prompt names them: images, audio clips by extension,
//! anything else a video with its soundtrack) through `oxinfer::minimax_h3::ref2va`.
//!
//! The output follows the extension of `-o` (all through oxideav, the `media` feature): `.mp4` is
//! H.264 and AAC in MP4 (the default), `.webm` VP9 and Opus in WebM, `.avi` Motion JPEG and 16-bit
//! PCM in AVI, `.webp` an animated WebP; anything else is a PNG sequence (`NAME_0000.png`, ...).
//! The stereo soundtrack (32 kHz, decoded when the model folder has `audio_vae/` or `--audio-vae`
//! names the audio VAE: a folder, or ComfyUI's `vae/minimax_h3_audio_vae_fp32.safetensors`) is the
//! file's audio stream; animated WebP has no audio and a PNG sequence is only frames, so both get
//! a sibling `NAME.wav`.
//!
//! `--model` is the release folder in diffusers' layout, or a single-file transformer in ComfyUI's
//! layout (`Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`) with
//! `--vae FILE`; `--transformer`, `--vae`, `--audio-vae` and `--text-encoder` replace a part of a
//! folder too.
//! `--lora FILE[:SCALE]` applies LoRA files; a turbo LoRA sets the step count and the shifts its
//! name stands for (`--steps`, `--shift`, `--audio-shift` override them).
//! `--corrector FILE` applies an H3 corrector (SulphurAI/Sulphur-3's `s3_corrector.safetensors`:
//! `oxinfer::minimax_h3::corrector`) with `--corrector-strength`, `--corrector-budget`,
//! `--corrector-floor`, `--corrector-start` and `--corrector-end`.
//! `--attention split|half|f32` sets the operands of the Metal transformer's attention.
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
        oxinfer::minimax_h3::encode::load_image(path)
    }
    #[cfg(not(feature = "media"))]
    bail!("built without the `media` feature: {} cannot be decoded", path.display())
}

/// A `--ref` file ([`oxinfer::minimax_h3::encode::load_reference`]).
fn load_ref(path: &Path) -> Result<oxinfer::minimax_h3::ref2va::RefInput> {
    #[cfg(feature = "media")]
    {
        oxinfer::minimax_h3::encode::load_reference(path)
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

#[cfg(feature = "media")]
use oxinfer::minimax_h3::encode::{VideoFormat, animated_webp, encode_video};

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
pub(crate) fn write_video(v: &Video, out: &Path) -> Result<String> {
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        #[cfg(feature = "media")]
        "mp4" | "m4v" | "webm" | "avi" => {
            let f = VideoFormat::parse(&ext).expect("listed above");
            std::fs::write(out, encode_video(v, f)?).with_context(|| format!("write {}", out.display()))?;
            let (vc, ac) = match f {
                VideoFormat::Mp4 => ("H.264", "AAC"),
                VideoFormat::Webm => ("VP9", "Opus"),
                VideoFormat::Avi => ("Motion JPEG", "16-bit PCM"),
            };
            let audio = match v.audio.as_ref().filter(|a| !a.is_empty()) {
                Some(a) => format!(", {ac} {} channels", a.channels.len()),
                None => String::new(),
            };
            Ok(format!("{} ({vc}, {} fps{audio})", out.display(), v.fps))
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
        "mp4" | "m4v" | "webm" | "avi" | "webp" => {
            bail!("built without the `media` feature: .{ext} cannot be written (use a .png name for a PNG sequence)")
        }
        "mkv" | "mov" | "gif" => {
            bail!(
                ".{ext} output is not supported: use .mp4 (H.264 + AAC), .webm (VP9 + Opus), .avi (Motion JPEG), .webp (animated) or .png (a PNG sequence)"
            )
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
    (p.guidance, p.still_prompt) =
        oxinfer::minimax_h3::video::guidance_fields(&|k| args.h3_guidance.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v.clone()))?;
    p.latents = std::env::var_os("OXINFER_H3_LATENTS").map(PathBuf::from);
    let cache = std::env::var_os("OXINFER_H3_PROMPT_EMBEDS").map(PathBuf::from);
    if cache.is_some() && (p.first.is_some() || p.last.is_some()) {
        bail!("OXINFER_H3_PROMPT_EMBEDS holds a text-only prompt: unset it with keyframes");
    }
    if !args.refs.is_empty() && (p.first.is_some() || p.last.is_some()) {
        bail!("--ref (Ref2VA/) and --init-image / --last-image (FL2VA/) are different tasks");
    }
    let refs = args.refs.iter().map(|r| load_ref(r)).collect::<Result<Vec<_>>>()?;
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
            None => crate::minimax_h3_cli::text_encoder(&args.model, args.cache_dir.as_deref()),
        }
    };
    let tpath = src.transformer.clone().unwrap_or_else(|| args.model.join("transformer"));
    let have_dit = has_dit(&tpath);
    let cond = match &cache {
        Some(_) if !refs.is_empty() => bail!("OXINFER_H3_PROMPT_EMBEDS holds a text-only prompt: unset it with --ref"),
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
    // A kept prompt embedding needs no text encoder, unless a still caption is to be encoded.
    src.text_encoder = if cond.is_some() && p.still_prompt.is_none() {
        PathBuf::new()
    } else {
        text_encoder()?
    };
    let t1 = Instant::now();
    let mut pipe = VideoPipeline::load_from(&src)?;
    if pipe.audio_vae.is_none() {
        eprintln!("[no audio VAE (audio_vae/ in the model folder, or --audio-vae): the video will have no soundtrack]");
    }
    for line in pipe.apply_loras(&args.loras)? {
        eprintln!("{line}");
    }
    if let Some(c) = &args.h3_corrector {
        pipe.load_corrector(&resolve_part(c, args)?)?;
    }
    for (k, v) in &args.h3_corrector_opts {
        pipe.corrector_settings.set(k, v)?;
    }
    if let Some(a) = &args.h3_attention {
        pipe.dit.gpu_attention =
            oxinfer::gpu::dit::AttnPrecision::parse(a).with_context(|| format!("--attention {a}: half, split or f32"))?;
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
    let video = if refs.is_empty() {
        pipe.generate(&p, cond, &mut progress)?
    } else {
        use oxinfer::minimax_h3::ref2va::{RefOptions, normalize};
        let mut o = RefOptions {
            short_edge: p.short_edge,
            max_pixels: p.max_pixels,
            ..Default::default()
        };
        if let Some(e) = args.ref_image_edge {
            o.image_short_edge = e;
        }
        let rate = pipe.audio_vae.as_ref().map_or(32000, |a| a.cfg.sampling_rate);
        let refs = normalize(refs, frames, rate, &o)?;
        pipe.generate_ref(&p, &refs, cond, &mut progress)?
    };
    let default = if cfg!(feature = "media") { "video.mp4" } else { "video.png" };
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
        eprint!("[transformer time by stage]\n{}", oxinfer::minimax_h3::dit::stage::report());
        eprint!("[time by operation]\n{}", oxinfer::sd::ops::timing::report());
    }
    Ok(())
}
