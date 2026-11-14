//! `oxinfer --model MiniMaxAI/MiniMax-H3 --prompt "..." --frames 124 --size 1344x768 -o out.avi`:
//! text to video, and first / last frame to video with `--init-image` / `--last-image`, through
//! `oxinfer::minimax_h3::video`.
//!
//! The output follows the extension of `-o`: `.avi` is Motion JPEG in an AVI container and
//! `.webp` an animated WebP (both through oxideav, the `media` feature: no pure-Rust H.264 / VP9 /
//! AV1 encoder is available); anything else is a PNG sequence (`NAME_0000.png`, ...). No audio is
//! written (the audio VAE is not implemented).
//!
//! `OXINFER_H3_PROMPT_EMBEDS=FILE` keeps the text encoder's output of a text-only prompt in FILE
//! (written when missing, read when present), so the 63 GB text encoder and the 66 GB transformer
//! need not be on disk at the same time; when the transformer is missing the run stops after
//! writing it.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::minimax_h3::video::{Conditioning, FPS, Video, VideoParams, VideoPipeline};
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

/// Whether the transformer's weights are on disk (not only its configuration).
fn has_dit(dir: &Path) -> bool {
    if dir.join("diffusion_pytorch_model.safetensors").exists() {
        return true;
    }
    std::fs::read(dir.join("diffusion_pytorch_model.safetensors.index.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| {
            v["weight_map"]
                .as_object()
                .is_some_and(|m| m.values().filter_map(|f| f.as_str()).all(|f| dir.join(f).exists()))
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

/// Motion JPEG frames in an AVI (RIFF) container with an index.
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
    // avih: microseconds per frame, max bytes / s, padding, flags (has index), frames, initial
    // frames, streams, buffer size, width, height, reserved x4.
    let mut avih = Vec::new();
    for x in [1_000_000 / FPS, max * FPS, 0, 0x10, n, 0, 1, max, v.width, v.height, 0, 0, 0, 0] {
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
    let strl = list(b"strl", &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat());
    let hdrl = list(b"hdrl", &[chunk(b"avih", &avih), strl].concat());
    let (mut movi, mut idx) = (Vec::new(), Vec::new());
    for j in &jpegs {
        idx.extend_from_slice(b"00dc");
        idx.extend_from_slice(&u(0x10));
        idx.extend_from_slice(&u(movi.len() + 4));
        idx.extend_from_slice(&u(j.len()));
        movi.extend(chunk(b"00dc", j));
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

/// Writes `v` as `out` says (see the module docs); returns what was written.
fn write_video(v: &Video, out: &Path) -> Result<String> {
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        #[cfg(feature = "media")]
        "avi" => {
            std::fs::write(out, avi_mjpeg(v)?).with_context(|| format!("write {}", out.display()))?;
            Ok(format!("{} (Motion JPEG, {FPS} fps)", out.display()))
        }
        #[cfg(feature = "media")]
        "webp" => {
            std::fs::write(out, animated_webp(v)?).with_context(|| format!("write {}", out.display()))?;
            Ok(format!("{} (animated WebP)", out.display()))
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
                "{}_0000.png ... _{:04}.png",
                stem.display(),
                v.frames.len().saturating_sub(1)
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
    let cache = std::env::var_os("OXINFER_H3_PROMPT_EMBEDS").map(PathBuf::from);
    if cache.is_some() && (p.first.is_some() || p.last.is_some()) {
        bail!("OXINFER_H3_PROMPT_EMBEDS holds a text-only prompt: unset it with keyframes");
    }
    let have_dit = has_dit(&args.model.join("transformer"));
    let cond = match &cache {
        Some(f) if f.exists() => Some(read_embeds(f)?),
        Some(f) => {
            let te = crate::minimax_h3_cli::text_encoder(&args.model)?;
            let c = oxinfer::minimax_h3::video::encode_prompt(&args.model, &te, &prompt, &[])?;
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
    let te = if cond.is_some() {
        PathBuf::new()
    } else {
        crate::minimax_h3_cli::text_encoder(&args.model)?
    };
    let t1 = Instant::now();
    let pipe = VideoPipeline::load(&args.model, &te)?;
    let (h, w, frames, g) = pipe.geometry(&p)?;
    let secs = frames as f64 / FPS as f64;
    if !(5.0..=15.0).contains(&secs) {
        eprintln!("[warning: {frames} frames is {secs:.1}s; MiniMax-H3 is released for 5 to 15 seconds (124 to 362 frames)]");
    }
    eprintln!(
        "[MiniMax-H3: {} blocks, {w}x{h}, {frames} frames at {FPS} fps ({} latent frames, {} video + {} audio rows), {} steps; loaded in {:.1}s]",
        pipe.dit.cfg.layers,
        g.lt,
        g.lt * (g.lh / 2) * (g.lw / 2),
        2 * g.n_audio,
        p.steps - 1,
        t1.elapsed().as_secs_f64()
    );
    let tp = Instant::now();
    let mut progress = |i: usize, n: usize| eprint!("\r[step {i}/{n}, {:.1}s]", tp.elapsed().as_secs_f64());
    let video = pipe.generate(&p, cond, &mut progress)?;
    let default = if cfg!(feature = "media") { "video.avi" } else { "video.png" };
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from(default));
    let what = write_video(&video, &out)?;
    eprintln!(
        "[{}x{}, {} frames | {:.1}s | wrote {what}; no audio track]",
        video.width,
        video.height,
        video.frames.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
