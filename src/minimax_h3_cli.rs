//! `oxinfer --model MiniMaxAI/MiniMax-H3-Image --prompt "..." --size 1024x1024 -o out.png`: text
//! to image through `oxinfer::minimax_h3`. The text encoder is MiniMax-H3's (`text_encoder/` of
//! `MiniMaxAI/MiniMax-H3`): `--text-encoder`, `OXINFER_H3_TEXT_ENCODER`, else found in the release
//! folder or the oxinfer / Hugging Face download caches.
//!
//! `--model` may also be ComfyUI's single-file transformer (`Comfy-Org/MiniMax-H3-Image`'s
//! `diffusion_models/minimax_h3_image_*.safetensors`) with `--vae` (its
//! `vae/minimax_h3_image_vae_fp16.safetensors`) and `--text-encoder` (a Qwen3-VL-32B-Instruct
//! GGUF, ComfyUI's text encoder file, or the release's folder). `--attention split|half|f32` sets
//! the operands of the Metal transformer's attention.

use crate::Args;
use anyhow::{Context, Result};
use oxinfer::minimax_h3::{Params, Pipeline};
use std::path::PathBuf;
use std::time::Instant;

/// MiniMax-H3's text encoder folder ([`oxinfer::minimax_h3::find_text_encoder`]).
pub(crate) fn text_encoder(model: &std::path::Path, cache_dir: Option<&std::path::Path>) -> Result<PathBuf> {
    oxinfer::minimax_h3::find_text_encoder(model, cache_dir)
}

pub fn run(args: &Args) -> Result<()> {
    let part = |spec: &str| -> Result<PathBuf> {
        let p = std::path::Path::new(spec);
        if p.exists() {
            Ok(p.to_path_buf())
        } else if spec.ends_with(".safetensors") || spec.ends_with(".gguf") {
            oxinfer::engine::resolve_file(spec, args.cache_dir.as_deref())
        } else {
            oxinfer::engine::resolve_model(spec, args.cache_dir.as_deref())
        }
    };
    let te = match &args.flux.text_encoder {
        Some(t) => part(t)?,
        None => text_encoder(&args.model, args.cache_dir.as_deref())?,
    };
    let t0 = Instant::now();
    let mut pipe = if args.model.is_file() {
        let vae = args.flux.vae.as_deref().with_context(|| {
            format!(
                "{} is a single-file transformer: name the image VAE with --vae (ComfyUI's minimax_h3_image_vae_fp16.safetensors)",
                args.model.display()
            )
        })?;
        Pipeline::load_files(&args.model, &part(vae)?, &te)?
    } else {
        Pipeline::load(&args.model, &te)?
    };
    if let Some(a) = &args.h3_attention {
        pipe.dit.gpu_attention =
            Some(oxinfer::gpu::dit::AttnPrecision::parse(a).with_context(|| format!("--attention {a}: half, split or f32"))?);
    }
    if let Some(a) = &args.h3_inputs {
        pipe.dit.gpu_inputs = Some(oxinfer::gpu::dit::InputPrecision::parse(a).with_context(|| format!("--inputs {a}: half or split"))?);
    }
    eprintln!(
        "[MiniMax-H3-Image: {} blocks, {} steps; text encoder {}; loaded in {:.1}s]",
        pipe.dit.cfg.layers,
        pipe.sigmas.len() - 1,
        te.display(),
        t0.elapsed().as_secs_f64()
    );
    let prompt = args.prompt.clone().context("--prompt is required")?;
    let mut p = Params::new(&prompt);
    if let Some((w, h)) = args.size {
        (p.width, p.height) = (w, h);
    }
    p.seed = args.seed;
    if let Ok(a) = std::env::var("OXINFER_H3_MIN_AREA") {
        p.min_area = a.parse().context("OXINFER_H3_MIN_AREA")?;
    }
    let tp = Instant::now();
    let mut progress = |i: usize, n: usize| eprint!("\r[step {i}/{n}, {:.1}s]", tp.elapsed().as_secs_f64());
    let img = pipe.generate(&p, &mut progress)?;
    eprintln!();
    let out = args.output.clone().unwrap_or_else(|| PathBuf::from("image.png"));
    std::fs::write(&out, oxinfer::png::encode_rgb(img.width, img.height, &img.rgb)).with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[{}x{} | {:.1}s | wrote {}]",
        img.width,
        img.height,
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}
