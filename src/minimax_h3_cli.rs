//! `oxinfer --model MiniMaxAI/MiniMax-H3-Image --prompt "..." --size 1024x1024 -o out.png`: text
//! to image through `oxinfer::minimax_h3`. The text encoder is MiniMax-H3's (`text_encoder/` of
//! `MiniMaxAI/MiniMax-H3`): `OXINFER_H3_TEXT_ENCODER`, else found in the release folder or the
//! oxinfer / Hugging Face download caches.

use crate::Args;
use anyhow::{Context, Result, bail};
use oxinfer::minimax_h3::{Params, Pipeline};
use std::path::PathBuf;
use std::time::Instant;

/// Whether a checkpoint folder holds its weights (not only its configuration).
fn has_weights(dir: &std::path::Path) -> bool {
    if dir.join("model.safetensors").exists() {
        return true;
    }
    let Ok(b) = std::fs::read(dir.join("model.safetensors.index.json")) else {
        return false;
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&b) else {
        return false;
    };
    v["weight_map"]
        .as_object()
        .is_some_and(|m| m.values().filter_map(|f| f.as_str()).all(|f| dir.join(f).exists()))
}

/// MiniMax-H3's text encoder folder.
fn text_encoder(model: &std::path::Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("OXINFER_H3_TEXT_ENCODER") {
        return Ok(PathBuf::from(p));
    }
    let mut roots = vec![model.to_path_buf()];
    let home = std::env::var("HOME").unwrap_or_default();
    for cache in [format!("{home}/.cache/oxinfer/hub"), format!("{home}/.cache/huggingface/hub")] {
        if let Ok(rd) = std::fs::read_dir(format!("{cache}/models--MiniMaxAI--MiniMax-H3/snapshots")) {
            roots.extend(rd.flatten().map(|e| e.path()));
        }
    }
    for r in &roots {
        for sub in ["text_encoder", "FL2VA/text_encoder"] {
            let p = r.join(sub);
            if p.join("config.json").exists() && has_weights(&p) {
                return Ok(p);
            }
        }
    }
    bail!(
        "MiniMax-H3's text encoder was not found: download it (`hf download MiniMaxAI/MiniMax-H3 --include 'text_encoder/*'`) or set OXINFER_H3_TEXT_ENCODER"
    )
}

pub fn run(args: &Args) -> Result<()> {
    let te = text_encoder(&args.model)?;
    let t0 = Instant::now();
    let pipe = Pipeline::load(&args.model, &te)?;
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
