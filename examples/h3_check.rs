//! Runs MiniMax-H3's DiT and VAE decoder (`oxinfer::minimax_h3`) on inputs written by
//! `tools/minimax_h3_tiny.py` and compares with diffusers' outputs:
//!
//! ```text
//! h3_check DIR
//! ```
//!
//! `DIR/case.json` describes the case; `DIR/*.bin` hold little-endian f32 arrays.

use anyhow::{Context, Result};
use oxinfer::minimax_h3::dit::{Dit, DitConfig, Packed};
use oxinfer::minimax_h3::vae::{VaeConfig, VaeDecoder};
use oxinfer::safetensors::ModelFiles;
use std::path::Path;

fn read(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| p.display().to_string())?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn cmp(name: &str, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let scale = b.iter().map(|v| v.abs()).sum::<f32>() / n.max(1) as f32;
    println!(
        "{name}: {} values (ref {}), max abs diff {max:.3e}, ref mean |v| {scale:.3e}",
        a.len(),
        b.len()
    );
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: h3_check DIR [--original]")?;
    let dir = Path::new(&dir);
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu)?;
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("case.json"))?)?;
    let u = |k: &str| case[k].as_u64().unwrap() as usize;
    // DiT: one forward per step head.
    // `--original`: the release's layout (tools/minimax_h3_tiny.py --original).
    let original = std::env::args().any(|a| a == "--original");
    let tdir = dir.join(if original { "dit_orig" } else { "dit" });
    let files = ModelFiles::open(&tdir, false)?;
    let heads_file = if original {
        Some(ModelFiles::open(&dir.join("pdd_fused_heads.safetensors"), false)?)
    } else {
        None
    };
    let dit = Dit::load(DitConfig::load(&tdir)?, &files, heads_file.as_ref())?;
    let text = read(&dir.join("text.bin"))?;
    let image = read(&dir.join("image.bin"))?;
    let pos: Vec<[f32; 3]> = read(&dir.join("positions.bin"))?.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let tags: Vec<u8> = read(&dir.join("tags.bin"))?.iter().map(|&v| v as u8).collect();
    let tidx: Vec<usize> = read(&dir.join("tidx.bin"))?.iter().map(|&v| v as usize).collect();
    let ts = read(&dir.join("timesteps.bin"))?;
    let emb = dit.embed_text(&text);
    for head in 0..u("heads") {
        let out = dit.forward(
            &Packed {
                text: &text,
                image: &image,
                positions: &pos,
                timesteps: &ts,
                timestep_index: &tidx,
                tags: &tags,
            },
            &emb,
            head,
        );
        cmp(&format!("DiT head {head}"), &out, &read(&dir.join(format!("dit_out_{head}.bin")))?);
    }
    // VAE decoder (tiled).
    let vdir = dir.join(if original { "vae_orig" } else { "vae" });
    let vfiles = ModelFiles::open(&vdir, false)?;
    let vae = VaeDecoder::load(VaeConfig::load(&vdir)?, &vfiles)?;
    let lat = read(&dir.join("latents.bin"))?;
    let img = vae.decode(&lat, u("lh"), u("lw"));
    cmp("VAE decode", &img, &read(&dir.join("vae_out.bin"))?);
    Ok(())
}
