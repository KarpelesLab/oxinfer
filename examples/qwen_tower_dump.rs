//! The Qwen vision tower alone on one image (no language model is loaded):
//!
//! ```text
//! qwen_tower_dump MODEL_DIR IMAGE PREFIX [--weights FILE.safetensors] [--cpu]
//! ```
//!
//! Writes `PREFIX_pixels.bin` (the processor's patches) and `PREFIX_image.bin` (the merged image
//! tokens), f32, as `oxinfer --dump-mm` does; `tools/ref_vl.py --tower` compares them with
//! transformers. `--weights` names the file holding the tower's tensors when the folder has only
//! that shard of the checkpoint (the configuration files are read from `MODEL_DIR`).

use anyhow::{Context, Result};
use oxinfer::qwen_vision::{QwenMmConfig, QwenVisionTower, preprocess_image};
use oxinfer::safetensors::ModelFiles;
use std::path::{Path, PathBuf};

fn save(path: String, v: &[f32]) -> Result<()> {
    std::fs::write(&path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).with_context(|| format!("write {path}"))
}

fn main() -> Result<()> {
    let mut pos = Vec::new();
    let (mut weights, mut cpu) = (None, false);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--weights" => weights = Some(PathBuf::from(args.next().context("--weights FILE")?)),
            "--cpu" => cpu = true,
            _ => pos.push(a),
        }
    }
    let [model, image, prefix] = pos.as_slice() else {
        anyhow::bail!("usage: qwen_tower_dump MODEL_DIR IMAGE PREFIX [--weights FILE.safetensors] [--cpu]");
    };
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = Path::new(model);
    let mm = QwenMmConfig::load(dir)?.context("not a Qwen vision-language checkpoint")?;
    let files = ModelFiles::open(weights.as_deref().unwrap_or(dir), false)?;
    let tower = QwenVisionTower::load(mm.vision.clone(), &files)?;
    let img = oxinfer::media::Media::new().load_image(Path::new(image))?;
    let px = preprocess_image(&img.data, img.width, img.height, &mm.image, 0)?;
    let t = std::time::Instant::now();
    let emb = tower.encode(&px);
    eprintln!(
        "[{}x{} -> {:?} patches -> {} tokens, tower {:.2}s]",
        img.width,
        img.height,
        px.grid,
        emb.len() / mm.vision.out_hidden_size,
        t.elapsed().as_secs_f64()
    );
    save(format!("{prefix}_pixels.bin"), &px.data)?;
    save(format!("{prefix}_image.bin"), &emb)
}
