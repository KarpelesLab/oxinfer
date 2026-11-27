//! Dumps for checking Nemotron-Labs-Image's tokenizer against the reference
//! (`tools/ref_nemotron_image.py` writes the same files):
//!
//! ```text
//! nemotron_image_check TOKDIR encode PIXELS.bin H W OUTDIR   codes.txt (first line "GH GW"), decoded.bin
//! nemotron_image_check TOKDIR decode CODES.txt OUT.bin       the decoder's output ([3, H, W] f32)
//! ```
//!
//! `OXINFER_VIT_F32=1` keeps the tokenizer's float32 weights (default bf16, as the reference's
//! `--bf16-weights`).

use anyhow::{Context, Result, bail};
use oxinfer::nemotron_image::vit::{Precision, Vit, VitConfig};
use std::path::Path;

fn save(path: impl AsRef<Path>, v: &[f32]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path.as_ref(), b).with_context(|| format!("write {}", path.as_ref().display()))
}

fn load_f32(path: &str) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("read {path}"))?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn read_codes(path: &str) -> Result<(Vec<u32>, (usize, usize))> {
    let s = std::fs::read_to_string(path)?;
    let (first, rest) = s.split_once('\n').context("codes file: no grid line")?;
    let g: Vec<usize> = first.split_whitespace().map(|t| t.parse()).collect::<Result<_, _>>()?;
    let codes = rest.split_whitespace().map(|t| t.parse()).collect::<Result<_, _>>()?;
    Ok((codes, (g[0], g[1])))
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        bail!("usage: nemotron_image_check TOKDIR encode|decode ...");
    }
    let dir = Path::new(&a[0]);
    let cfg = VitConfig::from_json(&serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?)?;
    let files = oxinfer::safetensors::ModelFiles::open(dir, false)?;
    let prec = if std::env::var_os("OXINFER_VIT_F32").is_some() {
        Precision::F32
    } else {
        Precision::Bf16
    };
    let t0 = std::time::Instant::now();
    match a[1].as_str() {
        "encode" => {
            let (h, w): (usize, usize) = (a[3].parse()?, a[4].parse()?);
            let vit = Vit::load(cfg, &files, prec, true, true)?;
            eprintln!("loaded in {:.2}s", t0.elapsed().as_secs_f64());
            let px = load_f32(&a[2])?;
            let t = std::time::Instant::now();
            let (codes, grid) = vit.encode(&px, h, w)?;
            eprintln!("encoded {}x{} in {:.3}s", grid.0, grid.1, t.elapsed().as_secs_f64());
            let out = Path::new(&a[5]);
            std::fs::create_dir_all(out)?;
            let line: Vec<String> = codes.iter().map(|c| c.to_string()).collect();
            std::fs::write(out.join("codes.txt"), format!("{} {}\n{}\n", grid.0, grid.1, line.join(" ")))?;
            let t = std::time::Instant::now();
            let img = vit.decode(&codes, grid)?;
            eprintln!("decoded in {:.3}s", t.elapsed().as_secs_f64());
            save(out.join("decoded.bin"), &img)?;
        }
        "decode" => {
            let vit = Vit::load(cfg, &files, prec, false, true)?;
            eprintln!("loaded in {:.2}s", t0.elapsed().as_secs_f64());
            let (codes, grid) = read_codes(&a[2])?;
            let t = std::time::Instant::now();
            let img = vit.decode(&codes, grid)?;
            eprintln!("decoded {}x{} in {:.3}s", grid.0, grid.1, t.elapsed().as_secs_f64());
            save(&a[3], &img)?;
        }
        m => bail!("unknown mode {m}"),
    }
    Ok(())
}
