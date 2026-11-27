//! Runs an H3 corrector (`oxinfer::minimax_h3::corrector`) on the cases written by
//! `tools/minimax_h3_corrector_tiny.py` and compares with the release node pack's network:
//!
//! ```text
//! h3_corrector_check DIR [--device cpu|gpu]
//! ```
//!
//! `DIR` holds `corrector.safetensors` and `io.safetensors`. Prints the largest difference of the
//! raw residual and of the capped one for each case.
//!
//! `h3_corrector_check --time FILE T H W [L]` instead times one forward of a real corrector file
//! on random inputs of that latent shape (24 channels; `L` caption rows of the file's width);
//! with `--device gpu` on the accelerator's session, then once on the CPU for a comparison.

use anyhow::{Context, Result};
use oxinfer::minimax_h3::corrector::{Corrector, bound_residual};
use oxinfer::safetensors::ModelFiles;
use std::path::Path;
use std::time::Instant;

fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let m = b.iter().map(|x| x.abs()).fold(0f32, f32::max);
    (d, m)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: h3_corrector_check DIR [--device cpu|gpu] | --time FILE T H W [L]";
    let gpu = args.windows(2).any(|w| w[0] == "--device" && w[1] == "gpu");
    let pref = if gpu {
        oxinfer::platform::DevicePref::Auto
    } else {
        oxinfer::platform::DevicePref::Cpu
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), pref)?;
    if args.first().map(String::as_str) == Some("--time") {
        let p = |i: usize| -> Result<usize> { args.get(i).context(usage)?.parse().context(usage) };
        let (t, h, w) = (p(2)?, p(3)?, p(4)?);
        let l = p(5).unwrap_or(124);
        let t0 = Instant::now();
        let mut net = Corrector::load(Path::new(args.get(1).context(usage)?))?;
        println!("loaded in {:.1}s: {:?}", t0.elapsed().as_secs_f64(), net.cfg);
        let mut rng = oxinfer::sd::ops::Rng::new(1);
        let c = net.cfg.video_channels;
        let (z, v, text) = (
            rng.normals(c * t * h * w),
            rng.normals(c * t * h * w),
            rng.normals(l * net.cfg.text_dim),
        );
        let mut last = Vec::new();
        for i in 0..3 {
            let t1 = Instant::now();
            let raw = net.forward(&z, &v, t, h, w, 0.9, 0.85, &text)?;
            let (_, ratio) = bound_residual(&raw, &v, 0.5, 1e-3);
            println!(
                "forward {i}: [{c}, {t}, {h}, {w}] ({} tokens), {l} text rows: {:.2}s, ratio {ratio:.3}",
                t * (h / 2) * (w / 2),
                t1.elapsed().as_secs_f64()
            );
            last = raw;
        }
        if gpu {
            net.gpu = false;
            let t1 = Instant::now();
            let cpu = net.forward(&z, &v, t, h, w, 0.9, 0.85, &text)?;
            let num: f64 = last.iter().zip(&cpu).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let den: f64 = cpu.iter().map(|b| (*b as f64).powi(2)).sum();
            let (d, m) = max_diff(&last, &cpu);
            println!(
                "CPU forward {:.2}s; device vs CPU: relative L2 {:.3e}, max abs {d:.3e} (values up to {m:.3})",
                t1.elapsed().as_secs_f64(),
                (num / den).sqrt()
            );
        }
        return Ok(());
    }
    let dir = Path::new(args.first().context(usage)?);
    let net = Corrector::load(&dir.join("corrector.safetensors"))?;
    let io = ModelFiles::open_file(&dir.join("io.safetensors"), false)?;
    let mut worst = 0f32;
    for i in 0.. {
        let k = |n: &str| format!("{i}.{n}");
        if !io.has(&k("z")) {
            break;
        }
        let shape = io.info(&k("z")).unwrap().shape.clone();
        let (t, h, w) = (shape[1], shape[2], shape[3]);
        let (z, v, text) = (io.f32(&k("z"))?, io.f32(&k("v"))?, io.f32(&k("text"))?);
        let s = io.f32(&k("sigma"))?;
        let raw = net.forward(&z, &v, t, h, w, s[0], s[1], &text)?;
        let (applied, ratio) = bound_residual(&raw, &v, 0.5, 1e-3);
        let (dr, mr) = max_diff(&raw, &io.f32(&k("raw"))?);
        let (da, ma) = max_diff(&applied, &io.f32(&k("applied"))?);
        let want = io.f32(&k("ratio"))?[0];
        println!("case {i} {shape:?}: raw {dr:.2e} (values up to {mr:.2}), applied {da:.2e} (up to {ma:.2}), ratio {ratio:.5} / {want:.5}");
        worst = worst.max(dr / mr).max(da / ma);
    }
    println!("worst relative max difference {worst:.2e}");
    Ok(())
}
