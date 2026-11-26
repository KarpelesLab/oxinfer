//! Batched decode steps (`TextLm::forward_batch`, the server's concurrent requests) against
//! one-token steps: milliseconds a step for 1, 2, 4 and 8 sessions of a Qwen3 / Llama model.
//!
//! `cargo run --release --example rows_bench -- MODEL.gguf [steps] [rows,...]`
//! (`OXINFER_BATCH_FAST=1`: the fast kernels for 5+ rows.)

use oxinfer::platform::DevicePref;
use oxinfer::safetensors::ModelFiles;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(&args[1]);
    let steps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(32);
    let counts: Vec<usize> = args
        .get(3)
        .map_or("1,2,4,8".into(), String::clone)
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    if std::env::var_os("OXINFER_BATCH_FAST").is_some() {
        oxinfer::batch::set_fast(true);
    }
    let plat = oxinfer::platform::detect();
    oxinfer::gpu::init(&plat, DevicePref::Auto).unwrap().expect("a GPU");
    let files = ModelFiles::open(path, false).unwrap();
    let cfg = oxinfer::config::QwenConfig::load(path).unwrap();
    let model = oxinfer::qwen::QwenModel::load(cfg, &files, oxinfer::model::Quant::Bf16).unwrap();
    for &b in &counts {
        let mut sess: Vec<_> = (0..b)
            .map(|i| {
                let mut s = model.new_session();
                let p: Vec<u32> = (0..137u32).map(|j| (j * 97 + 13 * i as u32 + 5) % 30000).collect();
                model.forward(&mut s, &p);
                s
            })
            .collect();
        let mut toks: Vec<u32> = (0..b as u32).map(|i| 100 + i).collect();
        let mut best = f64::MAX;
        let mut total = 0.0;
        for step in 0..steps + 2 {
            let t0 = Instant::now();
            let ls = if b == 1 {
                vec![model.forward(&mut sess[0], &[toks[0]])]
            } else {
                let mut refs: Vec<&mut _> = sess.iter_mut().collect();
                model.forward_batch(&mut refs, &toks)
            };
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            if step >= 2 {
                best = best.min(dt);
                total += dt;
            }
            toks = ls.iter().map(|l| oxinfer::sampler::argmax(l)).collect();
        }
        println!(
            "{b} rows: {:.2} ms a step (best {best:.2}), {:.1} tok/s",
            total / steps as f64,
            b as f64 * 1e3 * steps as f64 / total
        );
    }
}
