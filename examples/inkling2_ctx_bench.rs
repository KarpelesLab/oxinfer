//! Decoding speed and cache memory of an Inkling-2 model at a given context length, without
//! running a prompt of that length: the caches are filled with placeholder keys and values
//! (`Inkling2Model::fill_for_bench`; the output is meaningless, the work per token is that of a
//! real context of the same length).
//!
//! ```text
//! inkling2_ctx_bench MODEL CTX [--kv f32|f16|q8] [--tokens N] [--cpu]
//! ```
//!
//! Run it under `/usr/bin/time -l` for the process's peak memory.

use anyhow::{Context, Result};
use oxinfer::inkling::{Inkling2Config, Inkling2Model, KvCache};
use oxinfer::safetensors::ModelFiles;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: inkling2_ctx_bench MODEL CTX [--kv f32|f16|q8] [--tokens N] [--cpu]";
    let model = args.next().context(usage)?;
    let ctx: usize = args.next().context(usage)?.parse()?;
    let (mut kv, mut tokens, mut cpu) = (KvCache::F32, 16usize, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kv" => kv = KvCache::parse(&args.next().context("--kv MODE")?)?,
            "--tokens" => tokens = args.next().context("--tokens N")?.parse()?,
            "--cpu" => cpu = true,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let files = ModelFiles::open(&dir, true)?;
    let mut m = Inkling2Model::load(Inkling2Config::load(&dir)?, &files, oxinfer::model::Quant::Bf16)?;
    m.kv = kv;
    let mut sess = m.new_session();
    let t0 = std::time::Instant::now();
    m.fill_for_bench(&mut sess, ctx);
    let fill = t0.elapsed().as_secs_f64();
    let (per, win) = m.kv_bytes_per_token(kv);
    // A warm-up step, then the timed ones.
    let mut tok = 1000u32;
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
    tok = am(&m.forward(&mut sess, &[tok]));
    let t1 = std::time::Instant::now();
    for _ in 0..tokens {
        tok = am(&m.forward(&mut sess, &[tok]));
    }
    let dt = t1.elapsed().as_secs_f64();
    println!(
        "context {ctx}, kv {kv:?}: caches {:.2} GB ({:.1} KB/token + up to {:.0} MB of windows), decode {:.2} tok/s ({:.1} ms/token), fill {fill:.1}s",
        m.kv_bytes(&sess) as f64 / 1e9,
        per / 1024.0,
        win / 1e6,
        tokens as f64 / dt,
        dt * 1e3 / tokens as f64
    );
    Ok(())
}
