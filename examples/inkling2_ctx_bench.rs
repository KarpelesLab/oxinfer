//! Decoding speed and cache memory of an Inkling-2 model at a given context length, without
//! running a prompt of that length: the caches are filled with placeholder keys and values
//! (`Inkling2Model::fill_for_bench`; the output is meaningless, the work per token is that of a
//! real context of the same length).
//!
//! ```text
//! inkling2_ctx_bench MODEL CTX [--kv f32|f16|q8] [--tokens N] [--prefill N] [--reps R] [--cpu]
//! ```
//!
//! `--prefill N`: also times passes of N random tokens appended at that context (a chat turn's
//! new tokens after a long conversation), R of them one after the other (each at the context the
//! previous left), before decoding.
//!
//! `--text FILE`: instead, a real prompt (the file's text, tokenized, fed in passes of
//! `--prefill` tokens, default 2048, from position 0; CTX is then ignored), then `--tokens` greedy
//! tokens, printed with the top logits of the prompt's last position (to compare kernels and cache
//! types on a long prompt; `OXINFER_CTX_DUMP=FILE` writes those logits).
//!
//! Run it under `/usr/bin/time -l` for the process's peak memory.

use anyhow::{Context, Result};
use oxinfer::inkling::{Inkling2Config, Inkling2Model, KvCache};
use oxinfer::safetensors::ModelFiles;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: inkling2_ctx_bench MODEL CTX [--kv f32|f16|q8] [--tokens N] [--prefill N] [--reps R] [--cpu]";
    let model = args.next().context(usage)?;
    let ctx: usize = args.next().context(usage)?.parse()?;
    let (mut kv, mut tokens, mut cpu) = (KvCache::F32, 16usize, false);
    let (mut prefill, mut reps) = (0usize, 1usize);
    let mut text: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kv" => kv = KvCache::parse(&args.next().context("--kv MODE")?)?,
            "--tokens" => tokens = args.next().context("--tokens N")?.parse()?,
            "--prefill" => prefill = args.next().context("--prefill N")?.parse()?,
            "--reps" => reps = args.next().context("--reps R")?.parse()?,
            "--text" => text = Some(args.next().context("--text FILE")?),
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
    if let Some(path) = text {
        let tk = oxinfer::engine::load_tokenizer(&dir)?;
        let s = std::fs::read_to_string(path)?;
        let ids = tk
            .encode(s.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        let chunk = if prefill == 0 { 2048 } else { prefill };
        let t0 = std::time::Instant::now();
        let mut logits = Vec::new();
        for c in ids.chunks(chunk) {
            logits = m.forward(&mut sess, c);
        }
        let dt = t0.elapsed().as_secs_f64();
        let mut top: Vec<usize> = (0..logits.len()).collect();
        top.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let tops: Vec<String> = top[..5].iter().map(|&i| format!("{i}:{:.4}", logits[i])).collect();
        println!(
            "prompt {} tokens in {dt:.1}s ({:.0} tok/s); top {}",
            ids.len(),
            ids.len() as f64 / dt,
            tops.join(" ")
        );
        if let Some(f) = std::env::var_os("OXINFER_CTX_DUMP") {
            std::fs::write(f, logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        let mut tok = top[0] as u32;
        let mut out = vec![tok];
        let t1 = std::time::Instant::now();
        for _ in 1..tokens {
            let l = m.forward(&mut sess, &[tok]);
            tok = l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
            out.push(tok);
        }
        let dt = t1.elapsed().as_secs_f64();
        println!("greedy {:?} ({:.1} tok/s)", out, (tokens - 1) as f64 / dt);
        println!("text {:?}", tk.decode(&out, false).unwrap_or_default());
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    m.fill_for_bench(&mut sess, ctx);
    let fill = t0.elapsed().as_secs_f64();
    let (per, win) = m.kv_bytes_per_token(kv);
    // A warm-up step, then the timed ones.
    let mut tok = 1000u32;
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
    tok = am(&m.forward(&mut sess, &[tok]));
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..reps * (prefill > 0) as usize {
        let p: Vec<u32> = (0..prefill)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                1000 + (seed >> 33) as u32 % 100_000
            })
            .collect();
        let at = sess.pos;
        let t = std::time::Instant::now();
        let logits = m.forward(&mut sess, &p);
        let dt = t.elapsed().as_secs_f64();
        tok = am(&logits);
        // (`OXINFER_CTX_DUMP=FILE`: the first pass's logits, little-endian f32, to compare kernels.)
        if let Some(f) = std::env::var_os("OXINFER_CTX_DUMP").filter(|_| at == ctx + 1) {
            std::fs::write(f, logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        println!("prefill {prefill} at {at}: {:.1} tok/s ({:.0} ms)", prefill as f64 / dt, dt * 1e3);
    }
    let t1 = std::time::Instant::now();
    let mut last = Vec::new();
    for _ in 0..tokens {
        last = m.forward(&mut sess, &[tok]);
        tok = am(&last);
    }
    let dt = t1.elapsed().as_secs_f64();
    // (`OXINFER_CTX_DUMP_DECODE=FILE`: the last step's logits.)
    if let Some(f) = std::env::var_os("OXINFER_CTX_DUMP_DECODE") {
        std::fs::write(f, last.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
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
