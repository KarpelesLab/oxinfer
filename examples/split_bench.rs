//! A Qwen-path model on one machine or split across two ([`oxinfer::split`], docs/cluster.md):
//! prompt and decode speed, and the logits to compare between the two.
//!
//! ```text
//! # on the other machine
//! oxinfer split-worker MODEL --listen 10.100.80.2:8090 --key K
//! # here
//! cargo run --release --example split_bench -- MODEL --split 10.100.80.2:8090 --key K --dump split.f32
//! cargo run --release --example split_bench -- MODEL --dump whole.f32 && cmp whole.f32 split.f32
//! ```
//! Options: `--prompt N` (prompt tokens, default 4096), `--decode M` (greedy steps, default 64),
//! `--chunk C` (prefill chunk, default the engine's 1024, which the GPU prefill plan raises to
//! its own size), `--sessions B` (then also `M` steps of `B` sessions at once, default 4; 0:
//! none), `--device D`, `--quant Q`. The prompt is pseudo-random tokens: the timings and the
//! comparison need no text. `--dump FILE` writes the logits after the prompt and after each
//! step as raw `f32`.

use oxinfer::config::QwenConfig;
use oxinfer::engine::EngineOptions;
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use oxinfer::sampler::argmax;
use std::time::Instant;

fn ids(n: usize, seed: u64, vocab: usize) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
    let top = vocab.min(50_000) as u64 - 1000;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (1000 + (s >> 33) % top) as u32
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: split_bench MODEL [--split ADDR --key KEY] [--prompt N] [--decode M] [--chunk C] [--sessions B] [--dump FILE]";
    let spec = args.next().expect(usage);
    let (mut split, mut key, mut dump) = (None, std::env::var("OXINFER_SPLIT_KEY").unwrap_or_default(), None);
    let (mut n_prompt, mut n_decode, mut chunk, mut n_sess) = (4096usize, 64usize, 1024usize, 4usize);
    let mut eo = EngineOptions::default();
    while let Some(a) = args.next() {
        let mut val = || args.next().expect(usage);
        match a.as_str() {
            "--split" => split = Some(val()),
            "--key" => key = val(),
            "--dump" => dump = Some(val()),
            "--prompt" => n_prompt = val().parse()?,
            "--decode" => n_decode = val().parse()?,
            "--chunk" => chunk = val().parse()?,
            "--sessions" => n_sess = val().parse()?,
            "--device" => eo.set("device", &val())?,
            "--quant" => eo.set("quant", &val())?,
            other => anyhow::bail!("unknown option {other}\n{usage}"),
        }
    }
    let dir = oxinfer::engine::resolve_model(&spec, None)?;
    oxinfer::gpu::ensure(eo.device)?;
    let t0 = Instant::now();
    let files = ModelFiles::open(&dir, false)?;
    let model = match &split {
        Some(addr) => oxinfer::split::load_front(&dir, &files, eo.quant, addr, &key)?,
        None => QwenModel::load(QwenConfig::load(&dir)?, &files, eo.quant)?,
    };
    let (h, vocab) = (model.cfg.hidden_size, model.cfg.vocab_size);
    eprintln!(
        "[{} layers here{}, hidden {h}, loaded in {:.1}s]",
        model.cfg.num_hidden_layers,
        split.as_ref().map(|a| format!(" (the rest on {a})")).unwrap_or_default(),
        t0.elapsed().as_secs_f64()
    );
    let chunk = model.prefill_chunk(chunk);
    // Warm-up: the plans of both sides, and the weights paged in.
    let t0 = Instant::now();
    {
        let mut s = model.new_session();
        let hs = oxinfer::generate::prefill(&model, &mut s, &ids(chunk.min(n_prompt).max(2), 7, vocab), chunk, &[]);
        let mut t = argmax(&model.logits(&hs[hs.len() - h..], 1));
        for _ in 0..4 {
            t = argmax(&model.forward(&mut s, &[t]));
        }
    }
    eprintln!("[warm-up {:.1}s, prefill chunk {chunk}]", t0.elapsed().as_secs_f64());

    let mut logits_out: Vec<f32> = Vec::new();
    let mut toks = Vec::new();
    let mut s = model.new_session();
    let prompt = ids(n_prompt, 1, vocab);
    let t0 = Instant::now();
    let hs = oxinfer::generate::prefill(&model, &mut s, &prompt, chunk, &[]);
    let l = model.logits(&hs[hs.len() - h..], 1);
    let dt = t0.elapsed().as_secs_f64();
    println!("prompt: {n_prompt} tokens in {dt:.2}s = {:.0} tok/s", n_prompt as f64 / dt);
    let mut t = argmax(&l);
    logits_out.extend_from_slice(&l);
    let t0 = Instant::now();
    for _ in 0..n_decode {
        toks.push(t);
        let l = model.forward(&mut s, &[t]);
        t = argmax(&l);
        logits_out.extend_from_slice(&l);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "decode: {n_decode} tokens in {dt:.2}s = {:.1} tok/s ({:.2} ms per token)",
        n_decode as f64 / dt,
        dt * 1e3 / n_decode as f64
    );
    drop(s);

    if n_sess > 1 {
        let mut sessions: Vec<_> = (0..n_sess).map(|_| model.new_session()).collect();
        let mut next = Vec::new();
        for (i, s) in sessions.iter_mut().enumerate() {
            let hs = oxinfer::generate::prefill(&model, s, &ids(64, 100 + i as u64, vocab), chunk, &[]);
            next.push(argmax(&model.logits(&hs[hs.len() - h..], 1)));
        }
        let t0 = Instant::now();
        for _ in 0..n_decode {
            let mut refs: Vec<&mut _> = sessions.iter_mut().collect();
            let ls = model.forward_batch(&mut refs, &next);
            for (n, l) in next.iter_mut().zip(&ls) {
                *n = argmax(l);
            }
            logits_out.extend_from_slice(&ls[0]);
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "{n_sess} sessions at once: {} tokens in {dt:.2}s = {:.1} tok/s together",
            n_sess * n_decode,
            (n_sess * n_decode) as f64 / dt
        );
    }

    let mut hash = 0xcbf29ce484222325u64;
    for v in &logits_out {
        hash = (hash ^ v.to_bits() as u64).wrapping_mul(0x100000001b3);
    }
    println!(
        "logits: {} rows, hash {hash:016x}; first tokens {:?}",
        logits_out.len() / vocab,
        &toks[..toks.len().min(8)]
    );
    if let Some(path) = dump {
        let bytes: Vec<u8> = logits_out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes)?;
        eprintln!("[wrote {path}]");
    }
    Ok(())
}
