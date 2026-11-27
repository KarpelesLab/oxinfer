//! Inkling-2 on CUDA against the CPU path: the same prompt (pseudo-random tokens, in chunks of
//! `--chunk`) then `--steps` decode steps on the CPU's greedy tokens, comparing the logits after
//! the prompt (every position with `--all`) and after every step. Two runs, as the GPU backend is
//! process-wide:
//!
//! ```text
//! inkling2_cuda_check MODEL --cpu --save REF [--prompt-len N] [--chunk N] [--steps N] [--kv f32|f16|q8] [--all]
//! inkling2_cuda_check MODEL --load REF [same options]       # CUDA, compared with REF
//! inkling2_cuda_check MODEL [options] [--depth D]            # CUDA only: speed (two runs)
//! ```
//!
//! `--tail N` (with `--load` of an `--all` reference): the last N prompt tokens are fed as decode
//! steps instead, and their logits compared with the reference's prompt positions: decode steps
//! against the CPU's f32 prefill GEMM (its decode kernels round activations to int8).
//!
//! `--seed N` picks another pseudo-random prompt.
//!
//! `--lora DIR` attaches an adapter as the server attaches a profile's (behind a switch, turned
//! on; repeatable), `--lora-off` leaves the switches off. Each run tells whether its session
//! ended on the device.
//!
//! `--rollback R` (device only): after the decode steps, goes back R positions and feeds the same
//! tokens again, one at a time and then as one pass, comparing the logits with the first time
//! (a session's caches and convolution rings by position: they must match exactly).
//!
//! `--keep N` (with `--all`): only the logits of the last N prompt positions are kept.
//!
//! `--depth D` first runs D untimed tokens (in chunks of `--chunk`), as llama-bench's `-d`: the
//! prompt and the decode steps are then timed at that context.

use anyhow::{Context, Result};
use oxinfer::inkling::{Inkling2Config, Inkling2Model, KvCache};
use oxinfer::safetensors::ModelFiles;
use std::time::Instant;

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

/// KL(p || q) of the softmaxes of logits `a` (reference) and `b`.
fn kl(a: &[f32], b: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().fold(f32::NEG_INFINITY, |x, &y| x.max(y)) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(a), lse(b));
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let pa = (x as f64 - la).exp();
            pa * ((x as f64 - la) - (y as f64 - lb))
        })
        .sum()
}

struct Run {
    logits: Vec<Vec<f32>>,
    tokens: Vec<u32>,
    prefill: f64,
    decode: f64,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: inkling2_cuda_check MODEL [--cpu --save REF | --load REF] [--prompt-len N] [--chunk N] [--steps N] [--kv f32|f16|q8] [--all]";
    let model = args.next().context(usage)?;
    let (mut plen, mut chunk, mut steps, mut kv, mut all, mut cpu) = (300usize, 512usize, 16usize, KvCache::F32, false, false);
    let (mut save, mut load) = (None, None);
    let mut depth = 0usize;
    let mut tail = 0usize;
    let mut keep = usize::MAX;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rollback = 0usize;
    let (mut loras, mut lora_on) = (Vec::<String>::new(), true);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--prompt-len" => plen = args.next().context("N")?.parse()?,
            "--chunk" => chunk = args.next().context("N")?.parse()?,
            "--steps" => steps = args.next().context("N")?.parse()?,
            "--kv" => kv = KvCache::parse(&args.next().context("MODE")?)?,
            "--all" => all = true,
            "--cpu" => cpu = true,
            "--save" => save = Some(args.next().context("FILE")?),
            "--load" => load = Some(args.next().context("FILE")?),
            "--depth" => depth = args.next().context("D")?.parse()?,
            "--tail" => tail = args.next().context("N")?.parse()?,
            "--keep" => keep = args.next().context("N")?.parse()?,
            "--rollback" => rollback = args.next().context("R")?.parse()?,
            "--seed" => seed ^= args.next().context("N")?.parse::<u64>()?.wrapping_mul(0x9E37_79B9_7F4A_7C15),
            "--lora" => loras.push(args.next().context("DIR")?),
            "--lora-off" => lora_on = false,
            o => anyhow::bail!("unknown argument {o}"),
        }
    }
    let pref = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), pref)?;
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let files = ModelFiles::open(&dir, true)?;
    let mut m = Inkling2Model::load(Inkling2Config::load(&dir)?, &files, oxinfer::model::Quant::Bf16)?;
    m.kv = kv;
    // Attached as the server attaches a profile's adapter: behind a switch, on or off.
    let mut switches = Vec::new();
    for l in &loras {
        let sw = oxinfer::lora::attach_switched(&mut m, std::path::Path::new(l), &dir, 1.0)?;
        sw.set(lora_on);
        println!(
            "adapter {l}: rank {}, {} modules, switched {}",
            sw.adapter.rank,
            sw.adapter.modules,
            if lora_on { "on" } else { "off" }
        );
        switches.push(sw);
    }
    let on_device = std::cell::Cell::new(false);
    let mut s = seed;
    let tokens: Vec<u32> = (0..plen + depth)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            1000 + (s % 150_000) as u32
        })
        .collect();
    let u = m.cfg.unpadded_vocab_size;
    let (pre, tokens) = tokens.split_at(depth);
    let (tokens, tail_toks) = tokens.split_at(tokens.len() - tail);
    let run = |forced: Option<&[u32]>| -> Run {
        let h = m.cfg.hidden_size;
        let mut sess = m.new_session();
        for c in pre.chunks(chunk) {
            m.forward_hidden(&mut sess, c);
        }
        let mut out = Vec::new();
        let t0 = Instant::now();
        let mut last = Vec::new();
        let first_kept = tokens.len().saturating_sub(keep);
        let mut at = 0;
        for c in tokens.chunks(chunk) {
            let hs = m.forward_hidden(&mut sess, c);
            if all && at + c.len() > first_kept {
                let skip = first_kept.saturating_sub(at);
                out.extend(m.logits(&hs[skip * h..], c.len() - skip).chunks(u).map(|r| r.to_vec()));
            }
            at += c.len();
            last = hs[hs.len() - h..].to_vec();
        }
        let lg = m.logits(&last, 1);
        let prefill = t0.elapsed().as_secs_f64();
        if !all {
            out.push(lg.clone());
        }
        let mut toks = vec![argmax(&lg) as u32];
        for &tk in tail_toks {
            out.push(m.forward(&mut sess, &[tk]));
        }
        let t1 = Instant::now();
        for i in 0..steps {
            let tk = forced.map_or(toks[i], |f| f[i]);
            let lg = m.forward(&mut sess, &[tk]);
            toks.push(argmax(&lg) as u32);
            out.push(lg);
        }
        on_device.set(sess.on_device());
        Run {
            logits: out,
            tokens: toks,
            prefill,
            decode: t1.elapsed().as_secs_f64(),
        }
    };
    let report = |name: &str, r: &Run| {
        println!(
            "{name}: prefill {plen} tokens at depth {depth} in {:.2}s ({:.0} tok/s), {steps} decode steps {:.1} tok/s; session on the device: {}; tokens {:?}",
            r.prefill,
            plen as f64 / r.prefill,
            steps as f64 / r.decode,
            on_device.get(),
            &r.tokens[..r.tokens.len().min(12)]
        );
    };
    if let Some(f) = save {
        let r = run(None);
        report(if cpu { "cpu" } else { "device" }, &r);
        let mut b: Vec<u8> = Vec::new();
        b.extend((r.tokens.len() as u32).to_le_bytes());
        b.extend(r.tokens.iter().flat_map(|t| t.to_le_bytes()));
        b.extend((r.logits.len() as u32).to_le_bytes());
        for l in &r.logits {
            b.extend(l.iter().flat_map(|v| v.to_le_bytes()));
        }
        std::fs::write(f, b)?;
        return Ok(());
    }
    if rollback > 0 {
        let mut sess = m.new_session();
        for c in tokens.chunks(chunk) {
            m.forward_hidden(&mut sess, c);
        }
        let mut toks = vec![];
        let mut first = vec![];
        let mut tk = 1000u32;
        for _ in 0..steps {
            let lg = m.forward(&mut sess, &[tk]);
            toks.push(tk);
            tk = argmax(&lg) as u32;
            first.push(lg);
        }
        let back = rollback.min(steps);
        sess.truncate(sess.pos - back)?;
        let mut worst = 0f32;
        for i in steps - back..steps {
            let lg = m.forward(&mut sess, &[toks[i]]);
            worst = worst.max(lg.iter().zip(&first[i]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
        }
        println!("rollback {back}, one token at a time: max logit diff {worst:.3e} against the first time");
        sess.truncate(sess.pos - back)?;
        let hs = m.forward_hidden(&mut sess, &toks[steps - back..]);
        let lg = m.logits(&hs, back);
        let u = m.cfg.unpadded_vocab_size;
        let mut worst = 0f32;
        for (j, i) in (steps - back..steps).enumerate() {
            worst = worst.max(
                lg[j * u..(j + 1) * u]
                    .iter()
                    .zip(&first[i])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max),
            );
        }
        println!("rollback {back}, as one pass of {back}: max logit diff {worst:.3e} against one-token steps");
        return Ok(());
    }
    let Some(f) = load else {
        report("device", &run(None));
        report("device (again)", &run(None));
        return Ok(());
    };
    let b = std::fs::read(f)?;
    let rd = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    let nt = rd(0) as usize;
    let ref_toks: Vec<u32> = (0..nt).map(|i| rd(4 + 4 * i)).collect();
    let nl = rd(4 + 4 * nt) as usize;
    let base = 8 + 4 * nt;
    let ref_logits: Vec<Vec<f32>> = (0..nl)
        .map(|i| {
            b[base + 4 * u * i..base + 4 * u * (i + 1)]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        })
        .collect();
    let rg = run(Some(&ref_toks));
    report("device, the reference's tokens", &rg);
    let n_pf = if all { tokens.len().min(keep) } else { 1 };
    let (mut kmax, mut ksum, mut dmax, mut same) = (0f64, 0f64, 0f32, 0usize);
    for (i, (a, b)) in ref_logits.iter().zip(&rg.logits).enumerate() {
        let k = kl(a, b);
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        kmax = kmax.max(k);
        ksum += k;
        dmax = dmax.max(d);
        same += (argmax(a) == argmax(b)) as usize;
        if std::env::var("CHECK_VERBOSE").is_ok() && (k > 1e-3 || i % 100 == 0 || std::env::var("CHECK_VERBOSE").is_ok_and(|v| v == "2")) {
            println!("  position {i}: KL {k:.3e}, max diff {d:.3e}");
        }
        if i == n_pf - 1 {
            println!(
                "prefill: max logit diff {dmax:.3e}, mean KL {:.3e}, max KL {kmax:.3e}, same top {same}/{n_pf}",
                ksum / n_pf as f64
            );
            (kmax, ksum, dmax, same) = (0.0, 0.0, 0.0, 0);
        }
    }
    let n = ref_logits.len().min(rg.logits.len()) - n_pf;
    if n > 0 {
        let what = if tail > 0 { "decode steps on prompt tokens" } else { "decode" };
        println!(
            "{what}: max logit diff {dmax:.3e}, mean KL {:.3e}, max KL {kmax:.3e}, same top {same}/{n}",
            ksum / n as f64
        );
    }
    let r2 = run(None);
    report("device, greedy", &r2);
    println!(
        "greedy: the first {} of {} tokens are the reference's",
        r2.tokens.iter().zip(&ref_toks).take_while(|(a, b)| a == b).count(),
        ref_toks.len()
    );
    Ok(())
}
