//! Compares a Raon-1 or Ripple checkpoint run by oxinfer with reference outputs (as written by
//! `tools/ref_raon.py` and `tools/ref_ripple.py`: `ids` (I64), `logits` `[n, vocab]` at every prompt position, and optionally
//! `gen` (I64, greedy tokens) with `decode_logits` `[len(gen), vocab]`, the logits after feeding
//! each of them one at a time):
//!
//! ```text
//! lm_check MODEL REF.safetensors [--chunk N] [--cpu] [--mmap] [--quant bf16|q8|q4] [--layers K]
//! lm_check MODEL --encode FILE
//! ```
//!
//! Prints, for the prefill (in chunks of N tokens, default the whole prompt) and for the
//! token-by-token decode, the largest logit difference, the mean KL divergence and the agreement
//! of the top token. `--layers K` cuts a Ripple model to its first K layers (as
//! `ref_ripple.py --layers K`: a part of a checkpoint too big for the reference). `--encode` in place of the reference prints the ids the checkpoint's
//! tokenizer gives the text of FILE.

use anyhow::{Context, Result};
use oxinfer::generate::TextLm;
use oxinfer::safetensors::{ModelFiles, SafeTensors};
use std::path::Path;

fn i64s(st: &SafeTensors, name: &str) -> Result<Vec<u32>> {
    Ok(st
        .bytes(name)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c) as u32)
        .collect())
}

fn f32s(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    Ok(st.bytes(name)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// Max |a - b|, KL(softmax(b) || softmax(a)), whether the argmaxes agree.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64, bool) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(a), lse(b));
    let kl = b
        .iter()
        .zip(a)
        .map(|(&y, &x)| {
            let pb = (y as f64 - lb).exp();
            pb * ((y as f64 - lb) - (x as f64 - la))
        })
        .sum::<f64>();
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    (max, kl, am(a) == am(b))
}

fn report(what: &str, ours: &[f32], refs: &[f32], vocab: usize) {
    let n = refs.len() / vocab;
    let (mut max, mut kl, mut same, mut worst) = (0f32, 0f64, 0, 0);
    let mut off = Vec::new();
    for i in 0..n {
        let (m, k, s) = compare(&ours[i * vocab..(i + 1) * vocab], &refs[i * vocab..(i + 1) * vocab]);
        if m > max {
            (max, worst) = (m, i);
        }
        if !s {
            off.push(i);
        }
        kl += k;
        same += s as usize;
    }
    if !off.is_empty() {
        eprintln!("[{what}: another top token at positions {off:?}]");
    }
    let scale = refs.iter().fold(0f32, |a, b| a.max(b.abs()));
    println!(
        "{what}: {n} positions, max logit diff {max:.3e} (at {worst}; logits up to {scale:.1}), mean KL {:.3e}, same top-1 {same}/{n}",
        kl / n as f64
    );
}

/// The prefill and the decode of `m` against the reference file `st`.
fn check<M: TextLm>(m: &M, vocab: usize, st: &SafeTensors, chunk: Option<usize>) -> Result<()> {
    let ids = i64s(st, "ids")?;
    let refs = f32s(st, "logits")?;
    let chunk = chunk.unwrap_or(ids.len()).max(1);
    let mut sess = m.new_session();
    let mut ours = Vec::with_capacity(refs.len());
    let t0 = std::time::Instant::now();
    for c in ids.chunks(chunk) {
        let hs = m.forward_hidden_mm(&mut sess, c, &[]);
        ours.extend(m.logits(&hs, c.len()));
    }
    eprintln!(
        "[prefill {} tokens in chunks of {chunk}: {:.2}s]",
        ids.len(),
        t0.elapsed().as_secs_f64()
    );
    report("prefill", &ours, &refs, vocab);

    if st.info("gen").is_some() {
        let gen_ids = i64s(st, "gen")?;
        let dref = f32s(st, "decode_logits")?;
        let mut got = Vec::with_capacity(dref.len());
        let mut greedy = Vec::with_capacity(gen_ids.len());
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
        // Our own greedy continuation of the prompt, beside the logits after the reference's.
        let mut next = argmax(&ours[ours.len() - vocab..]);
        for &tk in &gen_ids {
            greedy.push(next);
            let lg = m.forward(&mut sess, &[tk]);
            if next == tk {
                next = argmax(&lg);
            }
            got.extend(lg);
        }
        report("decode", &got, &dref, vocab);
        let same = greedy.iter().zip(&gen_ids).take_while(|(a, b)| a == b).count();
        println!("greedy tokens: {same}/{} match the reference's", gen_ids.len());
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: lm_check MODEL REF.safetensors [--chunk N] [--cpu] [--mmap] [--quant Q]";
    let model = args.next().context(usage)?;
    let refp = args.next().context(usage)?;
    let (mut chunk, mut cpu, mut mmap, mut quant) = (None, false, false, oxinfer::model::Quant::Bf16);
    let (mut text, mut layers) = (None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = Some(args.next().context("--chunk N")?.parse::<usize>()?),
            "--cpu" => cpu = true,
            "--mmap" => mmap = true,
            "--layers" => layers = Some(args.next().context("--layers K")?.parse::<usize>()?),
            "--quant" => quant = oxinfer::model::Quant::parse(&args.next().context("--quant Q")?).context("--quant: bf16, q8 or q4")?,
            other if refp == "--encode" && text.is_none() => text = Some(other.to_string()),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    if refp == "--encode" {
        let text = std::fs::read_to_string(text.context("--encode FILE")?)?;
        let tok = oxinfer::engine::load_tokenizer(&dir)?;
        let enc = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        println!("{:?}", enc.get_ids());
        return Ok(());
    }
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let files = ModelFiles::open(&dir, !mmap)?;
    let st = SafeTensors::open(Path::new(&refp), true)?;
    match oxinfer::config::model_type(&dir)?.as_str() {
        "raon1_nano" => {
            let m = oxinfer::raon::RaonModel::load(oxinfer::raon::RaonConfig::load(&dir)?, &files, quant)?;
            check(&m, m.cfg.vocab_size, &st, chunk)
        }
        "ricursive" => {
            let mut cfg = oxinfer::ripple::RippleConfig::load(&dir)?;
            if let Some(k) = layers {
                cfg.num_hidden_layers = k;
                if let Some(lt) = &mut cfg.layer_types {
                    lt.truncate(k);
                }
            }
            let m = oxinfer::ripple::RippleModel::load(cfg, &files, quant)?;
            check(&m, m.cfg.vocab_size, &st, chunk)
        }
        other => anyhow::bail!("lm_check: model_type {other} is not handled here"),
    }
}
