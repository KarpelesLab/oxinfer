//! Compares a Nemotron-H checkpoint run by oxinfer with reference outputs (as written by
//! `tools/nemotron_tiny.py`: `ids` (I64), `logits` `[n, vocab]` at every prompt position, `gen`
//! (I64, greedy tokens), `decode_logits` `[len(gen), vocab]`, the logits after feeding each of
//! them through transformers' cache, and `decode_logits_full`, the same positions of one uncached
//! forward):
//!
//! ```text
//! nemotron_check MODEL REF.safetensors [--chunk N] [--quant bf16|q8|q4] [--dump FILE] [--cpu] [--mmap]
//! ```
//!
//! Prints, for the prefill (in chunks of N tokens, default the whole prompt) and for the
//! token-by-token decode, the largest logit difference, the mean KL divergence and the agreement
//! of the top token. The decode is run twice: with the time step floor (NVIDIA's module, and
//! this crate's default) against `decode_logits_full`, and without it against `decode_logits`
//! (transformers' cached step does not clamp). `--dump` writes the logits compared (prefill
//! rows, then the floored decode rows; f32).

use anyhow::{Context, Result};
use oxinfer::nemotron::{NemotronConfig, NemotronModel};
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
    for i in 0..n {
        let (m, k, s) = compare(&ours[i * vocab..(i + 1) * vocab], &refs[i * vocab..(i + 1) * vocab]);
        if m > max {
            (max, worst) = (m, i);
        }
        kl += k;
        same += s as usize;
    }
    println!(
        "{what}: {n} positions, max logit diff {max:.3e} (at {worst}), mean KL {:.3e}, same top-1 {same}/{n}",
        kl / n as f64
    );
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: nemotron_check MODEL REF.safetensors [--chunk N] [--quant bf16|q8|q4] [--dump FILE] [--cpu] [--mmap]";
    let model = args.next().context(usage)?;
    let refp = args.next().context(usage)?;
    let (mut chunk, mut cpu, mut mmap) = (None, false, false);
    let mut quant = oxinfer::model::Quant::Bf16;
    let mut dump: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = Some(args.next().context("--chunk N")?.parse::<usize>()?),
            "--quant" => quant = oxinfer::model::Quant::parse(&args.next().context("--quant Q")?).context("--quant bf16|q8|q4")?,
            "--cpu" => cpu = true,
            "--mmap" => mmap = true,
            "--dump" => dump = Some(args.next().context("--dump FILE")?),
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
    let cfg = NemotronConfig::load(&dir)?;
    let files = ModelFiles::open(&dir, !mmap)?;
    let t0 = std::time::Instant::now();
    let m = NemotronModel::load(cfg, &files, quant)?;
    eprintln!(
        "[{} blocks ({}), loaded in {:.2}s]",
        m.cfg.num_hidden_layers(),
        m.cfg.describe(),
        t0.elapsed().as_secs_f64()
    );
    let st = SafeTensors::open(Path::new(&refp), true)?;
    let ids = i64s(&st, "ids")?;
    let refs = f32s(&st, "logits")?;
    let vocab = m.cfg.vocab_size;
    let chunk = chunk.unwrap_or(ids.len()).max(1);

    let mut sess = m.new_session();
    let mut ours = Vec::with_capacity(refs.len());
    let t0 = std::time::Instant::now();
    for c in ids.chunks(chunk) {
        let hs = m.forward_hidden(&mut sess, c);
        ours.extend(m.logits(&hs, c.len()));
    }
    eprintln!(
        "[prefill {} tokens in chunks of {chunk}: {:.2}s]",
        ids.len(),
        t0.elapsed().as_secs_f64()
    );
    report("prefill", &ours, &refs, vocab);

    if st.info("gen").is_some() {
        let gen_ids = i64s(&st, "gen")?;
        // With the floor (the default), against the uncached forward.
        let mut floored = sess.clone();
        let mut got = Vec::new();
        for &tk in &gen_ids {
            got.extend(m.forward(&mut floored, &[tk]));
        }
        if st.info("decode_logits_full").is_some() {
            report(
                "decode (time step floor, vs the uncached forward)",
                &got,
                &f32s(&st, "decode_logits_full")?,
                vocab,
            );
        }
        // Without it, against transformers' cached steps.
        sess.dt_min = 0.0;
        let mut free = Vec::new();
        for &tk in &gen_ids {
            free.extend(m.forward(&mut sess, &[tk]));
        }
        report(
            "decode (no floor, vs transformers' cached steps)",
            &free,
            &f32s(&st, "decode_logits")?,
            vocab,
        );
        ours.extend(got);
    }
    if let Some(path) = dump {
        let bytes: Vec<u8> = ours.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&path, bytes).with_context(|| format!("write {path}"))?;
    }
    Ok(())
}
