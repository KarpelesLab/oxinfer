//! Compares a GLM-5.3 / GLM-5.3-Flash checkpoint run by oxinfer with reference outputs (as written by
//! `tools/glm5_tiny.py`: `ids` (I64), `logits` `[n, vocab]` at every prompt position, and
//! optionally `gen` (I64, greedy tokens) with `decode_logits` `[len(gen), vocab]`, the logits after
//! feeding each of them one at a time):
//!
//! ```text
//! glm5_check MODEL REF.safetensors [--chunk N] [--layers] [--dump FILE] [--cpu] [--mmap]
//! ```
//!
//! Prints, for the prefill (in chunks of N tokens, default the whole prompt) and for the
//! token-by-token decode, the largest logit difference, the mean KL divergence and the agreement
//! of the top token. `--dump` writes the logits compared (prefill rows, then decode rows; f32).

use anyhow::{Context, Result};
use oxinfer::glm5::Glm5Model;
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
    let usage = "usage: glm5_check MODEL REF.safetensors [--chunk N] [--layers] [--dump FILE] [--cpu] [--mmap]";
    let model = args.next().context(usage)?;
    let refp = args.next().context(usage)?;
    let (mut chunk, mut cpu, mut mmap, mut layers) = (None, false, false, false);
    let mut dump: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = Some(args.next().context("--chunk N")?.parse::<usize>()?),
            "--cpu" => cpu = true,
            "--mmap" => mmap = true,
            "--layers" => layers = true,
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
    let cfg = oxinfer::glm5::Glm5Config::load(&dir)?;
    let files = ModelFiles::open(&dir, !mmap)?;
    let m = Glm5Model::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
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

    // The residual stream after each layer (`hidden` `[layers, n, hidden]`), with `--layers`.
    if layers && st.info("hidden").is_some() {
        let href = f32s(&st, "hidden")?;
        let (n, h, t) = (m.cfg.num_hidden_layers, m.cfg.hidden_size, ids.len());
        let want: Vec<usize> = (1..=n).collect();
        let mut feats = Vec::new();
        m.forward_capture(&mut m.new_session(), &ids, &want, &mut feats);
        for l in 0..n {
            let (mut max, mut at, mut big) = (0f32, 0, 0f32);
            for ti in 0..t {
                for k in 0..h {
                    let (a, b) = (feats[(ti * n + l) * h + k], href[(l * t + ti) * h + k]);
                    big = big.max(b.abs());
                    if (a - b).abs() > max {
                        (max, at) = ((a - b).abs(), ti);
                    }
                }
            }
            println!("after layer {l}: max residual diff {max:.3e} (at position {at}; largest value {big:.2})");
        }
    }

    if st.info("gen").is_some() {
        let gen_ids = i64s(&st, "gen")?;
        let dref = f32s(&st, "decode_logits")?;
        let mut got = Vec::with_capacity(dref.len());
        for &tk in &gen_ids {
            got.extend(m.forward(&mut sess, &[tk]));
        }
        report("decode", &got, &dref, vocab);
        ours.extend(got);
    }
    if let Some(path) = dump {
        let bytes: Vec<u8> = ours.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&path, bytes).with_context(|| format!("write {path}"))?;
    }
    Ok(())
}
