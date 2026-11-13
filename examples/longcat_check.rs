//! Compares a LongCat checkpoint run by oxinfer with reference outputs (as written by
//! `tools/ref_longcat.py`: `ids` (I64), `logits` `[n, vocab]` at every prompt position, and
//! optionally `gen` (I64, greedy tokens) with `decode_logits` `[len(gen), vocab]`, the logits
//! after feeding each of them one at a time):
//!
//! ```text
//! longcat_check MODEL REF.safetensors [--chunk N] [--cpu] [--mmap]
//! longcat_check MODEL --load [TEXT...]
//! longcat_check MODEL --nll IDS.txt [--chunk N] [--from K] [--cpu] [--mmap]
//! ```
//!
//! Prints, for the prefill (in chunks of N tokens, default the whole prompt) and for the
//! token-by-token decode, the largest logit difference, the mean KL divergence and the agreement
//! of the top token. `--load` in place of the reference only opens the checkpoint (mapped: it
//! works on a file whose tensor data was not downloaded) and prints the ids its tokenizer gives
//! the texts that follow. `--nll` runs the token ids of a file (numbers separated by commas or
//! spaces) and prints the mean negative log-likelihood, in nats, the model gives each token after
//! the ones before it: over all of them, and over those from position K on (default 0).

use anyhow::{Context, Result};
use oxinfer::longcat::{LongcatConfig, LongcatModel};
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

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: longcat_check MODEL REF.safetensors [--chunk N] [--cpu] [--mmap]";
    let model = args.next().context(usage)?;
    let refp = args.next().context(usage)?;
    let (mut chunk, mut cpu, mut mmap) = (None, false, false);
    let mut texts = Vec::new();
    let (mut nll, mut from) = (None, 0usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = Some(args.next().context("--chunk N")?.parse::<usize>()?),
            "--cpu" => cpu = true,
            "--mmap" => mmap = true,
            "--from" => from = args.next().context("--from K")?.parse::<usize>()?,
            other if refp == "--load" => texts.push(other.to_string()),
            other if refp == "--nll" && nll.is_none() => nll = Some(other.to_string()),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    mmap |= refp == "--load" || refp == "--nll";
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let cfg = LongcatConfig::load(&dir)?;
    let files = ModelFiles::open(&dir, !mmap)?;
    let m = LongcatModel::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
    if refp == "--load" {
        let c = &m.cfg;
        println!(
            "loaded: {} layers of {} ({} attentions, {} with an indexer{}), {} experts and {} identity ones (top {}) of {}, {} n-gram tables, {:.2} GB read per token, {} tensors, {:.1} GB",
            c.num_layers,
            c.hidden_size,
            2 * c.num_layers,
            (0..2 * c.num_layers).filter(|&a| c.has_indexer(a)).count(),
            c.index.as_ref().map_or(String::new(), |d| format!(", top {}", d.topk)),
            c.n_routed_experts,
            c.zero_expert_num,
            c.moe_topk,
            c.expert_ffn_hidden_size,
            c.oe.as_ref().map_or(0, |o| o.tables()),
            m.streamed_bytes() as f64 / 1e9,
            files.names().count(),
            files.total_bytes() as f64 / 1e9
        );
        if !texts.is_empty() {
            let tok = oxinfer::engine::load_tokenizer(&dir)?;
            for text in texts {
                let enc = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
                println!("{:?}", enc.get_ids());
            }
        }
        return Ok(());
    }
    if refp == "--nll" {
        let text = std::fs::read_to_string(nll.context("--nll IDS.txt")?)?;
        let ids: Vec<u32> = text
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>())
            .collect::<Result<_, _>>()?;
        let vocab = m.cfg.vocab_size;
        let chunk = chunk.unwrap_or(256).max(1);
        let mut sess = m.new_session();
        let (mut all, mut tail, mut n_tail, mut hits) = (0f64, 0f64, 0usize, 0usize);
        let t0 = std::time::Instant::now();
        let mut pos = 0;
        for c in ids.chunks(chunk) {
            let hs = m.forward_hidden_mm(&mut sess, c, &[]);
            let lg = m.logits(&hs, c.len());
            for (i, row) in lg.chunks(vocab).enumerate() {
                // The row after token `pos + i` scores token `pos + i + 1`.
                let Some(&next) = ids.get(pos + i + 1) else { break };
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = mx + row.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
                let l = lse - row[next as usize] as f64;
                all += l;
                hits += (row[next as usize] as f64 == mx) as usize;
                if pos + i + 1 >= from {
                    tail += l;
                    n_tail += 1;
                }
            }
            pos += c.len();
        }
        let n = ids.len() - 1;
        println!(
            "nll: {} tokens in passes of {chunk}, {:.1}s: mean {:.4} nats (perplexity {:.2}), top-1 {}/{n}; from position {from}: mean {:.4} over {n_tail}",
            ids.len(),
            t0.elapsed().as_secs_f64(),
            all / n as f64,
            (all / n as f64).exp(),
            hits,
            tail / n_tail.max(1) as f64
        );
        return Ok(());
    }
    let st = SafeTensors::open(Path::new(&refp), true)?;
    let ids = i64s(&st, "ids")?;
    let refs = f32s(&st, "logits")?;
    let vocab = m.cfg.vocab_size;
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
        let gen_ids = i64s(&st, "gen")?;
        let dref = f32s(&st, "decode_logits")?;
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
