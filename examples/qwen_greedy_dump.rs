//! Greedy reference run of a Qwen3.5-family model, the counterpart of
//! `tools/llama_logits_dump.cpp` (same output format, so the two can be compared):
//!
//! ```text
//! qwen_greedy_dump MODEL TEXT|@IDS_FILE N_GEN OUT.bin [--cpu] [--ids-out FILE]
//! qwen_greedy_dump --compare A.bin B.bin VOCAB
//! ```
//!
//! Feeds the prompt (raw text, tokenized without a template, or `@file` of whitespace-separated
//! token ids) as one batch, then generates `N_GEN` tokens greedily one at a time. Writes the f32
//! logits row of the prompt's last position and of every generated token, and prints the chosen
//! ids; `--ids-out` writes the prompt's token ids (the other tool's `IDS_FILE`). `--compare`
//! reports, per step, both top tokens and the largest log-probability difference over the first
//! file's top 10.

use anyhow::{Context, Result};
use oxinfer::qwen::QwenModel;
use oxinfer::safetensors::ModelFiles;
use std::path::Path;

fn read(p: &str) -> Result<Vec<f32>> {
    Ok(std::fs::read(p)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0
}

fn compare(a: &str, b: &str, vocab: usize) -> Result<()> {
    let (x, y) = (read(a)?, read(b)?);
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&t| (t as f64 - m).exp()).sum::<f64>().ln()
    };
    let (mut same, mut worst, mut split) = (0, 0f64, None);
    let steps = (x.len() / vocab).min(y.len() / vocab);
    for (i, (p, q)) in x.chunks_exact(vocab).zip(y.chunks_exact(vocab)).enumerate() {
        let (lp, lq) = (lse(p), lse(q));
        let mut idx: Vec<usize> = (0..vocab).collect();
        idx.sort_unstable_by(|&a, &b| p[b].total_cmp(&p[a]));
        let d = idx[..10]
            .iter()
            .map(|&t| ((p[t] as f64 - lp) - (q[t] as f64 - lq)).abs())
            .fold(0f64, f64::max);
        let (tp, tq) = (argmax(p), argmax(q));
        if tp != tq {
            // After the first different token the two runs see different contexts.
            split.get_or_insert((i, tp, tq, (p[tp] as f64 - lp).exp(), (p[tq] as f64 - lp).exp()));
        }
        if split.is_none() {
            same += 1;
            worst = worst.max(d);
        }
    }
    println!("{same}/{steps} greedy steps identical, top-10 log-probabilities within {worst:.3} over them");
    if let Some((i, tp, tq, pp, pq)) = split {
        println!("first difference at step {i}: {tp} (p {pp:.3}) vs {tq} (p {pq:.3} in the first run)");
    }
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.first().map(String::as_str) == Some("--compare") {
        return compare(&a[1], &a[2], a[3].parse()?);
    }
    let usage = "usage: qwen_greedy_dump MODEL TEXT|@IDS_FILE N_GEN OUT.bin [--cpu] [--ids-out FILE]";
    let (model, text, n_gen, out) = (
        a.first().context(usage)?,
        a.get(1).context(usage)?,
        a.get(2).context(usage)?.parse::<usize>()?,
        a.get(3).context(usage)?,
    );
    let dev = if a.iter().any(|x| x == "--cpu") {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = oxinfer::engine::resolve_model(model, None)?;
    let dir = Path::new(&dir);
    let ids: Vec<u32> = match text.strip_prefix('@') {
        Some(f) => std::fs::read_to_string(f)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?,
        None => oxinfer::engine::load_tokenizer(dir)?
            .encode(text.as_str(), false)?
            .get_ids()
            .to_vec(),
    };
    if let Some(f) = a.iter().position(|x| x == "--ids-out").and_then(|i| a.get(i + 1)) {
        std::fs::write(f, ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" "))?;
    }
    let cfg = oxinfer::config::QwenConfig::load(dir)?;
    let files = ModelFiles::open(dir, false)?;
    let m = QwenModel::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
    let h = m.cfg.hidden_size;
    let mut sess = m.new_session();
    let mut rows: Vec<u8> = Vec::new();
    let mut chosen = Vec::with_capacity(n_gen);
    let mut next = ids.clone();
    for step in 0..=n_gen {
        let hs = m.forward_hidden_mm(&mut sess, &next, &[]);
        let l = m.logits(&hs[hs.len() - h..], 1);
        rows.extend(l.iter().flat_map(|v| v.to_le_bytes()));
        if step == n_gen {
            break;
        }
        let t = argmax(&l) as u32;
        chosen.push(t);
        next = vec![t];
    }
    std::fs::write(out, rows)?;
    println!("{}", chosen.iter().map(u32::to_string).collect::<Vec<_>>().join(" "));
    Ok(())
}
