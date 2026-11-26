//! A/B timing of decoding in one process: one engine per configuration (environment variables
//! read as its plans are built, e.g. `OXINFER_DEC_ATTN_ROPE=0`), then their runs of G greedy
//! tokens after a short prompt alternated one by one, so that other jobs' load on a shared GPU
//! hits them alike:
//!
//! ```text
//! dec_ab MODEL TOKENS ROUNDS LABEL[:KEY=VALUE,...] ... [-- key=value ...]
//! ```
//!
//! Prints, per configuration, the best and the median decode speed (the engine's decode tok/s,
//! as `parity_bench`'s tg rows) and hashes of its generated tokens (one per distinct reply over
//! the runs), so that configurations meant to decode the same tokens can be checked to. Options
//! after `--` are `EngineOptions` keys for every engine (default `draft=none lookup=0 mtp=0`); a
//! configuration's `opt.KEY=VALUE` pairs are `EngineOptions` keys of its engine alone, and
//! `chain=0` / `chain=1` turns the device's greedy chain off / on for its runs (the switch is
//! process-wide: `oxinfer::gpu::set_decode_chain` before each run), e.g. `steps:chain=0`.

use anyhow::{Context, Result};
use oxinfer::engine::{Engine, EngineOptions, Flow};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: dec_ab MODEL TOKENS ROUNDS LABEL[:KEY=VALUE,...] ... [-- key=value ...]";
    let model = a.first().context(usage)?;
    let n: usize = a.get(1).context(usage)?.parse()?;
    let rounds: usize = a.get(2).context(usage)?.parse()?;
    let split = a.iter().position(|x| x == "--").unwrap_or(a.len());
    let mut opts = EngineOptions::default();
    for kv in ["draft=none", "lookup=0", "mtp=0"]
        .iter()
        .map(|s| s.to_string())
        .chain(a[(split + 1).min(a.len())..].iter().cloned())
    {
        let (k, v) = kv.split_once('=').context("options are key=value")?;
        opts.set(k, v)?;
    }
    let corpus = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tools/bench/corpus.txt"))?;
    let run = |e: &mut Engine, prompt: &str| -> Result<(usize, f64, u64)> {
        e.reset();
        e.config_mut().max_tokens = n;
        let r = e.complete(prompt, &mut |_| Flow::Continue)?;
        // FNV-1a of the token ids.
        let h = r
            .tokens
            .iter()
            .fold(0xcbf29ce484222325u64, |h, &t| (h ^ t as u64).wrapping_mul(0x100000001b3));
        Ok((e.stats().generated_tokens, e.stats().decode_tok_s(), h))
    };
    let mut engines = Vec::new();
    for spec in &a[3..split] {
        let (label, envs) = spec.split_once(':').unwrap_or((spec, ""));
        let pairs: Vec<(&str, &str)> = envs
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let mut eo = opts.clone();
        let mut chain = None;
        for (k, v) in &pairs {
            match k.strip_prefix("opt.") {
                Some(k) => eo.set(k, v)?,
                None if *k == "chain" => chain = Some(*v != "0"),
                // SAFETY: single-threaded here (no engine is running).
                None => unsafe { std::env::set_var(k, v) },
            }
        }
        oxinfer::gpu::set_decode_chain(chain);
        let mut e = Engine::load(model, &eo)?;
        let ids = e.tokenizer().encode(corpus.as_str(), false)?.get_ids().to_vec();
        // `OXINFER_AB_PROMPT=FILE`: that raw text as the prompt instead.
        let prompt = match std::env::var("OXINFER_AB_PROMPT") {
            Ok(f) => std::fs::read_to_string(f)?,
            Err(_) => e.tokenizer().decode(&ids[..15.min(ids.len())], false)?,
        };
        // Warm-up: the plans are built (with this configuration's variables) and sized.
        run(&mut e, &prompt)?;
        for (k, _) in pairs.iter().filter(|(k, _)| !k.starts_with("opt.") && *k != "chain") {
            // SAFETY: as above.
            unsafe { std::env::remove_var(k) };
        }
        engines.push((label.to_string(), e, prompt, chain, Vec::new()));
    }
    for _ in 0..rounds {
        for (_, e, prompt, chain, v) in engines.iter_mut() {
            oxinfer::gpu::set_decode_chain(*chain);
            v.push(run(e, prompt)?);
        }
    }
    for (label, _, _, _, v) in &mut engines {
        let hashes: std::collections::BTreeSet<u64> = v.iter().map(|r| r.2).collect();
        let hashes: Vec<String> = hashes.iter().map(|h| format!("{h:016x}")).collect();
        v.sort_by(|x, y| y.1.total_cmp(&x.1));
        println!(
            "{label:<12} {} tokens: best {:.1} tok/s, median {:.1} tok/s, tokens {}",
            v[0].0,
            v[0].1,
            v[v.len() / 2].1,
            hashes.join(" ")
        );
    }
    Ok(())
}
