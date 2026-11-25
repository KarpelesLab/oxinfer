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
//! as `parity_bench`'s tg rows). Options after `--` are `EngineOptions` keys for every engine
//! (default `draft=none lookup=0 mtp=0`).

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
    let run = |e: &mut Engine, prompt: &str| -> Result<(usize, f64)> {
        e.reset();
        e.config_mut().max_tokens = n;
        e.complete(prompt, &mut |_| Flow::Continue)?;
        Ok((e.stats().generated_tokens, e.stats().decode_tok_s()))
    };
    let mut engines = Vec::new();
    for spec in &a[3..split] {
        let (label, envs) = spec.split_once(':').unwrap_or((spec, ""));
        let pairs: Vec<(&str, &str)> = envs
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|kv| kv.split_once('='))
            .collect();
        for (k, v) in &pairs {
            // SAFETY: single-threaded here (no engine is running).
            unsafe { std::env::set_var(k, v) };
        }
        let mut e = Engine::load(model, &opts)?;
        let ids = e.tokenizer().encode(corpus.as_str(), false)?.get_ids().to_vec();
        let prompt = e.tokenizer().decode(&ids[..15.min(ids.len())], false)?;
        // Warm-up: the plans are built (with this configuration's variables) and sized.
        run(&mut e, &prompt)?;
        for (k, _) in &pairs {
            // SAFETY: as above.
            unsafe { std::env::remove_var(k) };
        }
        engines.push((label.to_string(), e, prompt, Vec::new()));
    }
    for _ in 0..rounds {
        for (_, e, prompt, v) in engines.iter_mut() {
            v.push(run(e, prompt)?);
        }
    }
    for (label, _, _, v) in &mut engines {
        v.sort_by(|x, y| y.1.total_cmp(&x.1));
        println!(
            "{label:<12} {} tokens: best {:.1} tok/s, median {:.1} tok/s",
            v[0].0,
            v[0].1,
            v[v.len() / 2].1
        );
    }
    Ok(())
}
