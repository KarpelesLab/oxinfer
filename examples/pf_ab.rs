//! A/B timing of prompt processing in one process: one engine per configuration (environment
//! variables read as its plans are built, e.g. `OXINFER_PF_GPU_ROUTE=0`), then their passes
//! alternated one by one, so that other jobs' load on a shared GPU hits them alike:
//!
//! ```text
//! pf_ab MODEL TOKENS ROUNDS LABEL[:KEY=VALUE,...] ... [-- key=value ...]
//! ```
//!
//! Prints, per configuration, the fastest and the median prompt time (the engine's prefill
//! time: the pass and the last row's logits, tokenization excluded) and their tok/s. Options
//! after `--` are `EngineOptions` keys for every engine (default `draft=none lookup=0 mtp=0`).

use anyhow::{Context, Result};
use oxinfer::engine::{Engine, EngineOptions, Flow};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: pf_ab MODEL TOKENS ROUNDS LABEL[:KEY=VALUE,...] ... [-- key=value ...]";
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
        let prompt = e.tokenizer().decode(&ids[..(n - 1).min(ids.len())], false)?;
        let run = |e: &mut Engine| -> Result<f64> {
            e.reset();
            e.config_mut().max_tokens = 1;
            e.complete(&prompt, &mut |_| Flow::Continue)?;
            Ok(e.stats().prefill_secs)
        };
        // Warm-up: the plans are built (with this configuration's variables) and sized.
        run(&mut e)?;
        run(&mut e)?;
        for (k, _) in &pairs {
            // SAFETY: as above.
            unsafe { std::env::remove_var(k) };
        }
        engines.push((label.to_string(), e, prompt, Vec::new()));
    }
    for _ in 0..rounds {
        for (_, e, prompt, times) in engines.iter_mut() {
            e.reset();
            e.config_mut().max_tokens = 1;
            e.complete(prompt, &mut |_| Flow::Continue)?;
            times.push((e.stats().prefill_secs, e.stats().prompt_tokens));
        }
    }
    for (label, _, _, times) in &mut engines {
        times.sort_by(|x, y| x.0.total_cmp(&y.0));
        let (best, toks) = times[0];
        let med = times[times.len() / 2].0;
        println!(
            "{label:<12} {toks} tokens: best {:.2} ms ({:.0} tok/s), median {:.2} ms ({:.0} tok/s)",
            best * 1e3,
            toks as f64 / best,
            med * 1e3,
            toks as f64 / med
        );
    }
    Ok(())
}
