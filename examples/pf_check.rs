//! Prompt-processing check through the `engine` API: the greedy continuation of a raw text
//! prompt with the top log-probabilities of every step (the first from the prompt's prefill),
//! written to a file, or two such files compared (e.g. two builds, or `device=cpu` against the
//! GPU):
//!
//! ```text
//! pf_check MODEL TEXT_FILE OUT [n=64] [top=20] [tokens=N] [key=value ...]
//! pf_check --compare A B
//! ```
//!
//! `tokens=N` cuts the prompt to its first N tokens; other options are `EngineOptions` keys.
//! `--compare` prints how many greedy tokens agree, and for the first step (the prefill's
//! output) the largest log-probability difference over A's top tokens and the KL divergence
//! restricted to them.

use anyhow::{Context, Result};
use oxinfer::engine::{Engine, EngineOptions, Flow};

/// One step: the chosen token and the top (token, log-probability) pairs.
type Step = (u32, Vec<(u32, f32)>);

fn read(p: &str) -> Result<Vec<Step>> {
    let mut v = Vec::new();
    for line in std::fs::read_to_string(p)?.lines() {
        let mut it = line.split_whitespace();
        let tok: u32 = it.next().context("empty line")?.parse()?;
        let mut top = Vec::new();
        for kv in it {
            let (k, l) = kv.split_once(':').context("token:logprob")?;
            top.push((k.parse()?, l.parse()?));
        }
        v.push((tok, top));
    }
    Ok(v)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.first().map(String::as_str) == Some("--compare") {
        let (x, y) = (read(a.get(1).context("A")?)?, read(a.get(2).context("B")?)?);
        let same = x.iter().zip(&y).take_while(|(p, q)| p.0 == q.0).count();
        let (fx, fy) = (&x[0].1, &y[0].1);
        let (mut mx, mut kl) = (0f32, 0f64);
        for &(t, lp) in fx {
            if let Some(&(_, lq)) = fy.iter().find(|e| e.0 == t) {
                mx = mx.max((lp - lq).abs());
                kl += (lp as f64).exp() * (lp as f64 - lq as f64);
            } else {
                mx = f32::INFINITY;
            }
        }
        println!(
            "greedy {same}/{} tokens agree; first step: max logprob diff {mx:.3e} over {} top tokens, KL (top) {kl:.3e}",
            x.len().min(y.len()),
            fx.len()
        );
        return Ok(());
    }
    let usage = "usage: pf_check MODEL TEXT_FILE OUT [n=64] [top=20] [tokens=N] [key=value ...]";
    let (model, text, out) = (a.first().context(usage)?, a.get(1).context(usage)?, a.get(2).context(usage)?);
    let mut opts = EngineOptions::default();
    let (mut n, mut top, mut tokens) = (64usize, 20usize, None);
    for kv in &a[3..] {
        let (k, v) = kv.split_once('=').context("options are key=value")?;
        match k {
            "n" => n = v.parse()?,
            "top" => top = v.parse()?,
            "tokens" => tokens = Some(v.parse::<usize>()?),
            _ => opts.set(k, v)?,
        }
    }
    let mut e = Engine::load(model, &opts)?;
    let mut prompt = std::fs::read_to_string(text)?;
    if let Some(t) = tokens {
        let ids = e.tokenizer().encode(prompt.as_str(), false)?.get_ids().to_vec();
        prompt = e.tokenizer().decode(&ids[..t.min(ids.len())], false)?;
    }
    let c = e.config_mut();
    c.max_tokens = n;
    c.logprobs = Some(top);
    let r = e.complete(&prompt, &mut |_| Flow::Continue)?;
    eprintln!(
        "[{} prompt tokens, {} generated, prefill {:.1} tok/s]",
        r.stats.prompt_tokens,
        r.tokens.len(),
        r.stats.prefill_tok_s()
    );
    let mut s = String::new();
    for l in &r.logprobs {
        s.push_str(&l.token.to_string());
        for (t, p) in &l.top {
            s.push_str(&format!(" {t}:{p}"));
        }
        s.push('\n');
    }
    std::fs::write(out, s)?;
    Ok(())
}
