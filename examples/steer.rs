//! Runtime interventions through the `engine` API: runs every prompt of a file under a list of
//! intervention specs (each a fresh conversation), changing them between requests without
//! reloading, and writes one JSON line per (spec, prompt) with the reply:
//! `cargo run --release --example steer -- MODEL PROMPTS.txt OUT.jsonl SPEC.json|none ... [key=value ...]`
//! Options: `EngineOptions` keys plus `max_tokens=N`, `alpha=A,B,...` (each SPEC is then run
//! once per strength, with every `steer` alpha and `ablate`-free spec scaled), `limit=N` and
//! `make=POS.txt,NEG.txt,OUT.safetensors` (builds a difference-of-means direction first).

use oxinfer::engine::{Engine, EngineOptions, Flow, SystemPrompt};
use oxinfer::intervene::{Pool, Spec};
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let mut pos = Vec::new();
    let mut opts = EngineOptions::default();
    let (mut max_tokens, mut alphas, mut limit, mut make) = (256usize, vec![1.0f32], usize::MAX, None);
    for a in std::env::args().skip(1) {
        match a.split_once('=') {
            Some(("max_tokens", v)) => max_tokens = v.parse()?,
            Some(("alpha", v)) => alphas = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            Some(("limit", v)) => limit = v.parse()?,
            Some(("make", v)) => make = Some(v.to_string()),
            Some((k, v)) if !k.contains('/') && !k.contains('.') => opts.set(k, v)?,
            _ => pos.push(a),
        }
    }
    if pos.len() < 4 {
        anyhow::bail!("usage: steer MODEL PROMPTS OUT.jsonl SPEC.json|none ... [key=value ...]");
    }
    let mut engine = Engine::load(&pos[0], &opts)?;
    engine.config_mut().max_tokens = max_tokens;
    engine.config_mut().system = SystemPrompt::None;
    if let Some(m) = make {
        let p: Vec<&str> = m.split(',').collect();
        anyhow::ensure!(p.len() == 3, "make=POS.txt,NEG.txt,OUT.safetensors");
        let (a, b) = (
            oxinfer::intervene::read_prompts(p[0].as_ref())?,
            oxinfer::intervene::read_prompts(p[1].as_ref())?,
        );
        let t = std::time::Instant::now();
        let d = engine.make_direction(&a, &b, Pool::Last, true)?;
        oxinfer::intervene::save_vectors(p[2].as_ref(), &d, &[("kind", "difference of means".into())])?;
        eprintln!(
            "[direction from {} + {} prompts in {:.1}s -> {}]",
            a.len(),
            b.len(),
            t.elapsed().as_secs_f64(),
            p[2]
        );
    }
    let prompts: Vec<String> = oxinfer::intervene::read_prompts(pos[1].as_ref())?.into_iter().take(limit).collect();
    let mut out = std::fs::File::create(&pos[2])?;
    for spec_file in &pos[3..] {
        let base = if spec_file == "none" {
            None
        } else {
            Some(Spec::from_json(&std::fs::read_to_string(spec_file)?)?)
        };
        for &alpha in &alphas {
            // A new strength is a new spec on the same engine: nothing reloads.
            let spec = base.clone().map(|mut s| {
                for st in &mut s.steer {
                    st.alpha *= alpha;
                }
                s
            });
            engine.set_interventions(spec)?;
            let t = std::time::Instant::now();
            let mut toks = 0;
            for p in &prompts {
                engine.reset();
                let r = engine.send(p, &mut |_| Flow::Continue)?;
                toks += r.stats.generated_tokens;
                let line = serde_json::json!({"spec": spec_file, "alpha": alpha, "prompt": p, "reply": r.text, "decode_tok_s": r.stats.decode_tok_s()});
                writeln!(out, "{line}")?;
            }
            eprintln!(
                "[{spec_file} alpha {alpha}: {} prompts, {toks} tokens in {:.1}s]",
                prompts.len(),
                t.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}
