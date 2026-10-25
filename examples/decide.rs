//! Typed decisions (Jev models) and chat from one engine, through the library:
//! `cargo run --release --example decide -- MODEL [DECISION_FROM] [OPTION=VALUE ...]`
//!
//! MODEL is a Jev bundle (e.g. autotrust/JEV-9B) or a base model, with DECISION_FROM the bundle
//! whose decision assets to attach ("-" for none). Asks a question, then a decision, then the
//! same question again: the two replies are identical (the adapter is only on during the
//! decision), and their token ids are printed for comparison with a run without assets.

use oxinfer::decide::DecisionRequest;
use oxinfer::engine::{Engine, EngineOptions, Flow, SystemPrompt};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: decide MODEL [DECISION_FROM|-] [OPTION=VALUE ...]";
    let model = args.next().expect(usage);
    let mut opts = EngineOptions::default();
    if let Some(from) = args.next()
        && from != "-"
    {
        opts.set("decision_from", &from)?;
    }
    for kv in args {
        let (k, v) = kv.split_once('=').expect(usage);
        opts.set(k, v)?;
    }
    let mut e = Engine::load(&model, &opts)?;
    e.config_mut().max_tokens = 64;
    // The default system prompt has the time of day: none, so that runs compare.
    e.config_mut().system = SystemPrompt::None;
    let chat = |e: &mut Engine| -> anyhow::Result<Vec<u32>> {
        e.reset();
        let r = e.send("In two sentences, what is safety stock?", &mut |_| Flow::Continue)?;
        println!("chat: {}", r.text.replace('\n', " "));
        Ok(r.tokens)
    };
    let before = chat(&mut e)?;
    println!("tokens: {before:?}");
    if e.decision_assets().is_some() {
        let d = e.decide(&DecisionRequest::choice(
            "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
            "Supplier response for this scenario.",
            ["issue_warning", "renegotiate", "dual_source", "maintain"],
        ))?;
        for (o, p) in d.options.iter().zip(&d.probabilities) {
            println!("decision: {o:<14} {p:.4}");
        }
        println!(
            "decision: {} prompt tokens in {:.3}s ({:.0} tok/s)",
            d.stats.prompt_tokens,
            d.stats.prefill_secs,
            d.stats.prefill_tok_s()
        );
        let after = chat(&mut e)?;
        assert_eq!(before, after, "chat changed after a decision");
        println!("chat after the decision: identical");
    }
    Ok(())
}
