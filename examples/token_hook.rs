//! The token hook ([`Engine::set_token_hook`]) end to end: when the reply's text first ends a
//! sentence, the hook makes the model read a note of its own, then sampling resumes; a second
//! turn asks about the note (it is in the conversation), and the forced tokens are checked in the
//! reply. `cargo run --release --example token_hook -- MODEL [key=value ...]` (EngineOptions keys).

use oxinfer::engine::{Engine, EngineOptions, Flow, SamplingParams};
use std::sync::{Arc, Mutex};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: token_hook MODEL [key=value ...]");
    let mut opts = EngineOptions::default();
    for kv in args {
        let (k, v) = kv.split_once('=').expect("options are key=value");
        opts.set(k, v)?;
    }
    let mut engine = Engine::load(&model, &opts)?;
    // Greedy.
    engine.set_sampling(SamplingParams::default());
    engine.config_mut().think = false;
    // Not the default prompt with the time: the same reply on every run.
    engine.config_mut().system = oxinfer::engine::SystemPrompt::Text("You are a helpful assistant.".into());
    engine.reset();
    engine.config_mut().max_tokens = 300;
    let tok = engine.tokenizer().clone();
    let note = " (Receipt: the verification code is 4471-KX.)";
    let forced = tok.encode(note, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let fired = Arc::new(Mutex::new(None::<usize>));
    let f2 = fired.clone();
    engine.set_token_hook(Some(Box::new(move |reply: &[u32]| {
        let mut f = f2.lock().unwrap();
        if f.is_some() {
            return Vec::new();
        }
        let text = tok.decode(reply, true).unwrap_or_default();
        if text.trim_end().ends_with('.') {
            *f = Some(reply.len());
            return forced.clone();
        }
        Vec::new()
    })));
    let mut shown = String::new();
    let reply = engine.send("Describe the moon in three sentences.", &mut |d| {
        shown.push_str(d.text);
        Flow::Continue
    })?;
    // (Models that always think may have read it in their thinking.)
    let all = format!("{}{}", reply.thinking, reply.text);
    println!("reply: {all}");
    let at = fired.lock().unwrap().expect("the hook never fired");
    let want = engine
        .tokenizer()
        .encode(note, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    assert_eq!(
        &reply.tokens[at..at + want.len()],
        &want[..],
        "the forced tokens follow the trigger"
    );
    assert!(all.contains("4471-KX"), "the note is in the reply");
    assert_eq!(shown, all, "streamed as returned");
    println!(
        "[hook fired after token {at}; {} forced tokens; {} tokens in all, {:.1} tok/s, {:.2} tok/forward]",
        want.len(),
        reply.tokens.len(),
        reply.stats.decode_tok_s(),
        reply.stats.tokens_per_forward()
    );
    engine.set_token_hook(None);
    engine.config_mut().max_tokens = 400;
    let r2 = engine.send(
        "What was the verification code in your previous answer? Reply with just the code.",
        &mut |_| Flow::Continue,
    )?;
    let all2 = format!("{}{}", r2.thinking, r2.text);
    println!("follow-up: {}", all2.trim());
    assert!(all2.contains("4471"), "the model read the forced note");
    println!("OK");
    Ok(())
}
