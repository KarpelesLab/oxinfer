//! Measures how much of a conversation the engine reuses from one turn to the next: a first turn
//! on a long prompt, then a second turn that sends the conversation back with the reply as a chat
//! client returns it (without its thinking), as `Engine::chat` sees it from a server.
//!
//! ```sh
//! cargo run --release --example chat_turns -- MODEL PROMPT_FILE [--turns 1|2|second] [--think] \
//!     [--max-tokens N] [--save-tokens PREFIX] [key=value ...]
//! ```
//!
//! `key=value` are engine options (`prompt_checkpoints=0`, `prompt_cache=DIR`, ...). `--turns
//! second` runs only the second turn's request (its first reply is read from `PROMPT_FILE.reply`,
//! written by a run of both turns), to measure a restore from the prompt cache in a new process.
//! Each turn prints its prompt, cached and restored tokens, prefill and wall time, and a hash of
//! the reply's tokens (`--save-tokens`: the tokens themselves, to `PREFIX.turnN`).

use anyhow::{Context, Result, bail};
use oxinfer::engine::{Delta, Engine, EngineOptions, Flow, Message};
use std::time::Instant;

fn quiet(_: &Delta) -> Flow {
    Flow::Continue
}

fn main() -> Result<()> {
    let usage =
        "usage: chat_turns MODEL PROMPT_FILE [--turns 1|2|second] [--think] [--max-tokens N] [--save-tokens PREFIX] [key=value ...]";
    let mut args = std::env::args().skip(1);
    let model = args.next().context(usage)?;
    let prompt_file = args.next().context(usage)?;
    let prompt = std::fs::read_to_string(&prompt_file)?;
    let (mut turns, mut think, mut max_tokens) = ("2".to_string(), false, 64usize);
    let mut save_tokens = None::<String>;
    let mut opts = EngineOptions::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--turns" => turns = args.next().context("--turns 1|2|second")?,
            "--think" => think = true,
            "--max-tokens" => max_tokens = args.next().context("--max-tokens N")?.parse()?,
            "--save-tokens" => save_tokens = Some(args.next().context("--save-tokens PREFIX")?),
            kv => {
                let (k, v) = kv.split_once('=').with_context(|| format!("unknown argument {kv}"))?;
                opts.set(k, v)?;
            }
        }
    }
    let t = Instant::now();
    let mut e = Engine::load(&model, &opts)?;
    eprintln!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    e.config_mut().max_tokens = max_tokens;
    e.config_mut().think = think;
    let mut convo = vec![Message::system("You are a helpful assistant."), Message::user(prompt)];
    let reply_file = format!("{prompt_file}.reply");
    let turn = |e: &mut Engine, convo: &[Message], n: usize| -> Result<String> {
        let t = Instant::now();
        let r = e.chat(convo, &mut quiet)?;
        let wall = t.elapsed().as_secs_f64();
        let s = &r.stats;
        let h = oxinfer::snapshot::hash64(&[&r.tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>()]);
        println!(
            "turn {n}: {} prompt tokens ({} cached, {} restored in {:.2} s), prefill {:.2} s, {} reply tokens, wall {:.2} s, reply {h:016x}",
            s.prompt_tokens + s.cached_tokens,
            s.cached_tokens,
            s.restored_tokens,
            s.restore_secs,
            s.prefill_secs,
            r.tokens.len(),
            wall
        );
        if let Some(p) = &save_tokens {
            let line: Vec<String> = r.tokens.iter().map(|t| t.to_string()).collect();
            std::fs::write(format!("{p}.turn{n}"), line.join(" "))?;
        }
        Ok(r.text)
    };
    let first = match turns.as_str() {
        "second" => std::fs::read_to_string(&reply_file).with_context(|| reply_file.clone())?,
        "1" | "2" => {
            let text = turn(&mut e, &convo, 1)?;
            std::fs::write(&reply_file, &text)?;
            text
        }
        t => bail!("--turns {t}: 1, 2 or second"),
    };
    if turns != "1" {
        convo.push(Message::assistant(first));
        convo.push(Message::user("Summarize that in one sentence."));
        turn(&mut e, &convo, 2)?;
    }
    e.flush_prompt_cache()?;
    Ok(())
}
