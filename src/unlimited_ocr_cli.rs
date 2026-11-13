//! `oxinfer --model baidu/Unlimited-OCR-2 --image page.png`: document parsing through
//! `oxinfer::unlimited_ocr`. One `--image` per page, in order; `-p` replaces the instruction
//! (`Multi page parsing.`). The reply is printed with its special tokens (`<PAGE>` before each
//! page, the layout tags), as the reference returns it.

use crate::Args;
use anyhow::{Result, bail};
use oxinfer::generate::{self, Flow};
use oxinfer::sampler::Sampler;
use oxinfer::unlimited_ocr::{OcrChat, OcrConfig, OcrGuard, OcrModel, prompt_text};
use std::io::{Read, Write};
use std::time::Instant;

/// Unlimited-OCR flags.
#[derive(Default, Debug)]
pub struct OcrArgs {
    /// `--no-repeat-ngram N`: no n-gram of N generated tokens twice within the window.
    pub ngram: usize,
    /// `--ngram-window N`: the last tokens searched (default 100).
    pub window: usize,
    /// `--max-page-tokens N`: tokens of one page before the next is forced (default 20000).
    pub max_page_tokens: Option<usize>,
    /// `--no-page-guard`: do not hold the reply to the prompt's page count.
    pub no_guard: bool,
}

impl OcrArgs {
    /// Takes `flag` (with its value from `val`) if it is one of these.
    pub fn parse(&mut self, flag: &str, val: &mut dyn FnMut() -> Result<String>) -> Result<bool> {
        match flag {
            "--no-repeat-ngram" => self.ngram = val()?.parse()?,
            "--ngram-window" => self.window = val()?.parse()?,
            "--max-page-tokens" => self.max_page_tokens = Some(val()?.parse()?),
            "--no-page-guard" => self.no_guard = true,
            _ => return Ok(false),
        }
        Ok(true)
    }
}

pub fn run(args: &Args) -> Result<()> {
    let t0 = Instant::now();
    if args.mtp || args.draft.is_some() {
        bail!("Unlimited-OCR has no speculative decoding here");
    }
    if args.chat {
        bail!("Unlimited-OCR takes one prompt per run (the pages, then the instruction)");
    }
    let cfg = OcrConfig::load(&args.model)?;
    let files = oxinfer::safetensors::ModelFiles::open(&args.model, !args.mmap)?;
    let model = OcrModel::load(cfg, &files, args.quant)?;
    let tok = oxinfer::engine::load_tokenizer(&args.model)?;
    let c = &model.cfg;
    eprintln!(
        "[loaded Unlimited-OCR: {} layers, hidden {}, {:.1} GB checkpoint, {:?} weights, {} threads, {:.2}s]",
        c.num_hidden_layers,
        c.hidden_size,
        files.total_bytes() as f64 / 1e9,
        model.quant,
        oxinfer::pool::threads(),
        t0.elapsed().as_secs_f64()
    );
    let (_, feats) = crate::mm_for(args, &files, c.hidden_size)?;
    let counts: Vec<usize> = feats.iter().map(|f| f.n).collect();
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None if !counts.is_empty() => String::new(),
        None => {
            let mut s = String::new();
            std::io::stdin().lock().read_to_string(&mut s)?;
            s
        }
    };
    let text = if args.raw {
        prompt
    } else {
        use oxinfer::chat::ChatFormat;
        OcrChat.conversation_start(None, false) + &prompt_text(&prompt, &counts)?
    };
    if args.print_prompt {
        print!("{text}");
        return Ok(());
    }
    let ids = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    if args.show_tokens {
        eprintln!("[prompt ids ({}): {ids:?}]", ids.len());
    }
    if ids.is_empty() {
        bail!("empty prompt");
    }
    let mut sess = model.new_session();
    let overrides = oxinfer::mm::bind(&model, &mut sess, &ids, &feats)?;
    let t = Instant::now();
    let hs = generate::prefill(&model, &mut sess, &ids, usize::MAX, &overrides);
    let logits = model.logits(&hs[hs.len() - c.hidden_size..], 1);
    let prefill_secs = t.elapsed().as_secs_f64();
    if let Some(path) = &args.dump_logits {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let top: Vec<String> = idx[..10].iter().map(|&i| format!("{i}:{:.3}", logits[i])).collect();
        eprintln!("[wrote {} logits to {} | top10 {}]", logits.len(), path.display(), top.join(" "));
        eprintln!("[prefill {} tok in {:.2}s]", ids.len(), prefill_secs);
        return Ok(());
    }
    let mut guard = OcrGuard::new(c, &ids).with_ngram(args.ocr.ngram, args.ocr.window);
    if args.ocr.no_guard {
        guard.expected_pages = 0;
    }
    if let Some(n) = args.ocr.max_page_tokens {
        guard.max_page_tokens = n;
    }
    // The README's serving recipe budgets 8000 tokens per page.
    let pages = guard.expected_pages.max(1);
    let max_tokens = args.max_tokens_set.unwrap_or(8000 * pages);
    let mut sampler =
        Sampler::new(args.temperature, args.top_k, args.top_p, args.seed).with_repetition_penalty(args.repetition_penalty.unwrap_or(1.0));
    sampler.set_guide(Some(Box::new(guard)), None);
    let stop = [c.eos_token_id];
    let mut printed = 0;
    let mut show = |out: &[u32], last: bool| {
        // Special tokens kept; an incomplete character waits for its last bytes.
        let Ok(text) = tok.decode(out, false) else { return };
        let stable = if !last && text.ends_with('\u{FFFD}') {
            text.len().saturating_sub(3)
        } else {
            text.len()
        };
        if stable > printed && text.is_char_boundary(stable) {
            let mut lock = std::io::stdout().lock();
            let _ = lock.write_all(&text.as_bytes()[printed..stable]);
            let _ = lock.flush();
            printed = stable;
        }
    };
    let d = generate::decode(&model, &mut sess, logits, &mut sampler, &stop, max_tokens, &mut |out| {
        show(out, false);
        Flow::Continue
    });
    let out = match d.tokens.split_last() {
        Some((last, rest)) if stop.contains(last) => rest,
        _ => &d.tokens[..],
    };
    show(out, true);
    println!();
    if args.show_tokens {
        eprintln!("[output ids: {:?}]", d.tokens);
    }
    crate::report(ids.len(), prefill_secs, d.forwards, d.secs);
    Ok(())
}
