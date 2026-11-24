//! llama-bench style throughput through the `engine` API, warm (one warm-up pass first, as
//! llama-bench does): prompt processing of exactly N tokens, the BOS included as llama-bench
//! counts it (raw text cut from `tools/bench/corpus.txt`), and plain decoding of G tokens after
//! a short prompt, best and median of R repetitions:
//! `cargo run --release --example parity_bench -- MODEL [pp=512,2048] [tg=128] [reps=3] [key=value ...]`
//! (other options are `EngineOptions` keys, e.g. `device=cpu`, `lookup=0`, `draft=none`;
//! `chain_ab=1` measures decoding with the greedy chain off and on, run by run alternated;
//! `depth=8192,32768` measures decoding of G tokens after a prompt of each length instead;
//! `text=FILE` takes the prompts from FILE rather than the corpus).

use oxinfer::engine::{Engine, EngineOptions, Flow};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: parity_bench MODEL [pp=..] [tg=N] [reps=N] [key=value ...]");
    let mut opts = EngineOptions::default();
    let (mut pps, mut tg, mut reps) = (vec![512usize, 2048], 128usize, 3usize);
    let mut depths: Vec<usize> = Vec::new();
    let mut chain_ab = false;
    let mut text: Option<String> = None;
    for kv in args {
        let (k, v) = kv.split_once('=').expect("options are key=value");
        match k {
            "pp" => {
                pps = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?
            }
            "tg" => tg = v.parse()?,
            "depth" => {
                depths = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?
            }
            "reps" => reps = v.parse()?,
            "chain_ab" => chain_ab = v == "1",
            "text" => text = Some(v.to_string()),
            _ => opts.set(k, v)?,
        }
    }
    let mut e = Engine::load(&model, &opts)?;
    let i = e.info();
    eprintln!(
        "[{} ({:?}, {:?}) on {} with {} threads]",
        i.model_type, i.family, i.quant, i.device, i.threads
    );
    let corpus = std::fs::read_to_string(
        text.as_deref()
            .unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/tools/bench/corpus.txt")),
    )?;
    // Long prompts repeat the corpus (llama-bench's prompts are random tokens; the dense models'
    // speed depends only on the length).
    let longest = pps.iter().chain(&depths).copied().max().unwrap_or(0);
    let mut ids = e.tokenizer().encode(corpus.as_str(), false)?.get_ids().to_vec();
    let copies = longest / ids.len().max(1) + 1;
    if copies > 1 {
        let text = vec![corpus.as_str(); copies].join("\n\n");
        ids = e.tokenizer().encode(text.as_str(), false)?.get_ids().to_vec();
    }
    // Text of exactly n tokens (re-encoded: trim until it is).
    let text_of = |e: &Engine, n: usize| -> anyhow::Result<String> {
        let mut k = n.min(ids.len());
        loop {
            let s = e.tokenizer().decode(&ids[..k], false)?;
            let c = e.tokenizer().encode(s.as_str(), false)?.get_ids().len();
            if c <= n || k == 0 {
                return Ok(s);
            }
            k -= c - n;
        }
    };
    let run = |e: &mut Engine, prompt: &str, max: usize| -> anyhow::Result<(usize, f64, usize, f64)> {
        e.reset();
        e.config_mut().max_tokens = max;
        e.complete(prompt, &mut |_| Flow::Continue)?;
        let s = e.stats();
        Ok((s.prompt_tokens, s.prefill_tok_s(), s.generated_tokens, s.decode_tok_s()))
    };
    let warm = text_of(&e, 64)?;
    run(&mut e, &warm, 4)?;
    let stat = |mut v: Vec<f64>| {
        v.sort_by(|a, b| b.total_cmp(a));
        format!(
            "best {:.1} median {:.1} (all {:?})",
            v[0],
            v[v.len() / 2],
            v.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>()
        )
    };
    for &n in &pps {
        // Exactly n prompt tokens with the BOS the model adds, as llama-bench counts them.
        let mut p = text_of(&e, n)?;
        let (got, ..) = run(&mut e, &p, 1)?;
        if got > n {
            p = text_of(&e, n.saturating_sub(got - n))?;
        }
        // A warm-up at this size (plans and scratch are sized on first use).
        run(&mut e, &p, 1)?;
        let mut v = Vec::new();
        let mut toks = 0;
        // (A profile covers the measured passes only.)
        oxinfer::gpu::pf_reset();
        for _ in 0..reps {
            let (t, r, _, _) = run(&mut e, &p, 1)?;
            toks = t;
            v.push(r);
        }
        println!("pp{n} ({toks} tok): {}", stat(v));
    }
    // Decoding at depth (llama-bench `-d D -n G`): G tokens after a prompt of D tokens.
    for &d in &depths {
        let p = text_of(&e, d)?;
        run(&mut e, &p, 2)?;
        let mut v = Vec::new();
        let mut g = 0;
        for _ in 0..reps {
            let (_, _, n, r) = run(&mut e, &p, tg)?;
            g = n;
            v.push(r);
        }
        println!("tg{tg} @ d{d} ({g} tok): {}", stat(v));
    }
    if !depths.is_empty() {
        // (depth runs replace the short-prompt decode)
    } else if tg > 0 && chain_ab {
        let p = text_of(&e, 16)?;
        let (mut off, mut on) = (Vec::new(), Vec::new());
        let mut g = 0;
        for _ in 0..reps {
            for chain in [false, true] {
                oxinfer::gpu::set_decode_chain(Some(chain));
                let (_, _, n, r) = run(&mut e, &p, tg)?;
                g = n;
                if chain { on.push(r) } else { off.push(r) }
            }
        }
        oxinfer::gpu::set_decode_chain(None);
        println!("tg{tg} ({g} tok) chain off: {}", stat(off));
        println!("tg{tg} ({g} tok) chain on: {}", stat(on));
    } else if tg > 0 {
        let p = text_of(&e, 16)?;
        let mut v = Vec::new();
        let mut g = 0;
        for _ in 0..reps {
            let (_, _, n, r) = run(&mut e, &p, tg)?;
            g = n;
            v.push(r);
        }
        println!("tg{tg} ({g} tok): {}", stat(v));
    }
    if std::env::var_os("OXINFER_PF_PROFILE").is_some() {
        oxinfer::gpu::pf_report();
        // The op-by-op path's launches (copies up, device time, copies down) over the whole run.
        if let Some(s) = oxinfer::gpu::active().and_then(|b| b.stats()) {
            eprintln!("[gpu: {s}]");
        }
    }
    Ok(())
}
