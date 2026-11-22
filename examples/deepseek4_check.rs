//! Compares a DeepSeek V4, V4.1 or ERNIE 5.5 checkpoint run by oxinfer with reference outputs (as
//! written by `tools/ref_deepseek4.py` / `tools/ref_deepseek41.py` / `tools/ref_ernie55.py`: `ids`
//! (I64), `logits` `[n, vocab]` at
//! every prompt position, and optionally `gen` (I64, greedy tokens) with `decode_logits`
//! `[len(gen), vocab]`, the logits after feeding each of them one at a time; for V4.1 also
//! `image_at` / `image_embeds`, an image span of the prompt as the reference embeds it; for ERNIE
//! 5.5 also `mtp_logits` `[n - 1, vocab]`, the MTP layer over the prompt, and `draft` /
//! `draft_logits`, a chain of its drafts after the prompt):
//!
//! ```text
//! deepseek4_check MODEL REF.safetensors [--chunk N] [--cpu] [--mmap]
//! deepseek4_check MODEL --load [TEXT...]
//! deepseek4_check MODEL --greedy TEXT|@IDS_FILE N_GEN OUT.bin [--cpu] [--no-stream] [--ids-out FILE]
//! ```
//!
//! `--greedy` is the counterpart of `tools/llama_logits_dump.cpp` on a real checkpoint (same
//! output, compared by `qwen_greedy_dump --compare`): the prompt (raw text, tokenized without a
//! template, or `@file` of whitespace-separated ids) prefilled as the CLI does, then `N_GEN`
//! greedy tokens one at a time, writing the f32 logits after the prompt and after each of them.
//! A checkpoint larger than most of RAM is streamed as the CLI streams it.
//!
//! `--ppl TEXT_FILE N_CTX N_CHUNKS` (GGUF) is `llama-perplexity -c N_CTX --chunks N_CHUNKS` on the
//! same text: the text tokenized after a BOS, cut into chunks of `N_CTX` tokens, each run from an
//! empty context with its first token replaced by BOS, the second half of each chunk scored.
//!
//! Prints, for the prefill (in chunks of N tokens, default the whole prompt) and for the
//! token-by-token decode, the largest logit difference, the mean KL divergence and the agreement
//! of the top token. With an MTP layer (ERNIE 5.5), `--mtp N` then decodes N tokens greedily with
//! its drafts (`--drafts K` per round, default 3) and compares them with plain greedy decoding.

use anyhow::{Context, Result};
use oxinfer::deepseek4::DeepSeek4Model;
use oxinfer::safetensors::{ModelFiles, SafeTensors};
use std::path::Path;

fn i64s(st: &SafeTensors, name: &str) -> Result<Vec<u32>> {
    Ok(st
        .bytes(name)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c) as u32)
        .collect())
}

fn f32s(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    Ok(st.bytes(name)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// Max |a - b|, KL(softmax(b) || softmax(a)), whether the argmaxes agree.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64, bool) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(a), lse(b));
    let kl = b
        .iter()
        .zip(a)
        .map(|(&y, &x)| {
            let pb = (y as f64 - lb).exp();
            pb * ((y as f64 - lb) - (x as f64 - la))
        })
        .sum::<f64>();
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    (max, kl, am(a) == am(b))
}

fn report(what: &str, ours: &[f32], refs: &[f32], vocab: usize) {
    let n = refs.len() / vocab;
    let (mut max, mut kl, mut same, mut worst) = (0f32, 0f64, 0, 0);
    for i in 0..n {
        let (m, k, s) = compare(&ours[i * vocab..(i + 1) * vocab], &refs[i * vocab..(i + 1) * vocab]);
        if m > max {
            (max, worst) = (m, i);
        }
        if m > 0.05 && std::env::var_os("OXINFER_CHECK_VERBOSE").is_some() {
            eprintln!("{what} {i}: max logit diff {m:.3e}");
        }
        kl += k;
        same += s as usize;
    }
    println!(
        "{what}: {n} positions, max logit diff {max:.3e} (at {worst}), mean KL {:.3e}, same top-1 {same}/{n}",
        kl / n as f64
    );
}

/// `--ppl`: see the module documentation.
fn perplexity(m: &DeepSeek4Model, dir: &Path, text: &str, n_ctx: usize, n_chunks: usize) -> Result<()> {
    let hd = oxinfer::gguf::header(dir)?;
    let bos = hd
        .get("tokenizer.ggml.bos_token_id")
        .and_then(oxinfer::gguf::Value::as_u64)
        .context("no BOS token id in the GGUF header")? as u32;
    let tok = hd.tokenizer()?;
    let enc = tok
        .encode(std::fs::read_to_string(text)?.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let mut ids = vec![bos];
    ids.extend_from_slice(enc.get_ids());
    let n_chunks = n_chunks.min(ids.len() / n_ctx);
    let (h, vocab) = (m.cfg.hidden_size, m.cfg.vocab_size);
    let first = n_ctx / 2;
    let (mut nll, mut count) = (0f64, 0usize);
    let t0 = std::time::Instant::now();
    for ci in 0..n_chunks {
        let mut chunk = ids[ci * n_ctx..(ci + 1) * n_ctx].to_vec();
        chunk[0] = bos;
        let mut sess = m.new_session();
        let mut hs = Vec::with_capacity(n_ctx * h);
        for piece in chunk.chunks(256) {
            hs.extend(m.forward_hidden(&mut sess, piece));
        }
        let logits = m.logits(&hs[first * h..], n_ctx - first);
        for j in first..n_ctx - 1 {
            let row = &logits[(j - first) * vocab..(j - first + 1) * vocab];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + row.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
            nll += lse - row[chunk[j + 1] as usize] as f64;
            count += 1;
        }
        println!(
            "[{}] {:.4}  ({:.1}s)",
            ci + 1,
            (nll / count as f64).exp(),
            t0.elapsed().as_secs_f64()
        );
    }
    println!(
        "perplexity over {n_chunks} chunks of {n_ctx} ({} of {} tokens): {:.4} (mean NLL {:.4})",
        n_chunks * n_ctx,
        ids.len(),
        (nll / count as f64).exp(),
        nll / count as f64
    );
    Ok(())
}

/// `--greedy`: see the module documentation.
fn greedy_dump(m: &DeepSeek4Model, dir: &Path, prompt: &str, n_gen: usize, out: &str, ids_out: Option<&str>) -> Result<()> {
    let ids: Vec<u32> = match prompt.strip_prefix('@') {
        Some(f) => std::fs::read_to_string(f)?
            .split_whitespace()
            .map(|v| v.parse::<u32>())
            .collect::<Result<_, _>>()?,
        None => {
            let tok = oxinfer::gguf::header(dir)?.tokenizer()?;
            let enc = tok.encode(prompt, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            enc.get_ids().to_vec()
        }
    };
    if let Some(f) = ids_out {
        std::fs::write(f, ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "))?;
    }
    let h = m.cfg.hidden_size;
    let mut sess = m.new_session();
    let t0 = std::time::Instant::now();
    let hs = oxinfer::generate::prefill(m, &mut sess, &ids, 256, &[]);
    let mut logits = m.logits(&hs[hs.len() - h..], 1);
    eprintln!("[prefill {} tokens: {:.2}s]", ids.len(), t0.elapsed().as_secs_f64());
    let mut bytes = Vec::new();
    let mut chosen = Vec::new();
    let t1 = std::time::Instant::now();
    for step in 0..=n_gen {
        bytes.extend(logits.iter().flat_map(|v| v.to_le_bytes()));
        let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        chosen.push(best);
        if step < n_gen {
            logits = m.forward(&mut sess, &[best]);
        }
    }
    eprintln!("[decode {n_gen} tokens: {:.2}s]", t1.elapsed().as_secs_f64());
    std::fs::write(out, bytes)?;
    println!("{}", chosen.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: deepseek4_check MODEL REF.safetensors [--chunk N] [--cpu] [--mmap]";
    let model = args.next().context(usage)?;
    let refp = args.next().context(usage)?;
    let (mut chunk, mut cpu, mut mmap) = (None, false, false);
    let (mut mtp_n, mut drafts) = (0usize, 3usize);
    let mut texts = Vec::new();
    let ppl = if refp == "--ppl" {
        let g = "--ppl TEXT_FILE N_CTX N_CHUNKS";
        Some((
            args.next().context(g)?,
            args.next().context(g)?.parse::<usize>()?,
            args.next().context(g)?.parse::<usize>()?,
        ))
    } else {
        None
    };
    let greedy = if refp == "--greedy" {
        let g = "--greedy TEXT|@IDS_FILE N_GEN OUT.bin";
        Some((
            args.next().context(g)?,
            args.next().context(g)?.parse::<usize>()?,
            args.next().context(g)?,
        ))
    } else {
        None
    };
    let (mut no_stream, mut ids_out) = (false, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--no-stream" => no_stream = true,
            "--ids-out" => ids_out = Some(args.next().context("--ids-out FILE")?),
            "--chunk" => chunk = Some(args.next().context("--chunk N")?.parse::<usize>()?),
            "--cpu" => cpu = true,
            "--mtp" => mtp_n = args.next().context("--mtp N")?.parse()?,
            "--drafts" => drafts = args.next().context("--drafts K")?.parse()?,
            "--mmap" => mmap = true,
            other if refp == "--load" => texts.push(other.to_string()),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    mmap |= refp == "--load" || greedy.is_some() || ppl.is_some();
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let cfg = oxinfer::config::DeepSeek4Config::load(&dir)?;
    let files = ModelFiles::open(&dir, !mmap)?;
    let mut m = DeepSeek4Model::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
    if greedy.is_some() || ppl.is_some() {
        let size = oxinfer::safetensors::checkpoint_bytes(&dir);
        let streamer = if oxinfer::deepseek4::too_big(size) && !no_stream {
            let s = oxinfer::stream::attach_deepseek4(&mut m, &files, oxinfer::deepseek4::default_stream().with_env())?;
            let p = s.plan();
            eprintln!(
                "[stream: {:.2} GB resident, {:.2} GB streamed per pass]",
                p.resident_bytes as f64 / 1e9,
                p.streamed_bytes as f64 / 1e9
            );
            Some(s)
        } else {
            None
        };
        let r = match (greedy, ppl) {
            (Some((prompt, n_gen, out)), _) => greedy_dump(&m, &dir, &prompt, n_gen, &out, ids_out.as_deref()),
            (_, Some((text, n_ctx, n_chunks))) => perplexity(&m, &dir, &text, n_ctx, n_chunks),
            _ => unreachable!(),
        };
        m.set_stream(None);
        drop(streamer);
        return r;
    }
    // `--load` in place of the reference: only open the checkpoint (mapped), to try the loader
    // on the layout of a real file (even one whose tensor data was not downloaded), and print
    // the ids its tokenizer gives the texts that follow.
    if refp == "--load" {
        let c = &m.cfg;
        println!(
            "loaded: {} layers of {}, {} experts (top {}), engram layers {:?}, key sources {:?}, index sources {:?}, {:.2} GB read per token, {} tensors, {:.1} GB",
            c.num_hidden_layers,
            c.hidden_size,
            c.n_routed_experts,
            c.num_experts_per_tok,
            c.engram_layer_ids,
            c.kv_source_layer_ids,
            c.index_source_layer_ids,
            m.streamed_bytes() as f64 / 1e9,
            files.names().count(),
            files.total_bytes() as f64 / 1e9
        );
        if c.ernie.is_some() {
            println!("ERNIE 5.5: MTP layer {}", if m.has_mtp() { "loaded" } else { "absent" });
        }
        if dir.join("tokenizer.json").exists() {
            let tok =
                oxinfer::tokenizer::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
            for text in &texts {
                let enc = tok.encode(text.as_str(), true).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
                println!("{:?}", enc.get_ids());
            }
        }
        if oxinfer::gguf::is_gguf(&dir) {
            let tok = oxinfer::gguf::header(&dir)?.tokenizer()?;
            for text in texts {
                let enc = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
                println!("{:?}", enc.get_ids());
            }
        }
        return Ok(());
    }
    let st = SafeTensors::open(Path::new(&refp), true)?;
    let ids = i64s(&st, "ids")?;
    let refs = f32s(&st, "logits")?;
    let vocab = m.cfg.vocab_size;
    let chunk = chunk.unwrap_or(ids.len()).max(1);

    // DeepSeek V4.1: an image span of the prompt, as the reference's embeddings of its positions.
    let h = m.cfg.hidden_size;
    let (img_at, img) = if st.info("image_embeds").is_some() {
        (i64s(&st, "image_at")?[0] as usize, f32s(&st, "image_embeds")?)
    } else {
        (0, Vec::new())
    };
    let mut sess = m.new_session();
    let mut ours = Vec::with_capacity(refs.len());
    // ERNIE 5.5: the residual before the head at every position, for the MTP layer.
    let mtp = m.has_mtp() && st.info("mtp_logits").is_some();
    let mut resid = Vec::new();
    let t0 = std::time::Instant::now();
    for (ci, c) in ids.chunks(chunk).enumerate() {
        if mtp {
            let mut r = Vec::new();
            let hs = m.forward_residual(&mut sess, c, &mut r);
            resid.extend(r);
            ours.extend(m.logits(&hs, c.len()));
            continue;
        }
        let start = ci * chunk;
        let mm: Vec<(usize, &[f32])> = img
            .chunks(h)
            .enumerate()
            .map(|(i, row)| (img_at + i, row))
            .filter(|(p, _)| (start..start + c.len()).contains(p))
            .map(|(p, row)| (p - start, row))
            .collect();
        let hs = m.forward_capture_mm(&mut sess, c, &mm, &[], &mut Vec::new());
        ours.extend(m.logits(&hs, c.len()));
    }
    eprintln!(
        "[prefill {} tokens in chunks of {chunk}: {:.2}s]",
        ids.len(),
        t0.elapsed().as_secs_f64()
    );
    report("prefill", &ours, &refs, vocab);

    if mtp {
        // The MTP layer over the prompt (in the same chunks), then a round of drafts after it.
        let n = ids.len();
        let w = resid.len() / n;
        let mut ms = m.mtp_session().unwrap();
        let (mut got, mut last) = (Vec::new(), Vec::new());
        for c in (0..n - 1).collect::<Vec<_>>().chunks(chunk) {
            let (a, b) = (c[0], c[c.len() - 1] + 1);
            let hs = m.mtp_forward(&mut ms, &resid[a * w..b * w], &ids[a + 1..b + 1], &mut last);
            got.extend(m.logits(&hs, b - a));
        }
        report("mtp (prompt)", &got, &f32s(&st, "mtp_logits")?, vocab);
        let dref = f32s(&st, "draft_logits")?;
        let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
        let mut tok = am(&ours[(n - 1) * vocab..]);
        let mut r = resid[(n - 1) * w..].to_vec();
        let mut got = Vec::new();
        for _ in 0..dref.len() / vocab {
            let hs = m.mtp_forward(&mut ms, &r, &[tok], &mut last);
            let lg = m.logits(&hs, 1);
            tok = am(&lg);
            got.extend(lg);
            r = std::mem::take(&mut last);
        }
        report("mtp (drafts)", &got, &dref, vocab);
    }

    if st.info("gen").is_some() {
        let gen_ids = i64s(&st, "gen")?;
        let dref = f32s(&st, "decode_logits")?;
        let mut got = Vec::with_capacity(dref.len());
        for &tk in &gen_ids {
            got.extend(m.forward(&mut sess, &[tk]));
        }
        report("decode", &got, &dref, vocab);
    }
    if mtp_n > 0 {
        // Greedy decoding after the prompt, plain and with MTP drafts.
        let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
        let mut sess = m.new_session();
        let t0 = std::time::Instant::now();
        let mut plain = vec![am(&m.forward(&mut sess, &ids))];
        while plain.len() < mtp_n {
            let lg = m.forward(&mut sess, &plain[plain.len() - 1..]);
            plain.push(am(&lg));
        }
        let tp = t0.elapsed().as_secs_f64();
        let mut dec = oxinfer::deepseek4::ErnieMtpDecoder::new(&m, drafts)?;
        let t0 = std::time::Instant::now();
        let mut spec = vec![dec.prefill(&ids)];
        while spec.len() < mtp_n {
            spec.extend(dec.step());
        }
        spec.truncate(mtp_n);
        let same = plain.iter().zip(&spec).take_while(|(a, b)| a == b).count();
        let (rounds, drafted, kept) = dec.stats;
        println!(
            "mtp decode: {mtp_n} tokens, {same} match plain greedy decoding; {rounds} rounds, {kept}/{drafted} drafts kept; plain {tp:.2}s, with drafts {:.2}s",
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
