//! Checks a DSpark draft against a reference (`tools/ref_dspark.py` for Mistral Large 4,
//! `tools/ref_dspark_ds4.py` for DeepSeek V4, `tools/ref_dspark_hf.py` for Qwen / Gemma 4) and
//! measures its acceptance:
//!
//! ```text
//! dspark_check MODEL DRAFT PROMPT OUT_DIR [--k N] [--cpu] [--raw] [--gen N] [--draft-device cpu|gpu] [--bench N]
//!     [--quant Q] [--think] [--fast] [--drift]
//! ```
//!
//! Runs the chat-formatted `PROMPT` (as is with `--raw`; `--think` turns thinking on for Qwen
//! and Gemma 4) through the target, takes its greedy next token as the anchor, and writes to
//! `OUT_DIR` the target features of the prompt positions (`feats.f32`), the draft's logits for
//! `k` slots chained greedily (`logits.f32`) and `meta.json`. With `--gen N` it then decodes N
//! tokens greedily with and without DSpark and checks that both give the same tokens (the
//! environment's `DSPARK_P_MIN`, `DSPARK_CONF_MIN` and `DSPARK_FIXED=1` set the DSpark options).
//! `--draft-device` says where the draft's matmuls run (default: the GPU when there is one).
//! `--bench N` times N draft passes and injections of 3 positions on the CPU and on the GPU,
//! alternating. `--fast` runs the DSpark decoding of `--gen` with fast verification
//! (`DsparkOptions::fast_verify`); `--drift` (with `--gen`) then also feeds the plain greedy
//! output through 8-token verification passes, exact and fast (each in its own session, so the
//! fast one keeps its own keys and values as fast decoding would), and reports how the fast
//! logits differ from the exact ones (largest difference, KL divergence, top-1 changes).

use anyhow::{Context, Result, bail};
use oxinfer::chat::ChatFormat;
use oxinfer::dspark::{Backbone, Dspark, DsparkDecoder, DsparkOptions, Target};
use oxinfer::generate::{self, Flow};
use oxinfer::model::Quant;
use oxinfer::safetensors::ModelFiles;
use oxinfer::sampler::{Sampler, argmax};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

fn write_f32(path: &Path, v: &[f32]) -> Result<()> {
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(())
}

fn env_f32(k: &str) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)
}

struct Opts {
    out: PathBuf,
    k: Option<usize>,
    gen_n: usize,
    device: Option<bool>,
    bench: usize,
    stop: Vec<u32>,
    fast: bool,
    drift: bool,
}

/// KL divergence of softmax(b) from softmax(a) (a the reference) and the largest |a - b|.
fn kl_maxdiff(a: &[f32], b: &[f32]) -> (f64, f32) {
    let lse = |v: &[f32]| {
        let m = v.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x)) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(a), lse(b));
    let mut kl = 0f64;
    let mut dmax = 0f32;
    for (&x, &y) in a.iter().zip(b) {
        let pa = (x as f64 - la).exp();
        kl += pa * ((x as f64 - la) - (y as f64 - lb));
        dmax = dmax.max((x - y).abs());
    }
    (kl.max(0.0), dmax)
}

fn check<M: Target<Sess: Clone>>(m: &M, d: &Dspark, tok: &Tokenizer, ids: &[u32], o: &Opts) -> Result<()> {
    // Default: the whole block for Qwen / Gemma 4 drafts, 3 for the others.
    let gqa = matches!(d.cfg.backbone, Backbone::Gqa(_));
    let k = o.k.unwrap_or(if gqa { usize::MAX } else { 3 }).min(d.cfg.max_draft());
    let mut dec = DsparkDecoder::new(
        m,
        d,
        DsparkOptions {
            draft_tokens: k,
            device: o.device,
            // The check compares against one-token forwards bit for bit.
            fast_verify: false,
            ..Default::default()
        },
    );
    let t0 = std::time::Instant::now();
    let logits = dec.prefill(ids, 256);
    let anchor = argmax(&logits);
    eprintln!(
        "[prefill {:.2}s, anchor {anchor} {:?}]",
        t0.elapsed().as_secs_f64(),
        tok.decode(&[anchor], false)
    );
    let (dl, drafts) = d.draft_logits(m, dec.draft_session_mut(), anchor, k);
    eprintln!("[drafts {drafts:?} = {:?}]", tok.decode(&drafts, false));
    // The features of the prompt positions, recomputed (same kernels, so the same values).
    let mut s = m.new_session();
    let mut feats = Vec::new();
    let mut all = Vec::new();
    for c in ids.chunks(256) {
        m.forward_feats(&mut s, c, &[], &d.cfg.target_layers, &mut feats);
        all.extend_from_slice(&feats);
    }
    std::fs::create_dir_all(&o.out)?;
    write_f32(&o.out.join("feats.f32"), &all)?;
    write_f32(&o.out.join("logits.f32"), &dl)?;
    let meta = serde_json::json!({"n": ids.len(), "anchor": anchor, "k": drafts.len(), "drafts": drafts, "ids": ids});
    std::fs::write(o.out.join("meta.json"), meta.to_string())?;

    if o.bench > 0 {
        // Draft passes and injections of the last 3 positions' features, CPU and GPU in turn.
        let nf = all.len() / ids.len();
        let rows = &all[all.len() - 3 * nf..];
        let mut times = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
        for i in 0..2 * o.bench {
            let gpu = i % 2 == 1;
            d.set_device(gpu);
            let mut ds = dec.draft_session().clone();
            let t = std::time::Instant::now();
            let (l, _) = d.draft_logits(m, &mut ds, anchor, k);
            times[gpu as usize][0].push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(l);
            let t = std::time::Instant::now();
            d.inject(&mut ds, rows, 3);
            times[gpu as usize][1].push(t.elapsed().as_secs_f64() * 1e3);
        }
        for (dev, t) in ["cpu", "gpu"].iter().zip(&mut times) {
            for v in t.iter_mut() {
                v.sort_by(f64::total_cmp);
            }
            println!(
                "bench {dev}: draft pass median {:.1} ms (min {:.1}), inject 3 median {:.1} ms (min {:.1})",
                t[0][t[0].len() / 2],
                t[0][0],
                t[1][t[1].len() / 2],
                t[1][0]
            );
        }
        d.set_device(
            o.device
                .unwrap_or(oxinfer::dspark::DEVICE_BY_DEFAULT && oxinfer::gpu::active().is_some()),
        );
    }

    if o.gen_n > 0 {
        let mut plain_s = m.new_session();
        let lg = generate::prefill(m, &mut plain_s, ids, 256, &[]);
        let lg = m.logits(&lg[lg.len() - m.hidden_size()..], 1);
        let mut sp = Sampler::new(0.0, 0, 1.0, 1);
        let t1 = std::time::Instant::now();
        let plain = generate::decode(m, &mut plain_s, lg, &mut sp, &o.stop, o.gen_n, &mut |_| Flow::Continue);
        let plain_secs = t1.elapsed().as_secs_f64();
        let mut dec = DsparkDecoder::new(
            m,
            d,
            DsparkOptions {
                draft_tokens: k,
                stop: o.stop.clone(),
                device: o.device,
                p_min: env_f32("DSPARK_P_MIN"),
                conf_min: env_f32("DSPARK_CONF_MIN"),
                cost_aware: gqa && std::env::var_os("DSPARK_FIXED").is_none(),
                fast_verify: o.fast,
            },
        );
        let lg = dec.prefill(ids, 256);
        let t2 = std::time::Instant::now();
        let spec = generate::decode_rounds(&mut dec, &lg, &mut sp, &o.stop, o.gen_n, &mut |_| Flow::Continue);
        let spec_secs = t2.elapsed().as_secs_f64();
        let st = &dec.stats;
        println!(
            "plain {} tok {:.2} tok/s | dspark {} tok {:.2} tok/s, {} rounds, {:.2} tok/round, accepted {}/{} per-round {:?}, draft {:.2}s verify {:.2}s | same tokens: {}",
            plain.tokens.len(),
            plain.tokens.len() as f64 / plain_secs,
            spec.tokens.len(),
            spec.tokens.len() as f64 / spec_secs,
            st.rounds,
            st.tokens_per_round(),
            st.accepted,
            st.drafted,
            st.per_round,
            st.draft_secs,
            st.verify_secs,
            plain.tokens == spec.tokens
        );
        if plain.tokens != spec.tokens {
            let i = plain.tokens.iter().zip(&spec.tokens).position(|(a, b)| a != b);
            println!("first difference at {i:?}");
            println!("plain: {:?}", tok.decode(&plain.tokens, false));
        }
        println!("{:?}", tok.decode(&spec.tokens, false));
        if o.drift {
            let layers = d.cfg.target_layers.clone();
            let (mut se, mut sf) = (m.new_session(), m.new_session());
            generate::prefill(m, &mut se, ids, 256, &[]);
            generate::prefill(m, &mut sf, ids, 256, &[]);
            let v = m.vocab_size();
            let mut feats = Vec::new();
            let (mut n, mut flips, mut ties) = (0usize, 0usize, 0usize);
            let (mut kl_sum, mut kl_max, mut d_sum, mut d_max) = (0f64, 0f64, 0f64, 0f32);
            for chunk in plain.tokens.chunks(8).filter(|c| c.len() > 1) {
                let le = m.forward_verify(&mut se, chunk, &layers, &mut feats, false);
                let lf = m.forward_verify(&mut sf, chunk, &layers, &mut feats, true);
                for j in 0..chunk.len() {
                    let (a, b) = (&le[j * v..(j + 1) * v], &lf[j * v..(j + 1) * v]);
                    let (kl, dm) = kl_maxdiff(a, b);
                    n += 1;
                    kl_sum += kl;
                    kl_max = kl_max.max(kl);
                    d_sum += dm as f64;
                    d_max = d_max.max(dm);
                    let (ta, tb) = (argmax(a) as usize, argmax(b) as usize);
                    flips += (ta != tb) as usize;
                    // A near-tie: the exact top-2 margin is within the largest difference.
                    let top = a[ta];
                    let second = a
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != ta)
                        .fold(f32::NEG_INFINITY, |m, (_, &x)| m.max(x));
                    ties += (top - second <= dm) as usize;
                }
            }
            println!(
                "drift over {n} positions (8-token passes, fast vs exact): max |logit diff| mean {:.2e} max {:.2e}, KL mean {:.2e} max {:.2e}, top-1 changed at {flips}, exact top-2 margin within the difference at {ties}",
                d_sum / n.max(1) as f64,
                d_max,
                kl_sum / n.max(1) as f64,
                kl_max
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let usage = "usage: dspark_check MODEL DRAFT PROMPT OUT_DIR [--k N] [--cpu] [--raw] [--gen N] [--draft-device cpu|gpu] [--bench N] \
                 [--quant Q] [--think] [--fast] [--drift]";
    let mut args = std::env::args().skip(1);
    let model = args.next().context(usage)?;
    let draft_spec = args.next().context(usage)?;
    let prompt = args.next().context(usage)?;
    let out = PathBuf::from(args.next().context(usage)?);
    let (mut k, mut cpu, mut raw, mut gen_n, mut device, mut bench) = (None, false, false, 0usize, None, 0usize);
    let (mut quant, mut think, mut fast, mut drift) = (Quant::Bf16, false, false, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--k" => k = Some(args.next().context("--k N")?.parse()?),
            "--cpu" => cpu = true,
            "--raw" => raw = true,
            "--think" => think = true,
            "--fast" => fast = true,
            "--drift" => drift = true,
            "--gen" => gen_n = args.next().context("--gen N")?.parse()?,
            "--draft-device" => device = Some(args.next().context("--draft-device cpu|gpu")? == "gpu"),
            "--bench" => bench = args.next().context("--bench N")?.parse()?,
            "--quant" => quant = Quant::parse(&args.next().context("--quant Q")?).context("bad --quant")?,
            other => bail!("unknown argument {other}"),
        }
    }
    let dev = if cpu {
        oxinfer::platform::DevicePref::Cpu
    } else {
        oxinfer::platform::DevicePref::Auto
    };
    oxinfer::gpu::init(&oxinfer::platform::detect(), dev)?;
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let draft_path = oxinfer::engine::resolve_model(&draft_spec, None)?;
    let big = oxinfer::mistral4::too_big(oxinfer::safetensors::checkpoint_bytes(&dir));
    let files = ModelFiles::open(&dir, !big)?;
    let dfiles = ModelFiles::open(&draft_path, true)?;
    let tok = oxinfer::engine::load_tokenizer(&dir)?;
    let eos = oxinfer::config::load_eos_ids(&dir);
    let encode = |text: String| -> Result<Vec<u32>> { Ok(tok.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec()) };
    let mt = oxinfer::config::model_type(&dir)?;
    let fmt: &dyn ChatFormat = if mt == "deepseek_v4" {
        &oxinfer::chat::DeepSeekChat
    } else if mt.starts_with("mistral_large4") {
        &oxinfer::chat::MistralChat
    } else if mt.starts_with("gemma") {
        oxinfer::chat::GemmaChat::for_model(&dir)
    } else {
        oxinfer::chat::QwenChat::for_model(&dir)
    };
    let ids = encode(if raw {
        prompt.clone()
    } else {
        fmt.first_turn(None, think, &[], &prompt)
    })?;
    let o = Opts {
        out,
        k,
        gen_n,
        device,
        bench,
        stop: oxinfer::chat::with_turn_end(fmt, &tok, eos),
        fast,
        drift,
    };
    if mt == "deepseek_v4" {
        let cfg = oxinfer::config::DeepSeek4Config::load(&dir)?;
        let mut m = oxinfer::deepseek4::DeepSeek4Model::load(cfg, &files, quant)?;
        let streamer = if big {
            Some(oxinfer::stream::attach_deepseek4(
                &mut m,
                &files,
                oxinfer::deepseek4::default_stream(),
            )?)
        } else {
            None
        };
        let d = Dspark::load(&draft_path, &dfiles, m.cfg.hidden_size, m.cfg.vocab_size, quant)?;
        eprintln!("[prompt: {} tokens; draft {:?}]", ids.len(), d.cfg.target_layers);
        let r = check(&m, &d, &tok, &ids, &o);
        m.set_stream(None);
        drop(streamer);
        return r;
    }
    if mt.starts_with("mistral_large4") {
        let cfg = oxinfer::config::Mistral4Config::load(&dir)?;
        let mut m = oxinfer::mistral4::Mistral4Model::load(cfg, &files, quant)?;
        let streamer = if big {
            Some(oxinfer::stream::attach_mistral4(
                &mut m,
                &files,
                oxinfer::mistral4::default_stream(),
            )?)
        } else {
            None
        };
        let d = Dspark::load(&draft_path, &dfiles, m.cfg.hidden_size, m.cfg.vocab_size, quant)?;
        eprintln!("[prompt: {} tokens; draft {:?}]", ids.len(), d.cfg);
        let r = check(&m, &d, &tok, &ids, &o);
        m.set_stream(None);
        drop(streamer);
        return r;
    }
    if mt.starts_with("gemma") {
        let m = oxinfer::model::Model::load(oxinfer::config::TextConfig::load(&dir)?, &files, quant)?;
        let d = Dspark::load(&draft_path, &dfiles, m.cfg.hidden_size, m.cfg.vocab_size, quant)?;
        eprintln!("[prompt: {} tokens; draft {:?}]", ids.len(), d.cfg);
        return check(&m, &d, &tok, &ids, &o);
    }
    let m = oxinfer::qwen::QwenModel::load(oxinfer::config::QwenConfig::load(&dir)?, &files, quant)?;
    let d = Dspark::load(&draft_path, &dfiles, m.cfg.hidden_size, m.cfg.vocab_size, quant)?;
    eprintln!("[prompt: {} tokens; draft {:?}]", ids.len(), d.cfg);
    check(&m, &d, &tok, &ids, &o)
}
