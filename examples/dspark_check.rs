//! Checks a DSpark draft against a reference (`tools/ref_dspark.py`) and measures its acceptance:
//!
//! ```text
//! dspark_check MODEL DRAFT.gguf PROMPT OUT_DIR [--k N] [--cpu] [--raw] [--gen N] [--draft-device cpu|gpu] [--bench N]
//! ```
//!
//! Runs the chat-formatted `PROMPT` (as is with `--raw`) through the target, takes its greedy next
//! token as the anchor, and writes to `OUT_DIR` the target features of the prompt positions
//! (`feats.f32`), the draft's logits for `k` slots chained greedily (`logits.f32`) and
//! `meta.json`. With `--gen N` it then decodes N tokens greedily with and without DSpark and
//! checks that both give the same tokens. `--draft-device` says where the draft's matmuls run
//! (default: the GPU when there is one). `--bench N` times N draft passes and injections of 3
//! positions on the CPU and on the GPU, alternating.

use anyhow::{Context, Result, bail};
use oxinfer::chat::ChatFormat;
use oxinfer::dspark::{Dspark, DsparkDecoder, DsparkOptions, Target};
use oxinfer::generate::{self, Flow};
use oxinfer::mistral4::Mistral4Model;
use oxinfer::safetensors::ModelFiles;
use oxinfer::sampler::{Sampler, argmax};
use std::path::{Path, PathBuf};

fn write_f32(path: &Path, v: &[f32]) -> Result<()> {
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(())
}

fn main() -> Result<()> {
    let usage =
        "usage: dspark_check MODEL DRAFT.gguf PROMPT OUT_DIR [--k N] [--cpu] [--raw] [--gen N] [--draft-device cpu|gpu] [--bench N]";
    let mut args = std::env::args().skip(1);
    let model = args.next().context(usage)?;
    let draft_path = PathBuf::from(args.next().context(usage)?);
    let prompt = args.next().context(usage)?;
    let out = PathBuf::from(args.next().context(usage)?);
    let (mut k, mut cpu, mut raw, mut gen_n, mut device, mut bench) = (3usize, false, false, 0usize, None, 0usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--k" => k = args.next().context("--k N")?.parse()?,
            "--cpu" => cpu = true,
            "--raw" => raw = true,
            "--gen" => gen_n = args.next().context("--gen N")?.parse()?,
            "--draft-device" => device = Some(args.next().context("--draft-device cpu|gpu")? == "gpu"),
            "--bench" => bench = args.next().context("--bench N")?.parse()?,
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
    let cfg = oxinfer::config::Mistral4Config::load(&dir)?;
    let big = oxinfer::mistral4::too_big(oxinfer::safetensors::checkpoint_bytes(&dir));
    let files = ModelFiles::open(&dir, !big)?;
    let mut m = Mistral4Model::load(cfg, &files, oxinfer::model::Quant::Bf16)?;
    let streamer = if big {
        Some(oxinfer::stream::attach_mistral4(
            &mut m,
            &files,
            oxinfer::mistral4::default_stream(),
        )?)
    } else {
        None
    };
    let dfiles = ModelFiles::open(&draft_path, true)?;
    let d = Dspark::load(
        &draft_path,
        &dfiles,
        m.cfg.hidden_size,
        m.cfg.vocab_size,
        oxinfer::model::Quant::Bf16,
    )?;
    let tok = oxinfer::engine::load_tokenizer(&dir)?;
    let text = if raw {
        prompt.clone()
    } else {
        oxinfer::chat::MistralChat.first_turn(None, false, &[], &prompt)
    };
    let ids = tok.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    eprintln!("[prompt: {} tokens; draft {:?}]", ids.len(), d.cfg);

    let r = (|| -> Result<()> {
        let mut dec = DsparkDecoder::new(
            &m,
            &d,
            DsparkOptions {
                draft_tokens: k,
                device,
                ..Default::default()
            },
        );
        let t0 = std::time::Instant::now();
        let logits = dec.prefill(&ids, 256);
        let anchor = argmax(&logits);
        eprintln!(
            "[prefill {:.2}s, anchor {anchor} {:?}]",
            t0.elapsed().as_secs_f64(),
            tok.decode(&[anchor], false)
        );
        let (dl, drafts) = d.draft_logits(&m, dec.draft_session(), anchor, k);
        eprintln!("[drafts {drafts:?} = {:?}]", tok.decode(&drafts, false));
        // The features of the prompt positions, recomputed (same kernels, so the same values).
        let mut s = m.new_session();
        let mut feats = Vec::new();
        let mut all = Vec::new();
        for c in ids.chunks(256) {
            m.forward_feats(&mut s, c, &[], &d.cfg.target_layers, &mut feats);
            all.extend_from_slice(&feats);
        }
        std::fs::create_dir_all(&out)?;
        write_f32(&out.join("feats.f32"), &all)?;
        write_f32(&out.join("logits.f32"), &dl)?;
        let meta = serde_json::json!({"n": ids.len(), "anchor": anchor, "k": drafts.len(), "drafts": drafts});
        std::fs::write(out.join("meta.json"), meta.to_string())?;

        if bench > 0 {
            // Draft passes and injections of the last 3 positions' features, CPU and GPU in turn.
            let nf = all.len() / ids.len();
            let rows = &all[all.len() - 3 * nf..];
            let mut times = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
            for i in 0..2 * bench {
                let gpu = i % 2 == 1;
                d.set_device(gpu);
                let t = std::time::Instant::now();
                let (l, _) = d.draft_logits(&m, dec.draft_session(), anchor, k);
                times[gpu as usize][0].push(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(l);
                let mut ds = dec.draft_session().clone();
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
            d.set_device(device.unwrap_or(oxinfer::dspark::DEVICE_BY_DEFAULT && oxinfer::gpu::active().is_some()));
        }

        if gen_n > 0 {
            let stop = oxinfer::config::load_eos_ids(&dir);
            let mut plain_s = m.new_session();
            let lg = generate::prefill(&m, &mut plain_s, &ids, 256, &[]);
            let lg = m.logits(&lg[lg.len() - m.cfg.hidden_size..], 1);
            let mut sp = Sampler::new(0.0, 0, 1.0, 1);
            let t1 = std::time::Instant::now();
            let plain = generate::decode(&m, &mut plain_s, lg, &mut sp, &stop, gen_n, &mut |_| Flow::Continue);
            let plain_secs = t1.elapsed().as_secs_f64();
            let mut dec = DsparkDecoder::new(
                &m,
                &d,
                DsparkOptions {
                    draft_tokens: k,
                    stop: stop.clone(),
                    device,
                    ..Default::default()
                },
            );
            let lg = dec.prefill(&ids, 256);
            let t2 = std::time::Instant::now();
            let spec = generate::decode_rounds(&mut dec, &lg, &mut sp, &stop, gen_n, &mut |_| Flow::Continue);
            let spec_secs = t2.elapsed().as_secs_f64();
            let st = &dec.stats;
            println!(
                "plain {} tok {:.2} tok/s | dspark {} tok {:.2} tok/s, {} rounds, {:.2} tok/round, draft {:.1} ms/round, accepted {}/{} per-round {:?} | same tokens: {}",
                plain.tokens.len(),
                plain.tokens.len() as f64 / plain_secs,
                spec.tokens.len(),
                spec.tokens.len() as f64 / spec_secs,
                st.rounds,
                st.tokens_per_round(),
                1e3 * st.draft_secs / st.rounds.max(1) as f64,
                st.accepted,
                st.drafted,
                st.per_round,
                plain.tokens == spec.tokens
            );
            if plain.tokens != spec.tokens {
                let i = plain.tokens.iter().zip(&spec.tokens).position(|(a, b)| a != b);
                println!("first difference at {i:?}");
            }
            println!("{:?}", tok.decode(&spec.tokens, false));
        }
        Ok(())
    })();
    m.set_stream(None);
    drop(streamer);
    r
}
