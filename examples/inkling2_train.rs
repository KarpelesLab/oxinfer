//! Checks of Inkling-2 LoRA training (`oxinfer::inkling::train`, the Inkling side of
//! `oxinfer lora-train`), on the CPU:
//!
//! ```text
//! inkling2_train check MODEL REF.safetensors [--window N] [-v]
//! inkling2_train agree MODEL ADAPTER_DIR [--len N] [--text TEXT]
//! inkling2_train episodes MODEL PROFILE_DIR [--pair "PROMPT|||COMPLETION"]... [--neg "PROMPT|||COMPLETION"]...
//!                         [--random N,PROMPT_LEN,COMPLETION_LEN]
//! ```
//!
//! `check` compares the loss and the gradients with a reference written by
//! `tools/ref_inkling2_lora_grad.py` (PyTorch autograd on transformers' model, the same token ids,
//! targets and adapter): per tensor the largest absolute difference, the relative L2 error and the
//! cosine. `--window N` truncates the gradient to N positions before the first target (the
//! reference does not).
//! `agree` runs the training forward with a saved adapter and the inference forward with the same
//! adapter attached as `--lora` attaches it, and prints the largest logit difference.
//! `episodes` writes `episodes.jsonl` and `outcomes.jsonl` of synthetic episodes for
//! `oxinfer lora-train --profile PROFILE_DIR`: each `--pair` a user message and the reply (reward
//! 1), each `--neg` one with reward -1, `--random` N episodes of random tokens (reward 1).

use anyhow::{Context, Result, bail};
use oxinfer::inkling::train::Trainer;
use oxinfer::inkling::{Inkling2Config, Inkling2Model};
use oxinfer::lora_train::{Adapter, Objective};
use oxinfer::safetensors::{ModelFiles, SafeTensors};

fn f32s(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let info = st.info(name).with_context(|| format!("missing {name}"))?;
    let b = st.bytes(name)?;
    Ok(match info.dtype.as_str() {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "F64" => b.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c) as f32).collect(),
        d => bail!("{name}: dtype {d}"),
    })
}

fn i64s(st: &SafeTensors, name: &str) -> Result<Vec<i64>> {
    Ok(st.bytes(name)?.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect())
}

/// `[rows, cols]` to `[cols, rows]`.
fn transposed(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            y[c * rows + r] = x[r * cols + c];
        }
    }
    y
}

/// Max |a - b|, |a - b| / |b|, cosine.
fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut mx, mut d2, mut a2, mut b2, mut ab) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        mx = mx.max((x - y).abs());
        d2 += (x - y) * (x - y);
        a2 += x * x;
        b2 += y * y;
        ab += x * y;
    }
    (mx, (d2 / b2.max(1e-300)).sqrt(), ab / (a2 * b2).sqrt().max(1e-300))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().context("usage: inkling2_train check|agree|episodes MODEL ...")?;
    let model = args.next().context("MODEL")?;
    let rest: Vec<String> = args.collect();
    let opt = |name: &str| rest.iter().position(|a| a == name).and_then(|i| rest.get(i + 1)).cloned();
    let all = |name: &str| -> Vec<String> {
        rest.iter()
            .enumerate()
            .filter(|(_, a)| *a == name)
            .filter_map(|(i, _)| rest.get(i + 1).cloned())
            .collect()
    };
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let cfg = Inkling2Config::load(&dir)?;
    if mode == "episodes" {
        let out = std::path::PathBuf::from(rest.first().context("PROFILE_DIR")?);
        std::fs::create_dir_all(&out)?;
        let (mut eps, mut outs) = (String::new(), String::new());
        let mut push = |prompt: &[u32], completion: &[u32], reward: f32| {
            let id = format!("syn{}", eps.lines().count());
            eps += &format!(
                "{}\n",
                serde_json::json!({"id": id, "prompt_tokens": prompt, "completion_tokens": completion})
            );
            outs += &format!("{}\n", serde_json::json!({"id": id, "reward": reward, "kind": "synthetic"}));
        };
        for (flag, reward) in [("--pair", 1.0f32), ("--neg", -1.0)] {
            for p in all(flag) {
                let (prompt, reply) = p.split_once("|||").context("--pair PROMPT|||COMPLETION")?;
                let tok = oxinfer::engine::load_tokenizer(&dir)?;
                let enc = |s: &str| -> Result<Vec<u32>> {
                    Ok(tok.encode(s, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().to_vec())
                };
                // (The model opens its own messages: the reply's header is part of the completion.)
                let p = enc(&format!("<|message_user|><|content_text|>{prompt}<|end_message|>"))?;
                let c = enc(&format!(
                    "<|message_model|><|content_text|>{reply}<|end_message|><|content_model_end_sampling|>"
                ))?;
                push(&p, &c, reward);
            }
        }
        if let Some(r) = opt("--random") {
            let v: Vec<usize> = r.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
            let [n, pl, cl] = v[..] else {
                bail!("--random N,PROMPT_LEN,COMPLETION_LEN")
            };
            let mut seed = 0x2545_F491_4F6C_DD1Du64;
            let mut rnd = |n: usize| -> Vec<u32> {
                (0..n)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        (seed % (cfg.unpadded_vocab_size as u64 - 16)) as u32 + 8
                    })
                    .collect()
            };
            for _ in 0..n {
                let (p, c) = (rnd(pl), rnd(cl));
                push(&p, &c, 1.0);
            }
        }
        std::fs::write(out.join("episodes.jsonl"), &eps)?;
        std::fs::write(out.join("outcomes.jsonl"), &outs)?;
        println!("{} episodes written to {}", eps.lines().count(), out.display());
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let files = ModelFiles::open(&dir, true)?;
    let m = Inkling2Model::load(cfg.clone(), &files, oxinfer::model::Quant::Bf16)?;
    let window = opt("--window").map(|v| v.parse::<usize>()).transpose()?;
    let tr = Trainer::new(&m, window)?;
    println!(
        "loaded in {:.1} s; {} layer applications of {}; {} bytes kept per token",
        t0.elapsed().as_secs_f64(),
        cfg.steps().len(),
        cfg.hidden_size,
        tr.tape_bytes_per_token()
    );
    match mode.as_str() {
        "check" => {
            let st = SafeTensors::open(std::path::Path::new(rest.first().context("REF.safetensors")?), true)?;
            let tokens: Vec<u32> = i64s(&st, "ids")?.into_iter().map(|x| x as u32).collect();
            let first = i64s(&st, "first")?[0] as usize;
            let positive = i64s(&st, "positive")?[0] != 0;
            let obj = Objective {
                neg_clip: f32s(&st, "neg_clip")?[0],
            };
            let alpha = f32s(&st, "alpha")?[0];
            // The reference's pairs: `A.{layer}.{name}` and `B.{layer}.{name}`.
            let mut names: Vec<String> = Vec::new();
            let mut layers: Vec<usize> = Vec::new();
            let mut rank = 0;
            for n in st.names().filter(|n| n.starts_with("A.")) {
                let mut it = n.splitn(3, '.').skip(1);
                let (l, name) = (it.next().unwrap().parse::<usize>()?, it.next().unwrap().to_string());
                rank = st.info(n).unwrap().shape[0];
                if !names.contains(&name) {
                    names.push(name);
                }
                if !layers.contains(&l) {
                    layers.push(l);
                }
            }
            let mut ad = Adapter::new(&m, rank, alpha, &names, Some(&layers), 0)?;
            let mods: Vec<(usize, &'static str)> = ad.modules().map(|x| (x.0, x.1)).collect();
            for &(l, name) in &mods {
                let a = f32s(&st, &format!("A.{l}.{name}"))?;
                let b = f32s(&st, &format!("B.{l}.{name}"))?;
                ad.set_module(l, name, &a, &transposed(&b, b.len() / rank, rank))?;
            }
            println!(
                "{} tokens, targets from {first} ({}), {} modules of rank {rank}",
                tokens.len(),
                if positive { "rewarded" } else { "penalized" },
                mods.len()
            );
            let mut grads = ad.zero_grads();
            let out = tr.pass(Some(&ad), &tokens, first, positive, obj, Some((&mut grads, 1.0)))?;
            let ref_loss = f32s(&st, "loss")?[0] as f64;
            println!(
                "loss {:.6} reference {:.6} (diff {:.2e}); forward {:.2} s, backward {:.2} s, gradient from position {}",
                out.loss,
                ref_loss,
                (out.loss - ref_loss).abs(),
                out.forward_secs,
                out.backward_secs,
                out.grad_from
            );
            println!("log-probabilities: max diff {:.2e}", compare(&out.logp, &f32s(&st, "logp")?).0);
            let (mut worst_rel, mut worst_cos, mut worst_abs) = (0f64, 1f64, 0f64);
            let (mut all_mine, mut all_ref) = (Vec::new(), Vec::new());
            for (&(l, name), g) in mods.iter().zip(&grads) {
                let rb = f32s(&st, &format!("gB.{l}.{name}"))?;
                let rb = transposed(&rb, rb.len() / rank, rank);
                for (leaf, mine, r) in [("gA", &g.0, f32s(&st, &format!("gA.{l}.{name}"))?), ("gB", &g.1, rb)] {
                    let (mx, rel, cos) = compare(mine, &r);
                    let norm = r.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                    if rest.iter().any(|a| a == "-v") {
                        println!("{leaf}.{l}.{name}: |ref| {norm:.3e} max abs {mx:.2e} rel {rel:.2e} cos {cos:.6}");
                    }
                    worst_abs = worst_abs.max(mx);
                    worst_rel = worst_rel.max(rel);
                    worst_cos = worst_cos.min(cos);
                    all_mine.extend_from_slice(mine);
                    all_ref.extend(r);
                }
            }
            let (mx, rel, cos) = compare(&all_mine, &all_ref);
            println!(
                "gradients, {} tensors: worst per tensor max abs {worst_abs:.2e}, rel {worst_rel:.2e}, cos {worst_cos:.6}; all together max abs {mx:.2e}, rel {rel:.2e}, cos {cos:.8}",
                2 * mods.len()
            );
        }
        "agree" => {
            let adir = std::path::PathBuf::from(rest.first().context("ADAPTER_DIR")?);
            let ad = Adapter::load(&m, &adir)?;
            let tokens: Vec<u32> = match opt("--text") {
                Some(text) => oxinfer::engine::load_tokenizer(&dir)?
                    .encode(text.as_str(), false)
                    .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                    .get_ids()
                    .to_vec(),
                None => {
                    let n: usize = opt("--len").map_or(Ok(64), |v| v.parse())?;
                    (0..n)
                        .map(|i| ((i * 7919 + 13) % (cfg.unpadded_vocab_size - 16)) as u32 + 8)
                        .collect()
                }
            };
            let t = tokens.len();
            let pos: Vec<usize> = (0..t).collect();
            let train = tr.logits(Some(&ad), &tokens, &pos)?;
            let base = tr.logits(None, &tokens, &pos)?;
            // (The inference forward on the CPU, like the training one.)
            oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu)?;
            let files2 = ModelFiles::open(&dir, true)?;
            let mut m2 = Inkling2Model::load(cfg.clone(), &files2, oxinfer::model::Quant::Bf16)?;
            let spec = oxinfer::lora::LoraSpec { path: adir, scale: 1.0 };
            let at = oxinfer::lora::attach_specs(&mut m2, &[spec], &dir, false)?;
            let run = |m: &Inkling2Model| {
                let mut s = m.new_session();
                let hs = m.forward_hidden(&mut s, &tokens);
                m.logits(&hs, t)
            };
            let (adapted, plain) = (run(&m2), run(&m));
            println!(
                "{t} tokens, {} modules attached: training forward vs --lora max logit diff {:.2e}; base, training vs inference {:.2e}; the adapter moves the logits by up to {:.2e}",
                at[0].modules,
                compare(&train, &adapted).0,
                compare(&base, &plain).0,
                compare(&train, &base).0
            );
        }
        other => bail!("unknown mode {other}"),
    }
    Ok(())
}
