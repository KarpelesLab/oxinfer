//! LoRA training of an Inkling-2 model (`oxinfer::inkling::train`), on the CPU:
//!
//! ```text
//! inkling2_train check MODEL REF.safetensors [--window N]
//! inkling2_train agree MODEL ADAPTER_DIR [--len N] [--text TEXT]
//! inkling2_train fit MODEL [--pair "PROMPT|||COMPLETION"]... [--neg "PROMPT|||COMPLETION"]...
//!                    [--random N,LEN,TARGETS] [--steps S] [--lr LR] [--rank R] [--alpha A]
//!                    [--proj wo_ud,wq_du,..] [--window N] [--init ADAPTER_DIR] [--save ADAPTER_DIR]
//! ```
//!
//! `check` compares the loss and the gradients with a reference written by
//! `tools/ref_inkling2_lora_grad.py` (PyTorch autograd on transformers' model, the same token ids,
//! targets and adapter): per tensor the largest absolute difference, the relative L2 error and the
//! cosine. `agree` runs the training forward with a saved adapter and the inference forward with
//! the same adapter attached as `--lora` attaches it, and prints the largest logit difference.
//! `fit` runs AdamW steps over a few episodes (each `--pair` a user message and the reply to
//! imitate, `--neg` a reply to push down, `--random` N sequences of LEN random tokens whose last
//! TARGETS are targets) and prints the loss, the time and the memory of each step.
//! `--window N` truncates the gradient to N positions before the first target.

use anyhow::{Context, Result, bail};
use oxinfer::inkling::train::{Adapter, Grads, Proj, TrainOptions};
use oxinfer::inkling::{Inkling2Config, Inkling2Model};
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

/// Peak and current resident memory in GiB (`/proc/self/status`; 0 elsewhere).
fn rss() -> (f64, f64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |k: &str| {
        s.lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / (1 << 20) as f64
    };
    (get("VmHWM:"), get("VmRSS:"))
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

struct Episode {
    tokens: Vec<u32>,
    targets: Vec<(usize, f32)>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().context("usage: inkling2_train check|agree|fit MODEL ...")?;
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
    let window = opt("--window").map(|v| v.parse::<usize>()).transpose()?;
    let opts = TrainOptions { grad_window: window };
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu)?;
    let dir = oxinfer::engine::resolve_model(&model, None)?;
    let t0 = std::time::Instant::now();
    let files = ModelFiles::open(&dir, true)?;
    let cfg = Inkling2Config::load(&dir)?;
    let m = Inkling2Model::load(cfg.clone(), &files, oxinfer::model::Quant::Bf16)?;
    println!(
        "loaded in {:.1} s; {} layer applications of {}; tape {} bytes per token",
        t0.elapsed().as_secs_f64(),
        cfg.steps().len(),
        cfg.hidden_size,
        m.train_tape_bytes_per_token()
    );
    match mode.as_str() {
        "check" => {
            let st = SafeTensors::open(std::path::Path::new(rest.first().context("REF.safetensors")?), true)?;
            let tokens: Vec<u32> = i64s(&st, "ids")?.into_iter().map(|x| x as u32).collect();
            let targets: Vec<(usize, f32)> = i64s(&st, "tgt_idx")?
                .into_iter()
                .map(|x| x as usize)
                .zip(f32s(&st, "tgt_w")?)
                .collect();
            let alpha = f32s(&st, "alpha")?[0];
            let mut sites = Vec::new();
            for l in 0..cfg.num_hidden_layers {
                for p in Proj::ALL {
                    let an = format!("A.{l}.{}", p.name());
                    if st.info(&an).is_some() {
                        let (n_out, n_in) = p.shape(&cfg, l);
                        sites.push(oxinfer::inkling::train::Site {
                            layer: l,
                            proj: p,
                            n_out,
                            n_in,
                            a: f32s(&st, &an)?,
                            b: f32s(&st, &format!("B.{l}.{}", p.name()))?,
                        });
                    }
                }
            }
            let rank = sites[0].a.len() / sites[0].n_in;
            let ad = Adapter { rank, alpha, sites };
            println!(
                "{} tokens, {} targets, {} sites of rank {rank}",
                tokens.len(),
                targets.len(),
                ad.sites.len()
            );
            let out = m.train_grads(&ad, &tokens, &targets, opts)?;
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
            let lp = f32s(&st, "logp")?;
            println!("log-probabilities: max diff {:.2e}", compare(&out.logp, &lp).0);
            let (mut worst_rel, mut worst_cos, mut worst_abs) = (0f64, 1f64, 0f64);
            let (mut all_mine, mut all_ref) = (Vec::new(), Vec::new());
            for (i, s) in ad.sites.iter().enumerate() {
                for (leaf, mine) in [("gA", &out.grads.a[i]), ("gB", &out.grads.b[i])] {
                    let name = format!("{leaf}.{}.{}", s.layer, s.proj.name());
                    let r = f32s(&st, &name)?;
                    let (mx, rel, cos) = compare(mine, &r);
                    let norm = r.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                    if rest.iter().any(|a| a == "-v") {
                        println!("{name}: |ref| {norm:.3e} max abs {mx:.2e} rel {rel:.2e} cos {cos:.6}");
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
                2 * ad.sites.len()
            );
        }
        "agree" => {
            let adir = std::path::PathBuf::from(rest.first().context("ADAPTER_DIR")?);
            let ad = Adapter::load(&cfg, &adir)?;
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
            let train = m.train_logits(Some(&ad), &tokens, &pos)?;
            let base = m.train_logits(None, &tokens, &pos)?;
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
        "fit" => {
            let mut eps: Vec<Episode> = Vec::new();
            let tok = oxinfer::engine::load_tokenizer(&dir).ok();
            for (flag, w) in [("--pair", 1.0f32), ("--neg", -1.0)] {
                for p in all(flag) {
                    let (prompt, reply) = p.split_once("|||").context("--pair PROMPT|||COMPLETION")?;
                    let tok = tok.as_ref().context("the model has no tokenizer")?;
                    let enc = |s: &str| -> Result<Vec<u32>> {
                        Ok(tok.encode(s, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().to_vec())
                    };
                    let mut tokens = enc(&format!("<|message_user|><|content_text|>{prompt}<|end_message|>"))?;
                    let n0 = tokens.len();
                    tokens.extend(enc(&format!(
                        "<|message_model|><|content_text|>{reply}<|end_message|><|content_model_end_sampling|>"
                    ))?);
                    let targets = (n0..tokens.len()).map(|j| (j, w)).collect();
                    eps.push(Episode { tokens, targets });
                }
            }
            if let Some(r) = opt("--random") {
                let v: Vec<usize> = r.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
                let [n, len, nt] = v[..] else { bail!("--random N,LEN,TARGETS") };
                let mut seed = 0x2545_F491_4F6C_DD1Du64;
                for _ in 0..n {
                    let tokens: Vec<u32> = (0..len)
                        .map(|_| {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            (seed % (cfg.unpadded_vocab_size as u64 - 16)) as u32 + 8
                        })
                        .collect();
                    eps.push(Episode {
                        targets: (len - nt..len).map(|j| (j, 1.0)).collect(),
                        tokens,
                    });
                }
            }
            if eps.is_empty() {
                bail!("no episodes: --pair, --neg or --random");
            }
            let steps: usize = opt("--steps").map_or(Ok(5), |v| v.parse())?;
            let lr: f32 = opt("--lr").map_or(Ok(1e-3), |v| v.parse())?;
            let rank: usize = opt("--rank").map_or(Ok(8), |v| v.parse())?;
            let alpha: f32 = opt("--alpha").map_or(Ok(2.0 * rank as f32), |v| v.parse())?;
            let projs: Vec<Proj> = opt("--proj")
                .unwrap_or_else(|| "wo_ud".into())
                .split(',')
                .map(|n| Proj::parse(n).with_context(|| format!("unknown projection {n}")))
                .collect::<Result<_>>()?;
            let mut ad = match opt("--init") {
                Some(d) => Adapter::load(&cfg, std::path::Path::new(&d))?,
                None => Adapter::new(&cfg, rank, alpha, &projs, &[], 1),
            };
            let n_targets: usize = eps.iter().map(|e| e.targets.len()).sum();
            println!(
                "{} episodes ({} tokens, {n_targets} targets); {} sites of rank {}, {} parameters; lr {lr}",
                eps.len(),
                eps.iter().map(|e| e.tokens.len()).sum::<usize>(),
                ad.sites.len(),
                ad.rank,
                ad.n_params()
            );
            // AdamW (no weight decay), one step per pass over the episodes.
            let (b1, b2, e) = (0.9f32, 0.999f32, 1e-8f32);
            let mut mom: Grads = ad.zeros();
            let mut var: Grads = ad.zeros();
            for step in 1..=steps + 1 {
                let t0 = std::time::Instant::now();
                let mut g = ad.zeros();
                let (mut loss, mut fwd, mut bwd, mut tape) = (0f64, 0f64, 0f64, 0usize);
                let mut per = Vec::new();
                for ep in &eps {
                    if step == steps + 1 {
                        // After the last step: the losses only.
                        let (l, _) = m.train_loss(Some(&ad), &ep.tokens, &ep.targets)?;
                        loss += l;
                        per.push(l / ep.targets.len() as f64);
                        continue;
                    }
                    let out = m.train_grads(&ad, &ep.tokens, &ep.targets, opts)?;
                    loss += out.loss;
                    per.push(out.loss / ep.targets.len() as f64);
                    fwd += out.forward_secs;
                    bwd += out.backward_secs;
                    tape = tape.max(out.tape_bytes);
                    for (acc, part) in g.a.iter_mut().chain(g.b.iter_mut()).zip(out.grads.a.iter().chain(&out.grads.b)) {
                        for (a, &p) in acc.iter_mut().zip(part) {
                            *a += p / n_targets as f32;
                        }
                    }
                }
                let per: Vec<String> = per.iter().map(|l| format!("{l:.4}")).collect();
                let (peak, now) = rss();
                if step == steps + 1 {
                    println!(
                        "after {steps} steps: loss per target {:.5} [{}]",
                        loss / n_targets as f64,
                        per.join(" ")
                    );
                    break;
                }
                let gn =
                    g.a.iter()
                        .chain(&g.b)
                        .flatten()
                        .map(|&x| (x as f64) * (x as f64))
                        .sum::<f64>()
                        .sqrt();
                let (c1, c2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
                let sites = ad.sites.iter_mut().flat_map(|s| [&mut s.a, &mut s.b]);
                let gs = g.a.iter().zip(&g.b).flat_map(|(a, b)| [a, b]);
                let ms = mom.a.iter_mut().zip(mom.b.iter_mut()).flat_map(|(a, b)| [a, b]);
                let vs = var.a.iter_mut().zip(var.b.iter_mut()).flat_map(|(a, b)| [a, b]);
                for (((p, g), m1), v1) in sites.zip(gs).zip(ms).zip(vs) {
                    for i in 0..p.len() {
                        m1[i] = b1 * m1[i] + (1.0 - b1) * g[i];
                        v1[i] = b2 * v1[i] + (1.0 - b2) * g[i] * g[i];
                        p[i] -= lr * (m1[i] / c1) / ((v1[i] / c2).sqrt() + e);
                    }
                }
                println!(
                    "step {step}: loss per target {:.5} [{}] |grad| {gn:.3e}; {:.1} s (forward {fwd:.1}, backward {bwd:.1}); tape {:.2} GiB, RSS {now:.1} GiB, peak {peak:.1} GiB",
                    loss / n_targets as f64,
                    per.join(" "),
                    t0.elapsed().as_secs_f64(),
                    tape as f64 / (1u64 << 30) as f64
                );
            }
            if let Some(d) = opt("--save") {
                ad.save(std::path::Path::new(&d), &model)?;
                println!("adapter written to {d}");
            }
        }
        other => bail!("unknown mode {other}"),
    }
    Ok(())
}
