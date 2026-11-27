//! Tests of the LoRA trainer on tiny random dense checkpoints
//! ([`crate::qwen::tiny::write_dense`]).

use super::*;
use crate::config::QwenConfig;
use crate::model::Quant;
use crate::safetensors::ModelFiles;

fn tiny_dir(tag: &str, llama: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxinfer-lora-train-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    crate::qwen::tiny::write_dense(&dir, llama).unwrap();
    dir
}

fn load(dir: &Path) -> QwenModel<'static> {
    let files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(dir, true).unwrap()));
    QwenModel::load(QwenConfig::load(dir).unwrap(), files, Quant::Bf16).unwrap()
}

fn toks(n: usize, seed: u32) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + seed * 101 + i * i * 7 + 5) % 256).collect()
}

fn episode(id: &str, p: usize, c: usize, seed: u32, weight: f32) -> Episode {
    Episode {
        id: id.into(),
        prompt: toks(p, seed),
        completion: toks(c, seed + 50),
        weight,
    }
}

/// The inference model's logits of `tokens` (one prompt pass).
fn model_logits(m: &QwenModel, tokens: &[u32]) -> Vec<f32> {
    let mut s = m.new_session();
    let hs = m.forward_hidden_mm(&mut s, tokens, &[]);
    m.logits(&hs, tokens.len())
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Without an adapter the trainer's forward is the model's.
#[test]
fn base_forward_matches_model() {
    for llama in [false, true] {
        let dir = tiny_dir(&format!("fwd{llama}"), llama);
        let m = load(&dir);
        let tr = Trainer::new(&m, 0).unwrap();
        let t = toks(40, 3);
        let (ours, theirs) = (tr.logits(None, &t), model_logits(&m, &t));
        let d = max_diff(&ours, &theirs);
        let scale = theirs.iter().fold(0f32, |a, v| a.max(v.abs()));
        eprintln!("llama {llama}: trainer forward against the model's: max |diff| {d:e} (logits up to {scale:.2})");
        assert!(d <= 1e-4 * scale, "llama {llama}: {d}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The token loss's gradient against central differences of the loss, in each of its branches,
/// and its continuity where the capped branch starts.
#[test]
fn token_loss_gradient() {
    let base: Vec<f32> = (0..12).map(|i| ((i * 7 % 5) as f32 - 2.0) * 0.7).collect();
    // (positive, the target's logit, cap): imitation, uncapped and capped push-down.
    for (positive, z, cap) in [(true, 0.3f32, 4.0f32), (false, 0.3, 4.0), (false, 6.0, 4.0), (false, 6.0, 0.5)] {
        let mut row = base.clone();
        row[5] = z;
        let mut g = row.clone();
        token_loss(&mut g, 5, positive, cap);
        for j in 0..row.len() {
            let h = 1e-2f32;
            let (mut a, mut b) = (row.clone(), row.clone());
            a[j] += h;
            b[j] -= h;
            let step = (a[j] - b[j]) as f64;
            let fd = (token_loss(&mut a, 5, positive, cap) - token_loss(&mut b, 5, positive, cap)) / step;
            assert!(
                (fd - g[j] as f64).abs() < 2e-4 * (1.0 + fd.abs()),
                "positive {positive} z {z} cap {cap} logit {j}: {fd} vs {}",
                g[j]
            );
        }
    }
    // Either side of the odds reaching the cap (p = 0.8 at cap 4).
    let at = |z: f32| {
        let mut row = vec![0f32, z];
        let l = token_loss(&mut row, 1, false, 4.0);
        (l, row[1])
    };
    let z0 = 4f32.ln();
    let ((l0, g0), (l1, g1)) = (at(z0 - 1e-3), at(z0 + 1e-3));
    assert!((l0 - l1).abs() < 1e-2 && (g0 - g1).abs() < 2e-3, "{l0} {l1} {g0} {g1}");
    // Bounded: a token the model is sure of is pushed by the cap times (1 - p), not by p.
    assert!(at(30.0).1.abs() < 1e-3);
    // Unlikely tokens are left alone.
    assert!(at(-30.0).0.abs() < 1e-9 && at(-30.0).1.abs() < 1e-9);
}

/// A random adapter on every target with `B` nonzero.
fn random_adapter(m: &QwenModel, seed: u64) -> Adapter {
    let targets: Vec<String> = TARGETS.iter().map(|s| s.to_string()).collect();
    let mut ad = Adapter::new(m, 4, 8.0, &targets, None, seed).unwrap();
    let mut rng = Rng::new(seed + 1);
    for md in &mut ad.mods {
        md.bt.iter_mut().for_each(|v| *v = (rng.unit() - 0.5) * 0.2);
    }
    ad
}

/// The loss of `eps` (weighted as a training batch) and its gradients.
fn batch(tr: &Trainer, ad: &Adapter, eps: &[Episode], obj: Objective, grads: bool) -> (f64, Grads) {
    let den: f32 = eps.iter().map(|e| e.weight.abs()).sum();
    let mut g = ad.zero_grads();
    let mut loss = 0.0;
    for e in eps {
        let w = e.weight.abs() / den;
        loss += w as f64 * tr.episode(Some(ad), e, obj, grads.then_some((&mut g, w))).unwrap();
    }
    (loss, g)
}

/// The analytic gradients against central differences of the loss along random directions, one
/// per matrix of the adapter: Qwen3 (query / key norms) and Llama 3 (none), a rewarded and two
/// penalized episodes, the push-down both under and over its cap.
#[test]
fn gradients_match_finite_differences() {
    for llama in [false, true] {
        for neg_clip in [4.0f32, 0.002] {
            let dir = tiny_dir(&format!("fd{llama}{}", neg_clip < 1.0), llama);
            let m = load(&dir);
            let tr = Trainer::new(&m, 0).unwrap();
            let ad = random_adapter(&m, 5);
            let eps = [
                episode("a", 9, 14, 1, 1.0),
                episode("b", 5, 11, 2, -0.7),
                episode("c", 12, 6, 3, -1.0),
            ];
            let obj = Objective { neg_clip };
            let (_, g) = batch(&tr, &ad, &eps, obj, true);
            let mut rng = Rng::new(9);
            let mut worst = 0f64;
            let mut sum = 0f64;
            for i in 0..ad.mods.len() {
                for part in 0..2 {
                    let n = if part == 0 { ad.mods[i].a.len() } else { ad.mods[i].bt.len() };
                    let d: Vec<f32> = (0..n).map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 }).collect();
                    let analytic: f64 = if part == 0 { &g[i].0 } else { &g[i].1 }
                        .iter()
                        .zip(&d)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum();
                    // A fourth-order stencil: a step large enough for the f32 loss to resolve.
                    let h = 1e-2f32;
                    let shifted = |k: f32| {
                        let mut a2 = ad.clone();
                        let p = if part == 0 { &mut a2.mods[i].a } else { &mut a2.mods[i].bt };
                        p.iter_mut().zip(&d).for_each(|(v, dv)| *v += k * h * dv);
                        batch(&tr, &a2, &eps, obj, false).0
                    };
                    let fd = (8.0 * (shifted(1.0) - shifted(-1.0)) - (shifted(2.0) - shifted(-2.0))) / (12.0 * h as f64);
                    let err = (fd - analytic).abs() / fd.abs().max(analytic.abs());
                    worst = worst.max(err);
                    sum += err;
                    assert!(
                        err < 5e-2,
                        "llama {llama} cap {neg_clip} {} {}: finite difference {fd} against {analytic}",
                        module_name(ad.mods[i].layer, ad.mods[i].slot),
                        if part == 0 { "A" } else { "B" }
                    );
                }
            }
            eprintln!(
                "llama {llama} cap {neg_clip}: {} directional derivatives, relative error mean {:.2e}, worst {worst:.2e}",
                2 * ad.mods.len(),
                sum / (2 * ad.mods.len()) as f64
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// An adapter on layers 1-2 only gets the gradients the full backward gives those layers (the
/// lowest adapted layer stops early).
#[test]
fn lowest_layer_stops_early() {
    let dir = tiny_dir("low", false);
    let m = load(&dir);
    let full = random_adapter(&m, 5);
    let targets: Vec<String> = TARGETS.iter().map(|s| s.to_string()).collect();
    let mut part = Adapter::new(&m, 4, 8.0, &targets, Some(&[1, 2]), 5).unwrap();
    // The same matrices as `full` on layers 1-2, and `full`'s layer 0 switched off.
    let mut full0 = full.clone();
    for md in &mut full0.mods {
        if md.layer == 0 {
            md.bt.fill(0.0);
        }
    }
    for md in &mut part.mods {
        let src = &full.mods[full.index[md.layer][md.slot].unwrap()];
        (md.a, md.bt) = (src.a.clone(), src.bt.clone());
    }
    let eps = [episode("a", 9, 14, 1, 1.0)];
    let obj = Objective { neg_clip: 4.0 };
    let (l0, g0) = batch(&Trainer::new(&m, 0).unwrap(), &full0, &eps, obj, true);
    let (l1, g1) = batch(&Trainer::new(&m, 1).unwrap(), &part, &eps, obj, true);
    assert!((l0 - l1).abs() < 1e-6, "{l0} {l1}");
    for (i, md) in part.mods.iter().enumerate() {
        let j = full.index[md.layer][md.slot].unwrap();
        for (a, b) in [(&g1[i].0, &g0[j].0), (&g1[i].1, &g0[j].1)] {
            let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(max_diff(a, b) <= 1e-5 * scale.max(1e-6), "{}", module_name(md.layer, md.slot));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn episodes_and_outcomes() {
    let dir = std::env::temp_dir().join(format!("oxinfer-lora-train-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (e, o) = (dir.join("episodes.jsonl"), dir.join("outcomes.jsonl"));
    std::fs::write(
        &e,
        r#"{"id":"a","time":1,"model":"m","adapters":[],"prompt_tokens":[1,2],"completion_tokens":[3,4],"logprobs":[-0.1,-0.2],"finish_reason":"stop"}
{"id":"b","prompt_tokens":[1],"completion_tokens":[5]}
{"id":"c","prompt_tokens":[1],"completion_tokens":[6]}
{"id":"d","prompt_tokens":[1],"completion_tokens":[]}
{"id":"e","prompt_tokens":[1],"completion_tokens":[7]}
{"id":"f","prompt_tokens":[1],"completion_tok"#,
    )
    .unwrap();
    std::fs::write(
        &o,
        r#"{"id":"a","time":2,"reward":-1,"note":"first","kind":"tests"}
{"id":"a","time":3,"reward":0.5,"note":"then","kind":"tests"}
{"id":"b","reward":-3,"kind":"user_rejected"}
{"id":"d","reward":1}
{"id":"a","time":4,"reward":null,"kind":null,"note":"a note alone"}
{"id":"e","reward":1,"kind":"ignored"}
"#,
    )
    .unwrap();
    let kw = parse_kind_weights("user_rejected=0.3, ignored=0").unwrap();
    let (eps, stats) = load_episodes(&e, &o, &kw).unwrap();
    assert_eq!(eps.len(), 2);
    assert_eq!((eps[0].id.as_str(), eps[0].weight), ("a", 0.5));
    assert_eq!(
        (eps[0].prompt.as_slice(), eps[0].completion.as_slice()),
        (&[1u32, 2][..], &[3u32, 4][..])
    );
    // Clamped to -1, times its kind's weight.
    assert_eq!((eps[1].id.as_str(), eps[1].weight), ("b", -0.3));
    assert_eq!((stats.episodes, stats.without_outcome, stats.empty), (5, 1, 2));
    assert!(parse_kind_weights("tests").is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The split is by id: the same episodes stay held out as the log grows.
#[test]
fn split_is_stable() {
    let eps = |n: usize| (0..n).map(|i| episode(&format!("ep-{i}"), 3, 3, i as u32, 1.0)).collect::<Vec<_>>();
    let (t1, h1) = split(eps(40), 0.25);
    let (_, h2) = split(eps(80), 0.25);
    assert!(!h1.is_empty() && !t1.is_empty() && h1.len() + t1.len() == 40);
    assert!(h1.iter().all(|e| h2.iter().any(|f| f.id == e.id)));
    assert!((4..=16).contains(&h1.len()), "{} of 40 held out", h1.len());
    let (t, h) = split(eps(2), 0.01);
    assert_eq!((t.len(), h.len()), (1, 1));
    let (t, h) = split(eps(1), 0.5);
    assert_eq!((t.len(), h.len()), (1, 0));
}

/// The mean log-probability the inference model gives `ep`'s completion.
fn completion_logp(m: &QwenModel, ep: &Episode) -> f64 {
    let mut tokens = ep.prompt.clone();
    tokens.extend_from_slice(&ep.completion);
    let v = m.cfg.vocab_size;
    let logits = model_logits(m, &tokens[..tokens.len() - 1]);
    let first = ep.prompt.len() - 1;
    let mut sum = 0f64;
    for (i, row) in logits.chunks(v).enumerate().skip(first) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max as f64 + row.iter().map(|&z| ((z - max) as f64).exp()).sum::<f64>().ln();
        sum += row[tokens[i + 1] as usize] as f64 - lse;
    }
    sum / ep.completion.len() as f64
}

/// End to end on a tiny model: a profile directory with rewarded and penalized episodes is
/// trained on, the adapter written, and loaded through the path `--lora` takes
/// ([`crate::lora::attach_specs`]): the model with it gives the trainer's logits, the rewarded
/// completions became more likely and the penalized ones less. A second run starts from the
/// adapter and keeps the previous one; a run that makes the held-out loss worse leaves it alone.
#[test]
fn end_to_end() {
    for llama in [false, true] {
        let dir = tiny_dir(&format!("e2e{llama}"), llama);
        let profile = dir.join("learn");
        std::fs::create_dir_all(&profile).unwrap();
        // Twelve prompts, each with a rewarded completion (one fixed sequence) and a penalized
        // one (what the base model writes by itself, greedily: tokens it finds likely).
        let m = load(&dir);
        let good = toks(10, 900);
        let (mut el, mut ol) = (String::new(), String::new());
        let mut all = Vec::new();
        for i in 0..12u32 {
            let prompt = toks(6 + (i as usize % 4), i);
            let mut s = m.new_session();
            let mut l = m.forward(&mut s, &prompt);
            let mut bad = Vec::new();
            while bad.len() < 10 {
                bad.push(crate::sampler::argmax(&l));
                l = m.forward(&mut s, &bad[bad.len() - 1..]);
            }
            for (tag, comp, reward, kind) in [("g", &good, 1.0f32, "tests"), ("b", &bad, -1.0, "user_rejected")] {
                let id = format!("{tag}{i}");
                el.push_str(&serde_json::json!({"id": id, "prompt_tokens": prompt, "completion_tokens": comp, "adapters": []}).to_string());
                el.push('\n');
                ol.push_str(&serde_json::json!({"id": id, "reward": reward, "kind": kind}).to_string());
                ol.push('\n');
                all.push(Episode {
                    id,
                    prompt: prompt.clone(),
                    completion: comp.clone(),
                    weight: reward,
                });
            }
        }
        std::fs::write(profile.join("episodes.jsonl"), el).unwrap();
        std::fs::write(profile.join("outcomes.jsonl"), ol).unwrap();
        let paths = Paths::profile(&profile);
        let opts = TrainOptions {
            lr: 2e-3,
            steps: 40,
            holdout: 0.25,
            ..TrainOptions::default()
        };
        let mut lines = Vec::new();
        let rep = run(&m, "tiny", &paths, &[], &opts, &mut |l| lines.push(l.to_string())).unwrap();
        eprintln!(
            "llama {llama}: {} train / {} held out, training loss {:.4} -> {:.4}, held-out loss base {:.4} -> adapter {:.4}",
            rep.train_episodes,
            rep.holdout_episodes,
            rep.train_loss_before,
            rep.train_loss_after,
            rep.holdout_loss_base.unwrap(),
            rep.holdout_loss_new.unwrap()
        );
        assert!(rep.written && rep.better && !rep.resumed && rep.gate_against == "base", "{lines:?}");
        assert!(rep.train_loss_after < rep.train_loss_before);
        assert!(crate::lora::is_adapter_dir(&paths.adapter) && !paths.previous().exists());

        // Through the normal adapter path, on a model loaded again.
        let mut with = load(&dir);
        let spec = crate::lora::LoraSpec::parse(paths.adapter.to_str().unwrap()).unwrap();
        let attached = crate::lora::attach_specs(&mut with, &[spec], &dir, false).unwrap();
        assert_eq!(attached[0].modules, 2 * m.layers.len());
        let ad = Adapter::load(&m, &paths.adapter).unwrap();
        let tr = Trainer::new(&m, 0).unwrap();
        let t = toks(30, 77);
        let (ours, theirs, base) = (tr.logits(Some(&ad), &t), model_logits(&with, &t), model_logits(&m, &t));
        let scale = theirs.iter().fold(0f32, |a, v| a.max(v.abs()));
        let (d, moved) = (max_diff(&ours, &theirs), max_diff(&base, &theirs));
        eprintln!(
            "llama {llama}: trainer logits against the model with the adapter attached: max |diff| {d:e}; \
             the adapter moves the logits by up to {moved:.3} (logits up to {scale:.2})"
        );
        assert!(d <= 1e-4 * scale && moved > 100.0 * d, "{d} {moved}");
        let (mut up, mut down) = (0, 0);
        let (mut dg, mut db) = (0f64, 0f64);
        for e in &all {
            let (before, after) = (completion_logp(&m, e), completion_logp(&with, e));
            if e.weight > 0.0 {
                up += (after > before) as usize;
                dg += (after - before) / 12.0;
            } else {
                down += (after < before) as usize;
                db += (after - before) / 12.0;
            }
        }
        eprintln!(
            "llama {llama}: mean log-probability per token: rewarded completions {dg:+.4} ({up}/12 up), penalized {db:+.4} ({down}/12 down)"
        );
        // (Some of the 24 episodes were held out.)
        assert!(up == 12 && down >= 10 && db < 0.0, "{up} {down} {db}");

        // Again: from the adapter, compared with it, the previous one kept.
        let rep2 = run(&m, "tiny", &paths, &[], &opts, &mut |_| {}).unwrap();
        assert!(rep2.resumed && rep2.gate_against == "existing");
        assert!((rep2.holdout_loss_existing.unwrap() - rep.holdout_loss_new.unwrap()).abs() < 1e-6);
        assert!((rep2.train_loss_before - rep.train_loss_after).abs() < 1e-6);
        assert_eq!(rep2.written, rep2.better);
        if rep2.written {
            let prev = Adapter::load(&m, &paths.previous()).unwrap();
            assert!(prev.mods.iter().zip(&ad.mods).all(|(a, b)| a.a == b.a && a.bt == b.bt));
        }
        // A run that climbs the loss is not better: nothing changes on disk.
        let before = std::fs::read(paths.adapter.join("adapter_model.safetensors")).unwrap();
        let worse = TrainOptions { lr: -2e-3, ..opts.clone() };
        let rep3 = run(&m, "tiny", &paths, &[], &worse, &mut |_| {}).unwrap();
        assert!(!rep3.better && !rep3.written);
        assert_eq!(before, std::fs::read(paths.adapter.join("adapter_model.safetensors")).unwrap());
        // Another rank: a new adapter, still compared with the existing one.
        let other = TrainOptions {
            rank: 4,
            alpha: 8.0,
            ..opts.clone()
        };
        let rep4 = run(&m, "tiny", &paths, &[], &other, &mut |_| {}).unwrap();
        assert!(!rep4.resumed && rep4.existing_note.is_some() && rep4.gate_against == "existing");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A comparison with PyTorch + peft (`tools/ref_lora_train.py DIR` writes the checkpoint, the
/// episodes, a starting adapter, its gradients and the adapter after AdamW steps):
/// `OXINFER_LORA_TRAIN_REF=DIR cargo test --profile ci --lib lora_train::tests::reference -- --ignored --nocapture`.
#[test]
#[ignore]
fn reference() {
    let Some(dir) = std::env::var_os("OXINFER_LORA_TRAIN_REF").map(PathBuf::from) else {
        eprintln!("OXINFER_LORA_TRAIN_REF is not set");
        return;
    };
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("ref.json")).unwrap()).unwrap();
    let f = |k: &str| meta[k].as_f64().unwrap();
    let m = load(&dir.join("model"));
    let (eps, _) = load_episodes(&dir.join("episodes.jsonl"), &dir.join("outcomes.jsonl"), &[]).unwrap();
    let obj = Objective {
        neg_clip: f("neg_clip") as f32,
    };
    let mut ad = Adapter::load(&m, &dir.join("init")).unwrap();
    let tr = Trainer::new(&m, 0).unwrap();
    // Per-episode losses.
    for (e, want) in eps.iter().zip(meta["episode_losses"].as_array().unwrap()) {
        let got = tr.episode(Some(&ad), e, obj, None).unwrap();
        let want = want.as_f64().unwrap();
        eprintln!("episode {} (weight {}): loss {got:.6}, PyTorch {want:.6}", e.id, e.weight);
        assert!((got - want).abs() < 2e-5 * want.abs().max(1.0), "{got} {want}");
    }
    let (loss, mut g) = batch(&tr, &ad, &eps, obj, true);
    eprintln!("batch loss {loss:.7}, PyTorch {:.7}", f("loss"));
    // Gradients, as an adapter-shaped file (lora_A / lora_B hold dA / dB).
    let gref = Adapter::load(&m, &dir.join("grads")).unwrap();
    let (mut worst, mut num, mut den) = (0f64, 0f64, 0f64);
    for (i, md) in gref.mods.iter().enumerate() {
        for (ours, theirs) in [(&g[i].0, &md.a), (&g[i].1, &md.bt)] {
            let d: f64 = ours.iter().zip(theirs).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
            let n: f64 = theirs.iter().map(|b| (*b as f64).powi(2)).sum();
            worst = worst.max((d / n).sqrt());
            num += d;
            den += n;
        }
    }
    eprintln!(
        "gradients of {} matrices against PyTorch autograd: relative L2 error {:.2e} overall, worst matrix {worst:.2e}",
        2 * gref.mods.len(),
        (num / den).sqrt()
    );
    assert!(worst < 1e-3);
    // The same optimizer steps.
    let steps = meta["steps"].as_u64().unwrap();
    let mut opt = AdamW::new(&ad, f("lr") as f32, f("weight_decay") as f32);
    for s in 0..steps {
        if s > 0 {
            g = batch(&tr, &ad, &eps, obj, true).1;
        }
        clip_grads(&mut g, f("clip") as f32);
        opt.step(&mut ad, &g);
    }
    let want = Adapter::load(&m, &dir.join("stepped")).unwrap();
    let init = Adapter::load(&m, &dir.join("init")).unwrap();
    let (mut d, mut moved) = (0f64, 0f64);
    for ((a, b), c) in ad.mods.iter().zip(&want.mods).zip(&init.mods) {
        for ((x, y), z) in a.a.iter().chain(&a.bt).zip(b.a.iter().chain(&b.bt)).zip(c.a.iter().chain(&c.bt)) {
            d += (*x as f64 - *y as f64).powi(2);
            moved += (*y as f64 - *z as f64).powi(2);
        }
    }
    eprintln!(
        "after {steps} AdamW steps: parameters differ from peft's by {:.2e} of the distance they moved",
        (d / moved).sqrt()
    );
    assert!((d / moved).sqrt() < 1e-2);
    let after = batch(&tr, &ad, &eps, obj, false).0;
    eprintln!("batch loss after the steps {after:.7}, PyTorch {:.7}", f("loss_after"));
    assert!((after - f("loss_after")).abs() < 1e-4 * f("loss_after").abs().max(1.0));
}

/// On real weights (`OXINFER_LORA_TRAIN_MODEL`: a checkpoint folder or GGUF file;
/// `OXINFER_LORA_TRAIN_PROFILE`: a directory with episodes and, after `oxinfer lora-train`, its
/// adapter): the trainer's logits with the adapter against the model's with it attached, and the
/// analytic gradient of an episode's loss against a finite difference along itself (for
/// quantized weights the gradient is a straight-through estimate, see the module docs).
#[test]
#[ignore]
fn real_weights() {
    let (Some(model), Some(profile)) = (
        std::env::var_os("OXINFER_LORA_TRAIN_MODEL").map(PathBuf::from),
        std::env::var_os("OXINFER_LORA_TRAIN_PROFILE").map(PathBuf::from),
    ) else {
        eprintln!("OXINFER_LORA_TRAIN_MODEL / OXINFER_LORA_TRAIN_PROFILE are not set");
        return;
    };
    let m = load(&model);
    let paths = Paths::profile(&profile);
    let (eps, _) = load_episodes(&paths.episodes, &paths.outcomes, &[]).unwrap();
    let ad = Adapter::load(&m, &paths.adapter).unwrap();
    let tr = Trainer::new(&m, 0).unwrap();
    let mut with = load(&model);
    let spec = crate::lora::LoraSpec::parse(paths.adapter.to_str().unwrap()).unwrap();
    crate::lora::attach_specs(&mut with, &[spec], &model, false).unwrap();
    let mut t = eps[0].prompt.clone();
    t.extend_from_slice(&eps[0].completion);
    let (ours, theirs, base) = (tr.logits(Some(&ad), &t), model_logits(&with, &t), model_logits(&m, &t));
    let scale = theirs.iter().fold(0f32, |a, v| a.max(v.abs()));
    eprintln!(
        "{} ({}), {} tokens: trainer logits against the model with the adapter attached: max |diff| {:e}; \
         the adapter moves the logits by up to {:.3} (logits up to {scale:.2})",
        model.display(),
        weight_format(&m),
        t.len(),
        max_diff(&ours, &theirs),
        max_diff(&base, &theirs)
    );
    let obj = Objective { neg_clip: 4.0 };
    for e in eps.iter().take(2) {
        let (_, g) = batch(&tr, &ad, std::slice::from_ref(e), obj, true);
        let norm = g
            .iter()
            .flat_map(|(a, b)| a.iter().chain(b))
            .map(|v| *v as f64 * *v as f64)
            .sum::<f64>()
            .sqrt();
        // Along the normalized gradient the derivative is its norm.
        for h in [1e-3f64, 3e-3, 1e-2] {
            let shifted = |k: f64| {
                let mut a2 = ad.clone();
                for (md, gm) in a2.mods.iter_mut().zip(&g) {
                    md.a.iter_mut()
                        .zip(&gm.0)
                        .for_each(|(v, d)| *v += (k * h * *d as f64 / norm) as f32);
                    md.bt
                        .iter_mut()
                        .zip(&gm.1)
                        .for_each(|(v, d)| *v += (k * h * *d as f64 / norm) as f32);
                }
                batch(&tr, &a2, std::slice::from_ref(e), obj, false).0
            };
            let fd = (8.0 * (shifted(1.0) - shifted(-1.0)) - (shifted(2.0) - shifted(-2.0))) / (12.0 * h);
            eprintln!(
                "episode {} (weight {}): gradient norm {norm:.5}, finite difference along it (step {h}) {fd:.5}: relative error {:.2e}",
                e.id,
                e.weight,
                (fd - norm).abs() / norm
            );
        }
    }
}
