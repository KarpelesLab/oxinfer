//! Greedy decoding chained on the device (`TextLm::decode_greedy`, Metal's
//! `DecodePlan::greedy`) against one-token host steps with the host's argmax, on tiny random
//! checkpoints: a Llama (bf16, int8 and Q4 tables), a Gemma 4 with per-layer embeddings (the
//! second input row looked up on the device too) and a Qwen3.5 with Gated DeltaNet layers (the
//! recurrent states, whose passes alternate between two copies). The chain must give the host's
//! tokens, and the session after it the host's logits bit for bit: after a whole chain, after
//! a stop in the middle (the pass queued past it undone), after `Chain::Last`, past the device
//! KV copies' first growth, and with the passes limited.
//!
//! Its own test binary: it installs the process-wide GPU backend.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use oxinfer::generate::{Chain, TextLm};
use oxinfer::model::Quant;
use oxinfer::safetensors::ModelFiles;
use std::path::{Path, PathBuf};

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

/// `tensors` (name, shape, values) as one bf16 safetensors file.
fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, vals) in tensors {
        assert_eq!(vals.len(), shape.iter().product::<usize>(), "{name}");
        let start = data.len();
        for &v in vals {
            data.extend_from_slice(&((v.to_bits() + 0x8000) >> 16).to_le_bytes()[..2]);
        }
        header.insert(
            name.clone(),
            serde_json::json!({ "dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()] }),
        );
    }
    let mut json = serde_json::to_vec(&header).unwrap();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&json);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

/// One test at a time: `check` turns the chains off for a moment.
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

const H: usize = 256;
/// Not a multiple of the argmax's 1024-value blocks.
const VOCAB: usize = 1500;

fn tmp(tag: &str, cfg: serde_json::Value) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxinfer-gpu-chain-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    dir
}

type Tensors = Vec<(String, Vec<usize>, Vec<f32>)>;

fn rnd(s: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| lcg(s) * scale).collect()
}

fn norm(s: &mut u64, t: &mut Tensors, name: String, n: usize, offset: f32) {
    let v = rnd(s, n, 0.4).iter().map(|v| offset + v).collect();
    t.push((name, vec![n], v));
}

/// A random Llama (untied head).
fn tiny_llama() -> PathBuf {
    let (layers, heads, kv, inter) = (3usize, 4usize, 2usize, 512usize);
    let hd = H / heads;
    let dir = tmp(
        "llama",
        serde_json::json!({
            "model_type": "llama", "hidden_size": H, "intermediate_size": inter, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": kv, "vocab_size": VOCAB, "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false,
        }),
    );
    let mut s = 3u64;
    let si = 2.0 / (H as f32).sqrt();
    let mut t: Tensors = vec![("model.embed_tokens.weight".into(), vec![VOCAB, H], rnd(&mut s, VOCAB * H, 2.0))];
    t.push(("lm_head.weight".into(), vec![VOCAB, H], rnd(&mut s, VOCAB * H, si)));
    norm(&mut s, &mut t, "model.norm.weight".into(), H, 1.0);
    for l in 0..layers {
        let p = format!("model.layers.{l}.");
        for (n, o, i) in [("q", heads * hd, H), ("k", kv * hd, H), ("v", kv * hd, H), ("o", H, heads * hd)] {
            t.push((format!("{p}self_attn.{n}_proj.weight"), vec![o, i], rnd(&mut s, o * i, si)));
        }
        for (n, o, i) in [("gate", inter, H), ("up", inter, H), ("down", H, inter)] {
            t.push((format!("{p}mlp.{n}_proj.weight"), vec![o, i], rnd(&mut s, o * i, si)));
        }
        for n in ["input_layernorm", "post_attention_layernorm"] {
            norm(&mut s, &mut t, format!("{p}{n}.weight"), H, 1.0);
        }
    }
    write_safetensors(&dir.join("model.safetensors"), &t);
    dir
}

/// A random dense Gemma 4 with per-layer embeddings (sliding and full layers).
fn tiny_gemma() -> PathBuf {
    let (heads, kv, hd, ghd, inter, p) = (2usize, 1usize, 256usize, 512usize, 512usize, 64usize);
    let types = ["sliding_attention", "sliding_attention", "full_attention"];
    let nl = types.len();
    let dir = tmp(
        "gemma",
        serde_json::json!({
            "model_type": "gemma4_text", "hidden_size": H, "intermediate_size": inter, "num_hidden_layers": nl,
            "num_attention_heads": heads, "num_key_value_heads": kv, "num_global_key_value_heads": kv,
            "head_dim": hd, "global_head_dim": ghd, "vocab_size": VOCAB, "vocab_size_per_layer_input": VOCAB,
            "hidden_size_per_layer_input": p, "sliding_window": 16, "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6, "layer_types": types,
        }),
    );
    let mut s = 5u64;
    let si = 2.0 / (H as f32).sqrt();
    let mut t: Tensors = vec![("model.embed_tokens.weight".into(), vec![VOCAB, H], rnd(&mut s, VOCAB * H, 0.02))];
    t.push((
        "model.embed_tokens_per_layer.weight".into(),
        vec![VOCAB, nl * p],
        rnd(&mut s, VOCAB * nl * p, 2.0),
    ));
    t.push((
        "model.per_layer_model_projection.weight".into(),
        vec![nl * p, H],
        rnd(&mut s, nl * p * H, si),
    ));
    norm(&mut s, &mut t, "model.per_layer_projection_norm.weight".into(), p, 1.0);
    norm(&mut s, &mut t, "model.norm.weight".into(), H, 1.0);
    for (l, ty) in types.iter().enumerate() {
        let pf = format!("model.layers.{l}.");
        let d = if *ty == "sliding_attention" { hd } else { ghd };
        t.push((
            format!("{pf}self_attn.q_proj.weight"),
            vec![heads * d, H],
            rnd(&mut s, heads * d * H, si),
        ));
        t.push((format!("{pf}self_attn.k_proj.weight"), vec![kv * d, H], rnd(&mut s, kv * d * H, si)));
        t.push((format!("{pf}self_attn.v_proj.weight"), vec![kv * d, H], rnd(&mut s, kv * d * H, si)));
        t.push((
            format!("{pf}self_attn.o_proj.weight"),
            vec![H, heads * d],
            rnd(&mut s, H * heads * d, 1.0),
        ));
        // Small attention logits (Gemma 4 does not scale them).
        let q: Vec<f32> = rnd(&mut s, d, 0.4).iter().map(|v| 0.1 * (1.0 + v)).collect();
        t.push((format!("{pf}self_attn.q_norm.weight"), vec![d], q));
        norm(&mut s, &mut t, format!("{pf}self_attn.k_norm.weight"), d, 1.0);
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_per_layer_input_norm",
        ] {
            norm(&mut s, &mut t, format!("{pf}{n}.weight"), H, 1.0);
        }
        for (n, o, i) in [("gate", inter, H), ("up", inter, H), ("down", H, inter)] {
            t.push((format!("{pf}mlp.{n}_proj.weight"), vec![o, i], rnd(&mut s, o * i, si)));
        }
        t.push((format!("{pf}per_layer_input_gate.weight"), vec![p, H], rnd(&mut s, p * H, si)));
        t.push((format!("{pf}per_layer_projection.weight"), vec![H, p], rnd(&mut s, H * p, 0.2)));
    }
    write_safetensors(&dir.join("model.safetensors"), &t);
    dir
}

/// A random Qwen3.5 (three Gated DeltaNet layers, then a gated full-attention layer).
fn tiny_qwen35() -> PathBuf {
    let (nl, heads, kv, hd, inter) = (4usize, 2usize, 1usize, 128usize, 512usize);
    let (nk, nv, dk, dv, kc) = (2usize, 2usize, 128usize, 128usize, 4usize);
    let types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"];
    let dir = tmp(
        "qwen35",
        serde_json::json!({
            "model_type": "qwen3_5_text", "hidden_size": H, "intermediate_size": inter, "num_hidden_layers": nl,
            "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd, "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6, "layer_types": types, "full_attention_interval": 4, "attn_output_gate": true,
            "linear_conv_kernel_dim": kc, "linear_key_head_dim": dk, "linear_value_head_dim": dv,
            "linear_num_key_heads": nk, "linear_num_value_heads": nv, "tie_word_embeddings": true,
            "rope_parameters": { "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25 },
        }),
    );
    let mut s = 7u64;
    let si = 2.0 / (H as f32).sqrt();
    let mut t: Tensors = vec![("model.embed_tokens.weight".into(), vec![VOCAB, H], rnd(&mut s, VOCAB * H, 0.05))];
    // Zero-centred norms (1 + w).
    norm(&mut s, &mut t, "model.norm.weight".into(), H, 0.0);
    let conv_dim = 2 * nk * dk + nv * dv;
    for (l, ty) in types.iter().enumerate() {
        let p = format!("model.layers.{l}.");
        if *ty == "full_attention" {
            t.push((
                format!("{p}self_attn.q_proj.weight"),
                vec![2 * heads * hd, H],
                rnd(&mut s, 2 * heads * hd * H, si),
            ));
            t.push((
                format!("{p}self_attn.k_proj.weight"),
                vec![kv * hd, H],
                rnd(&mut s, kv * hd * H, si),
            ));
            t.push((
                format!("{p}self_attn.v_proj.weight"),
                vec![kv * hd, H],
                rnd(&mut s, kv * hd * H, si),
            ));
            t.push((
                format!("{p}self_attn.o_proj.weight"),
                vec![H, heads * hd],
                rnd(&mut s, H * heads * hd, 1.0),
            ));
            norm(&mut s, &mut t, format!("{p}self_attn.q_norm.weight"), hd, 0.0);
            norm(&mut s, &mut t, format!("{p}self_attn.k_norm.weight"), hd, 0.0);
        } else {
            let g = format!("{p}linear_attn.");
            t.push((format!("{g}in_proj_qkv.weight"), vec![conv_dim, H], rnd(&mut s, conv_dim * H, si)));
            t.push((format!("{g}in_proj_z.weight"), vec![nv * dv, H], rnd(&mut s, nv * dv * H, si)));
            t.push((format!("{g}in_proj_b.weight"), vec![nv, H], rnd(&mut s, nv * H, si)));
            t.push((format!("{g}in_proj_a.weight"), vec![nv, H], rnd(&mut s, nv * H, si)));
            t.push((format!("{g}conv1d.weight"), vec![conv_dim, 1, kc], rnd(&mut s, conv_dim * kc, 1.0)));
            t.push((format!("{g}A_log"), vec![nv], rnd(&mut s, nv, 1.0)));
            t.push((format!("{g}dt_bias"), vec![nv], rnd(&mut s, nv, 1.0)));
            norm(&mut s, &mut t, format!("{g}norm.weight"), dv, 1.0);
            t.push((format!("{g}out_proj.weight"), vec![H, nv * dv], rnd(&mut s, H * nv * dv, 1.0)));
        }
        for (n, o, i) in [("gate", inter, H), ("up", inter, H), ("down", H, inter)] {
            t.push((format!("{p}mlp.{n}_proj.weight"), vec![o, i], rnd(&mut s, o * i, si)));
        }
        for n in ["input_layernorm", "post_attention_layernorm"] {
            norm(&mut s, &mut t, format!("{p}{n}.weight"), H, 0.0);
        }
    }
    write_safetensors(&dir.join("model.safetensors"), &t);
    dir
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The chain against host steps on `model` (see the module docs); `n` passes in the long run.
fn check<M: TextLm>(name: &str, model: &M, n: usize)
where
    M::Sess: Clone,
{
    let prompt: Vec<u32> = (0..9u32).map(|i| (i * 211 + 17) % VOCAB as u32).collect();
    let mut base = model.new_session();
    let logits = model.forward(&mut base, &prompt);
    let first = oxinfer::sampler::argmax(&logits);
    // Host steps: toks[i] is fed at step i, `after[i]` the logits after it.
    let mut host = base.clone();
    let (mut toks, mut after) = (vec![first], Vec::new());
    for i in 0..=n {
        let l = model.forward(&mut host, &[toks[i]]);
        toks.push(oxinfer::sampler::argmax(&l));
        after.push(l);
    }
    let distinct = toks.iter().collect::<std::collections::BTreeSet<_>>().len();
    assert!(distinct > 3, "{name}: degenerate reference ({distinct} distinct tokens)");
    // A chain of `max` passes stopping (`Chain::Stop`) or ending (`Chain::Last`) at its `at`-th
    // token: the tokens it gave, then the host step after it against the reference.
    let run = |max: usize, at: Option<(usize, Chain)>| {
        let mut sess = base.clone();
        let mut got: Vec<u32> = Vec::new();
        let mut on = |t: u32| {
            got.push(t);
            match at {
                Some((k, c)) if got.len() == k => c,
                _ => Chain::Go,
            }
        };
        let kept = model
            .decode_greedy(&mut sess, first, max, &mut on)
            .unwrap_or_else(|| panic!("{name}: no chain"));
        assert_eq!(got.len(), kept, "{name} {max} {at:?}: tokens given");
        assert_eq!(got[..], toks[1..=kept], "{name} {max} {at:?}: tokens");
        assert_eq!(model.session_len(&sess), prompt.len() + kept, "{name}: session length");
        // The next host step: the kept passes' caches and states exactly, the undone pass gone.
        let next = got[kept - 1];
        let l = model.forward(&mut sess, &[next]);
        assert_eq!(bits(&l), bits(&after[kept]), "{name} {max} {at:?}: logits after {kept} passes");
    };
    run(n, None);
    run(n, Some((7, Chain::Stop)));
    run(n, Some((8, Chain::Stop)));
    run(n, Some((1, Chain::Stop)));
    run(n, Some((5, Chain::Last)));
    run(3, None);
    run(1, None);
    // Off: nothing runs.
    oxinfer::gpu::set_decode_chain(Some(false));
    let mut sess = base.clone();
    assert!(
        model.decode_greedy(&mut sess, first, 4, &mut |_| Chain::Go).is_none(),
        "{name}: chain off"
    );
    assert_eq!(model.session_len(&sess), prompt.len());
    oxinfer::gpu::set_decode_chain(None);
}

#[test]
fn chained_greedy_matches_host_steps() {
    let _one = ONE.lock().unwrap_or_else(|p| p.into_inner());
    let plat = oxinfer::platform::detect();
    let Ok(Some(_)) = oxinfer::gpu::init(&plat, oxinfer::platform::DevicePref::Auto) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let dir = tiny_llama();
    let files = ModelFiles::open(&dir, true).unwrap();
    for quant in [Quant::Bf16, Quant::Q8, Quant::Q4] {
        let cfg = oxinfer::config::QwenConfig::load(&dir).unwrap();
        let model = oxinfer::qwen::QwenModel::load(cfg, &files, quant).unwrap();
        // Past the device KV copies' first 256 positions.
        check(&format!("llama {quant:?}"), &model, if quant == Quant::Bf16 { 300 } else { 40 });
    }
    std::fs::remove_dir_all(&dir).unwrap();

    let dir = tiny_gemma();
    let files = ModelFiles::open(&dir, true).unwrap();
    for quant in [Quant::Bf16, Quant::Q8] {
        let cfg = oxinfer::config::TextConfig::load(&dir).unwrap();
        let model = oxinfer::model::Model::load(cfg, &files, quant).unwrap();
        check(&format!("gemma {quant:?}"), &model, 40);
    }
    std::fs::remove_dir_all(&dir).unwrap();

    let dir = tiny_qwen35();
    let files = ModelFiles::open(&dir, true).unwrap();
    let cfg = oxinfer::config::QwenConfig::load(&dir).unwrap();
    let model = oxinfer::qwen::QwenModel::load(cfg, &files, Quant::Bf16).unwrap();
    check("qwen3.5", &model, 40);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Qwen3.5 prompt checkpoints on the device: a session goes back to a checkpoint after a reply
/// decoded on the GPU (one-token steps and a chain, whose recurrent states are updated in place
/// and whose KV caches have device copies) and from there gives the logits of a fresh session
/// that ran the same passes, bit for bit (dense and MoE); the MTP decoder (verification passes
/// on the GPU) the same logits and greedy reply.
#[test]
fn qwen35_checkpoints_on_the_device() {
    let _one = ONE.lock().unwrap_or_else(|p| p.into_inner());
    use oxinfer::qwen::{QwenModel, QwenSession};
    let plat = oxinfer::platform::detect();
    let Ok(Some(_)) = oxinfer::gpu::init(&plat, oxinfer::platform::DevicePref::Auto) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let toks = |n: u32, seed: u32| -> Vec<u32> { (0..n).map(|i| (i * 37 + seed * 11 + 5) % 256).collect() };
    // Prefill, then one-token steps and a chain from the last logits: the logits after each.
    let run = |m: &QwenModel, s: &mut QwenSession, ids: &[u32]| -> Vec<Vec<u32>> {
        let l = m.forward(s, ids);
        let mut t = oxinfer::sampler::argmax(&l);
        let mut out = vec![bits(&l)];
        for _ in 0..3 {
            let l = m.forward(s, &[t]);
            t = oxinfer::sampler::argmax(&l);
            out.push(bits(&l));
        }
        let mut last = t;
        let n = m.decode_greedy(s, t, 12, &mut |x| {
            last = x;
            Chain::Go
        });
        assert_eq!(n, Some(12), "no chain");
        out.push(bits(&m.forward(s, &[last])));
        out
    };
    for moe in [false, true] {
        let dir = std::env::temp_dir().join(format!("oxinfer-gpu-chain-q35cp-{moe}-{}", std::process::id()));
        oxinfer::qwen::tiny::write(&dir, moe, true).unwrap();
        let files = ModelFiles::open(&dir, true).unwrap();
        let cfg = oxinfer::config::QwenConfig::load(&dir).unwrap();
        let m = QwenModel::load(cfg, &files, Quant::Bf16).unwrap();
        let (a, b, c) = (toks(40, 1), toks(9, 2), toks(7, 3));
        let mut s = m.new_session();
        m.forward(&mut s, &a[..20]);
        s.checkpoint(2);
        m.forward(&mut s, &a[20..39]);
        s.checkpoint(2);
        run(&m, &mut s, &a[39..]);
        s.cut_to(39).unwrap();
        let got = run(&m, &mut s, &b);
        let mut f = m.new_session();
        m.forward(&mut f, &a[..20]);
        m.forward(&mut f, &a[20..39]);
        assert_eq!(got, run(&m, &mut f, &b), "moe {moe}: back to the prompt");
        s.cut_to(20).unwrap();
        let got = run(&m, &mut s, &c);
        let mut f = m.new_session();
        m.forward(&mut f, &a[..20]);
        assert_eq!(got, run(&m, &mut f, &c), "moe {moe}: back to the prefix");

        // MTP.
        let mtp_layer = m.load_mtp(&files).unwrap();
        let opts = oxinfer::qwen_mtp::MtpOptions {
            draft_tokens: 3,
            stop: Vec::new(),
            ..Default::default()
        };
        let reply = |d: &mut oxinfer::qwen_mtp::MtpDecoder, l: &[f32], n: usize| {
            let mut g = oxinfer::sampler::Sampler::new(0.0, 0, 1.0, 1);
            let mut out = vec![oxinfer::sampler::argmax(l)];
            d.push(out[0]);
            while out.len() < n {
                out.extend(d.step(&mut g, n - out.len()));
            }
            out
        };
        let mut d = oxinfer::qwen_mtp::MtpDecoder::new(&m, &mtp_layer, opts.clone());
        d.prefill(&a[..20], 512);
        d.checkpoint(2);
        d.prefill(&a[20..39], 512);
        d.checkpoint(2);
        let l = d.prefill(&a[39..], 512);
        reply(&mut d, &l, 30);
        d.cut_to(39).unwrap();
        let got = d.prefill(&b, 512);
        let got_reply = reply(&mut d, &got, 20);
        let mut f = oxinfer::qwen_mtp::MtpDecoder::new(&m, &mtp_layer, opts.clone());
        f.prefill(&a[..20], 512);
        f.prefill(&a[20..39], 512);
        let want = f.prefill(&b, 512);
        assert_eq!(bits(&got), bits(&want), "moe {moe}: MTP");
        assert_eq!(got_reply, reply(&mut f, &want, 20), "moe {moe}: MTP reply");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Plain sampling chained on the device (`Inkling2Model::decode_sampled`, `ink_samp`: a
/// temperature draw over the whole vocabulary in token order from a uniform the host draws per
/// pass) against one-token host steps with the host's `Sampler::sample` on a tiny Inkling-2:
/// the same tokens, the sampler's generator left where the host's is, the session after a
/// stop the host's logits (to the rounding of the chain's embedding norm).
#[test]
fn inkling_sampled_chain_matches_host_steps() {
    let _one = ONE.lock().unwrap_or_else(|p| p.into_inner());
    use oxinfer::inkling::{Inkling2Config, Inkling2Model};
    use oxinfer::sampler::Sampler;
    let plat = oxinfer::platform::detect();
    let Ok(Some(_)) = oxinfer::gpu::init(&plat, oxinfer::platform::DevicePref::Auto) else {
        eprintln!("no GPU: skipped");
        return;
    };
    oxinfer::gpu::set_decode_chain(Some(true));
    let dir = std::env::temp_dir().join(format!("oxinfer-gpu-chain-inkling-{}", std::process::id()));
    oxinfer::inkling::tiny::write_tiny_metal(&dir, 4).unwrap();
    let files = ModelFiles::open(&dir, true).unwrap();
    let m = Inkling2Model::load(Inkling2Config::load(&dir).unwrap(), &files, Quant::Bf16).unwrap();
    let prompt: Vec<u32> = (0..13u32).map(|i| (i * 29 + 3) % 250).collect();
    let mut base = m.new_session();
    let l0 = m.forward(&mut base, &prompt);
    let n = 24;
    for temp in [0.7f32, 1.0, 1.6] {
        let mut hs = Sampler::new(temp, 0, 1.0, 9);
        let first = hs.sample(&l0);
        let mut host = base.clone();
        let (mut toks, mut after) = (vec![first], Vec::new());
        for i in 0..=n {
            let l = m.forward(&mut host, &[toks[i]]);
            toks.push(hs.sample(&l));
            after.push(l);
        }
        let distinct = toks.iter().collect::<std::collections::BTreeSet<_>>().len();
        assert!(distinct > 3, "degenerate reference ({distinct} distinct tokens)");
        for stop_at in [n, 9, 1] {
            let mut ds = Sampler::new(temp, 0, 1.0, 9);
            assert_eq!(ds.sample(&l0), first);
            let mut sess = base.clone();
            let got = std::cell::RefCell::new(Vec::new());
            let mut on = |t: u32| {
                got.borrow_mut().push(t);
                if got.borrow().len() == stop_at { Chain::Stop } else { Chain::Go }
            };
            let Some(kept) = oxinfer::generate::decode_chain(&m, &mut sess, &mut ds, first, n, &mut on, &|| got.borrow().len()) else {
                if m.decode_greedy(&mut base.clone(), first, 2, &mut |_| Chain::Go).is_none() {
                    eprintln!("no Inkling-2 chain on this device: skipped");
                    std::fs::remove_dir_all(&dir).unwrap();
                    return;
                }
                panic!("greedy chains but sampling does not");
            };
            let got = got.into_inner();
            assert_eq!(kept, stop_at, "temp {temp}: passes kept");
            assert_eq!(got[..], toks[1..=kept], "temp {temp}, stop at {stop_at}: tokens");
            // The generator where the host's is after as many draws.
            let mut hr = Sampler::new(temp, 0, 1.0, 9);
            hr.set_state(hr.state(), kept + 1);
            assert_eq!(ds.state(), hr.state(), "temp {temp}: generator state");
            // (The chain norms its embedding rows on the device, in another order than the
            // host: the logits agree to rounding.)
            let l = m.forward(&mut sess, &[got[kept - 1]]);
            let d = l.iter().zip(&after[kept]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(d < 1e-3, "temp {temp}: logits after {kept} passes differ by {d}");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Sampling with top-k, top-p, min-p and a repetition penalty chained on the device
/// (`ink_fsamp`, the penalty applied in `ink_softcap` and its token set kept on the device)
/// against one-token host steps with `Sampler::sample` on a tiny Inkling-2: the same tokens and
/// generator state, with the device choosing and with every cut draw left to the host (whose
/// tokens the chain then goes on from when they are not the device's guess).
#[test]
fn inkling_cut_sampling_chain_matches_host_steps() {
    let _one = ONE.lock().unwrap_or_else(|p| p.into_inner());
    use oxinfer::inkling::{Inkling2Config, Inkling2Model};
    use oxinfer::sampler::Sampler;
    let plat = oxinfer::platform::detect();
    let Ok(Some(_)) = oxinfer::gpu::init(&plat, oxinfer::platform::DevicePref::Auto) else {
        eprintln!("no GPU: skipped");
        return;
    };
    oxinfer::gpu::set_decode_chain(Some(true));
    let dir = std::env::temp_dir().join(format!("oxinfer-gpu-chain-inkling-cut-{}", std::process::id()));
    oxinfer::inkling::tiny::write_tiny_metal(&dir, 4).unwrap();
    let files = ModelFiles::open(&dir, true).unwrap();
    let m = Inkling2Model::load(Inkling2Config::load(&dir).unwrap(), &files, Quant::Bf16).unwrap();
    let prompt: Vec<u32> = (0..13u32).map(|i| (i * 29 + 3) % 250).collect();
    let mut base = m.new_session();
    let l0 = m.forward(&mut base, &prompt);
    let n = 24;
    // (temperature, top-k, top-p, min-p, repetition penalty)
    let settings = [
        (1.0f32, 40usize, 0.95f32, 0.0f32, 1.0f32),
        (1.3, 0, 0.9, 0.0, 1.0),
        (0.9, 0, 1.0, 0.05, 1.0),
        (1.0, 20, 0.8, 0.02, 1.3),
        (1.0, 0, 1.0, 0.0, 1.5),
        (0.0, 0, 1.0, 0.0, 1.4),
    ];
    let sampler = |s: &(f32, usize, f32, f32, f32)| {
        let mut x = Sampler::new(s.0, s.1, s.2, 9).with_min_p(s.3).with_repetition_penalty(s.4);
        x.observe(&prompt);
        x
    };
    for s in &settings {
        let mut hs = sampler(s);
        let first = hs.sample(&l0);
        let mut host = base.clone();
        let mut toks = vec![first];
        for i in 0..=n {
            let l = m.forward(&mut host, &[toks[i]]);
            toks.push(hs.sample(&l));
        }
        for cap in [None, Some(1)] {
            oxinfer::inkling::set_device_sample_cap(cap);
            for stop_at in [n, 7] {
                let mut ds = sampler(s);
                assert_eq!(ds.sample(&l0), first);
                let mut sess = base.clone();
                let got = std::cell::RefCell::new(Vec::new());
                let mut on = |t: u32| {
                    got.borrow_mut().push(t);
                    if got.borrow().len() == stop_at { Chain::Stop } else { Chain::Go }
                };
                let Some(kept) = oxinfer::generate::decode_chain(&m, &mut sess, &mut ds, first, n, &mut on, &|| got.borrow().len()) else {
                    if m.decode_greedy(&mut base.clone(), first, 2, &mut |_| Chain::Go).is_none() {
                        eprintln!("no Inkling-2 chain on this device: skipped");
                        oxinfer::inkling::set_device_sample_cap(None);
                        std::fs::remove_dir_all(&dir).unwrap();
                        return;
                    }
                    panic!("{s:?}: greedy chains but sampling does not");
                };
                let got = got.into_inner();
                assert_eq!(kept, stop_at, "{s:?} cap {cap:?}: passes kept");
                assert_eq!(got[..], toks[1..=kept], "{s:?} cap {cap:?}, stop at {stop_at}: tokens");
                assert_eq!(sess.pos, base.pos + kept, "{s:?} cap {cap:?}: session length");
                // The generator and the penalty's tokens where the host's are after as many
                // tokens.
                let mut hr = sampler(s);
                hr.sample(&l0);
                let mut hsess = base.clone();
                let mut t = first;
                for _ in 0..kept {
                    let l = m.forward(&mut hsess, &[t]);
                    t = hr.sample(&l);
                }
                assert_eq!(ds.state(), hr.state(), "{s:?} cap {cap:?}: generator state");
                assert_eq!(ds.seen_words(), hr.seen_words(), "{s:?} cap {cap:?}: tokens seen");
            }
        }
    }
    oxinfer::inkling::set_device_sample_cap(None);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A batched step of several sessions (`TextLm::forward_batch`, Metal's `DecodePlan::step_rows`:
/// the server's concurrent requests) against each session's own one-token steps: the same
/// greedy tokens, logits within rounding (bit for bit where the multi-token kernels are exact,
/// which this prints but does not require, as that depends on the chip's kernels).
fn check_rows<M: TextLm>(name: &str, model: &M, steps: usize)
where
    M::Sess: Clone,
{
    let lens = [3usize, 9, 17, 40, 250];
    let mut alone: Vec<M::Sess> = Vec::new();
    let mut toks: Vec<u32> = Vec::new();
    for (i, &n) in lens.iter().enumerate() {
        let prompt: Vec<u32> = (0..n as u32).map(|j| (j * 97 + 13 * i as u32 + 5) % VOCAB as u32).collect();
        let mut s = model.new_session();
        let l = model.forward(&mut s, &prompt);
        toks.push(oxinfer::sampler::argmax(&l));
        alone.push(s);
    }
    let mut batch: Vec<M::Sess> = alone.clone();
    let (mut exact, mut worst) = (true, 0f32);
    for step in 0..steps {
        let want: Vec<Vec<f32>> = alone.iter_mut().zip(&toks).map(|(s, &t)| model.forward(s, &[t])).collect();
        let mut refs: Vec<&mut M::Sess> = batch.iter_mut().collect();
        let got = model.forward_batch(&mut refs, &toks);
        for (j, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.len(), w.len());
            exact &= bits(g) == bits(w);
            let scale = w.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            let d = g.iter().zip(w).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
            worst = worst.max(d);
            assert!(d < 1e-3, "{name} step {step} row {j}: logits differ by {d}");
            assert_eq!(
                oxinfer::sampler::argmax(g),
                oxinfer::sampler::argmax(w),
                "{name} step {step} row {j}"
            );
        }
        for (j, s) in alone.iter().enumerate() {
            assert_eq!(model.session_len(s), model.session_len(&batch[j]));
        }
        toks = want.iter().map(|l| oxinfer::sampler::argmax(l)).collect();
    }
    eprintln!(
        "{name}: batched rows {} (worst relative difference {worst:e})",
        if exact { "bit-identical" } else { "within rounding" }
    );
}

#[test]
fn batched_rows_match_one_token_steps() {
    let _one = ONE.lock().unwrap_or_else(|p| p.into_inner());
    let plat = oxinfer::platform::detect();
    let Ok(Some(_)) = oxinfer::gpu::init(&plat, oxinfer::platform::DevicePref::Auto) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let dir = tiny_llama();
    let files = ModelFiles::open(&dir, true).unwrap();
    for quant in [Quant::Bf16, Quant::Q8, Quant::Q4] {
        let cfg = oxinfer::config::QwenConfig::load(&dir).unwrap();
        let model = oxinfer::qwen::QwenModel::load(cfg, &files, quant).unwrap();
        // (The longest prompt's device copies grow past 256 positions on the way.)
        check_rows(&format!("llama {quant:?}"), &model, 12);
    }
    std::fs::remove_dir_all(&dir).unwrap();

    let dir = tiny_gemma();
    let files = ModelFiles::open(&dir, true).unwrap();
    for quant in [Quant::Bf16, Quant::Q8] {
        let cfg = oxinfer::config::TextConfig::load(&dir).unwrap();
        let model = oxinfer::model::Model::load(cfg, &files, quant).unwrap();
        check_rows(&format!("gemma {quant:?}"), &model, 12);
    }
    std::fs::remove_dir_all(&dir).unwrap();

    // Gated DeltaNet layers: each row on its own recurrent states.
    let dir = tiny_qwen35();
    let files = ModelFiles::open(&dir, true).unwrap();
    for quant in [Quant::Bf16, Quant::Q8, Quant::Q4] {
        let cfg = oxinfer::config::QwenConfig::load(&dir).unwrap();
        let model = oxinfer::qwen::QwenModel::load(cfg, &files, quant).unwrap();
        check_rows(&format!("qwen3.5 {quant:?}"), &model, 12);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
