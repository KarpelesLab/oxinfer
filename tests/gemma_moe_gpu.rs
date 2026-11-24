//! A tiny random Gemma 4 MoE checkpoint (26B-A4B's layout: sliding layers with heads of 256,
//! global layers with heads of 512 and keys reused as values, a dense GELU MLP beside 64 routed
//! GELU experts with per-expert scales) run on the GPU against the exact path: the whole-pass
//! prefill plan (in steps shorter than the prompt, past the sliding window) and the GPU decode
//! steps must give the CPU's hidden states up to rounding: half-precision matrix units move the
//! tokens by 1e-3 to 3e-3, SiLU experts in the plan (an earlier bug) by up to 0.37. Without
//! matrix units there is no prefill plan and the test covers the per-operation path and decode.
//!
//! Its own test binary: it installs the process-wide GPU backend.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use oxinfer::config::TextConfig;
use oxinfer::model::{Model, Quant};
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

const H: usize = 256;
const VOCAB: usize = 512;
const LAYERS: usize = 4;
const N_EXP: usize = 64;
const MI: usize = 128;
/// Every expert routed (weights from the softmax over all of them, times the per-expert scales):
/// with a top 8 of 64, random routers put about one token in ten on a near-tie that rounding
/// flips, which hides everything else.
const TOP_K: usize = 64;

fn tiny_gemma_moe() -> PathBuf {
    tiny_gemma_moe_shared(0)
}

/// [`tiny_gemma_moe`] whose last `shared` layers read the caches of the last earlier layers of
/// their kind (Gemma 4 E2B / E4B's KV sharing).
fn tiny_gemma_moe_shared(shared: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxinfer-tiny-gemma-moe-{}-{shared}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (heads, kv, gkv, hd, ghd, inter) = (2usize, 1usize, 1usize, 256usize, 512usize, 256usize);
    let types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"];
    let cfg = serde_json::json!({
        "model_type": "gemma4_text", "hidden_size": H, "intermediate_size": inter, "num_hidden_layers": LAYERS,
        "num_attention_heads": heads, "num_key_value_heads": kv, "num_global_key_value_heads": gkv,
        "head_dim": hd, "global_head_dim": ghd, "attention_k_eq_v": true, "vocab_size": VOCAB,
        "hidden_size_per_layer_input": 0, "sliding_window": 48, "final_logit_softcapping": 30.0,
        "rms_norm_eps": 1e-6, "layer_types": types, "enable_moe_block": true, "num_experts": N_EXP,
        "top_k_experts": TOP_K, "moe_intermediate_size": MI, "num_kv_shared_layers": shared,
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let mut s = 7u64;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> { (0..n).map(|_| lcg(&mut s) * scale).collect() };
    let mut t = vec![("model.embed_tokens.weight".to_string(), vec![VOCAB, H], rnd(VOCAB * H, 0.2))];
    let norm = |rnd: &mut dyn FnMut(usize, f32) -> Vec<f32>, name: String, n: usize| {
        let v = rnd(n, 0.4).iter().map(|v| 1.0 + v).collect();
        (name, vec![n], v)
    };
    t.push(norm(&mut rnd, "model.norm.weight".into(), H));
    for (l, ty) in types.iter().enumerate() {
        let p = format!("model.layers.{l}.");
        let (d, n_kv) = if *ty == "sliding_attention" { (hd, kv) } else { (ghd, gkv) };
        let s_in = (H as f32).powf(-0.5) * 2.0;
        t.push((format!("{p}self_attn.q_proj.weight"), vec![heads * d, H], rnd(heads * d * H, s_in)));
        t.push((format!("{p}self_attn.k_proj.weight"), vec![n_kv * d, H], rnd(n_kv * d * H, s_in)));
        if *ty == "sliding_attention" {
            t.push((format!("{p}self_attn.v_proj.weight"), vec![n_kv * d, H], rnd(n_kv * d * H, s_in)));
        }
        t.push((format!("{p}self_attn.o_proj.weight"), vec![H, heads * d], rnd(H * heads * d, 0.1)));
        // Attention logits of a few units (Gemma 4 does not scale them): random queries and keys
        // of norm sqrt(d) would make every softmax nearly one-hot, flipped by rounding.
        let (name, shape, q_norm) = norm(&mut rnd, format!("{p}self_attn.q_norm.weight"), d);
        t.push((name, shape, q_norm.iter().map(|v| v * 0.1).collect()));
        t.push(norm(&mut rnd, format!("{p}self_attn.k_norm.weight"), d));
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_2",
        ] {
            t.push(norm(&mut rnd, format!("{p}{n}.weight"), H));
        }
        t.push((format!("{p}mlp.gate_proj.weight"), vec![inter, H], rnd(inter * H, s_in)));
        t.push((format!("{p}mlp.up_proj.weight"), vec![inter, H], rnd(inter * H, s_in)));
        t.push((format!("{p}mlp.down_proj.weight"), vec![H, inter], rnd(H * inter, 0.1)));
        t.push((format!("{p}router.proj.weight"), vec![N_EXP, H], rnd(N_EXP * H, 4.0)));
        t.push(norm(&mut rnd, format!("{p}router.scale"), H));
        t.push(norm(&mut rnd, format!("{p}router.per_expert_scale"), N_EXP));
        t.push((
            format!("{p}experts.gate_up_proj"),
            vec![N_EXP, 2 * MI, H],
            rnd(N_EXP * 2 * MI * H, s_in),
        ));
        t.push((format!("{p}experts.down_proj"), vec![N_EXP, H, MI], rnd(N_EXP * H * MI, 0.1)));
    }
    write_safetensors(&dir.join("model.safetensors"), &t);
    dir
}

/// A random PEFT adapter for [`tiny_gemma_moe`] in `dir`: attention query / output, the dense
/// MLP and the routed experts (one module per expert; layer 1 gate / up only, layer 3 down only).
fn tiny_adapter(dir: &Path) -> PathBuf {
    let ad = dir.join("adapter");
    std::fs::create_dir_all(&ad).unwrap();
    let (heads, hd, ghd, inter, r) = (2usize, 256usize, 512usize, 256usize, 4usize);
    let mut s = 99u64;
    let mut t = Vec::new();
    let mut pair = |name: String, n_out: usize, n_in: usize| {
        let a: Vec<f32> = (0..r * n_in).map(|_| lcg(&mut s) * 2.0 / (n_in as f32).sqrt()).collect();
        let b: Vec<f32> = (0..n_out * r).map(|_| lcg(&mut s) * 0.1).collect();
        t.push((format!("base_model.model.{name}.lora_A.weight"), vec![r, n_in], a));
        t.push((format!("base_model.model.{name}.lora_B.weight"), vec![n_out, r], b));
    };
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.");
        let d = if l % 2 == 0 { hd } else { ghd };
        pair(format!("{p}self_attn.q_proj"), heads * d, H);
        pair(format!("{p}self_attn.o_proj"), H, heads * d);
        pair(format!("{p}mlp.gate_proj"), inter, H);
        pair(format!("{p}mlp.up_proj"), inter, H);
        pair(format!("{p}mlp.down_proj"), H, inter);
        for e in 0..N_EXP {
            if l != 3 {
                pair(format!("{p}experts.{e}.gate_proj"), MI, H);
                pair(format!("{p}experts.{e}.up_proj"), MI, H);
            }
            if l != 1 {
                pair(format!("{p}experts.{e}.down_proj"), H, MI);
            }
        }
    }
    write_safetensors(&ad.join("adapter_model.safetensors"), &t);
    let cfg = serde_json::json!({ "peft_type": "LORA", "r": r, "lora_alpha": 2 * r, "target_modules": ["q_proj"] });
    std::fs::write(ad.join("adapter_config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    ad
}

/// Largest relative difference (L2) of a row of `a` from the same row of `b` (rows of `n`).
fn worst_row(a: &[f32], b: &[f32], n: usize) -> f32 {
    assert_eq!(a.len(), b.len());
    a.chunks(n)
        .zip(b.chunks(n))
        .map(|(x, y)| {
            let d: f32 = x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum();
            let n: f32 = y.iter().map(|q| q * q).sum();
            (d / n).sqrt()
        })
        .fold(0f32, f32::max)
}

#[test]
fn gpu_prefill_plan_and_decode_match_cpu() {
    // Prefill plan steps of 64 tokens: a 300-token prompt crosses four step boundaries.
    unsafe { std::env::set_var("OXINFER_PREFILL_TOKENS", "64") };
    let plat = oxinfer::platform::detect();
    let Ok(Some(_)) = oxinfer::gpu::init(&plat, oxinfer::platform::DevicePref::Auto) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let dir = tiny_gemma_moe();
    let cfg = TextConfig::load(&dir).unwrap();
    let files = ModelFiles::open(&dir, true).unwrap();
    let tokens: Vec<u32> = (0..300u32).map(|i| (i * 37 + 11) % VOCAB as u32).collect();
    let next = [5u32, 77, 300, 411];
    for quant in [Quant::Bf16, Quant::Q8, Quant::Q4] {
        let model = Model::load(cfg.clone(), &files, quant).unwrap();
        // Reference: no device plans, matmuls bit-identical to the CPU's.
        oxinfer::gpu::set_exact(true);
        oxinfer::gpu::set_decode(Some(false));
        // The decode steps' reference is the same pass over the prompt and the new tokens (one
        // f32 pass: the CPU's own decode steps run int8 activations under Q8 / Q4).
        let all: Vec<u32> = tokens.iter().chain(&next).copied().collect();
        let h_all = model.forward_hidden(&mut model.new_session(), &all);
        let h_ref = &h_all[..tokens.len() * H];
        let l_ref: Vec<Vec<f32>> = (tokens.len()..all.len())
            .map(|i| model.logits(&h_all[i * H..(i + 1) * H], 1))
            .collect();
        // The device path: matrix units (where the GPU has them), the prefill plan, GPU decode.
        oxinfer::gpu::set_exact(false);
        oxinfer::gpu::set_decode(Some(true));
        let mut s_gpu = model.new_session();
        let h_gpu = model.forward_hidden(&mut s_gpu, &tokens);
        let l_gpu: Vec<Vec<f32>> = next.iter().map(|&tk| model.forward(&mut s_gpu, &[tk])).collect();
        let prefill = worst_row(&h_gpu, h_ref, H);
        let decode = l_gpu.iter().zip(&l_ref).map(|(a, b)| worst_row(a, b, VOCAB)).fold(0f32, f32::max);
        eprintln!("{quant:?}: prefill worst token {prefill:.2e}, decode worst step {decode:.2e}");
        assert!(prefill < 1e-2, "{quant:?}: prefill hidden states differ by {prefill}");
        assert!(decode < 2e-2, "{quant:?}: decode logits differ by {decode}");
    }
    // A runtime LoRA adapter (dense layers and every expert), on the device's side paths, at two
    // strengths: the plans read the strength before every step.
    let ad = tiny_adapter(&dir);
    for quant in [Quant::Bf16, Quant::Q8] {
        let mut model = Model::load(cfg.clone(), &files, quant).unwrap();
        let spec = oxinfer::lora::LoraSpec {
            path: ad.clone(),
            scale: 1.0,
        };
        let att = oxinfer::lora::attach_specs(&mut model, &[spec], &dir, false).unwrap();
        let mut refs = Vec::new();
        for strength in [1.0f32, 0.4] {
            att[0].set_strength(strength).unwrap();
            oxinfer::gpu::set_exact(true);
            oxinfer::gpu::set_decode(Some(false));
            let all: Vec<u32> = tokens.iter().chain(&next).copied().collect();
            let h_all = model.forward_hidden(&mut model.new_session(), &all);
            let h_ref = &h_all[..tokens.len() * H];
            let l_ref: Vec<Vec<f32>> = (tokens.len()..all.len())
                .map(|i| model.logits(&h_all[i * H..(i + 1) * H], 1))
                .collect();
            oxinfer::gpu::set_exact(false);
            oxinfer::gpu::set_decode(Some(true));
            let mut s_gpu = model.new_session();
            let h_gpu = model.forward_hidden(&mut s_gpu, &tokens);
            let l_gpu: Vec<Vec<f32>> = next.iter().map(|&tk| model.forward(&mut s_gpu, &[tk])).collect();
            let prefill = worst_row(&h_gpu, h_ref, H);
            let decode = l_gpu.iter().zip(&l_ref).map(|(a, b)| worst_row(a, b, VOCAB)).fold(0f32, f32::max);
            eprintln!("{quant:?} + LoRA at {strength}: prefill worst token {prefill:.2e}, decode worst step {decode:.2e}");
            assert!(prefill < 1e-2, "{quant:?} + LoRA: prefill hidden states differ by {prefill}");
            assert!(decode < 2e-2, "{quant:?} + LoRA: decode logits differ by {decode}");
            refs.push(l_ref);
        }
        // The strengths give outputs further apart than the device is from the CPU.
        let apart = refs[0]
            .iter()
            .zip(&refs[1])
            .map(|(a, b)| worst_row(a, b, VOCAB))
            .fold(0f32, f32::max);
        assert!(apart > 0.1, "{quant:?}: the adapter's strength barely matters ({apart})");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// KV-sharing layers: a prompt's last token alone runs through them (the layers before them on
/// every token, across prefill plan steps), and the decode steps after it read the caches the
/// first layers filled; both must give the CPU's results up to rounding.
#[test]
fn gpu_prefill_skips_kv_shared_rows() {
    unsafe { std::env::set_var("OXINFER_PREFILL_TOKENS", "64") };
    let plat = oxinfer::platform::detect();
    let Ok(Some(_)) = oxinfer::gpu::init(&plat, oxinfer::platform::DevicePref::Auto) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let dir = tiny_gemma_moe_shared(2);
    let cfg = TextConfig::load(&dir).unwrap();
    assert_eq!(cfg.first_kv_shared_layer(), 2);
    let files = ModelFiles::open(&dir, true).unwrap();
    let tokens: Vec<u32> = (0..300u32).map(|i| (i * 37 + 11) % VOCAB as u32).collect();
    let next = [5u32, 77, 300, 411];
    for quant in [Quant::Bf16, Quant::Q8] {
        let model = Model::load(cfg.clone(), &files, quant).unwrap();
        oxinfer::gpu::set_exact(true);
        oxinfer::gpu::set_decode(Some(false));
        let all: Vec<u32> = tokens.iter().chain(&next).copied().collect();
        let h_all = model.forward_hidden(&mut model.new_session(), &all);
        let last_ref = &h_all[(tokens.len() - 1) * H..tokens.len() * H];
        let l_ref: Vec<Vec<f32>> = (tokens.len()..all.len())
            .map(|i| model.logits(&h_all[i * H..(i + 1) * H], 1))
            .collect();
        oxinfer::gpu::set_exact(false);
        oxinfer::gpu::set_decode(Some(true));
        let mut s_gpu = model.new_session();
        let last = model.prefill_hidden(&mut s_gpu, &tokens, &[], true);
        if oxinfer::gpu::active().is_some() {
            // (Only the last token's row: the shared layers ran on it alone.)
            assert_eq!(last.len(), H, "{quant:?}: the prefill did not skip the shared layers' rows");
        }
        assert_eq!(s_gpu.pos, tokens.len());
        let l_gpu: Vec<Vec<f32>> = next.iter().map(|&tk| model.forward(&mut s_gpu, &[tk])).collect();
        let prefill = worst_row(&last[last.len() - H..], last_ref, H);
        let decode = l_gpu.iter().zip(&l_ref).map(|(a, b)| worst_row(a, b, VOCAB)).fold(0f32, f32::max);
        eprintln!("{quant:?} (2 shared layers): prefill last token {prefill:.2e}, decode worst step {decode:.2e}");
        assert!(prefill < 1e-2, "{quant:?}: the last token's hidden state differs by {prefill}");
        assert!(decode < 2e-2, "{quant:?}: decode logits differ by {decode}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
