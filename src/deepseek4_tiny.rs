//! Tiny random DeepSeek V4 and V4.1 checkpoints in DeepSeek's naming (float32 noise from a
//! fixed-seed generator: deterministic, meaningless), for tests that run the model without a
//! real one and without Python ([`write_v4`], [`write_v41`]), and, with a byte-level
//! `tokenizer.json` carrying the chat format's tokens, through the engine and the server
//! ([`write_chat`]). `tools/deepseek4_tiny.py` makes checkpoints checked against DeepSeek's
//! code; these only need to run.
//!
//! - V4 ([`v4_json`]): four layers of 16 (2 heads of 8, a window of 4) with a hash layer, a CSA
//!   layer (rate 4) whose indexer chooses, a heavily compressed layer (rate 8) and a window-only
//!   one; 4 experts of which 2 per token.
//! - V4.1 ([`v41_json`]): six layers with key sources at rates 2 and 1, an index layer on another
//!   layer's keys, candidate blocks from layer 3 used by layer 4, engram lookups before two
//!   layers.

use crate::config::DeepSeek4Config;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;

/// The chat format's tokens, from id [`SPECIAL_BASE`] in [`write_chat`]'s tokenizer.
pub const SPECIAL: [&str; 7] = [
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜System｜>",
    "<think>",
    "</think>",
];
/// The id of the first special token.
pub const SPECIAL_BASE: u32 = 256;
/// The vocabulary of [`write_chat`]'s checkpoints: the bytes and [`SPECIAL`].
pub const CHAT_VOCAB: usize = 256 + SPECIAL.len();

/// The `config.json` of the tiny V4 model; `over`: keys replaced.
pub fn v4_json(over: Value) -> Value {
    let ratios = [0usize, 4, 8, 0];
    let mut c = json!({
        "model_type": "deepseek_v4",
        "hidden_size": 16, "num_hidden_layers": ratios.len(), "num_attention_heads": 2, "head_dim": 8,
        "qk_rope_head_dim": 4, "q_lora_rank": 8, "o_groups": 2, "o_lora_rank": 4, "vocab_size": 32,
        "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
        "routed_scaling_factor": 1.5, "swiglu_limit": 1.5, "num_hash_layers": 1, "sliding_window": 4,
        "compress_ratios": ratios, "index_n_heads": 2, "index_head_dim": 8, "index_topk": 2
    });
    for (k, v) in over.as_object().into_iter().flatten() {
        c[k] = v.clone();
    }
    c
}

/// The `config.json` of the tiny V4.1 model; `over`: keys of `text_config` replaced.
pub fn v41_json(over: Value) -> Value {
    let mut c = json!({
        "model_type": "deepseek_v41",
        "text_config": {
            "hidden_size": 16, "num_hidden_layers": 6, "num_attention_heads": 2, "head_dim": 8,
            "qk_rope_head_dim": 4, "q_lora_rank": 8, "o_groups": 2, "o_lora_rank": 4, "vocab_size": 32,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
            "routed_scaling_factor": 1.5, "swiglu_limit": 1.5, "sliding_window": 4, "rms_norm_eps": 1e-20,
            "compress_ratios": [0, 2, 2, 1, 1, 0], "kv_source_layer_ids": [1, 3],
            "index_source_layer_ids": [1, 2, 3, 4], "index_n_heads": 8, "index_head_dim": 8, "index_topk": 2,
            "candidate_source_layer_id": 3, "candidate_topk_blocks": 2, "candidate_block_size": 2,
            "engram_layer_ids": [1, 3], "engram_max_ngram_size": 3, "engram_vocab_size": 20,
            "engram_n_heads": 2, "engram_head_dim": 4, "engram_pad_token_id": 2,
            "engram_multipliers": [[7, 11, 13], [17, 19, 23]],
            "engram_token_map": (0..32).map(|i| i % 20).collect::<Vec<i32>>(),
        }
    });
    for (k, v) in over.as_object().into_iter().flatten() {
        c["text_config"][k] = v.clone();
    }
    c
}

/// A deterministic uniform generator in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// Tensors being written: name, dtype, shape, little-endian bytes.
struct Tensors {
    list: Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
    rng: Rng,
}

impl Tensors {
    /// `scale` times uniform noise, plus `bias`.
    fn add(&mut self, name: String, shape: &[usize], scale: f32, bias: f32) {
        let n: usize = shape.iter().product();
        let bytes = (0..n).flat_map(|_| (bias + scale * self.rng.next()).to_le_bytes()).collect();
        self.list.push((name, "F32", shape.to_vec(), bytes));
    }

    fn write(self, dir: &Path) -> Result<()> {
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in self.list {
            let entry = json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]});
            head.insert(name, entry);
            data.extend(bytes);
        }
        let mut head = Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
        let path = dir.join("model.safetensors");
        std::fs::write(&path, file).with_context(|| path.display().to_string())
    }
}

/// A matrix's noise scale: unit-variance outputs for unit-variance inputs, times `gain`.
fn mat(fan_in: usize, gain: f32) -> f32 {
    gain * (3.0 / fan_in as f32).sqrt()
}

/// The parts every layer of both versions has: norms, hyper-connections, the attention's
/// projections and sink.
fn common_layer(t: &mut Tensors, c: &DeepSeek4Config, p: &str) {
    let (h, nh, hd, ql) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank);
    let (g, r, hc) = (c.o_groups, c.o_lora_rank, c.hc_mult);
    let mix = (2 + hc) * hc;
    for part in ["attn", "ffn"] {
        t.add(format!("{p}{part}_norm.weight"), &[h], 0.2, 1.0);
        t.add(format!("{p}hc_{part}_fn"), &[mix, hc * h], mat(hc * h, 4.0), 0.0);
        t.add(format!("{p}hc_{part}_base"), &[mix], 0.5, 0.0);
        t.add(format!("{p}hc_{part}_scale"), &[3], 0.25, 0.75);
    }
    t.add(format!("{p}attn.wq_a.weight"), &[ql, h], mat(h, 2.0), 0.0);
    t.add(format!("{p}attn.q_norm.weight"), &[ql], 0.2, 1.0);
    t.add(format!("{p}attn.wq_b.weight"), &[nh * hd, ql], mat(ql, 2.0), 0.0);
    t.add(format!("{p}attn.wkv.weight"), &[hd, h], mat(h, 2.0), 0.0);
    t.add(format!("{p}attn.kv_norm.weight"), &[hd], 0.2, 1.0);
    t.add(format!("{p}attn.wo_a.weight"), &[g * r, nh * hd / g], mat(nh * hd / g, 1.0), 0.0);
    t.add(format!("{p}attn.wo_b.weight"), &[h, g * r], mat(g * r, 1.0), 0.0);
    t.add(format!("{p}attn.attn_sink"), &[nh], 1.0, 0.0);
}

/// The routed and shared experts of a layer.
fn experts(t: &mut Tensors, c: &DeepSeek4Config, p: &str) {
    let (h, ne, mi) = (c.hidden_size, c.n_routed_experts, c.moe_intermediate_size);
    for e in (0..ne).map(|e| format!("experts.{e}")).chain(["shared_experts".to_string()]) {
        t.add(format!("{p}ffn.{e}.w1.weight"), &[mi, h], mat(h, 2.0), 0.0);
        t.add(format!("{p}ffn.{e}.w3.weight"), &[mi, h], mat(h, 2.0), 0.0);
        t.add(format!("{p}ffn.{e}.w2.weight"), &[h, mi], mat(mi, 1.0), 0.0);
    }
}

fn top(t: &mut Tensors, c: &DeepSeek4Config) {
    let (h, v) = (c.hidden_size, c.vocab_size);
    t.add("embed.weight".into(), &[v, h], 1.7, 0.0);
    t.add("head.weight".into(), &[v, h], mat(h, 4.0), 0.0);
    t.add("norm.weight".into(), &[h], 0.2, 1.0);
}

/// Writes the tiny V4 model of `v4_json(over)` into `dir` (`config.json`,
/// `model.safetensors`) and returns its config.
pub fn write_v4(dir: &Path, over: Value) -> Result<DeepSeek4Config> {
    let json = v4_json(over);
    let c = DeepSeek4Config::from_json(json.clone())?;
    let (h, hc, v, ne, k) = (c.hidden_size, c.hc_mult, c.vocab_size, c.n_routed_experts, c.num_experts_per_tok);
    let (ni, di, hd, ql) = (c.index_n_heads, c.index_head_dim, c.head_dim, c.q_lora_rank);
    let mut t = Tensors {
        list: Vec::new(),
        rng: Rng(0x5eed),
    };
    top(&mut t, &c);
    t.add("hc_head_fn".into(), &[hc, hc * h], mat(hc * h, 4.0), 0.0);
    t.add("hc_head_base".into(), &[hc], 0.5, 0.0);
    t.add("hc_head_scale".into(), &[1], 0.25, 0.75);
    for l in 0..c.num_hidden_layers {
        let p = format!("layers.{l}.");
        let ratio = c.compress_ratios[l];
        common_layer(&mut t, &c, &p);
        let compressor = |t: &mut Tensors, p: String, d: usize| {
            let w = if ratio == 4 { 2 * d } else { d };
            t.add(format!("{p}wkv.weight"), &[w, h], mat(h, 1.0), 0.0);
            t.add(format!("{p}wgate.weight"), &[w, h], mat(h, 3.0), 0.0);
            t.add(format!("{p}ape"), &[ratio, w], 0.5, 0.0);
            t.add(format!("{p}norm.weight"), &[d], 0.2, 1.0);
        };
        if ratio > 0 {
            compressor(&mut t, format!("{p}attn.compressor."), hd);
        }
        if ratio == 4 {
            compressor(&mut t, format!("{p}attn.indexer.compressor."), di);
            t.add(format!("{p}attn.indexer.wq_b.weight"), &[ni * di, ql], mat(ql, 2.0), 0.0);
            t.add(format!("{p}attn.indexer.weights_proj.weight"), &[ni, h], mat(h, 1.0), 0.0);
        }
        t.add(format!("{p}ffn.gate.weight"), &[ne, h], mat(h, 3.0), 0.0);
        if l >= c.num_hash_layers {
            t.add(format!("{p}ffn.gate.bias"), &[ne], 0.3, 0.0);
        }
        experts(&mut t, &c, &p);
    }
    // The hash layers' experts of each token id (distinct).
    for l in 0..c.num_hash_layers {
        let tab: Vec<u8> = (0..v)
            .flat_map(|tk| (0..k).map(move |j| ((tk * 3 + j * (1 + tk % (ne - 1))) % ne) as i32))
            .flat_map(|e| e.to_le_bytes())
            .collect();
        t.list.push((format!("layers.{l}.ffn.gate.tid2eid"), "I32", vec![v, k], tab));
    }
    t.write(dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&json)?)?;
    Ok(c)
}

/// Writes the tiny V4.1 model of `v41_json(over)` into `dir` (`config.json`,
/// `model.safetensors`) and returns its config.
pub fn write_v41(dir: &Path, over: Value) -> Result<DeepSeek4Config> {
    let json = v41_json(over);
    let c = DeepSeek4Config::from_json(json.clone())?;
    let (h, hd, ql, hc, ne) = (c.hidden_size, c.head_dim, c.q_lora_rank, c.hc_mult, c.n_routed_experts);
    let (ni, di, ed) = (c.index_n_heads, c.index_head_dim, c.engram_head_dim);
    let cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads;
    let mut t = Tensors {
        list: Vec::new(),
        rng: Rng(0x5eed41),
    };
    top(&mut t, &c);
    // The primes above 19, two heads per n-gram size and layer: 23 + 29 + 31 + 37, then
    // 41 + 43 + 47 + 53.
    let rows = [120usize, 184];
    for l in 0..c.num_hidden_layers {
        let p = format!("layers.{l}.");
        let ratio = c.ratio(l);
        common_layer(&mut t, &c, &p);
        if c.kv_source_layer_ids.contains(&l) {
            t.add(format!("{p}attn.compressor.wkv.weight"), &[hd, h], mat(h, 2.0), 0.0);
            if ratio > 1 {
                t.add(format!("{p}attn.compressor.wgate.weight"), &[hd, h], mat(h, 3.0), 0.0);
            }
            t.add(format!("{p}attn.compressor.norm.weight"), &[hd], 0.2, 1.0);
            t.add(format!("{p}attn.indexer.wk.weight"), &[di, hd], mat(hd, 2.0), 0.0);
            t.add(format!("{p}attn.indexer.k_norm.weight"), &[di], 0.2, 1.0);
        }
        if c.index_source_layer_ids.contains(&l) {
            t.add(format!("{p}attn.indexer.wq_b.weight"), &[ni * di, ql], mat(ql, 2.0), 0.0);
            t.add(format!("{p}attn.indexer.weights_proj.weight"), &[ni, h], mat(h, 1.0), 0.0);
        }
        if let Some(li) = c.engram_layer_ids.iter().position(|&e| e == l) {
            t.add(format!("{p}engram.embed.weight"), &[rows[li], ed], 1.0, 0.0);
            t.add(
                format!("{p}engram.wkv.weight"),
                &[(hc + 1) * h, cols * ed],
                mat(cols * ed, 3.0),
                0.0,
            );
            t.add(format!("{p}engram.q_weight"), &[hc, h], 0.3, 1.0);
            t.add(format!("{p}engram.k_weight"), &[hc, h], 0.3, 1.0);
        }
        t.add(format!("{p}ffn.gate.weight"), &[ne, h], mat(h, 3.0), 0.0);
        t.add(format!("{p}ffn.gate.bias"), &[ne], 0.3, 0.0);
        t.add(format!("{p}ffn.gate.bias_vl"), &[ne], 2.0, 0.0);
        experts(&mut t, &c, &p);
    }
    t.write(dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&json)?)?;
    Ok(c)
}

/// Writes a tiny V4 (or with `v41`, V4.1) model the engine loads: the model of [`write_v4`] /
/// [`write_v41`] over a vocabulary of [`CHAT_VOCAB`] (plus the keys of `over`), beside a
/// byte-level `tokenizer.json` (one token per byte) with [`SPECIAL`] from [`SPECIAL_BASE`].
pub fn write_chat(dir: &Path, v41: bool, over: Value) -> Result<DeepSeek4Config> {
    let mut o = json!({"vocab_size": CHAT_VOCAB});
    if v41 {
        o["engram_token_map"] = json!((0..CHAT_VOCAB).map(|i| (i % 20) as i32).collect::<Vec<i32>>());
    }
    for (k, v) in over.as_object().into_iter().flatten() {
        o[k] = v.clone();
    }
    let c = if v41 { write_v41(dir, o)? } else { write_v4(dir, o)? };
    let tok = crate::inkling::tiny::byte_tokenizer(&SPECIAL, SPECIAL_BASE);
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tok)?)?;
    Ok(c)
}

/// A GGUF metadata value of [`write_draft`].
enum Meta {
    U32(u32),
    F32(f32),
    Str(&'static str),
    U32s(Vec<u32>),
}

/// Writes a tiny random DeepSeek V4 DSpark draft for the target `c` (a V4 model of [`write_v4`]
/// or [`write_chat`]) to the GGUF file `path`, as llama.cpp's `dflash` conversion of the
/// release's draft names it: one DeepSeek V4 stage (window-only attention, the target's shapes
/// and window, 4 experts of which 2 per token), blocks of 4 with `mask` as the noise token,
/// reading the streams entering the target's layers 1 and 2 and the last layer's output.
pub fn write_draft(path: &Path, c: &DeepSeek4Config, mask: u32, seed: u64) -> Result<()> {
    let (h, nh, hd, ql) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank);
    let (g, r, hc) = (c.o_groups, c.o_lora_rank, c.hc_mult);
    let (ne, mi) = (c.n_routed_experts, c.moe_intermediate_size);
    let targets = [1u32, 2, c.num_hidden_layers as u32];
    let n_feat = targets.len() * h;
    let meta: Vec<(&str, Meta)> = vec![
        ("general.architecture", Meta::Str("dflash")),
        ("dflash.block_count", Meta::U32(1)),
        ("dflash.vocab_size", Meta::U32(c.vocab_size as u32)),
        ("dflash.embedding_length", Meta::U32(h as u32)),
        ("dflash.attention.head_count", Meta::U32(nh as u32)),
        ("dflash.attention.key_length", Meta::U32(hd as u32)),
        ("dflash.rope.dimension_count", Meta::U32(c.qk_rope_head_dim as u32)),
        ("dflash.attention.q_lora_rank", Meta::U32(ql as u32)),
        ("dflash.attention.output_group_count", Meta::U32(g as u32)),
        ("dflash.attention.output_lora_rank", Meta::U32(r as u32)),
        ("dflash.attention.layer_norm_rms_epsilon", Meta::F32(1e-6)),
        ("dflash.expert_count", Meta::U32(ne as u32)),
        ("dflash.expert_used_count", Meta::U32(c.num_experts_per_tok as u32)),
        ("dflash.expert_feed_forward_length", Meta::U32(mi as u32)),
        ("dflash.expert_weights_scale", Meta::F32(1.5)),
        ("dflash.swiglu_clamp_exp", Meta::F32(1.5)),
        ("dflash.attention.sliding_window", Meta::U32(c.sliding_window as u32)),
        ("dflash.attention.compress_ratios", Meta::U32s(vec![0])),
        ("dflash.attention.indexer.head_count", Meta::U32(2)),
        ("dflash.attention.indexer.key_length", Meta::U32(8)),
        ("dflash.attention.indexer.top_k", Meta::U32(2)),
        ("dflash.hyper_connection.count", Meta::U32(hc as u32)),
        ("dflash.block_size", Meta::U32(4)),
        ("dflash.target_layers", Meta::U32s(targets.to_vec())),
        ("tokenizer.ggml.mask_token_id", Meta::U32(mask)),
    ];
    // (name, shape as the loader sees it (the GGUF's dimensions reversed), noise scale, bias)
    let mix = (2 + hc) * hc;
    let mut tensors: Vec<(String, Vec<usize>, f32, f32)> = vec![
        ("fc.weight".into(), vec![h, n_feat], mat(n_feat, 1.0), 0.0),
        ("enc.output_norm.weight".into(), vec![h], 0.2, 1.0),
        ("output_norm.weight".into(), vec![h], 0.2, 1.0),
        ("output_hc_fn.weight".into(), vec![hc, hc * h], mat(hc * h, 4.0), 0.0),
        ("output_hc_base.weight".into(), vec![hc], 0.5, 0.0),
        ("output_hc_scale.weight".into(), vec![1], 0.25, 0.75),
    ];
    let p = "blk.0.";
    for part in ["attn", "ffn"] {
        tensors.push((format!("{p}{part}_norm.weight"), vec![h], 0.2, 1.0));
        tensors.push((format!("{p}hc_{part}_fn.weight"), vec![mix, hc * h], mat(hc * h, 4.0), 0.0));
        tensors.push((format!("{p}hc_{part}_base.weight"), vec![mix], 0.5, 0.0));
        tensors.push((format!("{p}hc_{part}_scale.weight"), vec![3], 0.25, 0.75));
    }
    for (n, shape, scale, bias) in [
        ("attn_q_a.weight", vec![ql, h], mat(h, 2.0), 0.0),
        ("attn_q_a_norm.weight", vec![ql], 0.2, 1.0),
        ("attn_q_b.weight", vec![nh * hd, ql], mat(ql, 2.0), 0.0),
        ("attn_kv.weight", vec![hd, h], mat(h, 2.0), 0.0),
        ("attn_kv_a_norm.weight", vec![hd], 0.2, 1.0),
        ("attn_output_a.weight", vec![g * r, nh * hd / g], mat(nh * hd / g, 1.0), 0.0),
        ("attn_output_b.weight", vec![h, g * r], mat(g * r, 1.0), 0.0),
        ("attn_sinks.weight", vec![nh], 1.0, 0.0),
        ("ffn_gate_inp.weight", vec![ne, h], mat(h, 3.0), 0.0),
        ("exp_probs_b.bias", vec![ne], 0.3, 0.0),
        ("ffn_gate_exps.weight", vec![ne, mi, h], mat(h, 2.0), 0.0),
        ("ffn_up_exps.weight", vec![ne, mi, h], mat(h, 2.0), 0.0),
        ("ffn_down_exps.weight", vec![ne, h, mi], mat(mi, 1.0), 0.0),
        ("ffn_gate_shexp.weight", vec![mi, h], mat(h, 2.0), 0.0),
        ("ffn_up_shexp.weight", vec![mi, h], mat(h, 2.0), 0.0),
        ("ffn_down_shexp.weight", vec![h, mi], mat(mi, 1.0), 0.0),
    ] {
        tensors.push((format!("{p}{n}"), shape, scale, bias));
    }

    let put_str = |b: &mut Vec<u8>, s: &str| {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    };
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((meta.len() as u64).to_le_bytes());
    for (k, v) in &meta {
        put_str(&mut b, k);
        match v {
            Meta::U32(x) => {
                b.extend(4u32.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            Meta::F32(x) => {
                b.extend(6u32.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            Meta::Str(s) => {
                b.extend(8u32.to_le_bytes());
                put_str(&mut b, s);
            }
            Meta::U32s(a) => {
                b.extend(9u32.to_le_bytes());
                b.extend(4u32.to_le_bytes());
                b.extend((a.len() as u64).to_le_bytes());
                for x in a {
                    b.extend(x.to_le_bytes());
                }
            }
        }
    }
    let mut off = 0u64;
    for (name, shape, _, _) in &tensors {
        put_str(&mut b, name);
        b.extend((shape.len() as u32).to_le_bytes());
        for &d in shape.iter().rev() {
            b.extend((d as u64).to_le_bytes());
        }
        b.extend(0u32.to_le_bytes()); // F32
        b.extend(off.to_le_bytes());
        off += (shape.iter().product::<usize>() as u64 * 4).next_multiple_of(32);
    }
    b.resize(b.len().next_multiple_of(32), 0);
    let mut rng = Rng(seed ^ 0xd5_9a_4c);
    for (_, shape, scale, bias) in &tensors {
        let start = b.len();
        for _ in 0..shape.iter().product::<usize>() {
            b.extend((bias + scale * rng.next()).to_le_bytes());
        }
        b.resize(start + (b.len() - start).next_multiple_of(32), 0);
    }
    std::fs::write(path, b).with_context(|| path.display().to_string())
}
