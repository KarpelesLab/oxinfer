//! A tiny random Inkling-2-Mini (`inkling2_loop`) checkpoint, for tests that run the model end
//! to end (sessions going back to checkpoints, snapshots, the server's prompt reuse) without a
//! real one and without Python: 12 layers of 64 in blocks of 3, layers 3-8 looped three times
//! (24 cache slots), sliding layers (4 heads of 8 on 2 key/value heads, a window of 16) and full
//! ones (8 heads of 16, dynamic key / value convolutions, log scaling from position 32), 8
//! experts of which 2 per token, a vocabulary of 512 rows (500 tokens). The weights are noise
//! from a fixed-seed generator (deterministic, meaningless), in float32 with the release's tensor
//! names, beside a byte-level `tokenizer.json` (one token per byte) with Inkling-2's message
//! tokens from id 256 ([`write_tiny`]). `tools/inkling2_tiny.py` makes checkpoints checked
//! against transformers; this one only needs to run.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;

/// Rows of the embedding and unembedding, and of them the tokens.
pub const VOCAB: usize = 512;
pub const UNPADDED: usize = 500;
/// The sliding window.
pub const WINDOW: usize = 16;
/// Inkling-2's special tokens at their ids here.
pub const SPECIAL: [&str; 13] = [
    "<|message_system|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_xml|>",
    "<|content_tool_error|>",
    "<|content_image|>",
    "<|unused_200054|>",
];
/// The id of the first special token.
pub const SPECIAL_BASE: u32 = 256;

const H: usize = 64;
const LAYERS: usize = 12;
const BLOCK: usize = 3;
const LOOP: (usize, usize) = (3, 9);
const RANK: usize = 8;
const FULL: (usize, usize, usize) = (8, 2, 16);
const SWA: (usize, usize, usize) = (4, 2, 8);
const D_REL: usize = 8;
const EXTENT: usize = 24;
const KERNEL: usize = 4;
const EXPERTS: usize = 8;
const INTER: usize = 32;

struct Rng(u64);

impl Rng {
    /// Uniform in `[-1, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

struct Tensors {
    list: Vec<(String, Vec<usize>, Vec<f32>)>,
    rng: Rng,
}

impl Tensors {
    /// `bias + scale * U(-1, 1)`.
    fn uni(&mut self, name: &str, shape: &[usize], scale: f32, bias: f32) {
        let v = (0..shape.iter().product::<usize>())
            .map(|_| bias + scale * self.rng.next())
            .collect();
        self.list.push((format!("model.llm.{name}"), shape.to_vec(), v));
    }
    /// A matrix of unit-variance outputs for unit-variance inputs, times `gain`.
    fn mat(&mut self, name: &str, shape: &[usize], gain: f32) {
        let fan_in = *shape.last().unwrap();
        self.uni(name, shape, gain * (3.0 / fan_in as f32).sqrt(), 0.0);
    }
    fn ones(&mut self, name: &str, n: usize) {
        self.uni(name, &[n], 0.2, 1.0);
    }
    fn write(self, path: &Path) -> Result<()> {
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, v) in &self.list {
            let n = data.len();
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            head.insert(
                name.clone(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [n, data.len()]}),
            );
        }
        let mut head = Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        std::fs::write(path, file).with_context(|| path.display().to_string())
    }
}

fn sliding(l: usize) -> bool {
    l % BLOCK != BLOCK - 1
}

/// A byte-level BPE `tokenizer.json`: every byte one token (GPT-2's byte-to-character map, ids
/// 0 to 255), no merges, and [`SPECIAL`] from [`SPECIAL_BASE`].
fn tokenizer_json() -> Value {
    byte_tokenizer(&SPECIAL, SPECIAL_BASE)
}

/// A byte-level BPE `tokenizer.json`: every byte one token (GPT-2's byte-to-character map, ids
/// 0 to 255), no merges, and the special tokens `special` from id `base` (also for the other
/// tiny test models).
pub(crate) fn byte_tokenizer(special: &[&str], base: u32) -> Value {
    let printable = |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut vocab = serde_json::Map::new();
    let mut extra = 0;
    for b in 0..256u32 {
        let c = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
        vocab.insert(c.to_string(), json!(b));
    }
    let mut added = Vec::new();
    for (i, s) in special.iter().enumerate() {
        let id = base + i as u32;
        vocab.insert(s.to_string(), json!(id));
        added.push(
            json!({"id": id, "content": s, "single_word": false, "lstrip": false, "rstrip": false,
                          "normalized": false, "special": true}),
        );
    }
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": byte_level, "post_processor": byte_level, "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": [],
        },
    })
}

/// Writes the checkpoint folder `dir` (`config.json`, `model.safetensors`, `tokenizer.json`);
/// the weights depend on `seed` only.
pub fn write_tiny(dir: &Path, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    let end_turn = SPECIAL_BASE + SPECIAL.iter().position(|s| *s == "<|content_model_end_sampling|>").unwrap() as u32;
    let image = SPECIAL_BASE + SPECIAL.len() as u32 - 1;
    let layer_types: Vec<&str> = (0..LAYERS).map(|l| if sliding(l) { "hybrid_sliding" } else { "hybrid" }).collect();
    let config = json!({
        "model_type": "inkling2_loop",
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "image_token_id": image,
        "eos_token_id": end_turn,
        "text_config": {
            "vocab_size": VOCAB, "unpadded_vocab_size": UNPADDED, "hidden_size": H, "num_hidden_layers": LAYERS,
            "num_attention_heads": FULL.0, "num_key_value_heads": FULL.1, "head_dim": FULL.2,
            "swa_num_attention_heads": SWA.0, "swa_num_key_value_heads": SWA.1, "swa_head_dim": SWA.2,
            "sliding_window_size": WINDOW, "d_rel": D_REL, "rel_extent": EXTENT,
            "log_scaling_n_floor": 32, "log_scaling_alpha": 0.3, "conv_kernel_size": KERNEL,
            "moe_intermediate_size": INTER, "n_routed_experts": EXPERTS, "num_experts_per_tok": 2,
            "route_scale": 2.0, "qk_l2_clip_threshold": 4.0, "rel_l2_clip_threshold": 4.0,
            "n_layers_per_attn_res_block": BLOCK, "mlp_gate_softcap": 0.8, "mlp_up_softcap": 1.5,
            "final_logit_softcapping": 10.0, "rms_norm_eps": 1e-6, "eos_token_id": end_turn,
            "loop_layer_index_start": LOOP.0, "loop_layer_index_end": LOOP.1, "loop_num_passes": 3,
            "loop_gate_rank": RANK, "layer_types": layer_types,
        },
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config)?)?;
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tokenizer_json())?)?;

    let mut t = Tensors {
        list: Vec::new(),
        rng: Rng(seed ^ 0x1234_5678_9abc_def0),
    };
    t.uni("embed.weight", &[VOCAB, H], 1.0, 0.0);
    t.mat("unembed.weight", &[VOCAB, H], 1.0);
    t.ones("embed_norm.weight", H);
    t.ones("norm.weight", H);
    t.uni("attn_res.output_query", &[H], 1.0, 0.0);
    for l in 0..LAYERS {
        let (nh, nkv, d) = if sliding(l) { SWA } else { FULL };
        let (w, kw) = (nh * d, nkv * d);
        let a = format!("layers.{l}.attn.");
        t.mat(&format!("{a}wq_du.weight"), &[w, H], 1.0);
        t.mat(&format!("{a}wk_dv.weight"), &[kw, H], 1.0);
        t.mat(&format!("{a}wv_dv.weight"), &[kw, H], 1.0);
        t.mat(&format!("{a}wr_du.weight"), &[nh * D_REL, H], 1.0);
        t.mat(&format!("{a}gate_attn.weight"), &[w, H], 1.0);
        t.mat(&format!("{a}attn_sink_proj.weight"), &[nh, H], 0.5);
        t.mat(&format!("{a}wo_ud.weight"), &[H, w], 0.7);
        t.ones(&format!("{a}q_l2_clip.scale"), w);
        t.ones(&format!("{a}rel_l2_clip.scale"), nh * D_REL);
        let extent = if sliding(l) { WINDOW } else { EXTENT };
        t.uni(&format!("{a}rel_logits_proj.proj"), &[D_REL, extent], 0.5, 0.0);
        for kv in ["k_sconv", "v_sconv"] {
            t.uni(&format!("{a}{kv}.weight"), &[kw, 1, KERNEL], 0.3, 0.0);
            if !sliding(l) {
                t.mat(&format!("{a}{kv}.weight_proj"), &[nkv * KERNEL, H], 0.3);
            }
        }
        let p = format!("layers.{l}.");
        if sliding(l) {
            t.ones(&format!("{p}attn_norm.weight"), H);
        } else {
            for n in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                t.ones(&format!("{p}{n}.weight"), H);
            }
        }
        if l > 0 {
            let shape: &[usize] = if sliding(l) { &[H] } else { &[3, H] };
            t.uni(&format!("attn_res.attn_queries.{}", l - 1), shape, 1.0, 0.0);
        }
        t.uni(&format!("attn_res.mlp_queries.{l}"), &[H], 1.0, 0.0);
        t.ones(&format!("{p}mlp_norm.weight"), H);
        t.uni(&format!("{p}attn_sconv.weight"), &[H, 1, KERNEL], 0.3, 0.0);
        t.uni(&format!("{p}mlp_sconv.weight"), &[H, 1, KERNEL], 0.3, 0.0);
        t.mat(&format!("{p}mlp.gate.weight"), &[EXPERTS, H], 1.0);
        t.uni(&format!("{p}mlp.gate.bias"), &[EXPERTS], 0.1, 0.0);
        t.mat(&format!("{p}mlp.experts.w13_weight"), &[EXPERTS, 2 * INTER, H], 1.0);
        t.mat(&format!("{p}mlp.experts.w2_weight"), &[EXPERTS, H, INTER], 0.7);
        if (LOOP.0..LOOP.1).contains(&l) {
            for kind in ["attn", "k", "v", "mlp"] {
                let c = format!("{p}controllers.{kind}.");
                t.mat(&format!("{c}down_weight"), &[RANK, 2 * H], 1.0);
                t.mat(&format!("{c}forget_weight"), &[H, RANK], 0.5);
                t.mat(&format!("{c}gain_weight"), &[H, RANK], 0.5);
            }
            t.uni(&format!("{p}loop_msg_query"), &[4, H], 1.0, 0.0);
        }
    }
    t.write(&dir.join("model.safetensors"))
}

/// Layer applications of a pass of the tiny model (the looped layers run three times).
const STEPS: usize = LAYERS + (LOOP.1 - LOOP.0) * 2;

/// Adds a tiny random block drafter to the checkpoint folder `dir` of [`write_tiny`], as the
/// release's MTP head (`mtp.safetensors` and `config.json`'s `mtp_config`; a DSpark GGUF of it
/// is the same drafter): 3 layers of 64 (4 heads of 16 on 2 key / value heads), the layers of
/// `local` attending to the 8 positions before the anchor (the others to all), blocks of 4, a
/// Markov head of rank 8, reading the streams entering layer applications 2, 10 and 18 and the
/// final-normed output.
pub fn write_tiny_mtp(dir: &Path, seed: u64, local: &[usize]) -> Result<()> {
    const DL: usize = 3;
    const NH: usize = 4;
    const NKV: usize = 2;
    const HD: usize = 16;
    const DI: usize = 96;
    const DR: usize = 8;
    let cfg_path = dir.join("config.json");
    let raw = std::fs::read(&cfg_path).with_context(|| cfg_path.display().to_string())?;
    let mut config: Value = serde_json::from_slice(&raw)?;
    let targets = [2, 10, 18, STEPS];
    config["mtp_config"] = json!({
        "hidden_size": H, "num_hidden_layers": DL, "num_attention_heads": NH, "num_key_value_heads": NKV,
        "head_dim": HD, "intermediate_size": DI, "sliding_window_size": 8, "local_layer_ids": local,
        "block_size": 4, "target_layer_ids": targets, "markov_head_type": "rnn_hidden", "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
    });
    std::fs::write(&cfg_path, serde_json::to_vec_pretty(&config)?)?;
    let mut t = Tensors {
        list: Vec::new(),
        rng: Rng(seed ^ 0x0fed_cba9_8765_4321),
    };
    let (qw, kw) = (NH * HD, NKV * HD);
    for l in 0..DL {
        let p = format!("layers.{l}.");
        for n in ["input_layernorm", "post_attention_layernorm", "dflare_norm"] {
            t.ones(&format!("{p}{n}.weight"), H);
        }
        t.mat(&format!("{p}self_attn.q_proj.weight"), &[qw, H], 1.0);
        for n in ["k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"] {
            t.mat(&format!("{p}self_attn.{n}.weight"), &[kw, H], 1.0);
        }
        t.mat(&format!("{p}self_attn.o_proj.weight"), &[H, qw], 0.7);
        for n in ["q_norm", "k_norm", "k_norm_ctx"] {
            t.ones(&format!("{p}self_attn.{n}.weight"), HD);
        }
        t.mat(&format!("{p}mlp.gate_proj.weight"), &[DI, H], 1.0);
        t.mat(&format!("{p}mlp.up_proj.weight"), &[DI, H], 1.0);
        t.mat(&format!("{p}mlp.down_proj.weight"), &[H, DI], 0.7);
    }
    t.uni("fuse_weight", &[DL + 1, targets.len()], 1.0, 0.0);
    t.ones("markov_head.dflare_norm.weight", H);
    t.ones("norm.weight", H);
    t.uni("mask_embed", &[H], 1.0, 0.0);
    t.uni("markov_head.markov_w1.weight", &[VOCAB, DR], 1.0, 0.0);
    t.mat("markov_head.hidden_out.weight", &[H, DR], 1.0);
    t.uni("markov_head.hidden_out.bias", &[H], 0.1, 0.0);
    t.mat("markov_head.rnn_proj.weight", &[3 * DR, 2 * DR + 2 * H], 1.0);
    t.uni("markov_head.rnn_proj.bias", &[3 * DR], 0.1, 0.0);
    // The drafter's tensors have no prefix.
    for e in &mut t.list {
        e.0 = e.0.trim_start_matches("model.llm.").to_string();
    }
    t.write(&dir.join("mtp.safetensors"))
}
