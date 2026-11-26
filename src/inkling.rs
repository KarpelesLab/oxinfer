//! Thinking Machines' Inkling-2 text decoder, as transformers' `Inkling2TextModel` (`inkling2`:
//! thinkingmachines/Inkling-2-Small and Inkling-2-Small-MXFP8) and `Inkling2LoopTextModel`
//! (`inkling2_loop`: thinkingmachines/Inkling-2-Mini, see [`LoopConfig`]), from the releases or
//! from GGUF files ([`gguf_config`], [`gguf_name`]). The vision tower is in
//! [`crate::inkling_vision`].
//!
//! - The token embedding goes through an RMS norm. There are no position embeddings and no
//!   rotary positions: attention gets its positions from a relative bias and from short causal
//!   convolutions.
//! - Attention (grouped-query, softmax): queries scaled per channel then clipped to an L2 norm
//!   (`qk_l2_clip_threshold`), keys clipped, values RMS-normed without a gain. A fourth
//!   projection gives each query head a `d_rel` vector (scaled, clipped) that mixes a learned
//!   bank of bias-versus-distance profiles (`rel_logits_proj`, `[d_rel, extent]`) into one bias
//!   per backward distance below the extent. Every query also has a sink: one more logit (a
//!   projection of the input) with a zero value, which only absorbs probability. The attended
//!   values are gated per channel (`sigmoid(gate_proj x)`) before the output projection.
//! - Keys and values pass a short causal depthwise convolution (kernel 4) with a skip, as do the
//!   output of the attention and of the MLP: `(conv(x) + x) / sqrt(kernel + 1)`.
//! - "Sliding" layers (5 of 6) see the last `sliding_window_size` tokens and may have their own
//!   head counts (`swa_*`). The others see everything: there the relative extent is `rel_extent`,
//!   queries and relative vectors grow with `1 + alpha ln(max(1, (pos + 1) / n_floor))`, and the
//!   key / value convolutions add per-token taps (a projection of the input, shared by the
//!   channels of a key/value head) to the learned ones, over `sqrt(2 kernel + 1)`.
//! - Attention residual: there is no plain residual stream across blocks of
//!   `n_layers_per_attn_res_block` layers. The input of the first layer of each block is kept as
//!   a "block state"; before its attention, before its MLP and before the final norm, a layer
//!   reads a softmax mix (a learned query against the RMS-normed states, scaled by
//!   `hidden^-0.5`) of the block states and of the stream inside the current block. A full layer
//!   reads three mixes (for its queries, keys and values), each with its own norm. Inside a
//!   block the stream is the sum of the layers' outputs; the first layer of a block starts it
//!   again from its attention output.
//! - MoE without a shared expert: sigmoid scores, the top k of score plus a selection bias,
//!   renormalized and scaled by `route_scale`; experts are SwiGLU with soft caps on the linear
//!   gate factor and on `up` (`cap tanh(x / cap)`; [`crate::moe::Act::Situ`]).
//! - The logits are soft-capped (`final_logit_softcapping`) and cut to `unpadded_vocab_size`
//!   (the unembedding is padded).
//!
//! Not here: the multi-token prediction head (`mtp.safetensors`, which transformers does not
//! run either).
//!
//! Checkpoints: the release's tensor names (`model.llm.*`, which `save_pretrained` writes too).
//! The routed experts (`w13_weight`, gate and up rows interleaved, and `w2_weight`) are read as
//! stored: MXFP8 (E4M3 values with an E8M0 exponent per 32) in place, NVFP4 (E2M1 pairs with an
//! E4M3 scale per 16 and `amax / outer_scale_denominator` per expert) repacked at load into
//! ggml's NVFP4 blocks, or plain floats.
//!
//! A session can go back up to [`ROLLBACK`] positions ([`Inkling2Session::truncate`], what
//! speculative decoding needs): the convolutions keep their last inputs in rings of
//! `kernel - 1 + ROLLBACK` rows indexed by position, and the sliding-window caches keep that many
//! positions more than their window. Further back, it goes to a [`Checkpoint`] taken at that
//! position ([`Inkling2Model::checkpoint`], [`Inkling2Model::cut_to`]: the convolutions' last
//! inputs and the sliding slots' last window, the full-attention caches being kept whole), which
//! is how [`crate::engine::Engine::chat`] reuses a conversation's prompt after a mismatch.

#[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
#[path = "inkling_cuda.rs"]
mod cuda;

#[doc(hidden)]
#[path = "inkling_tiny.rs"]
pub mod tiny;

use crate::intervene::{NeuronHook, Target};
use crate::kernels::*;
use crate::model::{Lin, Quant, Table, linear, linear_multi, load_lin, load_lin_experts, table_quant};
use crate::moe::{Act, Experts, NeedExperts, route_sigmoid};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

/// The epsilon of the norm of the attention residual's states (the module's default, not
/// `rms_norm_eps`).
const MIX_EPS: f32 = 1e-6;
/// Added to a norm before an L2 clip divides by it.
const CLIP_EPS: f32 = 1e-6;
/// The prefix of the text decoder's tensors.
const P: &str = "model.llm.";
/// Positions a session can go back ([`Inkling2Session::truncate`]): the convolutions' rings keep
/// this many inputs beyond the `kernel - 1` they read, the sliding-window caches this many
/// positions beyond their window.
pub const ROLLBACK: usize = 64;
/// Most draft tokens a speculative round of this family verifies: a pass of `MAX_DRAFT + 1`
/// tokens stays on the CPU's decode kernels (routed experts included, `kernels::TN`), whose rows
/// are those of one-token forwards bit for bit, so greedy output is plain greedy output. (Longer
/// passes take the prefill path: other rounding, and several times slower on the CPU.)
pub const MAX_DRAFT: usize = crate::kernels::TN - 2;
/// [`MAX_DRAFT`] when the passes run on the accelerator, whose passes of up to 16 tokens give
/// one-token results ([`Inkling2Model::max_draft`]).
pub const DEVICE_MAX_DRAFT: usize = 15;

/// Whether `model_type` is one of this family's.
pub fn is_model_type(mt: &str) -> bool {
    matches!(mt, "inkling2" | "inkling2_text" | "inkling2_loop" | "inkling2_loop_text")
}

/// The looped middle of `inkling2_loop` (Inkling-2-Mini): layers `start..end` run `passes`
/// times with shared weights. Every pass after the first restarts from the stream and the block
/// states entering `start`, and its layers mix into their inputs messages read from the previous
/// pass (`gate_rank`: the rank of the gates that mix them).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopConfig {
    pub start: usize,
    pub end: usize,
    pub passes: usize,
    pub gate_rank: usize,
}

/// One application of a layer in the unrolled sequence of a pass through the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    /// The layer whose weights run.
    pub layer: usize,
    /// The pass through the loop (0 outside it).
    pub pass: usize,
    /// The cache slot (keys, values and convolution inputs) of this application.
    pub slot: usize,
}

/// The head layout of one kind of layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heads {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// The settings of an Inkling-2 text decoder (`text_config`, in the release's spelling or in
/// transformers').
#[derive(Debug, Clone)]
pub struct Inkling2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Rows of the embedding and of the unembedding.
    pub vocab_size: usize,
    /// Rows of the unembedding that are tokens (the logits).
    pub unpadded_vocab_size: usize,
    pub full: Heads,
    pub swa: Heads,
    pub sliding_window_size: usize,
    /// Per layer: sliding-window attention (otherwise full).
    pub sliding: Vec<bool>,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: f64,
    pub log_scaling_alpha: f64,
    pub conv_kernel_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub route_scale: f64,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,
    pub n_layers_per_attn_res_block: usize,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    pub final_logit_softcapping: f64,
    pub rms_norm_eps: f64,
    /// The placeholder token of an image feature.
    pub image_token_id: u32,
    /// The looped layers (`inkling2_loop`).
    pub loop_cfg: Option<LoopConfig>,
}

impl Inkling2Config {
    /// The config of a checkpoint folder (`config.json`) or GGUF file.
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: Value = if crate::gguf::is_gguf(dir) {
            crate::gguf::header(dir)?.config_json()?
        } else {
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?
        };
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &Value) -> Result<Self> {
        let mt = raw.get("model_type").and_then(Value::as_str).unwrap_or("");
        if !is_model_type(mt) {
            bail!("model_type {mt:?} is not an Inkling-2 model");
        }
        let t = raw.get("text_config").filter(|v| v.is_object()).unwrap_or(raw);
        let get = |keys: &[&str]| keys.iter().find_map(|k| t.get(*k).filter(|v| !v.is_null()));
        let int = |keys: &[&str], default: Option<usize>| -> Result<usize> {
            match get(keys) {
                Some(v) => v
                    .as_u64()
                    .map(|v| v as usize)
                    .with_context(|| format!("config: {} is not an integer", keys[0])),
                None => default.with_context(|| format!("config: {} is missing", keys[0])),
            }
        };
        let num = |keys: &[&str], default: f64| -> Result<f64> {
            match get(keys) {
                Some(v) => v.as_f64().with_context(|| format!("config: {} is not a number", keys[0])),
                None => Ok(default),
            }
        };
        // Variants of the architecture the release does not use and transformers does not have.
        for (key, want) in [
            ("q_bias", false),
            ("o_bias", false),
            ("pre_norm_pre_affine", false),
            ("final_norm_pre_affine", false),
            ("use_value_norm_no_gamma", true),
            ("attn_res_normalize_values", true),
            ("use_gate_bias", true),
        ] {
            if let Some(v) = t.get(key).and_then(Value::as_bool)
                && v != want
            {
                bail!("Inkling-2: {key} = {v} is not supported");
            }
        }
        if int(&["n_shared_experts"], Some(0))? != 0 {
            bail!("Inkling-2: shared experts are not supported");
        }
        if let Some(f) = get(&["scoring_func"]).and_then(Value::as_str)
            && f != "sigmoid"
        {
            bail!("Inkling-2: scoring_func {f:?} is not supported");
        }
        let n = int(&["num_hidden_layers"], Some(48))?;
        let block = int(&["n_layers_per_attn_res_block"], Some(6))?;
        let sliding: Vec<bool> = if let Some(lt) = get(&["layer_types"]).and_then(Value::as_array) {
            lt.iter()
                .map(|v| match v.as_str() {
                    Some("hybrid_sliding") => Ok(true),
                    Some("hybrid") => Ok(false),
                    other => bail!("Inkling-2: layer type {other:?}"),
                })
                .collect::<Result<_>>()?
        } else if let Some(ids) = get(&["local_layer_ids"]).and_then(Value::as_array) {
            let ids: Vec<u64> = ids.iter().filter_map(Value::as_u64).collect();
            (0..n as u64).map(|i| ids.contains(&i)).collect()
        } else {
            (0..n).map(|i| (i + 1) % 6 != 0).collect()
        };
        if sliding.len() != n {
            bail!("Inkling-2: {} layer types for {n} layers", sliding.len());
        }
        let full = Heads {
            heads: int(&["num_attention_heads"], Some(48))?,
            kv_heads: int(&["num_key_value_heads"], Some(8))?,
            head_dim: int(&["head_dim"], Some(128))?,
        };
        let swa = Heads {
            heads: int(&["swa_num_attention_heads"], Some(full.heads))?,
            kv_heads: int(&["swa_num_key_value_heads"], Some(full.kv_heads))?,
            head_dim: int(&["swa_head_dim"], Some(full.head_dim))?,
        };
        for hd in [full, swa] {
            if hd.kv_heads == 0 || !hd.heads.is_multiple_of(hd.kv_heads) {
                bail!("Inkling-2: {} heads over {} key/value heads", hd.heads, hd.kv_heads);
            }
        }
        let vocab_size = int(&["vocab_size"], Some(201024))?;
        let mut c = Self {
            hidden_size: int(&["hidden_size"], None)?,
            num_hidden_layers: n,
            vocab_size,
            unpadded_vocab_size: int(&["unpadded_vocab_size"], Some(vocab_size))?.min(vocab_size),
            full,
            swa,
            sliding_window_size: int(&["sliding_window_size", "sliding_window"], Some(1024))?,
            sliding,
            d_rel: int(&["d_rel"], Some(16))?,
            rel_extent: int(&["rel_extent"], Some(1024))?,
            log_scaling_n_floor: num(&["log_scaling_n_floor"], 128000.0)?,
            log_scaling_alpha: num(&["log_scaling_alpha"], 0.1)?,
            conv_kernel_size: int(&["conv_kernel_size", "sconv_kernel_size"], Some(4))?,
            moe_intermediate_size: int(&["moe_intermediate_size", "intermediate_size"], None)?,
            n_routed_experts: int(&["n_routed_experts", "num_local_experts"], None)?,
            num_experts_per_tok: int(&["num_experts_per_tok"], None)?,
            route_scale: num(&["route_scale", "routed_scaling_factor"], 8.0)?,
            qk_l2_clip_threshold: num(&["qk_l2_clip_threshold"], 22.627416997969522)?,
            rel_l2_clip_threshold: num(&["rel_l2_clip_threshold"], 8.0)?,
            n_layers_per_attn_res_block: block,
            mlp_gate_softcap: num(&["mlp_gate_softcap"], 4.0)?,
            mlp_up_softcap: num(&["mlp_up_softcap"], 25.0)?,
            final_logit_softcapping: num(&["final_logit_softcapping"], 30.0)?,
            rms_norm_eps: num(&["rms_norm_eps"], 1e-6)?,
            image_token_id: raw.get("image_token_id").and_then(Value::as_u64).unwrap_or(200054) as u32,
            loop_cfg: None,
        };
        if mt.starts_with("inkling2_loop") || t.get("loop_config").is_some_and(Value::is_object) {
            // The release's `loop_config`, or transformers' `loop_*` fields (its defaults otherwise).
            let lc = t.get("loop_config").filter(|v| v.is_object());
            let pick = |release: &str, tf: &str, default: usize| -> Result<usize> {
                match lc.and_then(|o| o.get(release)).or_else(|| t.get(tf)).filter(|v| !v.is_null()) {
                    Some(v) => v
                        .as_u64()
                        .map(|v| v as usize)
                        .with_context(|| format!("config: {release} is not an integer")),
                    None => Ok(default),
                }
            };
            if lc.and_then(|o| o.get("is_varloop")).and_then(Value::as_bool) == Some(true) {
                bail!("Inkling-2: a variable loop (is_varloop) is not supported");
            }
            let l = LoopConfig {
                start: pick("core_start", "loop_layer_index_start", 6)?,
                end: pick("core_end", "loop_layer_index_end", 18)?,
                passes: pick("num_passes", "loop_num_passes", 3)?,
                gate_rank: pick("gate_rank", "loop_gate_rank", 64)?,
            };
            if l.passes == 0
                || l.start == 0
                || l.start >= l.end
                || l.end > n
                || !l.start.is_multiple_of(block)
                || !l.end.is_multiple_of(block)
                || l.gate_rank == 0
            {
                bail!("Inkling-2: invalid loop {l:?} for {n} layers in blocks of {block}");
            }
            c.loop_cfg = Some(l);
        }
        if c.conv_kernel_size == 0 || c.n_layers_per_attn_res_block == 0 || c.sliding_window_size == 0 {
            bail!("Inkling-2: a kernel, a window or a block of 0");
        }
        if c.num_experts_per_tok == 0 || c.num_experts_per_tok > c.n_routed_experts {
            bail!("Inkling-2: {} experts per token of {}", c.num_experts_per_tok, c.n_routed_experts);
        }
        Ok(c)
    }

    /// The heads of layer `l`.
    pub fn heads(&self, l: usize) -> Heads {
        if self.sliding[l] { self.swa } else { self.full }
    }

    /// Whether layer `l` is in the loop.
    pub fn looped(&self, l: usize) -> bool {
        self.loop_cfg.is_some_and(|c| (c.start..c.end).contains(&l))
    }

    /// The layers in the order a pass runs them: the layers before the loop, the loop `passes`
    /// times, the rest. The first pass through the loop uses the layers' own cache slots, the
    /// others slots after them (as transformers' and llama.cpp's caches).
    pub fn steps(&self) -> Vec<Step> {
        let n = self.num_hidden_layers;
        let one = |layer| Step {
            layer,
            pass: 0,
            slot: layer,
        };
        let Some(c) = self.loop_cfg else {
            return (0..n).map(one).collect();
        };
        let core = c.end - c.start;
        let mut v: Vec<Step> = (0..c.end).map(one).collect();
        for pass in 1..c.passes {
            v.extend((c.start..c.end).map(|layer| Step {
                layer,
                pass,
                slot: n + (pass - 1) * core + layer - c.start,
            }));
        }
        v.extend((c.end..n).map(one));
        v
    }

    /// Cache slots of a session (one per layer application of a pass).
    pub fn n_slots(&self) -> usize {
        self.num_hidden_layers + self.loop_cfg.map_or(0, |c| (c.passes - 1) * (c.end - c.start))
    }

    /// The layer of each cache slot.
    pub fn slot_layers(&self) -> Vec<usize> {
        let mut v = vec![0usize; self.n_slots()];
        for s in self.steps() {
            v[s.slot] = s.layer;
        }
        v
    }
}

/// The release's `config.json` of a GGUF file of llama.cpp's `inkling2` / `inkling2_loop`
/// architectures.
pub(crate) fn gguf_config(hd: &crate::gguf::Header) -> Result<Value> {
    use crate::gguf::Value as G;
    let arch = hd.arch().to_string();
    let get = |k: &str| hd.get(&format!("{arch}.{k}"));
    let int = |k: &str| get(k).and_then(G::as_u64).map(|v| v as usize);
    let need = |k: &str| int(k).with_context(|| format!("GGUF metadata has no {arch}.{k}"));
    let float = |k: &str| {
        get(k)
            .and_then(G::as_f64)
            .with_context(|| format!("GGUF metadata has no {arch}.{k}"))
    };
    // The loop: llama.cpp's `inkling2_loop` (`loop.start`, ...), or unsloth's `inkling2` files
    // (`num_loops`, `loop.core_start`, ...), whose block count and per-layer arrays cover the
    // layer applications of a pass (the loop unrolled), not the layers.
    let lp = if arch == "inkling2_loop" {
        if matches!(get("loop.varying"), Some(G::Bool(true))) {
            bail!("GGUF {arch}: a variable loop (loop.varying) is not supported");
        }
        Some((
            need("loop.start")?,
            need("loop.end")?,
            need("loop.count")?,
            need("loop.gate_rank")?,
            false,
        ))
    } else if let Some(p) = int("num_loops").filter(|&p| p > 1) {
        Some((need("loop.core_start")?, need("loop.core_end")?, p, need("loop.gate_rank")?, true))
    } else {
        None
    };
    let applications = need("block_count")?;
    let n = match lp {
        Some((s, e, p, _, true)) => applications
            .checked_sub((p - 1) * (e.saturating_sub(s)))
            .context("GGUF: a loop longer than the blocks")?,
        _ => applications,
    };
    // Entry of layer `l` in a per-layer array of `len` (per application when unrolled).
    let entry = |l: usize, len: usize| -> usize {
        match lp {
            Some((s, e, p, _, true)) if len == applications && l >= e => l + (p - 1) * (e - s),
            _ => l,
        }
    };
    let flags: Vec<bool> = get("attention.sliding_window_pattern")
        .and_then(G::as_array)
        .context("GGUF metadata has no attention.sliding_window_pattern")?
        .iter()
        .map(|v| matches!(v, G::Bool(true)))
        .collect();
    if flags.len() != n && flags.len() != applications {
        bail!("GGUF {arch}: {} sliding-window flags for {n} layers", flags.len());
    }
    let pattern: Vec<bool> = (0..n).map(|l| flags[entry(l, flags.len())]).collect();
    // Per-layer head counts: those of a full layer and of a sliding one.
    let per_layer = |k: &str| -> Result<Vec<usize>> {
        match get(k) {
            Some(G::Array(a)) if a.len() == n || a.len() == applications => {
                Ok((0..n).map(|l| a[entry(l, a.len())].as_u64().unwrap_or(0) as usize).collect())
            }
            Some(v) if v.as_u64().is_some() => Ok(vec![v.as_u64().unwrap() as usize; n]),
            _ => bail!("GGUF metadata has no {arch}.{k} for {n} layers"),
        }
    };
    let (heads, kv) = (per_layer("attention.head_count")?, per_layer("attention.head_count_kv")?);
    let of = |v: &[usize], swa: bool| (0..n).find(|&l| pattern[l] == swa).map_or(v[0], |l| v[l]);
    let head_dim = need("attention.key_length")?;
    let mut text = serde_json::json!({
        "hidden_size": need("embedding_length")?,
        "num_hidden_layers": n,
        "vocab_size": need("vocab_size")?,
        "unpadded_vocab_size": need("vocab_size_unpadded")?,
        "num_attention_heads": of(&heads, false),
        "num_key_value_heads": of(&kv, false),
        "head_dim": head_dim,
        "swa_num_attention_heads": of(&heads, true),
        "swa_num_key_value_heads": of(&kv, true),
        "swa_head_dim": int("attention.key_length_swa").unwrap_or(head_dim),
        "sliding_window_size": need("attention.sliding_window")?,
        "local_layer_ids": (0..n).filter(|&l| pattern[l]).collect::<Vec<_>>(),
        "d_rel": need("attention.relative_length")?,
        "rel_extent": need("attention.relative_extent")?,
        "log_scaling_n_floor": need("attention.log_scale_floor")?,
        "log_scaling_alpha": float("attention.log_scale_alpha")?,
        "qk_l2_clip_threshold": float("attention.qk_clip")?,
        "rel_l2_clip_threshold": float("attention.relative_clip")?,
        "sconv_kernel_size": need("shortconv.l_cache")?,
        "n_layers_per_attn_res_block": need("attn_res.block_size")?,
        "n_routed_experts": need("expert_count")?,
        "num_experts_per_tok": need("expert_used_count")?,
        "n_shared_experts": 0,
        "moe_intermediate_size": need("expert_feed_forward_length")?,
        "route_scale": float("expert_weights_scale")?,
        "mlp_gate_softcap": float("activation.situ_beta")?,
        "mlp_up_softcap": float("activation.situ_linear_beta")?,
        "final_logit_softcapping": float("final_logit_softcapping")?,
        "rms_norm_eps": float("attention.layer_norm_rms_epsilon")?,
    });
    let mut mt = arch.clone();
    if let Some((s, e, p, rank, _)) = lp {
        text["loop_config"] = serde_json::json!({
            "core_start": s, "core_end": e, "num_passes": p, "gate_rank": rank, "is_varloop": false,
        });
        mt = "inkling2_loop".into();
    }
    Ok(serde_json::json!({"model_type": mt, "text_config": text}))
}

/// The release's name (`model.llm.*`) and shape of a tensor of a GGUF file of llama.cpp's
/// `inkling2` / `inkling2_loop` architectures, or None for tensors the loader does not read.
/// The relative-bias bank is transposed there (it goes under `..._t`); the experts' gate and up
/// rows are not interleaved (`gate_up_proj`, `down_proj`); each looped layer has its four
/// message queries (`loop_msg_query`).
pub(crate) fn gguf_name(name: &str, ne: &[usize]) -> Option<(String, Vec<usize>)> {
    let shape: Vec<usize> = ne.iter().rev().copied().collect();
    let top = match name {
        "token_embd.weight" => Some("embed.weight"),
        "token_embd_norm.weight" => Some("embed_norm.weight"),
        "output.weight" => Some("unembed.weight"),
        "output_norm.weight" => Some("norm.weight"),
        "output_res_query.weight" => Some("attn_res.output_query"),
        // unsloth's files: the release's two tables of message queries.
        "loop_res_query.weight" => Some("middle_cycle_cross_pass_queries"),
        "loop_kv_res_query.weight" => Some("middle_cycle_cross_pass_kv_queries"),
        _ => None,
    };
    if let Some(t) = top {
        return Some((format!("{P}{t}"), shape));
    }
    let (l, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    let l: usize = l.parse().ok()?;
    // ggml `{kernel, channels}` -> a depthwise Conv1d's `[channels, 1, kernel]`.
    let conv_shape = || vec![ne[1], 1, ne[0]];
    let (n, sh): (String, Vec<usize>) = match rest {
        "attn_q.weight" => ("attn.wq_du.weight".into(), shape),
        "attn_k.weight" => ("attn.wk_dv.weight".into(), shape),
        "attn_v.weight" => ("attn.wv_dv.weight".into(), shape),
        "attn_output.weight" => ("attn.wo_ud.weight".into(), shape),
        "attn_gate.weight" => ("attn.gate_attn.weight".into(), shape),
        "attn_rel.weight" => ("attn.wr_du.weight".into(), shape),
        "attn_sink_proj.weight" => ("attn.attn_sink_proj.weight".into(), shape),
        "attn_q_clip.weight" => ("attn.q_l2_clip.scale".into(), shape),
        "attn_rel_clip.weight" => ("attn.rel_l2_clip.scale".into(), shape),
        "attn_rel_bias.weight" => ("attn.rel_logits_proj.proj_t".into(), shape),
        "attn_k_conv.weight" => ("attn.k_sconv.weight".into(), conv_shape()),
        "attn_v_conv.weight" => ("attn.v_sconv.weight".into(), conv_shape()),
        "attn_k_conv_proj.weight" => ("attn.k_sconv.weight_proj".into(), shape),
        "attn_v_conv_proj.weight" => ("attn.v_sconv.weight_proj".into(), shape),
        "attn_post_conv.weight" => ("attn_sconv.weight".into(), conv_shape()),
        "ffn_post_conv.weight" => ("mlp_sconv.weight".into(), conv_shape()),
        "attn_norm.weight" => ("attn_norm.weight".into(), shape),
        "attn_q_inp_norm.weight" => ("q_input_norm.weight".into(), shape),
        "attn_k_inp_norm.weight" => ("k_input_norm.weight".into(), shape),
        "attn_v_inp_norm.weight" => ("v_input_norm.weight".into(), shape),
        "ffn_norm.weight" => ("mlp_norm.weight".into(), shape),
        "ffn_gate_inp.weight" => ("mlp.gate.weight".into(), shape),
        "exp_probs_b.bias" => ("mlp.gate.bias".into(), shape),
        "ffn_gate_up_exps.weight" => ("mlp.experts.gate_up_proj".into(), shape),
        "ffn_down_exps.weight" => ("mlp.experts.down_proj".into(), shape),
        "ffn_gate_up_exps.scale" => ("mlp.experts.gate_up_scale".into(), shape),
        "ffn_down_exps.scale" => ("mlp.experts.down_scale".into(), shape),
        "loop_msg_query.weight" => ("loop_msg_query".into(), shape),
        // (`attn_queries.{i}` belongs to layer `i + 1`.)
        "attn_res_query.weight" if l > 0 => return Some((format!("{P}attn_res.attn_queries.{}", l - 1), shape)),
        "ffn_res_query.weight" => return Some((format!("{P}attn_res.mlp_queries.{l}"), shape)),
        r => {
            // llama.cpp's `loop_q_ctrl_down`, unsloth's `loop_attn_down`.
            let r = r.strip_prefix("loop_")?.strip_suffix(".weight")?;
            let (kind, part) = r.split_once("_ctrl_").or_else(|| r.rsplit_once('_'))?;
            let kind = match kind {
                "q" | "attn" => "attn",
                "k" => "k",
                "v" => "v",
                "ffn" | "mlp" => "mlp",
                _ => return None,
            };
            let part = match part {
                "down" => "down_weight",
                "gain" => "gain_weight",
                "forget" => "forget_weight",
                _ => return None,
            };
            (format!("controllers.{kind}.{part}"), shape)
        }
    };
    Some((format!("{P}layers.{l}.{n}"), sh))
}

/// Whether `files` is an mlx-lm checkpoint of the family (`mlx-community/Inkling-2-Mini-4bit`:
/// mlx-lm's names, `model.language_model.*`), read under the release's names ([`mlx_name`]).
pub(crate) fn is_mlx_layout(files: &ModelFiles) -> bool {
    files.is_mlx() && files.has("model.language_model.final_input.query")
}

/// The release's name of a tensor of an mlx-lm checkpoint (mlx-lm's `sanitize` undone), or None
/// to keep it. Quantized matrices keep their `.weight` / `.scales` / `.biases` parts; the
/// experts (the release's bytes as words) go back under `w13_weight` / `w2_weight` with their
/// scales and global scales; the message queries of all looped layers (`[core, 4, H]`, query,
/// key, value, MLP) become one table, `loop_msg_query_all`.
pub(crate) fn mlx_name(n: &str) -> Option<String> {
    if let Some(sfx) = n.strip_prefix("lm_head.") {
        return Some(format!("{P}unembed.{sfx}"));
    }
    if let Some(r) = n.strip_prefix("model.vision_tower.") {
        return Some(mlx_vision_name(r));
    }
    let r = n.strip_prefix("model.language_model.")?;
    let top = match r {
        "embed_norm.weight" => Some("embed_norm.weight".to_string()),
        "norm.weight" => Some("norm.weight".to_string()),
        "final_input.query" => Some("attn_res.output_query".to_string()),
        "message_input.query" => Some("loop_msg_query_all".to_string()),
        _ => r.strip_prefix("embed_tokens.").map(|sfx| format!("embed.{sfx}")),
    };
    if let Some(t) = top {
        return Some(format!("{P}{t}"));
    }
    let (l, rest) = r.strip_prefix("layers.")?.split_once('.')?;
    let li: usize = l.parse().ok()?;
    match rest {
        // (Layer 0 has no attention residual query: `attn_queries.{i}` is layer `i + 1`'s.)
        "attention_input.query" => return Some(format!("{P}attn_res.attn_queries.{}", li.checked_sub(1)?)),
        "mlp_input.query" => return Some(format!("{P}attn_res.mlp_queries.{li}")),
        _ => {}
    }
    let own = |s: &str| Some(format!("{P}layers.{li}.{s}"));
    match rest {
        "input_layernorm.weight" => return own("attn_norm.weight"),
        "post_attention_layernorm.weight" => return own("mlp_norm.weight"),
        "q_input_layernorm.weight" => return own("q_input_norm.weight"),
        "k_input_layernorm.weight" => return own("k_input_norm.weight"),
        "v_input_layernorm.weight" => return own("v_input_norm.weight"),
        "post_attention_convolution.conv1d.weight" => return own("attn_sconv.weight"),
        "post_mlp_convolution.conv1d.weight" => return own("mlp_sconv.weight"),
        "mlp.gate.weight" => return own("mlp.gate.weight"),
        "mlp.gate.e_score_correction_bias" => return own("mlp.gate.bias"),
        _ => {}
    }
    if let Some((kind, part)) = rest.split_once("_message_gate.") {
        let kind = if kind == "q" { "attn" } else { kind };
        let part = match part {
            "down_proj.weight" => "down_weight",
            "forget_gate_proj.weight" => "forget_weight",
            "message_gate_proj.weight" => "gain_weight",
            _ => return None,
        };
        return own(&format!("controllers.{kind}.{part}"));
    }
    if let Some(e) = rest.strip_prefix("mlp.experts.") {
        let (proj, part) = e.split_once('.')?;
        let w = match proj {
            "gate_up_proj" => "w13_weight",
            "down_proj" => "w2_weight",
            _ => return None,
        };
        return match part {
            "weight" => own(&format!("mlp.experts.{w}")),
            "scales" => own(&format!("mlp.experts.{w}_scale")),
            "global_scale" => own(&format!("mlp.experts.{w}_global_scale")),
            _ => None,
        };
    }
    let a = rest.strip_prefix("self_attn.")?;
    for (from, to) in [
        ("q_proj.", "wq_du."),
        ("k_proj.", "wk_dv."),
        ("v_proj.", "wv_dv."),
        ("r_proj.", "wr_du."),
        ("o_proj.", "wo_ud."),
        ("gate_proj.", "gate_attn."),
        ("sink_proj.", "attn_sink_proj."),
    ] {
        if let Some(sfx) = a.strip_prefix(from) {
            return own(&format!("attn.{to}{sfx}"));
        }
    }
    for kv in ["k_sconv.", "v_sconv."] {
        if let Some(p) = a.strip_prefix(kv) {
            let to = match p {
                "conv1d.weight" => "weight".to_string(),
                "static_weight" => "static_w".to_string(),
                // The release's `weight_proj` has no `.weight` part (its MLX parts follow it).
                "weight_proj.weight" => "weight_proj".to_string(),
                _ => p.to_string(),
            };
            return own(&format!("attn.{kv}{to}"));
        }
    }
    // q_l2_clip.scale, rel_l2_clip.scale, rel_logits_proj.proj
    own(&format!("attn.{a}"))
}

/// The release's name (`model.visual.*`) of an mlx-lm vision tower tensor (after
/// `model.vision_tower.`).
fn mlx_vision_name(r: &str) -> String {
    if let Some(s) = r.strip_prefix("pre_layernorm.") {
        return format!("model.visual.backbone.ln_pre.{s}");
    }
    if let Some(s) = r.strip_prefix("merger.linear_1.") {
        return format!("model.visual.proj_in.{s}");
    }
    if let Some(s) = r.strip_prefix("merger.linear_2.") {
        return format!("model.visual.proj_out.{s}");
    }
    let mut s = r.to_string();
    if let Some(p) = s.strip_suffix("embeddings.position_embedding.weight") {
        s = format!("{p}embeddings.position_embedding");
    }
    if let Some(rest) = s.strip_prefix("layers.")
        && let Some((l, part)) = rest.split_once('.')
    {
        let part = match part {
            p if p.starts_with("layernorm_before.") => p.replacen("layernorm_before.", "ln_1.", 1),
            p if p.starts_with("layernorm_after.") => p.replacen("layernorm_after.", "ln_2.", 1),
            p if p.starts_with("mlp.fc1.") => p.replacen("mlp.fc1.", "mlp.c_fc.", 1),
            p if p.starts_with("mlp.fc2.") => p.replacen("mlp.fc2.", "mlp.c_proj.", 1),
            "lambda_1" => "ls_1.gamma".to_string(),
            "lambda_2" => "ls_2.gamma".to_string(),
            p => p.to_string(),
        };
        s = format!("layers.{l}.{part}");
    }
    format!("model.visual.backbone.{s}")
}

/// The prefill chunk when none is asked for: 2048 tokens on machines of 32 GB or more (each of
/// a layer's 128 experts then sees about 128 of them, enough for its weights to be decoded once
/// for many: prefill is 2-3x faster than with 256), 512 below (a chunk of 2048 holds about 3 GB
/// of activations).
pub fn default_prefill_chunk() -> usize {
    match crate::platform::total_memory() {
        Some(m) if m >= 32 << 30 => 2048,
        _ => 512,
    }
}

/// Checks that a DSpark draft reads streams this target has: Inkling-2's drafter
/// (`crate::inkling_draft`) reads layer applications of a pass and, last, the final-normed
/// output (index `steps().len() + 1` of [`Inkling2Model::forward_capture`]).
pub fn check_draft(m: &Inkling2Model, d: &crate::dspark::Dspark) -> Result<()> {
    let n = m.cfg.steps().len();
    let tl = &d.cfg.target_layers;
    if tl.iter().any(|&l| l > n + 1) || tl.last() != Some(&(n + 1)) {
        bail!("the draft reads target layers {tl:?}; this Inkling-2 model has {n} layer applications per pass");
    }
    Ok(())
}

/// Weight streaming for a checkpoint that does not fit in memory (as Mistral Large 4).
pub fn default_stream() -> crate::stream::StreamOptions {
    crate::mistral4::default_stream()
}

/// Whether a checkpoint of `bytes` is too big to hold in memory (over 60% of RAM).
pub fn too_big(bytes: u64) -> bool {
    crate::mistral4::too_big(bytes)
}

/// Memory Inkling-2 Mini needs beside its weights, measured (see docs/inkling.md, "Long
/// contexts and memory"): 1.3 GB of runtime (activations, prefill chunks of 512, the app), the
/// sliding windows' caches (692 MB in f32) and `context` tokens of the full-attention caches
/// (64 KB a token in f32), both scaled to the cache type. `context` None: 4096 tokens assumed.
pub fn runtime_bytes(kv: KvCache, context: Option<usize>) -> u64 {
    let r = kv.bytes_per_value() / 4.0;
    let ctx = context.unwrap_or(4096) as f64;
    (1.3e9 + 692e6 * r + ctx * 65536.0 * r) as u64
}

/// The convolution of keys or values.
// Two per layer, built once.
#[allow(clippy::large_enum_variant)]
pub(crate) enum KvConv<'a> {
    /// Learned taps, `[lag][channels]` (lag 0 is the current token).
    Static(Vec<f32>),
    /// Learned taps plus per-token ones: `proj` (`[kv_heads * kernel, H]`) gives one tap per
    /// key/value head and lag.
    Dynamic { taps: Vec<f32>, proj: Lin<'a> },
}

pub(crate) struct Attn<'a> {
    pub(crate) hd: Heads,
    /// Sliding-window attention over this many tokens (the current one included).
    pub(crate) window: Option<usize>,
    /// Backward distances that have a relative bias.
    pub(crate) extent: usize,
    pub(crate) q: Lin<'a>,        // [heads * d, H]
    pub(crate) k: Lin<'a>,        // [kv * d, H]
    pub(crate) v: Lin<'a>,        // [kv * d, H]
    pub(crate) r: Lin<'a>,        // [heads * d_rel, H]
    pub(crate) gate: Lin<'a>,     // [heads * d, H]
    pub(crate) sink: Lin<'a>,     // [heads, H]
    pub(crate) o: Lin<'a>,        // [H, heads * d]
    pub(crate) q_scale: Vec<f32>, // [heads * d]
    pub(crate) r_scale: Vec<f32>, // [heads * d_rel]
    pub(crate) rel: Vec<f32>,     // [d_rel, extent]
    pub(crate) k_conv: KvConv<'a>,
    pub(crate) v_conv: KvConv<'a>,
}

/// The norms before the attention: one, or one each for queries, keys and values.
pub(crate) enum InNorm {
    One(Vec<f32>),
    Qkv(Vec<f32>, Vec<f32>, Vec<f32>),
}

/// A message gate of a looped layer: `(1 - tanh(forget g)) x + tanh(gain g) m` for its input `x`
/// and message `m`, where `g = down [rms(x); rms(m)]`.
pub(crate) struct Gate<'a> {
    pub(crate) down: Lin<'a>,   // [rank, 2 H]
    pub(crate) forget: Lin<'a>, // [H, rank]
    pub(crate) gain: Lin<'a>,   // [H, rank]
}

/// The rows of a float matrix as stored (a gate's: f32 in the GGUF files, bf16 in the release),
/// for the passes of a few tokens, which take them one dot product at a time.
enum FloatRows<'m> {
    F32(&'m [f32]),
    Bf16(&'m [u16]),
}

impl<'m> FloatRows<'m> {
    /// None for a quantized matrix, one with a LoRA update, a folded input or block scales.
    fn of(l: &'m Lin) -> Option<Self> {
        if l.has_lora() || l.fold.is_some() {
            return None;
        }
        match &l.w {
            crate::model::Weight::Bf16(w, _) => Some(FloatRows::Bf16(&w[..])),
            crate::model::Weight::Gguf(m) if m.kind == crate::ggml::GgmlType::F32 && m.block_scales.is_none() => {
                let b = &m.data[..];
                // SAFETY: f32 values read in place where the bytes are aligned for them (a GGUF
                // tensor starts on a multiple of 32 bytes of a page-aligned buffer).
                (b.as_ptr() as usize)
                    .is_multiple_of(4)
                    .then(|| FloatRows::F32(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }))
            }
            _ => None,
        }
    }

    /// Row `r` (of `n` values) times `x`.
    #[inline]
    fn dot(&self, r: usize, x: &[f32]) -> f32 {
        let n = x.len();
        match self {
            FloatRows::F32(w) => dot(&w[r * n..(r + 1) * n], x),
            FloatRows::Bf16(w) => dot_bf16(&w[r * n..(r + 1) * n], x),
        }
    }
}

/// A message gate over the tokens of a short pass: `d = down cat` (`cat`: `[t][2 h]`, the
/// RMS-normed input and message), then `out = (1 - tanh(forget d)) x + tanh(gain d) m`.
///
/// The three float matrices are 2.4 MB of Inkling-2 Mini that a step reads once. Through the
/// prompt GEMM's tiles (one token in tiles of 16, three parallel regions) and with the two tanh
/// per value on the calling thread, a gate took 120 us of a DGX Spark's 20 cores, 96 times a
/// token. Here: two regions, the low-rank rows as plain dot products, then each output's two dot
/// products with its tanh mix. Every token is computed as it would be alone.
#[allow(clippy::too_many_arguments)]
fn gate_rows(
    wd: &FloatRows,
    wf: &FloatRows,
    wg: &FloatRows,
    cat: &[f32],
    x: &[f32],
    m: &[f32],
    h: usize,
    r: usize,
    d: &mut [f32],
    out: &mut [f32],
) {
    // (Tasks of whole parts of a token's rows: 16 for the low-rank rows, 24 for the outputs.)
    let dc = if r.is_multiple_of(16) { r / 16 } else { r };
    for_chunks(d, dc, true, |i, o| {
        let (ti, r0) = (i * dc / r, i * dc % r);
        let row = &cat[ti * 2 * h..(ti + 1) * 2 * h];
        for (k, v) in o.iter_mut().enumerate() {
            *v = wd.dot(r0 + k, row);
        }
    });
    let d = &*d;
    let oc = if h.is_multiple_of(24) { h / 24 } else { h };
    for_chunks(out, oc, true, |i, o| {
        let (ti, j0) = (i * oc / h, i * oc % h);
        let dr = &d[ti * r..(ti + 1) * r];
        for (k, v) in o.iter_mut().enumerate() {
            let (j, at) = (j0 + k, ti * h + j0 + k);
            *v = (1.0 - wf.dot(j, dr).tanh()) * x[at] + wg.dot(j, dr).tanh() * m[at];
        }
    });
}

/// What a looped layer has for the passes after the first.
pub(crate) struct LoopIn<'a> {
    /// The gates of the query, key, value and MLP inputs.
    pub(crate) gates: [Gate<'a>; 4],
    /// The four message queries (query, key, value, MLP), `[4, H]`.
    pub(crate) query: Vec<f32>,
}

pub(crate) struct Layer<'a> {
    /// The message gates and queries of a looped layer.
    pub(crate) looped: Option<LoopIn<'a>>,
    /// The first layer of a block: its input becomes a block state.
    pub(crate) boundary: bool,
    /// The attention residual's queries before the attention (none on layer 0): `[H]`, or
    /// `[3, H]` on a full layer.
    pub(crate) attn_query: Option<Vec<f32>>,
    pub(crate) mlp_query: Vec<f32>,
    pub(crate) ln_in: InNorm,
    pub(crate) ln_post: Vec<f32>,
    pub(crate) attn: Attn<'a>,
    /// The convolutions after the attention and after the MLP, `[lag][H]`.
    pub(crate) attn_conv: Vec<f32>,
    pub(crate) mlp_conv: Vec<f32>,
    pub(crate) router: Vec<f32>, // [n_exp, H]
    pub(crate) bias: Vec<f32>,   // [n_exp]
    pub(crate) experts: Experts<'a>,
}

/// How the attention caches store keys and values ([`Inkling2Model::kv`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvCache {
    /// float32 (exact).
    #[default]
    F32,
    /// IEEE half precision (keys are clipped to an L2 norm of 22.6 and values RMS-normed, so
    /// both fit easily).
    F16,
    /// int8 with one float32 scale per 32 values (as ggml's Q8_0, a quarter of float32).
    Q8,
}

impl KvCache {
    /// `auto` (or empty): None, for [`KvCache::auto`]; else as [`KvCache::parse`].
    pub fn parse_opt(s: &str) -> Result<Option<Self>> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => None,
            o => Some(Self::parse(o)?),
        })
    }
    /// The cache type when none is asked for: f16 when the passes run on Metal (decoding at 128k
    /// positions reads 4.3 GB of caches a token rather than 8.6: 1.6x faster with UD-IQ3_XXS; the
    /// prefill's flash attention reads half rows too), else f32 (exact).
    pub fn auto() -> Self {
        if cfg!(any(target_os = "macos", target_os = "ios")) && crate::gpu::active().is_some() {
            KvCache::F16
        } else {
            KvCache::F32
        }
    }
    /// `f32` (or `auto`), `f16` or `q8`.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" | "f32" | "fp32" => KvCache::F32,
            "f16" | "fp16" | "half" => KvCache::F16,
            "q8" | "q8_0" | "int8" => KvCache::Q8,
            o => bail!("kv cache {o:?}: f32, f16 or q8"),
        })
    }
    /// Bytes per stored value.
    pub fn bytes_per_value(self) -> f64 {
        match self {
            KvCache::F32 => 4.0,
            KvCache::F16 => 2.0,
            KvCache::Q8 => 1.125,
        }
    }
}

/// Rows of keys or values (`[width]` each) in one of the [`KvCache`] layouts.
#[derive(Clone, Default)]
pub(crate) struct KvRows {
    pub(crate) mode: KvCache,
    pub(crate) f: Vec<f32>,
    pub(crate) h: Vec<u16>,
    pub(crate) q: Vec<i8>,
    /// Q8: one scale per 32 values.
    pub(crate) s: Vec<f32>,
}

impl KvRows {
    pub(crate) fn new(mode: KvCache) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }
    /// Appends rows (`x.len()` a multiple of 32 for Q8).
    fn push(&mut self, x: &[f32]) {
        // Grown by an eighth at a time rather than doubled: at a long context the caches are
        // most of the memory.
        fn room<T>(v: &mut Vec<T>, add: usize) {
            if v.capacity() - v.len() < add {
                v.reserve_exact(add.max(v.len() / 8).max(1 << 16));
            }
        }
        match self.mode {
            KvCache::F32 => {
                room(&mut self.f, x.len());
                self.f.extend_from_slice(x)
            }
            KvCache::F16 => {
                room(&mut self.h, x.len());
                self.h.extend(x.iter().map(|&v| half::f16::from_f32(v).to_bits()))
            }
            KvCache::Q8 => {
                room(&mut self.q, x.len());
                room(&mut self.s, x.len() / 32);
                for b in x.chunks(32) {
                    let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let d = amax / 127.0;
                    let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                    self.q.extend(b.iter().map(|&v| (v * inv).round() as i8));
                    self.s.push(d);
                }
            }
        }
    }
    /// Keeps the first `n` values (`n` a multiple of 32 for Q8).
    fn truncate(&mut self, n: usize) {
        match self.mode {
            KvCache::F32 => self.f.truncate(n),
            KvCache::F16 => self.h.truncate(n),
            KvCache::Q8 => {
                self.q.truncate(n);
                self.s.truncate(n / 32);
            }
        }
    }
    /// Drops the first `n` values.
    fn drop_front(&mut self, n: usize) {
        match self.mode {
            KvCache::F32 => drop(self.f.drain(..n)),
            KvCache::F16 => drop(self.h.drain(..n)),
            KvCache::Q8 => {
                self.q.drain(..n);
                self.s.drain(..n / 32);
            }
        }
    }
    /// Values `at..at + out.len()` (`at` and the length multiples of 32 for Q8) as float32:
    /// borrowed when stored so, else decoded into `out`.
    #[inline]
    fn get<'b>(&'b self, at: usize, out: &'b mut [f32]) -> &'b [f32] {
        let n = out.len();
        match self.mode {
            KvCache::F32 => &self.f[at..at + n],
            KvCache::F16 => {
                use half::slice::HalfFloatSliceExt;
                // SAFETY: f16 and u16 have the same layout.
                let h: &[half::f16] = unsafe { std::slice::from_raw_parts(self.h[at..at + n].as_ptr() as *const half::f16, n) };
                h.convert_to_f32_slice(out);
                out
            }
            KvCache::Q8 => {
                for (b, (oc, qc)) in out.chunks_mut(32).zip(self.q[at..at + n].chunks(32)).enumerate() {
                    let d = self.s[at / 32 + b];
                    for (o, &q) in oc.iter_mut().zip(qc) {
                        *o = q as f32 * d;
                    }
                }
                out
            }
        }
    }
    /// Bytes held.
    fn bytes(&self) -> usize {
        self.f.len() * 4 + self.h.len() * 2 + self.q.len() + self.s.len() * 4
    }
    /// A copy of values `from..to` (multiples of 32 for Q8).
    fn range(&self, from: usize, to: usize) -> KvRows {
        let mut r = KvRows::new(self.mode);
        match self.mode {
            KvCache::F32 => r.f = self.f[from..to].to_vec(),
            KvCache::F16 => r.h = self.h[from..to].to_vec(),
            KvCache::Q8 => {
                r.q = self.q[from..to].to_vec();
                r.s = self.s[from / 32..to / 32].to_vec();
            }
        }
        r
    }
}

/// Whole passes on an accelerator (CUDA: `inkling_cuda.rs`; Metal: `crate::metal::inkling`),
/// chosen when the model first runs. A session's state then lives on the device ([`DevBox`]); a
/// pass the device does not take (interventions, captured layers on CUDA, more tokens than
/// [`DevEngine::max_tokens`]) first brings it back to the host caches ([`DevEngine::to_host`]).
pub(crate) trait DevEngine: Send + Sync {
    /// Most tokens of a pass the device takes.
    fn max_tokens(&self) -> usize {
        usize::MAX
    }
    /// Positions a session on the device can go back ([`Inkling2Session::truncate`]); 0 when it
    /// cannot (prompt lookup and draft models then stay off).
    fn rollback(&self) -> usize {
        0
    }
    /// Whether a session on the device can be written to a snapshot where it is
    /// ([`DevSession::download_into`]).
    fn snapshots(&self) -> bool {
        false
    }
    /// Most tokens of a pass whose rows are those of one-token passes (speculative verification
    /// stays within it: [`Inkling2Model::max_draft`]).
    fn exact_tokens(&self) -> usize {
        usize::MAX
    }
    /// Whether [`DevEngine::run`] takes captured streams (draft models read them at every round;
    /// without, their rounds would run on the CPU).
    fn captures(&self) -> bool {
        true
    }
    /// Runs the embedded tokens `x` (`[t, hidden]`, embedding norm applied) from `sess.pos`,
    /// advancing it, into `want`; false when the device does not take the pass (the session is
    /// then untouched). By default [`DevEngine::forward`] and the head, without captured streams.
    fn run(&self, m: &Inkling2Model, sess: &mut Inkling2Session, x: &[f32], t: usize, want: DevWant) -> bool {
        if want.capture.is_some() {
            return false;
        }
        let h = m.cfg.hidden_size;
        let hs = self.forward(m, sess, x, t);
        if let Some((l, rows)) = want.logits {
            let r = rows.min(t);
            *l = m.head(&hs[(t - r) * h..], r, true);
        }
        if let Some(hv) = want.hidden {
            *hv = hs;
        }
        true
    }
    /// Runs the embedded tokens `x` (`[t, hidden]`, embedding norm applied) from `sess.pos` and
    /// advances it: the final-normed hidden states.
    fn forward(&self, m: &Inkling2Model, sess: &mut Inkling2Session, x: &[f32], t: usize) -> Vec<f32>;
    /// The unembedding of `hidden` (`[t, hidden]`), soft-capped when `cap`; None leaves it to the
    /// CPU.
    fn head(&self, m: &Inkling2Model, hidden: &[f32], t: usize, cap: bool) -> Option<Vec<f32>>;
    /// Moves the device state of `sess`, if any, back into its host caches.
    fn to_host(&self, m: &Inkling2Model, sess: &mut Inkling2Session);
    /// Greedy decoding on the device ([`Inkling2Model::decode_greedy`]): one-token passes from
    /// `first`, each pass's input the argmax of the one before, at most `max` of them; `on` gets
    /// each kept pass's argmax in order ([`crate::generate::Chain`]). The passes run (the session
    /// advanced by them, a pass that ran past a stop included) and the passes kept; None when the
    /// device does not do it (the session is then untouched).
    fn greedy(
        &self,
        _m: &Inkling2Model,
        _sess: &mut Inkling2Session,
        _first: u32,
        _max: usize,
        _on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<(usize, usize)> {
        None
    }
    /// [`DevEngine::greedy`] where each pass's token is a temperature draw over the whole
    /// vocabulary ([`crate::sampler::sample_full`]) from the uniform `draw` gives as the pass is
    /// queued ([`Inkling2Model::decode_sampled`]); with `feats`, appending the streams `capture`
    /// lists as [`DevEngine::greedy_capture`].
    #[allow(clippy::too_many_arguments)]
    fn sampled(
        &self,
        _m: &Inkling2Model,
        _sess: &mut Inkling2Session,
        _first: u32,
        _max: usize,
        _capture: &[usize],
        _feats: Option<&mut Vec<f32>>,
        _temperature: f32,
        _draw: &mut dyn FnMut() -> f32,
        _on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<(usize, usize)> {
        None
    }
    /// [`DevEngine::greedy`] that also appends to `feats` the streams `capture` lists (as
    /// [`DevWant::capture`]: `[t, capture.len(), hidden]`) of each kept pass.
    #[allow(clippy::too_many_arguments)]
    fn greedy_capture(
        &self,
        _m: &Inkling2Model,
        _sess: &mut Inkling2Session,
        _first: u32,
        _max: usize,
        _capture: &[usize],
        _feats: &mut Vec<f32>,
        _on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<(usize, usize)> {
        None
    }
}

/// A session's state on a device ([`DevEngine`]).
pub(crate) trait DevSession: Send + Sync {
    fn clone_box(&self) -> Box<dyn DevSession>;
    /// Bytes of its attention caches.
    fn bytes(&self) -> usize;
    #[cfg_attr(
        not(all(
            feature = "gpu",
            any(not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32")), target_arch = "aarch64")
        )),
        allow(dead_code)
    )]
    fn as_any(&mut self) -> &mut dyn std::any::Any;
    /// Goes back from position `pos` to `len`; false when the device state cannot.
    fn truncate(&mut self, _pos: usize, _len: usize) -> bool {
        false
    }
    /// The slots' part of a [`Checkpoint`] of position `pos` (`shape`: the session's host
    /// layers, for the convolution rings' layout; `keep[slot]`: the positions before `pos` a
    /// sliding slot keeps, 0 for a full one); None when the device cannot (the session then
    /// moves to the host for it).
    fn checkpoint(&self, _pos: usize, _shape: &[LayerState], _keep: &[usize]) -> Option<Vec<SlotCheckpoint>> {
        None
    }
    /// Goes back to the position of `cp`, taken from this session at an earlier position (its
    /// full-attention rows before it unchanged since); false when the device cannot.
    fn restore(&mut self, _cp: &Checkpoint, _shape: &[LayerState]) -> bool {
        false
    }
    /// Writes the caches and rings at position `pos` into `layers` (fresh ones with the host's
    /// ring shapes) without leaving the device; false when the device cannot.
    fn download_into(&self, _layers: &mut [LayerState], _pos: usize) -> bool {
        false
    }
}

/// What a session needs to go back to an earlier position from any later one (further than
/// [`ROLLBACK`]): the convolutions' inputs before it and the sliding-window slots' last window
/// of keys and values (their rings and caches drop older positions); the full-attention slots
/// keep every row by position. [`Inkling2Model::checkpoint`] takes one,
/// [`Inkling2Model::cut_to`] goes back to it.
pub struct Checkpoint {
    pub(crate) pos: usize,
    pub(crate) slots: Vec<SlotCheckpoint>,
}

/// One cache slot's part of a [`Checkpoint`].
pub(crate) struct SlotCheckpoint {
    /// A sliding slot: the keys and values of positions `first..pos` (empty for a full slot,
    /// whose `first` is then `pos`).
    pub(crate) first: usize,
    pub(crate) k: KvRows,
    pub(crate) v: KvRows,
    /// The rows of positions `pos - (kernel - 1)..pos` of the four convolution rings, in order.
    pub(crate) conv: [Vec<f32>; 4],
}

impl Checkpoint {
    /// The position it goes back to.
    pub fn pos(&self) -> usize {
        self.pos
    }
    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| s.k.bytes() + s.v.bytes() + s.conv.iter().map(|c| c.len() * 4).sum::<usize>())
            .sum()
    }
}

/// See [`DevSession`]: authoritative while set (the host caches are then stale).
pub(crate) struct DevBox(pub(crate) Box<dyn DevSession>);

impl Clone for DevBox {
    fn clone(&self) -> Self {
        DevBox(self.0.clone_box())
    }
}

/// The last inputs of a short convolution by position: the row of position `p` (from
/// `-(kernel - 1)`, zeros before the first token) is at `(p + kernel - 1) % cap`; `cap` is
/// `kernel - 1 + ROLLBACK` rows, so that after going back up to [`ROLLBACK`] positions the
/// `kernel - 1` inputs before the new position are still there.
#[derive(Clone, Default)]
pub(crate) struct ConvRing {
    pub(crate) rows: Vec<f32>,
    pub(crate) ch: usize,
    pub(crate) back: usize,
}

impl ConvRing {
    fn new(kernel: usize, ch: usize) -> Self {
        let back = kernel - 1;
        Self {
            rows: vec![0f32; (back + ROLLBACK) * ch],
            ch,
            back,
        }
    }
    pub(crate) fn cap(&self) -> usize {
        self.back + ROLLBACK
    }
    /// The slot of position `p` (`p >= -(kernel - 1)`).
    pub(crate) fn slot(&self, p: isize) -> usize {
        (p + self.back as isize).rem_euclid(self.cap() as isize) as usize
    }
    fn row(&self, p: isize) -> &[f32] {
        let s = self.slot(p);
        &self.rows[s * self.ch..(s + 1) * self.ch]
    }
    fn row_mut(&mut self, p: isize) -> &mut [f32] {
        let s = self.slot(p);
        &mut self.rows[s * self.ch..(s + 1) * self.ch]
    }
}

/// The decoding state of one layer.
#[derive(Clone, Default)]
pub(crate) struct LayerState {
    /// Clipped keys and normed values of positions `first..`, `[kv * d]` each.
    pub(crate) k: KvRows,
    pub(crate) v: KvRows,
    pub(crate) first: usize,
    /// The recent inputs of the key, value, attention and MLP convolutions.
    pub(crate) hist: [ConvRing; 4],
}

#[derive(Clone)]
pub struct Inkling2Session {
    pub(crate) layers: Vec<LayerState>,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
    /// The state on the device, when the model runs there.
    pub(crate) dev: Option<DevBox>,
    /// The device could not hold this session's caches: it runs on the host from then on.
    pub(crate) dev_off: bool,
    /// Checkpoints of earlier positions, lowest first ([`Inkling2Model::checkpoint`]; shared by
    /// the session's copies).
    pub(crate) checkpoints: Vec<std::sync::Arc<Checkpoint>>,
    /// The earliest position [`Inkling2Session::truncate`] may go back to: a restored
    /// checkpoint's (the rings and windows hold nothing older).
    pub(crate) floor: usize,
}

/// What a device pass returns ([`DevEngine::run`]).
pub(crate) struct DevWant<'w> {
    /// The final-normed hidden states `[t, hidden]`.
    pub hidden: Option<&'w mut Vec<f32>>,
    /// The soft-capped logits `[rows, unpadded vocab]` of the last `rows` tokens.
    pub logits: Option<(&'w mut Vec<f32>, usize)>,
    /// The streams entering the given steps (`[t, steps.len() * hidden]`, see
    /// [`Inkling2Model::forward_capture`]).
    pub capture: Option<(&'w [usize], &'w mut Vec<f32>)>,
}

impl Inkling2Session {
    /// Goes back to position `len` (at most [`ROLLBACK`] before the current one), as if the
    /// tokens from `len` on had not been run.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len >= self.pos {
            return Ok(());
        }
        if self.pos - len > ROLLBACK || len < self.floor {
            bail!(
                "Inkling-2: a session goes back at most {ROLLBACK} positions and not before position {} ({len} asked at {})",
                self.floor,
                self.pos
            );
        }
        self.checkpoints.retain(|c| c.pos <= len);
        if let Some(d) = &mut self.dev {
            if !d.0.truncate(self.pos, len) {
                bail!("Inkling-2: this device's sessions cannot go back to an earlier position");
            }
            self.pos = len;
            return Ok(());
        }
        for st in &mut self.layers {
            // (A sliding layer keeps `ROLLBACK` positions more than its window.)
            let rows = len.saturating_sub(st.first);
            let row = st.hist[0].ch;
            st.k.truncate(rows * row);
            st.v.truncate(rows * row);
        }
        self.pos = len;
        Ok(())
    }
}

/// The snapshot code of a cache format.
fn kv_code(m: KvCache) -> u64 {
    match m {
        KvCache::F32 => 0,
        KvCache::F16 => 1,
        KvCache::Q8 => 2,
    }
}

/// Snapshots ([`crate::snapshot`]): the position (and the earliest one the session goes back
/// to), then per cache slot the position of its first cached row, its keys and values in the
/// cache's format and its four convolution rings. A session on the device is read from there
/// (Metal; it stays there); a restored session starts on the host and moves to the device at
/// its next pass. Checkpoints are not part of it.
impl crate::snapshot::SessionState for Inkling2Session {
    fn write_state<W: std::io::Write>(&self, w: &mut crate::snapshot::SnapWriter<W>) -> Result<()> {
        let fetched;
        let layers: &[LayerState] = match &self.dev {
            Some(d) => {
                let mut l: Vec<LayerState> = self
                    .layers
                    .iter()
                    .map(|st| LayerState {
                        hist: st.hist.clone(),
                        ..Default::default()
                    })
                    .collect();
                if !d.0.download_into(&mut l, self.pos) {
                    bail!("session snapshots are not supported for Inkling-2 on this device");
                }
                fetched = l;
                &fetched
            }
            None => &self.layers,
        };
        w.u64s("ink.pos", &[self.pos as u64, self.floor as u64])?;
        for (i, st) in layers.iter().enumerate() {
            w.u64s(&format!("ink.{i}.first"), &[st.first as u64, kv_code(st.k.mode)])?;
            for (n, r) in [("k", &st.k), ("v", &st.v)] {
                let name = format!("ink.{i}.{n}");
                match r.mode {
                    KvCache::F32 => w.f32s(&name, &r.f)?,
                    KvCache::F16 => w.u16s(&name, &r.h)?,
                    KvCache::Q8 => {
                        w.i8s(&name, &r.q)?;
                        w.f32s(&format!("{name}.s"), &r.s)?;
                    }
                }
            }
            for (j, h) in st.hist.iter().enumerate() {
                w.f32s(&format!("ink.{i}.conv{j}"), &h.rows)?;
            }
        }
        Ok(())
    }

    fn read_state(&mut self, r: &mut crate::snapshot::SnapReader) -> Result<()> {
        let p = r.u64s_exact("ink.pos", 2)?;
        let (pos, floor) = (p[0] as usize, p[1] as usize);
        if floor > pos {
            bail!("snapshot: bad positions");
        }
        for (i, st) in self.layers.iter_mut().enumerate() {
            let f = r.u64s_exact(&format!("ink.{i}.first"), 2)?;
            let first = f[0] as usize;
            if first > pos || f[1] != kv_code(st.k.mode) {
                bail!("snapshot: cache slot {i} does not match the model (another KV cache format?)");
            }
            let n = (pos - first).checked_mul(st.hist[0].ch).context("snapshot: cache size")?;
            for (name, rows) in [("k", &mut st.k), ("v", &mut st.v)] {
                let name = format!("ink.{i}.{name}");
                *rows = KvRows::new(rows.mode);
                match rows.mode {
                    KvCache::F32 => rows.f = r.f32s(&name, n)?,
                    KvCache::F16 => rows.h = r.u16s(&name, n)?,
                    KvCache::Q8 => {
                        rows.q = r.i8s(&name, n)?;
                        rows.s = r.f32s(&format!("{name}.s"), n / 32)?;
                    }
                }
            }
            st.first = first;
            for (j, h) in st.hist.iter_mut().enumerate() {
                r.f32s_into(&format!("ink.{i}.conv{j}"), &mut h.rows)?;
            }
        }
        self.pos = pos;
        self.floor = floor;
        self.dev = None;
        self.checkpoints.clear();
        Ok(())
    }

    fn truncatable(&self) -> bool {
        false
    }
}

pub struct Inkling2Model<'a> {
    pub cfg: Inkling2Config,
    pub quant: Quant,
    /// How new sessions store their keys and values (Q8 falls back to F16 for heads that are
    /// not whole blocks of 32).
    pub kv: KvCache,
    pub(crate) embed: Table<'a>,
    pub(crate) embed_norm: Vec<f32>,
    pub(crate) lm_head: Lin<'a>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) final_query: Vec<f32>,
    pub(crate) layers: Vec<Layer<'a>>,
    /// Weight streaming: unit `l` is layer `l`, unit `num_hidden_layers` the LM head.
    stream: Option<crate::stream::StreamHook>,
    /// Whole passes on the GPU, decided at the first pass ([`DevEngine`]).
    device: std::sync::OnceLock<Option<Box<dyn DevEngine>>>,
    /// Engines replaced since (see `attach_lora`).
    retired: Vec<Box<dyn DevEngine>>,
    /// Bytes of weights a GPU engine may keep mapped (None: no limit): above it the passes stay
    /// on the CPU (a phone's GPU working set or app limit, `crate::stream::DeviceMemory`).
    pub gpu_budget: Option<u64>,
    /// The checkpoint's size (what a GPU engine maps).
    weight_bytes: u64,
}

struct Loader<'a, 'c> {
    files: &'a ModelFiles,
    cfg: &'c Inkling2Config,
    quant: Quant,
}

impl<'a> Loader<'a, '_> {
    fn getf(&self, full: &str, n: usize) -> Result<Vec<f32>> {
        let info = self.files.info(full).with_context(|| format!("missing tensor {full}"))?;
        if info.shape.iter().product::<usize>() != n {
            bail!("tensor {full} has shape {:?}, expected {n} values", info.shape);
        }
        self.files.f32(full)
    }
    /// A matrix: float32 tensors exactly, others through `load_lin`.
    fn lin(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        if let Some(info) = self.files.info(full)
            && info.dtype == "F32"
        {
            if info.shape != [n_out, n_in] {
                bail!("tensor {full} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
            }
            self.files.note_use(full, true, false);
            return Lin::from_gguf(crate::ggml::GgmlType::F32, self.files.bytes(full)?, n_out, n_in);
        }
        load_lin(self.files, full, n_out, n_in, self.quant)
    }
    /// A small projection kept in float precision (never quantized at load).
    fn lin_fp(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        Loader {
            quant: Quant::Bf16,
            ..*self
        }
        .lin(full, n_out, n_in)
    }

    /// A depthwise convolution `[channels, 1, kernel]` as `[lag][channels]`: the last tap of the
    /// kernel is the current token's.
    fn conv(&self, full: &str, ch: usize) -> Result<Vec<f32>> {
        let k = self.cfg.conv_kernel_size;
        let w = self.getf(full, ch * k)?;
        let mut taps = vec![0f32; k * ch];
        for c in 0..ch {
            for j in 0..k {
                taps[(k - 1 - j) * ch + c] = w[c * k + j];
            }
        }
        Ok(taps)
    }

    fn kv_conv(&self, p: &str, ch: usize, kv: usize) -> Result<KvConv<'a>> {
        let k = self.cfg.conv_kernel_size;
        if self.files.has(&format!("{p}weight_proj")) {
            Ok(KvConv::Dynamic {
                // `static_w` is `[kernel, channels]`, tap index = lag; a GGUF file has the taps
                // in the layout of the static convolutions.
                taps: if self.files.has(&format!("{p}static_w")) {
                    self.getf(&format!("{p}static_w"), k * ch)?
                } else {
                    self.conv(&format!("{p}weight"), ch)?
                },
                proj: self.lin_fp(&format!("{p}weight_proj"), kv * k, self.cfg.hidden_size)?,
            })
        } else {
            Ok(KvConv::Static(self.conv(&format!("{p}weight"), ch)?))
        }
    }

    /// The experts of `[n_exp, n_out, n_in]` tensor `full`, in the format it is stored in.
    fn experts(&self, full: &str, n_out: usize, n_in: usize) -> Result<Vec<Lin<'a>>> {
        use crate::ggml::{GgmlType, GgufMatrix};
        use crate::quant::{PageVec, QBuf};
        let f = self.files;
        let ne = self.cfg.n_routed_experts;
        let info = f.info(full).with_context(|| format!("missing tensor {full}"))?;
        let sn = format!("{full}_scale");
        let scale_shape = |per: usize| -> Result<()> {
            let si = f.info(&sn).with_context(|| format!("missing tensor {sn}"))?;
            if !n_in.is_multiple_of(per) || si.shape != [ne, n_out, n_in / per] {
                bail!("{sn}: shape {:?} for [{ne}, {n_out}, {n_in}] in blocks of {per}", si.shape);
            }
            Ok(())
        };
        // mlx-lm's checkpoints: the release's bytes as 32-bit words (4 bits a weight: NVFP4, with
        // its scales as bytes and `global_scale = amax * 6 * 448 / denominator`; 8: MXFP8).
        let mlx = info.dtype == "U32" && info.shape.len() == 3 && info.shape[..2] == [ne, n_out];
        let dtype = match (mlx, info.shape.last().copied().unwrap_or(0) * 32) {
            (true, b) if b == 4 * n_in => "U8",
            (true, b) if b == 8 * n_in => "F8_E4M3",
            _ => info.dtype.as_str(),
        };
        match dtype {
            // NVFP4: E2M1 pairs, an E4M3 scale per 16, and per expert `amax / denominator`.
            "U8" => {
                let words = if mlx { n_in / 8 } else { n_in / 2 };
                if info.shape != [ne, n_out, words] || !n_in.is_multiple_of(16) {
                    bail!("{full}: NVFP4 shape {:?} for [{ne}, {n_out}, {n_in}]", info.shape);
                }
                scale_shape(16)?;
                if f.info(&sn).is_some_and(|i| i.dtype != "F8_E4M3" && !(mlx && i.dtype == "U8")) {
                    bail!("{sn}: NVFP4 scales must be E4M3");
                }
                let (amax, den) = if mlx {
                    // (`amax / denominator` is `global_scale / (6 * 448)`.)
                    (self.getf(&format!("{full}_global_scale"), ne)?, 2688.0)
                } else {
                    (
                        self.getf(&format!("{full}_amax"), ne)?,
                        self.getf(&format!("{full}_outer_scale_denominator"), 1)?[0],
                    )
                };
                if den == 0.0 {
                    bail!("{full}_outer_scale_denominator is 0");
                }
                let (data, sc) = (f.bytes(full)?, f.bytes(&sn)?);
                for n in [full, &sn] {
                    f.note_use(n, true, false);
                }
                let (per, sper) = (n_out * n_in / 2, n_out * n_in / 16);
                let lins: Result<Vec<Lin>> = (0..ne)
                    .map(|e| {
                        let (d, s) = (&data[e * per..(e + 1) * per], &sc[e * sper..(e + 1) * sper]);
                        let factor = amax[e] / den;
                        if n_in.is_multiple_of(64) {
                            return Ok(Lin::gguf(crate::model::nvfp4_blocks(d, s, factor, n_out, n_in)?));
                        }
                        // Rows that are not whole ggml blocks (tiny test checkpoints): as float32.
                        const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                        let mut out = PageVec::<u8>::zeroed(n_out * n_in * 4);
                        for (i, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                            let code = (d[i / 2] >> (4 * (i % 2))) & 0xF;
                            let v = E2M1[(code & 7) as usize] * if code & 8 != 0 { -1.0 } else { 1.0 };
                            *o = (v * crate::ggml::e4m3_to_f32(s[i / 16]) * factor).to_le_bytes();
                        }
                        Ok(Lin::gguf(GgufMatrix::new(GgmlType::F32, out.into(), n_out, n_in)?))
                    })
                    .collect();
                // (The repacked blocks are what runs: the file's codes and scales go.)
                if lins.is_ok() {
                    f.discard(full);
                    f.discard(&sn);
                }
                lins
            }
            // MXFP8: E4M3 values, an E8M0 exponent per 32 along each row, used in place.
            "F8_E4M3" => {
                if info.shape != [ne, n_out, if mlx { n_in / 4 } else { n_in }] {
                    bail!("{full}: MXFP8 shape {:?} for [{ne}, {n_out}, {n_in}]", info.shape);
                }
                scale_shape(32)?;
                if f.info(&sn).is_some_and(|i| i.dtype != "U8") {
                    bail!("{sn}: MXFP8 scales must be E8M0 bytes");
                }
                let (data, sc) = (f.bytes(full)?, f.bytes(&sn)?);
                for n in [full, &sn] {
                    f.note_use(n, true, false);
                }
                let (per, sper) = (n_out * n_in, n_out * n_in / 32);
                let in_place = crate::ggml::e8m0_in_place();
                let lins: Result<Vec<Lin>> = (0..ne)
                    .map(|e| {
                        if in_place {
                            // SAFETY: both borrow `files`, which outlives every model holding them.
                            let (d, s) = unsafe {
                                (
                                    QBuf::borrowed(&data[e * per..(e + 1) * per]),
                                    QBuf::borrowed(&sc[e * sper..(e + 1) * sper]),
                                )
                            };
                            return Ok(Lin::gguf(GgufMatrix::fp8_e8m0(d, s, n_out, n_in, (1, 32))?));
                        }
                        let mut s = PageVec::<f32>::zeroed(sper);
                        for (d, &b) in s.iter_mut().zip(&sc[e * sper..(e + 1) * sper]) {
                            *d = if b == 255 { f32::NAN } else { (2f32).powi(b as i32 - 127) };
                        }
                        // SAFETY: the weight borrows `files`, which outlives every model holding it.
                        let d = unsafe { QBuf::borrowed(&data[e * per..(e + 1) * per]) };
                        Ok(Lin::gguf(GgufMatrix::fp8(d, s.into(), n_out, n_in, (1, 32))?))
                    })
                    .collect();
                // (The values are read in place; the exponents too on the CPU, else expanded.)
                if lins.is_ok() && !in_place {
                    f.discard(&sn);
                }
                lins
            }
            _ => load_lin_experts(f, full, ne, n_out, n_in, self.quant),
        }
    }

    /// The bank of relative-bias profiles, `[d_rel, extent]` (a GGUF file stores it transposed,
    /// under `NAME_t`).
    fn rel(&self, full: &str, dr: usize, extent: usize) -> Result<Vec<f32>> {
        let tn = format!("{full}_t");
        if !self.files.has(&tn) {
            return self.getf(full, dr * extent);
        }
        let w = self.getf(&tn, dr * extent)?;
        let mut out = vec![0f32; dr * extent];
        for e in 0..extent {
            for d in 0..dr {
                out[d * extent + e] = w[e * dr + d];
            }
        }
        Ok(out)
    }

    /// The message gates and queries of looped layer `l`.
    fn looped(&self, l: usize) -> Result<Option<LoopIn<'a>>> {
        let c = self.cfg;
        let Some(lc) = c.loop_cfg.filter(|_| c.looped(l)) else {
            return Ok(None);
        };
        let (h, r) = (c.hidden_size, lc.gate_rank);
        let gate = |kind: &str| -> Result<Gate<'a>> {
            let p = format!("{P}layers.{l}.controllers.{kind}.");
            Ok(Gate {
                down: self.lin_fp(&format!("{p}down_weight"), r, 2 * h)?,
                forget: self.lin_fp(&format!("{p}forget_weight"), h, r)?,
                gain: self.lin_fp(&format!("{p}gain_weight"), h, r)?,
            })
        };
        let gates = [gate("attn")?, gate("k")?, gate("v")?, gate("mlp")?];
        // A GGUF file has each layer's four queries; the release two tables of `[2 * core, H]`:
        // the query and MLP queries of each looped layer, then its key and value queries.
        let own = format!("{P}layers.{l}.loop_msg_query");
        let all = format!("{P}loop_msg_query_all");
        let query = if self.files.has(&own) {
            self.getf(&own, 4 * h)?
        } else if self.files.has(&all) {
            // mlx-lm's table of every looped layer's four queries.
            let core = lc.end - lc.start;
            let t = self.getf(&all, core * 4 * h)?;
            t[(l - lc.start) * 4 * h..(l - lc.start + 1) * 4 * h].to_vec()
        } else {
            let core = lc.end - lc.start;
            let qm = self.getf(&format!("{P}middle_cycle_cross_pass_queries"), 2 * core * h)?;
            let kv = self.getf(&format!("{P}middle_cycle_cross_pass_kv_queries"), 2 * core * h)?;
            let i = l - lc.start;
            let row = |t: &[f32], j: usize| t[(2 * i + j) * h..(2 * i + j + 1) * h].to_vec();
            [row(&qm, 0), row(&kv, 0), row(&kv, 1), row(&qm, 1)].concat()
        };
        Ok(Some(LoopIn { gates, query }))
    }

    fn attn(&self, l: usize) -> Result<Attn<'a>> {
        let c = self.cfg;
        let (h, hd, dr) = (c.hidden_size, c.heads(l), c.d_rel);
        let (w, kw) = (hd.heads * hd.head_dim, hd.kv_heads * hd.head_dim);
        let ap = format!("{P}layers.{l}.attn.");
        let extent = if c.sliding[l] { c.sliding_window_size } else { c.rel_extent };
        Ok(Attn {
            hd,
            window: c.sliding[l].then_some(c.sliding_window_size),
            extent,
            q: self.lin(&format!("{ap}wq_du.weight"), w, h)?,
            k: self.lin(&format!("{ap}wk_dv.weight"), kw, h)?,
            v: self.lin(&format!("{ap}wv_dv.weight"), kw, h)?,
            r: self.lin(&format!("{ap}wr_du.weight"), hd.heads * dr, h)?,
            gate: self.lin(&format!("{ap}gate_attn.weight"), w, h)?,
            sink: self.lin_fp(&format!("{ap}attn_sink_proj.weight"), hd.heads, h)?,
            o: self.lin(&format!("{ap}wo_ud.weight"), h, w)?,
            q_scale: self.getf(&format!("{ap}q_l2_clip.scale"), w)?,
            r_scale: self.getf(&format!("{ap}rel_l2_clip.scale"), hd.heads * dr)?,
            rel: self.rel(&format!("{ap}rel_logits_proj.proj"), dr, extent)?,
            k_conv: self.kv_conv(&format!("{ap}k_sconv."), kw, hd.kv_heads)?,
            v_conv: self.kv_conv(&format!("{ap}v_sconv."), kw, hd.kv_heads)?,
        })
    }

    fn layer(&self, l: usize) -> Result<Layer<'a>> {
        let c = self.cfg;
        let h = c.hidden_size;
        let lp = format!("{P}layers.{l}.");
        let attn = self.attn(l)?;
        if matches!(attn.k_conv, KvConv::Dynamic { .. }) == c.sliding[l] || matches!(attn.v_conv, KvConv::Dynamic { .. }) == c.sliding[l] {
            bail!(
                "the key / value convolutions are not those of a {} layer",
                if c.sliding[l] { "sliding" } else { "full" }
            );
        }
        let ln_in = if c.sliding[l] {
            InNorm::One(self.getf(&format!("{lp}attn_norm.weight"), h)?)
        } else {
            InNorm::Qkv(
                self.getf(&format!("{lp}q_input_norm.weight"), h)?,
                self.getf(&format!("{lp}k_input_norm.weight"), h)?,
                self.getf(&format!("{lp}v_input_norm.weight"), h)?,
            )
        };
        // `attn_queries.{i}` belongs to layer `i + 1` (layer 0 reads its input as it is).
        let attn_query = match l {
            0 => None,
            _ => Some(self.getf(&format!("{P}attn_res.attn_queries.{}", l - 1), if c.sliding[l] { h } else { 3 * h })?),
        };
        let (ne, k, mi) = (c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size);
        let act = Act::Situ {
            beta: c.mlp_gate_softcap as f32,
            linear_beta: c.mlp_up_softcap as f32,
        };
        let ep = format!("{lp}mlp.experts.");
        let experts = if self.files.has(&format!("{ep}gate_up_proj")) {
            // A GGUF file: gate rows, then up rows; unsloth's files scale each expert's outputs
            // (`.scale`, one factor per expert).
            let load = |name: &str, n_out: usize, n_in: usize| -> Result<Vec<Lin<'a>>> {
                let mut v = crate::model::load_lin_experts(self.files, &format!("{ep}{name}_proj"), ne, n_out, n_in, self.quant)?;
                let sn = format!("{ep}{name}_scale");
                if self.files.has(&sn) {
                    for (l, f) in v.iter_mut().zip(self.getf(&sn, ne)?) {
                        match &mut l.w {
                            crate::model::Weight::Gguf(m) => m.set_factor(f).with_context(|| sn.clone())?,
                            _ => bail!("{sn}: the experts are not in a GGUF format"),
                        }
                    }
                }
                Ok(v)
            };
            Experts::fused(load("gate_up", 2 * mi, h)?, load("down", h, mi)?, k, mi, act)
        } else {
            Experts::interleaved(
                self.experts(&format!("{ep}w13_weight"), 2 * mi, h)?,
                self.experts(&format!("{ep}w2_weight"), h, mi)?,
                k,
                mi,
                act,
            )
        };
        Ok(Layer {
            looped: self.looped(l)?,
            boundary: l.is_multiple_of(c.n_layers_per_attn_res_block),
            attn_query,
            mlp_query: self.getf(&format!("{P}attn_res.mlp_queries.{l}"), h)?,
            ln_in,
            ln_post: self.getf(&format!("{lp}mlp_norm.weight"), h)?,
            attn,
            attn_conv: self.conv(&format!("{lp}attn_sconv.weight"), h)?,
            mlp_conv: self.conv(&format!("{lp}mlp_sconv.weight"), h)?,
            router: self.getf(&format!("{lp}mlp.gate.weight"), ne * h)?,
            bias: self.getf(&format!("{lp}mlp.gate.bias"), ne)?,
            experts,
        })
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scales `x` down to L2 norm `threshold` when it is longer.
#[inline]
fn l2_clip(x: &mut [f32], threshold: f32) {
    let f = threshold / (dot(x, x).sqrt() + CLIP_EPS);
    if f < 1.0 {
        scale_inplace(x, f);
    }
}

/// `x * rsqrt(mean(x^2) + eps)`.
#[inline]
fn rms_unit(x: &mut [f32], eps: f32) {
    let r = 1.0 / (dot(x, x) / x.len() as f32 + eps).sqrt();
    scale_inplace(x, r);
}

/// A causal depthwise convolution with a skip over `x` (`[t, ch]`, positions `start..`), in
/// place: `(sum_lag taps[lag] * x[t - lag] + x[t]) * norm`. `hist` holds the inputs before `x`
/// and gets those of `x`. With `dynamic` (`[t, groups * kernel]`), channel `c` of token `t` adds
/// `dynamic[t][c / (ch / groups) * kernel + lag]` to its taps.
fn short_conv(taps: &[f32], dynamic: Option<(&[f32], usize)>, hist: &mut ConvRing, x: &mut [f32], ch: usize, norm: f32, start: usize) {
    let k = taps.len() / ch;
    let t = x.len() / ch;
    debug_assert_eq!((hist.ch, hist.back), (ch, k - 1));
    // The `k - 1` inputs before `x`, then `x`.
    let back = (k - 1) * ch;
    let mut seq = Scratch::new(back + t * ch);
    for (i, p) in (start as isize - (k - 1) as isize..start as isize).enumerate() {
        seq[i * ch..(i + 1) * ch].copy_from_slice(hist.row(p));
    }
    {
        let x = &*x;
        for_chunks(&mut seq[back..], ch, t > 4, |ti, row| {
            row.copy_from_slice(&x[ti * ch..(ti + 1) * ch])
        });
    }
    let seq = &seq[..];
    for_chunks(x, ch, t > 1, |ti, row| {
        // Row `ti` of `x` is row `ti + k - 1` of `seq`.
        for (c, o) in row.iter_mut().enumerate() {
            *o = seq[back + ti * ch + c];
        }
        for lag in 0..k {
            let src = &seq[(ti + k - 1 - lag) * ch..(ti + k - lag) * ch];
            let w = &taps[lag * ch..(lag + 1) * ch];
            match dynamic {
                None => {
                    for ((o, &s), &wv) in row.iter_mut().zip(src).zip(w) {
                        *o += wv * s;
                    }
                }
                Some((d, groups)) => {
                    let per = ch / groups;
                    for (g, ((oc, sc), wc)) in row.chunks_mut(per).zip(src.chunks(per)).zip(w.chunks(per)).enumerate() {
                        let dw = d[(ti * groups + g) * k + lag];
                        for ((o, &s), &wv) in oc.iter_mut().zip(sc).zip(wc) {
                            *o += (wv + dw) * s;
                        }
                    }
                }
            }
        }
        scale_inplace(row, norm);
    });
    for ti in t.saturating_sub(hist.cap())..t {
        hist.row_mut((start + ti) as isize)
            .copy_from_slice(&seq[back + ti * ch..back + (ti + 1) * ch]);
    }
}

/// The attention residual: for each of the `nq` queries (`[nq, h]`) and each token, the softmax
/// mix (`query . state * h^-0.5`) of the block states (`[t, h]` each, already RMS-normed) and of
/// the RMS-normed `x` (`[t, h]`). One `[t, h]` result per query.
fn mix(states: &[Vec<f32>], x: &[f32], queries: &[f32], h: usize) -> Vec<Vec<f32>> {
    let _timer = prof::start(Cat::Norm);
    let nq = queries.len() / h;
    let t = x.len() / h;
    let scale = (h as f32).powf(-0.5);
    let mut out = scratch(t * nq * h);
    for_chunks(&mut out, nq * h, t > 4, |ti, o| {
        o.fill(0.0);
        let mut xn = x[ti * h..(ti + 1) * h].to_vec();
        rms_unit(&mut xn, MIX_EPS);
        let mut s = vec![0f32; states.len() + 1];
        for (q, oq) in queries.chunks(h).zip(o.chunks_mut(h)) {
            for (v, st) in s.iter_mut().zip(states) {
                *v = dot(q, &st[ti * h..(ti + 1) * h]) * scale;
            }
            *s.last_mut().unwrap() = dot(q, &xn) * scale;
            softmax_inplace(&mut s);
            for (&w, st) in s.iter().zip(states) {
                axpy(oq, w, &st[ti * h..(ti + 1) * h]);
            }
            axpy(oq, *s.last().unwrap(), &xn);
        }
    });
    if nq == 1 {
        return vec![out];
    }
    let per = (0..nq)
        .map(|q| {
            let mut v = scratch(t * h);
            for_chunks(&mut v, h, t > 4, |ti, r| {
                r.copy_from_slice(&out[(ti * nq + q) * h..(ti * nq + q + 1) * h])
            });
            v
        })
        .collect();
    recycle(out);
    per
}

impl<'a> Inkling2Model<'a> {
    pub fn load(cfg: Inkling2Config, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let en = format!("{P}embed.weight");
        if !files.has(&en) {
            bail!("cannot find the token embedding ({en}) in the checkpoint");
        }
        let ld = Loader { files, cfg: &cfg, quant };
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let embed = Table::load(files, &en, v, h, table_quant(quant))?;
        let lm_head = ld.lin(&format!("{P}unembed.weight"), v, h)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            layers.push(ld.layer(l).with_context(|| format!("layer {l}"))?);
        }
        let m = Self {
            embed,
            embed_norm: ld.getf(&format!("{P}embed_norm.weight"), h)?,
            lm_head,
            final_norm: ld.getf(&format!("{P}norm.weight"), h)?,
            final_query: ld.getf(&format!("{P}attn_res.output_query"), h)?,
            layers,
            cfg,
            quant,
            kv: KvCache::F32,
            stream: None,
            device: std::sync::OnceLock::new(),
            retired: Vec::new(),
            gpu_budget: None,
            weight_bytes: files.total_bytes(),
        };
        // Routed experts that cannot all stay in memory are read through the page cache as they
        // are routed: no accelerator mapping may pin them.
        if too_big(files.total_bytes()) {
            for l in &m.layers {
                for e in l.experts.lins().into_iter().flatten() {
                    e.set_transient_device(true);
                }
            }
        }
        Ok(m)
    }

    /// Attaches the modules of a runtime LoRA adapter (see `crate::lora`) by the release's module
    /// names (`layers.N.attn.wo_ud`, `layers.N.mlp.experts`, `layers.N.controllers.attn.down_weight`,
    /// `lm_head` for the unembedding), and its token-embedding update. A GGUF adapter's names
    /// map to these (`blk.N.attn_output.weight` is `layers.N.attn.wo_ud`, `ffn_down_exps` the
    /// experts' `down_proj`). Returns the number patched.
    pub fn attach_lora(&mut self, ad: &mut crate::lora::LoraAdapter) -> Result<usize> {
        // The GPU engine is built for the weights' adapters (Metal runs them as side paths, CUDA
        // leaves adapted models to the CPU): decided again at the next pass. The engine it
        // replaces stays to bring the sessions that ran on it back to the host caches.
        if let Some(Some(e)) = std::mem::take(&mut self.device).into_inner() {
            self.retired.push(e);
        }
        let c = &self.cfg;
        let (h, v, dr) = (c.hidden_size, c.vocab_size, c.d_rel);
        let rank = c.loop_cfg.map_or(0, |l| l.gate_rank);
        let k = c.conv_kernel_size;
        if let Some(d) = ad.take_embed(v, h)? {
            self.embed.add_lora(d);
        }
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("layers.{l}.");
            ad.attach_experts(&mut layer.experts, &format!("{p}mlp.experts"), h)?;
            let a = &mut layer.attn;
            let hd = a.hd;
            let (w, kw) = (hd.heads * hd.head_dim, hd.kv_heads * hd.head_dim);
            for (lin, name, n_out, n_in) in [
                (&mut a.q, "wq_du", w, h),
                (&mut a.k, "wk_dv", kw, h),
                (&mut a.v, "wv_dv", kw, h),
                (&mut a.r, "wr_du", hd.heads * dr, h),
                (&mut a.gate, "gate_attn", w, h),
                (&mut a.sink, "attn_sink_proj", hd.heads, h),
                (&mut a.o, "wo_ud", h, w),
            ] {
                ad.attach(lin, &format!("{p}attn.{name}"), n_out, n_in, None, None)?;
            }
            for (conv, name) in [(&mut a.k_conv, "k_sconv"), (&mut a.v_conv, "v_sconv")] {
                if let KvConv::Dynamic { proj, .. } = conv {
                    ad.attach(proj, &format!("{p}attn.{name}.weight_proj"), hd.kv_heads * k, h, None, None)?;
                }
            }
            if let Some(li) = &mut layer.looped {
                for (g, kind) in li.gates.iter_mut().zip(["attn", "k", "v", "mlp"]) {
                    let q = format!("{p}controllers.{kind}.");
                    ad.attach(&mut g.down, &format!("{q}down_weight"), rank, 2 * h, None, None)?;
                    ad.attach(&mut g.forget, &format!("{q}forget_weight"), h, rank, None, None)?;
                    ad.attach(&mut g.gain, &format!("{q}gain_weight"), h, rank, None, None)?;
                }
            }
        }
        ad.attach(&mut self.lm_head, "lm_head", v, h, None, None)?;
        ad.finish()
    }

    pub fn new_session(&self) -> Inkling2Session {
        let c = &self.cfg;
        let back = c.conv_kernel_size - 1;
        let slot_layer = c.slot_layers();
        Inkling2Session {
            layers: slot_layer
                .into_iter()
                .map(|l| {
                    let hd = c.heads(l);
                    let kw = hd.kv_heads * hd.head_dim;
                    let mode = match self.kv {
                        KvCache::Q8 if !hd.head_dim.is_multiple_of(32) => KvCache::F16,
                        m => m,
                    };
                    LayerState {
                        k: KvRows::new(mode),
                        v: KvRows::new(mode),
                        hist: [
                            ConvRing::new(back + 1, kw),
                            ConvRing::new(back + 1, kw),
                            ConvRing::new(back + 1, c.hidden_size),
                            ConvRing::new(back + 1, c.hidden_size),
                        ],
                        ..Default::default()
                    }
                })
                .collect(),
            pos: 0,
            iv: Default::default(),
            dev: None,
            dev_off: false,
            checkpoints: Vec::new(),
            floor: 0,
        }
    }

    /// Positions kept per cache slot by a [`Checkpoint`]: a sliding slot's window, 0 for a full
    /// slot.
    fn checkpoint_keep(&self) -> Vec<usize> {
        let c = &self.cfg;
        c.slot_layers()
            .into_iter()
            .map(|l| if c.sliding[l] { c.sliding_window_size } else { 0 })
            .collect()
    }

    /// Takes a [`Checkpoint`] of the current position of `sess` (none when it has one there or
    /// `max` is 0), keeping at most `max`: past that the lowest one (a shared prefix, say) and
    /// the newest stay, the second lowest goes.
    pub fn checkpoint(&self, sess: &mut Inkling2Session, max: usize) {
        let pos = sess.pos;
        if max == 0 || pos == 0 || sess.checkpoints.iter().any(|c| c.pos == pos) {
            return;
        }
        let keep = self.checkpoint_keep();
        let dev = sess.dev.as_ref().and_then(|d| d.0.checkpoint(pos, &sess.layers, &keep));
        let slots = match dev {
            Some(s) => s,
            None => {
                self.to_host(sess);
                sess.layers
                    .iter()
                    .zip(&keep)
                    .map(|(st, &k)| {
                        let kw = st.hist[0].ch;
                        let conv: [Vec<f32>; 4] = std::array::from_fn(|i| {
                            let r = &st.hist[i];
                            (0..r.back)
                                .flat_map(|j| r.row(pos as isize - (r.back - j) as isize).iter().copied())
                                .collect::<Vec<f32>>()
                        });
                        let (first, k, v) = if k > 0 {
                            let lo = pos.saturating_sub(k).max(st.first);
                            let (a, b) = ((lo - st.first) * kw, (pos - st.first) * kw);
                            (lo, st.k.range(a, b), st.v.range(a, b))
                        } else {
                            (pos, KvRows::new(st.k.mode), KvRows::new(st.v.mode))
                        };
                        SlotCheckpoint { first, k, v, conv }
                    })
                    .collect()
            }
        };
        let cp = std::sync::Arc::new(Checkpoint { pos, slots });
        let at = sess.checkpoints.partition_point(|c| c.pos < pos);
        sess.checkpoints.insert(at, cp);
        while sess.checkpoints.len() > max {
            let i = if max > 1 { 1 } else { 0 };
            sess.checkpoints.remove(i);
        }
    }

    /// Whether `sess` has a checkpoint of position `pos`.
    pub fn has_checkpoint(&self, sess: &Inkling2Session, pos: usize) -> bool {
        sess.checkpoints.iter().any(|c| c.pos == pos)
    }

    /// The latest position up to `len` that `sess` can go back to ([`Self::cut_to`]): `len`
    /// itself when within what it goes back ([`Self::rollback_limit`]), else its latest
    /// checkpoint up to `len`, else 0.
    pub fn reachable(&self, sess: &Inkling2Session, len: usize) -> usize {
        if len >= sess.pos {
            return sess.pos;
        }
        if len >= sess.floor && sess.pos - len <= self.rollback_limit() {
            return len;
        }
        sess.checkpoints.iter().map(|c| c.pos).filter(|&p| p <= len).max().unwrap_or(0)
    }

    /// Goes back to position `len`: within what the session goes back, or to a checkpoint of
    /// that position. The logits from there are those of a session that ran the same tokens up
    /// to `len` (in the same passes) and nothing after.
    pub fn cut_to(&self, sess: &mut Inkling2Session, len: usize) -> Result<()> {
        if len >= sess.pos {
            return Ok(());
        }
        if len >= sess.floor && sess.pos - len <= self.rollback_limit() {
            return sess.truncate(len);
        }
        let Some(cp) = sess.checkpoints.iter().find(|c| c.pos == len).cloned() else {
            bail!("Inkling-2: no checkpoint of position {len} to go back to (at {})", sess.pos);
        };
        let restored = match &mut sess.dev {
            Some(d) => d.0.restore(&cp, &sess.layers),
            None => false,
        };
        if !restored {
            self.to_host(sess);
            let keep = self.checkpoint_keep();
            for ((st, s), &k) in sess.layers.iter_mut().zip(&cp.slots).zip(&keep) {
                let kw = st.hist[0].ch;
                if k > 0 {
                    st.k = s.k.clone();
                    st.v = s.v.clone();
                    st.first = s.first;
                } else {
                    let rows = len.saturating_sub(st.first);
                    st.k.truncate(rows * kw);
                    st.v.truncate(rows * kw);
                }
                for (r, rows) in st.hist.iter_mut().zip(&s.conv) {
                    let (back, ch) = (r.back, r.ch);
                    for j in 0..back {
                        r.row_mut(len as isize - (back - j) as isize)
                            .copy_from_slice(&rows[j * ch..(j + 1) * ch]);
                    }
                }
            }
        }
        sess.pos = len;
        sess.floor = len;
        sess.checkpoints.retain(|c| c.pos <= len);
        Ok(())
    }

    /// Most draft tokens a speculative round verifies: [`DEVICE_MAX_DRAFT`] when the passes run
    /// on the accelerator (within its passes of one-token results: 3 on CUDA), else
    /// [`MAX_DRAFT`]; within what a session can go back ([`Self::rollback_limit`]).
    pub fn max_draft(&self) -> usize {
        let lim = self.rollback_limit().saturating_sub(1);
        match self.device() {
            Some(d) => DEVICE_MAX_DRAFT.min(lim).min(d.exact_tokens().saturating_sub(1)),
            None => MAX_DRAFT.min(lim),
        }
    }

    /// Whether draft models can run (sessions go back, and the device takes the passes that
    /// capture their streams: not on CUDA, where prompt lookup still runs).
    pub fn can_draft(&self) -> bool {
        self.can_rollback() && self.device().is_none_or(|d| d.captures())
    }

    /// Runs `tokens` on the device when it can (`want` as [`DevWant`]); false otherwise (the
    /// session then has its state on the host).
    fn run_device(&self, sess: &mut Inkling2Session, tokens: &[u32], mm: &[(usize, &[f32])], want: DevWant) -> bool {
        let t = tokens.len();
        if sess.iv.active() || t == 0 || sess.dev_off {
            self.to_host(sess);
            return false;
        }
        let Some(d) = self.device().filter(|d| t <= d.max_tokens()) else {
            self.to_host(sess);
            return false;
        };
        let x = self.embed_rows(tokens, mm);
        if d.run(self, sess, &x, t, want) {
            return true;
        }
        self.to_host(sess);
        false
    }

    /// The input rows of `tokens`: the normed embeddings, the positions listed in `mm` taking
    /// the given rows.
    fn embed_rows(&self, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let t = tokens.len();
        let _timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, t > 1, |ti, row| {
            self.embed.row(tokens[ti] as usize, row);
            rmsnorm_f32w_inplace(row, &self.embed_norm, eps);
        });
        for &(i, e) in mm {
            x[i * h..(i + 1) * h].copy_from_slice(e);
        }
        x
    }

    /// The device that runs whole passes, if any (decided once, at the first pass: the active
    /// backend, `--gpu-decode`, the weights' kinds and size).
    pub(crate) fn device(&self) -> Option<&dyn DevEngine> {
        if self.stream.is_some() || !crate::gpu::decode_enabled() {
            return None;
        }
        self.device
            .get_or_init(|| {
                if let Some(b) = self.gpu_budget.filter(|&b| self.weight_bytes > b) {
                    eprintln!(
                        "[inkling: the {:.1} GB of weights exceed the GPU budget ({:.1} GB): passes on the CPU]",
                        self.weight_bytes as f64 / 1e9,
                        b as f64 / 1e9
                    );
                    return None;
                }
                #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
                if let Some(e) = cuda::Engine::new(self) {
                    return Some(Box::new(e) as Box<dyn DevEngine>);
                }
                #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
                if let Some(e) = crate::metal::inkling::build(self) {
                    return Some(e);
                }
                None
            })
            .as_deref()
    }

    /// Brings the state of `sess` back to the host caches when it is on the device.
    pub(crate) fn to_host(&self, sess: &mut Inkling2Session) {
        if sess.dev.is_some() {
            // (The engine now, else the one an adapter's attach replaced: see `attach_lora`.)
            match self.device.get().and_then(|d| d.as_deref()).or(self.retired.last().map(|e| &**e)) {
                Some(d) => d.to_host(self, sess),
                None => sess.dev = None,
            }
        }
    }

    /// Positions a session of this model can go back ([`Inkling2Session::truncate`]): what its
    /// device allows (0 on CUDA), [`ROLLBACK`] on the CPU. Speculative decoding needs one.
    pub fn rollback_limit(&self) -> usize {
        self.device().map_or(ROLLBACK, |d| d.rollback())
    }

    /// Whether sessions of this model can be written to snapshots: on the CPU, or on a device
    /// that reads them where they are (Metal).
    pub fn snapshots(&self) -> bool {
        self.device().is_none_or(|d| d.snapshots())
    }

    /// Whether sessions of this model can go back at all (prompt lookup and drafts need it).
    pub fn can_rollback(&self) -> bool {
        self.rollback_limit() > 0
    }

    /// Measurements only: puts `n` positions of placeholder keys and values (pseudo-random,
    /// clipped and normed like real ones) in the caches of a fresh session, as a context of `n`
    /// tokens would (the sliding-window layers keep their last window), so that the memory and
    /// the speed of decoding at that context can be measured without running the prompt. The
    /// output is meaningless.
    #[doc(hidden)]
    pub fn fill_for_bench(&self, sess: &mut Inkling2Session, n: usize) {
        let c = &self.cfg;
        let mut slot_layer = vec![0usize; c.n_slots()];
        for s in c.steps() {
            slot_layer[s.slot] = s.layer;
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for (st, &l) in sess.layers.iter_mut().zip(&slot_layer) {
            let hd = c.heads(l);
            let kw = hd.kv_heads * hd.head_dim;
            let keep = if c.sliding[l] { n.min(c.sliding_window_size + ROLLBACK) } else { n };
            let mut row = vec![0f32; kw];
            for _ in 0..keep {
                for x in row.iter_mut() {
                    *x = rnd();
                }
                st.k.push(&row);
                st.v.push(&row);
            }
            st.first = n - keep;
        }
        sess.pos = n;
    }

    /// Bytes of the attention caches of `sess`.
    pub fn kv_bytes(&self, sess: &Inkling2Session) -> usize {
        if let Some(d) = &sess.dev {
            return d.0.bytes();
        }
        sess.layers.iter().map(|l| l.k.bytes() + l.v.bytes()).sum()
    }

    /// Attention-cache bytes per token of context, once the sliding-window layers are full (the
    /// full-attention applications: every pass of a looped layer keeps its own cache), and the
    /// most the sliding-window applications hold whatever the context.
    pub fn kv_bytes_per_token(&self, kv: KvCache) -> (f64, f64) {
        let c = &self.cfg;
        let (mut full, mut sliding) = (0f64, 0f64);
        for s in c.steps() {
            let hd = c.heads(s.layer);
            let per = 2.0 * (hd.kv_heads * hd.head_dim) as f64 * kv.bytes_per_value();
            if c.sliding[s.layer] {
                // At most twice the window and the rollback margin are kept (the cache is cut
                // a window at a time).
                sliding += per * (2 * c.sliding_window_size + ROLLBACK) as f64;
            } else {
                full += per;
            }
        }
        (full, sliding)
    }

    fn dense_lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let a = &l.attn;
        let mut v = vec![&a.q, &a.k, &a.v, &a.r, &a.gate, &a.sink, &a.o];
        for c in [&a.k_conv, &a.v_conv] {
            if let KvConv::Dynamic { proj, .. } = c {
                v.push(proj);
            }
        }
        v
    }

    /// Weight bytes read per decoded token (the routed experts at top-k).
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes();
        // (A looped layer is read once per pass.)
        for s in self.cfg.steps() {
            let l = &self.layers[s.layer];
            n += self.dense_lins(l).iter().map(|w| w.bytes()).sum::<usize>();
            n += l.experts.bytes_per_token();
        }
        n
    }

    /// The streaming units in execution order (see `crate::stream`): the layers (routed experts
    /// read on demand), then the LM head.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        let mut v: Vec<crate::stream::UnitSpec> = self
            .layers
            .iter()
            .enumerate()
            .map(|(l, layer)| crate::stream::UnitSpec {
                name: format!("layer.{l}"),
                ranges: self.dense_lins(layer).iter().flat_map(|w| w.mem_ranges()).collect(),
                uses: 1.0,
                parts: layer.experts.ranges(),
            })
            .collect();
        v.push(crate::stream::UnitSpec {
            name: "lm_head".into(),
            ranges: self.lm_head.mem_ranges(),
            uses: 1.0,
            parts: Vec::new(),
        });
        v
    }

    /// Marks the weights of the units with `streamed[i]` set as streamed (no accelerator state
    /// kept between calls), and every routed expert.
    pub fn mark_streamed(&self, streamed: &[bool]) {
        for (l, layer) in self.layers.iter().enumerate() {
            let s = streamed.get(l).copied().unwrap_or(false);
            for w in self.dense_lins(layer) {
                w.set_transient_device(s);
            }
            for e in layer.experts.lins().into_iter().flatten() {
                e.set_transient_device(true);
            }
        }
        self.lm_head
            .set_transient_device(streamed.get(self.layers.len()).copied().unwrap_or(false));
    }

    /// Brackets every layer and the LM head with `hook`'s units (None stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// The unembedding over `hidden` (`[t, hidden]`), cut to the tokens: `[t, unpadded_vocab]`.
    fn head(&self, hidden: &[f32], t: usize, cap: bool) -> Vec<f32> {
        if let Some(l) = self.device().and_then(|d| d.head(self, hidden, t, cap)) {
            return l;
        }
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let (v, u) = (c.vocab_size, c.unpadded_vocab_size);
        let mut logits = vec![0f32; t * v];
        {
            let _unit = crate::stream::enter(&self.stream, self.layers.len());
            linear(&self.lm_head, v, c.hidden_size, hidden, t, &mut logits);
        }
        if u < v {
            for ti in 1..t {
                logits.copy_within(ti * v..ti * v + u, ti * u);
            }
            logits.truncate(t * u);
        }
        if cap {
            let s = c.final_logit_softcapping as f32;
            let chunk = u.min(4096);
            for_chunks(&mut logits, chunk, t * u > 16 * chunk, |_, r| {
                for x in r.iter_mut() {
                    *x = (*x / s).tanh() * s;
                }
            });
        }
        logits
    }

    /// Soft-capped logits `[t, unpadded_vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.head(hidden, t, true)
    }

    /// The input embedding of `token` (`[hidden]`, before its norm).
    pub fn embed_row(&self, token: u32, out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        if let Some(s) = &self.stream {
            s.need_ranges(&self.embed.row_ranges(token as usize, h));
        }
        self.embed.row(token as usize, out);
    }

    /// Greedy decoding with the passes chained on the device: runs `first` and the tokens it leads
    /// to, each the argmax of the logits after the one before (the first index of the largest
    /// soft-capped logit, as [`crate::sampler::argmax`]), at most `max` passes, the device
    /// computing each pass's input row from the pass before so that the next pass is queued
    /// while the host takes a token. `on` gets each kept pass's argmax in order and says whether
    /// to go on ([`crate::generate::Chain`]: a token it stops at is not run, as
    /// [`crate::generate::decode`] leaves a stop token unfed). The passes kept, or None (the
    /// session untouched) when the device does not do it.
    pub fn decode_greedy(
        &self,
        sess: &mut Inkling2Session,
        first: u32,
        max: usize,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        if sess.iv.active() || sess.dev_off || crate::gpu::decode::check() || !crate::gpu::decode_chain_enabled() {
            return None;
        }
        let d = self.device()?;
        let pos0 = sess.pos;
        let (ran, kept) = d.greedy(self, sess, first, max, on)?;
        if kept < ran {
            sess.truncate(pos0 + kept).expect("a pass past the stop goes back");
        }
        Some(kept)
    }

    /// [`Self::decode_greedy`] for a temperature draw over the whole vocabulary
    /// ([`crate::generate::TextLm::decode_sampled`]): each pass's token drawn on the device from
    /// the uniform `draw` gives (one a pass queued, in order). None (the session untouched)
    /// when the device does not do it.
    pub fn decode_sampled(
        &self,
        sess: &mut Inkling2Session,
        first: u32,
        max: usize,
        temperature: f32,
        draw: &mut dyn FnMut() -> f32,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        if sess.iv.active() || sess.dev_off || crate::gpu::decode::check() || !crate::gpu::decode_chain_enabled() {
            return None;
        }
        let d = self.device()?;
        let pos0 = sess.pos;
        let (ran, kept) = d.sampled(self, sess, first, max, &[], None, temperature, draw, on)?;
        if kept < ran {
            sess.truncate(pos0 + kept).expect("a pass past the stop goes back");
        }
        Some(kept)
    }

    /// [`Self::decode_sampled`] that also appends to `feats` the streams entering `layers` of
    /// each kept pass (as [`Self::decode_greedy_feats`]).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_sampled_feats(
        &self,
        sess: &mut Inkling2Session,
        first: u32,
        max: usize,
        layers: &[usize],
        feats: &mut Vec<f32>,
        temperature: f32,
        draw: &mut dyn FnMut() -> f32,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        if sess.iv.active() || sess.dev_off || crate::gpu::decode::check() || !crate::gpu::decode_chain_enabled() {
            return None;
        }
        let d = self.device()?;
        let pos0 = sess.pos;
        let n0 = feats.len();
        let (ran, kept) = d.sampled(self, sess, first, max, layers, Some(feats), temperature, draw, on)?;
        if kept < ran {
            sess.truncate(pos0 + kept).expect("a pass past the stop goes back");
        }
        feats.truncate(n0 + kept * layers.len() * self.cfg.hidden_size);
        Some(kept)
    }

    /// [`Self::decode_greedy`] that also appends to `feats` the residual streams entering
    /// `layers` of each kept pass (as [`crate::dspark::Target::forward_feats`]), for a DSpark
    /// draft's context. None (the session and `feats` untouched) when the device does not do it.
    pub fn decode_greedy_feats(
        &self,
        sess: &mut Inkling2Session,
        first: u32,
        max: usize,
        layers: &[usize],
        feats: &mut Vec<f32>,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        if sess.iv.active() || sess.dev_off || crate::gpu::decode::check() || !crate::gpu::decode_chain_enabled() {
            return None;
        }
        let d = self.device()?;
        let pos0 = sess.pos;
        let n0 = feats.len();
        let (ran, kept) = d.greedy_capture(self, sess, first, max, layers, feats, on)?;
        if kept < ran {
            sess.truncate(pos0 + kept).expect("a pass past the stop goes back");
        }
        feats.truncate(n0 + kept * layers.len() * self.cfg.hidden_size);
        Some(kept)
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut Inkling2Session, tokens: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        // `OXINFER_GPU_DECODE_CHECK`: the same step on the CPU, from a host copy.
        let before = (crate::gpu::decode::check() && self.device().is_some()).then(|| sess.clone());
        let want = DevWant {
            hidden: None,
            logits: Some((&mut logits, 1)),
            capture: None,
        };
        if self.run_device(sess, tokens, &[], want) {
            if let Some(mut b) = before {
                let h = self.cfg.hidden_size;
                let hs = self.forward_host(&mut b, tokens, &[], &[], &mut Vec::new());
                crate::gpu::decode::report(b.pos - 1, &self.logits(&hs[hs.len() - h..], 1), &logits);
            }
            return logits;
        }
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden(sess, tokens);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos`, appending to the caches; returns the final-normed hidden
    /// states `[t, hidden]`.
    pub fn forward_hidden(&self, sess: &mut Inkling2Session, tokens: &[u32]) -> Vec<f32> {
        self.forward_capture(sess, tokens, &[], &[], &mut Vec::new())
    }

    /// [`Self::forward_hidden`] where the positions listed in `mm` (relative to this call) take
    /// the given embedding (an image feature, which skips the embedding norm), and that also
    /// writes to `feats` (`[t, layers.len() * hidden]`) the stream entering each of `layers` (the
    /// steps of a pass, [`Inkling2Config::steps`]), `steps().len()` standing for the last
    /// layer's output and `steps().len() + 1` for the final-normed output.
    pub fn forward_capture(
        &self,
        sess: &mut Inkling2Session,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        {
            let mut hidden = Vec::new();
            let want = DevWant {
                hidden: Some(&mut hidden),
                logits: None,
                capture: (!layers.is_empty()).then_some((layers, &mut *feats)),
            };
            if self.run_device(sess, tokens, mm, want) {
                if layers.is_empty() {
                    feats.clear();
                }
                self.capture_final(layers, feats, &hidden, t);
                return hidden;
            }
        }
        self.forward_host(sess, tokens, mm, layers, feats)
    }

    /// Writes the final-normed output `hidden` (`[t, H]`) into the `feats` slots of index
    /// `steps().len() + 1` (a device pass captures the streams entering steps only).
    fn capture_final(&self, layers: &[usize], feats: &mut [f32], hidden: &[f32], t: usize) {
        let (h, nf, fin) = (self.cfg.hidden_size, layers.len(), self.cfg.steps().len() + 1);
        for (k, _) in layers.iter().enumerate().filter(|e| *e.1 == fin) {
            for ti in 0..t {
                feats[(ti * nf + k) * h..(ti * nf + k + 1) * h].copy_from_slice(&hidden[ti * h..(ti + 1) * h]);
            }
        }
    }

    /// [`Self::forward_capture`] on the CPU.
    fn forward_host(
        &self,
        sess: &mut Inkling2Session,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        let t = tokens.len();
        self.to_host(sess);
        let c = &self.cfg;
        let h = c.hidden_size;
        let eps = c.rms_norm_eps as f32;
        let start = sess.pos;
        // A prompt chunk's temporaries (megabytes each, about forty per layer application) are
        // taken from and given back to the buffers of the pass: fresh ones are mapped page by
        // page by the threads that write them first, which on 32 cores cost more than the
        // kernels (a decode step's are too small to be kept).
        let _scratch = crate::kernels::ScratchScope::open();
        let mut x = self.embed_rows(tokens, mm);
        let iv = sess.iv.active().then(|| sess.iv.clone());
        let point = |p: usize, x: &mut [f32]| {
            if let Some(iv) = &iv {
                iv.point(p, x, h, start);
            }
        };
        point(0, &mut x);
        let nf = layers.len();
        feats.clear();
        feats.resize(t * nf * h, 0.0);
        let mut capture = |l: usize, x: &[f32]| {
            for (k, _) in layers.iter().enumerate().filter(|e| *e.1 == l) {
                for ti in 0..t {
                    feats[(ti * nf + k) * h..(ti * nf + k + 1) * h].copy_from_slice(&x[ti * h..(ti + 1) * h]);
                }
            }
        };
        // The inputs of the first layer of each block so far, RMS-normed (all that reads them).
        let mut states: Vec<Vec<f32>> = Vec::new();
        let conv_norm = 1.0 / ((c.conv_kernel_size + 1) as f32).sqrt();
        let steps = c.steps();
        // The stream and the number of block states entering the loop.
        let mut pre: (Vec<f32>, usize) = (Vec::new(), 0);
        // What the messages of a pass are read from: the previous pass's block states after its
        // first one, and its output.
        let mut bank: Option<(Vec<Vec<f32>>, Vec<f32>)> = None;
        for (si, step) in steps.iter().enumerate() {
            let l = step.layer;
            let layer = &self.layers[l];
            if let Some(lc) = c.loop_cfg
                && l == lc.start
            {
                if step.pass == 0 {
                    pre = (x.clone(), states.len());
                } else {
                    // Every pass starts again from the loop's input.
                    let prev = states.split_off(pre.1 + 1);
                    states.drain(pre.1..).for_each(recycle);
                    if let Some((bs, bx)) = bank.replace((prev, std::mem::replace(&mut x, pre.0.clone()))) {
                        bs.into_iter().for_each(recycle);
                        recycle(bx);
                    }
                }
            }
            capture(si, &x);
            let _unit = crate::stream::enter(&self.stream, l);
            let st = &mut sess.layers[step.slot];
            // The messages of this layer (query, key, value, MLP), in the passes after the first.
            let msgs = match (&bank, &layer.looped) {
                (Some((bs, bx)), Some(li)) if step.pass > 0 => Some((li, mix(bs, bx, &li.query, h))),
                _ => None,
            };
            let gated = |i: usize, src: &[f32]| -> Option<Scratch> {
                msgs.as_ref()
                    .map(|(li, m)| Scratch::from_vec(self.gate(&li.gates[i], src, &m[i], t)))
            };
            // What the attention reads: the layer's input, or mixes of it with the block states.
            let mixed = layer.attn_query.as_ref().map(|q| mix(&states, &x, q, h));
            let input = |i: usize| -> &[f32] {
                match &mixed {
                    Some(m) => &m[i.min(m.len() - 1)],
                    None => &x,
                }
            };
            let normed = |w: &[f32], src: &[f32]| {
                let mut y = Scratch::new(t * h);
                rmsnorm_f32w_rows(src, h, w, eps, &mut y);
                y
            };
            let mut a = scratch(t * h);
            match (&layer.ln_in, msgs.is_some()) {
                (InNorm::One(w), false) => {
                    let xn = normed(w, input(0));
                    self.attention(&layer.attn, [&xn, &xn, &xn], t, start, st, &mut a);
                }
                // The messages make the query, key and value inputs of a sliding layer differ.
                (InNorm::One(w), true) => {
                    let [xq, xk, xv] = [0, 1, 2].map(|i| normed(w, &gated(i, input(0)).unwrap()));
                    self.attention(&layer.attn, [&xq, &xk, &xv], t, start, st, &mut a);
                }
                (InNorm::Qkv(wq, wk, wv), m) => {
                    let src = |i: usize, w: &[f32]| match m {
                        true => normed(w, &gated(i, input(i)).unwrap()),
                        false => normed(w, input(i)),
                    };
                    let (xq, xk, xv) = (src(0, wq), src(1, wk), src(2, wv));
                    self.attention(&layer.attn, [&xq, &xk, &xv], t, start, st, &mut a);
                }
            }
            mixed.into_iter().flatten().for_each(recycle);
            short_conv(&layer.attn_conv, None, &mut st.hist[2], &mut a, h, conv_norm, start);
            // The stream inside the block: a block's first layer starts it from its attention.
            if layer.boundary {
                for_chunks(&mut x, h, t > 4, |_, r| rms_unit(r, MIX_EPS));
                states.push(std::mem::replace(&mut x, a));
            } else {
                add_inplace(&mut x, &a);
                recycle(a);
            }
            point(2 * si + 1, &mut x);
            let mut hn = Scratch::from_vec(mix(&states, &x, &layer.mlp_query, h).pop().unwrap());
            if let Some(g) = gated(3, &hn) {
                hn = g;
            }
            rmsnorm_f32w_rows_inplace(&mut hn, h, &layer.ln_post, eps);
            // (The experts add to it.)
            let mut o = Scratch::zeroed(t * h);
            let hook = iv.as_ref().and_then(|s| s.hook(si, start));
            {
                let need = |ids: &[usize]| {
                    if let Some(s) = &self.stream {
                        s.need_parts(l, ids)
                    }
                };
                let need: NeedExperts = self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize]));
                self.moe(layer, &hn, t, &mut o, need, hook.as_ref());
            }
            short_conv(&layer.mlp_conv, None, &mut st.hist[3], &mut o, h, conv_norm, start);
            add_inplace(&mut x, &o);
            point(2 * si + 2, &mut x);
            drop((hn, o));
            if let Some((_, m)) = msgs {
                m.into_iter().for_each(recycle);
            }
        }
        capture(steps.len(), &x);
        let mut out = mix(&states, &x, &self.final_query, h).pop().unwrap();
        rmsnorm_f32w_rows_inplace(&mut out, h, &self.final_norm, eps);
        // (One more index: the final-normed output, what Inkling-2's drafter reads last.)
        capture(steps.len() + 1, &out);
        sess.pos += t;
        states.into_iter().for_each(recycle);
        if let Some((bs, bx)) = bank {
            bs.into_iter().for_each(recycle);
            recycle(bx);
        }
        recycle(x);
        out
    }

    /// A message gate over the input `x` and the message `m` (`[t, H]` each).
    fn gate(&self, g: &Gate, x: &[f32], m: &[f32], t: usize) -> Vec<f32> {
        let _timer = prof::start(Cat::Norm);
        let h = self.cfg.hidden_size;
        let r = self.cfg.loop_cfg.map_or(0, |c| c.gate_rank);
        let eps = self.cfg.rms_norm_eps as f32;
        let mut cat = Scratch::new(t * 2 * h);
        for_chunks(&mut cat, 2 * h, t > 4, |ti, row| {
            row[..h].copy_from_slice(&x[ti * h..(ti + 1) * h]);
            row[h..].copy_from_slice(&m[ti * h..(ti + 1) * h]);
            rms_unit(&mut row[..h], eps);
            rms_unit(&mut row[h..], eps);
        });
        let mut d = vec![0f32; t * r];
        // Passes of a few tokens (decoding, verification) with float gates: [`gate_rows`]
        // (`OXINFER_INKLING_GATE_ROWS=0`: the GEMMs, as prompts).
        static ROWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if t < TN
            && r > 0
            && *ROWS.get_or_init(|| std::env::var("OXINFER_INKLING_GATE_ROWS").map_or(true, |v| v != "0"))
            && let (Some(wd), Some(wf), Some(wg)) = (FloatRows::of(&g.down), FloatRows::of(&g.forget), FloatRows::of(&g.gain))
        {
            let mut out = scratch(t * h);
            gate_rows(&wd, &wf, &wg, &cat, x, m, h, r, &mut d, &mut out);
            return out;
        }
        linear(&g.down, r, 2 * h, &cat, t, &mut d);
        let (mut f, mut gn) = (Scratch::new(t * h), Scratch::new(t * h));
        linear_multi(&mut [(&g.forget, h, &mut f[..]), (&g.gain, h, &mut gn[..])], r, &d, t);
        let mut out = scratch(t * h);
        for_chunks(&mut out, h, t > 4, |ti, o| {
            let at = ti * h;
            for (j, o) in o.iter_mut().enumerate() {
                *o = (1.0 - f[at + j].tanh()) * x[at + j] + gn[at + j].tanh() * m[at + j];
            }
        });
        out
    }

    /// Routes each token (sigmoid scores, the top k of score plus bias, renormalized and scaled)
    /// and runs its experts.
    fn moe(&self, layer: &Layer, x: &[f32], t: usize, y: &mut [f32], need: NeedExperts, hook: Option<&NeuronHook>) {
        let c = &self.cfg;
        let h = c.hidden_size;
        let ne = c.n_routed_experts;
        let mut logits = vec![0f32; t * ne];
        crate::moe::router_logits(&layer.router, x, h, t, &mut logits);
        let routes: Vec<_> = logits
            .chunks(ne)
            .map(|lg| route_sigmoid(lg, &layer.bias, c.num_experts_per_tok, c.route_scale as f32))
            .collect();
        layer.experts.forward(x, t, h, &routes, y, need, hook, None);
    }

    /// Attention over the normed inputs `xs` (`[t, H]` each: what the queries, the keys and the
    /// values are projected from) into `out`, appending to the layer's cache.
    fn attention(&self, a: &Attn, xs: [&[f32]; 3], t: usize, start: usize, st: &mut LayerState, out: &mut [f32]) {
        let c = &self.cfg;
        let (h, dr, kn) = (c.hidden_size, c.d_rel, c.conv_kernel_size);
        let Heads {
            heads: nh,
            kv_heads: nkv,
            head_dim: d,
        } = a.hd;
        let (w, kw) = (nh * d, nkv * d);
        let [xq, xk, xv] = xs;
        // (Each written whole by its projection.)
        let mut q = Scratch::new(t * w);
        let mut r = Scratch::new(t * nh * dr);
        let mut gate = Scratch::new(t * w);
        let mut sink = Scratch::new(t * nh);
        let mut k = Scratch::new(t * kw);
        let mut v = Scratch::new(t * kw);
        let dynamic = matches!(a.k_conv, KvConv::Dynamic { .. });
        let mut kd = vec![0f32; if dynamic { t * nkv * kn } else { 0 }];
        let mut vd = vec![0f32; if dynamic { t * nkv * kn } else { 0 }];
        {
            let mut jq: Vec<(&Lin, usize, &mut [f32])> = vec![
                (&a.q, w, &mut q[..]),
                (&a.r, nh * dr, &mut r[..]),
                (&a.gate, w, &mut gate[..]),
                (&a.sink, nh, &mut sink[..]),
            ];
            let mut jk: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.k, kw, &mut k[..])];
            let mut jv: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.v, kw, &mut v[..])];
            if let KvConv::Dynamic { proj, .. } = &a.k_conv {
                jk.push((proj, nkv * kn, &mut kd[..]));
            }
            if let KvConv::Dynamic { proj, .. } = &a.v_conv {
                jv.push((proj, nkv * kn, &mut vd[..]));
            }
            if std::ptr::eq(xq, xk) && std::ptr::eq(xq, xv) {
                // One input: all the projections in one parallel region.
                jq.append(&mut jk);
                jq.append(&mut jv);
                linear_multi(&mut jq, h, xq, t);
            } else {
                linear_multi(&mut jq, h, xq, t);
                linear_multi(&mut jk, h, xk, t);
                linear_multi(&mut jv, h, xv, t);
            }
        }
        let _timer = prof::start(Cat::Attention);
        for (conv, x, dy, hist) in [(&a.k_conv, &mut k, &kd, 0), (&a.v_conv, &mut v, &vd, 1)] {
            match conv {
                KvConv::Static(taps) => short_conv(taps, None, &mut st.hist[hist], x, kw, 1.0 / ((kn + 1) as f32).sqrt(), start),
                KvConv::Dynamic { taps, .. } => short_conv(
                    taps,
                    Some((dy, nkv)),
                    &mut st.hist[hist],
                    x,
                    kw,
                    1.0 / ((2 * kn + 1) as f32).sqrt(),
                    start,
                ),
            }
        }
        // The full layers scale queries and relative vectors up with the position.
        let tau = |ti: usize| -> f32 {
            if a.window.is_some() {
                return 1.0;
            }
            let n = ((start + ti + 1) as f64 / c.log_scaling_n_floor).max(1.0);
            (1.0 + c.log_scaling_alpha * n.ln()) as f32
        };
        let (qk_clip, rel_clip) = (c.qk_l2_clip_threshold as f32, c.rel_l2_clip_threshold as f32);
        for_chunks(&mut q, d, t > 1, |i, qh| {
            let hh = i % nh;
            for (x, &s) in qh.iter_mut().zip(&a.q_scale[hh * d..(hh + 1) * d]) {
                *x *= s;
            }
            l2_clip(qh, qk_clip);
            let f = tau(i / nh);
            if f != 1.0 {
                scale_inplace(qh, f);
            }
        });
        for_chunks(&mut r, dr, t > 1, |i, rh| {
            let hh = i % nh;
            for (x, &s) in rh.iter_mut().zip(&a.r_scale[hh * dr..(hh + 1) * dr]) {
                *x *= s;
            }
            l2_clip(rh, rel_clip);
            let f = tau(i / nh);
            if f != 1.0 {
                scale_inplace(rh, f);
            }
        });
        for_chunks(&mut k, d, t > 1, |_, kh| l2_clip(kh, qk_clip));
        let eps = c.rms_norm_eps as f32;
        for_chunks(&mut v, d, t > 1, |_, vh| rms_unit(vh, eps));
        st.k.push(&k);
        st.v.push(&v);

        let first = st.first;
        let (ck, cv) = (&st.k, &st.v);
        let mut att = Scratch::zeroed(t * w);
        // Prefill chunks: key blocks through the GEMM micro-kernel (`OXINFER_INKLING_ATTN=loop`
        // keeps the per-query loop below).
        #[cfg(has_simd)]
        let blocked = t >= ATT_QB && !loop_attention();
        #[cfg(not(has_simd))]
        let blocked = false;
        #[cfg(has_simd)]
        if blocked {
            attend_blocked(a, dr, &q, &r, &sink, ck, cv, first, start, t, &mut att);
        }
        if !blocked {
            attend_grouped(a, dr, &q, &r, &sink, ck, cv, first, start, t, &mut att);
        }
        // A sliding layer never reads further back than its window: what is older goes, a window's
        // worth at a time, keeping `ROLLBACK` positions more for a session that goes back.
        if let Some(win) = a.window {
            let keep = (start + t + 1).saturating_sub(win + ROLLBACK);
            if keep >= st.first + win {
                let cut = (keep - st.first) * kw;
                st.k.drop_front(cut);
                st.v.drop_front(cut);
                st.first = keep;
            }
        }
        for (x, &g) in att.iter_mut().zip(gate.iter()) {
            *x *= sigmoid(g);
        }
        drop(_timer);
        linear(&a.o, h, w, &att, t, out);
    }
}

/// The attention of a few queries (decoding, short passes): one task per (query, key/value
/// head, slice of the keys), the query heads of a key/value head together so that each cached
/// key and value is decoded once for all of them; the slices' partial softmaxes are merged with
/// the sink after. The same as one softmax over the keys and the sink up to rounding.
#[allow(clippy::too_many_arguments)]
fn attend_grouped(
    a: &Attn,
    dr: usize,
    q: &[f32],
    r: &[f32],
    sink: &[f32],
    ck: &KvRows,
    cv: &KvRows,
    first: usize,
    start: usize,
    t: usize,
    out: &mut [f32],
) {
    let Heads {
        heads: nh,
        kv_heads: nkv,
        head_dim: d,
    } = a.hd;
    let (kw, w, groups) = (nkv * d, nh * d, nh / nkv);
    let scaling = (d as f32).powf(-0.5);
    let extent = a.extent;
    let range = |i: usize| -> (usize, usize) {
        let p = start + i;
        let lo = match a.window {
            Some(win) => (p + 1).saturating_sub(win).max(first),
            None => first,
        };
        (lo, p + 1)
    };
    // Slices of at least 512 keys, as many as keep the cores busy.
    let longest = (0..t).map(|i| range(i).1 - range(i).0).max().unwrap_or(1);
    let want = crate::pool::threads().div_ceil(t * nkv).max(1);
    let slices = want.min(longest.div_ceil(512)).max(1);
    // Per (query, kv head, slice, head of the group): running max, sum, then `d` accumulators.
    let stride = d + 2;
    let mut part = vec![0f32; t * nkv * slices * groups * stride];
    let pp = SendMutPtr(part.as_mut_ptr());
    crate::pool::run(t * nkv * slices, &|task| {
        let (i, g, sl) = (task / (nkv * slices), task / slices % nkv, task % slices);
        let (lo, hi) = range(i);
        let n = hi - lo;
        let (j0, j1) = (lo + n * sl / slices, lo + n * (sl + 1) / slices);
        // SAFETY: each task writes its own slots of `part`.
        let res = unsafe { std::slice::from_raw_parts_mut(pp.get().add(task * groups * stride), groups * stride) };
        if j0 == j1 {
            for h in 0..groups {
                res[h * stride] = f32::NEG_INFINITY;
            }
            return;
        }
        let p = start + i;
        // The relative bias of the distances this slice covers, per head.
        let (dmin, dmax) = (p + 1 - j1, p - j0);
        let nb = (dmax + 1).min(extent).saturating_sub(dmin);
        let mut bias = vec![0f32; groups * nb];
        for h in 0..groups {
            let hh = g * groups + h;
            let rq = &r[(i * nh + hh) * dr..(i * nh + hh + 1) * dr];
            let bh = &mut bias[h * nb..(h + 1) * nb];
            for (&rv, prof) in rq.iter().zip(a.rel.chunks(extent)) {
                if nb > 0 {
                    axpy(bh, rv, &prof[dmin..dmin + nb]);
                }
            }
        }
        let len = j1 - j0;
        let mut s = vec![0f32; groups * len];
        let mut tmp = vec![0f32; d];
        let qs: Vec<&[f32]> = (0..groups)
            .map(|h| {
                let hh = g * groups + h;
                &q[i * w + hh * d..i * w + (hh + 1) * d]
            })
            .collect();
        for (jj, j) in (j0..j1).enumerate() {
            let kr = ck.get((j - first) * kw + g * d, &mut tmp);
            let dist = p - j;
            for (h, qh) in qs.iter().enumerate() {
                let b = if dist >= dmin && dist - dmin < nb {
                    bias[h * nb + dist - dmin]
                } else {
                    0.0
                };
                s[h * len + jj] = dot(qh, kr) * scaling + b;
            }
        }
        for h in 0..groups {
            let sh = &mut s[h * len..(h + 1) * len];
            let m = sh.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut l = 0f32;
            for x in sh.iter_mut() {
                *x = (*x - m).exp();
                l += *x;
            }
            res[h * stride] = m;
            res[h * stride + 1] = l;
        }
        for (jj, j) in (j0..j1).enumerate() {
            let vr = cv.get((j - first) * kw + g * d, &mut tmp);
            for h in 0..groups {
                let wj = s[h * len + jj];
                axpy(&mut res[h * stride + 2..(h + 1) * stride], wj, vr);
            }
        }
    });
    // Merge the slices with the sink (a logit with a zero value).
    for_chunks(out, d, t * nh > 1, |task, o| {
        let (i, hh) = (task / nh, task % nh);
        let (g, h) = (hh / groups, hh % groups);
        let at = |sl: usize| ((i * nkv + g) * slices + sl) * groups * stride + h * stride;
        let mut m = sink[i * nh + hh];
        for sl in 0..slices {
            m = m.max(part[at(sl)]);
        }
        let mut l = (sink[i * nh + hh] - m).exp();
        o.fill(0.0);
        for sl in 0..slices {
            let base = at(sl);
            if part[base] == f32::NEG_INFINITY {
                continue;
            }
            let c = (part[base] - m).exp();
            l += c * part[base + 1];
            axpy(o, c, &part[base + 2..base + stride]);
        }
        scale_inplace(o, 1.0 / l);
    });
}

/// `OXINFER_INKLING_ATTN=loop`: prefill attention one query at a time (as decoding).
#[cfg(has_simd)]
fn loop_attention() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OXINFER_INKLING_ATTN").is_ok_and(|v| v == "loop"))
}

/// Query rows per task of [`attend_blocked`] (a multiple of the GEMM tile's 8).
#[cfg(has_simd)]
const ATT_QB: usize = 32;
/// Keys per block of [`attend_blocked`] (a multiple of 8).
#[cfg(has_simd)]
const ATT_KB: usize = 64;

/// The attention of a prefill chunk (`t` queries at positions `start..start + t`, already in the
/// cache) as one task per (query head, block of 32 queries), each streaming blocks of 64 keys
/// through the f32 GEMM micro-kernel with an online softmax (as `sd::ops::attention_blocked`),
/// plus what Inkling-2's attention adds: the relative bias by backward distance, the sliding
/// window, and the sink (the running maximum starts at the sink's logit with a weight of 1).
/// The keys and values a chunk reads are copied out of the cache once per key/value head.
/// The same as the per-query loop of [`Inkling2Model::attention`] up to rounding.
#[cfg(has_simd)]
#[allow(clippy::too_many_arguments)]
fn attend_blocked(
    a: &Attn,
    dr: usize,
    q: &[f32],
    r: &[f32],
    sink: &[f32],
    ck: &KvRows,
    cv: &KvRows,
    first: usize,
    start: usize,
    t: usize,
    out: &mut [f32],
) {
    use crate::kernels::simd::gemm_8x8;
    let Heads {
        heads: nh,
        kv_heads: nkv,
        head_dim: d,
    } = a.hd;
    let (kw, w, groups) = (nkv * d, nh * d, nh / nkv);
    let scaling = (d as f32).powf(-0.5);
    let extent = a.extent;
    // The keys any query of the chunk may read: `kmin..start + t`.
    let kmin = match a.window {
        Some(win) => (start + 1).saturating_sub(win).max(first),
        None => first,
    };
    let nk = start + t - kmin;
    let nkb = nk.div_ceil(ATT_KB);
    let dhp = d.next_multiple_of(8);
    // One key/value head at a time (copies of its keys and values in the chunk's range): keys
    // per block, row-major `[KB, d]` (zero rows past the end); values transposed per block,
    // `[dhp][KB]`.
    let nqb = t.div_ceil(ATT_QB);
    let op = SendMutPtr(out.as_mut_ptr());
    let mut kbuf = vec![0f32; nkb * ATT_KB * d];
    let mut vbuf = vec![0f32; nkb * dhp * ATT_KB];
    for g in 0..nkv {
        for_chunks(&mut kbuf, ATT_KB * d, nkb > 1, |b, blk| {
            let mut tmp = vec![0f32; d];
            for jj in 0..ATT_KB.min(nk - b * ATT_KB) {
                let pos = kmin + b * ATT_KB + jj;
                blk[jj * d..(jj + 1) * d].copy_from_slice(ck.get((pos - first) * kw + g * d, &mut tmp));
            }
        });
        for_chunks(&mut vbuf, dhp * ATT_KB, nkb > 1, |b, blk| {
            let mut tmp = vec![0f32; d];
            for jj in 0..ATT_KB.min(nk - b * ATT_KB) {
                let pos = kmin + b * ATT_KB + jj;
                for (c, &x) in cv.get((pos - first) * kw + g * d, &mut tmp).iter().enumerate() {
                    blk[c * ATT_KB + jj] = x;
                }
            }
        });
        let (kh, vh) = (&kbuf[..], &vbuf[..]);
        crate::pool::run(groups * nqb, &|task| {
            let (hh, qb) = (g * groups + task / nqb, task % nqb);
            let q0 = qb * ATT_QB;
            let nq = (t - q0).min(ATT_QB);
            let tiles = nq.div_ceil(8);
            // The queries as GEMM panels `[tile][d][8]`, scaled.
            let mut qt = vec![0f32; tiles * d * 8];
            for i in 0..nq {
                let src = &q[(q0 + i) * w + hh * d..(q0 + i) * w + (hh + 1) * d];
                let (tl, ln) = (i / 8, i % 8);
                for (c, &x) in src.iter().enumerate() {
                    qt[(tl * d + c) * 8 + ln] = x * scaling;
                }
            }
            // Each query's bias by backward distance: its mix of the profiles.
            let mut bias = vec![0f32; nq * extent];
            for i in 0..nq {
                let rq = &r[((q0 + i) * nh + hh) * dr..((q0 + i) * nh + hh + 1) * dr];
                let bi = &mut bias[i * extent..(i + 1) * extent];
                for (&rv, prof) in rq.iter().zip(a.rel.chunks(extent)) {
                    axpy(bi, rv, prof);
                }
            }
            let mut s = vec![0f32; tiles * 8 * ATT_KB];
            let mut pt = vec![0f32; tiles * ATT_KB * 8];
            let mut tmp = vec![0f32; tiles * 8 * dhp];
            let mut acc = vec![0f32; nq * dhp];
            // The sink: a logit with a zero value.
            let mut m: Vec<f32> = (0..nq).map(|i| sink[(q0 + i) * nh + hh]).collect();
            let mut l = vec![1f32; nq];
            let (p_first, p_last) = (start + q0, start + q0 + nq - 1);
            let lo_first = match a.window {
                Some(win) => (p_first + 1).saturating_sub(win).max(kmin),
                None => kmin,
            };
            for b in (lo_first - kmin) / ATT_KB..(p_last - kmin) / ATT_KB + 1 {
                let j0 = kmin + b * ATT_KB;
                for gg in 0..ATT_KB / 8 {
                    for tl in 0..tiles {
                        // SAFETY: the panels and the score tile are sized for these offsets.
                        unsafe {
                            gemm_8x8::<f32>(
                                kh.as_ptr().add((b * ATT_KB + gg * 8) * d),
                                d,
                                qt.as_ptr().add(tl * d * 8),
                                s.as_mut_ptr(),
                                ATT_KB,
                                gg * 8,
                                tl * 8,
                                8,
                            );
                        }
                    }
                }
                for i in 0..nq {
                    let p = start + q0 + i;
                    let lo = match a.window {
                        Some(win) => (p + 1).saturating_sub(win).max(first),
                        None => first,
                    };
                    let row = &mut s[i * ATT_KB..(i + 1) * ATT_KB];
                    let (tl, ln) = (i / 8, i % 8);
                    let mut mx = m[i];
                    for (jj, x) in row.iter_mut().enumerate() {
                        let pos = j0 + jj;
                        if pos < lo || pos > p {
                            *x = f32::NEG_INFINITY;
                            continue;
                        }
                        let dist = p - pos;
                        if dist < extent {
                            *x += bias[i * extent + dist];
                        }
                        mx = mx.max(*x);
                    }
                    let corr = (m[i] - mx).exp();
                    let mut sum = 0f32;
                    for (jj, &x) in row.iter().enumerate() {
                        let e = if x == f32::NEG_INFINITY { 0.0 } else { (x - mx).exp() };
                        pt[(tl * ATT_KB + jj) * 8 + ln] = e;
                        sum += e;
                    }
                    l[i] = l[i] * corr + sum;
                    m[i] = mx;
                    if corr != 1.0 {
                        acc[i * dhp..(i + 1) * dhp].iter_mut().for_each(|v| *v *= corr);
                    }
                }
                for i in nq..tiles * 8 {
                    let (tl, ln) = (i / 8, i % 8);
                    for jj in 0..ATT_KB {
                        pt[(tl * ATT_KB + jj) * 8 + ln] = 0.0;
                    }
                }
                let vb = &vh[b * dhp * ATT_KB..(b + 1) * dhp * ATT_KB];
                for gg in 0..dhp / 8 {
                    for tl in 0..tiles {
                        // SAFETY: as above.
                        unsafe {
                            gemm_8x8::<f32>(
                                vb.as_ptr().add(gg * 8 * ATT_KB),
                                ATT_KB,
                                pt.as_ptr().add(tl * ATT_KB * 8),
                                tmp.as_mut_ptr(),
                                dhp,
                                gg * 8,
                                tl * 8,
                                8,
                            );
                        }
                    }
                }
                for (av, &x) in acc.iter_mut().zip(&tmp[..nq * dhp]) {
                    *av += x;
                }
            }
            for i in 0..nq {
                let inv = 1.0 / l[i];
                // SAFETY: each task writes its own (query, head) slices of `out`.
                let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((q0 + i) * w + hh * d), d) };
                for (o, &av) in o.iter_mut().zip(&acc[i * dhp..i * dhp + d]) {
                    *o = av * inv;
                }
            }
        });
    }
}

#[derive(Clone, Copy)]
struct SendMutPtr(*mut f32);
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}
impl SendMutPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

impl crate::generate::TextLm for Inkling2Model<'_> {
    type Sess = Inkling2Session;
    fn new_session(&self) -> Inkling2Session {
        Inkling2Model::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut Inkling2Session) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &Inkling2Session) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut Inkling2Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, &[], &mut Vec::new())
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        Inkling2Model::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut Inkling2Session, tokens: &[u32]) -> Vec<f32> {
        Inkling2Model::forward(self, sess, tokens)
    }
    fn decode_greedy(
        &self,
        sess: &mut Inkling2Session,
        first: u32,
        max: usize,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        Inkling2Model::decode_greedy(self, sess, first, max, on)
    }
    fn decode_sampled(
        &self,
        sess: &mut Inkling2Session,
        first: u32,
        max: usize,
        temperature: f32,
        draw: &mut dyn FnMut() -> f32,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        Inkling2Model::decode_sampled(self, sess, first, max, temperature, draw, on)
    }
}

/// Runtime interventions (`crate::intervene`): the stream after the embedding, after each
/// layer's attention and after each layer (inside a block: the sum of the layers' outputs so
/// far), and as neurons the input of `down_proj` of the routed experts (`Target::Expert`, before
/// the routing weight).
impl crate::intervene::Host for Inkling2Model<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        match target {
            Target::Expert(e) if e < self.layers.get(layer)?.experts.n_experts() => Some(self.cfg.moe_intermediate_size),
            _ => None,
        }
    }
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32> {
        let Target::Expert(e) = target else {
            unreachable!("Inkling-2 has routed experts only");
        };
        let (h, mi) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        let mut one = vec![0f32; mi];
        one[n] = 1.0;
        let mut y = vec![0f32; h];
        // A streamed checkpoint reads the layer (and the routed expert) as a pass would.
        let _unit = crate::stream::enter(&self.stream, layer);
        if let Some(s) = &self.stream {
            s.need_parts(layer, &[e]);
        }
        linear(&self.layers[layer].experts.down[e], h, mi, &one, 1, &mut y);
        y
    }
}

/// A speculative decoder's checkpoints ([`crate::dspark::DsparkDecoder::checkpoint`]): the
/// session's own.
impl crate::dspark::CheckpointTarget for Inkling2Model<'_> {
    fn checkpoint(&self, sess: &mut Inkling2Session, max: usize) {
        Inkling2Model::checkpoint(self, sess, max)
    }
    fn has_checkpoint(&self, sess: &Inkling2Session, pos: usize) -> bool {
        Inkling2Model::has_checkpoint(self, sess, pos)
    }
    fn reachable(&self, sess: &Inkling2Session, len: usize) -> usize {
        Inkling2Model::reachable(self, sess, len)
    }
    fn cut_to(&self, sess: &mut Inkling2Session, len: usize) -> Result<()> {
        Inkling2Model::cut_to(self, sess, len)
    }
}

/// What a speculative decoder needs of its target (a session goes back up to [`ROLLBACK`]
/// positions).
impl crate::dspark::Target for Inkling2Model<'_> {
    fn forward_feats(
        &self,
        sess: &mut Inkling2Session,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, layers, feats)
    }
    fn decode_greedy_feats(
        &self,
        sess: &mut Inkling2Session,
        first: u32,
        max: usize,
        layers: &[usize],
        feats: &mut Vec<f32>,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        Inkling2Model::decode_greedy_feats(self, sess, first, max, layers, feats, on)
    }
    fn decode_sampled_feats(
        &self,
        sess: &mut Inkling2Session,
        first: u32,
        max: usize,
        layers: &[usize],
        feats: &mut Vec<f32>,
        temperature: f32,
        draw: &mut dyn FnMut() -> f32,
        on: &mut dyn FnMut(u32) -> crate::generate::Chain,
    ) -> Option<usize> {
        Inkling2Model::decode_sampled_feats(self, sess, first, max, layers, feats, temperature, draw, on)
    }
    fn forward_verify(&self, sess: &mut Inkling2Session, tokens: &[u32], layers: &[usize], feats: &mut Vec<f32>, _fast: bool) -> Vec<f32> {
        let t = tokens.len();
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        let final_out = layers.contains(&(self.cfg.steps().len() + 1));
        let want = DevWant {
            hidden: final_out.then_some(&mut hidden),
            logits: Some((&mut logits, t)),
            capture: (!layers.is_empty()).then_some((layers, &mut *feats)),
        };
        if self.run_device(sess, tokens, &[], want) {
            self.capture_final(layers, feats, &hidden, t);
            return logits;
        }
        let hs = self.forward_capture(sess, tokens, &[], layers, feats);
        self.logits(&hs, t)
    }
    // The model's input: the embedding after its RMS norm (what a draft's anchor row takes).
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        Inkling2Model::embed_row(self, token, out);
        rmsnorm_f32w_inplace(out, &self.embed_norm, self.cfg.rms_norm_eps as f32);
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.head(hidden, t, false)
    }
    // A draft's few rows on the accelerator's matrix-vector kernels (4 rows per submission).
    fn lm_head(&self) -> Option<&Lin<'_>> {
        self.stream.is_none().then_some(&self.lm_head)
    }
    fn head_raw_device(&self, hidden: &[f32], t: usize) -> Option<Vec<f32>> {
        let c = &self.cfg;
        let (h, v, u) = (c.hidden_size, c.vocab_size, c.unpadded_vocab_size);
        if self.stream.is_some() {
            return None;
        }
        let mut out = Vec::with_capacity(t * u);
        let mut y = vec![0f32; 4 * v];
        for r0 in (0..t).step_by(4) {
            let n = (t - r0).min(4);
            let jobs = &mut [(&self.lm_head, None, v, &mut y[..n * v])];
            if !crate::model::linear_rows_device(jobs, h, &hidden[r0 * h..(r0 + n) * h], n) {
                return None;
            }
            for r in 0..n {
                out.extend_from_slice(&y[r * v..r * v + u]);
            }
        }
        Some(out)
    }
    // The logits cover the tokens only (the unembedding is padded).
    #[allow(clippy::misnamed_getters)]
    fn vocab_size(&self) -> usize {
        self.cfg.unpadded_vocab_size
    }
    fn truncate(&self, sess: &mut Inkling2Session, len: usize) {
        if let Err(e) = sess.truncate(len) {
            panic!("{e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The release's config spelling and transformers' give the same settings.
    #[test]
    fn config_spellings() {
        let hub = json!({"model_type": "inkling2", "text_config": {
            "hidden_size": 4608, "num_hidden_layers": 12, "vocab_size": 201024, "unpadded_vocab_size": 200058,
            "num_attention_heads": 48, "num_key_value_heads": 8, "head_dim": 128,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], "sconv_kernel_size": 4,
            "n_routed_experts": 384, "num_experts_per_tok": 8, "n_shared_experts": 0, "intermediate_size": 2304,
            "route_scale": 8.0, "use_gate_bias": true, "use_value_norm_no_gamma": true}});
        let c = Inkling2Config::from_json(&hub).unwrap();
        assert_eq!(
            (c.moe_intermediate_size, c.conv_kernel_size, c.unpadded_vocab_size),
            (2304, 4, 200058)
        );
        assert_eq!(c.swa, c.full);
        assert_eq!(c.sliding.iter().filter(|s| !**s).count(), 2);
        assert!(!c.sliding[5] && !c.sliding[11]);
        let tf = json!({"model_type": "inkling2", "text_config": {
            "hidden_size": 64, "num_hidden_layers": 6, "conv_kernel_size": 4, "moe_intermediate_size": 32,
            "n_routed_experts": 16, "num_experts_per_tok": 4, "swa_num_attention_heads": 4, "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "layer_types": ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid"]}});
        let c = Inkling2Config::from_json(&tf).unwrap();
        assert_eq!(c.sliding, [true, false, true, true, true, false]);
        assert_eq!((c.heads(0).heads, c.heads(1).heads), (4, 8));
        // A variant of the architecture that is not implemented is refused.
        let bad = json!({"model_type": "inkling2", "text_config": {"hidden_size": 64, "q_bias": true}});
        assert!(Inkling2Config::from_json(&bad).is_err());
    }

    /// The convolution against its definition, fed whole or token by token.
    #[test]
    fn short_conv_steps() {
        let (ch, k, t) = (4usize, 3usize, 5usize);
        let taps: Vec<f32> = (0..k * ch).map(|i| 0.1 * i as f32 - 0.3).collect();
        let x: Vec<f32> = (0..t * ch).map(|i| ((i * 7 % 11) as f32) * 0.25 - 1.0).collect();
        let dy: Vec<f32> = (0..t * 2 * k).map(|i| ((i * 5 % 7) as f32) * 0.1).collect();
        for dynamic in [false, true] {
            let want: Vec<f32> = (0..t * ch)
                .map(|i| {
                    let (ti, c) = (i / ch, i % ch);
                    let mut s = x[i];
                    for lag in 0..=ti.min(k - 1) {
                        let d = if dynamic { dy[(ti * 2 + c / 2) * k + lag] } else { 0.0 };
                        s += (taps[lag * ch + c] + d) * x[(ti - lag) * ch + c];
                    }
                    s * 0.5
                })
                .collect();
            let d = |a: usize, b: usize| dynamic.then(|| (&dy[a * 2 * k..b * 2 * k], 2));
            let mut whole = x.clone();
            short_conv(&taps, d(0, t), &mut ConvRing::new(k, ch), &mut whole, ch, 0.5, 0);
            let mut hist = ConvRing::new(k, ch);
            let mut steps = x.clone();
            for ti in 0..t {
                short_conv(&taps, d(ti, ti + 1), &mut hist, &mut steps[ti * ch..(ti + 1) * ch], ch, 0.5, ti);
            }
            // Going back: tokens 2.. again after junk at positions 2..4.
            let mut back = ConvRing::new(k, ch);
            let mut again = x.clone();
            short_conv(&taps, d(0, 2), &mut back, &mut again[..2 * ch], ch, 0.5, 0);
            let mut junk = vec![7.0f32; 3 * ch];
            short_conv(&taps, d(2, 5), &mut back, &mut junk, ch, 0.5, 2);
            short_conv(&taps, d(2, t), &mut back, &mut again[2 * ch..], ch, 0.5, 2);
            for (((a, b), c), w) in whole.iter().zip(&steps).zip(&again).zip(&want) {
                assert!(
                    (a - w).abs() < 1e-5 && (b - w).abs() < 1e-5 && (c - w).abs() < 1e-5,
                    "{a} {b} {c} {w}"
                );
            }
        }
    }

    /// mlx-lm's names go back to the release's (`sanitize` of mlx-lm's `inkling2`, undone).
    #[test]
    fn mlx_names() {
        let m = |n: &str| mlx_name(n).unwrap();
        for (mlx, release) in [
            ("lm_head.scales", "model.llm.unembed.scales"),
            ("model.language_model.embed_tokens.biases", "model.llm.embed.biases"),
            ("model.language_model.final_input.query", "model.llm.attn_res.output_query"),
            (
                "model.language_model.layers.7.attention_input.query",
                "model.llm.attn_res.attn_queries.6",
            ),
            ("model.language_model.layers.7.mlp_input.query", "model.llm.attn_res.mlp_queries.7"),
            (
                "model.language_model.layers.3.input_layernorm.weight",
                "model.llm.layers.3.attn_norm.weight",
            ),
            (
                "model.language_model.layers.5.k_input_layernorm.weight",
                "model.llm.layers.5.k_input_norm.weight",
            ),
            (
                "model.language_model.layers.9.q_message_gate.message_gate_proj.weight",
                "model.llm.layers.9.controllers.attn.gain_weight",
            ),
            (
                "model.language_model.layers.9.mlp_message_gate.down_proj.weight",
                "model.llm.layers.9.controllers.mlp.down_weight",
            ),
            (
                "model.language_model.layers.2.post_mlp_convolution.conv1d.weight",
                "model.llm.layers.2.mlp_sconv.weight",
            ),
            (
                "model.language_model.layers.2.self_attn.o_proj.scales",
                "model.llm.layers.2.attn.wo_ud.scales",
            ),
            (
                "model.language_model.layers.2.self_attn.sink_proj.weight",
                "model.llm.layers.2.attn.attn_sink_proj.weight",
            ),
            (
                "model.language_model.layers.2.self_attn.k_sconv.conv1d.weight",
                "model.llm.layers.2.attn.k_sconv.weight",
            ),
            (
                "model.language_model.layers.5.self_attn.v_sconv.static_weight",
                "model.llm.layers.5.attn.v_sconv.static_w",
            ),
            (
                "model.language_model.layers.5.self_attn.v_sconv.weight_proj.weight",
                "model.llm.layers.5.attn.v_sconv.weight_proj",
            ),
            (
                "model.language_model.layers.5.self_attn.v_sconv.weight_proj.biases",
                "model.llm.layers.5.attn.v_sconv.weight_proj.biases",
            ),
            (
                "model.language_model.layers.5.self_attn.q_l2_clip.scale",
                "model.llm.layers.5.attn.q_l2_clip.scale",
            ),
            (
                "model.language_model.layers.0.mlp.gate.e_score_correction_bias",
                "model.llm.layers.0.mlp.gate.bias",
            ),
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj.global_scale",
                "model.llm.layers.0.mlp.experts.w13_weight_global_scale",
            ),
            (
                "model.language_model.layers.21.mlp.experts.down_proj.scales",
                "model.llm.layers.21.mlp.experts.w2_weight_scale",
            ),
            ("model.vision_tower.layers.4.lambda_2", "model.visual.backbone.layers.4.ls_2.gamma"),
            (
                "model.vision_tower.layers.4.mlp.fc1.bias",
                "model.visual.backbone.layers.4.mlp.c_fc.bias",
            ),
            ("model.vision_tower.merger.linear_1.weight", "model.visual.proj_in.weight"),
            (
                "model.vision_tower.embeddings.position_embedding.weight",
                "model.visual.backbone.embeddings.position_embedding",
            ),
        ] {
            assert_eq!(m(mlx), release, "{mlx}");
        }
    }

    /// Without block states the mix is the RMS-normed input; a zero query averages.
    /// The short passes' message gate against its definition in f64, with f32 and bf16 rows,
    /// one token and three.
    #[test]
    fn gate_rows_match_definition() {
        crate::pool::init(3);
        let (h, r) = (96usize, 32usize);
        let mut seed = 9u64;
        let mut rnd = |n: usize, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * amp
                })
                .collect()
        };
        let (wd, wf, wg) = (rnd(r * 2 * h, 0.1), rnd(h * r, 0.3), rnd(h * r, 0.3));
        let bf = |w: &[f32]| -> Vec<u16> { w.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect() };
        let back = |w: &[u16]| -> Vec<f32> { w.iter().map(|&v| half::bf16::from_bits(v).to_f32()).collect() };
        let (bd, bfo, bg) = (bf(&wd), bf(&wf), bf(&wg));
        for t in [1usize, 3] {
            let (cat, x, m) = (rnd(t * 2 * h, 1.0), rnd(t * h, 2.0), rnd(t * h, 2.0));
            for bf16 in [false, true] {
                let (rows, vals) = if bf16 {
                    (
                        [FloatRows::Bf16(&bd), FloatRows::Bf16(&bfo), FloatRows::Bf16(&bg)],
                        [back(&bd), back(&bfo), back(&bg)],
                    )
                } else {
                    (
                        [FloatRows::F32(&wd), FloatRows::F32(&wf), FloatRows::F32(&wg)],
                        [wd.clone(), wf.clone(), wg.clone()],
                    )
                };
                let (mut d, mut out) = (vec![0f32; t * r], vec![0f32; t * h]);
                gate_rows(&rows[0], &rows[1], &rows[2], &cat, &x, &m, h, r, &mut d, &mut out);
                for ti in 0..t {
                    let dd: Vec<f64> = (0..r)
                        .map(|i| (0..2 * h).map(|c| vals[0][i * 2 * h + c] as f64 * cat[ti * 2 * h + c] as f64).sum())
                        .collect();
                    for j in 0..h {
                        let f: f64 = (0..r).map(|i| vals[1][j * r + i] as f64 * dd[i]).sum();
                        let g: f64 = (0..r).map(|i| vals[2][j * r + i] as f64 * dd[i]).sum();
                        let want = (1.0 - f.tanh()) * x[ti * h + j] as f64 + g.tanh() * m[ti * h + j] as f64;
                        let have = out[ti * h + j] as f64;
                        assert!(
                            (have - want).abs() < 2e-5,
                            "bf16 {bf16}, {t} tokens, token {ti} output {j}: {have} against {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mix_is_a_softmax_over_normed_states() {
        let h = 4;
        let x = [3.0f32, 0.0, 0.0, 0.0];
        let alone = mix(&[], &x, &[1.0, 2.0, 3.0, 4.0], h);
        assert!((alone[0][0] - 2.0).abs() < 1e-4 && alone[0][1] == 0.0);
        let state = vec![0.0f32, 2.0, 0.0, 0.0];
        let both = mix(&[state], &x, &[0.0; 8], h);
        assert_eq!(both.len(), 2);
        assert!((both[1][0] - 1.0).abs() < 1e-4 && (both[1][1] - 1.0).abs() < 1e-4);
    }

    /// The tiny random Inkling-2-Mini of [`tiny`], loaded with `kv` caches; on the CPU only
    /// unless `device` (then on the accelerator when there is one).
    fn tiny_model(tag: &str, kv: KvCache, device: bool) -> &'static Inkling2Model<'static> {
        let dir = std::env::temp_dir().join(format!("oxinfer-ink-tiny-{tag}-{}", std::process::id()));
        tiny::write_tiny(&dir, 7).unwrap();
        let cfg = Inkling2Config::load(&dir).unwrap();
        let files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(&dir, true).unwrap()));
        let mut m = Inkling2Model::load(cfg, files, Quant::Bf16).unwrap();
        m.kv = kv;
        // (The device engine on a backend of its own: the process's may be the CPU.)
        #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
        let dev = if device { crate::metal::inkling::build_own(&m) } else { None };
        #[cfg(not(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64")))]
        let dev: Option<Box<dyn DevEngine>> = {
            let _ = device;
            None
        };
        let _ = m.device.set(dev);
        Box::leak(Box::new(m))
    }

    /// Every position's logits of `toks` run from the session's position in passes of `pass`.
    fn run(m: &Inkling2Model, sess: &mut Inkling2Session, toks: &[u32], pass: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for c in toks.chunks(pass) {
            let hs = m.forward_hidden(sess, c);
            out.extend(m.logits(&hs, c.len()));
        }
        out
    }

    fn tokens(n: usize, salt: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 37 + salt * 11) % 250) as u32).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// With the drafter (the tiny one of [`tiny::write_tiny_mtp`], one layer attending to every
    /// position as the release's, or all three to a window): a decoder that went back to a
    /// checkpoint further than its draft's head rows (and windowed cache) reach drafts and
    /// decodes as a fresh one, and its snapshot continues as it does
    /// ([`crate::dspark::checks::checkpoint_matches_fresh`]).
    #[test]
    fn draft_checkpoint_cut_back_matches_fresh() {
        for local in [&[0usize, 2][..], &[0, 1, 2]] {
            let tag = format!("dcp{}", local.len());
            let m = tiny_model(&tag, KvCache::F32, false);
            let dir = std::env::temp_dir().join(format!("oxinfer-ink-tiny-{tag}-{}", std::process::id()));
            tiny::write_tiny_mtp(&dir, 9, local).unwrap();
            let path = dir.join("mtp.safetensors");
            let df: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(&path, true).unwrap()));
            let d = crate::dspark::Dspark::load(&path, df, m.cfg.hidden_size, m.cfg.vocab_size, Quant::Bf16).unwrap();
            check_draft(m, &d).unwrap();
            let first = tokens(100 + 560, 0);
            let next = tokens(30, 3);
            crate::dspark::checks::checkpoint_matches_fresh(m, &d, &first, 100, &next, 32, &format!("Inkling-2 MTP {local:?}"));
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// A session that ran past a checkpoint (further than it can go back otherwise), cut back to
    /// it and fed other tokens gives the logits of a fresh session that ran the same passes up
    /// to it, bit for bit; going back a few positions from there works as before.
    fn checkpoint_matches_fresh(m: &Inkling2Model, pass: usize) {
        let (c, past) = (100, 90);
        let first = tokens(c + past, 0);
        let next = tokens(40, 3);
        let mut a = m.new_session();
        run(m, &mut a, &first[..c], pass);
        m.checkpoint(&mut a, 2);
        run(m, &mut a, &first[c..], pass);
        assert!(a.pos - c > m.rollback_limit());
        assert_eq!(m.reachable(&a, c + 5), c);
        assert_eq!(m.reachable(&a, a.pos - 10), a.pos - 10);
        assert_eq!(m.reachable(&a, c - 1), 0);
        assert!(m.cut_to(&mut a, c + 5).is_err());
        m.cut_to(&mut a, c).unwrap();
        let got = run(m, &mut a, &next[..30], pass);
        let mut b = m.new_session();
        run(m, &mut b, &first[..c], pass);
        let want = run(m, &mut b, &next[..30], pass);
        assert_eq!(bits(&got), bits(&want), "after the checkpoint (passes of {pass})");
        // Not before the restored position; a few back from the end, as before.
        assert!(a.truncate(c - 1).is_err());
        a.truncate(c + 20).unwrap();
        b.truncate(c + 20).unwrap();
        let got = run(m, &mut a, &next[30..], pass);
        let want = run(m, &mut b, &next[30..], pass);
        assert_eq!(bits(&got), bits(&want), "going back after a restore (passes of {pass})");
    }

    #[test]
    fn checkpoint_cut_back_matches_fresh_cpu() {
        for kv in [KvCache::F32, KvCache::F16] {
            let m = tiny_model(&format!("cp{kv:?}"), kv, false);
            for pass in [1, 7, 64] {
                checkpoint_matches_fresh(m, pass);
            }
        }
    }

    /// The same on the accelerator, when there is one (Metal; skipped otherwise).
    #[test]
    fn checkpoint_cut_back_matches_fresh_device() {
        let m = tiny_model("cpdev", KvCache::F32, true);
        let mut s = m.new_session();
        run(m, &mut s, &tokens(8, 0), 8);
        if s.dev.is_none() || m.device().is_none_or(|d| !d.snapshots()) {
            eprintln!("[no Inkling-2 device engine here: skipped]");
            return;
        }
        eprintln!("[Inkling-2 checkpoints on the device]");
        for pass in [1, 16, 64] {
            checkpoint_matches_fresh(m, pass);
        }
    }

    /// Checkpoints past the cap: the lowest and the newest stay.
    #[test]
    fn checkpoint_cap_keeps_lowest_and_newest() {
        let m = tiny_model("cap", KvCache::F32, false);
        let mut s = m.new_session();
        let t = tokens(80, 1);
        for end in [10, 30, 50, 80] {
            let from = s.pos;
            run(m, &mut s, &t[from..end], 64);
            m.checkpoint(&mut s, 2);
        }
        let at: Vec<usize> = s.checkpoints.iter().map(|c| c.pos()).collect();
        assert_eq!(at, vec![10, 80]);
        assert!(m.has_checkpoint(&s, 10) && !m.has_checkpoint(&s, 30));
        m.cut_to(&mut s, 10).unwrap();
        assert_eq!(s.checkpoints.len(), 1);
    }

    /// A session written to a snapshot and read into a fresh one continues as the original does,
    /// bit for bit (host caches in each format; the device's read where they are).
    fn snapshot_round_trip(m: &Inkling2Model, tag: &str) {
        use crate::snapshot::{SessionState, SnapReader, SnapWriter};
        let t = tokens(150, 2);
        let mut a = m.new_session();
        run(m, &mut a, &t[..120], 32);
        let path = std::env::temp_dir().join(format!("oxinfer-ink-snap-{tag}-{}.oxsnap", std::process::id()));
        let mut w = SnapWriter::new(std::fs::File::create(&path).unwrap()).unwrap();
        a.write_state(&mut w).unwrap();
        w.finish(json!({})).unwrap();
        let mut b = m.new_session();
        let mut r = SnapReader::open(&path).unwrap();
        b.read_state(&mut r).unwrap();
        r.finish().unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(b.pos, 120);
        let got = run(m, &mut b, &t[120..], 1);
        let want = run(m, &mut a, &t[120..], 1);
        assert_eq!(bits(&got), bits(&want), "{tag}");
    }

    #[test]
    fn snapshot_round_trip_cpu() {
        for kv in [KvCache::F32, KvCache::F16] {
            snapshot_round_trip(tiny_model(&format!("snap{kv:?}"), kv, false), &format!("{kv:?}"));
        }
    }

    #[test]
    fn snapshot_round_trip_device() {
        let m = tiny_model("snapdev", KvCache::F32, true);
        if !m.snapshots() || m.device().is_none() {
            eprintln!("[no Inkling-2 device engine here: skipped]");
            return;
        }
        snapshot_round_trip(m, "device");
    }
}
