//! Gemma 4 / Gemma 3 text-model configuration, parsed from HuggingFace `config.json`.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

fn d256() -> usize {
    256
}
fn d512() -> usize {
    512
}
fn d_eps() -> f64 {
    1e-6
}
fn d_maxpos() -> usize {
    131_072
}
fn d_vocab() -> usize {
    262_144
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "d256")]
    pub head_dim: usize,
    #[serde(default = "d512")]
    pub global_head_dim: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_vocab")]
    pub vocab_size_per_layer_input: usize,
    #[serde(default = "d256")]
    pub hidden_size_per_layer_input: usize,
    #[serde(default = "d512")]
    pub sliding_window: usize,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, RopeParams>>,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default = "d_maxpos")]
    pub max_position_embeddings: usize,
    /// Gemma 3: the attention scale is `query_pre_attn_scalar^-0.5` (Gemma 4 uses 1.0).
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f64>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_local_base_freq: Option<f64>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// `true` / `"all"` (encoders: EmbeddingGemma): every position attends both ways (within the
    /// window on sliding layers). `"vision"` (Gemma 4 12B / 31B / 26B-A4B): the soft tokens of one
    /// image attend to each other both ways on sliding layers; everything else is causal.
    #[serde(default)]
    pub use_bidirectional_attention: Option<serde_json::Value>,
    /// Gemma 3 (set from `model_type`): RMSNorm scales by `1 + weight`, V is not normed, every
    /// layer has the same head dim, and the layer pattern does not force a global last layer.
    #[serde(skip)]
    pub gemma3: bool,
}

#[derive(Debug, Clone)]
pub struct RopeSpec {
    pub theta: f64,
    /// Fraction of the head dim that is rotated (1.0 = full RoPE).
    pub proportion: f64,
    /// Linear position scaling: frequencies are divided by this.
    pub factor: f64,
}

/// Gemma 3 configs (notably multimodal ones) omit fields that equal the transformers defaults.
const GEMMA3_DEFAULTS: [(&str, f64); 13] = [
    ("vocab_size", 262_208.0),
    ("hidden_size", 2304.0),
    ("intermediate_size", 9216.0),
    ("num_hidden_layers", 26.0),
    ("num_attention_heads", 8.0),
    ("num_key_value_heads", 4.0),
    ("head_dim", 256.0),
    ("query_pre_attn_scalar", 256.0),
    ("sliding_window", 4096.0),
    ("rope_theta", 1_000_000.0),
    ("rope_local_base_freq", 10_000.0),
    ("rms_norm_eps", 1e-6),
    ("max_position_embeddings", 131_072.0),
];

/// transformers 5.20+ saves Gemma 4's global-layer attention shape as `per_layer_config`
/// (`{"5": {"head_dim": 512, "num_key_value_heads": 2}, ...}` for the full-attention layers)
/// instead of `global_head_dim` / `num_global_key_value_heads`: maps it back.
fn per_layer_overrides(o: &mut serde_json::Map<String, serde_json::Value>) -> Result<()> {
    let Some(plc) = o.get("per_layer_config").and_then(|v| v.as_object()).cloned() else {
        return Ok(());
    };
    let mut seen: Option<&serde_json::Value> = None;
    for v in plc.values() {
        if seen.is_some_and(|s| s != v) {
            bail!("per_layer_config with different overrides per layer is not supported");
        }
        seen = Some(v);
    }
    if let Some(v) = seen {
        if let Some(extra) = v
            .as_object()
            .and_then(|m| m.keys().find(|k| !matches!(k.as_str(), "head_dim" | "num_key_value_heads")))
        {
            bail!("per_layer_config override {extra:?} is not supported");
        }
        for (from, to) in [
            ("head_dim", "global_head_dim"),
            ("num_key_value_heads", "num_global_key_value_heads"),
        ] {
            if let Some(x) = v.get(from) {
                o.insert(to.into(), x.clone());
            }
        }
    }
    Ok(())
}

impl TextConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = if crate::gguf::is_gguf(dir) {
            crate::gguf::header(dir)?.config_json()?
        } else {
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?
        };
        let mt = raw.get("model_type").and_then(|v| v.as_str()).unwrap_or_default();
        let gemma3 = mt == "gemma3" || mt == "gemma3_text";
        let mut text = raw.get("text_config").cloned().unwrap_or(raw);
        if gemma3 && let Some(o) = text.as_object_mut() {
            for (k, v) in GEMMA3_DEFAULTS {
                if o.get(k).is_none_or(|x| x.is_null()) {
                    let n = if v.fract() == 0.0 && v > 1.0 {
                        serde_json::Value::from(v as u64)
                    } else {
                        serde_json::Value::from(v)
                    };
                    o.insert(k.to_string(), n);
                }
            }
            let hd = o["head_dim"].clone();
            o.insert("global_head_dim".into(), hd);
            o.insert("hidden_size_per_layer_input".into(), 0.into());
            if o.get("layer_types").is_none_or(|x| x.is_null()) {
                let k = o.get("sliding_window_pattern").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
                let n = o["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
                let lt: Vec<&str> = (0..n)
                    .map(|i| if (i + 1) % k == 0 { "full_attention" } else { "sliding_attention" })
                    .collect();
                o.insert("layer_types".into(), lt.into());
            }
        }
        if let Some(o) = text.as_object_mut() {
            per_layer_overrides(o)?;
        }
        let mut cfg: TextConfig = serde_json::from_value(text).context("parse text config")?;
        cfg.gemma3 = gemma3;
        if cfg.bidirectional() && (gemma3 || cfg.use_bidirectional_attention.as_ref().is_some_and(|v| v == "all")) {
            // As transformers' Gemma3TextConfig: the stored window spans both directions, so each
            // side sees `window / 2` (keys less than `window / 2 + 1` positions away).
            cfg.sliding_window = cfg.sliding_window / 2 + 1;
        }
        cfg.finalize()?;
        Ok(cfg)
    }

    pub fn bidirectional(&self) -> bool {
        match &self.use_bidirectional_attention {
            Some(serde_json::Value::Bool(b)) => *b,
            Some(serde_json::Value::String(s)) => s == "all",
            _ => false,
        }
    }

    /// Image soft tokens attend to the other soft tokens of their image both ways (sliding layers).
    pub fn vision_bidirectional(&self) -> bool {
        matches!(&self.use_bidirectional_attention, Some(serde_json::Value::String(s)) if s == "vision")
    }

    /// Gemma 4 MoE layers (26B-A4B): a router and routed experts beside the dense MLP.
    pub fn moe(&self) -> Option<(usize, usize, usize)> {
        if !self.enable_moe_block {
            return None;
        }
        Some((
            self.num_experts.unwrap_or(0),
            self.top_k_experts.unwrap_or(0),
            self.moe_intermediate_size.unwrap_or(0),
        ))
    }

    /// Attention scale applied to the queries.
    pub fn query_scale(&self) -> f32 {
        self.query_pre_attn_scalar.map_or(1.0, |s| s.powf(-0.5) as f32)
    }

    /// RMSNorm weight offset: Gemma 3 scales by `1 + weight`.
    pub fn norm_offset(&self) -> f32 {
        if self.gemma3 { 1.0 } else { 0.0 }
    }

    fn finalize(&mut self) -> Result<()> {
        if self.layer_types.is_none() {
            // Default 5:1 pattern.
            self.layer_types = Some(
                (0..self.num_hidden_layers)
                    .map(|i| if (i + 1) % 6 == 0 { "full_attention" } else { "sliding_attention" }.to_string())
                    .collect(),
            );
        }
        let lt = self.layer_types.as_mut().unwrap();
        if lt.len() != self.num_hidden_layers {
            bail!("layer_types has {} entries, expected {}", lt.len(), self.num_hidden_layers);
        }
        if !self.gemma3
            && let Some(last) = lt.last_mut()
        {
            *last = "full_attention".to_string();
        }
        if let Some((n, k, mi)) = self.moe()
            && (n == 0 || k == 0 || k > n || mi == 0)
        {
            bail!("MoE config: {n} experts, top {k}, expert width {mi}");
        }
        Ok(())
    }

    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types.as_ref().unwrap()[layer] == "sliding_attention"
    }

    pub fn layer_head_dim(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }

    pub fn layer_n_kv(&self, layer: usize) -> usize {
        if !self.is_sliding(layer)
            && self.attention_k_eq_v
            && let Some(n) = self.num_global_key_value_heads
        {
            return n;
        }
        self.num_key_value_heads
    }

    /// Global layers reuse K as V when `attention_k_eq_v` is set.
    pub fn layer_k_eq_v(&self, layer: usize) -> bool {
        !self.is_sliding(layer) && self.attention_k_eq_v
    }

    pub fn first_kv_shared_layer(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers
    }

    pub fn is_kv_shared(&self, layer: usize) -> bool {
        self.num_kv_shared_layers > 0 && layer >= self.first_kv_shared_layer()
    }

    /// Index of the layer whose KV cache this layer reads from.
    pub fn kv_source(&self, layer: usize) -> usize {
        if !self.is_kv_shared(layer) {
            return layer;
        }
        let first = self.first_kv_shared_layer();
        let sliding = self.is_sliding(layer);
        (0..first)
            .rev()
            .find(|&l| self.is_sliding(l) == sliding)
            .expect("no non-shared layer of matching type")
    }

    /// Layers whose K/V are the model's "shared KV states" (consumed by KV-sharing layers and
    /// by the drafter): the last non-shared layer of each type, as (sliding, global).
    pub fn shared_kv_layers(&self) -> (usize, usize) {
        let first = self.first_kv_shared_layer();
        let s = (0..first).rev().find(|&l| self.is_sliding(l)).expect("no sliding layer");
        let g = (0..first).rev().find(|&l| !self.is_sliding(l)).expect("no global layer");
        (s, g)
    }

    pub fn layer_intermediate(&self, layer: usize) -> usize {
        if self.use_double_wide_mlp && self.is_kv_shared(layer) {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }

    pub fn rope_spec(&self, sliding: bool) -> RopeSpec {
        if self.gemma3 {
            // Local layers: base 10k, unscaled; global layers: rope_theta with `rope_scaling`.
            let factor = match &self.rope_scaling {
                Some(r) if !sliding && r["rope_type"].as_str().unwrap_or("linear") == "linear" => r["factor"].as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            return RopeSpec {
                theta: if sliding {
                    self.rope_local_base_freq.unwrap_or(10_000.0)
                } else {
                    self.rope_theta.unwrap_or(1_000_000.0)
                },
                proportion: 1.0,
                factor,
            };
        }
        let key = if sliding { "sliding_attention" } else { "full_attention" };
        match self.rope_parameters.as_ref().and_then(|m| m.get(key)) {
            Some(p) => RopeSpec {
                theta: p.rope_theta,
                proportion: if p.rope_type.as_deref() == Some("proportional") {
                    p.partial_rotary_factor.unwrap_or(1.0)
                } else {
                    1.0
                },
                factor: 1.0,
            },
            None => {
                if sliding {
                    RopeSpec {
                        theta: 10_000.0,
                        proportion: 1.0,
                        factor: 1.0,
                    }
                } else {
                    RopeSpec {
                        theta: 1_000_000.0,
                        proportion: 0.25,
                        factor: 1.0,
                    }
                }
            }
        }
    }

    pub fn has_ple(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }
}

/// Stop tokens: `eos_token_id` from `generation_config.json`, else from `config.json` (root or
/// `text_config`), else Gemma's `<eos>` and `<turn|>`. Callers add the chat format's end-of-turn
/// token on top (`chat::with_turn_end`), since some checkpoints only list the base model's EOS.
pub fn load_eos_ids(dir: &Path) -> Vec<u32> {
    if crate::gguf::is_gguf(dir) {
        return crate::gguf::header(dir).map(|h| h.eos_ids()).unwrap_or_default();
    }
    fn ids(v: &serde_json::Value) -> Option<Vec<u32>> {
        match v {
            serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as u32]),
            serde_json::Value::Array(a) => Some(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()),
            _ => None,
        }
    }
    let read = |name: &str| {
        crate::fsx::read(&dir.join(name))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    };
    if let Some(found) = read("generation_config.json").and_then(|v| ids(&v["eos_token_id"])) {
        return found;
    }
    if let Some(v) = read("config.json")
        && let Some(found) = ids(&v["eos_token_id"]).or_else(|| ids(&v["text_config"]["eos_token_id"]))
    {
        return found;
    }
    vec![1, 106]
}

/// `model_type` from the root of config.json (e.g. "gemma4", "qwen3_5").
pub fn model_type(dir: &Path) -> Result<String> {
    if crate::gguf::is_gguf(dir) {
        return Ok(crate::gguf::header(dir)?.model_type()?.to_string());
    }
    let raw: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
    raw.get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            raw.get("text_config")
                .and_then(|t| t.get("model_type"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .context("config.json has no model_type")
}

fn d4() -> usize {
    4
}
fn d128() -> usize {
    128
}
fn d16() -> usize {
    16
}
fn d32() -> usize {
    32
}
fn d_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct QwenRope {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    /// M-RoPE (vision-language checkpoints): rotary frequencies per (time, height, width) axis,
    /// interleaved.
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    /// "llama3" when transformers 5 folds Llama's `rope_scaling` in here.
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(flatten)]
    pub llama3: Option<Llama3Rope>,
}

/// Llama 3's RoPE frequency scaling (`rope_type` "llama3"): wavelengths beyond
/// `original_max_position_embeddings / low_freq_factor` are stretched by `factor`, those below
/// `.. / high_freq_factor` kept, and the band between interpolated.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Llama3Rope {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: f64,
}

impl Llama3Rope {
    /// Applies the scaling to plain RoPE inverse frequencies (transformers'
    /// `_compute_llama3_parameters`).
    pub fn scale(&self, inv_freq: &mut [f32]) {
        let low_wavelen = self.original_max_position_embeddings / self.low_freq_factor;
        let high_wavelen = self.original_max_position_embeddings / self.high_freq_factor;
        for f in inv_freq.iter_mut() {
            let freq = *f as f64;
            let wavelen = 2.0 * std::f64::consts::PI / freq;
            *f = if wavelen < high_wavelen {
                freq
            } else if wavelen > low_wavelen {
                freq / self.factor
            } else {
                let smooth = (self.original_max_position_embeddings / wavelen - self.low_freq_factor)
                    / (self.high_freq_factor - self.low_freq_factor);
                (1.0 - smooth) * freq / self.factor + smooth * freq
            } as f32;
        }
    }
}

/// `rope_scaling` as transformers 4 configs store it (Llama 3.1 / 3.2).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub rope_type: Option<String>,
    /// The older spelling of `rope_type` (configs may carry both).
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// Qwen2-VL / Qwen2.5-VL M-RoPE: frequencies per (time, height, width) axis, in contiguous
    /// sections (not interleaved).
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(flatten)]
    pub llama3: Option<Llama3Rope>,
}

/// Qwen3.5 / Qwen3.8 dense hybrid text config (Gated DeltaNet + gated full attention), or a
/// plain Qwen3 transformer (`model_type` "qwen3": full attention only, ungated, plain RMSNorm).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct QwenConfig {
    pub hidden_size: usize,
    /// Dense MLP width (absent in MoE configs).
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "d256")]
    pub head_dim: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "d4")]
    pub full_attention_interval: usize,
    #[serde(default = "d4")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "d128")]
    pub linear_key_head_dim: usize,
    #[serde(default = "d128")]
    pub linear_value_head_dim: usize,
    #[serde(default = "d16")]
    pub linear_num_key_heads: usize,
    #[serde(default = "d32")]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub rope_parameters: Option<QwenRope>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "d_true")]
    pub attn_output_gate: bool,
    /// Sparse MoE MLPs (Qwen3.5 MoE: routed experts plus a gated shared expert) when set.
    #[serde(default, alias = "num_local_experts")]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    /// Qwen3 keeps the RoPE base at the top level.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Multi-token-prediction ("NextN") layers stored under `mtp.*` (Qwen3.5: 1).
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    /// RMSNorm scales by `1 + weight` (Qwen3.5) rather than `weight` (Qwen3, and GGUF files,
    /// whose norm weights already include the 1).
    #[serde(skip, default = "d_true")]
    pub zero_centered_norm: bool,
    /// Gated DeltaNet value heads in tiled order (GGUF): value head `h` uses key head `h % n_k`
    /// rather than `h / (n_v / n_k)`.
    #[serde(skip)]
    pub v_heads_tiled: bool,
    /// Llama 3 (`model_type` "llama") runs on the same code: full attention without query / key
    /// norms or output gate, plain RMSNorm, RoPE with the llama3 frequency scaling.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// Attention has per-head RMSNorms on the queries and keys (all but Llama).
    #[serde(skip, default = "d_true")]
    pub qk_norm: bool,
    /// RoPE rotates adjacent pairs `(2i, 2i + 1)` rather than the halves `(i, i + d/2)`: Llama
    /// GGUF files, whose query / key rows llama.cpp permuted to that layout (see `crate::gguf`).
    #[serde(skip)]
    pub rope_interleaved: bool,
    /// Query / key / value projections have biases (Qwen2, Qwen2.5-VL).
    #[serde(skip)]
    pub attention_bias: bool,
    /// M-RoPE sections interleaved over the frequencies (Qwen3.5 / 3-VL: `i % 3` picks the axis)
    /// rather than contiguous (Qwen2-VL / 2.5-VL: the first `sections[0]` frequencies rotate with
    /// the time, the next `sections[1]` with the height, the rest with the width).
    #[serde(skip, default = "d_true")]
    pub mrope_interleaved: bool,
    /// Qwen4-Exp (Qwen3.8-Flash-Next, see `crate::qwen4exp`): residual streams (hyper-connections)
    /// in place of the layer norms; 0 for the other models.
    #[serde(default)]
    pub hc_count: usize,
    #[serde(default)]
    pub hc_lowrank: usize,
    /// The activation of the Gated DeltaNet output gate (`sigmoid` for Qwen4-Exp; default SiLU).
    #[serde(default)]
    pub output_gate_type: Option<String>,
    /// QSA indexer of the full-attention layers (Qwen4-Exp): `indexer_budget` tokens in blocks of
    /// `indexer_compress_ratio`.
    #[serde(default)]
    pub indexer_budget: usize,
    #[serde(default)]
    pub indexer_compress_ratio: usize,
    #[serde(default)]
    pub indexer_head_dim: usize,
    #[serde(default)]
    pub indexer_n_heads: usize,
    /// N-gram hash embedding layers (one-based, Qwen4-Exp), injected before those layers.
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default)]
    pub ple_embed_dim: usize,
    #[serde(default)]
    pub ple_conv_kernel_size: usize,
    #[serde(default)]
    pub ngram_size: usize,
    #[serde(default)]
    pub heads_per_ngram: usize,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    /// The n-gram hash constants when the configuration carries them (GGUF metadata; a
    /// safetensors checkpoint stores them as buffers instead).
    #[serde(default)]
    pub ple_layer_multipliers: Option<Vec<u64>>,
    #[serde(default)]
    pub ple_head_offsets: Option<Vec<u64>>,
    #[serde(default)]
    pub ple_head_vocab_sizes: Option<Vec<u64>>,
    /// The image placeholder token (hashed for image positions by the n-gram embedding).
    #[serde(default)]
    pub image_token_id: Option<u32>,
    /// MiniMax M2 (`model_type` "minimax_m2") runs on the same code: full attention whose query
    /// and key norms span all heads (`qk_norm_type` "per_layer"), partial RoPE given as
    /// `rotary_dim`, sigmoid-scored routing (see [`Self::router_sigmoid`]) without a shared expert.
    #[serde(default, rename = "rotary_dim")]
    pub rotary_dims: Option<usize>,
    /// MoE router scoring: "softmax" (Qwen) or "sigmoid" (MiniMax).
    #[serde(default)]
    pub scoring_func: Option<String>,
    /// Factor on the renormalized routing weights of sigmoid routers (1 when absent).
    #[serde(default)]
    pub routed_scaling_factor: Option<f64>,
    /// "per_layer": one RMSNorm over all heads' queries (keys); "per_head" (or absent): per head.
    #[serde(default)]
    pub qk_norm_type: Option<String>,
    /// MiniMax's shared expert width (0: none).
    #[serde(default)]
    pub shared_intermediate_size: usize,
    /// MiniMax M3 (`minimax_m3_vl`): the width of the leading dense layers' MLPs.
    #[serde(default)]
    pub dense_intermediate_size: usize,
    /// MiniMax M3: per-layer MLP kind, "dense" or "sparse" (MoE); from `moe_layer_freq` (0 / 1).
    #[serde(default)]
    pub mlp_layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub moe_layer_freq: Option<Vec<u8>>,
    /// "swigluoai" (MiniMax M3: see [`crate::moe::Act::SwigluOai`]); SiLU otherwise.
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub swiglu_alpha: Option<f64>,
    #[serde(default)]
    pub swiglu_limit: Option<f64>,
    /// MiniMax M3's lightning indexer (block-sparse attention, see `crate::minimax`): as the
    /// release spells it (`sparse_attention_config`) or as transformers saves it (`index_*`).
    #[serde(default)]
    pub sparse_attention_config: Option<MiniMaxSparse>,
    #[serde(default)]
    pub index_n_heads: Option<usize>,
    #[serde(default)]
    pub index_head_dim: Option<usize>,
    #[serde(default)]
    pub index_block_size: Option<usize>,
    #[serde(default)]
    pub index_topk_blocks: Option<usize>,
    #[serde(default)]
    pub index_local_blocks: Option<usize>,
    /// MiniMax M3's RMSNorm scales by `1 + weight` (Gemma style).
    #[serde(default)]
    pub use_gemma_norm: bool,
    /// MiniMax-Text-01 / M1 (`minimax`, `minimax_text_01`, `minimax_m1`): lightning attention
    /// (see `crate::minimax`) on the layers `attn_type_list` marks 0, softmax attention on those
    /// marked 1 (`layer_types` in the `-hf` releases).
    #[serde(default)]
    pub attn_type_list: Option<Vec<u8>>,
    /// MiniMax-01: the residual is the normed input (`postnorm`), and each sublayer's output is
    /// `alpha * residual + beta * sublayer`.
    #[serde(default)]
    pub postnorm: bool,
    /// The residual scales as the `-hf` releases name them, and as the original ones do
    /// (`layernorm_*`); either may be present, or both.
    #[serde(default)]
    pub full_attn_alpha_factor: Option<f64>,
    #[serde(default)]
    pub full_attn_beta_factor: Option<f64>,
    #[serde(default)]
    pub linear_attn_alpha_factor: Option<f64>,
    #[serde(default)]
    pub linear_attn_beta_factor: Option<f64>,
    #[serde(default)]
    pub mlp_alpha_factor: Option<f64>,
    #[serde(default)]
    pub mlp_beta_factor: Option<f64>,
    #[serde(default)]
    pub layernorm_full_attention_alpha: Option<f64>,
    #[serde(default)]
    pub layernorm_full_attention_beta: Option<f64>,
    #[serde(default)]
    pub layernorm_linear_attention_alpha: Option<f64>,
    #[serde(default)]
    pub layernorm_linear_attention_beta: Option<f64>,
    #[serde(default)]
    pub layernorm_mlp_alpha: Option<f64>,
    #[serde(default)]
    pub layernorm_mlp_beta: Option<f64>,
    /// Set for MiniMax-01 checkpoints: the linear layers are lightning attention.
    #[serde(skip)]
    pub lightning: bool,
}

/// `sparse_attention_config` of MiniMax M3's release.
#[derive(Debug, Clone, Deserialize)]
pub struct MiniMaxSparse {
    #[serde(default)]
    pub sparse_num_index_heads: Option<usize>,
    #[serde(default)]
    pub sparse_index_dim: Option<usize>,
    #[serde(default)]
    pub sparse_block_size: Option<usize>,
    #[serde(default)]
    pub sparse_topk_blocks: Option<usize>,
    #[serde(default)]
    pub sparse_local_block: Option<usize>,
}

/// MiniMax M3's lightning indexer: `heads` index heads (one per KV head) of `dim`, keys pooled
/// into blocks of `block` positions by their maximum score, the `topk` best blocks plus the `local`
/// ones ending at the query attended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexerConfig {
    pub heads: usize,
    pub dim: usize,
    pub block: usize,
    pub topk: usize,
    pub local: usize,
}

impl QwenConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let gguf = crate::gguf::is_gguf(dir);
        let raw: serde_json::Value = if gguf {
            crate::gguf::header(dir)?.config_json()?
        } else {
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?
        };
        let mt = raw.get("model_type").and_then(|v| v.as_str());
        let llama = mt == Some("llama");
        // Qwen2 / Qwen2.5 and their vision-language models: no query / key norms, biased q / k / v
        // projections, contiguous M-RoPE sections.
        let qwen2 = matches!(mt, Some("qwen2" | "qwen2_vl" | "qwen2_5_vl"));
        let minimax_m3 = matches!(mt, Some("minimax_m3_vl" | "minimax_m3_vl_text"));
        let minimax_01 = matches!(mt, Some("minimax" | "minimax_text_01" | "minimax_m1" | "minimax_vl_01"));
        let minimax_hf = mt == Some("minimax");
        let minimax_m2 = mt == Some("minimax_m2") || minimax_m3;
        let dense = mt == Some("qwen3") || llama || qwen2 || minimax_m2;
        let image_token = raw.get("image_token_id").and_then(|v| v.as_u64());
        let text = raw.get("text_config").cloned().unwrap_or(raw);
        let has_head_dim = text.get("head_dim").is_some_and(|v| v.is_u64());
        let mut cfg: QwenConfig = serde_json::from_value(text).context("parse qwen text config")?;
        if cfg.image_token_id.is_none() {
            cfg.image_token_id = image_token.map(|v| v as u32);
        }
        if qwen2 {
            cfg.qk_norm = false;
            cfg.attention_bias = true;
            cfg.mrope_interleaved = false;
            if !has_head_dim {
                cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads.max(1);
            }
            if cfg.rope_theta.is_none() && cfg.rope_parameters.is_none() {
                cfg.rope_theta = Some(1_000_000.0);
            }
        }
        if llama {
            cfg.qk_norm = false;
            if !has_head_dim {
                cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads.max(1);
            }
            if cfg.rope_theta.is_none() && cfg.rope_parameters.is_none() {
                cfg.rope_theta = Some(10_000.0);
            }
        }
        if minimax_m2 {
            // `intermediate_size` is the experts' width; every layer is a MoE.
            if cfg.moe_intermediate_size == 0 {
                cfg.moe_intermediate_size = cfg.intermediate_size;
            }
            cfg.shared_expert_intermediate_size = cfg.shared_intermediate_size;
            if cfg.scoring_func.is_none() {
                cfg.scoring_func = Some("sigmoid".to_string());
            }
            // transformers' MiniMaxM2Attention norms all heads together whatever the key says;
            // M3 norms each head.
            cfg.qk_norm_type = Some(if minimax_m3 { "per_head" } else { "per_layer" }.to_string());
            if cfg.num_experts.is_none() {
                bail!("MiniMax config has no num_local_experts");
            }
        }
        if minimax_m3 {
            if cfg.mlp_layer_types.is_none() {
                cfg.mlp_layer_types = cfg
                    .moe_layer_freq
                    .as_ref()
                    .map(|f| f.iter().map(|&v| if v != 0 { "sparse" } else { "dense" }.to_string()).collect());
            }
            if cfg.mlp_layer_types.as_ref().is_some_and(|m| m.len() != cfg.num_hidden_layers) {
                bail!("mlp_layer_types length mismatch");
            }
            if cfg.hidden_act.is_none() || cfg.swiglu_alpha.is_some() {
                // transformers rewrites "swigluoai" to "silu" on save, keeping alpha / limit.
                cfg.hidden_act = Some("swigluoai".to_string());
            }
        }
        if minimax_01 {
            // Lightning attention plus softmax attention every few layers (no query / key norms,
            // partial RoPE of `rotary_dim`), Mixtral-style softmax-routed experts.
            cfg.lightning = true;
            // transformers' `minimax` always takes the residual after the norm.
            if minimax_hf {
                cfg.postnorm = true;
            }
            cfg.qk_norm = false;
            cfg.attn_output_gate = false;
            cfg.zero_centered_norm = false;
            cfg.moe_intermediate_size = cfg.intermediate_size;
            if cfg.num_experts.is_none() {
                bail!("MiniMax-01 config has no num_local_experts");
            }
            if let Some(a) = &cfg.attn_type_list {
                cfg.layer_types = Some(
                    a.iter()
                        .map(|&v| if v == 1 { "full_attention" } else { "linear_attention" }.to_string())
                        .collect(),
                );
            }
            if cfg.layer_types.is_none() {
                bail!("MiniMax-01 config has neither attn_type_list nor layer_types");
            }
        }
        if dense {
            // MiniMax M3: the sparse layers are full attention whose indexer (present in the
            // checkpoint for those layers only) picks the key blocks.
            cfg.layer_types = Some(vec!["full_attention".to_string(); cfg.num_hidden_layers]);
            cfg.attn_output_gate = false;
            cfg.zero_centered_norm = minimax_m3;
        }
        if cfg.layer_types.is_none() {
            let k = cfg.full_attention_interval.max(1);
            cfg.layer_types = Some(
                (0..cfg.num_hidden_layers)
                    .map(|i| if (i + 1) % k == 0 { "full_attention" } else { "linear_attention" }.to_string())
                    .collect(),
            );
        }
        if gguf {
            cfg.zero_centered_norm = false;
            cfg.v_heads_tiled = cfg.linear_num_key_heads != cfg.linear_num_value_heads;
            cfg.rope_interleaved = llama;
        }
        if cfg.layer_types.as_ref().unwrap().len() != cfg.num_hidden_layers {
            bail!("layer_types length mismatch");
        }
        Ok(cfg)
    }
    pub fn is_full(&self, l: usize) -> bool {
        matches!(
            self.layer_types.as_ref().unwrap()[l].as_str(),
            "full_attention" | "indexed_attention"
        )
    }
    /// Qwen4-Exp (hyper-connections, n-gram embedding, QSA; see `crate::qwen4exp`).
    pub fn is_exp(&self) -> bool {
        self.hc_count > 0
    }
    /// The Gated DeltaNet output gate is a sigmoid (Qwen4-Exp) rather than SiLU.
    pub fn gdn_sigmoid_gate(&self) -> bool {
        self.output_gate_type.as_deref() == Some("sigmoid")
    }
    /// The primary end-of-sequence id of the text config (the last of a list).
    pub fn eos_id(&self) -> Option<u32> {
        match self.eos_token_id.as_ref()? {
            serde_json::Value::Number(n) => n.as_u64().map(|v| v as u32),
            serde_json::Value::Array(a) => a.last().and_then(|v| v.as_u64()).map(|v| v as u32),
            _ => None,
        }
    }
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10_000_000.0)
    }
    /// Llama 3 frequency scaling, from `rope_scaling` or `rope_parameters`.
    pub fn llama3_rope(&self) -> Option<Llama3Rope> {
        let is_llama3 = |t: &Option<String>| t.as_deref() == Some("llama3");
        match (&self.rope_scaling, &self.rope_parameters) {
            (Some(s), _) if is_llama3(&s.rope_type) || is_llama3(&s.kind) => s.llama3,
            (_, Some(p)) if is_llama3(&p.rope_type) => p.llama3,
            _ => None,
        }
    }
    /// The M-RoPE sections (time, height, width frequency counts), if the config has them.
    pub fn mrope_section(&self) -> Option<&[usize]> {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.mrope_section.as_deref())
            .or_else(|| self.rope_scaling.as_ref().and_then(|r| r.mrope_section.as_deref()))
            .filter(|s| s.len() == 3)
    }
    /// Number of rotated dimensions per head (partial RoPE).
    pub fn rotary_dim(&self) -> usize {
        let f = self
            .rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .or(self.partial_rotary_factor);
        match (f, self.rotary_dims) {
            (Some(f), _) => (self.head_dim as f64 * f) as usize,
            (None, Some(d)) => d,
            (None, None) => self.head_dim,
        }
    }
    /// Query / key norms over all heads at once (MiniMax M2's `qk_norm_type` "per_layer").
    pub fn qk_norm_full(&self) -> bool {
        self.qk_norm_type.as_deref() == Some("per_layer")
    }
    /// MiniMax-01's residual scales of layer `l`: `(alpha, beta)` of its attention, then of its MLP
    /// (`x = alpha * norm(x) + beta * f(norm(x))`); None for pre-norm models.
    pub fn postnorm_scales(&self, l: usize) -> Option<[(f32, f32); 2]> {
        if !self.postnorm {
            return None;
        }
        let f = |v: Option<f64>| v.unwrap_or(1.0) as f32;
        let attn = if self.is_full(l) {
            (
                f(self.full_attn_alpha_factor.or(self.layernorm_full_attention_alpha)),
                f(self.full_attn_beta_factor.or(self.layernorm_full_attention_beta)),
            )
        } else {
            (
                f(self.linear_attn_alpha_factor.or(self.layernorm_linear_attention_alpha)),
                f(self.linear_attn_beta_factor.or(self.layernorm_linear_attention_beta)),
            )
        };
        let mlp = (
            f(self.mlp_alpha_factor.or(self.layernorm_mlp_alpha)),
            f(self.mlp_beta_factor.or(self.layernorm_mlp_beta)),
        );
        Some([attn, mlp])
    }
    /// Layer `l` has a dense MLP rather than a MoE (MiniMax M3's leading layers).
    pub fn is_dense_mlp(&self, l: usize) -> bool {
        self.num_experts.is_none() || self.mlp_layer_types.as_ref().is_some_and(|m| m[l] == "dense")
    }
    /// The MLP activation: SiLU, or MiniMax M3's clamped "swigluoai".
    pub fn mlp_act(&self) -> crate::moe::Act {
        match self.hidden_act.as_deref() {
            Some("swigluoai") => crate::moe::Act::SwigluOai {
                alpha: self.swiglu_alpha.unwrap_or(1.702) as f32,
                limit: self.swiglu_limit.unwrap_or(7.0) as f32,
            },
            _ => crate::moe::Act::Silu,
        }
    }
    /// MiniMax M3's lightning indexer settings (layers with indexer weights use them).
    pub fn indexer(&self) -> Option<IndexerConfig> {
        let s = self.sparse_attention_config.as_ref();
        let pick = |flat: Option<usize>, nested: Option<usize>| flat.or(nested);
        Some(IndexerConfig {
            heads: pick(self.index_n_heads, s.and_then(|s| s.sparse_num_index_heads))?,
            dim: pick(self.index_head_dim, s.and_then(|s| s.sparse_index_dim))?,
            block: pick(self.index_block_size, s.and_then(|s| s.sparse_block_size))?,
            topk: pick(self.index_topk_blocks, s.and_then(|s| s.sparse_topk_blocks))?,
            local: pick(self.index_local_blocks, s.and_then(|s| s.sparse_local_block)).unwrap_or(1),
        })
    }
    /// Sigmoid-scored MoE routing (MiniMax): experts chosen by `sigmoid(logit) + bias`, weighted
    /// by their sigmoid scores renormalized over the chosen ones (times `routed_scaling_factor`).
    pub fn router_sigmoid(&self) -> bool {
        self.scoring_func.as_deref() == Some("sigmoid")
    }
}

fn d_ml4_one() -> f64 {
    1.0
}
fn d_ml4_dense() -> usize {
    1
}

/// YaRN RoPE parameters of Mistral Large 4 (`rope_parameters`).
#[derive(Debug, Clone, Deserialize)]
pub struct YarnParams {
    #[serde(default = "d_ml4_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub beta_fast: Option<f64>,
    #[serde(default)]
    pub beta_slow: Option<f64>,
    /// The cos / sin magnitude (None: YaRN's `0.1 ln(factor) + 1`, or the mscale ratio).
    #[serde(default)]
    pub attention_factor: Option<f64>,
    #[serde(default)]
    pub mscale: Option<f64>,
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
    #[serde(default)]
    pub truncate: Option<bool>,
}

fn d_ml4_theta() -> f64 {
    10_000.0
}

/// Mistral Large 4 text model (`mistral_large4_text`, transformers' `MistralLarge4TextConfig`):
/// MLA attention without query LoRA, DSA indexer, no-RoPE and KV-sharing layers, DeepSeek-style
/// MoE. Defaults are transformers'; a GGUF header is mapped onto the same fields
/// (`gguf::Header::config_json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Mistral4Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "d_eps5")]
    pub rms_norm_eps: f64,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "d_ml4_dense")]
    pub n_shared_experts: usize,
    #[serde(default = "d_ml4_one")]
    pub routed_scaling_factor: f64,
    #[serde(default = "d_ml4_dense")]
    pub n_group: usize,
    #[serde(default = "d_ml4_dense")]
    pub topk_group: usize,
    #[serde(default = "d_ml4_dense")]
    pub first_k_dense_replace: usize,
    /// "dense" or "sparse" per layer (default: `first_k_dense_replace` dense layers).
    #[serde(default)]
    pub mlp_layer_types: Option<Vec<String>>,
    pub index_topk: usize,
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    /// "full" (runs the indexer) or "shared" (reuses the previous full layer's top-k) per layer.
    #[serde(default)]
    pub indexer_types: Option<Vec<String>>,
    /// 1 = RoPE, 0 = NoPE, per layer (default: every `no_rope_layer_interval`-th layer is NoPE).
    #[serde(default)]
    pub no_rope_layers: Option<Vec<u8>>,
    #[serde(default = "d4")]
    pub no_rope_layer_interval: usize,
    /// The layer whose latent KV cache each layer attends to (default: its own).
    #[serde(default)]
    pub kv_source_layers: Option<Vec<usize>>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_parameters: Option<YarnParams>,
    /// Epsilon of the indexer's LayerNorms (transformers hardcodes 1e-6; GGUF stores it).
    #[serde(default = "d_eps")]
    pub indexer_norm_eps: f64,
}

fn d_eps5() -> f64 {
    1e-5
}

impl Mistral4Config {
    /// The text config of a Mistral Large 4 checkpoint folder (`config.json`, root or
    /// `text_config`) or GGUF file, with transformers' derived defaults filled in.
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = if crate::gguf::is_gguf(dir) {
            crate::gguf::header(dir)?.config_json()?
        } else {
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?
        };
        let text = raw.get("text_config").cloned().unwrap_or(raw);
        let mut c: Mistral4Config = serde_json::from_value(text.clone()).context("parse mistral_large4 text config")?;
        let n = c.num_hidden_layers;
        if c.no_rope_layers.is_none() {
            let k = c.no_rope_layer_interval.max(1);
            c.no_rope_layers = Some((0..n).map(|i| ((i + 1) % k != 0) as u8).collect());
        }
        if c.kv_source_layers.is_none() {
            c.kv_source_layers = Some((0..n).collect());
        }
        if c.indexer_types.is_none() {
            // `index_topk_pattern` ("FSSF...") or the freq / offset schedule, as transformers.
            c.indexer_types = Some(match text.get("index_topk_pattern").and_then(|v| v.as_str()) {
                Some(p) => p.chars().map(|ch| if ch == 'S' { "shared" } else { "full" }.to_string()).collect(),
                None => {
                    let freq = text.get("index_topk_freq").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
                    let off = text.get("index_skip_topk_offset").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
                    (0..n)
                        .map(|i| {
                            if (i + 1).saturating_sub(off) % freq == 0 {
                                "full"
                            } else {
                                "shared"
                            }
                            .to_string()
                        })
                        .collect()
                }
            });
        }
        if c.mlp_layer_types.is_none() {
            let d = c.first_k_dense_replace.min(n);
            c.mlp_layer_types = Some((0..n).map(|i| if i < d { "dense" } else { "sparse" }.to_string()).collect());
        }
        let lens = [
            c.no_rope_layers.as_ref().unwrap().len(),
            c.kv_source_layers.as_ref().unwrap().len(),
            c.indexer_types.as_ref().unwrap().len(),
            c.mlp_layer_types.as_ref().unwrap().len(),
        ];
        if lens.iter().any(|&l| l != n) {
            bail!("mistral_large4: per-layer lists of {lens:?} entries for {n} layers");
        }
        for l in 0..n {
            let s = c.kv_source(l);
            if s > l || c.kv_source(s) != s || c.has_rope(s) != c.has_rope(l) {
                bail!("mistral_large4: layer {l} cannot share the KV cache of layer {s}");
            }
        }
        if c.indexer_types.as_ref().unwrap().first().is_some_and(|t| t != "full") {
            bail!("mistral_large4: the first layer must run its own indexer");
        }
        if !c.n_routed_experts.is_multiple_of(c.n_group.max(1)) || c.topk_group > c.n_group || c.num_experts_per_tok > c.n_routed_experts {
            bail!("mistral_large4: bad expert grouping");
        }
        if !c.qk_rope_head_dim.is_multiple_of(2) || c.qk_rope_head_dim > c.index_head_dim {
            bail!("mistral_large4: bad rotary dimension {}", c.qk_rope_head_dim);
        }
        Ok(c)
    }
    pub fn has_rope(&self, l: usize) -> bool {
        self.no_rope_layers.as_ref().unwrap()[l] != 0
    }
    pub fn kv_source(&self, l: usize) -> usize {
        self.kv_source_layers.as_ref().unwrap()[l]
    }
    pub fn full_indexer(&self, l: usize) -> bool {
        self.indexer_types.as_ref().unwrap()[l] == "full"
    }
    pub fn is_moe(&self, l: usize) -> bool {
        self.mlp_layer_types.as_ref().unwrap()[l] == "sparse"
    }
    /// Query / key head size: the non-rotary part plus the rotary part (the same on NoPE layers,
    /// whose non-rotary part takes the whole head).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// The attention scale: `qk_head_dim^-0.5`, times `mscale^2` when YaRN has `mscale_all_dim`
    /// (transformers' `yarn_apply_mscale`).
    pub fn attn_scale(&self) -> f32 {
        let mut s = (self.qk_head_dim() as f64).powf(-0.5);
        if let Some(r) = &self.rope_parameters
            && r.rope_type.as_deref().is_some_and(|t| t != "default")
            && let Some(m) = r.mscale_all_dim.filter(|&m| m != 0.0)
        {
            let f = r.factor.unwrap_or(1.0);
            let ms = if f <= 1.0 { 1.0 } else { 0.1 * m * f.ln() + 1.0 };
            s *= ms * ms;
        }
        s as f32
    }
    /// The rotary inverse frequencies (`qk_rope_head_dim / 2` of them) and the cos / sin
    /// magnitude, as transformers' YaRN (`_compute_yarn_parameters`) or default RoPE computes
    /// them in float32; the learned LeRoPE factors are applied by the model.
    pub fn rope(&self) -> (Vec<f32>, f32) {
        let r = self.rope_parameters.clone().unwrap_or(YarnParams {
            rope_theta: 10_000.0,
            rope_type: Some("yarn".into()),
            factor: Some(2048.0),
            original_max_position_embeddings: Some(512),
            beta_fast: Some(32.0),
            beta_slow: Some(1.0),
            attention_factor: Some(1.0),
            mscale: None,
            mscale_all_dim: None,
            truncate: None,
        });
        r.inv_freq(self.qk_rope_head_dim)
    }
}

fn d_ds4_sqrtsoftplus() -> String {
    "sqrtsoftplus".into()
}
fn d_ds4_hc() -> usize {
    4
}
fn d_ds4_sinkhorn() -> usize {
    20
}
fn d_ds4_window() -> usize {
    128
}
fn d_ds4_ctheta() -> f64 {
    160_000.0
}

/// The `rope_scaling` of a DeepSeek V4 config (YaRN on the compressed layers).
#[derive(Debug, Clone, Deserialize)]
pub struct Ds4RopeScaling {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub beta_fast: Option<f64>,
    #[serde(default)]
    pub beta_slow: Option<f64>,
}

/// DeepSeek V4 (`deepseek_v4`, DeepSeek-V4-Flash / -Pro) in DeepSeek's own `config.json` keys (a
/// GGUF header is mapped onto the same fields, `gguf::Header::config_json`); see
/// [`crate::deepseek4`].
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeek4Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub vocab_size: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    pub n_routed_experts: usize,
    #[serde(default = "d_ml4_dense")]
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    #[serde(default = "d_ml4_one")]
    pub routed_scaling_factor: f64,
    /// `sqrtsoftplus`, `sigmoid` or `softmax`.
    #[serde(default = "d_ds4_sqrtsoftplus")]
    pub scoring_func: String,
    #[serde(default = "d_true")]
    pub norm_topk_prob: bool,
    /// Clamp of the experts' SwiGLU inputs (0: none).
    #[serde(default)]
    pub swiglu_limit: f64,
    /// The first layers route by token id (`ffn.gate.tid2eid`).
    #[serde(default)]
    pub num_hash_layers: usize,
    #[serde(default = "d_ds4_window")]
    pub sliding_window: usize,
    /// Per layer: 0 (sliding window only), 4 (compressed sparse attention with the indexer) or
    /// another rate (heavily compressed attention); may list the MTP / DSpark layers after them.
    pub compress_ratios: Vec<usize>,
    #[serde(default = "d_ds4_ctheta")]
    pub compress_rope_theta: f64,
    #[serde(default = "d_ml4_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<Ds4RopeScaling>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    #[serde(default = "d_ds4_hc")]
    pub hc_mult: usize,
    #[serde(default = "d_ds4_sinkhorn")]
    pub hc_sinkhorn_iters: usize,
    #[serde(default = "d_eps")]
    pub hc_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl DeepSeek4Config {
    /// The config of a DeepSeek V4 checkpoint folder (`config.json`) or GGUF file.
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = if crate::gguf::is_gguf(dir) {
            crate::gguf::header(dir)?.config_json()?
        } else {
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?
        };
        Self::from_json(raw)
    }

    pub fn from_json(raw: serde_json::Value) -> Result<Self> {
        let c: DeepSeek4Config = serde_json::from_value(raw).context("parse deepseek_v4 config")?;
        let n = c.num_hidden_layers;
        if c.compress_ratios.len() < n {
            bail!("deepseek_v4: {} compress_ratios for {n} layers", c.compress_ratios.len());
        }
        if !c.num_attention_heads.is_multiple_of(c.o_groups.max(1)) || c.qk_rope_head_dim > c.head_dim.min(c.index_head_dim) {
            bail!("deepseek_v4: bad head layout");
        }
        if !matches!(c.scoring_func.as_str(), "sqrtsoftplus" | "sigmoid" | "softmax") {
            bail!("deepseek_v4: scoring function {:?} is not supported", c.scoring_func);
        }
        if c.hc_mult == 0 || !c.qk_rope_head_dim.is_multiple_of(2) {
            bail!("deepseek_v4: bad hyper-connection or rotary settings");
        }
        Ok(c)
    }

    /// The compression rate of layer `l` (0: sliding window only).
    pub fn ratio(&self, l: usize) -> usize {
        self.compress_ratios[l]
    }

    /// Rotary inverse frequencies (`qk_rope_head_dim / 2`) of the sliding-window-only layers
    /// (plain RoPE, `rope_theta`) and of the compressed ones (`compress_rope_theta` with YaRN,
    /// without its magnitude scaling), as DeepSeek's `precompute_freqs_cis`.
    pub fn rope_freqs(&self) -> (Vec<f32>, Vec<f32>) {
        let rd = self.qk_rope_head_dim;
        let plain = YarnParams {
            rope_theta: self.rope_theta,
            rope_type: Some("default".into()),
            factor: None,
            original_max_position_embeddings: None,
            beta_fast: None,
            beta_slow: None,
            attention_factor: None,
            mscale: None,
            mscale_all_dim: None,
            truncate: None,
        };
        let main = plain.inv_freq(rd).0;
        let rs = self.rope_scaling.as_ref();
        let yarn =
            rs.is_some_and(|r| r.r#type.as_deref().unwrap_or("yarn") == "yarn" && r.original_max_position_embeddings.unwrap_or(0) > 0);
        let comp = YarnParams {
            rope_theta: self.compress_rope_theta,
            rope_type: Some(if yarn { "yarn" } else { "default" }.into()),
            factor: rs.and_then(|r| r.factor),
            original_max_position_embeddings: rs.and_then(|r| r.original_max_position_embeddings),
            beta_fast: rs.and_then(|r| r.beta_fast),
            beta_slow: rs.and_then(|r| r.beta_slow),
            attention_factor: Some(1.0),
            mscale: None,
            mscale_all_dim: None,
            truncate: None,
        };
        (main, comp.inv_freq(rd).0)
    }
}

impl YarnParams {
    /// The rotary inverse frequencies (`dim / 2` of them) and the cos / sin magnitude, as
    /// transformers' YaRN (`_compute_yarn_parameters`) or default RoPE computes them in float32.
    pub fn inv_freq(&self, dim: usize) -> (Vec<f32>, f32) {
        let r = self;
        let base = r.rope_theta;
        let pos: Vec<f32> = (0..dim / 2).map(|i| (base as f32).powf((2 * i) as f32 / dim as f32)).collect();
        if r.rope_type.as_deref() != Some("yarn") {
            return (pos.iter().map(|p| 1.0 / p).collect(), 1.0);
        }
        let factor = r.factor.unwrap_or(1.0);
        let orig = r.original_max_position_embeddings.unwrap_or(4096) as f64;
        let mscale = |s: f64, m: f64| if s <= 1.0 { 1.0 } else { 0.1 * m * s.ln() + 1.0 };
        let att = r.attention_factor.unwrap_or_else(|| match (r.mscale, r.mscale_all_dim) {
            (Some(a), Some(b)) if a != 0.0 && b != 0.0 => mscale(factor, a) / mscale(factor, b),
            _ => mscale(factor, 1.0),
        });
        let (bf, bs) = (
            r.beta_fast.filter(|&b| b != 0.0).unwrap_or(32.0),
            r.beta_slow.filter(|&b| b != 0.0).unwrap_or(1.0),
        );
        let corr = |rot: f64| (dim as f64 * (orig / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln());
        let (mut lo, mut hi) = (corr(bf), corr(bs));
        if r.truncate.unwrap_or(true) {
            (lo, hi) = (lo.floor(), hi.ceil());
        }
        let (lo, mut hi) = (lo.max(0.0) as f32, hi.min(dim as f64 - 1.0) as f32);
        if lo == hi {
            hi += 0.001;
        }
        let inv = (0..dim / 2)
            .map(|i| {
                let ramp = ((i as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
                let extra = 1.0 - ramp;
                let (e, ip) = (1.0 / pos[i], 1.0 / (factor as f32 * pos[i]));
                ip * (1.0 - extra) + e * extra
            })
            .collect();
        (inv, att as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 4 26B-A4B as transformers 5.20 saves it: `per_layer_config` for the global layers,
    /// `"vision"` bidirectional attention, a MoE block.
    #[test]
    fn gemma4_moe_config() {
        let dir = std::env::temp_dir().join(format!("oxinfer-cfg-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = serde_json::json!({"model_type": "gemma4", "text_config": {
            "hidden_size": 2816, "intermediate_size": 2112, "num_hidden_layers": 6, "num_attention_heads": 16,
            "num_key_value_heads": 8, "head_dim": 256, "attention_k_eq_v": true,
            "per_layer_config": {"5": {"head_dim": 512, "num_key_value_heads": 2}},
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "use_bidirectional_attention": "vision", "enable_moe_block": true, "num_experts": 128, "top_k_experts": 8,
            "moe_intermediate_size": 704, "hidden_size_per_layer_input": 0}});
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let c = TextConfig::load(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(
            (c.global_head_dim, c.layer_n_kv(5), c.layer_n_kv(0), c.layer_head_dim(5)),
            (512, 2, 8, 512)
        );
        assert!(c.vision_bidirectional() && !c.bidirectional());
        assert_eq!(c.moe(), Some((128, 8, 704)));
    }
}
