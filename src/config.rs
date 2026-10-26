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
    /// Encoder use (EmbeddingGemma): every position attends both ways (within the window on
    /// sliding layers).
    #[serde(default)]
    pub use_bidirectional_attention: Option<bool>,
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

impl TextConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&path)?)?;
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
        let mut cfg: TextConfig = serde_json::from_value(text).context("parse text config")?;
        cfg.gemma3 = gemma3;
        if gemma3 && cfg.bidirectional() {
            // As transformers' Gemma3TextConfig: the stored window spans both directions, so each
            // side sees `window / 2` (keys less than `window / 2 + 1` positions away).
            cfg.sliding_window = cfg.sliding_window / 2 + 1;
        }
        cfg.finalize()?;
        Ok(cfg)
    }

    pub fn bidirectional(&self) -> bool {
        self.use_bidirectional_attention.unwrap_or(false)
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
        if self.enable_moe_block {
            bail!("MoE models (26B-A4B) are not supported yet");
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
    #[serde(default, alias = "type")]
    pub rope_type: Option<String>,
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
    #[serde(default)]
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
        let dense = mt == Some("qwen3") || llama;
        let text = raw.get("text_config").cloned().unwrap_or(raw);
        let has_head_dim = text.get("head_dim").is_some_and(|v| v.is_u64());
        let mut cfg: QwenConfig = serde_json::from_value(text).context("parse qwen text config")?;
        if llama {
            cfg.qk_norm = false;
            if !has_head_dim {
                cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads.max(1);
            }
            if cfg.rope_theta.is_none() && cfg.rope_parameters.is_none() {
                cfg.rope_theta = Some(10_000.0);
            }
        }
        if dense {
            cfg.layer_types = Some(vec!["full_attention".to_string(); cfg.num_hidden_layers]);
            cfg.attn_output_gate = false;
            cfg.zero_centered_norm = false;
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
        self.layer_types.as_ref().unwrap()[l] == "full_attention"
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
            (Some(s), _) if is_llama3(&s.rope_type) => s.llama3,
            (_, Some(p)) if is_llama3(&p.rope_type) => p.llama3,
            _ => None,
        }
    }
    /// The M-RoPE sections (time, height, width frequency counts), if the config has them.
    pub fn mrope_section(&self) -> Option<&[usize]> {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.mrope_section.as_deref())
            .filter(|s| s.len() == 3)
    }
    /// Number of rotated dimensions per head (partial RoPE).
    pub fn rotary_dim(&self) -> usize {
        let f = self
            .rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .or(self.partial_rotary_factor)
            .unwrap_or(1.0);
        (self.head_dim as f64 * f) as usize
    }
}
