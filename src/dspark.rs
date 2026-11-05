//! DSpark speculative decoding (DeepSeek's DeepSpec, llama.cpp `--spec-type draft-dspark`, vLLM
//! `dspark`): a small draft model that reads the target model's hidden states and proposes a
//! whole block of tokens per forward pass.
//!
//! A draft is a DFlash block-diffusion backbone plus a semi-autoregressive "Markov" head:
//!
//! - Context: for every position the target model has processed, the residual streams entering
//!   a few target layers ([`DsparkConfig::target_layers`]; the layer count stands for the last
//!   layer's output) are concatenated, projected by `fc`, RMS-normed (`hidden_norm`), and each
//!   draft layer turns that into its own key / value cache entry (no attention norm on this
//!   path).
//! - Block: the last committed token (the "anchor", not yet run by the target) at position `n`,
//!   followed by mask embeddings at `n + 1..` (a learned table, or the embedding of a mask
//!   token), goes through the draft layers attending to the cached context of the positions
//!   before `n` (within a window on sliding layers) and to the block (all of it, or the rows up
//!   to its own on causal layers). The final-normed states go through the LM head (the draft's
//!   own or the target's, soft-capped with the draft's own cap); with `sample_from_anchor` slot
//!   `i` predicts the token at `n + i + 1`, otherwise the anchor row is unused and mask row `i`
//!   predicts the token at its own position (z-lab's DFlash layout).
//! - Markov head: each slot's logits get a low-rank bias keyed on the previous token
//!   (`markov_w2 * f(markov_w1[prev])`), chained across the block; the "rnn" variant carries a
//!   gated state through a joint projection of `[state, markov_w1[prev], hidden]`. The chain uses
//!   the drafted tokens, so the draft distribution of each slot is exactly what it was drawn from.
//! - Confidence head (optional): `sigmoid(w . [hidden, markov_w1[prev]] + b)` estimates the
//!   probability that a slot is accepted; [`DsparkOptions::conf_min`] cuts the block there.
//!
//! Backbones: the dense MLA layers of the Mistral Large 4 draft (latent K = V cache, llama.cpp's
//! `graph_mistral_large4`), and grouped-query attention layers: Qwen3 (DeepSpec
//! `Qwen3DSparkModel`, SpecForge / RadixArk `DSparkDraftModel`, NVIDIA Model Optimizer, z-lab
//! `DFlashDraftModel` with a Markov head, vLLM speculators `DSparkDraftModel`) and Gemma 4
//! (DeepSpec `Gemma4DSparkModel`: K = V attention, sandwich norms, proportional RoPE, GELU).
//! Drafts load from a Hugging Face folder (`config.json` + safetensors) or a llama.cpp `dflash`
//! GGUF.
//!
//! The target verifies `[anchor, drafts..]` in one pass (standard speculative sampling, as
//! [`crate::qwen_mtp`]); the features of the committed positions then join the draft's cache.
//! The target side is the [`Target`] trait.

use crate::generate::TextLm;
use crate::gguf::{Header, Value};
use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, linear, linear_multi, linear_rows_device, load_lin};
use crate::qwen_mtp::{MtpStats, speculative_accept};
use crate::safetensors::ModelFiles;
use crate::sampler::Sampler;
use anyhow::{Context, Result, bail};
use serde_json::Value as Json;
use std::path::{Path, PathBuf};

/// What a DSpark draft needs from its target model.
pub trait Target: TextLm {
    /// Runs `tokens` like [`TextLm::forward_hidden_mm`] (final-normed hidden states `[t,
    /// hidden]`) and writes to `feats` the residual stream entering each of `layers` (`[t,
    /// layers.len() * hidden]`; the layer count stands for the last layer's output before the
    /// final norm). Must allow [`Target::truncate`] back to any position inside `tokens`.
    fn forward_feats(
        &self,
        sess: &mut Self::Sess,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32>;
    /// The verification pass: [`Target::forward_feats`] without multimodal inputs, returning the
    /// logits `[t, vocab]` of every token (a GPU pass where the model has one). With `fast`, a
    /// GPU pass may use batched kernels whose rounding differs from one-token steps
    /// ([`DsparkOptions::fast_verify`]).
    fn forward_verify(&self, sess: &mut Self::Sess, tokens: &[u32], layers: &[usize], feats: &mut Vec<f32>, fast: bool) -> Vec<f32> {
        let _ = fast;
        let hs = self.forward_feats(sess, tokens, &[], layers, feats);
        self.logits(&hs, tokens.len())
    }
    /// The LM head matrix (`[vocab, hidden]`) for a draft that runs on the GPU, if the target
    /// has one in a form the GPU plans take.
    fn lm_head(&self) -> Option<&Lin<'_>> {
        None
    }
    /// The row of `token` in the input embedding table (`[hidden]`, before
    /// [`Target::embed_scale`]).
    fn embed_row(&self, token: u32, out: &mut [f32]);
    /// The factor the target applies to its input embeddings (Gemma: `sqrt(hidden)`).
    fn embed_scale(&self) -> f32 {
        1.0
    }
    /// LM-head logits `[t, vocab]` without the target's own soft-capping.
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32>;
    /// [`Target::head_raw`] for a draft's few rows (1 to 4) on the accelerator's matrix-vector
    /// kernels ([`crate::gpu::Backend::matvec_rows`]); None when it cannot (the caller then uses
    /// `head_raw`).
    fn head_raw_device(&self, _hidden: &[f32], _t: usize) -> Option<Vec<f32>> {
        None
    }
    fn vocab_size(&self) -> usize;
    /// Drops cached positions at or beyond `len`.
    fn truncate(&self, sess: &mut Self::Sess, len: usize);
}

/// The draft's decoder layers.
#[derive(Clone, Debug)]
pub enum Backbone {
    /// Mistral Large 4's dense MLA layers: latent (`kv_lora_rank`), non-rotary / rotary
    /// query-key and value head sizes.
    Mla { lat: usize, dn: usize, dr: usize, vd: usize },
    /// Grouped-query attention layers (Qwen3 or Gemma 4 style).
    Gqa(GqaConfig),
    /// Full DeepSeek V4 stages (hyper-connections, compressed attention, MoE; `crate::deepseek4`).
    Ds4(Box<crate::config::DeepSeek4Config>),
}

/// Shapes and variants of grouped-query attention draft layers.
#[derive(Clone, Debug)]
pub struct GqaConfig {
    pub n_kv: usize,
    pub head_dim: usize,
    /// No value projection: the values are the raw keys (Gemma 4 `attention_k_eq_v`).
    pub k_eq_v: bool,
    /// Values get a weightless RMSNorm (Gemma 4).
    pub v_norm: bool,
    /// Gemma 4's sandwich norms (post-attention, pre- and post-feed-forward) and layer scalars.
    pub sandwich: bool,
    /// GELU (tanh) feed-forward instead of SiLU.
    pub gelu: bool,
    pub attn_scale: f32,
    /// Leading dims of each head rotated (rotate-half pairs `(i, i + rope_dims / 2)`).
    pub rope_dims: usize,
    /// Gemma 4's "proportional" RoPE: of the `rope_dims / 2` frequencies (computed over
    /// `rope_dims`), only this fraction is non-zero.
    pub rope_proportion: f64,
}

/// How the block's mask rows are embedded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskEmbed {
    /// A learned table of `block_size - 1` rows (`mask_embd`).
    Learned,
    /// The embedding of this token id.
    Token(u32),
}

/// The draft's hyperparameters.
#[derive(Clone, Debug)]
pub struct DsparkConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub inter: usize,
    pub eps: f32,
    pub backbone: Backbone,
    /// Per layer: context positions the block attends to (0: all).
    pub windows: Vec<usize>,
    /// Per layer: whether block row `i` sees only rows `..=i` of the block.
    pub causal: Vec<bool>,
    /// Trained block: drafts per pass with `sample_from_anchor`, else one more slot than drafts.
    pub block_size: usize,
    pub sample_from_anchor: bool,
    /// Target layers whose input residual is read (the target layer count = its last output).
    pub target_layers: Vec<usize>,
    pub softcap: Option<f32>,
    pub rope: crate::config::YarnParams,
    pub mask: MaskEmbed,
    /// Factor on the block's token embeddings (None: the target's, [`Target::embed_scale`]).
    pub embed_scale: Option<f32>,
}

fn default_rope(theta: f64) -> crate::config::YarnParams {
    crate::config::YarnParams {
        rope_theta: theta,
        rope_type: Some("default".into()),
        factor: None,
        original_max_position_embeddings: None,
        beta_fast: None,
        beta_slow: None,
        attention_factor: None,
        mscale: None,
        mscale_all_dim: None,
        truncate: None,
    }
}

impl DsparkConfig {
    /// A llama.cpp `dflash` GGUF's metadata (or a `dspark` one, from llama.cpp's DSpark branch:
    /// the same keys under `dspark.`, without the Gemma 4 scales, which are then Gemma's).
    pub fn from_header(h: &Header) -> Result<Self> {
        let arch = h.arch();
        if !matches!(arch, "dflash" | "dspark") {
            bail!("not a DFlash / DSpark draft (GGUF architecture {arch:?})");
        }
        let get = |k: &str| h.get(&format!("{arch}.{k}"));
        // Per-layer arrays (`head_count_kv` in the DSpark branch's Gemma files) must be uniform.
        let u = |k: &str| match get(k) {
            Some(Value::Array(a)) => {
                let v: Vec<u64> = a.iter().filter_map(Value::as_u64).collect();
                (v.len() == a.len() && v.first().is_some_and(|f| v.iter().all(|x| x == f))).then(|| v[0] as usize)
            }
            v => v.and_then(Value::as_u64).map(|v| v as usize),
        };
        let need = |k: &str| u(k).with_context(|| format!("DSpark GGUF has no (uniform) {arch}.{k}"));
        let f = |k: &str| get(k).and_then(Value::as_f64);
        let b = |k: &str| match get(k) {
            Some(Value::Bool(v)) => Some(*v),
            _ => None,
        };
        let target_layers: Vec<usize> = get("target_layers")
            .and_then(Value::as_array)
            .context("DSpark GGUF has no dflash.target_layers")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as usize).context("bad dflash.target_layers"))
            .collect::<Result<_>>()?;
        // DeepSeek V4: full DeepSeek V4 stages (hyper-connections, MoE), `crate::deepseek4`.
        if get("hyper_connection.count").is_some() {
            let ds = crate::config::DeepSeek4Config::from_json(h.deepseek4_config()?)?;
            let Some(noise) = h.get("tokenizer.ggml.mask_token_id").and_then(Value::as_u64) else {
                bail!("bad DeepSeek V4 DSpark metadata (no mask token)");
            };
            let n_layers = ds.num_hidden_layers;
            let c = Self {
                hidden: ds.hidden_size,
                n_layers,
                n_heads: ds.num_attention_heads,
                inter: ds.moe_intermediate_size,
                eps: ds.rms_norm_eps as f32,
                windows: vec![ds.sliding_window; n_layers],
                causal: vec![false; n_layers],
                block_size: need("block_size")?,
                sample_from_anchor: !matches!(get("sample_from_anchor"), Some(Value::Bool(false))),
                target_layers,
                softcap: None,
                rope: default_rope(ds.rope_theta),
                mask: MaskEmbed::Token(noise as u32),
                embed_scale: None,
                backbone: Backbone::Ds4(Box::new(ds)),
            };
            if c.block_size < 2 || c.target_layers.is_empty() {
                bail!("bad DeepSeek V4 DSpark metadata (block size, target layers)");
            }
            return Ok(c);
        }
        let theta = f("rope.freq_base").unwrap_or(10_000.0);
        let mla = u("attention.kv_lora_rank");
        let rope = match get("rope.scaling.type").and_then(Value::as_str) {
            Some("yarn") => {
                // llama.cpp's magnitude: mscale(factor, 1) / mscale(factor, yarn_log_multiplier),
                // or mscale(factor, 1) without the multiplier; queries keep 1/sqrt(head dim).
                let log_mul = f("rope.scaling.yarn_log_multiplier").filter(|&m| m != 0.0);
                crate::config::YarnParams {
                    rope_theta: theta,
                    rope_type: Some("yarn".into()),
                    factor: f("rope.scaling.factor"),
                    original_max_position_embeddings: u("rope.scaling.original_context_length"),
                    beta_fast: f("rope.scaling.yarn_beta_fast"),
                    beta_slow: f("rope.scaling.yarn_beta_slow"),
                    attention_factor: f("rope.scaling.yarn_attn_factor").filter(|_| mla.is_none()),
                    mscale: log_mul.map(|_| 1.0),
                    mscale_all_dim: log_mul,
                    truncate: None,
                }
            }
            None | Some("none") => default_rope(theta),
            Some(t) => bail!("DSpark draft RoPE scaling {t:?} is not supported"),
        };
        let n_layers = need("block_count")?;
        let n_heads = need("attention.head_count")?;
        let hidden = need("embedding_length")?;
        let window = u("attention.sliding_window").unwrap_or(0);
        let (backbone, windows, causal) = match mla {
            Some(lat) => {
                let dr = need("rope.dimension_count")?;
                let qk = need("attention.key_length_mla")?;
                let bb = Backbone::Mla {
                    lat,
                    dn: qk.checked_sub(dr).context("dflash.attention.key_length_mla < rope dims")?,
                    dr,
                    vd: need("attention.value_length_mla")?,
                };
                (bb, vec![window; n_layers], vec![false; n_layers])
            }
            None => {
                let hd = u("attention.key_length").unwrap_or(hidden / n_heads.max(1));
                let rot = u("rope.dimension_count").unwrap_or(hd);
                let pattern: Option<Vec<bool>> = get("attention.sliding_window_pattern").and_then(Value::as_array).map(|a| {
                    a.iter()
                        .map(|v| matches!(v, Value::Bool(true)) || v.as_u64().is_some_and(|x| x != 0))
                        .collect()
                });
                let swa = |l: usize| window > 0 && pattern.as_ref().is_none_or(|p| p.get(l).copied().unwrap_or(false));
                let windows = (0..n_layers).map(|l| if swa(l) { window } else { 0 }).collect();
                // llama.cpp: one causal flag for every layer (default non-causal).
                let causal = vec![b("attention.causal").unwrap_or(false); n_layers];
                let shared_kv = !h.tensors.iter().any(|t| t.name == "blk.0.attn_v.weight");
                // Gemma 4's sandwich norms (the DSpark branch names Qwen's feed-forward input
                // norm `post_attention_norm` too, so the post-feed-forward norm tells them apart).
                let gemma = h.tensors.iter().any(|t| t.name == "blk.0.post_ffw_norm.weight");
                let bb = Backbone::Gqa(GqaConfig {
                    n_kv: need("attention.head_count_kv")?,
                    head_dim: hd,
                    k_eq_v: shared_kv,
                    v_norm: shared_kv || gemma,
                    sandwich: gemma,
                    gelu: match get("hidden_act").and_then(Value::as_str) {
                        Some(a) => matches!(a, "gelu" | "gelu_pytorch_tanh"),
                        None => gemma,
                    },
                    attn_scale: f("attention.scale")
                        .filter(|&s| s != 0.0)
                        .map_or(if gemma { 1.0 } else { (hd as f32).powf(-0.5) }, |s| s as f32),
                    rope_dims: rot,
                    rope_proportion: 1.0,
                });
                (bb, windows, causal)
            }
        };
        let mask = match mla {
            Some(_) => MaskEmbed::Learned,
            None => MaskEmbed::Token(
                h.get("tokenizer.ggml.mask_token_id")
                    .and_then(Value::as_u64)
                    .context("DSpark GGUF has no tokenizer.ggml.mask_token_id")? as u32,
            ),
        };
        // Without `embedding_scale`, a Gemma 4 draft scales its embeddings as its target does.
        let gemma = matches!(&backbone, Backbone::Gqa(g) if g.sandwich);
        let embed_scale = match f("embedding_scale").filter(|&s| s != 0.0) {
            Some(s) => Some(s as f32),
            None => (!gemma).then_some(1.0),
        };
        let c = Self {
            hidden,
            n_layers,
            n_heads,
            inter: need("feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
            backbone,
            windows,
            causal,
            block_size: need("block_size")?,
            sample_from_anchor: !matches!(get("sample_from_anchor"), Some(Value::Bool(false))),
            target_layers,
            softcap: f("final_logit_softcapping").filter(|&c| c > 0.0).map(|c| c as f32),
            rope,
            mask,
            embed_scale,
        };
        c.check()?;
        Ok(c)
    }

    /// A Hugging Face draft's `config.json`: DeepSpec (`Qwen3DSparkModel`, `Gemma4DSparkModel`),
    /// SpecForge / RadixArk (`DSparkDraftModel` with a `dflash_config`), NVIDIA Model Optimizer,
    /// z-lab `DFlashDraftModel` checkpoints with a Markov head, and vLLM speculators
    /// (`transformer_layer_config`).
    pub fn from_hf(raw: &Json) -> Result<Self> {
        let archs: Vec<&str> = raw
            .get("architectures")
            .and_then(Json::as_array)
            .map(|a| a.iter().filter_map(Json::as_str).collect())
            .unwrap_or_default();
        // Speculators keep the decoder layers' config apart.
        let speculators = raw.get("transformer_layer_config").is_some();
        let mut c = raw.clone();
        if let (Some(m), Some(Json::Object(t))) = (c.as_object_mut(), raw.get("transformer_layer_config")) {
            for (k, v) in t {
                m.insert(k.clone(), v.clone());
            }
        }
        let dfl = c
            .get("dflash_config")
            .or_else(|| c.get("dspark_config"))
            .cloned()
            .unwrap_or(Json::Null);
        let get = |k: &str| dfl.get(k).filter(|v| !v.is_null()).or_else(|| c.get(k).filter(|v| !v.is_null()));
        let u = |k: &str| get(k).and_then(Json::as_u64).map(|v| v as usize);
        let need = |k: &str| u(k).with_context(|| format!("DSpark config has no {k}"));
        let f = |k: &str| get(k).and_then(Json::as_f64);
        let b = |k: &str| get(k).and_then(Json::as_bool);
        let gemma = archs.iter().any(|a| a.starts_with("Gemma4"))
            || c.get("model_type").and_then(Json::as_str).is_some_and(|m| m.starts_with("gemma4"));
        let hidden = need("hidden_size")?;
        let n_heads = need("num_attention_heads")?;
        let n_layers = need("num_hidden_layers")?;
        // The target layers read: speculators name the layer whose input is read, the others
        // transformers' `hidden_states` index minus one (-1: the embeddings).
        let target_layers: Vec<usize> = if let Some(a) = c.get("aux_hidden_state_layer_ids").and_then(Json::as_array) {
            a.iter()
                .map(|v| v.as_u64().map(|x| x as usize).context("bad aux_hidden_state_layer_ids"))
                .collect::<Result<_>>()?
        } else {
            get("target_layer_ids")
                .and_then(Json::as_array)
                .context("DSpark config has no target_layer_ids")?
                .iter()
                .map(|v| {
                    v.as_i64()
                        .filter(|&x| x >= -1)
                        .map(|x| (x + 1) as usize)
                        .context("bad target_layer_ids")
                })
                .collect::<Result<_>>()?
        };
        let markov_type = get("markov_head_type").and_then(Json::as_str).unwrap_or("vanilla");
        if !matches!(markov_type, "vanilla" | "rnn") {
            bail!("DSpark Markov head type {markov_type:?} is not supported");
        }
        let (head_dim, n_kv, k_eq_v) = if gemma && b("attention_k_eq_v") == Some(true) {
            (need("global_head_dim")?, need("num_global_key_value_heads")?, true)
        } else {
            let hd = u("head_dim").unwrap_or(hidden / n_heads.max(1));
            (
                if gemma { u("global_head_dim").unwrap_or(hd) } else { hd },
                need("num_key_value_heads")?,
                false,
            )
        };
        // RoPE: `rope_parameters` (transformers 5; Gemma 4 nests it per attention type) or the
        // older `rope_theta` + `rope_scaling`.
        let mut rp = c.get("rope_parameters").cloned().unwrap_or(Json::Null);
        if let Some(full) = rp.get("full_attention") {
            rp = full.clone();
        }
        if rp.is_null() || rp.get("rope_theta").and_then(Json::as_f64).is_none() {
            let mut m = c
                .get("rope_scaling")
                .filter(|v| v.is_object())
                .cloned()
                .unwrap_or(serde_json::json!({}));
            let theta = c.get("rope_theta").and_then(Json::as_f64).unwrap_or(10_000.0);
            if let Some(o) = m.as_object_mut() {
                o.insert("rope_theta".into(), theta.into());
                if let Some(t) = o.remove("type") {
                    o.entry("rope_type").or_insert(t);
                }
            }
            rp = m;
        }
        let partial = rp
            .get("partial_rotary_factor")
            .or_else(|| c.get("partial_rotary_factor"))
            .and_then(Json::as_f64)
            .unwrap_or(1.0);
        let (rope_dims, proportion) = match rp.get("rope_type").and_then(Json::as_str) {
            Some("proportional") => (head_dim, partial),
            _ => ((head_dim as f64 * partial) as usize, 1.0),
        };
        if let Some(o) = rp.as_object_mut()
            && matches!(o.get("rope_type").and_then(Json::as_str), Some("proportional") | None)
        {
            o.insert("rope_type".into(), "default".into());
        }
        let rope: crate::config::YarnParams = serde_json::from_value(rp).context("DSpark RoPE parameters")?;
        if !matches!(rope.rope_type.as_deref(), Some("default" | "yarn")) {
            bail!("DSpark draft RoPE type {:?} is not supported", rope.rope_type);
        }
        // Sliding windows and causality per layer, as vLLM resolves them: `use_swa` makes every
        // layer sliding, else `layer_types`; sliding layers are causal within the block unless
        // `causal` (or speculators' `sliding_window_non_causal`) says otherwise.
        let layer_types: Vec<String> = c
            .get("layer_types")
            .and_then(Json::as_array)
            .map(|a| a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect())
            .unwrap_or_default();
        let use_swa = dfl.get("use_swa").and_then(Json::as_bool).unwrap_or(false);
        let any_sliding = layer_types.iter().any(|t| t == "sliding_attention");
        let window = dfl
            .get("swa_window_size")
            .and_then(Json::as_u64)
            .or_else(|| c.get("sliding_window").and_then(Json::as_u64))
            .unwrap_or(0) as usize;
        let causal_all = c
            .get("is_causal")
            .and_then(Json::as_bool)
            .or_else(|| dfl.get("causal").and_then(Json::as_bool));
        let mut windows = Vec::with_capacity(n_layers);
        let mut causal = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lt = layer_types.get(l).map(String::as_str);
            let sliding = if layer_types.is_empty() || (use_swa && !any_sliding) {
                use_swa
            } else {
                lt == Some("sliding_attention")
            };
            if sliding && window == 0 {
                bail!("DSpark sliding attention without a window size");
            }
            windows.push(if sliding { window } else { 0 });
            let sl_causal = if speculators {
                !c.get("sliding_window_non_causal").and_then(Json::as_bool).unwrap_or(false)
            } else {
                true
            };
            causal.push(causal_all.unwrap_or(lt == Some("sliding_attention") && sl_causal));
        }
        // DeepSpec / SpecForge / speculators drafts predict from the anchor row; z-lab's DFlash
        // layout (`DFlashDraftModel`) has the mask rows predict their own positions.
        let sample_from_anchor = c
            .get("sample_from_anchor")
            .and_then(Json::as_bool)
            .unwrap_or(!archs.contains(&"DFlashDraftModel"));
        let mask = MaskEmbed::Token(need("mask_token_id")? as u32);
        let cfg = Self {
            hidden,
            n_layers,
            n_heads,
            inter: need("intermediate_size")?,
            eps: f("rms_norm_eps").unwrap_or(1e-6) as f32,
            backbone: Backbone::Gqa(GqaConfig {
                n_kv,
                head_dim,
                k_eq_v,
                v_norm: gemma,
                sandwich: gemma,
                gelu: gemma
                    || matches!(
                        get("hidden_act").or_else(|| get("hidden_activation")).and_then(Json::as_str),
                        Some("gelu" | "gelu_pytorch_tanh")
                    ),
                attn_scale: if gemma { 1.0 } else { (head_dim as f32).powf(-0.5) },
                rope_dims,
                rope_proportion: proportion,
            }),
            windows,
            causal,
            block_size: need("block_size")?,
            sample_from_anchor,
            target_layers,
            softcap: f("final_logit_softcapping").filter(|&c| c > 0.0).map(|c| c as f32),
            rope,
            mask,
            // Speculators embed with the verifier's raw table; the others like the target.
            embed_scale: speculators.then_some(1.0),
        };
        cfg.check()?;
        Ok(cfg)
    }

    fn check(&self) -> Result<()> {
        let rot_ok = match &self.backbone {
            Backbone::Mla { dr, .. } => dr.is_multiple_of(2),
            Backbone::Gqa(g) => {
                g.rope_dims.is_multiple_of(2) && g.rope_dims <= g.head_dim && g.n_kv > 0 && self.n_heads.is_multiple_of(g.n_kv)
            }
            Backbone::Ds4(_) => true,
        };
        if self.block_size < 2 || self.target_layers.is_empty() || !rot_ok {
            bail!("bad DSpark metadata: {self:?}");
        }
        Ok(())
    }

    /// The most tokens one pass drafts.
    pub fn max_draft(&self) -> usize {
        if self.sample_from_anchor {
            self.block_size
        } else {
            self.block_size - 1
        }
    }

    /// Rows of one pass: the anchor and the mask rows.
    fn block_rows(&self) -> usize {
        self.max_draft() + usize::from(!self.sample_from_anchor)
    }
}

/// One dense MLA draft layer (llama.cpp `graph_mistral_large4`): absorbed latent attention, the
/// latent doubling as the value, then a SwiGLU MLP.
struct MlaLayer<'a> {
    attn_norm: Vec<f32>,
    q: Lin<'a>,    // [nh * (dn + dr), H]
    kv_a: Lin<'a>, // [lat + dr, H]
    kv_norm: Vec<f32>,
    /// Per head `[lat, dn]` (query onto the latent) and `[vd, lat]` (attended latent onto value).
    k_b: Vec<Lin<'a>>,
    v_b: Vec<Lin<'a>>,
    o: Lin<'a>, // [H, nh * vd]
}

/// One grouped-query attention draft layer (Qwen3 / Gemma 4).
struct GqaLayer<'a> {
    attn_norm: Vec<f32>,
    q: Lin<'a>,
    k: Lin<'a>,
    v: Option<Lin<'a>>,
    o: Lin<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn: Option<Vec<f32>>,
    post_ffn: Option<Vec<f32>>,
    scalar: f32,
}

#[allow(clippy::large_enum_variant)] // a handful of layers
enum Attn<'a> {
    Mla(MlaLayer<'a>),
    Gqa(GqaLayer<'a>),
}

struct Layer<'a> {
    attn: Attn<'a>,
    ffn_norm: Vec<f32>,
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
}

/// The Markov head: `markov_w1` `[V, R]` embeds the previous token, `markov_w2` `[V, R]` maps the
/// (tanh'd) state onto a logit bias; the rnn variant's joint projection `[3R, 2R + H]`.
struct Markov<'a> {
    w1: Table<'a>,
    w2: Lin<'a>,
    joint: Option<(Lin<'a>, Vec<f32>)>,
    rank: usize,
}

/// The confidence head: weights over `[hidden]` or `[hidden, markov_w1[prev]]`, and a bias.
struct Confidence {
    w: Vec<f32>,
    b: f32,
}

/// A loaded DSpark draft (weights borrowed from its files).
pub struct Dspark<'a> {
    pub cfg: DsparkConfig,
    /// The backbone: dense MLA or grouped-query attention layers, or DeepSeek V4 stages.
    dense: Option<Dense<'a>>,
    ds4: Option<crate::deepseek4::Ds4Draft<'a>>,
    markov: Option<Markov<'a>>,
    conf: Option<Confidence>,
    /// The draft's own embeddings / LM head, when it needs them (else the target's are used).
    embed: Option<Table<'a>>,
    head: Option<Lin<'a>>,
    target_hidden: usize,
    /// Run the matmuls of the draft's few rows on the accelerator (see [`Dspark::set_device`]).
    device: std::sync::atomic::AtomicBool,
    /// The GPU plans, once [`Dspark::init_gpu`] ran (None inside: the draft runs on the CPU).
    gpu: std::sync::OnceLock<Option<GpuDraft>>,
    /// The same on the fast (batched, not batch-invariant) kernels, for sessions with
    /// [`DsparkOptions::fast_verify`].
    gpu_fast: std::sync::OnceLock<Option<GpuDraft>>,
}

/// The dense backbones: MLA (llama.cpp `graph_mistral_large4`) or grouped-query attention
/// layers behind the context projection.
struct Dense<'a> {
    fc: Lin<'a>, // [H, n_feat * target hidden]
    enc_norm: Vec<f32>,
    out_norm: Vec<f32>,
    mask: Option<Table<'a>>, // [block_size - 1, H] (MaskEmbed::Learned)
    layers: Vec<Layer<'a>>,
    inv_freq: Vec<f32>,
    rope_mag: f32,
}

/// Positions the draft's context cache keeps beyond the window it attends to: a truncation by up
/// to this many positions leaves the cache exact (the dropped positions come back as the target
/// runs them again). The cache holds between `window + SLACK` and `window + 2 * SLACK`
/// positions once it is full; the oldest are dropped `SLACK` at a time.
pub const SLACK: usize = 256;

/// The draft's context cache: per layer and position, MLA's normed latent `[lat]` and rotated
/// rotary key `[dr]`, GQA's normed, rotated keys and (normed) values `[n_kv * hd]` each, or
/// DeepSeek V4's window keys `[head_dim]`, for positions `start..len` of what the target has
/// processed. When every layer attends to the last `window` positions only, the older ones are
/// dropped (see [`SLACK`]); otherwise all are kept. With the GPU draft plans, GQA keys and values
/// live in `kv` instead, all kept (the device copies index positions from 0).
#[derive(Clone, Default)]
pub struct DsparkSession {
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    wa: usize,
    wb: usize,
    /// Positions kept beyond `slack` (the draft's window; 0: all), and the slack.
    keep: usize,
    slack: usize,
    /// The first cached position.
    start: usize,
    /// GQA with the GPU plans: per layer keys and values.
    kv: Vec<LayerCache>,
    /// Device state of the GPU context and block plans (KV cache copies).
    gpu_ctx: crate::gpu::decode::Slot,
    gpu_block: crate::gpu::decode::Slot,
    /// Run the GPU passes on the fast plans ([`DsparkOptions::fast_verify`]).
    pub fast: bool,
    pub len: usize,
}

impl DsparkSession {
    /// Drops cached positions at or beyond `len`. Going back more than [`SLACK`] positions may
    /// leave less than the window before `len` cached (positions dropped earlier do not come
    /// back): the block then attends to what is left until new positions fill the window, which
    /// changes the drafts but not the output (the target verifies them).
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            let n = len.saturating_sub(self.start);
            for c in &mut self.a {
                c.truncate(n * self.wa);
            }
            for r in &mut self.b {
                r.truncate(n * self.wb);
            }
            for c in &mut self.kv {
                c.truncate(len);
            }
            self.start = self.start.min(len);
            self.len = len;
        }
    }

    /// The first position still cached (positions before it were dropped).
    pub fn start(&self) -> usize {
        self.start
    }

    /// Bytes of the cached context.
    pub fn bytes(&self) -> usize {
        let kv: usize = self.kv.iter().map(|c| c.len * c.n_kv * c.hd * 8).sum();
        self.a.iter().chain(&self.b).map(|v| v.len() * 4).sum::<usize>() + kv
    }

    /// Appends one layer's entries for new positions (the caller then advances `len`).
    fn push(&mut self, layer: usize, a: &[f32], b: &[f32]) {
        self.a[layer].extend_from_slice(a);
        self.b[layer].extend_from_slice(b);
    }

    /// Drops the oldest positions once more than `keep + 2 * slack` are cached, down to
    /// `keep + slack` (moving the rest down once per `slack` positions).
    fn trim(&mut self) {
        if self.keep == 0 || self.len - self.start < self.keep + 2 * self.slack {
            return;
        }
        let drop = self.len - self.start - (self.keep + self.slack);
        for c in &mut self.a {
            c.drain(..drop * self.wa);
        }
        for r in &mut self.b {
            r.drain(..drop * self.wb);
        }
        self.start += drop;
    }

    /// Cached entries of `layer` for positions `lo..hi` (within `start..len`).
    fn range(&self, layer: usize, lo: usize, hi: usize) -> (&[f32], &[f32]) {
        if let Some(c) = self.kv.get(layer) {
            let (k, v) = c.data();
            let w = c.n_kv * c.hd;
            return (&k[lo * w..hi * w], &v[lo * w..hi * w]);
        }
        let (a, b) = (lo - self.start, hi - self.start);
        (&self.a[layer][a * self.wa..b * self.wa], &self.b[layer][a * self.wb..b * self.wb])
    }
}

/// The draft's GPU plans (multi-token decode plans, see `crate::gpu::decode`): the context
/// (features to keys and values) and the block (to logits).
struct GpuDraft {
    ctx: std::sync::Arc<dyn crate::gpu::decode::DecodePlan>,
    block: std::sync::Arc<dyn crate::gpu::decode::DecodePlan>,
}

/// Whether a model uses a DSpark draft found beside it unless told otherwise (CLI
/// `--no-draft`, engine `draft=none`); which drafts that is, is [`default_draft`]'s choice.
pub const BY_DEFAULT: bool = true;

/// Whether the draft runs on the accelerator when one is active unless told otherwise
/// ([`DsparkOptions::device`]): measured faster on an M5 Max (see the README).
pub const DEVICE_BY_DEFAULT: bool = true;

/// The default `p_min` ([`DsparkOptions::p_min`]) when the target streams its weights. There a
/// verification pass reads the routed experts of every token it holds (experts shared by
/// consecutive tokens were page-cache hits anyway), so a rejected draft costs about one token's
/// reads: only drafts that are nearly always accepted pay. Measured on Mistral Large 4: drafts
/// the draft model gives at least 0.95 are accepted 98% of the time, 0.7-0.95 75%, 0.3-0.7 40%.
pub const STREAM_P_MIN: f32 = 0.85;

/// Default drafts per round with a DeepSeek V4 draft (its block holds 5): see the README.
pub const DS4_DRAFT_TOKENS: usize = 3;

/// Whether a [`KNOWN`] draft is used when none is asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByDefault {
    /// Only when asked for (`draft=auto`, `--draft auto`).
    No,
    /// Always.
    Yes,
    /// With fast verification ([`DsparkOptions::fast_verify`], the default) only: with exact
    /// verification it measured slower than plain decoding.
    Fast,
}

impl ByDefault {
    /// Whether the draft is a default with fast verification on (`fast`) or off.
    pub fn applies(self, fast: bool) -> bool {
        match self {
            ByDefault::No => false,
            ByDefault::Yes => true,
            ByDefault::Fast => fast,
        }
    }
}

/// Published DSpark drafts and their targets (Hugging Face repo ids), and whether that pair is
/// used by default: where it measured faster than plain decoding on average over code, prose,
/// chat and math prompts (Metal, M5 Max; the README has the numbers). The Q4 targets (Qwen3.8-27B,
/// Gemma 4 31B) are faster only with fast verification; Qwen3.5-35B-A3B (3B active: a plain step
/// takes 8 ms) and the pure-Q4_0 Gemma 4 draft were not faster.
/// A draft is found in the download cache (`org--name` folders beside the target's), or fetched
/// by `draft=auto` (CLI `--draft auto`) when it is not there; GGUF draft repositories (llama.cpp
/// conversions) are found but not fetched.
pub const KNOWN: &[(&str, &str, ByDefault)] = &[
    ("Qwen/Qwen3-4B", "deepseek-ai/dspark_qwen3_4b_block7", ByDefault::Yes),
    ("Qwen/Qwen3-8B", "deepseek-ai/dspark_qwen3_8b_block7", ByDefault::Yes),
    ("Qwen/Qwen3-8B", "ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF", ByDefault::Yes),
    ("Qwen/Qwen3-14B", "deepseek-ai/dspark_qwen3_14b_block7", ByDefault::Yes),
    ("Qwen/Qwen3-14B", "Anbeeld/Qwen3-14B-DSpark-GGUF", ByDefault::Yes),
    ("Qwen/Qwen3.8-27B", "RadixArk/Qwen3.8-27B-DSpark", ByDefault::Fast),
    ("Qwen/Qwen3.8-27B", "jialinyyzz/Qwen3.8-27B-DSpark-drafter", ByDefault::No),
    ("Qwen/Qwen3.5-35B-A3B", "jinzex/Qwen3.5-35B-A3B-DSpark", ByDefault::No),
    ("google/gemma-4-12B-it", "deepseek-ai/dspark_gemma4_12b_block7", ByDefault::Yes),
    ("google/gemma-4-12B-it", "ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF", ByDefault::No),
    (
        "google/gemma-4-31B-it",
        "RedHatAI/gemma-4-31B-it-speculator.dspark",
        ByDefault::Fast,
    ),
    ("google/gemma-4-31B-it", "Hikari07jp/DSpark-Gemma-4-31B-draft", ByDefault::No),
];

/// Whether a `-`-separated part of a GGUF file name is its quantization (`Q8_0`, `IQ4_XS`,
/// `BF16`, unsloth's `UD-...`).
fn is_quant_tag(p: &str) -> bool {
    let u = p.to_ascii_uppercase();
    let digit_at = |i: usize| u.as_bytes().get(i).is_some_and(u8::is_ascii_digit);
    matches!(u.as_str(), "UD" | "BF16" | "F16" | "F32") || (u.starts_with("IQ") && digit_at(2)) || (u.starts_with('Q') && digit_at(1))
}

/// The names (without the organization) the model at `model` may have on Hugging Face: its
/// folder's (`org--name` in the download cache, a converted copy's `name-oxinfer-q4`, a GGUF
/// repository's `name-GGUF`) and, for a GGUF file, its file name up to the quantization.
pub(crate) fn model_names(model: &Path) -> Vec<String> {
    let gguf = crate::gguf::is_gguf(model);
    let mut v = Vec::new();
    let folder = if gguf { model.parent() } else { Some(model) };
    if let Some(n) = folder.and_then(Path::file_name).and_then(|n| n.to_str()) {
        let n = n.rsplit("--").next().unwrap_or(n);
        let n = ["-oxinfer-q4", "-oxinfer-q8", "-GGUF", "-gguf"]
            .iter()
            .fold(n, |n, s| n.strip_suffix(s).unwrap_or(n));
        v.push(n.to_string());
    }
    if gguf && let Some(stem) = model.file_stem().and_then(|s| s.to_str()) {
        let parts: Vec<&str> = stem.split('-').collect();
        if let Some(i) = parts.iter().position(|p| is_quant_tag(p)).filter(|&i| i > 0) {
            v.push(parts[..i].join("-"));
        }
    }
    v
}

/// The [`KNOWN`] drafts of the model at `model` (repo id, used by default with fast
/// verification, the default), the default ones first.
pub fn known_drafts(model: &Path) -> Vec<(&'static str, bool)> {
    known_drafts_by(model).into_iter().map(|(d, b)| (d, b.applies(true))).collect()
}

/// [`known_drafts`] with when each is a default.
fn known_drafts_by(model: &Path) -> Vec<(&'static str, ByDefault)> {
    let names = model_names(model);
    let mut v: Vec<(&'static str, ByDefault)> = KNOWN
        .iter()
        .filter(|(t, _, _)| {
            t.rsplit('/')
                .next()
                .is_some_and(|t| names.iter().any(|n| n.eq_ignore_ascii_case(t)))
        })
        .map(|&(_, d, def)| (d, def))
        .collect();
    v.sort_by_key(|d| d.1 == ByDefault::No);
    v
}

/// The draft a folder holds: the folder itself (a Hugging Face draft), or its DFlash / DSpark
/// GGUF (the `Q8_0` one when a repository has several quantizations). A draft file is itself.
pub fn draft_in(path: &Path) -> Option<PathBuf> {
    if crate::gguf::is_gguf(path) || hf_config(path).is_some() {
        return is_draft(path).then(|| path.to_path_buf());
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(path)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| crate::gguf::is_gguf(p) && is_draft(p))
        .collect();
    v.sort();
    let q8 = v.iter().position(|p| p.to_string_lossy().to_ascii_uppercase().contains("Q8_0"));
    match q8 {
        Some(i) => Some(v.swap_remove(i)),
        None => v.into_iter().next(),
    }
}

/// The DSpark draft beside model `model`, and whether it is used by default:
///
/// - a `dspark*.gguf` file in a GGUF model's folder or the folder above (as unsloth lays them
///   out), or a `dspark*` draft folder or GGUF inside a model folder: by default for the Mistral
///   Large 4 and DeepSeek V4 drafts, and for a model with a default [`KNOWN`] draft;
/// - else a [`KNOWN`] draft of this model in the download cache (the folder holding the model's
///   folder), default or not as the table says (`fast`: with fast verification).
fn find_draft(model: &Path, fast: bool) -> Option<(PathBuf, bool)> {
    let known: Vec<(&str, bool)> = known_drafts_by(model).into_iter().map(|(d, b)| (d, b.applies(fast))).collect();
    let pick = |d: &Path| -> Option<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(d)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
                n.starts_with("dspark") && (n.ends_with(".gguf") || p.is_dir()) && is_draft(p)
            })
            .collect();
        v.sort();
        if v.len() == 1 { v.pop() } else { None }
    };
    let beside_default = |p: &Path| load_config(p).is_ok_and(|c| !matches!(c.backbone, Backbone::Gqa(_))) || known.iter().any(|d| d.1);
    let gguf = crate::gguf::is_gguf(model);
    let folder = if gguf {
        model.parent()?
    } else if model.is_dir() {
        model
    } else {
        return None;
    };
    let beside = if gguf {
        vec![Some(folder), folder.parent()]
    } else {
        vec![Some(folder)]
    };
    for d in beside.into_iter().flatten() {
        if let Some(p) = pick(d) {
            let def = beside_default(&p);
            return Some((p, def));
        }
    }
    let cache = folder.parent()?;
    known
        .iter()
        .find_map(|&(d, def)| draft_in(&cache.join(d.replace('/', "--"))).map(|p| (p, def)))
}

/// The DSpark draft beside `model` (see [`find_draft`]), if any: what `draft=auto` uses before
/// fetching one ([`auto_repo`]).
pub fn draft_for(model: &Path) -> Option<PathBuf> {
    find_draft(model, true).map(|d| d.0)
}

/// The draft used when none is asked for, with fast verification (the default): one beside the
/// model or in the download cache that measured faster ([`KNOWN`]; nothing is fetched).
pub fn default_draft(model: &Path) -> Option<PathBuf> {
    default_draft_for(model, true)
}

/// [`default_draft`] for fast (`fast`) or exact verification ([`DsparkOptions::fast_verify`]).
pub fn default_draft_for(model: &Path, fast: bool) -> Option<PathBuf> {
    find_draft(model, fast).filter(|d| d.1).map(|d| d.0)
}

/// The repository `draft=auto` fetches when no draft is beside `model`: its default [`KNOWN`]
/// draft, else its first known Hugging Face (safetensors) draft.
pub fn auto_repo(model: &Path) -> Option<&'static str> {
    known_drafts(model)
        .into_iter()
        .map(|d| d.0)
        .find(|d| !d.to_ascii_uppercase().ends_with("GGUF"))
}

/// The `config.json` of a Hugging Face DSpark / DFlash draft folder, if `path` is one.
fn hf_config(path: &Path) -> Option<Json> {
    if !path.is_dir() {
        return None;
    }
    let raw: Json = serde_json::from_slice(&std::fs::read(path.join("config.json")).ok()?).ok()?;
    let archs = raw.get("architectures").and_then(Json::as_array)?;
    let draft = archs
        .iter()
        .filter_map(Json::as_str)
        .any(|a| a.contains("DSpark") || a.contains("DFlash"))
        || raw.get("speculators_model_type").and_then(Json::as_str) == Some("dspark");
    draft.then_some(raw)
}

/// Whether `path` is a DSpark draft: a DFlash / DSpark GGUF, or a Hugging Face DSpark (or DFlash
/// with a Markov head) folder.
pub fn is_draft(path: &Path) -> bool {
    if crate::gguf::is_gguf(path) {
        return crate::gguf::header(path).is_ok_and(|h| matches!(h.arch(), "dflash" | "dspark"));
    }
    hf_config(path).is_some()
}

/// The draft's config: a GGUF header or a Hugging Face `config.json`.
pub fn load_config(path: &Path) -> Result<DsparkConfig> {
    if crate::gguf::is_gguf(path) {
        let h = crate::gguf::header(path)?;
        return DsparkConfig::from_header(&h);
    }
    let raw = hf_config(path).with_context(|| format!("{}: not a DSpark draft", path.display()))?;
    DsparkConfig::from_hf(&raw)
}

/// Tensor names of a llama.cpp GGUF or a Hugging Face checkpoint (with or without `model.`).
struct Names {
    gguf: bool,
    prefix: &'static str,
}

impl Names {
    fn top(&self, gguf: &str, hf: &str) -> String {
        if self.gguf {
            gguf.to_string()
        } else {
            format!("{}{hf}", self.prefix)
        }
    }
    fn layer(&self, l: usize, gguf: &str, hf: &str) -> String {
        if self.gguf {
            format!("blk.{l}.{gguf}")
        } else {
            format!("{}layers.{l}.{hf}", self.prefix)
        }
    }
}

impl<'a> Dspark<'a> {
    /// Loads the draft in `files` (opened from `path`: a `dflash` GGUF or a Hugging Face draft
    /// folder) for a target whose hidden size is `target_hidden` and vocabulary `vocab`.
    /// Unquantized weights become `quant` (the engine and the CLI pass the target's weight
    /// format: a Q4 draft of Qwen3.8-27B at Q4 accepted as many drafts as the bf16 one and ran
    /// its passes in 23-27 ms instead of 31-34).
    pub fn load(path: &Path, files: &'a ModelFiles, target_hidden: usize, vocab: usize, quant: Quant) -> Result<Self> {
        let gguf = crate::gguf::is_gguf(path);
        let cfg = load_config(path)?;
        let has = |n: &str| files.has(n);
        if has("d2t") || has("model.d2t") || files.names().any(|n| n.starts_with("selector_") || n.starts_with("lilicorr")) {
            bail!("this DFlash draft needs a reduced vocabulary, selector or LiLiCorr head (not supported)");
        }
        let nm = Names {
            gguf,
            prefix: if !gguf && has("model.fc.weight") { "model." } else { "" },
        };
        let c = &cfg;
        let (h, nh) = (c.hidden, c.n_heads);
        let shape = |n: &str, s: &[usize]| -> Result<()> {
            let info = files.info(n).with_context(|| format!("DSpark draft: missing tensor {n}"))?;
            if info.shape != s {
                bail!("DSpark draft: tensor {n} has shape {:?}, expected {s:?}", info.shape);
            }
            Ok(())
        };
        let lin = |n: &str, n_out: usize, n_in: usize| -> Result<Lin<'a>> {
            shape(n, &[n_out, n_in])?;
            load_lin(files, n, n_out, n_in, quant)
        };
        let vecf = |n: &str, len: usize| -> Result<Vec<f32>> {
            shape(n, &[len])?;
            files.f32(n)
        };
        // Per-head matrices `[nh, n_out, n_in]`, one Lin each (GGUF bytes split, or bf16 rows).
        let heads = |n: &str, n_out: usize, n_in: usize| -> Result<Vec<Lin<'a>>> {
            shape(n, &[nh, n_out, n_in])?;
            match files.ggml(n) {
                Some(g) => {
                    let (kind, bytes) = g?;
                    bytes
                        .chunks(bytes.len() / nh)
                        .map(|b| Lin::from_gguf(kind, b, n_out, n_in))
                        .collect()
                }
                None => Ok(files.bf16(n)?.chunks(n_out * n_in).map(Lin::bf16).collect()),
            }
        };
        let n_feat = c.target_layers.len() * target_hidden;
        // DeepSeek V4 stages load in `crate::deepseek4`.
        let n_dense = if matches!(c.backbone, Backbone::Ds4(_)) { 0 } else { c.n_layers };
        let mut layers = Vec::with_capacity(n_dense);
        for l in 0..n_dense {
            let ln = |g: &str, hf: &str| nm.layer(l, g, hf);
            let attn = match &c.backbone {
                Backbone::Mla { lat, dn, dr, vd } => {
                    if !gguf {
                        bail!("MLA DSpark drafts load from GGUF only");
                    }
                    let p = format!("blk.{l}.");
                    Attn::Mla(MlaLayer {
                        attn_norm: vecf(&format!("{p}attn_norm.weight"), h)?,
                        q: lin(&format!("{p}attn_q.weight"), nh * (dn + dr), h)?,
                        kv_a: lin(&format!("{p}attn_kv_a_mqa.weight"), lat + dr, h)?,
                        kv_norm: vecf(&format!("{p}attn_kv_a_norm.weight"), *lat)?,
                        k_b: heads(&format!("{p}attn_k_b.weight"), *lat, *dn)?,
                        v_b: heads(&format!("{p}attn_v_b.weight"), *vd, *lat)?,
                        o: lin(&format!("{p}attn_output.weight"), h, nh * vd)?,
                    })
                }
                Backbone::Gqa(g) => {
                    let (hd, kvw) = (g.head_dim, g.n_kv * g.head_dim);
                    // Gemma's `post_attention_layernorm` follows the attention; Qwen's is the
                    // feed-forward input norm.
                    let opt_vec =
                        |n: String, len: usize| -> Result<Option<Vec<f32>>> { if has(&n) { vecf(&n, len).map(Some) } else { Ok(None) } };
                    let (post_attn, post_ffn) = if g.sandwich {
                        (
                            Some(vecf(&ln("post_attention_norm.weight", "post_attention_layernorm.weight"), h)?),
                            Some(vecf(&ln("post_ffw_norm.weight", "post_feedforward_layernorm.weight"), h)?),
                        )
                    } else {
                        (None, None)
                    };
                    let scalar = {
                        let n = ln("layer_output_scale.weight", "layer_scalar");
                        match opt_vec(n, 1)? {
                            Some(v) => v[0],
                            None => 1.0,
                        }
                    };
                    Attn::Gqa(GqaLayer {
                        attn_norm: vecf(&ln("attn_norm.weight", "input_layernorm.weight"), h)?,
                        q: lin(&ln("attn_q.weight", "self_attn.q_proj.weight"), nh * hd, h)?,
                        k: lin(&ln("attn_k.weight", "self_attn.k_proj.weight"), kvw, h)?,
                        v: if g.k_eq_v {
                            None
                        } else {
                            Some(lin(&ln("attn_v.weight", "self_attn.v_proj.weight"), kvw, h)?)
                        },
                        o: lin(&ln("attn_output.weight", "self_attn.o_proj.weight"), h, nh * hd)?,
                        q_norm: vecf(&ln("attn_q_norm.weight", "self_attn.q_norm.weight"), hd)?,
                        k_norm: vecf(&ln("attn_k_norm.weight", "self_attn.k_norm.weight"), hd)?,
                        post_attn,
                        post_ffn,
                        scalar,
                    })
                }
                Backbone::Ds4(_) => unreachable!(),
            };
            let sandwich = matches!(&c.backbone, Backbone::Gqa(g) if g.sandwich);
            let ffn_norm = if sandwich {
                ln("ffn_norm.weight", "pre_feedforward_layernorm.weight")
            } else {
                // llama.cpp's DSpark branch keeps Qwen's name for it.
                let n = ln("ffn_norm.weight", "post_attention_layernorm.weight");
                if gguf && !has(&n) {
                    ln("post_attention_norm.weight", "")
                } else {
                    n
                }
            };
            layers.push(Layer {
                attn,
                ffn_norm: vecf(&ffn_norm, h)?,
                gate: lin(&ln("ffn_gate.weight", "mlp.gate_proj.weight"), c.inter, h)?,
                up: lin(&ln("ffn_up.weight", "mlp.up_proj.weight"), c.inter, h)?,
                down: lin(&ln("ffn_down.weight", "mlp.down_proj.weight"), h, c.inter)?,
            });
        }
        // llama.cpp's DSpark branch: `markov.w1`, `confidence.proj` (mainline: `markov_w1`,
        // `conf_proj`).
        let alt = |n: String| if gguf && !has(&n) { n.replacen('_', ".", 1) } else { n };
        let (w1n, w2n) = (
            alt(nm.top("markov_w1.weight", "markov_head.markov_w1.weight")),
            alt(nm.top("markov_w2.weight", "markov_head.markov_w2.weight")),
        );
        let markov = if has(&w1n) {
            let rank = files.info(&w1n).unwrap().shape.last().copied().unwrap_or(0);
            shape(&w1n, &[vocab, rank])?;
            let jn = nm.top("joint_proj.weight", "markov_head.joint_proj.weight");
            let joint = if has(&jn) {
                Some((
                    lin(&jn, 3 * rank, 2 * rank + h)?,
                    vecf(&nm.top("joint_proj.bias", "markov_head.joint_proj.bias"), 3 * rank)?,
                ))
            } else {
                None
            };
            Some(Markov {
                w1: Table::load(files, &w1n, vocab, rank, quant)?,
                w2: lin(&w2n, vocab, rank)?,
                joint,
                rank,
            })
        } else {
            None
        };
        let conf_alt = |n: String| {
            if gguf && !has(&n) {
                n.replacen("conf_proj", "confidence.proj", 1)
            } else {
                n
            }
        };
        let cn = conf_alt(nm.top("conf_proj.weight", "confidence_head.proj.weight"));
        let conf = if has(&cn) {
            let n_in = files.info(&cn).unwrap().shape.last().copied().unwrap_or(0);
            let rank = markov.as_ref().map_or(0, |m| m.rank);
            if n_in != h && n_in != h + rank {
                bail!("DSpark confidence head of width {n_in} (hidden {h}, Markov rank {rank})");
            }
            if files.info(&cn).unwrap().shape.len() > 1 {
                shape(&cn, &[1, n_in])?;
            }
            let bn = conf_alt(nm.top("conf_proj.bias", "confidence_head.proj.bias"));
            Some(Confidence {
                w: files.f32(&cn)?,
                b: if has(&bn) { files.f32(&bn)?[0] } else { 0.0 },
            })
        } else {
            None
        };
        let mask = if c.mask == MaskEmbed::Learned {
            let n = nm.top("mask_embd.weight", "mask_embedding.weight");
            shape(&n, &[c.block_size - 1, h])?;
            Some(Table::load(files, &n, c.block_size - 1, h, quant)?)
        } else {
            None
        };
        // A draft as wide as its target shares the target's embeddings and LM head (DeepSpec
        // drafts ship frozen copies of them, which GGUF conversions may quantize further: a Q4_0
        // copy of Gemma 4's head moved the draft logits by 18%); a narrower one needs its own.
        // The MLA and DeepSeek V4 GGUF drafts use theirs, as llama.cpp does.
        let own = h != target_hidden
            || (gguf && !matches!(c.backbone, Backbone::Gqa(_)))
            || std::env::var_os("OXINFER_DSPARK_OWN_HEAD").is_some();
        let en = nm.top("token_embd.weight", "embed_tokens.weight");
        let embed = if has(&en) && own {
            shape(&en, &[vocab, h])?;
            Some(Table::load(files, &en, vocab, h, quant)?)
        } else {
            if h != target_hidden {
                bail!("the DSpark draft (hidden {h}) has no embeddings of its own for a target of hidden size {target_hidden}");
            }
            None
        };
        let hn = nm.top("output.weight", "lm_head.weight");
        let head = if has(&hn) && own {
            Some(lin(&hn, vocab, h)?)
        } else {
            if h != target_hidden {
                bail!("the DSpark draft (hidden {h}) has no LM head of its own for a target of hidden size {target_hidden}");
            }
            None
        };
        if let Backbone::Ds4(ds) = &c.backbone {
            let ds4 = crate::deepseek4::Ds4Draft::load(files, (**ds).clone(), quant, n_feat)?;
            return Ok(Self {
                dense: None,
                ds4: Some(ds4),
                markov,
                conf,
                embed,
                head,
                target_hidden,
                device: std::sync::atomic::AtomicBool::new(false),
                gpu: Default::default(),
                gpu_fast: Default::default(),
                cfg,
            });
        }
        let (mut inv_freq, rope_mag) = match &c.backbone {
            Backbone::Mla { dr, .. } => cfg.rope.inv_freq(*dr),
            Backbone::Gqa(g) if g.rope_proportion < 1.0 => (rope_inv_freq(g.rope_dims, cfg.rope.rope_theta, g.rope_proportion), 1.0),
            Backbone::Gqa(g) => cfg.rope.inv_freq(g.rope_dims),
            Backbone::Ds4(_) => unreachable!(),
        };
        // llama.cpp's `rope_freqs` divide the frequencies (proportional RoPE: huge factors for
        // the pairs left unrotated).
        if gguf && has("rope_freqs.weight") {
            let ff = files.f32("rope_freqs.weight")?;
            if ff.len() != inv_freq.len() {
                bail!("DSpark draft: rope_freqs has {} entries, expected {}", ff.len(), inv_freq.len());
            }
            inv_freq = inv_freq.iter().zip(&ff).map(|(b, f)| if *f > 1e6 { 0.0 } else { b / f }).collect();
        }
        let dense = Dense {
            fc: lin(&nm.top("fc.weight", "fc.weight"), h, n_feat)?,
            enc_norm: vecf(&nm.top("enc.output_norm.weight", "hidden_norm.weight"), h)?,
            out_norm: vecf(&nm.top("output_norm.weight", "norm.weight"), h)?,
            mask,
            layers,
            inv_freq,
            rope_mag,
        };
        Ok(Self {
            dense: Some(dense),
            ds4: None,
            markov,
            conf,
            embed,
            head,
            target_hidden,
            device: std::sync::atomic::AtomicBool::new(false),
            gpu: Default::default(),
            gpu_fast: Default::default(),
            cfg,
        })
    }

    /// Runs the matmuls of a draft pass and of the injection of a round's committed tokens (the
    /// weights, the target's LM head) on the active accelerator's matrix-vector kernels
    /// (`Backend::matvec_rows`) instead of the CPU's; the attention and the small per-head
    /// matrices stay on the CPU. Without an accelerator that takes them, the CPU runs them.
    /// The drafts may differ in the last bits (the target's verification keeps the output
    /// exact). Off by default; [`DsparkDecoder::new`] sets it from [`DsparkOptions::device`].
    pub fn set_device(&self, on: bool) {
        self.device.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the draft runs its matmuls on the accelerator (see [`Dspark::set_device`]).
    pub fn device(&self) -> bool {
        self.device.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `linear_multi` of the draft: for a few rows on the accelerator's matrix-vector kernels
    /// when the draft runs there, else (or when they decline) as usual.
    fn lin(&self, jobs: &mut [(&Lin, usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
        if t <= 4 && self.device() {
            let mut dj: Vec<(&Lin, Option<&Lin>, usize, &mut [f32])> = jobs.iter_mut().map(|(l, n, y)| (*l, None, *n, &mut **y)).collect();
            if linear_rows_device(&mut dj, n_in, x, t) {
                return;
            }
        }
        linear_multi(jobs, n_in, x, t);
    }

    /// The MLP's `act(gate x) * up x` (`[t, inter]`, SiLU or GELU), as [`Dspark::lin`].
    fn glu(&self, l: &Layer, x: &[f32], t: usize) -> Vec<f32> {
        let (h, inter) = (self.cfg.hidden, self.cfg.inter);
        let gelu = matches!(&self.cfg.backbone, Backbone::Gqa(g) if g.gelu);
        let mut g = vec![0f32; t * inter];
        if !gelu && t <= 4 && self.device() && linear_rows_device(&mut [(&l.gate, Some(&l.up), inter, &mut g[..])], h, x, t) {
            return g;
        }
        let mut u = vec![0f32; t * inter];
        self.lin(&mut [(&l.gate, inter, &mut g[..]), (&l.up, inter, &mut u[..])], h, x, t);
        if gelu {
            gelu_mul(&mut g, &u);
        } else {
            silu_mul(&mut g, &u);
        }
        g
    }

    fn dense(&self) -> &Dense<'a> {
        self.dense.as_ref().expect("a dense DSpark backbone")
    }

    pub fn new_session(&self) -> DsparkSession {
        let gpu = matches!(self.gpu.get(), Some(Some(_)));
        let (n, wa, wb, keep) = match (&self.cfg.backbone, &self.ds4) {
            // DeepSeek V4: per stage its window keys, `head_dim` per position, indexed from
            // position 0 (no trimming).
            (_, Some(d)) => (d.n_stages(), d.cfg.head_dim, 0, 0),
            (Backbone::Mla { lat, dr, .. }, _) => (self.cfg.n_layers, *lat, *dr, 0),
            (Backbone::Gqa(g), _) => (self.cfg.n_layers, g.n_kv * g.head_dim, g.n_kv * g.head_dim, 0),
            (Backbone::Ds4(_), None) => unreachable!(),
        };
        // Bounded when every layer has a window (the GPU plans keep every position).
        let keep = match self.cfg.windows.iter().min() {
            Some(&w) if w > 0 && self.ds4.is_none() && !gpu => self.cfg.windows.iter().copied().max().unwrap_or(0),
            _ => keep,
        };
        let kv = match &self.cfg.backbone {
            Backbone::Gqa(g) if gpu => (0..n).map(|_| LayerCache::new(g.n_kv, g.head_dim)).collect(),
            _ => Vec::new(),
        };
        DsparkSession {
            a: vec![Vec::new(); n],
            b: vec![Vec::new(); n],
            wa,
            wb,
            keep,
            slack: SLACK,
            start: 0,
            kv,
            gpu_ctx: Default::default(),
            gpu_block: Default::default(),
            fast: false,
            len: 0,
        }
    }

    /// Weight bytes one draft pass reads (without the target's LM head).
    pub fn pass_bytes(&self) -> usize {
        let markov = self
            .markov
            .as_ref()
            .map_or(0, |m| m.w2.bytes() + m.joint.as_ref().map_or(0, |j| j.0.bytes()));
        if let Some(d) = &self.ds4 {
            return d.pass_bytes(self.cfg.block_size) + markov;
        }
        let m = self.dense();
        let mut n = m.fc.bytes();
        for l in &m.layers {
            n += [&l.gate, &l.up, &l.down].iter().map(|w| w.bytes()).sum::<usize>();
            n += match &l.attn {
                Attn::Mla(a) => {
                    [&a.q, &a.kv_a, &a.o].iter().map(|w| w.bytes()).sum::<usize>()
                        + a.k_b.iter().chain(&a.v_b).map(|w| w.bytes()).sum::<usize>()
                }
                Attn::Gqa(a) => [Some(&a.q), Some(&a.k), a.v.as_ref(), Some(&a.o)]
                    .iter()
                    .flatten()
                    .map(|w| w.bytes())
                    .sum(),
            };
        }
        n + markov + self.head.as_ref().map_or(0, |h| h.bytes())
    }

    /// Cos / sin tables (scaled by the YaRN magnitude) for `positions`.
    fn rope(&self, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let m = self.dense();
        let (mut cos, mut sin) = rope_tables(positions, &m.inv_freq);
        if m.rope_mag != 1.0 {
            cos.iter_mut().chain(sin.iter_mut()).for_each(|v| *v *= m.rope_mag);
        }
        (cos, sin)
    }

    /// An MLA layer's latent and rotated rotary key for `t` rows of its `kv_a` projection (`kv`,
    /// `[t, lat + dr]`) at positions `start..`: `([t, lat], [t, dr])`.
    fn latent(&self, l: &MlaLayer, kv: &[f32], t: usize, start: usize) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let Backbone::Mla { lat, dr, .. } = c.backbone else {
            unreachable!()
        };
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = self.rope(&positions);
        let half = dr / 2;
        let (mut cl, mut kr) = (vec![0f32; t * lat], vec![0f32; t * dr]);
        for ti in 0..t {
            let row = &kv[ti * (lat + dr)..(ti + 1) * (lat + dr)];
            rmsnorm_f32w(&row[..lat], &l.kv_norm, c.eps, &mut cl[ti * lat..(ti + 1) * lat]);
            let k = &mut kr[ti * dr..(ti + 1) * dr];
            k.copy_from_slice(&row[lat..]);
            apply_rope_interleaved(k, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
        }
        (cl, kr)
    }

    /// A GQA layer's keys and values from its projections of `t` rows at positions `start..`
    /// (`k` `[t, n_kv * hd]`; `v` the same, or None when the values are the raw keys): normed
    /// and rotated keys, (normed) values.
    fn keys_values(
        &self,
        l: &GqaLayer,
        g: &GqaConfig,
        mut k: Vec<f32>,
        v: Option<Vec<f32>>,
        t: usize,
        start: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let hd = g.head_dim;
        let mut v = v.unwrap_or_else(|| k.clone());
        if g.v_norm {
            for_chunks(&mut v, hd, t > 1, |_, vh| rmsnorm_inplace(vh, None, c.eps));
        }
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = self.rope(&positions);
        let half = g.rope_dims / 2;
        for_chunks(&mut k, hd, t > 1, |i, kh| {
            let ti = i / g.n_kv;
            rmsnorm_f32w_inplace(kh, &l.k_norm, c.eps);
            apply_rope(
                &mut kh[..2 * half],
                &cos[ti * half..(ti + 1) * half],
                &sin[ti * half..(ti + 1) * half],
            );
        });
        (k, v)
    }

    /// Every layer's key-side projections of the same rows `x` (`[t, H]`), in one batch: MLA's
    /// `kv_a` (`[t, lat + dr]`, second None), GQA's keys and values (None: the keys double as
    /// values).
    fn kv_projections(&self, x: &[f32], t: usize) -> Vec<(Vec<f32>, Option<Vec<f32>>)> {
        let m = self.dense();
        let h = self.cfg.hidden;
        let mut out: Vec<(Vec<f32>, Option<Vec<f32>>)> = m
            .layers
            .iter()
            .map(|l| match (&l.attn, &self.cfg.backbone) {
                (Attn::Mla(_), Backbone::Mla { lat, dr, .. }) => (vec![0f32; t * (lat + dr)], None),
                (Attn::Gqa(a), Backbone::Gqa(g)) => {
                    let w = g.n_kv * g.head_dim;
                    (vec![0f32; t * w], a.v.as_ref().map(|_| vec![0f32; t * w]))
                }
                _ => unreachable!(),
            })
            .collect();
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = Vec::new();
        for (l, (k, v)) in m.layers.iter().zip(out.iter_mut()) {
            match &l.attn {
                Attn::Mla(a) => jobs.push((&a.kv_a, k.len() / t, &mut k[..])),
                Attn::Gqa(a) => {
                    jobs.push((&a.k, k.len() / t, &mut k[..]));
                    if let (Some(wv), Some(v)) = (&a.v, v.as_mut()) {
                        jobs.push((wv, v.len() / t, &mut v[..]));
                    }
                }
            }
        }
        self.lin(&mut jobs, h, x, t);
        out
    }

    /// Adds the target features of the next `t` positions (`feats` `[t, n_feat]`, as
    /// [`Target::forward_feats`] returns them) to the draft's context cache.
    pub fn inject(&self, sess: &mut DsparkSession, feats: &[f32], t: usize) {
        if t == 0 {
            return;
        }
        let _timer = crate::prof::start(crate::prof::Cat::Other);
        let h = self.cfg.hidden;
        let n_feat = self.cfg.target_layers.len() * self.target_hidden;
        assert_eq!(feats.len(), t * n_feat);
        if let Some(d) = &self.ds4 {
            for (li, kv) in d.context(feats, t, sess.len).into_iter().enumerate() {
                sess.a[li].extend_from_slice(&kv);
            }
            sess.len += t;
            return;
        }
        if self.gpu_inject(sess, feats, t) {
            return;
        }
        let m = self.dense();
        let mut x = vec![0f32; t * h];
        self.lin(&mut [(&m.fc, h, &mut x[..])], n_feat, feats, t);
        rmsnorm_f32w_rows_inplace(&mut x, h, &m.enc_norm, self.cfg.eps);
        let kvs = self.kv_projections(&x, t);
        for (li, (l, (k, v))) in m.layers.iter().zip(kvs).enumerate() {
            match (&l.attn, &self.cfg.backbone) {
                (Attn::Mla(a), _) => {
                    let (cl, kr) = self.latent(a, &k, t, sess.len);
                    sess.push(li, &cl, &kr);
                }
                (Attn::Gqa(q), Backbone::Gqa(g)) => {
                    let (k, v) = self.keys_values(q, g, k, v, t, sess.len);
                    match sess.kv.get_mut(li) {
                        Some(c) => c.push(&k, &v, t),
                        None => sess.push(li, &k, &v),
                    }
                }
                _ => unreachable!(),
            }
        }
        sess.len += t;
        sess.trim();
    }

    /// The token embedding rows of the block: `anchor`, then the mask rows (`[nb, H]`).
    fn block_input<M: Target + ?Sized>(&self, target: &M, anchor: u32, nb: usize) -> Vec<f32> {
        let c = &self.cfg;
        let h = c.hidden;
        let mut x = vec![0f32; nb * h];
        let emb = |tok: u32, out: &mut [f32]| match &self.embed {
            Some(e) => e.row(tok as usize, out),
            None => target.embed_row(tok, out),
        };
        emb(anchor, &mut x[..h]);
        for i in 1..nb {
            let row = &mut x[i * h..(i + 1) * h];
            match (c.mask, self.dense.as_ref().and_then(|d| d.mask.as_ref())) {
                (MaskEmbed::Learned, Some(m)) => m.row(i - 1, row),
                (MaskEmbed::Token(id), _) => emb(id, row),
                _ => unreachable!(),
            }
        }
        // The learned mask rows are used as they are; token embeddings get the input scale.
        let scale = c.embed_scale.unwrap_or_else(|| target.embed_scale());
        if scale != 1.0 {
            let rows = if c.mask == MaskEmbed::Learned { 1 } else { nb };
            x[..rows * h].iter_mut().for_each(|v| *v *= scale);
        }
        x
    }

    /// Runs the block `[anchor, mask_1, ..]` of `nb` rows at positions `sess.len..` through the
    /// draft layers; returns the final-normed states `[nb, H]`.
    fn block<M: Target + ?Sized>(&self, target: &M, sess: &DsparkSession, anchor: u32, nb: usize) -> Vec<f32> {
        let c = &self.cfg;
        let h = c.hidden;
        let pos = sess.len;
        let mut x = self.block_input(target, anchor, nb);
        if let Some(d) = &self.ds4 {
            // The window of the last `window` positions.
            let hd = d.cfg.head_dim;
            let lo = pos.saturating_sub(c.windows.first().copied().unwrap_or(0));
            let ctx: Vec<&[f32]> = sess.a.iter().map(|kv| &kv[lo * hd..pos * hd]).collect();
            return d.block(&x, nb, pos, &ctx);
        }
        let m = self.dense();
        let positions: Vec<usize> = (pos..pos + nb).collect();
        let (cos, sin) = self.rope(&positions);
        let mut hn = vec![0f32; nb * h];
        let mut o = vec![0f32; nb * h];
        for (li, l) in m.layers.iter().enumerate() {
            // The context the block sees: the last `window` positions, whatever the slot (fewer
            // after a deep truncation, see `DsparkSession::truncate`).
            let lo = match c.windows[li] {
                0 => 0,
                w => pos.saturating_sub(w),
            };
            let lo = lo.max(sess.start);
            let (att_norm, post_attn) = match &l.attn {
                Attn::Mla(m) => (&m.attn_norm, None),
                Attn::Gqa(q) => (&q.attn_norm, q.post_attn.as_ref()),
            };
            rmsnorm_f32w_rows(&x, h, att_norm, c.eps, &mut hn);
            match (&l.attn, &c.backbone) {
                (Attn::Mla(m), _) => self.mla_attention(m, sess, li, &hn, nb, lo, &cos, &sin, &mut o),
                (Attn::Gqa(q), Backbone::Gqa(g)) => self.gqa_attention(q, g, sess, li, &hn, nb, lo, &mut o),
                _ => unreachable!(),
            }
            if let Some(w) = post_attn {
                rmsnorm_f32w_rows_inplace(&mut o, h, w, c.eps);
            }
            add_inplace(&mut x, &o);
            rmsnorm_f32w_rows(&x, h, &l.ffn_norm, c.eps, &mut hn);
            let g = self.glu(l, &hn, nb);
            self.lin(&mut [(&l.down, h, &mut o[..])], c.inter, &g, nb);
            if let Attn::Gqa(q) = &l.attn {
                if let Some(w) = &q.post_ffn {
                    rmsnorm_f32w_rows_inplace(&mut o, h, w, c.eps);
                }
                add_inplace(&mut x, &o);
                if q.scalar != 1.0 {
                    scale_inplace(&mut x, q.scalar);
                }
            } else {
                add_inplace(&mut x, &o);
            }
        }
        rmsnorm_f32w_rows_inplace(&mut x, h, &m.out_norm, c.eps);
        x
    }

    /// MLA attention of the block rows `hn` (`[nb, H]`) over context positions `lo..` and the
    /// whole block; writes the output projection to `o` (`[nb, H]`).
    #[allow(clippy::too_many_arguments)]
    fn mla_attention(
        &self,
        l: &MlaLayer,
        sess: &DsparkSession,
        li: usize,
        hn: &[f32],
        nb: usize,
        lo: usize,
        cos: &[f32],
        sin: &[f32],
        o: &mut [f32],
    ) {
        let c = &self.cfg;
        let Backbone::Mla { lat, dn, dr, vd } = c.backbone else {
            unreachable!()
        };
        let (h, nh) = (c.hidden, c.n_heads);
        let qk = dn + dr;
        let pos = sess.len;
        let half = dr / 2;
        let n_ctx = pos - lo;
        let scale = (qk as f32).powf(-0.5);
        let mut q = vec![0f32; nb * nh * qk];
        let mut kv = vec![0f32; nb * (lat + dr)];
        self.lin(&mut [(&l.q, nh * qk, &mut q[..]), (&l.kv_a, lat + dr, &mut kv[..])], h, hn, nb);
        let (bc, br) = self.latent(l, &kv, nb, pos);
        // Non-rotary queries per head [nh][nb][dn] onto the latent; rotary parts rotated.
        let mut qn = vec![0f32; nh * nb * dn];
        for i in 0..nb {
            for hh in 0..nh {
                let qh = &mut q[(i * nh + hh) * qk..(i * nh + hh + 1) * qk];
                qn[(hh * nb + i) * dn..(hh * nb + i + 1) * dn].copy_from_slice(&qh[..dn]);
                apply_rope_interleaved(&mut qh[dn..], &cos[i * half..(i + 1) * half], &sin[i * half..(i + 1) * half]);
            }
        }
        let ql = crate::mistral4::per_head(&l.k_b, &qn, nb, dn, lat);
        let (cc, cr) = sess.range(li, lo, pos);
        let mut ol = vec![0f32; nh * nb * lat];
        let op = SendPtr(ol.as_mut_ptr());
        crate::pool::run(nb * nh, &|task| {
            let (i, hh) = (task / nh, task % nh);
            let qlh = &ql[(hh * nb + i) * lat..(hh * nb + i + 1) * lat];
            let qr = &q[(i * nh + hh) * qk + dn..(i * nh + hh + 1) * qk];
            let key = |j: usize| -> (&[f32], &[f32]) {
                if j < n_ctx {
                    (&cc[j * lat..(j + 1) * lat], &cr[j * dr..(j + 1) * dr])
                } else {
                    let b = j - n_ctx;
                    (&bc[b * lat..(b + 1) * lat], &br[b * dr..(b + 1) * dr])
                }
            };
            let mut s: Vec<f32> = (0..n_ctx + nb)
                .map(|j| {
                    let (kc, kr) = key(j);
                    (dot(qlh, kc) + dot(qr, kr)) * scale
                })
                .collect();
            softmax_inplace(&mut s);
            // SAFETY: each (row, head) task writes its own latent row.
            let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((hh * nb + i) * lat), lat) };
            for (j, &p) in s.iter().enumerate() {
                axpy(o, p, key(j).0);
            }
        });
        let v = crate::mistral4::per_head(&l.v_b, &ol, nb, lat, vd);
        let mut att = vec![0f32; nb * nh * vd];
        for i in 0..nb {
            for hh in 0..nh {
                att[(i * nh + hh) * vd..(i * nh + hh + 1) * vd].copy_from_slice(&v[(hh * nb + i) * vd..(hh * nb + i + 1) * vd]);
            }
        }
        self.lin(&mut [(&l.o, h, o)], nh * vd, &att, nb);
    }

    /// Grouped-query attention of the block rows `hn` (`[nb, H]`) over context positions `lo..`
    /// and the block (causal or not); writes the output projection to `o` (`[nb, H]`).
    #[allow(clippy::too_many_arguments)]
    fn gqa_attention(&self, l: &GqaLayer, g: &GqaConfig, sess: &DsparkSession, li: usize, hn: &[f32], nb: usize, lo: usize, o: &mut [f32]) {
        let c = &self.cfg;
        let (h, nh, hd, nkv) = (c.hidden, c.n_heads, g.head_dim, g.n_kv);
        let kvw = nkv * hd;
        let pos = sess.len;
        let n_ctx = pos - lo;
        let causal = c.causal[li];
        let mut q = vec![0f32; nb * nh * hd];
        let mut k = vec![0f32; nb * kvw];
        let mut v = l.v.as_ref().map(|_| vec![0f32; nb * kvw]);
        {
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&l.q, nh * hd, &mut q[..]), (&l.k, kvw, &mut k[..])];
            if let (Some(wv), Some(v)) = (&l.v, v.as_mut()) {
                jobs.push((wv, kvw, &mut v[..]));
            }
            self.lin(&mut jobs, h, hn, nb);
        }
        let positions: Vec<usize> = (pos..pos + nb).collect();
        let (cos, sin) = self.rope(&positions);
        let half = g.rope_dims / 2;
        for_chunks(&mut q, hd, true, |i, qh| {
            let ti = i / nh;
            rmsnorm_f32w_inplace(qh, &l.q_norm, c.eps);
            apply_rope(
                &mut qh[..2 * half],
                &cos[ti * half..(ti + 1) * half],
                &sin[ti * half..(ti + 1) * half],
            );
        });
        let (bk, bv) = self.keys_values(l, g, k, v, nb, pos);
        let (ck, cv) = sess.range(li, lo, pos);
        let mut att = vec![0f32; nb * nh * hd];
        let ap = SendPtr(att.as_mut_ptr());
        let group = nh / nkv;
        crate::pool::run(nb * nh, &|task| {
            let (i, hh) = (task / nh, task % nh);
            let kvh = hh / group;
            let qh = &q[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
            let rows = if causal { i + 1 } else { nb };
            let kv = |j: usize| -> (&[f32], &[f32]) {
                if j < n_ctx {
                    let o = j * kvw + kvh * hd;
                    (&ck[o..o + hd], &cv[o..o + hd])
                } else {
                    let o = (j - n_ctx) * kvw + kvh * hd;
                    (&bk[o..o + hd], &bv[o..o + hd])
                }
            };
            let mut s: Vec<f32> = (0..n_ctx + rows).map(|j| dot(qh, kv(j).0) * g.attn_scale).collect();
            softmax_inplace(&mut s);
            // SAFETY: each (row, head) task writes its own output slice.
            let out = unsafe { std::slice::from_raw_parts_mut(ap.get().add((i * nh + hh) * hd), hd) };
            for (j, &p) in s.iter().enumerate() {
                axpy(out, p, kv(j).1);
            }
        });
        self.lin(&mut [(&l.o, h, o)], nh * hd, &att, nb);
    }

    /// Drafts up to `k` tokens (1 to [`DsparkConfig::max_draft`]) after `anchor`, the committed
    /// token at position `sess.len` the target has not run yet. The block is cut before the
    /// first draft whose probability under the draft's own softmax is below `p_min`, or whose
    /// confidence (the confidence head's estimate that it is accepted) is below `conf_min` (0:
    /// never), which leaves the output distribution unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn propose<M: Target + ?Sized>(
        &self,
        target: &M,
        sess: &mut DsparkSession,
        anchor: u32,
        k: usize,
        p_min: f32,
        conf_min: f32,
        sampler: &mut Sampler,
    ) -> Proposal {
        let mut p = Proposal::default();
        p.tokens = self.slots(target, sess, anchor, k, &mut |_, logits, conf| {
            if conf.is_some_and(|c| c < conf_min) {
                return None;
            }
            let q = sampler.distribution(logits);
            let d = if sampler.greedy() { q[0].0 } else { sampler.sample_from(&q) };
            let pd = softmax_at(logits, d);
            if pd < p_min {
                return None;
            }
            p.dists.push(q);
            p.probs.push(pd);
            p.conf.push(conf.unwrap_or(f32::NAN));
            Some(d)
        });
        p
    }

    /// The final logits of each of `k` slots (`[k, vocab]`) with the chain following the
    /// argmax, and those greedy drafts (for checks against a reference).
    pub fn draft_logits<M: Target + ?Sized>(&self, target: &M, sess: &mut DsparkSession, anchor: u32, k: usize) -> (Vec<f32>, Vec<u32>) {
        let mut out = Vec::new();
        let drafts = self.slots(target, sess, anchor, k, &mut |_, logits, _| {
            out.extend_from_slice(logits);
            Some(crate::sampler::argmax(logits))
        });
        (out, drafts)
    }

    /// One draft pass: the whole block, the (soft-capped) logits of the first `k` draft slots,
    /// and the Markov chain, `choose(slot, logits, confidence)` picking each slot's token (the
    /// next slot's "previous token"; None ends the block). Returns the chosen tokens.
    fn slots<M: Target + ?Sized>(&self, target: &M, sess: &mut DsparkSession, anchor: u32, k: usize, choose: &mut ChooseSlot) -> Vec<u32> {
        let _timer = crate::prof::start(crate::prof::Cat::Other);
        let c = &self.cfg;
        let k = k.clamp(1, c.max_draft());
        let first = usize::from(!c.sample_from_anchor);
        // The trained block runs whole (its rows attend to each other), except on the MLA
        // backbone (Mistral Large 4), which runs the anchor and the `k` slots.
        let h = c.hidden;
        let nb = if matches!(c.backbone, Backbone::Mla { .. }) {
            k + first
        } else {
            c.block_rows()
        };
        let (hid, mut logits) = match self.gpu_block(target, sess, anchor) {
            Some((hid, mut l)) => {
                let v = l.len() / c.block_rows();
                l.drain(..first * v);
                l.truncate(k * v);
                (hid, l)
            }
            None => {
                let hid = self.block(target, sess, anchor, nb);
                let rows = &hid[first * h..(first + k) * h];
                let mut logits = match &self.head {
                    Some(hd) => {
                        let v = target.vocab_size();
                        let mut y = vec![0f32; k * v];
                        self.lin(&mut [(hd, v, &mut y[..])], h, rows, k);
                        y
                    }
                    None => match self.device().then(|| target.head_raw_device(rows, k)).flatten() {
                        Some(y) => y,
                        None => target.head_raw(rows, k),
                    },
                };
                // The cap applies to the backbone's logits, not to the Markov bias.
                if let Some(cap) = c.softcap {
                    for_chunks(&mut logits, 8192, true, |_, ch| {
                        for v in ch.iter_mut() {
                            *v = (*v / cap).tanh() * cap;
                        }
                    });
                }
                (hid, logits)
            }
        };
        let rows = &hid[first * h..(first + k) * h];
        let v = logits.len() / k;
        let mut out = Vec::with_capacity(k);
        let mut prev = anchor;
        let mut state = self.markov.as_ref().map(|m| vec![0f32; m.rank]);
        for i in 0..k {
            let li = &mut logits[i * v..(i + 1) * v];
            let hid_i = &rows[i * h..(i + 1) * h];
            let mut w1 = Vec::new();
            if let Some(m) = &self.markov {
                let r = m.rank;
                w1 = vec![0f32; r];
                m.w1.row(prev as usize, &mut w1);
                // Vanilla head: the previous token's embedding; rnn head: gated state update,
                // `state = gate * state + (1 - gate) * cand`, output tanh(third chunk).
                let feat = match (&m.joint, state.as_mut()) {
                    (Some((jp, b)), Some(st)) => {
                        let mut inp = Vec::with_capacity(2 * r + h);
                        inp.extend_from_slice(st);
                        inp.extend_from_slice(&w1);
                        inp.extend_from_slice(hid_i);
                        let mut j = vec![0f32; 3 * r];
                        // Small (11 MB on Mistral Large 4's draft): not worth a submission.
                        linear(jp, 3 * r, 2 * r + h, &inp, 1, &mut j);
                        add_inplace(&mut j, b);
                        for e in 0..r {
                            let gate = 1.0 / (1.0 + (-j[e]).exp());
                            let cand = j[r + e].tanh();
                            st[e] = cand + gate * (st[e] - cand);
                        }
                        j[2 * r..].iter().map(|v| v.tanh()).collect()
                    }
                    _ => w1.clone(),
                };
                let mut bias = vec![0f32; v];
                self.lin(&mut [(&m.w2, v, &mut bias[..])], r, &feat, 1);
                add_inplace(li, &bias);
            }
            // Confidence: sigmoid(w . [hidden, markov_w1[prev]] + b).
            let conf = self.conf.as_ref().map(|cf| {
                let mut z = cf.b + dot(&cf.w[..h], hid_i);
                if cf.w.len() > h {
                    z += dot(&cf.w[h..], &w1);
                }
                1.0 / (1.0 + (-z).exp())
            });
            let Some(t) = choose(i, li, conf) else { break };
            prev = t;
            out.push(t);
        }
        out
    }
}

/// The GPU side of a GQA draft: the context and block passes as multi-token decode plans
/// (`crate::gpu::decode`), so a round runs the draft and the verification without CPU matmuls.
impl Dspark<'_> {
    /// Compiles the GPU plans for a target whose LM head is `head` (the draft's own if it has
    /// one), once; later calls do nothing. Without a GPU decode backend, or for the MLA backbone,
    /// the draft runs on the CPU.
    ///
    /// With `fast`, the plans on the fast batched kernels are compiled too (sessions with
    /// [`DsparkSession::fast`] use them; the drafts change in rounding only, the output not at
    /// all since the target verifies them).
    pub fn init_gpu(&self, head: Option<&Lin>, vocab: usize, fast: bool) {
        let build = |fast: bool| {
            if !crate::gpu::decode_enabled() || std::env::var_os("OXINFER_DSPARK_CPU").is_some() {
                return None;
            }
            let Backbone::Gqa(g) = &self.cfg.backbone else { return None };
            let be = crate::gpu::active()?;
            let head = self.head.as_ref().or(head)?;
            let rows = self.cfg.block_rows();
            let plan = |gr: &crate::gpu::decode::Graph, t: usize| {
                if fast {
                    be.decode_plan_multi_fast(gr, t)
                } else {
                    be.decode_plan_multi(gr, t)
                }
            };
            let block = plan(&self.block_graph(g, head, vocab)?, rows)?;
            let ctx = plan(&self.ctx_graph(g)?, 16)?;
            Some(GpuDraft {
                ctx: std::sync::Arc::from(ctx),
                block: std::sync::Arc::from(block),
            })
        };
        self.gpu.get_or_init(|| build(false));
        if fast && matches!(self.gpu.get(), Some(Some(_))) {
            self.gpu_fast.get_or_init(|| build(true));
        }
    }

    /// The GPU plans a session runs on (the fast ones when it asks for them and they exist).
    fn gpu_plans(&self, fast: bool) -> Option<&GpuDraft> {
        if fast && let Some(Some(g)) = self.gpu_fast.get() {
            return Some(g);
        }
        self.gpu.get().and_then(|g| g.as_ref())
    }

    /// The device weight of a matrix (None: the plan cannot use it).
    fn dev<'s>(l: &'s Lin, n_out: usize, n_in: usize) -> Option<&'s crate::gpu::DeviceWeight> {
        if l.has_lora() || l.is_transient() {
            return None;
        }
        l.device_weight(n_out, n_in)
    }

    /// Cos / sin rows of the draft's RoPE for `t` positions from `pos`, one token after the
    /// other (as `gpu::decode::Step::rope_cs`).
    fn rope_rows(&self, pos: usize, t: usize) -> Vec<f32> {
        let positions: Vec<usize> = (pos..pos + t).collect();
        let (cos, sin) = self.rope(&positions);
        let half = self.dense().inv_freq.len();
        let mut cs = Vec::with_capacity(2 * t * half);
        for j in 0..t {
            cs.extend_from_slice(&cos[j * half..(j + 1) * half]);
            cs.extend_from_slice(&sin[j * half..(j + 1) * half]);
        }
        cs
    }

    /// The context pass: features to every layer's keys and values at the step's positions.
    fn ctx_graph<'s>(&'s self, g: &GqaConfig) -> Option<crate::gpu::decode::Graph<'s>> {
        use crate::gpu::decode::{Graph, LinMode, Op};
        let c = &self.cfg;
        let (h, hd, nkv) = (c.hidden, g.head_dim, g.n_kv);
        let n_feat = c.target_layers.len() * self.target_hidden;
        let mut gr = Graph::new();
        let x = gr.buf(n_feat);
        let fx = gr.buf(h);
        let hn = gr.buf(h);
        let kv = gr.buf(2 * nkv * hd);
        let qo = gr.buf(1);
        gr.input = x;
        gr.n_input = n_feat;
        gr.hidden = hn;
        gr.n_hidden = h;
        gr.rope_halves = vec![g.rope_dims / 2];
        let lin = |l: &'s Lin, n_out: usize, n_in: usize, x, y| -> Option<Op<'s>> {
            Some(Op::Linear {
                w: l.as_ref(),
                dev: Self::dev(l, n_out, n_in)?,
                w2: None,
                n_out,
                n_in,
                x,
                y,
                mode: LinMode::Store,
            })
        };
        gr.ops.push(lin(&self.dense().fc, h, n_feat, x, fx)?);
        gr.ops.push(Op::norm(fx, hn, &self.dense().enc_norm, h, c.eps));
        for (li, l) in self.dense().layers.iter().enumerate() {
            let Attn::Gqa(a) = &l.attn else { return None };
            gr.kv.push((nkv, hd));
            gr.ops.push(lin(&a.k, nkv * hd, h, hn, kv)?);
            if let Some(v) = &a.v {
                gr.ops.push(lin(v, nkv * hd, h, hn, kv.at(nkv * hd))?);
            }
            gr.ops.push(Op::QkRope {
                qkv: kv,
                q_out: qo,
                kv: li,
                nh: 0,
                nkv,
                hd,
                half_rot: g.rope_dims / 2,
                q_stride: hd,
                k_off: 0,
                v_off: nkv * hd,
                interleaved: false,
                q_norm: None,
                k_norm: Some(&a.k_norm),
                eps: c.eps,
                rope: 0,
                v_norm: g.v_norm,
                v_from_k: a.v.is_none(),
            });
        }
        gr.head_from = gr.ops.len();
        Some(gr)
    }

    /// The block pass: the block's embeddings to the logits of every row (soft-capped, before
    /// the Markov bias).
    fn block_graph<'s>(&'s self, g: &GqaConfig, head: &'s Lin, vocab: usize) -> Option<crate::gpu::decode::Graph<'s>> {
        use crate::gpu::decode::{Ew, Graph, LinMode, Op, pairable};
        use std::borrow::Cow;
        let c = &self.cfg;
        let (h, nh, hd, nkv) = (c.hidden, c.n_heads, g.head_dim, g.n_kv);
        let mut gr = Graph::new();
        let x = gr.buf(h);
        let hn = gr.buf(h);
        let qkv = gr.buf((nh + 2 * nkv) * hd);
        let qo = gr.buf(nh * hd);
        let att = gr.buf(nh * hd);
        let ob = gr.buf(h);
        let act = gr.buf(c.inter);
        let act2 = gr.buf(c.inter);
        let hidden = gr.buf(h);
        let logits = gr.buf(vocab);
        gr.input = x;
        gr.hidden = hidden;
        gr.n_hidden = h;
        gr.rope_halves = vec![g.rope_dims / 2];
        let lin = |l: &'s Lin, n_out: usize, n_in: usize, x, y, mode| -> Option<Op<'s>> {
            Some(Op::Linear {
                w: l.as_ref(),
                dev: Self::dev(l, n_out, n_in)?,
                w2: None,
                n_out,
                n_in,
                x,
                y,
                mode,
            })
        };
        for (li, l) in self.dense().layers.iter().enumerate() {
            let Attn::Gqa(a) = &l.attn else { return None };
            if !g.sandwich && a.scalar != 1.0 {
                return None;
            }
            gr.kv.push((nkv, hd));
            gr.ops.push(Op::norm(x, hn, &a.attn_norm, h, c.eps));
            gr.ops.push(lin(&a.q, nh * hd, h, hn, qkv, LinMode::Store)?);
            gr.ops.push(lin(&a.k, nkv * hd, h, hn, qkv.at(nh * hd), LinMode::Store)?);
            if let Some(v) = &a.v {
                gr.ops.push(lin(v, nkv * hd, h, hn, qkv.at((nh + nkv) * hd), LinMode::Store)?);
            }
            gr.ops.push(Op::QkRope {
                qkv,
                q_out: qo,
                kv: li,
                nh,
                nkv,
                hd,
                half_rot: g.rope_dims / 2,
                q_stride: hd,
                k_off: nh * hd,
                v_off: (nh + nkv) * hd,
                interleaved: false,
                q_norm: Some(&a.q_norm),
                k_norm: Some(&a.k_norm),
                eps: c.eps,
                rope: 0,
                v_norm: g.v_norm,
                v_from_k: a.v.is_none(),
            });
            gr.ops.push(Op::Attend {
                q: qo,
                out: att,
                kv: li,
                nh,
                nkv,
                hd,
                scale: g.attn_scale,
                window: None,
                gate: None,
                block: Some((c.causal[li], c.windows[li])),
            });
            match &a.post_attn {
                Some(w) => {
                    gr.ops.push(lin(&a.o, h, nh * hd, att, ob, LinMode::Store)?);
                    gr.ops.push(Op::AddNorm {
                        y: ob,
                        x,
                        wy: Cow::Borrowed(w),
                        scale: 1.0,
                        next: Some((hn, &l.ffn_norm[..])),
                        n: h,
                        eps: c.eps,
                    });
                }
                None => {
                    gr.ops.push(lin(&a.o, h, nh * hd, att, x, LinMode::Add)?);
                    gr.ops.push(Op::norm(x, hn, &l.ffn_norm, h, c.eps));
                }
            }
            let (gw, uw) = (l.gate.as_ref(), l.up.as_ref());
            if pairable(&gw, &uw) {
                gr.ops.push(Op::Linear {
                    w: gw,
                    dev: Self::dev(&l.gate, c.inter, h)?,
                    w2: Some((uw, Self::dev(&l.up, c.inter, h)?)),
                    n_out: c.inter,
                    n_in: h,
                    x: hn,
                    y: act,
                    mode: if g.gelu { LinMode::GeluPair } else { LinMode::SiluPair },
                });
            } else {
                gr.ops.push(lin(&l.gate, c.inter, h, hn, act2, LinMode::Store)?);
                gr.ops.push(lin(&l.up, c.inter, h, hn, act, LinMode::Store)?);
                gr.ops.push(Op::Glu {
                    g: act2,
                    u: act,
                    out: act,
                    n: c.inter,
                    gelu: g.gelu,
                });
            }
            match &a.post_ffn {
                Some(w) => {
                    gr.ops.push(lin(&l.down, h, c.inter, act, ob, LinMode::Store)?);
                    gr.ops.push(Op::AddNorm {
                        y: ob,
                        x,
                        wy: Cow::Borrowed(w),
                        scale: a.scalar,
                        next: None,
                        n: h,
                        eps: c.eps,
                    });
                }
                None => gr.ops.push(lin(&l.down, h, c.inter, act, x, LinMode::Add)?),
            }
        }
        gr.ops.push(Op::norm(x, hidden, &self.dense().out_norm, h, c.eps));
        gr.head_from = gr.ops.len();
        gr.ops.push(lin(head, vocab, h, hidden, logits, LinMode::Store)?);
        if let Some(cap) = c.softcap {
            gr.ops.push(Op::Ew {
                kind: Ew::Softcap(cap),
                a: logits,
                b: None,
                out: logits,
                n: vocab,
            });
        }
        gr.logits = Some((logits, vocab));
        Some(gr)
    }

    /// The context pass on the GPU, when the plans exist and `t` fits; false otherwise (nothing
    /// changed).
    fn gpu_inject(&self, sess: &mut DsparkSession, feats: &[f32], t: usize) -> bool {
        let Some(gp) = self.gpu_plans(sess.fast) else { return false };
        if t > gp.ctx.tokens() || sess.kv.len() != self.dense().layers.len() {
            return false;
        }
        let h = self.cfg.hidden;
        let cs = self.rope_rows(sess.len, t);
        let mut hidden = vec![0f32; t * h];
        let mut kv: Vec<&mut LayerCache> = sess.kv.iter_mut().collect();
        let ok = gp.ctx.step(
            &mut sess.gpu_ctx,
            crate::gpu::decode::Step {
                pos: sess.len,
                t,
                rope_cs: &cs,
                x: feats,
                x2: &[],
                kv: &mut kv,
                states: &mut [],
                hidden: &mut hidden,
                logits: None,
                params: None,
                capture: None,
                scratch: false,
            },
        );
        if ok {
            sess.len += t;
        }
        ok
    }

    /// The block pass on the GPU: the final-normed rows and their (soft-capped) logits, all
    /// `block_rows` of them; the session is left as it was. None when the plans are missing or
    /// the device failed.
    fn gpu_block<M: Target + ?Sized>(&self, target: &M, sess: &mut DsparkSession, anchor: u32) -> Option<(Vec<f32>, Vec<f32>)> {
        let gp = self.gpu_plans(sess.fast)?;
        let _timer = crate::prof::start(crate::prof::Cat::Matmul);
        let nb = self.cfg.block_rows();
        let h = self.cfg.hidden;
        let pos = sess.len;
        let x = self.block_input(target, anchor, nb);
        let cs = self.rope_rows(pos, nb);
        let vocab = target.vocab_size();
        let mut hidden = vec![0f32; nb * h];
        let mut logits = vec![0f32; nb * vocab];
        let mut kv: Vec<&mut LayerCache> = sess.kv.iter_mut().collect();
        let ok = gp.block.step(
            &mut sess.gpu_block,
            crate::gpu::decode::Step {
                pos,
                t: nb,
                rope_cs: &cs,
                x: &x,
                x2: &[],
                kv: &mut kv,
                states: &mut [],
                hidden: &mut hidden,
                logits: Some(&mut logits),
                params: None,
                capture: None,
                scratch: true,
            },
        );
        ok.then_some((hidden, logits))
    }
}

/// Picks a slot's token from `(slot, logits, confidence)`; None ends the block.
type ChooseSlot<'c> = dyn FnMut(usize, &[f32], Option<f32>) -> Option<u32> + 'c;

/// The drafts of one pass.
#[derive(Clone, Debug, Default)]
pub struct Proposal {
    pub tokens: Vec<u32>,
    /// The distribution each draft was drawn from (the sampler's temperature / top-k / top-p
    /// over the draft logits; one token with probability 1 when greedy).
    pub dists: Vec<Vec<(u32, f32)>>,
    /// Each draft's probability under the draft's own softmax (temperature 1).
    pub probs: Vec<f32>,
    /// Each draft's confidence-head estimate (NaN without one).
    pub conf: Vec<f32>,
}

/// DSpark decoding settings.
#[derive(Clone, Debug)]
pub struct DsparkOptions {
    /// Draft tokens per round (at most the draft's block, [`DsparkConfig::max_draft`]; the
    /// default is the whole block).
    pub draft_tokens: usize,
    /// Tokens that end generation: nothing after one is committed within a round.
    pub stop: Vec<u32>,
    /// Choose each round between a draft pass and a plain step by their recently measured tokens
    /// per second, and verify the part of each block with the most expected tokens per second,
    /// from the measured cost of verifying each length and each draft's calibrated chance of
    /// being accepted ([`Adaptive`]); where the draft is rarely right, decoding stays near plain
    /// speed.
    pub cost_aware: bool,
    /// Cut each block before the first draft the draft model itself gives less than this
    /// probability (llama.cpp's `--spec-draft-p-min`; 0 = off): an unlikely draft costs a
    /// verification row, and its routed experts when they stream from storage.
    pub p_min: f32,
    /// Where the draft runs its matmuls ([`Dspark::set_device`]): the accelerator
    /// (`Some(true)`), the CPU (`Some(false)`), or by default [`DEVICE_BY_DEFAULT`] when an
    /// accelerator is active (None). A GQA draft with GPU decode plans ([`Dspark::init_gpu`])
    /// runs whole passes there instead, unless this is `Some(false)`.
    pub device: Option<bool>,
    /// Cut each block before the first draft whose confidence-head estimate of being accepted is
    /// below this (DeepSpec's `confidence_threshold`; 0 = off, and ignored by drafts without
    /// the head).
    pub conf_min: f32,
    /// "Fast verification": verify (and draft) on the GPU with batched matrix kernels that read
    /// the weights once for all the tokens but sum in another order than one-token decoding, so
    /// a token's logits differ from plain decoding's in rounding (greedy output can then drift
    /// from plain greedy output at near-ties; sampling stays a correct sample of the model up to
    /// that rounding). On by default; off, verification reproduces one-token decoding bit for
    /// bit. Only GPU decode plans have fast kernels (Metal); elsewhere this changes nothing.
    pub fast_verify: bool,
}

impl Default for DsparkOptions {
    fn default() -> Self {
        Self {
            draft_tokens: usize::MAX,
            stop: Vec::new(),
            cost_aware: true,
            p_min: 0.0,
            device: None,
            conf_min: 0.0,
            fast_verify: true,
        }
    }
}

/// Speculative generation with a DSpark draft: the plain decoding distribution exactly (greedy
/// gives the plain greedy tokens: the verification computes the same logits as one-token
/// forwards; sampling uses standard speculative sampling, see
/// [`crate::qwen_mtp::speculative_accept`]).
///
/// Usage: `prefill` the prompt, choose the first token from the returned logits and `push` it,
/// then call `step` until a stop token or the length limit; `prefill` again for a later turn.
pub struct DsparkDecoder<'m, M: Target> {
    model: &'m M,
    draft: &'m Dspark<'m>,
    pub sess: M::Sess,
    dsess: DsparkSession,
    pub opts: DsparkOptions,
    /// The committed token the target has not processed yet.
    next: Option<u32>,
    k_cur: usize,
    pub stats: MtpStats,
    adapt: Adaptive,
    /// Features of committed positions not yet in the draft's context (`pending.1` positions):
    /// they join it before the next draft pass, so rounds without a draft cost no context pass
    /// and consecutive ones share one.
    pending: (Vec<f32>, usize),
}

/// The cost-aware draft length policy ([`DsparkOptions::cost_aware`]), in two steps per round:
///
/// - Whether to draft: a draft pass costs the same whatever the number of drafts used, so a round
///   either drafts the whole block or none. It drafts when draft rounds have recently given more
///   tokens per second than plain steps; otherwise it decodes plainly and drafts again now and then
///   (after 6 plain rounds, twice as many after each probe that lost, up to 96) to follow the
///   text.
/// - How many drafts to verify ([`Adaptive::cut`]): with the drafts in hand, the length that
///   gives the most expected tokens per second of the round, from the measured cost of verifying
///   each length and each draft's chance of being accepted, estimated from its probability under
///   the draft model (calibrated online: drafts by probability bin, accepted or not).
#[derive(Clone, Debug)]
pub struct Adaptive {
    /// Recent cost of a draft pass, and of verifying (and committing) `t` drafts, with the number
    /// of measurements.
    pub draft: (f64, usize),
    pub verify: Vec<(f64, usize)>,
    /// Decayed counts of verified drafts and of those accepted, by the draft model's probability
    /// (tenths).
    pub bins: [(f64, f64); 10],
    /// Recent tokens and seconds per round with a draft pass.
    pub gain: (f64, f64, usize),
    /// Rounds in a row without a draft, and the current number of them before a probe.
    pub plain_run: usize,
    pub probe: usize,
    /// This round was a probe (drafted against the measured rates).
    probing: bool,
}

impl Default for Adaptive {
    fn default() -> Self {
        Self {
            draft: (0.0, 0),
            verify: Vec::new(),
            bins: [(0.0, 0.0); 10],
            gain: (0.0, 0.0, 0),
            plain_run: 0,
            probe: Self::PROBE,
            probing: false,
        }
    }
}

impl Adaptive {
    const DECAY: f64 = 0.8;
    /// Plain rounds before the first probe (doubling up to `PROBE_MAX` while probes lose).
    const PROBE: usize = 6;
    const PROBE_MAX: usize = 96;
    /// Weight of the draft model's own probability against the calibration counts of its bin.
    const PRIOR: f64 = 2.0;
    /// Decay of the calibration counts per verified draft (a window of about 500 drafts).
    const BIN_DECAY: f64 = 0.998;

    /// A moving average whose second sample replaces the first (which may include compiling a
    /// GPU plan or warming caches).
    fn ema(e: &mut (f64, usize), v: f64) {
        *e = (if e.1 < 2 { v } else { Self::DECAY * e.0 + (1.0 - Self::DECAY) * v }, e.1 + 1);
    }

    /// Records a round: `probs` the probabilities of the drafts proposed (empty: no draft pass),
    /// `verified` of them verified, `accepted` of those accepted, `emitted` tokens committed.
    fn record(&mut self, probs: &[f32], verified: usize, accepted: usize, emitted: usize, draft: f64, verify: f64) {
        if self.verify.len() <= verified {
            self.verify.resize(verified + 1, (0.0, 0));
        }
        Self::ema(&mut self.verify[verified], verify);
        if probs.is_empty() {
            self.plain_run += 1;
            return;
        }
        Self::ema(&mut self.draft, draft);
        for (j, &p) in probs.iter().enumerate().take(verified.min(accepted + 1)) {
            for b in &mut self.bins {
                b.0 *= Self::BIN_DECAY;
                b.1 *= Self::BIN_DECAY;
            }
            let b = &mut self.bins[Self::bin(p)];
            b.0 += f64::from(u8::from(j < accepted));
            b.1 += 1.0;
        }
        let (t, s) = (emitted as f64, draft + verify);
        self.gain = match self.gain.2 {
            0 | 1 => (t, s, self.gain.2 + 1),
            n => (
                Self::DECAY * self.gain.0 + (1.0 - Self::DECAY) * t,
                Self::DECAY * self.gain.1 + (1.0 - Self::DECAY) * s,
                n + 1,
            ),
        };
        if self.probing {
            // A probe that lost: probe half as often.
            let plain = self.verify.first().map_or(0.0, |v| v.0);
            if t * plain < s {
                self.probe = (self.probe * 2).min(Self::PROBE_MAX);
            } else {
                self.probe = Self::PROBE;
            }
        }
        self.plain_run = 0;
    }

    fn bin(p: f32) -> usize {
        ((p.max(0.0) * 10.0) as usize).min(9)
    }

    /// The chance that a draft of probability `p` (under the draft model) is accepted when the
    /// drafts before it were.
    pub fn accept_chance(&self, p: f32) -> f64 {
        let (a, n) = self.bins[Self::bin(p)];
        (a + Self::PRIOR * f64::from(p)) / (n + Self::PRIOR)
    }

    /// Verifying `k` drafts: measured, else interpolated (or extrapolated flat) between the
    /// nearest measured lengths.
    fn verify_time(&self, k: usize) -> Option<f64> {
        let m: Vec<(usize, f64)> = self
            .verify
            .iter()
            .enumerate()
            .filter(|e| e.1.1 > 0)
            .map(|(j, e)| (j, e.0))
            .collect();
        let lo = m.iter().rev().find(|e| e.0 <= k);
        let hi = m.iter().find(|e| e.0 >= k);
        match (lo, hi) {
            (Some(a), Some(b)) if a.0 == b.0 => Some(a.1),
            (Some(a), Some(b)) => Some(a.1 + (b.1 - a.1) * (k - a.0) as f64 / (b.0 - a.0) as f64),
            (Some(a), None) | (None, Some(a)) => Some(a.1),
            (None, None) => None,
        }
    }

    /// The next round's draft length: `k_max` (a draft pass) or 0 (a plain step).
    pub fn choose(&mut self, k_max: usize) -> usize {
        self.probing = false;
        // First two draft rounds, then two plain ones, to measure both (the first of each may
        // compile a plan).
        if self.draft.1 < 2 {
            return k_max;
        }
        let Some(plain) = self.verify.first().filter(|e| e.1 >= 2).map(|e| e.0) else {
            return 0;
        };
        if self.gain.0 * plain >= self.gain.1 {
            return k_max;
        }
        if self.plain_run >= self.probe {
            self.probing = true;
            return k_max;
        }
        0
    }

    /// How many of the proposed drafts (probabilities `probs`) to verify, given the draft pass
    /// that made them took `draft_secs`: the length with the most expected tokens per second.
    pub fn cut(&self, probs: &[f32], draft_secs: f64) -> usize {
        let mut best = (probs.len(), 0.0);
        let (mut e, mut p) = (1.0, 1.0);
        for k in 0..=probs.len() {
            if k > 0 {
                p *= self.accept_chance(probs[k - 1]);
                e += p;
            }
            let Some(v) = self.verify_time(k) else { return probs.len() };
            let rate = e / (draft_secs + v).max(1e-9);
            if rate > best.1 {
                best = (k, rate);
            }
        }
        best.0
    }
}

impl<'m, M: Target> DsparkDecoder<'m, M> {
    pub fn new(model: &'m M, draft: &'m Dspark<'m>, opts: DsparkOptions) -> Self {
        // At most the block, and at most what the batch-invariant decode kernels verify (15
        // tokens: a 16-token block drafts 14).
        let opts = DsparkOptions {
            draft_tokens: opts.draft_tokens.min(draft.cfg.max_draft()).min(crate::qwen_mtp::MAX_DRAFT),
            ..opts
        };
        // The verification of up to `draft + 1` tokens must stay on the decode kernels.
        if opts.draft_tokens + 1 > crate::kernels::decode_max_tokens() {
            crate::kernels::set_decode_max_tokens(opts.draft_tokens + 1);
        }
        draft.set_device(opts.device.unwrap_or(DEVICE_BY_DEFAULT && crate::gpu::active().is_some()));
        if opts.device != Some(false) {
            draft.init_gpu(model.lm_head(), model.vocab_size(), opts.fast_verify);
        }
        let mut dsess = draft.new_session();
        dsess.fast = opts.fast_verify;
        Self {
            model,
            draft,
            sess: model.new_session(),
            dsess,
            k_cur: opts.draft_tokens,
            opts,
            next: None,
            stats: MtpStats::default(),
            adapt: Adaptive::default(),
            pending: (Vec::new(), 0),
        }
    }

    pub fn model(&self) -> &'m M {
        self.model
    }

    pub fn draft(&self) -> &'m Dspark<'m> {
        self.draft
    }

    /// Current draft length.
    pub fn draft_len(&self) -> usize {
        self.k_cur
    }

    /// The cost-aware policy's state.
    pub fn adaptive(&self) -> &Adaptive {
        &self.adapt
    }

    /// The draft's context cache (for checks).
    pub fn draft_session(&mut self) -> &DsparkSession {
        self.flush();
        &self.dsess
    }

    /// The draft's context cache, for drafting outside a round (checks).
    pub fn draft_session_mut(&mut self) -> &mut DsparkSession {
        self.flush();
        &mut self.dsess
    }

    /// Adds the pending features to the draft's context, at most [`Self::FLUSH`] positions per
    /// pass.
    fn flush(&mut self) {
        let (feats, n) = std::mem::take(&mut self.pending);
        if let Some(nf) = feats.len().checked_div(n) {
            for c in feats.chunks(Self::FLUSH * nf) {
                self.draft.inject(&mut self.dsess, c, c.len() / nf);
            }
        }
        self.pending = (feats, 0);
        self.pending.0.clear();
    }

    /// Positions of one context pass when the pending features are added (the GPU context
    /// plan's capacity).
    const FLUSH: usize = 16;

    /// Runs `tokens` through the target and adds their features to the draft's cache; returns
    /// the target's final-normed hidden states.
    fn run(&mut self, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.flush();
        let mut feats = Vec::new();
        let hs = self
            .model
            .forward_feats(&mut self.sess, tokens, mm, &self.draft.cfg.target_layers, &mut feats);
        self.draft.inject(&mut self.dsess, &feats, tokens.len());
        hs
    }

    /// Appends `tokens` (a prompt, or the next turn of a conversation) in chunks of `chunk` and
    /// returns the logits after the last one. A committed token that was never processed (the
    /// stop token ending the previous reply; a reply cut short has its last token put back in
    /// front of `tokens` by the caller) is dropped.
    pub fn prefill(&mut self, tokens: &[u32], chunk: usize) -> Vec<f32> {
        self.prefill_mm(tokens, chunk, &[])
    }

    /// `prefill` with multimodal soft tokens replacing the embeddings at some positions of
    /// `tokens` (`(index in tokens, embedding)`).
    pub fn prefill_mm(&mut self, tokens: &[u32], chunk: usize, overrides: &[(usize, Vec<f32>)]) -> Vec<f32> {
        assert!(!tokens.is_empty());
        self.next = None;
        let h = self.model.hidden_size();
        let mut last = Vec::new();
        let mut start = 0;
        for c in tokens.chunks(chunk.max(1)) {
            let end = start + c.len();
            let local: Vec<(usize, &[f32])> = overrides
                .iter()
                .filter(|(p, _)| *p >= start && *p < end)
                .map(|(p, e)| (p - start, e.as_slice()))
                .collect();
            start = end;
            let hs = self.run(c, &local);
            last = hs[hs.len() - h..].to_vec();
        }
        self.model.logits(&last, 1)
    }

    /// Commits the token chosen from `prefill`'s logits (the first token of a reply).
    pub fn push(&mut self, token: u32) {
        assert!(self.next.is_none(), "push follows prefill; step commits its own tokens");
        self.next = Some(token);
    }

    /// One speculative round: drafts up to the current draft length (and at most
    /// `max_new - 1`), verifies them with one target forward and returns the committed tokens
    /// (1 to `draft + 1` of them, ending at the first stop token if any).
    pub fn step(&mut self, sampler: &mut Sampler, max_new: usize) -> Vec<u32>
    where
        M::Sess: Clone,
    {
        let next = self.next.expect("push a first token before step");
        let (model, draft) = (self.model, self.draft);
        let v = model.vocab_size();
        let n = model.session_len(&self.sess); // position of `next`
        debug_assert_eq!(self.dsess.len + self.pending.1, n);
        let k = self.k_cur.min(max_new.max(1) - 1);

        let t_round = crate::clock::Instant::now();
        let mut prop = if k > 0 {
            self.flush();
            draft.propose(model, &mut self.dsess, next, k, self.opts.p_min, self.opts.conf_min, sampler)
        } else {
            Proposal::default()
        };
        let draft_secs = t_round.elapsed().as_secs_f64();
        // `p_min` may have cut the block; the cost-aware policy verifies the most profitable part.
        let probs = prop.probs.clone();
        if self.opts.cost_aware && !prop.tokens.is_empty() {
            let m = self.adapt.cut(&prop.probs, draft_secs);
            prop.tokens.truncate(m);
            prop.dists.truncate(m);
        }
        let (drafts, qs) = (&prop.tokens, &prop.dists);
        let kd = drafts.len();

        let mut seq = Vec::with_capacity(kd + 1);
        seq.push(next);
        seq.extend_from_slice(drafts);
        let check = std::env::var_os("OXINFER_CHECK_SPEC").is_some().then(|| self.sess.clone());
        let mut feats = Vec::new();
        let logits = model.forward_verify(&mut self.sess, &seq, &draft.cfg.target_layers, &mut feats, self.opts.fast_verify);
        let verify_secs = t_round.elapsed().as_secs_f64() - draft_secs;
        if let Some(mut s1) = check {
            // Debug: the batched verification must match one-token forwards exactly.
            for (i, &tk) in seq.iter().enumerate() {
                let l1 = model.forward(&mut s1, &[tk]);
                let lk = &logits[i * v..(i + 1) * v];
                let diff = l1.iter().zip(lk).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                if diff > 0.0 {
                    eprintln!("[check: pos {} of {}: max logit diff {diff:e}]", n + i, seq.len());
                }
            }
        }

        let mut out = Vec::with_capacity(kd + 1);
        let mut accepted = 0;
        for (j, (&d, q)) in drafts.iter().zip(qs).enumerate() {
            let p = sampler.distribution(&logits[j * v..(j + 1) * v]);
            match speculative_accept(sampler, &p, q, d) {
                None => {
                    out.push(d);
                    accepted += 1;
                    if self.opts.stop.contains(&d) {
                        break;
                    }
                }
                Some(t) => {
                    out.push(t);
                    break;
                }
            }
        }
        if accepted == drafts.len() && !out.last().is_some_and(|t| self.opts.stop.contains(t)) {
            out.push(sampler.sample(&logits[kd * v..(kd + 1) * v]));
        }
        if let Some(i) = out.iter().position(|t| self.opts.stop.contains(t)) {
            out.truncate(i + 1);
        }

        // Commit: the target keeps [next, out[..c-1]] (positions n..n+c), whose features join
        // the draft's context; out[c-1] is the new `next`.
        let c = out.len();
        let t_acc = t_round.elapsed().as_secs_f64();
        model.truncate(&mut self.sess, n + c);
        let t_trunc = t_round.elapsed().as_secs_f64();
        let nf = feats.len() / seq.len();
        self.pending.0.extend_from_slice(&feats[..c * nf]);
        self.pending.1 += c;
        self.next = out.last().copied();
        let tail_secs = t_round.elapsed().as_secs_f64() - draft_secs;
        if std::env::var_os("OXINFER_DEBUG_DRAFT").is_some() {
            eprintln!(
                "[draft: p {:?} conf {:?} accepted {accepted}; ms: draft {:.1} verify {:.1} accept {:.1} rewind {:.1} inject {:.1}]",
                prop.probs,
                prop.conf,
                draft_secs * 1e3,
                verify_secs * 1e3,
                (t_acc - draft_secs - verify_secs) * 1e3,
                (t_trunc - t_acc) * 1e3,
                (draft_secs + tail_secs - t_trunc) * 1e3
            );
        }

        let st = &mut self.stats;
        st.rounds += 1;
        st.drafted += kd;
        st.accepted += accepted;
        st.emitted += c;
        if st.per_round.len() <= accepted {
            st.per_round.resize(accepted + 1, 0);
        }
        st.per_round[accepted] += 1;
        st.draft_secs += draft_secs;
        st.verify_secs += verify_secs;
        if k == self.k_cur {
            self.adapt.record(&probs, kd, accepted, c, draft_secs, tail_secs);
        }
        if self.opts.cost_aware {
            self.k_cur = self.adapt.choose(self.opts.draft_tokens);
        }
        out
    }
}

impl<M: Target> crate::generate::RoundDecoder for DsparkDecoder<'_, M>
where
    M::Sess: Clone,
{
    fn push(&mut self, token: u32) {
        DsparkDecoder::push(self, token)
    }
    fn step(&mut self, sampler: &mut Sampler, max_new: usize) -> Vec<u32> {
        DsparkDecoder::step(self, sampler, max_new)
    }
    fn rounds(&self) -> usize {
        self.stats.rounds
    }
}

/// `softmax(logits)[t]`.
fn softmax_at(logits: &[f32], t: u32) -> f32 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f32 = logits.iter().map(|&v| (v - m).exp()).sum();
    (logits[t as usize] - m).exp() / s
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::Flow;

    const V: usize = 64;
    const H: usize = 32;

    /// A deterministic toy target: the hidden state carries the token and its position, the next
    /// token is `token + 1` except after every fourth position (`token + 2`), with some spread
    /// for sampling.
    struct Toy;

    #[derive(Clone, Default)]
    struct ToySess(Vec<u32>);

    impl TextLm for Toy {
        type Sess = ToySess;
        fn new_session(&self) -> ToySess {
            ToySess::default()
        }
        fn hidden_size(&self) -> usize {
            H
        }
        fn n_layers(&self) -> usize {
            2
        }
        fn session_len(&self, s: &ToySess) -> usize {
            s.0.len()
        }
        fn forward_hidden_mm(&self, s: &mut ToySess, tokens: &[u32], _mm: &[(usize, &[f32])]) -> Vec<f32> {
            let mut out = vec![0f32; tokens.len() * H];
            for (i, &t) in tokens.iter().enumerate() {
                out[i * H] = t as f32;
                out[i * H + 1] = s.0.len() as f32;
                s.0.push(t);
            }
            out
        }
        fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
            let mut l = vec![0f32; t * V];
            for i in 0..t {
                let (tok, pos) = (hidden[i * H] as usize, hidden[i * H + 1] as usize);
                let next = (tok + if pos % 4 == 3 { 2 } else { 1 }) % V;
                for (j, x) in l[i * V..(i + 1) * V].iter_mut().enumerate() {
                    *x = ((j * 7 + pos * 3) % 11) as f32 * 0.1;
                }
                l[i * V + next] += 3.0;
            }
            l
        }
    }

    impl Target for Toy {
        fn forward_feats(
            &self,
            s: &mut ToySess,
            tokens: &[u32],
            mm: &[(usize, &[f32])],
            layers: &[usize],
            feats: &mut Vec<f32>,
        ) -> Vec<f32> {
            let hs = self.forward_hidden_mm(s, tokens, mm);
            feats.clear();
            for row in hs.chunks(H) {
                for _ in layers {
                    feats.extend(row.iter().map(|v| v * 0.01));
                }
            }
            hs
        }
        fn embed_row(&self, token: u32, out: &mut [f32]) {
            out.fill(0.0);
            out[token as usize % H] = 1.0;
        }
        fn head_raw(&self, _hidden: &[f32], t: usize) -> Vec<f32> {
            vec![0.0; t * V]
        }
        fn vocab_size(&self) -> usize {
            V
        }
        fn truncate(&self, s: &mut ToySess, len: usize) {
            s.0.truncate(len);
        }
    }

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }

    /// A one-layer MLA draft with zero backbone weights and a vanilla Markov head that maps the
    /// previous token `p` onto `p + 1` (bias 10): it always drafts `p + 1, p + 2, ..`. With
    /// `seed`, the backbone weights are random instead and the draft has its own LM head, so its
    /// logits depend on the attended context.
    fn toy_draft(path: &Path, seed: Option<u64>) {
        let mut st = seed.unwrap_or(0);
        let mut fill = |len: usize| -> Vec<f32> {
            match seed {
                None => vec![0.0; len],
                Some(_) => (0..len)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
                    })
                    .collect(),
            }
        };
        let (lat, dn, dr, vd, inter) = (32usize, 32usize, 32usize, 32usize, 32usize);
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        let mut kv: Vec<u8> = Vec::new();
        let mut n_kv = 0u64;
        let mut u32kv = |kv: &mut Vec<u8>, k: &str, v: u32| {
            put_str(kv, &format!("dflash.{k}"));
            kv.extend(4u32.to_le_bytes());
            kv.extend(v.to_le_bytes());
            n_kv += 1;
        };
        for (k, v) in [
            ("block_count", 1),
            ("embedding_length", H as u32),
            ("feed_forward_length", inter as u32),
            ("attention.head_count", 1),
            ("attention.kv_lora_rank", lat as u32),
            ("attention.key_length_mla", (dn + dr) as u32),
            ("attention.value_length_mla", vd as u32),
            ("rope.dimension_count", dr as u32),
            ("attention.sliding_window", 4),
            ("block_size", 3),
        ] {
            u32kv(&mut kv, k, v);
        }
        put_str(&mut kv, "dflash.target_layers");
        kv.extend(9u32.to_le_bytes());
        kv.extend(4u32.to_le_bytes());
        kv.extend(2u64.to_le_bytes());
        kv.extend(1u32.to_le_bytes());
        kv.extend(2u32.to_le_bytes());
        put_str(&mut kv, "general.architecture");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "dflash");
        n_kv += 2;
        // Tensors: (name, ggml ne, values).
        let ones = |n: usize| vec![1f32; n];
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            ("fc.weight".into(), vec![2 * H, H], fill(2 * H * H)),
            ("enc.output_norm.weight".into(), vec![H], ones(H)),
            ("output_norm.weight".into(), vec![H], ones(H)),
            ("mask_embd.weight".into(), vec![H, 2], fill(2 * H)),
        ];
        if seed.is_some() {
            tensors.push(("output.weight".into(), vec![H, V], fill(H * V)));
        }
        let mut w1 = vec![0f32; V * V];
        let mut w2 = vec![0f32; V * V];
        for p in 0..V {
            w1[p * V + p] = 1.0;
            w2[((p + 1) % V) * V + p] = 10.0;
        }
        tensors.push(("markov_w1.weight".into(), vec![V, V], w1));
        tensors.push(("markov_w2.weight".into(), vec![V, V], w2));
        for (n, ne) in [
            ("attn_norm.weight", vec![H]),
            ("attn_q.weight", vec![H, dn + dr]),
            ("attn_kv_a_mqa.weight", vec![H, lat + dr]),
            ("attn_kv_a_norm.weight", vec![lat]),
            ("attn_k_b.weight", vec![dn, lat, 1]),
            ("attn_v_b.weight", vec![lat, vd, 1]),
            ("attn_output.weight", vec![vd, H]),
            ("ffn_norm.weight", vec![H]),
            ("ffn_gate.weight", vec![H, inter]),
            ("ffn_up.weight", vec![H, inter]),
            ("ffn_down.weight", vec![inter, H]),
        ] {
            let len: usize = ne.iter().product();
            let v = if n.contains("norm") { ones(len) } else { fill(len) };
            tensors.push((format!("blk.0.{n}"), ne, v));
        }
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend(n_kv.to_le_bytes());
        b.extend(kv);
        let mut off = 0u64;
        for (name, ne, v) in &tensors {
            put_str(&mut b, name);
            b.extend((ne.len() as u32).to_le_bytes());
            for &d in ne {
                b.extend((d as u64).to_le_bytes());
            }
            b.extend(0u32.to_le_bytes());
            b.extend(off.to_le_bytes());
            off += (v.len() as u64 * 4).next_multiple_of(32);
        }
        b.resize(b.len().next_multiple_of(32), 0);
        for (_, _, v) in &tensors {
            let start = b.len();
            b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            b.resize(start + (v.len() * 4).next_multiple_of(32), 0);
        }
        std::fs::write(path, b).unwrap();
    }

    /// Greedy DSpark decoding commits exactly the plain greedy tokens, through rounds that accept
    /// every draft, some or none, and keeps the draft's context in step with the target.
    #[test]
    fn greedy_dspark_matches_plain_decoding() {
        let path = std::env::temp_dir().join(format!("oxinfer-dspark-test-{}.gguf", std::process::id()));
        toy_draft(&path, None);
        assert!(is_draft(&path));
        let files = ModelFiles::open(&path, true).unwrap();
        let d = Dspark::load(&path, &files, H, V, Quant::Bf16).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(d.cfg.max_draft(), 3);
        let toy = Toy;
        let prompt = [5u32, 9, 13];
        let mut plain_s = toy.new_session();
        let hs = crate::generate::prefill(&toy, &mut plain_s, &prompt, 2, &[]);
        let lg = toy.logits(&hs[hs.len() - H..], 1);
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        let plain = crate::generate::decode(&toy, &mut plain_s, lg, &mut s, &[], 40, &mut |_| Flow::Continue);
        for k in 1..=3 {
            let mut dec = DsparkDecoder::new(
                &toy,
                &d,
                DsparkOptions {
                    draft_tokens: k,
                    cost_aware: false,
                    fast_verify: false,
                    ..Default::default()
                },
            );
            let lg = dec.prefill(&prompt, 2);
            let spec = crate::generate::decode_rounds(&mut dec, &lg, &mut s, &[], 40, &mut |_| Flow::Continue);
            assert_eq!(spec.tokens, plain.tokens, "k = {k}");
            let dlen = dec.draft_session().len;
            assert_eq!(dlen, dec.sess.0.len());
            let st = &dec.stats;
            assert!(st.accepted > 0 && st.accepted < st.drafted, "k = {k}: {st:?}");
            assert!(st.per_round[k] > 0, "k = {k}: no fully accepted round: {:?}", st.per_round);
        }
    }

    /// The bounded context cache (window 4, slack 2) gives bit for bit the drafts of a cache that
    /// keeps every position, through injections of any size and truncations within the slack,
    /// and stays bounded: for an MLA draft and a sliding-window GQA one.
    #[test]
    fn bounded_context_cache_matches_full_cache() {
        let path = std::env::temp_dir().join(format!("oxinfer-dspark-window-{}.gguf", std::process::id()));
        toy_draft(&path, Some(7));
        let files = ModelFiles::open(&path, true).unwrap();
        let d = Dspark::load(&path, &files, H, V, Quant::Bf16).unwrap();
        std::fs::remove_file(&path).ok();
        bounded_cache_check(&d);
        let dir = std::env::temp_dir().join(format!("oxinfer-dspark-window-hf-{}", std::process::id()));
        toy_hf_draft(
            &dir,
            serde_json::json!({"layer_types": ["sliding_attention"], "sliding_window": 4}),
            Some(7),
        );
        let files = ModelFiles::open(&dir, true).unwrap();
        let d = Dspark::load(&dir, &files, H, V, Quant::Bf16).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert!(d.head.is_some() && d.embed.is_some());
        bounded_cache_check(&d);
    }

    fn bounded_cache_check(d: &Dspark) {
        assert_eq!(d.cfg.windows, [4]);
        let toy = Toy;
        let mut bounded = d.new_session();
        assert_eq!(bounded.keep, 4);
        bounded.slack = 2;
        let mut full = d.new_session();
        full.keep = 0;
        let n_feat = 2 * H;
        let mut rng = 1u64;
        let mut next = |m: u64| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) % m
        };
        let mut most = 0;
        let mut differ = false;
        let mut last: Option<Vec<f32>> = None;
        for step in 0..60 {
            let t = 1 + next(5) as usize;
            let feats: Vec<f32> = (0..t * n_feat).map(|_| next(1000) as f32 / 500.0 - 1.0).collect();
            d.inject(&mut bounded, &feats, t);
            d.inject(&mut full, &feats, t);
            if step % 3 == 2 {
                // A rejected tail: back by up to the slack.
                let back = (next(3) as usize).min(bounded.len - 1);
                bounded.truncate(bounded.len - back);
                full.truncate(full.len - back);
            }
            assert_eq!(bounded.len, full.len);
            // The window before `len` is always cached.
            assert!(bounded.len - bounded.start() >= bounded.len.min(4));
            most = most.max(bounded.len - bounded.start());
            let anchor = 3;
            let (lb, db) = d.draft_logits(&toy, &mut bounded, anchor, 3);
            let (lf, df) = d.draft_logits(&toy, &mut full, anchor, 3);
            assert_eq!(db, df, "step {step}");
            assert!(lb.iter().zip(&lf).all(|(a, b)| a.to_bits() == b.to_bits()), "step {step}");
            // The logits follow the context (the check is not vacuous).
            differ |= last.as_ref().is_some_and(|l| *l != lb);
            last = Some(lb);
        }
        assert!(differ);
        assert!(bounded.start() > 0 && most < 4 + 2 * 2, "start {} most {most}", bounded.start());
        assert!(bounded.bytes() < full.bytes() / 4);
        // A deep truncation keeps what is left; the block then sees fewer positions.
        let cut = bounded.start() - 1;
        bounded.truncate(cut);
        assert_eq!((bounded.start(), bounded.len), (cut, cut));
        d.inject(&mut bounded, &vec![0.5; n_feat], 1);
        assert_eq!(bounded.len - bounded.start(), 1);
        let (l, _) = d.draft_logits(&toy, &mut bounded, 3, 3);
        assert!(l.iter().all(|v| v.is_finite()));
    }

    /// Writes a Hugging Face DSpark draft folder (`config.json` + bf16 safetensors) with a
    /// one-layer Qwen3 GQA backbone of zero weights and the same Markov head as `toy_draft`
    /// (`p -> p + 1`), plus `extra` config keys. With `seed`, the draft is half as wide as the
    /// target, with random weights and its own embeddings and LM head, so its logits depend on
    /// the attended context.
    fn toy_hf_draft(dir: &Path, extra: serde_json::Value, seed: Option<u64>) {
        let (nh, nkv, hd, inter) = (2usize, 1usize, 16usize, 32usize);
        let dh = if seed.is_some() { H / 2 } else { H };
        let mut st = seed.unwrap_or(0);
        let mut fill = |len: usize| -> Vec<u16> {
            match seed {
                None => vec![0u16; len],
                Some(_) => (0..len)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let x = ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6;
                        (x.to_bits() >> 16) as u16
                    })
                    .collect(),
            }
        };
        std::fs::create_dir_all(dir).unwrap();
        let mut cfg = serde_json::json!({
            "architectures": ["Qwen3DSparkModel"], "model_type": "qwen3", "hidden_size": dh,
            "num_hidden_layers": 1, "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
            "intermediate_size": inter, "rms_norm_eps": 1e-6, "vocab_size": V, "block_size": 3,
            "target_layer_ids": [0, 1], "mask_token_id": V - 1, "markov_rank": V, "markov_head_type": "vanilla",
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "layer_types": ["full_attention"],
        });
        for (k, v) in extra.as_object().unwrap() {
            cfg[k] = v.clone();
        }
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let bf = |x: f32| (x.to_bits() >> 16) as u16;
        let ones = |n: usize| vec![bf(1.0); n];
        let zeros = |n: usize| vec![0u16; n];
        let mut w1 = zeros(V * V);
        let mut w2 = zeros(V * V);
        for p in 0..V {
            w1[p * V + p] = bf(1.0);
            w2[((p + 1) % V) * V + p] = bf(10.0);
        }
        let mut t: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
            ("fc.weight".into(), vec![dh, 2 * H], fill(2 * H * dh)),
            ("hidden_norm.weight".into(), vec![dh], ones(dh)),
            ("norm.weight".into(), vec![dh], ones(dh)),
            ("markov_head.markov_w1.weight".into(), vec![V, V], w1),
            ("markov_head.markov_w2.weight".into(), vec![V, V], w2),
            ("confidence_head.proj.weight".into(), vec![1, dh + V], zeros(dh + V)),
            ("confidence_head.proj.bias".into(), vec![1], ones(1)),
        ];
        if seed.is_some() {
            t.push(("embed_tokens.weight".into(), vec![V, dh], fill(V * dh)));
            t.push(("lm_head.weight".into(), vec![V, dh], fill(V * dh)));
        }
        for (n, s) in [
            ("input_layernorm.weight", vec![dh]),
            ("post_attention_layernorm.weight", vec![dh]),
            ("self_attn.q_norm.weight", vec![hd]),
            ("self_attn.k_norm.weight", vec![hd]),
            ("self_attn.q_proj.weight", vec![nh * hd, dh]),
            ("self_attn.k_proj.weight", vec![nkv * hd, dh]),
            ("self_attn.v_proj.weight", vec![nkv * hd, dh]),
            ("self_attn.o_proj.weight", vec![dh, nh * hd]),
            ("mlp.gate_proj.weight", vec![inter, dh]),
            ("mlp.up_proj.weight", vec![inter, dh]),
            ("mlp.down_proj.weight", vec![dh, inter]),
        ] {
            let len: usize = s.iter().product();
            let v = if n.contains("norm") { ones(len) } else { fill(len) };
            t.push((format!("layers.0.{n}"), s, v));
        }
        let mut header = serde_json::Map::new();
        let mut off = 0usize;
        for (name, shape, v) in &t {
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [off, off + 2 * v.len()]}),
            );
            off += 2 * v.len();
        }
        let hj = serde_json::to_vec(&header).unwrap();
        let mut b = (hj.len() as u64).to_le_bytes().to_vec();
        b.extend(hj);
        for (_, _, v) in &t {
            b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        std::fs::write(dir.join("model.safetensors"), b).unwrap();
    }

    /// The same check for Hugging Face GQA drafts: DeepSpec's layout (every slot predicts from
    /// the anchor), z-lab's DFlash layout (mask rows predict their own positions), and sliding,
    /// block-causal layers.
    #[test]
    fn greedy_dspark_gqa_matches_plain_decoding() {
        let toy = Toy;
        // Two prompt lengths: the rounds then start at every phase of the toy's 4-periodic rule.
        let prompts: [&[u32]; 2] = [&[5, 9, 13, 2, 7], &[5, 9, 13, 2, 7, 11]];
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        let plain: Vec<Vec<u32>> = prompts
            .iter()
            .map(|prompt| {
                let mut plain_s = toy.new_session();
                let hs = crate::generate::prefill(&toy, &mut plain_s, prompt, 2, &[]);
                let lg = toy.logits(&hs[hs.len() - H..], 1);
                crate::generate::decode(&toy, &mut plain_s, lg, &mut s, &[], 40, &mut |_| Flow::Continue).tokens
            })
            .collect();
        for (i, (extra, max_draft)) in [
            (serde_json::json!({}), 3),
            (serde_json::json!({"architectures": ["DFlashDraftModel"]}), 2),
            (
                serde_json::json!({"layer_types": ["sliding_attention"], "sliding_window": 2, "use_sliding_window": true}),
                3,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let dir = std::env::temp_dir().join(format!("oxinfer-dspark-hf-test-{}-{i}", std::process::id()));
            toy_hf_draft(&dir, extra, None);
            assert!(is_draft(&dir));
            let files = ModelFiles::open(&dir, true).unwrap();
            let d = Dspark::load(&dir, &files, H, V, Quant::Bf16).unwrap();
            std::fs::remove_dir_all(&dir).ok();
            assert_eq!(d.cfg.max_draft(), max_draft, "variant {i}");
            assert_eq!(d.cfg.target_layers, [1, 2]);
            if i == 2 {
                assert_eq!((d.cfg.windows[0], d.cfg.causal[0]), (2, true));
            }
            // Rounds that accepted no draft, some, and all, over the prompts and draft lengths.
            let (mut none, mut some, mut all) = (false, false, false);
            for (prompt, plain) in prompts.iter().zip(&plain) {
                for k in 1..=max_draft {
                    let mut dec = DsparkDecoder::new(
                        &toy,
                        &d,
                        DsparkOptions {
                            draft_tokens: k,
                            cost_aware: false,
                            fast_verify: false,
                            ..Default::default()
                        },
                    );
                    let lg = dec.prefill(prompt, 2);
                    let spec = crate::generate::decode_rounds(&mut dec, &lg, &mut s, &[], 40, &mut |_| Flow::Continue);
                    assert_eq!(&spec.tokens, plain, "variant {i}, k = {k}");
                    let dlen = dec.draft_session().len;
                    assert_eq!(dlen, dec.sess.0.len());
                    let st = &dec.stats;
                    assert!(st.accepted > 0, "variant {i}, k = {k}: {st:?}");
                    none |= st.per_round[0] > 0;
                    some |= st.per_round[1..k].iter().any(|&n| n > 0);
                    all |= st.per_round[k] > 0;
                }
            }
            assert!(
                none && some && all,
                "variant {i}: rounds with none / some / all drafts accepted: {none} {some} {all}"
            );
            // The confidence head (here sigmoid(1)) cuts every block before its first draft.
            let mut dec = DsparkDecoder::new(
                &toy,
                &d,
                DsparkOptions {
                    conf_min: 0.9,
                    fast_verify: false,
                    ..Default::default()
                },
            );
            let lg = dec.prefill(prompts[0], 2);
            let spec = crate::generate::decode_rounds(&mut dec, &lg, &mut s, &[], 40, &mut |_| Flow::Continue);
            assert_eq!(spec.tokens, plain[0]);
            assert_eq!(dec.stats.drafted, 0);
        }
    }

    /// Sampled DSpark decoding keeps the target's distribution: the frequencies of the first
    /// three sampled tokens match plain sampling's (the draft proposes `p + 1` chains, so some
    /// drafts are accepted, some rejected).
    #[test]
    fn sampled_dspark_keeps_the_target_distribution() {
        let dir = std::env::temp_dir().join(format!("oxinfer-dspark-hf-sample-{}", std::process::id()));
        toy_hf_draft(&dir, serde_json::json!({}), None);
        let files = ModelFiles::open(&dir, true).unwrap();
        let d = Dspark::load(&dir, &files, H, V, Quant::Bf16).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let toy = Toy;
        let prompt = [5u32, 9, 13];
        let n = 3000;
        let mut hist_plain = std::collections::HashMap::<(usize, u32), usize>::new();
        let mut hist_spec = hist_plain.clone();
        let (mut drafted, mut accepted) = (0, 0);
        for seed in 0..n {
            let mut s = Sampler::new(1.0, 0, 1.0, seed as u64 + 1);
            let mut ps = toy.new_session();
            let hs = crate::generate::prefill(&toy, &mut ps, &prompt, 4, &[]);
            let lg = toy.logits(&hs[hs.len() - H..], 1);
            let plain = crate::generate::decode(&toy, &mut ps, lg, &mut s, &[], 3, &mut |_| Flow::Continue);
            for (i, &t) in plain.tokens.iter().enumerate() {
                *hist_plain.entry((i, t)).or_default() += 1;
            }
            let mut s = Sampler::new(1.0, 0, 1.0, (seed + n) as u64 + 1);
            let mut dec = DsparkDecoder::new(&toy, &d, DsparkOptions::default());
            let lg = dec.prefill(&prompt, 4);
            let spec = crate::generate::decode_rounds(&mut dec, &lg, &mut s, &[], 3, &mut |_| Flow::Continue);
            assert_eq!(spec.tokens.len(), 3);
            for (i, &t) in spec.tokens.iter().enumerate() {
                *hist_spec.entry((i, t)).or_default() += 1;
            }
            drafted += dec.stats.drafted;
            accepted += dec.stats.accepted;
        }
        assert!(accepted > 0 && accepted < drafted, "{accepted} / {drafted}");
        for (k, &c) in hist_plain.iter().chain(hist_spec.iter()) {
            let a = *hist_plain.get(k).unwrap_or(&0) as f64 / n as f64;
            let b = *hist_spec.get(k).unwrap_or(&0) as f64 / n as f64;
            assert!((a - b).abs() < 0.035, "token {k:?}: plain {a} vs DSpark {b} ({c})");
        }
    }

    /// The cost-aware policy: drafts the measured rates favour, plain steps (with ever rarer
    /// probes) where drafting loses, and per block the most profitable verification length.
    #[test]
    fn adaptive_policy() {
        let mut a = Adaptive::default();
        // Two draft rounds, two plain ones to measure, then the rates decide.
        assert_eq!(a.choose(7), 7);
        a.record(&[0.99; 7], 7, 7, 8, 0.010, 0.020);
        assert_eq!(a.choose(7), 7);
        a.record(&[0.99; 7], 7, 7, 8, 0.010, 0.020);
        assert_eq!(a.choose(7), 0);
        a.record(&[], 0, 0, 1, 0.0, 0.010);
        assert_eq!(a.choose(7), 0);
        a.record(&[], 0, 0, 1, 0.0, 0.010);
        // 8 tokens in 30 ms beats 1 in 10 ms.
        assert_eq!(a.choose(7), 7);
        // Verification costs 10 ms plus 10 ms per draft: confident drafts are all verified, a
        // block whose drafts are all unlikely is not verified at all.
        a.verify = (0..=7).map(|k| (0.010 + 0.010 * k as f64, 2)).collect();
        assert_eq!(a.cut(&[0.999; 7], 0.010), 7);
        assert_eq!(a.cut(&[0.05; 7], 0.010), 0);
        assert_eq!(a.cut(&[0.999, 0.999, 0.05, 0.999, 0.999, 0.999, 0.999], 0.010), 2);
        // Draft rounds that lose: plain steps, with a probe after 6, then after 12.
        let mut a = Adaptive::default();
        for _ in 0..2 {
            a.choose(7);
            a.record(&[0.5; 7], 7, 0, 1, 0.010, 0.080);
        }
        for _ in 0..2 {
            a.choose(7);
            a.record(&[], 0, 0, 1, 0.0, 0.010);
        }
        let mut drafts = Vec::new();
        for r in 0..30 {
            let k = a.choose(7);
            if k > 0 {
                drafts.push(r);
                a.record(&[0.5; 7], 1, 0, 1, 0.010, 0.015);
            } else {
                a.record(&[], 0, 0, 1, 0.0, 0.010);
            }
        }
        assert_eq!(drafts, [4, 17], "probes at {drafts:?}");
    }

    /// Greedy DSpark decoding with the cost-aware policy (rounds with and without drafts, blocks
    /// cut by the calibrated estimate) gives the plain greedy tokens.
    #[test]
    fn cost_aware_dspark_matches_plain_decoding() {
        let dir = std::env::temp_dir().join(format!("oxinfer-dspark-hf-adaptive-{}", std::process::id()));
        toy_hf_draft(&dir, serde_json::json!({}), Some(7));
        let files = ModelFiles::open(&dir, true).unwrap();
        let d = Dspark::load(&dir, &files, H, V, Quant::Bf16).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let toy = Toy;
        let prompt = [5u32, 9, 13, 2, 7];
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        let mut ps = toy.new_session();
        let hs = crate::generate::prefill(&toy, &mut ps, &prompt, 2, &[]);
        let lg = toy.logits(&hs[hs.len() - H..], 1);
        let plain = crate::generate::decode(&toy, &mut ps, lg, &mut s, &[], 120, &mut |_| Flow::Continue).tokens;
        let mut dec = DsparkDecoder::new(&toy, &d, DsparkOptions::default());
        let lg = dec.prefill(&prompt, 2);
        let spec = crate::generate::decode_rounds(&mut dec, &lg, &mut s, &[], 120, &mut |_| Flow::Continue);
        assert_eq!(spec.tokens, plain);
        let dlen = dec.draft_session().len;
        assert_eq!(dlen, dec.sess.0.len());
    }

    /// Finding a draft: names of cached and GGUF models, the known drafts in the download cache,
    /// GGUF draft repositories.
    #[test]
    fn draft_discovery() {
        assert!(["Q8_0", "q4_k_m", "IQ4_XS", "BF16", "UD"].iter().all(|t| is_quant_tag(t)));
        assert!(!["Qwen3", "8B", "it", "Instruct", "Q"].iter().any(|t| is_quant_tag(t)));
        let root = std::env::temp_dir().join(format!("oxinfer-dspark-find-{}", std::process::id()));
        let model = root.join("Qwen--Qwen3-4B-oxinfer-q4");
        std::fs::create_dir_all(&model).unwrap();
        let gguf_dir = root.join("Qwen--Qwen3-8B-GGUF");
        std::fs::create_dir_all(&gguf_dir).unwrap();
        let gguf = gguf_dir.join("Qwen3-8B-Q8_0.gguf");
        std::fs::write(&gguf, b"GGUF\x03\0\0\0").unwrap();
        assert_eq!(model_names(&gguf), ["Qwen3-8B", "Qwen3-8B"]);
        assert_eq!(auto_repo(&gguf), Some("deepseek-ai/dspark_qwen3_8b_block7"));
        assert_eq!(auto_repo(&model), Some("deepseek-ai/dspark_qwen3_4b_block7"));
        assert!(draft_for(&model).is_none());
        // The known draft in the cache: found, and the default the table says.
        let draft = root.join("deepseek-ai--dspark_qwen3_4b_block7");
        toy_hf_draft(&draft, serde_json::json!({}), None);
        assert_eq!(draft_for(&model).as_deref(), Some(draft.as_path()));
        let def = KNOWN
            .iter()
            .any(|k| k.1 == "deepseek-ai/dspark_qwen3_4b_block7" && k.2 == ByDefault::Yes);
        assert_eq!(default_draft(&model).is_some(), def);
        assert_eq!(draft_in(&draft).as_deref(), Some(draft.as_path()));
        std::fs::remove_dir_all(&root).ok();
    }
}
