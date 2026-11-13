//! GLM-5.3 text models, as transformers' `Glm5NextTextModel` (`glm5_next`: zai-org/GLM-5.3-Flash,
//! nex-agi/Nex-N2.6; llama.cpp `glm5-next`) and `GlmMoeDsaModel` (`glm_moe_dsa`: zai-org/GLM-5.3;
//! llama.cpp `glm-dsa`). One implementation runs both; what a checkpoint has comes from its config.
//!
//! GLM-5.3-Flash:
//!
//! - Manifold-constrained hyper-connections: the residual stream is `hc_mult` (4) copies; before
//!   the attention and before the MLP a projection of the RMS-scaled copies gives the `pre` /
//!   `post` / `comb` weights (Sinkhorn-normalized `comb`). This is DeepSeek V4's arithmetic and
//!   runs on [`crate::deepseek4`]'s (`hc_pre`, `hc_post`); the final collapse is a plain mean.
//! - KDA linear attention (`layer_types[l] == "linear_attention"`, 34 of 45 layers): `q` / `k` /
//!   `v` projections through a depthwise causal convolution (kernel 4) and SiLU; `q` and `k`
//!   L2-normalized per head; a matrix state per head updated by the delta rule with a decay per
//!   key channel, `exp(lower_bound * sigmoid(exp(A_log) * (f_b f_a x + dt_bias)))`, and a write
//!   strength `sigmoid(b x)`; the read `S^T q` RMS-normed and gated by `sigmoid(g_b g_a x)`.
//!   No softmax, no cache that grows: the state is `heads * head_dim^2` floats.
//! - MLA attention without positions (NoPE) on the other layers: queries through a LoRA (`q_a`,
//!   RMS norm, `q_b`), keys and values expanded per head from a cached latent (`kv_lora_rank`).
//!   `kv_b` is absorbed into the query and the output, so only the latent is cached and scored.
//! - The DSA indexer of those layers scores pools of `index_kpool` (4) consecutive tokens: each
//!   pool's key is a per-channel softmax (gate projection plus a learned position bias) over its
//!   tokens' LayerNorm'd keys; `sum_h w_h relu(q_h . pool)` ranks the complete pools a query
//!   sees, the best `index_topk / index_kpool` are attended token by token, plus the tokens of
//!   the incomplete pool at the query's end. Up to `index_topk` complete-pool tokens every pool
//!   is taken, so short contexts are dense causal attention.
//! - Three dense layers, then MoE: sigmoid router with a selection bias, top 8 of 288 renormalized
//!   and scaled, one shared expert; every SwiGLU clamps its inputs (`swiglu_limit`).
//!
//! GLM-5.3 has no hyper-connections and no KDA: every layer is MLA with a rotary part (the last
//! `qk_rope_head_dim` dims of each query head against one shared rotary key, interleaved pairs),
//! the indexer scores single tokens (its first `qk_rope_head_dim` dims rotated) and "shared"
//! layers reuse the choice of the last "full" one; the SwiGLUs do not clamp.
//!
//! Not here: the vision tower (`model.visual.*`), and the MTP layer (`layers.{n}`, `eh_proj`),
//! which transformers does not run either.
//!
//! Checkpoints: the release's safetensors (FP8 block-quantized matrices with
//! `weight_scale_inv`, one module per expert; what `save_pretrained` writes too) and llama.cpp
//! GGUF files, named onto the same tensors by [`gguf_name`].
//!
//! Runtime interventions ([`crate::intervene`]): residual points as DeepSeek V4 on the copies
//! (the mean is read, edits go to every copy) or on the single stream; neurons are the input of
//! `down_proj` of the dense MLP, the shared expert and the routed experts.

use crate::config::DeepSeek4Config;
use crate::deepseek4::{Hc, hc_post, hc_pre};
use crate::intervene::{NeuronHook, Target};
use crate::kernels::*;
use crate::model::{Lin, Quant, Table, linear, linear_multi, load_lin, table_quant};
use crate::moe::{Act, Experts, NeedExperts, route_sigmoid};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::Path;

/// The indexer's LayerNorm epsilon (`nn.LayerNorm(head_dim, eps=1e-6)`).
const INDEX_NORM_EPS: f32 = 1e-6;

/// Whether `model_type` is one of this family's.
pub fn is_model_type(mt: &str) -> bool {
    matches!(mt, "glm5_next" | "glm5_next_text" | "glm_moe_dsa")
}

/// The settings of a GLM-5.3 text model (`text_config` of a `glm5_next` checkpoint, the root of a
/// `glm_moe_dsa` one, or a GGUF header through [`gguf_config`]).
#[derive(Debug, Clone)]
pub struct Glm5Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f64,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub index_topk: usize,
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    /// Tokens per indexer pool (1: the indexer scores single tokens).
    pub index_kpool: usize,
    /// Per layer: KDA linear attention (otherwise MLA).
    pub kda: Vec<bool>,
    /// Per layer: MoE (otherwise a dense MLP).
    pub sparse: Vec<bool>,
    /// Per layer: the indexer mode the config gives ("full"); the loader goes by the tensors.
    pub indexer_full: Vec<bool>,
    pub kda_heads: usize,
    pub kda_head_dim: usize,
    pub kda_kernel: usize,
    /// The forget gate's lower bound (None: `-exp(A_log) softplus(g)`).
    pub kda_lower_bound: Option<f64>,
    /// Clamp of every SwiGLU's inputs (0: none).
    pub swiglu_limit: f64,
    /// Residual copies (1: a plain residual stream).
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
}

impl Glm5Config {
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
        let t = raw.get("text_config").filter(|v| v.is_object()).unwrap_or(raw);
        let mt = t
            .get("model_type")
            .or_else(|| raw.get("model_type"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let flash = match mt {
            "glm5_next" | "glm5_next_text" => true,
            "glm_moe_dsa" => false,
            other => bail!("model_type {other:?} is not a GLM-5.3 model"),
        };
        let need = |k: &str| -> Result<usize> {
            t.get(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .with_context(|| format!("GLM-5.3 config has no {k}"))
        };
        let opt = |k: &str, d: usize| t.get(k).and_then(Value::as_u64).map_or(d, |v| v as usize);
        let optf = |k: &str, d: f64| t.get(k).and_then(Value::as_f64).unwrap_or(d);
        let strs = |k: &str| -> Option<Vec<String>> {
            t.get(k)?
                .as_array()
                .map(|a| a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect())
        };
        let n = need("num_hidden_layers")?;
        let kda: Vec<bool> = match strs("layer_types") {
            Some(v) => v.iter().map(|s| s == "linear_attention").collect(),
            None => (0..n).map(|i| flash && i % 4 != 3).collect(),
        };
        let sparse: Vec<bool> = match strs("mlp_layer_types") {
            Some(v) => v.iter().map(|s| s == "sparse").collect(),
            None => {
                let d = opt("first_k_dense_replace", 3).min(n);
                (0..n).map(|i| i >= d).collect()
            }
        };
        let indexer_full: Vec<bool> = match strs("indexer_types") {
            Some(v) => v.iter().map(|s| s != "shared").collect(),
            None => match t.get("index_topk_pattern").and_then(Value::as_str) {
                Some(p) => p.chars().map(|c| c != 'S').collect(),
                None => {
                    let freq = opt("index_topk_freq", 1).max(1);
                    let off = opt("index_skip_topk_offset", 2);
                    (0..n).map(|i| (i + 1).saturating_sub(off).is_multiple_of(freq)).collect()
                }
            },
        };
        if kda.len() < n || sparse.len() < n || indexer_full.len() < n {
            bail!("GLM-5.3 config: the per-layer lists are shorter than num_hidden_layers ({n})");
        }
        let la = t.get("linear_attn_config").filter(|v| v.is_object());
        let lin = |k: &str, flat: &str, d: usize| {
            la.and_then(|v| v.get(k))
                .or_else(|| t.get(flat))
                .and_then(Value::as_u64)
                .map_or(d, |v| v as usize)
        };
        let nh = need("num_attention_heads")?;
        // `safe_gate` (the default) gives -5 when no bound is set.
        let lower = match la.and_then(|v| v.get("gate_lower_bound")).or_else(|| t.get("linear_lower_bound")) {
            Some(Value::Null) | None => Some(-5.0),
            Some(v) => v.as_f64(),
        };
        let c = Glm5Config {
            hidden_size: need("hidden_size")?,
            num_hidden_layers: n,
            num_attention_heads: nh,
            vocab_size: need("vocab_size")?,
            intermediate_size: need("intermediate_size")?,
            moe_intermediate_size: opt("moe_intermediate_size", 2048),
            n_routed_experts: opt("n_routed_experts", opt("num_experts", 0)),
            n_shared_experts: opt("n_shared_experts", 1),
            num_experts_per_tok: opt("num_experts_per_tok", 8),
            routed_scaling_factor: optf("routed_scaling_factor", 2.5),
            kv_lora_rank: need("kv_lora_rank")?,
            q_lora_rank: need("q_lora_rank")?,
            qk_rope_head_dim: opt("qk_rope_head_dim", if flash { 0 } else { 64 }),
            qk_nope_head_dim: need("qk_nope_head_dim")?,
            v_head_dim: need("v_head_dim")?,
            index_topk: opt("index_topk", 2048),
            index_head_dim: opt("index_head_dim", 128),
            index_n_heads: opt("index_n_heads", 32),
            index_kpool: if flash { opt("index_kpool", 16) } else { 1 },
            kda,
            sparse,
            indexer_full,
            kda_heads: lin("num_heads", "linear_num_heads", nh),
            kda_head_dim: lin("head_dim", "linear_head_dim", 128),
            kda_kernel: lin("short_conv_kernel_size", "linear_conv_kernel_dim", 4),
            kda_lower_bound: lower,
            swiglu_limit: optf("swiglu_limit", if flash { 10.0 } else { 0.0 }),
            hc_mult: if flash { opt("hc_mult", 4) } else { 1 },
            hc_sinkhorn_iters: opt("hc_sinkhorn_iters", 20),
            hc_eps: optf("hc_eps", 1e-6),
            rms_norm_eps: optf("rms_norm_eps", 1e-5),
            rope_theta: t
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .or_else(|| t.get("rope_theta"))
                .and_then(Value::as_f64)
                .unwrap_or(10000.0),
            tie_word_embeddings: t.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
        };
        if c.sparse.iter().take(n).any(|&s| s) && c.n_routed_experts == 0 {
            bail!("GLM-5.3 config has no n_routed_experts");
        }
        if opt("n_group", 1) != 1 || opt("topk_group", 1) != 1 {
            bail!("GLM-5.3: expert groups are not supported");
        }
        if t.get("norm_topk_prob").and_then(Value::as_bool) == Some(false) {
            bail!("GLM-5.3: norm_topk_prob = false is not supported");
        }
        if matches!(t.get("scoring_func").and_then(Value::as_str), Some(s) if s != "sigmoid") {
            bail!("GLM-5.3: only the sigmoid router is supported");
        }
        if t.get("index_kpool_always_select_tail").and_then(Value::as_bool) == Some(false) && c.index_kpool > 1 {
            bail!("GLM-5.3: index_kpool_always_select_tail = false is not supported");
        }
        if c.index_kpool == 0 || !c.index_topk.is_multiple_of(c.index_kpool) {
            bail!(
                "GLM-5.3: index_topk ({}) must be a multiple of index_kpool ({})",
                c.index_topk,
                c.index_kpool
            );
        }
        if !c.qk_rope_head_dim.is_multiple_of(2) || c.qk_rope_head_dim > c.index_head_dim || c.hc_mult == 0 || c.kda_kernel == 0 {
            bail!("GLM-5.3: bad rotary, hyper-connection or convolution settings");
        }
        if let Some(r) = t.get("rope_parameters").and_then(|r| r.get("rope_type")).and_then(Value::as_str)
            && r != "default"
        {
            bail!("GLM-5.3: rope_type {r:?} is not supported");
        }
        Ok(c)
    }

    fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// The settings [`crate::deepseek4`]'s hyper-connection arithmetic reads, as the config type
    /// it takes (the other fields are placeholders it does not look at).
    fn mhc(&self) -> Result<DeepSeek4Config> {
        DeepSeek4Config::from_json(json!({
            "hidden_size": self.hidden_size,
            "rms_norm_eps": self.rms_norm_eps,
            "hc_mult": self.hc_mult,
            "hc_sinkhorn_iters": self.hc_sinkhorn_iters,
            "hc_eps": self.hc_eps,
            "num_hidden_layers": 0, "num_attention_heads": 1, "head_dim": 2, "qk_rope_head_dim": 0,
            "q_lora_rank": 1, "o_groups": 1, "o_lora_rank": 1, "vocab_size": 1, "n_routed_experts": 1,
            "num_experts_per_tok": 1, "moe_intermediate_size": 1, "compress_ratios": [],
            "index_n_heads": 1, "index_head_dim": 2, "index_topk": 1,
        }))
        .context("GLM-5.3 hyper-connection settings")
    }
}

/// The Hugging Face config (the keys [`Glm5Config::from_json`] reads) of a llama.cpp `glm5-next`
/// or `glm-dsa` header.
pub(crate) fn gguf_config(h: &crate::gguf::Header) -> Result<Value> {
    let arch = h.arch();
    let flash = arch == "glm5-next";
    let get = |k: &str| h.get(&format!("{arch}.{k}"));
    let need = |k: &str| -> Result<u64> {
        get(k)
            .and_then(crate::gguf::Value::as_u64)
            .with_context(|| format!("GGUF metadata has no {arch}.{k}"))
    };
    let opt = |k: &str| get(k).and_then(crate::gguf::Value::as_u64);
    let optf = |k: &str| get(k).and_then(crate::gguf::Value::as_f64);
    let flag = |v: &crate::gguf::Value| match v {
        crate::gguf::Value::Bool(b) => Some(*b),
        _ => None,
    };
    let n = h.n_layers();
    if matches!(opt("expert_gating_func"), Some(g) if g != 2) {
        bail!("GGUF {arch}: only the sigmoid expert gating function is supported");
    }
    // Per layer: an array (one entry per block, MTP blocks after the model's) or one value.
    let per_layer = |k: &str| -> Option<Vec<crate::gguf::Value>> {
        match get(k)? {
            crate::gguf::Value::Array(a) if a.len() >= n => Some(a[..n].to_vec()),
            crate::gguf::Value::Array(_) => None,
            v => Some(vec![v.clone(); n]),
        }
    };
    let lat = need("attention.kv_lora_rank")?;
    let dr = opt("rope.dimension_count").unwrap_or(0);
    let qk = need("attention.key_length_mla")?;
    let dense = opt("leading_dense_block_count").unwrap_or(3) as usize;
    let mut j = json!({
        "model_type": if flash { "glm5_next_text" } else { "glm_moe_dsa" },
        "hidden_size": need("embedding_length")?,
        "num_hidden_layers": n,
        "num_attention_heads": need("attention.head_count")?,
        "vocab_size": opt("vocab_size")
            .or_else(|| h.get("tokenizer.ggml.tokens").and_then(crate::gguf::Value::as_array).map(|a| a.len() as u64))
            .context("GGUF has no vocabulary size")?,
        "intermediate_size": need("feed_forward_length")?,
        "moe_intermediate_size": need("expert_feed_forward_length")?,
        "n_routed_experts": need("expert_count")?,
        "n_shared_experts": opt("expert_shared_count").unwrap_or(1),
        "num_experts_per_tok": need("expert_used_count")?,
        "routed_scaling_factor": optf("expert_weights_scale").unwrap_or(1.0),
        "norm_topk_prob": get("expert_weights_norm").and_then(flag).unwrap_or(true),
        "kv_lora_rank": lat,
        "q_lora_rank": need("attention.q_lora_rank")?,
        "qk_rope_head_dim": dr,
        "qk_nope_head_dim": qk.checked_sub(dr).context("GGUF: key_length_mla below the rotary size")?,
        "v_head_dim": need("attention.value_length_mla")?,
        "index_topk": need("attention.indexer.top_k")?,
        "index_head_dim": need("attention.indexer.key_length")?,
        "index_n_heads": need("attention.indexer.head_count")?,
        "mlp_layer_types": (0..n).map(|i| if i < dense { "dense" } else { "sparse" }).collect::<Vec<_>>(),
        "rms_norm_eps": optf("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
        "rope_theta": optf("rope.freq_base").unwrap_or(10000.0),
        "tie_word_embeddings": false,
    });
    let o = j.as_object_mut().unwrap();
    if let Some(v) = per_layer("attention.indexer.types") {
        let types: Vec<&str> = v.iter().map(|x| if flag(x).unwrap_or(true) { "full" } else { "shared" }).collect();
        o.insert("indexer_types".into(), json!(types));
    } else if !flash {
        // GLM-5.3's schedule (the header does not carry it; the loader goes by the tensors).
        o.insert("index_topk_freq".into(), json!(4));
        o.insert("index_skip_topk_offset".into(), json!(3));
    }
    if flash {
        // A KDA layer has no key-value heads.
        let kv = per_layer("attention.head_count_kv").context("GGUF glm5-next has no per-layer attention.head_count_kv")?;
        let types: Vec<&str> = kv
            .iter()
            .map(|x| {
                if x.as_u64() == Some(0) {
                    "linear_attention"
                } else {
                    "indexed_attention"
                }
            })
            .collect();
        o.insert("layer_types".into(), json!(types));
        o.insert("index_kpool".into(), json!(opt("attention.indexer.kpool").unwrap_or(1)));
        o.insert(
            "index_kpool_always_select_tail".into(),
            json!(get("attention.indexer.kpool_select_tail").and_then(flag).unwrap_or(true)),
        );
        o.insert("linear_head_dim".into(), json!(need("kda.head_dim")?));
        o.insert("linear_conv_kernel_dim".into(), json!(need("ssm.conv_kernel")?));
        o.insert("linear_lower_bound".into(), json!(optf("kda.gate_lower_bound")));
        o.insert("hc_mult".into(), json!(need("hyper_connection.count")?));
        o.insert(
            "hc_sinkhorn_iters".into(),
            json!(opt("hyper_connection.sinkhorn_iterations").unwrap_or(20)),
        );
        o.insert("hc_eps".into(), json!(optf("hyper_connection.epsilon").unwrap_or(1e-6)));
        let first = |k: &str| match get(k) {
            Some(crate::gguf::Value::Array(a)) => a.first().and_then(crate::gguf::Value::as_f64),
            Some(v) => v.as_f64(),
            None => None,
        };
        o.insert("swiglu_limit".into(), json!(first("swiglu_clamp_exp").unwrap_or(0.0)));
    }
    Ok(j)
}

/// The release's name and PyTorch-order shape of a llama.cpp `glm5-next` / `glm-dsa` tensor
/// (`ne`: ggml's shape); None for tensors not read (the MTP block among them).
pub(crate) fn gguf_name(name: &str, ne: &[usize], n_layers: usize) -> Option<(String, Vec<usize>)> {
    let shape: Vec<usize> = ne.iter().rev().copied().collect();
    let top = match name {
        "token_embd.weight" => Some("model.embed_tokens.weight"),
        "output.weight" => Some("lm_head.weight"),
        "output_norm.weight" => Some("model.norm.weight"),
        _ => None,
    };
    if let Some(t) = top {
        return Some((t.to_string(), shape));
    }
    let rest = name.strip_prefix("blk.")?;
    let (l, rest) = rest.split_once('.')?;
    let l: usize = l.parse().ok()?;
    if l >= n_layers {
        return None;
    }
    let n = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "hc_attn_fn.weight" => "hc_attn_fn",
        "hc_attn_base.weight" => "hc_attn_base",
        "hc_attn_scale.weight" => "hc_attn_scale",
        "hc_ffn_fn.weight" => "hc_ffn_fn",
        "hc_ffn_base.weight" => "hc_ffn_base",
        "hc_ffn_scale.weight" => "hc_ffn_scale",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        // llama.cpp stores `-exp(A_log)` (as for Qwen3.5).
        "ssm_a" => "self_attn.ssm_a",
        "ssm_dt.bias" => "self_attn.dt_bias",
        "ssm_beta.weight" => "self_attn.b_proj.weight",
        "ssm_f_a.weight" => "self_attn.f_a_proj.weight",
        "ssm_f_b.weight" => "self_attn.f_b_proj.weight",
        "ssm_g_a.weight" => "self_attn.g_a_proj.weight",
        "ssm_g_b.weight" => "self_attn.g_b_proj.weight",
        "ssm_norm.weight" => "self_attn.o_norm.weight",
        "ssm_conv1d_q.weight" => "self_attn.q_conv1d.weight",
        "ssm_conv1d_k.weight" => "self_attn.k_conv1d.weight",
        "ssm_conv1d_v.weight" => "self_attn.v_conv1d.weight",
        "attn_q_a.weight" => "self_attn.q_a_proj.weight",
        "attn_q_a_norm.weight" => "self_attn.q_a_layernorm.weight",
        "attn_q_b.weight" => "self_attn.q_b_proj.weight",
        "attn_kv_a_mqa.weight" => "self_attn.kv_a_proj_with_mqa.weight",
        "attn_kv_a_norm.weight" => "self_attn.kv_a_layernorm.weight",
        // `kv_b` split per head: `[heads, latent, nope]` (the key rows transposed) and
        // `[heads, value, latent]`.
        "attn_k_b.weight" => "self_attn.k_b",
        "attn_v_b.weight" => "self_attn.v_b",
        "indexer.attn_k.weight" => "self_attn.indexer.wk.weight",
        "indexer.attn_q_b.weight" => "self_attn.indexer.wq_b.weight",
        "indexer.k_norm.weight" => "self_attn.indexer.k_norm.weight",
        "indexer.k_norm.bias" => "self_attn.indexer.k_norm.bias",
        "indexer.proj.weight" => "self_attn.indexer.weights_proj.weight",
        "indexer_compressor_ape.weight" => "self_attn.indexer.index_kpool_compress_ape",
        "indexer_compressor_gate.weight" => "self_attn.indexer.index_kpool_compress_gate",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        "exp_probs_b.bias" => "mlp.gate.e_score_correction_bias",
        "ffn_gate_exps.weight" => "mlp.experts.gate_proj",
        "ffn_up_exps.weight" => "mlp.experts.up_proj",
        "ffn_down_exps.weight" => "mlp.experts.down_proj",
        "ffn_gate_shexp.weight" => "mlp.shared_experts.gate_proj.weight",
        "ffn_up_shexp.weight" => "mlp.shared_experts.up_proj.weight",
        "ffn_down_shexp.weight" => "mlp.shared_experts.down_proj.weight",
        _ => return None,
    };
    Some((format!("model.layers.{l}.{n}"), shape))
}

/// Weight streaming for a checkpoint that does not fit in memory (as Mistral Large 4).
pub fn default_stream() -> crate::stream::StreamOptions {
    crate::mistral4::default_stream()
}

/// Whether a checkpoint of `bytes` is too big to hold in memory (over 60% of RAM).
pub fn too_big(bytes: u64) -> bool {
    crate::mistral4::too_big(bytes)
}

struct Mlp<'a> {
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
    inter: usize,
}

struct Moe<'a> {
    router: Vec<f32>, // [n_exp, H]
    bias: Vec<f32>,   // [n_exp]
    experts: Experts<'a>,
    shared: Mlp<'a>,
}

enum Ffn<'a> {
    Dense(Mlp<'a>),
    Moe(Moe<'a>),
}

/// The k-pool compressor of a GLM-5.3-Flash indexer.
struct Pool<'a> {
    gate: Lin<'a>, // [D, H]
    ape: Vec<f32>, // [kpool, D]
}

struct Indexer<'a> {
    wq: Lin<'a>,                  // [n_heads * D, q_lora]
    wk: Lin<'a>,                  // [D, H]
    k_norm: (Vec<f32>, Vec<f32>), // LayerNorm [D]
    proj: Lin<'a>,                // [n_heads, H]
    pool: Option<Pool<'a>>,
}

struct Mla<'a> {
    q_a: Lin<'a>, // [q_lora, H]
    q_norm: Vec<f32>,
    q_b: Lin<'a>,  // [nh * (dn + dr), q_lora]
    kv_a: Lin<'a>, // [lat + dr, H]
    kv_norm: Vec<f32>,
    /// Per head: `k_b[h]` `[lat, dn]` maps a non-rotary query onto the latent (the transpose of
    /// `kv_b_proj`'s key rows), `v_b[h]` `[v, lat]` the attended latent onto the value.
    k_b: Vec<Lin<'a>>,
    v_b: Vec<Lin<'a>>,
    o: Lin<'a>, // [H, nh * v]
    indexer: Option<Indexer<'a>>,
}

struct Kda<'a> {
    q: Lin<'a>, // [nh * d, H]
    k: Lin<'a>,
    v: Lin<'a>,
    /// The depthwise kernels of the `q | k | v` channels, `[kernel][3 * nh * d]`.
    conv: Vec<f32>,
    f_a: Lin<'a>, // [d, H]
    f_b: Lin<'a>, // [nh * d, d]
    dt_bias: Vec<f32>,
    /// `exp(A_log)` per head.
    rate: Vec<f32>,
    b: Lin<'a>,   // [nh, H]
    g_a: Lin<'a>, // [d, H]
    g_b: Lin<'a>, // [nh * d, d]
    o_norm: Vec<f32>,
    o: Lin<'a>, // [H, nh * d]
}

// One per layer, built once.
#[allow(clippy::large_enum_variant)]
enum Attn<'a> {
    Kda(Kda<'a>),
    Mla(Mla<'a>),
}

struct Layer<'a> {
    ln_in: Vec<f32>,
    ln_post: Vec<f32>,
    /// The mixers before the attention and before the MLP (GLM-5.3-Flash).
    hc: Option<(Hc, Hc)>,
    attn: Attn<'a>,
    ffn: Ffn<'a>,
}

/// The decoding state of one layer.
#[derive(Clone)]
enum LayerState {
    /// The last `kernel - 1` rows of `q | k | v` before the convolution, and each head's
    /// `[d, d]` state (key rows, value columns).
    Kda {
        conv: Vec<f32>,
        s: Vec<f32>,
    },
    Mla(MlaState),
}

#[derive(Clone, Default)]
struct MlaState {
    /// Per position: the normed latent `[lat]` and the rotated rotary key `[dr]`.
    lat: Vec<f32>,
    rot: Vec<f32>,
    /// What the indexer scores: one key `[D]` per token, or per complete pool.
    keys: Vec<f32>,
    /// The keys and gate logits `[D]` of the tokens of the incomplete pool.
    pend_k: Vec<f32>,
    pend_g: Vec<f32>,
}

#[derive(Clone)]
pub struct Glm5Session {
    layers: Vec<LayerState>,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
}

impl Glm5Session {
    /// Drops cached positions at or beyond `pos`. Only a model without recurrent layers and
    /// without indexer pools (GLM-5.3) can go back; returns whether it did.
    pub fn truncate(&mut self, pos: usize) -> bool {
        if pos >= self.pos {
            return true;
        }
        let ok = self
            .layers
            .iter()
            .all(|l| matches!(l, LayerState::Mla(m) if m.pend_k.is_empty() && m.pend_g.is_empty()));
        if !ok {
            return false;
        }
        let n = self.pos;
        for l in &mut self.layers {
            if let LayerState::Mla(m) = l {
                for v in [&mut m.lat, &mut m.rot, &mut m.keys] {
                    let w = v.len() / n;
                    v.truncate(pos * w);
                }
            }
        }
        self.pos = pos;
        true
    }
}

/// The keys each query of a pass attends to: None = every earlier key (causal), or the sorted
/// positions the indexer chose.
type Selection = Vec<Option<Vec<u32>>>;

pub struct Glm5Model<'a> {
    pub cfg: Glm5Config,
    pub quant: Quant,
    embed: Table<'a>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
    /// The hyper-connection settings as [`crate::deepseek4`] takes them.
    mhc: DeepSeek4Config,
    /// Rotary inverse frequencies (`qk_rope_head_dim / 2`; empty without positions).
    inv_freq: Vec<f32>,
    /// Weight streaming: unit `l` is layer `l`, unit `num_hidden_layers` the LM head.
    stream: Option<crate::stream::StreamHook>,
}

struct Loader<'a, 'c> {
    files: &'a ModelFiles,
    cfg: &'c Glm5Config,
    quant: Quant,
    /// `model.language_model.` (a `glm5_next` checkpoint) or `model.`.
    prefix: &'static str,
}

impl<'a> Loader<'a, '_> {
    fn getf(&self, full: &str, n: usize) -> Result<Vec<f32>> {
        let info = self.files.info(full).with_context(|| format!("missing tensor {full}"))?;
        if info.shape.iter().product::<usize>() != n {
            bail!("tensor {full} has shape {:?}, expected {n} values", info.shape);
        }
        self.files.f32(full)
    }
    /// A matrix: GGUF blocks and FP8 as stored, float32 tensors exactly, others through `load_lin`.
    fn lin(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        if let Some(info) = self.files.info(full)
            && info.dtype == "F32"
            && self.files.ggml(full).is_none()
        {
            if info.shape.iter().product::<usize>() != n_out * n_in {
                bail!("tensor {full} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
            }
            self.files.note_use(full, true, false);
            return Lin::from_gguf(crate::ggml::GgmlType::F32, self.files.bytes(full)?, n_out, n_in);
        }
        load_lin(self.files, full, n_out, n_in, self.quant)
    }
    /// A small projection the reference keeps in float precision (never quantized at load).
    fn lin_fp(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        Loader {
            quant: Quant::Bf16,
            ..*self
        }
        .lin(full, n_out, n_in)
    }
    fn mlp(&self, p: &str, inter: usize) -> Result<Mlp<'a>> {
        let h = self.cfg.hidden_size;
        Ok(Mlp {
            gate: self.lin(&format!("{p}gate_proj.weight"), inter, h)?,
            up: self.lin(&format!("{p}up_proj.weight"), inter, h)?,
            down: self.lin(&format!("{p}down_proj.weight"), h, inter)?,
            inter,
        })
    }
    /// A hyper-connection mixer named `{p}fn` / `{p}base` / `{p}scale`.
    fn hc(&self, p: &str) -> Result<Hc> {
        let c = self.cfg;
        let rows = (2 + c.hc_mult) * c.hc_mult;
        Ok(Hc {
            f: self.getf(&format!("{p}fn"), rows * c.hc_mult * c.hidden_size)?,
            base: self.getf(&format!("{p}base"), rows)?,
            scale: self.getf(&format!("{p}scale"), 3)?,
        })
    }

    /// `kv_b` as per-head `k_b` (`[lat, dn]`) and `v_b` (`[v, lat]`): GGUF stores them so; a
    /// `kv_b_proj` (`[nh * (dn + v), lat]`) has its key rows transposed into a bf16 copy and its
    /// value rows borrowed.
    fn kv_b(&self, ap: &str) -> Result<(Vec<Lin<'a>>, Vec<Lin<'a>>)> {
        let c = self.cfg;
        let (nh, lat, v, dn) = (c.num_attention_heads, c.kv_lora_rank, c.v_head_dim, c.qk_nope_head_dim);
        let (kn, vn) = (format!("{ap}k_b"), format!("{ap}v_b"));
        if self.files.has(&kn) {
            let mut out = (Vec::with_capacity(nh), Vec::with_capacity(nh));
            for (name, n_out, n_in, dst) in [(&kn, lat, dn, &mut out.0), (&vn, v, lat, &mut out.1)] {
                let info = self.files.info(name).with_context(|| format!("missing tensor {name}"))?;
                if info.shape != [nh, n_out, n_in] {
                    bail!("tensor {name} has shape {:?}, expected [{nh}, {n_out}, {n_in}]", info.shape);
                }
                self.files.note_use(name, true, false);
                match self.files.ggml(name) {
                    Some(g) => {
                        let (kind, bytes) = g?;
                        for b in bytes.chunks(bytes.len() / nh) {
                            dst.push(Lin::from_gguf(kind, b, n_out, n_in)?);
                        }
                    }
                    None => {
                        let w = self.files.bf16(name)?;
                        dst.extend(w.chunks(n_out * n_in).map(Lin::bf16));
                    }
                }
            }
            return Ok(out);
        }
        let name = format!("{ap}kv_b_proj.weight");
        let info = self.files.info(&name).with_context(|| format!("missing tensor {name}"))?;
        if info.shape != [nh * (dn + v), lat] {
            bail!("tensor {name} has shape {:?}, expected [{}, {lat}]", info.shape, nh * (dn + v));
        }
        if info.dtype == "F8_E4M3" {
            bail!("{name}: an FP8 kv_b_proj is not supported (the release keeps it in bf16)");
        }
        let w: &'a [u16] = self.files.bf16(&name)?;
        self.files.note_use(&name, true, false);
        let (mut kb, mut vb) = (Vec::with_capacity(nh), Vec::with_capacity(nh));
        for h in 0..nh {
            let rows = &w[h * (dn + v) * lat..(h + 1) * (dn + v) * lat];
            let mut t = crate::quant::PageVec::<u16>::zeroed(lat * dn);
            for c in 0..dn {
                for r in 0..lat {
                    t[r * dn + c] = rows[c * lat + r];
                }
            }
            kb.push(Lin::bf16_owned(t));
            vb.push(Lin::bf16(&rows[dn * lat..]));
        }
        Ok((kb, vb))
    }

    fn mla(&self, l: usize) -> Result<Mla<'a>> {
        let c = self.cfg;
        let (h, nh, lat, ql, dr) = (
            c.hidden_size,
            c.num_attention_heads,
            c.kv_lora_rank,
            c.q_lora_rank,
            c.qk_rope_head_dim,
        );
        let ap = format!("{}layers.{l}.self_attn.", self.prefix);
        let ip = format!("{ap}indexer.");
        // A layer with indexer weights runs it ("full"); one without reuses the previous choice.
        let indexer = if self.files.has(&format!("{ip}wk.weight")) {
            let (ni, d) = (c.index_n_heads, c.index_head_dim);
            let pool = if c.index_kpool > 1 {
                Some(Pool {
                    gate: self.lin_fp(&format!("{ip}index_kpool_compress_gate"), d, h)?,
                    ape: self.getf(&format!("{ip}index_kpool_compress_ape"), c.index_kpool * d)?,
                })
            } else {
                None
            };
            Some(Indexer {
                wq: self.lin(&format!("{ip}wq_b.weight"), ni * d, ql)?,
                wk: self.lin(&format!("{ip}wk.weight"), d, h)?,
                k_norm: (
                    self.getf(&format!("{ip}k_norm.weight"), d)?,
                    self.getf(&format!("{ip}k_norm.bias"), d)?,
                ),
                proj: self.lin_fp(&format!("{ip}weights_proj.weight"), ni, h)?,
                pool,
            })
        } else {
            None
        };
        let (k_b, v_b) = self.kv_b(&ap)?;
        Ok(Mla {
            q_a: self.lin(&format!("{ap}q_a_proj.weight"), ql, h)?,
            q_norm: self.getf(&format!("{ap}q_a_layernorm.weight"), ql)?,
            q_b: self.lin(&format!("{ap}q_b_proj.weight"), nh * c.qk_head_dim(), ql)?,
            kv_a: self.lin(&format!("{ap}kv_a_proj_with_mqa.weight"), lat + dr, h)?,
            kv_norm: self.getf(&format!("{ap}kv_a_layernorm.weight"), lat)?,
            k_b,
            v_b,
            o: self.lin(&format!("{ap}o_proj.weight"), h, nh * c.v_head_dim)?,
            indexer,
        })
    }

    fn kda(&self, l: usize) -> Result<Kda<'a>> {
        let c = self.cfg;
        let (h, nh, d, k) = (c.hidden_size, c.kda_heads, c.kda_head_dim, c.kda_kernel);
        let w = nh * d;
        let ap = format!("{}layers.{l}.self_attn.", self.prefix);
        // `[channels, 1, kernel]` per projection (the release) or for all three (transformers'
        // module), as `[kernel][channels]`.
        let rows: Vec<f32> = if self.files.has(&format!("{ap}conv1d.weight")) {
            self.getf(&format!("{ap}conv1d.weight"), 3 * w * k)?
        } else {
            let mut v = Vec::with_capacity(3 * w * k);
            for n in ["q", "k", "v"] {
                v.extend(self.getf(&format!("{ap}{n}_conv1d.weight"), w * k)?);
            }
            v
        };
        let mut conv = vec![0f32; 3 * w * k];
        for (ch, r) in rows.chunks(k).enumerate() {
            for (j, &x) in r.iter().enumerate() {
                conv[j * 3 * w + ch] = x;
            }
        }
        // transformers' module keeps the forget gate in a submodule.
        let fg = if self.files.has(&format!("{ap}forget_gate.f_a_proj.weight")) {
            format!("{ap}forget_gate.")
        } else {
            ap.clone()
        };
        Ok(Kda {
            q: self.lin(&format!("{ap}q_proj.weight"), w, h)?,
            k: self.lin(&format!("{ap}k_proj.weight"), w, h)?,
            v: self.lin(&format!("{ap}v_proj.weight"), w, h)?,
            conv,
            f_a: self.lin_fp(&format!("{fg}f_a_proj.weight"), d, h)?,
            f_b: self.lin_fp(&format!("{fg}f_b_proj.weight"), w, d)?,
            dt_bias: self.getf(&format!("{fg}dt_bias"), w)?,
            rate: if self.files.has(&format!("{ap}ssm_a")) {
                self.getf(&format!("{ap}ssm_a"), nh)?.iter().map(|a| -a).collect()
            } else {
                self.getf(&format!("{fg}A_log"), nh)?.iter().map(|a| a.exp()).collect()
            },
            b: self.lin_fp(&format!("{ap}b_proj.weight"), nh, h)?,
            g_a: self.lin_fp(&format!("{ap}g_a_proj.weight"), d, h)?,
            g_b: self.lin_fp(&format!("{ap}g_b_proj.weight"), w, d)?,
            o_norm: self.getf(&format!("{ap}o_norm.weight"), d)?,
            o: self.lin(&format!("{ap}o_proj.weight"), h, w)?,
        })
    }

    fn layer(&self, l: usize) -> Result<Layer<'a>> {
        let c = self.cfg;
        let h = c.hidden_size;
        let lp = format!("{}layers.{l}.", self.prefix);
        let hc = if c.hc_mult > 1 {
            // The release's names, or transformers' modules.
            let (a, f) = if self.files.has(&format!("{lp}attn_hc.fn")) {
                (format!("{lp}attn_hc."), format!("{lp}ffn_hc."))
            } else {
                (format!("{lp}hc_attn_"), format!("{lp}hc_ffn_"))
            };
            Some((self.hc(&a)?, self.hc(&f)?))
        } else {
            None
        };
        let attn = if c.kda[l] {
            Attn::Kda(self.kda(l)?)
        } else {
            Attn::Mla(self.mla(l)?)
        };
        let mp = format!("{lp}mlp.");
        let act = if c.swiglu_limit > 0.0 {
            Act::SiluClamp(c.swiglu_limit as f32)
        } else {
            Act::Silu
        };
        let ffn = if c.sparse[l] {
            let (ne, k, mi) = (c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size);
            let bn = format!("{mp}gate.e_score_correction_bias");
            Ffn::Moe(Moe {
                router: self.getf(&format!("{mp}gate.weight"), ne * h)?,
                bias: if self.files.has(&bn) { self.getf(&bn, ne)? } else { vec![0.0; ne] },
                experts: Experts::load(self.files, &format!("{mp}experts."), ne, k, mi, h, self.quant, act)?,
                shared: self.mlp(&format!("{mp}shared_experts."), mi * c.n_shared_experts)?,
            })
        } else {
            Ffn::Dense(self.mlp(&mp, c.intermediate_size)?)
        };
        Ok(Layer {
            ln_in: self.getf(&format!("{lp}input_layernorm.weight"), h)?,
            ln_post: self.getf(&format!("{lp}post_attention_layernorm.weight"), h)?,
            hc,
            attn,
            ffn,
        })
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// LayerNorm with weight and bias over each row of `x` (`n` wide), in place.
fn layer_norm_rows(x: &mut [f32], n: usize, w: &[f32], b: &[f32], eps: f32) {
    for_chunks(x, n, x.len() > 64 * n, |_, r| {
        let mean = r.iter().sum::<f32>() / n as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ((v, &wi), &bi) in r.iter_mut().zip(w).zip(b) {
            *v = (*v - mean) * inv * wi + bi;
        }
    });
}

/// The `k` largest of `scores` (ties to the lower position), sorted by position.
fn top_positions(scores: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
    if k < idx.len() {
        idx.select_nth_unstable_by(k, |&a, &b| scores[b as usize].total_cmp(&scores[a as usize]).then(a.cmp(&b)));
        idx.truncate(k);
    }
    idx.sort_unstable();
    idx
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

#[derive(Clone, Copy)]
struct SendSel(*mut Option<Vec<u32>>);
unsafe impl Send for SendSel {}
unsafe impl Sync for SendSel {}
impl SendSel {
    #[inline(always)]
    fn get(self) -> *mut Option<Vec<u32>> {
        self.0
    }
}

impl<'a> Glm5Model<'a> {
    pub fn load(cfg: Glm5Config, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let prefix = if files.has("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else if files.has("model.embed_tokens.weight") {
            "model."
        } else {
            bail!("cannot find the token embedding (model.language_model.embed_tokens.weight) in the checkpoint");
        };
        let ld = Loader {
            files,
            cfg: &cfg,
            quant,
            prefix,
        };
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let en = format!("{prefix}embed_tokens.weight");
        let tied = cfg.tie_word_embeddings || !files.has("lm_head.weight");
        let embed = Table::load(files, &en, v, h, if tied { quant } else { table_quant(quant) })?;
        let lm_head = ld.lin(if tied { &en } else { "lm_head.weight" }, v, h)?;
        let final_norm = ld.getf(&format!("{prefix}norm.weight"), h)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut chosen = false;
        for l in 0..cfg.num_hidden_layers {
            let layer = ld.layer(l).with_context(|| format!("layer {l}"))?;
            if let Attn::Mla(m) = &layer.attn {
                chosen |= m.indexer.is_some();
                if !chosen {
                    bail!("layer {l}: a shared indexer layer before any layer with an indexer");
                }
            }
            layers.push(layer);
        }
        let dr = cfg.qk_rope_head_dim;
        let inv_freq = rope_inv_freq(dr, cfg.rope_theta, 1.0);
        let m = Self {
            mhc: cfg.mhc()?,
            cfg,
            quant,
            embed,
            lm_head,
            final_norm,
            layers,
            inv_freq,
            stream: None,
        };
        // Routed experts that cannot all stay in memory are read through the page cache as they
        // are routed: no accelerator mapping may pin them.
        if too_big(files.total_bytes()) {
            for l in &m.layers {
                if let Ffn::Moe(moe) = &l.ffn {
                    for e in moe.experts.lins().into_iter().flatten() {
                        e.set_transient_device(true);
                    }
                }
            }
        }
        Ok(m)
    }

    pub fn new_session(&self) -> Glm5Session {
        let c = &self.cfg;
        Glm5Session {
            layers: self
                .layers
                .iter()
                .map(|l| match &l.attn {
                    Attn::Kda(_) => LayerState::Kda {
                        conv: vec![0f32; (c.kda_kernel - 1) * 3 * c.kda_heads * c.kda_head_dim],
                        s: vec![0f32; c.kda_heads * c.kda_head_dim * c.kda_head_dim],
                    },
                    Attn::Mla(_) => LayerState::Mla(MlaState::default()),
                })
                .collect(),
            pos: 0,
            iv: Default::default(),
        }
    }

    fn dense_lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let mut v: Vec<&Lin<'a>> = match &l.attn {
            Attn::Kda(k) => vec![&k.q, &k.k, &k.v, &k.f_a, &k.f_b, &k.b, &k.g_a, &k.g_b, &k.o],
            Attn::Mla(a) => {
                let mut v = vec![&a.q_a, &a.q_b, &a.kv_a, &a.o];
                v.extend(a.k_b.iter().chain(&a.v_b));
                if let Some(ix) = &a.indexer {
                    v.extend([&ix.wq, &ix.wk, &ix.proj]);
                    v.extend(ix.pool.as_ref().map(|p| &p.gate));
                }
                v
            }
        };
        let m = match &l.ffn {
            Ffn::Dense(m) => m,
            Ffn::Moe(m) => &m.shared,
        };
        v.extend([&m.gate, &m.up, &m.down]);
        v
    }

    /// Weight bytes read per decoded token (the routed experts at top-k).
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes();
        for l in &self.layers {
            n += self.dense_lins(l).iter().map(|w| w.bytes()).sum::<usize>();
            if let Ffn::Moe(m) = &l.ffn {
                n += m.experts.bytes_per_token();
            }
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
                parts: match &layer.ffn {
                    Ffn::Moe(m) => m.experts.ranges(),
                    Ffn::Dense(_) => Vec::new(),
                },
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
            if let Ffn::Moe(m) = &layer.ffn {
                for e in m.experts.lins().into_iter().flatten() {
                    e.set_transient_device(true);
                }
            }
        }
        self.lm_head
            .set_transient_device(streamed.get(self.layers.len()).copied().unwrap_or(false));
    }

    /// Brackets every layer and the LM head with `hook`'s units (None stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// LM-head logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * c.vocab_size];
        let _unit = crate::stream::enter(&self.stream, self.layers.len());
        linear(&self.lm_head, c.vocab_size, c.hidden_size, hidden, t, &mut logits);
        logits
    }

    /// The input embedding of `token` (`[hidden]`).
    pub fn embed_row(&self, token: u32, out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        if let Some(s) = &self.stream {
            s.need_ranges(&self.embed.row_ranges(token as usize, h));
        }
        self.embed.row(token as usize, out);
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut Glm5Session, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden(sess, tokens);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos`, appending to the caches; returns the final-normed hidden
    /// states `[t, hidden]`.
    pub fn forward_hidden(&self, sess: &mut Glm5Session, tokens: &[u32]) -> Vec<f32> {
        self.forward_capture(sess, tokens, &[], &mut Vec::new())
    }

    /// [`Self::forward_hidden`] that also writes to `feats` (`[t, layers.len() * hidden]`) the
    /// residual stream (the mean of its copies) entering each of `layers`, `num_hidden_layers`
    /// standing for the last layer's output before the final norm.
    pub fn forward_capture(&self, sess: &mut Glm5Session, tokens: &[u32], layers: &[usize], feats: &mut Vec<f32>) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let c = &self.cfg;
        let (h, n) = (c.hidden_size, c.hc_mult);
        let eps = c.rms_norm_eps as f32;
        let start = sess.pos;
        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        // The residual stream: `n` copies of the embedding per token.
        let mut x = vec![0f32; t * n * h];
        for_chunks(&mut x, n * h, t > 1, |ti, row| {
            self.embed.row(tokens[ti] as usize, &mut row[..h]);
            for k in 1..n {
                row.copy_within(..h, k * h);
            }
        });
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = rope_tables(&positions, &self.inv_freq);
        drop(timer);
        let iv = sess.iv.active().then(|| sess.iv.clone());
        let point = |p: usize, x: &mut [f32]| {
            if let Some(iv) = &iv {
                if n > 1 {
                    iv.point_copies(p, x, h, n, start);
                } else {
                    iv.point(p, x, h, start);
                }
            }
        };
        point(0, &mut x);
        let nf = layers.len();
        feats.clear();
        feats.resize(t * nf * h, 0.0);
        let mut capture = |l: usize, x: &[f32]| {
            for (k, _) in layers.iter().enumerate().filter(|e| *e.1 == l) {
                for ti in 0..t {
                    mean_copies(
                        &x[ti * n * h..(ti + 1) * n * h],
                        &mut feats[(ti * nf + k) * h..(ti * nf + k + 1) * h],
                    );
                }
            }
        };
        let mut sel: Selection = Vec::new();
        let mut hn = vec![0f32; t * h];
        for (l, layer) in self.layers.iter().enumerate() {
            capture(l, &x);
            let _unit = crate::stream::enter(&self.stream, l);
            // Before a block: the collapsed (or the only) stream, normed.
            let pre = |w: Option<&Hc>, norm: &[f32], x: &[f32], hn: &mut Vec<f32>| -> Option<(Vec<f32>, Vec<f32>)> {
                match w {
                    Some(w) => {
                        let (y, post, comb) = hc_pre(&self.mhc, w, norm, x, t);
                        *hn = y;
                        Some((post, comb))
                    }
                    None => {
                        rmsnorm_f32w_rows(x, h, norm, eps, hn);
                        None
                    }
                }
            };
            let post = |x: &mut [f32], o: &[f32], mix: Option<(Vec<f32>, Vec<f32>)>| match mix {
                Some((post, comb)) => hc_post(&self.mhc, x, o, &post, &comb, t),
                None => add_inplace(x, o),
            };
            let mix = pre(layer.hc.as_ref().map(|p| &p.0), &layer.ln_in, &x, &mut hn);
            let mut o = vec![0f32; t * h];
            match (&layer.attn, &mut sess.layers[l]) {
                (Attn::Kda(k), LayerState::Kda { conv, s }) => self.kda_forward(k, &hn, t, conv, s, &mut o),
                (Attn::Mla(a), LayerState::Mla(st)) => self.mla_forward(a, &hn, t, start, st, &mut sel, (&cos, &sin), &mut o),
                _ => unreachable!("session of another model"),
            }
            post(&mut x, &o, mix);
            point(2 * l + 1, &mut x);
            let mix = pre(layer.hc.as_ref().map(|p| &p.1), &layer.ln_post, &x, &mut hn);
            o.fill(0.0);
            let hook = iv.as_ref().and_then(|s| s.hook(l, start));
            match &layer.ffn {
                Ffn::Dense(m) => self.mlp_forward(m, &hn, t, &mut o, hook.as_ref()),
                Ffn::Moe(m) => {
                    let need = |ids: &[usize]| {
                        if let Some(s) = &self.stream {
                            s.need_parts(l, ids)
                        }
                    };
                    let need: NeedExperts = self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize]));
                    self.moe_forward(m, &hn, t, &mut o, need, hook.as_ref());
                }
            }
            post(&mut x, &o, mix);
            point(2 * l + 2, &mut x);
        }
        capture(self.layers.len(), &x);
        // The final collapse is the mean of the copies.
        let mut out = vec![0f32; t * h];
        for_chunks(&mut out, h, t > 1, |ti, row| mean_copies(&x[ti * n * h..(ti + 1) * n * h], row));
        rmsnorm_f32w_rows_inplace(&mut out, h, &self.final_norm, eps);
        sess.pos += t;
        out
    }

    fn act(&self) -> Act {
        if self.cfg.swiglu_limit > 0.0 {
            Act::SiluClamp(self.cfg.swiglu_limit as f32)
        } else {
            Act::Silu
        }
    }

    /// A dense MLP: `down(silu(gate x) * up x)` with the clamp.
    fn mlp_forward(&self, m: &Mlp, x: &[f32], t: usize, y: &mut [f32], hook: Option<&NeuronHook>) {
        let (h, mi) = (self.cfg.hidden_size, m.inter);
        let mut g = vec![0f32; t * mi];
        let mut u = vec![0f32; t * mi];
        linear_multi(&mut [(&m.gate, mi, &mut g[..]), (&m.up, mi, &mut u[..])], h, x, t);
        self.act().apply(&mut g, &mut u);
        if let Some(k) = hook.filter(|k| k.edits(Target::Mlp)) {
            k.apply(Target::Mlp, &mut g, mi);
        }
        linear(&m.down, h, mi, &g, t, y);
    }

    /// Routes each token (sigmoid scores, the top k of score plus bias, renormalized and scaled)
    /// and runs the routed and shared experts.
    fn moe_forward(&self, m: &Moe, x: &[f32], t: usize, y: &mut [f32], need: NeedExperts, hook: Option<&NeuronHook>) {
        let c = &self.cfg;
        let h = c.hidden_size;
        let shared = crate::moe::Shared {
            gate: &m.shared.gate,
            up: &m.shared.up,
            down: &m.shared.down,
            inter: m.shared.inter,
            act: self.act(),
            hook: hook.filter(|k| k.edits(Target::Shared)),
            scale: None,
        };
        let ne = m.experts.n_experts();
        let mut logits = vec![0f32; t * ne];
        for_chunks(&mut logits, ne, t > 1, |ti, lg| {
            for (v, w) in lg.iter_mut().zip(m.router.chunks(h)) {
                *v = dot(w, &x[ti * h..(ti + 1) * h]);
            }
        });
        let routes: Vec<_> = logits
            .chunks(ne)
            .map(|lg| route_sigmoid(lg, &m.bias, m.experts.top_k, c.routed_scaling_factor as f32))
            .collect();
        m.experts.forward(x, t, h, &routes, y, need, hook, Some(&shared));
    }

    /// KDA linear attention over `xa` (`[t, H]`, the normed block input) into `out`, updating
    /// the convolution history `conv` and the heads' states `s`.
    fn kda_forward(&self, a: &Kda, xa: &[f32], t: usize, conv: &mut Vec<f32>, s: &mut [f32], out: &mut [f32]) {
        let c = &self.cfg;
        let (h, nh, d, kn) = (c.hidden_size, c.kda_heads, c.kda_head_dim, c.kda_kernel);
        let w = nh * d;
        let (mut q, mut k, mut v) = (vec![0f32; t * w], vec![0f32; t * w], vec![0f32; t * w]);
        let (mut fa, mut ga, mut beta) = (vec![0f32; t * d], vec![0f32; t * d], vec![0f32; t * nh]);
        linear_multi(
            &mut [
                (&a.q, w, &mut q[..]),
                (&a.k, w, &mut k[..]),
                (&a.v, w, &mut v[..]),
                (&a.f_a, d, &mut fa[..]),
                (&a.g_a, d, &mut ga[..]),
                (&a.b, nh, &mut beta[..]),
            ],
            h,
            xa,
            t,
        );
        let (mut g, mut gate) = (vec![0f32; t * w], vec![0f32; t * w]);
        linear(&a.f_b, w, d, &fa, t, &mut g);
        linear(&a.g_b, w, d, &ga, t, &mut gate);
        let _timer = prof::start(Cat::Attention);
        // The causal depthwise convolution over `q | k | v` (history first), then SiLU.
        let cw = 3 * w;
        let hist = kn - 1;
        let mut seq = std::mem::take(conv);
        seq.reserve(t * cw);
        for ti in 0..t {
            seq.extend_from_slice(&q[ti * w..(ti + 1) * w]);
            seq.extend_from_slice(&k[ti * w..(ti + 1) * w]);
            seq.extend_from_slice(&v[ti * w..(ti + 1) * w]);
        }
        let mut mixed = vec![0f32; t * cw];
        for_chunks(&mut mixed, cw, t > 1, |ti, row| {
            for j in 0..kn {
                let src = &seq[(ti + j) * cw..(ti + j + 1) * cw];
                let wj = &a.conv[j * cw..(j + 1) * cw];
                for ((o, &x), &wv) in row.iter_mut().zip(src).zip(wj) {
                    *o += wv * x;
                }
            }
            for o in row.iter_mut() {
                *o *= sigmoid(*o);
            }
        });
        *conv = seq[t * cw..].to_vec();
        debug_assert_eq!(conv.len(), hist * cw);
        // The decay (log space) per key channel and the write strength per head.
        let lower = c.kda_lower_bound.map(|b| b as f32);
        for_chunks(&mut g, w, t > 1, |_, row| {
            for (i, x) in row.iter_mut().enumerate() {
                let r = a.rate[i / d];
                let z = *x + a.dt_bias[i];
                *x = match lower {
                    Some(lb) => lb * sigmoid(r * z),
                    None => -r * if z > 20.0 { z } else { z.exp().ln_1p() },
                };
            }
        });
        beta.iter_mut().for_each(|b| *b = sigmoid(*b));
        // The delta rule, head by head over the tokens: `S <- S diag(decay) (rows = key
        // channels)`, `S += k (beta (v - S^T k))^T`, read `S^T q`.
        let scale = (d as f32).powf(-0.5);
        let eps = c.rms_norm_eps as f32;
        let mut att = vec![0f32; nh * t * d];
        let sp = SendPtr(s.as_mut_ptr());
        for_chunks(&mut att, t * d, nh > 1, |hh, oh| {
            // SAFETY: each head has its own state block.
            let st = unsafe { std::slice::from_raw_parts_mut(sp.get().add(hh * d * d), d * d) };
            let (mut qn, mut kk) = (vec![0f32; d], vec![0f32; d]);
            let mut delta = vec![0f32; d];
            for ti in 0..t {
                let at = ti * w + hh * d;
                let qr = &mixed[ti * cw + hh * d..ti * cw + (hh + 1) * d];
                let kr = &mixed[ti * cw + w + hh * d..ti * cw + w + (hh + 1) * d];
                let vr = &mixed[ti * cw + 2 * w + hh * d..ti * cw + 2 * w + (hh + 1) * d];
                let qs = scale / (dot(qr, qr) + 1e-6).sqrt();
                let ks = 1.0 / (dot(kr, kr) + 1e-6).sqrt();
                for i in 0..d {
                    qn[i] = qr[i] * qs;
                    kk[i] = kr[i] * ks;
                }
                // kv_mem = (decayed S)^T k
                delta.fill(0.0);
                for (i, row) in st.chunks_mut(d).enumerate() {
                    scale_inplace(row, g[at + i].exp());
                    axpy(&mut delta, kk[i], row);
                }
                let b = beta[ti * nh + hh];
                for (dv, &x) in delta.iter_mut().zip(vr) {
                    *dv = (x - *dv) * b;
                }
                let o = &mut oh[ti * d..(ti + 1) * d];
                for (i, row) in st.chunks_mut(d).enumerate() {
                    axpy(row, kk[i], &delta);
                    axpy(o, qn[i], row);
                }
                // The read, RMS-normed and gated.
                let gr = &gate[at..at + d];
                let r = 1.0 / (dot(o, o) / d as f32 + eps).sqrt();
                for ((x, &wn), &gv) in o.iter_mut().zip(&a.o_norm).zip(gr) {
                    *x = *x * r * wn * sigmoid(gv);
                }
            }
        });
        // [nh][t][d] -> [t][nh * d]
        let mut y = vec![0f32; t * w];
        for hh in 0..nh {
            for ti in 0..t {
                y[ti * w + hh * d..ti * w + (hh + 1) * d].copy_from_slice(&att[(hh * t + ti) * d..(hh * t + ti + 1) * d]);
            }
        }
        drop(_timer);
        linear(&a.o, h, w, &y, t, out);
    }

    /// MLA attention over `xa` (`[t, H]`, the normed block input) into `out`. `sel` holds the
    /// previous indexer layer's choice (replaced on layers with an indexer).
    #[allow(clippy::too_many_arguments)]
    fn mla_forward(
        &self,
        a: &Mla,
        xa: &[f32],
        t: usize,
        start: usize,
        st: &mut MlaState,
        sel: &mut Selection,
        rope: (&[f32], &[f32]),
        out: &mut [f32],
    ) {
        let c = &self.cfg;
        let (h, nh, lat, vd, ql) = (c.hidden_size, c.num_attention_heads, c.kv_lora_rank, c.v_head_dim, c.q_lora_rank);
        let (dn, dr) = (c.qk_nope_head_dim, c.qk_rope_head_dim);
        let qk = dn + dr;
        let eps = c.rms_norm_eps as f32;
        let (cos, sin) = rope;
        let half = dr / 2;
        let par = t > 1;
        let (ni, d, r) = (c.index_n_heads, c.index_head_dim, c.index_kpool);

        let mut qa = vec![0f32; t * ql];
        let mut kv = vec![0f32; t * (lat + dr)];
        let ix = a.indexer.as_ref();
        let mut ik = vec![0f32; if ix.is_some() { t * d } else { 0 }];
        let mut iw = vec![0f32; if ix.is_some() { t * ni } else { 0 }];
        let pool = ix.and_then(|i| i.pool.as_ref());
        let mut ig = vec![0f32; if pool.is_some() { t * d } else { 0 }];
        {
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.q_a, ql, &mut qa[..]), (&a.kv_a, lat + dr, &mut kv[..])];
            if let Some(ix) = ix {
                jobs.push((&ix.wk, d, &mut ik[..]));
                jobs.push((&ix.proj, ni, &mut iw[..]));
            }
            if let Some(p) = pool {
                jobs.push((&p.gate, d, &mut ig[..]));
            }
            linear_multi(&mut jobs, h, xa, t);
        }
        rmsnorm_f32w_rows_inplace(&mut qa, ql, &a.q_norm, eps);
        let mut q = vec![0f32; t * nh * qk];
        linear(&a.q_b, nh * qk, ql, &qa, t, &mut q);
        if dr > 0 {
            for_chunks(&mut q, qk, par, |i, qh| {
                let ti = i / nh;
                apply_rope_interleaved(&mut qh[dn..], &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            });
        }

        // The latent cache: normed latent and rotated rotary key per token.
        for ti in 0..t {
            let row = &mut kv[ti * (lat + dr)..(ti + 1) * (lat + dr)];
            let (cl, kr) = row.split_at_mut(lat);
            rmsnorm_f32w_inplace(cl, &a.kv_norm, eps);
            if dr > 0 {
                apply_rope_interleaved(kr, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            }
            st.lat.extend_from_slice(cl);
            st.rot.extend_from_slice(kr);
        }

        // The indexer: its keys (pooled for GLM-5.3-Flash) join its cache, and each query that
        // sees more than `index_topk / r` of them gets its best.
        if let Some(ix) = ix {
            layer_norm_rows(&mut ik, d, &ix.k_norm.0, &ix.k_norm.1, INDEX_NORM_EPS);
            if dr > 0 {
                for_chunks(&mut ik, d, par, |ti, kh| {
                    apply_rope_interleaved(&mut kh[..dr], &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                });
            }
            match pool {
                None => st.keys.extend_from_slice(&ik),
                Some(p) => {
                    for ti in 0..t {
                        st.pend_k.extend_from_slice(&ik[ti * d..(ti + 1) * d]);
                        st.pend_g.extend_from_slice(&ig[ti * d..(ti + 1) * d]);
                        if st.pend_k.len() == r * d {
                            // Per channel: a softmax over the pool's tokens of gate + bias.
                            let base = st.keys.len();
                            st.keys.resize(base + d, 0.0);
                            for ch in 0..d {
                                let lg = |j: usize| st.pend_g[j * d + ch] + p.ape[j * d + ch];
                                let m = (0..r).map(lg).fold(f32::NEG_INFINITY, f32::max);
                                let (mut tot, mut acc) = (0f32, 0f32);
                                for j in 0..r {
                                    let e = (lg(j) - m).exp();
                                    tot += e;
                                    acc += e * st.pend_k[j * d + ch];
                                }
                                st.keys[base + ch] = acc / tot;
                            }
                            st.pend_k.clear();
                            st.pend_g.clear();
                        }
                    }
                }
            }
            let units = c.index_topk / r;
            let rows: Vec<usize> = (0..t).filter(|&i| (start + i + 1) / r > units).collect();
            *sel = vec![None; t];
            if !rows.is_empty() {
                let _timer = prof::start(Cat::Attention);
                let mut iq = vec![0f32; t * ni * d];
                linear(&ix.wq, ni * d, ql, &qa, t, &mut iq);
                if dr > 0 {
                    for_chunks(&mut iq, d, par, |i, qh| {
                        let ti = i / ni;
                        apply_rope_interleaved(&mut qh[..dr], &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                    });
                }
                let ws = (d as f32).powf(-0.5) * (ni as f32).powf(-0.5);
                let keys = &st.keys;
                let sp = SendSel(sel.as_mut_ptr());
                crate::pool::run(rows.len(), &|ri| {
                    let i = rows[ri];
                    let p = start + i;
                    let nv = (p + 1) / r;
                    let mut s = vec![0f32; nv];
                    for (u, o) in s.iter_mut().enumerate() {
                        let key = &keys[u * d..(u + 1) * d];
                        let mut acc = 0f32;
                        for hh in 0..ni {
                            acc += iw[i * ni + hh] * dot(&iq[(i * ni + hh) * d..(i * ni + hh + 1) * d], key).max(0.0);
                        }
                        *o = acc * ws;
                    }
                    let mut list: Vec<u32> = Vec::with_capacity(units * r + r);
                    for u in top_positions(&s, units) {
                        list.extend((u * r as u32)..((u + 1) * r as u32));
                    }
                    // The tokens of the incomplete pool at the query's end.
                    list.extend((nv * r) as u32..=p as u32);
                    // SAFETY: each task writes its own query's slot.
                    unsafe { *sp.get().add(i) = Some(list) };
                });
            }
        } else if sel.len() != t {
            // (Checked at load: a shared layer follows a layer with an indexer.)
            *sel = vec![None; t];
        }

        // Scores against the latent (`kv_b` absorbed): each head's non-rotary query mapped onto
        // the latent, the attended latent mapped onto the value.
        let _timer = prof::start(Cat::Attention);
        let mut qn = vec![0f32; t * dn];
        let mut qlat = vec![0f32; nh * t * lat];
        for hh in 0..nh {
            for ti in 0..t {
                qn[ti * dn..(ti + 1) * dn].copy_from_slice(&q[(ti * nh + hh) * qk..(ti * nh + hh) * qk + dn]);
            }
            linear(&a.k_b[hh], lat, dn, &qn, t, &mut qlat[hh * t * lat..(hh + 1) * t * lat]);
        }
        let scale = (qk as f32).powf(-0.5);
        let mut acc = vec![0f32; nh * t * lat];
        let (clat, crot) = (&st.lat, &st.rot);
        let sel = &*sel;
        for_chunks(&mut acc, lat, true, |task, o| {
            let (hh, i) = (task / t, task % t);
            let ql = &qlat[task * lat..(task + 1) * lat];
            let qr = &q[(i * nh + hh) * qk + dn..(i * nh + hh + 1) * qk];
            let score = |j: usize| {
                let mut s = dot(ql, &clat[j * lat..(j + 1) * lat]);
                if dr > 0 {
                    s += dot(qr, &crot[j * dr..(j + 1) * dr]);
                }
                s * scale
            };
            let p = start + i;
            let mut s: Vec<f32> = match &sel[i] {
                Some(list) => list.iter().map(|&j| score(j as usize)).collect(),
                None => (0..=p).map(score).collect(),
            };
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut tot = 0f32;
            for v in s.iter_mut() {
                *v = (*v - m).exp();
                tot += *v;
            }
            let inv = 1.0 / tot;
            match &sel[i] {
                Some(list) => {
                    for (&j, &w) in list.iter().zip(&s) {
                        axpy(o, w * inv, &clat[j as usize * lat..(j as usize + 1) * lat]);
                    }
                }
                None => {
                    for (j, &w) in s.iter().enumerate() {
                        axpy(o, w * inv, &clat[j * lat..(j + 1) * lat]);
                    }
                }
            }
        });
        let mut att = vec![0f32; t * nh * vd];
        let mut vh = vec![0f32; t * vd];
        for hh in 0..nh {
            linear(&a.v_b[hh], vd, lat, &acc[hh * t * lat..(hh + 1) * t * lat], t, &mut vh);
            for ti in 0..t {
                att[(ti * nh + hh) * vd..(ti * nh + hh + 1) * vd].copy_from_slice(&vh[ti * vd..(ti + 1) * vd]);
            }
        }
        drop(_timer);
        linear(&a.o, h, nh * vd, &att, t, out);
    }
}

/// The mean of a token's residual copies `x` (`[n * H]`) into `y` (`[H]`).
fn mean_copies(x: &[f32], y: &mut [f32]) {
    let h = y.len();
    let n = x.len() / h;
    y.copy_from_slice(&x[..h]);
    for k in 1..n {
        add_inplace(y, &x[k * h..(k + 1) * h]);
    }
    if n > 1 {
        scale_inplace(y, 1.0 / n as f32);
    }
}

impl crate::generate::TextLm for Glm5Model<'_> {
    type Sess = Glm5Session;
    fn new_session(&self) -> Glm5Session {
        Glm5Model::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut Glm5Session) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &Glm5Session) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut Glm5Session, tokens: &[u32], _mm: &[(usize, &[f32])]) -> Vec<f32> {
        Glm5Model::forward_hidden(self, sess, tokens)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        Glm5Model::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut Glm5Session, tokens: &[u32]) -> Vec<f32> {
        Glm5Model::forward(self, sess, tokens)
    }
}

/// Runtime interventions (`crate::intervene`): the residual points around every layer (on the
/// copies for GLM-5.3-Flash), and as neurons the input of `down_proj` of each layer's dense MLP
/// (`Target::Mlp`), shared expert (`Target::Shared`) and routed experts (`Target::Expert`,
/// before the routing weight).
impl crate::intervene::Host for Glm5Model<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        match (&self.layers.get(layer)?.ffn, target) {
            (Ffn::Dense(m), Target::Mlp) => Some(m.inter),
            (Ffn::Moe(m), Target::Shared) => Some(m.shared.inter),
            (Ffn::Moe(m), Target::Expert(e)) if e < m.experts.n_experts() => Some(m.experts.inter),
            _ => None,
        }
    }
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32> {
        let (down, inter) = match (&self.layers[layer].ffn, target) {
            (Ffn::Dense(m), _) => (&m.down, m.inter),
            (Ffn::Moe(m), Target::Expert(e)) => (&m.experts.down[e], m.experts.inter),
            (Ffn::Moe(m), _) => (&m.shared.down, m.shared.inter),
        };
        let mut one = vec![0f32; inter];
        one[n] = 1.0;
        let h = self.cfg.hidden_size;
        let mut y = vec![0f32; h];
        // A streamed checkpoint reads the layer (and the routed expert) as a pass would.
        let _unit = crate::stream::enter(&self.stream, layer);
        if let (Some(s), Target::Expert(e)) = (&self.stream, target) {
            s.need_parts(layer, &[e]);
        }
        linear(down, h, inter, &one, 1, &mut y);
        y
    }
}

/// What a speculative decoder needs of its target. GLM-5.3-Flash's recurrent layers cannot go
/// back to an earlier position, so no draft is offered for it (`truncate` then panics).
impl crate::dspark::Target for Glm5Model<'_> {
    fn forward_feats(
        &self,
        sess: &mut Glm5Session,
        tokens: &[u32],
        _mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, layers, feats)
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        Glm5Model::embed_row(self, token, out)
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.logits(hidden, t)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut Glm5Session, len: usize) {
        assert!(
            sess.truncate(len),
            "GLM-5.3-Flash: the recurrent layers cannot go back to an earlier position"
        );
    }
}
