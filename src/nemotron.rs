//! NVIDIA Nemotron-H text models, as transformers' `NemotronHForCausalLM` (`nemotron_h`:
//! nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 and the earlier Nemotron 3 / Nemotron-H
//! releases; llama.cpp `nemotron_h` / `nemotron_h_moe`), and the language model of
//! nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (`nemo_speechlm`; the audio side is
//! [`crate::nemotron_asr`]).
//!
//! A stack of single-mixer blocks, `x += mixer(rms_norm(x))`, the mixer of each block given by
//! `layers_block_type`:
//!
//! - Mamba-2 (`mamba` / `linear_attention`): one projection gives a gate, the convolution input
//!   `x | B | C` and a time step per head; a causal depthwise convolution (kernel 4, with bias)
//!   and SiLU; per head a `[head_dim, state]` matrix, `S <- exp(dt A) S + dt x B^T` with
//!   `dt = max(softplus(dt + bias), time_step_min)` and `A = -exp(A_log)`, read as
//!   `S C + D x` (`B` and `C` shared by the heads of a group); the read times `silu(gate)`,
//!   RMS-normed per group, projected out. The state does not grow with the context, and the
//!   recurrence here is the reference's chunked scan in closed form.
//! - Attention (`attention` / `full_attention`): grouped-query causal attention without any
//!   position encoding (the Mamba blocks carry the order).
//! - MoE (`moe`): a sigmoid router with a selection bias, the top k renormalized and scaled;
//!   experts and the shared expert are plain two-layer MLPs with a squared ReLU,
//!   `down(relu(up x)^2)` (no gate projection, unlike the experts of [`crate::moe`]).
//! - MLP (`mlp`): the same two-layer MLP, dense (the Nemotron-H 8B / 47B / 56B models).
//!
//! The time step floor: NVIDIA's own module clamps `dt` at `time_step_min` in every pass, and so
//! does this one. transformers' port clamps in its multi-token scan only, so its cached
//! one-token steps differ from its own uncached forward wherever a head's `dt` is below the
//! floor ([`NemotronSession::dt_min`] reproduces either).
//!
//! Checkpoints, told apart by their tensor names:
//!
//! - the release (`backbone.layers.{l}.mixer.*`, one module per expert), and what transformers'
//!   `save_pretrained` writes (`model.layers.{l}.mixer.*`, experts stacked `[n, out, in]`);
//! - NeMo's speech LM export (`llm.model.layers.{l}.mixer.*`): Mamba's `A_log` / `D` /
//!   `dt_bias` under `_fp32_params`, and the experts stacked with the two matrix axes swapped
//!   (`gate_and_up_projs` `[n, in, out]`, `down_projs`), which are transposed into owned
//!   matrices at load (at the precision asked for);
//! - llama.cpp GGUF files, named onto transformers' tensors by [`gguf_name`].
//!
//! Not here: the multi-token prediction head (`mtp.*`), which transformers does not run
//! either; expert groups (`n_group > 1`) and the MoE latent projection (`moe_latent_size`),
//! which no released checkpoint uses.
//!
//! Runtime interventions ([`crate::intervene`]): the residual stream after the embeddings
//! (point 0) and after block `l` (point `2 l + 2`); neurons are the input of `down_proj` of a
//! dense MLP, of the shared expert and of the routed experts.

use crate::intervene::{NeuronHook, Target};
use crate::kernels::*;
use crate::model::{
    LayerCache, Lin, Quant, Table, attention, attention_prefill, linear, linear_grouped, linear_multi, load_lin, load_lin_experts,
    table_quant,
};
use crate::moe::route_sigmoid;
use crate::prof::{self, Cat};
use crate::quant::{PageVec, quantize_bf16, quantize_q4};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::Path;

/// Whether `model_type` is a Nemotron-H language model, alone or behind NeMo's speech front end.
pub fn is_model_type(mt: &str) -> bool {
    matches!(mt, "nemotron_h" | SPEECHLM)
}

/// `model_type` of NeMo's speech LM export (Nemotron 3.5 Transcribe).
pub const SPEECHLM: &str = "nemo_speechlm";

/// The mixer of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    Mamba,
    Attention,
    Moe,
    Mlp,
}

/// The settings of a Nemotron-H model (`config.json`, the `text_config` / `llm_config` of a
/// speech LM, or a GGUF header through [`gguf_config`]).
#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    /// The mixer of each block.
    pub blocks: Vec<Block>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,
    /// Groups sharing `B` and `C`, and of the gated RMS norm.
    pub n_groups: usize,
    pub ssm_state_size: usize,
    pub conv_kernel: usize,
    /// Width of a dense MLP block.
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub moe_shared_expert_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f64,
    pub layer_norm_epsilon: f64,
    /// The floor of the Mamba time step.
    pub time_step_min: f64,
}

impl NemotronConfig {
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
        // A speech LM keeps the language model's config under `text_config` (transformers) and
        // `llm_config` (NeMo).
        let t = ["text_config", "llm_config"]
            .iter()
            .filter_map(|k| raw.get(*k))
            .find(|v| v.is_object())
            .unwrap_or(raw);
        let num = |k: &str| t.get(k).and_then(Value::as_u64).map(|v| v as usize);
        let need = |k: &str| num(k).with_context(|| format!("Nemotron-H config has no {k}"));
        let float = |k: &str| t.get(k).and_then(Value::as_f64);
        let blocks: Vec<Block> = match (t.get("layers_block_type"), t.get("hybrid_override_pattern")) {
            (Some(Value::Array(a)), _) => a
                .iter()
                .map(|v| match v.as_str() {
                    Some("mamba" | "linear_attention") => Ok(Block::Mamba),
                    Some("attention" | "full_attention") => Ok(Block::Attention),
                    Some("moe") => Ok(Block::Moe),
                    Some("mlp") => Ok(Block::Mlp),
                    other => bail!("Nemotron-H: unknown block type {other:?}"),
                })
                .collect::<Result<_>>()?,
            (_, Some(Value::String(p))) => p
                .chars()
                .map(|c| match c {
                    'M' => Ok(Block::Mamba),
                    '*' => Ok(Block::Attention),
                    'E' => Ok(Block::Moe),
                    '-' => Ok(Block::Mlp),
                    other => bail!("Nemotron-H: unknown block {other:?} in hybrid_override_pattern"),
                })
                .collect::<Result<_>>()?,
            _ => bail!("Nemotron-H config has neither layers_block_type nor hybrid_override_pattern"),
        };
        if blocks.is_empty() {
            bail!("Nemotron-H config has no layers");
        }
        let kinds = blocks.clone();
        let has = |b: Block| kinds.contains(&b);
        for (k, want) in [("mlp_hidden_act", "relu2"), ("mamba_hidden_act", "silu")] {
            if let Some(a) = t.get(k).and_then(Value::as_str)
                && a != want
            {
                bail!("Nemotron-H: {k} {a:?} is not supported ({want} is)");
            }
        }
        if has(Block::Moe) {
            if num("n_group").unwrap_or(1) != 1 || num("topk_group").unwrap_or(1) != 1 {
                bail!("Nemotron-H: expert groups (n_group, topk_group) are not supported");
            }
            if t.get("moe_latent_size").is_some_and(|v| !v.is_null()) {
                bail!("Nemotron-H: moe_latent_size is not supported");
            }
            if t.get("norm_topk_prob").and_then(Value::as_bool) == Some(false) {
                bail!("Nemotron-H: norm_topk_prob=false is not supported");
            }
        }
        for k in ["use_bias", "mlp_bias", "attention_bias", "mamba_proj_bias"] {
            if t.get(k).and_then(Value::as_bool) == Some(true) {
                bail!("Nemotron-H: {k}=true is not supported");
            }
        }
        if t.get("use_conv_bias").and_then(Value::as_bool) == Some(false) {
            bail!("Nemotron-H: use_conv_bias=false is not supported");
        }
        let nh = num("num_attention_heads").unwrap_or(32);
        let hidden = need("hidden_size")?;
        let c = Self {
            hidden_size: hidden,
            vocab_size: need("vocab_size")?,
            num_attention_heads: nh,
            num_key_value_heads: num("num_key_value_heads").unwrap_or(nh),
            head_dim: num("head_dim").unwrap_or(hidden / nh.max(1)),
            mamba_num_heads: num("mamba_num_heads").unwrap_or(128),
            mamba_head_dim: num("mamba_head_dim").unwrap_or(64),
            n_groups: num("n_groups").or(num("mamba_n_groups")).unwrap_or(8),
            ssm_state_size: num("ssm_state_size").unwrap_or(128),
            conv_kernel: num("conv_kernel").or(num("mamba_d_conv")).unwrap_or(4),
            intermediate_size: num("intermediate_size").unwrap_or(0),
            moe_intermediate_size: num("moe_intermediate_size").unwrap_or(0),
            moe_shared_expert_intermediate_size: num("moe_shared_expert_intermediate_size").unwrap_or(0),
            n_routed_experts: num("n_routed_experts").unwrap_or(0),
            num_experts_per_tok: num("num_experts_per_tok").unwrap_or(0),
            routed_scaling_factor: float("routed_scaling_factor").unwrap_or(1.0),
            layer_norm_epsilon: float("layer_norm_epsilon").unwrap_or(1e-5),
            time_step_min: float("time_step_min").or(float("mamba_dt_min")).unwrap_or(0.001),
            blocks,
        };
        if has(Block::Attention) && (c.num_key_value_heads == 0 || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)) {
            bail!(
                "Nemotron-H: {} attention heads over {} key-value heads",
                c.num_attention_heads,
                c.num_key_value_heads
            );
        }
        if has(Block::Mamba) && (c.n_groups == 0 || !c.mamba_num_heads.is_multiple_of(c.n_groups) || c.conv_kernel == 0) {
            bail!("Nemotron-H: {} Mamba heads over {} groups", c.mamba_num_heads, c.n_groups);
        }
        if has(Block::Moe) && (c.n_routed_experts == 0 || c.num_experts_per_tok == 0 || c.num_experts_per_tok > c.n_routed_experts) {
            bail!("Nemotron-H: {} experts per token of {}", c.num_experts_per_tok, c.n_routed_experts);
        }
        Ok(c)
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Width of the Mamba inner stream.
    fn mamba_inner(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// Channels through the Mamba convolution: `x | B | C`.
    fn conv_dim(&self) -> usize {
        self.mamba_inner() + 2 * self.n_groups * self.ssm_state_size
    }

    fn count(&self, b: Block) -> usize {
        self.blocks.iter().filter(|x| **x == b).count()
    }

    /// "23 Mamba-2, 6 attention, 23 MoE" for the load line.
    pub fn describe(&self) -> String {
        [
            (Block::Mamba, "Mamba-2"),
            (Block::Attention, "attention"),
            (Block::Moe, "MoE"),
            (Block::Mlp, "MLP"),
        ]
        .iter()
        .filter(|(b, _)| self.count(*b) > 0)
        .map(|(b, n)| format!("{} {n}", self.count(*b)))
        .collect::<Vec<_>>()
        .join(", ")
    }
}

/// The Hugging Face config (the keys [`NemotronConfig::from_json`] reads) of a llama.cpp
/// `nemotron_h` / `nemotron_h_moe` header: a block with key-value heads is attention, one with a
/// feed-forward width is MoE (or a dense MLP in a file without experts), the others are Mamba-2.
pub(crate) fn gguf_config(h: &crate::gguf::Header) -> Result<Value> {
    let arch = h.arch();
    let get = |k: &str| h.get(&format!("{arch}.{k}"));
    let opt = |k: &str| get(k).and_then(crate::gguf::Value::as_u64);
    let need = |k: &str| opt(k).with_context(|| format!("GGUF metadata has no {arch}.{k}"));
    let optf = |k: &str| get(k).and_then(crate::gguf::Value::as_f64);
    let n = h.n_layers();
    let per_layer = |k: &str| -> Result<Vec<u64>> {
        match get(k) {
            Some(crate::gguf::Value::Array(a)) if a.len() >= n => Ok(a[..n].iter().map(|v| v.as_u64().unwrap_or(0)).collect()),
            Some(v) if v.as_u64().is_some() => Ok(vec![v.as_u64().unwrap(); n]),
            _ => bail!("GGUF {arch} has no per-layer {arch}.{k}"),
        }
    };
    let kv = per_layer("attention.head_count_kv")?;
    let ff = per_layer("feed_forward_length")?;
    let experts = opt("expert_count").unwrap_or(0);
    let blocks: Vec<&str> = (0..n)
        .map(|l| match (kv[l] > 0, ff[l] > 0) {
            (true, _) => "attention",
            (false, true) if experts > 0 => "moe",
            (false, true) => "mlp",
            (false, false) => "mamba",
        })
        .collect();
    let inner = need("ssm.inner_size")?;
    let heads = need("ssm.time_step_rank")?;
    if heads == 0 || inner % heads != 0 {
        bail!("GGUF {arch}: ssm.inner_size {inner} over ssm.time_step_rank {heads}");
    }
    let nh = need("attention.head_count")?;
    let hidden = need("embedding_length")?;
    Ok(json!({
        "model_type": "nemotron_h",
        "hidden_size": hidden,
        "vocab_size": opt("vocab_size")
            .or_else(|| h.get("tokenizer.ggml.tokens").and_then(crate::gguf::Value::as_array).map(|a| a.len() as u64))
            .context("GGUF has no vocabulary size")?,
        "layers_block_type": blocks,
        "num_attention_heads": nh,
        "num_key_value_heads": kv.iter().copied().find(|k| *k > 0).unwrap_or(nh),
        "head_dim": opt("attention.key_length").unwrap_or(hidden / nh.max(1)),
        "mamba_num_heads": heads,
        "mamba_head_dim": inner / heads,
        "n_groups": need("ssm.group_count")?,
        "ssm_state_size": need("ssm.state_size")?,
        "conv_kernel": need("ssm.conv_kernel")?,
        "intermediate_size": ff.iter().copied().find(|f| *f > 0).unwrap_or(0),
        "moe_intermediate_size": opt("expert_feed_forward_length").unwrap_or(0),
        "moe_shared_expert_intermediate_size": opt("expert_shared_feed_forward_length").unwrap_or(0),
        "n_routed_experts": experts,
        "num_experts_per_tok": opt("expert_used_count").unwrap_or(0),
        "n_group": opt("expert_group_count").unwrap_or(1),
        "topk_group": opt("expert_group_used_count").unwrap_or(1),
        "routed_scaling_factor": optf("expert_weights_scale").unwrap_or(1.0),
        "layer_norm_epsilon": optf("attention.layer_norm_rms_epsilon").or(optf("attention.layer_norm_epsilon")).unwrap_or(1e-5),
    }))
}

/// transformers' name and PyTorch-order shape of a llama.cpp `nemotron_h` tensor (`ne`: ggml's
/// shape); None for tensors not read (the MTP blocks among them).
pub(crate) fn gguf_name(name: &str, ne: &[usize], n_layers: usize) -> Option<(String, Vec<usize>)> {
    let shape: Vec<usize> = ne.iter().rev().copied().collect();
    let top = match name {
        "token_embd.weight" => Some("model.embeddings.weight"),
        "output.weight" => Some("lm_head.weight"),
        "output_norm.weight" => Some("model.norm_f.weight"),
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
        "attn_norm.weight" => "norm.weight",
        "ssm_in.weight" => "mixer.in_proj.weight",
        "ssm_conv1d.weight" => "mixer.conv1d.weight",
        "ssm_conv1d.bias" => "mixer.conv1d.bias",
        "ssm_dt.bias" => "mixer.dt_bias",
        // llama.cpp stores `-exp(A_log)`.
        "ssm_a" => "mixer.ssm_a",
        "ssm_d" => "mixer.D",
        "ssm_norm.weight" => "mixer.norm.weight",
        "ssm_out.weight" => "mixer.out_proj.weight",
        "attn_q.weight" => "mixer.q_proj.weight",
        "attn_k.weight" => "mixer.k_proj.weight",
        "attn_v.weight" => "mixer.v_proj.weight",
        "attn_output.weight" => "mixer.o_proj.weight",
        "ffn_gate_inp.weight" => "mixer.gate.weight",
        "exp_probs_b.bias" => "mixer.gate.e_score_correction_bias",
        "ffn_up_exps.weight" => "mixer.experts.up_proj",
        "ffn_down_exps.weight" => "mixer.experts.down_proj",
        "ffn_up_shexp.weight" => "mixer.shared_experts.up_proj.weight",
        "ffn_down_shexp.weight" => "mixer.shared_experts.down_proj.weight",
        "ffn_up.weight" => "mixer.up_proj.weight",
        "ffn_down.weight" => "mixer.down_proj.weight",
        _ => return None,
    };
    Some((format!("model.layers.{l}.{n}"), shape))
}

struct Mamba<'a> {
    /// `[inner + conv_dim + heads, H]`: gate, `x | B | C`, time step.
    in_proj: Lin<'a>,
    /// The depthwise kernels, `[kernel][conv_dim]` (the last multiplies the current token).
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    dt_bias: Vec<f32>,
    /// `-exp(A_log)` per head.
    a: Vec<f32>,
    d: Vec<f32>,
    norm: Vec<f32>,
    out_proj: Lin<'a>,
}

struct Attn<'a> {
    q: Lin<'a>,
    k: Lin<'a>,
    v: Lin<'a>,
    o: Lin<'a>,
}

/// `down(relu(up x)^2)`.
struct Mlp<'a> {
    up: Lin<'a>,
    down: Lin<'a>,
    inter: usize,
}

struct Moe<'a> {
    router: Vec<f32>, // [n_exp, H]
    bias: Vec<f32>,   // [n_exp]
    up: Vec<Lin<'a>>, // per expert [inter, H]
    down: Vec<Lin<'a>>,
    shared: Mlp<'a>,
}

// One per block, built once.
#[allow(clippy::large_enum_variant)]
enum Mixer<'a> {
    Mamba(Mamba<'a>),
    Attn(Attn<'a>),
    Moe(Moe<'a>),
    Mlp(Mlp<'a>),
}

struct Layer<'a> {
    norm: Vec<f32>,
    mixer: Mixer<'a>,
}

/// The decoding state of one block.
#[derive(Clone)]
enum LayerState {
    /// The last `kernel - 1` rows of `x | B | C` before the convolution, and each head's
    /// `[head_dim, state]` matrix.
    Mamba {
        conv: Vec<f32>,
        ssm: Vec<f32>,
    },
    Attn(LayerCache),
    None,
}

#[derive(Clone)]
pub struct NemotronSession {
    layers: Vec<LayerState>,
    pub pos: usize,
    /// The floor of the Mamba time step for the next passes (the config's `time_step_min`; 0
    /// is transformers' cached one-token step, which does not clamp).
    pub dt_min: f32,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
}

/// How a checkpoint names the language model's tensors.
struct Names {
    /// `backbone.layers.` and the like.
    layers: &'static str,
    embed: &'static str,
    norm: &'static str,
    head: &'static str,
}

const NAMES: [Names; 4] = [
    // transformers' save_pretrained, and GGUF files through `gguf_name`.
    Names {
        layers: "model.layers.",
        embed: "model.embeddings.weight",
        norm: "model.norm_f.weight",
        head: "lm_head.weight",
    },
    // The release.
    Names {
        layers: "backbone.layers.",
        embed: "backbone.embeddings.weight",
        norm: "backbone.norm_f.weight",
        head: "lm_head.weight",
    },
    // NeMo's speech LM export.
    Names {
        layers: "llm.model.layers.",
        embed: "llm.model.embed_tokens.weight",
        norm: "llm.model.norm.weight",
        head: "llm.lm_head.weight",
    },
    // A speech LM saved by transformers.
    Names {
        layers: "model.language_model.layers.",
        embed: "model.language_model.embeddings.weight",
        norm: "model.language_model.norm_f.weight",
        head: "lm_head.weight",
    },
];

pub struct NemotronModel<'a> {
    pub cfg: NemotronConfig,
    pub quant: Quant,
    embed: Table<'a>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
}

struct Loader<'a, 'c> {
    files: &'a ModelFiles,
    cfg: &'c NemotronConfig,
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

    /// A matrix: GGUF blocks as stored, float32 tensors exactly, others through `load_lin`.
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

    /// The matrices `[n_out, n_in]` of the `n` experts under `ep` (`...experts.`), `kind` being
    /// `up` or `down`: stacked (`up_proj`, `[n, n_out, n_in]`), one module per expert
    /// (`{e}.up_proj.weight`), or NeMo's stack with the matrix axes swapped (`gate_and_up_projs`
    /// / `down_projs`, `[n, n_in, n_out]`).
    fn experts(&self, ep: &str, kind: &str, n: usize, n_out: usize, n_in: usize) -> Result<Vec<Lin<'a>>> {
        let stacked = format!("{ep}{kind}_proj");
        if self.files.has(&stacked) || self.files.has(&format!("{stacked}.q8")) || self.files.has(&format!("{stacked}.q4")) {
            // Stacked float32 experts (a tiny reference checkpoint): exact, one view per expert.
            if let Some(info) = self.files.info(&stacked)
                && info.dtype == "F32"
                && self.files.ggml(&stacked).is_none()
            {
                if info.shape != [n, n_out, n_in] {
                    bail!("tensor {stacked} has shape {:?}, expected [{n}, {n_out}, {n_in}]", info.shape);
                }
                self.files.note_use(&stacked, true, false);
                return self
                    .files
                    .bytes(&stacked)?
                    .chunks(n_out * n_in * 4)
                    .map(|b| Lin::from_gguf(crate::ggml::GgmlType::F32, b, n_out, n_in))
                    .collect();
            }
            return load_lin_experts(self.files, &stacked, n, n_out, n_in, self.quant);
        }
        if self.files.has(&format!("{ep}0.{kind}_proj.weight")) {
            return (0..n)
                .map(|e| self.lin(&format!("{ep}{e}.{kind}_proj.weight"), n_out, n_in))
                .collect();
        }
        let swapped = format!("{ep}{}", if kind == "up" { "gate_and_up_projs" } else { "down_projs" });
        let info = self
            .files
            .info(&swapped)
            .with_context(|| format!("missing the experts' {kind} projections ({stacked}, {ep}0.{kind}_proj.weight or {swapped})"))?;
        if info.shape != [n, n_in, n_out] {
            bail!("tensor {swapped} has shape {:?}, expected [{n}, {n_in}, {n_out}]", info.shape);
        }
        self.files.note_use(&swapped, true, false);
        // bf16 bits of the stored values (a float32 tensor is rounded, exactly so when its
        // values are bf16's, as a tiny reference checkpoint's are).
        let owned: Vec<u16>;
        let src: &[u16] = if info.dtype == "BF16" {
            self.files.bf16(&swapped)?
        } else {
            owned = self
                .files
                .f32(&swapped)?
                .iter()
                .map(|x| half::bf16::from_f32(*x).to_bits())
                .collect();
            &owned
        };
        let quant = self.quant;
        let mut out = Vec::with_capacity(n);
        for w in src.chunks(n_in * n_out) {
            let mut t = PageVec::<u16>::zeroed(n_out * n_in);
            transpose_bf16(w, n_in, n_out, &mut t);
            out.push(match quant {
                Quant::Bf16 => Lin::bf16_owned(t),
                Quant::Q8 => Lin::q8(quantize_bf16(&t, n_out, n_in)),
                Quant::Q4 => Lin::q4(quantize_q4(&t, n_out, n_in)),
            });
        }
        Ok(out)
    }

    fn mlp(&self, p: &str, inter: usize) -> Result<Mlp<'a>> {
        let h = self.cfg.hidden_size;
        Ok(Mlp {
            up: self.lin(&format!("{p}up_proj.weight"), inter, h)?,
            down: self.lin(&format!("{p}down_proj.weight"), h, inter)?,
            inter,
        })
    }

    fn mamba(&self, mp: &str) -> Result<Mamba<'a>> {
        let c = self.cfg;
        let (h, nh, k) = (c.hidden_size, c.mamba_num_heads, c.conv_kernel);
        let (inner, cd) = (c.mamba_inner(), c.conv_dim());
        // `[channels, 1, kernel]` as `[kernel][channels]`.
        let rows = self.getf(&format!("{mp}conv1d.weight"), cd * k)?;
        let mut conv_w = vec![0f32; cd * k];
        for (ch, r) in rows.chunks(k).enumerate() {
            for (j, &x) in r.iter().enumerate() {
                conv_w[j * cd + ch] = x;
            }
        }
        // NeMo keeps the float32 parameters in a module of their own.
        let fp = if self.files.has(&format!("{mp}_fp32_params.dt_bias")) {
            format!("{mp}_fp32_params.")
        } else {
            mp.to_string()
        };
        let a = if self.files.has(&format!("{mp}ssm_a")) {
            self.getf(&format!("{mp}ssm_a"), nh)?
        } else {
            self.getf(&format!("{fp}A_log"), nh)?.iter().map(|a| -a.exp()).collect()
        };
        Ok(Mamba {
            in_proj: self.lin(&format!("{mp}in_proj.weight"), inner + cd + nh, h)?,
            conv_w,
            conv_b: self.getf(&format!("{mp}conv1d.bias"), cd)?,
            dt_bias: self.getf(&format!("{fp}dt_bias"), nh)?,
            a,
            d: self.getf(&format!("{fp}D"), nh)?,
            norm: self.getf(&format!("{mp}norm.weight"), inner)?,
            out_proj: self.lin(&format!("{mp}out_proj.weight"), h, inner)?,
        })
    }

    fn layer(&self, names: &Names, l: usize) -> Result<Layer<'a>> {
        let c = self.cfg;
        let h = c.hidden_size;
        let lp = format!("{}{l}.", names.layers);
        let mp = format!("{lp}mixer.");
        let mixer = match c.blocks[l] {
            Block::Mamba => Mixer::Mamba(self.mamba(&mp)?),
            Block::Attention => {
                let (q, kv) = (c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim);
                Mixer::Attn(Attn {
                    q: self.lin(&format!("{mp}q_proj.weight"), q, h)?,
                    k: self.lin(&format!("{mp}k_proj.weight"), kv, h)?,
                    v: self.lin(&format!("{mp}v_proj.weight"), kv, h)?,
                    o: self.lin(&format!("{mp}o_proj.weight"), h, q)?,
                })
            }
            Block::Moe => {
                let (ne, mi) = (c.n_routed_experts, c.moe_intermediate_size);
                let ep = format!("{mp}experts.");
                let bn = format!("{mp}gate.e_score_correction_bias");
                Mixer::Moe(Moe {
                    router: self.getf(&format!("{mp}gate.weight"), ne * h)?,
                    bias: if self.files.has(&bn) { self.getf(&bn, ne)? } else { vec![0.0; ne] },
                    up: self.experts(&ep, "up", ne, mi, h)?,
                    down: self.experts(&ep, "down", ne, h, mi)?,
                    shared: self.mlp(&format!("{mp}shared_experts."), c.moe_shared_expert_intermediate_size)?,
                })
            }
            Block::Mlp => Mixer::Mlp(self.mlp(&mp, c.intermediate_size)?),
        };
        Ok(Layer {
            norm: self.getf(&format!("{lp}norm.weight"), h)?,
            mixer,
        })
    }
}

/// `dst` (`[cols, rows]`) = the transpose of `src` (`[rows, cols]`), row by row of `dst`.
fn transpose_bf16(src: &[u16], rows: usize, cols: usize, dst: &mut [u16]) {
    debug_assert_eq!(src.len(), rows * cols);
    let sp = SendU16(dst.as_mut_ptr());
    crate::pool::run(cols, &|c| {
        // SAFETY: each task writes its own row of `dst`.
        let row = unsafe { std::slice::from_raw_parts_mut(sp.get().add(c * rows), rows) };
        for (r, o) in row.iter_mut().enumerate() {
            *o = src[r * cols + c];
        }
    });
}

#[derive(Clone, Copy)]
struct SendU16(*mut u16);
unsafe impl Send for SendU16 {}
unsafe impl Sync for SendU16 {}
impl SendU16 {
    #[inline(always)]
    fn get(self) -> *mut u16 {
        self.0
    }
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

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `relu(x)^2` in place.
fn relu2(x: &mut [f32]) {
    for v in x.iter_mut() {
        let r = v.max(0.0);
        *v = r * r;
    }
}

impl<'a> NemotronModel<'a> {
    pub fn load(cfg: NemotronConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let names = NAMES
            .iter()
            .find(|n| files.has(n.embed))
            .context("cannot find the token embedding (backbone.embeddings.weight, model.embeddings.weight or llm.model.embed_tokens.weight) in the checkpoint")?;
        let ld = Loader { files, cfg: &cfg, quant };
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let embed = Table::load(files, names.embed, v, h, table_quant(quant))?;
        let lm_head = ld.lin(names.head, v, h)?;
        let final_norm = ld.getf(names.norm, h)?;
        let mut layers = Vec::with_capacity(cfg.blocks.len());
        for l in 0..cfg.blocks.len() {
            layers.push(ld.layer(names, l).with_context(|| format!("layer {l}"))?);
        }
        Ok(Self {
            cfg,
            quant,
            embed,
            lm_head,
            final_norm,
            layers,
        })
    }

    pub fn new_session(&self) -> NemotronSession {
        let c = &self.cfg;
        NemotronSession {
            layers: self
                .layers
                .iter()
                .map(|l| match &l.mixer {
                    Mixer::Mamba(_) => LayerState::Mamba {
                        conv: vec![0f32; (c.conv_kernel - 1) * c.conv_dim()],
                        ssm: vec![0f32; c.mamba_inner() * c.ssm_state_size],
                    },
                    Mixer::Attn(_) => LayerState::Attn(LayerCache::new(c.num_key_value_heads, c.head_dim)),
                    _ => LayerState::None,
                })
                .collect(),
            pos: 0,
            dt_min: c.time_step_min as f32,
            iv: Default::default(),
        }
    }

    /// Weight bytes read per decoded token (the routed experts at top-k).
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes();
        let mlp = |m: &Mlp| m.up.bytes() + m.down.bytes();
        for l in &self.layers {
            n += match &l.mixer {
                Mixer::Mamba(m) => m.in_proj.bytes() + m.out_proj.bytes(),
                Mixer::Attn(a) => a.q.bytes() + a.k.bytes() + a.v.bytes() + a.o.bytes(),
                Mixer::Moe(m) => mlp(&m.shared) + self.cfg.num_experts_per_tok * (m.up[0].bytes() + m.down[0].bytes()),
                Mixer::Mlp(m) => mlp(m),
            };
        }
        n
    }

    /// LM-head logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * c.vocab_size];
        linear(&self.lm_head, c.vocab_size, c.hidden_size, hidden, t, &mut logits);
        logits
    }

    /// The input embedding of `token` (`[hidden]`).
    pub fn embed_row(&self, token: u32, out: &mut [f32]) {
        self.embed.row(token as usize, out);
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut NemotronSession, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden(sess, tokens);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos`, appending to the state; returns the final-normed hidden
    /// states `[t, hidden]`.
    pub fn forward_hidden(&self, sess: &mut NemotronSession, tokens: &[u32]) -> Vec<f32> {
        self.forward_capture(sess, tokens, &[], &[], &mut Vec::new())
    }

    /// [`Self::forward_hidden`] with the embeddings of some positions replaced (`(position in
    /// tokens, embedding)`: audio soft tokens).
    pub fn forward_hidden_mm(&self, sess: &mut NemotronSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, &[], &mut Vec::new())
    }

    /// [`Self::forward_hidden_mm`] that also writes to `feats` (`[t, layers.len() * hidden]`)
    /// the residual stream entering each of `layers`, `num_hidden_layers` standing for the last
    /// block's output before the final norm.
    pub fn forward_capture(
        &self,
        sess: &mut NemotronSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let c = &self.cfg;
        let h = c.hidden_size;
        let eps = c.layer_norm_epsilon as f32;
        let start = sess.pos;
        let timer = prof::start(Cat::Embed);
        let mut x = vec![0f32; t * h];
        // A token past the vocabulary (the audio locator, replaced below) reads row 0.
        for_chunks(&mut x, h, t > 1, |ti, row| {
            let tk = tokens[ti] as usize;
            self.embed.row(if tk < c.vocab_size { tk } else { 0 }, row);
        });
        for (p, e) in mm {
            x[p * h..(p + 1) * h].copy_from_slice(e);
        }
        drop(timer);
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
        let dt_min = sess.dt_min;
        let mut hn = vec![0f32; t * h];
        let mut o = vec![0f32; t * h];
        for (l, layer) in self.layers.iter().enumerate() {
            capture(l, &x);
            rmsnorm_f32w_rows(&x, h, &layer.norm, eps, &mut hn);
            o.fill(0.0);
            let hook = iv.as_ref().and_then(|s| s.hook(l, start));
            match (&layer.mixer, &mut sess.layers[l]) {
                (Mixer::Mamba(m), LayerState::Mamba { conv, ssm }) => self.mamba_forward(m, &hn, t, dt_min, conv, ssm, &mut o),
                (Mixer::Attn(a), LayerState::Attn(cache)) => self.attn_forward(a, &hn, t, start, cache, &mut o),
                (Mixer::Moe(m), _) => self.moe_forward(m, &hn, t, &mut o, hook.as_ref()),
                (Mixer::Mlp(m), _) => self.mlp_forward(m, &hn, t, &mut o, hook.as_ref().filter(|k| k.edits(Target::Mlp)), Target::Mlp),
                _ => unreachable!("session of another model"),
            }
            add_inplace(&mut x, &o);
            point(2 * l + 2, &mut x);
        }
        capture(self.layers.len(), &x);
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.final_norm, eps);
        sess.pos += t;
        x
    }

    /// `down(relu(up x)^2)` over `t` rows into `y`.
    fn mlp_forward(&self, m: &Mlp, x: &[f32], t: usize, y: &mut [f32], hook: Option<&NeuronHook>, target: Target) {
        let (h, mi) = (self.cfg.hidden_size, m.inter);
        let mut a = vec![0f32; t * mi];
        linear(&m.up, mi, h, x, t, &mut a);
        relu2(&mut a);
        if let Some(k) = hook {
            k.apply(target, &mut a, mi);
        }
        linear(&m.down, h, mi, &a, t, y);
    }

    /// The shared expert, then each token's routed experts (sigmoid scores, the top k of score
    /// plus bias, renormalized and scaled) added to it.
    fn moe_forward(&self, m: &Moe, x: &[f32], t: usize, y: &mut [f32], hook: Option<&NeuronHook>) {
        let c = &self.cfg;
        let (h, mi, ne) = (c.hidden_size, c.moe_intermediate_size, m.up.len());
        self.mlp_forward(&m.shared, x, t, y, hook.filter(|k| k.edits(Target::Shared)), Target::Shared);
        let mut logits = vec![0f32; t * ne];
        for_chunks(&mut logits, ne, t > 1, |ti, lg| {
            for (v, w) in lg.iter_mut().zip(m.router.chunks(h)) {
                *v = dot(w, &x[ti * h..(ti + 1) * h]);
            }
        });
        let routes: Vec<Vec<(usize, f32)>> = logits
            .chunks(ne)
            .map(|lg| route_sigmoid(lg, &m.bias, c.num_experts_per_tok, c.routed_scaling_factor as f32))
            .collect();
        let hook = hook.filter(|k| k.experts());
        if t == 1 {
            // Decode: every chosen expert sees the same row, so the up matmuls run as one batch.
            let r = &routes[0];
            let mut a = vec![vec![0f32; mi]; r.len()];
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> =
                r.iter().zip(a.iter_mut()).map(|(&(e, _), o)| (&m.up[e], mi, &mut o[..])).collect();
            linear_multi(&mut jobs, h, x, 1);
            let mut o = vec![0f32; h];
            for (&(e, w), a) in r.iter().zip(a.iter_mut()) {
                relu2(a);
                if let Some(k) = hook {
                    k.apply(Target::Expert(e), a, mi);
                }
                linear(&m.down[e], h, mi, a, 1, &mut o);
                axpy(y, w, &o);
            }
            return;
        }
        // Each expert's tokens gathered into one segment (padded to whole GEMM tiles), all
        // experts as one grouped matmul per projection.
        let mut count = vec![0usize; ne];
        for r in &routes {
            for &(e, _) in r {
                count[e] += 1;
            }
        }
        let mut segs = Vec::with_capacity(ne);
        let mut rows = 0;
        for &n in &count {
            let len = n.next_multiple_of(TN);
            segs.push((rows, len));
            rows += len;
        }
        let mut fill: Vec<usize> = segs.iter().map(|s| s.0).collect();
        let mut src = vec![usize::MAX; rows];
        let slot: Vec<Vec<(usize, f32)>> = routes
            .iter()
            .enumerate()
            .map(|(ti, r)| {
                r.iter()
                    .map(|&(e, w)| {
                        let i = fill[e];
                        fill[e] += 1;
                        src[i] = ti;
                        (i, w)
                    })
                    .collect()
            })
            .collect();
        let mut xs = vec![0f32; rows * h];
        for_chunks(&mut xs, h, true, |i, row| {
            if src[i] != usize::MAX {
                row.copy_from_slice(&x[src[i] * h..(src[i] + 1) * h]);
            }
        });
        let ups: Vec<&Lin> = m.up.iter().collect();
        let mut a = vec![0f32; rows * mi];
        linear_grouped(&ups, &segs, mi, h, &xs, &mut a);
        for_chunks(&mut a, mi, true, |_, r| relu2(r));
        if let Some(k) = hook {
            for (e, &(s0, len)) in segs.iter().enumerate() {
                if k.edits(Target::Expert(e)) {
                    k.apply_rows(Target::Expert(e), &mut a[s0 * mi..(s0 + len) * mi], mi, &src[s0..s0 + len]);
                }
            }
        }
        let downs: Vec<&Lin> = m.down.iter().collect();
        let mut o = vec![0f32; rows * h];
        linear_grouped(&downs, &segs, h, mi, &a, &mut o);
        for_chunks(y, h, true, |ti, yr| {
            for &(i, w) in &slot[ti] {
                axpy(yr, w, &o[i * h..(i + 1) * h]);
            }
        });
    }

    /// Causal grouped-query attention without positions over `xa` (`[t, H]`) into `out`.
    fn attn_forward(&self, a: &Attn, xa: &[f32], t: usize, start: usize, cache: &mut LayerCache, out: &mut [f32]) {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let mut q = vec![0f32; t * nh * hd];
        let mut k = vec![0f32; t * nkv * hd];
        let mut v = vec![0f32; t * nkv * hd];
        linear_multi(
            &mut [
                (&a.q, nh * hd, &mut q[..]),
                (&a.k, nkv * hd, &mut k[..]),
                (&a.v, nkv * hd, &mut v[..]),
            ],
            h,
            xa,
            t,
        );
        // The shared attention kernels score with scale 1.
        scale_inplace(&mut q, (hd as f32).powf(-0.5));
        cache.push(&k, &v, t);
        let mut att = vec![0f32; t * nh * hd];
        if t >= 32 {
            attention_prefill(&q, cache, nh, t, start, None, None, &mut att);
        } else {
            attention(&q, cache, nh, t, start, None, &mut att);
        }
        linear(&a.o, h, nh * hd, &att, t, out);
    }

    /// Mamba-2 over `xa` (`[t, H]`, the normed block input) into `out`, updating the convolution
    /// history `conv` and the heads' states `ssm`.
    #[allow(clippy::too_many_arguments)]
    fn mamba_forward(&self, m: &Mamba, xa: &[f32], t: usize, dt_min: f32, conv: &mut Vec<f32>, ssm: &mut [f32], out: &mut [f32]) {
        let c = &self.cfg;
        let (h, nh, hd, ng, ns, kn) = (
            c.hidden_size,
            c.mamba_num_heads,
            c.mamba_head_dim,
            c.n_groups,
            c.ssm_state_size,
            c.conv_kernel,
        );
        let (inner, cd) = (c.mamba_inner(), c.conv_dim());
        let pw = inner + cd + nh;
        let mut proj = vec![0f32; t * pw];
        linear(&m.in_proj, pw, h, xa, t, &mut proj);
        let _timer = prof::start(Cat::Attention);
        // The causal depthwise convolution over `x | B | C` (history first), then SiLU.
        let mut seq = std::mem::take(conv);
        seq.reserve(t * cd);
        for ti in 0..t {
            seq.extend_from_slice(&proj[ti * pw + inner..ti * pw + inner + cd]);
        }
        let mut xbc = vec![0f32; t * cd];
        for_chunks(&mut xbc, cd, t > 1, |ti, row| {
            row.copy_from_slice(&m.conv_b);
            for j in 0..kn {
                let src = &seq[(ti + j) * cd..(ti + j + 1) * cd];
                let wj = &m.conv_w[j * cd..(j + 1) * cd];
                for ((o, &x), &w) in row.iter_mut().zip(src).zip(wj) {
                    *o += w * x;
                }
            }
            for o in row.iter_mut() {
                *o = silu(*o);
            }
        });
        *conv = seq[t * cd..].to_vec();
        debug_assert_eq!(conv.len(), (kn - 1) * cd);
        // The recurrence, head by head over the tokens: `S <- exp(dt A) S + dt x B^T`, read
        // `S C + D x`.
        let per_group = nh / ng;
        let mut y = vec![0f32; nh * t * hd];
        let sp = SendPtr(ssm.as_mut_ptr());
        for_chunks(&mut y, t * hd, t > 1, |hh, yh| {
            // SAFETY: each head has its own state block.
            let st = unsafe { std::slice::from_raw_parts_mut(sp.get().add(hh * hd * ns), hd * ns) };
            let g = hh / per_group;
            for ti in 0..t {
                let row = &xbc[ti * cd..(ti + 1) * cd];
                let xs = &row[hh * hd..(hh + 1) * hd];
                let b = &row[inner + g * ns..inner + (g + 1) * ns];
                let cc = &row[inner + (ng + g) * ns..inner + (ng + g + 1) * ns];
                let z = proj[ti * pw + inner + cd + hh] + m.dt_bias[hh];
                let dt = (if z > 20.0 { z } else { z.exp().ln_1p() }).max(dt_min);
                let da = (dt * m.a[hh]).exp();
                for (p, s) in st.chunks_mut(ns).enumerate() {
                    scale_inplace(s, da);
                    axpy(s, dt * xs[p], b);
                    yh[ti * hd + p] = dot(s, cc) + m.d[hh] * xs[p];
                }
            }
        });
        // Gated, RMS-normed per group: [nh][t][hd] -> [t][inner].
        let gs = inner / ng;
        let eps = c.layer_norm_epsilon as f32;
        let mut z = vec![0f32; t * inner];
        for_chunks(&mut z, inner, t > 1, |ti, row| {
            let gate = &proj[ti * pw..ti * pw + inner];
            for hh in 0..nh {
                for p in 0..hd {
                    row[hh * hd + p] = y[(hh * t + ti) * hd + p] * silu(gate[hh * hd + p]);
                }
            }
            for (grp, w) in row.chunks_mut(gs).zip(m.norm.chunks(gs)) {
                let r = 1.0 / (dot(grp, grp) / gs as f32 + eps).sqrt();
                for (v, &wn) in grp.iter_mut().zip(w) {
                    *v = *v * r * wn;
                }
            }
        });
        drop(_timer);
        linear(&m.out_proj, h, inner, &z, t, out);
    }
}

impl crate::generate::TextLm for NemotronModel<'_> {
    type Sess = NemotronSession;
    fn new_session(&self) -> NemotronSession {
        NemotronModel::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.blocks.len()
    }
    fn iv_state<'s>(&self, sess: &'s mut NemotronSession) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &NemotronSession) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut NemotronSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        NemotronModel::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        NemotronModel::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut NemotronSession, tokens: &[u32]) -> Vec<f32> {
        NemotronModel::forward(self, sess, tokens)
    }
}

/// Runtime interventions (`crate::intervene`): the residual points after the embeddings and
/// after every block, and as neurons the input of `down_proj` of a dense MLP block
/// (`Target::Mlp`), of a MoE block's shared expert (`Target::Shared`) and of its routed experts
/// (`Target::Expert`, before the routing weight).
impl crate::intervene::Host for NemotronModel<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.blocks.len()
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        match (&self.layers.get(layer)?.mixer, target) {
            (Mixer::Mlp(m), Target::Mlp) => Some(m.inter),
            (Mixer::Moe(m), Target::Shared) => Some(m.shared.inter),
            (Mixer::Moe(m), Target::Expert(e)) if e < m.up.len() => Some(self.cfg.moe_intermediate_size),
            _ => None,
        }
    }
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32> {
        let (down, inter) = match (&self.layers[layer].mixer, target) {
            (Mixer::Mlp(m), _) => (&m.down, m.inter),
            (Mixer::Moe(m), Target::Expert(e)) => (&m.down[e], self.cfg.moe_intermediate_size),
            (Mixer::Moe(m), _) => (&m.shared.down, m.shared.inter),
            _ => panic!("layer {layer} has no MLP"),
        };
        let mut one = vec![0f32; inter];
        one[n] = 1.0;
        let h = self.cfg.hidden_size;
        let mut y = vec![0f32; h];
        linear(down, h, inter, &one, 1, &mut y);
        y
    }
}

/// What the command line's text loop needs of a model. The Mamba state cannot go back to an
/// earlier position, so no draft is offered (`truncate` then panics).
impl crate::dspark::Target for NemotronModel<'_> {
    fn forward_feats(
        &self,
        sess: &mut NemotronSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, layers, feats)
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        NemotronModel::embed_row(self, token, out)
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.logits(hidden, t)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut NemotronSession, len: usize) {
        assert!(len >= sess.pos, "Nemotron-H: the Mamba state cannot go back to an earlier position");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_and_names() {
        let j = json!({
            "model_type": "nemotron_h", "hidden_size": 32, "vocab_size": 64,
            "hybrid_override_pattern": "ME*E-", "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "mamba_num_heads": 4, "mamba_head_dim": 8, "n_groups": 2, "ssm_state_size": 8,
            "intermediate_size": 48, "moe_intermediate_size": 16, "moe_shared_expert_intermediate_size": 24,
            "n_routed_experts": 4, "num_experts_per_tok": 2,
        });
        let c = NemotronConfig::from_json(&j).unwrap();
        assert_eq!(c.blocks, [Block::Mamba, Block::Moe, Block::Attention, Block::Moe, Block::Mlp]);
        assert_eq!(c.describe(), "1 Mamba-2, 1 attention, 2 MoE, 1 MLP");
        assert_eq!(c.conv_dim(), 32 + 2 * 2 * 8);
        // The speech LM's config nests it, with NeMo's or transformers' block names.
        let s = json!({"model_type": "nemo_speechlm", "llm_config": {
            "hidden_size": 32, "vocab_size": 64, "layers_block_type": ["mamba", "moe", "attention"],
            "n_routed_experts": 4, "num_experts_per_tok": 2,
        }, "text_config": {
            "hidden_size": 32, "vocab_size": 64, "layers_block_type": ["linear_attention", "moe", "full_attention"],
            "n_routed_experts": 4, "num_experts_per_tok": 2,
        }});
        assert_eq!(
            NemotronConfig::from_json(&s).unwrap().blocks,
            [Block::Mamba, Block::Moe, Block::Attention]
        );
        assert!(
            NemotronConfig::from_json(&json!({"hidden_size": 8, "vocab_size": 8, "layers_block_type": ["moe"],
            "n_routed_experts": 4, "num_experts_per_tok": 2, "n_group": 2}))
            .is_err()
        );
        assert_eq!(
            gguf_name("blk.1.ffn_up_exps.weight", &[2688, 1856, 128], 52),
            Some(("model.layers.1.mixer.experts.up_proj".into(), vec![128, 1856, 2688]))
        );
        assert_eq!(
            gguf_name("blk.0.ssm_conv1d.weight", &[4, 6144], 52).map(|x| x.0).as_deref(),
            Some("model.layers.0.mixer.conv1d.weight")
        );
        // The MTP block after the model's.
        assert_eq!(gguf_name("blk.52.attn_q.weight", &[2688, 4096], 52), None);
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        }
    }

    /// A tiny random checkpoint with every block type, in one of the layouts: 0 transformers'
    /// (stacked experts), 1 the release's (one module per expert), 2 NeMo's (experts with the
    /// matrix axes swapped, `_fp32_params`). The same values in all three.
    fn tiny(dir: &Path, layout: usize) -> NemotronConfig {
        let j = json!({
            "model_type": "nemotron_h", "hidden_size": 32, "vocab_size": 48,
            "hybrid_override_pattern": "ME*EM-", "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "mamba_num_heads": 4, "mamba_head_dim": 8, "n_groups": 2, "ssm_state_size": 8,
            "intermediate_size": 40, "moe_intermediate_size": 16, "moe_shared_expert_intermediate_size": 24,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "routed_scaling_factor": 2.5,
        });
        let c = NemotronConfig::from_json(&j).unwrap();
        let (h, v, ne, mi) = (c.hidden_size, c.vocab_size, c.n_routed_experts, c.moe_intermediate_size);
        let mut rng = Rng(0x4e3);
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        // `scale` times uniform noise plus `bias`, rounded to bf16 (so that the transposed
        // experts, kept as bf16, hold the same values).
        let mut add = |name: String, shape: &[usize], scale: f32, bias: f32| {
            let n: usize = shape.iter().product();
            let vals = (0..n).map(|_| half::bf16::from_f32(bias + scale * rng.next()).to_f32()).collect();
            tensors.push((name, shape.to_vec(), vals));
        };
        let mat = |fan_in: usize, gain: f32| gain * (3.0 / fan_in as f32).sqrt();
        let n = &NAMES[layout];
        add(n.embed.into(), &[v, h], 1.5, 0.0);
        add(n.head.into(), &[v, h], mat(h, 4.0), 0.0);
        add(n.norm.into(), &[h], 0.2, 1.0);
        for (l, b) in c.blocks.iter().enumerate() {
            let lp = format!("{}{l}.", n.layers);
            let p = format!("{lp}mixer.");
            add(format!("{lp}norm.weight"), &[h], 0.2, 1.0);
            match b {
                Block::Mamba => {
                    let (inner, cd, nh) = (c.mamba_inner(), c.conv_dim(), c.mamba_num_heads);
                    let fp = if layout == 2 { format!("{p}_fp32_params.") } else { p.clone() };
                    add(format!("{p}in_proj.weight"), &[inner + cd + nh, h], mat(h, 2.0), 0.0);
                    add(format!("{p}conv1d.weight"), &[cd, 1, c.conv_kernel], 0.6, 0.0);
                    add(format!("{p}conv1d.bias"), &[cd], 0.2, 0.0);
                    add(format!("{fp}dt_bias"), &[nh], 3.0, -3.0);
                    add(format!("{fp}A_log"), &[nh], 1.0, 0.0);
                    add(format!("{fp}D"), &[nh], 0.5, 1.0);
                    add(format!("{p}norm.weight"), &[inner], 0.2, 1.0);
                    add(format!("{p}out_proj.weight"), &[h, inner], mat(inner, 2.0), 0.0);
                }
                Block::Attention => {
                    let (q, kv) = (c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim);
                    add(format!("{p}q_proj.weight"), &[q, h], mat(h, 3.0), 0.0);
                    add(format!("{p}k_proj.weight"), &[kv, h], mat(h, 3.0), 0.0);
                    add(format!("{p}v_proj.weight"), &[kv, h], mat(h, 2.0), 0.0);
                    add(format!("{p}o_proj.weight"), &[h, q], mat(q, 2.0), 0.0);
                }
                Block::Moe => {
                    add(format!("{p}gate.weight"), &[ne, h], mat(h, 3.0), 0.0);
                    add(format!("{p}gate.e_score_correction_bias"), &[ne], 0.3, 0.0);
                    add(format!("{p}experts.up_proj"), &[ne, mi, h], mat(h, 2.0), 0.0);
                    add(format!("{p}experts.down_proj"), &[ne, h, mi], mat(mi, 2.0), 0.0);
                    let si = c.moe_shared_expert_intermediate_size;
                    add(format!("{p}shared_experts.up_proj.weight"), &[si, h], mat(h, 2.0), 0.0);
                    add(format!("{p}shared_experts.down_proj.weight"), &[h, si], mat(si, 2.0), 0.0);
                }
                Block::Mlp => {
                    add(format!("{p}up_proj.weight"), &[c.intermediate_size, h], mat(h, 2.0), 0.0);
                    add(
                        format!("{p}down_proj.weight"),
                        &[h, c.intermediate_size],
                        mat(c.intermediate_size, 2.0),
                        0.0,
                    );
                }
            }
        }
        // The stacked experts in the layout's storage.
        let mut stored: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for (name, shape, vals) in tensors {
            let kind = ["up", "down"].into_iter().find(|k| name.ends_with(&format!("experts.{k}_proj")));
            match (kind, layout) {
                (Some(kind), 1) => {
                    let base = name.strip_suffix(&format!("{kind}_proj")).unwrap();
                    let per = shape[1] * shape[2];
                    for e in 0..shape[0] {
                        stored.push((
                            format!("{base}{e}.{kind}_proj.weight"),
                            shape[1..].to_vec(),
                            vals[e * per..(e + 1) * per].to_vec(),
                        ));
                    }
                }
                (Some(kind), 2) => {
                    let base = name.strip_suffix(&format!("{kind}_proj")).unwrap();
                    let (e, o, i) = (shape[0], shape[1], shape[2]);
                    let mut sw = vec![0f32; vals.len()];
                    for x in 0..e {
                        for r in 0..o {
                            for cc in 0..i {
                                sw[(x * i + cc) * o + r] = vals[(x * o + r) * i + cc];
                            }
                        }
                    }
                    let n = if kind == "up" { "gate_and_up_projs" } else { "down_projs" };
                    stored.push((format!("{base}{n}"), vec![e, i, o], sw));
                }
                _ => stored.push((name, shape, vals)),
            }
        }
        let mut head = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape, vals) in stored {
            let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
            let entry = json!({"dtype": "F32", "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]});
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
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        c
    }

    /// Without a reference (tools/nemotron_tiny.py has it): on a tiny random model the logits do
    /// not depend on how the tokens are split into passes (the Mamba state, the convolution
    /// history and the attention cache carry over), the three checkpoint layouts load the same
    /// model, an embedding override is the token whose embedding it is, and a steered residual
    /// point changes the logits.
    #[test]
    fn passes_layouts_and_overrides_agree() {
        crate::pool::init(3);
        let base = std::env::temp_dir().join(format!("oxinfer-nemotron-{}", std::process::id()));
        let ids: Vec<u32> = (0..41).map(|i| (i * 7 + 3) % 48).collect();
        let mut reference: Option<Vec<f32>> = None;
        for layout in 0..3 {
            let dir = base.join(layout.to_string());
            let cfg = tiny(&dir, layout);
            let files = ModelFiles::open(&dir, true).unwrap();
            let m = NemotronModel::load(cfg, &files, Quant::Bf16).unwrap();
            let v = m.cfg.vocab_size;
            // One pass, against chunks of 5 (the grouped expert path) and single tokens.
            let mut s = m.new_session();
            let hs = m.forward_hidden(&mut s, &ids);
            let whole = m.logits(&hs, ids.len());
            for chunk in [5usize, 1] {
                let mut s = m.new_session();
                let mut got = Vec::new();
                for c in ids.chunks(chunk) {
                    let hs = m.forward_hidden(&mut s, c);
                    got.extend(m.logits(&hs, c.len()));
                }
                let d = whole.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(d < 2e-4, "layout {layout}, chunks of {chunk}: max logit diff {d}");
            }
            assert!(whole.iter().all(|x| x.is_finite()));
            match &reference {
                None => reference = Some(whole.clone()),
                Some(r) => {
                    let d = whole.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    assert!(d < 2e-4, "layout {layout} differs from layout 0 by {d}");
                }
            }
            // An embedding override: token 5's own embedding in place of a token past the
            // vocabulary.
            let mut e = vec![0f32; m.cfg.hidden_size];
            m.embed_row(5, &mut e);
            let mut a = m.new_session();
            let ha = m.forward_hidden(&mut a, &[1, 5, 9]);
            let mut b = m.new_session();
            let hb = m.forward_hidden_mm(&mut b, &[1, v as u32, 9], &[(1, &e)]);
            assert_eq!(ha, hb);
            // The time step floor matters to this model, and a session carries it.
            let mut f = m.new_session();
            f.dt_min = 0.0;
            let hf = m.forward_hidden(&mut f, &ids);
            assert!(hs.iter().zip(&hf).any(|(a, b)| (a - b).abs() > 1e-6), "no head below the floor");
            // A steered residual point changes the logits.
            use crate::intervene::{Compiled, Ctx, Point, Positions, Spec, Steer};
            let resolve = |s: &str| -> Result<std::path::PathBuf> { Ok(s.into()) };
            let ctx = Ctx {
                resolve: &resolve,
                calibration: None,
                saes: None,
            };
            let steer = Spec {
                steer: vec![Steer {
                    vector: Some((0..m.cfg.hidden_size).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect()),
                    layer: Some(1),
                    alpha: 2.0,
                    point: Point::Post,
                    positions: Positions::All,
                    ..Default::default()
                }],
                ..Default::default()
            };
            let mut st = m.new_session();
            st.iv.iv = Some(std::sync::Arc::new(Compiled::new(&steer, &m, &ctx).unwrap()));
            let hi = m.forward_hidden(&mut st, &ids);
            assert!(hs.iter().zip(&hi).any(|(a, b)| (a - b).abs() > 1e-3), "steering changes nothing");
            let mut neutral = m.new_session();
            neutral.iv.iv = Some(std::sync::Arc::new(Compiled::new(&Spec::default(), &m, &ctx).unwrap()));
            assert_eq!(hs, m.forward_hidden(&mut neutral, &ids));
            let _ = std::fs::remove_dir_all(&dir);
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}
