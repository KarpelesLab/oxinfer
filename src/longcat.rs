//! Meituan's LongCat text models: LongCat-2.5 (`pro_v`, `LongcatCausalLM`: 38 layers of 8192),
//! LongCat-2.0 and LongCat-Flash-Lite-Sparse (the same architecture), and what they extend,
//! LongCat-Flash-Chat (`LongcatFlashForCausalLM`) and LongCat-Flash-Lite
//! (`LongcatFlashNgramForCausalLM`). One implementation; what a checkpoint has comes from its
//! config. The model code followed is transformers' `LongcatFlashModel`, the n-gram embedding of
//! LongCat-Flash-Lite's `modeling_longcat_ngram.py`, and, for what only LongCat-2.x has (the
//! sparse-attention indexer, the tokens left out of the n-grams, the `deepseek_yarn` name),
//! Meituan's SGLang-FluentLLM (branch `longcat-2.5-npu`, the deployment the model card names).
//!
//! - A layer is two sublayers and one MoE ("shortcut-connected MoE"): attention, dense MLP,
//!   attention, dense MLP on the residual stream, each behind its own RMS norm; the MoE reads the
//!   normed stream the first MLP reads and its output is added at the end of the layer, so it
//!   runs beside the second attention.
//! - MLA attention (DeepSeek-V3's): queries through a LoRA (`q_a`, RMS norm, `q_b`), keys and
//!   values expanded per head from a cached latent (`kv_lora_rank`) plus one rotary key shared by
//!   the heads (interleaved pairs, YaRN frequencies, the softmax scale times `mscale^2`). The
//!   query and the latent are scaled by `sqrt(hidden / rank)` (`mla_scale_q_lora`,
//!   `mla_scale_kv_lora`). `kv_b` is absorbed into the query and the output, so only the latent
//!   is cached and scored.
//! - Sparse attention (`index_topk`; LongCat-2.x): a lightning indexer on every `cli_factor`-th
//!   attention (the first of each layer), `I(t, s) = d^-0.5 H^-0.5 sum_h weights_proj(x_t)_h
//!   relu(wq_b(c_t)_h . k_s)` with `c_t` the normed query latent (before its scale) and `k_s =
//!   RMSNorm(wk(x_s))`, the first `qk_rope_head_dim` lanes of both rotated. A query attends to
//!   its `index_topk` best keys, the first `index_init_tokens` and its last `index_local_tokens`
//!   always among them (every key while it has no more than `index_topk`); the attentions
//!   without an indexer reuse the choice of the last one that has one.
//! - MoE: a softmax router over the routed experts and `zero_expert_num` identity experts; the
//!   `moe_topk` largest of probability plus a selection bias, weighted by their probabilities
//!   (not renormalized) times `routed_scaling_factor`. An identity expert returns its input.
//! - The n-gram ("over-encoding") embedding (`oe_*`, or `ngram_*` / `emb_*` in
//!   LongCat-Flash-Lite): `(n - 1) * k` hashed tables of `hidden / ((n - 1) k)` values, table
//!   `(g - 2) k + j` read at `sum_i token[t - i] vocab^i mod (m + 2 ((g - 2) k + j) + 1)` over the
//!   `g` last tokens (`g = 2..n`, stopping at the start, after an end-of-sequence token and at a
//!   token of `oe_ignore_tokens`), each row through its own projection onto the hidden size; the
//!   token's embedding is the mean of its word embedding and those. A token of
//!   `oe_ignore_tokens` (the image / audio / video markers) keeps its word embedding alone. The
//!   tables (16 of 16.5 M rows for LongCat-2.5, 270 GB) are read a few rows per token.
//!
//! Not here: the MTP layers (`model.mtp.*`, a speculative-decoding head), the vision tower and
//! its quantizer (`model.visual_*`), and LongCat-2.5-W4A4 (4-bit weights with 4-bit activations
//! behind learned transforms: its arithmetic is part of that checkpoint).
//!
//! Runtime interventions ([`crate::intervene`]): residual points only, `2 l + 1` after the first
//! sublayer of layer `l` (its attention and dense MLP) and `2 l + 2` at the layer's output.

use crate::config::YarnParams;
use crate::intervene::Target;
use crate::kernels::*;
use crate::mistral4::{SendPtrV, layer_norm_rows, load_kv_b, per_head, top_positions};
use crate::model::{Lin, Quant, Table, linear, linear_multi, load_lin, table_quant};
use crate::moe::{Act, Experts, NeedExperts};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

/// The epsilon of the indexer's key norm (the module default in SGLang-FluentLLM).
const INDEX_NORM_EPS: f32 = 1e-6;

/// The `model_type` oxinfer gives a config that has none (LongCat-2.0, LongCat-Flash-*): see
/// [`arch_model_type`].
pub const MODEL_TYPE: &str = "longcat";

/// Whether `model_type` is one of this family's (`pro_v`: LongCat-2.5; `longcat_flash`:
/// transformers' name for LongCat-Flash-Chat).
pub fn is_model_type(mt: &str) -> bool {
    matches!(mt, "pro_v" | "longcat" | "longcat_flash")
}

/// The model type of a config.json without `model_type`, from its `architectures`.
pub fn arch_model_type(raw: &Value) -> Option<&'static str> {
    let arch = raw.get("architectures")?.get(0)?.as_str()?;
    matches!(arch, "LongcatCausalLM" | "LongcatFlashForCausalLM" | "LongcatFlashNgramForCausalLM").then_some(MODEL_TYPE)
}

/// The sparse-attention indexer (`index_*`).
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    /// The key norm is an RMS norm (`index_k_norm_type: "rms"`), otherwise a LayerNorm.
    pub rms: bool,
    /// One attention in `every` has an indexer (`cli_factor`), counted over the sublayers.
    pub every: usize,
    /// Keys always attended: the first `init` and the query's last `local`.
    pub init: usize,
    pub local: usize,
    /// The rotary lanes are interleaved pairs (otherwise the two halves).
    pub interleave: bool,
}

/// The n-gram embedding (`oe_*` / `ngram_*`).
#[derive(Debug, Clone)]
pub struct OeConfig {
    /// The base table size `m = int(vocab * ratio)`: table `i` has `m + 2 i + 1` rows.
    pub m: usize,
    /// Tables per n-gram length (`oe_split_num`).
    pub split: usize,
    /// The longest n-gram (`oe_neighbor_num`).
    pub neighbors: usize,
    /// Tokens that keep their word embedding and end the n-grams of what follows.
    pub ignore: Vec<u32>,
}

impl OeConfig {
    pub fn tables(&self) -> usize {
        (self.neighbors - 1) * self.split
    }
    pub fn rows(&self, i: usize) -> usize {
        self.m + 2 * i + 1
    }
}

/// The settings of a LongCat text model (the root of its config.json).
#[derive(Debug, Clone)]
pub struct LongcatConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub expert_ffn_hidden_size: usize,
    /// Layers, each of two sublayers (`num_layers`).
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub mla_scale_q_lora: bool,
    pub mla_scale_kv_lora: bool,
    pub n_routed_experts: usize,
    pub zero_expert_num: usize,
    pub moe_topk: usize,
    pub routed_scaling_factor: f64,
    pub rms_norm_eps: f64,
    /// The epsilon of the RMS norms of the query LoRA and of the latent: `rms_norm_eps` in
    /// SGLang-FluentLLM (followed for `LongcatCausalLM`), the module default 1e-6 in
    /// transformers' `LongcatFlashMLA` (followed for the LongCat-Flash architectures).
    pub latent_norm_eps: f64,
    pub rope: YarnParams,
    /// The softmax scale's factor from YaRN (`mscale^2`, 1 without `mscale_all_dim`).
    pub attn_mscale2: f64,
    pub eos_token_id: u32,
    pub index: Option<IndexConfig>,
    pub oe: Option<OeConfig>,
    pub tie_word_embeddings: bool,
}

/// `"199:224"`, `"3,7:9"` or a list of ids: the tokens of `oe_ignore_tokens`, sorted.
fn ignore_tokens(v: Option<&Value>) -> Result<Vec<u32>> {
    let mut out = std::collections::BTreeSet::new();
    match v {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => {
            for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                match part.split_once(':') {
                    Some((a, b)) => out.extend(a.trim().parse::<u32>()?..b.trim().parse::<u32>()?),
                    None => {
                        out.insert(part.parse::<u32>()?);
                    }
                }
            }
        }
        Some(Value::Array(a)) => {
            for t in a {
                out.insert(t.as_u64().context("oe_ignore_tokens: not a token id")? as u32);
            }
        }
        Some(other) => bail!("oe_ignore_tokens: {other}"),
    }
    Ok(out.into_iter().collect())
}

impl LongcatConfig {
    /// The config of a checkpoint folder (`config.json`).
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        Self::from_json(&raw)
    }

    pub fn from_json(t: &Value) -> Result<Self> {
        let need = |k: &str| -> Result<usize> {
            t.get(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .with_context(|| format!("config.json: missing {k}"))
        };
        let opt = |k: &str| t.get(k).and_then(Value::as_u64).map(|v| v as usize);
        let flag = |k: &str, d: bool| t.get(k).and_then(Value::as_bool).unwrap_or(d);
        let num = |k: &str| t.get(k).and_then(Value::as_f64);
        if t.get("compression_config").is_some() || t.get("quantization_config").is_some() {
            bail!(
                "this LongCat checkpoint is quantized (W4A4 / INT8 / FP8: integer weights with quantized activations and learned transforms), which is not supported: use the bf16 release"
            );
        }
        let arch = t
            .get("architectures")
            .and_then(|a| a.get(0))
            .and_then(Value::as_str)
            .unwrap_or("LongcatCausalLM");
        let flash = arch.starts_with("LongcatFlash");
        if let Some(a) = t.get("attention_method").and_then(Value::as_str)
            && !matches!(a, "MLA" | "LSA")
        {
            bail!("attention_method {a:?} is not supported (MLA, or LSA: MLA with the sparse-attention indexer)");
        }
        if let Some(z) = t.get("zero_expert_type").and_then(Value::as_str)
            && z != "identity"
        {
            bail!("zero_expert_type {z:?} is not supported (identity)");
        }
        if flag("router_bias", false) || flag("norm_topk_prob", false) || flag("attention_bias", false) {
            bail!("router_bias, norm_topk_prob and attention_bias are not supported");
        }
        let hidden_size = need("hidden_size")?;
        let vocab_size = need("vocab_size")?;
        let rms_norm_eps = num("rms_norm_eps").unwrap_or(1e-5);
        let theta = num("rope_theta").unwrap_or(10_000_000.0);
        // `rope_scaling` (`rope_parameters` in transformers' own configs): SGLang-FluentLLM runs
        // every type as DeepSeek's YaRN, which is transformers' "yarn".
        let rs = t
            .get("rope_scaling")
            .or_else(|| t.get("rope_parameters"))
            .filter(|v| v.is_object() && v.get("factor").is_some());
        let rf = |k: &str| rs.and_then(|r| r.get(k)).and_then(Value::as_f64);
        let rope = YarnParams {
            rope_theta: rf("rope_theta").unwrap_or(theta),
            rope_type: Some(if rs.is_some() { "yarn" } else { "default" }.into()),
            factor: rf("factor"),
            original_max_position_embeddings: rf("original_max_position_embeddings").map(|v| v as usize),
            beta_fast: rf("beta_fast"),
            beta_slow: rf("beta_slow"),
            attention_factor: None,
            // (DeepSeek's defaults: the magnitude is the ratio of the two, 1 when equal.)
            mscale: Some(rf("mscale").unwrap_or(1.0)),
            mscale_all_dim: Some(rf("mscale_all_dim").filter(|&m| m != 0.0).unwrap_or(1.0)),
            truncate: None,
        };
        let attn_mscale2 = match (rs, rf("factor"), rf("mscale_all_dim")) {
            (Some(_), Some(f), Some(m)) if f > 1.0 && m != 0.0 => (0.1 * m * f.ln() + 1.0).powi(2),
            _ => 1.0,
        };
        if rs.is_some() && rf("mscale_all_dim").unwrap_or(0.0) == 0.0 && rf("mscale").is_some_and(|m| m != 0.0) {
            bail!("rope_scaling: mscale without mscale_all_dim is not supported");
        }
        let index = match opt("index_topk") {
            Some(topk) => Some(IndexConfig {
                n_heads: need("index_n_heads")?,
                head_dim: need("index_head_dim")?,
                topk,
                rms: match t.get("index_k_norm_type").and_then(Value::as_str).unwrap_or("layer") {
                    "rms" => true,
                    "layer" => false,
                    other => bail!("index_k_norm_type {other:?} is not supported (rms, layer)"),
                },
                every: opt("cli_factor").unwrap_or(1).max(1),
                init: opt("index_init_tokens").unwrap_or(0),
                local: opt("index_local_tokens").unwrap_or(0),
                interleave: flag("indexer_rope_interleave", true),
            }),
            None => None,
        };
        if t.get("indexer_types").is_some_and(|v| !v.is_null()) || flag("dsa_idxer_nope", false) {
            bail!("indexer_types / dsa_idxer_nope are not supported");
        }
        // The n-gram embedding under either set of names.
        let ratio = num("oe_vocab_size_ratio").or_else(|| num("ngram_vocab_size_ratio"));
        let oe = match ratio {
            Some(r) => {
                let neighbors = opt("oe_neighbor_num")
                    .or_else(|| opt("emb_neighbor_num"))
                    .context("config.json: missing oe_neighbor_num")?;
                let split = opt("oe_split_num")
                    .or_else(|| opt("emb_split_num"))
                    .context("config.json: missing oe_split_num")?;
                if neighbors < 2 || split == 0 {
                    bail!("oe_neighbor_num {neighbors} / oe_split_num {split}");
                }
                if t.get("embP").is_some_and(|v| !v.is_null()) {
                    bail!("embP (padded n-gram tables) is not supported");
                }
                let oe = OeConfig {
                    // (Python's `int(vocab * ratio)`, in doubles.)
                    m: (vocab_size as f64 * r) as usize,
                    split,
                    neighbors,
                    ignore: ignore_tokens(t.get("oe_ignore_tokens"))?,
                };
                if hidden_size < oe.tables() {
                    bail!("hidden_size {hidden_size} is below the {} n-gram tables", oe.tables());
                }
                Some(oe)
            }
            None => None,
        };
        let eos = match t.get("eos_token_id") {
            Some(Value::Array(a)) => a.first().and_then(Value::as_u64),
            Some(v) => v.as_u64(),
            None => None,
        };
        let c = LongcatConfig {
            vocab_size,
            hidden_size,
            ffn_hidden_size: need("ffn_hidden_size")?,
            expert_ffn_hidden_size: need("expert_ffn_hidden_size")?,
            num_layers: need("num_layers")?,
            num_attention_heads: need("num_attention_heads")?,
            kv_lora_rank: need("kv_lora_rank")?,
            q_lora_rank: need("q_lora_rank")?,
            qk_rope_head_dim: need("qk_rope_head_dim")?,
            qk_nope_head_dim: need("qk_nope_head_dim")?,
            v_head_dim: need("v_head_dim")?,
            // (transformers scales always; SGLang-FluentLLM goes by the config, default off.)
            mla_scale_q_lora: flag("mla_scale_q_lora", flash),
            mla_scale_kv_lora: flag("mla_scale_kv_lora", flash),
            n_routed_experts: need("n_routed_experts")?,
            zero_expert_num: opt("zero_expert_num").unwrap_or(0),
            moe_topk: need("moe_topk")?,
            routed_scaling_factor: num("routed_scaling_factor").unwrap_or(1.0),
            rms_norm_eps,
            latent_norm_eps: if flash { 1e-6 } else { rms_norm_eps },
            rope,
            attn_mscale2,
            eos_token_id: eos.unwrap_or(2) as u32,
            index,
            oe,
            tie_word_embeddings: flag("tie_word_embeddings", false),
        };
        if !c.qk_rope_head_dim.is_multiple_of(2) || c.qk_rope_head_dim == 0 {
            bail!("qk_rope_head_dim {}", c.qk_rope_head_dim);
        }
        if let Some(ix) = &c.index
            && ix.head_dim < c.qk_rope_head_dim
        {
            bail!("index_head_dim {} is below qk_rope_head_dim {}", ix.head_dim, c.qk_rope_head_dim);
        }
        if c.moe_topk > c.n_routed_experts + c.zero_expert_num {
            bail!("moe_topk {} exceeds the experts", c.moe_topk);
        }
        Ok(c)
    }

    fn qk(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Whether attention `a` (sublayer `a % 2` of layer `a / 2`) has an indexer.
    pub fn has_indexer(&self, a: usize) -> bool {
        self.index.as_ref().is_some_and(|ix| a.is_multiple_of(ix.every))
    }
}

/// Weight streaming for a checkpoint that does not fit in memory (as Mistral Large 4).
pub fn default_stream() -> crate::stream::StreamOptions {
    crate::mistral4::default_stream()
}

/// Whether a checkpoint of `bytes` is too big to hold in memory.
pub fn too_big(bytes: u64) -> bool {
    crate::mistral4::too_big(bytes)
}

/// Streams a model whose weights borrow `files` (memory-mapped): the n-gram projections, the
/// layers (routed experts read on demand), the LM head; the rows of the n-gram tables are read as
/// tokens need them. Detach (`set_stream(None)`) or drop the model before dropping `files`.
pub fn attach_stream(
    model: &mut LongcatModel<'_>,
    files: &ModelFiles,
    opts: crate::stream::StreamOptions,
) -> Result<std::sync::Arc<crate::stream::Streamer>> {
    let s = crate::stream::Streamer::new(&[files], model.stream_units(), opts)?;
    let streamed: Vec<bool> = s.plan().units.iter().map(|u| !u.2).collect();
    model.mark_streamed(&streamed);
    model.set_stream(Some(crate::stream::StreamHook::new(s.clone(), 0)));
    Ok(s)
}

struct Mlp<'a> {
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
    inter: usize,
}

impl Mlp<'_> {
    /// `y = down(silu(gate(x)) * up(x))` for `t` rows.
    fn forward(&self, x: &[f32], t: usize, h: usize, y: &mut [f32]) {
        let mut g = vec![0f32; t * self.inter];
        let mut u = vec![0f32; t * self.inter];
        linear_multi(
            &mut [(&self.gate, self.inter, &mut g[..]), (&self.up, self.inter, &mut u[..])],
            h,
            x,
            t,
        );
        Act::Silu.apply(&mut g, &mut u);
        linear(&self.down, h, self.inter, &g, t, y);
    }
}

struct Moe<'a> {
    /// `[n_routed + n_zero, H]`.
    router: Vec<f32>,
    bias: Vec<f32>,
    experts: Experts<'a>,
}

struct Indexer<'a> {
    wq: Lin<'a>, // [n_heads * D, q_lora]
    wk: Lin<'a>, // [D, H]
    /// The key norm's weight, and its bias when it is a LayerNorm.
    k_norm: (Vec<f32>, Option<Vec<f32>>),
    proj: Lin<'a>, // [n_heads, H]
}

struct Mla<'a> {
    q_a: Lin<'a>, // [q_lora, H]
    q_norm: Vec<f32>,
    q_b: Lin<'a>,  // [nh * (dn + dr), q_lora]
    kv_a: Lin<'a>, // [lat + dr, H]
    kv_norm: Vec<f32>,
    /// Per head: `k_b[h]` `[lat, dn]` maps a query's first `dn` lanes onto the latent, `v_b[h]`
    /// `[v, lat]` the attended latent onto the value.
    k_b: Vec<Lin<'a>>,
    v_b: Vec<Lin<'a>>,
    o: Lin<'a>, // [H, nh * v]
    indexer: Option<Indexer<'a>>,
}

struct Sub<'a> {
    ln_in: Vec<f32>,
    ln_post: Vec<f32>,
    attn: Mla<'a>,
    mlp: Mlp<'a>,
}

struct Layer<'a> {
    subs: [Sub<'a>; 2],
    moe: Moe<'a>,
}

/// The n-gram tables, their projections and the hash of each.
struct Oe<'a> {
    tables: Vec<Table<'a>>,
    /// `[H, dim]` each.
    projs: Vec<Lin<'a>>,
    /// Per table: its row count (the hash's modulus) and `vocab^i mod rows` for `i < neighbors`.
    mods: Vec<u64>,
    weights: Vec<Vec<u64>>,
    dim: usize,
    ignore: Vec<bool>,
}

/// An attention's cache: per position the normed (and scaled) latent `[lat]`, the rotated shared
/// key lanes `[dr]` and, with an indexer, the index key `[D]`.
#[derive(Clone, Default)]
struct MlaCache {
    c: Vec<f32>,
    r: Vec<f32>,
    idx: Vec<f32>,
    len: usize,
}

/// Decoding state.
#[derive(Clone)]
pub struct LongcatSession {
    /// One cache per attention (`2 * num_layers`).
    caches: Vec<MlaCache>,
    /// The tokens so far as the n-gram hash reads them: -1 where an n-gram ends (a token of
    /// `oe_ignore_tokens`, or a position whose embedding was given).
    hist: Vec<i64>,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
}

/// The keys each query of a pass attends to: None = every earlier key, or the sorted positions
/// the indexer chose.
type Selection = Vec<Option<Vec<u32>>>;

pub struct LongcatModel<'a> {
    pub cfg: LongcatConfig,
    pub quant: Quant,
    embed: Table<'a>,
    oe: Option<Oe<'a>>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
    inv_freq: Vec<f32>,
    rope_mag: f32,
    /// Weight streaming: unit 0 is the n-gram projections, unit `l + 1` layer `l`, the last one
    /// the LM head.
    stream: Option<crate::stream::StreamHook>,
}

/// Loads tensors by their names in Meituan's checkpoints, with shape checks.
struct Loader<'a, 'c> {
    files: &'a ModelFiles,
    cfg: &'c LongcatConfig,
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
        if self.files.ggml(full).is_none()
            && let Some(info) = self.files.info(full)
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
    fn mlp(&self, p: &str) -> Result<Mlp<'a>> {
        let (h, inter) = (self.cfg.hidden_size, self.cfg.ffn_hidden_size);
        Ok(Mlp {
            gate: self.lin(&format!("{p}gate_proj.weight"), inter, h)?,
            up: self.lin(&format!("{p}up_proj.weight"), inter, h)?,
            down: self.lin(&format!("{p}down_proj.weight"), h, inter)?,
            inter,
        })
    }

    fn mla(&self, l: usize, i: usize) -> Result<Mla<'a>> {
        let c = self.cfg;
        let (h, nh, lat, vd) = (c.hidden_size, c.num_attention_heads, c.kv_lora_rank, c.v_head_dim);
        let (dn, dr, ql) = (c.qk_nope_head_dim, c.qk_rope_head_dim, c.q_lora_rank);
        let ap = format!("model.layers.{l}.self_attn.{i}.");
        let (k_b, v_b) = load_kv_b(self.files, &ap, nh, lat, dn, vd)?;
        let indexer = match &c.index {
            Some(ix) if c.has_indexer(2 * l + i) => {
                let (ni, d) = (ix.n_heads, ix.head_dim);
                let ip = format!("{ap}indexer.");
                Some(Indexer {
                    wq: self.lin(&format!("{ip}wq_b.weight"), ni * d, ql)?,
                    wk: self.lin(&format!("{ip}wk.weight"), d, h)?,
                    k_norm: (
                        self.getf(&format!("{ip}k_norm.weight"), d)?,
                        if ix.rms {
                            None
                        } else {
                            Some(self.getf(&format!("{ip}k_norm.bias"), d)?)
                        },
                    ),
                    proj: self.lin(&format!("{ip}weights_proj.weight"), ni, h)?,
                })
            }
            _ => None,
        };
        Ok(Mla {
            q_a: self.lin(&format!("{ap}q_a_proj.weight"), ql, h)?,
            q_norm: self.getf(&format!("{ap}q_a_layernorm.weight"), ql)?,
            q_b: self.lin(&format!("{ap}q_b_proj.weight"), nh * (dn + dr), ql)?,
            kv_a: self.lin(&format!("{ap}kv_a_proj_with_mqa.weight"), lat + dr, h)?,
            kv_norm: self.getf(&format!("{ap}kv_a_layernorm.weight"), lat)?,
            k_b,
            v_b,
            o: self.lin(&format!("{ap}o_proj.weight"), h, nh * vd)?,
            indexer,
        })
    }

    fn sub(&self, l: usize, i: usize) -> Result<Sub<'a>> {
        let h = self.cfg.hidden_size;
        let lp = format!("model.layers.{l}.");
        Ok(Sub {
            ln_in: self.getf(&format!("{lp}input_layernorm.{i}.weight"), h)?,
            ln_post: self.getf(&format!("{lp}post_attention_layernorm.{i}.weight"), h)?,
            attn: self.mla(l, i)?,
            mlp: self.mlp(&format!("{lp}mlps.{i}."))?,
        })
    }

    fn moe(&self, l: usize) -> Result<Moe<'a>> {
        let c = self.cfg;
        let h = c.hidden_size;
        let mp = format!("model.layers.{l}.mlp.");
        let ne = c.n_routed_experts + c.zero_expert_num;
        Ok(Moe {
            router: self.getf(&format!("{mp}router.classifier.weight"), ne * h)?,
            bias: self.getf(&format!("{mp}router.e_score_correction_bias"), ne)?,
            experts: Experts::load(
                self.files,
                &format!("{mp}experts."),
                c.n_routed_experts,
                c.moe_topk,
                c.expert_ffn_hidden_size,
                h,
                self.quant,
                Act::Silu,
            )?,
        })
    }

    /// The n-gram tables (borrowed as stored: they are read a few rows per token) and their
    /// projections, under LongCat-2.x's names or LongCat-Flash-Lite's.
    fn oe(&self, oe: &OeConfig) -> Result<Oe<'a>> {
        let c = self.cfg;
        let n = oe.tables();
        let dim = c.hidden_size / n;
        let flash = self.files.has("model.ngram_embeddings.embedders.0.weight");
        let (mut tables, mut projs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut mods, mut weights) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let (tn, pn) = if flash {
                (
                    format!("model.ngram_embeddings.embedders.{i}.weight"),
                    format!("model.ngram_embeddings.post_projs.{i}.weight"),
                )
            } else {
                (format!("model.oe_embed_tokens{i}.weight"), format!("model.oe_embed_proj{i}.weight"))
            };
            let rows = oe.rows(i);
            let info = self.files.info(&tn).with_context(|| format!("missing tensor {tn}"))?;
            if info.shape != [rows, dim] {
                bail!("tensor {tn} has shape {:?}, expected [{rows}, {dim}]", info.shape);
            }
            tables.push(Table::load(self.files, &tn, rows, dim, Quant::Bf16)?);
            projs.push(self.lin(&pn, c.hidden_size, dim)?);
            let m = rows as u64;
            let mut w = Vec::with_capacity(oe.neighbors);
            let mut p = 1u64;
            for _ in 0..oe.neighbors {
                w.push(p);
                p = p * c.vocab_size as u64 % m;
            }
            mods.push(m);
            weights.push(w);
        }
        let mut ignore = vec![false; c.vocab_size];
        for &t in &oe.ignore {
            if let Some(f) = ignore.get_mut(t as usize) {
                *f = true;
            }
        }
        Ok(Oe {
            tables,
            projs,
            mods,
            weights,
            dim,
            ignore,
        })
    }
}

/// The table rows of the token at `p` of `hist` (see [`LongcatSession::hist`]): table
/// `(g - 2) * split + j` hashes the `g` last tokens, stopping at the start of the sequence, at an
/// ignored token and after an end-of-sequence token.
fn ngram_rows(oe: &OeConfig, mods: &[u64], weights: &[Vec<u64>], hist: &[i64], p: usize, eos: i64, out: &mut [usize]) {
    for g in 2..=oe.neighbors {
        for j in 0..oe.split {
            let i = (g - 2) * oe.split + j;
            let m = mods[i];
            let mut id = 0u64;
            for (back, w) in weights[i].iter().enumerate().take(g) {
                if back > p {
                    break;
                }
                let tk = hist[p - back];
                if tk < 0 || (tk == eos && back > 0) {
                    break;
                }
                id += tk as u64 * w % m;
            }
            out[i] = (id % m) as usize;
        }
    }
}

/// One token's experts: the `k` largest of `softmax(logits) + bias` (ties to the lower index),
/// each weighted by its probability times `scale`.
fn route(logits: &[f32], bias: &[f32], k: usize, scale: f32) -> Vec<(usize, f32)> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    let p: Vec<f32> = e.iter().map(|v| v / s).collect();
    let choice: Vec<f32> = p.iter().zip(bias).map(|(p, b)| p + b).collect();
    let mut idx: Vec<usize> = (0..p.len()).collect();
    let ord = |a: &usize, b: &usize| choice[*b].total_cmp(&choice[*a]).then(a.cmp(b));
    if k < idx.len() {
        idx.select_nth_unstable_by(k, ord);
        idx.truncate(k);
    }
    idx.sort_by(ord);
    idx.into_iter().map(|i| (i, p[i] * scale)).collect()
}

/// Rotates the two halves of `v` against each other (`(v[i], v[i + n / 2])` by angle `i`).
fn rope_halves(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = v.len() / 2;
    let (a, b) = v.split_at_mut(half);
    for i in 0..half {
        let (x1, x2) = (a[i], b[i]);
        a[i] = x1 * cos[i] - x2 * sin[i];
        b[i] = x2 * cos[i] + x1 * sin[i];
    }
}

impl<'a> LongcatModel<'a> {
    pub fn load(cfg: LongcatConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        if !files.has("model.embed_tokens.weight") {
            bail!("cannot find model.embed_tokens.weight in the checkpoint");
        }
        if files.has("model.layers.0.mlps.0.gate_proj.weight_scale") {
            bail!("this LongCat checkpoint holds integer-quantized weights (W4A4 / INT8), which are not supported: use the bf16 release");
        }
        let ld = Loader { files, cfg: &cfg, quant };
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let tied = cfg.tie_word_embeddings || !files.has("lm_head.weight");
        let embed = Table::load(
            files,
            "model.embed_tokens.weight",
            v,
            h,
            if tied { quant } else { table_quant(quant) },
        )?;
        let lm_head = ld.lin(if tied { "model.embed_tokens.weight" } else { "lm_head.weight" }, v, h)?;
        let oe = match &cfg.oe {
            Some(o) => Some(ld.oe(o)?),
            None => None,
        };
        let final_norm = ld.getf("model.norm.weight", h)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            layers.push(Layer {
                subs: [ld.sub(l, 0)?, ld.sub(l, 1)?],
                moe: ld.moe(l)?,
            });
        }
        let (inv_freq, rope_mag) = cfg.rope.inv_freq(cfg.qk_rope_head_dim);
        let m = Self {
            cfg,
            quant,
            embed,
            oe,
            lm_head,
            final_norm,
            layers,
            inv_freq,
            rope_mag,
            stream: None,
        };
        // Routed experts that cannot all stay in memory are read through the page cache as they
        // are routed: no accelerator mapping may pin them.
        let ram = crate::platform::total_memory().unwrap_or(u64::MAX);
        if files.total_bytes() as f64 > 0.6 * ram as f64 {
            for l in &m.layers {
                for e in l.moe.experts.lins().into_iter().flatten() {
                    e.set_transient_device(true);
                }
            }
        }
        Ok(m)
    }

    pub fn new_session(&self) -> LongcatSession {
        LongcatSession {
            caches: vec![MlaCache::default(); 2 * self.layers.len()],
            hist: Vec::new(),
            pos: 0,
            iv: Default::default(),
        }
    }

    /// Weight bytes read per decoded token (the routed experts at top-k, without the n-gram rows).
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes();
        if let Some(oe) = &self.oe {
            n += oe.projs.iter().map(|p| p.bytes()).sum::<usize>();
        }
        for l in &self.layers {
            n += self.dense_lins(l).iter().map(|w| w.bytes()).sum::<usize>();
            n += l.moe.experts.bytes_per_token();
        }
        n
    }

    /// A layer's weights apart from its routed experts.
    fn dense_lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let mut v: Vec<&Lin> = Vec::new();
        for s in &l.subs {
            let a = &s.attn;
            v.extend([&a.q_a, &a.q_b, &a.kv_a, &a.o]);
            v.extend(a.k_b.iter().chain(&a.v_b));
            if let Some(ix) = &a.indexer {
                v.extend([&ix.wq, &ix.wk, &ix.proj]);
            }
            v.extend([&s.mlp.gate, &s.mlp.up, &s.mlp.down]);
        }
        v
    }

    /// The streaming units in execution order (see `crate::stream`): the n-gram projections, the
    /// layers (routed experts read on demand), the LM head.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        let mut v = vec![crate::stream::UnitSpec {
            name: "oe_proj".into(),
            ranges: self.oe.iter().flat_map(|oe| oe.projs.iter().flat_map(|p| p.mem_ranges())).collect(),
            uses: 1.0,
            parts: Vec::new(),
        }];
        v.extend(self.layers.iter().enumerate().map(|(l, layer)| crate::stream::UnitSpec {
            name: format!("layer.{l}"),
            ranges: self.dense_lins(layer).iter().flat_map(|w| w.mem_ranges()).collect(),
            uses: 1.0,
            parts: layer.moe.experts.ranges(),
        }));
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
        let on = |i: usize| streamed.get(i).copied().unwrap_or(false);
        if let Some(oe) = &self.oe {
            for p in &oe.projs {
                p.set_transient_device(on(0));
            }
        }
        for (l, layer) in self.layers.iter().enumerate() {
            for w in self.dense_lins(layer) {
                w.set_transient_device(on(l + 1));
            }
            for e in layer.moe.experts.lins().into_iter().flatten() {
                e.set_transient_device(true);
            }
        }
        self.lm_head.set_transient_device(on(self.layers.len() + 1));
    }

    /// Brackets the n-gram projections, every layer and the LM head with `hook`'s units (None
    /// stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// LM-head logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * c.vocab_size];
        let _unit = crate::stream::enter(&self.stream, self.layers.len() + 1);
        linear(&self.lm_head, c.vocab_size, c.hidden_size, hidden, t, &mut logits);
        logits
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut LongcatSession, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos` (positions in `mm` take the given embeddings instead of the
    /// token's), appending to the caches; returns the final-normed hidden states `[t, hidden]`.
    pub fn forward_hidden_mm(&self, sess: &mut LongcatSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, &[], &mut Vec::new())
    }

    /// The input embeddings of `tokens` (`[t, H]`), which join `hist`: the word embedding, or its
    /// mean with the projected rows of the n-gram tables.
    fn embed_tokens(&self, hist: &mut Vec<i64>, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        let c = &self.cfg;
        let (h, t) = (c.hidden_size, tokens.len());
        let _timer = prof::start(Cat::Embed);
        let start = hist.len();
        let given = |i: usize| mm.iter().any(|m| m.0 == i);
        let plain = |i: usize| given(i) || self.oe.as_ref().is_some_and(|oe| oe.ignore[tokens[i] as usize]);
        hist.extend((0..t).map(|i| if plain(i) { -1 } else { tokens[i] as i64 }));
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, t > 1, |ti, row| self.embed.row(tokens[ti] as usize, row));
        if let (Some(oe), Some(oc)) = (&self.oe, &c.oe) {
            let n = oe.tables.len();
            let mut ids = vec![0usize; t * n];
            for (ti, row) in ids.chunks_mut(n).enumerate() {
                ngram_rows(oc, &oe.mods, &oe.weights, hist, start + ti, c.eos_token_id as i64, row);
            }
            if let Some(s) = &self.stream {
                let mut rows = Vec::with_capacity(t * n);
                for row in ids.chunks(n) {
                    for (i, &r) in row.iter().enumerate() {
                        rows.extend(oe.tables[i].row_ranges(r, oe.dim));
                    }
                }
                s.need_ranges(&rows);
            }
            let _unit = crate::stream::enter(&self.stream, 0);
            let mut sum = x.clone();
            let mut e = vec![0f32; t * oe.dim];
            let mut y = vec![0f32; t * h];
            for i in 0..n {
                for_chunks(&mut e, oe.dim, t > 1, |ti, row| oe.tables[i].row(ids[ti * n + i], row));
                linear(&oe.projs[i], h, oe.dim, &e, t, &mut y);
                add_inplace(&mut sum, &y);
            }
            let count = (n + 1) as f32;
            for ti in (0..t).filter(|&ti| !plain(ti)) {
                for (o, s) in x[ti * h..(ti + 1) * h].iter_mut().zip(&sum[ti * h..(ti + 1) * h]) {
                    *o = s / count;
                }
            }
        }
        for &(i, emb) in mm {
            x[i * h..(i + 1) * h].copy_from_slice(emb);
        }
        x
    }

    /// [`Self::forward_hidden_mm`] that also writes to `feats` (`[t, layers.len() * hidden]`) the
    /// residual stream entering each of `layers`, `num_layers` standing for the last layer's
    /// output.
    pub fn forward_capture(
        &self,
        sess: &mut LongcatSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let c = &self.cfg;
        let h = c.hidden_size;
        let start = sess.pos;
        debug_assert_eq!(sess.hist.len(), start);
        let eps = c.rms_norm_eps as f32;
        let mut x = self.embed_tokens(&mut sess.hist, tokens, mm);
        let positions: Vec<usize> = (start..start + t).collect();
        let (mut cos, mut sin) = rope_tables(&positions, &self.inv_freq);
        if self.rope_mag != 1.0 {
            cos.iter_mut().chain(sin.iter_mut()).for_each(|v| *v *= self.rope_mag);
        }
        let iv = sess.iv.active().then(|| sess.iv.clone());
        if let Some(iv) = &iv {
            iv.point(0, &mut x, h, start);
        }
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
        // The choice of the last attention with an indexer (the first one has one).
        let mut sel: Selection = vec![None; t];
        let mut hn = vec![0f32; t * h];
        let mut o = vec![0f32; t * h];
        let mut moe_out = vec![0f32; t * h];
        for (l, layer) in self.layers.iter().enumerate() {
            capture(l, &x);
            let _unit = crate::stream::enter(&self.stream, l + 1);
            for (i, sub) in layer.subs.iter().enumerate() {
                rmsnorm_f32w_rows(&x, h, &sub.ln_in, eps, &mut hn);
                self.mla_forward(
                    &sub.attn,
                    &mut sess.caches[2 * l + i],
                    &hn,
                    t,
                    start,
                    (&cos, &sin),
                    &mut sel,
                    &mut o,
                );
                add_inplace(&mut x, &o);
                rmsnorm_f32w_rows(&x, h, &sub.ln_post, eps, &mut hn);
                if i == 0 {
                    // The MoE reads what the first dense MLP reads; its output joins the stream
                    // at the end of the layer.
                    let need = |ids: &[usize]| {
                        if let Some(s) = &self.stream {
                            s.need_parts(l + 1, ids)
                        }
                    };
                    let need: NeedExperts = self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize]));
                    self.moe_forward(&layer.moe, &hn, t, &mut moe_out, need);
                }
                sub.mlp.forward(&hn, t, h, &mut o);
                add_inplace(&mut x, &o);
                if i == 0
                    && let Some(iv) = &iv
                {
                    iv.point(2 * l + 1, &mut x, h, start);
                }
            }
            add_inplace(&mut x, &moe_out);
            if let Some(iv) = &iv {
                iv.point(2 * l + 2, &mut x, h, start);
            }
        }
        capture(self.layers.len(), &x);
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.final_norm, eps);
        sess.pos += t;
        x
    }

    /// The MoE over `x` (`[t, H]`) into `y`: the routed experts, and the identity experts, which
    /// add the token itself times their weight.
    fn moe_forward(&self, m: &Moe, x: &[f32], t: usize, y: &mut [f32], need: NeedExperts) {
        let c = &self.cfg;
        let h = c.hidden_size;
        let (nr, ne) = (c.n_routed_experts, c.n_routed_experts + c.zero_expert_num);
        let mut logits = vec![0f32; t * ne];
        for_chunks(&mut logits, ne, t > 1, |ti, lg| {
            for (v, w) in lg.iter_mut().zip(m.router.chunks(h)) {
                *v = dot(w, &x[ti * h..(ti + 1) * h]);
            }
        });
        let all: Vec<_> = logits
            .chunks(ne)
            .map(|lg| route(lg, &m.bias, c.moe_topk, c.routed_scaling_factor as f32))
            .collect();
        let routes: Vec<Vec<(usize, f32)>> = all.iter().map(|r| r.iter().copied().filter(|e| e.0 < nr).collect()).collect();
        y.fill(0.0);
        if routes.iter().any(|r| !r.is_empty()) {
            m.experts.forward(x, t, h, &routes, y, need, None, None);
        }
        for (ti, r) in all.iter().enumerate() {
            let w: f32 = r.iter().filter(|e| e.0 >= nr).map(|e| e.1).sum();
            if w != 0.0 {
                axpy(&mut y[ti * h..(ti + 1) * h], w, &x[ti * h..(ti + 1) * h]);
            }
        }
    }

    /// An attention over `hn` (`[t, H]`, the normed input) into `out`, appending to `cache`.
    /// `sel`: the keys each query attends to, replaced when the attention has an indexer.
    #[allow(clippy::too_many_arguments)]
    fn mla_forward(
        &self,
        a: &Mla,
        cache: &mut MlaCache,
        hn: &[f32],
        t: usize,
        start: usize,
        rope: (&[f32], &[f32]),
        sel: &mut Selection,
        out: &mut [f32],
    ) {
        let c = &self.cfg;
        let (h, nh, lat, vd) = (c.hidden_size, c.num_attention_heads, c.kv_lora_rank, c.v_head_dim);
        let (dn, dr, ql, qk) = (c.qk_nope_head_dim, c.qk_rope_head_dim, c.q_lora_rank, c.qk());
        let eps = c.latent_norm_eps as f32;
        let half = dr / 2;
        let (cos, sin) = rope;
        let angle = |ti: usize| (&cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
        let ix = a.indexer.as_ref().zip(c.index.as_ref());
        let (ni, d) = ix.map_or((0, 0), |(_, ic)| (ic.n_heads, ic.head_dim));
        let mut qa = vec![0f32; t * ql];
        let mut kv = vec![0f32; t * (lat + dr)];
        let mut ik = vec![0f32; t * d];
        let mut iw = vec![0f32; t * ni];
        {
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.q_a, ql, &mut qa[..]), (&a.kv_a, lat + dr, &mut kv[..])];
            if let Some((x, _)) = ix {
                jobs.push((&x.wk, d, &mut ik[..]));
                jobs.push((&x.proj, ni, &mut iw[..]));
            }
            linear_multi(&mut jobs, h, hn, t);
        }
        rmsnorm_f32w_rows_inplace(&mut qa, ql, &a.q_norm, eps);
        let mut q = vec![0f32; t * nh * qk];
        linear(&a.q_b, nh * qk, ql, &qa, t, &mut q);
        for_chunks(&mut q, qk, t > 1, |i, qh| {
            let (cs, sn) = angle(i / nh);
            apply_rope_interleaved(&mut qh[dn..], cs, sn);
        });
        // The cache: the normed latent (scaled) and the rotated shared key lanes per token.
        let skv = if c.mla_scale_kv_lora { (h as f32 / lat as f32).sqrt() } else { 1.0 };
        for (ti, row) in kv.chunks_mut(lat + dr).enumerate() {
            let (cl, kr) = row.split_at_mut(lat);
            rmsnorm_f32w_inplace(cl, &a.kv_norm, eps);
            if skv != 1.0 {
                scale_inplace(cl, skv);
            }
            let (cs, sn) = angle(ti);
            apply_rope_interleaved(kr, cs, sn);
            cache.c.extend_from_slice(cl);
            cache.r.extend_from_slice(kr);
        }
        cache.len += t;
        debug_assert_eq!(cache.len, start + t);
        // The indexer: its keys join the cache; a query that sees more than `index_topk` keys
        // attends to its best ones only.
        if let Some((x, ic)) = ix {
            let rot = |v: &mut [f32], ti: usize| {
                let (cs, sn) = angle(ti);
                if ic.interleave {
                    apply_rope_interleaved(&mut v[..dr], cs, sn);
                } else {
                    rope_halves(&mut v[..dr], cs, sn);
                }
            };
            match &x.k_norm {
                (w, Some(b)) => layer_norm_rows(&mut ik, d, w, b, INDEX_NORM_EPS),
                (w, None) => rmsnorm_f32w_rows_inplace(&mut ik, d, w, INDEX_NORM_EPS),
            }
            for_chunks(&mut ik, d, t > 1, |ti, k| rot(k, ti));
            cache.idx.extend_from_slice(&ik);
            *sel = if start + t > ic.topk {
                let mut iq = vec![0f32; t * ni * d];
                linear(&x.wq, ni * d, ql, &qa, t, &mut iq);
                for_chunks(&mut iq, d, t > 1, |i, qh| rot(qh, i / ni));
                let ws = ((d * ni) as f32).powf(-0.5);
                iw.iter_mut().for_each(|w| *w *= ws);
                select(&iq, &iw, &cache.idx, ic, t, start)
            } else {
                vec![None; t]
            };
        }
        let mut att = vec![0f32; t * nh * vd];
        self.attend(a, &q, cache, t, start, sel, &mut att);
        linear(&a.o, h, nh * vd, &att, t, out);
    }

    /// Attention scored against the latent cache (`kv_b` absorbed): each head's first `dn` query
    /// lanes are mapped onto the latent (`k_b`) and scored, with its last `dr` lanes against the
    /// shared key lanes, at every chosen position; the attended latent is mapped onto the value
    /// (`v_b`). `out` `[t, nh * v]`.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, a: &Mla, q: &[f32], cache: &MlaCache, t: usize, start: usize, sel: &Selection, out: &mut [f32]) {
        let c = &self.cfg;
        let (nh, lat, vd) = (c.num_attention_heads, c.kv_lora_rank, c.v_head_dim);
        let (dn, dr, qk) = (c.qk_nope_head_dim, c.qk_rope_head_dim, c.qk());
        let mut qn = vec![0f32; nh * t * dn];
        for ti in 0..t {
            for hh in 0..nh {
                qn[(hh * t + ti) * dn..(hh * t + ti + 1) * dn].copy_from_slice(&q[(ti * nh + hh) * qk..(ti * nh + hh) * qk + dn]);
            }
        }
        let ql = per_head(&a.k_b, &qn, t, dn, lat);
        let _timer = prof::start(Cat::Attention);
        // The query's scale (both of its parts) goes into the softmax scale.
        let sq = if c.mla_scale_q_lora {
            (c.hidden_size as f64 / c.q_lora_rank as f64).sqrt()
        } else {
            1.0
        };
        let scale = ((qk as f64).powf(-0.5) * c.attn_mscale2 * sq) as f32;
        let lists: Vec<Option<&[u32]>> = (0..t).map(|ti| sel[ti].as_deref()).collect();
        let pos = |ti: usize, jj: usize| -> usize {
            match lists[ti] {
                Some(s) => s[jj] as usize,
                None => jj,
            }
        };
        let mut ol = vec![0f32; nh * t * lat];
        crate::attn::attend_shared(
            t,
            nh,
            lat,
            &|ti| lists[ti].map_or(start + ti + 1, |s| s.len()),
            &|ti, keys, sc| {
                let len = keys.len();
                let (mut s1, mut s2) = ([0f32; 8], [0f32; 8]);
                for h0 in (0..nh).step_by(8) {
                    let hb = 8.min(nh - h0);
                    let qls: Vec<&[f32]> = (h0..h0 + hb).map(|hh| &ql[(hh * t + ti) * lat..(hh * t + ti + 1) * lat]).collect();
                    let qrs: Vec<&[f32]> = (h0..h0 + hb)
                        .map(|hh| &q[(ti * nh + hh) * qk + dn..(ti * nh + hh + 1) * qk])
                        .collect();
                    for (jj, k) in keys.clone().enumerate() {
                        let j = pos(ti, k);
                        crate::attn::dots(&qls, &cache.c[j * lat..(j + 1) * lat], &mut s1[..hb]);
                        crate::attn::dots(&qrs, &cache.r[j * dr..(j + 1) * dr], &mut s2[..hb]);
                        for b in 0..hb {
                            sc[(h0 + b) * len + jj] = (s1[b] + s2[b]) * scale;
                        }
                    }
                }
            },
            &|_, _, row| softmax_inplace(row),
            &|ti, jj| {
                let j = pos(ti, jj);
                &cache.c[j * lat..(j + 1) * lat]
            },
            &|ti, hh| (hh * t + ti) * lat,
            &mut ol,
        );
        drop(_timer);
        let v = per_head(&a.v_b, &ol, t, lat, vd);
        for ti in 0..t {
            for hh in 0..nh {
                out[(ti * nh + hh) * vd..(ti * nh + hh + 1) * vd].copy_from_slice(&v[(hh * t + ti) * vd..(hh * t + ti + 1) * vd]);
            }
        }
    }

    /// Drops the positions at or beyond `len`.
    pub fn truncate(&self, sess: &mut LongcatSession, len: usize) {
        if len >= sess.pos {
            return;
        }
        let c = &self.cfg;
        let di = c.index.as_ref().map_or(0, |ix| ix.head_dim);
        for cache in &mut sess.caches {
            cache.c.truncate(len * c.kv_lora_rank);
            cache.r.truncate(len * c.qk_rope_head_dim);
            cache.idx.truncate(len * di);
            cache.len = cache.len.min(len);
        }
        sess.hist.truncate(len);
        sess.pos = len;
    }
}

/// Per query of the pass, the keys the indexer chooses (None when the query has no more than
/// `topk` keys): the first `init` and the query's last `local` keys, then the best scores, `topk`
/// in all (equal scores go to the earlier key). `iq` `[t, ni, d]`, `iw` `[t, ni]` (head weights,
/// scaled), `keys` `[start + t, d]`.
fn select(iq: &[f32], iw: &[f32], keys: &[f32], ic: &IndexConfig, t: usize, start: usize) -> Selection {
    let (ni, d, topk) = (ic.n_heads, ic.head_dim, ic.topk);
    let mut sel: Selection = vec![None; t];
    let rows: Vec<usize> = (0..t).filter(|&i| start + i + 1 > topk).collect();
    if rows.is_empty() {
        return sel;
    }
    let _timer = prof::start(Cat::Attention);
    let n = start + t;
    // Rows per GEMM keep the score block to ~32 MB.
    let rb = ((1usize << 23) / (ni * n)).clamp(1, 64);
    for blk in rows.chunks(rb) {
        let mut qb = vec![0f32; blk.len() * ni * d];
        for (r, &i) in blk.iter().enumerate() {
            qb[r * ni * d..(r + 1) * ni * d].copy_from_slice(&iq[i * ni * d..(i + 1) * ni * d]);
        }
        let sc = crate::sd::ops::matmul_f32(&keys[..n * d], n, d, &qb, blk.len() * ni);
        let mut picks = vec![Vec::new(); blk.len()];
        let op = SendPtrV(picks.as_mut_ptr());
        crate::pool::run(blk.len(), &|r| {
            let i = blk[r];
            let p = start + i;
            let mut s = vec![0f32; p + 1];
            for hh in 0..ni {
                let w = iw[i * ni + hh];
                let row = &sc[(r * ni + hh) * n..(r * ni + hh) * n + p + 1];
                for (o, &v) in s.iter_mut().zip(row) {
                    *o += w * v.max(0.0);
                }
            }
            // The keys always attended rank above every score.
            s[..ic.init.min(p + 1)].fill(f32::INFINITY);
            s[(p + 1).saturating_sub(ic.local)..].fill(f32::INFINITY);
            // SAFETY: each task writes its own slot.
            unsafe { *op.get().add(r) = top_positions(&s, topk) };
        });
        for (&i, pk) in blk.iter().zip(picks) {
            sel[i] = Some(pk);
        }
    }
    sel
}

impl crate::generate::TextLm for LongcatModel<'_> {
    type Sess = LongcatSession;
    fn new_session(&self) -> LongcatSession {
        LongcatModel::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut LongcatSession) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &LongcatSession) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut LongcatSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        LongcatModel::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        LongcatModel::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut LongcatSession, tokens: &[u32]) -> Vec<f32> {
        LongcatModel::forward(self, sess, tokens)
    }
}

/// The interface speculative decoding rounds use ([`crate::dspark::Target`]: prompt lookup; no
/// DSpark draft exists for this model). Every pass can be rolled back: the caches are cut.
impl crate::dspark::Target for LongcatModel<'_> {
    fn forward_feats(
        &self,
        sess: &mut LongcatSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, layers, feats)
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        if let Some(s) = &self.stream {
            s.need_ranges(&self.embed.row_ranges(token as usize, h));
        }
        self.embed.row(token as usize, out);
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        LongcatModel::logits(self, hidden, t)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut LongcatSession, len: usize) {
        LongcatModel::truncate(self, sess, len);
    }
}

/// Runtime interventions (`crate::intervene`): the residual points (see the module docs); no
/// neurons are offered (a layer has two dense MLPs and a MoE that reads beside them).
impl crate::intervene::Host for LongcatModel<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, _layer: usize, _target: Target) -> Option<usize> {
        None
    }
    fn down_column(&self, _layer: usize, _target: Target, _n: usize) -> Vec<f32> {
        vec![0.0; self.cfg.hidden_size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn released() -> Value {
        serde_json::from_str(
            r#"{
            "model_type": "pro_v", "architectures": ["LongcatCausalLM"], "vocab_size": 163840, "hidden_size": 8192,
            "ffn_hidden_size": 12288, "expert_ffn_hidden_size": 2048, "num_layers": 38, "num_attention_heads": 64,
            "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64, "v_head_dim": 128, "qk_nope_head_dim": 128,
            "mla_scale_q_lora": true, "mla_scale_kv_lora": true, "routed_scaling_factor": 9, "n_routed_experts": 768,
            "max_position_embeddings": 524288, "rms_norm_eps": 1e-05, "bos_token_id": 1, "eos_token_id": 2,
            "rope_theta": 1000000.0, "attention_method": "MLA", "zero_expert_num": 128, "zero_expert_type": "identity",
            "moe_topk": 12, "use_mla": 1, "moe_switch_token_num": 1024, "moe_impl": "mix",
            "oe_vocab_size_ratio": 100.567, "oe_neighbor_num": 5, "oe_split_num": 4, "oe_ignore_tokens": "199:224",
            "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048, "index_dtype": "bf16", "index_k_norm_type": "rms",
            "cli_factor": 2, "index_local_tokens": 1024, "index_init_tokens": 16,
            "rope_scaling": {"original_max_position_embeddings": 8192, "rope_type": "deepseek_yarn", "factor": 120,
                             "beta_fast": 32, "beta_slow": 1, "mscale": 1, "mscale_all_dim": 1}
        }"#,
        )
        .unwrap()
    }

    /// LongCat-2.5's config.json: the table sizes are those of the released tensors
    /// (`model.oe_embed_tokens0.weight` `[16476898, 512]`, `...15` `[16476928, 512]`).
    #[test]
    fn released_config() {
        let c = LongcatConfig::from_json(&released()).unwrap();
        let oe = c.oe.as_ref().unwrap();
        assert_eq!((oe.tables(), oe.rows(0), oe.rows(15)), (16, 16476898, 16476928));
        assert_eq!((oe.ignore.len(), oe.ignore[0], oe.ignore[24]), (25, 199, 223));
        let ix = c.index.as_ref().unwrap();
        assert!(ix.rms && ix.interleave);
        assert_eq!((ix.topk, ix.init, ix.local, ix.every), (2048, 16, 1024, 2));
        assert!(c.has_indexer(0) && !c.has_indexer(1) && c.has_indexer(74));
        assert_eq!(c.latent_norm_eps, 1e-5);
        // mscale = 0.1 ln(120) + 1.
        assert!((c.attn_mscale2 - 2.186_699).abs() < 1e-5, "{}", c.attn_mscale2);
        let (inv, mag) = c.rope.inv_freq(64);
        assert_eq!(mag, 1.0);
        // The slowest lane is stretched by the factor: 1 / (120 * 1e6^(62 / 64)).
        assert!((inv[0] - 1.0).abs() < 1e-6 && (inv[31] * 120.0 * 1e6f32.powf(62.0 / 64.0) - 1.0).abs() < 1e-4);
        assert_eq!(arch_model_type(&json!({"architectures": ["LongcatCausalLM"]})), Some(MODEL_TYPE));
        assert_eq!(arch_model_type(&json!({"architectures": ["LlamaForCausalLM"]})), None);
        let mut w = released();
        w["compression_config"] = json!({"format": "int-quantized"});
        assert!(LongcatConfig::from_json(&w).is_err());
    }

    /// The n-gram hash against Meituan's `NgramEmbedding` arithmetic written out: every token of
    /// the window times a power of the vocabulary size, modulo the table size.
    #[test]
    fn ngram_hash() {
        let oe = OeConfig {
            m: 1000,
            split: 2,
            neighbors: 3,
            ignore: vec![7],
        };
        let v = 50u64;
        let mods: Vec<u64> = (0..4).map(|i| oe.rows(i) as u64).collect();
        let weights: Vec<Vec<u64>> = mods.iter().map(|m| vec![1, v % m, v * v % m]).collect();
        // 9 is the end-of-sequence token; -1 an ignored token.
        let hist = [11i64, 12, 9, 13, 14, -1, 15, 16, 17];
        let rows = |p: usize| {
            let mut out = [0usize; 4];
            ngram_rows(&oe, &mods, &weights, &hist, p, 9, &mut out);
            out
        };
        let m = |i: usize| 1000 + 2 * i as u64 + 1;
        // The first token: itself in every table.
        assert_eq!(rows(0), [11; 4]);
        // Bigrams (tables 0, 1) and trigrams (2, 3).
        assert_eq!(
            rows(1),
            [12 + 11 * 50, 12 + 11 * 50, 12 + 11 * 50, 12 + 11 * 50].map(|x| x as usize)
        );
        let tri = |a: u64, b: u64, c: u64, i: usize| ((a + b * 50 + c * 2500) % m(i)) as usize;
        // The end-of-sequence token sees what is before it, and nothing sees past it.
        assert_eq!(rows(2), [9 + 12 * 50, 9 + 12 * 50, tri(9, 12, 11, 2), tri(9, 12, 11, 3)]);
        assert_eq!(rows(3), [13; 4]);
        assert_eq!(rows(4), [14 + 13 * 50; 4]);
        // An ignored token ends the window of what follows (and has no n-gram itself).
        assert_eq!(rows(5), [0; 4]);
        assert_eq!(rows(6), [15; 4]);
        assert_eq!(rows(8), [17 + 16 * 50, 17 + 16 * 50, tri(17, 16, 15, 2), tri(17, 16, 15, 3)]);
    }

    #[test]
    fn routing() {
        // Probabilities 0.5, 0.25, 0.125, 0.125; a bias lifts the last expert over the second.
        let logits = [3f32.ln() + 4f32.ln() - 6f32.ln(), 0.0, -(2f32.ln()), -(2f32.ln())];
        let r = route(&logits, &[0.0, 0.0, 0.0, 0.2], 2, 6.0);
        assert_eq!((r[0].0, r[1].0), (0, 3));
        let p: f32 = r.iter().map(|e| e.1).sum::<f32>() / 6.0;
        assert!((r[0].1 / 6.0 - 0.5).abs() < 1e-6 && (p - 0.625).abs() < 1e-6, "{r:?}");
        assert_eq!(ignore_tokens(Some(&json!("3, 7:9"))).unwrap(), vec![3, 7, 8]);
    }

    fn tiny_config(over: Value) -> LongcatConfig {
        let mut c: Value = serde_json::from_str(
            r#"{
            "architectures": ["LongcatCausalLM"], "vocab_size": 300, "hidden_size": 32, "ffn_hidden_size": 48,
            "expert_ffn_hidden_size": 24, "num_layers": 2, "num_attention_heads": 2, "kv_lora_rank": 8,
            "q_lora_rank": 12, "qk_rope_head_dim": 4, "v_head_dim": 8, "qk_nope_head_dim": 8,
            "mla_scale_q_lora": true, "mla_scale_kv_lora": true, "routed_scaling_factor": 3, "n_routed_experts": 4,
            "rms_norm_eps": 1e-5, "eos_token_id": 2, "rope_theta": 10000.0, "zero_expert_num": 2,
            "zero_expert_type": "identity", "moe_topk": 3, "oe_vocab_size_ratio": 2.5, "oe_neighbor_num": 3,
            "oe_split_num": 2, "oe_ignore_tokens": "199:224", "index_n_heads": 2, "index_head_dim": 8,
            "index_topk": 8, "index_k_norm_type": "rms", "cli_factor": 2, "index_local_tokens": 2,
            "index_init_tokens": 1,
            "rope_scaling": {"original_max_position_embeddings": 16, "rope_type": "deepseek_yarn", "factor": 8,
                             "beta_fast": 32, "beta_slow": 1, "mscale": 1, "mscale_all_dim": 1}
        }"#,
        )
        .unwrap();
        for (k, v) in over.as_object().unwrap() {
            c[k.as_str()] = v.clone();
        }
        LongcatConfig::from_json(&c).unwrap()
    }

    /// Writes a tiny random checkpoint of [`tiny_config`] in the released naming (float32).
    fn tiny_checkpoint(dir: &std::path::Path) {
        let c = tiny_config(json!({}));
        let (h, nh, v) = (c.hidden_size, c.num_attention_heads, c.vocab_size);
        let (ql, lat, dn, dr, vd) = (c.q_lora_rank, c.kv_lora_rank, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim);
        let (ne, nz, mi, ffn) = (c.n_routed_experts, c.zero_expert_num, c.expert_ffn_hidden_size, c.ffn_hidden_size);
        let ix = c.index.as_ref().unwrap();
        let oe = c.oe.as_ref().unwrap();
        let mut seed = 0x6c6f6e67u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        // `scale` times uniform noise, plus `bias`.
        let mut add = |name: String, shape: &[usize], scale: f32, bias: f32| {
            let n: usize = shape.iter().product();
            let bytes = (0..n).flat_map(|_| (bias + scale * rng()).to_le_bytes()).collect();
            tensors.push((name, shape.to_vec(), bytes));
        };
        let mat = |fan_in: usize, gain: f32| gain * (3.0 / fan_in as f32).sqrt();
        add("model.embed_tokens.weight".into(), &[v, h], 1.7, 0.0);
        add("lm_head.weight".into(), &[v, h], mat(h, 3.0), 0.0);
        add("model.norm.weight".into(), &[h], 0.2, 1.0);
        let dim = h / oe.tables();
        for i in 0..oe.tables() {
            add(format!("model.oe_embed_tokens{i}.weight"), &[oe.rows(i), dim], 1.7, 0.0);
            add(format!("model.oe_embed_proj{i}.weight"), &[h, dim], mat(dim, 1.0), 0.0);
        }
        for l in 0..c.num_layers {
            let lp = format!("model.layers.{l}.");
            for i in 0..2 {
                add(format!("{lp}input_layernorm.{i}.weight"), &[h], 0.2, 1.0);
                add(format!("{lp}post_attention_layernorm.{i}.weight"), &[h], 0.2, 1.0);
                let ap = format!("{lp}self_attn.{i}.");
                add(format!("{ap}q_a_proj.weight"), &[ql, h], mat(h, 1.0), 0.0);
                add(format!("{ap}q_a_layernorm.weight"), &[ql], 0.2, 1.0);
                add(format!("{ap}q_b_proj.weight"), &[nh * (dn + dr), ql], mat(ql, 0.7), 0.0);
                add(format!("{ap}kv_a_proj_with_mqa.weight"), &[lat + dr, h], mat(h, 1.0), 0.0);
                add(format!("{ap}kv_a_layernorm.weight"), &[lat], 0.2, 1.0);
                add(format!("{ap}kv_b_proj.weight"), &[nh * (dn + vd), lat], mat(lat, 0.5), 0.0);
                add(format!("{ap}o_proj.weight"), &[h, nh * vd], mat(nh * vd, 0.5), 0.0);
                if c.has_indexer(2 * l + i) {
                    add(
                        format!("{ap}indexer.wq_b.weight"),
                        &[ix.n_heads * ix.head_dim, ql],
                        mat(ql, 2.0),
                        0.0,
                    );
                    add(format!("{ap}indexer.wk.weight"), &[ix.head_dim, h], mat(h, 2.0), 0.0);
                    add(format!("{ap}indexer.k_norm.weight"), &[ix.head_dim], 0.2, 1.0);
                    add(format!("{ap}indexer.weights_proj.weight"), &[ix.n_heads, h], mat(h, 1.0), 0.0);
                }
                let mp = format!("{lp}mlps.{i}.");
                add(format!("{mp}gate_proj.weight"), &[ffn, h], mat(h, 1.0), 0.0);
                add(format!("{mp}up_proj.weight"), &[ffn, h], mat(h, 1.0), 0.0);
                add(format!("{mp}down_proj.weight"), &[h, ffn], mat(ffn, 0.5), 0.0);
            }
            add(format!("{lp}mlp.router.classifier.weight"), &[ne + nz, h], mat(h, 1.0), 0.0);
            add(format!("{lp}mlp.router.e_score_correction_bias"), &[ne + nz], 0.05, 0.0);
            for e in 0..ne {
                let ep = format!("{lp}mlp.experts.{e}.");
                add(format!("{ep}gate_proj.weight"), &[mi, h], mat(h, 1.0), 0.0);
                add(format!("{ep}up_proj.weight"), &[mi, h], mat(h, 1.0), 0.0);
                add(format!("{ep}down_proj.weight"), &[h, mi], mat(mi, 0.5), 0.0);
            }
        }
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, bytes) in tensors {
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
    }

    /// A tiny random checkpoint in the released naming, without a reference (`docs/longcat.md`
    /// has the agreement with transformers' and Meituan's code): the logits do not depend on how
    /// the tokens are split into passes, a rollback leaves no trace, and each mechanism (the
    /// indexer's choice, the keys always attended, the identity experts, the tokens left out of
    /// the n-grams, the n-gram tables, the LoRA scales, YaRN) changes the logits.
    #[test]
    fn tiny_model() {
        let dir = std::env::temp_dir().join(format!("oxinfer-longcat-test-{}", std::process::id()));
        tiny_checkpoint(&dir);
        let files = ModelFiles::open(&dir, true).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let load = |over: Value| LongcatModel::load(tiny_config(over), &files, Quant::Bf16).unwrap();
        let m = load(json!({}));
        let (h, v) = (m.cfg.hidden_size, m.cfg.vocab_size);
        // 40 tokens with an end-of-sequence token and two tokens left out of the n-grams.
        let mut ids: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % v as u32).collect();
        (ids[9], ids[20], ids[21]) = (2, 200, 205);
        let more = [5u32, 17, 2, 9];
        // The logits after the prompt (in passes of `chunk`) and after each of `more`.
        let run = |m: &LongcatModel, chunk: usize| -> Vec<f32> {
            let mut sess = m.new_session();
            let hs = crate::generate::prefill(m, &mut sess, &ids, chunk, &[]);
            let mut out = m.logits(&hs[hs.len() - h..], 1);
            for &t in &more {
                out.extend(m.forward(&mut sess, &[t]));
            }
            out
        };
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let base = run(&m, 40);
        assert!(base.iter().all(|x| x.is_finite()));
        let scale = base.iter().fold(0f32, |a, b| a.max(b.abs()));
        let tol = 2e-3 * scale;
        for chunk in [17, 7, 1] {
            let d = diff(&run(&m, chunk), &base);
            assert!(d <= tol, "passes of {chunk} differ from one pass by {d} (logits up to {scale})");
        }
        // A rollback: the tokens after it run as if the dropped ones never had.
        let mut sess = m.new_session();
        m.forward(&mut sess, &ids);
        m.forward(&mut sess, &[30, 200, 29]);
        m.truncate(&mut sess, ids.len());
        assert_eq!((sess.pos, sess.hist.len()), (ids.len(), ids.len()));
        let again: Vec<f32> = more.iter().flat_map(|&t| m.forward(&mut sess, &[t])).collect();
        let d = diff(&again, &base[v..]);
        assert!(d <= tol, "after a rollback the logits differ by {d}");
        // Each mechanism matters.
        for (what, over) in [
            ("the indexer's choice", json!({"index_topk": 64})),
            ("the keys always attended", json!({"index_local_tokens": 0, "index_init_tokens": 0})),
            ("the identity experts", json!({"zero_expert_num": 0, "moe_topk": 3})),
            ("the tokens left out of the n-grams", json!({"oe_ignore_tokens": ""})),
            ("the end of a sequence", json!({"eos_token_id": 299})),
            ("the query's LoRA scale", json!({"mla_scale_q_lora": false})),
            ("the latent's LoRA scale", json!({"mla_scale_kv_lora": false})),
            ("YaRN", json!({"rope_scaling": null})),
        ] {
            // (Without identity experts the router has fewer rows, and the loader checks the
            // tensor's size: that case is the loaded model with the router's last rows dropped.)
            let d = if what == "the identity experts" {
                let mut m2 = load(json!({}));
                m2.cfg.zero_expert_num = 0;
                for l in &mut m2.layers {
                    l.moe.router.truncate(m2.cfg.n_routed_experts * h);
                    l.moe.bias.truncate(m2.cfg.n_routed_experts);
                }
                diff(&run(&m2, 40), &base)
            } else {
                diff(&run(&load(over), 40), &base)
            };
            assert!(d > 20.0 * tol, "{what}: the logits moved by {d} only");
        }
    }

    #[test]
    fn forced_keys() {
        // One index head of 2 lanes; keys score by their first lane.
        let ic = IndexConfig {
            n_heads: 1,
            head_dim: 2,
            topk: 4,
            rms: true,
            every: 1,
            init: 1,
            local: 2,
            interleave: true,
        };
        let keys: Vec<f32> = [0.1f32, 0.9, 0.8, 0.2, 0.3, 0.7, 0.4, 0.5].iter().flat_map(|&s| [s, 0.0]).collect();
        // The queries at positions 6 and 7 of 8.
        let sel = select(&[1.0, 0.0, 1.0, 0.0], &[1.0, 1.0], &keys, &ic, 2, 6);
        // Position 6: key 0 (first), 5 and 6 (last two), then the best of the rest (1).
        assert_eq!(sel[0].as_deref(), Some(&[0u32, 1, 5, 6][..]));
        assert_eq!(sel[1].as_deref(), Some(&[0u32, 1, 6, 7][..]));
        // No more keys than `topk`: every key.
        assert!(select(&[1.0, 0.0], &[1.0], &keys[..8], &ic, 1, 3)[0].is_none());
    }
}
