//! Kimi K3 and K3.1 (`kimi_k3` / `kimi_k31`: Moonshot's Kimi-K3, Kimi-K3.1-Pro, 93 layers of 7168;
//! text model `KimiLinearForCausalLM`), as Moonshot's `modeling_kimi_linear.py` (published with
//! Kimi-K3; the K3.1 repository ships no modeling code):
//!
//! - No positions anywhere: order comes from the recurrent layers.
//! - KDA layers (Kimi Delta Attention, most layers): `q` / `k` / `v` projections through a short
//!   causal depthwise convolution and SiLU, then a gated delta rule per head over a `[key, value]`
//!   state: `S *= exp(g)` per key channel, `S += (beta k) (v - k^T S)`, `o = q^T S`, with `q` and
//!   `k` L2-normed, `beta = sigmoid(b_proj(x))` per head and the decay
//!   `g = lower_bound sigmoid(exp(A_log) (f_b(f_a(x)) + dt_bias))` per head and key channel. The
//!   output goes through a gated RMS norm (`norm(o) w sigmoid(g_proj(x))`) and `o_proj`. `A_log`
//!   is per head: the released checkpoints store it in a `[head_dim]` buffer whose first
//!   `num_heads` entries count (the rest is zero).
//! - MLA layers (every fourth layer and the last): DeepSeek-style latent attention with a query
//!   LoRA and no rotary part (`mla_use_nope`: the "rope" lanes of queries and keys are used as
//!   they are), served from the latent cache as for Mistral Large 4 (`kv_b` absorbed into the
//!   query and the output), and an output gate (`sigmoid(g_proj(x))` on the heads' values).
//! - Kimi K3.1's sparse attention (`dsa_config`, "gated DSA"): each MLA layer has a lightning
//!   indexer, `I(t, s) = d^-0.5 H^-0.5 sum_h weights_proj(x_t)_h relu(wq_b(c_t)_h . k_s)` with
//!   `c_t` the normed query latent and `k_s = LayerNorm(wk(x_s))`, and a query attends only to
//!   its `index_topk` best keys (all of them while it has no more than that). Moonshot published
//!   no code for it: this is what TokenSpeed (an engine the model card names) and llama.cpp
//!   implement, which agree.
//! - Cross-layer attention residuals (`attn_res_block_size`): the stream entering every
//!   `block_size`-th layer is banked and the running sum restarts from that layer's attention
//!   output; each sublayer reads a softmax mix of the banked streams and the running sum (scores:
//!   the RMS-normed stream times a learned vector), as does the final norm.
//! - A dense first layer, then MoE: a sigmoid router (top k of score plus bias, renormalized),
//!   routed experts in a narrower latent space (`routed_expert_hidden_size`: a down projection,
//!   the experts, an RMS norm, an up projection) and shared experts on the full width.
//! - "SiTU" in place of SwiGLU: `beta tanh(gate / beta) sigmoid(gate) * linear_beta
//!   tanh(up / linear_beta)`.
//!
//! Checkpoints: Moonshot's safetensors layout (tensors under `language_model.`, the routed
//! experts as compressed-tensors MXFP4, `weight_packed` / `weight_scale`, which are repacked into
//! ggml's MXFP4 blocks in memory) and llama.cpp GGUF files (`kimi-k31`, or `kimi-k3` with indexer
//! keys: the cross-layer scores fused, `-exp(A_log)` stored, `kv_b` split per head, the experts
//! stacked). The vision tower (MoonViT) is not read: text only.

use crate::gguf::Value;
use crate::intervene::{NeuronHook, Target};
use crate::kernels::*;
use crate::mistral4::{SendPtrV, layer_norm_rows, load_kv_b, per_head, top_positions};
use crate::model::{Lin, Quant, Table, linear, linear_multi, load_lin, table_quant};
use crate::moe::{Act, Experts, NeedExperts, route_sigmoid};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d_true() -> bool {
    true
}
fn d_one() -> usize {
    1
}
fn d_onef() -> f64 {
    1.0
}
fn d_eps() -> f64 {
    1e-6
}
fn d_sigmoid() -> String {
    "sigmoid".into()
}
fn d_silu() -> String {
    "silu".into()
}
fn d_layernorm() -> String {
    "layernorm".into()
}

/// `linear_attn_config`: which layers are KDA (1-based layer numbers) and the KDA shapes.
#[derive(Debug, Clone, Deserialize)]
pub struct KdaConfig {
    pub full_attn_layers: Vec<usize>,
    pub kda_layers: Vec<usize>,
    pub head_dim: usize,
    pub num_heads: usize,
    pub short_conv_kernel_size: usize,
    #[serde(default)]
    pub gate_lower_bound: Option<f64>,
    #[serde(default)]
    pub use_full_rank_gate: bool,
}

/// `dsa_config` (Kimi K3.1): the MLA layers with an indexer (1-based) and the indexer's shapes.
#[derive(Debug, Clone, Deserialize)]
pub struct DsaConfig {
    pub dsa_layers: Vec<usize>,
    pub index_topk: usize,
    pub index_head_dim: usize,
    pub index_num_heads: usize,
    #[serde(default)]
    pub index_rope_dim: usize,
    #[serde(default = "d_layernorm")]
    pub index_k_norm_type: String,
    #[serde(default = "d_one")]
    pub index_topk_freq: usize,
    /// The epsilon of the index keys' LayerNorm. The config names none: 1e-6 (DeepSeek V3.2's, as
    /// TokenSpeed); a GGUF file carries the one its converter wrote.
    #[serde(default = "d_eps")]
    pub index_norm_eps: f64,
}

/// The text config of a Kimi K3 / K3.1 checkpoint (`text_config` of its `config.json`,
/// `KimiLinearConfig`).
#[derive(Debug, Clone, Deserialize)]
pub struct KimiConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "d_silu")]
    pub hidden_act: String,
    #[serde(default)]
    pub activation_situ_beta: Option<f64>,
    #[serde(default)]
    pub activation_situ_linear_beta: Option<f64>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attn_res_block_size: Option<usize>,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "d_one")]
    pub moe_layer_freq: usize,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_token: Option<usize>,
    #[serde(default)]
    pub num_shared_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub routed_expert_hidden_size: Option<usize>,
    #[serde(default)]
    pub latent_moe_use_norm: bool,
    #[serde(default = "d_true")]
    pub moe_renormalize: bool,
    #[serde(default = "d_sigmoid")]
    pub moe_router_activation_func: String,
    #[serde(default = "d_onef")]
    pub routed_scaling_factor: f64,
    #[serde(default = "d_one")]
    pub num_expert_group: usize,
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    #[serde(default)]
    pub mla_use_nope: bool,
    #[serde(default)]
    pub mla_use_output_gate: bool,
    pub linear_attn_config: KdaConfig,
    #[serde(default)]
    pub dsa_config: Option<DsaConfig>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl KimiConfig {
    /// The text config of a checkpoint folder (`config.json`: its `text_config`, or the root of
    /// a bare `kimi_linear` one) or GGUF file.
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = if crate::gguf::is_gguf(dir) {
            crate::gguf::header(dir)?.config_json()?
        } else {
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?
        };
        Self::from_json(raw)
    }

    pub fn from_json(raw: serde_json::Value) -> Result<Self> {
        let text = raw.get("text_config").cloned().unwrap_or(raw);
        let c: KimiConfig = serde_json::from_value(text).context("parse the Kimi text config")?;
        let n = c.num_hidden_layers;
        let la = &c.linear_attn_config;
        let mut kinds = vec![0u8; n];
        for (&l, k) in la
            .kda_layers
            .iter()
            .map(|l| (l, 1))
            .chain(la.full_attn_layers.iter().map(|l| (l, 2)))
        {
            if l == 0 || l > n || kinds[l - 1] != 0 {
                bail!("kimi: layer {l} of linear_attn_config is out of range or listed twice");
            }
            kinds[l - 1] = k;
        }
        if kinds.contains(&0) {
            bail!("kimi: linear_attn_config does not name every layer");
        }
        // What Moonshot's modeling code allows but no released K3 uses is refused rather than
        // run untested.
        if !c.mla_use_nope {
            bail!("kimi: MLA layers with a rotary part (mla_use_nope = false) are not supported");
        }
        if c.q_lora_rank.is_none() {
            bail!("kimi: MLA layers without a query LoRA (q_lora_rank) are not supported");
        }
        if la.gate_lower_bound.is_none() || !la.use_full_rank_gate {
            bail!("kimi: only Kimi K3's KDA is supported (gate_lower_bound and use_full_rank_gate); Kimi-Linear-48B's is not");
        }
        if la.head_dim == 0 || la.num_heads == 0 || la.short_conv_kernel_size < 2 {
            bail!("kimi: bad KDA shapes");
        }
        if !matches!(c.hidden_act.as_str(), "situ" | "silu") {
            bail!("kimi: activation {:?} is not supported", c.hidden_act);
        }
        if c.num_experts.is_some() {
            if c.moe_router_activation_func != "sigmoid" || !c.moe_renormalize || c.num_expert_group > 1 {
                bail!("kimi: only a sigmoid router with renormalized weights and one expert group is supported");
            }
            if c.num_experts_per_token.is_none() || c.moe_intermediate_size.is_none() {
                bail!("kimi: num_experts without num_experts_per_token / moe_intermediate_size");
            }
        }
        if let Some(d) = &c.dsa_config {
            if d.index_rope_dim != 0 || d.index_k_norm_type != "layernorm" || d.index_topk_freq != 1 {
                bail!("kimi: only the released indexer is supported (index_rope_dim 0, LayerNorm'd keys, index_topk_freq 1)");
            }
            if d.index_topk == 0 || d.dsa_layers.iter().any(|&l| l == 0 || l > n || kinds[l - 1] != 2) {
                bail!("kimi: dsa_layers must be full-attention layers");
            }
        }
        if c.attn_res_block_size == Some(0) {
            bail!("kimi: attn_res_block_size 0");
        }
        Ok(c)
    }

    /// Whether layer `l` (0-based) is a KDA layer.
    pub fn is_kda(&self, l: usize) -> bool {
        self.linear_attn_config.kda_layers.contains(&(l + 1))
    }
    /// Whether MLA layer `l` (0-based) chooses its keys with an indexer.
    pub fn is_dsa(&self, l: usize) -> bool {
        self.dsa_config.as_ref().is_some_and(|d| d.dsa_layers.contains(&(l + 1)))
    }
    pub fn is_moe(&self, l: usize) -> bool {
        self.num_experts.is_some() && l >= self.first_k_dense_replace && l.is_multiple_of(self.moe_layer_freq.max(1))
    }
    /// The gated activation of every MLP.
    fn act(&self) -> Act {
        if self.hidden_act == "situ" {
            Act::Situ {
                beta: self.activation_situ_beta.filter(|&b| b != 0.0).unwrap_or(1.0) as f32,
                linear_beta: self.activation_situ_linear_beta.unwrap_or(0.0) as f32,
            }
        } else {
            Act::Silu
        }
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

pub(crate) struct Mlp<'a> {
    pub(crate) gate: Lin<'a>,
    pub(crate) up: Lin<'a>,
    pub(crate) down: Lin<'a>,
    pub(crate) inter: usize,
    pub(crate) act: Act,
}

impl Mlp<'_> {
    /// `y = down(act(gate(x), up(x)))` for `t` rows (`hook`: neuron edits of `target`).
    pub(crate) fn forward(&self, x: &[f32], t: usize, h: usize, y: &mut [f32], hook: Option<(&NeuronHook, Target)>) {
        let mut g = vec![0f32; t * self.inter];
        let mut u = vec![0f32; t * self.inter];
        linear_multi(
            &mut [(&self.gate, self.inter, &mut g[..]), (&self.up, self.inter, &mut u[..])],
            h,
            x,
            t,
        );
        self.act.apply(&mut g, &mut u);
        if let Some((hk, tg)) = hook {
            hk.apply(tg, &mut g, self.inter);
        }
        linear(&self.down, h, self.inter, &g, t, y);
    }
}

/// The routed experts' latent space: the projections around them and the norm of their output.
struct Latent<'a> {
    down: Lin<'a>, // [hl, H]
    up: Lin<'a>,   // [H, hl]
    norm: Option<Vec<f32>>,
    hl: usize,
}

struct Moe<'a> {
    router: Vec<f32>, // [n_exp, H]
    bias: Vec<f32>,
    experts: Experts<'a>,
    latent: Option<Latent<'a>>,
    shared: Option<Mlp<'a>>,
}

// One per layer, built once.
#[allow(clippy::large_enum_variant)]
enum Ffn<'a> {
    Dense(Mlp<'a>),
    Moe(Moe<'a>),
}

pub(crate) struct Kda<'a> {
    pub(crate) q: Lin<'a>, // [nh * hd, H]
    pub(crate) k: Lin<'a>,
    pub(crate) v: Lin<'a>,
    /// The q / k / v convolutions, `[nh * hd, kernel]` each.
    pub(crate) conv: [Vec<f32>; 3],
    pub(crate) f_a: Lin<'a>, // [rank, H]
    pub(crate) f_b: Lin<'a>, // [nh * hd, rank]
    /// `exp(A_log)` per head.
    pub(crate) exp_a: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>, // [nh * hd]
    pub(crate) b: Lin<'a>,        // [nh, H]
    /// Raon-1's write gate (`[nh, H]`): the value is written with `sigmoid(b + w)` while the
    /// state is erased with `sigmoid(b)`. None: both with `sigmoid(b)` (Kimi).
    pub(crate) w: Option<Lin<'a>>,
    pub(crate) g: Lin<'a>,       // [nh * hd, H]
    pub(crate) o_norm: Vec<f32>, // [hd]
    pub(crate) o: Lin<'a>,       // [H, nh * hd]
}

/// The shapes and constants of a KDA layer ([`kda_forward`]).
pub(crate) struct KdaDims {
    pub(crate) h: usize,
    pub(crate) nh: usize,
    pub(crate) hd: usize,
    /// The width between `f_a` and `f_b` (Kimi: `hd`).
    pub(crate) rank: usize,
    pub(crate) kc: usize,
    /// The decay's lower bound (`gate_lower_bound`).
    pub(crate) lower: f32,
    /// The epsilon of the gated norm of the output.
    pub(crate) eps: f32,
    /// The L2 norm of queries and keys: `x / max(|x|, 1e-6)` (PyTorch's `F.normalize`, Raon-1)
    /// instead of `x rsqrt(|x|^2 + 1e-6)` (fla-core's kernels, Kimi).
    pub(crate) clamp_norm: bool,
}

/// The shapes and constants of a gated NoPE MLA layer ([`mla_forward`]).
pub(crate) struct MlaDims {
    pub(crate) h: usize,
    pub(crate) nh: usize,
    pub(crate) lat: usize,
    pub(crate) vd: usize,
    pub(crate) dn: usize,
    pub(crate) dr: usize,
    pub(crate) ql: usize,
    /// The epsilon of the latent norms.
    pub(crate) eps: f32,
}

/// Kimi K3.1's lightning indexer.
pub(crate) struct Indexer<'a> {
    wq: Lin<'a>,                  // [n_heads * D, q_lora]
    wk: Lin<'a>,                  // [D, H]
    k_norm: (Vec<f32>, Vec<f32>), // LayerNorm [D]
    proj: Lin<'a>,                // [n_heads, H]
}

pub(crate) struct Mla<'a> {
    pub(crate) q_a: Lin<'a>, // [q_lora, H]
    pub(crate) q_norm: Vec<f32>,
    pub(crate) q_b: Lin<'a>,  // [nh * (dn + dr), q_lora]
    pub(crate) kv_a: Lin<'a>, // [lat + dr, H]
    pub(crate) kv_norm: Vec<f32>,
    /// Per head: `k_b[h]` `[lat, dn]` maps a query's first `dn` lanes onto the latent, `v_b[h]`
    /// `[v, lat]` the attended latent onto the value.
    pub(crate) k_b: Vec<Lin<'a>>,
    pub(crate) v_b: Vec<Lin<'a>>,
    pub(crate) gate: Option<Lin<'a>>, // [nh * v, H]
    pub(crate) o: Lin<'a>,            // [H, nh * v]
    pub(crate) indexer: Option<Indexer<'a>>,
}

#[allow(clippy::large_enum_variant)]
enum Attn<'a> {
    Kda(Kda<'a>),
    Mla(Mla<'a>),
}

struct Layer<'a> {
    ln_in: Vec<f32>,
    ln_post: Vec<f32>,
    /// The score vectors (`res_norm.weight * res_proj.weight`) of the mixes read by the attention
    /// and by the MLP.
    res: Option<(Vec<f32>, Vec<f32>)>,
    attn: Attn<'a>,
    ffn: Ffn<'a>,
}

/// A KDA layer's state: the delta rule's `[nh, hd, hd]` (key rows, value columns) and the last
/// `kernel - 1` inputs of the three convolutions (`[3][kernel - 1][nh * hd]`, oldest first).
#[derive(Clone)]
pub(crate) struct KdaState {
    pub(crate) s: Vec<f32>,
    pub(crate) conv: Vec<f32>,
}

/// An MLA layer's cache: per position the normed latent `[lat]`, the shared key lanes `[dr]` and
/// (K3.1) the index key `[D]`.
#[derive(Clone, Default)]
pub(crate) struct MlaCache {
    pub(crate) c: Vec<f32>,
    pub(crate) r: Vec<f32>,
    pub(crate) idx: Vec<f32>,
    pub(crate) len: usize,
}

#[derive(Clone)]
enum LayerState {
    Kda(KdaState),
    Mla(MlaCache),
}

/// What a recording pass keeps of one KDA layer so the session can roll back inside the pass: a
/// recurrent state cannot be cut back, so the state before the pass and the inputs of the
/// convolutions and of the delta rule are kept, and a rollback replays them over the kept tokens
/// (the same arithmetic as the pass, without reading a weight matrix). At Kimi K3's size that is
/// one state (6 MB) plus 0.2 MB per token, per layer.
#[derive(Clone, Default)]
pub(crate) struct KdaRecord {
    pub(crate) s0: Vec<f32>,
    pub(crate) conv0: Vec<f32>,
    /// The q / k / v projections before their convolutions, `[t, nh * hd]` each.
    pub(crate) raw: [Vec<f32>; 3],
    /// `f_b(f_a(x))` `[t, nh * hd]`, `b_proj(x)` `[t, nh]` and the write gate's projection
    /// (`[t, nh]`, empty without one).
    pub(crate) f: Vec<f32>,
    pub(crate) beta: Vec<f32>,
    pub(crate) write: Vec<f32>,
}

/// The records of the last recording pass, which started at `pos`.
#[derive(Clone)]
struct Record {
    pos: usize,
    kda: Vec<(usize, KdaRecord)>,
}

/// Decoding state.
#[derive(Clone)]
pub struct KimiSession {
    layers: Vec<LayerState>,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
    rec: Option<Box<Record>>,
}

/// The keys each query of a pass attends to: None = every earlier key, or the sorted positions
/// the indexer chose.
pub(crate) type Selection = Vec<Option<Vec<u32>>>;

pub struct KimiModel<'a> {
    pub cfg: KimiConfig,
    pub quant: Quant,
    embed: Table<'a>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    /// The score vector of the mix the final norm reads.
    out_res: Option<Vec<f32>>,
    layers: Vec<Layer<'a>>,
    /// Weight streaming: unit `l` is layer `l`, unit `num_hidden_layers` the LM head.
    stream: Option<crate::stream::StreamHook>,
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
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A matrix: GGUF blocks as stored, float32 tensors exactly, others through `load_lin`.
pub(crate) fn load_lin_exact<'a>(files: &'a ModelFiles, full: &str, n_out: usize, n_in: usize, quant: Quant) -> Result<Lin<'a>> {
    if files.ggml(full).is_none()
        && let Some(info) = files.info(full)
        && info.dtype == "F32"
    {
        if info.shape != [n_out, n_in] {
            bail!("tensor {full} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
        }
        files.note_use(full, true, false);
        return Lin::from_gguf(crate::ggml::GgmlType::F32, files.bytes(full)?, n_out, n_in);
    }
    load_lin(files, full, n_out, n_in, quant)
}

/// Loads tensors by their names in Moonshot's checkpoints, with shape checks.
struct Loader<'a, 'c> {
    files: &'a ModelFiles,
    cfg: &'c KimiConfig,
    quant: Quant,
    /// `language_model.model.` or `model.`.
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
    fn lin(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        load_lin_exact(self.files, full, n_out, n_in, self.quant)
    }
    fn mlp(&self, p: &str, inter: usize) -> Result<Mlp<'a>> {
        let h = self.cfg.hidden_size;
        Ok(Mlp {
            gate: self.lin(&format!("{p}gate_proj.weight"), inter, h)?,
            up: self.lin(&format!("{p}up_proj.weight"), inter, h)?,
            down: self.lin(&format!("{p}down_proj.weight"), h, inter)?,
            inter,
            act: self.cfg.act(),
        })
    }
    /// A cross-layer score vector: the product of `{p}_norm.weight` and `{p}_proj.weight`, or a
    /// GGUF file's `{p}_score`, which holds it.
    fn score(&self, p: &str) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let fused = format!("{p}_score");
        if self.files.has(&fused) {
            return self.getf(&fused, h);
        }
        let norm = self.getf(&format!("{p}_norm.weight"), h)?;
        let proj = self.getf(&format!("{p}_proj.weight"), h)?;
        Ok(norm.iter().zip(&proj).map(|(a, b)| a * b).collect())
    }

    fn kda(&self, l: usize) -> Result<Kda<'a>> {
        let c = self.cfg;
        let la = &c.linear_attn_config;
        let (h, nh, hd, kc) = (c.hidden_size, la.num_heads, la.head_dim, la.short_conv_kernel_size);
        let d = nh * hd;
        let ap = format!("{}layers.{l}.self_attn.", self.prefix);
        // `A_log` in a buffer of at least `num_heads` values (the released ones are `[head_dim]`
        // with the tail zero), or a GGUF file's `-exp(A_log)`.
        let neg = format!("{ap}neg_exp_a");
        let exp_a: Vec<f32> = if self.files.has(&neg) {
            self.getf(&neg, nh)?.iter().map(|v| -v).collect()
        } else {
            let name = format!("{ap}A_log");
            let a = self.files.f32(&name).with_context(|| format!("missing tensor {name}"))?;
            if a.len() < nh {
                bail!("tensor {name} has {} values for {nh} heads", a.len());
            }
            a[..nh].iter().map(|v| v.exp()).collect()
        };
        let conv = |n: &str| self.getf(&format!("{ap}{n}_conv1d.weight"), d * kc);
        Ok(Kda {
            q: self.lin(&format!("{ap}q_proj.weight"), d, h)?,
            k: self.lin(&format!("{ap}k_proj.weight"), d, h)?,
            v: self.lin(&format!("{ap}v_proj.weight"), d, h)?,
            conv: [conv("q")?, conv("k")?, conv("v")?],
            f_a: self.lin(&format!("{ap}f_a_proj.weight"), hd, h)?,
            f_b: self.lin(&format!("{ap}f_b_proj.weight"), d, hd)?,
            exp_a,
            dt_bias: self.getf(&format!("{ap}dt_bias"), d)?,
            b: self.lin(&format!("{ap}b_proj.weight"), nh, h)?,
            w: None,
            g: self.lin(&format!("{ap}g_proj.weight"), d, h)?,
            o_norm: self.getf(&format!("{ap}o_norm.weight"), hd)?,
            o: self.lin(&format!("{ap}o_proj.weight"), h, d)?,
        })
    }

    fn mla(&self, l: usize) -> Result<Mla<'a>> {
        let c = self.cfg;
        let (h, nh, lat, vd) = (c.hidden_size, c.num_attention_heads, c.kv_lora_rank, c.v_head_dim);
        let (dn, dr, ql) = (c.qk_nope_head_dim, c.qk_rope_head_dim, c.q_lora_rank.unwrap());
        let ap = format!("{}layers.{l}.self_attn.", self.prefix);
        let (k_b, v_b) = load_kv_b(self.files, &ap, nh, lat, dn, vd)?;
        let gname = format!("{ap}g_proj.weight");
        let gate = if self.files.has(&gname) {
            Some(self.lin(&gname, nh * vd, h)?)
        } else if c.mla_use_output_gate {
            bail!("missing tensor {gname}");
        } else {
            None
        };
        let indexer = match &c.dsa_config {
            Some(dc) if c.is_dsa(l) => {
                let (ni, d) = (dc.index_num_heads, dc.index_head_dim);
                let ip = format!("{ap}indexer.");
                Some(Indexer {
                    wq: self.lin(&format!("{ip}wq_b.weight"), ni * d, ql)?,
                    wk: self.lin(&format!("{ip}wk.weight"), d, h)?,
                    k_norm: (
                        self.getf(&format!("{ip}k_norm.weight"), d)?,
                        self.getf(&format!("{ip}k_norm.bias"), d)?,
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
            gate,
            o: self.lin(&format!("{ap}o_proj.weight"), h, nh * vd)?,
            indexer,
        })
    }

    fn ffn(&self, l: usize) -> Result<Ffn<'a>> {
        let c = self.cfg;
        let h = c.hidden_size;
        let lp = format!("{}layers.{l}.", self.prefix);
        if !c.is_moe(l) {
            return Ok(Ffn::Dense(self.mlp(&format!("{lp}mlp."), c.intermediate_size)?));
        }
        let mp = format!("{lp}block_sparse_moe.");
        let (ne, mi) = (c.num_experts.unwrap(), c.moe_intermediate_size.unwrap());
        let hl = c.routed_expert_hidden_size.unwrap_or(h);
        let latent = match c.routed_expert_hidden_size {
            Some(_) => Some(Latent {
                down: self.lin(&format!("{mp}routed_expert_down_proj.weight"), hl, h)?,
                up: self.lin(&format!("{mp}routed_expert_up_proj.weight"), h, hl)?,
                // (llama.cpp reads the norm when the file has it.)
                norm: if c.latent_moe_use_norm
                    && (self.files.has(&format!("{mp}routed_expert_norm.weight")) || self.files.metadata("format") != Some("gguf"))
                {
                    Some(self.getf(&format!("{mp}routed_expert_norm.weight"), hl)?)
                } else {
                    None
                },
                hl,
            }),
            None => None,
        };
        let shared = match c.num_shared_experts {
            Some(n) if n > 0 => Some(self.mlp(&format!("{mp}shared_experts."), mi * n)?),
            _ => None,
        };
        Ok(Ffn::Moe(Moe {
            router: self.getf(&format!("{mp}gate.weight"), ne * h)?,
            bias: self.getf(&format!("{mp}gate.e_score_correction_bias"), ne)?,
            experts: Experts::load(
                self.files,
                &format!("{mp}experts."),
                ne,
                c.num_experts_per_token.unwrap(),
                mi,
                hl,
                self.quant,
                c.act(),
            )?,
            latent,
            shared,
        }))
    }
}

impl<'a> KimiModel<'a> {
    pub fn load(cfg: KimiConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let (prefix, head) = if files.has("language_model.model.embed_tokens.weight") {
            ("language_model.model.", "language_model.lm_head.weight")
        } else if files.has("model.embed_tokens.weight") {
            ("model.", "lm_head.weight")
        } else {
            bail!("cannot find embed_tokens in the checkpoint");
        };
        // The released safetensors keep the routed experts as compressed-tensors MXFP4 (codes and
        // scales apart), which are repacked into ggml's blocks in memory: that copy cannot be
        // streamed from storage.
        if files.metadata("format") != Some("gguf")
            && !crate::platform::fits_in_memory(files.total_bytes())
            && files.names().any(|n| n.ends_with(".weight_packed"))
        {
            bail!(
                "the {:.0} GB of this checkpoint do not fit in memory, and its MXFP4 experts (compressed-tensors) are repacked in memory: use a GGUF conversion, which is streamed from storage (an MXFP4_MOE one holds the same weights)",
                files.total_bytes() as f64 / 1e9
            );
        }
        let ld = Loader {
            files,
            cfg: &cfg,
            quant,
            prefix,
        };
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let embed_name = format!("{prefix}embed_tokens.weight");
        let tied = cfg.tie_word_embeddings || !files.has(head);
        let embed = Table::load(files, &embed_name, v, h, if tied { quant } else { table_quant(quant) })?;
        let lm_head = if tied { ld.lin(&embed_name, v, h)? } else { ld.lin(head, v, h)? };
        let final_norm = ld.getf(&format!("{prefix}norm.weight"), h)?;
        let res = cfg.attn_res_block_size.is_some();
        let out_res = if res {
            Some(ld.score(&format!("{prefix}output_attn_res"))?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}layers.{l}.");
            layers.push(Layer {
                ln_in: ld.getf(&format!("{lp}input_layernorm.weight"), h)?,
                ln_post: ld.getf(&format!("{lp}post_attention_layernorm.weight"), h)?,
                res: if res {
                    Some((ld.score(&format!("{lp}self_attention_res"))?, ld.score(&format!("{lp}mlp_res"))?))
                } else {
                    None
                },
                attn: if cfg.is_kda(l) {
                    Attn::Kda(ld.kda(l)?)
                } else {
                    Attn::Mla(ld.mla(l)?)
                },
                ffn: ld.ffn(l)?,
            });
        }
        let m = Self {
            cfg,
            quant,
            embed,
            lm_head,
            final_norm,
            out_res,
            layers,
            stream: None,
        };
        // Routed experts that cannot all stay in memory are read through the page cache as they
        // are routed: no accelerator mapping may pin them.
        let ram = crate::platform::total_memory().unwrap_or(u64::MAX);
        if files.total_bytes() as f64 > 0.6 * ram as f64 {
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

    fn kda_dims(&self) -> KdaDims {
        let c = &self.cfg;
        let la = &c.linear_attn_config;
        KdaDims {
            h: c.hidden_size,
            nh: la.num_heads,
            hd: la.head_dim,
            rank: la.head_dim,
            kc: la.short_conv_kernel_size,
            lower: la.gate_lower_bound.unwrap() as f32,
            eps: c.rms_norm_eps as f32,
            clamp_norm: false,
        }
    }

    fn mla_dims(&self) -> MlaDims {
        let c = &self.cfg;
        MlaDims {
            h: c.hidden_size,
            nh: c.num_attention_heads,
            lat: c.kv_lora_rank,
            vd: c.v_head_dim,
            dn: c.qk_nope_head_dim,
            dr: c.qk_rope_head_dim,
            ql: c.q_lora_rank.unwrap(),
            eps: c.rms_norm_eps as f32,
        }
    }

    pub fn new_session(&self) -> KimiSession {
        let la = &self.cfg.linear_attn_config;
        let d = la.num_heads * la.head_dim;
        KimiSession {
            layers: self
                .layers
                .iter()
                .map(|l| match &l.attn {
                    Attn::Kda(_) => LayerState::Kda(KdaState {
                        s: vec![0.0; d * la.head_dim],
                        conv: vec![0.0; 3 * (la.short_conv_kernel_size - 1) * d],
                    }),
                    Attn::Mla(_) => LayerState::Mla(MlaCache::default()),
                })
                .collect(),
            pos: 0,
            iv: Default::default(),
            rec: None,
        }
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

    /// A layer's weights apart from its routed experts.
    fn dense_lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let mut v: Vec<&Lin> = match &l.attn {
            Attn::Kda(k) => vec![&k.q, &k.k, &k.v, &k.f_a, &k.f_b, &k.b, &k.g, &k.o],
            Attn::Mla(a) => {
                let mut v = vec![&a.q_a, &a.q_b, &a.kv_a, &a.o];
                v.extend(a.gate.as_ref());
                v.extend(a.k_b.iter().chain(&a.v_b));
                if let Some(ix) = &a.indexer {
                    v.extend([&ix.wq, &ix.wk, &ix.proj]);
                }
                v
            }
        };
        match &l.ffn {
            Ffn::Dense(m) => v.extend([&m.gate, &m.up, &m.down]),
            Ffn::Moe(m) => {
                if let Some(s) = &m.shared {
                    v.extend([&s.gate, &s.up, &s.down]);
                }
                if let Some(lt) = &m.latent {
                    v.extend([&lt.down, &lt.up]);
                }
            }
        }
        v
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

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut KimiSession, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos` (positions in `mm` take the given embeddings instead of the
    /// token's), appending to the caches; returns the final-normed hidden states `[t, hidden]`.
    pub fn forward_hidden_mm(&self, sess: &mut KimiSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, &[], &mut Vec::new(), false)
    }

    /// The softmax mix of the banked streams `blocks` and the running sum `x` that a sublayer
    /// reads: each scored by its RMS-normed values times `w`, the mix taken over the values
    /// themselves.
    fn res_mix(&self, x: &[f32], blocks: &[Vec<f32>], w: &[f32], t: usize, out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let nb = blocks.len();
        for_chunks(out, h, t > 1, |ti, o| {
            let row = |i: usize| -> &[f32] {
                if i < nb {
                    &blocks[i][ti * h..(ti + 1) * h]
                } else {
                    &x[ti * h..(ti + 1) * h]
                }
            };
            let mut p = vec![0f32; nb + 1];
            for (i, s) in p.iter_mut().enumerate() {
                let v = row(i);
                *s = dot(v, w) * (dot(v, v) / h as f32 + eps).powf(-0.5);
            }
            softmax_inplace(&mut p);
            o.fill(0.0);
            for (i, &pi) in p.iter().enumerate() {
                axpy(o, pi, row(i));
            }
        });
    }

    /// [`Self::forward_hidden_mm`] that also writes to `feats` (`[t, layers.len() * hidden]`) the
    /// running sum entering each of `layers`, `num_hidden_layers` standing for the last layer's
    /// output. `record`: keep what a rollback inside this pass needs ([`KimiSession::truncate`]).
    pub fn forward_capture(
        &self,
        sess: &mut KimiSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
        record: bool,
    ) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let c = &self.cfg;
        let h = c.hidden_size;
        let start = sess.pos;
        let eps = c.rms_norm_eps as f32;
        // (New tokens end what the last recorded pass could roll back.)
        sess.rec = None;
        let mut rec = record.then(|| Record {
            pos: start,
            kda: Vec::new(),
        });
        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, t > 1, |ti, row| self.embed.row(tokens[ti] as usize, row));
        for &(i, emb) in mm {
            x[i * h..(i + 1) * h].copy_from_slice(emb);
        }
        drop(timer);
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
        // The streams banked so far (this pass's tokens only: a token mixes its own).
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut mixed = vec![0f32; t * h];
        let mut hn = vec![0f32; t * h];
        let mut o = vec![0f32; t * h];
        for (l, layer) in self.layers.iter().enumerate() {
            capture(l, &x);
            let _unit = crate::stream::enter(&self.stream, l);
            // What the attention reads: the mix of the banked streams and the running sum.
            let banked = c.attn_res_block_size.is_some_and(|b| l.is_multiple_of(b));
            let input: &[f32] = match &layer.res {
                Some((w, _)) if !blocks.is_empty() => {
                    self.res_mix(&x, &blocks, w, t, &mut mixed);
                    &mixed
                }
                _ => &x,
            };
            rmsnorm_f32w_rows(input, h, &layer.ln_in, eps, &mut hn);
            match (&layer.attn, &mut sess.layers[l]) {
                (Attn::Kda(k), LayerState::Kda(st)) => {
                    let mut kr = rec.as_ref().map(|_| KdaRecord::default());
                    kda_forward(&self.kda_dims(), k, st, kr.as_mut(), &hn, t, &mut o);
                    if let (Some(r), Some(kr)) = (&mut rec, kr) {
                        r.kda.push((l, kr));
                    }
                }
                (Attn::Mla(a), LayerState::Mla(cache)) => {
                    mla_forward(&self.mla_dims(), c.dsa_config.as_ref(), a, cache, &hn, t, start, &mut o)
                }
                _ => unreachable!("layer state of another kind"),
            }
            if banked {
                // The stream entering the layer is banked; the sum restarts from the attention.
                blocks.push(std::mem::replace(&mut x, o.clone()));
            } else {
                add_inplace(&mut x, &o);
            }
            if let Some(iv) = &iv {
                iv.point(2 * l + 1, &mut x, h, start);
            }
            let input: &[f32] = match &layer.res {
                Some((_, w)) => {
                    self.res_mix(&x, &blocks, w, t, &mut mixed);
                    &mixed
                }
                None => &x,
            };
            rmsnorm_f32w_rows(input, h, &layer.ln_post, eps, &mut hn);
            let need = |ids: &[usize]| {
                if let Some(s) = &self.stream {
                    s.need_parts(l, ids)
                }
            };
            let need: NeedExperts = self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize]));
            let hook = iv.as_ref().and_then(|s| s.hook(l, start));
            match &layer.ffn {
                Ffn::Dense(m) => m.forward(&hn, t, h, &mut o, hook.as_ref().map(|k| (k, Target::Mlp))),
                Ffn::Moe(m) => self.moe_forward(m, &hn, t, &mut o, need, hook.as_ref()),
            }
            add_inplace(&mut x, &o);
            if let Some(iv) = &iv {
                iv.point(2 * l + 2, &mut x, h, start);
            }
        }
        capture(self.layers.len(), &x);
        if let Some(w) = &self.out_res {
            self.res_mix(&x, &blocks, w, t, &mut mixed);
            x.copy_from_slice(&mixed);
        }
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.final_norm, eps);
        sess.pos += t;
        sess.rec = rec.map(Box::new);
        x
    }

    /// A MoE layer over `x` (`[t, H]`) into `y`: the routed experts in their latent space, the
    /// shared experts on the full width.
    fn moe_forward(&self, m: &Moe, x: &[f32], t: usize, y: &mut [f32], need: NeedExperts, hook: Option<&NeuronHook>) {
        let c = &self.cfg;
        let h = c.hidden_size;
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
        y.fill(0.0);
        match &m.latent {
            Some(lt) => {
                let mut xl = vec![0f32; t * lt.hl];
                linear(&lt.down, lt.hl, h, x, t, &mut xl);
                let mut yl = vec![0f32; t * lt.hl];
                m.experts.forward(&xl, t, lt.hl, &routes, &mut yl, need, hook, None);
                if let Some(norm) = &lt.norm {
                    rmsnorm_f32w_rows_inplace(&mut yl, lt.hl, norm, c.rms_norm_eps as f32);
                }
                linear(&lt.up, h, lt.hl, &yl, t, y);
            }
            None => m.experts.forward(x, t, h, &routes, y, need, hook, None),
        }
        if let Some(s) = &m.shared {
            let mut ys = vec![0f32; t * h];
            s.forward(x, t, h, &mut ys, hook.map(|k| (k, Target::Shared)));
            add_inplace(y, &ys);
        }
    }
}

/// A KDA layer over `hn` (`[t, H]`, the normed input) into `y`, advancing `st`. `rec`: keep
/// what a rollback inside this pass needs.
pub(crate) fn kda_forward(dm: &KdaDims, kd: &Kda, st: &mut KdaState, rec: Option<&mut KdaRecord>, hn: &[f32], t: usize, y: &mut [f32]) {
    let (h, nh, hd, rank) = (dm.h, dm.nh, dm.hd, dm.rank);
    let d = nh * hd;
    let mut raw = [vec![0f32; t * d], vec![0f32; t * d], vec![0f32; t * d]];
    let mut fa = vec![0f32; t * rank];
    let mut beta = vec![0f32; t * nh];
    let mut write = vec![0f32; if kd.w.is_some() { t * nh } else { 0 }];
    let mut gate = vec![0f32; t * d];
    {
        let [rq, rk, rv] = &mut raw;
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![
            (&kd.q, d, &mut rq[..]),
            (&kd.k, d, &mut rk[..]),
            (&kd.v, d, &mut rv[..]),
            (&kd.f_a, rank, &mut fa[..]),
            (&kd.b, nh, &mut beta[..]),
            (&kd.g, d, &mut gate[..]),
        ];
        if let Some(w) = &kd.w {
            jobs.push((w, nh, &mut write[..]));
        }
        linear_multi(&mut jobs, h, hn, t);
    }
    let mut f = vec![0f32; t * d];
    linear(&kd.f_b, d, rank, &fa, t, &mut f);
    if let Some(r) = rec {
        r.s0.clone_from(&st.s);
        r.conv0.clone_from(&st.conv);
        r.raw.clone_from(&raw);
        r.f.clone_from(&f);
        r.beta.clone_from(&beta);
        r.write.clone_from(&write);
    }
    let mut out = vec![0f32; t * d];
    kda_scan(dm, kd, st, &raw, &f, &beta, &write, t, Some(&mut out));
    crate::qwen::gated_norm(&mut out, &gate, &kd.o_norm, dm.eps, true, t * nh > 64);
    linear(&kd.o, h, d, &out, t, y);
}

/// The recurrent part of a KDA layer over `t` tokens, advancing `st`: the convolutions of the
/// q / k / v projections `raw`, then the gated delta rule with the decay inputs `f` and the
/// `beta` inputs (and those of the write gate, `write`, empty without one), written to `out`
/// (`[t, nh * hd]`) when given.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kda_scan(
    dm: &KdaDims,
    kd: &Kda,
    st: &mut KdaState,
    raw: &[Vec<f32>; 3],
    f: &[f32],
    beta: &[f32],
    write: &[f32],
    t: usize,
    out: Option<&mut [f32]>,
) {
    let (nh, hd, kc) = (dm.nh, dm.hd, dm.kc);
    let d = nh * hd;
    let (lower, clamp) = (dm.lower, dm.clamp_norm);
    let _timer = prof::start(Cat::Attention);
    // Causal depthwise convolutions over time (left context from the state), then SiLU; the
    // state keeps the last `kernel - 1` inputs.
    let keep = kc - 1;
    let mut conv = [vec![0f32; t * d], vec![0f32; t * d], vec![0f32; t * d]];
    for (i, (out, x)) in conv.iter_mut().zip(raw).enumerate() {
        let w = &kd.conv[i];
        let past = &st.conv[i * keep * d..(i + 1) * keep * d];
        let src = |ti: isize, ch: usize| -> f32 {
            if ti >= 0 {
                x[ti as usize * d + ch]
            } else {
                past[(keep as isize + ti) as usize * d + ch]
            }
        };
        for_chunks(out, d, t > 1, |ti, row| {
            for (ch, o) in row.iter_mut().enumerate() {
                let mut s = 0f32;
                for j in 0..kc {
                    s += w[ch * kc + j] * src(ti as isize - keep as isize + j as isize, ch);
                }
                *o = s * sigmoid(s);
            }
        });
        let mut next = vec![0f32; keep * d];
        for slot in 0..keep {
            let ti = t as isize - keep as isize + slot as isize;
            let dst = &mut next[slot * d..(slot + 1) * d];
            if ti >= 0 {
                dst.copy_from_slice(&x[ti as usize * d..(ti as usize + 1) * d]);
            } else {
                let old = (keep as isize + ti) as usize;
                dst.copy_from_slice(&past[old * d..(old + 1) * d]);
            }
        }
        st.conv[i * keep * d..(i + 1) * keep * d].copy_from_slice(&next);
    }
    // The gated delta rule, recurrent over time, parallel over heads.
    let [cq, ck, cv] = &conv;
    let (op, sp) = (out.map(|o| SendPtr(o.as_mut_ptr())), SendPtr(st.s.as_mut_ptr()));
    let qscale = (hd as f32).powf(-0.5);
    crate::pool::run(nh, &|hh| {
        // SAFETY: each head owns its state slab and its output columns.
        let s = unsafe { std::slice::from_raw_parts_mut(sp.get().add(hh * hd * hd), hd * hd) };
        let (mut qn, mut kn) = (vec![0f32; hd], vec![0f32; hd]);
        let (mut mem, mut delta, mut decay) = (vec![0f32; hd], vec![0f32; hd], vec![0f32; hd]);
        for ti in 0..t {
            let at = ti * d + hh * hd;
            let (qv, kv, vv) = (&cq[at..at + hd], &ck[at..at + hd], &cv[at..at + hd]);
            let inv = |v: &[f32]| {
                if clamp {
                    dot(v, v).sqrt().max(1e-6).recip()
                } else {
                    (dot(v, v) + 1e-6).sqrt().recip()
                }
            };
            let (qi, ki) = (inv(qv) * qscale, inv(kv));
            for j in 0..hd {
                qn[j] = qv[j] * qi;
                kn[j] = kv[j] * ki;
                decay[j] = (lower * sigmoid(kd.exp_a[hh] * (f[at + j] + kd.dt_bias[hh * hd + j]))).exp();
            }
            let b = sigmoid(beta[ti * nh + hh]);
            // The value's share: written with `sigmoid(b + w)` where the state is erased with
            // `sigmoid(b)`.
            let wr = if write.is_empty() {
                1.0
            } else {
                sigmoid(beta[ti * nh + hh] + write[ti * nh + hh]) / b
            };
            // S *= exp(g) (per key row); mem = k^T S.
            mem.fill(0.0);
            for dk in 0..hd {
                let row = &mut s[dk * hd..(dk + 1) * hd];
                let (dc, kd_) = (decay[dk], kn[dk]);
                for (m, r) in mem.iter_mut().zip(row.iter_mut()) {
                    *r *= dc;
                    *m += *r * kd_;
                }
            }
            for j in 0..hd {
                delta[j] = (vv[j] * wr - mem[j]) * b;
            }
            // S += k (x) delta; o = q^T S.
            match op {
                Some(op) => {
                    // SAFETY: this (token, head) output slice is written by this head only.
                    let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(at), hd) };
                    o.fill(0.0);
                    for dk in 0..hd {
                        let row = &mut s[dk * hd..(dk + 1) * hd];
                        let (kd_, qd) = (kn[dk], qn[dk]);
                        for j in 0..hd {
                            row[j] += kd_ * delta[j];
                            o[j] += row[j] * qd;
                        }
                    }
                }
                None => {
                    for dk in 0..hd {
                        let row = &mut s[dk * hd..(dk + 1) * hd];
                        let kd_ = kn[dk];
                        for j in 0..hd {
                            row[j] += kd_ * delta[j];
                        }
                    }
                }
            }
        }
    });
}

/// An MLA layer over `hn` (`[t, H]`, the normed input) into `out`, appending to `cache`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_forward(
    dm: &MlaDims,
    dsa: Option<&DsaConfig>,
    a: &Mla,
    cache: &mut MlaCache,
    hn: &[f32],
    t: usize,
    start: usize,
    out: &mut [f32],
) {
    let (h, nh, lat, vd) = (dm.h, dm.nh, dm.lat, dm.vd);
    let (dr, ql, qk) = (dm.dr, dm.ql, dm.dn + dm.dr);
    let eps = dm.eps;
    let (ni, d, topk) = match (&a.indexer, dsa) {
        (Some(_), Some(dc)) => (dc.index_num_heads, dc.index_head_dim, dc.index_topk),
        _ => (0, 0, usize::MAX),
    };
    let mut qa = vec![0f32; t * ql];
    let mut kv = vec![0f32; t * (lat + dr)];
    let mut gate = vec![0f32; if a.gate.is_some() { t * nh * vd } else { 0 }];
    let mut ik = vec![0f32; t * d];
    let mut iw = vec![0f32; t * ni];
    {
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.q_a, ql, &mut qa[..]), (&a.kv_a, lat + dr, &mut kv[..])];
        if let Some(g) = &a.gate {
            jobs.push((g, nh * vd, &mut gate[..]));
        }
        if let Some(ix) = &a.indexer {
            jobs.push((&ix.wk, d, &mut ik[..]));
            jobs.push((&ix.proj, ni, &mut iw[..]));
        }
        linear_multi(&mut jobs, h, hn, t);
    }
    rmsnorm_f32w_rows_inplace(&mut qa, ql, &a.q_norm, eps);
    let mut q = vec![0f32; t * nh * qk];
    linear(&a.q_b, nh * qk, ql, &qa, t, &mut q);
    // The cache: normed latent and the shared key lanes per token.
    for row in kv.chunks_mut(lat + dr) {
        let (cl, kr) = row.split_at_mut(lat);
        rmsnorm_f32w_inplace(cl, &a.kv_norm, eps);
        cache.c.extend_from_slice(cl);
        cache.r.extend_from_slice(kr);
    }
    cache.len += t;
    debug_assert_eq!(cache.len, start + t);
    // The indexer: its keys join the cache; a query that sees more than `index_topk` keys
    // attends to its best ones only.
    let sel: Selection = match (&a.indexer, dsa) {
        (Some(ix), Some(dc)) => {
            layer_norm_rows(&mut ik, d, &ix.k_norm.0, &ix.k_norm.1, dc.index_norm_eps as f32);
            cache.idx.extend_from_slice(&ik);
            if start + t > topk {
                let mut iq = vec![0f32; t * ni * d];
                linear(&ix.wq, ni * d, ql, &qa, t, &mut iq);
                let ws = ((d * ni) as f32).powf(-0.5);
                iw.iter_mut().for_each(|w| *w *= ws);
                select(&iq, &iw, &cache.idx, ni, d, topk, t, start)
            } else {
                vec![None; t]
            }
        }
        _ => vec![None; t],
    };
    let mut att = vec![0f32; t * nh * vd];
    attend(dm, a, &q, cache, t, start, &sel, &mut att);
    if a.gate.is_some() {
        for (v, g) in att.iter_mut().zip(&gate) {
            *v *= sigmoid(*g);
        }
    }
    linear(&a.o, h, nh * vd, &att, t, out);
}

/// Attention scored against the latent cache (`kv_b` absorbed): each head's first `dn` query
/// lanes are mapped onto the latent (`k_b`) and scored, with its last `dr` lanes against the
/// shared key lanes, at every chosen position; the attended latent is mapped onto the value
/// (`v_b`). `out` `[t, nh * v]`.
#[allow(clippy::too_many_arguments)]
fn attend(dm: &MlaDims, a: &Mla, q: &[f32], cache: &MlaCache, t: usize, start: usize, sel: &Selection, out: &mut [f32]) {
    let (nh, lat, vd) = (dm.nh, dm.lat, dm.vd);
    let (dn, dr, qk) = (dm.dn, dm.dr, dm.dn + dm.dr);
    let mut qn = vec![0f32; nh * t * dn];
    for ti in 0..t {
        for hh in 0..nh {
            qn[(hh * t + ti) * dn..(hh * t + ti + 1) * dn].copy_from_slice(&q[(ti * nh + hh) * qk..(ti * nh + hh) * qk + dn]);
        }
    }
    let ql = per_head(&a.k_b, &qn, t, dn, lat);
    let _timer = prof::start(Cat::Attention);
    let scale = (qk as f32).powf(-0.5);
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
                    if dr > 0 {
                        crate::attn::dots(&qrs, &cache.r[j * dr..(j + 1) * dr], &mut s2[..hb]);
                    }
                    for b in 0..hb {
                        sc[(h0 + b) * len + jj] = (s1[b] + if dr > 0 { s2[b] } else { 0.0 }) * scale;
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

/// Per query of the pass, the keys the indexer ranks in its top `topk` (None when the query has
/// no more keys than that; equal scores go to the earlier key): `iq` `[t, ni, d]`, `iw` `[t, ni]`
/// (head weights, scaled), `keys` `[start + t, d]`.
#[allow(clippy::too_many_arguments)]
fn select(iq: &[f32], iw: &[f32], keys: &[f32], ni: usize, d: usize, topk: usize, t: usize, start: usize) -> Selection {
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
            // SAFETY: each task writes its own slot.
            unsafe { *op.get().add(r) = top_positions(&s, topk) };
        });
        for (&i, pk) in blk.iter().zip(picks) {
            sel[i] = Some(pk);
        }
    }
    sel
}

impl KimiSession {
    /// Whether the session can go back to `len` positions: the current length, or a position
    /// inside the last recorded pass.
    pub fn can_truncate(&self, len: usize) -> bool {
        len >= self.pos || self.rec.as_ref().is_some_and(|r| len >= r.pos)
    }
}

impl KimiModel<'_> {
    /// Drops the positions at or beyond `len`: the MLA caches are cut, and the recurrent states
    /// go back to those before the last recorded pass, whose kept tokens are replayed (their
    /// convolutions and delta rule only). Panics when `len` is before that pass (see
    /// [`KimiSession::can_truncate`]).
    pub fn truncate(&self, sess: &mut KimiSession, len: usize) {
        if len >= sess.pos {
            return;
        }
        let mut rec = match sess.rec.take() {
            Some(r) if len >= r.pos => r,
            _ => panic!("Kimi: the recurrent (KDA) state cannot go back before the last recorded pass"),
        };
        let c = &self.cfg;
        let la = &c.linear_attn_config;
        let (nh, d) = (la.num_heads, la.num_heads * la.head_dim);
        let n = len - rec.pos;
        for (l, kr) in &mut rec.kda {
            let (Attn::Kda(kd), LayerState::Kda(st)) = (&self.layers[*l].attn, &mut sess.layers[*l]) else {
                unreachable!("a KDA record of another layer kind")
            };
            st.s.copy_from_slice(&kr.s0);
            st.conv.copy_from_slice(&kr.conv0);
            kr.raw.iter_mut().for_each(|r| r.truncate(n * d));
            kr.f.truncate(n * d);
            kr.beta.truncate(n * nh);
            if n > 0 {
                kda_scan(&self.kda_dims(), kd, st, &kr.raw, &kr.f, &kr.beta, &kr.write, n, None);
            }
        }
        let di = c.dsa_config.as_ref().map_or(0, |dc| dc.index_head_dim);
        for st in &mut sess.layers {
            if let LayerState::Mla(cache) = st {
                cache.c.truncate(len * c.kv_lora_rank);
                cache.r.truncate(len * c.qk_rope_head_dim);
                cache.idx.truncate(len * di);
                cache.len = cache.len.min(len);
            }
        }
        sess.pos = len;
        // (Still good for a rollback further into the pass.)
        sess.rec = Some(rec);
    }
}

impl crate::generate::TextLm for KimiModel<'_> {
    type Sess = KimiSession;
    fn new_session(&self) -> KimiSession {
        KimiModel::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut KimiSession) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &KimiSession) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut KimiSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        KimiModel::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        KimiModel::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut KimiSession, tokens: &[u32]) -> Vec<f32> {
        KimiModel::forward(self, sess, tokens)
    }
}

/// The interface speculative decoding rounds use ([`crate::dspark::Target`]: prompt lookup; no
/// DSpark draft exists for this model). Its passes are recorded, so they can be rolled back.
impl crate::dspark::Target for KimiModel<'_> {
    fn forward_feats(
        &self,
        sess: &mut KimiSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, layers, feats, true)
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        if let Some(s) = &self.stream {
            s.need_ranges(&self.embed.row_ranges(token as usize, h));
        }
        self.embed.row(token as usize, out);
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        KimiModel::logits(self, hidden, t)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut KimiSession, len: usize) {
        KimiModel::truncate(self, sess, len);
    }
}

/// Runtime interventions (`crate::intervene`): the points around every sublayer act on the
/// running sum (which restarts from the attention output on the layers that bank it), and as
/// neurons the input of `down_proj` of the dense MLP (`Target::Mlp`) and of a MoE layer's shared
/// experts (`Target::Shared`). The routed experts write into a latent space, not the stream:
/// they are not offered.
impl crate::intervene::Host for KimiModel<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        match (&self.layers.get(layer)?.ffn, target) {
            (Ffn::Dense(m), Target::Mlp) => Some(m.inter),
            (Ffn::Moe(m), Target::Shared) => m.shared.as_ref().map(|s| s.inter),
            (Ffn::Moe(m), Target::Expert(e)) if m.latent.is_none() && e < m.experts.n_experts() => Some(m.experts.inter),
            _ => None,
        }
    }
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let (down, inter) = match (&self.layers[layer].ffn, target) {
            (Ffn::Dense(m), _) => (&m.down, m.inter),
            (Ffn::Moe(m), Target::Expert(e)) if m.latent.is_none() => (&m.experts.down[e], m.experts.inter),
            (Ffn::Moe(m), _) => match &m.shared {
                Some(s) => (&s.down, s.inter),
                None => return vec![0.0; h],
            },
        };
        let mut one = vec![0f32; inter];
        one[n] = 1.0;
        let mut y = vec![0f32; h];
        linear(down, h, inter, &one, 1, &mut y);
        y
    }
}

/// The name (in Moonshot's layout, without `language_model.`) and PyTorch-order shape of a
/// `kimi-k3` / `kimi-k31` GGUF tensor (`ne`: ggml's shape), as llama.cpp's converter writes them.
/// What it fuses or splits keeps a name of its own: `..._res_score` (norm times projection of a
/// cross-layer mix), `self_attn.neg_exp_a` (`-exp(A_log)` per head), `self_attn.k_b` / `v_b`
/// (`kv_b_proj` per head), `experts.{gate,up,down}_proj` (the experts stacked).
pub(crate) fn gguf_name(name: &str, ne: &[usize]) -> Option<(String, Vec<usize>)> {
    let shape: Vec<usize> = ne.iter().rev().copied().collect();
    let top = match name {
        "token_embd.weight" => Some("model.embed_tokens.weight"),
        "output.weight" => Some("lm_head.weight"),
        "output_norm.weight" => Some("model.norm.weight"),
        "output_res_score.weight" => Some("model.output_attn_res_score"),
        _ => None,
    };
    if let Some(t) = top {
        return Some((t.to_string(), shape));
    }
    let (l, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    l.parse::<usize>().ok()?;
    let n = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_res_score.weight" => "self_attention_res_score",
        "ffn_res_score.weight" => "mlp_res_score",
        // KDA.
        "ssm_a" => "self_attn.neg_exp_a",
        "ssm_beta.weight" => "self_attn.b_proj.weight",
        "ssm_dt.bias" => "self_attn.dt_bias",
        "ssm_f_a.weight" => "self_attn.f_a_proj.weight",
        "ssm_f_b.weight" => "self_attn.f_b_proj.weight",
        "ssm_g.weight" => "self_attn.g_proj.weight",
        "ssm_conv1d_q.weight" => "self_attn.q_conv1d.weight",
        "ssm_conv1d_k.weight" => "self_attn.k_conv1d.weight",
        "ssm_conv1d_v.weight" => "self_attn.v_conv1d.weight",
        "ssm_norm.weight" => "self_attn.o_norm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        // MLA and its indexer.
        "attn_q_a.weight" => "self_attn.q_a_proj.weight",
        "attn_q_a_norm.weight" => "self_attn.q_a_layernorm.weight",
        "attn_q_b.weight" => "self_attn.q_b_proj.weight",
        "attn_kv_a_mqa.weight" => "self_attn.kv_a_proj_with_mqa.weight",
        "attn_kv_a_norm.weight" => "self_attn.kv_a_layernorm.weight",
        "attn_k_b.weight" => "self_attn.k_b",
        "attn_v_b.weight" => "self_attn.v_b",
        "attn_gate.weight" => "self_attn.g_proj.weight",
        "indexer.attn_q_b.weight" => "self_attn.indexer.wq_b.weight",
        "indexer.attn_k.weight" => "self_attn.indexer.wk.weight",
        "indexer.k_norm.weight" => "self_attn.indexer.k_norm.weight",
        "indexer.k_norm.bias" => "self_attn.indexer.k_norm.bias",
        "indexer.proj.weight" => "self_attn.indexer.weights_proj.weight",
        // MLPs.
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "ffn_gate_inp.weight" => "block_sparse_moe.gate.weight",
        "exp_probs_b.bias" => "block_sparse_moe.gate.e_score_correction_bias",
        "ffn_gate_exps.weight" => "block_sparse_moe.experts.gate_proj",
        "ffn_up_exps.weight" => "block_sparse_moe.experts.up_proj",
        "ffn_down_exps.weight" => "block_sparse_moe.experts.down_proj",
        "ffn_routed_down.weight" => "block_sparse_moe.routed_expert_down_proj.weight",
        "ffn_routed_up.weight" => "block_sparse_moe.routed_expert_up_proj.weight",
        "ffn_routed_norm.weight" => "block_sparse_moe.routed_expert_norm.weight",
        "ffn_gate_shexp.weight" => "block_sparse_moe.shared_experts.gate_proj.weight",
        "ffn_up_shexp.weight" => "block_sparse_moe.shared_experts.up_proj.weight",
        "ffn_down_shexp.weight" => "block_sparse_moe.shared_experts.down_proj.weight",
        _ => return None,
    };
    Some((format!("model.layers.{l}.{n}"), shape))
}

/// The `config.json` equivalent of a `kimi-k3` / `kimi-k31` GGUF header (llama.cpp's metadata
/// keys). The file is Kimi K3.1 when it has indexer keys; the indexer layers are those of
/// `attention.indexer.types`, or every full-attention layer.
pub(crate) fn gguf_config(hd: &crate::gguf::Header) -> Result<serde_json::Value> {
    let arch = hd.arch().to_string();
    let get = |k: &str| hd.get(&format!("{arch}.{k}"));
    let need = |k: &str| {
        get(k)
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .with_context(|| format!("GGUF metadata has no {arch}.{k}"))
    };
    let int = |k: &str| get(k).and_then(Value::as_u64).map(|v| v as usize);
    let float = |k: &str| get(k).and_then(Value::as_f64);
    let n = hd.n_layers();
    // 2 = sigmoid (llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID).
    if let Some(g) = int("expert_gating_func")
        && g != 2
    {
        bail!("GGUF {arch} with expert gating function {g} is not supported (sigmoid only)");
    }
    let kv = get("attention.head_count_kv")
        .and_then(Value::as_array)
        .with_context(|| format!("GGUF metadata has no {arch}.attention.head_count_kv array"))?;
    if kv.len() != n {
        bail!("GGUF {arch}: {} head_count_kv entries for {n} layers", kv.len());
    }
    // A layer without key / value heads is a KDA layer.
    let full: Vec<usize> = (0..n).filter(|&l| kv[l].as_u64().unwrap_or(0) > 0).map(|l| l + 1).collect();
    let kda: Vec<usize> = (1..=n).filter(|l| !full.contains(l)).collect();
    let heads = need("attention.head_count")?;
    let rot = need("rope.dimension_count")?;
    let mut c = serde_json::json!({
        "model_type": "kimi_linear",
        "vocab_size": need("vocab_size")?,
        "hidden_size": need("embedding_length")?,
        "intermediate_size": int("feed_forward_length").unwrap_or(0),
        "num_hidden_layers": n,
        "num_attention_heads": heads,
        "hidden_act": "situ",
        "activation_situ_beta": float("activation.situ_beta").context("GGUF metadata has no activation.situ_beta")?,
        "activation_situ_linear_beta": float("activation.situ_linear_beta"),
        "rms_norm_eps": float("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
        "attn_res_block_size": int("attn_res.block_size").filter(|&b| b > 0),
        "first_k_dense_replace": int("leading_dense_block_count").unwrap_or(0),
        "num_experts": int("expert_count"),
        "num_experts_per_token": int("expert_used_count"),
        "num_shared_experts": int("expert_shared_count"),
        "moe_intermediate_size": int("expert_feed_forward_length"),
        "routed_expert_hidden_size": int("expert_latent_length").filter(|&v| v > 0),
        // (The loader reads the norm when the file has it.)
        "latent_moe_use_norm": true,
        "moe_renormalize": !matches!(get("expert_weights_norm"), Some(Value::Bool(false))),
        "routed_scaling_factor": float("expert_weights_scale").unwrap_or(1.0),
        "q_lora_rank": int("attention.q_lora_rank"),
        "kv_lora_rank": need("attention.kv_lora_rank")?,
        "qk_rope_head_dim": rot,
        "qk_nope_head_dim": need("attention.key_length_mla")? - rot,
        "v_head_dim": need("attention.value_length_mla")?,
        "mla_use_nope": true,
        "linear_attn_config": {
            "full_attn_layers": full, "kda_layers": kda,
            "head_dim": need("kda.head_dim")?, "num_heads": heads,
            "short_conv_kernel_size": need("ssm.conv_kernel")?,
            "gate_lower_bound": float("kda.gate_lower_bound"),
            "use_full_rank_gate": true,
        },
    });
    if let Some(topk) = int("attention.indexer.top_k").filter(|&k| k > 0) {
        let layers: Vec<usize> = match get("attention.indexer.types").and_then(Value::as_array) {
            Some(v) => (0..n.min(v.len()))
                .filter(|&l| matches!(v[l], Value::Bool(true)) && full.contains(&(l + 1)))
                .map(|l| l + 1)
                .collect(),
            None => full.clone(),
        };
        c["dsa_config"] = serde_json::json!({
            "dsa_layers": layers,
            "index_topk": topk,
            "index_head_dim": need("attention.indexer.key_length")?,
            "index_num_heads": need("attention.indexer.head_count")?,
            "index_rope_dim": int("attention.indexer.rope_dimension_count").unwrap_or(0),
            "index_topk_freq": int("attention.indexer.top_k_freq").unwrap_or(1),
            // llama.cpp's default when the file names none.
            "index_norm_eps": float("attention.layer_norm_epsilon").unwrap_or(1e-5),
        });
    }
    Ok(c)
}

/// The pre-tokenizer pattern of Kimi's `tokenization_kimi.py` (`pat_str`): runs of Han
/// characters apart, words with an optional contraction, digits in groups of up to three.
pub const PRETOKENIZER: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// GPT-2's byte to character map (`bytes_to_unicode`): printable Latin-1 bytes as themselves,
/// the others from U+0100 up.
pub(crate) fn byte_chars() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for (b, slot) in out.iter_mut().enumerate() {
        let keep = matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
        *slot = char::from_u32(if keep { b as u32 } else { 256 + n }).unwrap();
        n += u32::from(!keep);
    }
    out
}

/// A `tokenizer.json` equivalent to a tiktoken vocabulary: byte-level BPE over `normal` (token
/// strings in GPT-2's byte characters, with their ids, which are the tiktoken ranks) whose merges
/// are every split of a token into two tokens, ranked by the id of the token they make
/// (`rank_by_id`). That is tiktoken's rule: of all adjacent pairs that form a token, merge the
/// one forming the lowest-ranked token, the leftmost on ties. `added`: the special tokens
/// (content, id, special).
pub(crate) fn bpe_json(normal: &[(String, u32)], added: &[(String, u32, bool)]) -> serde_json::Value {
    use serde_json::json;
    let ids: std::collections::HashSet<&str> = normal.iter().map(|t| t.0.as_str()).collect();
    let mut merges = Vec::new();
    for (s, _) in normal {
        for (i, _) in s.char_indices().skip(1) {
            let (a, b) = s.split_at(i);
            if ids.contains(a) && ids.contains(b) {
                merges.push(json!([a, b]));
            }
        }
    }
    // (The special tokens are in the model's vocabulary too: an added token that is not would be
    // renumbered after it, which moves it when the ids have a gap.)
    let vocab: serde_json::Map<String, serde_json::Value> = normal
        .iter()
        .map(|(s, id)| (s.clone(), json!(id)))
        .chain(added.iter().map(|(s, id, _)| (s.clone(), json!(id))))
        .collect();
    let added: Vec<serde_json::Value> = added
        .iter()
        .map(|(s, id, special)| {
            json!({"id": id, "content": s, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": special})
        })
        .collect();
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": PRETOKENIZER}, "behavior": "Isolated", "invert": false},
            byte_level.clone(),
        ]},
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "rank_by_id": true, "vocab": vocab, "merges": merges,
        },
    })
}

/// The tokenizer of a Kimi checkpoint folder, which has no `tokenizer.json`: `tiktoken.model`
/// (a base64 token and its rank per line) and the special tokens after it, named by
/// `tokenizer_config.json` (`added_tokens_decoder`) or `<|reserved_token_N|>`, as
/// `tokenization_kimi.py` builds them.
pub fn tiktoken_tokenizer(dir: &Path) -> Result<crate::tokenizer::Tokenizer> {
    const RESERVED: usize = 256;
    let text = String::from_utf8(crate::fsx::read(&dir.join("tiktoken.model"))?).context("tiktoken.model is not text")?;
    let chars = byte_chars();
    let mut normal = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (tok, rank) = line.split_once(' ').with_context(|| format!("tiktoken.model: bad line {line:?}"))?;
        let bytes = crate::b64::decode(tok).with_context(|| format!("tiktoken.model: bad base64 {tok:?}"))?;
        let rank: u32 = rank
            .trim()
            .parse()
            .with_context(|| format!("tiktoken.model: bad rank in {line:?}"))?;
        normal.push((bytes.iter().map(|&b| chars[b as usize]).collect::<String>(), rank));
    }
    let n = normal.len();
    let names: serde_json::Value = crate::fsx::read(&dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let added: Vec<(String, u32, bool)> = (n..n + RESERVED)
        .map(|id| {
            let e = &names["added_tokens_decoder"][id.to_string()];
            match e["content"].as_str() {
                Some(c) => (c.to_string(), id as u32, e["special"].as_bool().unwrap_or(true)),
                None => (format!("<|reserved_token_{id}|>"), id as u32, true),
            }
        })
        .collect();
    crate::tokenizer::from_value(&bpe_json(&normal, &added)).map_err(|e| anyhow::anyhow!("tiktoken.model: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let n = |name: &str, ne: &[usize]| gguf_name(name, ne);
        assert_eq!(
            n("blk.59.attn_k_b.weight", &[128, 512, 96]),
            Some(("model.layers.59.self_attn.k_b".into(), vec![96, 512, 128]))
        );
        assert_eq!(
            n("blk.1.ffn_down_exps.weight", &[3072, 3584, 896]),
            Some(("model.layers.1.block_sparse_moe.experts.down_proj".into(), vec![896, 3584, 3072]))
        );
        assert_eq!(
            n("blk.0.ssm_conv1d_q.weight", &[4, 1, 12288, 1]).map(|x| x.0).as_deref(),
            Some("model.layers.0.self_attn.q_conv1d.weight")
        );
        // The two gates share Moonshot's name; the layer kind tells them apart.
        assert_eq!(n("blk.0.ssm_g.weight", &[8, 4]), n("blk.0.attn_gate.weight", &[8, 4]));
        assert_eq!(
            n("output_res_score.weight", &[7168]),
            Some(("model.output_attn_res_score".into(), vec![7168]))
        );
        assert_eq!(n("blk.3.unknown.weight", &[1]), None);
    }

    /// The released Kimi-K3.1-Pro config (its text keys, with the ones this loader ignores).
    #[test]
    fn released_config() {
        let full: Vec<usize> = (1..=93).filter(|l| l % 4 == 0 || *l == 93).collect();
        let kda: Vec<usize> = (1..=93).filter(|l| !full.contains(l)).collect();
        // (In two halves: one `json!` of this many keys passes the macro recursion limit.)
        let mut text = serde_json::json!({
            "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "attn_res_block_size": 12,
            "bad_words_ids": null, "bos_token_id": 163584, "eos_token_id": 163586,
            "dsa_config": {
                "dsa_layers": full, "index_topk": 2048, "index_head_dim": 128, "index_num_heads": 32,
                "index_rope_dim": 0, "index_k_norm_type": "layernorm", "index_topk_freq": 1,
                "index_skip_topk_offset": 1,
            },
            "first_k_dense_replace": 1, "hidden_act": "situ", "hidden_size": 7168, "intermediate_size": 33792,
            "kv_lora_rank": 512, "latent_moe_use_norm": true,
            "linear_attn_config": {
                "full_attn_layers": full, "gate_lower_bound": -5.0, "head_dim": 128, "kda_layers": kda,
                "num_heads": 96, "short_conv_kernel_size": 4, "use_full_rank_gate": true,
            },
            "max_position_embeddings": 1048576, "mla_use_nope": true, "mla_use_output_gate": true,
        });
        let more = serde_json::json!({
            "model_type": "kimi_linear", "moe_intermediate_size": 3072, "moe_layer_freq": 1,
            "moe_renormalize": true, "moe_router_activation_func": "sigmoid", "num_attention_heads": 96,
            "num_expert_group": 1, "num_experts": 896, "num_experts_per_token": 16, "num_hidden_layers": 93,
            "num_key_value_heads": 96, "num_nextn_predict_layers": 0, "num_shared_experts": 2,
            "q_lora_rank": 1536, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "quantization_config": {"format": "mxfp4-pack-quantized", "quant_method": "compressed-tensors"},
            "rms_norm_eps": 1e-05, "routed_expert_hidden_size": 3584, "routed_scaling_factor": 1.0,
            "tie_word_embeddings": false, "topk_group": 1, "topk_method": "noaux_tc", "use_grouped_topk": true,
            "v_head_dim": 128, "vocab_size": 163840,
        });
        for (k, v) in more.as_object().unwrap() {
            text[k] = v.clone();
        }
        let c = KimiConfig::from_json(serde_json::json!({
            "architectures": ["KimiK31ForConditionalGeneration"], "model_type": "kimi_k31",
            "media_placeholder_token_id": 163605, "text_config": text,
        }))
        .unwrap();
        assert_eq!((c.num_hidden_layers, c.hidden_size, c.num_experts), (93, 7168, Some(896)));
        assert!(c.is_kda(0) && !c.is_kda(3) && c.is_dsa(3) && c.is_dsa(92) && !c.is_dsa(0));
        assert!(!c.is_moe(0) && c.is_moe(1) && c.is_moe(92));
        assert_eq!(c.dsa_config.as_ref().unwrap().index_norm_eps, 1e-6);
        assert_eq!(
            c.act(),
            Act::Situ {
                beta: 4.0,
                linear_beta: 25.0
            }
        );
        // Kimi K3: the same without `dsa_config`; Kimi-Linear-48B's KDA is refused.
        let mut raw = serde_json::to_value(serde_json::json!({
            "vocab_size": 8, "hidden_size": 8, "intermediate_size": 8, "num_hidden_layers": 2, "num_attention_heads": 1,
            "q_lora_rank": 4, "kv_lora_rank": 4, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4, "v_head_dim": 4,
            "mla_use_nope": true,
            "linear_attn_config": {
                "full_attn_layers": [2], "kda_layers": [1], "head_dim": 4, "num_heads": 2, "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0, "use_full_rank_gate": true,
            },
        }))
        .unwrap();
        assert!(KimiConfig::from_json(raw.clone()).is_ok_and(|c| !c.is_dsa(1)));
        raw["linear_attn_config"]["use_full_rank_gate"] = false.into();
        assert!(KimiConfig::from_json(raw).is_err());
    }

    /// The config of the tiny model: KDA layers, MLA layers with an indexer (the last two layers
    /// both, as in the release), streams banked every three layers, a dense first layer, latent
    /// MoE with its norm, shared experts, SiTU. `over`: keys replaced.
    fn tiny_config(over: serde_json::Value) -> KimiConfig {
        let mut c = serde_json::json!({
            "model_type": "kimi_k31",
            "text_config": {
                "vocab_size": 48, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 7,
                "num_attention_heads": 2, "hidden_act": "situ", "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0, "rms_norm_eps": 1e-5, "attn_res_block_size": 3,
                "first_k_dense_replace": 1, "num_experts": 4, "num_experts_per_token": 2, "num_shared_experts": 2,
                "moe_intermediate_size": 16, "routed_expert_hidden_size": 16, "latent_moe_use_norm": true,
                "q_lora_rank": 12, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8,
                "mla_use_nope": true, "mla_use_output_gate": true,
                "linear_attn_config": {
                    "full_attn_layers": [3, 6, 7], "kda_layers": [1, 2, 4, 5], "head_dim": 8, "num_heads": 3,
                    "short_conv_kernel_size": 4, "gate_lower_bound": -5.0, "use_full_rank_gate": true,
                },
                "dsa_config": {
                    "dsa_layers": [3, 6, 7], "index_topk": 5, "index_head_dim": 8, "index_num_heads": 8,
                    "index_rope_dim": 0, "index_k_norm_type": "layernorm", "index_topk_freq": 1,
                },
            }
        });
        for (k, v) in over.as_object().unwrap() {
            c["text_config"][k] = v.clone();
        }
        KimiConfig::from_json(c).unwrap()
    }

    /// Writes a tiny random checkpoint of [`tiny_config`] in Moonshot's naming (float32).
    fn tiny_checkpoint(dir: &std::path::Path) {
        let c = tiny_config(serde_json::json!({}));
        let la = &c.linear_attn_config;
        let (h, nh, v) = (c.hidden_size, c.num_attention_heads, c.vocab_size);
        let (kh, hd, kc) = (la.num_heads, la.head_dim, la.short_conv_kernel_size);
        let d = kh * hd;
        let (ql, lat, dn, dr, vd) = (
            c.q_lora_rank.unwrap(),
            c.kv_lora_rank,
            c.qk_nope_head_dim,
            c.qk_rope_head_dim,
            c.v_head_dim,
        );
        let (ne, mi, hl) = (
            c.num_experts.unwrap(),
            c.moe_intermediate_size.unwrap(),
            c.routed_expert_hidden_size.unwrap(),
        );
        let dc = c.dsa_config.as_ref().unwrap();
        let (ni, di) = (dc.index_num_heads, dc.index_head_dim);
        let mut seed = 0x6b696d69u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        // `scale` times uniform noise, plus `bias`.
        let mut add = |name: String, shape: &[usize], scale: f32, bias: f32| {
            let n: usize = shape.iter().product();
            let bytes = (0..n).flat_map(|_| (bias + scale * rng()).to_le_bytes()).collect();
            tensors.push((format!("language_model.{name}"), shape.to_vec(), bytes));
        };
        let mat = |fan_in: usize, gain: f32| gain * (3.0 / fan_in as f32).sqrt();
        add("model.embed_tokens.weight".into(), &[v, h], 1.7, 0.0);
        add("lm_head.weight".into(), &[v, h], mat(h, 3.0), 0.0);
        add("model.norm.weight".into(), &[h], 0.2, 1.0);
        add("model.output_attn_res_norm.weight".into(), &[h], 0.2, 1.0);
        add("model.output_attn_res_proj.weight".into(), &[1, h], mat(h, 3.0), 0.0);
        for l in 0..c.num_hidden_layers {
            let p = format!("model.layers.{l}.");
            for n in [
                "input_layernorm",
                "post_attention_layernorm",
                "self_attention_res_norm",
                "mlp_res_norm",
            ] {
                add(format!("{p}{n}.weight"), &[h], 0.2, 1.0);
            }
            add(format!("{p}self_attention_res_proj.weight"), &[1, h], mat(h, 3.0), 0.0);
            add(format!("{p}mlp_res_proj.weight"), &[1, h], mat(h, 3.0), 0.0);
            let ap = format!("{p}self_attn.");
            if c.is_kda(l) {
                for n in ["q", "k", "v"] {
                    add(format!("{ap}{n}_proj.weight"), &[d, h], mat(h, 1.5), 0.0);
                    add(format!("{ap}{n}_conv1d.weight"), &[d, 1, kc], 0.7, 0.0);
                }
                // A buffer longer than the heads, as released.
                add(format!("{ap}A_log"), &[hd], 0.6, 0.0);
                add(format!("{ap}dt_bias"), &[d], 0.5, 0.0);
                add(format!("{ap}f_a_proj.weight"), &[hd, h], mat(h, 1.0), 0.0);
                add(format!("{ap}f_b_proj.weight"), &[d, hd], mat(hd, 2.0), 0.0);
                add(format!("{ap}b_proj.weight"), &[kh, h], mat(h, 1.5), 0.0);
                add(format!("{ap}g_proj.weight"), &[d, h], mat(h, 1.5), 0.0);
                add(format!("{ap}o_norm.weight"), &[hd], 0.2, 1.0);
                add(format!("{ap}o_proj.weight"), &[h, d], mat(d, 1.0), 0.0);
            } else {
                add(format!("{ap}q_a_proj.weight"), &[ql, h], mat(h, 1.0), 0.0);
                add(format!("{ap}q_a_layernorm.weight"), &[ql], 0.2, 1.0);
                add(format!("{ap}q_b_proj.weight"), &[nh * (dn + dr), ql], mat(ql, 2.0), 0.0);
                add(format!("{ap}kv_a_proj_with_mqa.weight"), &[lat + dr, h], mat(h, 1.0), 0.0);
                add(format!("{ap}kv_a_layernorm.weight"), &[lat], 0.2, 1.0);
                add(format!("{ap}kv_b_proj.weight"), &[nh * (dn + vd), lat], mat(lat, 2.0), 0.0);
                add(format!("{ap}g_proj.weight"), &[nh * vd, h], mat(h, 1.5), 0.0);
                add(format!("{ap}o_proj.weight"), &[h, nh * vd], mat(nh * vd, 1.0), 0.0);
                add(format!("{ap}indexer.wq_b.weight"), &[ni * di, ql], mat(ql, 2.0), 0.0);
                add(format!("{ap}indexer.wk.weight"), &[di, h], mat(h, 2.0), 0.0);
                add(format!("{ap}indexer.k_norm.weight"), &[di], 0.2, 1.0);
                add(format!("{ap}indexer.k_norm.bias"), &[di], 0.2, 0.0);
                add(format!("{ap}indexer.weights_proj.weight"), &[ni, h], mat(h, 1.0), 0.0);
            }
            if !c.is_moe(l) {
                add(format!("{p}mlp.gate_proj.weight"), &[c.intermediate_size, h], mat(h, 1.0), 0.0);
                add(format!("{p}mlp.up_proj.weight"), &[c.intermediate_size, h], mat(h, 1.0), 0.0);
                add(
                    format!("{p}mlp.down_proj.weight"),
                    &[h, c.intermediate_size],
                    mat(c.intermediate_size, 1.0),
                    0.0,
                );
                continue;
            }
            let mp = format!("{p}block_sparse_moe.");
            add(format!("{mp}gate.weight"), &[ne, h], mat(h, 2.0), 0.0);
            add(format!("{mp}gate.e_score_correction_bias"), &[ne], 0.3, 0.0);
            add(format!("{mp}routed_expert_down_proj.weight"), &[hl, h], mat(h, 1.0), 0.0);
            add(format!("{mp}routed_expert_up_proj.weight"), &[h, hl], mat(hl, 1.0), 0.0);
            add(format!("{mp}routed_expert_norm.weight"), &[hl], 0.2, 1.0);
            for e in 0..ne {
                add(format!("{mp}experts.{e}.w1.weight"), &[mi, hl], mat(hl, 1.0), 0.0);
                add(format!("{mp}experts.{e}.w3.weight"), &[mi, hl], mat(hl, 1.0), 0.0);
                add(format!("{mp}experts.{e}.w2.weight"), &[hl, mi], mat(mi, 1.0), 0.0);
            }
            let si = mi * c.num_shared_experts.unwrap();
            add(format!("{mp}shared_experts.gate_proj.weight"), &[si, h], mat(h, 1.0), 0.0);
            add(format!("{mp}shared_experts.up_proj.weight"), &[si, h], mat(h, 1.0), 0.0);
            add(format!("{mp}shared_experts.down_proj.weight"), &[h, si], mat(si, 1.0), 0.0);
        }
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, bytes) in tensors {
            let entry = serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]});
            head.insert(name, entry);
            data.extend(bytes);
        }
        let mut head = serde_json::Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
    }

    /// A toy tiktoken vocabulary (the bytes, `aa`, `aaa`) and Kimi's tag tokens.
    fn toy_tokenizer() -> crate::tokenizer::Tokenizer {
        let chars = byte_chars();
        let mut normal: Vec<(String, u32)> = chars.iter().enumerate().map(|(b, c)| (c.to_string(), b as u32)).collect();
        normal.push(("aa".into(), 256));
        normal.push(("aaa".into(), 257));
        let added: Vec<(String, u32, bool)> = ["<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"]
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), 300 + i as u32, i == 3))
            .collect();
        crate::tokenizer::from_value(&bpe_json(&normal, &added)).unwrap()
    }

    /// Merges follow tiktoken: the pair forming the lowest-ranked token first, the leftmost on
    /// ties, whichever way the token splits.
    #[test]
    fn tiktoken_merges() {
        let tok = toy_tokenizer();
        let ids = |s: &str| tok.encode(s, false).unwrap().get_ids().to_vec();
        let a = b'a' as u32;
        assert_eq!(ids("a"), [a]);
        assert_eq!(ids("aaa"), [257]);
        // `aa` (rank 256) forms before `aaa` (257) can: aa, aa.
        assert_eq!(ids("aaaa"), [256, 256]);
        assert_eq!(ids("aaaaa"), [256, 257]);
        assert_eq!(ids("ab<|sep|>é"), [a, b'b' as u32, 302, 0xc3, 0xa9]);
        assert_eq!(tok.decode(&ids("aaaa é"), false).unwrap(), "aaaa é");
    }

    /// A reply's tags never reach the text: the think block is reported as thinking, the
    /// response as text, and the text stops where the tool calls open.
    #[test]
    fn reply_tags_are_filtered() {
        use crate::tools::kimi::{close, open};
        let tok = toy_tokenizer();
        let reply = format!(
            "hm{}{}Yes.{}{}{}",
            close("think"),
            open("response", &[]),
            close("response"),
            open("tools", &[]),
            open("call", &[("tool", "f"), ("index", "1")]),
        );
        let ids = tok.encode(reply.as_str(), false).unwrap().get_ids().to_vec();
        let mut stream = crate::generate::TextStream::new(&crate::chat::KimiChat, true, &tok);
        assert!(stream.thinking());
        let (mut thinking, mut text) = (String::new(), String::new());
        for n in 1..=ids.len() {
            stream.on_token(ids[n - 1]);
            let d = stream.delta(&tok, &ids[..n], n == ids.len());
            if stream.thinking() { &mut thinking } else { &mut text }.push_str(&d);
        }
        assert_eq!((thinking.as_str(), text.as_str()), ("hm", "Yes."));
        assert!(stream.called() && !stream.thinking());
        // Without thinking the reply starts in its response.
        let ids = tok
            .encode(format!("No.{}", close("response")).as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        let mut stream = crate::generate::TextStream::new(&crate::chat::KimiChat, false, &tok);
        ids.iter().for_each(|&t| {
            stream.on_token(t);
        });
        assert_eq!(stream.delta(&tok, &ids, true), "No.");
        assert!(!stream.called() && !stream.thinking());
    }

    /// The routed experts as released (compressed-tensors MXFP4): the repacked matrix decodes to
    /// the values of the codes and scales, and a product against it is that of those values up
    /// to the kernels' int8 activations, for passes of any size.
    #[test]
    fn mxfp4_experts() {
        const VALUES: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut seed = 0x4d58u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for (case, (n_out, n_in)) in [(64usize, 32usize), (32, 64), (48, 128), (16, 352)].into_iter().enumerate() {
            let packed: Vec<u8> = (0..n_out * n_in / 2).map(|_| rng() as u8).collect();
            let scale: Vec<u8> = (0..n_out * n_in / 32).map(|_| 120 + (rng() % 6) as u8).collect();
            let head = serde_json::json!({
                "w.weight_packed": {"dtype": "U8", "shape": [n_out, n_in / 2], "data_offsets": [0, packed.len()]},
                "w.weight_scale": {"dtype": "U8", "shape": [n_out, n_in / 32], "data_offsets": [packed.len(), packed.len() + scale.len()]},
            });
            let mut head = head.to_string();
            while !head.len().is_multiple_of(8) {
                head.push(' ');
            }
            let mut file = (head.len() as u64).to_le_bytes().to_vec();
            file.extend_from_slice(head.as_bytes());
            file.extend(&packed);
            file.extend(&scale);
            let dir = std::env::temp_dir().join(format!("oxinfer-kimi-mxfp4-{}-{case}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), file).unwrap();
            let files = ModelFiles::open(&dir, true).unwrap();
            std::fs::remove_dir_all(&dir).ok();
            let lin = load_lin(&files, "w.weight", n_out, n_in, Quant::Bf16).unwrap();
            let want: Vec<f32> = (0..n_out * n_in)
                .map(|i| {
                    let code = (packed[i / 2] >> (4 * (i % 2))) & 0xF;
                    VALUES[code as usize] * 2f32.powi(scale[i / 32] as i32 - 127)
                })
                .collect();
            let crate::model::Weight::Gguf(m) = &lin.w else {
                panic!("not repacked into ggml blocks")
            };
            let mut got = vec![0f32; n_out * n_in];
            m.dequantize_rows(0, n_out, &mut got);
            assert_eq!(got, want, "[{n_out}, {n_in}]");
            for t in [1usize, 3, 8, 13] {
                let x: Vec<f32> = (0..t * n_in).map(|_| (rng() % 2001) as f32 / 1000.0 - 1.0).collect();
                let mut y = vec![0f32; t * n_out];
                linear(&lin, n_out, n_in, &x, t, &mut y);
                let (mut err, mut size) = (0f32, 0f32);
                for ti in 0..t {
                    for r in 0..n_out {
                        let exact = dot(&want[r * n_in..(r + 1) * n_in], &x[ti * n_in..(ti + 1) * n_in]);
                        err = err.max((y[ti * n_out + r] - exact).abs());
                        size = size.max(exact.abs());
                    }
                }
                assert!(
                    err <= 0.02 * size,
                    "[{n_out}, {n_in}] x {t} rows: off by {err} (products up to {size})"
                );
            }
        }
    }

    /// MXFP4 experts in a short pass (the decode kernels: int8 activations) and in a long one
    /// (a grouped matmul over the decoded weights) give the same output up to that rounding.
    #[test]
    fn mxfp4_experts_short_and_long_passes() {
        let (ne, mi, hl) = (4usize, 64usize, 32usize);
        let mut seed = 0x4b33u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut head = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for e in 0..ne {
            for (w, n_out, n_in) in [("w1", mi, hl), ("w3", mi, hl), ("w2", hl, mi)] {
                for (suffix, n, lo, span) in [("packed", n_out * n_in / 2, 0u32, 256u32), ("scale", n_out * n_in / 32, 122, 4)] {
                    let bytes: Vec<u8> = (0..n).map(|_| (lo + rng() % span) as u8).collect();
                    let shape = [n_out, n / n_out];
                    head.insert(
                        format!("experts.{e}.{w}.weight_{suffix}"),
                        serde_json::json!({"dtype": "U8", "shape": shape, "data_offsets": [data.len(), data.len() + n]}),
                    );
                    data.extend(bytes);
                }
            }
        }
        let mut head = serde_json::Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        let dir = std::env::temp_dir().join(format!("oxinfer-kimi-mxfp4-moe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        let files = ModelFiles::open(&dir, true).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let act = Act::Situ {
            beta: 4.0,
            linear_beta: 25.0,
        };
        let experts = Experts::load(&files, "experts.", ne, ne, mi, hl, Quant::Bf16, act).unwrap();
        let t = 16;
        let x: Vec<f32> = (0..t * hl).map(|_| (rng() % 2001) as f32 / 1000.0 - 1.0).collect();
        let routes: Vec<Vec<(usize, f32)>> = (0..t)
            .map(|ti| (0..ne).map(|e| (e, 0.1 + 0.1 * ((ti + e) % 3) as f32)).collect())
            .collect();
        let mut long = vec![0f32; t * hl];
        experts.forward(&x, t, hl, &routes, &mut long, None, None, None);
        let size = long.iter().fold(0f32, |a, b| a.max(b.abs()));
        for n in [1usize, 3] {
            let mut short = vec![0f32; t * hl];
            for (i, (xr, yr)) in x.chunks(n * hl).zip(short.chunks_mut(n * hl)).enumerate() {
                let rows = xr.len() / hl;
                experts.forward(xr, rows, hl, &routes[i * n..i * n + rows], yr, None, None, None);
            }
            let err = short.iter().zip(&long).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(err <= 0.03 * size, "passes of {n}: off by {err} (outputs up to {size})");
        }
    }

    /// A tiny random K3.1 model, without a reference (tools/ref_kimi_k31.py measures the
    /// agreement with Moonshot's code): the logits do not depend on how the tokens are split into
    /// passes, a rollback inside a recorded pass leaves no trace, and each mechanism (the
    /// indexer's choice, the banked streams, the latent norm, the MLA output gate's layer kind,
    /// SiTU's second beta) changes the logits.
    #[test]
    fn tiny_model() {
        let dir = std::env::temp_dir().join(format!("oxinfer-kimi-test-{}", std::process::id()));
        tiny_checkpoint(&dir);
        let files = ModelFiles::open(&dir, true).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let load = |over: serde_json::Value| KimiModel::load(tiny_config(over), &files, Quant::Bf16).unwrap();
        let m = load(serde_json::json!({}));
        let (h, v) = (m.cfg.hidden_size, m.cfg.vocab_size);
        let ids: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % v as u32).collect();
        let more = [5u32, 17, 2, 9];
        // The logits after the prompt (in passes of `chunk`) and after each of `more`.
        let run = |m: &KimiModel, chunk: usize| -> Vec<f32> {
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
        // A rollback inside a recorded pass: the tokens after it run as if the dropped ones never had.
        use crate::dspark::Target as _;
        let mut sess = m.new_session();
        m.forward(&mut sess, &ids[..30]);
        m.forward_feats(&mut sess, &ids[30..], &[], &[], &mut Vec::new());
        m.forward_feats(&mut sess, &[30, 31, 29], &[], &[], &mut Vec::new());
        assert!(sess.can_truncate(ids.len() + 1) && !sess.can_truncate(ids.len() - 1));
        m.truncate(&mut sess, ids.len() + 1);
        assert_eq!(sess.pos, ids.len() + 1);
        m.truncate(&mut sess, ids.len());
        let again: Vec<f32> = more.iter().flat_map(|&t| m.forward(&mut sess, &[t])).collect();
        let d = diff(&again, &base[v..]);
        assert!(d <= tol, "after a rollback the logits differ by {d}");
        // Each mechanism matters.
        let dense = serde_json::json!({"dsa_layers": [3, 6, 7], "index_topk": 64, "index_head_dim": 8, "index_num_heads": 8});
        for (what, over) in [
            ("the indexer's choice", serde_json::json!({"dsa_config": dense})),
            ("the latent norm", serde_json::json!({"latent_moe_use_norm": false})),
            ("SiTU's second beta", serde_json::json!({"activation_situ_linear_beta": null})),
            ("the banking period", serde_json::json!({"attn_res_block_size": 2})),
        ] {
            let d = diff(&run(&load(over), 40), &base);
            assert!(d > 20.0 * tol, "{what}: the logits moved by {d} only");
        }
    }
}
