//! DeepSeek V4 (`deepseek_v4`: DeepSeek-V4-Flash, 284B with 13B active, and DeepSeek-V4-Pro) text
//! model, as DeepSeek's own inference code (`inference/model.py` of the checkpoint):
//!
//! - Manifold-constrained hyper-connections (mHC): the residual stream is `hc_mult` (4) copies of
//!   the hidden state. Before the attention and before the MoE, a projection of the RMS-scaled
//!   copies gives per-token mixing weights: `pre` (sigmoid) collapses the copies into the block's
//!   input, `post` (twice a sigmoid) spreads the block's output over them, and `comb` (a softmax
//!   made doubly stochastic by Sinkhorn iterations) mixes the copies into each other. A last
//!   collapse (`hc_head`) feeds the final norm.
//! - Attention: one shared key / value head (MQA, the keys are the values) of `head_dim` (512);
//!   queries through a LoRA (`wq_a`, RMS norm, `wq_b`) and a weightless RMS norm per head; RoPE on
//!   the last `qk_rope_head_dim` (64) dimensions (interleaved pairs), undone on the attention
//!   output at the query's position; a learned sink logit per head in the softmax; the output
//!   projected per group of heads (`wo_a`) then mixed (`wo_b`).
//! - Every layer attends to a sliding window of raw keys (128 positions, itself included).
//!   Layers with a compression rate `r` add pooled keys: every `r` tokens become one key (a per
//!   channel softmax over learned gates plus a position bias weighs their projections; RMS norm;
//!   RoPE at the window's first position), visible once the window is complete. Rate 4 ("CSA")
//!   pools overlapping windows of 8 (two series of 4) and a lightning indexer (`index_n_heads` of
//!   `index_head_dim` on its own pooled keys, scores `sum_h w_h relu(q_h . k)`) chooses the
//!   `index_topk` best per query; other rates ("HCA", 128 in the release) attend to all. Compressed
//!   layers rotate with `compress_rope_theta` and YaRN (no magnitude scaling), window-only layers
//!   with plain RoPE.
//! - MoE on every layer: routed SiLU experts with clamped inputs (`swiglu_limit`) scored by
//!   `sqrt(softplus(logit))`; the first `num_hash_layers` take their experts from a token-id table
//!   (`tid2eid`), the others the top-k of score plus a selection bias; weights renormalized and
//!   scaled; one shared expert.
//!
//! The reference's quantization-aware-training simulations (FP8 rounding of the cached keys, FP4
//! of the indexer's queries and keys, the Hadamard rotation before them) are left out, as
//! transformers and llama.cpp do (the rotation alone changes no dot product).
//!
//! Checkpoints: DeepSeek's naming (`layers.N.attn.wq_a.weight`, `layers.N.ffn.experts.E.w1.weight`,
//! ...; float tensors; the experts are renamed `gate_proj` / `down_proj` / `up_proj` on open,
//! [`native_name`]) and llama.cpp GGUF files (`deepseek4`, named onto the same by [`gguf_name`]).
//! The model's DSpark draft is in [`crate::dspark`].
//!
//! Runtime interventions ([`crate::intervene`]). The residual stream is the `hc_mult` copies, and
//! "the residual" at a point is their mean (what a DSpark draft reads too): residual point `2l`
//! is the stream entering layer `l` (`Block.forward`'s input; point 0: the embedding, repeated),
//! `2l + 1` the stream after the attention's `hc_post`, `2l + 2` the layer's output (after the
//! MoE's `hc_post`). Hooks go through [`crate::intervene::State::point_copies`]: reads (capture,
//! calibration, SAE encoding) see the mean, edits write to every copy so that the mean changes
//! as a single stream would (steering adds to each copy, ablation projects each copy, SAE
//! feature changes computed on the mean are added to each copy). Neurons are the input of `w2`
//! (`down_proj`) of the shared expert and of each routed expert, `silu(min(gate, limit)) *
//! clamp(up, -limit, limit)`, before the routing weight (the reference multiplies it in before
//! `w2`; `w2` has no bias, so that is the same edit of an activation scaled afterwards). Every
//! layer is a MoE: there is no dense MLP.

use crate::config::DeepSeek4Config;
use crate::intervene::{NeuronHook, Target};
use crate::kernels::*;
use crate::model::{Lin, Quant, Table, linear, linear_multi, load_lin, table_quant};
use crate::moe::{Act, Experts, NeedExperts};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// Chunks of at least this many tokens run attention and the indexer as block GEMMs; shorter
/// ones (decode, speculative verification) per query, so a verified token gets the bits of a
/// one-token pass.
const GEMM_MIN: usize = 32;
/// Positions kept beyond what attention needs (raw keys, compressor rows), so a speculative round
/// can roll back.
const SLACK: usize = 64;

/// Whether `files` is a checkpoint in DeepSeek's naming.
pub(crate) fn is_native(files: &ModelFiles) -> bool {
    files.has("embed.weight") && files.has("layers.0.attn.wq_a.weight") && files.has("layers.0.hc_attn_fn")
}

/// The name of an expert projection of DeepSeek's naming as the loaders read it (`w1` ->
/// `gate_proj`, `w2` -> `down_proj`, `w3` -> `up_proj`, also for their `.scale`); None for other
/// tensors.
pub(crate) fn native_name(n: &str) -> Option<String> {
    let (head, tail) = n.rsplit_once('.')?;
    let (pre, w) = head.rsplit_once('.')?;
    if !(pre.contains(".ffn.experts.") || pre.ends_with(".ffn.shared_experts")) {
        return None;
    }
    let p = match w {
        "w1" => "gate_proj",
        "w2" => "down_proj",
        "w3" => "up_proj",
        _ => return None,
    };
    Some(format!("{pre}.{p}.{tail}"))
}

/// The DeepSeek-naming name and PyTorch-order shape of a llama.cpp `deepseek4` tensor (also the
/// stages of a `dflash` DSpark draft); `ne` is ggml's shape. None for tensors not read.
pub(crate) fn gguf_name(name: &str, ne: &[usize]) -> Option<(String, Vec<usize>)> {
    let mut shape: Vec<usize> = ne.iter().rev().copied().collect();
    let top = match name {
        "token_embd.weight" => Some("embed.weight"),
        "output.weight" => Some("head.weight"),
        "output_norm.weight" => Some("norm.weight"),
        "output_hc_fn.weight" => Some("hc_head_fn"),
        "output_hc_base.weight" => Some("hc_head_base"),
        "output_hc_scale.weight" => Some("hc_head_scale"),
        _ => None,
    };
    if let Some(t) = top {
        return Some((t.to_string(), shape));
    }
    let rest = name.strip_prefix("blk.")?;
    let (l, rest) = rest.split_once('.')?;
    let l: usize = l.parse().ok()?;
    let (base, _suffix) = rest.rsplit_once('.')?;
    let n = match base {
        "attn_norm" => "attn_norm.weight",
        "ffn_norm" => "ffn_norm.weight",
        "hc_attn_fn" => "hc_attn_fn",
        "hc_attn_base" => "hc_attn_base",
        "hc_attn_scale" => "hc_attn_scale",
        "hc_ffn_fn" => "hc_ffn_fn",
        "hc_ffn_base" => "hc_ffn_base",
        "hc_ffn_scale" => "hc_ffn_scale",
        "attn_sinks" => "attn.attn_sink",
        "attn_q_a" => "attn.wq_a.weight",
        "attn_q_a_norm" => "attn.q_norm.weight",
        "attn_q_b" => "attn.wq_b.weight",
        "attn_kv" => "attn.wkv.weight",
        "attn_kv_a_norm" | "attn_kv_norm" => "attn.kv_norm.weight",
        "attn_output_a" => {
            // [groups, o_lora, in] -> [groups * o_lora, in]: the same rows.
            if shape.len() == 3 {
                shape = vec![shape[0] * shape[1], shape[2]];
            }
            "attn.wo_a.weight"
        }
        "attn_output_b" => "attn.wo_b.weight",
        "attn_compressor_ape" => "attn.compressor.ape",
        "attn_compressor_kv" => "attn.compressor.wkv.weight",
        "attn_compressor_gate" => "attn.compressor.wgate.weight",
        "attn_compressor_norm" => "attn.compressor.norm.weight",
        "indexer.attn_q_b" => "attn.indexer.wq_b.weight",
        "indexer.proj" => "attn.indexer.weights_proj.weight",
        "indexer_compressor_ape" => "attn.indexer.compressor.ape",
        "indexer_compressor_kv" => "attn.indexer.compressor.wkv.weight",
        "indexer_compressor_gate" => "attn.indexer.compressor.wgate.weight",
        "indexer_compressor_norm" => "attn.indexer.compressor.norm.weight",
        "ffn_gate_inp" => "ffn.gate.weight",
        "exp_probs_b" => "ffn.gate.bias",
        "ffn_gate_tid2eid" => "ffn.gate.tid2eid",
        "ffn_gate_exps" => "ffn.experts.gate_proj",
        "ffn_up_exps" => "ffn.experts.up_proj",
        "ffn_down_exps" => "ffn.experts.down_proj",
        "ffn_gate_shexp" => "ffn.shared_experts.gate_proj.weight",
        "ffn_up_shexp" => "ffn.shared_experts.up_proj.weight",
        "ffn_down_shexp" => "ffn.shared_experts.down_proj.weight",
        _ => return None,
    };
    Some((format!("layers.{l}.{n}"), shape))
}

/// Weight streaming for a checkpoint that does not fit in memory: a page-cache budget of 70% of
/// RAM (the dense weights stay resident, the rest caches recently routed experts).
pub fn default_stream() -> crate::stream::StreamOptions {
    crate::mistral4::default_stream()
}

/// Whether a checkpoint of `bytes` is too big to hold in memory (over 60% of RAM).
pub fn too_big(bytes: u64) -> bool {
    crate::mistral4::too_big(bytes)
}

/// The shared expert (run with the routed ones: `moe::Shared`, with DeepSeek's SwiGLU clamp
/// `silu(min(g, limit)) * clamp(u, -limit, limit)` when `limit` > 0).
pub(crate) struct Mlp<'a> {
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
    inter: usize,
    limit: f32,
}

impl Mlp<'_> {
    fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

/// A hyper-connection mixer: `f` `[(2 + hc) * hc, hc * H]` (or `[hc, hc * H]` for the head),
/// `base`, and the `pre` / `post` / `comb` scales.
pub(crate) struct Hc {
    f: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
}

/// Pools `r` tokens (two overlapping series of `r` with `overlap`) into one key of `d`.
pub(crate) struct Compressor<'a> {
    wkv: Lin<'a>,   // [coff * d, H]
    wgate: Lin<'a>, // [coff * d, H]
    ape: Vec<f32>,  // [r, coff * d]
    norm: Vec<f32>, // [d]
    ratio: usize,
    d: usize,
}

impl Compressor<'_> {
    fn overlap(&self) -> bool {
        self.ratio == 4
    }
    /// Width of a projected row: both series with `overlap`.
    fn width(&self) -> usize {
        if self.overlap() { 2 * self.d } else { self.d }
    }
}

pub(crate) struct Indexer<'a> {
    wq: Lin<'a>,   // [heads * dim, q_lora]
    proj: Lin<'a>, // [heads, H]
    comp: Compressor<'a>,
}

pub(crate) struct Attn<'a> {
    wq_a: Lin<'a>, // [q_lora, H]
    q_norm: Vec<f32>,
    wq_b: Lin<'a>, // [heads * hd, q_lora]
    wkv: Lin<'a>,  // [hd, H]
    kv_norm: Vec<f32>,
    /// Per group of heads `[o_lora, heads * hd / groups]`.
    wo_a: Vec<Lin<'a>>,
    wo_b: Lin<'a>, // [H, groups * o_lora]
    sink: Vec<f32>,
    comp: Option<Compressor<'a>>,
    indexer: Option<Indexer<'a>>,
}

/// How a MoE layer picks its experts.
pub(crate) enum Pick {
    /// `tid2eid`: the experts of each token id (`[vocab, k]`).
    Hash(Vec<u32>),
    /// The top-k of score plus this bias.
    Score(Option<Vec<f32>>),
}

pub(crate) struct Moe<'a> {
    router: Vec<f32>, // [n_exp, H]
    pick: Pick,
    pub(crate) experts: Experts<'a>,
    shared: Mlp<'a>,
}

pub(crate) struct Layer<'a> {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    hc_attn: Hc,
    hc_ffn: Hc,
    pub(crate) attn: Attn<'a>,
    pub(crate) moe: Moe<'a>,
    ratio: usize,
}

/// The compressed keys of one layer (or of its indexer) and the projected rows of the tokens of
/// the windows still needed (`kv`, `sc`: gate logits plus the position bias, `width` each).
#[derive(Clone, Default)]
struct CompState {
    keys: Vec<f32>,
    n: usize,
    kv: Vec<f32>,
    sc: Vec<f32>,
    rows_start: usize,
    d: usize,
    width: usize,
    ratio: usize,
}

impl CompState {
    fn rows(&self) -> usize {
        self.kv.len() / self.width.max(1)
    }
    fn truncate(&mut self, pos: usize) {
        let n = pos / self.ratio;
        if n < self.n {
            self.n = n;
            self.keys.truncate(n * self.d);
        }
        if pos < self.rows_start + self.rows() {
            assert!(pos >= self.rows_start, "rollback beyond the kept compressor rows");
            let keep = (pos - self.rows_start) * self.width;
            self.kv.truncate(keep);
            self.sc.truncate(keep);
        }
    }
    /// Drops rows no window can need again (with room for a rollback of [`SLACK`] tokens).
    fn trim(&mut self, pos: usize) {
        let r = self.ratio;
        let keep_from = (pos.saturating_sub(SLACK) / r).saturating_sub(1) * r;
        if keep_from >= self.rows_start + 2 * SLACK {
            let drop = (keep_from - self.rows_start) * self.width;
            self.kv.drain(..drop);
            self.sc.drain(..drop);
            self.rows_start = keep_from;
        }
    }
}

#[derive(Clone, Default)]
struct LayerState {
    /// Normed, rotated key rows `[hd]` of positions `raw_start..`.
    raw: Vec<f32>,
    raw_start: usize,
    comp: Option<CompState>,
    idx: Option<CompState>,
}

/// Decoding state: per layer the recent raw keys, the compressed keys and the compressors'
/// pending rows; the position.
#[derive(Clone)]
pub struct DeepSeek4Session {
    layers: Vec<LayerState>,
    hd: usize,
    window: usize,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
}

impl DeepSeek4Session {
    /// Drops cached positions at or beyond `pos`.
    pub fn truncate(&mut self, pos: usize) {
        if pos >= self.pos {
            return;
        }
        for st in &mut self.layers {
            assert!(pos >= st.raw_start, "rollback beyond the kept raw keys");
            st.raw.truncate((pos - st.raw_start) * self.hd);
            for c in [&mut st.comp, &mut st.idx].into_iter().flatten() {
                c.truncate(pos);
            }
        }
        self.pos = pos;
    }

    fn trim(&mut self) {
        let keep_from = self.pos.saturating_sub(self.window + SLACK);
        for st in &mut self.layers {
            if keep_from >= st.raw_start + 2 * SLACK {
                st.raw.drain(..(keep_from - st.raw_start) * self.hd);
                st.raw_start = keep_from;
            }
            for c in [&mut st.comp, &mut st.idx].into_iter().flatten() {
                c.trim(self.pos);
            }
        }
    }
}

pub struct DeepSeek4Model<'a> {
    pub cfg: DeepSeek4Config,
    pub quant: Quant,
    embed: Table<'a>,
    head: Lin<'a>,
    norm: Vec<f32>,
    hc_head: Hc,
    pub(crate) layers: Vec<Layer<'a>>,
    /// Rotary inverse frequencies of the window-only layers and of the compressed ones.
    inv_main: Vec<f32>,
    inv_comp: Vec<f32>,
    /// Weight streaming: unit `l` is layer `l`, unit `num_hidden_layers` the LM head.
    stream: Option<crate::stream::StreamHook>,
}

/// Loads tensors in DeepSeek's naming with shape checks.
pub(crate) struct Loader<'a, 'c> {
    pub(crate) files: &'a ModelFiles,
    pub(crate) cfg: &'c DeepSeek4Config,
    pub(crate) quant: Quant,
}

impl<'a> Loader<'a, '_> {
    fn getf(&self, full: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.files.info(full).with_context(|| format!("missing tensor {full}"))?;
        if info.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
            bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
        }
        self.files.f32(full)
    }
    /// A matrix: GGUF blocks as stored, float32 tensors exactly, others through `load_lin`.
    fn lin(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        if let Some(info) = self.files.info(full)
            && info.dtype == "F32"
        {
            if info.shape.iter().product::<usize>() != n_out * n_in {
                bail!("tensor {full} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
            }
            self.files.note_use(full, true, false);
            return Lin::from_gguf(crate::ggml::GgmlType::F32, self.files.bytes(full)?, n_out, n_in);
        }
        load_lin(self.files, full, n_out, n_in, self.quant)
    }
    /// The `parts` row blocks of matrix `full` (`[parts * n_out, n_in]`) as one linear layer each.
    fn row_parts(&self, full: &str, parts: usize, n_out: usize, n_in: usize) -> Result<Vec<Lin<'a>>> {
        let info = self.files.info(full).with_context(|| format!("missing tensor {full}"))?;
        if info.shape.iter().product::<usize>() != parts * n_out * n_in {
            bail!("tensor {full} has shape {:?}, expected [{}, {n_in}]", info.shape, parts * n_out);
        }
        self.files.note_use(full, true, false);
        if let Some(g) = self.files.ggml(full) {
            let (kind, bytes) = g?;
            return bytes
                .chunks(bytes.len() / parts)
                .map(|b| Lin::from_gguf(kind, b, n_out, n_in))
                .collect();
        }
        if info.dtype == "F32" {
            let b = self.files.bytes(full)?;
            return b
                .chunks(b.len() / parts)
                .map(|b| Lin::from_gguf(crate::ggml::GgmlType::F32, b, n_out, n_in))
                .collect();
        }
        Ok(self.files.bf16(full)?.chunks(n_out * n_in).map(Lin::bf16).collect())
    }
    fn hc(&self, p: &str, rows: usize) -> Result<Hc> {
        let c = self.cfg;
        let w = c.hc_mult * c.hidden_size;
        Ok(Hc {
            f: self.getf(&format!("{p}_fn"), &[rows, w])?,
            base: self.getf(&format!("{p}_base"), &[rows])?,
            scale: self.getf(&format!("{p}_scale"), &[if rows == c.hc_mult { 1 } else { 3 }])?,
        })
    }
    fn mlp(&self, p: &str, inter: usize) -> Result<Mlp<'a>> {
        let h = self.cfg.hidden_size;
        Ok(Mlp {
            gate: self.lin(&format!("{p}gate_proj.weight"), inter, h)?,
            up: self.lin(&format!("{p}up_proj.weight"), inter, h)?,
            down: self.lin(&format!("{p}down_proj.weight"), h, inter)?,
            inter,
            limit: self.cfg.swiglu_limit as f32,
        })
    }
    fn compressor(&self, p: &str, ratio: usize, d: usize) -> Result<Compressor<'a>> {
        let h = self.cfg.hidden_size;
        let coff = if ratio == 4 { 2 } else { 1 };
        Ok(Compressor {
            wkv: self.lin(&format!("{p}wkv.weight"), coff * d, h)?,
            wgate: self.lin(&format!("{p}wgate.weight"), coff * d, h)?,
            ape: self.getf(&format!("{p}ape"), &[ratio, coff * d])?,
            norm: self.getf(&format!("{p}norm.weight"), &[d])?,
            ratio,
            d,
        })
    }

    /// Layer `l` (`prefix` `layers.{l}.` or a DSpark stage's), attention with rate `ratio`.
    pub(crate) fn layer(&self, prefix: &str, l: usize, ratio: usize) -> Result<Layer<'a>> {
        let c = self.cfg;
        let (h, nh, hd, ql) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank);
        let hcn = c.hc_mult;
        let mix = (2 + hcn) * hcn;
        let ap = format!("{prefix}attn.");
        let (comp, indexer) = if ratio > 0 {
            let comp = self.compressor(&format!("{ap}compressor."), ratio, hd)?;
            let indexer = if ratio == 4 {
                let (ni, di) = (c.index_n_heads, c.index_head_dim);
                Some(Indexer {
                    wq: self.lin(&format!("{ap}indexer.wq_b.weight"), ni * di, ql)?,
                    // Kept in float precision by the reference.
                    proj: {
                        let n = format!("{ap}indexer.weights_proj.weight");
                        match self.files.info(&n).map(|i| i.dtype.as_str()) {
                            Some("F32") => self.lin(&n, ni, h)?,
                            _ => load_lin(self.files, &n, ni, h, Quant::Bf16)?,
                        }
                    },
                    comp: self.compressor(&format!("{ap}indexer.compressor."), 4, di)?,
                })
            } else {
                None
            };
            (Some(comp), indexer)
        } else {
            (None, None)
        };
        let g = c.o_groups;
        let attn = Attn {
            wq_a: self.lin(&format!("{ap}wq_a.weight"), ql, h)?,
            q_norm: self.getf(&format!("{ap}q_norm.weight"), &[ql])?,
            wq_b: self.lin(&format!("{ap}wq_b.weight"), nh * hd, ql)?,
            wkv: self.lin(&format!("{ap}wkv.weight"), hd, h)?,
            kv_norm: self.getf(&format!("{ap}kv_norm.weight"), &[hd])?,
            wo_a: self.row_parts(&format!("{ap}wo_a.weight"), g, c.o_lora_rank, nh * hd / g)?,
            wo_b: self.lin(&format!("{ap}wo_b.weight"), h, g * c.o_lora_rank)?,
            sink: self.getf(&format!("{ap}attn_sink"), &[nh])?,
            comp,
            indexer,
        };
        let fp = format!("{prefix}ffn.");
        let (ne, k, mi) = (c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size);
        let pick = if l < c.num_hash_layers {
            let n = format!("{fp}gate.tid2eid");
            let info = self.files.info(&n).with_context(|| format!("missing tensor {n}"))?;
            if info.shape.iter().product::<usize>() != c.vocab_size * k {
                bail!("tensor {n} has shape {:?}, expected [{}, {k}]", info.shape, c.vocab_size);
            }
            let b = self.files.bytes(&n)?;
            let tab: Vec<u32> = match info.dtype.as_str() {
                "I32" | "U32" => b.as_chunks::<4>().0.iter().map(|x| u32::from_le_bytes(*x)).collect(),
                "I64" => b.as_chunks::<8>().0.iter().map(|x| u64::from_le_bytes(*x) as u32).collect(),
                d => bail!("tensor {n}: dtype {d}"),
            };
            if tab.iter().any(|&e| e as usize >= ne) {
                bail!("tensor {n}: expert index out of range");
            }
            Pick::Hash(tab)
        } else {
            let bn = format!("{fp}gate.bias");
            Pick::Score(if self.files.has(&bn) { Some(self.getf(&bn, &[ne])?) } else { None })
        };
        let moe = Moe {
            router: self.getf(&format!("{fp}gate.weight"), &[ne, h])?,
            pick,
            experts: Experts::load(
                self.files,
                &format!("{fp}experts."),
                ne,
                k,
                mi,
                h,
                self.quant,
                Act::SiluClamp(c.swiglu_limit as f32),
            )?,
            shared: self.mlp(&format!("{fp}shared_experts."), mi * c.n_shared_experts)?,
        };
        Ok(Layer {
            attn_norm: self.getf(&format!("{prefix}attn_norm.weight"), &[h])?,
            ffn_norm: self.getf(&format!("{prefix}ffn_norm.weight"), &[h])?,
            hc_attn: self.hc(&format!("{prefix}hc_attn"), mix)?,
            hc_ffn: self.hc(&format!("{prefix}hc_ffn"), mix)?,
            attn,
            moe,
            ratio,
        })
    }

    /// The final collapse of the residual copies (`{prefix}hc_head_*`).
    pub(crate) fn hc_head(&self, prefix: &str) -> Result<Hc> {
        self.hc(&format!("{prefix}hc_head"), self.cfg.hc_mult)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The mixing weights of one token's residual copies `x` (`[hc * H]`): `pre` `[hc]`, `post`
/// `[hc]`, `comb` `[hc * hc]` (row `j` = source copy `j`) as `hc_split_sinkhorn`.
pub(crate) fn hc_mix(c: &DeepSeek4Config, w: &Hc, x: &[f32], pre: &mut [f32], post: &mut [f32], comb: &mut [f32]) {
    let n = c.hc_mult;
    let (eps, heps) = (c.rms_norm_eps as f32, c.hc_eps as f32);
    let r = 1.0 / (dot(x, x) / x.len() as f32 + eps).sqrt();
    let m = |i: usize| dot(&w.f[i * x.len()..(i + 1) * x.len()], x) * r;
    for i in 0..n {
        pre[i] = sigmoid(m(i) * w.scale[0] + w.base[i]) + heps;
        post[i] = 2.0 * sigmoid(m(n + i) * w.scale[1] + w.base[n + i]);
    }
    for (i, v) in comb.iter_mut().enumerate() {
        *v = m(2 * n + i) * w.scale[2] + w.base[2 * n + i];
    }
    for row in comb.chunks_mut(n) {
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            s += *v;
        }
        for v in row.iter_mut() {
            *v = *v / s + heps;
        }
    }
    let cols = |comb: &mut [f32]| {
        for k in 0..n {
            let s: f32 = (0..n).map(|j| comb[j * n + k]).sum::<f32>() + heps;
            for j in 0..n {
                comb[j * n + k] /= s;
            }
        }
    };
    cols(comb);
    for _ in 1..c.hc_sinkhorn_iters {
        for row in comb.chunks_mut(n) {
            let s: f32 = row.iter().sum::<f32>() + heps;
            for v in row.iter_mut() {
                *v /= s;
            }
        }
        cols(comb);
    }
}

/// The weighted sum of a token's residual copies `x` (`[hc * H]`) into `y` (`[H]`).
fn hc_collapse(pre: &[f32], x: &[f32], y: &mut [f32]) {
    let h = y.len();
    y.fill(0.0);
    for (i, &p) in pre.iter().enumerate() {
        axpy(y, p, &x[i * h..(i + 1) * h]);
    }
}

/// Before a block: per token the mixing weights of `w`, and the collapsed, `norm`-ed input
/// (`[t, H]`); returns (input, post `[t, hc]`, comb `[t, hc * hc]`).
pub(crate) fn hc_pre(c: &DeepSeek4Config, w: &Hc, norm: &[f32], x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (n, h) = (c.hc_mult, c.hidden_size);
    let mut y = vec![0f32; t * h];
    let mut post = vec![0f32; t * n];
    let mut comb = vec![0f32; t * n * n];
    let (pp, cp) = (SendPtr(post.as_mut_ptr()), SendPtr(comb.as_mut_ptr()));
    for_chunks(&mut y, h, t > 1, |ti, yr| {
        // SAFETY: each token writes its own rows.
        let (po, co) = unsafe {
            (
                std::slice::from_raw_parts_mut(pp.get().add(ti * n), n),
                std::slice::from_raw_parts_mut(cp.get().add(ti * n * n), n * n),
            )
        };
        let xr = &x[ti * n * h..(ti + 1) * n * h];
        let mut pre = vec![0f32; n];
        hc_mix(c, w, xr, &mut pre, po, co);
        hc_collapse(&pre, xr, yr);
        rmsnorm_f32w_inplace(yr, norm, c.rms_norm_eps as f32);
    });
    (y, post, comb)
}

/// After a block: `x[k] = post[k] * f + sum_j comb[j][k] * x[j]` per token (`f` `[t, H]`).
pub(crate) fn hc_post(c: &DeepSeek4Config, x: &mut [f32], f: &[f32], post: &[f32], comb: &[f32], t: usize) {
    let (n, h) = (c.hc_mult, c.hidden_size);
    for_chunks(x, n * h, t > 1, |ti, xr| {
        let res = xr.to_vec();
        let (po, co) = (&post[ti * n..(ti + 1) * n], &comb[ti * n * n..(ti + 1) * n * n]);
        let fr = &f[ti * h..(ti + 1) * h];
        for k in 0..n {
            let o = &mut xr[k * h..(k + 1) * h];
            for (v, &a) in o.iter_mut().zip(fr) {
                *v = po[k] * a;
            }
            for j in 0..n {
                axpy(o, co[j * n + k], &res[j * h..(j + 1) * h]);
            }
        }
    });
}

/// The final collapse (`hc_head`): per token `sum_i (sigmoid(mix_i scale + base_i) + eps) x_i`.
pub(crate) fn hc_head(c: &DeepSeek4Config, w: &Hc, x: &[f32], t: usize) -> Vec<f32> {
    let (n, h) = (c.hc_mult, c.hidden_size);
    let (eps, heps) = (c.rms_norm_eps as f32, c.hc_eps as f32);
    let mut y = vec![0f32; t * h];
    for_chunks(&mut y, h, t > 1, |ti, yr| {
        let xr = &x[ti * n * h..(ti + 1) * n * h];
        let r = 1.0 / (dot(xr, xr) / xr.len() as f32 + eps).sqrt();
        let pre: Vec<f32> = (0..n)
            .map(|i| sigmoid(dot(&w.f[i * n * h..(i + 1) * n * h], xr) * r * w.scale[0] + w.base[i]) + heps)
            .collect();
        hc_collapse(&pre, xr, yr);
    });
    y
}

/// Rotates the last `cos.len() * 2` dimensions of `v` (interleaved pairs), backwards with `inv`.
#[inline]
fn rope_tail(v: &mut [f32], cos: &[f32], sin: &[f32], inv: bool) {
    let n = v.len();
    let tail = &mut v[n - 2 * cos.len()..];
    for (i, p) in tail.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let s = if inv { -sin[i] } else { sin[i] };
        let (x1, x2) = (p[0], p[1]);
        p[0] = x1 * cos[i] - x2 * s;
        p[1] = x2 * cos[i] + x1 * s;
    }
}

/// Weighted RMS norm then RoPE on the tail of each `d`-wide row of `x` (positions `start..`).
fn norm_rope_rows(x: &mut [f32], d: usize, w: Option<&[f32]>, eps: f32, cos: &[f32], sin: &[f32]) {
    let half = cos.len() / (x.len() / d).max(1);
    for_chunks(x, d, x.len() > 64 * d, |ti, r| {
        match w {
            Some(w) => rmsnorm_f32w_inplace(r, w, eps),
            None => {
                let s = 1.0 / (dot(r, r) / d as f32 + eps).sqrt();
                scale_inplace(r, s);
            }
        }
        rope_tail(r, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], false);
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

impl<'a> DeepSeek4Model<'a> {
    pub fn load(cfg: DeepSeek4Config, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        if !files.has("embed.weight") {
            bail!("cannot find embed.weight in checkpoint (DeepSeek V4 is read in DeepSeek's naming or from GGUF)");
        }
        if files.info("layers.0.attn.wq_a.weight").is_some_and(|i| i.dtype == "F8_E4M3") {
            bail!("DeepSeek V4: the FP8 / FP4 release is not supported yet (use a GGUF conversion)");
        }
        let ld = Loader { files, cfg: &cfg, quant };
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let tied = cfg.tie_word_embeddings || !files.has("head.weight");
        let embed = Table::load(files, "embed.weight", v, h, if tied { quant } else { table_quant(quant) })?;
        let head = ld.lin(if tied { "embed.weight" } else { "head.weight" }, v, h)?;
        let norm = ld.getf("norm.weight", &[h])?;
        let hc_head = ld.hc_head("")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            layers.push(
                ld.layer(&format!("layers.{l}."), l, cfg.ratio(l))
                    .with_context(|| format!("layer {l}"))?,
            );
        }
        let (inv_main, inv_comp) = cfg.rope_freqs();
        let m = Self {
            cfg,
            quant,
            embed,
            head,
            norm,
            hc_head,
            layers,
            inv_main,
            inv_comp,
            stream: None,
        };
        // Routed experts that cannot all stay in memory are read through the page cache as they
        // are routed: no accelerator mapping may pin them.
        if too_big(files.total_bytes()) {
            for l in &m.layers {
                for e in l.moe.experts.lins().into_iter().flatten() {
                    e.set_transient_device(true);
                }
            }
        }
        Ok(m)
    }

    pub fn new_session(&self) -> DeepSeek4Session {
        let c = &self.cfg;
        let cs = |d: usize, width: usize, ratio: usize| CompState {
            d,
            width,
            ratio,
            ..Default::default()
        };
        DeepSeek4Session {
            layers: self
                .layers
                .iter()
                .map(|l| LayerState {
                    comp: l.attn.comp.as_ref().map(|cp| cs(cp.d, cp.width(), cp.ratio)),
                    idx: l.attn.indexer.as_ref().map(|ix| cs(ix.comp.d, ix.comp.width(), ix.comp.ratio)),
                    ..Default::default()
                })
                .collect(),
            hd: c.head_dim,
            window: c.sliding_window,
            pos: 0,
            iv: Default::default(),
        }
    }

    fn attn_lins<'s>(&'s self, a: &'s Attn<'a>) -> Vec<&'s Lin<'a>> {
        let mut v = vec![&a.wq_a, &a.wq_b, &a.wkv, &a.wo_b];
        v.extend(a.wo_a.iter());
        if let Some(cp) = &a.comp {
            v.extend([&cp.wkv, &cp.wgate]);
        }
        if let Some(ix) = &a.indexer {
            v.extend([&ix.wq, &ix.proj, &ix.comp.wkv, &ix.comp.wgate]);
        }
        v
    }

    fn dense_lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let mut v = self.attn_lins(&l.attn);
        let s = &l.moe.shared;
        v.extend([&s.gate, &s.up, &s.down]);
        v
    }

    /// Weight bytes read per decoded token (the routed experts at top-k).
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.head.bytes();
        for l in &self.layers {
            n += self.attn_lins(&l.attn).iter().map(|w| w.bytes()).sum::<usize>();
            n += l.moe.shared.bytes() + l.moe.experts.bytes_per_token();
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
                parts: layer.moe.experts.ranges(),
            })
            .collect();
        v.push(crate::stream::UnitSpec {
            name: "lm_head".into(),
            ranges: self.head.mem_ranges(),
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
            for e in layer.moe.experts.lins().into_iter().flatten() {
                e.set_transient_device(true);
            }
        }
        self.head
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
        linear(&self.head, c.vocab_size, c.hidden_size, hidden, t, &mut logits);
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
    pub fn forward(&self, sess: &mut DeepSeek4Session, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden(sess, tokens);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos`, appending to the caches; returns the final-normed hidden
    /// states `[t, hidden]`.
    pub fn forward_hidden(&self, sess: &mut DeepSeek4Session, tokens: &[u32]) -> Vec<f32> {
        self.forward_capture(sess, tokens, &[], &mut Vec::new())
    }

    /// [`Self::forward_hidden`] that also writes to `feats` (`[t, layers.len() * hidden]`) the
    /// mean of the residual copies entering each of `layers`, `num_hidden_layers` standing for
    /// the last layer's output (the target features of a DSpark draft).
    pub fn forward_capture(&self, sess: &mut DeepSeek4Session, tokens: &[u32], layers: &[usize], feats: &mut Vec<f32>) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let c = &self.cfg;
        let (h, n) = (c.hidden_size, c.hc_mult);
        let start = sess.pos;
        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut x = vec![0f32; t * n * h];
        for_chunks(&mut x, n * h, t > 1, |ti, row| {
            self.embed.row(tokens[ti] as usize, &mut row[..h]);
            for k in 1..n {
                row.copy_within(..h, k * h);
            }
        });
        let positions: Vec<usize> = (start..start + t).collect();
        let ropes = [rope_tables(&positions, &self.inv_main), rope_tables(&positions, &self.inv_comp)];
        drop(timer);
        let nf = layers.len();
        feats.clear();
        feats.resize(t * nf * h, 0.0);
        // Interventions: residual points on the copies (see the module docs).
        let iv = sess.iv.active().then(|| sess.iv.clone());
        let point = |p: usize, x: &mut [f32]| {
            if let Some(iv) = &iv {
                iv.point_copies(p, x, h, n, start);
            }
        };
        point(0, &mut x);
        let mut capture = |l: usize, x: &[f32]| {
            for (k, _) in layers.iter().enumerate().filter(|e| *e.1 == l) {
                for ti in 0..t {
                    let o = &mut feats[(ti * nf + k) * h..(ti * nf + k + 1) * h];
                    hc_collapse(&vec![1.0 / n as f32; n], &x[ti * n * h..(ti + 1) * n * h], o);
                }
            }
        };
        for (l, layer) in self.layers.iter().enumerate() {
            capture(l, &x);
            let _unit = crate::stream::enter(&self.stream, l);
            let (xa, post, comb) = hc_pre(c, &layer.hc_attn, &layer.attn_norm, &x, t);
            let rope = &ropes[usize::from(layer.ratio > 0)];
            let mut o = vec![0f32; t * h];
            self.attn_forward(l, layer, &xa, t, start, &mut sess.layers[l], (&rope.0, &rope.1), &mut o);
            hc_post(c, &mut x, &o, &post, &comb, t);
            point(2 * l + 1, &mut x);
            let (xf, post, comb) = hc_pre(c, &layer.hc_ffn, &layer.ffn_norm, &x, t);
            o.fill(0.0);
            let need = |ids: &[usize]| {
                if let Some(s) = &self.stream {
                    s.need_parts(l, ids)
                }
            };
            let need: NeedExperts = self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize]));
            let hook = iv.as_ref().and_then(|s| s.hook(l, start));
            self.moe_forward(&layer.moe, &xf, tokens, t, &mut o, need, hook.as_ref());
            hc_post(c, &mut x, &o, &post, &comb, t);
            point(2 * l + 2, &mut x);
        }
        capture(self.layers.len(), &x);
        let mut out = hc_head(c, &self.hc_head, &x, t);
        rmsnorm_f32w_rows_inplace(&mut out, h, &self.norm, c.rms_norm_eps as f32);
        sess.pos += t;
        sess.trim();
        out
    }

    /// Routes each token and runs the routed and shared experts of `m` over `x` (`[t, H]`) into
    /// `y` (`hook`: neuron edits).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_forward(
        &self,
        m: &Moe,
        x: &[f32],
        tokens: &[u32],
        t: usize,
        y: &mut [f32],
        need: NeedExperts,
        hook: Option<&NeuronHook>,
    ) {
        moe_forward(&self.cfg, m, x, tokens, t, y, need, hook)
    }

    /// One layer's attention over `xa` (`[t, H]`, the normed block input) into `out`.
    #[allow(clippy::too_many_arguments)]
    fn attn_forward(
        &self,
        l: usize,
        layer: &Layer,
        xa: &[f32],
        t: usize,
        start: usize,
        st: &mut LayerState,
        rope: (&[f32], &[f32]),
        out: &mut [f32],
    ) {
        let _ = l;
        let c = &self.cfg;
        let a = &layer.attn;
        let (h, nh, hd, ql) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank);
        let eps = c.rms_norm_eps as f32;
        let (cos, sin) = rope;
        // Projections of the block input.
        let mut qa = vec![0f32; t * ql];
        let mut kv = vec![0f32; t * hd];
        let cw = a.comp.as_ref().map_or(0, |cp| cp.width());
        let (mut ckv, mut cg) = (vec![0f32; t * cw], vec![0f32; t * cw]);
        let (iw_n, ic_w) = match &a.indexer {
            Some(ix) => (c.index_n_heads, ix.comp.width()),
            None => (0, 0),
        };
        // The indexer only chooses for a query that sees more than `index_topk` compressed keys
        // (otherwise every one is selected, see `select`): before that its query projection is
        // not computed (its compressor still runs, for later queries). The head weights stay in
        // the fused batch below, whose composition decides the accelerator's kernel (with the
        // same jobs, the other projections keep their bits).
        let choose = a.indexer.as_ref().is_some_and(|ix| (start + t) / ix.comp.ratio > c.index_topk);
        let (mut iw, mut ikv, mut ig) = (vec![0f32; t * iw_n], vec![0f32; t * ic_w], vec![0f32; t * ic_w]);
        {
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.wq_a, ql, &mut qa[..]), (&a.wkv, hd, &mut kv[..])];
            if let Some(cp) = &a.comp {
                jobs.push((&cp.wkv, cw, &mut ckv[..]));
                jobs.push((&cp.wgate, cw, &mut cg[..]));
            }
            if let Some(ix) = &a.indexer {
                jobs.push((&ix.proj, iw_n, &mut iw[..]));
                jobs.push((&ix.comp.wkv, ic_w, &mut ikv[..]));
                jobs.push((&ix.comp.wgate, ic_w, &mut ig[..]));
            }
            linear_multi(&mut jobs, h, xa, t);
        }
        rmsnorm_f32w_rows_inplace(&mut qa, ql, &a.q_norm, eps);
        let mut q = vec![0f32; t * nh * hd];
        linear(&a.wq_b, nh * hd, ql, &qa, t, &mut q);
        // Per head: weightless RMS norm, rotated tail.
        let half = cos.len() / t;
        for_chunks(&mut q, hd, t * nh > 64, |i, qh| {
            let ti = i / nh;
            let s = 1.0 / (dot(qh, qh) / hd as f32 + eps).sqrt();
            scale_inplace(qh, s);
            rope_tail(qh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], false);
        });
        norm_rope_rows(&mut kv, hd, Some(&a.kv_norm), eps, cos, sin);
        debug_assert_eq!(st.raw_start + st.raw.len() / hd, start);
        st.raw.extend_from_slice(&kv);
        if let (Some(cp), Some(cs)) = (&a.comp, st.comp.as_mut()) {
            compress(c, cp, cs, &ckv, &cg, t, start, &self.inv_comp);
        }
        // The indexer's choice per query (None: every visible compressed key).
        let mut sel: Vec<Option<Vec<u32>>> = vec![None; t];
        if let (Some(ix), Some(is)) = (&a.indexer, st.idx.as_mut()) {
            compress(c, &ix.comp, is, &ikv, &ig, t, start, &self.inv_comp);
        }
        if let (true, Some(ix), Some(is)) = (choose, &a.indexer, st.idx.as_ref()) {
            let (ni, di) = (c.index_n_heads, c.index_head_dim);
            let mut iq = vec![0f32; t * ni * di];
            linear(&ix.wq, ni * di, ql, &qa, t, &mut iq);
            for_chunks(&mut iq, di, t * ni > 64, |i, qh| {
                let ti = i / ni;
                rope_tail(qh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], false);
            });
            let ws = (di as f32).powf(-0.5) * (ni as f32).powf(-0.5);
            iw.iter_mut().for_each(|w| *w *= ws);
            sel = select(c, &iq, &iw, is, t, start);
        }
        let mut att = vec![0f32; t * nh * hd];
        attend(c, &a.sink, &q, st, &sel, layer.ratio, t, start, &mut att);
        // Undo the rotation of the values' tails at each query's position.
        for_chunks(&mut att, hd, t * nh > 64, |i, oh| {
            let ti = i / nh;
            rope_tail(oh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], true);
        });
        grouped_out(c, a, &att, t, out);
    }
}

/// The grouped output projection: each group of heads through its `wo_a` block, then `wo_b`.
fn grouped_out(c: &DeepSeek4Config, a: &Attn, att: &[f32], t: usize, out: &mut [f32]) {
    let (h, nh, hd, g, r) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.o_groups, c.o_lora_rank);
    let gd = nh * hd / g;
    if t < GEMM_MIN {
        // Short passes: each group on the decode kernels (the grouped GEMM pads to whole tiles).
        let mut y = vec![0f32; t * g * r];
        let mut xi = vec![0f32; t * gd];
        let mut yi = vec![0f32; t * r];
        for (gi, w) in a.wo_a.iter().enumerate() {
            for ti in 0..t {
                xi[ti * gd..(ti + 1) * gd].copy_from_slice(&att[ti * nh * hd + gi * gd..ti * nh * hd + (gi + 1) * gd]);
            }
            linear(w, r, gd, &xi, t, &mut yi);
            for ti in 0..t {
                y[(ti * g + gi) * r..(ti * g + gi + 1) * r].copy_from_slice(&yi[ti * r..(ti + 1) * r]);
            }
        }
        linear(&a.wo_b, h, g * r, &y, t, out);
        return;
    }
    let mut xg = vec![0f32; g * t * gd];
    for ti in 0..t {
        for gi in 0..g {
            xg[(gi * t + ti) * gd..(gi * t + ti + 1) * gd].copy_from_slice(&att[ti * nh * hd + gi * gd..ti * nh * hd + (gi + 1) * gd]);
        }
    }
    let yg = crate::mistral4::per_head(&a.wo_a, &xg, t, gd, r);
    let mut y = vec![0f32; t * g * r];
    for ti in 0..t {
        for gi in 0..g {
            y[(ti * g + gi) * r..(ti * g + gi + 1) * r].copy_from_slice(&yg[(gi * t + ti) * r..(gi * t + ti + 1) * r]);
        }
    }
    linear(&a.wo_b, h, g * r, &y, t, out);
}

/// Appends the projected rows of `t` tokens at `start..` to a compressor's state (`kvr`, `gr`:
/// `[t, width]`, gate logits without the position bias) and pools every window they complete.
#[allow(clippy::too_many_arguments)]
fn compress(c: &DeepSeek4Config, cp: &Compressor, cs: &mut CompState, kvr: &[f32], gr: &[f32], t: usize, start: usize, inv: &[f32]) {
    let (r, w) = (cp.ratio, cp.width());
    debug_assert_eq!(cs.rows_start + cs.rows(), start);
    for ti in 0..t {
        let p = start + ti;
        cs.kv.extend_from_slice(&kvr[ti * w..(ti + 1) * w]);
        let j = p % r;
        cs.sc
            .extend(gr[ti * w..(ti + 1) * w].iter().zip(&cp.ape[j * w..(j + 1) * w]).map(|(a, b)| a + b));
        if (p + 1).is_multiple_of(r) {
            let w0 = p + 1 - r;
            debug_assert_eq!(cs.n, w0 / r);
            let key = pool(c, cp, cs, w0, inv);
            cs.keys.extend_from_slice(&key);
            cs.n += 1;
        }
    }
}

/// The compressed key of the window starting at `w0`.
fn pool(c: &DeepSeek4Config, cp: &Compressor, cs: &CompState, w0: usize, inv: &[f32]) -> Vec<f32> {
    let (r, d, w) = (cp.ratio, cp.d, cp.width());
    let row = |p: usize| (p - cs.rows_start) * w;
    // Slots: with overlap, the previous window's first series, then this window's second series.
    let mut slots: Vec<usize> = Vec::with_capacity(2 * r);
    let mut cols: Vec<usize> = Vec::with_capacity(2 * r);
    if cp.overlap() && w0 >= r {
        for p in w0 - r..w0 {
            slots.push(row(p));
            cols.push(0);
        }
    }
    for p in w0..w0 + r {
        slots.push(row(p));
        cols.push(if cp.overlap() { d } else { 0 });
    }
    let mut key = vec![0f32; d];
    let mut s = vec![0f32; slots.len()];
    for (ch, k) in key.iter_mut().enumerate() {
        for (i, (&o, &co)) in slots.iter().zip(&cols).enumerate() {
            s[i] = cs.sc[o + co + ch];
        }
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut tot = 0.0;
        let mut acc = 0.0;
        for (i, (&o, &co)) in slots.iter().zip(&cols).enumerate() {
            let e = (s[i] - m).exp();
            tot += e;
            acc += e * cs.kv[o + co + ch];
        }
        *k = acc / tot;
    }
    rmsnorm_f32w_inplace(&mut key, &cp.norm, c.rms_norm_eps as f32);
    let (cos, sin) = rope_tables(&[w0], inv);
    rope_tail(&mut key, &cos, &sin, false);
    key
}

/// The lightning indexer's choice for each query of the pass: the `index_topk` compressed keys of
/// highest `sum_h w_h relu(q_h . k)` among the visible ones (None when there are no more than
/// that). `iq` `[t, heads, dim]` (rotated), `iw` `[t, heads]` (scaled).
fn select(c: &DeepSeek4Config, iq: &[f32], iw: &[f32], is: &CompState, t: usize, start: usize) -> Vec<Option<Vec<u32>>> {
    let (ni, di, topk, r) = (c.index_n_heads, c.index_head_dim, c.index_topk, is.ratio);
    let mut sel: Vec<Option<Vec<u32>>> = vec![None; t];
    let rows: Vec<usize> = (0..t).filter(|&i| (start + i + 1) / r > topk).collect();
    if rows.is_empty() {
        return sel;
    }
    let _timer = prof::start(Cat::Attention);
    let keys = &is.keys;
    let score_row = |i: usize, sc: &dyn Fn(usize, usize) -> f32| -> Vec<u32> {
        let nv = (start + i + 1) / r;
        let mut s = vec![0f32; nv];
        for (wi, o) in s.iter_mut().enumerate() {
            let mut acc = 0f32;
            for hh in 0..ni {
                acc += iw[i * ni + hh] * sc(hh, wi).max(0.0);
            }
            *o = acc;
        }
        top_positions(&s, topk)
    };
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); rows.len()];
    let op = SendPtrV(out.as_mut_ptr());
    if t < GEMM_MIN && rows.len() < crate::pool::threads() {
        // Few queries (decode, verification): each one's scores in parallel over blocks of
        // compressed keys (the same per-key arithmetic as `score_row`), then its top-k.
        for (ri, &i) in rows.iter().enumerate() {
            let nv = (start + i + 1) / r;
            let mut s = vec![0f32; nv];
            let sp = SendPtr(s.as_mut_ptr());
            let qs: Vec<&[f32]> = (0..ni).map(|hh| &iq[(i * ni + hh) * di..(i * ni + hh + 1) * di]).collect();
            let kb = nv.div_ceil(4 * crate::pool::threads()).max(16);
            crate::pool::run(nv.div_ceil(kb), &|b| {
                let mut d = vec![0f32; ni];
                for wi in b * kb..((b + 1) * kb).min(nv) {
                    crate::attn::dots(&qs, &keys[wi * di..(wi + 1) * di], &mut d);
                    let mut acc = 0f32;
                    for hh in 0..ni {
                        acc += iw[i * ni + hh] * d[hh].max(0.0);
                    }
                    // SAFETY: each task writes its own block of `s`.
                    unsafe { *sp.get().add(wi) = acc };
                }
            });
            out[ri] = top_positions(&s, topk);
        }
    } else if t < GEMM_MIN {
        crate::pool::run(rows.len(), &|ri| {
            let i = rows[ri];
            let v = score_row(i, &|hh, wi| {
                dot(&iq[(i * ni + hh) * di..(i * ni + hh + 1) * di], &keys[wi * di..(wi + 1) * di])
            });
            // SAFETY: each task writes its own slot.
            unsafe { *op.get().add(ri) = v };
        });
    } else {
        let n = (start + t) / r;
        let rb = ((1usize << 22) / (ni * n.max(1))).clamp(1, 64);
        for (bi, blk) in rows.chunks(rb).enumerate() {
            let mut qb = vec![0f32; blk.len() * ni * di];
            for (k, &i) in blk.iter().enumerate() {
                qb[k * ni * di..(k + 1) * ni * di].copy_from_slice(&iq[i * ni * di..(i + 1) * ni * di]);
            }
            let sc = crate::sd::ops::matmul_f32(&keys[..n * di], n, di, &qb, blk.len() * ni);
            crate::pool::run(blk.len(), &|k| {
                let v = score_row(blk[k], &|hh, wi| sc[(k * ni + hh) * n + wi]);
                // SAFETY: each task writes its own slot.
                unsafe { *op.get().add(bi * rb + k) = v };
            });
        }
    }
    for (&i, v) in rows.iter().zip(out) {
        sel[i] = Some(v);
    }
    sel
}

/// Attention of each query to its window of raw keys and its compressed keys, with the sinks;
/// `q` `[t, heads, hd]`, `out` the same.
#[allow(clippy::too_many_arguments)]
fn attend(
    c: &DeepSeek4Config,
    sink: &[f32],
    q: &[f32],
    st: &LayerState,
    sel: &[Option<Vec<u32>>],
    ratio: usize,
    t: usize,
    start: usize,
    out: &mut [f32],
) {
    let _timer = prof::start(Cat::Attention);
    let (nh, hd, win) = (c.num_attention_heads, c.head_dim, c.sliding_window);
    let scale = (hd as f32).powf(-0.5);
    let comp = st.comp.as_ref();
    let raw = |p: usize| &st.raw[(p - st.raw_start) * hd..(p - st.raw_start + 1) * hd];
    let ckey = |w: usize| &comp.unwrap().keys[w * hd..(w + 1) * hd];
    // The keys of query `i`: window positions, then compressed entries.
    let keys_of = |i: usize| -> (std::ops::Range<usize>, Vec<u32>) {
        let p = start + i;
        let lo = (p + 1).saturating_sub(win);
        let cks: Vec<u32> = match &sel[i] {
            Some(s) => s.clone(),
            None => (0..(p + 1).checked_div(ratio).unwrap_or(0) as u32).collect(),
        };
        (lo..p + 1, cks)
    };
    if t < GEMM_MIN {
        // Every head shares the one key-value head: the shared-row kernel reads each key once
        // per phase for all 64 heads (see `crate::attn`). Per head: scaled scores, softmax with
        // the sink, `o = sum_j (e_j / tot) k_j` in key order, as a per-head loop would.
        let lists: Vec<(std::ops::Range<usize>, Vec<u32>)> = (0..t).map(keys_of).collect();
        let key = |i: usize, j: usize| -> &[f32] {
            let (wr, cks) = &lists[i];
            if j < wr.len() {
                raw(wr.start + j)
            } else {
                ckey(cks[j - wr.len()] as usize)
            }
        };
        crate::attn::attend_shared(
            t,
            nh,
            hd,
            &|i| lists[i].0.len() + lists[i].1.len(),
            &|i, keys, sc| {
                let qs: Vec<&[f32]> = (0..nh).map(|hh| &q[(i * nh + hh) * hd..(i * nh + hh + 1) * hd]).collect();
                crate::attn::score_rows(&qs, keys, &|j| key(i, j), Some(scale), sc);
            },
            &|_, hh, s| {
                let m = s.iter().copied().fold(sink[hh], f32::max);
                let mut tot = (sink[hh] - m).exp();
                for v in s.iter_mut() {
                    *v = (*v - m).exp();
                    tot += *v;
                }
                s.iter_mut().for_each(|p| *p /= tot);
            },
            &key,
            &|i, hh| (i * nh + hh) * hd,
            out,
        );
        return;
    }
    // Blocks of queries: scores and outputs as GEMMs over the union of their keys.
    const B: usize = 32;
    for b0 in (0..t).step_by(B) {
        let b1 = (b0 + B).min(t);
        let (pf, pl) = (start + b0, start + b1 - 1);
        let wlo = (pf + 1).saturating_sub(win);
        let nw = pl + 1 - wlo;
        let nc = (pl + 1).checked_div(ratio).unwrap_or(0);
        let nk = nw + nc;
        let nkp = nk.next_multiple_of(32);
        let mut km = vec![0f32; nkp * hd];
        for j in 0..nw {
            km[j * hd..(j + 1) * hd].copy_from_slice(raw(wlo + j));
        }
        for w in 0..nc {
            km[(nw + w) * hd..(nw + w + 1) * hd].copy_from_slice(ckey(w));
        }
        let rows = (b1 - b0) * nh;
        let sc = crate::sd::ops::matmul_f32(&km, nkp, hd, &q[b0 * nh * hd..b1 * nh * hd], rows);
        let mut p = vec![0f32; rows * nkp];
        for_chunks(&mut p, nkp, true, |r, pr| {
            let i = b0 + r / nh;
            let hh = r % nh;
            let (wr, cks) = keys_of(i);
            let s = &sc[r * nkp..(r + 1) * nkp];
            let mut m = sink[hh];
            for j in wr.clone() {
                m = m.max(s[j - wlo] * scale);
            }
            for &w in &cks {
                m = m.max(s[nw + w as usize] * scale);
            }
            let mut tot = (sink[hh] - m).exp();
            for j in wr {
                let e = (s[j - wlo] * scale - m).exp();
                pr[j - wlo] = e;
                tot += e;
            }
            for &w in &cks {
                let e = (s[nw + w as usize] * scale - m).exp();
                pr[nw + w as usize] = e;
                tot += e;
            }
            let inv = 1.0 / tot;
            pr.iter_mut().for_each(|v| *v *= inv);
        });
        // O = P K: the keys transposed as the "weights".
        let mut kt = vec![0f32; hd * nkp];
        for_chunks(&mut kt, nkp, true, |d, row| {
            for (j, x) in row.iter_mut().enumerate() {
                *x = km[j * hd + d];
            }
        });
        let o = crate::sd::ops::matmul_f32(&kt, hd, nkp, &p, rows);
        out[b0 * nh * hd..b1 * nh * hd].copy_from_slice(&o);
    }
}

/// The routing of one token (`logits` over the experts) as the reference's `Gate`.
fn route(c: &DeepSeek4Config, m: &Moe, logits: &[f32], token: u32) -> Vec<(usize, f32)> {
    let k = c.num_experts_per_tok;
    let scores: Vec<f32> = match c.scoring_func.as_str() {
        "softmax" => {
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = logits.iter().map(|&v| (v - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            e.into_iter().map(|v| v / s).collect()
        }
        "sigmoid" => logits.iter().map(|&v| sigmoid(v)).collect(),
        // sqrt(softplus), softplus as torch (linear above 20).
        _ => logits.iter().map(|&v| if v > 20.0 { v } else { v.exp().ln_1p() }.sqrt()).collect(),
    };
    let idx: Vec<usize> = match &m.pick {
        Pick::Hash(tab) => tab[token as usize * k..(token as usize + 1) * k]
            .iter()
            .map(|&e| e as usize)
            .collect(),
        Pick::Score(bias) => {
            let sel: Vec<f32> = match bias {
                Some(b) => scores.iter().zip(b).map(|(s, b)| s + b).collect(),
                None => scores.clone(),
            };
            let mut idx: Vec<usize> = (0..sel.len()).collect();
            idx.sort_by(|&a, &b| sel[b].total_cmp(&sel[a]).then(a.cmp(&b)));
            idx.truncate(k);
            idx
        }
    };
    let mut w: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
    if c.scoring_func != "softmax" {
        let s: f32 = w.iter().sum();
        w.iter_mut().for_each(|v| *v /= s);
    }
    let rs = c.routed_scaling_factor as f32;
    idx.into_iter().zip(w).map(|(i, v)| (i, v * rs)).collect()
}

/// The MoE block of the target or of a DSpark stage (`hook`: neuron edits of the shared expert
/// and of the routed ones, on the input of their down projection).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_forward(
    c: &DeepSeek4Config,
    m: &Moe,
    x: &[f32],
    tokens: &[u32],
    t: usize,
    y: &mut [f32],
    need: NeedExperts,
    hook: Option<&NeuronHook>,
) {
    let h = c.hidden_size;
    let s = &m.shared;
    let shared = crate::moe::Shared {
        gate: &s.gate,
        up: &s.up,
        down: &s.down,
        inter: s.inter,
        act: if s.limit > 0.0 { Act::SiluClamp(s.limit) } else { Act::Silu },
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
    let routes: Vec<_> = logits.chunks(ne).zip(tokens).map(|(lg, &tk)| route(c, m, lg, tk)).collect();
    m.experts.forward(x, t, h, &routes, y, need, hook, Some(&shared));
}

/// The DSpark draft of DeepSeek V4 (the `mtp.*` stages of deepseek-ai/DeepSeek-V4-Flash-DSpark;
/// llama.cpp's `dflash` GGUF with hyper-connections, its tensors named as the target's by
/// [`gguf_name`]), as the checkpoint's `inference/model.py`:
///
/// - Context: the target's residual (mean of its copies) after a few layers (`dspark_target_layer_ids`,
///   40..42 of 43), concatenated, projected (`main_proj`, GGUF `fc`) and RMS-normed (`main_norm`,
///   `enc.output_norm`); each stage turns that into a key of its sliding window (`kv_norm(wkv(.))`,
///   rotated at the position), the last `sliding_window` positions.
/// - Block: the anchor token (committed, not yet run by the target) at position `n` and
///   `block_size - 1` noise tokens (`dspark_noise_token_id`) after it, through full DeepSeek V4
///   blocks (hyper-connections, MoE, window-only attention) whose queries attend to the window and
///   to the whole block (no causal mask); the draft's own `hc_head` and norm, the target's head.
///   Slot `i` predicts the token at `n + i + 1`.
/// - Markov head: each slot's logits get `markov_w2 markov_w1[prev]`, chained on the drafted tokens
///   (in [`crate::dspark`]).
pub(crate) struct Ds4Draft<'a> {
    pub cfg: DeepSeek4Config,
    stages: Vec<Layer<'a>>,
    fc: Lin<'a>, // [H, n_feat]
    main_norm: Vec<f32>,
    hc_head: Hc,
    norm: Vec<f32>,
    inv: Vec<f32>,
}

impl<'a> Ds4Draft<'a> {
    /// Loads the stages of a draft GGUF (`files`, named by [`gguf_name`]) whose features are
    /// `n_feat` wide.
    pub(crate) fn load(files: &'a ModelFiles, cfg: DeepSeek4Config, quant: Quant, n_feat: usize) -> Result<Self> {
        if cfg.compress_ratios.iter().take(cfg.num_hidden_layers).any(|&r| r != 0) {
            bail!("DeepSeek V4 DSpark draft: compressed attention in a stage is not supported");
        }
        let ld = Loader { files, cfg: &cfg, quant };
        let h = cfg.hidden_size;
        let stages = (0..cfg.num_hidden_layers)
            .map(|l| {
                ld.layer(&format!("layers.{l}."), usize::MAX, 0)
                    .with_context(|| format!("DSpark stage {l}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let (inv, _) = cfg.rope_freqs();
        Ok(Self {
            fc: ld.lin("fc.weight", h, n_feat)?,
            main_norm: ld.getf("enc.output_norm.weight", &[h])?,
            hc_head: ld.hc_head("")?,
            norm: ld.getf("norm.weight", &[h])?,
            stages,
            inv,
            cfg,
        })
    }

    pub(crate) fn n_stages(&self) -> usize {
        self.stages.len()
    }

    /// Weight bytes one draft pass reads (routed experts at top-k per slot).
    pub(crate) fn pass_bytes(&self, slots: usize) -> usize {
        let mut n = self.fc.bytes();
        for l in &self.stages {
            let a = &l.attn;
            n += [&a.wq_a, &a.wq_b, &a.wkv, &a.wo_b].iter().map(|w| w.bytes()).sum::<usize>();
            n += a.wo_a.iter().map(|w| w.bytes()).sum::<usize>();
            n += l.moe.shared.bytes() + slots * l.moe.experts.bytes_per_token();
        }
        n
    }

    /// Each stage's window keys (`[t, head_dim]`) for `t` rows of target features (`[t, n_feat]`)
    /// at positions `start..`.
    pub(crate) fn context(&self, feats: &[f32], t: usize, start: usize) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let (h, hd) = (c.hidden_size, c.head_dim);
        let eps = c.rms_norm_eps as f32;
        let mut x = vec![0f32; t * h];
        linear(&self.fc, h, feats.len() / t, feats, t, &mut x);
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.main_norm, eps);
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = rope_tables(&positions, &self.inv);
        self.stages
            .iter()
            .map(|l| {
                let mut kv = vec![0f32; t * hd];
                linear(&l.attn.wkv, hd, h, &x, t, &mut kv);
                norm_rope_rows(&mut kv, hd, Some(&l.attn.kv_norm), eps, &cos, &sin);
                kv
            })
            .collect()
    }

    /// Runs the block whose input embeddings are `emb` (`[nb, H]`: the anchor's, then the noise
    /// token's) at positions `pos..` over each stage's window keys `ctx` (`[n_ctx, head_dim]`);
    /// returns the final-normed states `[nb, H]`.
    pub(crate) fn block(&self, emb: &[f32], nb: usize, pos: usize, ctx: &[&[f32]]) -> Vec<f32> {
        let c = &self.cfg;
        let (h, n) = (c.hidden_size, c.hc_mult);
        let mut x = vec![0f32; nb * n * h];
        for (i, row) in x.chunks_mut(n * h).enumerate() {
            for k in 0..n {
                row[k * h..(k + 1) * h].copy_from_slice(&emb[i * h..(i + 1) * h]);
            }
        }
        let positions: Vec<usize> = (pos..pos + nb).collect();
        let (cos, sin) = rope_tables(&positions, &self.inv);
        let tokens = vec![0u32; nb];
        for (l, layer) in self.stages.iter().enumerate() {
            let (xa, post, comb) = hc_pre(c, &layer.hc_attn, &layer.attn_norm, &x, nb);
            let mut o = vec![0f32; nb * h];
            self.attn(&layer.attn, &xa, nb, ctx[l], (&cos, &sin), &mut o);
            hc_post(c, &mut x, &o, &post, &comb, nb);
            let (xf, post, comb) = hc_pre(c, &layer.hc_ffn, &layer.ffn_norm, &x, nb);
            o.fill(0.0);
            moe_forward(c, &layer.moe, &xf, &tokens, nb, &mut o, None, None);
            hc_post(c, &mut x, &o, &post, &comb, nb);
        }
        let mut out = hc_head(c, &self.hc_head, &x, nb);
        rmsnorm_f32w_rows_inplace(&mut out, h, &self.norm, c.rms_norm_eps as f32);
        out
    }

    /// A stage's attention: the block's queries over the window keys `ctx` and the block's own
    /// keys, no mask, with the sinks.
    fn attn(&self, a: &Attn, xa: &[f32], nb: usize, ctx: &[f32], rope: (&[f32], &[f32]), out: &mut [f32]) {
        let c = &self.cfg;
        let (h, nh, hd, ql) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank);
        let eps = c.rms_norm_eps as f32;
        let (cos, sin) = rope;
        let half = cos.len() / nb;
        let mut qa = vec![0f32; nb * ql];
        let mut kv = vec![0f32; nb * hd];
        linear_multi(&mut [(&a.wq_a, ql, &mut qa[..]), (&a.wkv, hd, &mut kv[..])], h, xa, nb);
        rmsnorm_f32w_rows_inplace(&mut qa, ql, &a.q_norm, eps);
        let mut q = vec![0f32; nb * nh * hd];
        linear(&a.wq_b, nh * hd, ql, &qa, nb, &mut q);
        for_chunks(&mut q, hd, false, |i, qh| {
            let ti = i / nh;
            let s = 1.0 / (dot(qh, qh) / hd as f32 + eps).sqrt();
            scale_inplace(qh, s);
            rope_tail(qh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], false);
        });
        norm_rope_rows(&mut kv, hd, Some(&a.kv_norm), eps, cos, sin);
        let n_ctx = ctx.len() / hd;
        let key = |j: usize| -> &[f32] {
            if j < n_ctx {
                &ctx[j * hd..(j + 1) * hd]
            } else {
                &kv[(j - n_ctx) * hd..(j - n_ctx + 1) * hd]
            }
        };
        let scale = (hd as f32).powf(-0.5);
        let mut att = vec![0f32; nb * nh * hd];
        let op = SendPtr(att.as_mut_ptr());
        crate::pool::run(nb * nh, &|task| {
            let (i, hh) = (task / nh, task % nh);
            let qh = &q[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
            let mut s: Vec<f32> = (0..n_ctx + nb).map(|j| dot(qh, key(j)) * scale).collect();
            let m = s.iter().copied().fold(a.sink[hh], f32::max);
            let mut tot = (a.sink[hh] - m).exp();
            for v in s.iter_mut() {
                *v = (*v - m).exp();
                tot += *v;
            }
            // SAFETY: each (row, head) task writes its own row.
            let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((i * nh + hh) * hd), hd) };
            for (j, &p) in s.iter().enumerate() {
                axpy(o, p / tot, key(j));
            }
        });
        for_chunks(&mut att, hd, false, |i, oh| {
            let ti = i / nh;
            rope_tail(oh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], true);
        });
        grouped_out(c, a, &att, nb, out);
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

#[derive(Clone, Copy)]
struct SendPtrV(*mut Vec<u32>);
unsafe impl Send for SendPtrV {}
unsafe impl Sync for SendPtrV {}
impl SendPtrV {
    #[inline(always)]
    fn get(self) -> *mut Vec<u32> {
        self.0
    }
}

impl crate::generate::TextLm for DeepSeek4Model<'_> {
    type Sess = DeepSeek4Session;
    fn new_session(&self) -> DeepSeek4Session {
        DeepSeek4Model::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut DeepSeek4Session) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &DeepSeek4Session) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut DeepSeek4Session, tokens: &[u32], _mm: &[(usize, &[f32])]) -> Vec<f32> {
        DeepSeek4Model::forward_hidden(self, sess, tokens)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        DeepSeek4Model::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut DeepSeek4Session, tokens: &[u32]) -> Vec<f32> {
        DeepSeek4Model::forward(self, sess, tokens)
    }
}

/// Runtime interventions (`crate::intervene`): the residual points around every layer on the
/// hyper-connection copies (see the module docs), and as neurons the input of `down_proj` (`w2`)
/// of each layer's shared expert (`Target::Shared`) and routed experts (`Target::Expert`, before
/// the routing weight). No layer has a dense MLP.
impl crate::intervene::Host for DeepSeek4Model<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        let m = &self.layers.get(layer)?.moe;
        match target {
            Target::Mlp => None,
            Target::Shared => Some(m.shared.inter),
            Target::Expert(e) => (e < m.experts.n_experts()).then_some(m.experts.inter),
        }
    }
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32> {
        let m = &self.layers[layer].moe;
        let (down, inter) = match target {
            Target::Expert(e) => (&m.experts.down[e], m.experts.inter),
            Target::Shared | Target::Mlp => (&m.shared.down, m.shared.inter),
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

/// The target of a DSpark draft ([`crate::dspark`]).
impl crate::dspark::Target for DeepSeek4Model<'_> {
    fn forward_feats(
        &self,
        sess: &mut DeepSeek4Session,
        tokens: &[u32],
        _mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, layers, feats)
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        DeepSeek4Model::embed_row(self, token, out)
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.logits(hidden, t)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut DeepSeek4Session, len: usize) {
        sess.truncate(len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(
            native_name("layers.3.ffn.experts.17.w1.weight").as_deref(),
            Some("layers.3.ffn.experts.17.gate_proj.weight")
        );
        assert_eq!(
            native_name("layers.3.ffn.shared_experts.w2.scale").as_deref(),
            Some("layers.3.ffn.shared_experts.down_proj.scale")
        );
        assert_eq!(native_name("layers.3.attn.wq_a.weight"), None);
        assert_eq!(
            gguf_name("blk.5.attn_output_a.weight", &[4096, 1024, 8]),
            Some(("layers.5.attn.wo_a.weight".into(), vec![8192, 4096]))
        );
        assert_eq!(
            gguf_name("blk.2.indexer.attn_q_b.weight", &[1024, 8192]).map(|x| x.0).as_deref(),
            Some("layers.2.attn.indexer.wq_b.weight")
        );
        assert_eq!(gguf_name("output_hc_fn.weight", &[16384, 4]).map(|x| x.1), Some(vec![4, 16384]));
    }

    /// A deterministic uniform generator in [-1, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
    }

    /// Writes a tiny random checkpoint in DeepSeek's naming (float32) with a hash layer, a CSA
    /// layer whose indexer chooses, a heavily compressed layer and a window-only one; returns its
    /// config.
    fn tiny_checkpoint(dir: &std::path::Path) -> DeepSeek4Config {
        let ratios = [0usize, 4, 8, 0];
        let c = DeepSeek4Config::from_json(serde_json::json!({
            "hidden_size": 16, "num_hidden_layers": ratios.len(), "num_attention_heads": 2, "head_dim": 8,
            "qk_rope_head_dim": 4, "q_lora_rank": 8, "o_groups": 2, "o_lora_rank": 4, "vocab_size": 32,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
            "routed_scaling_factor": 1.5, "swiglu_limit": 1.5, "num_hash_layers": 1, "sliding_window": 4,
            "compress_ratios": ratios, "index_n_heads": 2, "index_head_dim": 8, "index_topk": 2
        }))
        .unwrap();
        let (h, nh, hd, ql) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank);
        let (g, r, hc) = (c.o_groups, c.o_lora_rank, c.hc_mult);
        let (ne, k, mi, v) = (c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.vocab_size);
        let (ni, di) = (c.index_n_heads, c.index_head_dim);
        let mut rng = Rng(0x5eed);
        let mut tensors: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> = Vec::new();
        // `scale` times uniform noise, plus `bias`.
        let mut add = |name: String, shape: &[usize], scale: f32, bias: f32| {
            let n: usize = shape.iter().product();
            let bytes = (0..n).flat_map(|_| (bias + scale * rng.next()).to_le_bytes()).collect();
            tensors.push((name, "F32", shape.to_vec(), bytes));
        };
        let mat = |fan_in: usize, gain: f32| gain * (3.0 / fan_in as f32).sqrt();
        add("embed.weight".into(), &[v, h], 1.7, 0.0);
        add("head.weight".into(), &[v, h], mat(h, 4.0), 0.0);
        add("norm.weight".into(), &[h], 0.2, 1.0);
        add("hc_head_fn".into(), &[hc, hc * h], mat(hc * h, 4.0), 0.0);
        add("hc_head_base".into(), &[hc], 0.5, 0.0);
        add("hc_head_scale".into(), &[1], 0.25, 0.75);
        for (l, &ratio) in ratios.iter().enumerate() {
            let p = format!("layers.{l}.");
            let mix = (2 + hc) * hc;
            for part in ["attn", "ffn"] {
                add(format!("{p}{part}_norm.weight"), &[h], 0.2, 1.0);
                add(format!("{p}hc_{part}_fn"), &[mix, hc * h], mat(hc * h, 4.0), 0.0);
                add(format!("{p}hc_{part}_base"), &[mix], 0.5, 0.0);
                add(format!("{p}hc_{part}_scale"), &[3], 0.25, 0.75);
            }
            add(format!("{p}attn.wq_a.weight"), &[ql, h], mat(h, 2.0), 0.0);
            add(format!("{p}attn.q_norm.weight"), &[ql], 0.2, 1.0);
            add(format!("{p}attn.wq_b.weight"), &[nh * hd, ql], mat(ql, 2.0), 0.0);
            add(format!("{p}attn.wkv.weight"), &[hd, h], mat(h, 2.0), 0.0);
            add(format!("{p}attn.kv_norm.weight"), &[hd], 0.2, 1.0);
            add(format!("{p}attn.wo_a.weight"), &[g * r, nh * hd / g], mat(nh * hd / g, 1.0), 0.0);
            add(format!("{p}attn.wo_b.weight"), &[h, g * r], mat(g * r, 1.0), 0.0);
            add(format!("{p}attn.attn_sink"), &[nh], 1.0, 0.0);
            let mut compressor = |p: String, d: usize| {
                let w = if ratio == 4 { 2 * d } else { d };
                add(format!("{p}wkv.weight"), &[w, h], mat(h, 1.0), 0.0);
                add(format!("{p}wgate.weight"), &[w, h], mat(h, 3.0), 0.0);
                add(format!("{p}ape"), &[ratio, w], 0.5, 0.0);
                add(format!("{p}norm.weight"), &[d], 0.2, 1.0);
            };
            if ratio > 0 {
                compressor(format!("{p}attn.compressor."), hd);
            }
            if ratio == 4 {
                compressor(format!("{p}attn.indexer.compressor."), di);
                add(format!("{p}attn.indexer.wq_b.weight"), &[ni * di, ql], mat(ql, 2.0), 0.0);
                add(format!("{p}attn.indexer.weights_proj.weight"), &[ni, h], mat(h, 1.0), 0.0);
            }
            add(format!("{p}ffn.gate.weight"), &[ne, h], mat(h, 3.0), 0.0);
            if l >= c.num_hash_layers {
                add(format!("{p}ffn.gate.bias"), &[ne], 0.3, 0.0);
            }
            for e in (0..ne).map(|e| format!("experts.{e}")).chain(["shared_experts".to_string()]) {
                add(format!("{p}ffn.{e}.w1.weight"), &[mi, h], mat(h, 2.0), 0.0);
                add(format!("{p}ffn.{e}.w3.weight"), &[mi, h], mat(h, 2.0), 0.0);
                add(format!("{p}ffn.{e}.w2.weight"), &[h, mi], mat(mi, 1.0), 0.0);
            }
        }
        // The hash layer's experts of each token id (distinct).
        let tab: Vec<u8> = (0..v)
            .flat_map(|t| (0..k).map(move |j| ((t * 3 + j * (1 + t % (ne - 1))) % ne) as i32))
            .flat_map(|e| e.to_le_bytes())
            .collect();
        tensors.push(("layers.0.ffn.gate.tid2eid".into(), "I32", vec![v, k], tab));
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let entry = serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]});
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
        c
    }

    /// Runtime interventions on a tiny random model, without a reference: every kind of hook
    /// changes the logits, a neutral spec does not, the result does not depend on how the tokens
    /// are split into passes (prefill, short passes and one-token decode steps take different
    /// expert paths), an ablated direction is absent from the residual, and the calibration reads
    /// the mean of the hyper-connection copies. (Agreement with PyTorch hooks is measured by
    /// tools/ref_intervene.py, see the README.)
    #[test]
    fn interventions_on_a_tiny_model() {
        use crate::intervene::{self, Ablate, Compiled, Ctx, ExpertRef, NeuronEdit, NeuronOp, Point, Positions, Spec, Steer};
        use std::sync::Arc;
        let dir = std::env::temp_dir().join(format!("oxinfer-ds4-iv-test-{}", std::process::id()));
        let cfg = tiny_checkpoint(&dir);
        let files = ModelFiles::open(&dir, true).unwrap();
        let m = DeepSeek4Model::load(cfg, &files, Quant::Bf16).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let (h, nl, v) = (m.cfg.hidden_size, m.cfg.num_hidden_layers, m.cfg.vocab_size);
        let ids: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % v as u32).collect();
        let more = [5u32, 17, 2];
        let resolve = |s: &str| -> Result<std::path::PathBuf> { Ok(s.into()) };
        let compile = |spec: &Spec| {
            let ctx = Ctx {
                resolve: &resolve,
                calibration: None,
                saes: None,
            };
            Arc::new(Compiled::new(spec, &m, &ctx).unwrap())
        };
        // The logits after the prompt (in passes of `chunk`) and after each of `more`.
        let run = |iv: Option<Arc<Compiled>>, chunk: usize| -> Vec<f32> {
            let mut sess = m.new_session();
            sess.iv.iv = iv;
            let hs = crate::generate::prefill(&m, &mut sess, &ids, chunk, &[]);
            let mut out = m.logits(&hs[hs.len() - h..], 1);
            for &t in &more {
                out.extend(m.forward(&mut sess, &[t]));
            }
            out
        };
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let vecf = |seed: usize| -> Vec<f32> { (0..h).map(|i| ((i * 5 + seed * 3) % 7) as f32 - 3.0).collect() };
        let neuron = |layer: usize, neuron: usize, expert: Option<usize>, op: NeuronOp, value: f32, positions: Positions| NeuronEdit {
            layer,
            neuron,
            expert: Some(expert.map_or(ExpertRef::Name("shared".into()), ExpertRef::Id)),
            op: Some(op),
            value: Some(value),
            positions,
            ..Default::default()
        };
        let mut neurons = vec![
            neuron(0, 1, None, NeuronOp::Add, 1.5, Positions::All),
            neuron(2, 3, None, NeuronOp::Scale, -2.0, Positions::Generated),
        ];
        // A neuron of every routed expert of a hash layer and of a scored one: whatever the routes.
        for e in 0..m.cfg.n_routed_experts {
            neurons.push(neuron(0, 2, Some(e), NeuronOp::Add, 0.8, Positions::All));
            neurons.push(neuron(3, 5, Some(e), NeuronOp::Set, 0.6, Positions::Prompt));
            neurons.push(neuron(1, 4, Some(e), NeuronOp::Scale, 3.0, Positions::Generated));
        }
        let steer = |layer: usize, point: Point, positions: Positions, alpha: f32| Steer {
            vector: Some(vecf(layer)),
            layer: Some(layer),
            alpha,
            point,
            positions,
            ..Default::default()
        };
        let steers = vec![
            steer(0, Point::Pre, Positions::All, 0.3),
            steer(1, Point::Mid, Positions::Generated, 0.5),
            steer(2, Point::Post, Positions::Prompt, -0.4),
        ];
        let r = vecf(11);
        let ablate = vec![Ablate {
            vector: Some(r.clone()),
            ..Default::default()
        }];
        let only_neurons = Spec {
            neurons: neurons.clone(),
            ..Default::default()
        };
        let only_steer = Spec {
            steer: steers.clone(),
            ..Default::default()
        };
        let only_ablate = Spec {
            ablate: ablate.clone(),
            ..Default::default()
        };
        let all = Spec {
            neurons,
            steer: steers,
            ablate,
            ..Default::default()
        };

        let base = run(None, 40);
        let scale = base.iter().fold(0f32, |a, b| a.max(b.abs()));
        let tol = 1e-3 * scale;
        // A neutral spec (hooks installed, nothing changed) gives the plain logits.
        let neutral = Spec {
            neurons: vec![
                neuron(1, 0, None, NeuronOp::Scale, 1.0, Positions::All),
                neuron(1, 0, Some(2), NeuronOp::Scale, 1.0, Positions::All),
            ],
            steer: vec![steer(2, Point::Mid, Positions::All, 0.0)],
            ..Default::default()
        };
        let d = diff(&run(Some(compile(&neutral)), 40), &base);
        assert!(d <= tol, "neutral spec moved the logits by {d}");
        // Every kind of hook does something, and the same thing however the passes are cut.
        for (what, spec) in [
            ("neurons", &only_neurons),
            ("steer", &only_steer),
            ("ablate", &only_ablate),
            ("all", &all),
        ] {
            let whole = run(Some(compile(spec)), 40);
            assert!(whole.iter().all(|x| x.is_finite()));
            let moved = diff(&whole, &base);
            assert!(moved > 50.0 * tol, "{what}: the logits moved by {moved} only");
            for chunk in [7, 1] {
                let d = diff(&run(Some(compile(spec)), chunk), &whole);
                assert!(
                    d <= tol,
                    "{what}: passes of {chunk} differ from one pass by {d} (logits up to {scale})"
                );
            }
        }
        // The ablated direction is absent from the residual (the mean of the copies, as a DSpark
        // draft reads it) entering every layer and leaving the last.
        let layers: Vec<usize> = (0..=nl).collect();
        let feats_of = |iv: Option<Arc<Compiled>>| {
            let mut sess = m.new_session();
            sess.iv.iv = iv;
            let mut feats = Vec::new();
            m.forward_capture(&mut sess, &ids, &layers, &mut feats);
            feats
        };
        let rn = dot(&r, &r).sqrt();
        let along = |f: &[f32]| f.chunks(h).map(|x| (dot(x, &r) / rn).abs()).fold(0f32, f32::max);
        let plain = feats_of(None);
        let ablated = feats_of(Some(compile(&only_ablate)));
        assert!(
            along(&plain) > 0.1 && along(&ablated) < 1e-4,
            "{} {}",
            along(&plain),
            along(&ablated)
        );
        // Calibration (the `resid_median` unit) and capture read that same residual.
        let cal = intervene::calibrate(&m, &ids, 16).unwrap();
        for l in 0..nl {
            let norms: Vec<f32> = (1..ids.len())
                .map(|t| {
                    let x = &plain[(t * (nl + 1) + l + 1) * h..(t * (nl + 1) + l + 2) * h];
                    dot(x, x).sqrt()
                })
                .collect();
            let want = intervene::median(&norms);
            assert!(
                (cal.resid_median[l] - want).abs() <= 1e-3 * want,
                "layer {l}: {} vs {want}",
                cal.resid_median[l]
            );
        }
        let cap = intervene::capture_layers(&m, std::slice::from_ref(&ids), intervene::Pool::Last, 16).unwrap();
        let t = ids.len() - 1;
        for l in 0..nl {
            let x = &plain[(t * (nl + 1) + l + 1) * h..(t * (nl + 1) + l + 2) * h];
            assert!(diff(&cap[0][&l], x) <= 1e-3 * dot(x, x).sqrt(), "layer {l}");
        }
    }

    /// The Sinkhorn mixing matrix is (nearly) doubly stochastic.
    #[test]
    fn sinkhorn_is_doubly_stochastic() {
        let c = DeepSeek4Config::from_json(serde_json::json!({
            "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1, "head_dim": 8,
            "qk_rope_head_dim": 4, "q_lora_rank": 4, "o_groups": 1, "o_lora_rank": 4, "vocab_size": 4,
            "n_routed_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 4,
            "compress_ratios": [0], "index_n_heads": 1, "index_head_dim": 8, "index_topk": 1
        }))
        .unwrap();
        let n = c.hc_mult;
        let w = Hc {
            f: (0..(2 + n) * n * n * 8).map(|i| ((i * 37 % 17) as f32 - 8.0) * 0.1).collect(),
            base: (0..(2 + n) * n).map(|i| (i as f32 * 0.3).sin()).collect(),
            scale: vec![0.7, 1.1, 1.3],
        };
        let x: Vec<f32> = (0..n * 8).map(|i| (i as f32 * 0.7).cos()).collect();
        let (mut pre, mut post, mut comb) = (vec![0f32; n], vec![0f32; n], vec![0f32; n * n]);
        hc_mix(&c, &w, &x, &mut pre, &mut post, &mut comb);
        for j in 0..n {
            let row: f32 = comb[j * n..(j + 1) * n].iter().sum();
            let col: f32 = (0..n).map(|k| comb[k * n + j]).sum();
            assert!((row - 1.0).abs() < 1e-3 && (col - 1.0).abs() < 1e-5, "{row} {col}");
        }
        assert!(pre.iter().all(|&p| p > 0.0 && p < 1.01) && post.iter().all(|&p| p > 0.0 && p < 2.0));
    }
}
