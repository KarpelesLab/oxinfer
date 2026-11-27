//! Single-token decoding on an accelerator: a model describes one token's forward pass as a list
//! of ops over activation buffers ([`Graph`], built once per model), the backend turns it into a
//! [`DecodePlan`] that runs a whole token in one submission (one command buffer and one wait on
//! Metal) instead of one per matmul. The host only looks up the input embedding and samples from
//! the logits.
//!
//! The KV caches stay authoritative on the host ([`crate::model::LayerCache`]): the backend
//! keeps a device copy per session ([`Slot`]), appends what the host added since the last step
//! (prefill chunks, speculative verification passes run on the CPU), forgets what a rewind
//! dropped, and pushes each step's new keys and values back. Recurrent states (Gated DeltaNet)
//! are page-aligned host memory the device reads and writes in place.

use super::{DeviceWeight, WeightRef};
use crate::model::LayerCache;
use std::any::Any;
use std::borrow::Cow;

/// An activation buffer of the graph.
pub type Buf = usize;

/// A position inside an activation buffer (in floats).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub buf: Buf,
    pub off: usize,
}

impl View {
    pub fn at(self, off: usize) -> View {
        View {
            buf: self.buf,
            off: self.off + off,
        }
    }
}

/// How a matvec writes its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinMode {
    /// `y = W x`
    Store,
    /// `y += W x` (residual)
    Add,
    /// `y = silu(W x) * (W2 x)`
    SiluPair,
    /// `y = gelu_tanh(W x) * (W2 x)`
    GeluPair,
    /// `y += sigmoid(g) * W x`, with `g` the scalar at this view (a gated shared expert).
    AddGated(View),
    /// `y = relu(W x)^2` (Nemotron-H's squared-ReLU MLPs).
    Relu2,
}

pub enum Op<'a> {
    /// RMSNorm of `rows` rows of `n` values (row strides in floats) of `x * pre`: `out = x' /
    /// rms(x') * w`, or `out += ...` with `add`. `w` holds `n` weights shared by the rows, or
    /// `rows * n`, each row its own (Nemotron-H's grouped norm; one-token plans only).
    Norm {
        x: View,
        out: View,
        w: Option<Cow<'a, [f32]>>,
        pre: f32,
        n: usize,
        rows: usize,
        x_stride: usize,
        out_stride: usize,
        eps: f32,
        add: bool,
    },
    /// `y[n_out] = W x[n_in]` (see [`LinMode`]; `w2` is the second matrix of a pair).
    Linear {
        w: WeightRef<'a>,
        dev: &'a DeviceWeight,
        w2: Option<(WeightRef<'a>, &'a DeviceWeight)>,
        n_out: usize,
        n_in: usize,
        x: View,
        y: View,
        mode: LinMode,
    },
    /// Attention inputs of the token: in `qkv`, query head `h` starts at `h * q_stride`, key
    /// head `h` at `k_off + h * hd`, value head `h` at `v_off + h * hd` (or the raw key head with
    /// `v_from_k`). Queries (optionally RMS-normed per head with `q_norm`) get RoPE (table `rope`
    /// of [`Graph::rope_halves`]) on their first `2 * half_rot` dimensions and go to `q_out`
    /// (`[nh, hd]`); keys likewise into KV cache `kv` at the token's position; values (RMS-normed
    /// without weight with `v_norm`) go there too.
    QkRope {
        qkv: View,
        q_out: View,
        kv: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        half_rot: usize,
        q_stride: usize,
        k_off: usize,
        v_off: usize,
        interleaved: bool,
        q_norm: Option<&'a [f32]>,
        k_norm: Option<&'a [f32]>,
        eps: f32,
        rope: usize,
        v_norm: bool,
        v_from_k: bool,
    },
    /// Sandwich-norm residual step of one row of `n`: `x = (x + rmsnorm(y) * wy) * scale`, then
    /// with `next`, `out = rmsnorm(x) * w` (the following norm, fused).
    AddNorm {
        y: View,
        x: View,
        wy: Cow<'a, [f32]>,
        scale: f32,
        next: Option<(View, &'a [f32])>,
        n: usize,
        eps: f32,
    },
    /// Elementwise over `n` values into `out` (see [`Ew`]).
    Ew {
        kind: Ew,
        a: View,
        b: Option<View>,
        out: View,
        n: usize,
    },
    /// Attention of the token's queries `q` (`[nh, hd]`) over KV cache `kv` (positions up to and
    /// including the token's; the last `window` only, if set) into `out` (`[nh, hd]`), scores
    /// scaled by `scale`. `gate`: `out *= sigmoid(gate)`, with head `h` of the gate at
    /// `gate.off + h * stride`.
    Attend {
        q: View,
        out: View,
        kv: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        window: Option<usize>,
        gate: Option<(View, usize)>,
        /// Multi-token plans of a DSpark draft's block: `(causal, window)`: every token of the step
        /// sees the cached positions before the step's first (the last `window`, if not 0) and
        /// the step's tokens (those up to its own when `causal`), instead of the positions up to
        /// its own.
        block: Option<(bool, usize)>,
    },
    /// Causal depthwise conv of kernel `kc` over `dim` channels with SiLU: `out = silu(sum of
    /// w[ch][j] * input j)`, inputs being the `kc - 1` earlier ones in recurrent state `state`
    /// (`[kc - 1, dim]`, oldest first) and `x`; the state then shifts in `x`. With `bias`,
    /// `out = silu(bias + sum ...)` (Mamba-2's convolution; one-token plans only).
    GdnConv {
        x: View,
        out: View,
        state: usize,
        w: &'a [f32],
        dim: usize,
        kc: usize,
        bias: Option<&'a [f32]>,
    },
    /// Mamba-2's selective state update of one token (Nemotron-H, see `crate::nemotron`).
    Mamba2(Mamba2<'a>),
    GdnRule(GdnRule<'a>),
    /// MoE routing of one token: router logits `router . x` (`[n_exp, h]` rows; with `gate`, one
    /// more logit `gate . x` after them, for a gated shared expert) into `logits`, then the top
    /// `k` experts (ties to the lower index) and their softmax weights into `sel`: `k` ids (u32
    /// bits) then `k` weights.
    Route {
        x: View,
        router: &'a [f32],
        gate: Option<&'a [f32]>,
        n_exp: usize,
        h: usize,
        k: usize,
        logits: View,
        sel: View,
        /// Per-expert factors the chosen weights are multiplied by after the softmax (Gemma).
        expert_scale: Option<&'a [f32]>,
        /// Sigmoid routing (Nemotron-H): the experts of the `k` largest `sigmoid(logit) + bias`
        /// (ties to the lower index), weighted by their `sigmoid(logit)` renormalized over the
        /// `k`, times the scale, instead of the softmax. `(bias, scale)`.
        sigmoid: Option<(&'a [f32], f32)>,
    },
    /// The routed experts chosen in `sel` (see [`Op::Route`]): `act[s] = silu(gate_e x) *
    /// (up_e x)` for each slot, then `y += sum of weight_s * down_e act[s]`.
    Experts(Experts<'a>),
    /// Residual hooks of `crate::intervene` on one row `x` of `n`: with `n_add` coefficients
    /// `c` then `n_items` entries of [`crate::intervene::GPU_ENTRY`] floats at `hot`, and the add
    /// vectors, encoder rows and decoder rows at `cold`: `x += sum c_a add_a + sum (a'_j - a_j)
    /// dec_j` where `a_j = act(enc_j . x + bias_j)` (see [`crate::intervene::GpuPoint`]).
    Steer {
        x: View,
        n: usize,
        hot: View,
        cold: View,
        n_add: usize,
        n_items: usize,
    },
    /// Neuron edits on `act` (`n` entries at `hot`, see [`crate::intervene::GpuNeurons`]).
    NeuronEdit {
        act: View,
        hot: View,
        n: usize,
    },
    /// `out = act(g) * u` over `n` values (SiLU, or GELU with the tanh approximation): the
    /// gate / up pair when the two matrices cannot share one matvec (different formats).
    Glu {
        g: View,
        u: View,
        out: View,
        n: usize,
        gelu: bool,
    },
    /// Hyper-connection read (Qwen4-Exp, see `crate::qwen4exp`): `out[i] = (1 / hc) sum_s
    /// sigmoid(gate[s n + i]) xn[s n + i]` over the `hc` streams of `n`.
    HcPre {
        gate: View,
        xn: View,
        out: View,
        n: usize,
        hc: usize,
    },
    /// Hyper-connection write: `x[s n + i] += y[i] * 2 sigmoid(inj[s] / hc)`.
    HcPost {
        x: View,
        y: View,
        inj: View,
        n: usize,
        hc: usize,
    },
    /// The n-gram embedding gate of Qwen4-Exp: per stream `s`, `a = k_s . q_s / sqrt(n)`, then
    /// `out[s n + i] = sigmoid(sign(a) sqrt(max(|a|, 1e-6))) * v[i]`.
    PleGate {
        k: View,
        q: View,
        v: View,
        out: View,
        n: usize,
        hc: usize,
    },
    /// QSA indexer keys (Qwen4-Exp, see `crate::qwen4exp`): the token's raw key `k` (`d` floats)
    /// goes to key row `pos` of cache `kv` (`(1, d)`); value row `pos` gets, when the token
    /// completes a block of `ratio` (`(pos + 1) % ratio == 0`), the block's pooled key: the
    /// mean of its raw keys, RMS-normed with `norm`, RoPE'd (first `2 * half_rot` dimensions,
    /// table `rope` of [`Graph::rope_halves`], which holds the angles of the block's first
    /// position), else zeros. The ops follow the CPU path's order (RMSNorm factors may differ
    /// in the last bit).
    QsaKey {
        k: View,
        kv: usize,
        d: usize,
        ratio: usize,
        norm: &'a [f32],
        eps: f32,
        half_rot: usize,
        rope: usize,
    },
    /// QSA block selection: past `top` complete blocks, the raw queries `q` (`[nh, d]`) are
    /// RMS-normed (`norm`) and RoPE'd (table `rope`), every complete block scored against the
    /// pooled keys of cache `kv` (`scale * sum_h relu(q_h . k)`, into `scores`, one float per
    /// block), and the positions attended written to `sel` (u32 bits): the `top` best blocks
    /// (ties to the lower block) in ascending order, `ratio` positions each, then the
    /// incomplete tail up to the token. Nothing runs while every block fits the budget.
    QsaSelect {
        q: View,
        kv: usize,
        nh: usize,
        d: usize,
        ratio: usize,
        top: usize,
        norm: &'a [f32],
        eps: f32,
        scale: f32,
        half_rot: usize,
        rope: usize,
        scores: View,
        /// Floats of `scores` (most blocks scored).
        cap: usize,
        sel: View,
    },
    /// The first half of the runtime LoRA updates of a linear op (see `crate::lora`): `u = A x`
    /// (`a` `[r, n_in]`) as partial sums over chunks of [`LORA_CHUNK`] inputs, `u[c * r + k]`
    /// the sum of chunk `c` for component `k` ([`lora_chunks`] chunks). It only reads `x`, so it
    /// runs beside the matvec.
    LoraDown {
        a: Cow<'a, [f32]>,
        x: View,
        u: View,
        r: usize,
        n_in: usize,
    },
    /// The second half, after the matvec: `y += g * sum_k u_k S[scale[k]] B^T[k]` with `u_k` the
    /// sum of the chunks of [`Op::LoraDown`] in order, `bt` `[r, n_out]`, `S` the strengths of
    /// [`Graph::strengths`] (written before every step) and `g` the sigmoid of the scalar at
    /// `gate` (a gated shared expert, [`LinMode::AddGated`]) or 1.
    LoraUp {
        bt: Cow<'a, [f32]>,
        scale: Vec<u32>,
        u: View,
        y: View,
        gate: Option<View>,
        r: usize,
        n_in: usize,
        n_out: usize,
    },
    /// [`Op::Attend`] under QSA: over the positions [`Op::QsaSelect`] wrote to `sel`, or all of
    /// them while every block fits the budget (`top` blocks of `ratio`).
    AttendSel {
        q: View,
        out: View,
        kv: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        gate: Option<(View, usize)>,
        sel: View,
        ratio: usize,
        top: usize,
    },
    /// The input transform of a Hadamard-folded weight ([`crate::hadamard::Fold::apply`]) of a
    /// row of `n` values: `out = H (D P x)` (P the value-head permutation, D the signs, H the
    /// blocked Walsh-Hadamard rotation), for the [`Op::Linear`]s that read `out`.
    Fold {
        x: View,
        out: View,
        n: usize,
        fold: &'a crate::hadamard::Fold,
    },
}

/// The largest rank (all chained adapters of a layer together) of [`Op::LoraUp`].
pub const LORA_MAX_RANK: usize = 256;

/// Inputs per partial sum of [`Op::LoraDown`].
pub const LORA_CHUNK: usize = 1024;

/// Partial sums per component of [`Op::LoraDown`] over `n_in` inputs.
pub fn lora_chunks(n_in: usize) -> usize {
    n_in.div_ceil(LORA_CHUNK)
}

/// Low-rank updates on the device ([`Op::LoraDown`], [`Op::LoraUp`]): `a` is `[r, n_in]`, `bt`
/// (B transposed) `[r, n_out]`, component `k` taking strength `scale[k]` of
/// [`Graph::strengths`]. Chained adapters are concatenated along the rank.
pub struct LoraSide<'a> {
    pub a: Cow<'a, [f32]>,
    pub bt: Cow<'a, [f32]>,
    pub scale: Vec<u32>,
    pub r: usize,
    pub n_in: usize,
    pub n_out: usize,
}

/// The LoRA updates of every expert of an MoE layer ([`Experts::lora`]), padded to one rank `r`
/// (zero components where an expert has fewer): `a` `[n_exp, r, n_in]`, `bt` `[n_exp, r,
/// n_out]`, `scale` `[n_exp, r]` (strength ids, see [`LoraSide`]).
pub struct LoraStack {
    pub a: Vec<f32>,
    pub bt: Vec<f32>,
    pub scale: Vec<u32>,
    pub r: usize,
    pub n_in: usize,
    pub n_out: usize,
}

/// Runtime LoRA updates of routed experts: `gate_up` on the fused gate then up rows (`n_out = 2
/// * inter`, either storage), `down` on the down projections. With `gate_up` the pair matvecs
/// write the raw gate / up outputs to `raw` (`[k, 2 * inter]`), the updates are added and the
/// activation taken from there; `down`'s update is added to `y` with each slot's routing weight.
/// `u` is `[k, max rank]` scratch.
pub struct ExpertLora {
    pub gate_up: Option<LoraStack>,
    pub down: Option<LoraStack>,
    pub raw: View,
    pub u: View,
}

/// Elementwise ops ([`Op::Ew`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ew {
    /// `out = a * s`
    Scale(f32),
    /// `out = tanh(a / cap) * cap`
    Softcap(f32),
    /// `out = (a * sa + b * sb) * s`
    Combine { sa: f32, sb: f32, s: f32 },
    /// `out = silu(a * s)`
    SiluScale(f32),
}

impl<'a> Op<'a> {
    /// RMSNorm of one row of `n` values with weights `w`.
    pub fn norm(x: View, out: View, w: &'a [f32], n: usize, eps: f32) -> Self {
        Op::Norm {
            x,
            out,
            w: Some(Cow::Borrowed(w)),
            pre: 1.0,
            n,
            rows: 1,
            x_stride: n,
            out_stride: n,
            eps,
            add: false,
        }
    }
}

/// The routed experts of an MoE layer: per expert, `gate[e]` is `[inter, h]` (with `up[e]`
/// `[inter, h]`), or without `up` the fused `[2 * inter, h]` gate then up rows; `down[e]` is
/// `[h, inter]`. One storage format per projection.
pub struct Experts<'a> {
    pub gate: Vec<WeightRef<'a>>,
    pub up: Option<Vec<WeightRef<'a>>>,
    pub down: Vec<WeightRef<'a>>,
    pub h: usize,
    pub inter: usize,
    pub k: usize,
    pub x: View,
    pub sel: View,
    /// `[k, inter]` scratch.
    pub act: View,
    pub y: View,
    /// GELU-tanh experts (else SiLU).
    pub gelu: bool,
    /// Ungated squared-ReLU experts (Nemotron-H): `act[s] = relu(up_e x)^2`, the up matrices
    /// `[inter, h]` in `gate` and `up` None (instead of a fused gate / up).
    pub relu2: bool,
    /// `y +=` the weighted sum (else `y =`).
    pub add: bool,
    /// Neuron edits of the routed experts' activations (entries with expert ids, see
    /// [`crate::intervene::GpuNeurons`]): `(hot, n)`.
    pub edits: Option<(View, usize)>,
    /// Runtime LoRA updates of the experts.
    pub lora: Option<Box<ExpertLora>>,
}

/// Whether two weights can form one [`LinMode::SiluPair`] / [`LinMode::GeluPair`] matvec (same
/// storage format); otherwise use two matvecs and [`Op::Glu`].
pub fn pairable(a: &WeightRef, b: &WeightRef) -> bool {
    match (a, b) {
        (WeightRef::Bf16(_), WeightRef::Bf16(_)) | (WeightRef::Q8(_), WeightRef::Q8(_)) => true,
        (WeightRef::Q4(x), WeightRef::Q4(y)) => x.offsets.as_ref().map(|o| o.group) == y.offsets.as_ref().map(|o| o.group),
        (WeightRef::Gguf(x), WeightRef::Gguf(y)) => x.kind == y.kind,
        _ => false,
    }
}

/// The gated delta rule of Gated DeltaNet (Qwen3.5 / 3.8 linear attention, see `crate::qwen`)
/// for one token over all value heads, then the gated RMSNorm.
/// Mamba-2's state update of one token ([`Op::Mamba2`]): per head `h` (group `g = h / (nh /
/// ng)`), `dt = softplus(dt_raw[h] + dt_bias[h])` clamped to `[dt_min, dt_max]`, `S_h <-
/// exp(dt a[h]) S_h + dt x_h B_g^T` (`S_h` `[hd, ns]`), then `out_h = (S_h C_g + d[h] x_h) *
/// silu(z_h)`, the gated output before its grouped RMSNorm.
pub struct Mamba2<'a> {
    /// The convolution's output `[x (nh * hd) | B (ng * ns) | C (ng * ns)]`.
    pub xbc: View,
    /// The gate `z` (`nh * hd`).
    pub z: View,
    /// The raw time steps (`nh`).
    pub dt: View,
    /// `nh * hd`.
    pub out: View,
    /// Recurrent state `[nh, hd, ns]`.
    pub state: usize,
    pub nh: usize,
    pub hd: usize,
    pub ng: usize,
    pub ns: usize,
    /// `A = -exp(A_log)`, `D` and the time step bias (`nh` each).
    pub a: &'a [f32],
    pub d: &'a [f32],
    pub dt_bias: &'a [f32],
    pub dt_min: f32,
    pub dt_max: f32,
}

pub struct GdnRule<'a> {
    /// Conv outputs `[q | k | v]` (`2 * nk * hdk + nv * hdv`), gate inputs `z` (`nv * hdv`),
    /// `b` and `a` (`nv` each).
    pub c: View,
    pub z: View,
    pub b: View,
    pub a: View,
    /// `[nv * hdv]`: the rule's output after the gated RMSNorm.
    pub out: View,
    /// Recurrent state `[nv, hdk, hdv]`.
    pub state: usize,
    pub nk: usize,
    pub nv: usize,
    pub hdk: usize,
    pub hdv: usize,
    /// Value head `h` reads key head `h % nk` (GGUF) rather than `h / (nv / nk)`.
    pub tiled: bool,
    /// The output gate is `sigmoid(z)` (Qwen4-Exp) rather than `silu(z)`.
    pub sigmoid_gate: bool,
    pub decay_coef: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm_w: &'a [f32],
    pub eps: f32,
}

/// One token's forward pass.
pub struct Graph<'a> {
    /// Activation buffer sizes (floats).
    pub bufs: Vec<usize>,
    pub ops: Vec<Op<'a>>,
    /// Where the input embedding goes (`hidden` floats).
    pub input: View,
    /// The final hidden state read back after the pass.
    pub hidden: View,
    pub n_hidden: usize,
    /// Floats of the input per token (0: `n_hidden`).
    pub n_input: usize,
    /// Ops from this index on compute the logits (skipped when the caller wants the hidden
    /// state only).
    pub head_from: usize,
    /// The logits (`n` floats) the head ops produce.
    pub logits: Option<(View, usize)>,
    /// `(n_kv, hd)` of each KV cache the ops use.
    pub kv: Vec<(usize, usize)>,
    /// cos / sin entries per position of each RoPE table `QkRope` uses (half the rotated
    /// dimensions).
    pub rope_halves: Vec<usize>,
    /// A second host input (e.g. per-layer embeddings), `n` floats, if any.
    pub input2: Option<(View, usize)>,
    /// Sizes (floats) of the recurrent states the ops use (host memory, see [`Step::states`]).
    pub states: Vec<usize>,
    /// Intervention parameters (`crate::intervene`): the per-step part (`hot`, floats) and the
    /// part refreshed only when its key changes (`cold`), see [`Step::params`].
    pub params: Option<((View, usize), (View, usize))>,
    /// A buffer the ops copy residual streams into (`n` floats per token), read back after the
    /// pass ([`Step::capture`]; DSpark target features).
    pub capture: Option<(View, usize)>,
    /// Weight streaming: `(first op, unit)` where each streaming unit of the model starts (in
    /// op order; the ops before the first belong to none), and the streamer that brings the
    /// units in. Linear ops of units that are not resident carry [`streamed_weight`] as their
    /// device state: the plan wraps their memory per step, after the unit arrived, and drops
    /// the wrap once the device has read it, so the pages can leave the cache.
    pub units: Vec<(usize, usize)>,
    pub gate: Option<std::sync::Arc<dyn UnitGate>>,
    /// The strengths of the LoRA updates ([`Op::LoraUp`], [`Experts::lora`]), read before every
    /// step.
    pub strengths: Vec<crate::lora::Strength>,
    /// Greedy chains ([`DecodePlan::greedy`]): the token embedding table the input row is read
    /// from (`n_input` columns) and the factor it is multiplied by, as the host computes the
    /// input (`Table::row`, then `* scale`).
    pub embed: Option<(&'a crate::model::Table<'a>, f32)>,
    /// The table of the second input ([`Graph::input2`]), when it is a token's row.
    pub embed2: Option<&'a crate::model::Table<'a>>,
    /// Identifies the model whose plans share slots (device copies of the caches): plans of
    /// graphs with the same caches, states and key share them. None: the graph's first weight
    /// (graphs of part of a model's layers set the model's key).
    pub layout_key: Option<usize>,
}

/// The weight streamer as a decode plan sees it (see [`Graph::units`]).
pub trait UnitGate: Send + Sync {
    /// Waits until unit `u` is in memory; it stays in use until [`Self::end`].
    fn begin(&self, u: usize);
    fn end(&self, u: usize);
    /// Whether unit `u` stays in memory (its weights are wrapped once, as without streaming).
    fn resident(&self, u: usize) -> bool;
    /// The memory of unit `u`: page-aligned `(address, length)` spans.
    fn spans(&self, u: usize) -> Vec<(usize, usize)>;
    /// On-demand parts of unit `u` (its routed experts, read once routing chose them).
    fn part_count(&self, _u: usize) -> usize {
        0
    }
    /// The memory of part `p` of unit `u`, as page-aligned spans.
    fn part_spans(&self, _u: usize, _p: usize) -> Vec<(usize, usize)> {
        Vec::new()
    }
    /// Inside `begin(u)`: reads parts `ids` of unit `u` and waits for them.
    fn need_parts(&self, _u: usize, _ids: &[usize]) {}
    /// Whether part `p` of unit `u` is still wanted in memory (in use or cached): a backend
    /// keeping it mapped between steps lets go of it otherwise, so that its pages can be dropped.
    fn part_cached(&self, _u: usize, _p: usize) -> bool {
        false
    }
    /// The device reads the streamed units (a plan was built over them): the streamer may leave
    /// mapping their pages to it.
    fn device_reads(&self, _on: bool) {}
    /// Backend state kept as long as the streamed memory (wraps of cached parts).
    fn device_state(&self) -> Option<&std::sync::Mutex<Option<Box<dyn std::any::Any + Send + Sync>>>> {
        None
    }
}

impl UnitGate for crate::stream::StreamHook {
    fn begin(&self, u: usize) {
        crate::stream::StreamHook::begin(self, u)
    }
    fn end(&self, u: usize) {
        self.leave(u)
    }
    fn resident(&self, u: usize) -> bool {
        self.is_resident(u)
    }
    fn spans(&self, u: usize) -> Vec<(usize, usize)> {
        self.unit_spans(u)
    }
    fn part_count(&self, u: usize) -> usize {
        crate::stream::StreamHook::part_count(self, u)
    }
    fn part_spans(&self, u: usize, p: usize) -> Vec<(usize, usize)> {
        crate::stream::StreamHook::part_spans(self, u, p)
    }
    fn need_parts(&self, u: usize, ids: &[usize]) {
        crate::stream::StreamHook::need_parts(self, u, ids)
    }
    fn part_cached(&self, u: usize, p: usize) -> bool {
        crate::stream::StreamHook::part_cached(self, u, p)
    }
    fn device_reads(&self, on: bool) {
        self.set_device_reads(on)
    }
    fn device_state(&self) -> Option<&std::sync::Mutex<Option<Box<dyn std::any::Any + Send + Sync>>>> {
        Some(crate::stream::StreamHook::device_state(self))
    }
}

/// The device state of a weight that streams ([`Graph::units`]): a marker, nothing is wrapped.
pub struct StreamedWeight;

/// [`StreamedWeight`] as a [`DeviceWeight`].
pub fn streamed_weight() -> &'static DeviceWeight {
    static W: std::sync::OnceLock<DeviceWeight> = std::sync::OnceLock::new();
    W.get_or_init(|| Box::new(StreamedWeight))
}

/// Whether `dev` is [`streamed_weight`].
pub fn is_streamed(dev: &DeviceWeight) -> bool {
    dev.is::<StreamedWeight>()
}

impl<'a> Graph<'a> {
    pub fn new() -> Self {
        Graph {
            bufs: Vec::new(),
            ops: Vec::new(),
            input: View { buf: 0, off: 0 },
            hidden: View { buf: 0, off: 0 },
            n_hidden: 0,
            n_input: 0,
            head_from: 0,
            logits: None,
            kv: Vec::new(),
            rope_halves: Vec::new(),
            input2: None,
            states: Vec::new(),
            params: None,
            capture: None,
            units: Vec::new(),
            gate: None,
            strengths: Vec::new(),
            embed: None,
            embed2: None,
            layout_key: None,
        }
    }

    /// The id of strength `s` in [`Graph::strengths`] (added if new).
    pub fn strength_id(&mut self, s: crate::lora::Strength) -> u32 {
        match self.strengths.iter().position(|o| o.same(&s)) {
            Some(i) => i as u32,
            None => {
                self.strengths.push(s);
                (self.strengths.len() - 1) as u32
            }
        }
    }

    /// The device form of `d` and the updates chained after it (None when they have no
    /// components).
    pub fn lora_side(&mut self, d: &'a crate::lora::LoraDelta) -> Option<LoraSide<'a>> {
        let (n_out, n_in) = d.shape();
        let links: Vec<&crate::lora::LoraDelta> = d.links().filter(|l| l.factors().2 > 0).collect();
        let mut scale = Vec::new();
        for l in &links {
            let id = self.strength_id(l.strength_handle());
            scale.extend(std::iter::repeat_n(id, l.factors().2));
        }
        let (a, bt): (Cow<'a, [f32]>, Cow<'a, [f32]>) = match links[..] {
            [] => return None,
            [l] => (Cow::Borrowed(l.factors().0), Cow::Borrowed(l.factors().1)),
            _ => (
                Cow::Owned(links.iter().flat_map(|l| l.factors().0).copied().collect()),
                Cow::Owned(links.iter().flat_map(|l| l.factors().1).copied().collect()),
            ),
        };
        Some(LoraSide {
            a,
            bt,
            r: scale.len(),
            scale,
            n_in,
            n_out,
        })
    }

    /// The runtime LoRA updates of the linear op of `l` (input `x`, output `y`, `gate` for
    /// [`LinMode::AddGated`]): [`Op::LoraDown`], to go beside the matvec, and [`Op::LoraUp`],
    /// after it; `Some(None)` without updates, None when the device cannot take them (a rank
    /// above [`LORA_MAX_RANK`]).
    #[allow(clippy::type_complexity)]
    pub fn lora_ops(&mut self, l: &'a crate::model::Lin, x: View, y: View, gate: Option<View>) -> Option<Option<(Op<'a>, Op<'a>)>> {
        let Some(LoraSide {
            a,
            bt,
            scale,
            r,
            n_in,
            n_out,
        }) = l.lora_delta().and_then(|d| self.lora_side(d))
        else {
            return Some(None);
        };
        if r > LORA_MAX_RANK {
            return None;
        }
        let u = self.buf(r * lora_chunks(n_in));
        Some(Some((
            Op::LoraDown { a, x, u, r, n_in },
            Op::LoraUp {
                bt,
                scale,
                u,
                y,
                gate,
                r,
                n_in,
                n_out,
            },
        )))
    }

    /// The runtime LoRA updates of routed experts `ex` ([`Experts::lora`]; `Some(None)` without
    /// any), or None when the device cannot take them (interleaved gate / up rows).
    pub(crate) fn expert_lora(&mut self, ex: &'a crate::moe::Experts, k: usize) -> Option<Option<Box<ExpertLora>>> {
        use crate::moe::GateUp;
        let (ne, mi) = (ex.n_experts(), ex.inter);
        let mut gu: Vec<Vec<(&crate::lora::LoraDelta, usize)>> = Vec::with_capacity(ne);
        let mut dn: Vec<Vec<(&crate::lora::LoraDelta, usize)>> = Vec::with_capacity(ne);
        let chain = |l: &'a crate::model::Lin, row0: usize| -> Vec<(&'a crate::lora::LoraDelta, usize)> {
            l.lora_delta()
                .map(|d| d.links().filter(|l| l.factors().2 > 0).map(|l| (l, row0)).collect())
                .unwrap_or_default()
        };
        for (g, d) in ex.gate_up.iter().zip(&ex.down) {
            gu.push(match g {
                GateUp::Fused(l) => chain(l, 0),
                GateUp::Split(a, b) => chain(a, 0).into_iter().chain(chain(b, mi)).collect(),
            });
            dn.push(chain(d, 0));
        }
        let any = |v: &[Vec<(&crate::lora::LoraDelta, usize)>]| v.iter().any(|l| !l.is_empty());
        if !any(&gu) && !any(&dn) {
            return Some(None);
        }
        if ex.interleaved && any(&gu) {
            return None;
        }
        let mut stack = |v: &[Vec<(&crate::lora::LoraDelta, usize)>], n_in: usize, n_out: usize| -> Option<LoraStack> {
            let r = v.iter().map(|l| l.iter().map(|(d, _)| d.factors().2).sum::<usize>()).max()?;
            if r == 0 {
                return None;
            }
            let mut a = vec![0f32; ne * r * n_in];
            let mut bt = vec![0f32; ne * r * n_out];
            let mut scale = vec![0u32; ne * r];
            for (e, links) in v.iter().enumerate() {
                let mut c = 0;
                for &(d, row0) in links {
                    let (da, dbt, dr) = d.factors();
                    let (dn_out, dn_in) = d.shape();
                    if dn_in != n_in || row0 + dn_out > n_out {
                        return None;
                    }
                    let id = self.strength_id(d.strength_handle());
                    for j in 0..dr {
                        let row = e * r + c + j;
                        a[row * n_in..(row + 1) * n_in].copy_from_slice(&da[j * n_in..(j + 1) * n_in]);
                        bt[row * n_out + row0..row * n_out + row0 + dn_out].copy_from_slice(&dbt[j * dn_out..(j + 1) * dn_out]);
                        scale[row] = id;
                    }
                    c += dr;
                }
            }
            Some(LoraStack {
                a,
                bt,
                scale,
                r,
                n_in,
                n_out,
            })
        };
        let gate_up = if any(&gu) {
            let h = gu.iter().flatten().next().map(|(d, _)| d.shape().1)?;
            Some(stack(&gu, h, 2 * mi)?)
        } else {
            None
        };
        let down = if any(&dn) {
            let h = dn.iter().flatten().next().map(|(d, _)| d.shape().0)?;
            Some(stack(&dn, mi, h)?)
        } else {
            None
        };
        let r = gate_up.as_ref().map_or(0, |s| s.r).max(down.as_ref().map_or(0, |s| s.r));
        let raw = self.buf(if gate_up.is_some() { k * 2 * mi } else { 1 });
        let u = self.buf(k * r);
        Some(Some(Box::new(ExpertLora { gate_up, down, raw, u })))
    }
    /// Adds an activation buffer of `n` floats.
    pub fn buf(&mut self, n: usize) -> View {
        self.bufs.push(n);
        View {
            buf: self.bufs.len() - 1,
            off: 0,
        }
    }
}

impl Default for Graph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// One step's inputs and outputs. A multi-token plan ([`DecodePlan::tokens`] > 1) runs `t` tokens
/// at positions `pos..pos + t` (each attending to the ones before it), with every per-token input
/// and output below holding `t` rows.
pub struct Step<'s, 'c> {
    /// Tokens already in the caches (the position of the first token).
    pub pos: usize,
    /// Tokens of this step (1 for single-token plans).
    pub t: usize,
    /// For each table of `Graph::rope_halves`, one after the other: per token, cos then sin of
    /// its rotary angles.
    pub rope_cs: &'s [f32],
    /// The input embeddings.
    pub x: &'s [f32],
    /// The second input (`Graph::input2`), if the graph has one.
    pub x2: &'s [f32],
    /// The host KV caches, in the graph's order (each holds exactly `pos` positions).
    pub kv: &'s mut [&'c mut LayerCache],
    /// The recurrent states, in the graph's order: page-aligned allocations (`PageVec`) that
    /// stay put, which the device reads and updates in place.
    pub states: &'s mut [&'c mut [f32]],
    pub hidden: &'s mut [f32],
    /// Computes the head and copies the logits here when set.
    pub logits: Option<&'s mut [f32]>,
    /// [`Graph::params`] values: hot (every step) and cold with its key (copied when the key
    /// differs from the last step's).
    pub params: Option<(&'s [f32], &'s [f32], u64)>,
    /// Receives the [`Graph::capture`] rows (`[t, n]`).
    pub capture: Option<&'s mut [f32]>,
    /// The step's keys and values are scratch (a DSpark draft block): they are not appended to
    /// the host caches, and the next step overwrites them.
    pub scratch: bool,
    /// The recurrent states are exactly as the last step on this slot left them (nothing on the
    /// host wrote them since): a backend holding device copies may skip copying them up.
    pub states_kept: bool,
}

/// One sequence's token of a batched step ([`DecodePlan::step_rows`]): its session's slot, its
/// position, and its inputs as a one-token [`Step`] takes them.
pub struct Row<'s, 'c> {
    pub slot: &'s mut Slot,
    pub pos: usize,
    pub rope_cs: &'s [f32],
    pub x: &'s [f32],
    pub x2: &'s [f32],
    /// The session's host KV caches, holding exactly `pos` positions each.
    pub kv: &'s mut [&'c mut LayerCache],
    /// The session's recurrent states, as [`Step::states`] (each row its own; empty for plans
    /// without recurrent layers).
    pub states: &'s mut [&'c mut [f32]],
}

/// Greedy decoding chained on the device ([`DecodePlan::greedy`]).
pub struct Greedy<'s, 'c> {
    /// Tokens already in the caches (the position of the first pass).
    pub pos: usize,
    /// The token of the first pass.
    pub first: u32,
    /// Passes at most.
    pub max: usize,
    /// The RoPE rows of a position (as [`Step::rope_cs`] for one token).
    pub rope_cs: &'s dyn Fn(usize) -> Vec<f32>,
    /// The host KV caches (as [`Step::kv`]): the kept passes' keys and values are appended.
    pub kv: &'s mut [&'c mut LayerCache],
    /// The recurrent states (as [`Step::states`]): left as the kept passes leave them.
    pub states: &'s mut [&'c mut [f32]],
    /// As [`Step::states_kept`].
    pub states_kept: bool,
    /// Gets each kept pass's argmax in order and says what to do next.
    pub on: &'s mut dyn FnMut(u32) -> crate::generate::Chain,
}

/// A compiled graph.
pub trait DecodePlan: Send + Sync {
    /// Runs one step. Returns false (with the host caches unchanged) when the device failed.
    fn step(&self, slot: &mut Slot, s: Step) -> bool;
    /// Most tokens of a step.
    fn tokens(&self) -> usize {
        1
    }
    /// Multi-token plans: runs the recurrent-state ops (Gated DeltaNet conv and rule) of the last
    /// step again over its first `keep` tokens, from `states` the caller put back as they were
    /// before that step (the inputs those ops read are kept in per-layer buffers), so a rewind
    /// inside the step leaves exactly the states one-token steps would have. False if the plan
    /// cannot.
    fn replay_recurrent(&self, _slot: &mut Slot, _states: &mut [&mut [f32]], _keep: usize) -> bool {
        false
    }
    /// Greedy decoding with the passes chained on the device (one-token plans of graphs with
    /// [`Graph::embed`]): pass k runs the token chosen by pass k - 1 (the first index of the
    /// largest logit, as `crate::sampler::argmax`; `first` for pass 0), its input row looked up on
    /// the device, so that pass k + 1 is queued before the host has read pass k's token. `on`
    /// gets each kept pass's token ([`crate::generate::Chain`]); a pass queued past a stop is
    /// undone (its keys and values are not appended, the recurrent states are those before it).
    /// The passes kept, or None (nothing run, nothing changed) when the plan cannot.
    fn greedy(&self, _slot: &mut Slot, _g: Greedy) -> Option<usize> {
        None
    }
    /// Multi-token plans: one token of each of several independent sequences (`rows`, each with
    /// its own slot, caches and position) in one step, the weights read once for all: their
    /// final hidden states (`[rows, hidden]`) and, with `logits`, their logits. Each row gets
    /// what a one-token step of its sequence would give where the plan's multi-token kernels
    /// are exact (not on fast-verification plans). False (nothing changed) when the plan cannot
    /// (interventions, streamed weights, more rows than its tokens).
    fn step_rows(&self, _rows: &mut [Row], _hidden: &mut [f32], _logits: Option<&mut [f32]>) -> bool {
        false
    }
}

/// Per-session device state of a plan (the KV cache copies). Cloning a session starts its copy
/// empty (it is filled again from the host caches on the next step).
#[derive(Default)]
pub struct Slot(pub Option<Box<dyn Any + Send + Sync>>);

impl Slot {
    /// Bytes of device memory it holds (the KV cache copies; 0 for backends that do not tell).
    pub fn bytes(&self) -> usize {
        let Some(_s) = &self.0 else { return 0 };
        #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
        if let Some(n) = crate::metal::decode::slot_bytes(&**_s) {
            return n;
        }
        0
    }
}

impl Clone for Slot {
    fn clone(&self) -> Self {
        Slot(None)
    }
}

/// Compiled plans of a model by key (one per intervention layout; 0 is the plain model), and
/// a lock per key held while its plan builds (different keys build concurrently).
#[derive(Default)]
#[allow(clippy::type_complexity)]
pub struct PlanCache(
    std::sync::Mutex<Vec<(u64, SharedPlan)>>,
    std::sync::Mutex<Vec<(u64, std::sync::Arc<std::sync::Mutex<()>>)>>,
);

/// A compiled plan shared by the sessions of a model (None: the model cannot use one).
pub type SharedPlan = Option<std::sync::Arc<dyn DecodePlan>>;

impl PlanCache {
    /// Plans kept besides the plain model's (the least recently built go first).
    const KEEP: usize = 4;

    /// The plan for `key`, built by `make` the first time (a failure is remembered too).
    pub fn get(&self, key: u64, make: impl FnOnce() -> Option<Box<dyn DecodePlan>>) -> Option<std::sync::Arc<dyn DecodePlan>> {
        let built = || -> Option<SharedPlan> {
            let v = self.0.lock().unwrap_or_else(|p| p.into_inner());
            v.iter().find(|(k, _)| *k == key).map(|(_, p)| p.clone())
        };
        if let Some(p) = built() {
            return p;
        }
        let lock = {
            let mut b = self.1.lock().unwrap_or_else(|p| p.into_inner());
            match b.iter().find(|(k, _)| *k == key) {
                Some((_, l)) => l.clone(),
                None => {
                    let l = std::sync::Arc::new(std::sync::Mutex::new(()));
                    b.push((key, l.clone()));
                    l
                }
            }
        };
        let _building = lock.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = built() {
            return p;
        }
        let p: Option<std::sync::Arc<dyn DecodePlan>> = make().map(std::sync::Arc::from);
        let mut v = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if key != 0 && v.iter().filter(|(k, _)| *k != 0).count() >= Self::KEEP {
            let i = v.iter().position(|(k, _)| *k != 0).unwrap();
            v.remove(i);
        }
        v.push((key, p.clone()));
        p
    }
}

/// Where a streamed model with on-demand experts decodes when left to decide
/// ([`crate::gpu::stream_decode_forced`] unset): each MoE layer of a GPU step waits for its
/// routing to come back before the chosen experts can be read and wrapped, which on an M5 Max
/// costs about what decoding on the CPU does. The first steps alternate between the two (after
/// [`DeviceChoice::WARM`] each), then the faster one is kept.
#[derive(Default)]
pub struct DeviceChoice(std::sync::Mutex<ChoiceState>);

#[derive(Default)]
struct ChoiceState {
    /// Steps and seconds measured: `[cpu, gpu]`.
    n: [u32; 2],
    secs: [f64; 2],
    decided: Option<bool>,
}

impl DeviceChoice {
    /// Steps of each kind not counted (plans compiled, caches filled).
    pub const WARM: u32 = 2;
    /// Steps of each kind measured.
    pub const SAMPLES: u32 = 4;

    /// Whether the next step should run on the device.
    pub fn want_gpu(&self) -> bool {
        let s = self.0.lock().unwrap();
        s.decided.unwrap_or(s.n[1] <= s.n[0])
    }
    /// Whether steps are still being measured.
    pub fn measuring(&self) -> bool {
        self.0.lock().unwrap().decided.is_none()
    }
    /// Records a step's wall time; returns the decision once made (`Some(true)`: the device).
    pub fn record(&self, gpu: bool, secs: f64) -> Option<bool> {
        let mut s = self.0.lock().unwrap();
        if s.decided.is_some() {
            return None;
        }
        let i = gpu as usize;
        s.n[i] += 1;
        if s.n[i] > Self::WARM {
            s.secs[i] += secs;
        }
        if s.n.iter().all(|&n| n >= Self::WARM + Self::SAMPLES) {
            let per = [0, 1].map(|i| s.secs[i] / (s.n[i] - Self::WARM) as f64);
            let g = per[1] <= per[0];
            s.decided = Some(g);
            eprintln!(
                "[streamed decoding: {:.0} ms per token on the GPU, {:.0} on the CPU: {} from here on]",
                per[1] * 1e3,
                per[0] * 1e3,
                if g { "GPU" } else { "CPU" }
            );
            return s.decided;
        }
        None
    }
}

/// `OXINFER_GPU_DECODE_CHECK=1`: models also run every GPU decode step on the CPU and
/// [`report`] the difference.
pub fn check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_GPU_DECODE_CHECK").is_some_and(|v| v != "0"))
}

/// Prints how the GPU logits of the token at `pos` differ from the CPU's: argmax of each, the
/// KL divergence of the GPU distribution from the CPU one, and the largest logit difference.
pub fn report(pos: usize, cpu: &[f32], gpu: &[f32]) {
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a })
            .0
    };
    let lse = |v: &[f32]| {
        let m = v.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x)) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lc, lg) = (lse(cpu), lse(gpu));
    let mut kl = 0f64;
    let mut dmax = 0f32;
    for (&c, &g) in cpu.iter().zip(gpu) {
        let pc = (c as f64 - lc).exp();
        kl += pc * ((c as f64 - lc) - (g as f64 - lg));
        dmax = dmax.max((c - g).abs());
    }
    eprintln!(
        "[gpu-check pos {pos}: argmax cpu {} gpu {}, KL {kl:.2e}, max |diff| {dmax:.2e}]",
        argmax(cpu),
        argmax(gpu)
    );
}
