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
}

pub enum Op<'a> {
    /// RMSNorm of `rows` rows of `n` values (row strides in floats) of `x * pre`: `out = x' /
    /// rms(x') * w`, or `out += ...` with `add`.
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
    },
    /// Causal depthwise conv of kernel `kc` over `dim` channels with SiLU: `out = silu(sum of
    /// w[ch][j] * input j)`, inputs being the `kc - 1` earlier ones in recurrent state `state`
    /// (`[kc - 1, dim]`, oldest first) and `x`; the state then shifts in `x`.
    GdnConv {
        x: View,
        out: View,
        state: usize,
        w: &'a [f32],
        dim: usize,
        kc: usize,
    },
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
    /// `y +=` the weighted sum (else `y =`).
    pub add: bool,
    /// Neuron edits of the routed experts' activations (entries with expert ids, see
    /// [`crate::intervene::GpuNeurons`]): `(hot, n)`.
    pub edits: Option<(View, usize)>,
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
}

impl Graph<'_> {
    pub fn new() -> Self {
        Graph {
            bufs: Vec::new(),
            ops: Vec::new(),
            input: View { buf: 0, off: 0 },
            hidden: View { buf: 0, off: 0 },
            n_hidden: 0,
            head_from: 0,
            logits: None,
            kv: Vec::new(),
            rope_halves: Vec::new(),
            input2: None,
            states: Vec::new(),
            params: None,
        }
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

/// One step's inputs and outputs.
pub struct Step<'s, 'c> {
    /// Tokens already in the caches (the position of this token).
    pub pos: usize,
    /// cos then sin of this token's rotary angles for each table of `Graph::rope_halves`, one
    /// table after the other.
    pub rope_cs: &'s [f32],
    /// The input embedding.
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
}

/// A compiled graph.
pub trait DecodePlan: Send + Sync {
    /// Runs one token. Returns false (with the host caches unchanged) when the device failed.
    fn step(&self, slot: &mut Slot, s: Step) -> bool;
}

/// Per-session device state of a plan (the KV cache copies). Cloning a session starts its copy
/// empty (it is filled again from the host caches on the next step).
#[derive(Default)]
pub struct Slot(pub Option<Box<dyn Any + Send + Sync>>);

impl Clone for Slot {
    fn clone(&self) -> Self {
        Slot(None)
    }
}

/// Compiled plans of a model by key (one per intervention layout; 0 is the plain model).
#[derive(Default)]
pub struct PlanCache(std::sync::Mutex<Vec<(u64, SharedPlan)>>);

/// A compiled plan shared by the sessions of a model (None: the model cannot use one).
pub type SharedPlan = Option<std::sync::Arc<dyn DecodePlan>>;

impl PlanCache {
    /// Plans kept besides the plain model's (the least recently built go first).
    const KEEP: usize = 4;

    /// The plan for `key`, built by `make` the first time (a failure is remembered too).
    pub fn get(&self, key: u64, make: impl FnOnce() -> Option<Box<dyn DecodePlan>>) -> Option<std::sync::Arc<dyn DecodePlan>> {
        let mut v = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, p)) = v.iter().find(|(k, _)| *k == key) {
            return p.clone();
        }
        let p: Option<std::sync::Arc<dyn DecodePlan>> = make().map(std::sync::Arc::from);
        if key != 0 && v.iter().filter(|(k, _)| *k != 0).count() >= Self::KEEP {
            let i = v.iter().position(|(k, _)| *k != 0).unwrap();
            v.remove(i);
        }
        v.push((key, p.clone()));
        p
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
