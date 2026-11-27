//! Accelerator backends: a trait the model's linear layers dispatch through, plus a global
//! registry chosen at start-up from the platform detection. Every backend is optional at run time;
//! the CPU kernels remain the reference path and the fallback.

use crate::ggml::GgufMatrix;
use crate::quant::{Q4Matrix, QMatrix};
use std::any::Any;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

#[path = "gpu_decode.rs"]
pub mod decode;
#[path = "gpu_dit.rs"]
pub mod dit;
#[path = "gpu_unet.rs"]
pub mod unet;
#[path = "gpu_whisper.rs"]
pub mod whisper;

/// A weight matrix in one of the supported storage formats (`[n_out, n_in]`).
pub enum WeightRef<'a> {
    Bf16(&'a [u16]),
    Q8(&'a QMatrix),
    Q4(&'a Q4Matrix),
    /// llama.cpp block formats, read as stored (Metal and CUDA have kernels for most; a backend
    /// declines the ones it has none for, which stay on the CPU).
    Gguf(&'a GgufMatrix),
}

impl WeightRef<'_> {
    pub fn bytes(&self) -> usize {
        match self {
            WeightRef::Bf16(w) => w.len() * 2,
            WeightRef::Q8(m) => m.bytes(),
            WeightRef::Q4(m) => m.bytes(),
            WeightRef::Gguf(m) => m.bytes(),
        }
    }
}

/// The gated activation of MoE experts: `act(gate) * up`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MoeAct {
    Silu,
    GeluTanh,
    /// SiLU with DeepSeek V4's clamp: `silu(min(gate, l)) * clamp(up, -l, l)`.
    SiluClamp(f32),
    /// MiniMax M3's "swigluoai" (GPT-OSS's, gate and up not interleaved): with
    /// `g = min(gate, limit)`, `(clamp(up, -limit, limit) + 1) * g * sigmoid(alpha * g)`.
    /// CPU only for now: the accelerator paths decline it.
    SwigluOai {
        alpha: f32,
        limit: f32,
    },
    /// ERNIE 5.5's and Kimi K3's SiTU: `beta tanh(gate / beta) sigmoid(gate) * linear_beta tanh(up /
    /// linear_beta)` (the up side unclipped when `linear_beta` is 0). CPU only for now.
    Situ {
        beta: f32,
        linear_beta: f32,
    },
}

/// Device-side state for one weight matrix (buffers / handles), owned by the layer.
pub type DeviceWeight = Box<dyn Any + Send + Sync>;

/// One matmul of a batch sharing the same input `x`: `y[t][n_out] = x[t][n_in] * W^T`.
pub struct Job<'a> {
    pub w: WeightRef<'a>,
    pub dev: &'a DeviceWeight,
    pub n_out: usize,
    pub y: &'a mut [f32],
}

/// One matrix-vector product of [`Backend::matvec_rows`]: `y[t][n_out] = W x[t]`, or with `up`
/// (the same shape and format) `y[t] = silu(W x[t]) * (U x[t])`.
pub struct RowsJob<'a> {
    pub w: WeightRef<'a>,
    pub dev: &'a DeviceWeight,
    pub up: Option<(WeightRef<'a>, &'a DeviceWeight)>,
    pub n_out: usize,
    pub y: &'a mut [f32],
}

/// A DSpark draft's vanilla Markov head, greedy, in one submission ([`Backend::markov_greedy`]):
/// slot `i`'s logits (`logits` row `i`) get the bias `w2 * w1[prev]`, where `prev` is `first`
/// for slot 0 and slot `i - 1`'s argmax after its bias otherwise.
pub struct MarkovJob<'a> {
    /// `[vocab, rank]` bf16 rows: the previous token's embedding.
    pub w1: &'a [u16],
    pub w1_dev: &'a DeviceWeight,
    /// `[vocab, rank]`: the bias.
    pub w2: WeightRef<'a>,
    pub w2_dev: &'a DeviceWeight,
    pub rank: usize,
    pub vocab: usize,
    pub first: u32,
    /// `[k, vocab]`: the slots' logits before the bias, replaced by the logits after it.
    pub logits: &'a mut [f32],
    /// `[k]`: each slot's argmax (the first index of the largest logit, as
    /// [`crate::sampler::argmax`]).
    pub tokens: &'a mut [u32],
}

/// Attention of a prefill chunk over a host KV cache ([`Backend::attention_cached`]): query
/// `i` (position `start + i`, `[nh, hd]`, already scaled) attends to the cached positions
/// `sel[i]` when set (QSA, Qwen4-Exp), else to positions `0..=start + i`; heads share key / value
/// heads in groups of `nh / nkv`.
pub struct AttnJob<'a> {
    pub q: &'a [f32],
    /// `[len, nkv, hd]` each.
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub start: usize,
    pub t: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub sel: Option<&'a [Option<Vec<u32>>]>,
    /// `[t, nh, hd]`.
    pub out: &'a mut [f32],
}

/// The gated delta rule of Gated DeltaNet over `t` tokens in order ([`Backend::gdn_rule`]; the
/// CPU path is `crate::qwen`'s `DeltaRule::run`).
pub struct GdnJob<'a> {
    /// `[t, 2 nk hdk + nv hdv]`: the conv outputs (q, k, v).
    pub c: &'a [f32],
    /// `[t, 2 nv]`: per token `b`, then `a`.
    pub ba: &'a [f32],
    /// `[nv, hdk, hdv]`, updated.
    pub state: &'a mut [f32],
    /// `[t, nv hdv]`: the rule's outputs (before the gated norm).
    pub out: &'a mut [f32],
    pub t: usize,
    pub nk: usize,
    pub nv: usize,
    pub hdk: usize,
    pub hdv: usize,
    pub tiled: bool,
    pub decay_coef: &'a [f32],
    pub dt_bias: &'a [f32],
}

/// Geometry of a k x k convolution over channels-last input `[h*w, cin]`, zero padded by `pad`
/// pixels before the first row and column (k/2 for a "same" convolution; the zero padding after
/// the last row and column is whatever `oh`, `ow` and `stride` reach), producing `[oh*ow, n_out]`;
/// the weights are in the unfolded `[n_out][ky][kx][cin]` layout.
#[derive(Clone, Copy, Debug)]
pub struct ConvGeom {
    pub h: usize,
    pub w: usize,
    pub cin: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub oh: usize,
    pub ow: usize,
}

/// A normalization applied to a convolution's input as it is loaded (so it runs on the device
/// without an extra pass): `GroupNorm(groups)` over the channels of all pixels, or `Layer` over each
/// pixel's (row's) channels; then `* gamma + beta`, then SiLU if `silu`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormKind {
    Group(usize),
    Layer,
}

#[derive(Clone, Copy, Debug)]
pub struct NormSpec<'a> {
    pub kind: NormKind,
    pub eps: f32,
    pub gamma: &'a [f32],
    pub beta: &'a [f32],
    pub silu: bool,
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    /// Whether this backend should run a matmul of these dimensions. Dispatch has a fixed cost
    /// (~0.15 ms on Metal), so small ops and single-token decode stay on the CPU.
    fn wants(&self, n_out: usize, n_in: usize, t: usize) -> bool;
    /// Prepares device-side state for a weight matrix (zero-copy where the API allows).
    fn upload(&self, w: &WeightRef, n_out: usize, n_in: usize) -> Option<DeviceWeight>;
    /// Runs a batch of matmuls over one shared input in a single submission. Returns false if
    /// nothing was computed (the caller then runs the whole batch on the CPU).
    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool;
    /// Runs a batch of matrix-vector products over `t` (1 to 4) input rows in one submission,
    /// whatever their size ([`Backend::wants`] does not apply): the few rows of a draft model's
    /// pass, where the device reads the weights faster than the CPU. Returns false if nothing was
    /// computed (the caller then runs the batch on the CPU).
    fn matvec_rows(&self, _x: &[f32], _t: usize, _n_in: usize, _jobs: &mut [RowsJob]) -> bool {
        false
    }
    /// See [`MarkovJob`]: the whole chain in one submission. Returns false if nothing was
    /// computed (the caller then runs the chain itself).
    fn markov_greedy(&self, _j: &mut MarkovJob) -> bool {
        false
    }
    /// Whether [`Backend::markov_greedy`] runs at all (the caller then prepares its table).
    fn has_markov_greedy(&self) -> bool {
        false
    }
    /// Runs a convolution directly on the device (gathering the k x k neighbourhoods itself, so
    /// the unfolded input never exists in host memory), optionally normalizing its input on the
    /// way in. `job.y` is `[oh*ow, n_out]`. Returns false when the backend does not handle this
    /// case (the caller normalizes and unfolds on the CPU).
    fn conv2d(&self, _x: &[f32], _g: &ConvGeom, _norm: Option<&NormSpec>, _job: &mut Job) -> bool {
        false
    }
    /// Multi-head attention: `q` `[tq, heads*dh]`, `k` and `v` `[tk, heads*dh]` into `out`
    /// (`[tq, heads*dh]`), softmax(q k^T / sqrt(dh)) v, keys after each query masked if `causal`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _tq: usize,
        _tk: usize,
        _heads: usize,
        _dh: usize,
        _causal: bool,
        _out: &mut [f32],
    ) -> bool {
        false
    }
    /// A session running whole diffusion-transformer passes on the device (see [`dit`]); None
    /// when the backend has none.
    fn dit_session(&'static self) -> Option<Box<dyn dit::Session>> {
        None
    }
    /// A session running whole FLUX.1 transformer passes on the device; None when the backend
    /// has none.
    fn flux_session(&'static self) -> Option<Box<dyn dit::FluxSession>> {
        None
    }
    /// A session running whole T5 encoder passes on the device; None when the backend has none.
    fn t5_session(&'static self) -> Option<Box<dyn dit::T5Session>> {
        None
    }
    /// A session running whole LTX-2.5 transformer passes on the device; None when the backend
    /// has none.
    fn ltx_session(&'static self) -> Option<Box<dyn dit::LtxSession>> {
        None
    }
    /// A session running LTX-2.5 DiffVAE blocks on the device; None when the backend has none.
    fn na_session(&'static self) -> Option<Box<dyn dit::NaSession>> {
        None
    }
    /// 3D neighborhood attention (`natten.na3d`, LTX-2.5's DiffVAE): queries over a `qd` grid
    /// (`[t, h, w]`) attend to the `ker` window of keys / values over a `kd` grid centered at
    /// the query's index plus `off` per axis, shifted inward at the borders; rows of
    /// `heads * hd`, queries already scaled. False when the backend has no kernel for it.
    #[allow(clippy::too_many_arguments)]
    fn na3d(
        &self,
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _qd: [usize; 3],
        _kd: [usize; 3],
        _off: [usize; 3],
        _ker: [usize; 3],
        _heads: usize,
        _hd: usize,
        _out: &mut [f32],
    ) -> bool {
        false
    }
    /// A session running whole LTX-2.5 video VAE decodes on the device; None when the backend
    /// has none.
    fn ltx_vae_session(&'static self) -> Option<Box<dyn dit::LtxVaeSession>> {
        None
    }
    /// A session running whole LTX-2.5 vocoder generator passes on the device; None when the
    /// backend has none.
    fn ltx_voc_session(&'static self) -> Option<Box<dyn dit::LtxVocSession>> {
        None
    }
    /// A session running whole CLIP text-encoder passes on the device; None when the backend
    /// has none.
    fn clip_session(&'static self) -> Option<Box<dyn dit::ClipSession>> {
        None
    }
    /// A session running whole convolutional UNet passes on the device (see [`unet`]); None
    /// when the backend has none.
    fn unet_session(&'static self) -> Option<Box<dyn unet::Session>> {
        None
    }
    /// A session running whole Whisper encoder passes on the device (see [`whisper`]); None when
    /// the backend has none.
    fn whisper_session(&'static self) -> Option<Box<dyn whisper::Session>> {
        None
    }
    /// See [`AttnJob`]. False when the backend does not run it (the caller then does).
    fn attention_cached(&self, _j: &mut AttnJob) -> bool {
        false
    }
    /// See [`GdnJob`]. False (the state untouched) when the backend does not run it.
    fn gdn_rule(&self, _j: &mut GdnJob) -> bool {
        false
    }
    /// Device state for the routed experts of an MoE layer: expert `e` has the fused gate/up
    /// projection `gate_up[e]` (`[2 * inter, h]`, gate rows then up rows), or with `up` the gate
    /// projection `gate_up[e]` (`[inter, h]`) and the up projection `up[e]` (GGUF files store
    /// them apart), and the down projection `down[e]` (`[h, inter]`), with activation `act`. None
    /// if the backend does not run MoE layers.
    #[allow(clippy::too_many_arguments)]
    fn upload_moe(
        &self,
        _gate_up: &[WeightRef],
        _up: Option<&[WeightRef]>,
        _down: &[WeightRef],
        _h: usize,
        _inter: usize,
        _act: MoeAct,
    ) -> Option<DeviceWeight> {
        None
    }
    /// [`Backend::upload_moe`] for one pass over experts that stream from storage: the state is
    /// dropped after the call, and nothing keeps the weights resident beyond the submissions
    /// that use them.
    #[allow(clippy::too_many_arguments)]
    fn upload_moe_transient(
        &self,
        _gate_up: &[WeightRef],
        _up: Option<&[WeightRef]>,
        _down: &[WeightRef],
        _h: usize,
        _inter: usize,
        _act: MoeAct,
    ) -> Option<DeviceWeight> {
        None
    }
    /// The routed experts of an MoE layer (`dev` from `upload_moe`) in one submission: token `i`
    /// of `x` (`[t, h]`) goes through the experts `routes[i]` lists (expert index, weight), and
    /// `y[i] += sum of weight * down(act(gate) * up)`. Returns false if nothing was computed
    /// (the caller then runs the experts itself).
    #[allow(clippy::too_many_arguments)]
    fn moe(
        &self,
        _x: &[f32],
        _t: usize,
        _h: usize,
        _inter: usize,
        _routes: &[Vec<(usize, f32)>],
        _dev: &DeviceWeight,
        _y: &mut [f32],
    ) -> bool {
        false
    }
    /// Whether [`Backend::decode_plan`] can return a plan at all (the start-up banner says where
    /// decoding runs).
    fn decodes(&self) -> bool {
        false
    }
    /// Bytes the device can keep resident for this process without losing performance (Metal's
    /// `recommendedMaxWorkingSetSize`), when known: the GPU-resident weights of a streamed model
    /// and its read-ahead window must fit in it.
    fn working_set(&self) -> Option<u64> {
        None
    }
    /// `(free, total)` bytes of the device's own memory, when it has some (a discrete GPU) and
    /// says: what loading another model can still put there.
    fn device_memory(&self) -> Option<(u64, u64)> {
        None
    }
    /// Bytes of device memory this process has allocated through the backend (Metal's
    /// `currentAllocatedSize`), when it says: the server's status page.
    fn allocated_memory(&self) -> Option<u64> {
        None
    }
    /// Whether checkpoints are better mapped from their files than read into memory: the device
    /// keeps a copy of every weight it uses in memory it shares with the host (a unified-memory
    /// CUDA device), so the host's own copy would hold the same physical memory a second time,
    /// where a mapped file's pages are given back when something else needs them.
    fn prefers_mapped_weights(&self) -> bool {
        false
    }
    /// Whether this backend's plain decoding outruns speculative decoding (whose drafts run on the
    /// CPU): the speculative defaults (MTP, DSpark drafts) then give way to plain decoding; asking
    /// for them explicitly still uses them.
    fn prefers_plain_decode(&self) -> bool {
        false
    }
    /// Compiles a single-token decode graph (see [`decode`]); None when the backend cannot run
    /// some op or weight of it (the model then decodes on the CPU).
    fn decode_plan<'s>(&'s self, _g: &decode::Graph) -> Option<Box<dyn decode::DecodePlan + 's>> {
        None
    }
    /// Compiles a decode graph for steps of up to `tokens` tokens (speculative verification;
    /// see [`decode::DecodePlan::tokens`]); None when the backend cannot.
    fn decode_plan_multi<'s>(&'s self, _g: &decode::Graph, _tokens: usize) -> Option<Box<dyn decode::DecodePlan + 's>> {
        None
    }
    /// [`Backend::decode_plan_multi`] for "fast verification": the multi-token steps may use
    /// batched kernels whose arithmetic differs from the single-token kernels' (a token's results
    /// then depend on the batch in rounding), for speed. Backends without such kernels give the
    /// exact plan.
    fn decode_plan_multi_fast<'s>(&'s self, g: &decode::Graph, tokens: usize) -> Option<Box<dyn decode::DecodePlan + 's>> {
        self.decode_plan_multi(g, tokens)
    }
    /// Compiles a decode graph as a prefill plan: steps of up to `tokens` tokens (at least; the
    /// plan may round up, see [`decode::DecodePlan::tokens`]) on batched kernels, the
    /// activations kept on the device for the whole step. Its results differ from one-token
    /// steps in rounding. None when the backend cannot.
    fn decode_plan_prefill<'s>(&'s self, _g: &decode::Graph, _tokens: usize) -> Option<Box<dyn decode::DecodePlan + 's>> {
        None
    }
    /// Cumulative timing summary for `--profile`.
    fn stats(&self) -> Option<String> {
        None
    }
    /// The backend as its concrete type, for model code that runs kernels of its own on it
    /// (Inkling-2's CUDA and Metal engines). None by default.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

static ACTIVE: OnceLock<Option<Box<dyn Backend>>> = OnceLock::new();

/// What [`warn`] recorded: reasons this process runs slower than its hardware allows.
static WARNINGS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Records that the process runs in a degraded mode (the matrix-unit kernels failed to compile,
/// a GPU kernel's pipeline could not be built, the GPU backend failed to start and the CPU took
/// over...): printed to stderr at once as a `WARNING` line (the CLI's and the server's log), and
/// kept for [`warnings`], which the server reports (`GET /health`, `"degraded"` / `"warnings"`)
/// and the C API returns (`oxi_warnings`). The same message is recorded once.
pub fn warn(msg: impl Into<String>) {
    let msg = msg.into();
    let mut w = WARNINGS.lock().unwrap_or_else(|e| e.into_inner());
    if !w.contains(&msg) {
        eprintln!("[WARNING: {msg}]");
        w.push(msg);
    }
}

/// The degraded-mode warnings recorded so far ([`warn`]), oldest first.
pub fn warnings() -> Vec<String> {
    WARNINGS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Whether any degraded-mode warning was recorded ([`warnings`] is not empty).
pub fn degraded() -> bool {
    !WARNINGS.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
}

/// Process-wide switches: 0 = unset (environment default), 1 = on, 2 = off.
static EXACT: AtomicU8 = AtomicU8::new(0);
static MOE: AtomicU8 = AtomicU8::new(0);
static DECODE: AtomicU8 = AtomicU8::new(0);
static CHAIN: AtomicU8 = AtomicU8::new(0);
static KV_HALF: AtomicU8 = AtomicU8::new(0);

fn switch(a: &AtomicU8, env: &'static OnceLock<Option<String>>, var: &str, on_env: fn(&str) -> bool, default: bool) -> bool {
    match a.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => env.get_or_init(|| std::env::var(var).ok()).as_deref().map_or(default, on_env),
    }
}

/// Keeps GPU matmuls bit-identical to the CPU: no reduced-precision matrix units (Metal on
/// M5/A19 converts inputs to half with f32 accumulation, a KL divergence of 1e-7 to 1e-3 from
/// the CPU). Process-wide, takes effect on the next dispatch. Default: off, or
/// `OXINFER_METAL_MMA=0`.
pub fn set_exact(on: bool) {
    EXACT.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// Keeps attention in f32 on the GPU (f32 operands on the matrix units where the GPU has them)
/// while the matmuls may still use half-precision operands: for models whose attention logits
/// are large (Qwen-Image's run into the thousands, where half-precision queries and keys move the
/// softmax visibly). Process-wide. Default: off.
/// Returns the previous setting.
pub fn set_exact_attention(on: bool) -> bool {
    EXACT_ATTN.swap(on, Ordering::Relaxed)
}

/// See [`set_exact_attention`] (also on with [`exact`]).
pub fn exact_attention() -> bool {
    EXACT_ATTN.load(Ordering::Relaxed) || exact()
}

static EXACT_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`set_exact`] when it was called (or `OXINFER_METAL_MMA` set), else None: for backends whose
/// default is the exact path.
pub fn exact_forced() -> Option<bool> {
    match EXACT.load(Ordering::Relaxed) {
        1 => Some(true),
        2 => Some(false),
        _ => std::env::var("OXINFER_METAL_MMA").ok().map(|v| v == "0"),
    }
}

/// See [`set_exact`].
pub fn exact() -> bool {
    static ENV: OnceLock<Option<String>> = OnceLock::new();
    switch(&EXACT, &ENV, "OXINFER_METAL_MMA", |v| v == "0", false)
}

/// Runs MoE expert matmuls on the GPU when the backend can (prefill). Process-wide; affects
/// models loaded afterwards. Default: on, or `OXINFER_METAL_MOE=0` for off.
pub fn set_moe_offload(on: bool) {
    MOE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// See [`set_moe_offload`].
pub fn moe_offload() -> bool {
    static ENV: OnceLock<Option<String>> = OnceLock::new();
    switch(&MOE, &ENV, "OXINFER_METAL_MOE", |v| v != "0", true)
}

/// Runs single-token decode steps entirely on the GPU (one submission per token) for the model
/// families and weight formats the backend supports; prefill, speculative verification and
/// anything unsupported stay where they were. Process-wide, read when a decode step starts.
/// None restores the default (on, or `OXINFER_GPU_DECODE=0|1`).
pub fn set_decode(on: Option<bool>) {
    DECODE.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// See [`set_decode`].
pub fn decode_enabled() -> bool {
    static ENV: OnceLock<Option<String>> = OnceLock::new();
    switch(&DECODE, &ENV, "OXINFER_GPU_DECODE", |v| v != "0", true)
}

/// Chains greedy decoding on the device: where a model decodes whole tokens on the GPU, each
/// pass ends with the argmax of its logits and the next pass starts from that token's embedding
/// row, both computed on the device, so the next pass is queued while the host reads a token
/// (plain greedy decoding, the plain rounds of prompt lookup and DSpark). The tokens are those of
/// the host's argmax, bit for bit. Process-wide, read when a reply starts. None restores the
/// default (on, or `OXINFER_GPU_CHAIN=0|1`).
pub fn set_decode_chain(on: Option<bool>) {
    CHAIN.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// See [`set_decode_chain`].
pub fn decode_chain_enabled() -> bool {
    static ENV: OnceLock<Option<String>> = OnceLock::new();
    switch(&CHAIN, &ENV, "OXINFER_GPU_CHAIN", |v| v != "0", true)
}

/// Keeps the device's copies of the attention caches as half-precision rows (`true`, the default)
/// or f32 ones (`false`), where whole-pass GPU plans run (Metal with matrix units: decoding,
/// verification and prefill plans; QSA caches stay f32). Half rows halve the attention's memory
/// traffic at long contexts, as llama.cpp's f16 cache; the host's caches stay f32 (filled from
/// the device's rows). Process-wide, read when a model builds its plans. None restores the
/// default (on, or `OXINFER_GPU_KV=f32|f16`).
pub fn set_kv_half(on: Option<bool>) {
    KV_HALF.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// Copies the routed experts of quantized GGUF checkpoints into a tiled layout for the GPU
/// (Metal: each band of 64 rows stored block column by block column, so that a grouped GEMM
/// tile reads each K step's weights as one run of memory; the same values, so the same
/// results). The copies are made when a model builds its first decode or prefill plan, once per
/// model, and both the prefill GEMMs and the decode kernels read them; the checkpoint's experts
/// are then let go (their pages dropped from memory when mapped from the file, read again from
/// it if a CPU path needs them), so the experts take their memory once. Experts read into memory
/// (not mapped) stay: their copies take as much memory again. Forced on, the per-layer MoE
/// uploads of models without plans make copies too (for their GEMMs, beside the experts).
/// `Some(true)`: always (also on GPUs without matrix units); `Some(false)`: never; None (the
/// default, or `OXINFER_GPU_EXPERT_TILES=0|1`): on GPUs with matrix units, not where the OS
/// limits an app's memory (iOS), and for experts read into memory while the copies stay within
/// a quarter of the machine's memory. Not for streamed models. Process-wide, read when a model
/// builds its plans.
pub fn set_expert_tiles(on: Option<bool>) {
    EXPERT_TILES.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// See [`set_expert_tiles`] (None: automatic).
pub fn expert_tiles() -> Option<bool> {
    match EXPERT_TILES.load(Ordering::Relaxed) {
        1 => Some(true),
        2 => Some(false),
        // (Read as each model builds its plans, so that `pf_ab` can compare.)
        _ => std::env::var("OXINFER_GPU_EXPERT_TILES").ok().map(|v| v != "0"),
    }
}

static EXPERT_TILES: AtomicU8 = AtomicU8::new(0);

/// See [`set_kv_half`].
pub fn kv_half_enabled() -> bool {
    static ENV: OnceLock<Option<String>> = OnceLock::new();
    switch(&KV_HALF, &ENV, "OXINFER_GPU_KV", |v| v != "f32", true)
}

/// Decodes on the device also when the weights stream from storage (`crate::stream`): the
/// resident layers stay wrapped, every other layer is wrapped per pass as it arrives and dropped
/// after (see `crate::gpu::decode::Graph::units`). Process-wide, read when a model builds its
/// plans. None restores the default (auto: dense models decode on the GPU, models with on-demand
/// experts measure both and keep the faster, see [`decode::DeviceChoice`]; or
/// `OXINFER_STREAM_GPU=0|1`); without device decoding ([`decode_enabled`]) streamed models decode
/// on the CPU either way.
pub fn set_stream_decode(on: Option<bool>) {
    STREAM_DECODE.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// See [`set_stream_decode`].
pub fn stream_decode_enabled() -> bool {
    static ENV: OnceLock<Option<String>> = OnceLock::new();
    decode_enabled() && switch(&STREAM_DECODE, &ENV, "OXINFER_STREAM_GPU", |v| v != "0", true)
}

/// [`set_stream_decode`] when set (or `OXINFER_STREAM_GPU`); None: the default, where models
/// with on-demand experts measure both and keep the faster
/// ([`decode::DeviceChoice`]).
pub fn stream_decode_forced() -> Option<bool> {
    match STREAM_DECODE.load(Ordering::Relaxed) {
        1 => Some(true),
        2 => Some(false),
        _ => std::env::var("OXINFER_STREAM_GPU").ok().map(|v| v != "0"),
    }
}

static STREAM_DECODE: AtomicU8 = AtomicU8::new(0);

/// Layers ahead whose experts a streamed GPU decode predicts while experts miss (DeepSeek V4.1 on
/// Metal: each layer also runs the routers of the next `n` layers on its own input, and the
/// experts they choose that are not in memory are read with the misses, so the next layers do not
/// miss them). 0 turns it off; None restores the default (2, or `OXINFER_DS41_PREDICT`, which
/// overrides any setting). Process-wide, read when a model builds its decode plan; at most 4.
pub fn set_stream_predict(layers: Option<usize>) {
    STREAM_PREDICT.store(layers.map_or(usize::MAX, |n| n.min(STREAM_PREDICT_MAX)), Ordering::Relaxed);
}

/// See [`set_stream_predict`].
pub fn stream_predict() -> usize {
    if let Some(n) = std::env::var("OXINFER_DS41_PREDICT")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        return n.min(STREAM_PREDICT_MAX);
    }
    match STREAM_PREDICT.load(Ordering::Relaxed) {
        usize::MAX => 2,
        n => n,
    }
}

/// The most layers [`set_stream_predict`] takes.
pub const STREAM_PREDICT_MAX: usize = 4;

static STREAM_PREDICT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// True when the active backend decodes on the device and prefers plain decoding there (see
/// [`Backend::prefers_plain_decode`]).
pub fn plain_decode_preferred() -> bool {
    decode_enabled() && active().is_some_and(|b| b.decodes() && b.prefers_plain_decode())
}

/// Installs the backend for this process (call once, before loading models).
pub fn install(b: Option<Box<dyn Backend>>) {
    let _ = ACTIVE.set(b);
}

/// Whether a backend choice (possibly "none") was installed already.
pub fn initialized() -> bool {
    ACTIVE.get().is_some()
}

pub fn active() -> Option<&'static dyn Backend> {
    ACTIVE.get().and_then(|b| b.as_deref())
}

/// Picks and initialises a backend for the detected platform and the user's preference. `Auto`
/// tries each detected backend in order (Metal or CUDA, then Vulkan) and falls back to the CPU
/// when none can start; a device asked for by name that is missing
/// or cannot start is an error (and nothing is installed, so the caller can still choose another).
pub fn init(plat: &crate::platform::Platform, pref: crate::platform::DevicePref) -> anyhow::Result<Option<&'static dyn Backend>> {
    use crate::platform::DevicePref;
    let start = |g: &crate::platform::GpuInfo| -> Result<Box<dyn Backend>, String> {
        match g.kind {
            #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
            crate::platform::GpuKind::Metal => crate::metal::MetalBackend::new()
                .map(|b| Box::new(b) as Box<dyn Backend>)
                .map_err(|e| e.to_string()),
            #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
            crate::platform::GpuKind::Cuda => crate::cuda::CudaBackend::new(g.index)
                .map(|b| Box::new(b) as Box<dyn Backend>)
                .map_err(|e| e.to_string()),
            #[cfg(all(feature = "gpu", target_pointer_width = "64", not(target_arch = "wasm32")))]
            crate::platform::GpuKind::Vulkan => crate::vulkan::VulkanBackend::new(g.index)
                .map(|b| Box::new(b) as Box<dyn Backend>)
                .map_err(|e| e.to_string()),
            #[allow(unreachable_patterns)]
            _ => Err(format!("{} compute is not implemented in this build", g.kind.as_str())),
        }
    };
    let chosen: Option<Box<dyn Backend>> = match pref {
        DevicePref::Cpu => None,
        // The first backend that starts: Metal or CUDA when present, else Vulkan.
        DevicePref::Auto => plat.auto_order().find_map(|g| match start(g) {
            Ok(b) => Some(b),
            Err(e) => {
                if !e.contains("not implemented") {
                    warn(format!("the {} backend failed to start ({e}); falling back", g.kind.as_str()));
                }
                None
            }
        }),
        DevicePref::Gpu(k) => match plat.pick(pref) {
            None => {
                let why = if plat.notes.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", plat.notes.join("; "))
                };
                anyhow::bail!("device {}: not available{why}", k.as_str());
            }
            Some(g) => match start(g) {
                Ok(b) => Some(b),
                Err(e) => anyhow::bail!("device {}: {e}", g.kind.as_str()),
            },
        },
    };
    install(chosen);
    Ok(active())
}

/// The process's backend, initialised on first use: [`init`] the first time, afterwards the
/// installed one, which must agree with a device asked for by name (the backend is process-wide,
/// so a second load cannot pick another).
pub fn ensure(pref: crate::platform::DevicePref) -> anyhow::Result<Option<&'static dyn Backend>> {
    use crate::platform::DevicePref;
    if !initialized() {
        return init(&crate::platform::detect(), pref);
    }
    let b = active();
    let current = b.map_or("cpu", |b| b.name());
    match pref {
        DevicePref::Auto => {}
        DevicePref::Cpu if b.is_none() => {}
        DevicePref::Gpu(k) if current.starts_with(k.as_str()) => {}
        DevicePref::Cpu => anyhow::bail!("device cpu: this process already runs on {current}"),
        DevicePref::Gpu(k) => anyhow::bail!("device {}: this process already runs on {current}", k.as_str()),
    }
    Ok(b)
}

/// With `OXINFER_PF_PROFILE` set, prints the GPU time per kind of op of the prefill plans
/// (Metal) so far.
pub fn pf_report() {
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
    crate::metal::decode::pf_report();
}

/// Clears the times [`pf_report`] adds up (to profile one pass after warm-up ones).
pub fn pf_reset() {
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
    crate::metal::decode::pf_reset();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{DevicePref, GpuInfo, GpuKind, detect};

    #[test]
    fn a_device_asked_for_by_name_never_falls_back() {
        let mut plat = detect();
        plat.gpus.clear();
        let e = init(&plat, DevicePref::Gpu(GpuKind::Cuda)).err().expect("missing device must fail");
        assert!(e.to_string().starts_with("device cuda: not available"), "{e}");
        // Detected, but with no compute backend in this build.
        let kind = if cfg!(any(target_os = "macos", target_os = "ios")) {
            GpuKind::Cuda
        } else {
            GpuKind::Metal
        };
        plat.gpus.push(GpuInfo {
            kind,
            index: 0,
            name: "test".into(),
            memory_bytes: None,
            driver: String::new(),
            unified_memory: false,
        });
        let e = init(&plat, DevicePref::Gpu(kind)).err().expect("no compute for this kind here");
        assert!(e.to_string().contains("not implemented"), "{e}");
    }

    #[test]
    fn warnings_are_recorded_once() {
        let msg = "test: a degraded mode recorded by warnings_are_recorded_once";
        warn(msg);
        warn(msg);
        assert_eq!(warnings().iter().filter(|w| *w == msg).count(), 1);
        assert!(degraded());
    }
}
