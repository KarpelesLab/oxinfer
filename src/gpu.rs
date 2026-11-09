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

/// A weight matrix in one of the supported storage formats (`[n_out, n_in]`).
pub enum WeightRef<'a> {
    Bf16(&'a [u16]),
    Q8(&'a QMatrix),
    Q4(&'a Q4Matrix),
    /// llama.cpp block formats: no device kernels yet, every backend declines them (CPU path).
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
    /// Cumulative timing summary for `--profile`.
    fn stats(&self) -> Option<String> {
        None
    }
}

static ACTIVE: OnceLock<Option<Box<dyn Backend>>> = OnceLock::new();

/// Process-wide switches: 0 = unset (environment default), 1 = on, 2 = off.
static EXACT: AtomicU8 = AtomicU8::new(0);
static MOE: AtomicU8 = AtomicU8::new(0);
static DECODE: AtomicU8 = AtomicU8::new(0);

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

/// Decodes on the device also when the weights stream from storage (`crate::stream`): the
/// resident layers stay wrapped, every other layer is wrapped per pass as it arrives and dropped
/// after (see `crate::gpu::decode::Graph::units`). Process-wide, read when a model builds its
/// plans. None restores the default (on, or `OXINFER_STREAM_GPU=0|1`); without device decoding
/// ([`decode_enabled`]) streamed models decode on the CPU either way.
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

static STREAM_DECODE: AtomicU8 = AtomicU8::new(0);

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
/// falls back to the CPU when no GPU backend can start; a device asked for by name that is missing
/// or cannot start is an error (and nothing is installed, so the caller can still choose another).
pub fn init(plat: &crate::platform::Platform, pref: crate::platform::DevicePref) -> anyhow::Result<Option<&'static dyn Backend>> {
    use crate::platform::DevicePref;
    let chosen: Option<Box<dyn Backend>> = match plat.pick(pref) {
        None => {
            if let DevicePref::Gpu(k) = pref {
                let why = if plat.notes.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", plat.notes.join("; "))
                };
                anyhow::bail!("device {}: not available{why}", k.as_str());
            }
            None
        }
        Some(g) => {
            let r: Result<Box<dyn Backend>, String> = match g.kind {
                #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
                crate::platform::GpuKind::Metal => crate::metal::MetalBackend::new()
                    .map(|b| Box::new(b) as Box<dyn Backend>)
                    .map_err(|e| e.to_string()),
                #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
                crate::platform::GpuKind::Cuda => crate::cuda::CudaBackend::new(g.index)
                    .map(|b| Box::new(b) as Box<dyn Backend>)
                    .map_err(|e| e.to_string()),
                _ => Err(format!("{} compute is not implemented in this build", g.kind.as_str())),
            };
            match r {
                Ok(b) => Some(b),
                Err(e) if pref != DevicePref::Auto => anyhow::bail!("device {}: {e}", g.kind.as_str()),
                Err(e) => {
                    if !e.contains("not implemented") {
                        eprintln!("[{}: {e}; using CPU]", g.kind.as_str());
                    }
                    None
                }
            }
        }
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
        // Detected, but with no compute backend.
        plat.gpus.push(GpuInfo {
            kind: GpuKind::Vulkan,
            index: 0,
            name: "test".into(),
            memory_bytes: None,
            driver: String::new(),
            unified_memory: false,
        });
        let e = init(&plat, DevicePref::Gpu(GpuKind::Vulkan)).err().expect("vulkan has no compute");
        assert!(e.to_string().contains("not implemented"), "{e}");
    }
}
