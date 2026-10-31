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

/// Device-side state for one weight matrix (buffers / handles), owned by the layer.
pub type DeviceWeight = Box<dyn Any + Send + Sync>;

/// One matmul of a batch sharing the same input `x`: `y[t][n_out] = x[t][n_in] * W^T`.
pub struct Job<'a> {
    pub w: WeightRef<'a>,
    pub dev: &'a DeviceWeight,
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
    /// them apart), and the down projection `down[e]` (`[h, inter]`); the activation is GELU-tanh
    /// when `gelu` is set, else SiLU. None if the backend does not run MoE layers.
    #[allow(clippy::too_many_arguments)]
    fn upload_moe(
        &self,
        _gate_up: &[WeightRef],
        _up: Option<&[WeightRef]>,
        _down: &[WeightRef],
        _h: usize,
        _inter: usize,
        _gelu: bool,
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
    /// Compiles a single-token decode graph (see [`decode`]); None when the backend cannot run
    /// some op or weight of it (the model then decodes on the CPU).
    fn decode_plan<'s>(&'s self, _g: &decode::Graph) -> Option<Box<dyn decode::DecodePlan + 's>> {
        None
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

/// Picks and initialises a backend for the detected platform and the user's preference.
pub fn init(plat: &crate::platform::Platform, pref: crate::platform::DevicePref) -> Option<&'static dyn Backend> {
    use crate::platform::DevicePref;
    let chosen: Option<Box<dyn Backend>> = match plat.pick(pref) {
        None => None,
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
                Err(e) => {
                    if pref != DevicePref::Auto || !e.contains("not implemented") {
                        eprintln!("[{}: {e}; using CPU]", g.kind.as_str());
                    }
                    None
                }
            }
        }
    };
    install(chosen);
    active()
}
