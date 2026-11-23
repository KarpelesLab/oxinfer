//! MiniMax-H3's transformer blocks on CUDA: a block stack ([`super::stack_dev::Source`]: the
//! DiT's, the video VAE decoder's) with the sequence kept on the device, kernels in `cuda.cu`
//! (compiled by NVRTC at first use). For FP8 "scaled" weights (`*_fp8_scaled.safetensors`: one
//! byte per weight, one scale per matrix) and float16 ones (ComfyUI's fp16 VAE), with or without
//! biases and LoRA side paths.
//!
//! Per block: the RMS norm and AdaLN modulation, the four projections and their low-rank side
//! paths, the per-head query / key norms and RoPE, the attention, SwiGLU and the gated or
//! scaled residuals. The host computes what comes before and after the blocks, and the AdaLN
//! tables.
//!
//! The attention of heads 64 or 128 wide is one kernel that walks the keys 64 at a time with a
//! running softmax, the scores staying in the accumulators (`attn_flash`); other widths take
//! three launches per head and query chunk (scores, softmax, product with the values).
//!
//! The products run on the tensor cores (f16 operands, f32 accumulation). FP8 and float16
//! weights are exact in a half. An f32 operand (activations, attention operands, LoRA factors
//! that are not exact in a half) goes in as two halves, `hi + lo 2^-11`, each product as two or
//! three tensor-core products: ~1e-6 relative instead of the ~5e-4 of plain f16 operands (which
//! `OXINFER_H3_GPU_EXACT=0` or `gpu_exact=0` select).
//!
//! Weights: as many blocks as fit stay on the device (a block of the FP8 transformer is 0.45 GB
//! with a turbo LoRA's factors, the 50 of them 22 GB); the others are copied into one of two
//! slots while the block before them runs. Activations are sized for the sequence (1.2 GB for
//! the transformer's 4682 rows of a 448x256 clip, 9.4 GB for the 37,834 of a 1344x768 one); the
//! feed-forward runs in row chunks and the attention keeps no scores (or runs in query chunks)
//! so that neither grows with the square of the sequence.
//!
//! Declined (the CPU path runs): other weight formats, q / k / v as separate matrices,
//! Hadamard-folded weights, heads that are not a multiple of 64 wide, widths that are not
//! multiples of 32, and a sequence whose activations do not fit (`OXINFER_GPU_PLAN_DEBUG=1` says
//! why; `OXINFER_H3_GPU=0` keeps the CPU path).

use super::stack_dev::{AdaIn, BlockSpec, Source, StackSpec, WSpec};
use crate::cuda::CudaBackend;
use anyhow::{Result, anyhow, bail};
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

dlsym::library! {
    /// The driver entry points this engine calls.
    pub struct Cuh {
        unsafe extern "C" {
            pub fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int;
            pub fn cuMemFree_v2(ptr: u64) -> c_int;
            pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
            pub fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> c_int;
            pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> c_int;
            pub fn cuStreamCreate(stream: *mut *mut c_void, flags: u32) -> c_int;
            pub fn cuStreamSynchronize(stream: *mut c_void) -> c_int;
            pub fn cuStreamDestroy_v2(stream: *mut c_void) -> c_int;
            pub fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> c_int;
            pub fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
            pub fn cuEventSynchronize(event: *mut c_void) -> c_int;
            pub fn cuEventDestroy_v2(event: *mut c_void) -> c_int;
            pub fn cuModuleGetFunction(f: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> c_int;
            pub fn cuModuleUnload(module: *mut c_void) -> c_int;
            pub fn cuFuncSetAttribute(f: *mut c_void, attrib: c_int, value: c_int) -> c_int;
            pub fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
            pub fn cuLaunchKernel(
                f: *mut c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32,
                shmem: u32, stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void,
            ) -> c_int;
        }
    }
}

type F = *mut c_void;

/// Pointers to the kernel arguments, for `cuLaunchKernel`.
macro_rules! args {
    ($($v:expr),* $(,)?) => { [$( &$v as *const _ as *mut c_void ),*] };
}

/// Kept free on the device beyond what the engine allocates.
const RESERVE: usize = 512 << 20;
/// The scores of one attention launch (queries x keys, f32), at most.
const SCORE_BYTES: usize = 256 << 20;
/// The weights of one group of a GEMM's outputs, at most (`OXINFER_H3_GPU_HOT`, in MB, overrides).
const HOT_BYTES: usize = 16 << 20;
/// Alignment of every piece of a block's weights.
const ALIGN: usize = 256;
/// "No such piece" among a block's offsets.
const NONE: u64 = u64::MAX;

fn debug(what: &str, why: &str) {
    if std::env::var("OXINFER_GPU_PLAN_DEBUG").is_ok_and(|v| v != "0") {
        eprintln!("[minimax-h3 cuda, {what}: {why}]");
    }
}

/// Split operands (see the module docs) unless `OXINFER_H3_GPU_EXACT=0` or `gpu_exact=0`.
fn exact() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| std::env::var("OXINFER_H3_GPU_EXACT").ok().map(|v| v != "0"));
    env.or_else(crate::gpu::exact_forced).unwrap_or(true)
}

/// `GemmP` of `cuda.cu`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmP {
    xh: u64,
    xl: u64,
    w: u64,
    wl: u64,
    rs: u64,
    bias: u64,
    y: u64,
    ys: u64,
    n_in: u32,
    n_out: u32,
    rows: u32,
    add: u32,
    f: f32,
    /// Tiles of 64 outputs per group ([`Engine::gemm`] sets it).
    cg: u32,
}

/// `AttP` of `cuda.cu`.
#[repr(C)]
#[derive(Clone, Copy)]
struct AttP {
    qh: u64,
    ql: u64,
    kh: u64,
    kl: u64,
    vh: u64,
    vl: u64,
    vs: u64,
    out: u64,
    os: u64,
    n: u32,
    nk: u32,
    scale: f32,
    pad: u32,
}

/// The shared memory of an `attn_flash` block for heads of `hd` (bytes): two planes each of 64
/// queries, of 64 keys or `hd` value rows, and of 64 rows of probabilities, and eight floats per
/// query.
fn flash_smem(hd: usize) -> u32 {
    let (q, p) = (64 * (hd / 2 + 4), 64 * 36);
    ((2 * q + 2 * q.max(hd * 36) + 2 * p + 8 * 64) * 4) as u32
}
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
const MAX_DYNAMIC_SHARED: c_int = 8;

/// A float matrix as half planes: `rows` of `cols / 2` words (two halves each; `cols` padded to a
/// multiple of 32 with zeros), the values times a power of two that brings the largest into
/// `[2^13, 2^14)`; `factor` undoes it. `lo`: the second halves (`(v - hi) 2^11`) when the
/// matrix is not exact in the first.
struct Planes {
    hi: Vec<u32>,
    lo: Option<Vec<u32>>,
    factor: f32,
}

fn to_planes(v: &[f32], rows: usize, cols: usize) -> Planes {
    use half::f16;
    assert_eq!(v.len(), rows * cols);
    let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let (f, factor) = if max > 0.0 && max.is_finite() {
        let s = (13 - max.log2().floor() as i32).clamp(-100, 100);
        (2f32.powi(s), 2f32.powi(-s))
    } else {
        (1.0, 1.0)
    };
    let per = cols.next_multiple_of(32) / 2;
    let pair = |r: usize, i: usize| -> (u32, u32) {
        let at = |c: usize| if c < cols { v[r * cols + c] * f } else { 0.0 };
        let (a, b) = (at(2 * i), at(2 * i + 1));
        let (ha, hb) = (f16::from_f32(a), f16::from_f32(b));
        let la = f16::from_f32((a - ha.to_f32()) * 2048.0);
        let lb = f16::from_f32((b - hb.to_f32()) * 2048.0);
        (
            ha.to_bits() as u32 | (hb.to_bits() as u32) << 16,
            la.to_bits() as u32 | (lb.to_bits() as u32) << 16,
        )
    };
    let mut hi = vec![0u32; rows * per];
    let mut lo = vec![0u32; rows * per];
    let threads = crate::pool::threads().clamp(1, 32);
    let rows_per = rows.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (t, (h, l)) in hi.chunks_mut(rows_per * per).zip(lo.chunks_mut(rows_per * per)).enumerate() {
            let pair = &pair;
            s.spawn(move || {
                for (j, (hw, lw)) in h.iter_mut().zip(l.iter_mut()).enumerate() {
                    (*hw, *lw) = pair(t * rows_per + j / per, j % per);
                }
            });
        }
    });
    // Zero second halves (of either sign): the matrix is exact in the first.
    let has_lo = lo.iter().any(|&w| w & 0x7fff_7fff != 0);
    Planes {
        hi,
        lo: has_lo.then_some(lo),
        factor,
    }
}

/// `[cols, rows]` from `m` (`[rows, cols]`), across threads.
fn transposed(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(m.len(), rows * cols);
    let mut t = vec![0f32; rows * cols];
    let per = cols.div_ceil(crate::pool::threads().clamp(1, 32)).max(1);
    std::thread::scope(|s| {
        for (i, part) in t.chunks_mut(per * rows).enumerate() {
            s.spawn(move || {
                for (j, row) in part.chunks_mut(rows).enumerate() {
                    for (k, v) in row.iter_mut().enumerate() {
                        *v = m[k * cols + i * per + j];
                    }
                }
            });
        }
    });
    t
}

/// One low-rank side path on the host: `A` `[r, n_in]`, `B` `[n_out, r]`.
struct HostLora {
    a: Planes,
    b: Planes,
    r: usize,
    strength: crate::lora::Strength,
}

/// Planes on the device.
#[derive(Clone, Copy, Default)]
struct DPlanes {
    hi: u64,
    lo: u64,
    factor: f32,
}

#[derive(Clone, Copy, Default)]
struct DLora {
    a: DPlanes,
    b: DPlanes,
    r: usize,
}

/// One projection on the device: its weights (FP8 bytes, or float16: half planes as they are),
/// its bias, the factor on its outputs, its side paths.
#[derive(Clone, Default)]
struct DLinW {
    w: u64,
    fp8: bool,
    bias: u64,
    n_out: usize,
    n_in: usize,
    f: f32,
    lora: Vec<DLora>,
}

/// One block on the device: pointers, or offsets from the block's base ([`NONE`] for a piece the
/// block does not have).
#[derive(Clone, Default)]
struct DBlock {
    lins: [DLinW; 4],
    norm1: u64,
    norm2: u64,
    norm_q: u64,
    norm_k: u64,
    scale1: u64,
    scale2: u64,
}

impl DBlock {
    /// The pointers of this layout at `base` (0 for an absent piece).
    fn at(&self, base: u64) -> DBlock {
        let mut b = self.clone();
        let fix = |p: &mut u64| *p = if *p == NONE { 0 } else { *p + base };
        for l in b.lins.iter_mut() {
            fix(&mut l.w);
            fix(&mut l.bias);
            for d in l.lora.iter_mut() {
                for p in [&mut d.a, &mut d.b] {
                    fix(&mut p.hi);
                    fix(&mut p.lo);
                }
            }
        }
        for p in [
            &mut b.norm1,
            &mut b.norm2,
            &mut b.norm_q,
            &mut b.norm_k,
            &mut b.scale1,
            &mut b.scale2,
        ] {
            fix(p);
        }
        b
    }
}

fn bytes_of<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain numeric data viewed as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The pieces of a block in device order, each `Some(bytes)` or None for one it does not have:
/// per projection its weights, bias and side paths' planes (A, A's second halves, B, B's), then
/// the norm weights, the query / key norm weights and the residual scales.
fn pieces<'d>(b: &'d BlockSpec<'d>, host: &'d [Vec<HostLora>; 4]) -> Vec<Option<&'d [u8]>> {
    let mut v: Vec<Option<&[u8]>> = Vec::new();
    for (l, hl) in b.lins.iter().zip(host) {
        v.push(Some(match l.w {
            WSpec::F8(w, _) | WSpec::F16(w) => w,
        }));
        v.push(l.bias.map(bytes_of));
        for d in hl {
            for p in [&d.a, &d.b] {
                v.push(Some(bytes_of(&p.hi)));
                v.push(p.lo.as_deref().map(bytes_of));
            }
        }
    }
    v.push(Some(bytes_of(b.norm1)));
    v.push(Some(bytes_of(b.norm2)));
    v.push(b.norm_qk.map(|n| bytes_of(n.0)));
    v.push(b.norm_qk.map(|n| bytes_of(n.1)));
    v.push(b.scale.map(|n| bytes_of(n.0)));
    v.push(b.scale.map(|n| bytes_of(n.1)));
    v
}

/// The layout of a block (offsets from its base) and its size.
fn layout(b: &BlockSpec, host: &[Vec<HostLora>; 4]) -> (DBlock, usize) {
    let mut o = 0usize;
    let offs: Vec<u64> = pieces(b, host)
        .iter()
        .map(|p| match p {
            Some(p) => {
                let at = o as u64;
                o += p.len().next_multiple_of(ALIGN);
                at
            }
            None => NONE,
        })
        .collect();
    let mut it = offs.into_iter();
    let mut next = || it.next().unwrap();
    let mut d = DBlock::default();
    for ((l, hl), dl) in b.lins.iter().zip(host).zip(d.lins.iter_mut()) {
        let (fp8, f) = match l.w {
            // The halves hold the FP8 values times 2^-8.
            WSpec::F8(_, scale) => (true, 256.0 * scale),
            WSpec::F16(_) => (false, 1.0),
        };
        *dl = DLinW {
            w: next(),
            fp8,
            bias: next(),
            n_out: l.n_out,
            n_in: l.n_in,
            f,
            lora: Vec::new(),
        };
        for hlo in hl {
            let mut pl = |p: &Planes| DPlanes {
                hi: next(),
                lo: next(),
                factor: p.factor,
            };
            let a = pl(&hlo.a);
            let b = pl(&hlo.b);
            dl.lora.push(DLora { a, b, r: hlo.r });
        }
    }
    d.norm1 = next();
    d.norm2 = next();
    d.norm_q = next();
    d.norm_k = next();
    d.scale1 = next();
    d.scale2 = next();
    (d, o)
}

/// Whether the engine has kernels for `b` of a stack of `s`.
fn check_block(s: &StackSpec, b: &BlockSpec) -> Result<()> {
    let (h, inner, ffn) = (s.hidden, s.heads * s.head_dim, s.ffn);
    let shapes = [(3 * inner, h), (h, inner), (2 * ffn, h), (h, ffn)];
    for (l, (n_out, n_in)) in b.lins.iter().zip(shapes) {
        let bytes = match l.w {
            WSpec::F8(w, _) => w.len(),
            WSpec::F16(w) => w.len() / 2,
        };
        if l.n_out != n_out || l.n_in != n_in || bytes != n_out * n_in || l.bias.is_some_and(|b| b.len() != n_out) {
            bail!("a projection of [{}, {}] where [{n_out}, {n_in}] is expected", l.n_out, l.n_in);
        }
    }
    let hd = s.head_dim;
    if b.norm1.len() != h
        || b.norm2.len() != h
        || b.norm_qk.is_some_and(|n| n.0.len() != hd || n.1.len() != hd)
        || b.scale.is_some_and(|n| n.0.len() != h || n.1.len() != h)
    {
        bail!("norm weights or residual scales of unexpected sizes");
    }
    Ok(())
}

/// The activation buffers and the sequence length they hold.
#[derive(Default)]
struct Arena {
    /// The capacities the buffers were made with ([`Engine::sizes`]) and their pointers.
    caps: Vec<usize>,
    all: Vec<u64>,
}

/// The buffers of [`Arena::all`], in the order of [`Engine::sizes`].
#[derive(Clone, Copy)]
struct Bufs {
    x: u64,
    hn: u64,
    qp: [u64; 2],
    kp: [u64; 2],
    vp: [u64; 2],
    xs: [u64; 2],
    us: [u64; 2],
    u: u64,
    rs: u64,
    urs: u64,
    prs: u64,
    vs: u64,
    cos: u64,
    sin: u64,
    key: u64,
    tabs: u64,
    big: u64,
}

impl Arena {
    fn bufs(&self) -> Bufs {
        let p = &self.all;
        Bufs {
            x: p[0],
            hn: p[1],
            qp: [p[2], p[3]],
            kp: [p[4], p[5]],
            vp: [p[6], p[7]],
            xs: [p[8], p[9]],
            us: [p[10], p[11]],
            u: p[12],
            rs: p[13],
            urs: p[14],
            prs: p[15],
            vs: p[16],
            cos: p[17],
            sin: p[18],
            key: p[19],
            tabs: p[20],
            big: p[21],
        }
    }
}

struct State {
    arena: Arena,
    /// Blocks kept on the device (base pointers), and the two slots of the others.
    resident: Vec<Option<u64>>,
    /// Whether a kept block's weights are there yet (copied when the block first runs, while
    /// the block before it computes).
    filled: Vec<bool>,
    placed: bool,
    slots: [u64; 2],
    events: [F; 2],
}

struct Funcs {
    gemm_f8: [F; 2],
    gemm_h: [F; 3],
    split_rows: F,
    norm_mod: F,
    resid: F,
    swiglu: F,
    qk_prep: F,
    v_scale: F,
    v_planes: F,
    softmax_split: F,
    attn_flash: F,
}

pub(super) struct Engine {
    what: &'static str,
    x: Cuh,
    ctx: *mut c_void,
    module: F,
    stream: F,
    f: Funcs,
    spec: StackSpec,
    /// Per block: its layout (offsets from the block's base), its bytes on the device, and its
    /// side paths (per projection).
    blocks: Vec<(DBlock, usize)>,
    loras: Vec<[Vec<HostLora>; 4]>,
    block_cap: usize,
    rmax: usize,
    state: Mutex<State>,
    failed: AtomicBool,
    prof: bool,
    /// The scores of one attention launch (bytes) and the rows of a feed-forward chunk, at most.
    score_bytes: usize,
    chunk_rows: usize,
    /// The attention runs without a score matrix (`attn_flash`: heads 64 or 128 wide, and a device that
    /// gives a block its shared memory; `OXINFER_H3_GPU_FLASH=0` keeps the three-launch form).
    flash: bool,
    /// The weights of one group of a GEMM's outputs (bytes), at most.
    hot: usize,
}

// SAFETY: the raw pointers are driver handles; every use binds the context and goes through
// `state`'s lock.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if self.x.cuCtxSetCurrent(self.ctx) != 0 {
                return;
            }
            self.x.cuStreamSynchronize(self.stream);
            let st = self.state.get_mut().unwrap();
            for p in st.arena.all.drain(..).chain(st.resident.drain(..).flatten()).chain(st.slots) {
                if p != 0 {
                    self.x.cuMemFree_v2(p);
                }
            }
            for e in st.events {
                if !e.is_null() {
                    self.x.cuEventDestroy_v2(e);
                }
            }
            self.x.cuStreamDestroy_v2(self.stream);
            self.x.cuModuleUnload(self.module);
        }
    }
}

fn ceil(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

impl Engine {
    /// The engine for `src` on the active backend, when that is CUDA and the stack is one it
    /// takes (see the module docs). `what` names the stack in messages.
    pub(super) fn new(src: &dyn Source, what: &'static str) -> Option<Engine> {
        if std::env::var("OXINFER_H3_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        let be = crate::gpu::active()?.as_any()?.downcast_ref::<CudaBackend>()?;
        match Self::build(be, src, what) {
            Ok(e) => Some(e),
            Err(e) => {
                debug(what, &format!("{e:#}"));
                None
            }
        }
    }

    fn build(be: &CudaBackend, src: &dyn Source, what: &'static str) -> Result<Engine> {
        let spec = src.spec();
        let (h, inner, ffn, hd) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn, spec.head_dim);
        if !hd.is_multiple_of(64) || [h, inner, ffn].iter().any(|d| !d.is_multiple_of(32)) || 2 * spec.half > hd || src.n_blocks() == 0 {
            bail!("widths {h} / {inner} / {ffn} or heads of {hd} have no kernel");
        }
        // Layouts and side paths.
        let t0 = crate::clock::Instant::now();
        let mut blocks = Vec::with_capacity(src.n_blocks());
        let mut loras = Vec::with_capacity(src.n_blocks());
        let (mut block_cap, mut rmax) = (0usize, 0usize);
        for bi in 0..src.n_blocks() {
            let b = src
                .block(bi)
                .ok_or_else(|| anyhow!("block {bi}: weights in a format without a kernel"))?;
            check_block(&spec, &b)?;
            let mut host: [Vec<HostLora>; 4] = Default::default();
            for (l, hl) in b.lins.iter().zip(host.iter_mut()) {
                for d in l.lora.into_iter().flat_map(|d| d.links()) {
                    let (a, bt, r) = d.factors();
                    if r == 0 {
                        continue;
                    }
                    let bm = transposed(bt, r, l.n_out);
                    rmax = rmax.max(r.next_multiple_of(32));
                    hl.push(HostLora {
                        a: to_planes(a, r, l.n_in),
                        b: to_planes(&bm, l.n_out, r),
                        r,
                        strength: d.strength_handle(),
                    });
                }
            }
            let (lay, bytes) = layout(&b, &host);
            block_cap = block_cap.max(bytes);
            blocks.push((lay, bytes));
            loras.push(host);
        }
        debug(
            what,
            &format!("layouts and side paths as halves in {:.2} s", t0.elapsed().as_secs_f64()),
        );
        let x: Cuh = {
            use dlsym::Bindings;
            // SAFETY: libcuda is already initialised by the backend.
            unsafe { Cuh::open_first(crate::platform::gpu::cuda::CANDIDATES.iter().copied()) }.map_err(|e| anyhow!("libcuda: {e}"))?
        };
        let ctx = be.context().ok_or_else(|| anyhow!("no context"))?;
        let module = be.cached_module(include_str!("cuda.cu"), "minimax-h3-{sm}", "minimax_h3.cu")?;
        let get = |name: &str| -> Result<F> {
            let cn = CString::new(name)?;
            let mut f = std::ptr::null_mut();
            if unsafe { x.cuModuleGetFunction(&mut f, module, cn.as_ptr()) } != 0 {
                bail!("kernel {name} missing");
            }
            Ok(f)
        };
        let f = Funcs {
            gemm_f8: [get("gemm_f8_1")?, get("gemm_f8_2")?],
            gemm_h: [get("gemm_h_1")?, get("gemm_h_2")?, get("gemm_h_3")?],
            split_rows: get("split_rows")?,
            norm_mod: get("norm_mod")?,
            resid: get("resid")?,
            swiglu: get("swiglu")?,
            qk_prep: get("qk_prep")?,
            v_scale: get("v_scale")?,
            v_planes: get("v_planes")?,
            softmax_split: get("softmax_split")?,
            attn_flash: get(if hd == 128 { "attn_flash_128" } else { "attn_flash_64" })?,
        };
        let flash = (hd == 64 || hd == 128)
            && !std::env::var("OXINFER_H3_GPU_FLASH").is_ok_and(|v| v == "0")
            && unsafe { x.cuFuncSetAttribute(f.attn_flash, MAX_DYNAMIC_SHARED, flash_smem(hd) as c_int) } == 0;
        let mut stream = std::ptr::null_mut();
        // Non-blocking: host copies do not wait for the kernels in flight.
        if unsafe { x.cuStreamCreate(&mut stream, 1) } != 0 {
            bail!("cuStreamCreate failed");
        }
        let mut events = [std::ptr::null_mut(); 2];
        for e in events.iter_mut() {
            if unsafe { x.cuEventCreate(e, 0) } != 0 {
                bail!("cuEventCreate failed");
            }
        }
        let n_blocks = blocks.len();
        Ok(Engine {
            what,
            x,
            ctx,
            module,
            stream,
            f,
            spec,
            blocks,
            loras,
            block_cap,
            rmax,
            state: Mutex::new(State {
                arena: Arena::default(),
                resident: vec![None; n_blocks],
                filled: vec![false; n_blocks],
                placed: false,
                slots: [0; 2],
                events,
            }),
            failed: AtomicBool::new(false),
            // The stages of one stack: the transformer's, or the VAE decoder's with `vae`.
            prof: std::env::var("OXINFER_H3_GPU_PROF").is_ok_and(|v| v != "0" && (v == "vae") == (what != "transformer")),
            score_bytes: SCORE_BYTES,
            chunk_rows: usize::MAX,
            flash,
            hot: std::env::var("OXINFER_H3_GPU_HOT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map_or(HOT_BYTES, |mb| mb << 20),
        })
    }

    /// Whether the attention runs as `attn_flash` (split operands only).
    fn flash_on(&self) -> bool {
        self.flash && exact()
    }

    fn alloc(&self, bytes: usize) -> Option<u64> {
        let mut d = 0u64;
        (unsafe { self.x.cuMemAlloc_v2(&mut d, bytes.max(ALIGN)) } == 0).then_some(d)
    }

    fn free_mem(&self) -> usize {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe { self.x.cuMemGetInfo_v2(&mut free, &mut total) };
        crate::cuda::usable_free(free, total)
    }

    /// Copies to the device and returns once the bytes are there: `cuMemcpyHtoD` may return while
    /// a copy of 64 KB or less from pageable memory is still on its way, ordered on the default
    /// stream only, which the kernels' stream (`CU_STREAM_NON_BLOCKING`) does not wait for.
    fn htod(&self, dst: u64, src: &[u8]) -> bool {
        crate::cuda::touch_before_upload(src);
        src.is_empty()
            || unsafe {
                self.x.cuMemcpyHtoD_v2(dst, src.as_ptr() as *const c_void, src.len()) == 0
                    && self.x.cuStreamSynchronize(std::ptr::null_mut()) == 0
            }
    }

    /// Copies block `bi` to `base`.
    fn upload_block(&self, src: &dyn Source, bi: usize, base: u64) -> bool {
        let Some(b) = src.block(bi) else { return false };
        let mut o = 0usize;
        for p in pieces(&b, &self.loras[bi]).into_iter().flatten() {
            if !self.htod(base + o as u64, p) {
                return false;
            }
            o += p.len().next_multiple_of(ALIGN);
        }
        o == self.blocks[bi].1
    }

    /// The sizes of the activation buffers for `n` rows (in the order of [`Arena::bufs`]), the
    /// key count padded to 32, the query rows per attention launch and the rows per feed-forward
    /// chunk.
    fn sizes(&self, n: usize, tab_floats: usize) -> ([usize; 22], usize, usize, usize) {
        let s = &self.spec;
        let (h, inner, ffn, hd, heads) = (s.hidden, s.heads * s.head_dim, s.ffn, s.head_dim, s.heads);
        let nk = n.next_multiple_of(32);
        let n64 = n.next_multiple_of(64);
        let qc = (self.score_bytes / (nk * 4) / 64 * 64).clamp(64, n64).min(n);
        // The largest of: q, k, v of every row; the attention's output, one launch's scores and
        // their planes; a feed-forward chunk's gate / value rows and activations.
        let a = n * 3 * inner * 4;
        let b = n * inner * 4 + if self.flash_on() { 0 } else { 2 * qc * nk * 4 };
        let rc = (a.max(b) / (3 * ffn * 4) / 64 * 64).clamp(64, n64).min(n).min(self.chunk_rows);
        let big = a.max(b).max(rc * 3 * ffn * 4);
        let plane = heads * nk * hd * 2;
        let xs = (n * h.max(inner)).max(rc * ffn.max(h)) * 2;
        let rm = self.rmax.max(32);
        let rope = n * s.half * 4;
        let sizes = [
            n * h * 4,
            n * h * 4,
            plane,
            plane,
            plane,
            plane,
            plane,
            plane,
            xs,
            xs,
            n * rm * 2,
            n * rm * 2,
            n * rm * 4,
            n * 4,
            n * 4,
            n * 4,
            heads * 4,
            rope,
            rope,
            n * 4,
            self.blocks.len() * tab_floats * 4,
            big,
        ];
        (sizes, nk, qc, rc)
    }

    /// Makes the buffers hold `n` rows (made again, larger, when they do not).
    fn make_arena(&self, st: &mut State, n: usize, tab_floats: usize) -> bool {
        // Room for the AdaLN sets of four timesteps and three modalities at least (their number
        // changes from one step to the next).
        let (need, ..) = self.sizes(n, tab_floats.max(72 * self.spec.hidden * usize::from(tab_floats > 0)));
        if st.arena.caps.len() == need.len() && need.iter().zip(&st.arena.caps).all(|(n, c)| n <= c) {
            return true;
        }
        unsafe { self.x.cuStreamSynchronize(self.stream) };
        let caps: Vec<usize> = match st.arena.caps.len() == need.len() {
            true => need.iter().zip(&st.arena.caps).map(|(&n, &c)| n.max(c)).collect(),
            false => need.to_vec(),
        };
        for p in st.arena.all.drain(..) {
            unsafe { self.x.cuMemFree_v2(p) };
        }
        st.arena = Arena::default();
        let total: usize = caps.iter().sum();
        // Blocks kept on the device make room for a longer sequence.
        if st.placed && self.free_mem() < total + RESERVE {
            self.release_blocks(st);
        }
        // (The blocks' two slots are there already when blocks are.)
        let slots = if st.placed { 0 } else { 2 * self.block_cap };
        if self.free_mem() < total + RESERVE + slots {
            debug(
                self.what,
                &format!(
                    "{n} rows need {:.1} GB of activations, {:.1} GB are free",
                    total as f64 / 1e9,
                    self.free_mem() as f64 / 1e9
                ),
            );
            return false;
        }
        let mut ptrs = Vec::with_capacity(caps.len());
        for &s in &caps {
            match self.alloc(s) {
                Some(p) => ptrs.push(p),
                None => {
                    for p in ptrs {
                        unsafe { self.x.cuMemFree_v2(p) };
                    }
                    debug(self.what, "out of device memory for the activations");
                    return false;
                }
            }
        }
        debug(
            self.what,
            &format!(
                "{n} rows of {} ({} heads of {}, feed-forward {}): {:.2} GB of activations {caps:?}",
                self.spec.hidden,
                self.spec.heads,
                self.spec.head_dim,
                self.spec.ffn,
                total as f64 / 1e9
            ),
        );
        st.arena = Arena { caps, all: ptrs };
        true
    }

    fn release_blocks(&self, st: &mut State) {
        unsafe { self.x.cuStreamSynchronize(self.stream) };
        for p in st.resident.iter_mut().filter_map(|r| r.take()).chain(std::mem::take(&mut st.slots)) {
            if p != 0 {
                unsafe { self.x.cuMemFree_v2(p) };
            }
        }
        st.filled.fill(false);
        st.placed = false;
    }

    /// Room on the device for as many blocks as fit beside two slots for the others.
    fn place_blocks(&self, st: &mut State) -> bool {
        if st.placed {
            return true;
        }
        let total: usize = self.blocks.iter().map(|b| b.1).sum();
        let all_fit = self.free_mem() >= total + RESERVE;
        for bi in 0..self.blocks.len() {
            let bytes = self.blocks[bi].1;
            let need = bytes + RESERVE + if all_fit { 0 } else { 2 * self.block_cap };
            if self.free_mem() < need {
                break;
            }
            let Some(p) = self.alloc(bytes) else { break };
            st.resident[bi] = Some(p);
        }
        if st.resident.iter().any(|r| r.is_none()) {
            for s in st.slots.iter_mut() {
                match self.alloc(self.block_cap) {
                    Some(p) => *s = p,
                    None => return false,
                }
            }
        }
        st.placed = true;
        debug(
            self.what,
            &format!(
                "{} of {} blocks on the device, {:.1} GB free",
                st.resident.iter().filter(|r| r.is_some()).count(),
                self.blocks.len(),
                self.free_mem() as f64 / 1e9
            ),
        );
        true
    }

    fn go(&self, f: F, grid: (u32, u32), block: u32, args: &mut [*mut c_void]) {
        self.go_shared(f, grid, block, 0, args)
    }

    /// A launch with `shmem` bytes of dynamic shared memory per block.
    fn go_shared(&self, f: F, grid: (u32, u32), block: u32, shmem: u32, args: &mut [*mut c_void]) {
        let rc = unsafe {
            self.x.cuLaunchKernel(
                f,
                grid.0.max(1),
                grid.1.max(1),
                1,
                block,
                1,
                1,
                shmem,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            self.failed.store(true, Relaxed);
        }
    }

    /// Ends a stage of the profile (`OXINFER_H3_GPU_PROF=1`, or `vae` for the VAE decoder's
    /// blocks: waits for the device).
    fn mark(&self, t: &mut crate::clock::Instant, k: usize) {
        if self.prof {
            unsafe { self.x.cuStreamSynchronize(self.stream) };
            super::dit::stage::add(k, t.elapsed());
            *t = crate::clock::Instant::now();
        }
    }

    /// Launches a GEMM over weights of `w_row` bytes per output: the outputs go in groups whose
    /// weights fit [`Engine::hot`], every row passing one group before the next (the weights of
    /// a projection are 77 to 154 MB, the cache a fraction of that: in one group, each tile of
    /// 64 rows would read them all from the device's memory again).
    fn gemm(&self, f: F, mut p: GemmP, w_row: usize) {
        let (ct, rt) = (ceil(p.n_out as usize, 64), ceil(p.rows as usize, 64));
        let mut cg = ((self.hot / (64 * w_row.max(1))) as u32).clamp(1, ct);
        // (Weights of up to three groups stay in one: measured faster at 38 MB.)
        if rt * ct.div_ceil(cg) > 65535 || ct <= 3 * cg {
            cg = ct;
        }
        p.cg = cg;
        self.go(f, (cg, rt * ct.div_ceil(cg)), 128, &mut args![p]);
    }

    /// `y (= or +=) f rs W x (+ bias)` with the planes of `w` (`[n_out][n_in]`).
    #[allow(clippy::too_many_arguments)]
    fn gemm_h(
        &self,
        xs: [u64; 2],
        w: DPlanes,
        bias: u64,
        y: u64,
        ys: usize,
        n_in: usize,
        n_out: usize,
        rows: usize,
        add: bool,
        f: f32,
        rs: u64,
    ) {
        let np = if !exact() {
            1
        } else if w.lo != 0 {
            3
        } else {
            2
        };
        let p = GemmP {
            xh: xs[0],
            xl: xs[1],
            w: w.hi,
            wl: w.lo,
            rs,
            bias,
            y,
            ys: ys as u64,
            n_in: n_in as u32,
            n_out: n_out as u32,
            rows: rows as u32,
            add: add as u32,
            f,
            cg: 0,
        };
        self.gemm(self.f.gemm_h[np - 1], p, n_in * 2 * (np - 1).max(1));
    }

    /// `y = W x + bias` of projection `l` and its side paths over `rows` rows whose planes are
    /// in `xs` (row scales in `rs`).
    fn lin(&self, bf: &Bufs, l: &DLinW, strengths: &[f32], y: u64, ys: usize, rows: usize) {
        if l.fp8 {
            let p = GemmP {
                xh: bf.xs[0],
                xl: bf.xs[1],
                w: l.w,
                wl: 0,
                rs: bf.rs,
                bias: l.bias,
                y,
                ys: ys as u64,
                n_in: l.n_in as u32,
                n_out: l.n_out as u32,
                rows: rows as u32,
                add: 0,
                f: l.f,
                cg: 0,
            };
            self.gemm(self.f.gemm_f8[exact() as usize], p, l.n_in);
        } else {
            let w = DPlanes {
                hi: l.w,
                lo: 0,
                factor: 1.0,
            };
            self.gemm_h(bf.xs, w, l.bias, y, ys, l.n_in, l.n_out, rows, false, l.f, bf.rs);
        }
        for (d, &s) in l.lora.iter().zip(strengths) {
            if s == 0.0 {
                continue;
            }
            let rp = d.r.next_multiple_of(32);
            // u = A x, then y += s B u.
            self.gemm_h(bf.xs, d.a, 0, bf.u, rp, l.n_in, d.r, rows, false, d.a.factor, bf.rs);
            self.split(bf.u, rp, rp, rows, bf.us, bf.urs);
            self.gemm_h(bf.us, d.b, 0, y, ys, rp, l.n_out, rows, true, d.b.factor * s, bf.urs);
        }
    }

    /// The planes and row scales of `rows` rows of `k` values (`xs` floats apart).
    fn split(&self, x: u64, xs: usize, k: usize, rows: usize, planes: [u64; 2], rs: u64) {
        self.go(
            self.f.split_rows,
            (rows as u32, 1),
            256,
            &mut args![x, xs as u64, k as u32, planes[0], planes[1], rs],
        );
    }

    fn run(&self, src: &dyn Source, x: &mut [f32], ada: Option<AdaIn>, cos: &[f32], sin: &[f32]) -> bool {
        let s = self.spec;
        let (h, inner, ffn, hd, heads) = (s.hidden, s.heads * s.head_dim, s.ffn, s.head_dim, s.heads);
        let n = x.len() / h;
        if n == 0
            || self.failed.load(Relaxed)
            || src.n_blocks() != self.blocks.len()
            || cos.len() != n * s.half
            || sin.len() != n * s.half
            || ada.is_some_and(|a| a.tables.len() != self.blocks.len() || a.key.len() != n)
        {
            return false;
        }
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        if unsafe { self.x.cuCtxSetCurrent(self.ctx) } != 0 {
            return false;
        }
        let tab_floats = ada.map_or(0, |a| a.tables[0].len());
        if !self.make_arena(st, n, tab_floats) || !self.place_blocks(st) {
            self.release_blocks(st);
            return false;
        }
        let bf = st.arena.bufs();
        let (_, nk, qc, rc) = self.sizes(n, tab_floats);
        let mut ok = self.htod(bf.x, bytes_of(x)) && self.htod(bf.cos, bytes_of(cos)) && self.htod(bf.sin, bytes_of(sin));
        if let Some(a) = ada {
            let keys: Vec<u32> = a.key.iter().map(|&k| k as u32).collect();
            ok = ok && self.htod(bf.key, bytes_of(&keys));
            for (bi, t) in a.tables.iter().enumerate() {
                ok = ok && t.len() == tab_floats && self.htod(bf.tabs + (bi * tab_floats * 4) as u64, bytes_of(t));
            }
        }
        if !ok {
            return false;
        }
        // Where q, k, v of head 0 start in a row of the fused projection, and the heads' stride.
        let (offs, hstep) = if s.stacked {
            ([0, inner, 2 * inner], hd)
        } else {
            ([0, hd, 2 * hd], 3 * hd)
        };
        let scale = 1.0 / (hd as f32).sqrt();
        let (hu, nu, hdu) = (h as u32, n as u32, hd as u32);
        let np_att = if exact() { 3 } else { 1 };
        let mut slot = 0usize;
        let mut tm = crate::clock::Instant::now();
        use super::dit::stage;
        for bi in 0..self.blocks.len() {
            // The block's weights: on the device (copied there at its first run), or copied into
            // the slot the block before the last one used (once that one is done). Either copy
            // runs while the device computes the block before.
            let base = match st.resident[bi] {
                Some(p) => {
                    if !st.filled[bi] {
                        if !self.upload_block(src, bi, p) {
                            self.failed.store(true, Relaxed);
                            break;
                        }
                        st.filled[bi] = true;
                    }
                    p
                }
                None => {
                    let p = st.slots[slot];
                    unsafe { self.x.cuEventSynchronize(st.events[slot]) };
                    if !self.upload_block(src, bi, p) {
                        self.failed.store(true, Relaxed);
                        break;
                    }
                    p
                }
            };
            let d = self.blocks[bi].0.at(base);
            let strengths: Vec<Vec<f32>> = self.loras[bi]
                .iter()
                .map(|l| l.iter().map(|d| d.strength.get()).collect())
                .collect();
            // AdaLN: the block's table and every row's set; else no modulation, and the
            // residual scales as the one "set" of every row.
            let (tab, key) = match ada {
                Some(_) => (bf.tabs + (bi * tab_floats * 4) as u64, bf.key),
                None => (0, 0),
            };
            let gates = |which: u32| match ada {
                Some(_) => (tab, which),
                None => (if which == 2 { d.scale1 } else { d.scale2 }, 0),
            };
            self.mark(&mut tm, stage::IO);

            // Attention.
            self.go(
                self.f.norm_mod,
                (nu, 1),
                256,
                &mut args![bf.x, d.norm1, tab, key, bf.hn, hu, s.norm_eps, 0u32],
            );
            self.mark(&mut tm, stage::NORM);
            self.split(bf.hn, h, h, n, bf.xs, bf.rs);
            let qkv = bf.big;
            self.lin(&bf, &d.lins[0], &strengths[0], qkv, 3 * inner, n);
            self.mark(&mut tm, stage::QKV);
            let row = (3 * inner) as u64;
            for (part, (norm, planes)) in [(d.norm_q, bf.qp), (d.norm_k, bf.kp)].into_iter().enumerate() {
                self.go(
                    self.f.qk_prep,
                    (nu, heads as u32),
                    32,
                    &mut args![
                        qkv,
                        row,
                        offs[part] as u32,
                        hstep as u32,
                        norm,
                        bf.cos,
                        bf.sin,
                        s.half as u32,
                        hdu,
                        nu,
                        s.qk_eps,
                        planes[0],
                        planes[1]
                    ],
                );
            }
            self.go(
                self.f.v_scale,
                (heads as u32, 1),
                256,
                &mut args![qkv, row, offs[2] as u32, hstep as u32, hdu, nu, bf.vs],
            );
            self.go(
                self.f.v_planes,
                (ceil(nk / 2, 256), (heads * hd) as u32),
                256,
                &mut args![
                    qkv,
                    row,
                    offs[2] as u32,
                    hstep as u32,
                    hdu,
                    nu,
                    nk as u32,
                    bf.vs,
                    bf.vp[0],
                    bf.vp[1]
                ],
            );
            self.mark(&mut tm, stage::QK);
            // q, k, v are in their planes: the big buffer now holds the attention's output, one
            // launch's scores and their planes.
            let att = bf.big;
            let scores = att + (n * inner * 4) as u64;
            let ps = [scores + (qc * nk * 4) as u64, scores + (qc * nk * 6) as u64];
            let words = |rows: usize| (rows * (hd / 2) * 4) as u64;
            let flash = self.flash_on();
            if flash {
                let p = AttP {
                    qh: bf.qp[0],
                    ql: bf.qp[1],
                    kh: bf.kp[0],
                    kl: bf.kp[1],
                    vh: bf.vp[0],
                    vl: bf.vp[1],
                    vs: bf.vs,
                    out: att,
                    os: inner as u64,
                    n: nu,
                    nk: nk as u32,
                    scale,
                    pad: 0,
                };
                self.go_shared(self.f.attn_flash, (ceil(n, 64), heads as u32), 256, flash_smem(hd), &mut args![p]);
            }
            for q0 in (0..n).step_by(qc).filter(|_| !flash) {
                let qn = qc.min(n - q0);
                for hh in 0..heads {
                    let p = GemmP {
                        xh: bf.qp[0] + words(hh * n + q0),
                        xl: bf.qp[1] + words(hh * n + q0),
                        w: bf.kp[0] + words(hh * n),
                        wl: bf.kp[1] + words(hh * n),
                        rs: 0,
                        bias: 0,
                        y: scores,
                        ys: nk as u64,
                        n_in: hdu,
                        n_out: nu,
                        rows: qn as u32,
                        add: 0,
                        f: 1.0,
                        cg: 0,
                    };
                    self.gemm(self.f.gemm_h[np_att - 1], p, hd * 4);
                    self.go(
                        self.f.softmax_split,
                        (qn as u32, 1),
                        256,
                        &mut args![scores, nu, nk as u32, scale, ps[0], ps[1], bf.prs, bf.vs + (hh * 4) as u64],
                    );
                    let vw = (hh * hd * (nk / 2) * 4) as u64;
                    let p = GemmP {
                        xh: ps[0],
                        xl: ps[1],
                        w: bf.vp[0] + vw,
                        wl: bf.vp[1] + vw,
                        rs: bf.prs,
                        bias: 0,
                        y: att + ((q0 * inner + hh * hd) * 4) as u64,
                        ys: inner as u64,
                        n_in: nk as u32,
                        n_out: hdu,
                        rows: qn as u32,
                        add: 0,
                        f: 1.0,
                        cg: 0,
                    };
                    self.gemm(self.f.gemm_h[np_att - 1], p, nk * 4);
                }
            }
            self.mark(&mut tm, stage::ATTN);
            self.split(att, inner, inner, n, bf.xs, bf.rs);
            self.lin(&bf, &d.lins[1], &strengths[1], bf.hn, h, n);
            self.mark(&mut tm, stage::OUT);
            let (g, which) = gates(2);
            self.go(self.f.resid, (ceil(h, 256), nu), 256, &mut args![bf.x, bf.hn, g, key, hu, which]);
            self.mark(&mut tm, stage::RES);

            // Feed-forward, in row chunks.
            self.go(
                self.f.norm_mod,
                (nu, 1),
                256,
                &mut args![bf.x, d.norm2, tab, key, bf.hn, hu, s.norm_eps, 3u32],
            );
            self.mark(&mut tm, stage::NORM);
            let gu = bf.big;
            let act = gu + (rc * 2 * ffn * 4) as u64;
            for r0 in (0..n).step_by(rc) {
                let rn = rc.min(n - r0);
                let rows = bf.hn + (r0 * h * 4) as u64;
                self.split(rows, h, h, rn, bf.xs, bf.rs);
                self.lin(&bf, &d.lins[2], &strengths[2], gu, 2 * ffn, rn);
                self.mark(&mut tm, stage::FC1);
                self.go(
                    self.f.swiglu,
                    (ceil(rn * ffn, 256), 1),
                    256,
                    &mut args![gu, act, rn as u32, ffn as u32, s.gate_first as u32],
                );
                self.mark(&mut tm, stage::ACT);
                self.split(act, ffn, ffn, rn, bf.xs, bf.rs);
                // The chunk's rows of the normed input are in their planes no longer: its output
                // takes their place.
                self.lin(&bf, &d.lins[3], &strengths[3], rows, h, rn);
                self.mark(&mut tm, stage::FC2);
            }
            let (g, which) = gates(5);
            self.go(self.f.resid, (ceil(h, 256), nu), 256, &mut args![bf.x, bf.hn, g, key, hu, which]);
            self.mark(&mut tm, stage::RES);
            if st.resident[bi].is_none() {
                unsafe { self.x.cuEventRecord(st.events[slot], self.stream) };
                slot ^= 1;
            }
            if self.failed.load(Relaxed) {
                break;
            }
        }
        let done = unsafe { self.x.cuStreamSynchronize(self.stream) } == 0;
        if !done || self.failed.load(Relaxed) {
            // A failed launch or copy: this engine is not used again.
            self.failed.store(true, Relaxed);
            eprintln!("[minimax-h3 cuda, {}: a kernel or a copy failed; the CPU path runs]", self.what);
            return false;
        }
        let got = unsafe { self.x.cuMemcpyDtoH_v2(x.as_mut_ptr() as *mut c_void, bf.x, x.len() * 4) } == 0;
        if !got {
            self.failed.store(true, Relaxed);
        }
        if !self.prof && self.what == "transformer" {
            stage::add(stage::DEV, tm.elapsed());
        }
        got
    }
}

impl super::stack_dev::Engine for Engine {
    fn stack(&self, src: &dyn Source, x: &mut [f32], ada: Option<AdaIn>, cos: &[f32], sin: &[f32]) -> bool {
        self.run(src, x, ada, cos, sin)
    }

    fn release(&self) {
        let mut st = self.state.lock().unwrap();
        if unsafe { self.x.cuCtxSetCurrent(self.ctx) } != 0 {
            return;
        }
        self.release_blocks(&mut st);
        for p in st.arena.all.drain(..) {
            unsafe { self.x.cuMemFree_v2(p) };
        }
        st.arena = Arena::default();
    }
}

#[cfg(test)]
mod tests {
    use super::super::stack_dev::LinSpec;
    use super::*;
    use crate::lora::LoraDelta;

    fn backend() -> Option<&'static CudaBackend> {
        static B: std::sync::OnceLock<Option<CudaBackend>> = std::sync::OnceLock::new();
        B.get_or_init(|| CudaBackend::new(0).ok()).as_ref()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32, bias: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * scale + bias).collect()
        }
    }

    struct TLin {
        f8: Option<(Vec<u8>, f32)>,
        f16: Vec<u8>,
        bias: Option<Vec<f32>>,
        lora: Option<LoraDelta>,
        n_out: usize,
        n_in: usize,
        /// The layer as one f64 matrix (side paths merged).
        dense: Vec<f64>,
    }

    /// A random layer: FP8 with a scale (`fp8`) or float16, with a bias (`bias`) and `lora` side
    /// paths of rank 32 (the first exact in a half, the second not).
    fn tlin(r: &mut Rng, n_out: usize, n_in: usize, fp8: bool, bias: bool, lora: usize) -> TLin {
        let gain = (3.0 / n_in as f32).sqrt();
        let (mut f8, mut f16) = (None, Vec::new());
        let mut dense: Vec<f64>;
        if fp8 {
            // Every byte but the NaNs.
            let b: Vec<u8> = (0..n_out * n_in)
                .map(|_| {
                    let v = (r.next().abs() * 255.9) as u8;
                    if v & 0x7f == 0x7f { 0x33 } else { v }
                })
                .collect();
            let scale = gain / 100.0;
            dense = b.iter().map(|&v| crate::ggml::e4m3_to_f32(v) as f64 * scale as f64).collect();
            f8 = Some((b, scale));
        } else {
            let h: Vec<half::f16> = (0..n_out * n_in).map(|_| half::f16::from_f32(r.next() * gain)).collect();
            dense = h.iter().map(|v| v.to_f64()).collect();
            f16 = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        }
        let mut delta: Option<LoraDelta> = None;
        for i in 0..lora {
            let rank = 32;
            let (a, b): (Vec<f32>, Vec<f32>) = if i == 0 {
                (
                    (0..rank * n_in).map(|_| (r.next() * 64.0).round() / 512.0).collect(),
                    (0..n_out * rank).map(|_| (r.next() * 64.0).round() / 4096.0).collect(),
                )
            } else {
                (r.vec(rank * n_in, gain, 0.0), r.vec(n_out * rank, 0.05, 0.0))
            };
            for o in 0..n_out {
                for c in 0..n_in {
                    dense[o * n_in + c] += (0..rank).map(|k| b[o * rank + k] as f64 * a[k * n_in + c] as f64).sum::<f64>();
                }
            }
            let d = LoraDelta::fixed(a, &b, rank, n_in, n_out);
            match &mut delta {
                Some(first) => first.push(Box::new(d)),
                None => delta = Some(d),
            }
        }
        TLin {
            f8,
            f16,
            bias: bias.then(|| r.vec(n_out, 0.1, 0.0)),
            lora: delta,
            n_out,
            n_in,
            dense,
        }
    }

    struct TBlock {
        lins: [TLin; 4],
        norm1: Vec<f32>,
        norm2: Vec<f32>,
        norm_qk: Option<(Vec<f32>, Vec<f32>)>,
        scale: Option<(Vec<f32>, Vec<f32>)>,
    }

    struct TStack {
        spec: StackSpec,
        blocks: Vec<TBlock>,
    }

    impl Source for TStack {
        fn spec(&self) -> StackSpec {
            self.spec
        }
        fn n_blocks(&self) -> usize {
            self.blocks.len()
        }
        fn block(&self, i: usize) -> Option<BlockSpec<'_>> {
            let b = &self.blocks[i];
            fn lin(l: &TLin) -> LinSpec<'_> {
                LinSpec {
                    w: match &l.f8 {
                        Some((b, s)) => WSpec::F8(b, *s),
                        None => WSpec::F16(&l.f16),
                    },
                    bias: l.bias.as_deref(),
                    n_out: l.n_out,
                    n_in: l.n_in,
                    lora: l.lora.as_ref(),
                }
            }
            Some(BlockSpec {
                lins: [lin(&b.lins[0]), lin(&b.lins[1]), lin(&b.lins[2]), lin(&b.lins[3])],
                norm1: &b.norm1,
                norm2: &b.norm2,
                norm_qk: b.norm_qk.as_ref().map(|n| (&n.0[..], &n.1[..])),
                scale: b.scale.as_ref().map(|n| (&n.0[..], &n.1[..])),
            })
        }
    }

    /// The stack from its definition, in f64.
    fn reference(st: &TStack, x: &[f32], ada: Option<AdaIn>, cos: &[f32], sin: &[f32]) -> Vec<f64> {
        let s = st.spec;
        let (h, hd, heads, ffn, half) = (s.hidden, s.head_dim, s.heads, s.ffn, s.half);
        let inner = heads * hd;
        let n = x.len() / h;
        let mut x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let lin = |l: &TLin, v: &[f64]| -> Vec<f64> {
            let mut y = vec![0f64; n * l.n_out];
            for r in 0..n {
                for o in 0..l.n_out {
                    let w = &l.dense[o * l.n_in..(o + 1) * l.n_in];
                    let dot: f64 = w.iter().zip(&v[r * l.n_in..(r + 1) * l.n_in]).map(|(a, b)| a * b).sum();
                    y[r * l.n_out + o] = dot + l.bias.as_ref().map_or(0.0, |b| b[o] as f64);
                }
            }
            y
        };
        let rms = |v: &mut [f64], w: Option<&[f32]>, eps: f32| {
            let r = (v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64 + eps as f64).powf(-0.5);
            for (i, a) in v.iter_mut().enumerate() {
                *a *= r * w.map_or(1.0, |w| w[i] as f64);
            }
        };
        for (bi, b) in st.blocks.iter().enumerate() {
            let set = |r: usize, which: usize| ada.map(|a| &a.tables[bi][(a.key[r] * 6 + which) * h..(a.key[r] * 6 + which + 1) * h]);
            let normed = |x: &[f64], w: &[f32], which: usize| -> Vec<f64> {
                let mut hn = x.to_vec();
                for (r, row) in hn.chunks_mut(h).enumerate() {
                    rms(row, Some(w), s.norm_eps);
                    if let (Some(shift), Some(scale)) = (set(r, which), set(r, which + 1)) {
                        for i in 0..h {
                            row[i] = row[i] * (1.0 + scale[i] as f64) + shift[i] as f64;
                        }
                    }
                }
                hn
            };
            let residual = |x: &mut [f64], y: &[f64], which: usize, sc: Option<&Vec<f32>>| {
                for r in 0..n {
                    for i in 0..h {
                        let g = match (set(r, which), sc) {
                            (Some(g), _) => g[i],
                            (None, Some(g)) => g[i],
                            _ => 1.0,
                        };
                        x[r * h + i] += g as f64 * y[r * h + i];
                    }
                }
            };
            let f = lin(&b.lins[0], &normed(&x, &b.norm1, 0));
            // [part][head][row][hd]
            let mut qkv = vec![vec![vec![0f64; n * hd]; heads]; 3];
            for (part, of_part) in qkv.iter_mut().enumerate() {
                for (hh, of_head) in of_part.iter_mut().enumerate() {
                    for r in 0..n {
                        let o = if s.stacked { part * inner + hh * hd } else { (hh * 3 + part) * hd };
                        let dst = &mut of_head[r * hd..(r + 1) * hd];
                        dst.copy_from_slice(&f[r * 3 * inner + o..r * 3 * inner + o + hd]);
                        if part < 2 {
                            let w = b.norm_qk.as_ref().map(|nw| if part == 0 { &nw.0[..] } else { &nw.1[..] });
                            rms(dst, w, s.qk_eps);
                            for i in 0..half {
                                let (c, sn) = (cos[r * half + i] as f64, sin[r * half + i] as f64);
                                let (a, bb) = (dst[i], dst[i + half]);
                                dst[i] = a * c - bb * sn;
                                dst[i + half] = bb * c + a * sn;
                            }
                        }
                    }
                }
            }
            let mut att = vec![0f64; n * inner];
            for hh in 0..heads {
                for i in 0..n {
                    let q = &qkv[0][hh][i * hd..(i + 1) * hd];
                    let sc: Vec<f64> = (0..n)
                        .map(|j| q.iter().zip(&qkv[1][hh][j * hd..(j + 1) * hd]).map(|(a, b)| a * b).sum::<f64>() / (hd as f64).sqrt())
                        .collect();
                    let m = sc.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = sc.iter().map(|v| (v - m).exp()).collect();
                    let sum: f64 = e.iter().sum();
                    for j in 0..n {
                        for c in 0..hd {
                            att[i * inner + hh * hd + c] += e[j] / sum * qkv[2][hh][j * hd + c];
                        }
                    }
                }
            }
            let y = lin(&b.lins[1], &att);
            residual(&mut x, &y, 2, b.scale.as_ref().map(|v| &v.0));
            let gu = lin(&b.lins[2], &normed(&x, &b.norm2, 3));
            let mut act = vec![0f64; n * ffn];
            for r in 0..n {
                for j in 0..ffn {
                    let (a, c) = (gu[r * 2 * ffn + j], gu[r * 2 * ffn + ffn + j]);
                    let (g, u) = if s.gate_first { (a, c) } else { (c, a) };
                    act[r * ffn + j] = u * g / (1.0 + (-g).exp());
                }
            }
            let y = lin(&b.lins[3], &act);
            residual(&mut x, &y, 5, b.scale.as_ref().map(|v| &v.1));
        }
        x
    }

    /// Both kinds of stack (the DiT's: FP8, AdaLN, weighted query / key norms, `[q; k; v]`, side
    /// paths; the VAE decoder's: float16 with biases, scaled residuals, `[q k v]` per head)
    /// against their definition, whole and in query / row chunks.
    #[test]
    fn stacks_match_definition() {
        use super::super::stack_dev::Engine as _;
        let Some(be) = backend() else { return };
        let (h, heads, ffn, half, n) = (64usize, 2usize, 96usize, 24usize, 150usize);
        // Heads of 128 as the transformer's, of 64 as the VAE's.
        for (dit, hd) in [(true, 64usize), (false, 64), (true, 128)] {
            let inner = heads * hd;
            let mut r = Rng(if dit { 7 } else { 11 } + hd as u64);
            let blocks: Vec<TBlock> = (0..2)
                .map(|bi| TBlock {
                    lins: [
                        tlin(&mut r, 3 * inner, h, dit, !dit, if dit { 2 } else { 0 }),
                        tlin(&mut r, h, inner, dit, !dit, usize::from(dit && bi == 0)),
                        tlin(&mut r, 2 * ffn, h, dit, !dit, 0),
                        tlin(&mut r, h, ffn, dit, !dit, usize::from(dit)),
                    ],
                    norm1: r.vec(h, 0.2, 1.0),
                    norm2: r.vec(h, 0.2, 1.0),
                    norm_qk: dit.then(|| (r.vec(hd, 0.3, 1.5), r.vec(hd, 0.3, 1.5))),
                    scale: (!dit).then(|| (r.vec(h, 0.5, 0.0), r.vec(h, 0.5, 0.0))),
                })
                .collect();
            let st = TStack {
                spec: StackSpec {
                    hidden: h,
                    heads,
                    head_dim: hd,
                    ffn,
                    half,
                    norm_eps: 1e-5,
                    qk_eps: 1e-6,
                    stacked: dit,
                    gate_first: dit,
                },
                blocks,
            };
            let x = r.vec(n * h, 3.0, 0.0);
            let ang = r.vec(n * half, 3.0, 0.0);
            let (cos, sin): (Vec<f32>, Vec<f32>) = ang.iter().map(|a| (a.cos(), a.sin())).unzip();
            let tables: Vec<Vec<f32>> = (0..2).map(|_| r.vec(3 * 6 * h, 0.3, 0.0)).collect();
            let key: Vec<usize> = (0..n).map(|i| i % 3).collect();
            let ada = dit.then_some(AdaIn {
                tables: &tables,
                key: &key,
            });
            let want = reference(&st, &x, ada, &cos, &sin);
            let peak = want.iter().fold(0f64, |m, v| m.max(v.abs()));
            for (chunked, flash) in [(false, true), (true, true), (false, false), (true, false)] {
                let mut e = Engine::build(be, &st, "test").unwrap();
                if chunked {
                    (e.score_bytes, e.chunk_rows) = (1, 64);
                }
                assert!(e.flash, "no shared memory for the attention kernel");
                e.flash = flash;
                let mut got = x.clone();
                assert!(e.run(&st, &mut got, ada, &cos, &sin), "the device declined");
                let err = got.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max);
                assert!(
                    err < 2e-5 * peak,
                    "dit {dit}, heads of {hd}, chunked {chunked}, flash {flash}: max error {err:e} of {peak:e}"
                );
                // The same again after a release.
                e.release();
                let mut again = x.clone();
                assert!(e.run(&st, &mut again, ada, &cos, &sin));
                assert_eq!(got, again);
                // More AdaLN sets and fewer rows on the same buffers (the sets of a sampling
                // step depend on its timesteps).
                if dit {
                    let m = n - 37;
                    let tables: Vec<Vec<f32>> = (0..2).map(|_| r.vec(15 * 6 * h, 0.3, 0.0)).collect();
                    let key: Vec<usize> = (0..m).map(|i| i * 7 % 15).collect();
                    let ada = Some(AdaIn {
                        tables: &tables,
                        key: &key,
                    });
                    let (xm, cm, sm) = (&x[..m * h], &cos[..m * half], &sin[..m * half]);
                    let want = reference(&st, xm, ada, cm, sm);
                    let mut got = xm.to_vec();
                    assert!(e.run(&st, &mut got, ada, cm, sm));
                    let err = got.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max);
                    assert!(err < 2e-5 * peak, "15 sets, chunked {chunked}: max error {err:e} of {peak:e}");
                }
            }
        }
    }

    #[test]
    fn transposes() {
        let m: Vec<f32> = (0..3 * 70).map(|i| i as f32).collect();
        let t = transposed(&m, 3, 70);
        for r in 0..3 {
            for c in 0..70 {
                assert_eq!(t[c * 3 + r], m[r * 70 + c]);
            }
        }
    }

    #[test]
    fn planes_are_exact_or_split() {
        let exact: Vec<f32> = (0..96).map(|i| (i as f32 - 40.0) / 256.0).collect();
        let p = to_planes(&exact, 3, 32);
        assert!(p.lo.is_none());
        let back = |p: &Planes, i: usize| {
            let h = |w: u32, k: usize| half::f16::from_bits((w >> (16 * k)) as u16).to_f64();
            let lo = p.lo.as_ref().map_or(0.0, |l| h(l[i / 2], i % 2) / 2048.0);
            (h(p.hi[i / 2], i % 2) + lo) * p.factor as f64
        };
        for (i, &v) in exact.iter().enumerate() {
            assert_eq!(back(&p, i), v as f64);
        }
        // 40 columns pad to 64; values that need the second half.
        let odd: Vec<f32> = (0..80).map(|i| 1.0 / (i as f32 + 3.0)).collect();
        let p = to_planes(&odd, 2, 40);
        assert!(p.lo.is_some());
        assert_eq!(p.hi.len(), 2 * 32);
        for (i, &v) in odd.iter().enumerate() {
            let at = i / 40 * 64 + i % 40;
            assert!((back(&p, at) - v as f64).abs() < 3e-7 * v as f64, "{i}");
        }
    }
}
