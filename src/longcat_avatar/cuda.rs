//! LongCat-Video-Avatar's block stack on CUDA, with the sequence kept on the device. The kernels
//! are MiniMax-H3's (`minimax_h3/cuda.cu`: tensor-core GEMM with f16 operands and f32
//! accumulation, flash attention without a score matrix) plus `cuda.cu` of this folder: the
//! GEMM over int8 rows with a scale each (the int8
//! bytes are read as stored and are exact in a half), attention over a range of queries and a
//! range of keys (the conditioning frames' restricted self-attention, the text and the per-frame
//! audio cross-attention), layer norms with the per-frame AdaLN sets, and the query / key
//! preparation with adjacent-pair RoPE.
//!
//! Activations stay float32 between the kernels. Into a product they go as float16 after a power
//! of two per row (about 3e-4 relative on an operand; the reference computes everything in
//! bfloat16, whose mantissa is three bits shorter), or, with `OXINFER_AVATAR_GPU_EXACT=1`, as
//! two halves `hi + lo 2^-11` (two products per GEMM, three per attention product: float32
//! results to about 1e-6, at about twice the time).
//!
//! The default path is laid out for a GPU whose kernels wait on memory rather than on the
//! tensor cores (measured on a GB10, which shares the machine's memory: with the products
//! removed a GEMM took 90% of its time, with its reads removed 60%): GEMMs in tiles of 128 x 128
//! (half the reads per product of `gemm_body`'s 64 x 64), attention in blocks of 128 queries
//! (each read of a step's keys and values serves twice the queries), the weights of a GEMM in
//! groups of outputs that stay in the cache (12 MB on a GPU sharing the machine's memory), no
//! planes for the operands' second halves, and what follows a matrix as part of its GEMM's
//! output instead of a pass of its own over the activations: the LoRAs' side paths, the gated
//! residual, SwiGLU's product.
//!
//! Weights: `base_model_int8/` goes to the device byte for byte (0.33 GB a block); the bf16
//! `base_model/` as float16 (exact but for values under 2^-14; 0.66 GB a block). The LoRAs are
//! side paths (`u = A x`, `y += s B u` per output section) with their factors as float16 after a
//! power of two (28 MB a block and LoRA). As many blocks as fit stay on the device; the others
//! are copied into one of two slots while the block before them runs (MiniMax-H3's scheme), so a
//! 16 GB card streams most of the 48 blocks every step.
//!
//! Declined (the CPU path runs): several speakers (the reference-attention map and the 1-D
//! RoPE of the audio attention), `audio_prenorm`, heads that are not 64 or 128 wide, widths
//! that are not multiples of 32, a sequence whose activations do not fit
//! (`OXINFER_GPU_PLAN_DEBUG=1` says why; `OXINFER_AVATAR_GPU=0` keeps the CPU path).

use super::dit::{Block, Ctx, Dit};
use super::lora::LoraNet;
use super::weights::{Mat, Store};
use crate::cuda::CudaBackend;
use anyhow::{Result, anyhow, bail};
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

dlsym::library! {
    /// The driver entry points this engine calls.
    struct Cuh {
        unsafe extern "C" {
            fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int;
            fn cuMemFree_v2(ptr: u64) -> c_int;
            fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
            fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> c_int;
            fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> c_int;
            fn cuStreamCreate(stream: *mut *mut c_void, flags: u32) -> c_int;
            fn cuStreamSynchronize(stream: *mut c_void) -> c_int;
            fn cuStreamDestroy_v2(stream: *mut c_void) -> c_int;
            fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> c_int;
            fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
            fn cuEventSynchronize(event: *mut c_void) -> c_int;
            fn cuEventDestroy_v2(event: *mut c_void) -> c_int;
            fn cuModuleGetFunction(f: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> c_int;
            fn cuModuleUnload(module: *mut c_void) -> c_int;
            fn cuFuncSetAttribute(f: *mut c_void, attrib: c_int, value: c_int) -> c_int;
            fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
            fn cuLaunchKernel(
                f: *mut c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32,
                shmem: u32, stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void,
            ) -> c_int;
        }
    }
}

type F = *mut c_void;

macro_rules! args {
    ($($v:expr),* $(,)?) => { [$( &$v as *const _ as *mut c_void ),*] };
}

/// Kept free on the device beyond what the engine allocates.
const RESERVE: usize = 512 << 20;
/// The weights of one group of a GEMM's outputs, at most.
const HOT_BYTES: usize = 16 << 20;
const ALIGN: usize = 256;
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
const MAX_DYNAMIC_SHARED: c_int = 8;
/// The block's matrices, in the order of [`mats`].
const N_MATS: usize = 11;
/// The matrices LoRAs target, by their index in [`mats`] and their module name.
const LORA_TARGETS: [(usize, &str); 7] = [
    (0, "attn.qkv"),
    (1, "attn.proj"),
    (2, "cross_attn.q_linear"),
    (3, "cross_attn.kv_linear"),
    (8, "ffn.w1"),
    (9, "ffn.w3"),
    (10, "ffn.w2"),
];

fn debug(why: &str) {
    if std::env::var("OXINFER_GPU_PLAN_DEBUG").is_ok_and(|v| v != "0") {
        eprintln!("[longcat-avatar cuda: {why}]");
    }
}

fn mats<'b>(b: &'b Block<'static>) -> [&'b Mat<'static>; N_MATS] {
    [
        &b.qkv,
        &b.attn_proj,
        &b.cq,
        &b.ckv,
        &b.cproj,
        &b.aq,
        &b.akv,
        &b.aproj,
        &b.w1,
        &b.w3,
        &b.w2,
    ]
}

/// The shared memory of an `attn_flash` block of `fq` queries (as `flash_smem` of
/// `minimax_h3/cuda.rs`); `planes`: 2 with the operands' second halves, else 1.
fn flash_smem(hd: usize, fq: usize, planes: usize) -> u32 {
    let (q, b, p) = (fq * (hd / 2 + 4), (64 * (hd / 2 + 4)).max(hd * 36), fq * 36);
    ((planes * (q + b + p) + 8 * fq) * 4) as u32
}

/// `GemmP` of `minimax_h3/cuda.cu`.
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
    cg: u32,
}

/// `LcGemmP` of `cuda.cu`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LcGemmP {
    p: GemmP,
    cs: u64,
}

/// `LcFastP` of `cuda.cu`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LcFastP {
    p: GemmP,
    cs: u64,
    uh: [u64; 2],
    bw: [u64; 2],
    urs: [u64; 2],
    tab: u64,
    uf: [f32; 2],
    ur: u32,
    us: u32,
    mode: u32,
    s: u32,
    frame0: u32,
    stride: u32,
    gate_at: u32,
    row0: u32,
}

/// What a matrix's output becomes on the default path (`mode` of `LcFastP`).
#[derive(Clone, Copy)]
enum Out {
    /// `y = v`.
    Store,
    /// `y += gate v`, `y` the residual stream: the gate at `gate_at` of the row's frame in `tab`
    /// (`stride` floats per frame; row r is row `row0 + r` of frame `frame0 + (row0 + r) / s`),
    /// or 1 with `tab` 0.
    Resid {
        tab: u64,
        s: u32,
        frame0: u32,
        stride: u32,
        gate_at: u32,
        row0: u32,
    },
    /// `y = v silu(y)`: `y` holds the gate rows of the SwiGLU.
    Swiglu,
}

/// `LcAttP` of `cuda.cu` (`AttP`, then the ranges).
#[repr(C)]
#[derive(Clone, Copy)]
struct LcAttP {
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
    qlo: u32,
    qhi: u32,
    klo: u32,
    khi: u32,
}

/// A float matrix as float16 after a power of two that brings its largest value into
/// `[2^13, 2^14)`: rows of `cols` padded to a multiple of 32 with zeros; and the factor that
/// undoes the power of two.
fn half_planes(v: &[f32], rows: usize, cols: usize) -> (Vec<u16>, f32) {
    use half::f16;
    let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let (f, factor) = if max > 0.0 && max.is_finite() {
        let s = (13 - max.log2().floor() as i32).clamp(-100, 100);
        (2f32.powi(s), 2f32.powi(-s))
    } else {
        (1.0, 1.0)
    };
    let cp = cols.next_multiple_of(32);
    let mut out = vec![0u16; rows * cp];
    let per = rows.div_ceil(crate::pool::threads().clamp(1, 32)).max(1);
    std::thread::scope(|s| {
        for (t, part) in out.chunks_mut(per * cp).enumerate() {
            s.spawn(move || {
                for (j, row) in part.chunks_mut(cp).enumerate() {
                    let r = t * per + j;
                    for (o, &x) in row.iter_mut().zip(&v[r * cols..(r + 1) * cols]) {
                        *o = f16::from_f32(x * f).to_bits();
                    }
                }
            });
        }
    });
    (out, factor)
}

/// A float matrix as stored, in float16 (`[n_out, n_in]`).
fn mat_halves(m: &Mat) -> Vec<u8> {
    use half::f16;
    let (n_out, n_in) = (m.n_out, m.n_in);
    let mut out = vec![0u8; n_out * n_in * 2];
    if let Store::Float("F16", b) = m.store {
        out.copy_from_slice(b);
        return out;
    }
    let per = n_out.div_ceil(crate::pool::threads().clamp(1, 32)).max(1);
    std::thread::scope(|s| {
        for (t, part) in out.chunks_mut(per * n_in * 2).enumerate() {
            s.spawn(move || {
                let rows = part.len() / (n_in * 2);
                match m.store {
                    Store::Float("BF16", b) => {
                        let src = &b[t * per * n_in * 2..];
                        for (o, c) in part.as_chunks_mut::<2>().0.iter_mut().zip(src.as_chunks::<2>().0) {
                            let v = f32::from_bits((u16::from_le_bytes(*c) as u32) << 16);
                            *o = f16::from_f32(v).to_bits().to_le_bytes();
                        }
                    }
                    _ => {
                        let mut buf = vec![0f32; rows * n_in];
                        m.rows(t * per, t * per + rows, &mut buf);
                        for (o, &v) in part.as_chunks_mut::<2>().0.iter_mut().zip(&buf) {
                            *o = f16::from_f32(v).to_bits().to_le_bytes();
                        }
                    }
                }
            });
        }
    });
    out
}

fn bytes_of<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain numeric slices.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// One side path on the device (offsets from the block's base).
#[derive(Clone, Default)]
struct DLora {
    /// Which of the engine's LoRAs.
    net: usize,
    a: u64,
    a_f: f32,
    /// Per output section: planes and factor.
    bs: Vec<(u64, f32)>,
    r: usize,
    alpha: f32,
}

/// One matrix on the device (offsets from the block's base).
#[derive(Clone, Default)]
struct DLin {
    w: u64,
    int8: bool,
    cs: u64,
    bias: u64,
    n_out: usize,
    n_in: usize,
    lora: Vec<DLora>,
}

/// One block's layout: its matrices, then a blob of everything small (scales, biases, norm
/// weights, LoRA factors), made once on the host.
#[derive(Default)]
struct BlockImg {
    lins: Vec<DLin>,
    /// Offsets into the block: q / k norm weights of the three attentions, the two affine norms.
    norms: [u64; 10],
    blob_at: usize,
    blob: Vec<u8>,
    bytes: usize,
}

#[derive(Default)]
struct Arena {
    caps: Vec<usize>,
    all: Vec<u64>,
}

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
    vs: u64,
    cos: u64,
    sin: u64,
    tabs: u64,
    big: u64,
    text: u64,
    tkv: u64,
    audio: u64,
    akv: u64,
    /// Floats per row of a side path's `u`, the halves per row of its planes (per section), and
    /// the rows a GEMM may have.
    u_stride: usize,
    u_rp: usize,
    rmax: usize,
}

impl Arena {
    /// `exact`: with the planes of the operands' second halves (else null: not allocated).
    fn bufs(&self, exact: bool) -> Bufs {
        let p = &self.all;
        let lo = |i: usize| if exact { p[i] } else { 0 };
        Bufs {
            x: p[0],
            hn: p[1],
            qp: [p[2], lo(3)],
            kp: [p[4], lo(5)],
            vp: [p[6], lo(7)],
            xs: [p[8], lo(9)],
            us: [p[10], lo(11)],
            u: p[12],
            rs: p[13],
            urs: p[14],
            vs: p[15],
            cos: p[16],
            sin: p[17],
            tabs: p[18],
            big: p[19],
            text: p[20],
            tkv: p[21],
            audio: p[22],
            akv: p[23],
            u_stride: 0,
            u_rp: 0,
            rmax: 0,
        }
    }
}

struct State {
    /// The LoRAs the layouts were made with (their addresses).
    nets: Vec<usize>,
    blocks: Vec<BlockImg>,
    block_cap: usize,
    /// Width of a row of the side paths' `u`, and of a section's planes (halves).
    u_stride: usize,
    u_rp: usize,
    /// Whether the LoRAs of each matrix have the same sections and rank (what the default path's
    /// GEMM takes as side paths).
    sides_ok: bool,
    arena: Arena,
    resident: Vec<Option<u64>>,
    filled: Vec<bool>,
    placed: bool,
    slots: [u64; 2],
    events: [F; 2],
}

struct Funcs {
    gemm_i8: F,
    gemm_h: F,
    /// The default path's GEMMs with side paths (int8 rows, float16 planes).
    gemm_sides: [F; 2],
    /// Rows and outputs per block of the two GEMMs, and its threads.
    tile: usize,
    threads: u32,
    /// The weights of a group of a GEMM's outputs, at most, and how many groups' worth of
    /// outputs still make one group.
    hot: usize,
    merge: u32,
    split_rows: F,
    swiglu: F,
    v_scale: F,
    v_planes: F,
    attn: F,
    ln_mod: F,
    ln_affine: F,
    resid: F,
    qk_prep: F,
    /// The queries and the shared memory of an attention block.
    fq: usize,
    smem: u32,
}

pub(super) struct Engine {
    x: Cuh,
    ctx: *mut c_void,
    module: F,
    stream: F,
    f: Funcs,
    state: Mutex<State>,
    failed: AtomicBool,
    prof: bool,
    /// Split operands (`OXINFER_AVATAR_GPU_EXACT`).
    exact: bool,
    /// What follows a matrix (side paths, residual, SwiGLU) as part of its GEMM's output.
    fuse: bool,
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

/// Stage times of the last forwards (`OXINFER_AVATAR_GPU_PROF=1`).
static PROF: Mutex<[f64; 14]> = Mutex::new([0.0; 14]);
const STAGES: [&str; 14] = [
    "copy",
    "norm",
    "qkv",
    "prep",
    "attn",
    "proj",
    "text q",
    "text attn",
    "text proj",
    "audio q",
    "audio attn",
    "audio proj",
    "ffn in",
    "ffn out",
];

impl Engine {
    /// The engine for `dit` on the active backend, when that is CUDA and the model is one it
    /// takes (see the module docs).
    pub(super) fn new(dit: &Dit) -> Option<Engine> {
        if std::env::var("OXINFER_AVATAR_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        let be = crate::gpu::active()?.as_any()?.downcast_ref::<CudaBackend>()?;
        // Split operands when asked (`OXINFER_AVATAR_GPU_EXACT=1`, or the crate-wide `gpu_exact`).
        let exact = std::env::var("OXINFER_AVATAR_GPU_EXACT")
            .ok()
            .map(|v| v != "0")
            .or_else(crate::gpu::exact_forced)
            .unwrap_or(false);
        match Self::build(be, dit, exact) {
            Ok(e) => Some(e),
            Err(e) => {
                debug(&format!("{e:#}"));
                None
            }
        }
    }

    fn build(be: &CudaBackend, dit: &Dit, exact: bool) -> Result<Engine> {
        let c = &dit.cfg;
        let hd = dit.head_dim();
        let Some(b0) = dit.blocks.first() else { bail!("no blocks") };
        if (hd != 64 && hd != 128) || c.audio_prenorm || mats(b0).iter().any(|m| !m.n_in.is_multiple_of(32)) {
            bail!("heads of {hd}, audio_prenorm or a width that is not a multiple of 32: no kernel");
        }
        let x: Cuh = {
            use dlsym::Bindings;
            // SAFETY: libcuda is already initialised by the backend.
            unsafe { Cuh::open_first(crate::platform::gpu::cuda::CANDIDATES.iter().copied()) }.map_err(|e| anyhow!("libcuda: {e}"))?
        };
        let ctx = be.context().ok_or_else(|| anyhow!("no context"))?;
        let src = concat!(include_str!("../minimax_h3/cuda.cu"), "\n", include_str!("cuda.cu"));
        // The default path: 128 queries per attention block, GEMMs in tiles of 128.
        let fq = if exact { 64 } else { 128 };
        let smem = flash_smem(hd, fq, if exact { 2 } else { 1 });
        let module = be.cached_module(src, "longcat-avatar-{sm}", "longcat_avatar.cu")?;
        let get = |name: &str| -> Result<F> {
            let cn = CString::new(name)?;
            let mut f = std::ptr::null_mut();
            if unsafe { x.cuModuleGetFunction(&mut f, module, cn.as_ptr()) } != 0 {
                bail!("kernel {name} missing");
            }
            Ok(f)
        };
        let f = Funcs {
            gemm_i8: get(if exact { "lc_gemm_i8_2" } else { "lc_gemm_fast_i8" })?,
            gemm_h: get(if exact { "gemm_h_2" } else { "lc_gemm_fast_h" })?,
            gemm_sides: [get("lc_gemm_sides_i8")?, get("lc_gemm_sides_h")?],
            tile: if exact { 64 } else { 128 },
            threads: if exact { 128 } else { 256 },
            // A GPU that shares the machine's memory (GB10) keeps less of a GEMM's weights in
            // its cache: 12 MB per group measured best there (8 and 6 within 2%, one group of
            // 32 MB 10% slower over a step), 16 MB and matrices up to 48 MB whole on the
            // RTX 5080.
            hot: if crate::cuda::integrated() { 12 << 20 } else { HOT_BYTES },
            merge: if crate::cuda::integrated() { 1 } else { 3 },
            split_rows: get("split_rows")?,
            swiglu: get("swiglu")?,
            v_scale: get("v_scale")?,
            v_planes: get("v_planes")?,
            attn: get(match (hd, exact) {
                (128, true) => "lc_attn_flash_128",
                (128, false) => "lc_attn_flash_128_wide",
                (_, true) => "lc_attn_flash_64",
                (_, false) => "lc_attn_flash_64_wide",
            })?,
            ln_mod: get("lc_ln_mod")?,
            ln_affine: get("lc_ln_affine")?,
            resid: get("lc_resid")?,
            qk_prep: get("lc_qk_prep")?,
            fq,
            smem,
        };
        if unsafe { x.cuFuncSetAttribute(f.attn, MAX_DYNAMIC_SHARED, smem as c_int) } != 0 {
            bail!("the device does not give a block {smem} bytes of shared memory");
        }
        let mut stream = std::ptr::null_mut();
        if unsafe { x.cuStreamCreate(&mut stream, 1) } != 0 {
            bail!("cuStreamCreate failed");
        }
        let mut events = [std::ptr::null_mut(); 2];
        for e in events.iter_mut() {
            if unsafe { x.cuEventCreate(e, 0) } != 0 {
                bail!("cuEventCreate failed");
            }
        }
        let n = dit.blocks.len();
        Ok(Engine {
            x,
            ctx,
            module,
            stream,
            f,
            state: Mutex::new(State {
                nets: vec![usize::MAX],
                blocks: Vec::new(),
                block_cap: 0,
                u_stride: 32,
                u_rp: 32,
                sides_ok: true,
                arena: Arena::default(),
                resident: vec![None; n],
                filled: vec![false; n],
                placed: false,
                slots: [0; 2],
                events,
            }),
            failed: AtomicBool::new(false),
            exact,
            // On a GPU sharing the machine's memory, where a pass over the activations costs
            // what it reads and writes (GB10: 48.0 s a forward against 56.9 s); a discrete
            // card runs the passes as fast (RTX 5080: 27.1 s fused, 26.9 s not).
            fuse: !exact && crate::cuda::integrated(),
            prof: std::env::var("OXINFER_AVATAR_GPU_PROF").is_ok_and(|v| v != "0"),
        })
    }

    /// The blocks' layouts and blobs for `nets`.
    fn layouts(dit: &Dit, nets: &[&LoraNet]) -> (Vec<BlockImg>, usize, [usize; 2], bool) {
        let t0 = crate::clock::Instant::now();
        let mut out = Vec::with_capacity(dit.blocks.len());
        let (mut cap, mut u_stride, mut u_rp, mut sides_ok) = (0usize, 32usize, 32usize, true);
        for (bi, b) in dit.blocks.iter().enumerate() {
            let ms = mats(b);
            let mut img = BlockImg::default();
            let mut at = 0usize;
            for m in ms {
                img.lins.push(DLin {
                    w: at as u64,
                    int8: m.is_int8(),
                    n_out: m.n_out,
                    n_in: m.n_in,
                    ..Default::default()
                });
                at += (if m.is_int8() { m.n_out * m.n_in } else { m.n_out * m.n_in * 2 }).next_multiple_of(ALIGN);
            }
            img.blob_at = at;
            let mut blob: Vec<u8> = Vec::new();
            let mut put = |bytes: &[u8]| -> u64 {
                let o = blob.len();
                blob.extend_from_slice(bytes);
                blob.resize(blob.len().next_multiple_of(ALIGN), 0);
                (at + o) as u64
            };
            for (l, m) in img.lins.iter_mut().zip(ms) {
                if let Some(s) = m.scales() {
                    l.cs = put(bytes_of(s));
                }
                l.bias = m.bias.as_ref().map_or(u64::MAX, |b| put(bytes_of(b)));
            }
            let vecs: [&[f32]; 10] = [
                &b.q_norm,
                &b.k_norm,
                &b.cq_norm,
                &b.ck_norm,
                &b.aq_norm,
                &b.ak_norm,
                &b.pre_crs.w,
                &b.pre_crs.b,
                &b.pre_video.w,
                &b.pre_video.b,
            ];
            for (o, v) in img.norms.iter_mut().zip(vecs) {
                *o = put(bytes_of(v));
            }
            for (ni, net) in nets.iter().enumerate() {
                for (mi, name) in LORA_TARGETS {
                    let Some(l) = net.mods.get(&format!("blocks.{bi}.{name}")) else {
                        continue;
                    };
                    let lin = &mut img.lins[mi];
                    if l.n_in != lin.n_in || l.n_out != lin.n_out {
                        continue;
                    }
                    let sep = l.ups.len();
                    let (a, a_f) = half_planes(&l.down, sep * l.r, l.n_in);
                    let a_at = put(bytes_of(&a));
                    let mut bs = Vec::with_capacity(sep);
                    for up in &l.ups {
                        let (p, f) = half_planes(up, l.n_out / sep, l.r);
                        bs.push((put(bytes_of(&p)), f));
                    }
                    u_stride = u_stride.max((sep * l.r + 32).next_multiple_of(32));
                    u_rp = u_rp.max(l.r.next_multiple_of(32));
                    sides_ok &= sep <= 3 && lin.lora.iter().all(|d| d.bs.len() == sep && d.r == l.r) && lin.lora.len() < 2;
                    lin.lora.push(DLora {
                        net: ni,
                        a: a_at,
                        a_f,
                        bs,
                        r: l.r,
                        alpha: l.alpha_scale,
                    });
                }
            }
            img.bytes = at + blob.len();
            img.blob = blob;
            cap = cap.max(img.bytes);
            out.push(img);
        }
        debug(&format!(
            "layouts of {} blocks ({:.2} GB, {} LoRAs) in {:.2} s",
            out.len(),
            out.iter().map(|b| b.bytes).sum::<usize>() as f64 / 1e9,
            nets.len(),
            t0.elapsed().as_secs_f64()
        ));
        (out, cap, [u_stride, u_rp], sides_ok)
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

    /// Copies to the device and returns once the bytes are there (see `minimax_h3/cuda.rs`).
    fn htod(&self, dst: u64, src: &[u8]) -> bool {
        crate::cuda::touch_before_upload(src);
        src.is_empty()
            || unsafe {
                self.x.cuMemcpyHtoD_v2(dst, src.as_ptr() as *const c_void, src.len()) == 0
                    && self.x.cuStreamSynchronize(std::ptr::null_mut()) == 0
            }
    }

    fn upload_block(&self, dit: &Dit, img: &BlockImg, bi: usize, base: u64) -> bool {
        for (l, m) in img.lins.iter().zip(mats(&dit.blocks[bi])) {
            let ok = match m.store {
                Store::Int8(b) => self.htod(base + l.w, b),
                Store::Float(..) => self.htod(base + l.w, &mat_halves(m)),
            };
            if !ok {
                return false;
            }
        }
        self.htod(base + img.blob_at as u64, &img.blob)
    }

    /// The sizes of the activation buffers (in the order of [`Arena::bufs`]), the key count
    /// padded, and the rows per feed-forward chunk.
    fn sizes(&self, dit: &Dit, ctx: &Ctx, u_stride: usize, u_rp: usize) -> ([usize; 24], usize, usize) {
        let c = &dit.cfg;
        let (h, hd, heads) = (c.hidden_size, dit.head_dim(), c.num_heads);
        let ffn = dit.blocks[0].w1.n_out;
        let n = ctx.n_t * ctx.s;
        let nk = n.next_multiple_of(64);
        let big = n * 3 * h * 4;
        let rc = (big / (3 * ffn * 4) / 64 * 64).clamp(64, n.next_multiple_of(64)).min(n);
        let big = big.max(rc * 3 * ffn * 4);
        let plane = heads * nk * hd * 2;
        let l = ctx.text_len;
        let arows = ctx.n_t * ctx.na;
        let xs = (n * h).max(rc * ffn).max(l * h).max(arows * c.output_dim.next_multiple_of(32)) * 2;
        let tab = dit.blocks.len() * ctx.n_t * 9 * h * 4;
        let rmax = n.max(l).max(arows);
        // The side paths: one `u`, its planes and its row scales with split operands; on the
        // default path those of two LoRAs, the planes and scales per section (three at most).
        let (us, u, urs) = if self.exact {
            (n * u_stride * 2, n * u_stride * 4, rmax * 4)
        } else {
            (6 * rmax * u_rp * 2, 2 * rmax * u_stride * 4, 6 * rmax * 4)
        };
        // The operands' second halves: with split operands only.
        let lo = |bytes: usize| if self.exact { bytes } else { 0 };
        let sizes = [
            n * h * 4,
            n * h * 4,
            plane,
            lo(plane),
            plane,
            lo(plane),
            plane,
            lo(plane),
            xs,
            lo(xs),
            us,
            lo(us),
            u,
            rmax * 4,
            urs,
            heads * 4,
            n * hd / 2 * 4,
            n * hd / 2 * 4,
            tab,
            big,
            l * h * 4,
            l * 2 * h * 4,
            arows * c.output_dim * 4,
            arows * 2 * h * 4,
        ];
        (sizes, nk, rc)
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

    fn make_arena(&self, st: &mut State, need: &[usize; 24]) -> bool {
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
        if st.placed && self.free_mem() < total + RESERVE {
            self.release_blocks(st);
        }
        let slots = if st.placed { 0 } else { 2 * st.block_cap };
        if self.free_mem() < total + RESERVE + slots {
            debug(&format!(
                "{:.1} GB of activations and two block slots of {:.2} GB, {:.1} GB are free",
                total as f64 / 1e9,
                st.block_cap as f64 / 1e9,
                self.free_mem() as f64 / 1e9
            ));
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
                    debug("out of device memory for the activations");
                    return false;
                }
            }
        }
        debug(&format!("{:.2} GB of activations", total as f64 / 1e9));
        st.arena = Arena { caps, all: ptrs };
        // The side paths' `u` rows are read past their last section (see `lin`): zeros there.
        let b = st.arena.bufs(self.exact);
        let z = vec![0u8; st.arena.caps[12]];
        self.htod(b.u, &z)
    }

    fn place_blocks(&self, st: &mut State) -> bool {
        if st.placed {
            return true;
        }
        let total: usize = st.blocks.iter().map(|b| b.bytes).sum();
        let all_fit = self.free_mem() >= total + RESERVE;
        for bi in 0..st.blocks.len() {
            let bytes = st.blocks[bi].bytes;
            let need = bytes + RESERVE + if all_fit { 0 } else { 2 * st.block_cap };
            if self.free_mem() < need {
                break;
            }
            let Some(p) = self.alloc(bytes) else { break };
            st.resident[bi] = Some(p);
        }
        if st.resident.iter().any(|r| r.is_none()) {
            for i in 0..2 {
                match self.alloc(st.block_cap) {
                    Some(p) => st.slots[i] = p,
                    None => return false,
                }
            }
        }
        st.placed = true;
        debug(&format!(
            "{} of {} blocks on the device, {:.1} GB free",
            st.resident.iter().filter(|r| r.is_some()).count(),
            st.blocks.len(),
            self.free_mem() as f64 / 1e9
        ));
        true
    }

    fn go(&self, f: F, grid: (u32, u32), block: u32, shmem: u32, args: &mut [*mut c_void]) {
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

    fn mark(&self, t: &mut crate::clock::Instant, k: usize) {
        if self.prof {
            unsafe { self.x.cuStreamSynchronize(self.stream) };
            PROF.lock().unwrap()[k] += t.elapsed().as_secs_f64();
            *t = crate::clock::Instant::now();
        }
    }

    /// The grid of a GEMM whose weights take `w_row` bytes per output (as `Engine::gemm` of
    /// `minimax_h3/cuda.rs`: the outputs in groups whose weights stay in the cache), in blocks of
    /// `tile` rows and outputs.
    fn grid(&self, p: &mut GemmP, w_row: usize) -> (u32, u32) {
        let tile = self.f.tile;
        let (ct, rt) = (ceil(p.n_out as usize, tile), ceil(p.rows as usize, tile));
        let mut cg = ((self.f.hot / (tile * w_row.max(1))) as u32).clamp(1, ct);
        if rt * ct.div_ceil(cg) > 65535 || ct <= self.f.merge * cg {
            cg = ct;
        }
        p.cg = cg;
        (cg, rt * ct.div_ceil(cg))
    }

    /// A GEMM of the default path: `p` (its grid is set here), the int8 rows' scales `cs` (0:
    /// float16 planes), the side paths and the output of `a`.
    fn gemm_fast(&self, mut a: LcFastP, int8: bool, out: Out) {
        match out {
            Out::Store => a.mode = 0,
            Out::Resid {
                tab,
                s,
                frame0,
                stride,
                gate_at,
                row0,
            } => (a.mode, a.tab, a.s, a.frame0, a.stride, a.gate_at, a.row0) = (1, tab, s.max(1), frame0, stride, gate_at, row0),
            Out::Swiglu => a.mode = 2,
        }
        a.s = a.s.max(1);
        let w_row = if int8 { a.p.n_in as usize } else { a.p.n_in as usize * 2 };
        let g = self.grid(&mut a.p, w_row);
        let f = match (a.uh[0] != 0, int8) {
            (true, _) => self.f.gemm_sides[!int8 as usize],
            (false, true) => self.f.gemm_i8,
            (false, false) => self.f.gemm_h,
        };
        self.go(f, g, self.f.threads, 0, &mut args![a]);
    }

    /// `y (= or +=) f rs W x` with float16 planes `w`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_h(
        &self,
        xs: [u64; 2],
        rs: u64,
        w: u64,
        bias: u64,
        y: u64,
        ys: usize,
        n_in: usize,
        n_out: usize,
        rows: usize,
        add: bool,
        f: f32,
    ) {
        let mut p = GemmP {
            xh: xs[0],
            xl: xs[1],
            w,
            wl: 0,
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
        if !self.exact {
            let out = match add {
                true => Out::Resid {
                    tab: 0,
                    s: 1,
                    frame0: 0,
                    stride: 0,
                    gate_at: 0,
                    row0: 0,
                },
                false => Out::Store,
            };
            return self.gemm_fast(LcFastP { p, ..Default::default() }, false, out);
        }
        let g = self.grid(&mut p, n_in * 2);
        self.go(self.f.gemm_h, g, self.f.threads, 0, &mut args![p]);
    }

    /// `W x + bias` of matrix `l` (at `base`) and its side paths, over `rows` rows whose planes
    /// are in `bf.xs` (row scales in `bf.rs`), into `y` as `out` says (with split operands:
    /// stored).
    #[allow(clippy::too_many_arguments)]
    fn lin(&self, bf: &Bufs, base: u64, l: &DLin, mult: &[f32], y: u64, ys: usize, rows: usize, out: Out) {
        let bias = if l.bias == u64::MAX { 0 } else { base + l.bias };
        let u_stride = bf.u_stride;
        if !self.exact {
            // The side paths first: u = A x of each LoRA in use, as planes per output section.
            let sides: Vec<(&DLora, f32)> = l
                .lora
                .iter()
                .map(|d| (d, mult.get(d.net).copied().unwrap_or(0.0)))
                .filter(|(_, m)| *m != 0.0)
                .take(2)
                .collect();
            let sep = sides.first().map_or(1, |(d, _)| d.bs.len());
            let rp = sides.first().map_or(32, |(d, _)| d.r.next_multiple_of(32));
            let (plane, scales) = ((bf.rmax * bf.u_rp * 2) as u64, (bf.rmax * 4) as u64);
            for (i, (d, _)) in sides.iter().enumerate() {
                let u = bf.u + (i * bf.rmax * u_stride * 4) as u64;
                self.gemm_h(bf.xs, bf.rs, base + d.a, 0, u, u_stride, l.n_in, sep * d.r, rows, false, d.a_f);
                for j in 0..sep {
                    let k = (3 * i + j) as u64;
                    self.split(
                        u + (j * d.r * 4) as u64,
                        u_stride,
                        rp,
                        rows,
                        [bf.us[0] + k * plane, 0],
                        bf.urs + k * scales,
                    );
                }
            }
            let per = l.n_out / sep;
            let w_row = if l.int8 { l.n_in } else { l.n_in * 2 };
            for j in 0..sep {
                let at = |bytes: usize| (j * per * bytes) as u64;
                let mut a = LcFastP {
                    p: GemmP {
                        xh: bf.xs[0],
                        xl: 0,
                        w: base + l.w + at(w_row),
                        wl: 0,
                        rs: bf.rs,
                        bias: if bias == 0 { 0 } else { bias + at(4) },
                        y: y + at(4),
                        ys: ys as u64,
                        n_in: l.n_in as u32,
                        n_out: per as u32,
                        rows: rows as u32,
                        add: 0,
                        f: 1.0,
                        cg: 0,
                    },
                    cs: if l.int8 { base + l.cs + at(4) } else { 0 },
                    ur: rp as u32,
                    us: (rp / 2) as u32,
                    ..Default::default()
                };
                for (i, (d, m)) in sides.iter().enumerate() {
                    if !self.fuse {
                        break;
                    }
                    let k = (3 * i + j) as u64;
                    a.uh[i] = bf.us[0] + k * plane;
                    a.bw[i] = base + d.bs[j].0;
                    a.urs[i] = bf.urs + k * scales;
                    a.uf[i] = d.bs[j].1 * m * d.alpha;
                }
                self.gemm_fast(a, l.int8, out);
                // Not fused: each side path's section as a GEMM of its own, added.
                for (i, (d, m)) in sides.iter().enumerate() {
                    if self.fuse {
                        break;
                    }
                    let k = (3 * i + j) as u64;
                    let planes = [bf.us[0] + k * plane, 0];
                    let f = d.bs[j].1 * m * d.alpha;
                    self.gemm_h(
                        planes,
                        bf.urs + k * scales,
                        base + d.bs[j].0,
                        0,
                        y + at(4),
                        ys,
                        rp,
                        per,
                        rows,
                        true,
                        f,
                    );
                }
            }
            return;
        }
        if l.int8 {
            let mut p = LcGemmP {
                p: GemmP {
                    xh: bf.xs[0],
                    xl: bf.xs[1],
                    w: base + l.w,
                    wl: 0,
                    rs: bf.rs,
                    bias,
                    y,
                    ys: ys as u64,
                    n_in: l.n_in as u32,
                    n_out: l.n_out as u32,
                    rows: rows as u32,
                    add: 0,
                    f: 1.0,
                    cg: 0,
                },
                cs: base + l.cs,
            };
            let g = self.grid(&mut p.p, l.n_in);
            self.go(self.f.gemm_i8, g, self.f.threads, 0, &mut args![p]);
        } else {
            self.gemm_h(bf.xs, bf.rs, base + l.w, bias, y, ys, l.n_in, l.n_out, rows, false, 1.0);
        }
        for d in &l.lora {
            let m = mult.get(d.net).copied().unwrap_or(0.0);
            if m == 0.0 {
                continue;
            }
            let sep = d.bs.len();
            let rp = d.r.next_multiple_of(32);
            let per = l.n_out / sep;
            // u = A x, then each output section += m alpha B_j u_j.
            self.gemm_h(bf.xs, bf.rs, base + d.a, 0, bf.u, u_stride, l.n_in, sep * d.r, rows, false, d.a_f);
            for (j, &(bp, bf_)) in d.bs.iter().enumerate() {
                self.split(bf.u + (j * d.r * 4) as u64, u_stride, rp, rows, bf.us, bf.urs);
                self.gemm_h(
                    bf.us,
                    bf.urs,
                    base + bp,
                    0,
                    y + (j * per * 4) as u64,
                    ys,
                    rp,
                    per,
                    rows,
                    true,
                    bf_ * m * d.alpha,
                );
            }
        }
    }

    /// The planes and row scales of `rows` rows of `k` values (`xs` floats apart).
    fn split(&self, x: u64, xs: usize, k: usize, rows: usize, planes: [u64; 2], rs: u64) {
        self.go(
            self.f.split_rows,
            (rows as u32, 1),
            256,
            0,
            &mut args![x, xs as u64, k as u32, planes[0], planes[1], rs],
        );
    }

    /// Queries `q` against keys `k` of the planes in `bf` (heads `stride` rows apart), into `out`.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        bf: &Bufs,
        out: u64,
        h: usize,
        hd: usize,
        heads: usize,
        stride: usize,
        nk: usize,
        q: std::ops::Range<usize>,
        k: std::ops::Range<usize>,
    ) {
        let p = LcAttP {
            qh: bf.qp[0],
            ql: bf.qp[1],
            kh: bf.kp[0],
            kl: bf.kp[1],
            vh: bf.vp[0],
            vl: bf.vp[1],
            vs: bf.vs,
            out,
            os: h as u64,
            n: stride as u32,
            nk: nk as u32,
            scale: 1.0 / (hd as f32).sqrt(),
            pad: 0,
            qlo: q.start as u32,
            qhi: q.end as u32,
            klo: k.start as u32,
            khi: k.end as u32,
        };
        let fq = self.f.fq;
        self.go(
            self.f.attn,
            (ceil(q.len(), fq), heads as u32),
            4 * fq as u32,
            self.f.smem,
            &mut args![p],
        );
    }

    /// The values of `rows` keys (at `off0` of rows `xs` floats apart in `src`) as planes.
    #[allow(clippy::too_many_arguments)]
    fn values(&self, bf: &Bufs, src: u64, xs: usize, off0: usize, hd: usize, heads: usize, rows: usize, nk: usize) {
        let (hdu, ru) = (hd as u32, rows as u32);
        self.go(
            self.f.v_scale,
            (heads as u32, 1),
            256,
            0,
            &mut args![src, xs as u64, off0 as u32, hdu, hdu, ru, bf.vs],
        );
        self.go(
            self.f.v_planes,
            (ceil(nk / 2, 256), (heads * hd) as u32),
            256,
            0,
            &mut args![src, xs as u64, off0 as u32, hdu, hdu, ru, nk as u32, bf.vs, bf.vp[0], bf.vp[1]],
        );
    }

    pub(super) fn run(&self, dit: &Dit, x: &mut [f32], ctx: &Ctx) -> bool {
        let c = &dit.cfg;
        let (h, hd, heads) = (c.hidden_size, dit.head_dim(), c.num_heads);
        let (s, n_t) = (ctx.s, ctx.n_t);
        let n = n_t * s;
        if n == 0 || self.failed.load(Relaxed) || x.len() != n * h || ctx.token_masks.is_some() {
            return false;
        }
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        if unsafe { self.x.cuCtxSetCurrent(self.ctx) } != 0 {
            return false;
        }
        // The layouts follow the LoRAs given (whatever their multipliers).
        let nets: Vec<&LoraNet> = ctx.lora.nets.iter().map(|(n, _)| *n).collect();
        let ids: Vec<usize> = nets.iter().map(|n| *n as *const LoraNet as usize).collect();
        if st.nets != ids {
            self.release_blocks(st);
            let (blocks, cap, [u_stride, u_rp], sides_ok) = Self::layouts(dit, &nets);
            (st.blocks, st.block_cap, st.u_stride, st.u_rp, st.sides_ok, st.nets) = (blocks, cap, u_stride, u_rp, sides_ok, ids);
            // `u` may be wider now.
            for p in st.arena.all.drain(..) {
                unsafe { self.x.cuMemFree_v2(p) };
            }
            st.arena = Arena::default();
        }
        let mult: Vec<f32> = ctx.lora.nets.iter().map(|(_, m)| *m).collect();
        if !self.exact && !st.sides_ok {
            debug("LoRAs of different ranks or sections on one matrix");
            return false;
        }
        let (need, nk, rc) = self.sizes(dit, ctx, st.u_stride, st.u_rp);
        if !self.make_arena(st, &need) || !self.place_blocks(st) {
            self.release_blocks(st);
            return false;
        }
        let mut bf = st.arena.bufs(self.exact);
        (bf.u_stride, bf.u_rp, bf.rmax) = (st.u_stride, st.u_rp, n.max(ctx.text_len).max(n_t * ctx.na));
        let mut tm = crate::clock::Instant::now();
        // The AdaLN sets of every block: [n_t, 6 H] then [n_t, 3 H].
        let tab_floats = n_t * 9 * h;
        let mut tabs = vec![0f32; dit.blocks.len() * tab_floats];
        for (bi, t) in tabs.chunks_mut(tab_floats).enumerate() {
            let (m, a) = dit.block_mods(bi, ctx);
            t[..n_t * 6 * h].copy_from_slice(&m);
            t[n_t * 6 * h..].copy_from_slice(&a);
        }
        if !(self.htod(bf.x, bytes_of(x))
            && self.htod(bf.cos, bytes_of(&ctx.cos))
            && self.htod(bf.sin, bytes_of(&ctx.sin))
            && self.htod(bf.tabs, bytes_of(&tabs))
            && self.htod(bf.text, bytes_of(&ctx.text))
            && self.htod(bf.audio, bytes_of(&ctx.audio)))
        {
            return false;
        }
        drop(tabs);
        let ffn = dit.blocks[0].w1.n_out;
        let (hu, su, hdu, headsu) = (h as u32, s as u32, hd as u32, heads as u32);
        let nc = ctx.num_cond;
        let nn = n - nc * s;
        let (l, na, od) = (ctx.text_len, ctx.na, c.output_dim);
        let frames = n_t - nc;
        let (nk_text, nk_audio) = (l.next_multiple_of(64), (n_t * na).next_multiple_of(64));
        let segs = ctx.segments();
        let mut slot = 0usize;
        for bi in 0..st.blocks.len() {
            let base = match st.resident[bi] {
                Some(p) => {
                    if !st.filled[bi] {
                        if !self.upload_block(dit, &st.blocks[bi], bi, p) {
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
                    if !self.upload_block(dit, &st.blocks[bi], bi, p) {
                        self.failed.store(true, Relaxed);
                        break;
                    }
                    p
                }
            };
            let img = &st.blocks[bi];
            let lin_out = |i: usize, y: u64, ys: usize, rows: usize, out: Out| self.lin(&bf, base, &img.lins[i], &mult, y, ys, rows, out);
            let lin = |i: usize, y: u64, ys: usize, rows: usize| lin_out(i, y, ys, rows, Out::Store);
            // The gated residual of the row's frame (`tab` 0: no gate), for rows from `row0` on.
            let resid = |tab: u64, gate_at: u32, row0: usize| Out::Resid {
                tab,
                s: su,
                frame0: 0,
                stride: 6 * hu,
                gate_at,
                row0: row0 as u32,
            };
            let nrm = |i: usize| base + img.norms[i];
            let tab = bf.tabs + (bi * tab_floats * 4) as u64;
            let atab = tab + (n_t * 6 * h * 4) as u64;
            let (m6, m3) = (6 * hu, 3 * hu);
            let null = 0u64;
            self.mark(&mut tm, 0);

            // Self-attention.
            self.go(
                self.f.ln_mod,
                (n as u32, 1),
                256,
                0,
                &mut args![bf.x, tab, bf.hn, hu, su, 0u32, m6, 0u32, hu],
            );
            self.mark(&mut tm, 1);
            self.split(bf.hn, h, h, n, bf.xs, bf.rs);
            lin(0, bf.big, 3 * h, n);
            self.mark(&mut tm, 2);
            for (part, planes) in [bf.qp, bf.kp].into_iter().enumerate() {
                self.go(
                    self.f.qk_prep,
                    (n as u32, headsu),
                    32,
                    0,
                    &mut args![
                        bf.big,
                        (3 * h) as u64,
                        (part * h) as u32,
                        nrm(part),
                        bf.cos,
                        bf.sin,
                        hdu,
                        n as u32,
                        planes[0],
                        planes[1]
                    ],
                );
            }
            self.values(&bf, bf.big, 3 * h, 2 * h, hd, heads, n, nk);
            self.mark(&mut tm, 3);
            for (q, k) in &segs {
                self.attend(&bf, bf.big, h, hd, heads, n, nk, q.clone(), k.clone());
            }
            self.mark(&mut tm, 4);
            self.split(bf.big, h, h, n, bf.xs, bf.rs);
            if !self.fuse {
                lin(1, bf.hn, h, n);
                self.go(
                    self.f.resid,
                    (ceil(h, 256), n as u32),
                    256,
                    0,
                    &mut args![bf.x, bf.hn, tab, hu, su, 0u32, m6, 2 * hu],
                );
            } else {
                lin_out(1, bf.x, h, n, resid(tab, 2 * hu, 0));
            }
            self.mark(&mut tm, 5);

            if nn > 0 {
                let xn = bf.x + (nc * s * h * 4) as u64;
                let att = bf.big + (nn * h * 4) as u64;
                // Text cross-attention.
                self.go(self.f.ln_affine, (nn as u32, 1), 256, 0, &mut args![xn, nrm(6), nrm(7), bf.hn, hu]);
                self.split(bf.hn, h, h, nn, bf.xs, bf.rs);
                lin(2, bf.big, h, nn);
                self.go(
                    self.f.qk_prep,
                    (nn as u32, headsu),
                    32,
                    0,
                    &mut args![bf.big, h as u64, 0u32, nrm(2), null, null, hdu, n as u32, bf.qp[0], bf.qp[1]],
                );
                self.mark(&mut tm, 6);
                self.split(bf.text, h, h, l, bf.xs, bf.rs);
                lin(3, bf.tkv, 2 * h, l);
                self.go(
                    self.f.qk_prep,
                    (l as u32, headsu),
                    32,
                    0,
                    &mut args![bf.tkv, (2 * h) as u64, 0u32, nrm(3), null, null, hdu, n as u32, bf.kp[0], bf.kp[1]],
                );
                self.values(&bf, bf.tkv, 2 * h, h, hd, heads, l, nk_text);
                self.attend(&bf, att, h, hd, heads, n, nk_text, 0..nn, 0..l);
                self.mark(&mut tm, 7);
                self.split(att, h, h, nn, bf.xs, bf.rs);
                if !self.fuse {
                    lin(4, bf.hn, h, nn);
                    self.go(
                        self.f.resid,
                        (ceil(h, 256), nn as u32),
                        256,
                        0,
                        &mut args![xn, bf.hn, null, hu, su, 0u32, m6, 0u32],
                    );
                } else {
                    lin_out(4, xn, h, nn, resid(0, 0, 0));
                }
                self.mark(&mut tm, 8);

                // Audio cross-attention, frame by frame.
                self.go(self.f.ln_affine, (nn as u32, 1), 256, 0, &mut args![xn, nrm(8), nrm(9), bf.hn, hu]);
                self.split(bf.hn, h, h, nn, bf.xs, bf.rs);
                lin(5, bf.big, h, nn);
                self.go(
                    self.f.qk_prep,
                    (nn as u32, headsu),
                    32,
                    0,
                    &mut args![bf.big, h as u64, 0u32, nrm(4), null, null, hdu, n as u32, bf.qp[0], bf.qp[1]],
                );
                self.mark(&mut tm, 9);
                let arows = frames * na;
                self.split(bf.audio + (nc * na * od * 4) as u64, od, od, arows, bf.xs, bf.rs);
                lin(6, bf.akv, 2 * h, arows);
                self.go(
                    self.f.qk_prep,
                    (arows as u32, headsu),
                    32,
                    0,
                    &mut args![bf.akv, (2 * h) as u64, 0u32, nrm(5), null, null, hdu, n as u32, bf.kp[0], bf.kp[1]],
                );
                self.values(&bf, bf.akv, 2 * h, h, hd, heads, arows, nk_audio);
                for f in 0..frames {
                    self.attend(&bf, att, h, hd, heads, n, nk_audio, f * s..(f + 1) * s, f * na..(f + 1) * na);
                }
                self.mark(&mut tm, 10);
                self.split(att, h, h, nn, bf.xs, bf.rs);
                lin(7, bf.hn, h, nn);
                self.go(
                    self.f.ln_mod,
                    (nn as u32, 1),
                    256,
                    0,
                    &mut args![bf.hn, atab, bf.big, hu, su, nc as u32, m3, 0u32, hu],
                );
                self.go(
                    self.f.resid,
                    (ceil(h, 256), nn as u32),
                    256,
                    0,
                    &mut args![xn, bf.big, atab, hu, su, nc as u32, m3, 2 * hu],
                );
                self.mark(&mut tm, 11);
            }

            // Feed-forward, in row chunks.
            self.go(
                self.f.ln_mod,
                (n as u32, 1),
                256,
                0,
                &mut args![bf.x, tab, bf.hn, hu, su, 0u32, m6, 3 * hu, 4 * hu],
            );
            let gu = bf.big;
            let act = gu + (rc * 2 * ffn * 4) as u64;
            for r0 in (0..n).step_by(rc) {
                let rn = rc.min(n - r0);
                let rows = bf.hn + (r0 * h * 4) as u64;
                self.split(rows, h, h, rn, bf.xs, bf.rs);
                if self.fuse {
                    // The gate rows, the value rows times their SiLU over them, and the result
                    // into the residual stream.
                    lin(8, gu, ffn, rn);
                    lin_out(9, gu, ffn, rn, Out::Swiglu);
                    self.mark(&mut tm, 12);
                    self.split(gu, ffn, ffn, rn, bf.xs, bf.rs);
                    lin_out(10, bf.x + (r0 * h * 4) as u64, h, rn, resid(tab, 5 * hu, r0));
                    self.mark(&mut tm, 13);
                    continue;
                }
                lin(8, gu, 2 * ffn, rn);
                lin(9, gu + (ffn * 4) as u64, 2 * ffn, rn);
                self.go(
                    self.f.swiglu,
                    (ceil(rn * ffn, 256), 1),
                    256,
                    0,
                    &mut args![gu, act, rn as u32, ffn as u32, 1u32],
                );
                self.mark(&mut tm, 12);
                self.split(act, ffn, ffn, rn, bf.xs, bf.rs);
                lin(10, rows, h, rn);
            }
            if !self.fuse {
                self.go(
                    self.f.resid,
                    (ceil(h, 256), n as u32),
                    256,
                    0,
                    &mut args![bf.x, bf.hn, tab, hu, su, 0u32, m6, 5 * hu],
                );
            }
            self.mark(&mut tm, 13);
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
            self.failed.store(true, Relaxed);
            eprintln!("[longcat-avatar cuda: a kernel or a copy failed; the CPU path runs]");
            return false;
        }
        let got = unsafe { self.x.cuMemcpyDtoH_v2(x.as_mut_ptr() as *mut c_void, bf.x, x.len() * 4) } == 0;
        if !got {
            self.failed.store(true, Relaxed);
        }
        if self.prof {
            let mut p = PROF.lock().unwrap();
            let line: Vec<String> = STAGES.iter().zip(p.iter()).map(|(n, t)| format!("{n} {t:.2}")).collect();
            eprintln!("[longcat-avatar cuda: {} s]", line.join(", "));
            *p = [0.0; 14];
        }
        got
    }

    /// Frees the device memory (weights and activations); the next forward brings them back.
    pub(super) fn release(&self) {
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
    use super::super::dit::Cond;
    use super::super::lora::Active;
    use super::*;

    fn backend() -> Option<&'static CudaBackend> {
        static B: std::sync::OnceLock<Option<CudaBackend>> = std::sync::OnceLock::new();
        B.get_or_init(|| CudaBackend::new(0).ok()).as_ref()
    }

    /// The device's block stack against the CPU's on the tiny release (skipped without an NVIDIA
    /// GPU): text only, a reference image, a continuation with a masked reference; float and
    /// int8 weights; without LoRAs and with a mix of the two.
    #[test]
    fn stack_matches_cpu() {
        let Some(be) = backend() else { return };
        let dir = std::env::temp_dir().join(format!("oxinfer-longcat-avatar-cuda-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (avatar, _) = super::super::tiny::write_tiny_release(&dir).unwrap();
        let nets = [
            LoraNet::load(&avatar.join("loraA/lora.safetensors")).unwrap(),
            LoraNet::load(&avatar.join("loraB/lora.safetensors")).unwrap(),
        ];
        let mut seed = 7u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        for sub in ["base_model", "base_model_int8"] {
            let dit = Dit::load(&avatar.join(sub)).unwrap();
            for (exact, fuse) in [(true, false), (false, false), (false, true)] {
                let mut e = Engine::build(be, &dit, exact).unwrap();
                e.fuse = fuse;
                let c = &dit.cfg;
                let (lh, lw) = (8, 12);
                for (n_t, num_cond, num_ref, mask) in [(4usize, 0usize, 0usize, 0usize), (4, 1, 0, 0), (6, 3, 1, 1)] {
                    for mix in [None, Some((0.3f32, 0.7f32))] {
                        let frames = 1 + c.vae_scale * (n_t - num_ref - 1);
                        let audio: Vec<f32> = (0..frames * c.audio_window * c.audio_block * c.audio_channel)
                            .map(|_| rnd())
                            .collect();
                        let text: Vec<f32> = (0..5 * c.caption_channels).map(|_| rnd()).collect();
                        let ts: Vec<f32> = (0..n_t).map(|f| if f < num_cond { 0.0 } else { 640.0 }).collect();
                        let active: Vec<(&LoraNet, f32)> = match mix {
                            Some((a, b)) => vec![(&nets[0], a), (&nets[1], b)],
                            None => Vec::new(),
                        };
                        let cond = Cond {
                            text: &text,
                            text_len: 16,
                            audio: &audio,
                            humans: 1,
                            num_cond,
                            num_ref,
                            ref_img_index: 2,
                            mask_frame_range: mask,
                            ref_masks: None,
                            lora: Active { nets: &active },
                        };
                        let ctx = dit.context((n_t, lh, lw), &ts, &cond).unwrap();
                        let n = n_t * ctx.s;
                        let x: Vec<f32> = (0..n * c.hidden_size).map(|_| rnd()).collect();
                        let (mut dev, mut cpu) = (x.clone(), x);
                        assert!(e.run(&dit, &mut dev, &ctx), "{sub}: the engine declined");
                        dit.blocks_cpu(&mut cpu, &ctx);
                        let scale = cpu.iter().map(|v| v.abs()).sum::<f32>() / cpu.len() as f32;
                        let max = dev.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                        // The side paths' factors are float16 on the device; without split operands
                        // so are the activations.
                        let tol = match (exact, mix.is_some()) {
                            (true, false) => 2e-5,
                            (true, true) => 2e-3,
                            (false, _) => 2e-2,
                        };
                        assert!(
                            max < tol * scale.max(1.0),
                            "{sub} exact {exact} fused {fuse} cond {num_cond} ref {num_ref} mix {mix:?}: {max:.3e} (mean |v| {scale:.3e})"
                        );
                    }
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
