//! Inkling-2 on CUDA: whole passes of the decoder (prefill chunks and decode steps) and the
//! unembedding on the device, kernels in `inkling_cuda.cu` (compiled by NVRTC at first use, only
//! for the weight kinds the model has).
//!
//! What runs there, per layer application (48 for Mini: layers 6-17 three times): the attention
//! residual's mixes of the block states, the loop's messages and their gates, the RMS norms, the
//! projections (matvecs for decode, f16 tensor-core GEMMs for prefill), the key / value
//! convolutions (static and per-token taps) with their histories, the query / key / relative
//! vector scales, clips and log scaling, the value norm, the cache appends (f32, f16 or q8 as
//! `--kv-cache`), the attention with its relative bias, sink and window (split over the keys and
//! merged), the output gate, the short convolutions after the attention and the MLP, the router
//! and its top k, and the routed experts (decode: the chosen experts' rows read directly; prefill:
//! segments per expert built on the device, grouped GEMMs with the activation in the epilogue).
//! The host looks up the token embeddings and samples.
//!
//! A session's caches and convolution histories live on the device while it runs there
//! ([`DevSess`]); a pass the device does not take (interventions, captured layers) first brings
//! them back to the host, and a session that ran on the host goes up at its next device pass.
//! Decode steps are captured once per session as a CUDA graph (positions are read on the device),
//! so a token is one graph launch.
//!
//! Declined (the CPU path runs): weight kinds without a kernel (the ternary and 1-bit GGUF types,
//! affine Q4), LoRA adapters, Hadamard-folded weights, streamed weights, head sizes other than
//! 128, more than 4 query heads per key/value head, and models that do not fit in VRAM
//! (`OXINFER_GPU_PLAN_DEBUG=1` says why; `OXINFER_INKLING_GPU=0` keeps the CPU path).

use super::{
    DevBox, DevEngine, DevSession, Gate, InNorm, Inkling2Model, Inkling2Session, KvCache, KvConv, KvRows, LayerState, MIX_EPS, ROLLBACK,
};
use crate::cuda::{Cu, CudaBackend};
use crate::ggml::GgmlType;
use crate::model::{Lin, Weight};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use std::ffi::{CString, c_int, c_void};
use std::sync::{Arc, Mutex};

dlsym::library! {
    /// Driver entry points beyond the backend's: streams, async copies and graphs.
    pub struct Cux {
        unsafe extern "C" {
            pub fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: *mut c_void) -> c_int;
            pub fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: *mut c_void) -> c_int;
            pub fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, bytes: usize, stream: *mut c_void) -> c_int;
            pub fn cuMemsetD8Async(dst: u64, v: u8, n: usize, stream: *mut c_void) -> c_int;
            pub fn cuStreamBeginCapture_v2(stream: *mut c_void, mode: c_int) -> c_int;
            pub fn cuStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> c_int;
            pub fn cuGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64) -> c_int;
            pub fn cuGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> c_int;
            pub fn cuGraphExecDestroy(exec: *mut c_void) -> c_int;
            pub fn cuGraphDestroy(graph: *mut c_void) -> c_int;
            pub fn cuStreamCreate(stream: *mut *mut c_void, flags: u32) -> c_int;
            pub fn cuStreamSynchronize(stream: *mut c_void) -> c_int;
            pub fn cuStreamDestroy_v2(stream: *mut c_void) -> c_int;
            pub fn cuModuleGetFunction(f: *mut *mut c_void, module: *mut c_void, name: *const std::ffi::c_char) -> c_int;
            pub fn cuModuleUnload(module: *mut c_void) -> c_int;
            pub fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int;
            pub fn cuMemFree_v2(ptr: u64) -> c_int;
            pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
            pub fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
            pub fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> c_int;
            pub fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
            pub fn cuEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> c_int;
            pub fn cuEventDestroy_v2(event: *mut c_void) -> c_int;
            pub fn cuLaunchKernel(
                f: *mut c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32,
                shmem: u32, stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void,
            ) -> c_int;
        }
    }
}

type F = *mut c_void;

/// Pointers to the kernel arguments, for `cuLaunchKernel` (in one statement with the launch, so
/// temporaries outlive it).
macro_rules! args {
    ($($v:expr),* $(,)?) => { [$( &$v as *const _ as *mut c_void ),*] };
}

/// Most tokens a pass runs on the decode kernels (more take the GEMMs and the tiled attention):
/// every token of such a pass is computed as in a pass of its own (speculative verification
/// stays exact).
const MV_T: usize = 16;
/// Most tokens of one device pass (longer calls run in chunks).
const CHUNK: usize = 1024;
/// Key splits of a decode step's attention: full and sliding layers.
const SPLITS_FULL: usize = 32;
const SPLITS_SLIDING: usize = 16;
/// Block states the mixes take (`MAXS` in the kernels).
const MAX_STATES: usize = 11;
/// Most captured streams per token (`DevWant::capture`; DSpark drafts read four).
const MAX_CAPTURE: usize = 8;
/// Kept free on the device beyond the weights and activations.
const RESERVE: usize = 1 << 30;
/// Kept free when a session's caches are allocated or grown (the session then moves to the host).
const SESS_RESERVE: usize = 256 << 20;

fn debug(why: &str) {
    if std::env::var("OXINFER_GPU_PLAN_DEBUG").is_ok_and(|v| v != "0") {
        eprintln!("[inkling cuda: {why}]");
    }
}

/// A weight matrix as the kernels take it (`W`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WArg {
    q: u64,
    s: u64,
    ef: u64,
    rb: u64,
    es: u64,
    ss: u64,
    f: f32,
    sr: u32,
    sc: u32,
    ncb: u32,
}

/// A device matrix (or stack of experts) and its kernels.
#[derive(Clone, Copy)]
struct Dw {
    a: WArg,
    kind: u32,
    n_out: usize,
    n_in: usize,
    mv: F,
    gemm: F,
    gu: F,
    dn: F,
    mvm: F,
    gdown: F,
    gout: F,
    /// A message gate's down projection for prefill (a block per token).
    gdown_t: F,
    /// The split GEMM with the weights' exact values and per-16 scales (`gemmi_*`).
    gemmi: F,
}

impl Dw {
    /// The prefill GEMM: by default the activations split into pairs of halves and the weights'
    /// exact values with their scales applied per 16 in f32 (5e-8 of the summed magnitudes, as
    /// f32: plain f16 operands moved this model's logits by KL ~2e-3), or f16 operands (~10%
    /// faster) with `gpu_exact=0` ([`crate::gpu::set_exact`]) or `OXINFER_INKLING_EXACT=0`
    /// (which wins).
    fn gemm_f(&self) -> F {
        if !split_gemm() {
            self.gemm
        } else {
            // f32 / bf16 weights take the split of both operands (`gemmi_*` falls back to it).
            self.gemmi
        }
    }
}

/// See [`Dw::gemm_f`].
fn split_gemm() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| std::env::var("OXINFER_INKLING_EXACT").ok().map(|v| v != "0"));
    env.or_else(crate::gpu::exact_forced).unwrap_or(true)
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MvJob {
    w: WArg,
    y: u64,
    n_out: u32,
    ys: u32,
    b0: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MvJobs {
    j: [MvJob; 8],
    x: u64,
    n: u32,
    n_in: u32,
    xs: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VecJob {
    x: u64,
    scale: u64,
    alpha: f64,
    n: u32,
    nv: u32,
    stride: u32,
    total: u32,
    mode: u32,
    b0: u32,
    clip: f32,
    _p: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VecJobs {
    j: [VecJob; 4],
    info: u64,
    floor: f64,
    eps: f32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ConvJob {
    x: u64,
    hist: u64,
    taps: u64,
    dyn_: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ConvJobs {
    j: [ConvJob; 2],
    info: u64,
    ch: u32,
    k: u32,
    t: u32,
    groups: u32,
    ring: u32,
    norm: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmP {
    x: u64,
    y: u64,
    gidx: u64,
    tiles: u64,
    ntiles: u64,
    wrow: u64,
    xh: u64,
    xl: u64,
    n_in: u32,
    n_out: u32,
    rows: u32,
    xs: u32,
    ys: u32,
    mode: u32,
    mi: u32,
    inter: u32,
    beta: f32,
    lbeta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MixP {
    st: [u64; MAX_STATES],
    out: [u64; 4],
    nw: [u64; 4],
    x: u64,
    q: u64,
    ns: u32,
    nq: u32,
    h: u32,
    eps: f32,
    neps: f32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AttP {
    q: u64,
    r: u64,
    kc: u64,
    vc: u64,
    ks: u64,
    vs: u64,
    rel: u64,
    part: u64,
    info: u64,
    nh: u32,
    nkv: u32,
    dr: u32,
    extent: u32,
    window: u32,
    cap: u32,
    ns: u32,
    qs: u32,
    rs: u32,
    scaling: f32,
}

/// A weight as stored on the host: kernel kind, bytes, row bytes, block scales
/// (`(scales, rows, cols)`), factor.
struct HostW<'m> {
    kind: u32,
    data: &'m [u8],
    rb: usize,
    scales: Option<(std::borrow::Cow<'m, [f32]>, usize, usize)>,
    f: f32,
}

/// Kernel kinds: the GGUF ids, plus oxinfer's Q8 (200), Q4 (201) and FP8 E4M3 (202).
const KINDS: [u32; 27] = [
    0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 39, 40, 200, 201, 202,
];
const K_OQ8: u32 = 200;
const K_OQ4: u32 = 201;
const K_F8: u32 = 202;

fn host_w<'m>(l: &'m Lin, n_in: usize) -> Option<HostW<'m>> {
    if l.has_lora() || l.fold.is_some() {
        return None;
    }
    Some(match &l.w {
        Weight::Bf16(b, _) => {
            // SAFETY: u16 to bytes of the same memory.
            let data = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 2) };
            HostW {
                kind: 30,
                data,
                rb: n_in * 2,
                scales: None,
                f: 1.0,
            }
        }
        Weight::Q8(m) => HostW {
            kind: K_OQ8,
            // SAFETY: i8 to bytes of the same memory.
            data: unsafe { std::slice::from_raw_parts(m.q.as_ptr() as *const u8, m.q.len()) },
            rb: m.n_in,
            scales: Some((std::borrow::Cow::Borrowed(&m.scales[..]), 1, 32)),
            f: 1.0,
        },
        Weight::Q4(m) => {
            if m.offsets.is_some() {
                return None;
            }
            let s: Vec<f32> = m.scales.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
            HostW {
                kind: K_OQ4,
                data: &m.q[..],
                rb: m.n_in / 2,
                scales: Some((std::borrow::Cow::Owned(s), 1, 32)),
                f: 1.0,
            }
        }
        Weight::Gguf(m) => {
            let kind = match m.kind {
                GgmlType::F8E4M3 => K_F8,
                GgmlType::I8 => K_OQ8,
                k => k as u32,
            };
            if !KINDS.contains(&kind) {
                return None;
            }
            let (mut scales, mut f) = (None, 1.0);
            if let Some(bs) = &m.block_scales {
                if kind == K_F8 || kind == K_OQ8 {
                    let cols = bs.cols.min(m.n_in);
                    if !cols.is_multiple_of(32) || bs.rows == 0 {
                        return None;
                    }
                    scales = Some((std::borrow::Cow::Borrowed(&bs.s[..]), bs.rows, cols));
                } else if bs.rows >= m.n_out && bs.cols >= m.n_in && bs.s.len() == 1 {
                    f = bs.s[0];
                } else {
                    return None;
                }
            } else if kind == K_OQ8 {
                return None;
            }
            HostW {
                kind,
                data: &m.data[..],
                rb: m.row_bytes(),
                scales,
                f,
            }
        }
    })
}

/// The decoding state of one layer application on the device: keys and values at rows
/// `pos % cap` (`cap` positions; q8 scales beside them), and the four convolution histories.
struct Slot {
    k: u64,
    v: u64,
    ks: u64,
    vs: u64,
    cap: usize,
    kw: usize,
    hist: [u64; 4],
    hist_len: [usize; 4],
}

/// A session's state on the device (authoritative while it exists).
pub(super) struct DevSess {
    x: Arc<Cux>,
    ctx: *mut c_void,
    slots: Vec<Slot>,
    /// 0 f32, 1 f16, 2 q8.
    fmt: u32,
    /// Captured passes of up to `MV_T` tokens.
    graphs: Vec<PassGraph>,
    /// The engine whose buffers a graph uses.
    engine: usize,
}
/// A captured pass: (tokens, captured steps, the cache capacities it was captured with,
/// executable, graph).
type PassGraph = (usize, Vec<usize>, Vec<usize>, F, F);

// SAFETY: device addresses and driver handles; the driver API is thread-safe and the context is
// made current before use.
unsafe impl Send for DevSess {}
unsafe impl Sync for DevSess {}

impl DevSess {
    fn bind(&self) -> bool {
        unsafe { self.x.cuCtxSetCurrent(self.ctx) == 0 }
    }
    fn drop_graph(&mut self) {
        for (_, _, _, exec, g) in self.graphs.drain(..) {
            unsafe {
                self.x.cuGraphExecDestroy(exec);
                self.x.cuGraphDestroy(g);
            }
        }
    }
    fn alloc(x: &Cux, bytes: usize) -> Option<u64> {
        let mut d = 0u64;
        (unsafe { x.cuMemAlloc_v2(&mut d, bytes.max(256)) } == 0).then_some(d)
    }
    fn bytes_of(&self, s: &Slot) -> usize {
        let per = match self.fmt {
            0 => 4.0,
            1 => 2.0,
            _ => 1.125,
        };
        (2.0 * per * (s.cap * s.kw) as f64) as usize
    }
}

impl Drop for DevSess {
    fn drop(&mut self) {
        if !self.bind() {
            return;
        }
        self.drop_graph();
        for s in &self.slots {
            for p in [s.k, s.v, s.ks, s.vs].into_iter().chain(s.hist) {
                if p != 0 {
                    unsafe { self.x.cuMemFree_v2(p) };
                }
            }
        }
    }
}

impl DevSession for DevSess {
    fn clone_box(&self) -> Box<dyn DevSession> {
        assert!(self.bind(), "inkling cuda: context");
        let copy = |p: u64, n: usize| -> u64 {
            if p == 0 {
                return 0;
            }
            let d = DevSess::alloc(&self.x, n).expect("inkling cuda: out of device memory copying a session");
            unsafe { self.x.cuMemcpyDtoDAsync_v2(d, p, n, std::ptr::null_mut()) };
            d
        };
        let el = match self.fmt {
            0 => 4,
            1 => 2,
            _ => 1,
        };
        let slots = self
            .slots
            .iter()
            .map(|s| {
                let n = s.cap * s.kw;
                Slot {
                    k: copy(s.k, n * el),
                    v: copy(s.v, n * el),
                    ks: copy(s.ks, n / 32 * 4),
                    vs: copy(s.vs, n / 32 * 4),
                    cap: s.cap,
                    kw: s.kw,
                    hist: std::array::from_fn(|i| copy(s.hist[i], s.hist_len[i] * 4)),
                    hist_len: s.hist_len,
                }
            })
            .collect();
        unsafe { self.x.cuStreamSynchronize(std::ptr::null_mut()) };
        Box::new(DevSess {
            x: self.x.clone(),
            ctx: self.ctx,
            slots,
            fmt: self.fmt,
            graphs: Vec::new(),
            engine: self.engine,
        })
    }
    fn bytes(&self) -> usize {
        self.slots.iter().map(|s| self.bytes_of(s)).sum()
    }
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
    /// Nothing to undo: the caches are by position (the sliding rings keep `CHUNK` rows more
    /// than the window), and so are the convolutions' rings.
    fn truncate(&mut self, pos: usize, len: usize) -> bool {
        pos - len <= ROLLBACK
    }
}

struct DLoop {
    /// down, forget, gain of the query, key, value and MLP gates.
    gates: [(Dw, Dw, Dw); 4],
    /// `[4, H]`.
    query: u64,
}

struct DLayer {
    q: Dw,
    k: Dw,
    v: Dw,
    r: Dw,
    gate: Dw,
    sink: Dw,
    o: Dw,
    kproj: Option<Dw>,
    vproj: Option<Dw>,
    q_scale: u64,
    r_scale: u64,
    rel: u64,
    k_taps: u64,
    v_taps: u64,
    attn_taps: u64,
    mlp_taps: u64,
    /// One norm, or the query, key and value ones.
    ln: Vec<u64>,
    ln_post: u64,
    /// The attention residual's queries before the attention and their count (1 or 3).
    attn_query: Option<(u64, usize)>,
    mlp_query: u64,
    router: Dw,
    bias: u64,
    gu: Dw,
    dn: Dw,
    inter: bool,
    looped: Option<DLoop>,
    window: usize,
    extent: usize,
    nh: usize,
    nkv: usize,
    full: bool,
    boundary: bool,
}

/// The activation buffers of a pass of up to [`CHUNK`] tokens (one pass at a time).
struct Arena {
    info: u64,
    x: u64,
    pre_x: u64,
    bank_x: u64,
    a: u64,
    hn: u64,
    hn2: u64,
    o: u64,
    out: u64,
    states: Vec<u64>,
    msgs: [u64; 4],
    mixed: [u64; 3],
    gated: [u64; 3],
    xn: [u64; 3],
    cat: u64,
    gd: u64,
    gf: u64,
    gg: u64,
    q: u64,
    r: u64,
    g: u64,
    sink: u64,
    k: u64,
    v: u64,
    kd: u64,
    vd: u64,
    att: u64,
    part: u64,
    logits: u64,
    ids: u64,
    wts: u64,
    mid: u64,
    dmoe: u64,
    perm: u64,
    inv: u64,
    tok: u64,
    wrow: u64,
    tiles: u64,
    ntiles: u64,
    cnt: u64,
    /// Captured streams, `[MAX_CAPTURE][tokens][hidden]`.
    cap: u64,
    /// The unembedding of up to 16 rows.
    head: u64,
    head_out: u64,
}

pub(super) struct Engine {
    cu: Arc<Cu>,
    x: Arc<Cux>,
    ctx: *mut c_void,
    module: F,
    stream: F,
    funcs: HashMap<&'static str, F>,
    kfuncs: HashMap<u32, [F; 9]>,
    /// `OXINFER_INKLING_PROFILE=1`: time per kernel (calls, seconds), launches synchronized.
    prof: Option<Mutex<HashMap<usize, (u64, f64)>>>,
    /// Two buffers of a pass's `[tokens, hidden]` for the prefill convolutions (0: none yet).
    conv_scratch: (u64, u64),
    /// The split halves of a GEMM's input (hi, lo, capacity in half pairs).
    xsplit: (u64, u64, usize),
    /// Launches whose events are not read yet (function, start, end).
    pending: Mutex<Vec<(usize, usize, usize)>>,
    owned: Mutex<Vec<u64>>,
    layers: Vec<DLayer>,
    head: Dw,
    final_norm: u64,
    final_query: u64,
    arena: Mutex<Arena>,
    h: usize,
    vocab: usize,
    unpadded: usize,
    softcap: f32,
    eps: f32,
    conv_k: usize,
    d_rel: usize,
    window: usize,
    k_top: usize,
    ne: usize,
    mi: usize,
    route_scale: f32,
    beta: f32,
    lbeta: f32,
    clip_qk: f32,
    clip_rel: f32,
    alpha: f64,
    n_floor: f64,
    max_heads: usize,
    max_kv_heads: usize,
    rank: usize,
}
// SAFETY: as DevSess; one pass at a time (the arena's lock).
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(p) = &self.prof {
            let p = p.lock().unwrap();
            let mut v: Vec<(String, (u64, f64))> = p.iter().map(|(&f, &e)| (self.name_of(f), e)).collect();
            v.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
            let tot: f64 = v.iter().map(|e| e.1.1).sum();
            eprintln!("[inkling cuda profile: {:.3} s in kernels (CUDA events around each launch)]", tot);
            for (n, (c, t)) in v {
                eprintln!(
                    "  {n:<14} {c:>8} calls {:>9.3} ms {:>6.1}%  {:>7.1} us/call",
                    t * 1e3,
                    100.0 * t / tot,
                    t * 1e6 / c as f64
                );
            }
        }
        if unsafe { self.x.cuCtxSetCurrent(self.ctx) } != 0 {
            return;
        }
        unsafe {
            self.x.cuStreamSynchronize(self.stream);
            for &p in self.owned.lock().unwrap().iter() {
                self.x.cuMemFree_v2(p);
            }
            self.x.cuStreamDestroy_v2(self.stream);
            self.x.cuModuleUnload(self.module);
        }
    }
}

/// The i-quant grids and value tables, as device arrays.
fn tables_src() -> String {
    use crate::ggml::tables::*;
    use std::fmt::Write;
    let mut s = String::new();
    let mut arr = |ty: &str, name: &str, vals: Vec<String>| {
        let _ = write!(s, "__device__ const {ty} {name}[{}] = {{", vals.len());
        s.push_str(&vals.join(","));
        s.push_str("};\n");
    };
    arr("float", "kvalues_iq4nl", KVALUES_IQ4NL.iter().map(|v| format!("{v}.f")).collect());
    arr("float", "kvalues_mxfp4", KVALUES_MXFP4.iter().map(|v| format!("{v}.f")).collect());
    for (name, t) in [
        ("iq2xxs_grid", &IQ2XXS_GRID[..]),
        ("iq2xs_grid", &IQ2XS_GRID[..]),
        ("iq2s_grid", &IQ2S_GRID[..]),
        ("iq1s_grid", &IQ1S_GRID[..]),
    ] {
        arr("unsigned long long", name, t.iter().map(|v| format!("{v:#x}ull")).collect());
    }
    for (name, t) in [("iq3xxs_grid", &IQ3XXS_GRID[..]), ("iq3s_grid", &IQ3S_GRID[..])] {
        arr("unsigned int", name, t.iter().map(|v| format!("{v:#x}u")).collect());
    }
    s
}

/// The module for the weight kinds `kinds`.
pub(super) fn kernel_source(kinds: &[u32]) -> String {
    let mut s = tables_src();
    for k in kinds {
        s.push_str(&format!("#define HAS_{k} 1\n"));
    }
    s.push_str(include_str!("inkling_cuda.cu"));
    s
}

/// The module for `src`, compiled once per machine: kept in `$OXINFER_KERNEL_CACHE` (default
/// `~/.cache/oxinfer/kernels`; `0` turns it off) as one file holding the source and the binary,
/// used again only for exactly that source (NVRTC takes seconds for a dozen weight kinds).
fn cached_module(be: &CudaBackend, src: &str, kinds: &[u32]) -> Result<F> {
    let dir = match std::env::var("OXINFER_KERNEL_CACHE") {
        Ok(v) if v == "0" => None,
        Ok(v) => Some(std::path::PathBuf::from(v)),
        Err(_) => std::env::var("HOME")
            .ok()
            .map(|h| std::path::Path::new(&h).join(".cache").join("oxinfer").join("kernels")),
    };
    let (ma, mi) = be.sm();
    let ks: Vec<String> = kinds.iter().map(|k| k.to_string()).collect();
    let name = format!("inkling-sm{ma}{mi}-{}.bin", ks.join("-"));
    // [source length, u64][source][image]
    if let Some(d) = &dir
        && let Ok(f) = std::fs::read(d.join(&name))
        && f.len() >= 8
    {
        let n = u64::from_le_bytes(f[..8].try_into().unwrap()) as usize;
        if f.len() > 8 + n
            && &f[8..8 + n] == src.as_bytes()
            && let Ok(m) = be.load_image(&f[8 + n..])
        {
            return Ok(m);
        }
    }
    let img = be.compile(src, "inkling.cu")?;
    if let Some(d) = &dir
        && std::fs::create_dir_all(d).is_ok()
    {
        // Written under a temporary name, then renamed: a reader sees a whole file.
        let tmp = d.join(format!("{name}.{}", std::process::id()));
        let mut f = (src.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(src.as_bytes());
        f.extend_from_slice(&img);
        if std::fs::write(&tmp, &f).is_ok() && std::fs::rename(&tmp, d.join(&name)).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    be.load_image(&img)
}

const FIXED: [&str; 40] = [
    "attnt_f32_1",
    "attnt_f16_1",
    "attnt_q8_1",
    "attnt_f32_2",
    "attnt_f16_2",
    "attnt_q8_2",
    "attnt_f32_3",
    "attnt_f16_3",
    "attnt_q8_3",
    "split_rows",
    "sconv_par",
    "sconv_hist",
    "rms_rows",
    "mix_1",
    "mix_3",
    "mix_4",
    "gate_cat",
    "gate_mix",
    "add_k",
    "sconv",
    "vec_post",
    "kv_store",
    "attn_f32_1",
    "attn_f16_1",
    "attn_q8_1",
    "attn_f32_2",
    "attn_f16_2",
    "attn_q8_2",
    "attn_f32_3",
    "attn_f16_3",
    "attn_q8_3",
    "attn_f32_4",
    "attn_f16_4",
    "attn_q8_4",
    "attn_merge",
    "route_k",
    "moe_plan",
    "moe_rows",
    "moe_combine",
    "softcap_k",
];

fn ceil(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

impl Engine {
    /// The engine for `m` on the active CUDA backend, or None (the CPU path).
    pub(super) fn new(m: &Inkling2Model) -> Option<Engine> {
        if std::env::var("OXINFER_INKLING_GPU").is_ok_and(|v| v == "0") || !crate::gpu::decode_enabled() {
            return None;
        }
        let be: &'static CudaBackend = crate::gpu::active()?.as_any()?.downcast_ref::<CudaBackend>()?;
        if m.stream.is_some() {
            debug("streamed weights");
            return None;
        }
        let c = &m.cfg;
        for hd in [c.full, c.swa] {
            if hd.head_dim != 128 || hd.heads % hd.kv_heads != 0 || hd.heads / hd.kv_heads > 4 {
                debug("head layout (needs heads of 128, at most 4 query heads per key/value head)");
                return None;
            }
        }
        let h = c.hidden_size;
        let mi = c.moe_intermediate_size;
        if !h.is_multiple_of(64)
            || !mi.is_multiple_of(32)
            || c.d_rel > 32
            || c.conv_kernel_size > 8
            || c.conv_kernel_size < 2
            || c.n_routed_experts > 256
            || c.num_experts_per_tok > 32
        {
            debug("sizes");
            return None;
        }
        let rank = c.loop_cfg.map_or(0, |l| l.gate_rank);
        if !rank.is_multiple_of(32) {
            debug("gate rank");
            return None;
        }
        // Every weight as stored, and the kinds they need.
        let mut kinds: Vec<u32> = vec![0];
        let mut total = 0usize;
        let mut lins: Vec<&Lin> = vec![&m.lm_head];
        for l in &m.layers {
            let a = &l.attn;
            lins.extend([&a.q, &a.k, &a.v, &a.r, &a.gate, &a.sink, &a.o]);
            for cv in [&a.k_conv, &a.v_conv] {
                if let KvConv::Dynamic { proj, .. } = cv {
                    lins.push(proj);
                }
            }
            if let Some(li) = &l.looped {
                for g in &li.gates {
                    lins.extend([&g.down, &g.forget, &g.gain]);
                }
            }
            for gu in &l.experts.gate_up {
                match gu {
                    crate::moe::GateUp::Fused(w) => lins.push(w),
                    crate::moe::GateUp::Split(..) => {
                        debug("split gate / up experts");
                        return None;
                    }
                }
            }
            lins.extend(l.experts.down.iter());
            if !matches!(l.experts.act, crate::moe::Act::Situ { .. }) {
                debug("expert activation");
                return None;
            }
        }
        for l in &lins {
            let Some(hw) = host_w(l, 1) else {
                debug("a weight kind without a kernel, a LoRA or a folded weight");
                return None;
            };
            if !kinds.contains(&hw.kind) {
                kinds.push(hw.kind);
            }
            total += hw.data.len() + hw.scales.as_ref().map_or(0, |s| s.0.len() * 4);
        }
        kinds.sort_unstable();
        let mut e = match Self::bare(be, &kinds) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("[inkling cuda: {err}]");
                return None;
            }
        };
        let (mut free, mut tot) = (0usize, 0usize);
        unsafe { e.x.cuMemGetInfo_v2(&mut free, &mut tot) };
        let arena_bytes = Self::arena_bytes(m);
        if free < total + arena_bytes + RESERVE {
            debug(&format!(
                "{:.1} GB of weights and {:.1} GB of activations do not fit in {:.1} GB of free device memory",
                total as f64 / 1e9,
                arena_bytes as f64 / 1e9,
                free as f64 / 1e9
            ));
            return None;
        }
        let t0 = std::time::Instant::now();
        e.h = h;
        e.vocab = c.vocab_size;
        e.unpadded = c.unpadded_vocab_size;
        e.softcap = c.final_logit_softcapping as f32;
        e.eps = c.rms_norm_eps as f32;
        e.conv_k = c.conv_kernel_size;
        e.d_rel = c.d_rel;
        e.window = c.sliding_window_size;
        e.k_top = c.num_experts_per_tok;
        e.ne = c.n_routed_experts;
        e.mi = mi;
        e.route_scale = c.route_scale as f32;
        e.clip_qk = c.qk_l2_clip_threshold as f32;
        e.clip_rel = c.rel_l2_clip_threshold as f32;
        e.alpha = c.log_scaling_alpha;
        e.n_floor = c.log_scaling_n_floor;
        e.max_heads = c.full.heads.max(c.swa.heads);
        e.max_kv_heads = c.full.kv_heads.max(c.swa.kv_heads);
        e.rank = rank;
        let kfuncs = e.kfuncs.clone();
        match e.upload_model(m, &kfuncs) {
            Ok(()) => {}
            Err(err) => {
                debug(&format!("upload: {err}"));
                return None;
            }
        }
        let arena = e.make_arena(m)?;
        *e.arena.lock().unwrap() = arena;
        let cw = (CHUNK * h.max(e.max_kv_heads * 128)) * 4;
        e.conv_scratch = (e.alloc(cw).ok()?, e.alloc(cw).ok()?);
        // The largest GEMM input: a chunk of the widest projection input, or the experts' rows.
        let widest = (2 * h).max(e.max_heads * 128).max(mi);
        let pairs = (CHUNK * widest).max(CHUNK * e.k_top * mi) / 2;
        e.xsplit = (e.alloc(pairs * 4).ok()?, e.alloc(pairs * 4).ok()?, pairs);
        debug(&format!(
            "{:.1} GB of weights on the device in {:.1} s (kinds {kinds:?})",
            total as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        ));
        Some(e)
    }

    /// The kernels for weight kinds `kinds` (sorted) on `be`'s context, and nothing else.
    fn bare(be: &CudaBackend, kinds: &[u32]) -> Result<Engine> {
        let x: Arc<Cux> = {
            use dlsym::Bindings;
            // SAFETY: libcuda is already initialised by the backend.
            Arc::new(
                unsafe { Cux::open_first(crate::platform::gpu::cuda::CANDIDATES.iter().copied()) }.map_err(|e| anyhow!("libcuda: {e}"))?,
            )
        };
        let ctx = be.context().ok_or_else(|| anyhow!("no context"))?;
        let module = cached_module(be, &kernel_source(kinds), kinds)?;
        let get = |name: &str| -> Result<F> {
            let cn = CString::new(name)?;
            let mut f = std::ptr::null_mut();
            if unsafe { x.cuModuleGetFunction(&mut f, module, cn.as_ptr()) } != 0 {
                bail!("kernel {name} missing");
            }
            Ok(f)
        };
        let mut funcs: HashMap<&'static str, F> = HashMap::new();
        for n in FIXED {
            funcs.insert(n, get(n)?);
        }
        let mut kfuncs: HashMap<u32, [F; 9]> = HashMap::new();
        for &k in kinds {
            kfuncs.insert(
                k,
                [
                    get(&format!("mv_{k}"))?,
                    get(&format!("gemm_{k}"))?,
                    get(&format!("moe_gu_{k}"))?,
                    get(&format!("moe_dn_{k}"))?,
                    get(&format!("mvm_{k}"))?,
                    get(&format!("gate_down_{k}"))?,
                    get(&format!("gate_out_{k}"))?,
                    get(&format!("gemmi_{k}"))?,
                    get(&format!("gate_down_t_{k}"))?,
                ],
            );
        }
        let mut stream = std::ptr::null_mut();
        if unsafe { x.cuStreamCreate(&mut stream, 1) } != 0 {
            bail!("cuStreamCreate failed");
        }
        let null = std::ptr::null_mut();
        Ok(Engine {
            cu: be.driver().clone(),
            x,
            ctx,
            module,
            stream,
            funcs,
            kfuncs,
            prof: std::env::var("OXINFER_INKLING_PROFILE")
                .is_ok_and(|v| v != "0")
                .then(|| Mutex::new(HashMap::new())),
            pending: Mutex::new(Vec::new()),
            conv_scratch: (0, 0),
            xsplit: (0, 0, 0),
            owned: Mutex::new(Vec::new()),
            layers: Vec::new(),
            head: Dw {
                a: WArg::default(),
                kind: 0,
                n_out: 0,
                n_in: 0,
                mv: null,
                gemm: null,
                gu: null,
                dn: null,
                mvm: null,
                gdown: null,
                gout: null,
                gdown_t: null,
                gemmi: null,
            },
            final_norm: 0,
            final_query: 0,
            arena: Mutex::new(Self::empty_arena()),
            h: 0,
            vocab: 0,
            unpadded: 0,
            softcap: 0.0,
            eps: 1e-6,
            conv_k: 4,
            d_rel: 0,
            window: 0,
            k_top: 0,
            ne: 0,
            mi: 0,
            route_scale: 1.0,
            beta: 0.0,
            lbeta: 0.0,
            clip_qk: 0.0,
            clip_rel: 0.0,
            alpha: 0.0,
            n_floor: 1.0,
            max_heads: 0,
            max_kv_heads: 0,
            rank: 0,
        })
    }

    fn arena_bytes(m: &Inkling2Model) -> usize {
        let c = &m.cfg;
        let (h, t) = (c.hidden_size, CHUNK);
        let nh = c.full.heads.max(c.swa.heads);
        let kw = (c.full.kv_heads * c.full.head_dim).max(c.swa.kv_heads * c.swa.head_dim);
        let tk = t * c.num_experts_per_tok;
        4 * (t * h * 30
            + t * 2 * h
            + 3 * t * nh * 128
            + 2 * t * kw
            + t * nh * 130 * 2
            + tk * (c.moe_intermediate_size + h + 4)
            + 16 * c.vocab_size * 2)
    }

    fn empty_arena() -> Arena {
        Arena {
            info: 0,
            x: 0,
            pre_x: 0,
            bank_x: 0,
            a: 0,
            hn: 0,
            hn2: 0,
            o: 0,
            out: 0,
            states: Vec::new(),
            msgs: [0; 4],
            mixed: [0; 3],
            gated: [0; 3],
            xn: [0; 3],
            cat: 0,
            gd: 0,
            gf: 0,
            gg: 0,
            q: 0,
            r: 0,
            g: 0,
            sink: 0,
            k: 0,
            v: 0,
            kd: 0,
            vd: 0,
            att: 0,
            part: 0,
            logits: 0,
            ids: 0,
            wts: 0,
            mid: 0,
            dmoe: 0,
            perm: 0,
            inv: 0,
            tok: 0,
            wrow: 0,
            tiles: 0,
            ntiles: 0,
            cnt: 0,
            cap: 0,
            head: 0,
            head_out: 0,
        }
    }

    /// A device allocation owned by the engine.
    fn alloc(&self, bytes: usize) -> Result<u64> {
        let mut d = 0u64;
        if unsafe { self.x.cuMemAlloc_v2(&mut d, bytes.max(256)) } != 0 {
            bail!("out of device memory ({bytes} bytes)");
        }
        self.owned.lock().unwrap().push(d);
        Ok(d)
    }

    fn up_bytes(&self, b: &[u8]) -> Result<u64> {
        let d = self.alloc(b.len())?;
        if !b.is_empty() && unsafe { self.cu.cuMemcpyHtoD_v2(d, b.as_ptr() as *const c_void, b.len()) } != 0 {
            bail!("copy to the device failed");
        }
        Ok(d)
    }

    fn upf(&self, v: &[f32]) -> Result<u64> {
        // SAFETY: f32 to bytes of the same memory.
        self.up_bytes(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) })
    }

    /// One matrix, or (several) experts stacked at a common stride.
    fn up_lins(&self, ls: &[&Lin], n_out: usize, n_in: usize, kf: &HashMap<u32, [F; 9]>) -> Result<Dw> {
        let hs: Vec<HostW> = ls
            .iter()
            .map(|l| host_w(l, n_in).ok_or_else(|| anyhow!("unsupported weight")))
            .collect::<Result<_>>()?;
        let k0 = hs[0].kind;
        let rb = hs[0].rb;
        if hs
            .iter()
            .any(|h| h.kind != k0 || h.rb != rb || h.data.len() != rb * n_out || h.scales.is_some() != hs[0].scales.is_some())
        {
            bail!("experts of different kinds or shapes");
        }
        if !n_in.is_multiple_of(32) || (k0 == 40 && !n_in.is_multiple_of(64)) {
            bail!("rows of {n_in} values");
        }
        let es = (rb * n_out).next_multiple_of(256);
        let q = self.alloc(es * hs.len())?;
        for (i, hw) in hs.iter().enumerate() {
            if unsafe {
                self.cu
                    .cuMemcpyHtoD_v2(q + (i * es) as u64, hw.data.as_ptr() as *const c_void, hw.data.len())
            } != 0
            {
                bail!("copy to the device failed");
            }
        }
        let mut a = WArg {
            q,
            rb: rb as u64,
            es: es as u64,
            f: 1.0,
            sr: 1,
            sc: 32,
            ncb: 1,
            ..Default::default()
        };
        if let Some((s0, sr, sc)) = &hs[0].scales {
            let n = s0.len();
            let s = self.alloc(n * 4 * hs.len())?;
            for (i, hw) in hs.iter().enumerate() {
                let sv = &hw.scales.as_ref().unwrap().0;
                if sv.len() != n {
                    bail!("experts with different scale tables");
                }
                if unsafe { self.cu.cuMemcpyHtoD_v2(s + (i * n * 4) as u64, sv.as_ptr() as *const c_void, n * 4) } != 0 {
                    bail!("copy to the device failed");
                }
            }
            a.s = s;
            a.ss = n as u64;
            a.sr = *sr as u32;
            a.sc = *sc as u32;
            a.ncb = n_in.div_ceil(*sc) as u32;
        }
        let fs: Vec<f32> = hs.iter().map(|h| h.f).collect();
        if fs.iter().all(|&f| f == fs[0]) {
            a.f = fs[0];
        } else {
            a.ef = self.upf(&fs)?;
        }
        let f = kf.get(&k0).ok_or_else(|| anyhow!("kind {k0} not compiled"))?;
        Ok(Dw {
            a,
            kind: k0,
            n_out,
            n_in,
            mv: f[0],
            gemm: f[1],
            gu: f[2],
            dn: f[3],
            mvm: f[4],
            gdown: f[5],
            gout: f[6],
            gemmi: f[7],
            gdown_t: f[8],
        })
    }

    /// A float32 matrix from host values.
    fn up_f32(&self, w: &[f32], n_out: usize, n_in: usize, kf: &HashMap<u32, [F; 9]>) -> Result<Dw> {
        let f = kf[&0];
        Ok(Dw {
            a: WArg {
                q: self.upf(w)?,
                rb: (n_in * 4) as u64,
                f: 1.0,
                sr: 1,
                sc: 32,
                ncb: 1,
                ..Default::default()
            },
            kind: 0,
            n_out,
            n_in,
            mv: f[0],
            gemm: f[1],
            gu: f[2],
            dn: f[3],
            mvm: f[4],
            gdown: f[5],
            gout: f[6],
            gemmi: f[7],
            gdown_t: f[8],
        })
    }

    fn upload_model(&mut self, m: &Inkling2Model, kf: &HashMap<u32, [F; 9]>) -> Result<()> {
        let c = &m.cfg;
        let (h, kn) = (c.hidden_size, c.conv_kernel_size);
        self.head = self.up_lins(&[&m.lm_head], c.vocab_size, h, kf)?;
        self.final_norm = self.upf(&m.final_norm)?;
        self.final_query = self.upf(&m.final_query)?;
        let mut layers = Vec::new();
        let mut acts = (0f32, 0f32);
        for (li, l) in m.layers.iter().enumerate() {
            let a = &l.attn;
            let hd = a.hd;
            let (w, kw) = (hd.heads * hd.head_dim, hd.kv_heads * hd.head_dim);
            let dr = c.d_rel;
            let one = |lin: &Lin, n_out: usize, n_in: usize| self.up_lins(&[lin], n_out, n_in, kf);
            let (k_taps, kproj) = match &a.k_conv {
                KvConv::Static(t) => (self.upf(t)?, None),
                KvConv::Dynamic { taps, proj } => (self.upf(taps)?, Some(one(proj, hd.kv_heads * kn, h)?)),
            };
            let (v_taps, vproj) = match &a.v_conv {
                KvConv::Static(t) => (self.upf(t)?, None),
                KvConv::Dynamic { taps, proj } => (self.upf(taps)?, Some(one(proj, hd.kv_heads * kn, h)?)),
            };
            let ln = match &l.ln_in {
                InNorm::One(w) => vec![self.upf(w)?],
                InNorm::Qkv(a, b, cc) => vec![self.upf(a)?, self.upf(b)?, self.upf(cc)?],
            };
            let looped = match &l.looped {
                None => None,
                Some(lp) => {
                    let g = |gt: &Gate| -> Result<(Dw, Dw, Dw)> {
                        Ok((
                            one(&gt.down, self.rank, 2 * h)?,
                            one(&gt.forget, h, self.rank)?,
                            one(&gt.gain, h, self.rank)?,
                        ))
                    };
                    Some(DLoop {
                        gates: [g(&lp.gates[0])?, g(&lp.gates[1])?, g(&lp.gates[2])?, g(&lp.gates[3])?],
                        query: self.upf(&lp.query)?,
                    })
                }
            };
            let ex = &l.experts;
            let gu: Vec<&Lin> = ex
                .gate_up
                .iter()
                .map(|g| match g {
                    crate::moe::GateUp::Fused(w) => w,
                    crate::moe::GateUp::Split(w, _) => w,
                })
                .collect();
            let dn: Vec<&Lin> = ex.down.iter().collect();
            if let crate::moe::Act::Situ { beta, linear_beta } = ex.act {
                acts = (beta, linear_beta);
            }
            layers.push(DLayer {
                q: one(&a.q, w, h)?,
                k: one(&a.k, kw, h)?,
                v: one(&a.v, kw, h)?,
                r: one(&a.r, hd.heads * dr, h)?,
                gate: one(&a.gate, w, h)?,
                sink: one(&a.sink, hd.heads, h)?,
                o: one(&a.o, h, w)?,
                kproj,
                vproj,
                q_scale: self.upf(&a.q_scale)?,
                r_scale: self.upf(&a.r_scale)?,
                rel: self.upf(&a.rel)?,
                k_taps,
                v_taps,
                attn_taps: self.upf(&l.attn_conv)?,
                mlp_taps: self.upf(&l.mlp_conv)?,
                ln,
                ln_post: self.upf(&l.ln_post)?,
                attn_query: match &l.attn_query {
                    Some(q) => Some((self.upf(q)?, q.len() / h)),
                    None => None,
                },
                mlp_query: self.upf(&l.mlp_query)?,
                router: self.up_f32(&l.router, c.n_routed_experts, h, kf)?,
                bias: self.upf(&l.bias)?,
                gu: self.up_lins(&gu, 2 * self.mi, h, kf)?,
                dn: self.up_lins(&dn, h, self.mi, kf)?,
                inter: ex.interleaved,
                looped,
                window: a.window.unwrap_or(0),
                extent: a.extent,
                nh: hd.heads,
                nkv: hd.kv_heads,
                full: a.window.is_none(),
                boundary: l.boundary,
            });
            let _ = li;
        }
        self.layers = layers;
        (self.beta, self.lbeta) = acts;
        Ok(())
    }

    fn make_arena(&self, m: &Inkling2Model) -> Option<Arena> {
        let c = &m.cfg;
        let (h, t) = (c.hidden_size, CHUNK);
        let nh = self.max_heads;
        let kw = self.max_kv_heads * 128;
        let tk = t * self.k_top;
        let f = |n: usize| self.alloc(n * 4).ok();
        let part = (t * nh).max(16 * nh * SPLITS_FULL) * 130;
        // Block states: the deepest stack plus the previous pass's.
        let n_states = (c.num_hidden_layers / c.n_layers_per_attn_res_block + 3).min(MAX_STATES + 2);
        Some(Arena {
            info: f(4)?,
            x: f(t * h)?,
            pre_x: f(t * h)?,
            bank_x: f(t * h)?,
            a: f(t * h)?,
            hn: f(t * h)?,
            hn2: f(t * h)?,
            o: f(t * h)?,
            out: f(t * h)?,
            states: (0..n_states).map(|_| f(t * h)).collect::<Option<_>>()?,
            msgs: [f(t * h)?, f(t * h)?, f(t * h)?, f(t * h)?],
            mixed: [f(t * h)?, f(t * h)?, f(t * h)?],
            gated: [f(t * h)?, f(t * h)?, f(t * h)?],
            xn: [f(t * h)?, f(t * h)?, f(t * h)?],
            cat: f(t * 2 * h)?,
            gd: f(t * self.rank.max(1))?,
            gf: f(t * h)?,
            gg: f(t * h)?,
            q: f(t * nh * 128)?,
            r: f(t * nh * self.d_rel.max(1))?,
            g: f(t * nh * 128)?,
            sink: f(t * nh)?,
            k: f(t * kw)?,
            v: f(t * kw)?,
            kd: f(t * self.max_kv_heads * self.conv_k)?,
            vd: f(t * self.max_kv_heads * self.conv_k)?,
            att: f(t * nh * 128)?,
            part: f(part)?,
            logits: f(t * self.ne)?,
            ids: f(tk)?,
            wts: f(tk)?,
            mid: f(tk * self.mi)?,
            dmoe: f(tk * h)?,
            perm: f(tk)?,
            inv: f(tk)?,
            tok: f(tk)?,
            wrow: f(tk)?,
            tiles: f(3 * (tk / 64 + self.ne + 1))?,
            ntiles: f(1)?,
            cnt: f(self.ne)?,
            cap: f(MAX_CAPTURE * t * h)?,
            head: f(16 * self.vocab)?,
            head_out: f(16 * self.vocab)?,
        })
    }

    fn f(&self, name: &str) -> F {
        self.funcs[name]
    }

    /// The attention kernel for a cache format and query heads per key/value head.
    fn attn_f(&self, fmt: u32, g: usize) -> F {
        let n = [
            ["attn_f32_1", "attn_f32_2", "attn_f32_3", "attn_f32_4"],
            ["attn_f16_1", "attn_f16_2", "attn_f16_3", "attn_f16_4"],
            ["attn_q8_1", "attn_q8_2", "attn_q8_3", "attn_q8_4"],
        ][fmt as usize][g - 1];
        self.f(n)
    }

    /// Launches `f` on the engine's stream.
    fn go(&self, f: F, grid: (u32, u32, u32), block: u32, args: &mut [*mut c_void]) {
        let ev = self.prof.as_ref().map(|_| {
            let (mut a, mut b) = (std::ptr::null_mut(), std::ptr::null_mut());
            unsafe {
                self.x.cuEventCreate(&mut a, 0);
                self.x.cuEventCreate(&mut b, 0);
                self.x.cuEventRecord(a, self.stream);
            }
            (a, b)
        });
        let rc = unsafe {
            self.x.cuLaunchKernel(
                f,
                grid.0.max(1),
                grid.1.max(1),
                grid.2.max(1),
                block,
                1,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert!(rc == 0, "inkling cuda: kernel launch failed ({rc})");
        if let Some((a, b)) = ev {
            unsafe { self.x.cuEventRecord(b, self.stream) };
            self.pending.lock().unwrap().push((f as usize, a as usize, b as usize));
        }
    }

    /// Adds the timed launches since the last call to the profile (after a synchronization).
    fn prof_collect(&self) {
        let Some(p) = &self.prof else { return };
        let mut p = p.lock().unwrap();
        for (f, a, b) in self.pending.lock().unwrap().drain(..) {
            let mut ms = 0f32;
            unsafe {
                self.x.cuEventElapsedTime(&mut ms, a as F, b as F);
                self.x.cuEventDestroy_v2(a as F);
                self.x.cuEventDestroy_v2(b as F);
            }
            let e = p.entry(f).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms as f64 / 1e3;
        }
    }

    /// The kernel name of a function (profiles).
    fn name_of(&self, f: usize) -> String {
        if let Some((n, _)) = self.funcs.iter().find(|(_, g)| **g as usize == f) {
            return n.to_string();
        }
        for (k, fs) in &self.kfuncs {
            for (i, &g) in fs.iter().enumerate() {
                if g as usize == f {
                    return format!(
                        "{}_{k}",
                        [
                            "mv",
                            "gemm",
                            "moe_gu",
                            "moe_dn",
                            "mvm",
                            "gate_down",
                            "gate_out",
                            "gemmi",
                            "gate_down_t"
                        ][i]
                    );
                }
            }
        }
        format!("{f:#x}")
    }

    fn copy(&self, dst: u64, src: u64, floats: usize) {
        unsafe { self.x.cuMemcpyDtoDAsync_v2(dst, src, floats * 4, self.stream) };
    }

    /// y (= or +=) W x over t rows (strides in floats).
    #[allow(clippy::too_many_arguments)]
    fn lin(&self, w: &Dw, x: u64, y: u64, t: usize, xs: usize, ys: usize, add: bool) {
        let (n_in, n_out) = (w.n_in as u32, w.n_out as u32);
        if t <= MV_T {
            self.go(
                w.mv,
                (ceil(w.n_out, 4), t as u32, 1),
                128,
                &mut args![w.a, x, y, n_in, n_out, xs as u32, ys as u32, add as u32],
            );
        } else {
            let sp = self.split(x, xs, t, w.n_in);
            self.gemm(w, x, y, t, xs, ys, add, sp);
        }
    }

    /// The split halves of `rows` rows of `k` values (for the split GEMMs), in the engine's
    /// buffer; (0, 0) when the GEMMs take f16 operands or the buffer is too small.
    fn split(&self, x: u64, xs: usize, rows: usize, k: usize) -> (u64, u64) {
        let (hi, lo, cap) = self.xsplit;
        if !split_gemm() || hi == 0 || rows * k / 2 > cap {
            return (0, 0);
        }
        let n = (rows * k / 2) as u32;
        self.go(
            self.f("split_rows"),
            (n.div_ceil(256), 1, 1),
            256,
            &mut args![x, xs as u32, rows as u32, k as u32, hi, lo],
        );
        (hi, lo)
    }

    /// The GEMM of `w` over `t` rows of x (pre-split halves `sp`, or (0, 0)).
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, w: &Dw, x: u64, y: u64, t: usize, xs: usize, ys: usize, add: bool, sp: (u64, u64)) {
        let p = GemmP {
            x,
            y,
            xh: sp.0,
            xl: sp.1,
            n_in: w.n_in as u32,
            n_out: w.n_out as u32,
            rows: t as u32,
            xs: xs as u32,
            ys: ys as u32,
            mode: add as u32,
            ..Default::default()
        };
        self.go(w.gemm_f(), (ceil(w.n_out, 64), ceil(t, 64), 1), 128, &mut args![w.a, p]);
    }

    /// Several matrices over one input (`jobs`: matrix, output, output stride): one launch per
    /// run of up to 8 of a kind on the matvec path.
    fn lin_many(&self, jobs: &[(&Dw, u64, usize)], x: u64, xs: usize, t: usize) {
        if t > MV_T {
            // One split of the input for all of them.
            let n_in = jobs[0].0.n_in;
            let sp = if jobs.iter().all(|j| j.0.n_in == n_in) {
                self.split(x, xs, t, n_in)
            } else {
                (0, 0)
            };
            for &(w, y, ys) in jobs {
                self.gemm(w, x, y, t, xs, ys, false, sp);
            }
            return;
        }
        let mut i = 0;
        while i < jobs.len() {
            let w0 = jobs[i].0;
            let mut p = MvJobs {
                x,
                n_in: w0.n_in as u32,
                xs: xs as u32,
                ..Default::default()
            };
            let mut blocks = 0u32;
            while i < jobs.len() && (p.n as usize) < 8 && jobs[i].0.kind == w0.kind && jobs[i].0.n_in == w0.n_in {
                let (w, y, ys) = jobs[i];
                p.j[p.n as usize] = MvJob {
                    w: w.a,
                    y,
                    n_out: w.n_out as u32,
                    ys: ys as u32,
                    b0: blocks,
                    _p: 0,
                };
                blocks += ceil(w.n_out, 8);
                p.n += 1;
                i += 1;
            }
            self.go(w0.mvm, (blocks, t as u32, 1), 256, &mut args![p]);
        }
    }

    fn rms(&self, x: u64, y: u64, w: u64, n: usize, eps: f32, t: usize) {
        self.go(
            self.f("rms_rows"),
            (t as u32, 1, 1),
            256,
            &mut args![x, y, w, n as u32, eps, n as u32, n as u32],
        );
    }

    /// The softmax mixes of `states` and `x` for `nq` queries into `outs`, RMS-normed with the
    /// weights `norms` (one per query) when given.
    #[allow(clippy::too_many_arguments)]
    fn mix(&self, states: &[u64], x: u64, q: u64, nq: usize, outs: &[u64], norms: &[u64], t: usize) {
        assert!(states.len() <= MAX_STATES && matches!(nq, 1 | 3 | 4) && (norms.is_empty() || norms.len() == nq));
        let mut p = MixP {
            x,
            q,
            ns: states.len() as u32,
            nq: nq as u32,
            h: self.h as u32,
            eps: MIX_EPS,
            neps: self.eps,
            ..Default::default()
        };
        p.st[..states.len()].copy_from_slice(states);
        p.out[..nq].copy_from_slice(&outs[..nq]);
        p.nw[..norms.len()].copy_from_slice(norms);
        let f = match nq {
            1 => self.f("mix_1"),
            2 | 3 => self.f("mix_3"),
            _ => self.f("mix_4"),
        };
        // A query slot without a query reads the last one again (its output is unused).
        for i in nq..[1, 1, 3, 3, 4][nq] {
            p.out[i] = p.out[nq - 1];
            p.nw[i] = p.nw[nq - 1];
        }
        // Decode: as many threads as the kernel takes (latency); prefill: 256.
        let threads = if t > 16 {
            256
        } else if nq == 1 {
            1024
        } else {
            512
        };
        self.go(f, (t as u32, 1, 1), threads, &mut args![p]);
    }

    /// A message gate over input `x` and message `msg` into `out`.
    fn gate(&self, ar: &Arena, g: &(Dw, Dw, Dw), x: u64, msg: u64, out: u64, t: usize) {
        let h = self.h;
        if t <= MV_T && g.1.kind == g.2.kind {
            let r = self.rank as u32;
            self.go(
                g.0.gdown,
                (r, t as u32, 1),
                256,
                &mut args![g.0.a, x, msg, ar.gd, h as u32, r, self.eps],
            );
            self.go(
                g.1.gout,
                (ceil(h, 8), t as u32, 1),
                256,
                &mut args![g.1.a, g.2.a, ar.gd, x, msg, out, h as u32, r],
            );
            return;
        }
        if 2 * h <= 4608 {
            // A block per token (`gate_down_t`'s shared input holds 4608 values).
            let r = self.rank as u32;
            self.go(
                g.0.gdown_t,
                (t as u32, 1, 1),
                256,
                &mut args![g.0.a, x, msg, ar.gd, h as u32, r, self.eps],
            );
        } else {
            self.go(
                self.f("gate_cat"),
                (t as u32, 1, 1),
                256,
                &mut args![x, msg, ar.cat, h as u32, self.eps],
            );
            self.lin(&g.0, ar.cat, ar.gd, t, 2 * h, self.rank, false);
        }
        self.lin(&g.1, ar.gd, ar.gf, t, self.rank, h, false);
        self.lin(&g.2, ar.gd, ar.gg, t, self.rank, h, false);
        let n = (t * h) as u32;
        self.go(
            self.f("gate_mix"),
            (n.div_ceil(256), 1, 1),
            256,
            &mut args![x, msg, ar.gf, ar.gg, out, n],
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn sconv(&self, x: u64, hist: u64, taps: u64, dyn_: u64, ch: usize, groups: usize, t: usize, norm: f32, info: u64) {
        self.sconvs(&[ConvJob { x, hist, taps, dyn_ }], ch, groups, t, norm, info);
    }

    /// One or two convolutions of the same shape in one launch, for the positions from
    /// `info[0]`, their histories rings of `kernel - 1 + ROLLBACK` rows by position.
    #[allow(clippy::too_many_arguments)]
    fn sconvs(&self, jobs: &[ConvJob], ch: usize, groups: usize, t: usize, norm: f32, info: u64) {
        let ring = self.conv_k - 1 + ROLLBACK;
        let mut p = ConvJobs {
            info,
            ch: ch as u32,
            k: self.conv_k as u32,
            t: t as u32,
            groups: groups as u32,
            ring: ring as u32,
            norm,
            ..Default::default()
        };
        p.j[..jobs.len()].copy_from_slice(jobs);
        let ar = self.conv_scratch;
        if t < 8 || ar.0 == 0 {
            self.go(self.f("sconv"), (ceil(ch, 128), jobs.len() as u32, 1), 128, &mut args![p]);
            return;
        }
        // Prefill: a thread per (channel, token) into scratch, then the history, then the copy.
        let n = jobs.len() as u32;
        self.go(self.f("sconv_par"), (ceil(ch, 128), t as u32, n), 128, &mut args![p, ar.0, ar.1]);
        self.go(self.f("sconv_hist"), (ceil(t.min(ring) * ch, 128), 1, n), 128, &mut args![p]);
        for (j, y) in jobs.iter().zip([ar.0, ar.1]) {
            self.copy(j.x, y, t * ch);
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn vec_post(&self, x: u64, scale: u64, n: usize, nv: usize, stride: usize, t: usize, mode: u32, clip: f32, alpha: f64, info: u64) {
        self.vec_posts(&[(x, scale, n, nv, stride, mode, clip, alpha)], t, info);
    }

    /// Up to 4 [`Self::vec_post`]s (x, scale, n, nv, stride, mode, clip, alpha) in one launch.
    #[allow(clippy::type_complexity)]
    fn vec_posts(&self, jobs: &[(u64, u64, usize, usize, usize, u32, f32, f64)], t: usize, info: u64) {
        let mut p = VecJobs {
            info,
            floor: self.n_floor,
            eps: self.eps,
            n: jobs.len() as u32,
            ..Default::default()
        };
        let mut blocks = 0u32;
        for (i, &(x, scale, n, nv, stride, mode, clip, alpha)) in jobs.iter().enumerate() {
            let total = t * nv;
            p.j[i] = VecJob {
                x,
                scale,
                alpha,
                n: n as u32,
                nv: nv as u32,
                stride: stride as u32,
                total: total as u32,
                mode,
                b0: blocks,
                clip,
                _p: 0.0,
            };
            blocks += ceil(total, 8);
        }
        self.go(self.f("vec_post"), (blocks, 1, 1), 256, &mut args![p]);
    }

    /// The attention of layer `l` over the normed inputs into `ar.a`, appending to slot `s`.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, ar: &Arena, l: &DLayer, xq: u64, xk: u64, xv: u64, s: &Slot, fmt: u32, t: usize) {
        let (nh, nkv, d, dr, kn) = (l.nh, l.nkv, 128usize, self.d_rel, self.conv_k);
        let (w, kw, h) = (nh * d, nkv * d, self.h);
        let mut jq: Vec<(&Dw, u64, usize)> = vec![(&l.q, ar.q, w), (&l.r, ar.r, nh * dr), (&l.gate, ar.g, w), (&l.sink, ar.sink, nh)];
        let mut jk: Vec<(&Dw, u64, usize)> = vec![(&l.k, ar.k, kw)];
        let mut jv: Vec<(&Dw, u64, usize)> = vec![(&l.v, ar.v, kw)];
        if let Some(p) = &l.kproj {
            jk.push((p, ar.kd, nkv * kn));
        }
        if let Some(p) = &l.vproj {
            jv.push((p, ar.vd, nkv * kn));
        }
        if xq == xk && xq == xv {
            jq.append(&mut jk);
            jq.append(&mut jv);
            // Same kinds next to each other (fewer launches).
            jq.sort_by_key(|j| j.0.kind);
            self.lin_many(&jq, xq, h, t);
        } else {
            self.lin_many(&jq, xq, h, t);
            self.lin_many(&jk, xk, h, t);
            self.lin_many(&jv, xv, h, t);
        }
        let (dyn_k, dyn_v, norm) = match (&l.kproj, &l.vproj) {
            (Some(_), Some(_)) => (ar.kd, ar.vd, 1.0 / ((2 * kn + 1) as f32).sqrt()),
            _ => (0, 0, 1.0 / ((kn + 1) as f32).sqrt()),
        };
        self.sconvs(
            &[
                ConvJob {
                    x: ar.k,
                    hist: s.hist[0],
                    taps: l.k_taps,
                    dyn_: dyn_k,
                },
                ConvJob {
                    x: ar.v,
                    hist: s.hist[1],
                    taps: l.v_taps,
                    dyn_: dyn_v,
                },
            ],
            kw,
            nkv,
            t,
            norm,
            ar.info,
        );
        let alpha = if l.full { self.alpha } else { 0.0 };
        self.vec_posts(
            &[
                (ar.q, l.q_scale, d, nh, w, 0, self.clip_qk, alpha),
                (ar.r, l.r_scale, dr, nh, nh * dr, 0, self.clip_rel, alpha),
                (ar.k, 0, d, nkv, kw, 0, self.clip_qk, 0.0),
                (ar.v, 0, d, nkv, kw, 1, 0.0, 0.0),
            ],
            t,
            ar.info,
        );
        let groups = (2 * t * kw / 32) as u32;
        self.go(
            self.f("kv_store"),
            (groups.div_ceil(128), 1, 1),
            128,
            &mut args![ar.k, ar.v, s.k, s.v, s.ks, s.vs, kw as u32, t as u32, s.cap as u32, fmt, ar.info],
        );
        // Up to MV_T tokens: each token's keys split as in a pass of its own.
        let ns = if t <= MV_T {
            if l.full { SPLITS_FULL } else { SPLITS_SLIDING }
        } else {
            (256 / (t * nkv)).clamp(1, if l.full { SPLITS_FULL } else { SPLITS_SLIDING })
        };
        let p = AttP {
            q: ar.q,
            r: ar.r,
            kc: s.k,
            vc: s.v,
            ks: s.ks,
            vs: s.vs,
            rel: l.rel,
            part: ar.part,
            info: ar.info,
            nh: nh as u32,
            nkv: nkv as u32,
            dr: dr as u32,
            extent: l.extent as u32,
            window: l.window as u32,
            cap: s.cap as u32,
            ns: ns as u32,
            qs: w as u32,
            rs: (nh * dr) as u32,
            scaling: (d as f32).powf(-0.5),
        };
        if t > MV_T && nh / nkv <= 3 {
            // Prefill: query tiles of 8 sharing their keys and values (`attnt_*`); splits of the
            // keys when the tiles alone do not fill the device.
            let tiles = t.div_ceil(8);
            // (The partial results hold 1024 x heads rows: tokens x splits stays within that.)
            let ns = (512 / (tiles * nkv)).clamp(1, (1024 / t).clamp(1, 32)) as u32;
            let mut p = p;
            p.ns = ns;
            let name = format!("attnt_{}_{}", ["f32", "f16", "q8"][fmt as usize], nh / nkv);
            let f = *self.funcs.iter().find(|(k, _)| **k == name).unwrap().1;
            self.go(f, (nkv as u32, tiles as u32, ns), 128, &mut args![p, t as u32]);
            self.go(
                self.f("attn_merge"),
                (nh as u32, t as u32, 1),
                128,
                &mut args![ar.part, ar.sink, ar.g, ar.att, nh as u32, ns, nh as u32, w as u32],
            );
            self.lin(&l.o, ar.att, ar.a, t, w, h, false);
            return;
        }
        self.go(self.attn_f(fmt, nh / nkv), (nkv as u32, t as u32, ns as u32), 128, &mut args![p]);
        self.go(
            self.f("attn_merge"),
            (nh as u32, t as u32, 1),
            128,
            &mut args![ar.part, ar.sink, ar.g, ar.att, nh as u32, ns as u32, nh as u32, w as u32],
        );
        self.lin(&l.o, ar.att, ar.a, t, w, h, false);
    }

    /// The routed experts of `l` over `x` into `y`.
    fn moe(&self, ar: &Arena, l: &DLayer, x: u64, y: u64, t: usize) {
        self.experts(ar, &l.router, l.bias, &l.gu, &l.dn, l.inter, x, y, t)
    }

    /// Routing (`router`, `bias`) and the routed experts (`gu`, `dn`) over `x` into `y`.
    #[allow(clippy::too_many_arguments)]
    fn experts(&self, ar: &Arena, router: &Dw, bias: u64, gu: &Dw, dn: &Dw, inter: bool, x: u64, y: u64, t: usize) {
        let (h, ne, k, mi) = (self.h, self.ne, self.k_top, self.mi);
        // The router in f32 on the matvec kernel whatever t (the choice of experts is sensitive).
        self.go(
            router.mv,
            (ceil(ne, 4), t as u32, 1),
            128,
            &mut args![router.a, x, ar.logits, h as u32, ne as u32, h as u32, ne as u32, 0u32],
        );
        self.go(
            self.f("route_k"),
            (ceil(t, 4), 1, 1),
            128,
            &mut args![ar.logits, bias, ar.ids, ar.wts, ne as u32, k as u32, self.route_scale, t as u32],
        );
        if t <= MV_T {
            self.go(
                gu.gu,
                (ceil(mi, 8), (t * k) as u32, 1),
                256,
                &mut args![
                    gu.a,
                    ar.ids,
                    x,
                    ar.mid,
                    h as u32,
                    mi as u32,
                    k as u32,
                    h as u32,
                    inter as u32,
                    self.beta,
                    self.lbeta
                ],
            );
            self.go(
                dn.dn,
                (ceil(h, 2), t as u32, 1),
                256,
                &mut args![dn.a, ar.ids, ar.wts, ar.mid, y, h as u32, mi as u32, k as u32, h as u32],
            );
            return;
        }
        let n = (t * k) as u32;
        self.go(
            self.f("moe_plan"),
            (1, 1, 1),
            1024,
            &mut args![ar.ids, n, ne as u32, ar.perm, ar.inv, ar.tiles, ar.ntiles, ar.cnt],
        );
        self.go(
            self.f("moe_rows"),
            (n.div_ceil(256), 1, 1),
            256,
            &mut args![ar.perm, ar.wts, ar.tok, ar.wrow, n, k as u32],
        );
        let max_tiles = (t * k).div_ceil(64) + ne;
        let sp = self.split(x, h, t, h);
        let p = GemmP {
            x,
            xh: sp.0,
            xl: sp.1,
            y: ar.mid,
            gidx: ar.tok,
            tiles: ar.tiles,
            ntiles: ar.ntiles,
            n_in: h as u32,
            n_out: 2 * mi as u32,
            rows: n,
            xs: h as u32,
            ys: mi as u32,
            mode: 2,
            mi: mi as u32,
            inter: inter as u32,
            beta: self.beta,
            lbeta: self.lbeta,
            ..Default::default()
        };
        self.go(gu.gemm_f(), (ceil(mi, 32), max_tiles as u32, 1), 128, &mut args![gu.a, p]);
        let sp = self.split(ar.mid, mi, t * k, mi);
        let p = GemmP {
            x: ar.mid,
            xh: sp.0,
            xl: sp.1,
            y: ar.dmoe,
            tiles: ar.tiles,
            ntiles: ar.ntiles,
            wrow: ar.wrow,
            n_in: mi as u32,
            n_out: h as u32,
            rows: n,
            xs: mi as u32,
            ys: h as u32,
            mode: 3,
            ..Default::default()
        };
        self.go(dn.gemm_f(), (ceil(h, 64), max_tiles as u32, 1), 128, &mut args![dn.a, p]);
        let tot = (t * h) as u32;
        self.go(
            self.f("moe_combine"),
            (tot.div_ceil(256), 1, 1),
            256,
            &mut args![ar.dmoe, ar.inv, y, h as u32, k as u32, t as u32],
        );
    }

    /// One pass of `t` tokens (their embeddings in `ar.x`, the position in `ar.info`) into
    /// `ar.out` (final-normed hidden states), as `Inkling2Model::forward_capture`.
    /// `capture`: the steps whose entering stream goes to `ar.cap` (`[capture.len()][t][hidden]`;
    /// `steps().len()` the last layer's output).
    fn pass(&self, m: &Inkling2Model, ar: &Arena, s: &DevSess, t: usize, capture: &[usize]) {
        let c = &m.cfg;
        let h = self.h;
        let n = t * h;
        let conv_norm = 1.0 / ((self.conv_k + 1) as f32).sqrt();
        let mut free: Vec<u64> = ar.states.iter().rev().copied().collect();
        let mut states: Vec<u64> = Vec::new();
        let mut pre_n = 0usize;
        let mut bank: Option<Vec<u64>> = None;
        let steps = c.steps();
        let grab = |si: usize| {
            for (k, _) in capture.iter().enumerate().filter(|e| *e.1 == si) {
                self.copy(ar.cap + (k * n * 4) as u64, ar.x, n);
            }
        };
        for (si, step) in steps.iter().enumerate() {
            let l = &self.layers[step.layer];
            let slot = &s.slots[step.slot];
            if let Some(lc) = c.loop_cfg
                && step.layer == lc.start
            {
                if step.pass == 0 {
                    self.copy(ar.pre_x, ar.x, n);
                    pre_n = states.len();
                } else {
                    let prev = states.split_off(pre_n + 1);
                    free.extend(states.drain(pre_n..));
                    if let Some(old) = bank.replace(prev) {
                        free.extend(old);
                    }
                    self.copy(ar.bank_x, ar.x, n);
                    self.copy(ar.x, ar.pre_x, n);
                }
            }
            // (The stream entering the step: after the loop's restart, as the CPU captures it.)
            grab(si);
            let msgs = match (&bank, &l.looped) {
                (Some(bs), Some(lp)) if step.pass > 0 => {
                    self.mix(bs, ar.bank_x, lp.query, 4, &ar.msgs, &[], t);
                    Some(lp)
                }
                _ => None,
            };
            // Without messages the mixes go straight into the input norms.
            let fused = msgs.is_none() && l.attn_query.is_some();
            let input: Vec<u64> = match l.attn_query {
                Some((q, nq)) if fused => {
                    self.mix(&states, ar.x, q, nq, &ar.xn, &l.ln, t);
                    vec![]
                }
                Some((q, nq)) => {
                    self.mix(&states, ar.x, q, nq, &ar.mixed, &[], t);
                    (0..3).map(|i| ar.mixed[i.min(nq - 1)]).collect()
                }
                None => vec![ar.x; 3],
            };
            let xs: [u64; 3] = match (l.ln.len(), msgs) {
                (1, None) if fused => [ar.xn[0]; 3],
                (_, None) if fused => ar.xn,
                (1, None) => {
                    self.rms(input[0], ar.xn[0], l.ln[0], h, self.eps, t);
                    [ar.xn[0]; 3]
                }
                (1, Some(lp)) => {
                    for i in 0..3 {
                        self.gate(ar, &lp.gates[i], input[0], ar.msgs[i], ar.gated[i], t);
                        self.rms(ar.gated[i], ar.xn[i], l.ln[0], h, self.eps, t);
                    }
                    ar.xn
                }
                (_, mg) => {
                    for i in 0..3 {
                        let src = match mg {
                            Some(lp) => {
                                self.gate(ar, &lp.gates[i], input[i], ar.msgs[i], ar.gated[i], t);
                                ar.gated[i]
                            }
                            None => input[i],
                        };
                        self.rms(src, ar.xn[i], l.ln[i], h, self.eps, t);
                    }
                    ar.xn
                }
            };
            self.attention(ar, l, xs[0], xs[1], xs[2], slot, s.fmt, t);
            self.sconv(ar.a, slot.hist[2], l.attn_taps, 0, h, 1, t, conv_norm, ar.info);
            if l.boundary {
                let st = free.pop().expect("inkling cuda: block states");
                self.rms(ar.x, st, 0, h, MIX_EPS, t);
                states.push(st);
                self.copy(ar.x, ar.a, n);
            } else {
                self.go(
                    self.f("add_k"),
                    ((n as u32).div_ceil(256), 1, 1),
                    256,
                    &mut args![ar.x, ar.a, n as u32],
                );
            }
            let hn = match msgs {
                Some(lp) => {
                    self.mix(&states, ar.x, l.mlp_query, 1, &[ar.hn], &[], t);
                    self.gate(ar, &lp.gates[3], ar.hn, ar.msgs[3], ar.hn2, t);
                    self.rms(ar.hn2, ar.hn2, l.ln_post, h, self.eps, t);
                    ar.hn2
                }
                None => {
                    self.mix(&states, ar.x, l.mlp_query, 1, &[ar.hn], &[l.ln_post], t);
                    ar.hn
                }
            };
            self.moe(ar, l, hn, ar.o, t);
            self.sconv(ar.o, slot.hist[3], l.mlp_taps, 0, h, 1, t, conv_norm, ar.info);
            self.go(
                self.f("add_k"),
                ((n as u32).div_ceil(256), 1, 1),
                256,
                &mut args![ar.x, ar.o, n as u32],
            );
        }
        grab(steps.len());
        self.mix(&states, ar.x, self.final_query, 1, &[ar.out], &[self.final_norm], t);
    }

    /// The cache capacity a slot needs for `pos` positions (sliding layers: a fixed ring).
    fn cap_for(&self, l: &DLayer, pos: usize) -> usize {
        if l.full {
            pos.max(4096).next_multiple_of(1024)
        } else {
            (self.window + CHUNK).next_multiple_of(256)
        }
    }

    /// A zeroed device buffer, if the device keeps [`SESS_RESERVE`] free after it.
    fn try_alloc(&self, n: usize) -> Option<u64> {
        // Tests: `OXINFER_INKLING_SESS_LIMIT=BYTES` refuses larger buffers (the host fallback).
        static LIMIT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        if LIMIT
            .get_or_init(|| std::env::var("OXINFER_INKLING_SESS_LIMIT").ok().and_then(|v| v.parse().ok()))
            .is_some_and(|l| n > l)
        {
            return None;
        }
        let (mut free, mut tot) = (0usize, 0usize);
        unsafe { self.x.cuMemGetInfo_v2(&mut free, &mut tot) };
        if free < n + SESS_RESERVE {
            return None;
        }
        let p = DevSess::alloc(&self.x, n)?;
        unsafe { self.x.cuMemsetD8Async(p, 0, n.max(256), self.stream) };
        Some(p)
    }

    /// A fresh device session for `m` able to hold `pos` positions.
    fn new_sess(&self, m: &Inkling2Model, pos: usize) -> Option<DevSess> {
        let c = &m.cfg;
        let fmt = match m.kv {
            KvCache::F32 => 0,
            KvCache::F16 => 1,
            KvCache::Q8 => 2,
        };
        let el = [4usize, 2, 1][fmt as usize];
        let back = c.conv_kernel_size - 1;
        let mut slot_layer = vec![0usize; c.n_slots()];
        for s in c.steps() {
            slot_layer[s.slot] = s.layer;
        }
        let mut sess = DevSess {
            x: self.x.clone(),
            ctx: self.ctx,
            slots: Vec::new(),
            fmt,
            graphs: Vec::new(),
            engine: self as *const Engine as usize,
        };
        for &li in &slot_layer {
            let l = &self.layers[li];
            let kw = l.nkv * 128;
            let cap = self.cap_for(l, pos);
            // The convolutions' rings by position, as the host's (`ConvRing`).
            let rows = back + ROLLBACK;
            let hist_len = [rows * kw, rows * kw, rows * self.h, rows * self.h];
            // Every buffer of the slot, or none (what was allocated goes with `sess`).
            let mut slot = Slot {
                k: 0,
                v: 0,
                ks: 0,
                vs: 0,
                cap,
                kw,
                hist: [0; 4],
                hist_len,
            };
            let q8 = fmt == 2;
            let ok = (|| -> Option<()> {
                slot.k = self.try_alloc(cap * kw * el)?;
                slot.v = self.try_alloc(cap * kw * el)?;
                if q8 {
                    slot.ks = self.try_alloc(cap * kw / 32 * 4)?;
                    slot.vs = self.try_alloc(cap * kw / 32 * 4)?;
                }
                for i in 0..4 {
                    slot.hist[i] = self.try_alloc(hist_len[i] * 4)?;
                }
                Some(())
            })();
            sess.slots.push(slot);
            ok?;
        }
        unsafe { self.x.cuStreamSynchronize(self.stream) };
        Some(sess)
    }

    /// Grows the full layers' caches of `s` to hold `need` positions (copying what they hold), by
    /// a quarter at least; false (nothing changed) when the device cannot hold them.
    fn grow(&self, m: &Inkling2Model, s: &mut DevSess, need: usize) -> bool {
        let el = [4usize, 2, 1][s.fmt as usize];
        let c = &m.cfg;
        let mut slot_layer = vec![0usize; c.n_slots()];
        for st in c.steps() {
            slot_layer[st.slot] = st.layer;
        }
        // The new buffers first: (slot, cap, [k, v, ks, vs]).
        let mut new: Vec<(usize, usize, [u64; 4])> = Vec::new();
        let free_all = |new: &[(usize, usize, [u64; 4])]| {
            for (_, _, b) in new {
                for &p in b {
                    if p != 0 {
                        unsafe { self.x.cuMemFree_v2(p) };
                    }
                }
            }
        };
        for (i, (sl, &li)) in s.slots.iter().zip(&slot_layer).enumerate() {
            let l = &self.layers[li];
            if !l.full || sl.cap >= need {
                continue;
            }
            let cap = need.max(sl.cap + sl.cap / 4).next_multiple_of(1024);
            let n = cap * sl.kw;
            let mut b = [0u64; 4];
            let sizes = [
                n * el,
                n * el,
                if sl.ks != 0 { n / 32 * 4 } else { 0 },
                if sl.vs != 0 { n / 32 * 4 } else { 0 },
            ];
            for (p, &bytes) in b.iter_mut().zip(&sizes) {
                if bytes > 0 {
                    match self.try_alloc(bytes) {
                        Some(d) => *p = d,
                        None => {
                            new.push((i, cap, b));
                            free_all(&new);
                            debug(&format!("no device memory for {need} positions of context"));
                            return false;
                        }
                    }
                }
            }
            new.push((i, cap, b));
        }
        if new.is_empty() {
            return true;
        }
        for (i, cap, b) in new {
            let sl = &mut s.slots[i];
            let n_old = sl.cap * sl.kw;
            for (old, (d, bytes)) in [&mut sl.k, &mut sl.v, &mut sl.ks, &mut sl.vs].into_iter().zip(b.into_iter().zip([
                n_old * el,
                n_old * el,
                n_old / 32 * 4,
                n_old / 32 * 4,
            ])) {
                if *old != 0 {
                    unsafe { self.x.cuMemcpyDtoDAsync_v2(d, *old, bytes, self.stream) };
                    let o = std::mem::replace(old, d);
                    unsafe {
                        self.x.cuStreamSynchronize(self.stream);
                        self.x.cuMemFree_v2(o);
                    }
                }
            }
            sl.cap = cap;
        }
        s.drop_graph();
        true
    }

    /// Copies the host caches and histories of `sess` into a new device session with room for
    /// `room` more positions (None when the device cannot hold it).
    fn upload_sess(&self, m: &Inkling2Model, sess: &Inkling2Session, room: usize) -> Option<DevSess> {
        let pos = sess.pos;
        let ds = self.new_sess(m, pos + room)?;
        let el = [4usize, 2, 1][ds.fmt as usize];
        for (hs, d) in sess.layers.iter().zip(&ds.slots) {
            let kw = d.kw;
            for (i, hp) in d.hist.iter().enumerate() {
                // The host's ring by position, in the device's layout (a fresh session's is the
                // device's zeros already).
                if pos == 0 {
                    break;
                }
                let v = &hs.hist[i].rows;
                debug_assert_eq!(v.len(), d.hist_len[i]);
                unsafe { self.cu.cuMemcpyHtoD_v2(*hp, v.as_ptr() as *const c_void, v.len() * 4) };
            }
            // Rows first..pos to rows p % cap: at most two runs (the ring wraps once).
            for (rows, dev, devs) in [(&hs.k, d.k, d.ks), (&hs.v, d.v, d.vs)] {
                // A ring holds the last `cap` positions (all a sliding layer reads).
                let from = hs.first.max(pos.saturating_sub(d.cap));
                for (p0, p1) in runs(from, pos, d.cap) {
                    let (r, at, n) = (p0 - hs.first, p0 % d.cap, p1 - p0);
                    let (src, bytes): (*const c_void, usize) = match rows.mode {
                        KvCache::F32 => (rows.f[r * kw..].as_ptr() as *const c_void, n * kw * 4),
                        KvCache::F16 => (rows.h[r * kw..].as_ptr() as *const c_void, n * kw * 2),
                        KvCache::Q8 => (rows.q[r * kw..].as_ptr() as *const c_void, n * kw),
                    };
                    unsafe { self.cu.cuMemcpyHtoD_v2(dev + (at * kw * el) as u64, src, bytes) };
                    if rows.mode == KvCache::Q8 {
                        let g = kw / 32;
                        unsafe {
                            self.cu
                                .cuMemcpyHtoD_v2(devs + (at * g * 4) as u64, rows.s[r * g..].as_ptr() as *const c_void, n * g * 4)
                        };
                    }
                }
            }
        }
        Some(ds)
    }

    /// Writes a device session back into the host caches of `sess` (the device copy is dropped).
    fn download_sess(&self, m: &Inkling2Model, ds: &DevSess, sess: &mut Inkling2Session) {
        let pos = sess.pos;
        let c = &m.cfg;
        let mut slot_layer = vec![0usize; c.n_slots()];
        for st in c.steps() {
            slot_layer[st.slot] = st.layer;
        }
        let el = [4usize, 2, 1][ds.fmt as usize];
        let mode = [KvCache::F32, KvCache::F16, KvCache::Q8][ds.fmt as usize];
        for ((hs, d), &li) in sess.layers.iter_mut().zip(&ds.slots).zip(&slot_layer) {
            let kw = d.kw;
            for (i, hp) in d.hist.iter().enumerate() {
                let v = &mut hs.hist[i].rows;
                v.resize(d.hist_len[i], 0.0);
                unsafe { self.cu.cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, *hp, v.len() * 4) };
            }
            let l = &self.layers[li];
            // (A sliding layer keeps `ROLLBACK` positions more than its window, as on the host.)
            let first = if l.full { 0 } else { pos.saturating_sub(self.window + ROLLBACK) };
            let n = pos - first;
            let rows_of = |dev: u64, devs: u64| -> KvRows {
                let mut kr = KvRows::new(mode);
                let mut raw = vec![0u8; n * kw * el];
                let mut sc = vec![0f32; if mode == KvCache::Q8 { n * kw / 32 } else { 0 }];
                for (p0, p1) in runs(first, pos, d.cap) {
                    let (r, at, c) = (p0 - first, p0 % d.cap, p1 - p0);
                    unsafe {
                        self.cu.cuMemcpyDtoH_v2(
                            raw[r * kw * el..].as_mut_ptr() as *mut c_void,
                            dev + (at * kw * el) as u64,
                            c * kw * el,
                        )
                    };
                    if mode == KvCache::Q8 {
                        let g = kw / 32;
                        unsafe {
                            self.cu
                                .cuMemcpyDtoH_v2(sc[r * g..].as_mut_ptr() as *mut c_void, devs + (at * g * 4) as u64, c * g * 4)
                        };
                    }
                }
                match mode {
                    KvCache::F32 => kr.f = raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
                    KvCache::F16 => kr.h = raw.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect(),
                    KvCache::Q8 => {
                        kr.q = raw.iter().map(|&b| b as i8).collect();
                        kr.s = sc;
                    }
                }
                kr
            };
            let k = rows_of(d.k, d.ks);
            let v = rows_of(d.v, d.vs);
            *hs = LayerState {
                k,
                v,
                first,
                hist: std::mem::take(&mut hs.hist),
            };
        }
    }

    /// Runs `x` (`[t, hidden]`) through the device session of `sess`.
    /// None, with `sess` as it was, when the device cannot hold its caches for this call.
    /// `capture`: the steps whose entering streams go to `feats` (`[t][capture.len()][hidden]`;
    /// the final-normed one, `steps().len() + 1`, is the caller's).
    fn run_pass(
        &self,
        m: &Inkling2Model,
        sess: &mut Inkling2Session,
        x: &[f32],
        t: usize,
        capture: &[usize],
        mut feats: Option<&mut Vec<f32>>,
    ) -> Option<Vec<f32>> {
        let nf = capture.len();
        if let Some(f) = feats.as_deref_mut() {
            f.clear();
            f.resize(t * nf * self.h, 0.0);
        }
        let h = self.h;
        assert!(unsafe { self.x.cuCtxSetCurrent(self.ctx) } == 0, "inkling cuda: context");
        if sess.dev.is_none() {
            let ds = self.upload_sess(m, sess, t.max(CHUNK))?;
            sess.dev = Some(DevBox(Box::new(ds)));
        }
        // Room for the whole call before any of it runs.
        {
            let ds: &mut DevSess = sess.dev.as_mut().unwrap().0.as_any().downcast_mut::<DevSess>().unwrap();
            if !self.grow(m, ds, sess.pos + t) {
                return None;
            }
        }
        let ar = self.arena.lock().unwrap();
        let mut out = vec![0f32; t * h];
        let mut done = 0;
        while done < t {
            let n = (t - done).min(CHUNK);
            let pos = sess.pos;
            let ds: &mut DevSess = sess.dev.as_mut().unwrap().0.as_any().downcast_mut::<DevSess>().unwrap();
            if ds.engine != self as *const Engine as usize {
                ds.drop_graph();
                ds.engine = self as *const Engine as usize;
            }
            let info = [pos as i32, 0, 0, 0];
            unsafe {
                self.x
                    .cuMemcpyHtoDAsync_v2(ar.info, info.as_ptr() as *const c_void, 16, self.stream);
                self.x
                    .cuMemcpyHtoDAsync_v2(ar.x, x[done * h..].as_ptr() as *const c_void, n * h * 4, self.stream);
            }
            if n <= MV_T && graphs() && self.prof.is_none() {
                // Passes of a few tokens as CUDA graphs, one per token count and captured steps.
                let caps: Vec<usize> = ds.slots.iter().map(|s| s.cap).collect();
                if ds.graphs.first().is_some_and(|g| g.2 != caps) {
                    ds.drop_graph();
                }
                if !ds.graphs.iter().any(|g| g.0 == n && g.1 == capture)
                    && let Some((e, g)) = self.capture(m, &ar, ds, n, capture)
                {
                    ds.graphs.push((n, capture.to_vec(), caps, e, g));
                }
                match ds.graphs.iter().find(|g| g.0 == n && g.1 == capture) {
                    Some(g) => assert!(unsafe { self.x.cuGraphLaunch(g.3, self.stream) } == 0, "inkling cuda: graph launch"),
                    None => self.pass(m, &ar, ds, n, capture),
                }
            } else {
                self.pass(m, &ar, ds, n, capture);
            }
            let mut cap = vec![0f32; if feats.is_some() { nf * n * h } else { 0 }];
            unsafe {
                self.x
                    .cuMemcpyDtoHAsync_v2(out[done * h..].as_mut_ptr() as *mut c_void, ar.out, n * h * 4, self.stream);
                if !cap.is_empty() {
                    self.x
                        .cuMemcpyDtoHAsync_v2(cap.as_mut_ptr() as *mut c_void, ar.cap, cap.len() * 4, self.stream);
                }
                let rc = self.x.cuStreamSynchronize(self.stream);
                assert!(rc == 0, "inkling cuda: a pass failed ({rc})");
            }
            if let Some(f) = feats.as_deref_mut() {
                for k in 0..nf {
                    for ti in 0..n {
                        let dst = ((done + ti) * nf + k) * h;
                        f[dst..dst + h].copy_from_slice(&cap[(k * n + ti) * h..(k * n + ti + 1) * h]);
                    }
                }
            }
            self.prof_collect();
            sess.pos += n;
            done += n;
        }
        Some(out)
    }

    /// Captures a pass of `t` tokens as a graph.
    fn capture(&self, m: &Inkling2Model, ar: &Arena, ds: &DevSess, t: usize, capture: &[usize]) -> Option<(F, F)> {
        unsafe {
            if self.x.cuStreamBeginCapture_v2(self.stream, 1) != 0 {
                return None;
            }
        }
        self.pass(m, ar, ds, t, capture);
        let mut g = std::ptr::null_mut();
        if unsafe { self.x.cuStreamEndCapture(self.stream, &mut g) } != 0 || g.is_null() {
            debug("graph capture failed");
            return None;
        }
        let mut exec = std::ptr::null_mut();
        if unsafe { self.x.cuGraphInstantiateWithFlags(&mut exec, g, 0) } != 0 {
            unsafe { self.x.cuGraphDestroy(g) };
            debug("graph instantiation failed");
            return None;
        }
        Some((exec, g))
    }

    /// The unembedding of `hidden` (`[t, hidden]`), soft-capped when `cap`.
    fn logits(&self, hidden: &[f32], t: usize, cap: bool) -> Vec<f32> {
        let (h, v, u) = (self.h, self.vocab, self.unpadded);
        assert!(unsafe { self.x.cuCtxSetCurrent(self.ctx) } == 0, "inkling cuda: context");
        let ar = self.arena.lock().unwrap();
        let mut out = vec![0f32; t * u];
        let mut done = 0;
        while done < t {
            let n = (t - done).min(16);
            unsafe {
                self.x
                    .cuMemcpyHtoDAsync_v2(ar.x, hidden[done * h..].as_ptr() as *const c_void, n * h * 4, self.stream);
            }
            self.lin(&self.head, ar.x, ar.head, n, h, v, false);
            let tot = (n * u) as u32;
            let cp = if cap { self.softcap } else { 0.0 };
            self.go(
                self.f("softcap_k"),
                (tot.div_ceil(256), 1, 1),
                256,
                &mut args![ar.head, ar.head_out, v as u32, u as u32, n as u32, cp],
            );
            unsafe {
                self.x
                    .cuMemcpyDtoHAsync_v2(out[done * u..].as_mut_ptr() as *mut c_void, ar.head_out, n * u * 4, self.stream);
                assert!(self.x.cuStreamSynchronize(self.stream) == 0, "inkling cuda: the unembedding failed");
            }
            self.prof_collect();
            done += n;
        }
        out
    }
}

/// Positions `a..b` of a ring of `cap` rows as runs of consecutive rows (at most two when
/// `b - a <= cap`).
fn runs(a: usize, b: usize, cap: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut p = a;
    while p < b {
        let end = b.min((p / cap + 1) * cap);
        v.push((p, end));
        p = end;
    }
    v
}

/// Decode steps as CUDA graphs (`OXINFER_INKLING_GRAPH=0`: launched op by op).
fn graphs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OXINFER_INKLING_GRAPH").map_or(true, |v| v != "0"))
}

impl DevEngine for Engine {
    fn rollback(&self) -> usize {
        ROLLBACK
    }
    /// Passes of up to `MV_T` tokens run on the decode kernels, each token as alone.
    fn exact_tokens(&self) -> usize {
        MV_T
    }
    fn run(&self, m: &Inkling2Model, sess: &mut Inkling2Session, x: &[f32], t: usize, want: super::DevWant) -> bool {
        let (capture, feats) = match want.capture {
            Some((c, f)) => (c, Some(f)),
            None => (&[][..], None),
        };
        if capture.len() > MAX_CAPTURE {
            return false;
        }
        let Some(hs) = self.run_pass(m, sess, x, t, capture, feats) else {
            // The device cannot hold this session: it stays on the host from now on.
            sess.dev_off = true;
            return false;
        };
        if let Some((l, rows)) = want.logits {
            let r = rows.min(t);
            *l = self.logits(&hs[(t - r) * self.h..], r, true);
        }
        if let Some(hv) = want.hidden {
            *hv = hs;
        }
        true
    }
    fn forward(&self, m: &Inkling2Model, sess: &mut Inkling2Session, x: &[f32], t: usize) -> Vec<f32> {
        self.run_pass(m, sess, x, t, &[], None)
            .expect("inkling cuda: no device memory for the session")
    }
    fn head(&self, _m: &Inkling2Model, hidden: &[f32], t: usize, cap: bool) -> Option<Vec<f32>> {
        Some(self.logits(hidden, t, cap))
    }
    fn to_host(&self, m: &Inkling2Model, sess: &mut Inkling2Session) {
        let Some(mut d) = sess.dev.take() else { return };
        assert!(unsafe { self.x.cuCtxSetCurrent(self.ctx) } == 0, "inkling cuda: context");
        let ds = d.0.as_any().downcast_mut::<DevSess>().unwrap();
        self.download_sess(m, ds, sess);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::{GgufMatrix, random_blocks};
    use crate::quant::PageVec;

    fn backend() -> Option<&'static CudaBackend> {
        static B: std::sync::OnceLock<Option<CudaBackend>> = std::sync::OnceLock::new();
        B.get_or_init(|| CudaBackend::new(0).ok()).as_ref()
    }

    fn rnd(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn rnds(n: usize, seed: &mut u64, scale: f32) -> Vec<f32> {
        (0..n).map(|_| rnd(seed) * scale).collect()
    }

    fn zeros(e: &Engine, n: usize) -> u64 {
        let p = e.alloc(n * 4).unwrap();
        unsafe { e.x.cuMemsetD8Async(p, 0, (n * 4).max(256), e.stream) };
        p
    }
    fn down(e: &Engine, p: u64, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        unsafe {
            assert_eq!(e.x.cuStreamSynchronize(e.stream), 0, "a kernel failed");
            e.cu.cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, p, n * 4);
        }
        v
    }
    fn down_bytes(e: &Engine, p: u64, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        unsafe {
            assert_eq!(e.x.cuStreamSynchronize(e.stream), 0, "a kernel failed");
            e.cu.cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, p, n);
        }
        v
    }
    fn set_info(e: &Engine, p: u64, pos: usize) {
        let info = [pos as i32, 0, 0, 0];
        unsafe {
            e.x.cuStreamSynchronize(e.stream);
            e.cu.cuMemcpyHtoD_v2(p, info.as_ptr() as *const c_void, 16);
        }
    }

    fn gguf(kind: GgmlType, n_out: usize, n_in: usize, seed: &mut u64) -> Lin<'static> {
        let raw = random_blocks(kind, n_out * n_in / kind.block_size(), seed);
        let mut pv = PageVec::<u8>::zeroed(raw.len());
        pv.copy_from_slice(&raw);
        Lin::gguf(GgufMatrix::new(kind, pv.into(), n_out, n_in).unwrap())
    }

    /// The matrix as float32 rows, as the CPU reads it.
    fn dense(l: &Lin, n_out: usize, n_in: usize) -> Vec<f32> {
        let mut d = vec![0f32; n_out * n_in];
        match &l.w {
            Weight::Gguf(m) => m.dequantize_rows(0, n_out, &mut d),
            Weight::Q8(m) => crate::quant::dequantize_rows(m, 0, n_out, &mut d),
            Weight::Q4(m) => crate::quant::dequantize_rows_q4(m, 0, n_out, &mut d),
            Weight::Bf16(b, _) => {
                for (o, &v) in d.iter_mut().zip(b.iter()) {
                    *o = half::bf16::from_bits(v).to_f32();
                }
            }
        }
        d
    }

    /// Every weight kind through the matvec (1 and 3 rows) and the GEMM (70 rows, partial tiles),
    /// against the product of the CPU's dequantized rows in f64.
    #[test]
    fn matvecs_and_gemms_match_definition() {
        let Some(be) = backend() else { return };
        let e = Engine::bare(be, &KINDS).unwrap();
        let (n_out, n_in) = (70usize, 512usize);
        let mut seed = 7u64;
        let mut mats: Vec<(String, Lin<'static>)> = Vec::new();
        for &k in &KINDS {
            let lin = match k {
                K_OQ8 => Lin::q8(crate::quant::quantize_f32(&rnds(n_out * n_in, &mut seed, 0.5), n_out, n_in)),
                K_OQ4 => Lin::q4(crate::quant::quantize_q4_f32(&rnds(n_out * n_in, &mut seed, 0.5), n_out, n_in)),
                K_F8 => {
                    let mut q = PageVec::<u8>::zeroed(n_out * n_in);
                    for b in q.iter_mut() {
                        let v = (rnd(&mut seed) * 128.0 + 128.0) as u8;
                        *b = if v & 0x7f == 0x7f { 0 } else { v };
                    }
                    let mut s = PageVec::<f32>::zeroed(n_out * n_in / 32);
                    for v in s.iter_mut() {
                        *v = (2f32).powi(-((rnd(&mut seed).abs() * 8.0) as i32) - 4);
                    }
                    Lin::gguf(GgufMatrix::fp8(q.into(), s.into(), n_out, n_in, (1, 32)).unwrap())
                }
                k => gguf(GgmlType::from_id(k).unwrap(), n_out, n_in, &mut seed),
            };
            mats.push((format!("kind {k}"), lin));
        }
        let mut scaled = gguf(GgmlType::Q4_K, n_out, n_in, &mut seed);
        if let Weight::Gguf(m) = &mut scaled.w {
            m.set_factor(0.37).unwrap();
        }
        mats.push(("Q4_K with a factor".into(), scaled));
        let kf = e.kfuncs.clone();
        for (name, lin) in &mats {
            let dw = e.up_lins(&[lin], n_out, n_in, &kf).unwrap();
            let w = dense(lin, n_out, n_in);
            for t in [1usize, 3, 70] {
                let x = rnds(t * n_in, &mut seed, 1.0);
                let y0 = rnds(t * n_out, &mut seed, 1.0);
                let (px, py) = (e.upf(&x).unwrap(), e.upf(&y0).unwrap());
                let add = t == 3;
                e.lin(&dw, px, py, t, n_in, n_out, add);
                let y = down(&e, py, t * n_out);
                // Of the summed magnitudes: the matvec sums in f32, the GEMM in split halves
                // (measured: ~3e-8 and ~7e-8).
                let tol = if t <= MV_T { 2e-7 } else { 4e-7 };
                for tok in 0..t {
                    for r in 0..n_out {
                        let (mut s, mut mag) = (0f64, 0f64);
                        for k in 0..n_in {
                            let p = w[r * n_in + k] as f64 * x[tok * n_in + k] as f64;
                            s += p;
                            mag += p.abs();
                        }
                        if add {
                            s += y0[tok * n_out + r] as f64;
                        }
                        let got = y[tok * n_out + r] as f64;
                        assert!(
                            (got - s).abs() <= tol * mag + 1e-9,
                            "{name}, {t} rows, row {r} token {tok}: {got} against {s} (magnitude {mag})"
                        );
                    }
                }
            }
        }
    }

    fn situ(g: f64, u: f64, beta: f64, lbeta: f64) -> f64 {
        beta * (g / beta).tanh() / (1.0 + (-g).exp()) * lbeta * (u / lbeta).tanh()
    }

    /// The router, the top k and the experts, on the decode path (2 tokens) and the prefill path
    /// (segments, grouped GEMMs; 50 tokens), interleaved and fused gate / up rows, per-expert factors.
    #[test]
    fn moe_matches_definition() {
        let Some(be) = backend() else { return };
        let mut e = Engine::bare(be, &[0, 8, 12]).unwrap();
        let (h, mi, ne, k) = (256usize, 256usize, 16usize, 4usize);
        (e.h, e.mi, e.ne, e.k_top, e.route_scale, e.beta, e.lbeta) = (h, mi, ne, k, 2.5, 0.8, 1.5);
        let mut seed = 11u64;
        let tmax = 64;
        let mut ar = Engine::empty_arena();
        let tk = tmax * k;
        ar.logits = zeros(&e, tmax * ne);
        ar.ids = zeros(&e, tk);
        ar.wts = zeros(&e, tk);
        ar.mid = zeros(&e, tk * mi);
        ar.dmoe = zeros(&e, tk * h);
        ar.perm = zeros(&e, tk);
        ar.inv = zeros(&e, tk);
        ar.tok = zeros(&e, tk);
        ar.wrow = zeros(&e, tk);
        ar.tiles = zeros(&e, 3 * (tk / 64 + ne + 1));
        ar.ntiles = zeros(&e, 1);
        ar.cnt = zeros(&e, ne);
        let kf = e.kfuncs.clone();
        let router = rnds(ne * h, &mut seed, 0.2);
        let bias = rnds(ne, &mut seed, 0.05);
        let r_dw = e.up_f32(&router, ne, h, &kf).unwrap();
        let b_dev = e.upf(&bias).unwrap();
        let gus: Vec<Lin> = (0..ne).map(|_| gguf(GgmlType::Q8_0, 2 * mi, h, &mut seed)).collect();
        let mut dns: Vec<Lin> = (0..ne).map(|_| gguf(GgmlType::Q4_K, h, mi, &mut seed)).collect();
        for (i, l) in dns.iter_mut().enumerate() {
            if let Weight::Gguf(m) = &mut l.w {
                m.set_factor(0.5 + 0.1 * i as f32).unwrap();
            }
        }
        let gu_dw = e.up_lins(&gus.iter().collect::<Vec<_>>(), 2 * mi, h, &kf).unwrap();
        let dn_dw = e.up_lins(&dns.iter().collect::<Vec<_>>(), h, mi, &kf).unwrap();
        let gw: Vec<Vec<f32>> = gus.iter().map(|l| dense(l, 2 * mi, h)).collect();
        let dw: Vec<Vec<f32>> = dns.iter().map(|l| dense(l, h, mi)).collect();
        for inter in [false, true] {
            for t in [2usize, 50] {
                let x = rnds(t * h, &mut seed, 1.0);
                let px = e.upf(&x).unwrap();
                let py = zeros(&e, t * h);
                e.experts(&ar, &r_dw, b_dev, &gu_dw, &dn_dw, inter, px, py, t);
                let y = down(&e, py, t * h);
                let ids = down_bytes(&e, ar.ids, t * k * 4);
                let tol = 5e-6;
                for tok in 0..t {
                    let xr = &x[tok * h..(tok + 1) * h];
                    let lg: Vec<f32> = (0..ne)
                        .map(|j| router[j * h..(j + 1) * h].iter().zip(xr).map(|(a, b)| a * b).sum())
                        .collect();
                    let routes = crate::moe::route_sigmoid(&lg, &bias, k, 2.5);
                    for (sl, &(ex, _)) in routes.iter().enumerate() {
                        let at = 4 * (tok * k + sl);
                        assert_eq!(
                            i32::from_le_bytes(ids[at..at + 4].try_into().unwrap()) as usize,
                            ex,
                            "token {tok} slot {sl}"
                        );
                    }
                    let mut yr = vec![0f64; h];
                    let mut mag = vec![0f64; h];
                    for &(ex, wt) in &routes {
                        let act: Vec<f64> = (0..mi)
                            .map(|j| {
                                let (rg, ru) = if inter { (2 * j, 2 * j + 1) } else { (j, mi + j) };
                                let dot = |r: usize| -> f64 { (0..h).map(|c| gw[ex][r * h + c] as f64 * xr[c] as f64).sum() };
                                situ(dot(rg), dot(ru), 0.8, 1.5)
                            })
                            .collect();
                        for r in 0..h {
                            for j in 0..mi {
                                let p = wt as f64 * dw[ex][r * mi + j] as f64 * act[j];
                                yr[r] += p;
                                mag[r] += p.abs();
                            }
                        }
                    }
                    for r in 0..h {
                        let got = y[tok * h + r] as f64;
                        assert!(
                            (got - yr[r]).abs() <= tol * mag[r] + 1e-5,
                            "interleaved {inter}, {t} tokens, token {tok} row {r}: {got} against {}",
                            yr[r]
                        );
                    }
                }
            }
        }
    }

    /// Cache appends in the three formats, then the attention (relative bias, sink, window, a
    /// ring that wraps, splits of the keys) and the gated merge, against the definition over the
    /// stored keys and values.
    #[test]
    fn attention_matches_definition() {
        let Some(be) = backend() else { return };
        let e = Engine::bare(be, &[0]).unwrap();
        let (nh, nkv, d, dr, extent) = (24usize, 8usize, 128usize, 16usize, 1024usize);
        let (kw, w) = (nkv * d, nh * d);
        let mut seed = 5u64;
        let info = zeros(&e, 4);
        let rel = rnds(dr * extent, &mut seed, 0.5);
        let p_rel = e.upf(&rel).unwrap();
        for fmt in 0..3u32 {
            // (window, queries, position of the first): three queries on the per-query kernel, more
            // on the tiled one (`attnt_*`, tiles of 8), the model's window and extent last.
            for (window, t, t0) in [
                (0usize, 3usize, 40usize),
                (13, 3, 40),
                (0, 21, 40),
                (13, 21, 40),
                (0, 100, 40),
                (37, 100, 40),
                (1024, 200, 900),
            ] {
                let cap = 2048;
                let el = [4usize, 2, 1][fmt as usize];
                let kc = e.alloc(cap * kw * el).unwrap();
                let vc = e.alloc(cap * kw * el).unwrap();
                let ks = e.alloc(cap * kw / 32 * 4).unwrap();
                let vs = e.alloc(cap * kw / 32 * 4).unwrap();
                let store = |pos: usize, n: usize, seed: &mut u64| {
                    let kv = (e.upf(&rnds(n * kw, seed, 1.0)).unwrap(), e.upf(&rnds(n * kw, seed, 1.0)).unwrap());
                    set_info(&e, info, pos);
                    let groups = (2 * n * kw / 32) as u32;
                    e.go(
                        e.f("kv_store"),
                        (groups.div_ceil(128), 1, 1),
                        128,
                        &mut args![kv.0, kv.1, kc, vc, ks, vs, kw as u32, n as u32, cap as u32, fmt, info],
                    );
                    down(&e, kv.0, 1);
                };
                if t0 >= 20 {
                    store(0, t0 - 20, &mut seed);
                    store(t0 - 20, 20, &mut seed);
                }
                store(t0, t, &mut seed);
                set_info(&e, info, t0);
                // The stored rows, decoded.
                let decode = |c: u64, s: u64| -> Vec<f32> {
                    let raw = down_bytes(&e, c, cap * kw * el);
                    let sc = down(&e, s, cap * kw / 32);
                    (0..cap * kw)
                        .map(|i| match fmt {
                            0 => f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()),
                            1 => half::f16::from_bits(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])).to_f32(),
                            _ => raw[i] as i8 as f32 * sc[i / 32],
                        })
                        .collect()
                };
                let (kd, vd) = (decode(kc, ks), decode(vc, vs));
                let q = rnds(t * w, &mut seed, 3.0);
                let r = rnds(t * nh * dr, &mut seed, 1.0);
                let sink = rnds(t * nh, &mut seed, 1.0);
                let gate = rnds(t * w, &mut seed, 2.0);
                let (pq, pr, ps, pg) = (e.upf(&q).unwrap(), e.upf(&r).unwrap(), e.upf(&sink).unwrap(), e.upf(&gate).unwrap());
                for ns in [1usize, 4] {
                    let part = zeros(&e, t * nh * ns * 130);
                    let out = zeros(&e, t * w);
                    let p = AttP {
                        q: pq,
                        r: pr,
                        kc,
                        vc,
                        ks,
                        vs,
                        rel: p_rel,
                        part,
                        info,
                        nh: nh as u32,
                        nkv: nkv as u32,
                        dr: dr as u32,
                        extent: extent as u32,
                        window: window as u32,
                        cap: cap as u32,
                        ns: ns as u32,
                        qs: w as u32,
                        rs: (nh * dr) as u32,
                        scaling: (d as f32).powf(-0.5),
                    };
                    if t >= 16 {
                        let name = format!("attnt_{}_{}", ["f32", "f16", "q8"][fmt as usize], nh / nkv);
                        let f = *e.funcs.iter().find(|(k, _)| **k == name).unwrap().1;
                        e.go(f, (nkv as u32, t.div_ceil(8) as u32, ns as u32), 128, &mut args![p, t as u32]);
                    } else {
                        e.go(e.attn_f(fmt, nh / nkv), (nkv as u32, t as u32, ns as u32), 128, &mut args![p]);
                    }
                    e.go(
                        e.f("attn_merge"),
                        (nh as u32, t as u32, 1),
                        128,
                        &mut args![part, ps, pg, out, nh as u32, ns as u32, nh as u32, w as u32],
                    );
                    let got = down(&e, out, t * w);
                    for i in 0..t {
                        let pos = t0 + i;
                        let lo = if window > 0 { (pos + 1).saturating_sub(window) } else { 0 };
                        for hh in 0..nh {
                            let g = hh / (nh / nkv);
                            let qh = &q[i * w + hh * d..i * w + (hh + 1) * d];
                            let mut sc: Vec<f64> = (lo..=pos)
                                .map(|j| {
                                    let row = (j % cap) * kw + g * d;
                                    let dot: f64 = (0..d).map(|c| qh[c] as f64 * kd[row + c] as f64).sum::<f64>() / (d as f64).sqrt();
                                    let dist = pos - j;
                                    let b: f64 = if dist < extent {
                                        (0..dr)
                                            .map(|z| r[(i * nh + hh) * dr + z] as f64 * rel[z * extent + dist] as f64)
                                            .sum()
                                    } else {
                                        0.0
                                    };
                                    dot + b
                                })
                                .collect();
                            let sk = sink[i * nh + hh] as f64;
                            let m = sc.iter().fold(sk, |a, &b| a.max(b));
                            let mut l = (sk - m).exp();
                            for v in sc.iter_mut() {
                                *v = (*v - m).exp();
                                l += *v;
                            }
                            for c in 0..d {
                                let o: f64 = (lo..=pos)
                                    .zip(&sc)
                                    .map(|(j, &pj)| pj * vd[(j % cap) * kw + g * d + c] as f64)
                                    .sum::<f64>()
                                    / l;
                                let gv = gate[i * w + hh * d + c] as f64;
                                let want = o / (1.0 + (-gv).exp());
                                let have = got[i * w + hh * d + c] as f64;
                                assert!(
                                    (have - want).abs() < 2e-6 * (1.0 + want.abs()),
                                    "format {fmt} window {window} splits {ns}, {t} queries: query {i} head {hh} dim {c}: {have} against {want}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The top k is the CPU's (`route_sigmoid`) on ordinary logits, ties included, and picks k
    /// distinct valid experts whatever NaNs the logits hold.
    #[test]
    fn routing_matches_cpu_and_survives_nan() {
        let Some(be) = backend() else { return };
        let e = Engine::bare(be, &[0]).unwrap();
        let (ne, k, t) = (128usize, 8usize, 6usize);
        let mut seed = 9u64;
        let bias = rnds(ne, &mut seed, 0.05);
        let mut lg = rnds(t * ne, &mut seed, 3.0);
        // Token 1: ties; token 3: a few NaNs; token 4: all NaN; token 5: infinities.
        for j in 0..ne {
            lg[ne + j] = (j % 4) as f32;
        }
        for j in [3usize, 50, 90] {
            lg[3 * ne + j] = f32::NAN;
        }
        for j in 0..ne {
            lg[4 * ne + j] = f32::NAN;
        }
        lg[5 * ne + 7] = f32::INFINITY;
        lg[5 * ne + 8] = f32::NEG_INFINITY;
        let (pl, pb, ids, wts) = (e.upf(&lg).unwrap(), e.upf(&bias).unwrap(), zeros(&e, t * k), zeros(&e, t * k));
        e.go(
            e.f("route_k"),
            (ceil(t, 4), 1, 1),
            128,
            &mut args![pl, pb, ids, wts, ne as u32, k as u32, 2.5f32, t as u32],
        );
        let raw = down_bytes(&e, ids, t * k * 4);
        let got: Vec<i32> = raw.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
        let w = down(&e, wts, t * k);
        for tok in 0..t {
            let row = &got[tok * k..(tok + 1) * k];
            let mut u: Vec<i32> = row.to_vec();
            u.sort_unstable();
            u.dedup();
            assert_eq!(u.len(), k, "token {tok}: {row:?}");
            assert!(row.iter().all(|&i| (0..ne as i32).contains(&i)), "token {tok}: {row:?}");
            if [0usize, 1, 2].contains(&tok) {
                let want = crate::moe::route_sigmoid(&lg[tok * ne..(tok + 1) * ne], &bias, k, 2.5);
                for (sl, &(ex, wt)) in want.iter().enumerate() {
                    assert_eq!(row[sl] as usize, ex, "token {tok} slot {sl}");
                    assert!(
                        (w[tok * k + sl] - wt).abs() < 1e-5,
                        "token {tok} slot {sl}: {} against {wt}",
                        w[tok * k + sl]
                    );
                }
            }
        }
    }

    /// The convolutions, the mixes, the vector post-processing and the message gate's pieces
    /// against the CPU's functions.
    #[test]
    fn small_ops_match_cpu() {
        let Some(be) = backend() else { return };
        crate::pool::init(4);
        let mut e = Engine::bare(be, &[0]).unwrap();
        let mut seed = 3u64;
        // Short convolutions, static and with per-token taps, over rings by position: one token
        // and seven (the decode kernel), 70 (the prefill kernels; more than the ring holds), from
        // position 100 (the ring wraps).
        let (ch, kn, groups) = (96usize, 4usize, 3usize);
        e.conv_k = kn;
        e.conv_scratch = (zeros(&e, 70 * ch), zeros(&e, 70 * ch));
        let info = zeros(&e, 4);
        for dynamic in [false, true] {
            for t in [1usize, 7, 70] {
                let pos = 100usize;
                set_info(&e, info, pos);
                let taps = rnds(kn * ch, &mut seed, 0.5);
                let mut ring = super::super::ConvRing::new(kn, ch);
                ring.rows = rnds(ring.rows.len(), &mut seed, 1.0);
                let mut x = rnds(t * ch, &mut seed, 1.0);
                let dy = rnds(t * groups * kn, &mut seed, 0.3);
                let (px, ph, pt) = (e.upf(&x).unwrap(), e.upf(&ring.rows).unwrap(), e.upf(&taps).unwrap());
                let pd = if dynamic { e.upf(&dy).unwrap() } else { 0 };
                e.sconv(px, ph, pt, pd, ch, groups, t, 0.4, info);
                let (gx, gh) = (down(&e, px, t * ch), down(&e, ph, ring.rows.len()));
                super::super::short_conv(&taps, dynamic.then_some((&dy[..], groups)), &mut ring, &mut x, ch, 0.4, pos);
                for (a, b) in gx.iter().zip(&x).chain(gh.iter().zip(&ring.rows)) {
                    assert!((a - b).abs() < 1e-5, "short conv (dynamic {dynamic}, {t} tokens): {a} against {b}");
                }
            }
        }
        // Mixes: 3 states, 3 queries, 5 tokens, and none of the states.
        let (h, t) = (256usize, 5usize);
        e.h = h;
        for ns in [0usize, 3] {
            let states: Vec<Vec<f32>> = (0..ns)
                .map(|_| {
                    let mut s = rnds(t * h, &mut seed, 1.0);
                    for r in s.chunks_mut(h) {
                        super::super::rms_unit(r, MIX_EPS);
                    }
                    s
                })
                .collect();
            let x = rnds(t * h, &mut seed, 3.0);
            let q = rnds(3 * h, &mut seed, 2.0);
            let pst: Vec<u64> = states.iter().map(|s| e.upf(s).unwrap()).collect();
            let (px, pq) = (e.upf(&x).unwrap(), e.upf(&q).unwrap());
            let outs = [zeros(&e, t * h), zeros(&e, t * h), zeros(&e, t * h)];
            e.mix(&pst, px, pq, 3, &outs, &[], t);
            let want = super::super::mix(&states, &x, &q, h);
            for (o, w) in outs.iter().zip(&want) {
                for (a, b) in down(&e, *o, t * h).iter().zip(w) {
                    assert!((a - b).abs() < 1e-5, "mix ({ns} states): {a} against {b}");
                }
            }
            // The same mixes RMS-normed with weights.
            let nws: Vec<Vec<f32>> = (0..3).map(|_| rnds(h, &mut seed, 1.0)).collect();
            let pnw: Vec<u64> = nws.iter().map(|w| e.upf(w).unwrap()).collect();
            e.mix(&pst, px, pq, 3, &outs, &pnw, t);
            for ((o, w), nw) in outs.iter().zip(&want).zip(&nws) {
                let got = down(&e, *o, t * h);
                for (gr, wr) in got.chunks(h).zip(w.chunks(h)) {
                    let r = 1.0 / (wr.iter().map(|v| v * v).sum::<f32>() / h as f32 + e.eps).sqrt();
                    for ((a, b), g) in gr.iter().zip(wr).zip(nw) {
                        assert!((a - b * r * g).abs() < 2e-5, "normed mix ({ns} states): {a} against {}", b * r * g);
                    }
                }
            }
        }
        // Scale, clip and log scaling (position 9000 against a floor of 8192); RMS norm.
        let (n, nv) = (128usize, 6usize);
        e.n_floor = 8192.0;
        let info = zeros(&e, 4);
        set_info(&e, info, 9000);
        let x = rnds(t * nv * n, &mut seed, 3.0);
        let sc = rnds(nv * n, &mut seed, 1.0);
        let (px, ps) = (e.upf(&x).unwrap(), e.upf(&sc).unwrap());
        e.vec_post(px, ps, n, nv, nv * n, t, 0, 5.0, 0.1, info);
        let got = down(&e, px, t * nv * n);
        for (i, v) in x.chunks(n).enumerate() {
            let mut v: Vec<f32> = v.iter().zip(&sc[(i % nv) * n..(i % nv + 1) * n]).map(|(a, b)| a * b).collect();
            super::super::l2_clip(&mut v, 5.0);
            let tau = (1.0 + 0.1 * ((9000 + i / nv + 1) as f64 / 8192.0).max(1.0).ln()) as f32;
            for (a, b) in got[i * n..(i + 1) * n].iter().zip(&v) {
                assert!((a - b * tau).abs() < 1e-5, "clip and scale: {a} against {}", b * tau);
            }
        }
        let px = e.upf(&x).unwrap();
        e.vec_post(px, 0, n, nv, nv * n, t, 1, 0.0, 0.0, info);
        let got = down(&e, px, t * nv * n);
        for (i, v) in x.chunks(n).enumerate() {
            let mut v = v.to_vec();
            super::super::rms_unit(&mut v, e.eps);
            for (a, b) in got[i * n..(i + 1) * n].iter().zip(&v) {
                assert!((a - b).abs() < 1e-5, "value norm: {a} against {b}");
            }
        }
        // A message gate (rank 32) against its definition.
        let r = 32usize;
        e.rank = r;
        let kf = e.kfuncs.clone();
        let (wd, wf, wg) = (
            rnds(r * 2 * h, &mut seed, 0.1),
            rnds(h * r, &mut seed, 0.3),
            rnds(h * r, &mut seed, 0.3),
        );
        let g = (
            e.up_f32(&wd, r, 2 * h, &kf).unwrap(),
            e.up_f32(&wf, h, r, &kf).unwrap(),
            e.up_f32(&wg, h, r, &kf).unwrap(),
        );
        let mut ar = Engine::empty_arena();
        (ar.cat, ar.gd, ar.gf, ar.gg) = (zeros(&e, t * 2 * h), zeros(&e, t * r), zeros(&e, t * h), zeros(&e, t * h));
        let x = rnds(t * h, &mut seed, 2.0);
        let m = rnds(t * h, &mut seed, 2.0);
        let (px, pm, po) = (e.upf(&x).unwrap(), e.upf(&m).unwrap(), zeros(&e, t * h));
        e.gate(&ar, &g, px, pm, po, t);
        let got = down(&e, po, t * h);
        for tok in 0..t {
            let mut cat = [&x[tok * h..(tok + 1) * h], &m[tok * h..(tok + 1) * h]].concat();
            super::super::rms_unit(&mut cat[..h], e.eps);
            super::super::rms_unit(&mut cat[h..], e.eps);
            let dd: Vec<f64> = (0..r)
                .map(|i| (0..2 * h).map(|c| wd[i * 2 * h + c] as f64 * cat[c] as f64).sum())
                .collect();
            for j in 0..h {
                let f: f64 = (0..r).map(|i| wf[j * r + i] as f64 * dd[i]).sum();
                let gn: f64 = (0..r).map(|i| wg[j * r + i] as f64 * dd[i]).sum();
                let want = (1.0 - f.tanh()) * x[tok * h + j] as f64 + gn.tanh() * m[tok * h + j] as f64;
                let have = got[tok * h + j] as f64;
                assert!((have - want).abs() < 2e-5, "gate token {tok} {j}: {have} against {want}");
            }
        }
    }
}
