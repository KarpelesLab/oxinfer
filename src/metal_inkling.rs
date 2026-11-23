//! Inkling-2 ([`crate::inkling`]) on Metal: whole passes of up to [`MAX_T`] tokens (decoding and
//! speculative verification) in one submission, every layer application on the GPU with the
//! activations, the attention caches and the convolution rings resident there.
//!
//! The pass is the CPU's (`Inkling2Model::forward_capture`), op for op: the attention residual's
//! mixes over the block states (`ink_mix`), the message gates of the looped layers (norms,
//! three matvecs, `ink_gmix`), the projections on the decode library's matvec kernels (one
//! token) or its exact multi-token ones (several), the key / value convolutions with their
//! clip and norm written to the caches (`ink_kv_*`, f32 / f16 / int8 rows), the scaled and
//! clipped queries and relative vectors (`ink_q`), attention with the relative bias, the
//! sliding window, the sink and the output gate (`ink_attn_*` in splits of keys, then
//! `ink_attn_merge`), the convolutions after the attention and the MLP (`ink_conv`), sigmoid
//! routing (`ink_mvf` or the f32 GEMM, `ink_route`) and the soft-capped experts in their stored format, NVFP4
//! and MXFP8 included (`ink_x_*`), the final mix, the LM head and the logit cap.
//!
//! Every token of a pass is computed exactly as it would be alone (the token is a grid
//! dimension of every kernel; the multi-token matvecs keep the one-token arithmetic; attention
//! splits depend on a token's own range), so speculative verification gives the logits of
//! plain decoding.
//!
//! A session's state moves to the device on its first pass there (`InkDev::upload`) and back
//! to the host when a pass runs on the CPU (`InkDev::download`): sliding-window caches are
//! rings of `window + ROLLBACK` rows by position, full-attention caches grow; the convolution
//! rings have the host's layout.

use super::decode::{
    FastEnc, Fns, MvFmt, MvMode, MvParams, MvWeight, begin_enc, encode_mv, end_commit, end_timed, mvk_rows, run_concurrent, wait_timed,
};
use super::{GGUF_GPU_TYPES, Id, IntoInnerUnsafe, MetalBackend, MetalWeight, MsgNewBufLen, MsgVoid};
use crate::ggml::GgmlType;
use crate::gpu::WeightRef;
use crate::inkling::{
    DevBox, DevEngine, DevSession, DevWant, InNorm, Inkling2Model, Inkling2Session, KvCache, KvConv, LayerState, ROLLBACK,
};
use crate::model::Lin;
use std::fmt::Write as _;

const SRC: &str = include_str!("metal_inkling.metal");

/// Dispatches and barriers encoded so far (`OXINFER_INKLING_PROFILE` reports them per pass).
static DISPATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static BARRIERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Where each pipeline of a profile was set (`Pe::pipe`).
static PIPE_SITES: std::sync::Mutex<std::collections::BTreeMap<usize, String>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// `OXINFER_INKLING_PROFILE=N`: passes of up to [`MAX_T`] tokens run cut at every barrier, each
/// stage in a command buffer of its own, and every N passes (default 16) the GPU time per kernel
/// of a pass is printed with the dispatch and barrier counts (slow: for profiling).
fn profile_every() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("OXINFER_INKLING_PROFILE") {
        Ok(v) => v.parse().unwrap_or(16).max(1),
        Err(_) => 0,
    })
}

/// A profile in progress: the command buffer being encoded, the pipelines of its stage, the
/// GPU time per kind of stage (its sorted pipelines), the buffers each encoder makes resident.
struct Prof<'f> {
    be: &'static MetalBackend,
    f: &'f Fns,
    res: &'f [Id],
    cb: std::cell::Cell<Id>,
    stage: std::cell::RefCell<Vec<usize>>,
    times: std::cell::RefCell<std::collections::HashMap<Vec<usize>, (f64, usize)>>,
    /// The stages' command buffers, committed back to back (the GPU stays busy and at speed) and
    /// timed once the pass is done.
    pending: std::cell::RefCell<Vec<(Id, Vec<usize>)>>,
}

/// The encoder of a pass: the decode library's, counting what it encodes; in a profile it is
/// cut at every barrier ([`Prof`]).
struct Pe<'f> {
    enc: std::cell::Cell<Id>,
    f: &'f Fns,
    prof: Option<&'f Prof<'f>>,
}

impl<'f> Pe<'f> {
    fn new(e: &FastEnc<'f>, prof: Option<&'f Prof<'f>>) -> Self {
        Pe {
            enc: std::cell::Cell::new(e.enc),
            f: e.f,
            prof,
        }
    }
    fn fe(&self) -> FastEnc<'f> {
        FastEnc {
            enc: self.enc.get(),
            f: self.f,
        }
    }
    /// Notes a pipeline set by another encoding function (for the profile).
    fn note(&self, p: Id) {
        if let Some(pr) = self.prof {
            pr.stage.borrow_mut().push(p as usize);
        }
    }
    #[track_caller]
    fn pipe(&self, p: Id) {
        if self.prof.is_some() {
            // (Pipelines of other libraries are named by where they were set.)
            let l = std::panic::Location::caller();
            PIPE_SITES
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(p as usize, format!("{}:{}", l.file(), l.line()));
        }
        self.note(p);
        self.fe().pipe(p);
    }
    fn buf(&self, b: Id, offset: usize, index: usize) {
        self.fe().buf(b, offset, index);
    }
    fn bytes<T>(&self, v: &T, index: usize) {
        self.fe().bytes(v, index);
    }
    fn dispatch(&self, groups: (usize, usize), threads: usize) {
        DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.fe().dispatch(groups, threads);
    }
    fn dispatch3(&self, groups: (usize, usize, usize), threads: usize) {
        DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.fe().dispatch3(groups, threads);
    }
    fn use_resources(&self, ids: &[Id]) {
        self.fe().use_resources(ids);
    }
    fn barrier(&self) {
        BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match self.prof {
            None => self.fe().barrier(),
            Some(pr) => self.cut(pr),
        }
    }
    /// Profiles: ends the stage's command buffer, times it, and starts the next one.
    fn cut(&self, pr: &Prof) {
        let mut st = std::mem::take(&mut *pr.stage.borrow_mut());
        if st.is_empty() {
            return;
        }
        st.sort_unstable();
        st.dedup();
        end_commit(pr.f, pr.cb.get(), self.enc.get());
        pr.pending.borrow_mut().push((pr.cb.get(), st));
        let (cb, enc) = begin_enc(pr.be, pr.f);
        pr.cb.set(cb);
        self.enc.set(enc);
        self.use_resources(pr.res);
    }
}

/// The lanes per row of the grouped expert kernels (`ink_*_gpair` / `_gsum`) for rows of `n_in`
/// values in chunks of `ch`: the largest power of two dividing the chunk count, at most `max`
/// (every lane then walks the same number of chunks).
fn lanes_per_row(n_in: usize, ch: usize, max: usize) -> u32 {
    let nch = (n_in / ch).max(1);
    1u32 << nch.trailing_zeros().min(max.trailing_zeros())
}

/// The chunk of an expert format's kernels (values per `mdot`).
fn ch_of(fmt: &str) -> usize {
    expert_formats().iter().find(|f| f.0 == fmt).map_or(32, |f| f.3)
}

/// Tokens of a prefill pass at most (longer ones run in pieces), and their key splits.
const PF_T: usize = 512;
const PF_SPLIT: usize = 8;
/// Most captured streams per token of a prefill pass (DSpark drafts read four).
const PF_CAPTURE: usize = 8;

/// `OXINFER_INKLING_PREFILL=0`: prefill passes on the CPU (the matmuls on Metal), as before the
/// engine had them.
fn pf_off() -> bool {
    std::env::var("OXINFER_INKLING_PREFILL").is_ok_and(|v| v == "0")
}

const SK_DENSE: u32 = 1;
const SK_ATTN: u32 = 2;
const SK_EXP: u32 = 4;
const SK_MIX: u32 = 8;
const SK_GATES: u32 = 16;
const SK_CONV: u32 = 32;
const SK_ROUTE: u32 = 64;
const SK_HEAD: u32 = 128;
const SK_QKV: u32 = 256;
const SK_ATTK: u32 = 512;
const SK_MERGE: u32 = 1024;
const SK_MVF: u32 = 4096;
const SK_RT: u32 = 8192;
const SK_NORM: u32 = 16384;

/// Tokens of a pass at most (the decode library's multi-token matvecs take 16).
pub(crate) const MAX_T: usize = 16;

/// Splits of a token's keys at most (`ink_chunk` in the shader).
const MAX_SPLIT: usize = 64;
/// The epsilon of the attention residual's norms (`inkling::MIX_EPS`).
const MIX_EPS: f32 = 1e-6;
/// Added to a norm before an L2 clip divides by it (`inkling::CLIP_EPS`).
const CLIP_EPS: f32 = 1e-6;

/// The formats of routed experts: kernel suffix, weight type, x chunk type and size, and the
/// constructor of the weight type from a table entry `e` (rows of `n` values).
fn expert_formats() -> Vec<(String, String, &'static str, usize, String)> {
    let raw = |n: &str, t: &str, x: &'static str, ch: usize| {
        (
            n.to_string(),
            t.to_string(),
            x,
            ch,
            format!("{t}{{ (device const uchar*)(e).q, (e).rb }}"),
        )
    };
    let mut v = vec![
        raw("bf", "MBf", "X16", 16),
        (
            "q8".into(),
            "MQ8".into(),
            "X16",
            16,
            "MQ8{ (device const char*)(e).q, (device const float*)(e).s, n }".into(),
        ),
        (
            "nvfp4".into(),
            "MGen<NVFP4W>".into(),
            "X32",
            32,
            "MGen<NVFP4W>{ NVFP4W{ { (device const uchar*)(e).q, (e).rb } } }".into(),
        ),
        (
            "mxf8".into(),
            "MGen<MXF8W>".into(),
            "X32",
            32,
            "MGen<MXF8W>{ MXF8W{ (device const uchar*)(e).q, (device const float*)(e).s, n } }".into(),
        ),
        (
            "mxf8b".into(),
            "MGen<MXF8B>".into(),
            "X32",
            32,
            "MGen<MXF8B>{ MXF8B{ (device const uchar*)(e).q, (device const uchar*)(e).s, n } }".into(),
        ),
        (
            "MXFP4".into(),
            "MGen<MXFP4W>".into(),
            "X32",
            32,
            "MGen<MXFP4W>{ MXFP4W{ { (device const uchar*)(e).q, (e).rb } } }".into(),
        ),
        (
            "Q2_0".into(),
            "MGen<Q2_0W>".into(),
            "X32",
            32,
            "MGen<Q2_0W>{ Q2_0W{ { (device const uchar*)(e).q, (e).rb } } }".into(),
        ),
    ];
    for &k in GGUF_GPU_TYPES.iter() {
        let n = k.name();
        v.push(match k {
            GgmlType::F16 => raw(n, "MF16", "X16", 16),
            GgmlType::Q4_0 => raw(n, "MQ4_0", "X32", 32),
            GgmlType::Q8_0 => raw(n, "MQ8_0", "X32", 32),
            GgmlType::Q4_K => raw(n, "MQ4_K", "XKQ", 32),
            GgmlType::Q5_K => raw(n, "MQ5_K", "XKQ", 32),
            GgmlType::Q6_K => raw(n, "MQ6_K", "X6", 32),
            _ => {
                let w = if k == GgmlType::F32 { "F32W2".to_string() } else { format!("{n}W") };
                (
                    n.to_string(),
                    format!("MGen<{w}>"),
                    "X32",
                    32,
                    format!("MGen<{w}>{{ {w}{{ {{ (device const uchar*)(e).q, (e).rb }} }} }}"),
                )
            }
        });
    }
    v
}

/// The Inkling-2 kernels and their expert instantiations, after the sources they use (the
/// weight loaders, the decode kernels' matvec templates); with `mma`, the matrix-unit GEMMs and
/// the prefill kernels too. `OXINFER_INKLING_METAL=FILE` takes the kernels from a file (kernel
/// work).
pub(crate) fn shader_src(mma: bool) -> String {
    let mut s = format!(
        "{}{}{}#define MVK_GROUP 4\n{}",
        super::SHADER_SRC,
        super::gguf_shader_tables(),
        super::GGUF_SHADER_SRC,
        super::decode::SRC
    );
    if mma {
        s = format!("#define GTM {}\n#define MMA_UNITS 1\n{s}{}", super::MMA_GTM, super::MMA_SHADER_SRC);
    }
    match std::env::var_os("OXINFER_INKLING_METAL").and_then(|f| std::fs::read_to_string(f).ok()) {
        Some(src) => s.push_str(&src),
        None => s.push_str(SRC),
    }
    for (n, t, x, ch, mk) in expert_formats() {
        let _ = writeln!(s, "#define IMK_{n}(e, n) {mk}");
        let _ = writeln!(s, "INK_XP(ink_x_{n}, {t}, {x}, {ch}, IMK_{n})");
        let _ = writeln!(s, "INK_XG(ink_x_{n}, {t}, {x}, {ch}, IMK_{n})");
    }
    if mma {
        s.push_str(PF_SRC);
    }
    s
}

const PF_SRC: &str = include_str!("metal_inkling_pf.metal");

// ---- Kernel parameters (as the shader's structs) -------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MixP {
    h: u32,
    ns: u32,
    norm: u32,
    x: u32,
    q: u32,
    out: u32,
    out_qs: u32,
    nw: u32,
    scale: f32,
    eps: f32,
    mix_eps: f32,
    _p: u32,
    s: [u32; 16],
    cat: u32,
    cat_qs: u32,
    cat_ts: u32,
    cat_last: u32,
    cat_n: u32,
    _p1: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NormP {
    h: u32,
    eps: f32,
    x: u32,
    out: u32,
    w: u32,
    has_w: u32,
    xs: u32,
    os: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CopyP {
    n: u32,
    src: u32,
    dst: u32,
    ds: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct KvP {
    kw: u32,
    hd: u32,
    kn: u32,
    nkv: u32,
    dynamic: u32,
    norm: f32,
    clip: f32,
    eps: f32,
    k: u32,
    v: u32,
    kd: u32,
    vd: u32,
    tk: u32,
    tv: u32,
    pos: u32,
    ring_cap: u32,
    cap: u32,
    clip_eps: f32,
    ring_write: u32,
    row_base: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QP {
    w: u32,
    hd: u32,
    dr: u32,
    nh: u32,
    q: u32,
    r: u32,
    qs: u32,
    rs: u32,
    qclip: f32,
    rclip: f32,
    alpha: f32,
    floor_: f32,
    pos: u32,
    clip_eps: f32,
    _p0: u32,
    _p1: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AttnP {
    nh: u32,
    nkv: u32,
    hd: u32,
    dr: u32,
    pos: u32,
    window: u32,
    cap: u32,
    extent: u32,
    cmin: u32,
    maxsplit: u32,
    scale: f32,
    q: u32,
    r: u32,
    part: u32,
    rel: u32,
    kw: u32,
    tbase: u32,
    tmp: u32,
    _p0b: u32,
    _p1b: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MergeP {
    nh: u32,
    hd: u32,
    maxsplit: u32,
    part: u32,
    sink: u32,
    gate: u32,
    out: u32,
    pos: u32,
    window: u32,
    cmin: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ConvP {
    h: u32,
    kn: u32,
    x: u32,
    out: u32,
    taps: u32,
    pos: u32,
    mode: u32,
    norm: f32,
    ring_cap: u32,
    ntok: u32,
    ring_write: u32,
    _p1: u32,
    od: u32,
    ok: u32,
    _q0: u32,
    _q1: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RouteP {
    n: u32,
    k: u32,
    logits: u32,
    sel: u32,
    bias: u32,
    scale: f32,
    single: u32,
    grp: u32,
    asg: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MvfP {
    n_out: u32,
    n_in: u32,
    w: u32,
    x: u32,
    y: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CapP {
    v: u32,
    u: u32,
    x: u32,
    out: u32,
    cap: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
/// An expert's projection in the address tables (`IXT`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XEntry {
    q: u64,
    s: u64,
    factor: f32,
    rb: u32,
    _p: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XP {
    inter: u32,
    h: u32,
    k: u32,
    il: u32,
    off: u32,
    x: u32,
    sel: u32,
    act: u32,
    y: u32,
    cap_g: f32,
    cap_u: f32,
    add: u32,
}
/// The decode library's multi-token matvec parameters (`MVK`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MvkP {
    m: MvParams,
    ntok: u32,
    sx: u32,
    sy: u32,
    sgate: u32,
}

// ---- The model on the device ------------------------------------------------------------------

/// A dense matrix and its matvec pipelines: one token (pipeline, rows and SIMD-groups per
/// threadgroup) and multi-token (8 and 16 tokens: pipeline, rows per threadgroup, threads).
struct Dw {
    w: MvWeight,
    /// Runtime LoRA updates of the matrix (`crate::lora`), applied beside it.
    lora: Vec<LoraDev>,
    one: (Id, usize, usize),
    multi: [(Id, usize, usize); 2],
    /// Prefill: the matrix-unit GEMM, where the format has one; float32 weights (the loop's
    /// gates): their buffer and offset in floats, for the f32 GEMM.
    g: Option<Gemm>,
    f32g: Option<(Id, u32)>,
}

/// A runtime LoRA update on the device: `A` and `B` transposed (copies), its rank and shape, and
/// the update itself (its strength is read at each pass: adapters switch on and off).
struct LoraDev {
    a: Id,
    bt: Id,
    r: usize,
    n_out: usize,
    n_in: usize,
    d: *const crate::lora::LoraDelta,
}

/// The routed experts' down-projection LoRA updates of a layer: the address table (`IXL`, its
/// strengths written at each pass) and its contents, the updates, the largest rank.
struct XLora {
    tab: Id,
    tab_p: *mut u8,
    deltas: Vec<Option<*const crate::lora::LoraDelta>>,
    rmax: usize,
}

/// A dense matrix-unit GEMM (`mma_*`): pipelines and outputs per threadgroup for 128-, 32- and
/// 64-token tiles, the weight's buffers, its offsets (elements, and scales or row bytes) and
/// affine Q4's offsets.
#[derive(Clone, Copy)]
struct Gemm {
    pipes: [(Id, usize); 3],
    data: Id,
    scales: Id,
    w_off: u32,
    s_off: u32,
    offs: Option<(Id, [u32; 4])>,
}

/// Prefill: the routed experts of a layer as grouped matrix-unit GEMMs (`mmag_*`, tiles of 32,
/// 128 and 16 rows) over the tokens of each expert: pipelines, `{data, scales}` address tables,
/// row bytes of the GGUF formats, and the experts' output factors.
struct Grouped {
    gu: [Id; 3],
    d: [Id; 3],
    ep_gu: Id,
    ep_d: Id,
    rb: (u32, u32),
    f_gu: Vec<f32>,
    f_d: Vec<f32>,
}

struct GateDev {
    down: Dw,
    forget: Dw,
    gain: Dw,
}

/// The routed experts of a layer: the pair and sum pipelines (formats of gate/up and of down)
/// and their address tables.
struct ExpDev {
    pair: Id,
    sum: Id,
    /// The same over the pass's experts grouped by expert (passes of up to MAX_T tokens).
    gpair: Id,
    gsum: Id,
    tab_gu: Id,
    tab_d: Id,
    il: u32,
    off: u32,
    /// Lanes per row of the grouped gate / up and down kernels.
    lpr: (u32, u32),
    grouped: Option<Grouped>,
    xlora: Option<XLora>,
    /// The gate / up projections' updates: `A x` per slot first, the pair kernels add the rest
    /// before SiTU (prefill passes then take the pair kernels too, not the grouped GEMMs).
    xlora_gu: Option<XLora>,
}

/// An expert's down-projection LoRA update in the address table (`IXL`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XlEntry {
    a: u64,
    bt: u64,
    r: u32,
    s: f32,
    _p: u64,
}

struct LayerDev {
    sliding: bool,
    boundary: bool,
    nh: usize,
    nkv: usize,
    hd: usize,
    window: usize,
    extent: usize,
    /// Consts offsets: the residual queries before the attention (and their count), the MLP
    /// query, the input norms (1 or 3, contiguous), the post-attention norm.
    attn_query: Option<(usize, usize)>,
    mlp_query: usize,
    ln_in: (usize, usize),
    ln_post: usize,
    q_scale: usize,
    r_scale: usize,
    rel: usize,
    k_taps: usize,
    v_taps: usize,
    attn_conv: usize,
    mlp_conv: usize,
    router: usize,
    bias: usize,
    q: Dw,
    r: Dw,
    gate: Dw,
    sink: Dw,
    k: Dw,
    v: Dw,
    o: Dw,
    kd: Option<Dw>,
    vd: Option<Dw>,
    gates: Option<([GateDev; 4], usize)>,
    exp: ExpDev,
}

/// What a step of the pass does with the loop and the block states (fixed by the config).
#[derive(Clone, Default)]
struct StepPlan {
    layer: usize,
    slot: usize,
    /// At the loop's start: save the stream (first pass), or keep it as the bank and restore the
    /// saved one (later passes).
    save: bool,
    swap: bool,
    /// The message bank of a later pass (block states, stream).
    bank: Option<Vec<usize>>,
    /// The block states before the attention, the state this layer pushes, those after.
    attn_states: Vec<usize>,
    push: Option<usize>,
    mlp_states: Vec<usize>,
}

/// Arena offsets (floats), each sized for `tm` tokens.
#[derive(Default)]
struct Bufs {
    tm: usize,
    /// The block states (`tm * hidden` floats each).
    states: usize,
    x: usize,
    xin: usize,
    bankx: usize,
    msgs: usize,
    mixed: usize,
    xn: usize,
    gcat: usize,
    gd: usize,
    gf: usize,
    gg: usize,
    q: usize,
    r: usize,
    gate: usize,
    sink: usize,
    k: usize,
    v: usize,
    kd: usize,
    vd: usize,
    att: usize,
    ab: usize,
    hn: usize,
    hm: usize,
    rlog: usize,
    sel: usize,
    act: usize,
    o: usize,
    part: usize,
    hidden: usize,
    logits: usize,
    capped: usize,
    cap: usize,
    /// LoRA scratch (the down halves' outputs) and the largest rank.
    lu: usize,
    lora_r: usize,
    /// Passes of up to MAX_T tokens: the experts grouped (`ink_xgroup`), the down rows per
    /// (token, slot).
    grp: usize,
    asg: usize,
    od: usize,
    h: usize,
    /// Most key splits of a token's attention.
    maxsplit: usize,
    /// Most captured streams per token.
    nf: usize,
    /// Prefill arenas: the GEMM scratch (half rows, their scales), a pass's keys and values of a
    /// sliding window (in the cache's layout, f32-sized), the experts' staging.
    pf: Option<PfBufs>,
}

#[derive(Default)]
struct PfBufs {
    xh: usize,
    rs: usize,
    kt: usize,
    vt: usize,
    kts: usize,
    vts: usize,
    /// Staging of the grouped experts: rows (gathered tokens), tile lists, source rows, the
    /// (token, slot) -> row map, per-row factors, half inputs and their scales, gate/up
    /// outputs, half activations and scales, down outputs.
    max_rows: usize,
    rexp: usize,
    tiles: usize,
    src: usize,
    map: usize,
    f_gu: usize,
    f_d: usize,
    xe: usize,
    rse: usize,
    gu: usize,
    ah: usize,
    rs2: usize,
    od: usize,
}

/// The arena of a kind of pass.
struct Arena {
    id: Id,
    b: Bufs,
}

struct Pipes {
    mix: Id,
    /// `ink_mix` for at most 8 block states (fewer registers).
    mix8: Id,
    norm: Id,
    copy: Id,
    kv: [Id; 3],
    q: Id,
    attn: [Id; 3],
    /// Passes of up to MAX_T tokens: lanes across the channels (`ink_attv`).
    attv: [Id; 3],
    merge: Id,
    conv: Id,
    route: Id,
    mvf: Id,
    softcap: Id,
    xgroup: Id,
    xcombine: Id,
    gmix_norm: Id,
    gfg: Id,
    mvr: Id,
    lora_a: Id,
    lora_b: Id,
    xlora_a: Id,
    xlora_b: Id,
}

/// The prefill kernels.
struct PfPipes {
    x_half: Id,
    gemm_f32: Id,
    ring_store: Id,
    kv_ring: Id,
    situ_half: Id,
    combine: Id,
}

pub(crate) struct InkMetal {
    be: &'static MetalBackend,
    f: Fns,
    /// Passes of up to `MAX_T` tokens, and prefill passes of up to `PF_T`.
    dec: Arena,
    pfa: Option<Arena>,
    consts: Id,
    p: Pipes,
    pfp: Option<PfPipes>,
    layers: Vec<LayerDev>,
    plan: Vec<StepPlan>,
    final_states: Vec<usize>,
    final_query: usize,
    final_norm: usize,
    lm_head: Dw,
    /// Buffers reached through the experts' address tables (made resident).
    indirect: Vec<Id>,
    /// Buffers made here (released on drop) and shared wraps (given back on drop).
    owned: Vec<Id>,
    shared: Vec<Id>,
    // The config's numbers the passes use.
    h: usize,
    vocab: usize,
    unpadded: usize,
    kn: usize,
    dr: usize,
    rank: usize,
    ne: usize,
    topk: usize,
    inter: usize,
    eps: f32,
    route_scale: f32,
    caps: (f32, f32, f32),
    clips: (f32, f32),
    log_scale: (f32, f32),
    ring_rows: usize,
    window: usize,
    lock: std::sync::Mutex<()>,
}
unsafe impl Send for InkMetal {}
unsafe impl Sync for InkMetal {}

impl Drop for InkMetal {
    fn drop(&mut self) {
        let release: MsgVoid = self.be.o.msg_send();
        let last: Vec<Id> = self.shared.iter().copied().filter(|&b| self.be.unwrap_shared(b)).collect();
        if let Some(r) = &self.be.residency
            && !last.is_empty()
        {
            r.update(&last, false);
        }
        for &b in self.owned.iter().chain(&last) {
            unsafe { release(b, self.be.sels.release) };
        }
    }
}

fn specialized(f: MvFmt) -> bool {
    !matches!(f, MvFmt::Gguf(k) if !matches!(k, GgmlType::F16 | GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K))
}

fn chunk(f: MvFmt) -> usize {
    match f {
        MvFmt::Bf16 | MvFmt::Q8 | MvFmt::Gguf(GgmlType::F16) => 16,
        _ => 32,
    }
}

/// An expert projection as the kernels read it: format suffix, data and scales memory, the
/// factor of its outputs and the row bytes.
struct XWeight {
    name: String,
    data: (*const u8, usize),
    scales: Option<(*const u8, usize)>,
    factor: f32,
    rb: u32,
}

fn expert_weight(w: &WeightRef, n_out: usize, n_in: usize) -> Option<XWeight> {
    Some(match w {
        WeightRef::Bf16(v) => XWeight {
            name: "bf".into(),
            data: (v.as_ptr() as *const u8, v.len() * 2),
            scales: None,
            factor: 1.0,
            rb: (n_in * 2) as u32,
        },
        WeightRef::Q8(m) => XWeight {
            name: "q8".into(),
            data: (m.q.as_ptr() as *const u8, m.q.len()),
            scales: Some((m.scales.as_ptr() as *const u8, m.scales.len() * 4)),
            factor: 1.0,
            rb: 0,
        },
        WeightRef::Q4(_) => return None,
        WeightRef::Gguf(g) => {
            if g.n_out != n_out || g.n_in != n_in {
                return None;
            }
            let mut factor = 1.0;
            let mut scales = None;
            let name = match (&g.block_scales, g.kind) {
                (Some(b), GgmlType::F8E4M3) if b.rows == 1 && b.cols == 32 => match &b.e8m0 {
                    // The E8M0 bytes as stored (decoded in the kernels), or their f32 expansion.
                    Some(e) => {
                        scales = Some((e.as_ptr(), e.len()));
                        "mxf8b".to_string()
                    }
                    None => {
                        scales = Some((b.s.as_ptr() as *const u8, b.s.len() * 4));
                        "mxf8".to_string()
                    }
                },
                (Some(_), GgmlType::F8E4M3) => return None,
                (Some(b), k) if !k.is_float() && b.rows >= n_out && b.cols >= n_in && b.s.len() == 1 => {
                    factor = b.s[0];
                    kind_name(k)?
                }
                (Some(_), _) => return None,
                (None, k) => kind_name(k)?,
            };
            XWeight {
                name,
                data: (g.data.as_ptr(), g.data.len()),
                scales,
                factor,
                rb: g.row_bytes() as u32,
            }
        }
    })
}

fn kind_name(k: GgmlType) -> Option<String> {
    Some(match k {
        GgmlType::NVFP4 => "nvfp4".into(),
        GgmlType::BF16 => "bf".into(),
        GgmlType::MXFP4 => "MXFP4".into(),
        GgmlType::Q2_0 => "Q2_0".into(),
        k if GGUF_GPU_TYPES.contains(&k) => k.name().into(),
        _ => return None,
    })
}

/// The device state of a session's cache slot.
struct SlotDev {
    kw: usize,
    fmt: KvCache,
    /// Sliding-window slots: rows of the ring (`window + ROLLBACK`); else 0 (rows by position).
    ring: usize,
    /// Rows the buffers hold.
    rows: usize,
    k: (Id, *mut u8),
    v: (Id, *mut u8),
    ks: (Id, *mut u8),
    vs: (Id, *mut u8),
    rings: [(Id, *mut u8, usize); 4],
}

/// A session's caches and rings on the device, and the session's position (what they hold).
pub(crate) struct InkDev {
    slots: Vec<SlotDev>,
    be: &'static MetalBackend,
    pos: usize,
}
unsafe impl Send for InkDev {}
unsafe impl Sync for InkDev {}

impl Drop for InkDev {
    fn drop(&mut self) {
        let release: MsgVoid = self.be.o.msg_send();
        for s in &self.slots {
            for b in [s.k.0, s.v.0, s.ks.0, s.vs.0].into_iter().chain(s.rings.iter().map(|r| r.0)) {
                if !b.is_null() {
                    unsafe { release(b, self.be.sels.release) };
                }
            }
        }
    }
}

fn row_bytes(fmt: KvCache, kw: usize) -> usize {
    match fmt {
        KvCache::F32 => kw * 4,
        KvCache::F16 => kw * 2,
        KvCache::Q8 => kw,
    }
}

impl MetalBackend {
    /// A shared buffer of `bytes` (zeroed) and its contents.
    fn ink_buffer(&self, bytes: usize) -> Option<(Id, *mut u8)> {
        let new_len: MsgNewBufLen = self.o.msg_send();
        let id = unsafe { new_len(self.device, self.sels.new_buffer_len, bytes.max(16), 0) };
        if id.is_null() {
            return None;
        }
        let contents: super::MsgPtr = self.o.msg_send();
        let p = unsafe { contents(id, self.sels.contents) } as *mut u8;
        unsafe { std::ptr::write_bytes(p, 0, bytes.max(16)) };
        Some((id, p))
    }
    fn ink_release(&self, id: Id) {
        if !id.is_null() {
            let release: MsgVoid = self.o.msg_send();
            unsafe { release(id, self.sels.release) };
        }
    }
}

impl InkDev {
    /// Makes room for `rows` positions in the full-attention slots (rows kept).
    fn reserve(&mut self, rows: usize) -> bool {
        let be = self.be;
        for s in &mut self.slots {
            if s.ring != 0 || s.rows >= rows {
                continue;
            }
            let n = rows.next_power_of_two().max(1024);
            let rb = row_bytes(s.fmt, s.kw);
            let sb = if s.fmt == KvCache::Q8 { s.kw / 32 * 4 } else { 4 };
            let (Some(k), Some(v), Some(ks), Some(vs)) = (
                be.ink_buffer(n * rb),
                be.ink_buffer(n * rb),
                be.ink_buffer(n * sb),
                be.ink_buffer(n * sb),
            ) else {
                return false;
            };
            unsafe {
                std::ptr::copy_nonoverlapping(s.k.1, k.1, s.rows * rb);
                std::ptr::copy_nonoverlapping(s.v.1, v.1, s.rows * rb);
                if s.fmt == KvCache::Q8 {
                    std::ptr::copy_nonoverlapping(s.ks.1, ks.1, s.rows * sb);
                    std::ptr::copy_nonoverlapping(s.vs.1, vs.1, s.rows * sb);
                }
            }
            for b in [s.k.0, s.v.0, s.ks.0, s.vs.0] {
                be.ink_release(b);
            }
            (s.k, s.v, s.ks, s.vs, s.rows) = (k, v, ks, vs, n);
        }
        true
    }
}

impl DevSession for InkDev {
    fn clone_box(&self) -> Box<dyn DevSession> {
        // A copy of every buffer (shared memory: copied by the CPU).
        let be = self.be;
        let copy = |b: (Id, *mut u8), n: usize| -> (Id, *mut u8) {
            let (id, p) = be.ink_buffer(n).expect("Metal: a buffer for a session copy");
            unsafe { std::ptr::copy_nonoverlapping(b.1, p, n) };
            (id, p)
        };
        let slots = self
            .slots
            .iter()
            .map(|s| {
                let rb = row_bytes(s.fmt, s.kw);
                let sb = if s.fmt == KvCache::Q8 { s.kw / 32 * 4 } else { 4 };
                SlotDev {
                    kw: s.kw,
                    fmt: s.fmt,
                    ring: s.ring,
                    rows: s.rows,
                    k: copy(s.k, s.rows * rb),
                    v: copy(s.v, s.rows * rb),
                    ks: copy(s.ks, s.rows * sb),
                    vs: copy(s.vs, s.rows * sb),
                    rings: s.rings.map(|r| {
                        let (id, p) = copy((r.0, r.1), r.2 * 4);
                        (id, p, r.2)
                    }),
                }
            })
            .collect();
        Box::new(InkDev { slots, be, pos: self.pos })
    }
    fn bytes(&self) -> usize {
        self.kv_bytes(self.pos)
    }
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
    /// The caches and rings are read by position: going back is forgetting the later rows.
    fn truncate(&mut self, pos: usize, len: usize) -> bool {
        if len > pos || pos - len > ROLLBACK {
            return false;
        }
        self.pos = len;
        true
    }
}

impl InkDev {
    /// Writes the caches and rings of a session at position `pos` into the host's `layers`.
    fn download(&self, layers: &mut [LayerState], pos: usize) {
        for (s, st) in self.slots.iter().zip(layers.iter_mut()) {
            let first = if s.ring != 0 { pos.saturating_sub(s.ring) } else { 0 };
            let rb = row_bytes(s.fmt, s.kw);
            for (rows, src, sc) in [(&mut st.k, s.k.1, s.ks.1), (&mut st.v, s.v.1, s.vs.1)] {
                *rows = crate::inkling::KvRows::new(s.fmt);
                for p in first..pos {
                    let r = if s.ring != 0 { p % s.ring } else { p };
                    unsafe {
                        let at = src.add(r * rb);
                        match s.fmt {
                            KvCache::F32 => rows.f.extend_from_slice(std::slice::from_raw_parts(at as *const f32, s.kw)),
                            KvCache::F16 => rows.h.extend_from_slice(std::slice::from_raw_parts(at as *const u16, s.kw)),
                            KvCache::Q8 => {
                                rows.q.extend_from_slice(std::slice::from_raw_parts(at as *const i8, s.kw));
                                rows.s
                                    .extend_from_slice(std::slice::from_raw_parts((sc as *const f32).add(r * s.kw / 32), s.kw / 32));
                            }
                        }
                    }
                }
            }
            st.first = first;
            for (ring, (_, p, n)) in st.hist.iter_mut().zip(&s.rings) {
                debug_assert_eq!(ring.rows.len(), *n);
                unsafe { std::ptr::copy_nonoverlapping(*p as *const f32, ring.rows.as_mut_ptr(), *n) };
            }
        }
    }
    /// Bytes of the attention caches at position `pos`.
    fn kv_bytes(&self, pos: usize) -> usize {
        self.slots
            .iter()
            .map(|s| {
                let n = if s.ring != 0 { pos.min(s.ring) } else { pos };
                let per = row_bytes(s.fmt, s.kw) + if s.fmt == KvCache::Q8 { s.kw / 32 * 4 } else { 0 };
                2 * n * per
            })
            .sum()
    }
}

/// Consts: float tables appended at 64-float boundaries.
#[derive(Default)]
struct Consts(Vec<f32>);
impl Consts {
    fn add(&mut self, v: &[f32]) -> usize {
        let o = self.0.len();
        self.0.extend_from_slice(v);
        self.0.resize(self.0.len().next_multiple_of(64), 0.0);
        o
    }
}

/// The library of [`shader_src`], compiled once per process, and its pipelines by name.
struct Lib {
    lib: Id,
    /// With the matrix-unit GEMMs (Metal 4) and the prefill kernels.
    mma: bool,
    pipes: std::sync::Mutex<std::collections::HashMap<String, usize>>,
}
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

fn library(be: &MetalBackend) -> anyhow::Result<&'static Lib> {
    use super::{Msg0, MsgNewLib};
    static LIB: std::sync::OnceLock<Result<Lib, String>> = std::sync::OnceLock::new();
    let r = LIB.get_or_init(|| {
        let o = &be.o;
        let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id = unsafe { o.objc.get(c"objc_getClass") }
            .map_err(|e| e.to_string())?
            .into_inner_unsafe();
        let msg_str: unsafe extern "C" fn(Id, super::Sel, *const std::ffi::c_char) -> Id = o.msg_send();
        let msg0: Msg0 = o.msg_send();
        let mma = be.mma().is_some_and(|m| m.units);
        let src = std::ffi::CString::new(shader_src(mma)).map_err(|e| e.to_string())?;
        let t0 = std::time::Instant::now();
        let ns = unsafe { msg_str(get_class(c"NSString".as_ptr()), o.sel(c"stringWithUTF8String:"), src.as_ptr()) };
        let new_lib: MsgNewLib = o.msg_send();
        let mut err: Id = std::ptr::null_mut();
        let mut opts: Id = std::ptr::null_mut();
        if mma {
            let cls = unsafe { get_class(c"MTLCompileOptions".as_ptr()) };
            if !cls.is_null() {
                opts = unsafe { msg0(msg0(cls, o.sel(c"alloc")), o.sel(c"init")) };
                let set_ver: unsafe extern "C" fn(Id, super::Sel, u64) = o.msg_send();
                unsafe { set_ver(opts, o.sel(c"setLanguageVersion:"), 0x4_0000) }; // MTLLanguageVersion4_0
            }
        }
        let lib = unsafe { new_lib(be.device, o.sel(c"newLibraryWithSource:options:error:"), ns, opts, &mut err) };
        if !opts.is_null() {
            let void: MsgVoid = o.msg_send();
            unsafe { void(opts, be.sels.release) };
        }
        if lib.is_null() {
            let text = if err.is_null() {
                String::new()
            } else {
                unsafe { o.nsstring_to_string(msg0(err, o.sel(c"localizedDescription"))) }
            };
            return Err(format!("Inkling-2 kernels: {text}"));
        }
        if std::env::var_os("OXINFER_DEBUG").is_some() {
            eprintln!("[inkling metal: kernels compiled in {:.2}s]", t0.elapsed().as_secs_f64());
        }
        Ok(Lib {
            lib,
            mma,
            pipes: Default::default(),
        })
    });
    r.as_ref().map_err(|e| anyhow::anyhow!("{e}"))
}

impl Lib {
    fn pipeline(&self, be: &MetalBackend, name: &str) -> Option<Id> {
        use super::{Msg1, MsgNewPipe, MsgVoid};
        let mut m = self.pipes.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(&p) = m.get(name) {
            return Some(p as Id);
        }
        let o = &be.o;
        let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id =
            unsafe { o.objc.get(c"objc_getClass") }.ok()?.into_inner_unsafe();
        let msg_str: unsafe extern "C" fn(Id, super::Sel, *const std::ffi::c_char) -> Id = o.msg_send();
        let cname = std::ffi::CString::new(name).ok()?;
        let ns = unsafe { msg_str(get_class(c"NSString".as_ptr()), o.sel(c"stringWithUTF8String:"), cname.as_ptr()) };
        let msg_fn: Msg1 = o.msg_send();
        let f = unsafe { msg_fn(self.lib, o.sel(c"newFunctionWithName:"), ns) };
        if f.is_null() {
            return None;
        }
        let new_pipe: MsgNewPipe = o.msg_send();
        let mut err: Id = std::ptr::null_mut();
        let p = unsafe { new_pipe(be.device, o.sel(c"newComputePipelineStateWithFunction:error:"), f, &mut err) };
        let void: MsgVoid = o.msg_send();
        unsafe { void(f, be.sels.release) };
        if p.is_null() {
            return None;
        }
        m.insert(name.to_string(), p as usize);
        Some(p)
    }
}

/// Builds the engine of `m` on the active Metal backend; None when the model or its weights are
/// not for these kernels (it then runs on the CPU).
pub(crate) fn build(m: &Inkling2Model) -> Option<Box<dyn DevEngine>> {
    let be: &'static MetalBackend = crate::gpu::active()?.as_any()?.downcast_ref::<MetalBackend>()?;
    match InkMetal::new(be, m) {
        Ok(e) => Some(Box::new(e)),
        Err(e) => {
            if std::env::var_os("OXINFER_DEBUG").is_some() || std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                eprintln!("[inkling metal: {e}]");
            }
            None
        }
    }
}

impl InkMetal {
    fn new(be: &'static MetalBackend, m: &Inkling2Model) -> anyhow::Result<Self> {
        use anyhow::{Context, bail};
        let c = &m.cfg;
        let h = c.hidden_size;
        if !h.is_multiple_of(32) || c.conv_kernel_size == 0 {
            bail!("hidden size {h}");
        }
        let mut consts = Consts::default();
        let mut shared: Vec<Id> = Vec::new();
        let mut indirect: Vec<Id> = Vec::new();
        let mut owned: Vec<Id> = Vec::new();
        let lib = library(be)?;
        let pipe = |n: &str| lib.pipeline(be, n).with_context(|| format!("kernel {n}"));
        // A dense matrix on the decode matvecs.
        let lora_bufs = std::cell::RefCell::new(Vec::<Id>::new());
        let rmax = std::cell::Cell::new(1usize);
        let dense = |l: &Lin, n_out: usize, n_in: usize, shared: &mut Vec<Id>| -> anyhow::Result<Dw> {
            let w = match l.as_ref() {
                WeightRef::Gguf(g) if g.kind == GgmlType::BF16 && g.block_scales.is_none() => {
                    let (id, addr) = be.wrap_shared(g.data.as_ptr(), g.data.len()).context("wrap")?;
                    shared.push(id);
                    let off = addr - be.buffer_gpu_address(id);
                    MvWeight {
                        fmt: MvFmt::Bf16,
                        n_out,
                        n_in,
                        data: id,
                        data_off: off,
                        scales: id,
                        scales_off: 0,
                        offs: id,
                        offs_off: 0,
                        gpr: (n_in * 2) as u32,
                        gk: 0,
                    }
                }
                WeightRef::Gguf(g) if g.block_scales.is_some() => bail!("a dense {} matrix with block scales", g.kind),
                wr => {
                    let dev = l.device_weight(n_out, n_in).context("no device weight")?;
                    if let Some(mw) = dev.downcast_ref::<MetalWeight>() {
                        mw.keep_resident(&be.residency);
                    }
                    MvWeight::new(&wr, dev, n_out, n_in).with_context(|| format!("no matvec for [{n_out}, {n_in}]"))?
                }
            };
            let ks = if specialized(w.fmt) {
                (n_in / chunk(w.fmt) / 64).clamp(1, 4).next_power_of_two().min(4)
            } else {
                4
            };
            let (nr1, ks1) = if specialized(w.fmt) { (1, ks) } else { (2, 4) };
            let one = (be.mv_pipe(w.fmt, nr1, ks1).context("mv pipeline")?, nr1, ks1);
            if profile_every() > 0 {
                eprintln!(
                    "[inkling profile: matvec [{n_out}, {n_in}] {:?}: pipeline {:#x}]",
                    w.fmt, one.0 as usize
                );
            }
            let mk = |tt: usize| -> anyhow::Result<(Id, usize, usize)> {
                Ok((
                    be.mvk_pipe(w.fmt, mvk_rows(tt), ks, tt).context("mvk pipeline")?,
                    mvk_rows(tt),
                    32 * ks,
                ))
            };
            // Prefill: the matrix-unit GEMM.
            let g = (|| -> Option<Gemm> {
                let wr = l.as_ref();
                if matches!(&wr, WeightRef::Gguf(g) if g.kind == GgmlType::BF16) {
                    return None;
                }
                let mw = l.device_weight(n_out, n_in)?.downcast_ref::<MetalWeight>()?;
                let affine = mw.offsets.is_some();
                let pd = be.mma_pipe(&wr, affine, n_out, n_in, super::Tile::Dense)?;
                let ps = be.mma_pipe(&wr, affine, n_out, n_in, super::Tile::Small)?;
                let pm = be.mma_pipe(&wr, affine, n_out, n_in, super::Tile::Mid)?;
                let (_, eb, sb) = super::format_index(&wr);
                let s_off = match &wr {
                    WeightRef::Gguf(m) => super::gguf_s_off(m, mw),
                    _ => mw.scales.as_ref().map_or(0, |s| s.offset_bytes / sb),
                };
                Some(Gemm {
                    pipes: [pd, ps, pm],
                    data: mw.data.id,
                    scales: mw.scales.as_ref().map_or(mw.data.id, |s| s.id),
                    w_off: (mw.data.offset_bytes / eb) as u32,
                    s_off: s_off as u32,
                    offs: mw
                        .offsets
                        .as_ref()
                        .map(|(ob, gpr, gk)| (ob.id, [(ob.offset_bytes / 2) as u32, *gpr, *gk, 0])),
                })
            })();
            let f32g = match l.as_ref() {
                WeightRef::Gguf(m) if m.kind == GgmlType::F32 && n_out.is_multiple_of(32) && n_in.is_multiple_of(8) => {
                    Some((w.data, (w.data_off / 4) as u32))
                }
                _ => None,
            };
            // The adapters' updates, copied (they are small and do not change).
            let mut lora = Vec::new();
            for x in l.lora_delta().into_iter().flat_map(|d| d.links()) {
                let (a, bt, r) = x.factors();
                let mk = |v: &[f32]| -> anyhow::Result<Id> {
                    let (id, p) = be.ink_buffer(v.len().max(1) * 4).context("LoRA buffer")?;
                    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, p, v.len() * 4) };
                    Ok(id)
                };
                let (ia, ib) = (mk(a)?, mk(bt)?);
                lora_bufs.borrow_mut().extend([ia, ib]);
                lora.push(LoraDev {
                    a: ia,
                    bt: ib,
                    r,
                    n_out,
                    n_in,
                    d: x as *const _,
                });
                rmax.set(rmax.get().max(r));
            }
            Ok(Dw {
                w,
                lora,
                one,
                multi: [mk(8)?, mk(16)?],
                g,
                f32g,
            })
        };
        let (kn, dr) = (c.conv_kernel_size, c.d_rel);
        // (`ink_attv` loads the relative vectors' up to 16 values at once.)
        if dr > 16 {
            bail!("relative vectors of {dr} values");
        }
        let rank = c.loop_cfg.map_or(1, |l| l.gate_rank);
        let ne = c.n_routed_experts;
        let (topk, mi) = (c.num_experts_per_tok, c.moe_intermediate_size);
        if ne > 1024 || topk > 32 || !mi.is_multiple_of(32) {
            bail!("{ne} experts of {mi}, {topk} per token");
        }
        // The experts' tables: per expert, the gate/up and down projections.
        let wrap = |p: *const u8, len: usize, shared: &mut Vec<Id>, indirect: &mut Vec<Id>| -> anyhow::Result<u64> {
            let (id, addr) = be.wrap_shared(p, len).context("wrap")?;
            shared.push(id);
            indirect.push(id);
            Ok(addr)
        };
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        let (mut max_w, mut max_nh, mut max_kw, mut max_nkv) = (0, 0, 0, 0);
        for (li, l) in m.layers.iter().enumerate() {
            let a = &l.attn;
            let hd = a.hd;
            if hd.head_dim != 128 || hd.heads % hd.kv_heads != 0 || !(1..=4).contains(&(hd.heads / hd.kv_heads)) || a.extent > 1024 {
                bail!("layer {li}: {hd:?}, extent {}", a.extent);
            }
            let (w, kw) = (hd.heads * hd.head_dim, hd.kv_heads * hd.head_dim);
            (max_w, max_nh, max_kw, max_nkv) = (max_w.max(w), max_nh.max(hd.heads), max_kw.max(kw), max_nkv.max(hd.kv_heads));
            let (k_taps, kd) = match &a.k_conv {
                KvConv::Static(t) => (consts.add(t), None),
                KvConv::Dynamic { taps, proj } => (consts.add(taps), Some(dense(proj, hd.kv_heads * kn, h, &mut shared)?)),
            };
            let (v_taps, vd) = match &a.v_conv {
                KvConv::Static(t) => (consts.add(t), None),
                KvConv::Dynamic { taps, proj } => (consts.add(taps), Some(dense(proj, hd.kv_heads * kn, h, &mut shared)?)),
            };
            let ln_in = match &l.ln_in {
                InNorm::One(w) => (consts.add(w), 1),
                InNorm::Qkv(a, b, cc) => (consts.add(&[&a[..], &b[..], &cc[..]].concat()), 3),
            };
            let gates = match &l.looped {
                Some(lp) => {
                    let g = |i: usize, shared: &mut Vec<Id>| -> anyhow::Result<GateDev> {
                        Ok(GateDev {
                            down: dense(&lp.gates[i].down, rank, 2 * h, shared)?,
                            forget: dense(&lp.gates[i].forget, h, rank, shared)?,
                            gain: dense(&lp.gates[i].gain, h, rank, shared)?,
                        })
                    };
                    Some((
                        [g(0, &mut shared)?, g(1, &mut shared)?, g(2, &mut shared)?, g(3, &mut shared)?],
                        consts.add(&lp.query),
                    ))
                }
                None => None,
            };
            // Experts: one format per projection kind.
            let ex = &l.experts;
            let (mut gu_fmt, mut d_fmt) = (None::<String>, None::<String>);
            let mut tab_gu = vec![XEntry::default(); ne];
            let mut tab_d = vec![XEntry::default(); ne];
            for e in 0..ne {
                let gl = match &ex.gate_up[e] {
                    crate::moe::GateUp::Fused(l) => l,
                    crate::moe::GateUp::Split(..) => bail!("split expert projections"),
                };
                for (lin, n_out, n_in, tab, fmt) in [
                    (gl, 2 * mi, h, &mut tab_gu, &mut gu_fmt),
                    (&ex.down[e], h, mi, &mut tab_d, &mut d_fmt),
                ] {
                    let xw = expert_weight(&lin.as_ref(), n_out, n_in).with_context(|| format!("layer {li}: an expert format"))?;
                    if fmt.as_ref().is_some_and(|f| *f != xw.name) {
                        bail!("layer {li}: experts in several formats");
                    }
                    *fmt = Some(xw.name.clone());
                    let q = wrap(xw.data.0, xw.data.1, &mut shared, &mut indirect)?;
                    let s = match xw.scales {
                        Some((p, n)) => wrap(p, n, &mut shared, &mut indirect)?,
                        None => q,
                    };
                    tab[e] = XEntry {
                        q,
                        s,
                        factor: xw.factor,
                        rb: xw.rb,
                        _p: 0,
                    };
                }
            }
            let mk_tab = |t: &[XEntry], owned: &mut Vec<Id>| -> anyhow::Result<Id> {
                let bytes = std::mem::size_of_val(t);
                let (id, p) = be.ink_buffer(bytes).context("table buffer")?;
                unsafe { std::ptr::copy_nonoverlapping(t.as_ptr() as *const u8, p, bytes) };
                owned.push(id);
                Ok(id)
            };
            let (gu_fmt, d_fmt) = (gu_fmt.unwrap(), d_fmt.unwrap());
            // The experts' LoRA updates of one projection kind (one per expert at most).
            let xlora_of = |lins: Vec<&Lin>, owned: &mut Vec<Id>, indirect: &mut Vec<Id>| -> anyhow::Result<Option<XLora>> {
                if !lins.iter().any(|l| l.has_lora()) {
                    return Ok(None);
                }
                let mut data: Vec<f32> = Vec::new();
                let mut at = Vec::with_capacity(ne);
                let mut deltas = Vec::with_capacity(ne);
                let mut rx = 0usize;
                for l in lins {
                    match l.lora_delta() {
                        Some(d) => {
                            if d.links().nth(1).is_some() {
                                bail!("layer {li}: chained expert LoRAs");
                            }
                            let (a, bt, r) = d.factors();
                            let o = data.len();
                            data.extend_from_slice(a);
                            data.resize(data.len().next_multiple_of(4), 0.0);
                            let ob = data.len();
                            data.extend_from_slice(bt);
                            data.resize(data.len().next_multiple_of(4), 0.0);
                            at.push((o, ob, r));
                            deltas.push(Some(d as *const crate::lora::LoraDelta));
                            rx = rx.max(r);
                        }
                        None => {
                            at.push((0, 0, 0));
                            deltas.push(None);
                        }
                    }
                }
                let (db, dp) = be.ink_buffer(data.len().max(1) * 4).context("LoRA buffer")?;
                unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, dp, data.len() * 4) };
                owned.push(db);
                indirect.push(db);
                let base = be.buffer_gpu_address(db);
                let ents: Vec<XlEntry> = at
                    .iter()
                    .map(|&(o, ob, r)| XlEntry {
                        a: base + (o * 4) as u64,
                        bt: base + (ob * 4) as u64,
                        r: r as u32,
                        s: 0.0,
                        _p: 0,
                    })
                    .collect();
                let bytes = std::mem::size_of_val(&ents[..]);
                let (tab, tab_p) = be.ink_buffer(bytes).context("LoRA table")?;
                unsafe { std::ptr::copy_nonoverlapping(ents.as_ptr() as *const u8, tab_p, bytes) };
                owned.push(tab);
                rmax.set(rmax.get().max(rx));
                Ok(Some(XLora {
                    tab,
                    tab_p,
                    deltas,
                    rmax: rx,
                }))
            };
            let xlora = xlora_of(ex.down.iter().collect(), &mut owned, &mut indirect)?;
            let gu_lins: Vec<&Lin> = ex
                .gate_up
                .iter()
                .map(|g| match g {
                    crate::moe::GateUp::Fused(l) => l,
                    crate::moe::GateUp::Split(l, _) => l,
                })
                .collect();
            let xlora_gu = xlora_of(gu_lins, &mut owned, &mut indirect)?;
            // Prefill: the grouped GEMMs, where both formats have them.
            let mmag = |f: &str| -> Option<[Id; 3]> {
                let n = match f {
                    "bf" | "q8" | "nvfp4" | "mxf8" | "mxf8b" => format!("mmag_{f}"),
                    "F32" => return None,
                    k => format!("mmag_{k}"),
                };
                Some([
                    lib.pipeline(be, &n)?,
                    lib.pipeline(be, &format!("{n}_l"))?,
                    lib.pipeline(be, &format!("{n}_s"))?,
                ])
            };
            let grouped = match (lib.mma && xlora_gu.is_none(), mmag(&gu_fmt), mmag(&d_fmt)) {
                (true, Some(gu), Some(d)) => {
                    let ep = |t: &[XEntry], owned: &mut Vec<Id>| -> anyhow::Result<Id> {
                        let v: Vec<[u64; 2]> = t.iter().map(|x| [x.q, x.s]).collect();
                        let bytes = v.len() * 16;
                        let (id, p) = be.ink_buffer(bytes).context("table buffer")?;
                        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, p, bytes) };
                        owned.push(id);
                        Ok(id)
                    };
                    Some(Grouped {
                        gu,
                        d,
                        ep_gu: ep(&tab_gu, &mut owned)?,
                        ep_d: ep(&tab_d, &mut owned)?,
                        rb: (tab_gu[0].rb, tab_d[0].rb),
                        f_gu: tab_gu.iter().map(|x| x.factor).collect(),
                        f_d: tab_d.iter().map(|x| x.factor).collect(),
                    })
                }
                _ => None,
            };
            let exp = ExpDev {
                pair: pipe(&format!("ink_x_{gu_fmt}_pair"))?,
                sum: pipe(&format!("ink_x_{d_fmt}_sum"))?,
                gpair: pipe(&format!("ink_x_{gu_fmt}_gpair"))?,
                gsum: pipe(&format!("ink_x_{d_fmt}_gsum"))?,
                tab_gu: mk_tab(&tab_gu, &mut owned)?,
                tab_d: mk_tab(&tab_d, &mut owned)?,
                il: if ex.interleaved { 2 } else { 1 },
                off: if ex.interleaved { 1 } else { mi as u32 },
                lpr: (lanes_per_row(h, ch_of(&gu_fmt), 16), lanes_per_row(mi, ch_of(&d_fmt), 32)),
                grouped,
                xlora,
                xlora_gu,
            };
            layers.push(LayerDev {
                sliding: c.sliding[li],
                boundary: l.boundary,
                nh: hd.heads,
                nkv: hd.kv_heads,
                hd: hd.head_dim,
                window: a.window.unwrap_or(0),
                extent: a.extent,
                attn_query: l.attn_query.as_ref().map(|q| (consts.add(q), q.len() / h)),
                mlp_query: consts.add(&l.mlp_query),
                ln_in,
                ln_post: consts.add(&l.ln_post),
                q_scale: consts.add(&a.q_scale),
                r_scale: consts.add(&a.r_scale),
                rel: consts.add(&a.rel),
                k_taps,
                v_taps,
                attn_conv: consts.add(&l.attn_conv),
                mlp_conv: consts.add(&l.mlp_conv),
                router: consts.add(&l.router),
                bias: consts.add(&l.bias),
                q: dense(&a.q, w, h, &mut shared)?,
                r: dense(&a.r, hd.heads * dr, h, &mut shared)?,
                gate: dense(&a.gate, w, h, &mut shared)?,
                sink: dense(&a.sink, hd.heads, h, &mut shared)?,
                k: dense(&a.k, kw, h, &mut shared)?,
                v: dense(&a.v, kw, h, &mut shared)?,
                o: dense(&a.o, h, w, &mut shared)?,
                kd,
                vd,
                gates,
                exp,
            });
        }
        let lm_head = dense(&m.lm_head, c.vocab_size, h, &mut shared)?;
        owned.extend(lora_bufs.borrow_mut().drain(..));
        let lora_r = rmax.get();
        let final_query = consts.add(&m.final_query);
        let final_norm = consts.add(&m.final_norm);
        // The step plan (as `forward_capture`'s bookkeeping) and the block-state slots.
        let steps = c.steps();
        let mut plan = Vec::with_capacity(steps.len());
        let mut states: Vec<usize> = Vec::new();
        let mut pushes = 0usize;
        let mut pre_n = 0usize;
        let mut push_slots: Vec<usize> = Vec::new();
        for st in &steps {
            let l = &m.layers[st.layer];
            let mut sp = StepPlan {
                layer: st.layer,
                slot: st.slot,

                ..Default::default()
            };
            if let Some(lc) = c.loop_cfg
                && st.layer == lc.start
            {
                if st.pass == 0 {
                    sp.save = true;
                    pre_n = states.len();
                } else {
                    sp.swap = true;
                    let prev = states.split_off(pre_n + 1);
                    states.truncate(pre_n);
                    // (Bank states are push slots; the bank stream is `bankx`.)
                    push_slots = prev;
                }
            }
            if st.pass > 0 && l.looped.is_some() {
                sp.bank = Some(push_slots.clone());
            }
            sp.attn_states = states.clone();
            if l.boundary {
                sp.push = Some(pushes);
                states.push(pushes);
                pushes += 1;
            }
            sp.mlp_states = states.clone();
            plan.push(sp);
        }
        if states.len() > 16 || plan.iter().any(|p| p.bank.as_ref().is_some_and(|b| b.len() > 15)) {
            bail!("{} block states", states.len());
        }
        let final_states = states.clone();
        // The arenas: passes of up to MAX_T tokens, and prefill passes of up to PF_T.
        let make = |t: usize, nf: usize, pf: bool| -> anyhow::Result<Arena> {
            let mut off = 0usize;
            let mut take = |n: usize| -> usize {
                let o = off;
                off += n.next_multiple_of(64);
                o
            };
            let th = t * h;
            let maxsplit = if pf { PF_SPLIT } else { MAX_SPLIT };
            let b = Bufs {
                tm: t,
                states: take(pushes * th),
                nf,
                pf: pf.then(|| {
                    let max_rows = (t * topk + 15 * ne).next_multiple_of(128);
                    PfBufs {
                        xh: take(t * 2 * h.max(max_w).max(mi) / 2 + 64),
                        rs: take(t),
                        kt: take(t * max_kw),
                        vt: take(t * max_kw),
                        kts: take(t * max_kw / 32),
                        vts: take(t * max_kw / 32),
                        max_rows,
                        rexp: take(max_rows),
                        tiles: take(4 * (max_rows / 16 + ne)),
                        src: take(max_rows),
                        map: take(t * topk),
                        f_gu: take(max_rows),
                        f_d: take(max_rows),
                        xe: take(max_rows * h / 2),
                        rse: take(max_rows),
                        gu: take(max_rows * 2 * mi),
                        ah: take(max_rows * mi / 2),
                        rs2: take(max_rows),
                        od: take(max_rows * h),
                    }
                }),
                x: take(th),
                xin: take(th),
                bankx: take(th),
                msgs: take(4 * th),
                mixed: take(3 * th),
                xn: take(3 * th),
                gcat: take(4 * 2 * th),
                gd: take(4 * t * rank),
                gf: take(4 * th),
                gg: take(4 * th),
                q: take(t * max_w),
                r: take(t * max_nh * dr),
                gate: take(t * max_w),
                sink: take(t * max_nh),
                k: take(t * max_kw),
                v: take(t * max_kw),
                kd: take(t * max_nkv * kn),
                vd: take(t * max_nkv * kn),
                att: take(t * max_w),
                ab: take(th),
                hn: take(th),
                hm: take(th),
                rlog: take(t * ne),
                sel: take(t * 2 * topk),
                act: take(t * topk * mi),
                o: take(th),
                part: take(t * max_nh * maxsplit * (128 + 2)),
                hidden: take(th),
                // (Prefill: the logits of the last tokens only.)
                logits: take(t.min(MAX_T) * c.vocab_size),
                capped: take(t.min(MAX_T) * c.unpadded_vocab_size),
                cap: take(t * nf * h),
                lu: take((t * topk + 15 * ne + 128) * lora_r),
                grp: take(1 + 3 * MAX_T * topk),
                asg: take(MAX_T * topk),
                od: take(MAX_T * topk * h),
                lora_r,
                h,
                maxsplit,
            };
            let total = off;
            if total >= u32::MAX as usize {
                bail!("arena of {total} floats");
            }
            let (id, _) = be.ink_buffer(total * 4).context("arena")?;
            Ok(Arena { id, b })
        };
        if consts.0.len() >= u32::MAX as usize {
            bail!("{} floats of constants", consts.0.len());
        }
        let dec = make(MAX_T, steps.len() + 1, false)?;
        owned.push(dec.id);
        let pfa = match lib.mma && !pf_off() {
            true => {
                let a = make(PF_T, PF_CAPTURE, true)?;
                owned.push(a.id);
                Some(a)
            }
            false => None,
        };
        let (cbuf, cp) = be.ink_buffer(consts.0.len() * 4).context("constants")?;
        unsafe { std::ptr::copy_nonoverlapping(consts.0.as_ptr() as *const u8, cp, consts.0.len() * 4) };
        owned.push(cbuf);
        if let Some(r) = &be.residency {
            r.update(&indirect, true);
        }
        let p = Pipes {
            mix: pipe("ink_mix")?,
            norm: pipe("ink_norm")?,
            copy: pipe("ink_copy")?,
            kv: [pipe("ink_kv_0")?, pipe("ink_kv_1")?, pipe("ink_kv_2")?],
            q: pipe("ink_q")?,
            attn: {
                let g = m.layers[0].attn.hd.heads / m.layers[0].attn.hd.kv_heads;
                if m.layers.iter().any(|l| l.attn.hd.heads / l.attn.hd.kv_heads != g) {
                    bail!("layers with different groups of query heads");
                }
                [
                    pipe(&format!("ink_attn_4_{g}_0"))?,
                    pipe(&format!("ink_attn_4_{g}_1"))?,
                    pipe(&format!("ink_attn_4_{g}_2"))?,
                ]
            },
            attv: {
                let g = m.layers[0].attn.hd.heads / m.layers[0].attn.hd.kv_heads;
                if m.layers.iter().any(|l| l.attn.hd.heads / l.attn.hd.kv_heads != g) {
                    bail!("layers with different groups of query heads");
                }
                [
                    pipe(&format!("ink_attv_4_{g}_0"))?,
                    pipe(&format!("ink_attv_4_{g}_1"))?,
                    pipe(&format!("ink_attv_4_{g}_2"))?,
                ]
            },
            merge: pipe("ink_attn_merge")?,
            conv: pipe("ink_conv")?,
            route: pipe("ink_route")?,
            mvf: pipe("ink_mvf")?,
            softcap: pipe("ink_softcap")?,
            xgroup: pipe("ink_xgroup")?,
            mix8: pipe("ink_mix8")?,
            gmix_norm: pipe("ink_gmix_norm")?,
            gfg: pipe("ink_gfg")?,
            mvr: pipe("ink_mvr")?,
            xcombine: pipe("ink_xcombine")?,
            lora_a: pipe("ink_lora_a")?,
            lora_b: pipe("ink_lora_b")?,
            xlora_a: pipe("ink_xlora_a")?,
            xlora_b: pipe("ink_xlora_b")?,
        };
        let pfp = match pfa {
            Some(_) => Some(PfPipes {
                x_half: be.named_pipeline("x_half").context("x_half")?,
                gemm_f32: be.named_pipeline("gemm_f32_s").context("gemm_f32_s")?,
                ring_store: pipe("ink_ring_store")?,
                kv_ring: pipe("ink_kv_ring")?,
                situ_half: pipe("ink_situ_half")?,
                combine: pipe("ink_combine")?,
            }),
            None => None,
        };
        Ok(InkMetal {
            be,
            f: Fns::new(be),
            dec,
            pfa,
            consts: cbuf,
            p,
            pfp,
            layers,
            plan,
            final_states,
            final_query,
            final_norm,
            lm_head,
            indirect,
            owned,
            shared,
            h,
            vocab: c.vocab_size,
            unpadded: c.unpadded_vocab_size,
            kn,
            dr,
            rank,
            ne,
            topk,
            inter: mi,
            eps: c.rms_norm_eps as f32,
            route_scale: c.route_scale as f32,
            caps: (c.mlp_gate_softcap as f32, c.mlp_up_softcap as f32, c.final_logit_softcapping as f32),
            clips: (c.qk_l2_clip_threshold as f32, c.rel_l2_clip_threshold as f32),
            log_scale: (c.log_scaling_alpha as f32, c.log_scaling_n_floor as f32),
            ring_rows: kn - 1 + ROLLBACK,
            window: c.sliding_window_size,
            lock: std::sync::Mutex::new(()),
        })
    }

    /// The session's state on the device, from its host state.
    fn upload(&self, m: &Inkling2Model, sess: &mut Inkling2Session) -> bool {
        let c = &m.cfg;
        let be = self.be;
        let pos = sess.pos;
        let slot_layers = c.slot_layers();
        let mut slots = Vec::with_capacity(slot_layers.len());
        for (st, &l) in sess.layers.iter().zip(&slot_layers) {
            let hd = c.heads(l);
            let kw = hd.kv_heads * hd.head_dim;
            let fmt = st.k.mode;
            let ring = if c.sliding[l] { self.window + ROLLBACK } else { 0 };
            let rows = if ring != 0 {
                ring
            } else {
                (pos + MAX_T).next_power_of_two().max(1024)
            };
            let rb = row_bytes(fmt, kw);
            let sb = if fmt == KvCache::Q8 { kw / 32 * 4 } else { 4 };
            let mk = |n: usize| be.ink_buffer(n);
            let (Some(k), Some(v), Some(ks), Some(vs)) = (mk(rows * rb), mk(rows * rb), mk(rows * sb), mk(rows * sb)) else {
                return false;
            };
            let mut rings = [(std::ptr::null_mut(), std::ptr::null_mut(), 0usize); 4];
            for (r, hr) in rings.iter_mut().zip(&st.hist) {
                let Some((id, p)) = mk(hr.rows.len() * 4) else { return false };
                unsafe { std::ptr::copy_nonoverlapping(hr.rows.as_ptr() as *const u8, p, hr.rows.len() * 4) };
                *r = (id, p, hr.rows.len());
            }
            let sd = SlotDev {
                kw,
                fmt,
                ring,
                rows,
                k,
                v,
                ks,
                vs,
                rings,
            };
            // The host rows of positions `first..pos` that the device keeps.
            let lo = if ring != 0 {
                pos.saturating_sub(ring).max(st.first)
            } else {
                st.first
            };
            for p in lo..pos {
                let hr = (p - st.first) * kw;
                let r = if ring != 0 { p % ring } else { p };
                for (src, dst, dsc) in [(&st.k, sd.k.1, sd.ks.1), (&st.v, sd.v.1, sd.vs.1)] {
                    unsafe {
                        let at = dst.add(r * rb);
                        match fmt {
                            KvCache::F32 => std::ptr::copy_nonoverlapping(src.f[hr..].as_ptr() as *const u8, at, rb),
                            KvCache::F16 => std::ptr::copy_nonoverlapping(src.h[hr..].as_ptr() as *const u8, at, rb),
                            KvCache::Q8 => {
                                std::ptr::copy_nonoverlapping(src.q[hr..].as_ptr() as *const u8, at, rb);
                                std::ptr::copy_nonoverlapping(
                                    src.s[hr / 32..].as_ptr() as *const u8,
                                    dsc.add(r * kw / 32 * 4),
                                    kw / 32 * 4,
                                );
                            }
                        }
                    }
                }
            }
            slots.push(sd);
        }
        sess.dev = Some(DevBox(Box::new(InkDev { slots, be, pos })));
        true
    }

    // ---- Encoding --------------------------------------------------------------------------

    /// Measurements only (`OXINFER_INKLING_SKIP=dense,attn,exp,mix,gates,conv,route,head`, and
    /// within those `qkv,attk,merge,mvf,rt,norm`: the query / key stage, the attention kernel, its
    /// merge, the router's matvec, the routing, the plain norms): the kinds of ops left out of the passes (whose results are then
    /// meaningless).
    fn skip(&self, what: u32) -> bool {
        static SKIP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let m = *SKIP.get_or_init(|| {
            let v = std::env::var("OXINFER_INKLING_SKIP").unwrap_or_default();
            let names = [
                "dense", "attn", "exp", "mix", "gates", "conv", "route", "head", "qkv", "attk", "merge", "", "mvf", "rt", "norm",
            ];
            names
                .iter()
                .enumerate()
                .filter(|(_, n)| v.split(',').any(|x| x == **n))
                .map(|(i, _)| 1 << i)
                .sum()
        });
        m & what != 0
    }

    /// `y = W x` for `t` tokens: the one-token matvec, the exact multi-token ones (up to 16), and
    /// in prefill passes the matrix-unit GEMM (or the multi-token matvecs 16 tokens at a time).
    #[allow(clippy::too_many_arguments)]
    fn mv(&self, e: &Pe, c: &Cx, d: &Dw, x: usize, y: usize, t: usize, mode: MvMode) {
        // The LoRA updates beside it: `u = A x` in the matvec's own stage (both read x only), then
        // `y += s Bt u` once the matvec wrote y. (Each update has its `u` rows in the scratch, so
        // that the `A x` of all of them run with the matvec.)
        // SAFETY: the updates live with the model, which outlives its engine.
        let active: Vec<(&LoraDev, f32)> = d
            .lora
            .iter()
            .map(|l| (l, unsafe { (*l.d).strength() }))
            .filter(|(l, s)| *s != 0.0 && l.r != 0)
            .collect();
        let mut u = c.b.lu;
        let mut us = Vec::with_capacity(active.len());
        for (l, _) in &active {
            #[repr(C)]
            struct Pa {
                n_in: u32,
                r: u32,
                x: u32,
                u: u32,
            }
            e.pipe(self.p.lora_a);
            e.buf(c.a, 0, 0);
            e.buf(l.a, 0, 1);
            e.bytes(
                &Pa {
                    n_in: l.n_in as u32,
                    r: l.r as u32,
                    x: x as u32,
                    u: u as u32,
                },
                2,
            );
            e.dispatch((l.r, t), 32);
            us.push(u);
            u += t * l.r;
        }
        self.mv_base(e, c, d, x, y, t, mode);
        for ((l, s), u) in active.iter().zip(us) {
            #[repr(C)]
            struct Pb {
                n_out: u32,
                r: u32,
                u: u32,
                y: u32,
                s: f32,
                _p: [u32; 3],
            }
            e.barrier();
            e.pipe(self.p.lora_b);
            e.buf(c.a, 0, 0);
            e.buf(l.bt, 0, 1);
            e.bytes(
                &Pb {
                    n_out: l.n_out as u32,
                    r: l.r as u32,
                    u: u as u32,
                    y: y as u32,
                    s: *s,
                    _p: [0; 3],
                },
                2,
            );
            e.dispatch((l.n_out.div_ceil(256), t), 256);
        }
        if !active.is_empty() {
            e.barrier();
        }
    }

    /// [`Self::mv`] without the LoRA updates.
    #[allow(clippy::too_many_arguments)]
    fn mv_base(&self, e: &Pe, c: &Cx, d: &Dw, x: usize, y: usize, t: usize, mode: MvMode) {
        let a = c.a;
        if t == 1 {
            let (pipe, nr, ks) = d.one;
            // (The profile's label: the matvec's shape.)
            e.note(((d.w.n_out << 44) | (d.w.n_in << 24) | (pipe as usize & 0xff_ffff)) as Id);
            encode_mv(&e.fe(), pipe, nr, ks, &d.w, None, (a, x * 4), (a, y * 4), mode, (a, 0));
            return;
        }
        if t > MAX_T {
            if let (Some(g), Some(pf), Some(pp)) = (&d.g, &c.b.pf, &self.pfp) {
                self.gemm(e, c, g, pf, pp, d.w.n_out, d.w.n_in, x, y, t, mode == MvMode::Add);
                return;
            }
            if let (Some((buf, off)), Some(pp), MvMode::Store) = (d.f32g, &self.pfp, mode) {
                // (Rows padded to 32 tokens: the arena's rows beyond `t` are scratch.)
                e.pipe(pp.gemm_f32);
                e.buf(buf, 0, 0);
                e.buf(buf, 0, 1);
                e.buf(c.a, x * 4, 2);
                e.buf(c.a, y * 4, 3);
                e.bytes(&[d.w.n_out as u32, d.w.n_in as u32, off, 0], 4);
                e.dispatch((d.w.n_out / 32, t.div_ceil(32)), 128);
                return;
            }
            for t0 in (0..t).step_by(MAX_T) {
                let n = (t - t0).min(MAX_T);
                self.mv_base(e, c, d, x + t0 * d.w.n_in, y + t0 * d.w.n_out, n, mode);
            }
            return;
        }
        let (pipe, nr, threads) = d.multi[usize::from(t > 8)];
        let w = &d.w;
        let p = MvkP {
            m: MvParams {
                n_out: w.n_out as u32,
                n_in: w.n_in as u32,
                mode: mode as u32,
                rows_per_tg: nr as u32,
                w_off: w.data_off,
                s_off: w.scales_off,
                w2_off: w.data_off,
                s2_off: w.scales_off,
                o_off: w.offs_off,
                o2_off: w.offs_off,
                gpr: w.gpr,
                gk: w.gk,
                ldy: 0,
                _p: 0,
            },
            ntok: t as u32,
            sx: w.n_in as u32,
            sy: w.n_out as u32,
            sgate: 0,
        };
        e.pipe(pipe);
        e.buf(w.data, 0, 0);
        e.buf(w.scales, 0, 1);
        e.buf(a, x * 4, 2);
        e.buf(a, y * 4, 3);
        e.bytes(&p, 4);
        e.buf(w.data, 0, 5);
        e.buf(w.scales, 0, 6);
        e.buf(w.offs, 0, 7);
        e.buf(w.offs, 0, 8);
        e.buf(a, 0, 9);
        e.dispatch((w.n_out.div_ceil(nr), 1), threads);
    }

    /// Rows of `x` (`n` floats each) to the GEMM scratch as scaled half rows (`rows` of them,
    /// the tile padding included).
    #[allow(clippy::too_many_arguments)]
    fn x_half(&self, e: &Pe, c: &Cx, pf: &PfBufs, pp: &PfPipes, x: usize, n: usize, rows: usize) {
        #[repr(C)]
        struct Xh {
            n: u32,
            gather: u32,
            stride: u32,
            _p: u32,
        }
        e.pipe(pp.x_half);
        e.buf(c.a, x * 4, 0);
        e.buf(c.a, pf.xh * 4, 1);
        e.buf(c.a, pf.rs * 4, 2);
        e.bytes(
            &Xh {
                n: n as u32,
                gather: 0,
                stride: n as u32,
                _p: 0,
            },
            3,
        );
        e.buf(c.a, 0, 4);
        e.dispatch((rows, 1), 256);
    }

    /// `y = W x` (`+=` with `add`) for `t` tokens on the matrix units.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        e: &Pe,
        c: &Cx,
        g: &Gemm,
        pf: &PfBufs,
        pp: &PfPipes,
        n_out: usize,
        n_in: usize,
        x: usize,
        y: usize,
        t: usize,
        add: bool,
    ) {
        // Few 128-token tiles (narrow outputs): 64- or 32-token ones keep more of the GPU busy.
        let few = |tm: usize| (n_out / g.pipes[0].1) * t.div_ceil(tm) < 160;
        let (tile, tm) = match std::env::var("OXINFER_INKLING_TILE").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(0) => (0, 128),
            Some(2) => (2, 64),
            Some(_) => (1, 32),
            None if !few(128) => (0, 128),
            None if !few(64) => (2, 64),
            None => (1, 32),
        };
        let rows = t.next_multiple_of(tm);
        self.x_half(e, c, pf, pp, x, n_in, rows);
        e.barrier();
        let (pipe, tn) = g.pipes[tile];
        e.pipe(pipe);
        e.buf(g.data, 0, 0);
        e.buf(g.scales, 0, 1);
        e.buf(c.a, pf.xh * 4, 2);
        e.buf(c.a, y * 4, 3);
        e.bytes(
            &super::MmaParams {
                n_out: n_out as u32,
                n_in: n_in as u32,
                w_off: g.w_off,
                s_off: g.s_off,
                rows: rows as u32,
                _p: [0, 0, add as u32],
            },
            4,
        );
        e.buf(c.a, pf.rs * 4, 5);
        if let Some((ob, op)) = &g.offs {
            e.buf(*ob, 0, 6);
            e.bytes(op, 7);
        }
        e.dispatch((n_out / tn, rows / tm), 128);
    }

    fn std(&self, e: &Pe, c: &Cx, pipe: Id) {
        e.pipe(pipe);
        e.buf(c.a, 0, 0);
        e.buf(self.consts, 0, 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn mix(&self, e: &Pe, c: &Cx, states: &[usize], x: usize, q: usize, nq: usize, out: usize, norm: Option<usize>, t: usize) {
        self.mix_cat(e, c, states, x, q, nq, out, norm, t, None);
    }

    /// [`Self::mix`], with `cat` (gate inputs `i0 ..`, the last query into `n` of them) also the
    /// output normed without a weight as the message gates' first stage takes it ([`Self::gate_cat`]).
    #[allow(clippy::too_many_arguments)]
    fn mix_cat(
        &self,
        e: &Pe,
        c: &Cx,
        states: &[usize],
        x: usize,
        q: usize,
        nq: usize,
        out: usize,
        norm: Option<usize>,
        t: usize,
        cat: Option<(usize, bool, usize)>,
    ) {
        let mut p = MixP {
            h: self.h as u32,
            ns: states.len() as u32,
            norm: norm.is_some() as u32,
            x: x as u32,
            q: q as u32,
            out: out as u32,
            out_qs: (c.b.tm * self.h) as u32,
            nw: norm.unwrap_or(0) as u32,
            scale: (self.h as f32).powf(-0.5),
            eps: self.eps,
            mix_eps: MIX_EPS,
            ..Default::default()
        };
        for (d, &s) in p.s.iter_mut().zip(states) {
            *d = c.state(s) as u32;
        }
        if let Some(cat) = cat {
            self.gate_cat(c, &mut p, cat);
        }
        // (At most 8 block states: fewer registers.)
        self.std(e, c, if states.len() <= 8 { self.p.mix8 } else { self.p.mix });
        e.bytes(&p, 2);
        e.dispatch((nq, t), 1024);
    }

    /// Passes of up to [`MAX_T`] tokens: the router's logits spread over the GPU (`ink_mvr`, a
    /// threadgroup per expert) rather than a SIMD-group per expert. `OXINFER_INKLING_NARROW=1`
    /// (measurements): `ink_mvf`.
    fn wide(&self, t: usize) -> bool {
        static NARROW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        t <= MAX_T && !*NARROW.get_or_init(|| std::env::var("OXINFER_INKLING_NARROW").is_ok_and(|v| v == "1"))
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(&self, e: &Pe, c: &Cx, x: usize, out: usize, w: Option<usize>, eps: f32, xs: usize, os: usize, t: usize) {
        let p = NormP {
            h: self.h as u32,
            eps,
            x: x as u32,
            out: out as u32,
            w: w.unwrap_or(0) as u32,
            has_w: w.is_some() as u32,
            xs: xs as u32,
            os: os as u32,
        };
        self.std(e, c, self.p.norm);
        e.bytes(&p, 2);
        if !self.skip(SK_NORM) {
            e.dispatch((t, 1), 256);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn copy(&self, e: &Pe, c: &Cx, src: usize, dst: usize, n: usize, ds: usize, t: usize) {
        let p = CopyP {
            n: n as u32,
            src: src as u32,
            dst: dst as u32,
            ds: ds as u32,
        };
        e.pipe(self.p.copy);
        e.buf(c.a, 0, 0);
        e.bytes(&p, 1);
        e.dispatch((n.div_ceil(256), t), 256);
    }

    /// Message gates `which` (gate index, input, norm weights, output) with messages `msgs + i`:
    /// the gated inputs RMS-normed into the outputs.
    fn gates(&self, e: &Pe, c: &Cx, g: &[GateDev; 4], which: &[(usize, usize, usize, usize)], t: usize, normed: bool) {
        let (h, th) = (self.h, c.b.tm * self.h);
        let b = c.b;
        // (`normed`: the mixes that made the inputs wrote them normed, `mix_cat`.)
        if !normed {
            for &(i, src, _, _) in which {
                let cat = b.gcat + i * 2 * th;
                self.norm(e, c, src, cat, None, self.eps, h, 2 * h, t);
                self.norm(e, c, b.msgs + i * th, cat + h, None, self.eps, h, 2 * h, t);
            }
            e.barrier();
        }
        for &(i, ..) in which {
            self.mv(
                e,
                c,
                &g[i].down,
                b.gcat + i * 2 * th,
                b.gd + i * c.b.tm * self.rank,
                t,
                MvMode::Store,
            );
            if c.b.pf.is_some() {
                e.barrier();
            }
        }
        e.barrier();
        // Float32 or bf16 forget / gain rows of the rank: both matvecs and the mix per channel in
        // one kernel (`ink_gfg`), then the norm.
        let fast = |d: &Dw| -> Option<u32> {
            let el = match d.w.fmt {
                MvFmt::Gguf(GgmlType::F32) => 4,
                MvFmt::Bf16 => 2,
                _ => return None,
            };
            (d.lora.is_empty() && self.rank.is_multiple_of(16) && d.w.gpr as usize == self.rank * el).then_some((el == 2) as u32)
        };
        if let Some(bf) = which
            .iter()
            .map(|&(i, ..)| fast(&g[i].forget).zip(fast(&g[i].gain)))
            .try_fold(None, |acc: Option<u32>, f| {
                let (a1, a2) = f?;
                (a1 == a2 && acc.is_none_or(|x| x == a1)).then_some(Some(a1))
            })
            .flatten()
        {
            for &(i, src, _, _) in which {
                #[repr(C)]
                struct P {
                    h: u32,
                    rank: u32,
                    d: u32,
                    x: u32,
                    m: u32,
                    out: u32,
                    bf: u32,
                    _p: u32,
                }
                e.pipe(self.p.gfg);
                e.buf(c.a, 0, 0);
                e.buf(g[i].forget.w.data, g[i].forget.w.data_off as usize, 1);
                e.buf(g[i].gain.w.data, g[i].gain.w.data_off as usize, 2);
                e.bytes(
                    &P {
                        h: h as u32,
                        rank: self.rank as u32,
                        d: (b.gd + i * c.b.tm * self.rank) as u32,
                        x: src as u32,
                        m: (b.msgs + i * th) as u32,
                        out: (b.gf + i * th) as u32,
                        bf,
                        _p: 0,
                    },
                    3,
                );
                e.dispatch((h.div_ceil(64), t), 256);
            }
            e.barrier();
            for &(i, _, w, out) in which {
                self.norm(e, c, b.gf + i * th, out, Some(w), self.eps, h, h, t);
            }
            e.barrier();
            return;
        }
        for &(i, ..) in which {
            let d = b.gd + i * c.b.tm * self.rank;
            self.mv(e, c, &g[i].forget, d, b.gf + i * th, t, MvMode::Store);
            if c.b.pf.is_some() {
                e.barrier();
            }
            self.mv(e, c, &g[i].gain, d, b.gg + i * th, t, MvMode::Store);
            if c.b.pf.is_some() {
                e.barrier();
            }
        }
        e.barrier();
        for &(i, src, w, out) in which {
            #[repr(C)]
            struct P {
                h: u32,
                x: u32,
                m: u32,
                f: u32,
                g: u32,
                nout: u32,
                w: u32,
                eps: f32,
            }
            self.std(e, c, self.p.gmix_norm);
            e.bytes(
                &P {
                    h: h as u32,
                    x: src as u32,
                    m: (b.msgs + i * th) as u32,
                    f: (b.gf + i * th) as u32,
                    g: (b.gg + i * th) as u32,
                    nout: out as u32,
                    w: w as u32,
                    eps: self.eps,
                },
                2,
            );
            e.dispatch((t, 1), 1024);
        }
        e.barrier();
    }

    /// The parameters of `ink_conv` (`od`: the input as the sum of the experts' slots there).
    #[allow(clippy::too_many_arguments)]
    fn conv_p(&self, x: usize, out: usize, taps: usize, pos: usize, add: bool, t: usize, od: Option<usize>) -> ConvP {
        ConvP {
            h: self.h as u32,
            kn: self.kn as u32,
            x: x as u32,
            out: out as u32,
            taps: taps as u32,
            pos: pos as u32,
            mode: add as u32,
            norm: 1.0 / ((self.kn + 1) as f32).sqrt(),
            ring_cap: self.ring_rows as u32,
            ntok: t as u32,
            ring_write: self.inline_rings(t) as u32,
            od: od.unwrap_or(0) as u32,
            ok: if od.is_some() { self.topk as u32 } else { 0 },
            ..Default::default()
        }
    }

    /// Points the mix `p` at the message gates' normed inputs: gate `i0` on (`msgs`: the second half
    /// of each gate's row, the messages' norm), its last query into gates up to 2 (the attention's
    /// three gates read the last of fewer queries).
    fn gate_cat(&self, c: &Cx, p: &mut MixP, (i0, msgs, last): (usize, bool, usize)) {
        let (h, th) = (self.h, c.b.tm * self.h);
        p.cat = (c.b.gcat + i0 * 2 * th + if msgs { h } else { 0 }) as u32;
        p.cat_qs = (2 * th) as u32;
        p.cat_ts = (2 * h) as u32;
        p.cat_last = last as u32;
        p.cat_n = if msgs { 1 } else { (3 - i0).saturating_sub(last).max(1) as u32 };
    }

    /// Whether a pass of `t` tokens writes its convolution inputs to the rings in the
    /// convolutions themselves (else a separate kernel does after, `ring_store`).
    fn inline_rings(&self, t: usize) -> bool {
        t + self.kn <= self.ring_rows
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, e: &Pe, c: &Cx, x: usize, out: usize, taps: usize, ring: Id, pos: usize, add: bool, t: usize) {
        let p = ConvP {
            h: self.h as u32,
            kn: self.kn as u32,
            x: x as u32,
            out: out as u32,
            taps: taps as u32,
            pos: pos as u32,
            mode: add as u32,
            norm: 1.0 / ((self.kn + 1) as f32).sqrt(),
            ring_cap: self.ring_rows as u32,
            ntok: t as u32,
            ring_write: self.inline_rings(t) as u32,
            ..Default::default()
        };
        self.std(e, c, self.p.conv);
        e.bytes(&p, 2);
        e.buf(ring, 0, 3);
        e.dispatch((self.h.div_ceil(256), t), 256);
    }

    /// The raw inputs `x` (`ch` floats per token) of the last tokens into a ring (prefill).
    #[allow(clippy::too_many_arguments)]
    fn ring_store(&self, e: &Pe, c: &Cx, x: usize, ch: usize, ring: Id, pos: usize, t: usize) {
        let pp = self.pfp.as_ref().expect("prefill kernels");
        #[repr(C)]
        struct P {
            ch: u32,
            x: u32,
            pos: u32,
            ntok: u32,
            ring_cap: u32,
            kn: u32,
            _p: [u32; 2],
        }
        e.pipe(pp.ring_store);
        e.buf(c.a, 0, 0);
        e.bytes(
            &P {
                ch: ch as u32,
                x: x as u32,
                pos: pos as u32,
                ntok: t as u32,
                ring_cap: self.ring_rows as u32,
                kn: self.kn as u32,
                _p: [0; 2],
            },
            1,
        );
        e.buf(ring, 0, 2);
        e.dispatch((ch.div_ceil(256), t), 256);
    }

    /// The final mix, and the LM head with its cap over the last `head_rows` tokens.
    fn enc_final(&self, e: &Pe, c: &Cx, t: usize, capture: &[usize], head_rows: usize) {
        let (h, b) = (self.h, c.b);
        if self.capture(e, c, self.plan.len(), t, capture) {
            e.barrier();
        }
        if !self.skip(SK_MIX) {
            self.mix(
                e,
                c,
                &self.final_states,
                b.x,
                self.final_query,
                1,
                b.hidden,
                Some(self.final_norm),
                t,
            );
        }
        if head_rows > 0 {
            e.barrier();
            if !self.skip(SK_HEAD) {
                self.mv(
                    e,
                    c,
                    &self.lm_head,
                    b.hidden + (t - head_rows) * h,
                    b.logits,
                    head_rows,
                    MvMode::Store,
                );
            }
            e.barrier();
            let p = CapP {
                v: self.vocab as u32,
                u: self.unpadded as u32,
                x: b.logits as u32,
                out: b.capped as u32,
                cap: self.caps.2,
                ..Default::default()
            };
            e.pipe(self.p.softcap);
            e.buf(c.a, 0, 0);
            e.bytes(&p, 1);
            e.dispatch((self.unpadded.div_ceil(256), head_rows), 256);
        }
    }

    /// Copies the stream into the capture rows of the steps listed (true if any).
    fn capture(&self, e: &Pe, c: &Cx, si: usize, t: usize, capture: &[usize]) -> bool {
        let (h, nf) = (self.h, capture.len());
        let mut any = false;
        for (k, _) in capture.iter().enumerate().filter(|x| *x.1 == si) {
            self.copy(e, c, c.b.x, c.b.cap + k * h, h, nf * h, t);
            any = true;
        }
        any
    }

    /// A step's attention half and its routing (everything before the experts).
    #[allow(clippy::too_many_arguments)]
    fn enc_attn(&self, e: &Pe, c: &Cx, si: usize, t: usize, pos: usize, dev: &InkDev, capture: &[usize]) {
        let (h, th) = (self.h, c.b.tm * self.h);
        let b = c.b;
        let sp = &self.plan[si];
        let ld = &self.layers[sp.layer];
        let sd = &dev.slots[sp.slot];
        if sp.save {
            self.copy(e, c, b.x, b.xin, h, h, t);
            e.barrier();
        }
        if sp.swap {
            self.copy(e, c, b.x, b.bankx, h, h, t);
            e.barrier();
            self.copy(e, c, b.xin, b.x, h, h, t);
            e.barrier();
        }
        if self.capture(e, c, si, t, capture) {
            e.barrier();
        }
        let msgs = match (&sp.bank, &ld.gates) {
            (Some(bank), Some((_, lq))) => {
                if !self.skip(SK_MIX) {
                    self.mix_cat(e, c, bank, b.bankx, *lq, 4, b.msgs, None, t, Some((0, true, 3)));
                }
                true
            }
            _ => false,
        };
        if let Some(s) = sp.push {
            self.norm(e, c, b.x, c.state(s), None, MIX_EPS, h, h, t);
        }
        // The attention's inputs (normed): `xn + i * th`.
        let n_in = match (ld.attn_query, msgs) {
            (Some((q, nq)), false) => {
                if !self.skip(SK_MIX) {
                    self.mix(e, c, &sp.attn_states, b.x, q, nq, b.xn, Some(ld.ln_in.0), t);
                }
                nq
            }
            (None, false) => {
                self.norm(e, c, b.x, b.xn, Some(ld.ln_in.0), self.eps, h, h, t);
                1
            }
            (Some((q, nq)), true) => {
                if !self.skip(SK_MIX) {
                    self.mix_cat(e, c, &sp.attn_states, b.x, q, nq, b.mixed, None, t, Some((0, false, nq - 1)));
                }
                e.barrier();
                let (g, _) = ld.gates.as_ref().unwrap();
                let which: Vec<(usize, usize, usize, usize)> = (0..3)
                    .map(|i| {
                        let w = ld.ln_in.0 + if ld.ln_in.1 == 3 { i * h } else { 0 };
                        (i, b.mixed + i.min(nq - 1) * th, w, b.xn + i * th)
                    })
                    .collect();
                if !self.skip(SK_GATES) {
                    self.gates(e, c, g, &which, t, !self.skip(SK_MIX));
                }
                3
            }
            (None, true) => unreachable!("a looped layer reads the block states"),
        };
        e.barrier();
        let pf = c.b.pf.is_some() && t > MAX_T;
        // (Prefill GEMMs share one half-row scratch: one at a time.)
        let sep = |e: &Pe| {
            if pf {
                e.barrier();
            }
        };
        let xi = |i: usize| b.xn + if n_in == 3 { i * th } else { 0 };
        if !self.skip(SK_DENSE) {
            for (d, x, y) in [
                (&ld.q, xi(0), b.q),
                (&ld.r, xi(0), b.r),
                (&ld.gate, xi(0), b.gate),
                (&ld.sink, xi(0), b.sink),
                (&ld.k, xi(1), b.k),
                (&ld.v, xi(2), b.v),
            ] {
                self.mv(e, c, d, x, y, t, MvMode::Store);
                sep(e);
            }
            if let Some(kd) = &ld.kd {
                self.mv(e, c, kd, xi(1), b.kd, t, MvMode::Store);
                sep(e);
            }
            if let Some(vd) = &ld.vd {
                self.mv(e, c, vd, xi(2), b.vd, t, MvMode::Store);
            }
        }
        e.barrier();
        let (nh, nkv, hd) = (ld.nh, ld.nkv, ld.hd);
        let (w, kw) = (nh * hd, nkv * hd);
        let qp = QP {
            w: w as u32,
            hd: hd as u32,
            dr: self.dr as u32,
            nh: nh as u32,
            q: b.q as u32,
            r: b.r as u32,
            qs: ld.q_scale as u32,
            rs: ld.r_scale as u32,
            qclip: self.clips.0,
            rclip: self.clips.1,
            alpha: if ld.sliding { 0.0 } else { self.log_scale.0 },
            floor_: self.log_scale.1,
            pos: pos as u32,
            clip_eps: CLIP_EPS,
            ..Default::default()
        };
        self.std(e, c, self.p.q);
        e.bytes(&qp, 2);
        if !self.skip(SK_ATTN | SK_QKV) {
            e.dispatch((nh, t), hd);
        }
        let dynamic = ld.kd.is_some();
        let kn = self.kn;
        // A prefill pass longer than a sliding window's spare rows writes its keys and values
        // by position from `pos` into the arena's rows (copied to the ring after).
        let tmp = sd.ring != 0 && !self.inline_rings(t);
        let pfb = c.b.pf.as_ref();
        let kp = KvP {
            kw: kw as u32,
            hd: hd as u32,
            kn: kn as u32,
            nkv: nkv as u32,
            dynamic: dynamic as u32,
            norm: 1.0 / ((if dynamic { 2 * kn + 1 } else { kn + 1 }) as f32).sqrt(),
            clip: self.clips.0,
            eps: self.eps,
            k: b.k as u32,
            v: b.v as u32,
            kd: b.kd as u32,
            vd: b.vd as u32,
            tk: ld.k_taps as u32,
            tv: ld.v_taps as u32,
            pos: pos as u32,
            ring_cap: self.ring_rows as u32,
            cap: if tmp { 0 } else { sd.ring as u32 },
            clip_eps: CLIP_EPS,
            ring_write: self.inline_rings(t) as u32,
            row_base: if tmp { pos as u32 } else { 0 },
        };
        let fi = match sd.fmt {
            KvCache::F32 => 0,
            KvCache::F16 => 1,
            KvCache::Q8 => 2,
        };
        self.std(e, c, self.p.kv[fi]);
        e.bytes(&kp, 2);
        e.buf(sd.rings[0].0, 0, 3);
        e.buf(sd.rings[1].0, 0, 4);
        match (tmp, pfb) {
            (true, Some(pb)) => {
                e.buf(c.a, pb.kt * 4, 5);
                e.buf(c.a, pb.vt * 4, 6);
                e.buf(c.a, pb.kts * 4, 7);
                e.buf(c.a, pb.vts * 4, 8);
            }
            _ => {
                e.buf(sd.k.0, 0, 5);
                e.buf(sd.v.0, 0, 6);
                e.buf(sd.ks.0, 0, 7);
                e.buf(sd.vs.0, 0, 8);
            }
        }
        if !self.skip(SK_ATTN | SK_QKV) {
            e.dispatch((nkv, t), hd);
        }
        e.barrier();
        if !self.inline_rings(t) {
            self.ring_store(e, c, b.k, kw, sd.rings[0].0, pos, t);
            self.ring_store(e, c, b.v, kw, sd.rings[1].0, pos, t);
        }
        // Splits: the most any token of the step has (`ink_chunk`).
        let ms = c.b.maxsplit;
        let cmin = if t > MAX_T { 256 } else { 64 };
        let chunk = |len: usize| {
            cmin.max(256.min(len.div_ceil(5).div_ceil(32) * 32))
                .max(len.div_ceil(ms).div_ceil(32) * 32)
        };
        let maxsplit = (0..t)
            .map(|ti| {
                let hi = pos + ti + 1;
                let lo = if ld.window != 0 { hi.saturating_sub(ld.window) } else { 0 };
                (hi - lo).div_ceil(chunk(hi - lo)).max(1)
            })
            .max()
            .unwrap_or(1);
        let ap = AttnP {
            nh: nh as u32,
            nkv: nkv as u32,
            hd: hd as u32,
            dr: self.dr as u32,
            pos: pos as u32,
            window: ld.window as u32,
            cap: sd.ring as u32,
            extent: ld.extent as u32,
            cmin: cmin as u32,
            maxsplit: c.b.maxsplit as u32,
            scale: (hd as f32).powf(-0.5),
            q: b.q as u32,
            r: b.r as u32,
            part: b.part as u32,
            rel: ld.rel as u32,
            kw: kw as u32,
            tbase: pos as u32,
            tmp: tmp as u32,
            ..Default::default()
        };
        self.std(e, c, if t <= MAX_T { self.p.attv[fi] } else { self.p.attn[fi] });
        e.bytes(&ap, 2);
        e.buf(sd.k.0, 0, 3);
        e.buf(sd.v.0, 0, 4);
        e.buf(sd.ks.0, 0, 5);
        e.buf(sd.vs.0, 0, 6);
        match (tmp, pfb) {
            (true, Some(pb)) => {
                e.buf(c.a, pb.kt * 4, 7);
                e.buf(c.a, pb.vt * 4, 8);
                e.buf(c.a, pb.kts * 4, 9);
                e.buf(c.a, pb.vts * 4, 10);
            }
            _ => {
                e.buf(sd.k.0, 0, 7);
                e.buf(sd.v.0, 0, 8);
                e.buf(sd.ks.0, 0, 9);
                e.buf(sd.vs.0, 0, 10);
            }
        }
        if !self.skip(SK_ATTN | SK_ATTK) {
            e.dispatch3((nkv, maxsplit, t), 256);
        }
        e.barrier();
        if let (true, Some(pb), Some(pp)) = (tmp, pfb, &self.pfp) {
            // The pass's last rows to the window's ring.
            #[repr(C)]
            struct P {
                rb: u32,
                sb: u32,
                base: u32,
                ntok: u32,
                cap: u32,
                _p: [u32; 3],
            }
            let rb = row_bytes(sd.fmt, kw);
            let sb = if sd.fmt == KvCache::Q8 { kw / 32 * 4 } else { 0 };
            e.pipe(pp.kv_ring);
            e.bytes(
                &P {
                    rb: rb as u32,
                    sb: sb as u32,
                    base: pos as u32,
                    ntok: t as u32,
                    cap: sd.ring as u32,
                    _p: [0; 3],
                },
                0,
            );
            e.buf(c.a, pb.kt * 4, 1);
            e.buf(c.a, pb.vt * 4, 2);
            e.buf(c.a, pb.kts * 4, 3);
            e.buf(c.a, pb.vts * 4, 4);
            e.buf(sd.k.0, 0, 5);
            e.buf(sd.v.0, 0, 6);
            e.buf(sd.ks.0, 0, 7);
            e.buf(sd.vs.0, 0, 8);
            e.dispatch((rb.max(sb).div_ceil(4).div_ceil(256), t), 256);
        }
        let mp = MergeP {
            nh: nh as u32,
            hd: hd as u32,
            maxsplit: c.b.maxsplit as u32,
            part: b.part as u32,
            sink: b.sink as u32,
            gate: b.gate as u32,
            out: b.att as u32,
            pos: pos as u32,
            window: ld.window as u32,
            cmin: cmin as u32,
            ..Default::default()
        };
        e.pipe(self.p.merge);
        e.buf(c.a, 0, 0);
        e.bytes(&mp, 1);
        if !self.skip(SK_ATTN | SK_MERGE) {
            e.dispatch((nh, t), 128);
        }
        e.barrier();
        if !self.skip(SK_DENSE) {
            self.mv(e, c, &ld.o, b.att, b.ab, t, MvMode::Store);
        }
        e.barrier();
        if !self.skip(SK_CONV) {
            self.conv(e, c, b.ab, b.x, ld.attn_conv, sd.rings[2].0, pos, !ld.boundary, t);
        }
        e.barrier();
        if !self.inline_rings(t) {
            self.ring_store(e, c, b.ab, h, sd.rings[2].0, pos, t);
        }
        // The MLP's input.
        if msgs {
            if !self.skip(SK_MIX) {
                self.mix_cat(e, c, &sp.mlp_states, b.x, ld.mlp_query, 1, b.hm, None, t, Some((3, false, 0)));
            }
            e.barrier();
            let (g, _) = ld.gates.as_ref().unwrap();
            if !self.skip(SK_GATES) {
                self.gates(e, c, g, &[(3, b.hm, ld.ln_post, b.hn)], t, !self.skip(SK_MIX));
            }
        } else if !self.skip(SK_MIX) {
            self.mix(e, c, &sp.mlp_states, b.x, ld.mlp_query, 1, b.hn, Some(ld.ln_post), t);
        }
        e.barrier();
        self.route(e, c, ld, t);
    }

    /// The router's logits of `hn` (the f32 GEMM in prefill passes, else a SIMD-group per
    /// expert) and the routing into `sel` (a pass of one token also gets its expert groups).
    fn route(&self, e: &Pe, c: &Cx, ld: &LayerDev, t: usize) {
        let (h, b) = (self.h, c.b);
        match &self.pfp {
            Some(pp) if t > MAX_T => {
                // (32 x 32 tiles; the rows are padded to 32.)
                e.pipe(pp.gemm_f32);
                e.buf(self.consts, 0, 0);
                e.buf(self.consts, 0, 1);
                e.buf(c.a, b.hn * 4, 2);
                e.buf(c.a, b.rlog * 4, 3);
                e.bytes(&[self.ne as u32, h as u32, ld.router as u32, 0], 4);
                if !self.skip(SK_ROUTE) {
                    e.dispatch((self.ne / 32, t.div_ceil(32)), 128);
                }
            }
            _ if self.wide(t) && h % 4 == 0 && h <= 4096 => {
                let rp = MvfP {
                    n_out: self.ne as u32,
                    n_in: h as u32,
                    w: ld.router as u32,
                    x: b.hn as u32,
                    y: b.rlog as u32,
                    ..Default::default()
                };
                self.std(e, c, self.p.mvr);
                e.bytes(&rp, 2);
                if !self.skip(SK_ROUTE | SK_MVF) {
                    e.dispatch((self.ne, t), 128);
                }
            }
            _ => {
                let rp = MvfP {
                    n_out: self.ne as u32,
                    n_in: h as u32,
                    w: ld.router as u32,
                    x: b.hn as u32,
                    y: b.rlog as u32,
                    ..Default::default()
                };
                self.std(e, c, self.p.mvf);
                e.bytes(&rp, 2);
                if !self.skip(SK_ROUTE | SK_MVF) {
                    e.dispatch((self.ne.div_ceil(8), t), 256);
                }
            }
        }
        e.barrier();
        self.std(e, c, self.p.route);
        e.bytes(&self.route_params(c, ld, t), 2);
        if !self.skip(SK_RT) {
            e.dispatch((t, 1), self.ne.next_multiple_of(32));
        }
        e.barrier();
    }

    fn route_params(&self, c: &Cx, ld: &LayerDev, t: usize) -> RouteP {
        let b = c.b;
        RouteP {
            n: self.ne as u32,
            k: self.topk as u32,
            logits: b.rlog as u32,
            sel: b.sel as u32,
            bias: ld.bias as u32,
            scale: self.route_scale,
            single: (t == 1) as u32,
            grp: b.grp as u32,
            asg: b.asg as u32,
            ..Default::default()
        }
    }

    /// A step's experts (one token at a time) and the convolution after them.
    #[allow(clippy::too_many_arguments)]
    fn enc_moe(&self, e: &Pe, c: &Cx, si: usize, t: usize, pos: usize, dev: &InkDev, grouped: bool) {
        let (h, b) = (self.h, c.b);
        let sp = &self.plan[si];
        let ld = &self.layers[sp.layer];
        let sd = &dev.slots[sp.slot];
        if !grouped && t <= MAX_T {
            // The experts grouped by expert: tokens choosing the same one read it once.
            let n = t * self.topk;
            #[repr(C)]
            struct Pg {
                k: u32,
                n: u32,
                sel: u32,
                grp: u32,
                asg: u32,
                _p: [u32; 3],
            }
            // (A pass of one token has its groups from the routing.)
            if t > 1 {
                e.pipe(self.p.xgroup);
                e.buf(c.a, 0, 0);
                e.bytes(
                    &Pg {
                        k: self.topk as u32,
                        n: n as u32,
                        sel: b.sel as u32,
                        grp: b.grp as u32,
                        asg: b.asg as u32,
                        _p: [0; 3],
                    },
                    1,
                );
                e.dispatch((1, 1), n.next_multiple_of(32));
                e.barrier();
            }
            let xp = XP {
                inter: self.inter as u32,
                h: h as u32,
                k: self.topk as u32,
                il: ld.exp.il,
                off: ld.exp.off,
                x: b.hn as u32,
                sel: b.sel as u32,
                act: b.act as u32,
                y: b.od as u32,
                cap_g: self.caps.0,
                cap_u: self.caps.1,
                add: 0,
            };
            let gq = [b.grp as u32, b.asg as u32, ld.exp.lpr.0, ld.exp.lpr.1];
            let lq = self.gu_lora(e, c, ld, t);
            e.pipe(ld.exp.gpair);
            e.buf(c.a, 0, 0);
            e.buf(ld.exp.tab_gu, 0, 1);
            e.bytes(&xp, 2);
            e.bytes(&gq, 3);
            e.buf(lq.0, 0, 4);
            e.bytes(&lq.1, 5);
            if !self.skip(SK_EXP) {
                e.dispatch((self.inter.div_ceil(4 * 16 / ld.exp.lpr.0 as usize), n), 128);
            }
            e.barrier();
            e.pipe(ld.exp.gsum);
            e.buf(c.a, 0, 0);
            e.buf(ld.exp.tab_d, 0, 1);
            e.bytes(&xp, 2);
            e.bytes(&gq, 3);
            if !self.skip(SK_EXP) {
                e.dispatch((h.div_ceil(4 * 32 / ld.exp.lpr.1 as usize), n), 128);
            }
            e.barrier();
            if ld.exp.xlora.is_none() {
                // The convolution sums the slots itself.
                if !self.skip(SK_CONV) {
                    let cp = self.conv_p(b.o, b.x, ld.mlp_conv, pos, true, t, Some(b.od));
                    self.std(e, c, self.p.conv);
                    e.bytes(&cp, 2);
                    e.buf(sd.rings[3].0, 0, 3);
                    e.dispatch((self.h.div_ceil(256), t), 256);
                }
                e.barrier();
                return;
            }
            #[repr(C)]
            struct Pc {
                h: u32,
                k: u32,
                od: u32,
                y: u32,
                add: u32,
                _p: [u32; 3],
            }
            e.pipe(self.p.xcombine);
            e.buf(c.a, 0, 0);
            e.bytes(
                &Pc {
                    h: h as u32,
                    k: self.topk as u32,
                    od: b.od as u32,
                    y: b.o as u32,
                    add: 0,
                    _p: [0; 3],
                },
                1,
            );
            e.dispatch((h.div_ceil(256), t), 256);
            e.barrier();
            if let Some(xl) = &ld.exp.xlora {
                self.xlora(e, c, xl, t, None);
            }
        } else if !grouped {
            let xp = XP {
                inter: self.inter as u32,
                h: h as u32,
                k: self.topk as u32,
                il: ld.exp.il,
                off: ld.exp.off,
                x: b.hn as u32,
                sel: b.sel as u32,
                act: b.act as u32,
                y: b.o as u32,
                cap_g: self.caps.0,
                cap_u: self.caps.1,
                add: 0,
            };
            let lq = self.gu_lora(e, c, ld, t);
            e.pipe(ld.exp.pair);
            e.buf(c.a, 0, 0);
            e.buf(ld.exp.tab_gu, 0, 1);
            e.bytes(&xp, 2);
            e.buf(lq.0, 0, 3);
            e.bytes(&lq.1, 4);
            if !self.skip(SK_EXP) {
                e.dispatch3((self.inter.div_ceil(8), self.topk, t), 128);
            }
            e.barrier();
            e.pipe(ld.exp.sum);
            e.buf(c.a, 0, 0);
            e.buf(ld.exp.tab_d, 0, 1);
            e.bytes(&xp, 2);
            if !self.skip(SK_EXP) {
                e.dispatch((h.div_ceil(8), t), 128);
            }
            e.barrier();
            if let Some(xl) = &ld.exp.xlora {
                self.xlora(e, c, xl, t, None);
            }
        }
        if !self.skip(SK_CONV) {
            self.conv(e, c, b.o, b.x, ld.mlp_conv, sd.rings[3].0, pos, true, t);
        }
        e.barrier();
        if !self.inline_rings(t) {
            self.ring_store(e, c, b.o, h, sd.rings[3].0, pos, t);
            e.barrier();
        }
    }

    /// The down projections' LoRA updates of the routed experts, added to `o`: per (token, slot)
    /// from `act`, or with `rows` (prefill: the grouped rows, their experts and half
    /// activations) from those.
    fn xlora(&self, e: &Pe, c: &Cx, xl: &XLora, t: usize, rows: Option<(usize, &PfBufs)>) {
        self.xlora_a(e, c, xl, t, rows, false);
        #[repr(C)]
        struct Pb {
            n_out: u32,
            k: u32,
            rmax: u32,
            u: u32,
            sel: u32,
            y: u32,
            mapped: u32,
            _p: u32,
        }
        let h = self.h;
        e.pipe(self.p.xlora_b);
        e.buf(c.a, 0, 0);
        e.buf(xl.tab, 0, 1);
        e.bytes(
            &Pb {
                n_out: h as u32,
                k: self.topk as u32,
                rmax: c.b.lora_r as u32,
                u: c.b.lu as u32,
                sel: c.b.sel as u32,
                y: c.b.o as u32,
                mapped: rows.is_some() as u32,
                _p: 0,
            },
            2,
        );
        e.buf(c.a, rows.map_or(0, |r| r.1.map * 4), 3);
        e.dispatch((h.div_ceil(256), t), 256);
        e.barrier();
    }

    /// The gate / up projections' updates of `ld`, if any: their first half (then a barrier), and
    /// the table and parameters the pair kernels take (`IXQ`; off without updates).
    fn gu_lora(&self, e: &Pe, c: &Cx, ld: &LayerDev, t: usize) -> (Id, [u32; 4]) {
        match &ld.exp.xlora_gu {
            Some(xl) => {
                self.xlora_a(e, c, xl, t, None, true);
                (xl.tab, [1, c.b.lu as u32, c.b.lora_r as u32, 0])
            }
            None => (c.a, [0; 4]),
        }
    }

    /// The first half of the experts' updates `xl`: `u = A_e x` per slot into `lu`, x the slot's
    /// activations (down projections) or with `tok` the token's MLP input (gate / up).
    #[allow(clippy::too_many_arguments)]
    fn xlora_a(&self, e: &Pe, c: &Cx, xl: &XLora, t: usize, rows: Option<(usize, &PfBufs)>, tok: bool) {
        let (h, k, mi) = (self.h, self.topk, self.inter);
        let (n_in, act) = if tok { (h, c.b.hn) } else { (mi, c.b.act) };
        #[repr(C)]
        struct Pa {
            n_in: u32,
            k: u32,
            rmax: u32,
            act: u32,
            u: u32,
            sel: u32,
            rows: u32,
            tok: u32,
        }
        e.pipe(self.p.xlora_a);
        e.buf(c.a, 0, 0);
        e.buf(xl.tab, 0, 1);
        e.bytes(
            &Pa {
                n_in: n_in as u32,
                k: k as u32,
                rmax: c.b.lora_r as u32,
                act: act as u32,
                u: c.b.lu as u32,
                sel: c.b.sel as u32,
                rows: rows.map_or(0, |r| r.0) as u32,
                tok: tok as u32,
            },
            2,
        );
        match rows {
            Some((n, pb)) => {
                e.buf(c.a, pb.ah * 4, 3);
                e.buf(c.a, pb.rs2 * 4, 4);
                e.buf(c.a, pb.rexp * 4, 5);
                e.dispatch3((xl.rmax, n, 1), 32);
            }
            None => {
                e.buf(c.a, 0, 3);
                e.buf(c.a, 0, 4);
                e.buf(c.a, 0, 5);
                e.dispatch3((xl.rmax, k, t), 32);
            }
        }
        e.barrier();
    }

    /// The strengths of the experts' LoRA updates now, into their tables (adapters switch on and
    /// off at run time).
    fn lora_strengths(&self) {
        for xl in self.layers.iter().flat_map(|ld| ld.exp.xlora.iter().chain(&ld.exp.xlora_gu)) {
            {
                for (i, d) in xl.deltas.iter().enumerate() {
                    // SAFETY: the table holds one entry per expert; the deltas live with the model.
                    let s = d.map_or(0.0, |d| unsafe { (*d).strength() });
                    unsafe {
                        let ent = (xl.tab_p as *mut XlEntry).add(i);
                        (*ent).s = s;
                    }
                }
            }
        }
    }

    /// The grouped experts of step `si` (prefill), from the routing read back (`sel`, `[t][2 k]`
    /// as in the arena): the tokens of each expert gathered into tiles, gate / up GEMMs, SiTU,
    /// down GEMMs, and the weighted sum into `o`.
    #[allow(clippy::too_many_arguments)]
    fn enc_grouped(&self, e: &Pe, c: &Cx, si: usize, t: usize, gr: &Grouped, pb: &PfBufs, pp: &PfPipes, sel: &[f32]) {
        let (h, mi, k) = (self.h, self.inter, self.topk);
        let base = self.f.contents(c.a);
        let ld = &self.layers[self.plan[si].layer];
        // Rows per expert, in expert order; the row of every (token, slot).
        let mut count = vec![0usize; self.ne];
        for ti in 0..t {
            for s in 0..k {
                count[sel[ti * 2 * k + s].to_bits() as usize] += 1;
            }
        }
        let (tiles, ncls, start, rows) = super::group_tiles(&count);
        assert!(rows <= pb.max_rows && tiles.len() * 4 <= 4 * (pb.max_rows / 16 + self.ne));
        let mut fill = start.clone();
        unsafe {
            let src = std::slice::from_raw_parts_mut(base.add(pb.src * 4) as *mut u32, rows);
            let map = std::slice::from_raw_parts_mut(base.add(pb.map * 4) as *mut u32, t * k);
            let fgu = std::slice::from_raw_parts_mut(base.add(pb.f_gu * 4) as *mut f32, rows);
            let fd = std::slice::from_raw_parts_mut(base.add(pb.f_d * 4) as *mut f32, rows);
            let rexp = std::slice::from_raw_parts_mut(base.add(pb.rexp * 4) as *mut u32, rows);
            rexp.fill(u32::MAX);
            src.fill(u32::MAX);
            fgu.fill(0.0);
            fd.fill(0.0);
            for ti in 0..t {
                for s in 0..k {
                    let ex = sel[ti * 2 * k + s].to_bits() as usize;
                    let r = fill[ex];
                    fill[ex] += 1;
                    src[r] = ti as u32;
                    rexp[r] = ex as u32;
                    map[ti * k + s] = r as u32;
                    fgu[r] = gr.f_gu[ex];
                    fd[r] = gr.f_d[ex];
                }
            }
            std::ptr::copy_nonoverlapping(tiles.as_ptr() as *const u8, base.add(pb.tiles * 4), tiles.len() * 16);
        }
        #[repr(C)]
        struct Xh {
            n: u32,
            gather: u32,
            stride: u32,
            _p: u32,
        }
        e.pipe(pp.x_half);
        e.buf(c.a, c.b.hn * 4, 0);
        e.buf(c.a, pb.xe * 4, 1);
        e.buf(c.a, pb.rse * 4, 2);
        e.bytes(
            &Xh {
                n: h as u32,
                gather: 1,
                stride: h as u32,
                _p: 0,
            },
            3,
        );
        e.buf(c.a, pb.src * 4, 4);
        e.dispatch((rows, 1), 256);
        e.barrier();
        let params = |n_out: usize, n_in: usize, s_off: u32| super::MmaParams {
            n_out: n_out as u32,
            n_in: n_in as u32,
            w_off: 0,
            s_off,
            rows: rows as u32,
            _p: [0; 3],
        };
        // One grid per tile size (disjoint rows: no barrier between them).
        let grouped = |pipes: [Id; 3], ep: Id, xh: usize, out: usize, rs: usize, p: &super::MmaParams| {
            let mut first = 0;
            for (pipe, n) in [pipes[1], pipes[0], pipes[2]].into_iter().zip(ncls) {
                if n > 0 {
                    e.pipe(pipe);
                    e.buf(ep, 0, 0);
                    e.buf(c.a, (pb.tiles + first * 4) * 4, 1);
                    e.buf(c.a, xh * 4, 2);
                    e.buf(c.a, out * 4, 3);
                    e.bytes(p, 4);
                    e.buf(c.a, rs * 4, 5);
                    e.dispatch((p.n_out as usize / 64, n), 128);
                }
                first += n;
            }
        };
        if !self.skip(SK_EXP) {
            grouped(gr.gu, gr.ep_gu, pb.xe, pb.gu, pb.rse, &params(2 * mi, h, gr.rb.0));
        }
        e.barrier();
        #[repr(C)]
        struct Si {
            n: u32,
            il: u32,
            cap_g: f32,
            cap_u: f32,
        }
        e.pipe(pp.situ_half);
        e.buf(c.a, pb.gu * 4, 0);
        e.buf(c.a, pb.ah * 4, 1);
        e.buf(c.a, pb.rs2 * 4, 2);
        e.bytes(
            &Si {
                n: mi as u32,
                il: ld.exp.il,
                cap_g: self.caps.0,
                cap_u: self.caps.1,
            },
            3,
        );
        e.buf(c.a, pb.f_gu * 4, 4);
        e.dispatch((rows, 1), 256);
        e.barrier();
        if !self.skip(SK_EXP) {
            grouped(gr.d, gr.ep_d, pb.ah, pb.od, pb.rs2, &params(h, mi, gr.rb.1));
        }
        e.barrier();
        #[repr(C)]
        struct Cb {
            h: u32,
            k: u32,
            sel: u32,
            y: u32,
        }
        e.pipe(pp.combine);
        e.buf(c.a, 0, 0);
        e.bytes(
            &Cb {
                h: h as u32,
                k: k as u32,
                sel: c.b.sel as u32,
                y: c.b.o as u32,
            },
            1,
        );
        e.buf(c.a, pb.od * 4, 2);
        e.buf(c.a, pb.map * 4, 3);
        e.buf(c.a, pb.f_d * 4, 4);
        e.dispatch((h.div_ceil(256), t), 256);
        e.barrier();
        if let Some(xl) = &ld.exp.xlora {
            self.xlora(e, c, xl, t, Some((rows, pb)));
        }
    }

    /// A prefill pass of `t` tokens (at most the prefill arena's): command buffers that end
    /// at each MoE layer's routing, read back for the grouped experts.
    fn run_prefill(&self, c: &Cx, t: usize, pos: usize, dev: &InkDev, capture: &[usize], head_rows: usize) -> bool {
        let pb = c.b.pf.as_ref().expect("prefill arena");
        let pp = self.pfp.as_ref().expect("prefill kernels");
        let indirect = if self.be.residency.is_some() { &[][..] } else { &self.indirect[..] };
        let n = self.plan.len();
        let base = self.f.contents(c.a) as *const f32;
        let k = self.topk;
        let mut ok = true;
        // Step si's experts need its routing: [attn 0] wait [moe 0, attn 1] wait ...
        for si in 0..=n {
            let prev = si.checked_sub(1);
            let sel: Vec<f32> = match prev {
                Some(p) if self.layers[self.plan[p].layer].exp.grouped.is_some() => unsafe {
                    std::slice::from_raw_parts(base.add(c.b.sel), t * 2 * k).to_vec()
                },
                _ => Vec::new(),
            };
            let (r, _) = run_concurrent(self.be, &self.f, std::slice::from_ref(&(0..1)), |e, _| {
                let e = &Pe::new(e, None);
                e.use_resources(indirect);
                if let Some(p) = prev {
                    let gr = self.layers[self.plan[p].layer].exp.grouped.as_ref();
                    if let Some(gr) = gr {
                        self.enc_grouped(e, c, p, t, gr, pb, pp, &sel);
                    }
                    self.enc_moe(e, c, p, t, pos, dev, gr.is_some());
                }
                if si < n {
                    self.enc_attn(e, c, si, t, pos, dev, capture);
                } else {
                    self.enc_final(e, c, t, capture, head_rows);
                }
            });
            ok &= r;
            if !ok {
                break;
            }
        }
        ok
    }

    /// A pass of up to [`MAX_T`] tokens as `run` encodes it, cut at every barrier and timed
    /// stage by stage (`OXINFER_INKLING_PROFILE`); prints the time per kernel every few passes.
    #[allow(clippy::too_many_arguments)]
    fn run_profiled(&self, c: &Cx, t: usize, pos: usize, dev: &InkDev, capture: &[usize], head_rows: usize, res: &[Id]) -> bool {
        // (Passes of one token count apart: a pass of another size starts the count again.)
        type Acc = (usize, usize, usize, Vec<(Vec<usize>, f64, usize)>, usize);
        static ACC: std::sync::Mutex<Acc> = std::sync::Mutex::new((0, 0, 0, Vec::new(), 0));
        let pool = unsafe { (self.be.sels.pool_push)() };
        let (cb, enc) = begin_enc(self.be, &self.f);
        let pr = Prof {
            be: self.be,
            f: &self.f,
            res,
            cb: std::cell::Cell::new(cb),
            stage: Default::default(),
            times: Default::default(),
            pending: Default::default(),
        };
        let fe = FastEnc { enc, f: &self.f };
        let e = Pe::new(&fe, Some(&pr));
        e.use_resources(res);
        let (d0, b0) = (
            DISPATCHES.load(std::sync::atomic::Ordering::Relaxed),
            BARRIERS.load(std::sync::atomic::Ordering::Relaxed),
        );
        for si in 0..=self.plan.len() {
            if si == self.plan.len() {
                self.enc_final(&e, c, t, capture, head_rows);
            } else {
                self.enc_attn(&e, c, si, t, pos, dev, capture);
                self.enc_moe(&e, c, si, t, pos, dev, false);
            }
        }
        pr.stage.borrow_mut().push(0);
        e.cut(&pr);
        end_timed(self.be, &self.f, pr.cb.get(), e.enc.get());
        for (cb, st) in pr.pending.take() {
            let g = wait_timed(self.be, &self.f, cb);
            let mut t = pr.times.borrow_mut();
            let v = t.entry(st).or_insert((0.0, 0));
            v.0 += g;
            v.1 += 1;
        }
        unsafe { (self.be.sels.pool_pop)(pool) };
        let mut acc = ACC.lock().unwrap_or_else(|p| p.into_inner());
        if acc.4 != t {
            *acc = (0, 0, 0, Vec::new(), t);
        }
        acc.0 += 1;
        acc.1 += DISPATCHES.load(std::sync::atomic::Ordering::Relaxed) - d0;
        acc.2 += BARRIERS.load(std::sync::atomic::Ordering::Relaxed) - b0;
        for (k, (g, n)) in pr.times.into_inner() {
            match acc.3.iter_mut().find(|x| x.0 == k) {
                Some(x) => (x.1, x.2) = (x.1 + g, x.2 + n),
                None => acc.3.push((k, g, n)),
            }
        }
        if acc.0 >= profile_every() {
            let names: std::collections::HashMap<usize, String> = library(self.be)
                .map(|l| {
                    l.pipes
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .iter()
                        .map(|(k, v)| (*v, k.clone()))
                        .collect()
                })
                .unwrap_or_default();
            let passes = acc.0 as f64;
            let mut rows: Vec<(String, f64, f64)> = acc
                .3
                .iter()
                .map(|(k, g, n)| {
                    let label: Vec<String> = k
                        .iter()
                        .filter(|&&p| p != 0)
                        .map(|p| {
                            names.get(p).cloned().unwrap_or_else(|| {
                                if p >> 44 != 0 {
                                    format!("matvec {}x{}", p >> 44, (p >> 24) & 0xf_ffff)
                                } else {
                                    let sites = PIPE_SITES.lock().unwrap_or_else(|e| e.into_inner());
                                    format!("pipeline set at {}", sites.get(p).map_or("?", |s| s.as_str()))
                                }
                            })
                        })
                        .collect();
                    (label.join("+"), g / passes * 1e6, *n as f64 / passes)
                })
                .collect();
            // (One kind of stage: the same kernels, whatever their pipelines' variants.)
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            rows.dedup_by(|a, b| {
                a.0 == b.0 && {
                    b.1 += a.1;
                    b.2 += a.2;
                    true
                }
            });
            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total: f64 = rows.iter().map(|r| r.1).sum();
            eprintln!(
                "[inkling profile: {} passes of {t} token(s); per pass {:.0} dispatches, {:.0} barriers, {:.0} us of GPU time in its stages]",
                acc.0,
                acc.1 as f64 / passes,
                acc.2 as f64 / passes,
                total
            );
            for (l, us, n) in rows.iter().take(40) {
                eprintln!("  {us:8.1} us {:5.1}%  {n:6.1} stages  {l}", us / total * 100.0);
            }
            *acc = (0, 0, 0, Vec::new(), t);
        }
        true
    }
}

/// The arena and buffers of a pass.
struct Cx<'a> {
    a: Id,
    b: &'a Bufs,
}

impl Cx<'_> {
    /// The arena offset of block state `s`.
    fn state(&self, s: usize) -> usize {
        self.b.states + s * self.b.tm * self.b.h
    }
}

impl DevEngine for InkMetal {
    fn max_tokens(&self) -> usize {
        match &self.pfa {
            Some(_) => usize::MAX,
            None => MAX_T,
        }
    }

    fn rollback(&self) -> usize {
        ROLLBACK
    }

    fn forward(&self, m: &Inkling2Model, sess: &mut Inkling2Session, x: &[f32], t: usize) -> Vec<f32> {
        let mut hidden = Vec::new();
        let want = DevWant {
            hidden: Some(&mut hidden),
            logits: None,
            capture: None,
        };
        assert!(self.run(m, sess, x, t, want), "Metal: an Inkling-2 pass the engine does not take");
        hidden
    }

    /// (The head runs inside the passes that want logits.)
    fn head(&self, _m: &Inkling2Model, _hidden: &[f32], _t: usize, _cap: bool) -> Option<Vec<f32>> {
        None
    }

    fn to_host(&self, _m: &Inkling2Model, sess: &mut Inkling2Session) {
        if let Some(mut d) = sess.dev.take()
            && let Some(dev) = d.0.as_any().downcast_mut::<InkDev>()
        {
            dev.download(&mut sess.layers, sess.pos);
        }
    }

    fn run(&self, m: &Inkling2Model, sess: &mut Inkling2Session, x: &[f32], t: usize, mut want: DevWant) -> bool {
        let h = self.h;
        if t == 0 || x.len() != t * h || (t > MAX_T && self.pfa.is_none()) {
            return false;
        }
        let capture: &[usize] = want.capture.as_ref().map_or(&[], |c| c.0);
        let arena = if t > MAX_T { self.pfa.as_ref().unwrap() } else { &self.dec };
        if capture.len() > arena.b.nf {
            return false;
        }
        let _g = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        if sess.dev.is_none() && !self.upload(m, sess) {
            sess.dev = None;
            return false;
        }
        let pos0 = sess.pos;
        let Some(dev) = sess.dev.as_mut().and_then(|d| d.0.as_any().downcast_mut::<InkDev>()) else {
            return false;
        };
        dev.pos = pos0;
        if !dev.reserve(pos0 + t) {
            return false;
        }
        let dev: &InkDev = dev;
        self.lora_strengths();
        let c = Cx { a: arena.id, b: &arena.b };
        let base = self.f.contents(c.a) as *mut f32;
        let nf = capture.len();
        let mut hidden = want.hidden;
        let mut feats = want.capture.map(|c| c.1);
        if let Some(hs) = hidden.as_deref_mut() {
            hs.clear();
        }
        if let Some(f) = feats.as_deref_mut() {
            f.clear();
            f.resize(t * nf * h, 0.0);
        }
        let head = want.logits.as_ref().map_or(0, |l| l.1.min(t));
        // Pieces of at most the arena's tokens.
        let mut t0 = 0;
        while t0 < t {
            let n = (t - t0).min(arena.b.tm);
            let pos = pos0 + t0;
            let last = t0 + n == t;
            let head_rows = if last { head.min(n) } else { 0 };
            unsafe { std::ptr::copy_nonoverlapping(x[t0 * h..].as_ptr(), base.add(c.b.x), n * h) };
            let ok = if n > MAX_T {
                self.run_prefill(&c, n, pos, dev, capture, head_rows)
            } else {
                let steps = self.plan.len() + 1;
                // Command buffers of a few steps each: the GPU starts on the first while the rest is
                // encoded (the first of two steps only, so that it starts sooner: 0.07 ms a token).
                let parts: Vec<std::ops::Range<usize>> = std::iter::once(0..2.min(steps))
                    .chain((2..steps).step_by(8).map(|s| s..(s + 8).min(steps)))
                    .collect();
                let indirect = if self.be.residency.is_some() { &[][..] } else { &self.indirect[..] };
                if profile_every() > 0 {
                    self.run_profiled(&c, n, pos, dev, capture, head_rows, indirect)
                } else {
                    run_concurrent(self.be, &self.f, &parts, |e, r| {
                        let e = &Pe::new(e, None);
                        e.use_resources(indirect);
                        for si in r {
                            if si == self.plan.len() {
                                self.enc_final(e, &c, n, capture, head_rows);
                            } else {
                                self.enc_attn(e, &c, si, n, pos, dev, capture);
                                self.enc_moe(e, &c, si, n, pos, dev, false);
                            }
                        }
                    })
                    .0
                }
            };
            if !ok {
                panic!("Metal: an Inkling-2 pass failed (OXINFER_GPU_PLAN_DEBUG=1 says why)");
            }
            unsafe {
                if let Some(hs) = hidden.as_deref_mut() {
                    hs.extend_from_slice(std::slice::from_raw_parts(base.add(c.b.hidden), n * h));
                }
                if let Some(f) = feats.as_deref_mut() {
                    let src = std::slice::from_raw_parts(base.add(c.b.cap), n * nf * h);
                    f[t0 * nf * h..(t0 + n) * nf * h].copy_from_slice(src);
                }
                if last && let Some((l, _)) = &mut want.logits {
                    l.clear();
                    l.extend_from_slice(std::slice::from_raw_parts(base.add(c.b.capped), head_rows * self.unpadded));
                }
            }
            t0 += n;
        }
        sess.pos += t;
        if let Some(d) = sess.dev.as_mut().and_then(|d| d.0.as_any().downcast_mut::<InkDev>()) {
            d.pos = sess.pos;
        }
        true
    }
}

#[cfg(test)]
mod mxf8_tests {
    use super::*;

    fn lcg(s: &mut u64) -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }

    /// The MXFP8 expert kernels reading the E8M0 bytes (`mxf8b`) against the ones reading their
    /// f32 expansion (`mxf8`), bit for bit: the decode path's grouped down kernel (`_gsum`) and
    /// the prefill's grouped matrix-unit GEMMs (`mmag_*`, every tile size).
    #[test]
    fn mxf8_byte_scales_match_f32_scales() {
        let Ok(be) = MetalBackend::new() else { return };
        let lib = library(&be).unwrap();
        let f = Fns::new(&be);
        let (ne, n_out, n_in) = (2usize, 128usize, 512usize);
        let mut seed = 5u64;
        // E4M3 values without NaNs (0x7f, 0xff); exponents over a wide range, 0 included.
        let q: Vec<u8> = (0..ne * n_out * n_in)
            .map(|_| match lcg(&mut seed) as u8 {
                0x7f | 0xff => 0x10,
                v => v,
            })
            .collect();
        let e: Vec<u8> = (0..ne * n_out * n_in / 32)
            .map(|i| if i % 97 == 0 { 0 } else { 100 + (lcg(&mut seed) % 50) as u8 })
            .collect();
        let s: Vec<f32> = e.iter().map(|&b| crate::ggml::e8m0_to_f32(b)).collect();
        let buf = |bytes: &[u8]| {
            let (id, p) = be.ink_buffer(bytes.len()).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
            let addr = be.gpu_address(&super::super::Buf { id, offset_bytes: 0 });
            (id, addr)
        };
        let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let (qb, qa) = buf(&q);
        let (eb, ea) = buf(&e);
        let (sb, sa) = buf(&f32b(&s));
        let (per, sper) = ((n_out * n_in) as u64, (n_out * n_in / 32) as u64);
        let run = |enc: &dyn Fn(&FastEnc)| {
            let (ok, _) = run_concurrent(&be, &f, &[std::ops::Range { start: 0, end: 1 }], |fe, _| {
                fe.use_resources(&[qb, eb, sb]);
                enc(fe)
            });
            assert!(ok);
        };

        // Decode: tokens 0..t through expert 1's down rows (`gsum`, `h` = n_out, `inter` = n_in).
        let t = 5usize;
        let (g_off, asg_off, sel_off, act_off) = (0usize, 16, 32, 64);
        let y_off = act_off + t * n_in;
        let arena = y_off + t * n_out;
        let mut outs: Vec<Vec<u32>> = Vec::new();
        for (fmt, sc, sstride) in [("mxf8", sa, sper * 4), ("mxf8b", ea, sper)] {
            let pipe = lib.pipeline(&be, &format!("ink_x_{fmt}_gsum")).unwrap();
            let tab: Vec<XEntry> = (0..ne as u64)
                .map(|x| XEntry {
                    q: qa + x * per,
                    s: sc + x * sstride,
                    factor: 1.0,
                    rb: n_in as u32,
                    _p: 0,
                })
                .collect();
            let tab_bytes = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, std::mem::size_of_val(&tab[..])) };
            let (tb, _) = buf(tab_bytes);
            let (ab, ap) = be.ink_buffer(arena * 4).unwrap();
            let a = unsafe { std::slice::from_raw_parts_mut(ap as *mut f32, arena) };
            let ua = |i: usize, v: u32| unsafe { *(ap as *mut u32).add(i) = v };
            ua(g_off, 1);
            ua(g_off + 1, 1);
            ua(g_off + 2, t as u32);
            ua(g_off + 3, 0);
            for i in 0..t {
                ua(asg_off + i, i as u32);
                a[sel_off + 2 * i + 1] = 1.0;
            }
            let mut xs = 9u64;
            for v in &mut a[act_off..y_off] {
                *v = (lcg(&mut xs) as f32 / u32::MAX as f32 - 0.5) * 4.0;
            }
            let xp = XP {
                inter: n_in as u32,
                h: n_out as u32,
                k: 1,
                sel: sel_off as u32,
                act: act_off as u32,
                y: y_off as u32,
                ..Default::default()
            };
            let lpr = lanes_per_row(n_in, 32, 8);
            let gq = [g_off as u32, asg_off as u32, lpr, lpr];
            run(&|fe| {
                fe.use_resources(&[tb]);
                fe.pipe(pipe);
                fe.buf(ab, 0, 0);
                fe.buf(tb, 0, 1);
                fe.bytes(&xp, 2);
                fe.bytes(&gq, 3);
                fe.dispatch((n_out.div_ceil(4 * 32 / lpr as usize), 1), 128);
            });
            outs.push(a[y_off..].iter().map(|v| v.to_bits()).collect());
            assert!(a[y_off..].iter().any(|v| *v != 0.0), "{fmt}: no output");
            be.ink_release(ab);
            be.ink_release(tb);
        }
        assert_eq!(outs[0], outs[1], "decode (gsum): byte scales differ from f32 scales");

        // Prefill: the grouped GEMMs over 150 rows of expert 0 and 37 of expert 1.
        if lib.mma {
            let (tiles, ncls, _, rows) = super::super::group_tiles(&[150, 37]);
            let mut outs: Vec<Vec<u32>> = Vec::new();
            for (fmt, sc, sstride) in [("mxf8", sa, sper * 4), ("mxf8b", ea, sper)] {
                let pipes: Vec<Id> = ["", "_l", "_s"]
                    .iter()
                    .map(|x| lib.pipeline(&be, &format!("mmag_{fmt}{x}")).unwrap())
                    .collect();
                let ep: Vec<[u64; 2]> = (0..ne as u64).map(|x| [qa + x * per, sc + x * sstride]).collect();
                let (epb, _) = buf(unsafe { std::slice::from_raw_parts(ep.as_ptr() as *const u8, ep.len() * 16) });
                let (tlb, _) = buf(unsafe { std::slice::from_raw_parts(tiles.as_ptr() as *const u8, tiles.len() * 16) });
                let mut xs = 11u64;
                let xh: Vec<u8> = (0..rows * n_in)
                    .flat_map(|_| {
                        // Halves of magnitude 1/4 to 2, either sign.
                        let r = lcg(&mut xs);
                        (((r >> 31) << 15 | (13 + (r >> 12) % 3) << 10 | (r & 0x3ff)) as u16).to_le_bytes()
                    })
                    .collect();
                let (xb, _) = buf(&xh);
                let (rb, _) = buf(&f32b(&vec![0.5f32; rows]));
                let (yb, yp) = be.ink_buffer(rows * n_out * 4).unwrap();
                let p = super::super::MmaParams {
                    n_out: n_out as u32,
                    n_in: n_in as u32,
                    w_off: 0,
                    s_off: 0,
                    rows: rows as u32,
                    _p: [0; 3],
                };
                run(&|fe| {
                    let mut first = 0;
                    for (pipe, n) in [pipes[1], pipes[0], pipes[2]].into_iter().zip(ncls) {
                        if n > 0 {
                            fe.pipe(pipe);
                            fe.buf(epb, 0, 0);
                            fe.buf(tlb, first * 16, 1);
                            fe.buf(xb, 0, 2);
                            fe.buf(yb, 0, 3);
                            fe.bytes(&p, 4);
                            fe.buf(rb, 0, 5);
                            fe.dispatch((n_out / 64, n), 128);
                        }
                        first += n;
                    }
                });
                let y = unsafe { std::slice::from_raw_parts(yp as *const f32, rows * n_out) };
                assert!(y.iter().any(|v| *v != 0.0), "{fmt}: no output");
                outs.push(y.iter().map(|v| v.to_bits()).collect());
                for b in [epb, tlb, xb, rb, yb] {
                    be.ink_release(b);
                }
            }
            assert_eq!(outs[0], outs[1], "prefill (mmag): byte scales differ from f32 scales");
        }
        for b in [qb, eb, sb] {
            be.ink_release(b);
        }
    }
}
