//! DeepSeek V4.1 single-token decoding on Metal: a whole token in one stream of command buffers,
//! with no round trip to the CPU inside it ([`Plan::step`]).
//!
//! The model side (`deepseek41_gpu.rs`) describes the layers ([`ModelDesc`]), keeps the host
//! session in step with the device copies of the caches ([`State`]) and computes what only the
//! host has: the input embedding, the engram rows (looked up in a 110 GB mapped table) and the
//! rotary angles. Everything else runs here: hyper-connection mixes, projections (the decode
//! matvec kernels of `metal_decode.metal`), the compressors and indexers with their top-k, the
//! window + compressed attention with sinks, the grouped output projection, routing and the
//! routed and shared experts, the LM head (`metal_ds41.metal`).
//!
//! Routed experts are chosen on the GPU: each layer has a table of every expert's GPU address
//! (EPX entries of the `mvx_*` kernels), filled for the experts whose memory is wrapped. With
//! streamed weights those are the experts in the streamer's part cache, wrapped (and so wired)
//! while it keeps them: the part cache is the GPU's expert cache. A chosen expert without an
//! address is a miss: the router kernel records it and runs a zero expert instead; after the
//! pass the host reads the missed experts, wraps them, and runs the token again from the first
//! layer that missed (every cache write of a step goes to a fixed place, so a rerun is exact).

use super::decode::{FastEnc, Fns, MvFmt, MvMode, MvWeight, encode_mv, mv_variant};
use super::{Id, IntoInnerUnsafe, MetalBackend, Msg0, Msg1, MsgNewBufLen, MsgNewLib, MsgNewPipe, MsgVoid, Sel};
use crate::gpu::{Backend, DeviceWeight, WeightRef};
use crate::model::Lin;
use std::sync::Mutex;
use std::time::Instant;

const SRC: &str = include_str!("metal_ds41.metal");

const NAMES: [&str; 14] = [
    "d41_dots",
    "d41_mix_fin",
    "d41_collapse",
    "d41_norm_rope",
    "d41_pool",
    "d41_attn",
    "d41_attn_comb",
    "d41_hc_post",
    "d41_engram",
    "d41_route",
    "d41_idx_score",
    "d41_block_max",
    "d41_mask",
    "d41_topk",
];
const P_DOTS: usize = 0;
const P_MIX: usize = 1;
const P_COLL: usize = 2;
const P_NR: usize = 3;
const P_POOL: usize = 4;
const P_ATTN: usize = 5;
const P_COMB: usize = 6;
const P_POST: usize = 7;
const P_ENG: usize = 8;
const P_ROUTE: usize = 9;
const P_IDX: usize = 10;
const P_BLK: usize = 11;
const P_MASK: usize = 12;
const P_TOPK: usize = 13;

/// Splits of the hyper-connection dot products.
const SPLIT: usize = 8;
/// Keys per attention partial.
const CHUNK: usize = 32;
/// The kernels' fixed sizes: attention head, indexer heads and their width.
const HD: usize = 512;
const NI: usize = 32;
const DI: usize = 128;

/// `OXINFER_DS41_PROFILE=1`: every stage of a step in a command buffer of its own, the GPU time
/// summed per kind of op and printed every 32 steps (slower; for the breakdown only).
fn profile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_DS41_PROFILE").is_some_and(|v| v != "0"))
}

/// Layers per command stream of a step while experts miss (`OXINFER_DS41_CHUNK`, default 4).
fn cold_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("OXINFER_DS41_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(4)
    })
}

/// `OXINFER_DS41_TRACE=1`: one line per step (host and GPU time, misses).
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_DS41_TRACE").is_some_and(|v| v != "0"))
}

// ---- What the model describes ---------------------------------------------------------------

/// A matrix `[n_out, n_in]`.
pub(crate) struct Mat<'m> {
    pub lin: &'m Lin<'m>,
    pub n_out: usize,
    pub n_in: usize,
}

/// A hyper-connection mixer: `f` `[(2 + n) n, n h]`, `base` `[(2 + n) n]`, `scale` `[3]`.
pub(crate) struct HcDesc<'m> {
    pub f: &'m [f32],
    pub base: &'m [f32],
    pub scale: &'m [f32],
}

/// A key source's compressor.
pub(crate) struct CompDesc<'m> {
    pub wkv: Mat<'m>,
    pub wgate: Option<Mat<'m>>,
    pub norm: &'m [f32],
}

/// An index source's indexer (`wk` / `k_norm` on key sources).
pub(crate) struct IdxDesc<'m> {
    pub wq: Mat<'m>,
    pub proj: Mat<'m>,
    pub wk: Option<Mat<'m>>,
    pub k_norm: &'m [f32],
}

/// An engram layer's projection and gate weights (`[n, h]`).
pub(crate) struct EngDesc<'m> {
    pub wkv: Mat<'m>,
    pub weight: &'m [f32],
}

pub(crate) enum RouterDesc<'m> {
    F32(&'m [f32]),
    Bf16(&'m [u16]),
}

pub(crate) struct ExpertDesc<'m> {
    pub gate: &'m Lin<'m>,
    pub up: &'m Lin<'m>,
    pub down: &'m Lin<'m>,
}

pub(crate) struct LayerDesc<'m> {
    pub ratio: usize,
    pub attn_norm: &'m [f32],
    pub ffn_norm: &'m [f32],
    pub hc_attn: HcDesc<'m>,
    pub hc_ffn: HcDesc<'m>,
    pub wq_a: Mat<'m>,
    pub q_norm: &'m [f32],
    pub wq_b: Mat<'m>,
    pub wkv: Mat<'m>,
    pub kv_norm: &'m [f32],
    pub wo_a: Vec<Mat<'m>>,
    pub wo_b: Mat<'m>,
    pub sink: &'m [f32],
    pub comp: Option<CompDesc<'m>>,
    pub idx: Option<IdxDesc<'m>>,
    pub engram: Option<EngDesc<'m>>,
    /// The key source whose compressed keys the layer attends to.
    pub kv_src: Option<usize>,
    pub router: RouterDesc<'m>,
    pub bias: Option<&'m [f32]>,
    pub experts: Vec<ExpertDesc<'m>>,
    pub shared: [Mat<'m>; 3],
    pub shared_limit: f32,
}

pub(crate) struct ModelDesc<'m> {
    pub h: usize,
    /// Residual copies.
    pub n: usize,
    pub nh: usize,
    pub ql: usize,
    pub o_lora: usize,
    pub topk: usize,
    pub window: usize,
    /// Rotary pairs (half the rotated dimensions).
    pub half: usize,
    pub eps: f32,
    pub heps: f32,
    pub sinkhorn: usize,
    pub rsf: f32,
    pub limit: f32,
    /// 0 softmax, 1 sigmoid, 2 sqrt(softplus).
    pub scoring: u32,
    pub norm_topk: bool,
    pub k: usize,
    pub inter: usize,
    /// `(layer, blocks kept, block size)` of the candidate source.
    pub cand: Option<(usize, usize, usize)>,
    /// Floats of a layer's engram rows (`cols * head_dim`).
    pub eng_in: usize,
    pub vocab: usize,
    pub layers: Vec<LayerDesc<'m>>,
    pub norm: &'m [f32],
    pub head: Mat<'m>,
}

// ---- Parameters of the kernels (as in metal_ds41.metal) --------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DotsP {
    len: u32,
    rows: u32,
    split: u32,
    _p: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MixP {
    n: u32,
    split: u32,
    len: u32,
    iters: u32,
    eps: f32,
    heps: f32,
    _p: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CollP {
    h: u32,
    n: u32,
    eps: f32,
    _p: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NrP {
    d: u32,
    flags: u32,
    half: u32,
    ss: u32,
    ds: u32,
    eps: f32,
    _p: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PoolP {
    d: u32,
    r: u32,
    eps: f32,
    _p: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AttnP {
    nh: u32,
    nw: u32,
    lo: u32,
    win: u32,
    nc: u32,
    use_sel: u32,
    chunk: u32,
    half: u32,
    scale: f32,
    splits: u32,
    _p: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PostP {
    h: u32,
    n: u32,
    has_o2: u32,
    _p: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EngP {
    h: u32,
    n: u32,
    eps: f32,
    _p: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RouteP {
    n: u32,
    k: u32,
    rsf: f32,
    limit: f32,
    scoring: u32,
    norm: u32,
    layer: u32,
    has_bias: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IdxP {
    nv: u32,
    half: u32,
    ws: f32,
    bs: u32,
    has_mask: u32,
    _p: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BlkP {
    nv: u32,
    bs: u32,
    nb: u32,
    _p: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MaskP {
    nb: u32,
    k: u32,
    _p: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TopkP {
    n: u32,
    k: u32,
    _p: [u32; 2],
}

// ---- Device helpers --------------------------------------------------------------------------

fn new_buf(be: &MetalBackend, bytes: usize) -> Option<Id> {
    let f: MsgNewBufLen = be.o.msg_send();
    let id = unsafe { f(be.device, be.sels.new_buffer_len, bytes.max(16), 0) };
    (!id.is_null()).then_some(id)
}

fn release(be: &MetalBackend, id: Id) {
    if !id.is_null() {
        let void: MsgVoid = be.o.msg_send();
        unsafe { void(id, be.sels.release) };
    }
}

/// The library of `SRC` and its pipelines (in [`NAMES`] order).
fn compile(be: &MetalBackend) -> Option<Vec<Id>> {
    let o = &be.o;
    let get_class: unsafe extern "C" fn(*const std::ffi::c_char) -> Id = unsafe { o.objc.get(c"objc_getClass") }.ok()?.into_inner_unsafe();
    let msg_str: unsafe extern "C" fn(Id, Sel, *const std::ffi::c_char) -> Id = o.msg_send();
    let msg0: Msg0 = o.msg_send();
    let ns = |s: &std::ffi::CStr| unsafe { msg_str(get_class(c"NSString".as_ptr()), o.sel(c"stringWithUTF8String:"), s.as_ptr()) };
    let src = std::ffi::CString::new(SRC).ok()?;
    let new_lib: MsgNewLib = o.msg_send();
    let mut err: Id = std::ptr::null_mut();
    let lib = unsafe {
        new_lib(
            be.device,
            o.sel(c"newLibraryWithSource:options:error:"),
            ns(&src),
            std::ptr::null_mut(),
            &mut err,
        )
    };
    if lib.is_null() {
        if !err.is_null() {
            let text = unsafe { o.nsstring_to_string(msg0(err, o.sel(c"localizedDescription"))) };
            eprintln!("[deepseek v4.1 metal decode: the kernels do not build: {text}]");
        }
        return None;
    }
    let msg_fn: Msg1 = o.msg_send();
    let new_pipe: MsgNewPipe = o.msg_send();
    let mut pipes = Vec::with_capacity(NAMES.len());
    for name in NAMES {
        let c = std::ffi::CString::new(name).ok()?;
        let f = unsafe { msg_fn(lib, o.sel(c"newFunctionWithName:"), ns(&c)) };
        if f.is_null() {
            return None;
        }
        let p = unsafe { new_pipe(be.device, o.sel(c"newComputePipelineStateWithFunction:error:"), f, &mut err) };
        release(be, f);
        if p.is_null() {
            return None;
        }
        pipes.push(p);
    }
    Some(pipes)
}

/// A matvec: the weight as the kernels address it, its pipeline and variant.
#[derive(Clone, Copy)]
struct Mv {
    w: MvWeight,
    pipe: Id,
    nr: usize,
    ks: usize,
}

/// The constant f32 tables of the plan (norm weights, mixers, sinks, ...), copied into one
/// buffer: offsets in bytes.
#[derive(Default)]
struct Consts(Vec<u8>);

impl Consts {
    fn add(&mut self, v: &[f32]) -> usize {
        let at = self.0.len().next_multiple_of(256);
        self.0.resize(at, 0);
        self.0.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        at
    }
}

/// Activation buffers inside the arena: offsets in bytes.
#[derive(Default)]
struct Arena(usize);

impl Arena {
    fn take(&mut self, floats: usize) -> usize {
        let at = self.0.next_multiple_of(256);
        self.0 = at + floats.max(1) * 4;
        at
    }
}

/// What a layer runs, compiled.
struct LayerPlan {
    ratio: usize,
    kv_src: Option<usize>,
    /// Index of the layer among the key sources (its slot in [`State`]), and its rate.
    src: Option<usize>,
    /// Index among the index sources.
    isrc: Option<usize>,
    /// The index source whose choice this layer's attention takes.
    sel_from: Option<usize>,
    eng: Option<usize>,
    c_attn_norm: usize,
    c_ffn_norm: usize,
    c_hc: [(usize, usize, usize); 2],
    c_q_norm: usize,
    c_kv_norm: usize,
    c_sink: usize,
    c_comp_norm: usize,
    c_k_norm: usize,
    c_eng_w: usize,
    c_bias: Option<usize>,
    c_shared_limit: usize,
    hc_f: [Id; 2],
    wq_a: Mv,
    wkv: Mv,
    wq_b: Mv,
    wo_a: Vec<Mv>,
    wo_b: Mv,
    comp_wkv: Option<Mv>,
    comp_wgate: Option<Mv>,
    idx_wq: Option<Mv>,
    idx_proj: Option<Mv>,
    idx_wk: Option<Mv>,
    eng_wkv: Option<Mv>,
    router: Mv,
    sh: [Mv; 3],
    shared_clamp: bool,
    /// Expert address tables (gate, up, down; `n_exp + 1` EPX entries, the last the zero expert),
    /// the pair and sum pipelines, and the gate / up and down row bytes.
    tabs: [Id; 3],
    pair: Id,
    sum: Id,
    gpr: [u32; 2],
    /// Per expert: (address, length) of the gate, up and down data.
    exp_mem: Vec<[(usize, usize); 3]>,
}

/// Arena offsets (bytes).
struct Offs {
    x: Vec<usize>,
    xe: usize,
    xa_res: usize,
    part: usize,
    mix: Vec<[usize; 2]>,
    one_hot: usize,
    xa: usize,
    qa: usize,
    kvr: usize,
    iw: usize,
    q: usize,
    iq: usize,
    lat: usize,
    ik: usize,
    att: usize,
    apart: usize,
    og: usize,
    o1: usize,
    xf: usize,
    rlog: usize,
    sel: Vec<usize>,
    act: usize,
    act_s: usize,
    y1: usize,
    y2: usize,
    eng_kv: usize,
    eng_in: usize,
    cs: usize,
    isel: usize,
    cand: usize,
    miss: usize,
    hid: usize,
    logits: usize,
}

/// Experts wrapped for the device: their buffers per (layer, expert).
#[derive(Default)]
struct Wraps {
    map: std::collections::HashMap<(usize, usize), Vec<Id>>,
    /// The streamer's part changes are followed (a first full list was taken).
    tracking: bool,
}

/// The compiled decode of a model.
pub(crate) struct Plan {
    be: &'static MetalBackend,
    f: Fns,
    pipes: Vec<Id>,
    consts: Id,
    arena: Id,
    off: Offs,
    layers: Vec<LayerPlan>,
    head: Mv,
    c_norm: usize,
    _dev: Vec<DeviceWeight>,
    /// Plain buffers of the plan (released on drop).
    own: Vec<Id>,
    wraps: Mutex<Wraps>,
    /// The last step's experts per layer, for the part cache (see [`Self::step`]).
    pending: Mutex<Vec<(usize, Vec<usize>)>>,
    /// The last step missed experts: the next runs in chunks of layers.
    cold: std::sync::atomic::AtomicBool,
    lock: Mutex<()>,
    id: u64,
    // Sizes.
    h: usize,
    n: usize,
    nh: usize,
    ql: usize,
    o_lora: usize,
    groups: usize,
    topk: usize,
    window: usize,
    half: usize,
    eps: f32,
    heps: f32,
    sinkhorn: usize,
    rsf: f32,
    limit: f32,
    scoring: u32,
    norm_topk: bool,
    k: usize,
    n_exp: usize,
    inter: usize,
    cand: Option<(usize, usize, usize)>,
    eng_in: usize,
    n_eng: usize,
    vocab: usize,
    /// Rates of the key sources and the index sources (in layer order).
    src_rates: Vec<usize>,
    prof: Mutex<ProfTimes>,
}
unsafe impl Send for Plan {}
unsafe impl Sync for Plan {}

impl Drop for Plan {
    fn drop(&mut self) {
        let w = self.wraps.get_mut().unwrap();
        let ids: Vec<Id> = w.map.drain().flat_map(|(_, v)| v).collect();
        if let Some(r) = &self.be.residency
            && !ids.is_empty()
        {
            r.update(&ids, false);
        }
        for id in ids {
            release(self.be, id);
        }
        for &b in &self.own {
            release(self.be, b);
        }
    }
}

/// The device side of a session: the caches the steps read and write (shared memory, kept in
/// step with the host session by the model side).
pub(crate) struct State {
    be: &'static MetalBackend,
    plan: u64,
    /// `[layers, window, 512]`: each layer's last window keys, position p in row p % window.
    ring: Id,
    /// Per key source: compressed keys `[cap, 512]` and index keys `[cap, 128]`, rows valid.
    comp: Vec<(Id, Id, usize)>,
    /// Per key source: the rows of the group pooling (`[r, 512]` kv then `[r, 512]` gates, slot p % r).
    pend: Vec<Id>,
    /// Indexer scores, block scores and the candidate mask (`cap` positions).
    scores: Id,
    blk: Id,
    mask: Id,
    cap: usize,
    window: usize,
    /// Positions the device copies hold (the host may have moved on, or back).
    pub dev_pos: usize,
}
unsafe impl Send for State {}
unsafe impl Sync for State {}

impl Drop for State {
    fn drop(&mut self) {
        for (a, b, _) in &self.comp {
            release(self.be, *a);
            release(self.be, *b);
        }
        for &p in &self.pend {
            release(self.be, p);
        }
        for b in [self.ring, self.scores, self.blk, self.mask] {
            release(self.be, b);
        }
    }
}

fn contents(be: &MetalBackend, b: Id) -> *mut u8 {
    let f: super::MsgPtr = be.o.msg_send();
    unsafe { f(b, be.sels.contents) as *mut u8 }
}

impl State {
    /// Layer `l`'s window row of position `p`.
    pub(crate) fn ring_row(&mut self, l: usize, p: usize) -> &mut [f32] {
        let base = contents(self.be, self.ring) as *mut f32;
        let at = (l * self.window + p % self.window) * HD;
        // SAFETY: the ring holds layers * window rows of 512.
        unsafe { std::slice::from_raw_parts_mut(base.add(at), HD) }
    }
    /// Key source `s`'s compressed keys `from..to` (grown to hold them).
    pub(crate) fn comp_rows(&mut self, s: usize, from: usize, to: usize) -> Option<&mut [f32]> {
        self.reserve(to)?;
        let base = contents(self.be, self.comp[s].0) as *mut f32;
        // SAFETY: the buffer holds `cap` rows.
        Some(unsafe { std::slice::from_raw_parts_mut(base.add(from * HD), (to - from) * HD) })
    }
    /// Key source `s`'s index keys `from..to`.
    pub(crate) fn idx_rows(&mut self, s: usize, from: usize, to: usize) -> Option<&mut [f32]> {
        self.reserve(to)?;
        let base = contents(self.be, self.comp[s].1) as *mut f32;
        // SAFETY: the buffer holds `cap` rows.
        Some(unsafe { std::slice::from_raw_parts_mut(base.add(from * DI), (to - from) * DI) })
    }
    /// Key source `s`'s pooling rows: `[r, 512]` kv rows then `[r, 512]` gate rows.
    pub(crate) fn pend(&mut self, s: usize, r: usize) -> &mut [f32] {
        let base = contents(self.be, self.pend[s]) as *mut f32;
        // SAFETY: the buffer holds 2 r rows.
        unsafe { std::slice::from_raw_parts_mut(base, 2 * r * HD) }
    }
    /// The rows valid in key source `s`'s buffers (the model side sets them as it syncs).
    pub(crate) fn valid(&mut self, s: usize) -> &mut usize {
        &mut self.comp[s].2
    }

    /// Room for `rows` positions in every growing buffer.
    fn reserve(&mut self, rows: usize) -> Option<()> {
        if rows <= self.cap {
            return Some(());
        }
        let cap = rows.next_power_of_two().max(2 * self.cap);
        for (a, b, valid) in &mut self.comp {
            for (buf, w) in [(&mut *a, HD), (&mut *b, DI)] {
                let nb = new_buf(self.be, cap * w * 4)?;
                // SAFETY: both hold at least `valid` rows of w floats.
                unsafe {
                    std::ptr::copy_nonoverlapping(contents(self.be, *buf), contents(self.be, nb), *valid * w * 4);
                }
                release(self.be, *buf);
                *buf = nb;
            }
        }
        for b in [&mut self.scores, &mut self.blk, &mut self.mask] {
            release(self.be, *b);
            *b = new_buf(self.be, cap * 4)?;
        }
        self.cap = cap;
        Some(())
    }
}

/// GPU seconds and count per kind of stage, and the steps profiled.
type ProfTimes = (Vec<(&'static str, f64, usize)>, usize);

/// An expert matrix's (address, length) of data, scales and offsets, its row bytes and affine
/// group field.
type ExpertMem = ([(usize, usize); 3], u32, u32);

static PLAN_IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The device addresses an expert matvec table entry needs: data, scales, offsets (0: none), and
/// the row bytes / affine group fields.
fn expert_parts(w: &WeightRef) -> Option<ExpertMem> {
    let sp = |p: *const u8, n: usize| (p as usize, n);
    Some(match w {
        WeightRef::Gguf(g) => {
            if g.block_scales.is_some() || g.x4 {
                return None;
            }
            ([sp(g.data.as_ptr(), g.data.len()), (0, 0), (0, 0)], g.row_bytes() as u32, 0)
        }
        WeightRef::Bf16(v) => ([sp(v.as_ptr() as *const u8, v.len() * 2), (0, 0), (0, 0)], 0, 0),
        WeightRef::Q8(m) => (
            [
                sp(m.q.as_ptr() as *const u8, m.q.len()),
                sp(m.scales.as_ptr() as *const u8, m.scales.len() * 4),
                (0, 0),
            ],
            0,
            0,
        ),
        WeightRef::Q4(m) => {
            if m.offsets.is_some() {
                return None;
            }
            (
                [
                    sp(m.q.as_ptr(), m.q.len()),
                    sp(m.scales.as_ptr() as *const u8, m.scales.len() * 2),
                    (0, 0),
                ],
                0,
                0,
            )
        }
    })
}

impl Plan {
    /// Compiles the decode of `d` on `be` (None when some part cannot run there; the model then
    /// decodes on the CPU). Experts are wrapped by [`Self::step`] (`stream`: from the part cache;
    /// without, all of them on the first step).
    pub(crate) fn build(be: &'static MetalBackend, d: &ModelDesc) -> Option<Plan> {
        let dbg = |why: &str| {
            if std::env::var_os("OXINFER_DEBUG").is_some() {
                eprintln!("[deepseek v4.1 metal decode: {why}]");
            }
        };
        let (h, n, nh) = (d.h, d.n, d.nh);
        let n_exp = d.layers.first()?.experts.len();
        if !nh.is_multiple_of(8) || n > 8 || n * h % (SPLIT * 4) != 0 || h > 8192 || n_exp >= 1024 || d.k > 16 || d.ql > 4096 {
            dbg("unsupported sizes");
            return None;
        }
        be.residency.as_ref().or_else(|| {
            dbg("no residency set");
            None
        })?;
        let pipes = compile(be)?;
        let mut dev: Vec<DeviceWeight> = Vec::new();
        let mut mv = |m: &Mat| -> Option<Mv> {
            let w = m.lin.as_ref();
            if m.lin.active_lora().is_some() {
                return None;
            }
            if let WeightRef::Gguf(g) = &w
                && g.x4
            {
                return None;
            }
            let dw = be.upload(&w, m.n_out, m.n_in)?;
            let mw = MvWeight::new(&w, &dw, m.n_out, m.n_in)?;
            dev.push(dw);
            let (nr, ks) = mv_variant(mw.fmt, m.n_in);
            Some(Mv {
                w: mw,
                pipe: be.mv_pipe(mw.fmt, nr, ks)?,
                nr,
                ks,
            })
        };
        let mut c = Consts::default();
        let one_hot: Vec<f32> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let c_one_hot = c.add(&one_hot);
        let mut own: Vec<Id> = Vec::new();
        let mut layers = Vec::with_capacity(d.layers.len());
        let mut src_rates = Vec::new();
        let mut n_isrc = 0;
        let mut n_eng = 0;
        let mut last_isrc: Option<usize> = None;
        let mut max_exp = 0usize;
        let mut ok = true;
        for (l, ld) in d.layers.iter().enumerate() {
            if ld.experts.len() != n_exp || ld.wo_a.is_empty() || ld.sink.len() != nh {
                dbg("layer layout");
                return None;
            }
            let src = ld.comp.as_ref().map(|_| {
                src_rates.push(ld.ratio);
                src_rates.len() - 1
            });
            let isrc = ld.idx.as_ref().map(|_| {
                n_isrc += 1;
                n_isrc - 1
            });
            if isrc.is_some() {
                last_isrc = isrc;
            }
            let eng = ld.engram.as_ref().map(|_| {
                n_eng += 1;
                n_eng - 1
            });
            // The mixers' fn rows stay in a buffer of their own each (2 MB).
            let mut hc_f = [std::ptr::null_mut(); 2];
            let mut c_hc = [(0, 0, 0); 2];
            for (j, hc) in [&ld.hc_attn, &ld.hc_ffn].into_iter().enumerate() {
                if hc.f.len() != (2 + n) * n * n * h || hc.base.len() != (2 + n) * n || hc.scale.len() < 3 {
                    dbg("mixer shape");
                    return None;
                }
                let b = new_buf(be, hc.f.len() * 4)?;
                // SAFETY: the buffer holds the floats.
                unsafe { std::ptr::copy_nonoverlapping(hc.f.as_ptr(), contents(be, b) as *mut f32, hc.f.len()) };
                own.push(b);
                hc_f[j] = b;
                c_hc[j] = (0, c.add(hc.base), c.add(hc.scale));
            }
            // Routed experts: one format per projection.
            let e0 = &ld.experts[0];
            let (Some(fg), Some(fu), Some(fd)) = (
                MvFmt::of(&e0.gate.as_ref()),
                MvFmt::of(&e0.up.as_ref()),
                MvFmt::of(&e0.down.as_ref()),
            ) else {
                dbg("expert format");
                return None;
            };
            if fg != fu {
                dbg("gate and up formats differ");
                return None;
            }
            let mut exp_mem = Vec::with_capacity(n_exp);
            let mut gpr = [0u32; 2];
            for e in &ld.experts {
                let (Some(g), Some(u), Some(dn)) = (
                    expert_parts(&e.gate.as_ref()),
                    expert_parts(&e.up.as_ref()),
                    expert_parts(&e.down.as_ref()),
                ) else {
                    dbg("expert layout");
                    return None;
                };
                if g.0[1].0 != 0 || dn.0[1].0 != 0 {
                    // (Formats with separate scales: not on this path.)
                    dbg("expert scales");
                    return None;
                }
                gpr = [g.1, dn.1];
                max_exp = max_exp.max(g.0[0].1).max(u.0[0].1).max(dn.0[0].1);
                exp_mem.push([g.0[0], u.0[0], dn.0[0]]);
            }
            if gpr[0] == 0 {
                gpr[0] = (d.h * 2) as u32;
            }
            if gpr[1] == 0 {
                gpr[1] = (d.inter * 2) as u32;
            }
            let mut tabs = [std::ptr::null_mut(); 3];
            for t in &mut tabs {
                *t = new_buf(be, (n_exp + 1) * 32)?;
                // SAFETY: zeroed table (no expert addressable yet).
                unsafe { std::ptr::write_bytes(contents(be, *t), 0, (n_exp + 1) * 32) };
                own.push(*t);
            }
            let router = match &ld.router {
                RouterDesc::Bf16(w) => {
                    let (id, off) = be.wrap_raw(w.as_ptr() as *const u8, w.len() * 2)?;
                    own.push(id);
                    MvWeight {
                        fmt: MvFmt::Bf16,
                        n_out: n_exp,
                        n_in: h,
                        data: id,
                        data_off: off as u64,
                        scales: id,
                        scales_off: 0,
                        offs: id,
                        offs_off: 0,
                        gpr: (h * 2) as u32,
                        gk: 0,
                    }
                }
                RouterDesc::F32(w) => {
                    let b = new_buf(be, w.len() * 4)?;
                    // SAFETY: the buffer holds the floats.
                    unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), contents(be, b) as *mut f32, w.len()) };
                    own.push(b);
                    MvWeight {
                        fmt: MvFmt::Gguf(crate::ggml::GgmlType::F32),
                        n_out: n_exp,
                        n_in: h,
                        data: b,
                        data_off: 0,
                        scales: b,
                        scales_off: 0,
                        offs: b,
                        offs_off: 0,
                        gpr: (h * 4) as u32,
                        gk: 0,
                    }
                }
            };
            let (nr, ks) = mv_variant(router.fmt, h);
            let router = Mv {
                w: router,
                pipe: be.mv_pipe(router.fmt, nr, ks)?,
                nr,
                ks,
            };
            let opt = |m: Option<&Mat>, mv: &mut dyn FnMut(&Mat) -> Option<Mv>, ok: &mut bool| -> Option<Mv> {
                let m = m?;
                let r = mv(m);
                *ok &= r.is_some();
                r
            };
            let lp = LayerPlan {
                ratio: ld.ratio,
                kv_src: ld.kv_src,
                src,
                isrc,
                sel_from: if ld.kv_src.is_some() { last_isrc } else { None },
                eng,
                c_attn_norm: c.add(ld.attn_norm),
                c_ffn_norm: c.add(ld.ffn_norm),
                c_hc,
                c_q_norm: c.add(ld.q_norm),
                c_kv_norm: c.add(ld.kv_norm),
                c_sink: c.add(ld.sink),
                c_comp_norm: c.add(ld.comp.as_ref().map_or(&[0.0][..], |cp| cp.norm)),
                c_k_norm: c.add(
                    ld.idx
                        .as_ref()
                        .map_or(&[0.0][..], |ix| if ix.k_norm.is_empty() { &[0.0][..] } else { ix.k_norm }),
                ),
                c_eng_w: c.add(ld.engram.as_ref().map_or(&[0.0][..], |e| e.weight)),
                c_bias: ld.bias.map(|b| c.add(b)),
                c_shared_limit: c.add(&[ld.shared_limit]),
                hc_f,
                wq_a: mv(&ld.wq_a)?,
                wkv: mv(&ld.wkv)?,
                wq_b: mv(&ld.wq_b)?,
                wo_a: ld.wo_a.iter().map(&mut mv).collect::<Option<Vec<_>>>()?,
                wo_b: mv(&ld.wo_b)?,
                comp_wkv: opt(ld.comp.as_ref().map(|c| &c.wkv), &mut mv, &mut ok),
                comp_wgate: opt(ld.comp.as_ref().and_then(|c| c.wgate.as_ref()), &mut mv, &mut ok),
                idx_wq: opt(ld.idx.as_ref().map(|c| &c.wq), &mut mv, &mut ok),
                idx_proj: opt(ld.idx.as_ref().map(|c| &c.proj), &mut mv, &mut ok),
                idx_wk: opt(ld.idx.as_ref().and_then(|c| c.wk.as_ref()), &mut mv, &mut ok),
                eng_wkv: opt(ld.engram.as_ref().map(|e| &e.wkv), &mut mv, &mut ok),
                router,
                sh: [mv(&ld.shared[0])?, mv(&ld.shared[1])?, mv(&ld.shared[2])?],
                shared_clamp: ld.shared_limit > 0.0,
                tabs,
                pair: be.named_pipeline(&format!("mvx_{}_pair", fg.name()))?,
                sum: be.named_pipeline(&format!("mvx_{}_sum", fd.name()))?,
                gpr,
                exp_mem,
            };
            if !ok {
                dbg(&format!("layer {l}: a matrix the decode kernels do not take"));
                return None;
            }
            if lp.sh[0].w.fmt != lp.sh[1].w.fmt {
                dbg("shared expert formats");
                return None;
            }
            layers.push(lp);
        }
        if layers.iter().any(|lp| lp.kv_src.is_some() && lp.sel_from.is_none()) {
            dbg("a layer reads compressed keys before any indexer");
            return None;
        }
        let head = mv(&d.head)?;
        let c_norm = c.add(d.norm);
        // The arena.
        let nl = d.layers.len();
        let mut a = Arena::default();
        let max_nc = d.window + d.topk;
        let off = Offs {
            x: (0..=nl).map(|_| a.take(n * h)).collect(),
            xe: a.take(n * h),
            xa_res: a.take(n * h),
            part: a.take(((2 + n) * n + 1) * SPLIT),
            mix: (0..nl).map(|_| [a.take((2 + n) * n), a.take((2 + n) * n)]).collect(),
            one_hot: 0,
            xa: a.take(h),
            qa: a.take(d.ql),
            kvr: a.take(HD),
            iw: a.take(NI),
            q: a.take(nh * HD),
            iq: a.take(NI * DI),
            lat: a.take(HD),
            ik: a.take(DI),
            att: a.take(nh * HD),
            apart: a.take(nh * max_nc.div_ceil(CHUNK) * (HD + 2)),
            og: a.take(d.layers[0].wo_a.len() * d.o_lora),
            o1: a.take(h),
            xf: a.take(h),
            rlog: a.take(n_exp),
            sel: (0..nl).map(|_| a.take(2 * d.k + 1)).collect(),
            act: a.take(d.k * d.inter),
            act_s: a.take(d.layers[0].shared[0].n_out),
            y1: a.take(h),
            y2: a.take(h),
            eng_kv: a.take((n + 1) * h),
            eng_in: a.take(n_eng.max(1) * d.eng_in),
            cs: a.take(3 * 2 * d.half),
            isel: a.take(n_isrc.max(1) * d.topk),
            cand: a.take(d.cand.map_or(1, |c| c.1)),
            miss: a.take(1 + nl * d.k),
            hid: a.take(h),
            logits: a.take(d.vocab),
        };
        let arena = new_buf(be, a.0)?;
        own.push(arena);
        let mut off = off;
        off.one_hot = c_one_hot;
        let consts = new_buf(be, c.0.len())?;
        // SAFETY: the buffer holds the bytes.
        unsafe { std::ptr::copy_nonoverlapping(c.0.as_ptr(), contents(be, consts), c.0.len()) };
        own.push(consts);
        let zero = new_buf(be, max_exp.next_multiple_of(16384))?;
        // SAFETY: zeroed (new buffers are, but make it explicit).
        unsafe { std::ptr::write_bytes(contents(be, zero), 0, max_exp) };
        own.push(zero);
        let za = be.buffer_gpu_address(zero);
        for lp in &layers {
            for &t in &lp.tabs {
                // SAFETY: the table has n_exp + 1 entries of 4 u64.
                unsafe {
                    let p = (contents(be, t) as *mut u64).add(4 * n_exp);
                    for j in 0..4 {
                        *p.add(j) = za;
                    }
                }
            }
        }
        if let Some(r) = &be.residency {
            r.update(&[zero], true);
        }
        Some(Plan {
            be,
            f: Fns::new(be),
            pipes,
            consts,
            arena,
            off,
            layers,
            head,
            c_norm,
            _dev: dev,
            own,
            wraps: Mutex::new(Wraps::default()),
            pending: Mutex::new(Vec::new()),
            cold: std::sync::atomic::AtomicBool::new(true),
            lock: Mutex::new(()),
            id: PLAN_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            h,
            n,
            nh,
            ql: d.ql,
            o_lora: d.o_lora,
            groups: d.layers[0].wo_a.len(),
            topk: d.topk,
            window: d.window,
            half: d.half,
            eps: d.eps,
            heps: d.heps,
            sinkhorn: d.sinkhorn,
            rsf: d.rsf,
            limit: d.limit,
            scoring: d.scoring,
            norm_topk: d.norm_topk,
            k: d.k,
            n_exp,
            inter: d.inter,
            cand: d.cand,
            eng_in: d.eng_in,
            n_eng,
            vocab: d.vocab,
            src_rates,
            prof: Mutex::new((Vec::new(), 0)),
        })
    }

    /// Wraps the experts in memory now (a first step would), e.g. while a prefill runs.
    pub(crate) fn prewrap(&self, hook: Option<&crate::stream::StreamHook>) {
        let mut w = self.wraps.lock().unwrap();
        self.sync_wraps(&mut w, hook, &[]);
    }

    /// A fresh device state for a session.
    pub(crate) fn new_state(&self) -> Option<State> {
        let cap = 4096;
        let nl = self.layers.len();
        let mut comp = Vec::new();
        for _ in &self.src_rates {
            comp.push((new_buf(self.be, cap * HD * 4)?, new_buf(self.be, cap * DI * 4)?, 0));
        }
        Some(State {
            be: self.be,
            plan: self.id,
            ring: new_buf(self.be, nl * self.window * HD * 4)?,
            comp,
            pend: self
                .src_rates
                .iter()
                .map(|&r| new_buf(self.be, 2 * r.max(1) * HD * 4))
                .collect::<Option<_>>()?,
            scores: new_buf(self.be, cap * 4)?,
            blk: new_buf(self.be, cap * 4)?,
            mask: new_buf(self.be, cap * 4)?,
            cap,
            window: self.window,
            dev_pos: 0,
        })
    }

    /// Whether `st` is a state of this plan.
    pub(crate) fn owns(&self, st: &State) -> bool {
        st.plan == self.id
    }

    /// The rates of the key sources (their order is the slots of [`State`]).
    pub(crate) fn src_rates(&self) -> &[usize] {
        &self.src_rates
    }

    // (The arena is device-shared memory, read and written under the step lock only.)
    #[allow(clippy::mut_from_ref)]
    fn arena_f32(&self, off: usize, n: usize) -> &mut [f32] {
        // SAFETY: offsets come from the arena layout; the GPU is idle between steps.
        unsafe { std::slice::from_raw_parts_mut(contents(self.be, self.arena).add(off) as *mut f32, n) }
    }

    /// Wraps expert `e` of layer `l` and writes its table entries. Returns its buffers.
    fn wrap_expert(&self, l: usize, e: usize) -> Option<Vec<Id>> {
        let lp = &self.layers[l];
        let mut ids = Vec::with_capacity(3);
        for (j, &(a, len)) in lp.exp_mem[e].iter().enumerate() {
            let Some((id, off)) = self.be.wrap_raw(a as *const u8, len) else {
                for id in ids {
                    release(self.be, id);
                }
                return None;
            };
            let addr = self.be.buffer_gpu_address(id) + off as u64;
            // SAFETY: entry e of a table of n_exp + 1.
            unsafe {
                let p = (contents(self.be, lp.tabs[j]) as *mut u64).add(4 * e);
                for k in 0..4 {
                    *p.add(k) = addr;
                }
            }
            ids.push(id);
        }
        Some(ids)
    }

    fn unwrap_expert(&self, l: usize, e: usize) {
        let lp = &self.layers[l];
        for &t in &lp.tabs {
            // SAFETY: entry e of a table of n_exp + 1.
            unsafe { std::ptr::write_bytes((contents(self.be, t) as *mut u64).add(4 * e), 0, 4) };
        }
    }

    /// Brings the wrapped experts in line with what is in memory: without a streamer every
    /// expert (once); with one, the parts its cache holds (all of them the first time, then its
    /// changes). `extra`: parts in use now (just read) to wrap as well.
    fn sync_wraps(&self, w: &mut Wraps, hook: Option<&crate::stream::StreamHook>, extra: &[(usize, usize)]) -> bool {
        let mut add: Vec<(usize, usize)> = Vec::new();
        let mut remove: Vec<(usize, usize)> = Vec::new();
        match hook {
            None if !w.tracking => {
                w.tracking = true;
                for l in 0..self.layers.len() {
                    add.extend((0..self.n_exp).map(|e| (l, e)));
                }
            }
            None => {}
            Some(hk) => {
                if !w.tracking {
                    w.tracking = true;
                    add = hk
                        .track_parts()
                        .into_iter()
                        .filter(|&(l, e)| l < self.layers.len() && e < self.n_exp)
                        .collect();
                } else {
                    for (l, e, on) in hk.part_changes() {
                        if l >= self.layers.len() || e >= self.n_exp {
                            continue;
                        }
                        if on {
                            remove.retain(|&k| k != (l, e));
                            add.push((l, e));
                        } else {
                            add.retain(|&k| k != (l, e));
                            remove.push((l, e));
                        }
                    }
                }
            }
        }
        add.extend_from_slice(extra);
        let mut gone: Vec<Id> = Vec::new();
        for k in remove {
            if let Some(v) = w.map.remove(&k) {
                self.unwrap_expert(k.0, k.1);
                gone.extend(v);
            }
        }
        let resid = self.be.residency.as_ref().unwrap();
        if !gone.is_empty() {
            resid.update(&gone, false);
            for id in gone {
                release(self.be, id);
            }
        }
        add.sort_unstable();
        add.dedup();
        add.retain(|k| !w.map.contains_key(k));
        if add.is_empty() {
            return true;
        }
        // Wrapping costs ~10-50 us an expert: in parallel for a first fill.
        let t0 = Instant::now();
        let results: Vec<Option<Vec<Id>>> = if add.len() > 64 {
            let out: Vec<Mutex<Option<Vec<usize>>>> = add.iter().map(|_| Mutex::new(None)).collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|sc| {
                for _ in 0..8 {
                    sc.spawn(|| {
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= add.len() {
                                break;
                            }
                            *out[i].lock().unwrap() = self
                                .wrap_expert(add[i].0, add[i].1)
                                .map(|v| v.into_iter().map(|id| id as usize).collect());
                        }
                    });
                }
            });
            out.into_iter()
                .map(|m| m.into_inner().unwrap().map(|v| v.into_iter().map(|id| id as Id).collect()))
                .collect()
        } else {
            add.iter().map(|&(l, e)| self.wrap_expert(l, e)).collect()
        };
        let mut ok = true;
        let mut new_ids = Vec::new();
        for (k, r) in add.iter().zip(results) {
            match r {
                Some(v) => {
                    new_ids.extend(v.iter().copied());
                    w.map.insert(*k, v);
                }
                None => ok = false,
            }
        }
        if !new_ids.is_empty() {
            resid.update(&new_ids, true);
        }
        if add.len() > 64 && trace() {
            eprintln!(
                "[ds41 gpu: wrapped {} experts in {:.0} ms ({} wrapped)]",
                add.len(),
                t0.elapsed().as_secs_f64() * 1e3,
                w.map.len()
            );
        }
        ok
    }

    /// One token at position `inp.pos` (the device caches hold the positions before it). Writes
    /// the logits (or, with `hidden_only`, the final-normed hidden state) to `out`. False when
    /// the device failed (nothing of the host side changed).
    pub(crate) fn step(
        &self,
        st: &mut State,
        inp: &StepIn,
        hook: Option<&crate::stream::StreamHook>,
        out: &mut [f32],
        hidden_only: bool,
    ) -> bool {
        let _g = self.lock.lock().unwrap();
        let t0 = Instant::now();
        let (h, n, k) = (self.h, self.n, self.k);
        let pos = inp.pos;
        // Room for this position's keys and scores.
        if st.reserve(pos + 1).is_none() {
            return false;
        }
        let mut wraps = self.wraps.lock().unwrap();
        if !self.sync_wraps(&mut wraps, hook, &[]) {
            return false;
        }
        let tw = t0.elapsed().as_secs_f64();
        // Inputs.
        let x0 = self.arena_f32(self.off.x[0], n * h);
        for c in 0..n {
            x0[c * h..(c + 1) * h].copy_from_slice(inp.x);
        }
        self.arena_f32(self.off.cs, inp.cs.len()).copy_from_slice(inp.cs);
        if self.n_eng > 0 {
            self.arena_f32(self.off.eng_in, inp.eng.len()).copy_from_slice(inp.eng);
        }
        let nl = self.layers.len();
        // While experts miss (a cold cache), a step runs in chunks of layers, each waited for:
        // a chunk's misses are read before the layers after it run on a wrong guess (which
        // would miss experts of their own). Warm, the whole token is one stream.
        let chunk = if self.cold.load(std::sync::atomic::Ordering::Relaxed) {
            cold_chunk()
        } else {
            nl
        };
        let mut from = 0;
        let mut upto = chunk.min(nl);
        let mut passes = 0;
        let mut missed_total = 0;
        let mut gpu = 0f64;
        let mut tio = 0f64;
        let mut tbk = 0f64;
        loop {
            self.arena_f32(self.off.miss, 1)[0] = 0.0;
            // The part cache's bookkeeping of the last step's experts, while the GPU runs.
            let (ok, g) = self.run(st, pos, from, upto, hidden_only, &mut || {
                if let Some(hk) = hook {
                    let ti = Instant::now();
                    for (l, ids) in std::mem::take(&mut *self.pending.lock().unwrap()) {
                        hk.begin(l);
                        hk.need_parts(l, &ids);
                        hk.leave(l);
                    }
                    tbk += ti.elapsed().as_secs_f64();
                }
            });
            gpu += g;
            passes += 1;
            if !ok {
                return false;
            }
            // SAFETY: the miss list: a count, then layer << 16 | expert.
            let miss: Vec<u32> = unsafe {
                let p = contents(self.be, self.arena).add(self.off.miss) as *const u32;
                let cnt = (*p as usize).min(self.layers.len() * k);
                (0..cnt).map(|i| *p.add(1 + i)).collect()
            };
            if miss.is_empty() {
                if upto == nl {
                    break;
                }
                from = upto;
                upto = (upto + chunk).min(nl);
                continue;
            }
            missed_total += miss.len();
            let Some(hk) = hook else {
                // Every expert is wrapped without a streamer: a miss is a failed wrap.
                return false;
            };
            if passes > 4 * nl {
                return false;
            }
            let ti = Instant::now();
            let mut by_layer: Vec<Vec<usize>> = vec![Vec::new(); self.layers.len()];
            for &m in &miss {
                let (l, e) = ((m >> 16) as usize, (m & 0xffff) as usize);
                if l < self.layers.len() && e < self.n_exp && !by_layer[l].contains(&e) {
                    by_layer[l].push(e);
                }
            }
            for (l, ids) in by_layer.iter().enumerate() {
                if !ids.is_empty() {
                    hk.prefetch_parts(l, ids);
                }
            }
            let mut extra = Vec::new();
            for (l, ids) in by_layer.iter().enumerate() {
                if ids.is_empty() {
                    continue;
                }
                hk.begin(l);
                hk.need_parts(l, ids);
                extra.extend(ids.iter().map(|&e| (l, e)));
                // Wrapped while in use (the part cache takes them when the unit is left).
                if !self.sync_wraps(&mut wraps, Some(hk), &extra) {
                    hk.leave(l);
                    return false;
                }
                hk.leave(l);
            }
            // Parts the cache would not keep (it is full of parts in use): dropped by the next
            // sync, after this step.
            tio += ti.elapsed().as_secs_f64();
            from = by_layer.iter().position(|v| !v.is_empty()).unwrap_or(0);
        }
        self.cold.store(missed_total > 0, std::sync::atomic::Ordering::Relaxed);
        // The experts used, for the part cache's least-recently-used order (taken while the next
        // step runs on the GPU).
        if hook.is_some() {
            let mut pend = self.pending.lock().unwrap();
            pend.clear();
            for l in 0..self.layers.len() {
                let sel = self.arena_f32(self.off.sel[l], k);
                pend.push((l, sel.iter().map(|v| v.to_bits() as usize).filter(|&e| e < self.n_exp).collect()));
            }
        }
        drop(wraps);
        if hidden_only {
            out[..h].copy_from_slice(self.arena_f32(self.off.hid, h));
        } else {
            out[..self.vocab].copy_from_slice(self.arena_f32(self.off.logits, self.vocab));
        }
        if trace() {
            eprintln!(
                "[ds41 gpu step {pos}: {:.2} ms, gpu {:.2} ms, {} pass(es), {} misses, wraps {:.2} ms, expert io {:.2} ms, cache bookkeeping {:.2} ms]",
                t0.elapsed().as_secs_f64() * 1e3,
                gpu * 1e3,
                passes,
                missed_total,
                tw * 1e3,
                tio * 1e3,
                tbk * 1e3
            );
        }
        true
    }

    /// Encodes and runs layers `from..` and the head. Returns (ok, GPU seconds).
    /// Encodes and runs layers `from..upto` (and the head when `upto` is the last layer).
    /// Returns (ok, GPU seconds).
    #[allow(clippy::too_many_arguments)]
    fn run(&self, st: &State, pos: usize, from: usize, upto: usize, hidden_only: bool, while_busy: &mut dyn FnMut()) -> (bool, f64) {
        let be = self.be;
        let f = &self.f;
        let prof = profile();
        let pool = unsafe { (be.sels.pool_push)() };
        let mut cbs: Vec<(Id, &'static str)> = Vec::new();
        let new_enc = || unsafe {
            let cb = (f.msg0)(be.queue, f.s_cb);
            (cb, (f.enc_type)(cb, f.s_enc_type, 1))
        };
        let mut cur = new_enc();
        let mut rec = Rec {
            f,
            cur: &mut cur,
            cbs: &mut cbs,
            prof,
            label: "mix",
            new_enc: &new_enc,
        };
        let nl = self.layers.len();
        for l in from..upto {
            self.encode_layer(&mut rec, st, l, pos);
            // A few command buffers per token: the GPU starts on the first ones while the rest
            // is encoded.
            if !prof && (l == from + 1 || l % 10 == 9) {
                rec.cut();
            }
        }
        if upto == nl {
            self.encode_head(&mut rec, hidden_only);
        }
        rec.cut();
        drop(rec);
        while_busy();
        let ok;
        let mut g = 0.0;
        unsafe {
            let mut all = true;
            for &(cb, _) in cbs.iter() {
                (f.void)(cb, f.s_wait);
                all &= (f.status)(cb, f.s_status) == 4;
            }
            ok = all;
            let time: unsafe extern "C" fn(Id, Sel) -> f64 = be.o.msg_send();
            if !cbs.is_empty() {
                let (s0, e0) = (be.o.sel(c"GPUStartTime"), be.o.sel(c"GPUEndTime"));
                if prof {
                    let mut p = self.prof.lock().unwrap();
                    for &(cb, label) in &cbs {
                        let t = time(cb, e0) - time(cb, s0);
                        match p.0.iter_mut().find(|e| e.0 == label) {
                            Some(e) => {
                                e.1 += t;
                                e.2 += 1;
                            }
                            None => p.0.push((label, t, 1)),
                        }
                    }
                    p.1 += 1;
                    if p.1.is_multiple_of(32) {
                        let tot: f64 = p.0.iter().map(|e| e.1).sum();
                        let mut v = p.0.clone();
                        v.sort_by(|a, b| b.1.total_cmp(&a.1));
                        let line: Vec<String> = v.iter().map(|e| format!("{} {:.2}", e.0, e.1 * 1e3 / p.1 as f64)).collect();
                        eprintln!(
                            "[ds41 gpu profile, ms per step over {} steps: total {:.2}; {}]",
                            p.1,
                            tot * 1e3 / p.1 as f64,
                            line.join(", ")
                        );
                    }
                }
                g = time(cbs.last().unwrap().0, e0) - time(cbs[0].0, s0);
            }
            (be.sels.pool_pop)(pool);
        }
        (ok, g)
    }

    fn d(&self, e: &FastEnc, p: usize, groups: (usize, usize), threads: usize) {
        e.pipe(self.pipes[p]);
        e.dispatch(groups, threads);
    }

    #[allow(clippy::too_many_arguments)]
    fn mv(&self, e: &FastEnc, m: &Mv, w2: Option<&Mv>, x: (Id, usize), y: (Id, usize), mode: MvMode, gate: (Id, usize)) {
        encode_mv(e, m.pipe, m.nr, m.ks, &m.w, w2.map(|w| &w.w), x, y, mode, gate);
    }

    /// The hyper-connection weights of sublayer `j` of layer `l` from the copies at `x` (two
    /// stages: the dot products, then the weights; call [`Self::mix_fin`] after a barrier).
    fn mix_dots(&self, e: &FastEnc, l: usize, j: usize, x: usize) {
        let a = self.arena;
        let lp = &self.layers[l];
        e.buf(lp.hc_f[j], 0, 0);
        e.buf(a, x, 1);
        e.buf(a, self.off.part, 2);
        e.bytes(
            &DotsP {
                len: (self.n * self.h) as u32,
                rows: ((2 + self.n) * self.n) as u32,
                split: SPLIT as u32,
                _p: 0,
            },
            3,
        );
        self.d(e, P_DOTS, ((2 + self.n) * self.n + 1, SPLIT), 256);
    }

    fn mix_fin(&self, e: &FastEnc, l: usize, j: usize) {
        let (a, c) = (self.arena, self.consts);
        let lp = &self.layers[l];
        e.buf(a, self.off.part, 0);
        e.buf(c, lp.c_hc[j].1, 1);
        e.buf(c, lp.c_hc[j].2, 2);
        e.buf(a, self.off.mix[l][j], 3);
        e.bytes(
            &MixP {
                n: self.n as u32,
                split: SPLIT as u32,
                len: (self.n * self.h) as u32,
                iters: self.sinkhorn as u32,
                eps: self.eps,
                heps: self.heps,
                _p: [0; 2],
            },
            4,
        );
        self.d(e, P_MIX, (1, 1), 32);
    }

    /// `out = rmsnorm(sum_k pre[k] x[k]) * w`.
    fn collapse(&self, e: &FastEnc, x: usize, pre: (Id, usize), w: usize, out: usize) {
        let a = self.arena;
        e.buf(a, x, 0);
        e.buf(pre.0, pre.1, 1);
        e.buf(self.consts, w, 2);
        e.buf(a, out, 3);
        e.bytes(
            &CollP {
                h: self.h as u32,
                n: self.n as u32,
                eps: self.eps,
                _p: 0,
            },
            4,
        );
        self.d(e, P_COLL, (1, 1), 1024);
    }

    /// Rows of `d`: norm (with weight `w` when given) and / or the rotation by table `table` of
    /// the step's angles.
    #[allow(clippy::too_many_arguments)]
    fn norm_rope(
        &self,
        e: &FastEnc,
        src: (Id, usize),
        dst: (Id, usize),
        d: usize,
        rows: usize,
        norm: bool,
        w: Option<usize>,
        table: Option<usize>,
    ) {
        e.buf(src.0, src.1, 0);
        e.buf(dst.0, dst.1, 1);
        e.buf(self.consts, w.unwrap_or(0), 2);
        e.buf(self.arena, self.off.cs + table.unwrap_or(0) * 2 * self.half * 4, 3);
        e.bytes(
            &NrP {
                d: d as u32,
                flags: norm as u32 | ((w.is_some() as u32) << 1),
                half: if table.is_some() { self.half as u32 } else { 0 },
                ss: d as u32,
                ds: d as u32,
                eps: self.eps,
                _p: [0; 2],
            },
            4,
        );
        self.d(e, P_NR, (rows, 1), 256);
    }

    fn encode_layer(&self, r: &mut Rec, st: &State, l: usize, pos: usize) {
        let lp = &self.layers[l];
        let (a, c) = (self.arena, self.consts);
        let (h, n, nh, k) = (self.h, self.n, self.nh, self.k);
        let o = &self.off;
        let mut xin = o.x[l];
        // Rotary tables: 0 main at pos, 1 compressed at pos, 2 compressed at pos - 1.
        let tq = usize::from(lp.ratio > 0);
        if let (Some(li), Some(w)) = (lp.eng, &lp.eng_wkv) {
            r.barrier("engram");
            let e = r.e();
            self.mv(
                &e,
                w,
                None,
                (a, o.eng_in + li * self.eng_in * 4),
                (a, o.eng_kv),
                MvMode::Store,
                (a, 0),
            );
            r.barrier("engram");
            let e = r.e();
            e.buf(a, xin, 0);
            e.buf(a, o.eng_kv, 1);
            e.buf(c, lp.c_eng_w, 2);
            e.buf(a, o.xe, 3);
            e.bytes(
                &EngP {
                    h: h as u32,
                    n: n as u32,
                    eps: self.eps,
                    _p: 0,
                },
                4,
            );
            self.d(&e, P_ENG, (n, 1), 1024);
            xin = o.xe;
        }
        r.barrier("mix");
        let pre_prev = if l == 0 { (c, o.one_hot) } else { (a, o.mix[l - 1][1]) };
        {
            let e = r.e();
            self.mix_dots(&e, l, 0, xin);
            self.collapse(&e, xin, pre_prev, lp.c_attn_norm, o.xa);
        }
        r.barrier("attn in-proj");
        // In-projections, beside the mixing weights.
        let src = lp.src;
        let rate = lp.ratio;
        let completes = src.is_some() && (pos + 1).is_multiple_of(rate.max(1));
        {
            let e = r.e();
            self.mix_fin(&e, l, 0);
            self.mv(&e, &lp.wq_a, None, (a, o.xa), (a, o.qa), MvMode::Store, (a, 0));
            self.mv(&e, &lp.wkv, None, (a, o.xa), (a, o.kvr), MvMode::Store, (a, 0));
            if let Some(s) = src {
                let slot = pos % rate.max(1);
                let pend = st.pend[s];
                if let Some(w) = &lp.comp_wkv {
                    self.mv(&e, w, None, (a, o.xa), (pend, slot * HD * 4), MvMode::Store, (a, 0));
                }
                if let Some(w) = &lp.comp_wgate {
                    self.mv(&e, w, None, (a, o.xa), (pend, (rate + slot) * HD * 4), MvMode::Store, (a, 0));
                }
            }
            if let Some(w) = &lp.idx_proj {
                self.mv(&e, w, None, (a, o.xa), (a, o.iw), MvMode::Store, (a, 0));
            }
        }
        r.barrier("attn norms");
        {
            let e = r.e();
            self.norm_rope(&e, (a, o.qa), (a, o.qa), self.ql, 1, true, Some(lp.c_q_norm), None);
            let ring_row = ((l * self.window) + pos % self.window) * HD * 4;
            self.norm_rope(&e, (a, o.kvr), (st.ring, ring_row), HD, 1, true, Some(lp.c_kv_norm), Some(tq));
            if completes {
                let s = src.unwrap();
                e.buf(st.pend[s], 0, 0);
                e.buf(st.pend[s], rate * HD * 4, 1);
                e.buf(c, lp.c_comp_norm, 2);
                e.buf(a, o.lat, 3);
                e.bytes(
                    &PoolP {
                        d: HD as u32,
                        r: rate as u32,
                        eps: self.eps,
                        _p: 0,
                    },
                    4,
                );
                self.d(&e, P_POOL, (1, 1), 1024);
            }
        }
        r.barrier("wq_b");
        // The indexer chooses when its query sees more compressed keys than it keeps.
        let choose = lp.isrc.is_some() && (pos + 1) / rate.max(1) > self.topk;
        // The group's key stands for its first position.
        let tw0 = if rate > 1 { 2 } else { 1 };
        {
            let e = r.e();
            self.mv(&e, &lp.wq_b, None, (a, o.qa), (a, o.q), MvMode::Store, (a, 0));
            if completes {
                let s = src.unwrap();
                let row = (pos + 1) / rate - 1;
                if let Some(w) = &lp.idx_wk {
                    self.mv(&e, w, None, (a, o.lat), (a, o.ik), MvMode::Store, (a, 0));
                }
                self.norm_rope(&e, (a, o.lat), (st.comp[s].0, row * HD * 4), HD, 1, false, None, Some(tw0));
            }
            if choose && let Some(w) = &lp.idx_wq {
                self.mv(&e, w, None, (a, o.qa), (a, o.iq), MvMode::Store, (a, 0));
            }
        }
        if completes && lp.idx_wk.is_some() {
            r.barrier("key source");
            let e = r.e();
            let s = src.unwrap();
            let row = (pos + 1) / rate - 1;
            self.norm_rope(
                &e,
                (a, o.ik),
                (st.comp[s].1, row * DI * 4),
                DI,
                1,
                true,
                Some(lp.c_k_norm),
                Some(tw0),
            );
        }
        // Index source: scores over the key source's index keys, the top k (and, for the
        // candidate source, the blocks the later index layers choose in).
        if choose {
            let ks = self.layers[lp.kv_src.unwrap()].src.unwrap();
            let isrc = lp.isrc.unwrap();
            let nv = (pos + 1) / rate;
            let cand = self.cand.filter(|&(cl, tb, bs)| cl < l && nv.div_ceil(bs) > tb);
            let makes = self.cand.filter(|&(cl, tb, bs)| cl == l && nv.div_ceil(bs) > tb);
            r.barrier("indexer");
            let e = r.e();
            e.buf(a, o.iq, 0);
            e.buf(a, o.iw, 1);
            e.buf(st.comp[ks].1, 0, 2);
            e.buf(a, o.cs + tq * 2 * self.half * 4, 3);
            e.buf(st.mask, 0, 4);
            e.buf(st.scores, 0, 5);
            e.bytes(
                &IdxP {
                    nv: nv as u32,
                    half: self.half as u32,
                    ws: (DI as f32).powf(-0.5) * (NI as f32).powf(-0.5),
                    bs: cand.map_or(1, |c| c.2) as u32,
                    has_mask: cand.is_some() as u32,
                    _p: [0; 3],
                },
                6,
            );
            self.d(&e, P_IDX, (nv.div_ceil(64), 1), 256);
            r.barrier("indexer");
            let e = r.e();
            e.buf(st.scores, 0, 0);
            e.buf(a, o.isel + isrc * self.topk * 4, 1);
            e.bytes(
                &TopkP {
                    n: nv as u32,
                    k: self.topk as u32,
                    _p: [0; 2],
                },
                2,
            );
            self.d(&e, P_TOPK, (1, 1), 1024);
            if let Some((_, tb, bs)) = makes {
                let nb = nv.div_ceil(bs);
                e.buf(st.scores, 0, 0);
                e.buf(st.blk, 0, 1);
                e.bytes(
                    &BlkP {
                        nv: nv as u32,
                        bs: bs as u32,
                        nb: nb as u32,
                        _p: 0,
                    },
                    2,
                );
                self.d(&e, P_BLK, (nb.div_ceil(256), 1), 256);
                r.barrier("indexer");
                let e = r.e();
                e.buf(st.blk, 0, 0);
                e.buf(a, o.cand, 1);
                e.bytes(
                    &TopkP {
                        n: nb as u32,
                        k: tb as u32,
                        _p: [0; 2],
                    },
                    2,
                );
                self.d(&e, P_TOPK, (1, 1), 1024);
                r.barrier("indexer");
                let e = r.e();
                e.buf(a, o.cand, 0);
                e.buf(st.mask, 0, 1);
                e.bytes(
                    &MaskP {
                        nb: nb as u32,
                        k: tb as u32,
                        _p: [0; 2],
                    },
                    2,
                );
                self.d(&e, P_MASK, (1, 1), 1024);
            }
        }
        // Attention.
        r.barrier("attend");
        let lo = (pos + 1).saturating_sub(self.window);
        let nw = pos + 1 - lo;
        let (comp_buf, nc, use_sel, sel_off) = match lp.kv_src {
            Some(kvs) => {
                let s = self.layers[kvs].src.unwrap();
                let srate = self.layers[kvs].ratio;
                let nvis = (pos + 1) / srate;
                // The choice of the last index source, when it chose.
                let chosen = lp.sel_from.filter(|_| nvis > self.topk);
                match chosen {
                    Some(is) => (st.comp[s].0, self.topk, 1u32, o.isel + is * self.topk * 4),
                    None => (st.comp[s].0, nvis, 0, o.isel),
                }
            }
            None => (st.ring, 0, 0, o.isel),
        };
        let total = nw + nc;
        let splits = total.div_ceil(CHUNK);
        let ap = AttnP {
            nh: nh as u32,
            nw: nw as u32,
            lo: lo as u32,
            win: self.window as u32,
            nc: nc as u32,
            use_sel,
            chunk: CHUNK as u32,
            half: self.half as u32,
            scale: (HD as f32).powf(-0.5),
            splits: splits as u32,
            _p: [0; 2],
        };
        {
            let e = r.e();
            e.buf(a, o.q, 0);
            e.buf(st.ring, l * self.window * HD * 4, 1);
            e.buf(comp_buf, 0, 2);
            e.buf(a, sel_off, 3);
            e.buf(a, o.cs + tq * 2 * self.half * 4, 4);
            e.buf(a, o.apart, 5);
            e.bytes(&ap, 6);
            self.d(&e, P_ATTN, (nh / 8, splits), 256);
        }
        r.barrier("attend");
        {
            let e = r.e();
            e.buf(a, o.apart, 0);
            e.buf(c, lp.c_sink, 1);
            e.buf(a, o.cs + tq * 2 * self.half * 4, 2);
            e.buf(a, o.att, 3);
            e.bytes(&ap, 4);
            self.d(&e, P_COMB, (nh, 1), 256);
        }
        r.barrier("attn out-proj");
        {
            let e = r.e();
            let gd = nh * HD / self.groups;
            for (g, w) in lp.wo_a.iter().enumerate() {
                self.mv(
                    &e,
                    w,
                    None,
                    (a, o.att + g * gd * 4),
                    (a, o.og + g * self.o_lora * 4),
                    MvMode::Store,
                    (a, 0),
                );
            }
        }
        r.barrier("attn out-proj");
        {
            let e = r.e();
            self.mv(&e, &lp.wo_b, None, (a, o.og), (a, o.o1), MvMode::Store, (a, 0));
        }
        r.barrier("hc post");
        {
            let e = r.e();
            self.hc_post(&e, xin, o.o1, None, o.mix[l][0], o.xa_res);
        }
        // Feed-forward.
        r.barrier("mix");
        {
            let e = r.e();
            self.mix_dots(&e, l, 1, o.xa_res);
            self.collapse(&e, o.xa_res, (a, o.mix[l][0]), lp.c_ffn_norm, o.xf);
        }
        r.barrier("router");
        {
            let e = r.e();
            self.mix_fin(&e, l, 1);
            self.mv(&e, &lp.router, None, (a, o.xf), (a, o.rlog), MvMode::Store, (a, 0));
            let mode = if lp.shared_clamp { MvMode::SiluClampPair } else { MvMode::SiluPair };
            self.mv(
                &e,
                &lp.sh[0],
                Some(&lp.sh[1]),
                (a, o.xf),
                (a, o.act_s),
                mode,
                (c, lp.c_shared_limit),
            );
        }
        r.barrier("router");
        {
            let e = r.e();
            e.buf(a, o.rlog, 0);
            e.buf(c, lp.c_bias.unwrap_or(0), 1);
            e.buf(lp.tabs[0], 0, 2);
            e.buf(a, o.sel[l], 3);
            e.buf(a, o.miss, 4);
            e.bytes(
                &RouteP {
                    n: self.n_exp as u32,
                    k: k as u32,
                    rsf: self.rsf,
                    limit: self.limit,
                    scoring: self.scoring,
                    norm: self.norm_topk as u32,
                    layer: l as u32,
                    has_bias: lp.c_bias.is_some() as u32,
                },
                5,
            );
            self.d(&e, P_ROUTE, (1, 1), 32);
            self.mv(&e, &lp.sh[2], None, (a, o.act_s), (a, o.y2), MvMode::Store, (a, 0));
        }
        r.barrier("experts");
        {
            let e = r.e();
            // (DeepSeek's experts always clamp: `Act::SiluClamp`.)
            let mode = MvMode::SiluClampPair;
            let p = super::decode::MvParams {
                n_out: self.inter as u32,
                n_in: h as u32,
                mode: mode as u32,
                rows_per_tg: 1,
                gpr: lp.gpr[0],
                ldy: k as u32,
                ..Default::default()
            };
            e.pipe(lp.pair);
            e.buf(lp.tabs[0], 0, 0);
            e.buf(lp.tabs[1], 0, 1);
            e.buf(a, o.xf, 2);
            e.buf(a, o.act, 3);
            e.bytes(&p, 4);
            e.buf(a, o.sel[l], 5);
            e.dispatch((self.inter, k), 32);
        }
        r.barrier("experts");
        {
            let e = r.e();
            let p = super::decode::MvParams {
                n_out: h as u32,
                n_in: self.inter as u32,
                mode: MvMode::Store as u32,
                rows_per_tg: 1,
                gpr: lp.gpr[1],
                ldy: k as u32,
                ..Default::default()
            };
            e.pipe(lp.sum);
            e.buf(lp.tabs[2], 0, 0);
            e.buf(lp.tabs[2], 0, 1);
            e.buf(a, o.act, 2);
            e.buf(a, o.y1, 3);
            e.bytes(&p, 4);
            e.buf(a, o.sel[l], 5);
            e.dispatch((h, 1), 32);
        }
        r.barrier("hc post");
        {
            let e = r.e();
            self.hc_post(&e, o.xa_res, o.y1, Some(o.y2), o.mix[l][1], o.x[l + 1]);
        }
    }

    fn hc_post(&self, e: &FastEnc, xin: usize, o1: usize, o2: Option<usize>, mix: usize, xout: usize) {
        let a = self.arena;
        e.buf(a, xin, 0);
        e.buf(a, o1, 1);
        e.buf(a, o2.unwrap_or(o1), 2);
        e.buf(a, mix, 3);
        e.buf(a, xout, 4);
        e.bytes(
            &PostP {
                h: self.h as u32,
                n: self.n as u32,
                has_o2: o2.is_some() as u32,
                _p: 0,
            },
            5,
        );
        self.d(e, P_POST, (self.h.div_ceil(256), 1), 256);
    }

    fn encode_head(&self, r: &mut Rec, hidden_only: bool) {
        let (a, o) = (self.arena, &self.off);
        let nl = self.layers.len();
        r.barrier("head");
        {
            let e = r.e();
            self.collapse(&e, o.x[nl], (a, o.mix[nl - 1][1]), self.c_norm, o.hid);
        }
        if !hidden_only {
            r.barrier("head");
            let e = r.e();
            self.mv(&e, &self.head, None, (a, o.hid), (a, o.logits), MvMode::Store, (a, 0));
        }
    }
}

/// The host's inputs of a step.
pub(crate) struct StepIn<'s> {
    pub pos: usize,
    /// The input embedding (`h`).
    pub x: &'s [f32],
    /// Rotary cos then sin (`half` each) of: the main table at `pos`, the compressed table at
    /// `pos`, the compressed table at `pos - 1`.
    pub cs: &'s [f32],
    /// Per engram layer, its rows (`eng_in` floats).
    pub eng: &'s [f32],
}

/// The command buffers of a step being encoded: barriers between dependent stages, cuts into a
/// new command buffer (each stage in one of its own when profiling, labelled).
struct Rec<'r, 'f> {
    f: &'f Fns,
    cur: &'r mut (Id, Id),
    cbs: &'r mut Vec<(Id, &'static str)>,
    prof: bool,
    label: &'static str,
    new_enc: &'r dyn Fn() -> (Id, Id),
}

impl Rec<'_, '_> {
    fn e(&self) -> FastEnc<'_> {
        FastEnc {
            enc: self.cur.1,
            f: self.f,
        }
    }
    /// The next stage depends on the ones before.
    fn barrier(&mut self, next: &'static str) {
        if self.prof {
            self.cut();
        } else {
            self.e().barrier();
        }
        self.label = next;
    }
    /// Commits the command buffer and opens another.
    fn cut(&mut self) {
        let (cb, enc) = *self.cur;
        unsafe {
            (self.f.void)(enc, self.f.s_end);
            (self.f.void)(cb, self.f.s_commit);
        }
        self.cbs.push((cb, self.label));
        *self.cur = (self.new_enc)();
    }
}

impl Drop for Rec<'_, '_> {
    fn drop(&mut self) {
        // The last (empty) encoder opened by a cut.
        let (cb, enc) = *self.cur;
        unsafe {
            (self.f.void)(enc, self.f.s_end);
            (self.f.void)(cb, self.f.s_commit);
        }
    }
}

/// Each kernel against a reference written out on the CPU (in f64 where sums matter).
#[cfg(test)]
mod tests {
    use super::*;

    struct Kit {
        be: &'static MetalBackend,
        f: Fns,
        pipes: Vec<Id>,
    }

    fn kit() -> Option<Kit> {
        // A backend of its own: the process-wide one would put the other tests on the GPU.
        let be: &'static MetalBackend = Box::leak(Box::new(MetalBackend::new().ok()?));
        Some(Kit {
            be,
            f: Fns::new(be),
            pipes: compile(be).expect("the kernels build"),
        })
    }

    impl Kit {
        fn buf(&self, v: &[f32]) -> Id {
            let b = new_buf(self.be, v.len() * 4).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), contents(self.be, b) as *mut f32, v.len()) };
            b
        }
        fn bufu(&self, v: &[u32]) -> Id {
            let f: Vec<f32> = v.iter().map(|&x| f32::from_bits(x)).collect();
            self.buf(&f)
        }
        fn read(&self, b: Id, n: usize) -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(contents(self.be, b) as *const f32, n).to_vec() }
        }
        fn run(&self, enc: impl Fn(&FastEnc)) {
            let (ok, _) = super::super::decode::run_concurrent(self.be, &self.f, std::slice::from_ref(&(0..1)), |e, _| enc(e));
            assert!(ok, "the command buffer failed");
        }
        fn go(&self, e: &FastEnc, p: usize, bufs: &[Id], groups: (usize, usize), threads: usize) {
            e.pipe(self.pipes[p]);
            for (i, &b) in bufs.iter().enumerate() {
                e.buf(b, 0, i);
            }
            e.dispatch(groups, threads);
        }
    }

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
    }

    fn close(a: &[f32], b: &[f64], tol: f64, what: &str) {
        let scale = b.iter().fold(1e-30f64, |m, x| m.max(x.abs()));
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(((x as f64) - y).abs() <= tol * scale, "{what}[{i}]: {x} vs {y}");
        }
    }

    /// The k largest (ties to the lower position) in position order, as `top_positions`.
    fn top(s: &[f32], k: usize) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..s.len() as u32).collect();
        idx.sort_by(|&a, &b| s[b as usize].total_cmp(&s[a as usize]).then(a.cmp(&b)));
        idx.truncate(k);
        idx.sort_unstable();
        idx
    }

    #[test]
    fn topk_matches_cpu() {
        let Some(k) = kit() else { return };
        let mut r = rng(3);
        for (n, kk) in [(7usize, 5usize), (600, 512), (5000, 512), (40000, 2048)] {
            // Many ties (values on a coarse grid), some -inf.
            let s: Vec<f32> = (0..n)
                .map(|i| {
                    if i % 97 == 5 {
                        f32::NEG_INFINITY
                    } else {
                        (r() * 20.0).round() / 4.0
                    }
                })
                .collect();
            let (sb, ob) = (k.buf(&s), k.buf(&vec![0f32; kk]));
            let p = k.buf(&[f32::from_bits(n as u32), f32::from_bits(kk as u32), 0.0, 0.0]);
            k.run(|e| k.go(e, P_TOPK, &[sb, ob, p], (1, 1), 1024));
            let got: Vec<u32> = k.read(ob, kk).iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, top(&s, kk), "n {n}, k {kk}");
        }
    }

    #[test]
    fn mix_matches_cpu() {
        let Some(k) = kit() else { return };
        let mut r = rng(5);
        let (n, h) = (4usize, 512usize);
        let m = (2 + n) * n;
        let x: Vec<f32> = (0..n * h).map(|_| r()).collect();
        let w: Vec<f32> = (0..m * n * h).map(|_| r() * 0.05).collect();
        let base: Vec<f32> = (0..m).map(|_| r()).collect();
        let scale = [0.7f32, 0.9, 1.1];
        let (xb, wb, pb, bb, sb, ob) = (
            k.buf(&x),
            k.buf(&w),
            k.buf(&vec![0f32; (m + 1) * SPLIT]),
            k.buf(&base),
            k.buf(&scale),
            k.buf(&vec![0f32; m]),
        );
        let dp = k.buf(&[(n * h) as u32, m as u32, SPLIT as u32, 0].map(f32::from_bits));
        let (eps, heps, iters) = (1e-6f32, 1e-6f32, 3u32);
        let mp = k.buf(&[
            f32::from_bits(n as u32),
            f32::from_bits(SPLIT as u32),
            f32::from_bits((n * h) as u32),
            f32::from_bits(iters),
            eps,
            heps,
            0.0,
            0.0,
        ]);
        k.run(|e| {
            k.go(e, P_DOTS, &[wb, xb, pb, dp], (m + 1, SPLIT), 256);
            e.barrier();
            k.go(e, P_MIX, &[pb, bb, sb, ob, mp], (1, 1), 32);
        });
        // hc_split_sinkhorn in f64.
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&p, &q)| p as f64 * q as f64).sum::<f64>();
        let rs = 1.0 / (dot(&x, &x) / (n * h) as f64 + eps as f64).sqrt();
        let mm: Vec<f64> = (0..m).map(|i| dot(&w[i * n * h..(i + 1) * n * h], &x) * rs).collect();
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        let (sc, bs, he) = (scale.map(|v| v as f64), &base, heps as f64);
        let mut want = vec![0f64; m];
        for i in 0..n {
            want[i] = sig(mm[i] * sc[0] + bs[i] as f64) + he;
            want[n + i] = 2.0 * sig(mm[n + i] * sc[1] + bs[n + i] as f64);
        }
        let mut c: Vec<f64> = (0..n * n).map(|i| mm[2 * n + i] * sc[2] + bs[2 * n + i] as f64).collect();
        for row in c.chunks_mut(n) {
            let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let s: f64 = row.iter().map(|v| (v - mx).exp()).sum();
            for v in row.iter_mut() {
                *v = (*v - mx).exp() / s + he;
            }
        }
        for it in 0..iters {
            if it > 0 {
                for row in c.chunks_mut(n) {
                    let s: f64 = row.iter().sum::<f64>() + he;
                    row.iter_mut().for_each(|v| *v /= s);
                }
            }
            for col in 0..n {
                let s: f64 = (0..n).map(|j| c[j * n + col]).sum::<f64>() + he;
                (0..n).for_each(|j| c[j * n + col] /= s);
            }
        }
        want[2 * n..].copy_from_slice(&c);
        close(&k.read(ob, m), &want, 1e-5, "mix");
    }

    #[test]
    fn collapse_post_norm_rope_pool_match_cpu() {
        let Some(k) = kit() else { return };
        let mut r = rng(7);
        let (n, h) = (4usize, 1280usize);
        let x: Vec<f32> = (0..n * h).map(|_| r()).collect();
        let pre = [0.3f32, 1.2, -0.4, 0.8];
        let w: Vec<f32> = (0..h).map(|_| 1.0 + 0.3 * r()).collect();
        let eps = 1e-6f32;
        let (xb, prb, wb, ob) = (k.buf(&x), k.buf(&pre), k.buf(&w), k.buf(&vec![0f32; h]));
        let cp = k.buf(&[f32::from_bits(h as u32), f32::from_bits(n as u32), eps, 0.0]);
        // hc_post: mix = pre [n], post [n], comb [n * n].
        let mix: Vec<f32> = (0..(2 + n) * n).map(|_| r()).collect();
        let (o1, o2): (Vec<f32>, Vec<f32>) = ((0..h).map(|_| r()).collect(), (0..h).map(|_| r()).collect());
        let (mb, o1b, o2b, xo) = (k.buf(&mix), k.buf(&o1), k.buf(&o2), k.buf(&vec![0f32; n * h]));
        let pp = k.buf(&[h as u32, n as u32, 1, 0].map(f32::from_bits));
        // norm_rope: 3 rows of 512, weighted norm then the last 64 rotated.
        let (d, half) = (512usize, 32usize);
        let rows: Vec<f32> = (0..3 * d).map(|_| r()).collect();
        let nw: Vec<f32> = (0..d).map(|_| 1.0 + 0.2 * r()).collect();
        let cs: Vec<f32> = (0..half)
            .map(|i| (i as f32 * 0.3).cos())
            .chain((0..half).map(|i| (i as f32 * 0.3).sin()))
            .collect();
        let (rb, nwb, csb, rob) = (k.buf(&rows), k.buf(&nw), k.buf(&cs), k.buf(&vec![0f32; 3 * d]));
        let np = k.buf(
            &[d as u32, 3, half as u32, d as u32, d as u32]
                .map(f32::from_bits)
                .into_iter()
                .chain([eps, 0.0, 0.0])
                .collect::<Vec<_>>(),
        );
        // pool: rate 2.
        let kv: Vec<f32> = (0..2 * d).map(|_| r()).collect();
        let sc: Vec<f32> = (0..2 * d).map(|_| 3.0 * r()).collect();
        let (kvb, scb, plo) = (k.buf(&kv), k.buf(&sc), k.buf(&vec![0f32; d]));
        let plp = k.buf(&[f32::from_bits(d as u32), f32::from_bits(2), eps, 0.0]);
        k.run(|e| {
            k.go(e, P_COLL, &[xb, prb, wb, ob, cp], (1, 1), 1024);
            k.go(e, P_POST, &[xb, o1b, o2b, mb, xo, pp], (h.div_ceil(256), 1), 256);
            k.go(e, P_NR, &[rb, rob, nwb, csb, np], (3, 1), 256);
            k.go(e, P_POOL, &[kvb, scb, nwb, plo, plp], (1, 1), 1024);
        });
        let rms = |v: &[f64]| 1.0 / (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64 + eps as f64).sqrt();
        let y: Vec<f64> = (0..h).map(|i| (0..n).map(|c| pre[c] as f64 * x[c * h + i] as f64).sum()).collect();
        let s = rms(&y);
        close(
            &k.read(ob, h),
            &y.iter().zip(&w).map(|(v, &w)| v * s * w as f64).collect::<Vec<_>>(),
            1e-5,
            "collapse",
        );
        let mut want = vec![0f64; n * h];
        for c in 0..n {
            for i in 0..h {
                want[c * h + i] = mix[n + c] as f64 * (o1[i] + o2[i]) as f64
                    + (0..n).map(|j| mix[2 * n + j * n + c] as f64 * x[j * h + i] as f64).sum::<f64>();
            }
        }
        close(&k.read(xo, n * h), &want, 1e-5, "hc_post");
        let mut want = vec![0f64; 3 * d];
        for row in 0..3 {
            let v: Vec<f64> = rows[row * d..(row + 1) * d].iter().map(|&x| x as f64).collect();
            let s = rms(&v);
            let mut y: Vec<f64> = v.iter().zip(&nw).map(|(x, &w)| x * s * w as f64).collect();
            for i in 0..half {
                let (a, b) = (y[d - 2 * half + 2 * i], y[d - 2 * half + 2 * i + 1]);
                let (c, sn) = (cs[i] as f64, cs[half + i] as f64);
                y[d - 2 * half + 2 * i] = a * c - b * sn;
                y[d - 2 * half + 2 * i + 1] = b * c + a * sn;
            }
            want[row * d..(row + 1) * d].copy_from_slice(&y);
        }
        close(&k.read(rob, 3 * d), &want, 1e-5, "norm_rope");
        let lat: Vec<f64> = (0..d)
            .map(|c| {
                let (a, b) = (sc[c] as f64, sc[d + c] as f64);
                let m = a.max(b);
                let (ea, eb) = ((a - m).exp(), (b - m).exp());
                (ea * kv[c] as f64 + eb * kv[d + c] as f64) / (ea + eb)
            })
            .collect();
        let s = rms(&lat);
        close(
            &k.read(plo, d),
            &lat.iter().zip(&nw).map(|(v, &w)| v * s * w as f64).collect::<Vec<_>>(),
            1e-5,
            "pool",
        );
    }

    #[test]
    fn attention_matches_cpu() {
        let Some(k) = kit() else { return };
        let mut r = rng(11);
        let (nh, win, half) = (16usize, 128usize, 32usize);
        let pos = 300usize;
        let ncomp = 700usize;
        let q: Vec<f32> = (0..nh * HD).map(|_| r()).collect();
        let ring: Vec<f32> = (0..win * HD).map(|_| r()).collect();
        let comp: Vec<f32> = (0..ncomp * HD).map(|_| r()).collect();
        let sink: Vec<f32> = (0..nh).map(|_| 2.0 * r()).collect();
        let sel: Vec<u32> = (0..512u32).map(|i| (i * 7 + 3) % ncomp as u32).collect();
        let cs: Vec<f32> = (0..half)
            .map(|i| (i as f32 * 0.37).cos())
            .chain((0..half).map(|i| (i as f32 * 0.37).sin()))
            .collect();
        let lo = pos + 1 - win;
        let total = win + sel.len();
        let splits = total.div_ceil(CHUNK);
        let ap = AttnP {
            nh: nh as u32,
            nw: win as u32,
            lo: lo as u32,
            win: win as u32,
            nc: sel.len() as u32,
            use_sel: 1,
            chunk: CHUNK as u32,
            half: half as u32,
            scale: (HD as f32).powf(-0.5),
            splits: splits as u32,
            _p: [0; 2],
        };
        let (qb, rb, cb, sb, csb, pb, skb, ob) = (
            k.buf(&q),
            k.buf(&ring),
            k.buf(&comp),
            k.bufu(&sel),
            k.buf(&cs),
            k.buf(&vec![0f32; nh * splits * (HD + 2)]),
            k.buf(&sink),
            k.buf(&vec![0f32; nh * HD]),
        );
        k.run(|e| {
            e.pipe(k.pipes[P_ATTN]);
            for (i, b) in [qb, rb, cb, sb, csb, pb].into_iter().enumerate() {
                e.buf(b, 0, i);
            }
            e.bytes(&ap, 6);
            e.dispatch((nh / 8, splits), 256);
            e.barrier();
            e.pipe(k.pipes[P_COMB]);
            for (i, b) in [pb, skb, csb, ob].into_iter().enumerate() {
                e.buf(b, 0, i);
            }
            e.bytes(&ap, 4);
            e.dispatch((nh, 1), 256);
        });
        let rot = |v: &mut [f64], inv: bool| {
            for i in 0..half {
                let (a, b) = (v[HD - 2 * half + 2 * i], v[HD - 2 * half + 2 * i + 1]);
                let (c, s) = (cs[i] as f64, if inv { -cs[half + i] } else { cs[half + i] } as f64);
                v[HD - 2 * half + 2 * i] = a * c - b * s;
                v[HD - 2 * half + 2 * i + 1] = b * c + a * s;
            }
        };
        let keys: Vec<&[f32]> = (lo..=pos)
            .map(|p| &ring[(p % win) * HD..(p % win + 1) * HD])
            .chain(sel.iter().map(|&w| &comp[w as usize * HD..(w as usize + 1) * HD]))
            .collect();
        let mut want = vec![0f64; nh * HD];
        for hh in 0..nh {
            let mut qh: Vec<f64> = q[hh * HD..(hh + 1) * HD].iter().map(|&v| v as f64).collect();
            rot(&mut qh, false);
            let s: Vec<f64> = keys
                .iter()
                .map(|kr| kr.iter().zip(&qh).map(|(&a, b)| a as f64 * b).sum::<f64>() / (HD as f64).sqrt())
                .collect();
            let m = s.iter().copied().fold(sink[hh] as f64, f64::max);
            let tot = (sink[hh] as f64 - m).exp() + s.iter().map(|v| (v - m).exp()).sum::<f64>();
            let o = &mut want[hh * HD..(hh + 1) * HD];
            for (kr, v) in keys.iter().zip(&s) {
                let p = (v - m).exp() / tot;
                for (od, &kd) in o.iter_mut().zip(kr.iter()) {
                    *od += p * kd as f64;
                }
            }
            rot(o, true);
        }
        close(&k.read(ob, nh * HD), &want, 2e-5, "attention");
    }

    /// The routed experts' pair (clamped SwiGLU) and sum kernels through address tables, on the
    /// grid formats of the real checkpoint, with a zero expert in a slot.
    #[test]
    fn experts_match_cpu() {
        use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
        let Some(k) = kit() else { return };
        let (h, inter, ne, kk) = (512usize, 256usize, 4usize, 3usize);
        for (fg, fd) in [
            (GgmlType::IQ1_S, GgmlType::IQ1_M),
            (GgmlType::IQ2_XXS, GgmlType::IQ1_S),
            (GgmlType::IQ1_M, GgmlType::IQ2_XXS),
        ] {
            let mut seed = 17u64;
            let mats = |kind: GgmlType, n_out: usize, n_in: usize, seed: &mut u64| -> Vec<Vec<u8>> {
                (0..ne).map(|_| random_blocks(kind, n_out * n_in / 256, seed)).collect()
            };
            let (g, u, d) = (
                mats(fg, inter, h, &mut seed),
                mats(fg, inter, h, &mut seed),
                mats(fd, h, inter, &mut seed),
            );
            // One page-aligned allocation of every expert, wrapped once.
            let all: Vec<&Vec<u8>> = g.iter().chain(&u).chain(&d).collect();
            let offs: Vec<usize> = all
                .iter()
                .scan(0usize, |at, v| {
                    let o = *at;
                    *at += v.len().next_multiple_of(256);
                    Some(o)
                })
                .collect();
            let total = offs.last().unwrap() + all.last().unwrap().len();
            let mut mem = crate::quant::PageVec::<u8>::zeroed(total.next_multiple_of(16384));
            for (v, &o) in all.iter().zip(&offs) {
                mem[o..o + v.len()].copy_from_slice(v);
            }
            let (id, off) = k.be.wrap_raw(mem.as_ptr(), mem.len()).unwrap();
            let base = k.be.buffer_gpu_address(id) + off as u64;
            let tab = |j: usize| -> Vec<u32> {
                let mut t = vec![0u32; (ne + 1) * 8];
                for e in 0..ne {
                    let a = base + offs[j * ne + e] as u64;
                    for q in 0..4 {
                        t[e * 8 + 2 * q] = a as u32;
                        t[e * 8 + 2 * q + 1] = (a >> 32) as u32;
                    }
                }
                t
            };
            let (tg, tu, td) = (k.bufu(&tab(0)), k.bufu(&tab(1)), k.bufu(&tab(2)));
            let mut r = rng(19);
            let x: Vec<f32> = (0..h).map(|_| r()).collect();
            let ids = [2u32, 0, 3];
            let w = [0.7f32, -0.4, 1.3];
            let limit = 0.9f32;
            let sel: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).chain(w).chain([limit]).collect();
            let (xb, sb, ab, yb) = (k.buf(&x), k.buf(&sel), k.buf(&vec![0f32; kk * inter]), k.buf(&vec![0f32; h]));
            let pipe = |n: String| k.be.named_pipeline(&n).unwrap();
            let (pair, sum) = (pipe(format!("mvx_{}_pair", fg.name())), pipe(format!("mvx_{}_sum", fd.name())));
            let p1 = super::super::decode::MvParams {
                n_out: inter as u32,
                n_in: h as u32,
                mode: MvMode::SiluClampPair as u32,
                rows_per_tg: 1,
                gpr: fg.row_bytes(h) as u32,
                ldy: kk as u32,
                ..Default::default()
            };
            let p2 = super::super::decode::MvParams {
                n_out: h as u32,
                n_in: inter as u32,
                mode: MvMode::Store as u32,
                rows_per_tg: 1,
                gpr: fd.row_bytes(inter) as u32,
                ldy: kk as u32,
                ..Default::default()
            };
            k.run(|e| {
                e.use_resources(&[id]);
                e.pipe(pair);
                for (i, b) in [tg, tu, xb, ab].into_iter().enumerate() {
                    e.buf(b, 0, i);
                }
                e.bytes(&p1, 4);
                e.buf(sb, 0, 5);
                e.dispatch((inter, kk), 32);
                e.barrier();
                e.pipe(sum);
                for (i, b) in [td, td, ab, yb].into_iter().enumerate() {
                    e.buf(b, 0, i);
                }
                e.bytes(&p2, 4);
                e.buf(sb, 0, 5);
                e.dispatch((h, 1), 32);
            });
            let deq = |kind: GgmlType, b: &[u8], n_out: usize, n_in: usize| {
                let mut o = vec![0f32; n_out * n_in];
                GgufMatrix::from_bytes(kind, b, n_out, n_in)
                    .unwrap()
                    .dequantize_rows(0, n_out, &mut o);
                o
            };
            let mut want = vec![0f64; h];
            for (s, &e) in ids.iter().enumerate() {
                let e = e as usize;
                let (gm, um, dm) = (deq(fg, &g[e], inter, h), deq(fg, &u[e], inter, h), deq(fd, &d[e], h, inter));
                let act: Vec<f64> = (0..inter)
                    .map(|i| {
                        let a: f64 = (0..h).map(|j| gm[i * h + j] as f64 * x[j] as f64).sum();
                        let b: f64 = (0..h).map(|j| um[i * h + j] as f64 * x[j] as f64).sum();
                        let a = a.min(limit as f64);
                        a / (1.0 + (-a).exp()) * b.clamp(-limit as f64, limit as f64)
                    })
                    .collect();
                close(
                    &k.read(ab, kk * inter)[s * inter..(s + 1) * inter],
                    &act,
                    1e-4,
                    &format!("{fg} pair, slot {s}"),
                );
                for (o, wo) in want.iter_mut().enumerate() {
                    *wo += w[s] as f64 * (0..inter).map(|i| dm[o * inter + i] as f64 * act[i]).sum::<f64>();
                }
            }
            close(&k.read(yb, h), &want, 1e-4, &format!("{fd} sum"));
            release(k.be, id);
        }
    }

    #[test]
    fn indexer_and_route_and_engram_match_cpu() {
        let Some(k) = kit() else { return };
        let mut r = rng(13);
        // Indexer scores over 1000 keys, blocks of 8 masked.
        let (nv, half, bs) = (1000usize, 32usize, 8usize);
        let iq: Vec<f32> = (0..NI * DI).map(|_| r()).collect();
        let iw: Vec<f32> = (0..NI).map(|_| r()).collect();
        let keys: Vec<f32> = (0..nv * DI).map(|_| r()).collect();
        let mask: Vec<u32> = (0..nv.div_ceil(bs) as u32).map(|b| (b % 3 != 1) as u32).collect();
        let cs: Vec<f32> = (0..half)
            .map(|i| (i as f32 * 0.21).cos())
            .chain((0..half).map(|i| (i as f32 * 0.21).sin()))
            .collect();
        let ws = 0.0123f32;
        let ip = IdxP {
            nv: nv as u32,
            half: half as u32,
            ws,
            bs: bs as u32,
            has_mask: 1,
            _p: [0; 3],
        };
        let (qb, wb, kb, csb, mb, sb) = (
            k.buf(&iq),
            k.buf(&iw),
            k.buf(&keys),
            k.buf(&cs),
            k.bufu(&mask),
            k.buf(&vec![0f32; nv]),
        );
        // Routing: 384 experts, sqrt(softplus), bias, top 6, a few experts without an address.
        let (ne, kk) = (384usize, 6usize);
        let lg: Vec<f32> = (0..ne).map(|_| 3.0 * r()).collect();
        let bias: Vec<f32> = (0..ne).map(|_| 0.5 * r()).collect();
        let mut tab = vec![0f32; (ne + 1) * 8];
        let missing = |e: usize| e % 5 == 2;
        for e in 0..=ne {
            if !missing(e) {
                tab[e * 8] = f32::from_bits(1);
            }
        }
        let (lb, bb, tb, selb, missb) = (
            k.buf(&lg),
            k.buf(&bias),
            k.buf(&tab),
            k.buf(&vec![0f32; 2 * kk + 1]),
            k.buf(&vec![0f32; 1 + kk]),
        );
        let rp = RouteP {
            n: ne as u32,
            k: kk as u32,
            rsf: 1.5,
            limit: 7.0,
            scoring: 2,
            norm: 1,
            layer: 9,
            has_bias: 1,
        };
        // Engram: 4 copies of 1024.
        let (n, h) = (4usize, 1024usize);
        let x: Vec<f32> = (0..n * h).map(|_| r()).collect();
        let kv: Vec<f32> = (0..(n + 1) * h).map(|_| r()).collect();
        let gw: Vec<f32> = (0..n * h).map(|_| 1.0 + 0.3 * r()).collect();
        let (xb, kvb, gwb, xo) = (k.buf(&x), k.buf(&kv), k.buf(&gw), k.buf(&vec![0f32; n * h]));
        let eps = 1e-6f32;
        let ep = EngP {
            h: h as u32,
            n: n as u32,
            eps,
            _p: 0,
        };
        k.run(|e| {
            e.pipe(k.pipes[P_IDX]);
            for (i, b) in [qb, wb, kb, csb, mb, sb].into_iter().enumerate() {
                e.buf(b, 0, i);
            }
            e.bytes(&ip, 6);
            e.dispatch((nv.div_ceil(64), 1), 256);
            e.pipe(k.pipes[P_ROUTE]);
            for (i, b) in [lb, bb, tb, selb, missb].into_iter().enumerate() {
                e.buf(b, 0, i);
            }
            e.bytes(&rp, 5);
            e.dispatch((1, 1), 32);
            e.pipe(k.pipes[P_ENG]);
            for (i, b) in [xb, kvb, gwb, xo].into_iter().enumerate() {
                e.buf(b, 0, i);
            }
            e.bytes(&ep, 4);
            e.dispatch((n, 1), 1024);
        });
        // Indexer.
        let mut qr: Vec<f64> = iq.iter().map(|&v| v as f64).collect();
        for hh in 0..NI {
            for i in 0..half {
                let j = hh * DI + DI - 2 * half + 2 * i;
                let (a, b) = (qr[j], qr[j + 1]);
                let (c, s) = (cs[i] as f64, cs[half + i] as f64);
                qr[j] = a * c - b * s;
                qr[j + 1] = b * c + a * s;
            }
        }
        let got = k.read(sb, nv);
        let mut scale = 0f64;
        let want: Vec<f64> = (0..nv)
            .map(|w| {
                if mask[w / bs] == 0 {
                    return f64::NEG_INFINITY;
                }
                let s: f64 = (0..NI)
                    .map(|hh| {
                        let d: f64 = (0..DI).map(|i| qr[hh * DI + i] * keys[w * DI + i] as f64).sum();
                        iw[hh] as f64 * ws as f64 * d.max(0.0)
                    })
                    .sum();
                scale = scale.max(s.abs());
                s
            })
            .collect();
        for (w, (&g, &v)) in got.iter().zip(&want).enumerate() {
            if v.is_infinite() {
                assert!(g == f32::NEG_INFINITY, "masked key {w}: {g}");
            } else {
                assert!((g as f64 - v).abs() < 1e-5 * scale, "score {w}: {g} vs {v}");
            }
        }
        // Routing.
        let score: Vec<f32> = lg.iter().map(|&v| if v > 20.0 { v } else { v.exp().ln_1p() }.sqrt()).collect();
        let selv: Vec<f32> = score.iter().zip(&bias).map(|(s, b)| s + b).collect();
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|&a, &b| selv[b].total_cmp(&selv[a]).then(a.cmp(&b)));
        idx.truncate(kk);
        let tot: f32 = idx.iter().map(|&i| score[i]).sum::<f32>() + 1e-20;
        let got = k.read(selb, 2 * kk + 1);
        let miss = k.read(missb, 1 + kk);
        let mut nmiss = 0;
        for (j, &e) in idx.iter().enumerate() {
            let (id, w) = (got[j].to_bits() as usize, got[kk + j]);
            if missing(e) {
                assert_eq!((id, w), (ne, 0.0), "slot {j}: expert {e} has no address");
                assert!(
                    miss[1..1 + miss[0].to_bits() as usize]
                        .iter()
                        .any(|m| m.to_bits() == (9 << 16) | e as u32)
                );
                nmiss += 1;
            } else {
                assert_eq!(id, e, "slot {j}");
                assert!((w - score[e] / tot * 1.5).abs() < 1e-5, "slot {j}: weight {w}");
            }
        }
        assert_eq!(miss[0].to_bits() as usize, nmiss);
        assert_eq!(got[2 * kk], 7.0);
        // Engram.
        let mut want = vec![0f64; n * h];
        for c in 0..n {
            let xc = &x[c * h..(c + 1) * h];
            let key = &kv[c * h..(c + 1) * h];
            let d: f64 = (0..h).map(|i| xc[i] as f64 * gw[c * h + i] as f64 * key[i] as f64).sum();
            let ms = |v: &[f32]| (v.iter().map(|&a| a as f64 * a as f64).sum::<f64>() / h as f64 + eps as f64).sqrt();
            let d = d / (ms(xc) * ms(key)) / (h as f64).sqrt();
            let g = 1.0 / (1.0 + (-(d.abs().max(1e-6).sqrt().copysign(d))).exp());
            for i in 0..h {
                want[c * h + i] = xc[i] as f64 + g * kv[n * h + i] as f64;
            }
        }
        close(&k.read(xo, n * h), &want, 1e-5, "engram");
    }
}
