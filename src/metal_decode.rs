//! Single-token decode on the GPU: the kernels of `metal_decode.metal` and their encoding.

use super::*;
use std::fmt::Write as _;

const SRC: &str = include_str!("metal_decode.metal");

/// (rows per threadgroup, SIMD-groups splitting each row) variants compiled for each format.
pub(crate) const MV_VARIANTS: [(usize, usize); 4] = [(1, 1), (1, 2), (1, 4), (2, 4)];

/// A weight format with a decode matvec kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MvFmt {
    Bf16,
    Q8,
    Q4,
    Gguf(GgmlType),
}

impl MvFmt {
    pub(crate) fn of(w: &WeightRef) -> Option<MvFmt> {
        Some(match w {
            WeightRef::Bf16(_) => MvFmt::Bf16,
            WeightRef::Q8(_) => MvFmt::Q8,
            WeightRef::Q4(_) => MvFmt::Q4,
            WeightRef::Gguf(g) if GGUF_GPU_TYPES.contains(&g.kind) => MvFmt::Gguf(g.kind),
            WeightRef::Gguf(_) => return None,
        })
    }
    fn name(self) -> String {
        match self {
            MvFmt::Bf16 => "bf".into(),
            MvFmt::Q8 => "q8".into(),
            MvFmt::Q4 => "q4".into(),
            MvFmt::Gguf(k) => k.name().into(),
        }
    }
    /// The instantiation macro call for this format.
    fn instantiate(self, nr: usize, ks: usize) -> String {
        let n = format!("mv_{}_{nr}_{ks}", self.name());
        match self {
            MvFmt::Bf16 => format!("MV_RAW({n}, MBf, X16, 16, {nr}, {ks})"),
            MvFmt::Q8 => format!("MV_Q8({n}, {nr}, {ks})"),
            MvFmt::Q4 => format!("MV_Q4({n}, {nr}, {ks})"),
            MvFmt::Gguf(k) => match k {
                GgmlType::F16 => format!("MV_RAW({n}, MF16, X16, 16, {nr}, {ks})"),
                GgmlType::Q4_0 => format!("MV_RAW({n}, MQ4_0, X32, 32, {nr}, {ks})"),
                GgmlType::Q8_0 => format!("MV_RAW({n}, MQ8_0, X32, 32, {nr}, {ks})"),
                GgmlType::Q4_K => format!("MV_RAW({n}, MQ4_K, XKQ, 32, {nr}, {ks})"),
                GgmlType::Q5_K => format!("MV_RAW({n}, MQ5_K, XKQ, 32, {nr}, {ks})"),
                GgmlType::Q6_K => format!("MV_RAW({n}, MQ6_K, X6, 32, {nr}, {ks})"),
                GgmlType::F32 => format!("MV_GEN({n}, F32W2, {nr}, {ks})"),
                k => format!("MV_GEN({n}, {}W, {nr}, {ks})", k.name()),
            },
        }
    }
    /// The kernel's weight type, x chunk type and chunk size, and the `MKX_` constructor of the
    /// weight type from an expert table entry `e`.
    fn kernel_types(self) -> (String, &'static str, usize, String) {
        let raw = |t: &str, x: &'static str, ch: usize| (t.to_string(), x, ch, format!("{t}{{ (e).q, p.gpr }}"));
        let gen_ = |w: String| {
            (
                format!("MGen<{w}>"),
                "X32",
                32,
                format!("MGen<{w}>{{ {w}{{ {{ (e).q, p.gpr }} }} }}"),
            )
        };
        match self {
            MvFmt::Bf16 => raw("MBf", "X16", 16),
            MvFmt::Q8 => (
                "MQ8".into(),
                "X16",
                16,
                "MQ8{ (device const char*)(e).q, (device const float*)(e).s, p.n_in }".into(),
            ),
            MvFmt::Q4 => (
                "MQ4".into(),
                "X32",
                32,
                "MQ4{ (e).q, (device const half*)(e).s, p.n_in, (device const ushort*)(e).o, p.gpr, p.gk }".into(),
            ),
            MvFmt::Gguf(k) => match k {
                GgmlType::F16 => raw("MF16", "X16", 16),
                GgmlType::Q4_0 => raw("MQ4_0", "X32", 32),
                GgmlType::Q8_0 => raw("MQ8_0", "X32", 32),
                GgmlType::Q4_K => raw("MQ4_K", "XKQ", 32),
                GgmlType::Q5_K => raw("MQ5_K", "XKQ", 32),
                GgmlType::Q6_K => raw("MQ6_K", "X6", 32),
                GgmlType::F32 => gen_("F32W2".into()),
                k => gen_(format!("{}W", k.name())),
            },
        }
    }
    /// Values per chunk (one lane's unit of work).
    fn chunk(self) -> usize {
        match self {
            MvFmt::Bf16 | MvFmt::Q8 | MvFmt::Gguf(GgmlType::F16) => 16,
            _ => 32,
        }
    }
    fn specialized(self) -> bool {
        !matches!(self, MvFmt::Gguf(k) if !matches!(k, GgmlType::F16 | GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K))
    }
}

fn all_formats() -> Vec<MvFmt> {
    let mut v = vec![MvFmt::Bf16, MvFmt::Q8, MvFmt::Q4];
    v.extend(GGUF_GPU_TYPES.iter().map(|&k| MvFmt::Gguf(k)));
    v
}

/// The decode kernels' source with the matvec instantiations.
pub(crate) fn shader_src() -> String {
    let mut s = String::from(SRC);
    for f in all_formats() {
        // MoE experts: `mvx_<fmt>_pair` / `_sum`.
        let (t, x, ch, mk) = f.kernel_types();
        let n = f.name();
        let _ = writeln!(s, "#define MKX_{n}(e) {mk}");
        let _ = writeln!(s, "MVX(mvx_{n}, {t}, {x}, {ch}, MKX_{n})");
        for &(nr, ks) in &MV_VARIANTS {
            if f.specialized() || (nr, ks) == (2, 4) {
                let _ = writeln!(s, "{}", f.instantiate(nr, ks));
            }
        }
    }
    s
}

/// Parameters of a matvec (`MV` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct MvParams {
    pub n_out: u32,
    pub n_in: u32,
    pub mode: u32,
    pub rows_per_tg: u32,
    pub w_off: u64,
    pub s_off: u64,
    pub w2_off: u64,
    pub s2_off: u64,
    pub o_off: u64,
    pub o2_off: u64,
    pub gpr: u32,
    pub gk: u32,
    pub ldy: u32,
    pub _p: u32,
}

/// A weight matrix as the matvec kernels address it.
#[derive(Clone, Copy)]
pub(crate) struct MvWeight {
    pub fmt: MvFmt,
    pub n_out: usize,
    pub n_in: usize,
    pub data: Id,
    pub data_off: u64,
    pub scales: Id,
    pub scales_off: u64,
    pub offs: Id,
    pub offs_off: u64,
    /// Row bytes (raw formats) or offset groups per row (affine Q4).
    pub gpr: u32,
    /// Affine Q4: blocks per offset group (0: not affine).
    pub gk: u32,
}

impl MvWeight {
    pub(crate) fn new(w: &WeightRef, dev: &DeviceWeight, n_out: usize, n_in: usize) -> Option<Self> {
        let mw = dev.downcast_ref::<MetalWeight>()?;
        let fmt = MvFmt::of(w)?;
        if !n_in.is_multiple_of(fmt.chunk()) || (matches!(fmt, MvFmt::Gguf(_)) && !n_in.is_multiple_of(32)) {
            return None;
        }
        let (gpr, gk) = match w {
            WeightRef::Gguf(g) => (g.row_bytes() as u32, 0),
            WeightRef::Bf16(_) => ((n_in * 2) as u32, 0),
            _ => match &mw.offsets {
                Some((_, gpr, gk)) => (*gpr, *gk),
                None => (0, 0),
            },
        };
        let (offs, offs_off) = match &mw.offsets {
            Some((b, _, _)) => (b.id, b.offset_bytes as u64),
            None => (mw.data.id, 0),
        };
        Some(MvWeight {
            fmt,
            n_out,
            n_in,
            data: mw.data.id,
            data_off: mw.data.offset_bytes as u64,
            scales: mw.scales.as_ref().map_or(mw.data.id, |s| s.id),
            scales_off: mw.scales.as_ref().map_or(0, |s| s.offset_bytes as u64),
            offs,
            offs_off,
            gpr,
            gk,
        })
    }
}

/// Matvec output modes (`MV::mode`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MvMode {
    Store = 0,
    Add = 1,
    SiluPair = 2,
    GeluPair = 3,
    AddGated = 4,
}

impl MetalBackend {
    /// The matvec pipeline for a format and variant.
    pub(crate) fn mv_pipe(&self, fmt: MvFmt, nr: usize, ks: usize) -> Option<Id> {
        if !fmt.specialized() && (nr, ks) != (2, 4) {
            return None;
        }
        self.named_pipeline(&format!("mv_{}_{nr}_{ks}", fmt.name()))
    }
}

/// `objc_msgSend` typed for the encoder calls, and their selectors, looked up once (`Objc::msg_send`
/// and `Objc::sel` go through `dlsym`, too slow for the ~1000 calls of a decode step).
pub(crate) struct Fns {
    set_pipe: MsgSetPipe,
    set_buf: MsgSetBuffer,
    set_bytes: MsgSetBytes,
    dispatch: MsgDispatch,
    barrier: unsafe extern "C" fn(Id, Sel, u64),
    use_res: unsafe extern "C" fn(Id, Sel, *const Id, usize, usize),
    enc_type: unsafe extern "C" fn(Id, Sel, u64) -> Id,
    msg0: Msg0,
    void: MsgVoid,
    status: MsgU,
    contents: MsgPtr,
    s_pipe: Sel,
    s_buf: Sel,
    s_bytes: Sel,
    s_dispatch: Sel,
    s_barrier: Sel,
    s_use_res: Sel,
    s_enc_type: Sel,
    s_cb: Sel,
    s_end: Sel,
    s_commit: Sel,
    s_wait: Sel,
    s_status: Sel,
    s_contents: Sel,
    s_release: Sel,
}

impl Fns {
    pub(crate) fn new(b: &MetalBackend) -> Self {
        let o = &b.o;
        Fns {
            set_pipe: o.msg_send(),
            set_buf: o.msg_send(),
            set_bytes: o.msg_send(),
            dispatch: o.msg_send(),
            barrier: o.msg_send(),
            use_res: o.msg_send(),
            enc_type: o.msg_send(),
            msg0: o.msg_send(),
            void: o.msg_send(),
            status: o.msg_send(),
            contents: o.msg_send(),
            s_pipe: b.sels.set_pipeline,
            s_buf: b.sels.set_buffer,
            s_bytes: b.sels.set_bytes,
            s_dispatch: b.sels.dispatch,
            s_barrier: o.sel(c"memoryBarrierWithScope:"),
            s_use_res: b.sels.use_resources,
            s_enc_type: o.sel(c"computeCommandEncoderWithDispatchType:"),
            s_cb: b.sels.command_buffer,
            s_end: b.sels.end_encoding,
            s_commit: b.sels.commit,
            s_wait: b.sels.wait,
            s_status: b.sels.status,
            s_contents: b.sels.contents,
            s_release: b.sels.release,
        }
    }
    pub(crate) fn contents(&self, b: Id) -> *mut u8 {
        unsafe { (self.contents)(b, self.s_contents) as *mut u8 }
    }
}

/// A compute encoder being filled through [`Fns`].
pub(crate) struct FastEnc<'f> {
    pub enc: Id,
    pub f: &'f Fns,
}

impl FastEnc<'_> {
    pub(crate) fn pipe(&self, p: Id) {
        unsafe { (self.f.set_pipe)(self.enc, self.f.s_pipe, p) };
    }
    pub(crate) fn buf(&self, b: Id, offset: usize, index: usize) {
        unsafe { (self.f.set_buf)(self.enc, self.f.s_buf, b, offset, index) };
    }
    pub(crate) fn bytes<T>(&self, v: &T, index: usize) {
        unsafe {
            (self.f.set_bytes)(
                self.enc,
                self.f.s_bytes,
                v as *const T as *const c_void,
                std::mem::size_of::<T>(),
                index,
            )
        };
    }
    pub(crate) fn bytes_slice(&self, v: &[f32], index: usize) {
        unsafe {
            (self.f.set_bytes)(
                self.enc,
                self.f.s_bytes,
                v.as_ptr() as *const c_void,
                std::mem::size_of_val(v),
                index,
            )
        };
    }
    pub(crate) fn dispatch(&self, groups: (usize, usize), threads: usize) {
        let g = MtlSize {
            w: groups.0,
            h: groups.1,
            d: 1,
        };
        let t = MtlSize { w: threads, h: 1, d: 1 };
        unsafe { (self.f.dispatch)(self.enc, self.f.s_dispatch, &g, &t) };
    }
    /// Makes buffers reached only through addresses resident for this encoder.
    pub(crate) fn use_resources(&self, ids: &[Id]) {
        if !ids.is_empty() {
            unsafe { (self.f.use_res)(self.enc, self.f.s_use_res, ids.as_ptr(), ids.len(), 1) };
        }
    }
    /// Makes the writes of earlier dispatches visible to later ones (concurrent encoders).
    pub(crate) fn barrier(&self) {
        unsafe { (self.f.barrier)(self.enc, self.f.s_barrier, 1) }; // MTLBarrierScopeBuffers
    }
}

/// `OXINFER_METAL_TRACE` is set.
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_METAL_TRACE").is_some())
}

/// Runs `encode` over the ranges of `parts`, one command buffer each with a concurrent compute
/// encoder, committed as soon as it is encoded (the GPU starts on the first while the rest is
/// encoded), and waits for the last; returns whether all completed and the GPU time in seconds.
pub(crate) fn run_concurrent(
    b: &MetalBackend,
    f: &Fns,
    parts: &[std::ops::Range<usize>],
    encode: impl Fn(&FastEnc, std::ops::Range<usize>),
) -> (bool, f64) {
    unsafe {
        let t0 = Instant::now();
        let pool = (b.sels.pool_push)();
        let mut cbs = Vec::with_capacity(parts.len());
        for r in parts {
            let cb = (f.msg0)(b.queue, f.s_cb);
            let enc = (f.enc_type)(cb, f.s_enc_type, 1); // MTLDispatchTypeConcurrent
            encode(&FastEnc { enc, f }, r.clone());
            (f.void)(enc, f.s_end);
            (f.void)(cb, f.s_commit);
            cbs.push(cb);
        }
        let t1 = Instant::now();
        let mut ok = true;
        for &cb in &cbs {
            (f.void)(cb, f.s_wait);
            ok &= (f.status)(cb, f.s_status) == 4;
        }
        let mut g = 0.0;
        if trace() && !cbs.is_empty() {
            let time: unsafe extern "C" fn(Id, Sel) -> f64 = b.o.msg_send();
            g = time(*cbs.last().unwrap(), b.o.sel(c"GPUEndTime")) - time(cbs[0], b.o.sel(c"GPUStartTime"));
            eprintln!(
                "[metal decode: encode {:.0} us, wait {:.0} us, gpu {:.0} us]",
                (t1 - t0).as_secs_f64() * 1e6,
                t1.elapsed().as_secs_f64() * 1e6,
                g * 1e6
            );
        }
        (b.sels.pool_pop)(pool);
        (ok, g)
    }
}

/// Encodes one matvec: `y = W x` (or the `mode` variants; `w2` is the up projection of a pair).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mv(
    e: &FastEnc,
    pipe: Id,
    nr: usize,
    ks: usize,
    w: &MvWeight,
    w2: Option<&MvWeight>,
    x: (Id, usize),
    y: (Id, usize),
    mode: MvMode,
    gate: (Id, usize),
) {
    let w2 = w2.unwrap_or(w);
    let p = MvParams {
        n_out: w.n_out as u32,
        n_in: w.n_in as u32,
        mode: mode as u32,
        rows_per_tg: nr as u32,
        w_off: w.data_off,
        s_off: w.scales_off,
        w2_off: w2.data_off,
        s2_off: w2.scales_off,
        o_off: w.offs_off,
        o2_off: w2.offs_off,
        gpr: w.gpr,
        gk: w.gk,
        ldy: 0,
        _p: 0,
    };
    e.pipe(pipe);
    e.buf(w.data, 0, 0);
    e.buf(w.scales, 0, 1);
    e.buf(x.0, x.1, 2);
    e.buf(y.0, y.1, 3);
    e.bytes(&p, 4);
    e.buf(w2.data, 0, 5);
    e.buf(w2.scales, 0, 6);
    e.buf(w.offs, 0, 7);
    e.buf(w2.offs, 0, 8);
    e.buf(gate.0, gate.1, 9);
    e.dispatch((w.n_out.div_ceil(nr), 1), 32 * ks);
}

// ---- Decode plans -------------------------------------------------------------------------------

use crate::gpu::decode::{DecodePlan, Graph, LinMode, Op, Slot, Step, View};

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnRuleParams {
    nk: u32,
    nv: u32,
    key_dim: u32,
    tiled: u32,
    eps: f32,
    qscale: f32,
    hdk: u32,
    hdv: u32,
}

/// The matvec variant (rows per threadgroup, SIMD-groups per row) for a shape: enough lanes per
/// row to cover it in one or two passes (measured: short rows want 1 SIMD-group, 8192 4).
fn mv_variant(fmt: MvFmt, n_in: usize) -> (usize, usize) {
    if !fmt.specialized() {
        return (2, 4); // the only variant compiled for the generic kernel
    }
    let nch = n_in / fmt.chunk();
    (1, (nch / 64).clamp(1, 4).next_power_of_two().min(4))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormParams {
    n: u32,
    eps: f32,
    has_w: u32,
    add: u32,
    x_stride: u32,
    o_stride: u32,
    pre: f32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QkParams {
    nh: u32,
    nkv: u32,
    hd: u32,
    half_rot: u32,
    q_stride: u32,
    k_off: u32,
    v_off: u32,
    pos: u32,
    interleaved: u32,
    has_qn: u32,
    has_kn: u32,
    eps: f32,
    v_norm: u32,
    v_from_k: u32,
    _p: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParams {
    nh: u32,
    nkv: u32,
    len: u32,
    lo: u32,
    scale: f32,
    has_gate: u32,
    g_stride: u32,
    g_off: u32,
    nsplit: u32,
    chunk: u32,
    _p: [u32; 2],
}

/// Most key chunks a head's attention is split into (`AttnParams::nsplit`).
const ATTN_MAX_SPLIT: usize = 64;

/// A compiled op: byte offsets into the plan's activation arena (`a`) or constants (`c`).
enum MOp {
    Barrier,
    Norm {
        pipe: Id,
        x: usize,
        out: usize,
        w: Option<usize>,
        p: NormParams,
        rows: usize,
        threads: usize,
    },
    Mv {
        pipe: Id,
        nr: usize,
        ks: usize,
        w: MvWeight,
        w2: Option<MvWeight>,
        x: usize,
        y: usize,
        mode: MvMode,
        /// AddGated: the scalar gate's arena offset.
        gate: usize,
    },
    Topk {
        pipe: Id,
        logits: usize,
        sel: usize,
        n: u32,
        k: u32,
        scale: Option<usize>,
    },
    Experts {
        pair: Id,
        sum: Id,
        /// Address tables (gate or fused gate/up, up, down).
        tabs: [Id; 3],
        fused: bool,
        gelu: bool,
        add: bool,
        /// (row bytes or offset groups, blocks per group) of gate / up and of down.
        g: (u32, u32),
        d: (u32, u32),
        x: usize,
        sel: usize,
        act: usize,
        y: usize,
        h: usize,
        inter: usize,
        k: usize,
        /// Neuron edits between the two kernels: (pipeline, hot offset, entries).
        edit: Option<(Id, usize, u32)>,
    },
    QkRope {
        pipe: Id,
        qkv: usize,
        qo: usize,
        kv: usize,
        qn: Option<usize>,
        kn: Option<usize>,
        p: QkParams,
        /// Where this op's RoPE table starts in the step's cos / sin floats.
        cs: usize,
    },
    AddNorm {
        pipe: Id,
        y: usize,
        x: usize,
        out: usize,
        wy: usize,
        wn: usize,
        p: [u32; 4],
        threads: usize,
    },
    Ew {
        pipe: Id,
        a: usize,
        b: usize,
        out: usize,
        n: u32,
        p: [f32; 4],
    },
    Glu {
        pipe: Id,
        g: usize,
        u: usize,
        out: usize,
        n: u32,
        gelu: u32,
    },
    Steer {
        pipe: Id,
        x: usize,
        hot: usize,
        cold: usize,
        p: [u32; 4],
    },
    NeuronEdit {
        pipe: Id,
        act: usize,
        hot: usize,
        n: u32,
    },
    GdnConv {
        pipe: Id,
        x: usize,
        out: usize,
        state: usize,
        w: usize,
        dim: u32,
        kc: u32,
    },
    GdnRule {
        pipe: Id,
        c: usize,
        z: usize,
        b: usize,
        a: usize,
        out: usize,
        state: usize,
        cst: usize,
        p: GdnRuleParams,
    },
    Attn {
        /// Query heads per threadgroup.
        grp: usize,
        pipe: Id,
        combine: Id,
        q: usize,
        out: usize,
        kv: usize,
        gate: Option<usize>,
        window: Option<usize>,
        p: AttnParams,
        threads: usize,
        hd: u32,
    },
}

pub(crate) struct MetalPlan<'b> {
    be: &'b MetalBackend,
    f: Fns,
    arena: Id,
    consts: Id,
    /// Arena offset of the split-attention partials.
    part: usize,
    /// Buffers made for the plan (MoE weight spans, expert tables), released with it.
    owned: Vec<Id>,
    /// Buffers reached only through addresses, made resident on every step when the queue has
    /// no residency set.
    indirect: Vec<Id>,
    ops: Vec<MOp>,
    head_from: usize,
    input: usize,
    hidden: (usize, usize),
    logits: Option<(usize, usize)>,
    kv: Vec<(usize, usize)>,
    rope_halves: Vec<usize>,
    input2: Option<(usize, usize)>,
    states: Vec<usize>,
    /// Intervention parameters: (hot offset, floats), (cold offset, floats), and the key of the
    /// cold values in the arena.
    params: Option<((usize, usize), (usize, usize))>,
    params_key: AtomicU64,
    /// One step at a time (the arena is shared).
    lock: Mutex<()>,
    /// Identifies this plan's slots.
    id: u64,
}
unsafe impl Send for MetalPlan<'_> {}
unsafe impl Sync for MetalPlan<'_> {}

impl Drop for MetalPlan<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.f.void)(self.arena, self.f.s_release);
            (self.f.void)(self.consts, self.f.s_release);
            if let Some(r) = &self.be.residency
                && !self.indirect.is_empty()
            {
                r.update(&self.indirect, false);
            }
            for &b in &self.owned {
                (self.f.void)(b, self.f.s_release);
            }
        }
    }
}

/// Device copies of a session's KV caches.
struct KvMirror {
    k: Id,
    v: Id,
    /// Positions the buffers hold, and how many match the host cache.
    cap: usize,
    synced: usize,
    release: (MsgVoid, Sel),
}

impl Drop for KvMirror {
    fn drop(&mut self) {
        for b in [self.k, self.v] {
            if !b.is_null() {
                unsafe { (self.release.0)(b, self.release.1) };
            }
        }
    }
}

struct MetalSlot {
    plan: u64,
    kv: Vec<KvMirror>,
    /// The recurrent states wrapped in place: (address, length, buffer).
    states: Vec<(usize, usize, Option<Buf>)>,
    release: (MsgVoid, Sel),
}

impl Drop for MetalSlot {
    fn drop(&mut self) {
        for (_, _, b) in &self.states {
            if let Some(b) = b {
                unsafe { (self.release.0)(b.id, self.release.1) };
            }
        }
    }
}
unsafe impl Send for MetalSlot {}
unsafe impl Sync for MetalSlot {}

static PLAN_IDS: AtomicU64 = AtomicU64::new(1);

/// What an op reads and writes, for barrier placement: (resource, start, end) in floats, where a
/// resource is an activation buffer or (`KV_RES + i`) a KV cache.
type Range = (usize, usize, usize);
const KV_RES: usize = 1 << 20;

fn overlaps(a: &[Range], b: &[Range]) -> bool {
    a.iter().any(|x| b.iter().any(|y| x.0 == y.0 && x.1 < y.2 && y.1 < x.2))
}

impl MetalBackend {
    pub(crate) fn build_plan(&self, g: &Graph) -> Option<MetalPlan<'_>> {
        let f = Fns::new(self);
        // Activation arena: every buffer 256-byte aligned.
        let mut boff = Vec::with_capacity(g.bufs.len());
        let mut total = 0usize;
        for &n in &g.bufs {
            boff.push(total);
            total += (n * 4).next_multiple_of(256);
        }
        let byte = |v: View| boff[v.buf] + v.off * 4;
        // Constants: every vector 64-byte aligned.
        let mut consts: Vec<f32> = Vec::new();
        let mut cst = |v: &[f32]| -> usize {
            consts.resize(consts.len().next_multiple_of(16), 0.0);
            let o = consts.len() * 4;
            consts.extend_from_slice(v);
            o
        };
        let mut ops: Vec<MOp> = Vec::new();
        let mut part_floats = 0usize;
        let mut owned: Vec<Id> = Vec::new();
        let mut indirect: Vec<Id> = Vec::new();
        let mut head_from = usize::MAX;
        let (mut rd, mut wr): (Vec<Range>, Vec<Range>) = (Vec::new(), Vec::new());
        let rng = |v: View, n: usize| (v.buf, v.off, v.off + n);
        for (i, op) in g.ops.iter().enumerate() {
            if i == g.head_from {
                // The head starts after a barrier, so skipping it leaves no dangling dependency.
                ops.push(MOp::Barrier);
                rd.clear();
                wr.clear();
                head_from = ops.len();
            }
            let mut emit: Vec<MOp> = Vec::new();
            let (r, w): (Vec<Range>, Vec<Range>) = match *op {
                Op::Norm {
                    x,
                    out,
                    ref w,
                    pre,
                    n,
                    rows,
                    x_stride,
                    out_stride,
                    eps,
                    add,
                } => {
                    let threads = n.next_multiple_of(32).min(1024);
                    emit.push(MOp::Norm {
                        pipe: self.named_pipeline("dec_rms_norm")?,
                        x: byte(x),
                        out: byte(out),
                        w: w.as_deref().map(&mut cst),
                        p: NormParams {
                            n: n as u32,
                            eps,
                            has_w: w.is_some() as u32,
                            add: add as u32,
                            x_stride: x_stride as u32,
                            o_stride: out_stride as u32,
                            pre,
                            _p: 0,
                        },
                        rows,
                        threads,
                    });
                    let span = |v: View, s: usize| rng(v, (rows - 1) * s + n);
                    let mut r = vec![span(x, x_stride)];
                    if add {
                        r.push(span(out, out_stride));
                    }
                    (r, vec![span(out, out_stride)])
                }
                Op::Linear {
                    ref w,
                    dev,
                    ref w2,
                    n_out,
                    n_in,
                    x,
                    y,
                    mode,
                } => {
                    let mw = MvWeight::new(w, dev, n_out, n_in)?;
                    let mw2 = match w2 {
                        Some((w2, d2)) => {
                            let m = MvWeight::new(w2, d2, n_out, n_in)?;
                            if m.fmt != mw.fmt || m.gk != mw.gk || m.gpr != mw.gpr {
                                return None; // a pair shares one kernel
                            }
                            Some(m)
                        }
                        None => None,
                    };
                    let (mode, gate) = match mode {
                        LinMode::Store => (MvMode::Store, None),
                        LinMode::Add => (MvMode::Add, None),
                        LinMode::SiluPair => (MvMode::SiluPair, None),
                        LinMode::GeluPair => (MvMode::GeluPair, None),
                        LinMode::AddGated(v) => (MvMode::AddGated, Some(v)),
                    };
                    if (mw2.is_some()) != matches!(mode, MvMode::SiluPair | MvMode::GeluPair) {
                        return None;
                    }
                    let (nr, ks) = mv_variant(mw.fmt, n_in);
                    emit.push(MOp::Mv {
                        pipe: self.mv_pipe(mw.fmt, nr, ks)?,
                        nr,
                        ks,
                        w: mw,
                        w2: mw2,
                        x: byte(x),
                        y: byte(y),
                        mode,
                        gate: gate.map_or(0, byte),
                    });
                    let mut r = vec![rng(x, n_in)];
                    if matches!(mode, MvMode::Add | MvMode::AddGated) {
                        r.push(rng(y, n_out));
                    }
                    if let Some(v) = gate {
                        r.push(rng(v, 1));
                    }
                    (r, vec![rng(y, n_out)])
                }
                Op::QkRope {
                    qkv,
                    q_out,
                    kv,
                    nh,
                    nkv,
                    hd,
                    half_rot,
                    q_stride,
                    k_off,
                    v_off,
                    interleaved,
                    q_norm,
                    k_norm,
                    eps,
                    rope,
                    v_norm,
                    v_from_k,
                } => {
                    if hd > 512 || !hd.is_multiple_of(32) || 2 * half_rot > hd || g.rope_halves.get(rope) != Some(&half_rot) {
                        return None;
                    }
                    emit.push(MOp::QkRope {
                        pipe: self.named_pipeline("dec_qk_rope")?,
                        qkv: byte(qkv),
                        qo: byte(q_out),
                        kv,
                        qn: q_norm.map(&mut cst),
                        kn: k_norm.map(&mut cst),
                        p: QkParams {
                            nh: nh as u32,
                            nkv: nkv as u32,
                            hd: hd as u32,
                            half_rot: half_rot as u32,
                            q_stride: q_stride as u32,
                            k_off: k_off as u32,
                            v_off: v_off as u32,
                            pos: 0,
                            interleaved: interleaved as u32,
                            has_qn: q_norm.is_some() as u32,
                            has_kn: k_norm.is_some() as u32,
                            eps,
                            v_norm: v_norm as u32,
                            v_from_k: v_from_k as u32,
                            _p: [0; 2],
                        },
                        cs: g.rope_halves[..rope].iter().map(|h| 2 * h).sum(),
                    });
                    let end = (nh * q_stride)
                        .max(k_off + nkv * hd)
                        .max(if v_from_k { 0 } else { v_off + nkv * hd });
                    let mut w = vec![rng(q_out, nh * hd)];
                    if nkv > 0 {
                        w.push((KV_RES + kv, 0, usize::MAX));
                    }
                    (vec![rng(qkv, end)], w)
                }
                Op::Route {
                    x,
                    router,
                    gate,
                    n_exp,
                    h,
                    k,
                    logits,
                    sel,
                    expert_scale,
                } => {
                    if expert_scale.is_some_and(|s| s.len() != n_exp) {
                        return None;
                    }
                    if router.len() != n_exp * h || gate.is_some_and(|g| g.len() != h) || n_exp > 1024 || k == 0 || k > 64 || k > n_exp {
                        return None;
                    }
                    let mut w = router.to_vec();
                    w.extend_from_slice(gate.unwrap_or(&[]));
                    let rows = n_exp + gate.is_some() as usize;
                    let fmt = MvFmt::Gguf(GgmlType::F32);
                    // The router rows live in the constants buffer (patched in below).
                    let mw = MvWeight {
                        fmt,
                        n_out: rows,
                        n_in: h,
                        data: std::ptr::null_mut(),
                        data_off: cst(&w) as u64,
                        scales: std::ptr::null_mut(),
                        scales_off: 0,
                        offs: std::ptr::null_mut(),
                        offs_off: 0,
                        gpr: (h * 4) as u32,
                        gk: 0,
                    };
                    let (nr, ks) = mv_variant(fmt, h);
                    emit.push(MOp::Mv {
                        pipe: self.mv_pipe(fmt, nr, ks)?,
                        nr,
                        ks,
                        w: mw,
                        w2: None,
                        x: byte(x),
                        y: byte(logits),
                        mode: MvMode::Store,
                        gate: 0,
                    });
                    emit.push(MOp::Barrier);
                    emit.push(MOp::Topk {
                        pipe: self.named_pipeline("dec_topk")?,
                        logits: byte(logits),
                        sel: byte(sel),
                        n: n_exp as u32,
                        k: k as u32,
                        scale: expert_scale.map(&mut cst),
                    });
                    (vec![rng(x, h), rng(logits, rows)], vec![rng(logits, rows), rng(sel, 2 * k)])
                }
                Op::Experts(ref ex) => {
                    let (tabs, fused, g, d, pf) = self.expert_tables(ex, &mut owned, &mut indirect)?;
                    let edit = match ex.edits {
                        Some((v, n)) => Some((self.named_pipeline("dec_neuron_edit")?, byte(v), n as u32)),
                        None => None,
                    };
                    emit.push(MOp::Experts {
                        pair: self.named_pipeline(&format!("mvx_{}_pair", pf.0.name()))?,
                        sum: self.named_pipeline(&format!("mvx_{}_sum", pf.1.name()))?,
                        tabs,
                        fused,
                        gelu: ex.gelu,
                        add: ex.add,
                        g,
                        d,
                        x: byte(ex.x),
                        sel: byte(ex.sel),
                        act: byte(ex.act),
                        y: byte(ex.y),
                        h: ex.h,
                        inter: ex.inter,
                        k: ex.k,
                        edit,
                    });
                    let mut r = vec![
                        rng(ex.x, ex.h),
                        rng(ex.sel, 2 * ex.k),
                        rng(ex.act, ex.k * ex.inter),
                        rng(ex.y, ex.h),
                    ];
                    if let Some((v, n)) = ex.edits {
                        r.push(rng(v, n * crate::intervene::GPU_ENTRY));
                    }
                    (r, vec![rng(ex.act, ex.k * ex.inter), rng(ex.y, ex.h)])
                }
                Op::AddNorm {
                    y,
                    x,
                    ref wy,
                    scale,
                    next,
                    n,
                    eps,
                } => {
                    let threads = n.next_multiple_of(32).min(1024);
                    if n > 8 * threads || wy.len() != n || next.is_some_and(|(_, w)| w.len() != n) {
                        return None;
                    }
                    emit.push(MOp::AddNorm {
                        pipe: self.named_pipeline("dec_add_norm")?,
                        y: byte(y),
                        x: byte(x),
                        out: next.map_or(byte(x), |(o, _)| byte(o)),
                        wy: cst(wy),
                        wn: next.map_or(0, |(_, w)| cst(w)),
                        p: [n as u32, eps.to_bits(), scale.to_bits(), next.is_some() as u32],
                        threads,
                    });
                    let mut w = vec![rng(x, n)];
                    w.extend(next.map(|(o, _)| rng(o, n)));
                    (vec![rng(y, n), rng(x, n)], w)
                }
                Op::Ew { kind, a, b, out, n } => {
                    use crate::gpu::decode::Ew;
                    let p = match kind {
                        Ew::Scale(s) => [0.0, s, 0.0, 0.0],
                        Ew::Softcap(c) => [1.0, c, 0.0, 0.0],
                        Ew::Combine { sa, sb, s } => [2.0, sa, sb, s],
                    };
                    if matches!(kind, Ew::Combine { .. }) && b.is_none() {
                        return None;
                    }
                    emit.push(MOp::Ew {
                        pipe: self.named_pipeline("dec_ew")?,
                        a: byte(a),
                        b: byte(b.unwrap_or(a)),
                        out: byte(out),
                        n: n as u32,
                        p,
                    });
                    let mut r = vec![rng(a, n)];
                    r.extend(b.map(|b| rng(b, n)));
                    (r, vec![rng(out, n)])
                }
                Op::Glu { g: gv, u, out, n, gelu } => {
                    emit.push(MOp::Glu {
                        pipe: self.named_pipeline("dec_glu")?,
                        g: byte(gv),
                        u: byte(u),
                        out: byte(out),
                        n: n as u32,
                        gelu: gelu as u32,
                    });
                    (vec![rng(gv, n), rng(u, n)], vec![rng(out, n)])
                }
                Op::Steer {
                    x,
                    n,
                    hot,
                    cold,
                    n_add,
                    n_items,
                } => {
                    if n_items > crate::intervene::GPU_MAX_ITEMS {
                        return None;
                    }
                    emit.push(MOp::Steer {
                        pipe: self.named_pipeline("dec_steer")?,
                        x: byte(x),
                        hot: byte(hot),
                        cold: byte(cold),
                        p: [n as u32, n_add as u32, n_items as u32, 0],
                    });
                    (
                        vec![
                            rng(x, n),
                            rng(hot, n_add.next_multiple_of(4) + crate::intervene::GPU_ENTRY * n_items),
                            rng(cold, (n_add + 2 * n_items) * n),
                        ],
                        vec![rng(x, n)],
                    )
                }
                Op::NeuronEdit { act, hot, n } => {
                    emit.push(MOp::NeuronEdit {
                        pipe: self.named_pipeline("dec_neuron_edit")?,
                        act: byte(act),
                        hot: byte(hot),
                        n: n as u32,
                    });
                    // The edited neurons are anywhere in the activation: the whole buffer.
                    (
                        vec![(act.buf, 0, usize::MAX), rng(hot, n * crate::intervene::GPU_ENTRY)],
                        vec![(act.buf, 0, usize::MAX)],
                    )
                }
                Op::GdnConv { x, out, state, w, dim, kc } => {
                    if kc < 2 || w.len() != dim * kc || g.states.get(state) != Some(&((kc - 1) * dim)) {
                        return None;
                    }
                    emit.push(MOp::GdnConv {
                        pipe: self.named_pipeline("dec_gdn_conv")?,
                        x: byte(x),
                        out: byte(out),
                        state,
                        w: cst(w),
                        dim: dim as u32,
                        kc: kc as u32,
                    });
                    (
                        vec![rng(x, dim), (KV_RES * 2 + state, 0, usize::MAX)],
                        vec![rng(out, dim), (KV_RES * 2 + state, 0, usize::MAX)],
                    )
                }
                Op::GdnRule(ref r) => {
                    let ok_dims = r.hdk.is_multiple_of(32)
                        && r.hdk <= 256
                        && r.hdv.is_multiple_of(32)
                        && r.hdk / 32 * r.hdv <= 1024
                        && r.hdv >= r.hdk.min(32);
                    if !ok_dims || r.nk == 0 || !r.nv.is_multiple_of(r.nk) {
                        return None;
                    }
                    if g.states.get(r.state) != Some(&(r.nv * r.hdk * r.hdv)) || r.norm_w.len() != r.hdv {
                        return None;
                    }
                    let mut v = r.decay_coef.to_vec();
                    v.extend_from_slice(r.dt_bias);
                    v.extend_from_slice(r.norm_w);
                    let key_dim = r.nk * r.hdk;
                    emit.push(MOp::GdnRule {
                        pipe: self.named_pipeline("dec_gdn_rule")?,
                        c: byte(r.c),
                        z: byte(r.z),
                        b: byte(r.b),
                        a: byte(r.a),
                        out: byte(r.out),
                        state: r.state,
                        cst: cst(&v),
                        p: GdnRuleParams {
                            nk: r.nk as u32,
                            nv: r.nv as u32,
                            key_dim: key_dim as u32,
                            tiled: r.tiled as u32,
                            eps: r.eps,
                            qscale: (r.hdk as f32).powf(-0.5),
                            hdk: r.hdk as u32,
                            hdv: r.hdv as u32,
                        },
                    });
                    let vd = r.nv * r.hdv;
                    (
                        vec![
                            rng(r.c, 2 * key_dim + vd),
                            rng(r.z, vd),
                            rng(r.b, r.nv),
                            rng(r.a, r.nv),
                            (KV_RES * 2 + r.state, 0, usize::MAX),
                        ],
                        vec![rng(r.out, vd), (KV_RES * 2 + r.state, 0, usize::MAX)],
                    )
                }
                Op::Attend {
                    q,
                    out,
                    kv,
                    nh,
                    nkv,
                    hd,
                    scale,
                    window,
                    gate,
                } => {
                    let d = hd / 32;
                    if !hd.is_multiple_of(32) || !matches!(d, 2 | 4 | 8 | 16) || nkv == 0 || !nh.is_multiple_of(nkv) {
                        return None;
                    }
                    if g.kv.get(kv) != Some(&(nkv, hd)) {
                        return None;
                    }
                    part_floats = part_floats.max(nh * ATTN_MAX_SPLIT * (hd + 2));
                    // Query heads per threadgroup: as many of a key head's as fit (G * D <= 16).
                    let rep = nh / nkv;
                    let grp = [8, 4, 2, 1]
                        .into_iter()
                        .find(|&g| rep.is_multiple_of(g) && g * d <= 16)
                        .unwrap_or(1);
                    emit.push(MOp::Attn {
                        grp,
                        pipe: self.named_pipeline(&format!("dec_attn_{d}_{grp}"))?,
                        combine: self.named_pipeline("dec_attn_combine")?,
                        hd: hd as u32,
                        q: byte(q),
                        out: byte(out),
                        kv,
                        gate: gate.map(|(v, _)| boff[v.buf]),
                        window,
                        p: AttnParams {
                            nh: nh as u32,
                            nkv: nkv as u32,
                            len: 0,
                            lo: 0,
                            scale,
                            has_gate: gate.is_some() as u32,
                            g_stride: gate.map_or(0, |(_, s)| s as u32),
                            g_off: gate.map_or(0, |(v, _)| v.off as u32),
                            nsplit: 1,
                            chunk: 0,
                            _p: [0; 2],
                        },
                        threads: 256,
                    });
                    let mut r = vec![rng(q, nh * hd), (KV_RES + kv, 0, usize::MAX)];
                    if let Some((v, s)) = gate {
                        r.push(rng(v, nh * s));
                    }
                    (r, vec![rng(out, nh * hd)])
                }
            };
            // A barrier when this op touches what earlier ones of the stage wrote, or writes what
            // they read.
            if overlaps(&r, &wr) || overlaps(&w, &wr) || overlaps(&w, &rd) {
                ops.push(MOp::Barrier);
                rd.clear();
                wr.clear();
            }
            ops.extend(emit);
            rd.extend(r);
            wr.extend(w);
        }
        if head_from == usize::MAX {
            head_from = ops.len();
        }
        // Split-attention partials after the activations.
        let part = total;
        total += part_floats * 4;
        let new_len: MsgNewBufLen = self.o.msg_send();
        let arena = unsafe { new_len(self.device, self.sels.new_buffer_len, total.max(256), 0) };
        if arena.is_null() {
            return None;
        }
        consts.push(0.0);
        let copy: MsgNewBufBytes = self.o.msg_send();
        let cbuf = unsafe {
            copy(
                self.device,
                self.sels.new_buffer_bytes,
                consts.as_ptr() as *const c_void,
                consts.len() * 4,
                0,
            )
        };
        if cbuf.is_null() {
            unsafe { (f.void)(arena, f.s_release) };
            return None;
        }
        for op in &mut ops {
            if let MOp::Mv { w, .. } = op
                && w.data.is_null()
            {
                (w.data, w.scales, w.offs) = (cbuf, cbuf, cbuf);
            }
        }
        if let Some(r) = &self.residency
            && !indirect.is_empty()
        {
            r.update(&indirect, true);
        }
        Some(MetalPlan {
            be: self,
            arena,
            consts: cbuf,
            part,
            owned,
            indirect,
            ops,
            head_from,
            input: byte(g.input),
            hidden: (byte(g.hidden), g.n_hidden),
            logits: g.logits.map(|(v, n)| (byte(v), n)),
            kv: g.kv.clone(),
            rope_halves: g.rope_halves.clone(),
            input2: g.input2.map(|(v, n)| (byte(v), n)),
            states: g.states.clone(),
            params: g.params.map(|((h, hn), (c, cn))| ((byte(h), hn), (byte(c), cn))),
            params_key: AtomicU64::new(0),
            lock: Mutex::new(()),
            id: PLAN_IDS.fetch_add(1, Ordering::Relaxed),
            f,
        })
    }
}

impl MetalBackend {
    /// Address tables of the experts of an MoE layer: entry e = (data, scales, affine offsets,
    /// unused) GPU addresses of expert e. The experts' memory is wrapped in a few buffers (adjacent
    /// regions merged: experts are usually slices of one tensor), added to `owned` and `indirect`.
    /// Returns the tables (gate or fused gate/up, up, down), whether gate/up are fused, the
    /// (row bytes or groups, blocks per group) parameters of gate/up and of down, and their formats.
    #[allow(clippy::type_complexity)]
    fn expert_tables(
        &self,
        ex: &crate::gpu::decode::Experts,
        owned: &mut Vec<Id>,
        indirect: &mut Vec<Id>,
    ) -> Option<([Id; 3], bool, (u32, u32), (u32, u32), (MvFmt, MvFmt))> {
        let n = ex.gate.len();
        if n == 0 || ex.down.len() != n || ex.up.as_ref().is_some_and(|u| u.len() != n) || ex.k > 64 {
            return None;
        }
        let fused = ex.up.is_none();
        let g_rows = if fused { 2 * ex.inter } else { ex.inter };
        // Format, (gpr, gk) and memory regions (data, scales, offsets) of one weight.
        type Info = (MvFmt, (u32, u32), [Option<(usize, usize)>; 3]);
        let info = |w: &WeightRef, n_out: usize, n_in: usize| -> Option<Info> {
            fn span<T>(v: &[T]) -> Option<(usize, usize)> {
                Some((v.as_ptr() as usize, std::mem::size_of_val(v)))
            }
            let fmt = MvFmt::of(w)?;
            if !n_in.is_multiple_of(fmt.chunk()) || !n_in.is_multiple_of(32) {
                return None;
            }
            Some(match w {
                WeightRef::Bf16(v) => (fmt, ((n_in * 2) as u32, 0), [span(v), None, None]),
                WeightRef::Q8(m) => (fmt, (0, 0), [span(&m.q[..]), span(&m.scales[..]), None]),
                WeightRef::Q4(m) => {
                    let q = match &m.offsets {
                        Some(o) => ((n_in / o.group) as u32, (o.group / 32) as u32),
                        None => (0, 0),
                    };
                    (
                        fmt,
                        q,
                        [span(&m.q[..]), span(&m.scales[..]), m.offsets.as_ref().and_then(|o| span(&o.v[..]))],
                    )
                }
                WeightRef::Gguf(g) => {
                    if g.n_out != n_out || g.n_in != n_in {
                        return None;
                    }
                    (fmt, (g.row_bytes() as u32, 0), [span(&g.data[..]), None, None])
                }
            })
        };
        let lists: [(&[WeightRef], usize, usize); 3] = [
            (&ex.gate, g_rows, ex.h),
            (ex.up.as_deref().unwrap_or(&[]), ex.inter, ex.h),
            (&ex.down, ex.h, ex.inter),
        ];
        let mut infos: Vec<Vec<Info>> = Vec::new();
        for (ws, n_out, n_in) in lists {
            let v = ws.iter().map(|w| info(w, n_out, n_in)).collect::<Option<Vec<_>>>()?;
            if v.iter().any(|i| i.0 != v[0].0 || i.1 != v[0].1) {
                return None; // one kernel per projection
            }
            infos.push(v);
        }
        if !fused && (infos[1][0].0 != infos[0][0].0 || infos[1][0].1 != infos[0][0].1) {
            return None; // gate and up share the pair kernel
        }
        let mut all: Vec<(usize, usize)> = infos.iter().flatten().flat_map(|i| i.2.iter().flatten().copied()).collect();
        all.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (p, n) in all {
            match spans.last_mut() {
                Some((s, e)) if p <= *e => *e = (*e).max(p + n),
                _ => spans.push((p, p + n)),
            }
        }
        let mut addr = Vec::with_capacity(spans.len());
        for &(s, e) in &spans {
            let b = self.wrap(s as *const u8, e - s)?;
            addr.push(self.gpu_address(&b));
            owned.push(b.id);
            indirect.push(b.id);
        }
        let at = |p: usize| {
            let i = spans.partition_point(|&(s, _)| s <= p) - 1;
            addr[i] + (p - spans[i].0) as u64
        };
        let copy: MsgNewBufBytes = self.o.msg_send();
        let mut tabs = [std::ptr::null_mut(); 3];
        for (t, v) in tabs.iter_mut().zip(&infos) {
            if v.is_empty() {
                continue;
            }
            let tab: Vec<[u64; 4]> = v
                .iter()
                .map(|i| {
                    let d = at(i.2[0].unwrap().0);
                    [d, i.2[1].map_or(d, |r| at(r.0)), i.2[2].map_or(d, |r| at(r.0)), 0]
                })
                .collect();
            *t = unsafe {
                copy(
                    self.device,
                    self.sels.new_buffer_bytes,
                    tab.as_ptr() as *const c_void,
                    tab.len() * 32,
                    0,
                )
            };
            if t.is_null() {
                return None;
            }
            owned.push(*t);
        }
        if fused {
            tabs[1] = tabs[0];
        }
        Some((tabs, fused, infos[0][0].1, infos[2][0].1, (infos[0][0].0, infos[2][0].0)))
    }
}

impl MetalPlan<'_> {
    /// Brings the device KV copies in line with the host caches (`pos` positions each) and makes
    /// room for one more position.
    fn sync_kv(&self, slot: &mut MetalSlot, kv: &mut [&mut crate::model::LayerCache], pos: usize) -> bool {
        let be = self.be;
        let new_len: MsgNewBufLen = be.o.msg_send();
        for (i, (m, c)) in slot.kv.iter_mut().zip(kv.iter_mut()).enumerate() {
            let row = self.kv[i].0 * self.kv[i].1;
            if c.len != pos || c.n_kv * c.hd != row {
                return false;
            }
            // A truncation (rewind) since the last step invalidates the copy from there on, even
            // if the host cache has grown back past it.
            m.synced = m.synced.min(c.len).min(c.kept);
            if m.cap < pos + 1 {
                let cap = (pos + 1).next_power_of_two().max(256);
                let bytes = cap * row * 4;
                let (k, v) = unsafe {
                    (
                        new_len(be.device, be.sels.new_buffer_len, bytes, 0),
                        new_len(be.device, be.sels.new_buffer_len, bytes, 0),
                    )
                };
                if k.is_null() || v.is_null() {
                    for b in [k, v] {
                        if !b.is_null() {
                            unsafe { (self.f.void)(b, self.f.s_release) };
                        }
                    }
                    return false;
                }
                if m.synced > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.f.contents(m.k), self.f.contents(k), m.synced * row * 4);
                        std::ptr::copy_nonoverlapping(self.f.contents(m.v), self.f.contents(v), m.synced * row * 4);
                    }
                }
                for b in [m.k, m.v] {
                    if !b.is_null() {
                        unsafe { (self.f.void)(b, self.f.s_release) };
                    }
                }
                (m.k, m.v, m.cap) = (k, v, cap);
            }
            if m.synced < c.len {
                let (hk, hv) = c.data();
                let r = m.synced * row..c.len * row;
                unsafe {
                    let dk = (self.f.contents(m.k) as *mut f32).add(r.start);
                    let dv = (self.f.contents(m.v) as *mut f32).add(r.start);
                    super::par_copy(dk, hk[r.clone()].as_ptr(), r.len());
                    super::par_copy(dv, hv[r.clone()].as_ptr(), r.len());
                }
                m.synced = c.len;
            }
            c.kept = c.len;
        }
        true
    }
}

impl MetalPlan<'_> {
    /// Wraps the host recurrent states as device buffers (in place; re-wrapped only when one
    /// moved). False if one cannot be wrapped without a copy.
    fn wrap_states(&self, slot: &mut MetalSlot, states: &[&mut [f32]]) -> bool {
        slot.states.resize_with(states.len(), || (0, 0, None));
        for ((addr, len, buf), st) in slot.states.iter_mut().zip(states) {
            let (a, l) = (st.as_ptr() as usize, st.len() * 4);
            if *addr == a && *len == l && buf.is_some() {
                continue;
            }
            if let Some(b) = buf.take() {
                unsafe { (self.f.void)(b.id, self.f.s_release) };
            }
            let Some(b) = self.be.wrap(st.as_ptr() as *const u8, l) else {
                return false;
            };
            // A copy would leave the host state behind.
            if self.f.contents(b.id) as usize + b.offset_bytes != a {
                unsafe { (self.f.void)(b.id, self.f.s_release) };
                return false;
            }
            (*addr, *len, *buf) = (a, l, Some(b));
        }
        true
    }
}

impl DecodePlan for MetalPlan<'_> {
    fn step(&self, slot: &mut Slot, s: Step) -> bool {
        let t_start = Instant::now();
        let _g = self.lock.lock().unwrap();
        if s.kv.len() != self.kv.len()
            || s.rope_cs.len() != 2 * self.rope_halves.iter().sum::<usize>()
            || self.input2.is_some_and(|(_, n)| s.x2.len() != n)
            || s.hidden.len() < self.hidden.1
            || s.states.len() != self.states.len()
            || s.states.iter().zip(&self.states).any(|(a, &n)| a.len() != n)
        {
            return false;
        }
        let fresh = !matches!(slot.0.as_ref().and_then(|b| b.downcast_ref::<MetalSlot>()), Some(m) if m.plan == self.id);
        if fresh {
            let release = (self.f.void, self.f.s_release);
            slot.0 = Some(Box::new(MetalSlot {
                plan: self.id,
                kv: (0..self.kv.len())
                    .map(|_| KvMirror {
                        k: std::ptr::null_mut(),
                        v: std::ptr::null_mut(),
                        cap: 0,
                        synced: 0,
                        release,
                    })
                    .collect(),
                states: Vec::new(),
                release,
            }));
        }
        let ms = slot.0.as_mut().unwrap().downcast_mut::<MetalSlot>().unwrap();
        let pos = s.pos;
        if !self.sync_kv(ms, s.kv, pos) || !self.wrap_states(ms, s.states) {
            return false;
        }
        let stb: Vec<(Id, usize)> = ms
            .states
            .iter()
            .map(|(_, _, b)| b.as_ref().map_or((std::ptr::null_mut(), 0), |b| (b.id, b.offset_bytes)))
            .collect();
        let t_synced = Instant::now();
        let arena = self.arena;
        let base = self.f.contents(arena);
        unsafe { std::ptr::copy_nonoverlapping(s.x.as_ptr(), base.add(self.input) as *mut f32, self.hidden.1) };
        if let Some((off, n)) = self.input2 {
            unsafe { std::ptr::copy_nonoverlapping(s.x2.as_ptr(), base.add(off) as *mut f32, n) };
        }
        match (self.params, s.params) {
            (Some(((ho, hn), (co, cn))), Some((hot, cold, key))) if hot.len() == hn && cold.len() == cn => {
                unsafe { std::ptr::copy_nonoverlapping(hot.as_ptr(), base.add(ho) as *mut f32, hn) };
                if self.params_key.load(Ordering::Relaxed) != key {
                    unsafe { std::ptr::copy_nonoverlapping(cold.as_ptr(), base.add(co) as *mut f32, cn) };
                    self.params_key.store(key, Ordering::Relaxed);
                }
            }
            (None, None) => {}
            _ => return false,
        }
        let want_logits = s.logits.is_some() && self.logits.is_some();
        let end = if want_logits { self.ops.len() } else { self.head_from };
        let kvb: Vec<(Id, Id)> = ms.kv.iter().map(|m| (m.k, m.v)).collect();
        let consts = self.consts;
        let cs = s.rope_cs;
        // The first part is small, so the GPU starts while the rest is encoded.
        let at = |from: usize| (from..end).find(|&i| matches!(self.ops[i], MOp::Barrier)).unwrap_or(end);
        static SPLITS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
        let splits = SPLITS.get_or_init(|| {
            std::env::var("OXINFER_DEC_SPLITS")
                .ok()
                .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_else(|| vec![16])
        });
        let mut parts = Vec::new();
        let mut from = 0;
        for &d in splits {
            let s = at((end / d.max(1)).max(from));
            parts.push(from..s);
            from = s;
        }
        parts.push(from..end);
        let indirect = if self.be.residency.is_some() { &[][..] } else { &self.indirect[..] };
        let (ok, gpu) = run_concurrent(self.be, &self.f, &parts, |e, range| {
            e.use_resources(indirect);
            // OXINFER_DEC_ONLY=barrier,norm,mv,qk,attn (debugging): encode only these ops.
            static ONLY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
            let only = ONLY.get_or_init(|| std::env::var("OXINFER_DEC_ONLY").ok());
            for op in &self.ops[range] {
                if let Some(o) = only {
                    let k = match op {
                        MOp::Barrier => "barrier",
                        MOp::Norm { .. } => "norm",
                        MOp::Mv { .. } => "mv",
                        MOp::Topk { .. } | MOp::Experts { .. } => "moe",
                        MOp::QkRope { .. } => "qk",
                        MOp::GdnConv { .. } | MOp::GdnRule { .. } => "gdn",
                        MOp::Glu { .. } | MOp::Ew { .. } => "glu",
                        MOp::Steer { .. } | MOp::NeuronEdit { .. } => "iv",
                        MOp::AddNorm { .. } => "norm",
                        MOp::Attn { .. } => "attn",
                    };
                    if !o.split(',').any(|x| x == k) {
                        continue;
                    }
                }
                match op {
                    MOp::Barrier => e.barrier(),
                    MOp::Norm {
                        pipe,
                        x,
                        out,
                        w,
                        p,
                        rows,
                        threads,
                    } => {
                        e.pipe(*pipe);
                        e.buf(arena, *x, 0);
                        e.buf(arena, *out, 1);
                        e.buf(consts, w.unwrap_or(0), 2);
                        e.bytes(p, 3);
                        e.dispatch((*rows, 1), *threads);
                    }
                    MOp::Mv {
                        pipe,
                        nr,
                        ks,
                        w,
                        w2,
                        x,
                        y,
                        mode,
                        gate,
                    } => encode_mv(e, *pipe, *nr, *ks, w, w2.as_ref(), (arena, *x), (arena, *y), *mode, (arena, *gate)),
                    MOp::Topk {
                        pipe,
                        logits,
                        sel,
                        n,
                        k,
                        scale,
                    } => {
                        e.pipe(*pipe);
                        e.buf(arena, *logits, 0);
                        e.buf(arena, *sel, 1);
                        e.bytes(&[*n, *k, scale.is_some() as u32, 0], 2);
                        e.buf(consts, scale.unwrap_or(0), 3);
                        e.dispatch((1, 1), (*n as usize).next_multiple_of(32));
                    }
                    MOp::Experts {
                        pair,
                        sum,
                        tabs,
                        fused,
                        gelu,
                        add,
                        g,
                        d,
                        x,
                        sel,
                        act,
                        y,
                        h,
                        inter,
                        k,
                        edit,
                    } => {
                        let p = |n_out: usize, n_in: usize, mode: MvMode, q: (u32, u32), r2: usize| MvParams {
                            n_out: n_out as u32,
                            n_in: n_in as u32,
                            mode: mode as u32,
                            rows_per_tg: 1,
                            gpr: q.0,
                            gk: q.1,
                            ldy: *k as u32,
                            _p: r2 as u32,
                            ..Default::default()
                        };
                        e.pipe(*pair);
                        e.buf(tabs[0], 0, 0);
                        e.buf(if *fused { tabs[0] } else { tabs[1] }, 0, 1);
                        e.buf(arena, *x, 2);
                        e.buf(arena, *act, 3);
                        let am = if *gelu { MvMode::GeluPair } else { MvMode::SiluPair };
                        e.bytes(&p(*inter, *h, am, *g, if *fused { *inter } else { 0 }), 4);
                        e.buf(arena, *sel, 5);
                        e.dispatch((*inter, *k), 32);
                        e.barrier();
                        if let Some((pipe, hot, n)) = edit {
                            e.pipe(*pipe);
                            e.buf(arena, *act, 0);
                            e.buf(arena, *hot, 1);
                            e.buf(arena, *sel, 2);
                            e.bytes(&[*n, *inter as u32, *k as u32, 1u32], 3);
                            e.dispatch(((*n as usize).div_ceil(64), 1), 64);
                            e.barrier();
                        }
                        e.pipe(*sum);
                        e.buf(tabs[2], 0, 0);
                        e.buf(tabs[2], 0, 1);
                        e.buf(arena, *act, 2);
                        e.buf(arena, *y, 3);
                        e.bytes(&p(*h, *inter, if *add { MvMode::Add } else { MvMode::Store }, *d, 0), 4);
                        e.buf(arena, *sel, 5);
                        e.dispatch((*h, 1), 32);
                    }
                    MOp::QkRope {
                        pipe,
                        qkv,
                        qo,
                        kv,
                        qn,
                        kn,
                        p,
                        cs: c0,
                    } => {
                        let mut p = *p;
                        p.pos = pos as u32;
                        e.pipe(*pipe);
                        e.buf(arena, *qkv, 0);
                        e.buf(arena, *qo, 1);
                        e.buf(kvb[*kv].0, 0, 2);
                        e.buf(kvb[*kv].1, 0, 3);
                        e.buf(consts, qn.unwrap_or(0), 4);
                        e.buf(consts, kn.unwrap_or(0), 5);
                        e.bytes(&p, 6);
                        e.bytes_slice(&cs[*c0..*c0 + 2 * p.half_rot as usize], 7);
                        e.dispatch(((p.nh + 2 * p.nkv) as usize, 1), p.hd as usize);
                    }
                    MOp::AddNorm {
                        pipe,
                        y,
                        x,
                        out,
                        wy,
                        wn,
                        p,
                        threads,
                    } => {
                        e.pipe(*pipe);
                        e.buf(arena, *y, 0);
                        e.buf(arena, *x, 1);
                        e.buf(arena, *out, 2);
                        e.buf(consts, *wy, 3);
                        e.buf(consts, *wn, 4);
                        e.bytes(p, 5);
                        e.dispatch((1, 1), *threads);
                    }
                    MOp::Ew { pipe, a, b, out, n, p } => {
                        e.pipe(*pipe);
                        e.buf(arena, *a, 0);
                        e.buf(arena, *b, 1);
                        e.buf(arena, *out, 2);
                        e.bytes(p, 3);
                        e.bytes(n, 4);
                        e.dispatch(((*n as usize).div_ceil(256), 1), 256);
                    }
                    MOp::Steer { pipe, x, hot, cold, p } => {
                        e.pipe(*pipe);
                        e.buf(arena, *x, 0);
                        e.buf(arena, *hot, 1);
                        e.buf(arena, *cold, 2);
                        e.bytes(p, 3);
                        e.dispatch((1, 1), 1024);
                    }
                    MOp::NeuronEdit { pipe, act, hot, n } => {
                        e.pipe(*pipe);
                        e.buf(arena, *act, 0);
                        e.buf(arena, *hot, 1);
                        e.buf(arena, *hot, 2);
                        e.bytes(&[*n, 0u32, 0, 0], 3);
                        e.dispatch(((*n as usize).div_ceil(64), 1), 64);
                    }
                    MOp::Glu { pipe, g, u, out, n, gelu } => {
                        e.pipe(*pipe);
                        e.buf(arena, *g, 0);
                        e.buf(arena, *u, 1);
                        e.buf(arena, *out, 2);
                        e.bytes(&[*n, *gelu], 3);
                        e.dispatch(((*n as usize).div_ceil(256), 1), 256);
                    }
                    MOp::GdnConv {
                        pipe,
                        x,
                        out,
                        state,
                        w,
                        dim,
                        kc,
                    } => {
                        #[repr(C)]
                        struct Gc {
                            dim: u32,
                            kc: u32,
                        }
                        e.pipe(*pipe);
                        e.buf(arena, *x, 0);
                        e.buf(arena, *out, 1);
                        e.buf(stb[*state].0, stb[*state].1, 2);
                        e.buf(consts, *w, 3);
                        e.bytes(&Gc { dim: *dim, kc: *kc }, 4);
                        e.dispatch(((*dim as usize).div_ceil(256), 1), 256);
                    }
                    MOp::GdnRule {
                        pipe,
                        c,
                        z,
                        b,
                        a,
                        out,
                        state,
                        cst,
                        p,
                    } => {
                        e.pipe(*pipe);
                        e.buf(arena, *c, 0);
                        e.buf(arena, *z, 1);
                        e.buf(arena, *b, 2);
                        e.buf(arena, *a, 3);
                        e.buf(arena, *out, 4);
                        e.buf(stb[*state].0, stb[*state].1, 5);
                        e.buf(consts, *cst, 6);
                        e.bytes(p, 7);
                        e.dispatch((p.nv as usize, 1), (p.hdk / 32 * p.hdv) as usize);
                    }
                    MOp::Attn {
                        grp,
                        pipe,
                        combine,
                        q,
                        out,
                        kv,
                        gate,
                        window,
                        p,
                        threads,
                        hd,
                    } => {
                        let mut p = *p;
                        p.len = (pos + 1) as u32;
                        p.lo = window.map_or(0, |w| (pos + 1).saturating_sub(w)) as u32;
                        // Keys per threadgroup: one pass of its 8 SIMD-groups (256 keys) at least, so
                        // long contexts spread over up to ATTN_MAX_SPLIT threadgroups per head.
                        let n = (p.len - p.lo) as usize;
                        let chunk = n.div_ceil(ATTN_MAX_SPLIT).next_multiple_of(256).max(256);
                        p.nsplit = n.div_ceil(chunk) as u32;
                        p.chunk = chunk as u32;
                        e.pipe(*pipe);
                        e.buf(arena, *q, 0);
                        e.buf(kvb[*kv].0, 0, 1);
                        e.buf(kvb[*kv].1, 0, 2);
                        e.buf(arena, *out, 3);
                        e.buf(arena, gate.unwrap_or(0), 4);
                        e.bytes(&p, 5);
                        e.buf(arena, self.part, 6);
                        e.dispatch((p.nh as usize / grp, p.nsplit as usize), *threads);
                        if p.nsplit > 1 {
                            e.barrier();
                            e.pipe(*combine);
                            e.buf(arena, self.part, 0);
                            e.buf(arena, *out, 1);
                            e.buf(arena, gate.unwrap_or(0), 2);
                            e.bytes(&p, 3);
                            e.bytes(hd, 4);
                            e.dispatch((p.nh as usize, 1), 256);
                        }
                    }
                }
            }
        });
        if !ok {
            return false;
        }
        let t_ran = Instant::now();
        let _ = gpu;
        unsafe {
            std::ptr::copy_nonoverlapping(base.add(self.hidden.0) as *const f32, s.hidden.as_mut_ptr(), self.hidden.1);
            if let (Some(l), Some((off, n))) = (s.logits, self.logits)
                && want_logits
            {
                super::par_copy(l.as_mut_ptr(), base.add(off) as *const f32, n.min(l.len()));
            }
        }
        // The new keys and values go back to the host caches.
        for (i, (m, c)) in ms.kv.iter_mut().zip(s.kv.iter_mut()).enumerate() {
            let row = self.kv[i].0 * self.kv[i].1;
            unsafe {
                let k = std::slice::from_raw_parts((self.f.contents(m.k) as *const f32).add(pos * row), row);
                let v = std::slice::from_raw_parts((self.f.contents(m.v) as *const f32).add(pos * row), row);
                c.push(k, v, 1);
            }
            m.synced = pos + 1;
            c.kept = c.len;
        }
        if trace() {
            eprintln!(
                "[metal decode step: sync {:.0} us, run {:.0} us, read back {:.0} us]",
                (t_synced - t_start).as_secs_f64() * 1e6,
                (t_ran - t_synced).as_secs_f64() * 1e6,
                t_ran.elapsed().as_secs_f64() * 1e6
            );
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::GgufMatrix;
    use crate::quant::{PageVec, Q4Matrix, QBuf, QMatrix};

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    /// One command buffer of `encode` calls; returns the GPU time in seconds.
    fn timed(be: &MetalBackend, encode: impl Fn(&FastEnc)) -> f64 {
        let f = Fns::new(be);
        let time: unsafe extern "C" fn(Id, Sel) -> f64 = be.o.msg_send();
        unsafe {
            let pool = (be.sels.pool_push)();
            let cb = (f.msg0)(be.queue, f.s_cb);
            let enc = if std::env::var_os("OXINFER_MV_CONC").is_some() {
                (f.enc_type)(cb, f.s_enc_type, 1)
            } else {
                (f.msg0)(cb, be.sels.compute_encoder)
            };
            encode(&FastEnc { enc, f: &f });
            (f.void)(enc, f.s_end);
            (f.void)(cb, f.s_commit);
            (f.void)(cb, f.s_wait);
            let g = time(cb, be.o.sel(c"GPUEndTime")) - time(cb, be.o.sel(c"GPUStartTime"));
            (be.sels.pool_pop)(pool);
            g
        }
    }

    /// A random matrix of `fmt`: its bytes, its scale bytes (oxinfer Q8 / Q4) and its values.
    fn matrix(fmt: MvFmt, n_out: usize, n_in: usize, seed: &mut u64) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        let w16: Vec<u16> = (0..n_out * n_in).map(|_| (lcg(seed).to_bits() >> 16) as u16).collect();
        let mut deq = vec![0f32; n_out * n_in];
        let (data, scales) = match fmt {
            MvFmt::Bf16 => {
                deq.iter_mut().zip(&w16).for_each(|(o, &b)| *o = f32::from_bits((b as u32) << 16));
                (w16.iter().flat_map(|v| v.to_le_bytes()).collect(), vec![0u8; 16])
            }
            MvFmt::Q8 => {
                let m = crate::quant::quantize_bf16(&w16, n_out, n_in);
                crate::quant::dequantize_rows(&m, 0, n_out, &mut deq);
                (
                    m.q.iter().map(|&v| v as u8).collect(),
                    m.scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )
            }
            MvFmt::Q4 => {
                let m = crate::quant::quantize_q4(&w16, n_out, n_in);
                crate::quant::dequantize_rows_q4(&m, 0, n_out, &mut deq);
                (m.q.to_vec(), m.scales.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            MvFmt::Gguf(k) => {
                let raw = crate::ggml::random_blocks(k, n_out * n_in / k.block_size(), seed);
                GgufMatrix::from_bytes(k, &raw, n_out, n_in)
                    .unwrap()
                    .dequantize_rows(0, n_out, &mut deq);
                (raw, vec![0u8; 16])
            }
        };
        (data, scales, deq)
    }

    /// Every matvec kernel variant against an exact (f64) product, in the store, add and SiLU
    /// pair modes, with a row count that leaves a partial threadgroup.
    #[test]
    fn decode_mv_matches_reference() {
        let Ok(be) = MetalBackend::new() else { return };
        let f = Fns::new(&be);
        let (n_out, n_in) = (37, 512);
        let mut seed = 3u64;
        let fmts = [
            MvFmt::Bf16,
            MvFmt::Q8,
            MvFmt::Q4,
            MvFmt::Gguf(GgmlType::F16),
            MvFmt::Gguf(GgmlType::Q4_0),
            MvFmt::Gguf(GgmlType::Q8_0),
            MvFmt::Gguf(GgmlType::Q4_K),
            MvFmt::Gguf(GgmlType::Q5_K),
            MvFmt::Gguf(GgmlType::Q6_K),
            MvFmt::Gguf(GgmlType::Q3_K),
            MvFmt::Gguf(GgmlType::IQ4_XS),
        ];
        for fmt in fmts {
            let (d1, s1, w1) = matrix(fmt, n_out, n_in, &mut seed);
            let (d2, s2, w2) = matrix(fmt, n_out, n_in, &mut seed);
            // Page-aligned copies for wrapping: [d1 | d2] and [s1 | s2].
            let dl = d1.len().next_multiple_of(4096);
            let sl = s1.len().next_multiple_of(4096);
            let mut data = PageVec::<u8>::zeroed(2 * dl);
            let mut sc = PageVec::<u8>::zeroed(2 * sl);
            data[..d1.len()].copy_from_slice(&d1);
            data[dl..dl + d2.len()].copy_from_slice(&d2);
            sc[..s1.len()].copy_from_slice(&s1);
            sc[sl..sl + s2.len()].copy_from_slice(&s2);
            let db = be.wrap(data.as_ptr(), data.len()).unwrap();
            let sb = be.wrap(sc.as_ptr(), sc.len()).unwrap();
            let wt = |i: usize| MvWeight {
                fmt,
                n_out,
                n_in,
                data: db.id,
                data_off: (db.offset_bytes + i * dl) as u64,
                scales: sb.id,
                scales_off: (sb.offset_bytes + i * sl) as u64,
                offs: db.id,
                offs_off: 0,
                gpr: match fmt {
                    MvFmt::Gguf(k) => k.row_bytes(n_in) as u32,
                    MvFmt::Bf16 => (n_in * 2) as u32,
                    _ => 0,
                },
                gk: 0,
            };
            let x: Vec<f32> = (0..n_in).map(|_| lcg(&mut seed)).collect();
            let y0: Vec<f32> = (0..n_out).map(|_| lcg(&mut seed)).collect();
            let dot = |w: &[f32], r: usize| {
                w[r * n_in..(r + 1) * n_in]
                    .iter()
                    .zip(&x)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>()
            };
            let mut st = be.staging.lock().unwrap();
            let st = &mut *st;
            assert!(be.ensure_staging(st, n_in * 4, n_out * 4));
            let variants: Vec<(usize, usize)> = if fmt.specialized() { MV_VARIANTS.to_vec() } else { vec![(2, 4)] };
            for (nr, ks) in variants {
                let pipe = be.mv_pipe(fmt, nr, ks).unwrap();
                for mode in [MvMode::Store, MvMode::Add, MvMode::SiluPair] {
                    unsafe {
                        std::ptr::copy_nonoverlapping(x.as_ptr(), f.contents(st.x) as *mut f32, n_in);
                        std::ptr::copy_nonoverlapping(y0.as_ptr(), f.contents(st.y) as *mut f32, n_out);
                    }
                    let up = wt(1);
                    let pair = (mode == MvMode::SiluPair).then_some(&up);
                    let (ok, _) = run_concurrent(&be, &f, &[std::ops::Range { start: 0, end: 1 }], |e, _| {
                        encode_mv(e, pipe, nr, ks, &wt(0), pair, (st.x, 0), (st.y, 0), mode, (st.x, 0))
                    });
                    assert!(ok);
                    let y = unsafe { std::slice::from_raw_parts(f.contents(st.y) as *const f32, n_out) };
                    for r in 0..n_out {
                        let a = dot(&w1, r);
                        let want = match mode {
                            MvMode::Store => a,
                            MvMode::Add => y0[r] as f64 + a,
                            _ => a / (1.0 + (-a).exp()) * dot(&w2, r),
                        };
                        let err = (y[r] as f64 - want).abs() / (want.abs() + 1e-3);
                        assert!(err < 1e-4, "{fmt:?} {nr}x{ks} {mode:?} row {r}: {} vs {want}", y[r]);
                    }
                }
            }
            for b in [db.id, sb.id] {
                unsafe { (f.void)(b, f.s_release) };
            }
        }
    }

    /// Kernel throughput over copies of a matrix too large for the caches, and agreement with the
    /// CPU kernels. `OXINFER_MV_FMTS=Q4_K,q8,...` and `OXINFER_MV_SHAPES=4096x4096,...` pick cases.
    #[test]
    #[ignore]
    fn decode_mv_bench() {
        let be = MetalBackend::new().unwrap();
        let fmts_env = std::env::var("OXINFER_MV_FMTS").unwrap_or("bf,q8,q4,Q4_0,Q8_0,Q4_K,Q5_K,Q6_K,F16,Q3_K".into());
        let shapes_env = std::env::var("OXINFER_MV_SHAPES").unwrap_or("2048x2048,512x2048,8192x2048,2048x8192,128256x2048".into());
        let shapes: Vec<(usize, usize)> = shapes_env
            .split(',')
            .map(|s| {
                let (a, b) = s.split_once('x').unwrap();
                (a.parse().unwrap(), b.parse().unwrap())
            })
            .collect();
        let mut seed = 7u64;
        for f in fmts_env.split(',') {
            let fmt = match f {
                "bf" => MvFmt::Bf16,
                "q8" => MvFmt::Q8,
                "q4" => MvFmt::Q4,
                k => MvFmt::Gguf(GGUF_GPU_TYPES.iter().copied().find(|t| t.name() == k).unwrap()),
            };
            for &(n_out, n_in) in &shapes {
                // One matrix (data, scales), as bytes.
                let w16: Vec<u16> = (0..n_out * n_in).map(|_| (lcg(&mut seed).to_bits() >> 16) as u16).collect();
                let (data, scales): (Vec<u8>, Vec<u8>) = match fmt {
                    MvFmt::Bf16 => (w16.iter().flat_map(|v| v.to_le_bytes()).collect(), vec![0u8; 16]),
                    MvFmt::Q8 => {
                        let m = crate::quant::quantize_bf16(&w16, n_out, n_in);
                        (
                            m.q.iter().map(|&v| v as u8).collect(),
                            m.scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        )
                    }
                    MvFmt::Q4 => {
                        let m = crate::quant::quantize_q4(&w16, n_out, n_in);
                        (m.q.to_vec(), m.scales.iter().flat_map(|v| v.to_le_bytes()).collect())
                    }
                    MvFmt::Gguf(k) => (
                        crate::ggml::random_blocks(k, n_out * n_in / k.block_size(), &mut seed),
                        vec![0u8; 16],
                    ),
                };
                let copies = (768usize << 20).div_ceil(data.len()).clamp(1, 64);
                let dlen = data.len().next_multiple_of(4096);
                let slen = scales.len().next_multiple_of(4096);
                let mut big = PageVec::<u8>::zeroed(copies * dlen);
                let mut bigs = PageVec::<u8>::zeroed(copies * slen);
                for c in 0..copies {
                    big[c * dlen..c * dlen + data.len()].copy_from_slice(&data);
                    bigs[c * slen..c * slen + scales.len()].copy_from_slice(&scales);
                }
                let x: Vec<f32> = (0..n_in).map(|_| lcg(&mut seed) * 2.0).collect();
                // Exact reference (f64 over dequantized rows) for the first rows of copy 0.
                let nref = n_out.min(2048);
                let mut deq = vec![0f32; nref * n_in];
                match fmt {
                    MvFmt::Bf16 => deq.iter_mut().zip(&w16).for_each(|(o, &b)| *o = f32::from_bits((b as u32) << 16)),
                    MvFmt::Q8 => {
                        let m = QMatrix {
                            q: unsafe { QBuf::borrowed(std::slice::from_raw_parts(big.as_ptr() as *const i8, n_out * n_in)) },
                            scales: unsafe { QBuf::borrowed(std::slice::from_raw_parts(bigs.as_ptr() as *const f32, n_out * n_in / 32)) },
                            n_out,
                            n_in,
                        };
                        crate::quant::dequantize_rows(&m, 0, nref, &mut deq)
                    }
                    MvFmt::Q4 => {
                        let m = Q4Matrix {
                            q: unsafe { QBuf::borrowed(&big[..n_out * n_in / 2]) },
                            scales: unsafe { QBuf::borrowed(std::slice::from_raw_parts(bigs.as_ptr() as *const u16, n_out * n_in / 32)) },
                            n_out,
                            n_in,
                            offsets: None,
                        };
                        crate::quant::dequantize_rows_q4(&m, 0, nref, &mut deq)
                    }
                    MvFmt::Gguf(k) => GgufMatrix::from_bytes(k, &data, n_out, n_in)
                        .unwrap()
                        .dequantize_rows(0, nref, &mut deq),
                }
                let yr: Vec<f32> = (0..nref)
                    .map(|r| {
                        deq[r * n_in..(r + 1) * n_in]
                            .iter()
                            .zip(&x)
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum::<f64>() as f32
                    })
                    .collect();
                let wb = be.wrap(big.as_ptr(), big.len()).unwrap();
                let sbuf = be.wrap(bigs.as_ptr(), bigs.len()).unwrap();
                let mut st = be.staging.lock().unwrap();
                let st = &mut *st;
                assert!(be.ensure_staging(st, n_in * 4, n_out * 4));
                unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), be.contents(st.x) as *mut f32, n_in) };
                let reps = 64;
                let mut line = format!("{:>6} {n_out:>6}x{n_in:<5}", fmt.name());
                for &(nr, ks) in &MV_VARIANTS {
                    let suffix = std::env::var("OXINFER_MV_SUFFIX").unwrap_or_default();
                    let Some(pipe) = (if suffix.is_empty() {
                        be.mv_pipe(fmt, nr, ks)
                    } else {
                        be.named_pipeline(&format!("mv_{}_{nr}_{ks}{suffix}", fmt.name()))
                    }) else {
                        continue;
                    };
                    if !fmt.specialized() && (nr, ks) != (2, 4) {
                        continue;
                    }
                    let wt = |c: usize| MvWeight {
                        fmt,
                        n_out,
                        n_in,
                        data: wb.id,
                        data_off: (wb.offset_bytes + c * dlen) as u64,
                        scales: sbuf.id,
                        scales_off: (sbuf.offset_bytes + c * slen) as u64,
                        offs: wb.id,
                        offs_off: 0,
                        gpr: match fmt {
                            MvFmt::Gguf(k) => k.row_bytes(n_in) as u32,
                            MvFmt::Bf16 => (n_in * 2) as u32,
                            _ => 0,
                        },
                        gk: 0,
                    };
                    let mut best = f64::MAX;
                    let barrier = std::env::var_os("OXINFER_MV_BARRIER").is_some();
                    // Warm up (the GPU clocks down when idle).
                    let t_warm = std::time::Instant::now();
                    while t_warm.elapsed().as_millis() < 150 {
                        timed(&be, |e| {
                            for r in 0..reps {
                                encode_mv(
                                    e,
                                    pipe,
                                    nr,
                                    ks,
                                    &wt((r + 1) % copies),
                                    None,
                                    (st.x, 0),
                                    (st.y, 0),
                                    MvMode::Store,
                                    (st.x, 0),
                                );
                            }
                        });
                    }
                    // OXINFER_MV_PAIR=1: SiLU pairs of two copies (gate / up) instead.
                    let pair = std::env::var_os("OXINFER_MV_PAIR").is_some();
                    for _ in 0..5 {
                        best = best.min(timed(&be, |e| {
                            for r in 0..reps {
                                let w2 = wt((r + 2) % copies);
                                encode_mv(
                                    e,
                                    pipe,
                                    nr,
                                    ks,
                                    &wt((r + 1) % copies),
                                    pair.then_some(&w2),
                                    (st.x, 0),
                                    (st.y, 0),
                                    if pair { MvMode::SiluPair } else { MvMode::Store },
                                    (st.x, 0),
                                );
                                if barrier {
                                    e.barrier();
                                }
                            }
                        }));
                    }
                    let y = unsafe { std::slice::from_raw_parts(be.contents(st.y) as *const f32, n_out) };
                    let (mut e, mut m) = (0f32, 0f32);
                    for (a, b) in y[..nref].iter().zip(&yr) {
                        e = e.max((a - b).abs());
                        m = m.max(b.abs());
                    }
                    let gbs = ((1 + pair as usize) * reps * data.len()) as f64 / best / 1e9;
                    let e = if pair { 0.0 } else { e };
                    let _ = write!(
                        line,
                        " {nr}x{ks}:{gbs:>4.0}/{:.1}us{}",
                        best / reps as f64 * 1e6,
                        if e / m > 2e-5 {
                            format!("(ERR {:.1e})", e / m)
                        } else {
                            String::new()
                        }
                    );
                }
                println!("{line}");
            }
        }
    }
}
