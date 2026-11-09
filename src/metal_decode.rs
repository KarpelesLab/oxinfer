//! Single-token decode on the GPU: the kernels of `metal_decode.metal` and their encoding.

use super::*;
use std::fmt::Write as _;

const SRC: &str = include_str!("metal_decode.metal");

/// (rows per threadgroup, SIMD-groups splitting each row) variants compiled for each format.
pub(crate) const MV_VARIANTS: [(usize, usize); 4] = [(1, 1), (1, 2), (1, 4), (2, 4)];

/// Token capacities of the multi-token matvec kernels (a plan of up to N tokens uses the
/// smallest that holds N).
pub(crate) const MVK_TOKENS: [usize; 2] = [8, 16];

/// Tokens the multi-token matvecs take per sweep over the rows (more run as further sweeps, the
/// rows' weights then coming from the cache; `OXINFER_MVK_GROUP` overrides it for measurements).
/// With 2 rows per threadgroup, 4 measured best on an M5 Max (`decode_mvk_bench`: 8 tokens of a
/// 17408x5120 Q4 matrix in ~500 us, Q8 ~400 us, against ~100 / ~180 us for one token).
const MVK_GROUP: usize = 4;

/// Rows per threadgroup of the multi-token matvecs for up to `tt` tokens (accumulators: rows x
/// tokens per lane; each token's x chunk is loaded once for the rows). 2 measured best on an M5
/// Max end to end (`decode_mvk_bench` for single matrices; `OXINFER_MVK_ROWS` overrides it).
pub(crate) fn mvk_rows(_tt: usize) -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("OXINFER_MVK_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|r| [1, 2, 4].contains(r))
            .unwrap_or(2)
    })
}

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
    /// The multi-token instantiation (up to `tt` tokens; see `mvk_body` in the shader).
    fn instantiate_k(self, nr: usize, ks: usize, tt: usize) -> String {
        let one = self.instantiate(nr, ks);
        let (mac, rest) = one.split_once('(').unwrap();
        let rest = rest
            .strip_suffix(')')
            .unwrap()
            .replacen(&format!("mv_{}_{nr}_{ks}", self.name()), &self.mvk_name(nr, ks, tt), 1);
        format!("{}({rest}, {tt})", mac.replacen("MV_", "MVK_", 1))
    }
    fn mvk_name(self, nr: usize, ks: usize, tt: usize) -> String {
        format!("mvk_{}_{nr}_{ks}_{tt}", self.name())
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
    let group = std::env::var("OXINFER_MVK_GROUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|g| (1..=16).contains(g))
        .unwrap_or(MVK_GROUP);
    let mut s = format!("#define MVK_GROUP {group}\n");
    s.push_str(SRC);
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
        // A few rows at once (`mvt_<fmt>_1_<ks>`, see `MetalBackend::mvt_pipe`).
        for ks in [1, 2, 4] {
            let name = format!("mvt_{n}_1_{ks}");
            let _ = match f {
                MvFmt::Bf16 => writeln!(s, "MVT_RAW({name}, MBf, X16, 16, 1, {ks})"),
                MvFmt::Q8 => writeln!(s, "MVT_Q8({name}, 1, {ks})"),
                MvFmt::Q4 => writeln!(s, "MVT_Q4({name}, 1, {ks})"),
                MvFmt::Gguf(GgmlType::F32) => writeln!(s, "MVT_LOAD({name}, F32W2, 1, {ks})"),
                MvFmt::Gguf(k) => writeln!(s, "MVT_LOAD({name}, {}W, 1, {ks})", k.name()),
            };
        }
        // Multi-token: the SIMD-group splits of the single-token variants (the same per-row
        // arithmetic), more rows per threadgroup (each x chunk loaded once for them).
        for ks in [1, 2, 4] {
            if f.specialized() || ks == 4 {
                for tt in MVK_TOKENS {
                    let _ = writeln!(s, "{}", f.instantiate_k(mvk_rows(tt), ks, tt));
                    // decode_mvk_bench: the other row counts too.
                    if std::env::var_os("OXINFER_MVK_ALL").is_some() {
                        for nr in [1, 2, 4].into_iter().filter(|&n| n != mvk_rows(tt)) {
                            let _ = writeln!(s, "{}", f.instantiate_k(nr, ks, tt));
                        }
                    }
                }
            }
        }
    }
    // The sources appended after these (the matrix-unit GEMMs) keep the default (fast) math.
    s.push_str("#pragma clang fp reassociate(on)\n");
    // Fast verification matmuls (`fv_<fmt>_<tokens>`).
    s.push_str(FV_SRC);
    for f in FV_FORMATS {
        let (w, mk) = fv_types(f);
        if let MvFmt::Gguf(_) = f {
            let _ = writeln!(s, "#define {mk}(W, S, O, WO, SO, OO) {w}{{ {{ (W) + (WO), p.gpr }} }}");
        }
        for tt in MVK_TOKENS {
            let _ = writeln!(s, "FV_INST({}, {w}, {mk}, {tt})", fv_name(f, false, tt));
            if f == MvFmt::Q4 {
                let _ = writeln!(s, "FV_INST({}, Q4A, FV_MK_Q4A, {tt})", fv_name(f, true, tt));
            }
        }
    }
    s
}

/// The weight loader type of a format for the fast matmuls and its constructor macro.
fn fv_types(f: MvFmt) -> (String, String) {
    match f {
        MvFmt::Bf16 => ("BfW".into(), "FV_MK_BF".into()),
        MvFmt::Q8 => ("Q8W".into(), "FV_MK_Q8".into()),
        MvFmt::Q4 => ("Q4W".into(), "FV_MK_Q4".into()),
        MvFmt::Gguf(k) => (
            if k == GgmlType::F32 {
                "F32W2".to_string()
            } else {
                format!("{}W", k.name())
            },
            format!("FV_MK_{}", k.name()),
        ),
    }
}

impl MetalBackend {
    /// The fast matmul of a weight for up to `tt` tokens (one of [`MVK_TOKENS`]), and its rows
    /// per threadgroup.
    fn fv_pipe(&self, w: &MvWeight, tt: usize) -> Option<(Id, usize)> {
        if !FV_FORMATS.contains(&w.fmt) {
            return None;
        }
        Some((self.named_pipeline(&fv_name(w.fmt, w.gk != 0, tt))?, FV_ROWS))
    }
}

const FV_SRC: &str = include_str!("metal_fastv.metal");
/// Rows per threadgroup of the fast verification matmuls (`FV_R` in the shader).
const FV_ROWS: usize = 32;
/// The formats with fast matmuls (a chunk loader `fv_load8` in the shader); other formats keep the
/// exact multi-token kernels in fast plans.
const FV_FORMATS: [MvFmt; 9] = [
    MvFmt::Bf16,
    MvFmt::Q8,
    MvFmt::Q4,
    MvFmt::Gguf(GgmlType::F16),
    MvFmt::Gguf(GgmlType::Q4_0),
    MvFmt::Gguf(GgmlType::Q8_0),
    MvFmt::Gguf(GgmlType::Q4_K),
    MvFmt::Gguf(GgmlType::Q5_K),
    MvFmt::Gguf(GgmlType::Q6_K),
];

/// The fast verification matmul of a format (affine: oxinfer Q4 with MLX offsets) for up to `tt`
/// tokens (one of [`MVK_TOKENS`]).
fn fv_name(f: MvFmt, affine: bool, tt: usize) -> String {
    format!("fv_{}{}_{tt}", f.name(), if affine { "a" } else { "" })
}

/// Whether fast plans use the fast matmuls (`OXINFER_FAST_KERNELS=0` keeps the exact multi-token
/// kernels in them, for measurements).
fn fast_kernels() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXINFER_FAST_KERNELS").is_none_or(|v| v != "0"))
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

impl MvWeight {
    /// A streamed weight (`crate::gpu::decode::StreamedWeight`): no buffers yet (the plan wraps
    /// the unit's memory per step, see [`MvWeight::bind`]); returns the addresses of the data,
    /// scales and affine offsets (0 when absent).
    fn streamed(w: &WeightRef, n_out: usize, n_in: usize) -> Option<(Self, [usize; 3])> {
        let fmt = MvFmt::of(w)?;
        if !n_in.is_multiple_of(fmt.chunk()) || (matches!(fmt, MvFmt::Gguf(_)) && !n_in.is_multiple_of(32)) {
            return None;
        }
        let at = |p: *const u8| p as usize;
        let (gpr, gk, addr) = match w {
            WeightRef::Gguf(g) => {
                if g.block_scales.is_some() {
                    return None; // FP8 block scales: not on the decode kernels
                }
                (g.row_bytes() as u32, 0, [at(g.data.as_ptr()), 0, 0])
            }
            WeightRef::Bf16(v) => ((n_in * 2) as u32, 0, [at(v.as_ptr() as *const u8), 0, 0]),
            WeightRef::Q8(m) => (0, 0, [at(m.q.as_ptr() as *const u8), at(m.scales.as_ptr() as *const u8), 0]),
            WeightRef::Q4(m) => {
                let (gpr, gk, o) = match &m.offsets {
                    Some(o) => ((n_in / o.group) as u32, (o.group / 32) as u32, at(o.v.as_ptr() as *const u8)),
                    None => (0, 0, 0),
                };
                (gpr, gk, [at(m.q.as_ptr()), at(m.scales.as_ptr() as *const u8), o])
            }
        };
        let null = std::ptr::null_mut();
        Some((
            MvWeight {
                fmt,
                n_out,
                n_in,
                data: null,
                data_off: 0,
                scales: null,
                scales_off: 0,
                offs: null,
                offs_off: 0,
                gpr,
                gk,
            },
            addr,
        ))
    }

    /// Points a streamed weight at its memory inside `bufs`: `(span start, span end, buffer,
    /// buffer offset of the span start)`. False when an address lies in none of them.
    fn bind(&mut self, addr: &[usize; 3], bufs: &[(usize, usize, Id, usize)]) -> bool {
        let find = |a: usize| bufs.iter().find(|b| a >= b.0 && a < b.1).map(|b| (b.2, (a - b.0 + b.3) as u64));
        let Some((d, doff)) = find(addr[0]) else { return false };
        (self.data, self.data_off) = (d, doff);
        (self.scales, self.scales_off) = if addr[1] != 0 {
            match find(addr[1]) {
                Some(v) => v,
                None => return false,
            }
        } else {
            (d, 0)
        };
        (self.offs, self.offs_off) = if addr[2] != 0 {
            match find(addr[2]) {
                Some(v) => v,
                None => return false,
            }
        } else {
            (d, 0)
        };
        true
    }
}

/// A matvec of a plan whose weights stream (see `Graph::units`): its op index and the addresses
/// of its weight (and the pair's second weight).
struct StreamedMv {
    mop: usize,
    w: [usize; 3],
    w2: Option<[usize; 3]>,
}

/// Per expert, per projection (gate or fused gate/up, up, down): the addresses of its data,
/// scales and affine offsets (0: none).
type ExpertAddrs = [[usize; 3]; 3];

/// The routed experts of a plan that stream (on-demand parts of their unit): the op, the unit,
/// the address tables the op reads (filled per step), each expert's memory, and where the op
/// finds the chosen experts (`k` ids per token, `sel_stride` bytes apart).
struct StreamedExperts {
    mop: usize,
    unit: usize,
    tabs: [Id; 3],
    addrs: Vec<ExpertAddrs>,
    sel: usize,
    sel_stride: usize,
    k: usize,
}

/// A streaming unit of a plan: where its ops start, the streamer's unit, and for a unit that is
/// not resident the memory to wrap per step.
struct PlanUnit {
    start: usize,
    unit: usize,
    spans: Option<Vec<(usize, usize)>>,
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

    /// The few-rows matvec pipeline for a format and a row length: one output row per
    /// threadgroup, with enough SIMD-groups to cover the row's chunks in one or two passes (GGUF
    /// formats take chunks of 32 values, one dequantized block of their GEMM loader).
    pub(crate) fn mvt_pipe(&self, fmt: MvFmt, n_in: usize) -> Option<(Id, usize, usize)> {
        let chunk = if matches!(fmt, MvFmt::Gguf(_)) { 32 } else { fmt.chunk() };
        let ks = (n_in / chunk / 64).clamp(1, 4).next_power_of_two().min(4);
        Some((self.named_pipeline(&format!("mvt_{}_1_{ks}", fmt.name()))?, 1, ks))
    }
    /// The multi-token matvec pipeline for up to `tt` tokens (one of [`MVK_TOKENS`]).
    pub(crate) fn mvk_pipe(&self, fmt: MvFmt, nr: usize, ks: usize, tt: usize) -> Option<Id> {
        if !fmt.specialized() && ks != 4 {
            return None;
        }
        self.named_pipeline(&fmt.mvk_name(nr, ks, tt))
    }
}

/// Parameters of a multi-token matvec (`MVK` in the shader): the token count and the x, y and
/// gate strides in floats.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MvkParams {
    m: MvParams,
    ntok: u32,
    sx: u32,
    sy: u32,
    sgate: u32,
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
            let st = (f.status)(cb, f.s_status);
            if st != 4 && std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                let msg0: unsafe extern "C" fn(Id, Sel) -> Id = b.o.msg_send();
                let err = msg0(cb, b.o.sel(c"error"));
                let s = if err.is_null() {
                    String::new()
                } else {
                    b.o.nsstring_to_string(msg0(err, b.o.sel(c"localizedDescription")))
                };
                eprintln!("[metal decode: command buffer status {st}: {s}]");
            }
            ok &= st == 4;
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

impl MetalBackend {
    /// [`crate::gpu::Backend::matvec_rows`]: every job in one command buffer (a concurrent
    /// encoder: the jobs write disjoint outputs), on the few-rows matvec kernels.
    pub(crate) fn matvec_rows_impl(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [crate::gpu::RowsJob]) -> bool {
        if jobs.is_empty() || !(1..=4).contains(&t) || x.len() < t * n_in {
            return false;
        }
        let mut ws = Vec::with_capacity(jobs.len());
        let mut y_total = 0;
        for j in jobs.iter() {
            let Some(w) = MvWeight::new(&j.w, j.dev, j.n_out, n_in) else {
                return false;
            };
            let up = match &j.up {
                Some((u, d)) => match MvWeight::new(u, d, j.n_out, n_in) {
                    Some(u) if u.fmt == w.fmt => Some(u),
                    _ => return false,
                },
                None => None,
            };
            let Some(pipe) = self.mvt_pipe(w.fmt, n_in) else {
                return false;
            };
            if j.y.len() < t * j.n_out {
                return false;
            }
            for d in std::iter::once(j.dev).chain(j.up.as_ref().map(|u| u.1)) {
                if let Some(mw) = d.downcast_ref::<MetalWeight>() {
                    mw.keep_resident(&self.residency);
                }
            }
            ws.push((w, up, pipe, y_total));
            y_total += t * j.n_out;
        }
        let f = Fns::new(self);
        let mut st = self.staging.lock().unwrap();
        let st = &mut *st;
        if !self.ensure_staging(st, t * n_in * 4, y_total * 4) {
            return false;
        }
        let t0 = Instant::now();
        // SAFETY: the staging buffers were just sized for these copies.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), f.contents(st.x) as *mut f32, t * n_in) };
        let t1 = Instant::now();
        let (ok, _) = run_concurrent(self, &f, &[std::ops::Range { start: 0, end: 1 }], |e, _| {
            for (w, up, (pipe, nr, ks), y_off) in &ws {
                let w2 = up.as_ref().unwrap_or(w);
                let p = MvParams {
                    n_out: w.n_out as u32,
                    n_in: n_in as u32,
                    mode: if up.is_some() { MvMode::SiluPair } else { MvMode::Store } as u32,
                    rows_per_tg: *nr as u32,
                    w_off: w.data_off,
                    s_off: w.scales_off,
                    w2_off: w2.data_off,
                    s2_off: w2.scales_off,
                    o_off: w.offs_off,
                    o2_off: w2.offs_off,
                    gpr: w.gpr,
                    gk: w.gk,
                    ldy: t as u32,
                    _p: 0,
                };
                e.pipe(*pipe);
                e.buf(w.data, 0, 0);
                e.buf(w.scales, 0, 1);
                e.buf(st.x, 0, 2);
                e.buf(st.y, y_off * 4, 3);
                e.bytes(&p, 4);
                e.buf(w2.data, 0, 5);
                e.buf(w2.scales, 0, 6);
                e.buf(w.offs, 0, 7);
                e.buf(w2.offs, 0, 8);
                e.buf(st.x, 0, 9);
                e.dispatch((w.n_out.div_ceil(*nr), 1), 32 * ks);
            }
        });
        let t2 = Instant::now();
        if ok {
            let yp = f.contents(st.y) as *const f32;
            for (j, (_, _, _, y_off)) in jobs.iter_mut().zip(&ws) {
                // SAFETY: the job's rows were written at its offset of the staging output.
                unsafe { std::ptr::copy_nonoverlapping(yp.add(*y_off), j.y.as_mut_ptr(), t * j.n_out) };
            }
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
        self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        ok
    }
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
#[derive(Clone)]
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
        /// Multi-token plans: the x and out token strides in floats, and whether the kernel takes
        /// the tokens as rows (one row per token) rather than one dispatch per token.
        t: Option<(usize, usize, bool)>,
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
        /// Multi-token plans: the pipeline, the x, y, gate strides in floats, the rows and the
        /// threads per threadgroup.
        t: Option<(Id, u32, u32, u32, usize, usize)>,
        /// Multi-token plans without `t` (the MoE router): the x and y strides in bytes, the
        /// single-token kernel then running once per token.
        each: (usize, usize),
    },
    Topk {
        pipe: Id,
        logits: usize,
        sel: usize,
        n: u32,
        k: u32,
        scale: Option<usize>,
        /// Multi-token plans: the logits and selection strides in bytes (one dispatch per token).
        t: Option<(usize, usize)>,
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
        /// Multi-token plans: the x, selection, activation and y strides in bytes (each token's
        /// experts through the single-token kernels: tokens route to different experts).
        t: Option<[usize; 4]>,
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
        /// Multi-token plans: the pipeline, the qkv and query-output strides in floats, and the
        /// arena offset of this table's per-token cos / sin rows.
        t: Option<(Id, u32, u32, usize)>,
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
        /// Multi-token plans: the pipeline and the y, x, out strides.
        t: Option<(Id, [u32; 4])>,
    },
    Ew {
        pipe: Id,
        a: usize,
        b: usize,
        out: usize,
        n: u32,
        p: [f32; 4],
        /// Multi-token plans: the pipeline and the a, b, out strides.
        t: Option<(Id, [u32; 4])>,
    },
    Glu {
        pipe: Id,
        g: usize,
        u: usize,
        out: usize,
        n: u32,
        gelu: u32,
        /// Multi-token plans: the pipeline and the g, u, out strides.
        t: Option<(Id, [u32; 4])>,
    },
    /// See `Op::PleGate`.
    PleGate {
        pipe: Id,
        k: usize,
        q: usize,
        v: usize,
        out: usize,
        n: u32,
        hc: u32,
        /// Multi-token plans: the pipeline and the k, q, v, out strides.
        t: Option<(Id, [u32; 4])>,
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
        /// Multi-token plans: the pipeline and the x, out strides.
        t: Option<(Id, u32, u32)>,
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
        /// Multi-token plans: the pipeline and the c, projection (z / b / a), out strides.
        t: Option<(Id, u32, u32, u32)>,
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
        /// Multi-token plans: the query, output and gate strides in bytes (one dispatch per
        /// token, each attending to the positions up to its own).
        t: Option<(usize, usize, usize)>,
        /// A draft block's span (see `Op::Attend::block`).
        block: Option<(bool, usize)>,
    },
}

pub(crate) struct MetalPlan<'b> {
    be: &'b MetalBackend,
    f: Fns,
    arena: Id,
    consts: Id,
    /// Arena offset of the split-attention partials.
    part: usize,
    /// Buffers made for the plan (expert tables), released with it.
    owned: Vec<Id>,
    /// Expert weight spans shared with the model's other plans (`MetalBackend::wrap_shared`).
    shared: Vec<Id>,
    /// Buffers reached only through addresses, made resident on every step when the queue has
    /// no residency set.
    indirect: Vec<Id>,
    ops: Vec<MOp>,
    head_from: usize,
    input: usize,
    /// Floats of the input per token.
    n_input: usize,
    hidden: (usize, usize),
    logits: Option<(usize, usize)>,
    /// Tokens per step at most (1: a single-token plan), and the byte strides between the
    /// tokens' rows of the input, the hidden state, the logits, the second input and the
    /// capture.
    tokens: usize,
    strides: [usize; 5],
    /// Multi-token plans: the arena offset of the per-token RoPE rows, and the bytes of
    /// split-attention partials per token.
    rope_off: usize,
    part_stride: usize,
    /// Multi-token plans: the capture rows (offset, floats per token).
    capture: Option<(usize, usize)>,
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
    /// Identifies the slots this plan can use: plans of the same model (one-token, multi-token,
    /// with or without captured features) share a slot, so its device copies of the caches stay
    /// valid when steps alternate between them (one copy, one truncation watermark).
    id: u64,
    /// Weight streaming (see `Graph::units`): the streamer, the matvecs of streamed weights (by
    /// op index) and the units in op order. None / empty when every weight is resident.
    gate: Option<std::sync::Arc<dyn crate::gpu::decode::UnitGate>>,
    streamed: Vec<StreamedMv>,
    experts: Vec<StreamedExperts>,
    units: Vec<PlanUnit>,
}
unsafe impl Send for MetalPlan<'_> {}
unsafe impl Sync for MetalPlan<'_> {}

impl Drop for MetalPlan<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.f.void)(self.arena, self.f.s_release);
            (self.f.void)(self.consts, self.f.s_release);
            // Shared spans other plans still use stay resident and wrapped.
            let last: Vec<Id> = self.shared.iter().copied().filter(|&b| self.be.unwrap_shared(b)).collect();
            let gone: Vec<Id> = self
                .indirect
                .iter()
                .copied()
                .filter(|b| !self.shared.contains(b) || last.contains(b))
                .collect();
            if let Some(r) = &self.be.residency
                && !gone.is_empty()
            {
                r.update(&gone, false);
            }
            for &b in self.owned.iter().chain(&last) {
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

/// The slot layout of a graph: its KV caches' and recurrent states' shapes, and the weights they
/// belong to (the address of the graph's first weight), so that only plans of one model share
/// slots.
fn layout_id(g: &Graph) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    g.kv.hash(&mut h);
    g.states.hash(&mut h);
    g.ops
        .iter()
        .find_map(|op| match op {
            // A streamed weight has no device state of its own: its memory identifies it.
            Op::Linear { w, dev, n_out, n_in, .. } if crate::gpu::decode::is_streamed(dev) => {
                MvWeight::streamed(w, *n_out, *n_in).map(|(_, a)| a[0])
            }
            Op::Linear { dev, .. } => Some(*dev as *const DeviceWeight as usize),
            _ => None,
        })
        .hash(&mut h);
    h.finish() | 1
}

/// What an op reads and writes, for barrier placement: (resource, start, end) in floats, where a
/// resource is an activation buffer or (`KV_RES + i`) a KV cache.
type Range = (usize, usize, usize);
const KV_RES: usize = 1 << 20;

fn overlaps(a: &[Range], b: &[Range]) -> bool {
    a.iter().any(|x| b.iter().any(|y| x.0 == y.0 && x.1 < y.2 && y.1 < x.2))
}

impl MetalBackend {
    pub(crate) fn build_plan(&self, g: &Graph) -> Option<MetalPlan<'_>> {
        self.build_plan_t(g, 1, false)
    }

    /// Compiles `g` for steps of up to `tokens` tokens (1: the single-token plan). A multi-token
    /// plan keeps `tokens` rows of every activation buffer, runs the matvecs with the multi-token
    /// kernels (weights read once for all the tokens) and the other ops over every token, each
    /// computed exactly as a single-token step would (same kernels, same reduction orders).
    ///
    /// With `fast` ("fast verification"), the matvecs of a multi-token plan run on batched
    /// matrix kernels (`fv_*`, simdgroup MMAs over all the tokens) where the format has one: the
    /// weights are read once at little more than one token's cost, but the sums are taken in
    /// another order, so the results differ from single-token steps in rounding.
    pub(crate) fn build_plan_t(&self, g: &Graph, tokens: usize, fast: bool) -> Option<MetalPlan<'_>> {
        let f = Fns::new(self);
        let mt = tokens > 1;
        let fast = fast && mt && fast_kernels();
        let tt = match MVK_TOKENS.iter().find(|&&t| t >= tokens) {
            Some(&t) => t,
            None if !mt => 1,
            None => return None,
        };
        // The fast kernels read whole 8-token tiles of x: keep that many rows.
        let tokens = if fast { tokens.next_multiple_of(8) } else { tokens };
        // Activation arena: every buffer 256-byte aligned; in a multi-token plan, `tokens` rows of
        // `tsz` floats (256-byte aligned too).
        let tsz: Vec<usize> = g.bufs.iter().map(|&n| if mt { n.next_multiple_of(64) } else { n }).collect();
        let mut boff = Vec::with_capacity(g.bufs.len());
        let mut total = 0usize;
        for &n in &tsz {
            boff.push(total);
            total += (n * tokens * 4).next_multiple_of(256);
        }
        let byte = |v: View| boff[v.buf] + v.off * 4;
        // The token stride of a view's buffer, in floats.
        let ts = |v: View| tsz[v.buf];
        // Multi-token plans: the per-token RoPE rows, table after table.
        let rope_off = total;
        total += (tokens * 2 * g.rope_halves.iter().sum::<usize>() * 4).next_multiple_of(256);
        let rope_table = |r: usize| rope_off + tokens * 2 * g.rope_halves[..r].iter().sum::<usize>() * 4;
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
        let mut shared: Vec<Id> = Vec::new();
        let mut indirect: Vec<Id> = Vec::new();
        let mut head_from = usize::MAX;
        let (mut rd, mut wr): (Vec<Range>, Vec<Range>) = (Vec::new(), Vec::new());
        let rng = |v: View, n: usize| (v.buf, v.off, v.off + n + (tokens - 1) * tsz[v.buf]);
        // Streaming: the first compiled op of each graph op, and the matvecs of streamed weights.
        let mut op_mop: Vec<usize> = Vec::with_capacity(g.ops.len());
        let mut streamed: Vec<StreamedMv> = Vec::new();
        let mut experts: Vec<StreamedExperts> = Vec::new();
        for (i, op) in g.ops.iter().enumerate() {
            op_mop.push(ops.len());
            let mut smv: Option<([usize; 3], Option<[usize; 3]>)> = None;
            let mut sxp: Option<StreamedExperts> = None;
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
                    // One row per token when the op has one row, else one dispatch per token.
                    let native = mt && rows == 1;
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
                            x_stride: if native { ts(x) } else { x_stride } as u32,
                            o_stride: if native { ts(out) } else { out_stride } as u32,
                            pre,
                            _p: 0,
                        },
                        rows,
                        threads,
                        t: mt.then(|| (ts(x), ts(out), native)),
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
                    use crate::gpu::decode::is_streamed;
                    let (mw, sa) = if is_streamed(dev) {
                        let (m, a) = MvWeight::streamed(w, n_out, n_in)?;
                        (m, Some(a))
                    } else {
                        (MvWeight::new(w, dev, n_out, n_in)?, None)
                    };
                    let (mw2, sa2) = match w2 {
                        Some((w2, d2)) => {
                            let (m, a) = if is_streamed(d2) {
                                let (m, a) = MvWeight::streamed(w2, n_out, n_in)?;
                                (m, Some(a))
                            } else {
                                (MvWeight::new(w2, d2, n_out, n_in)?, None)
                            };
                            if m.fmt != mw.fmt || m.gk != mw.gk || m.gpr != mw.gpr {
                                return None; // a pair shares one kernel
                            }
                            (Some(m), a)
                        }
                        None => (None, None),
                    };
                    match (sa, sa2, mw2.is_some()) {
                        (None, None, _) => {}
                        (Some(a), a2, pair) if a2.is_some() == pair => smv = Some((a, a2)),
                        _ => return None, // a pair streams whole or not at all
                    }
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
                    let fv = if fast && n_in.is_multiple_of(32) {
                        self.fv_pipe(&mw, tt)
                    } else {
                        None
                    };
                    let t = match fv {
                        Some((p, rows)) => Some((p, ts(x) as u32, ts(y) as u32, gate.map_or(0, ts) as u32, rows, 128)),
                        None if mt => Some((
                            self.mvk_pipe(mw.fmt, mvk_rows(tt), ks, tt)?,
                            ts(x) as u32,
                            ts(y) as u32,
                            gate.map_or(0, ts) as u32,
                            mvk_rows(tt),
                            32 * ks,
                        )),
                        None => None,
                    };
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
                        t,
                        each: (0, 0),
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
                        t: if mt {
                            Some((
                                self.named_pipeline("dec_qk_rope_t")?,
                                ts(qkv) as u32,
                                ts(q_out) as u32,
                                rope_table(rope),
                            ))
                        } else {
                            None
                        },
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
                    // Multi-token plans route each token with the single-token kernels.
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
                        t: None,
                        each: (ts(x) * 4, ts(logits) * 4),
                    });
                    emit.push(MOp::Barrier);
                    emit.push(MOp::Topk {
                        pipe: self.named_pipeline("dec_topk")?,
                        logits: byte(logits),
                        sel: byte(sel),
                        n: n_exp as u32,
                        k: k as u32,
                        scale: expert_scale.map(&mut cst),
                        t: mt.then(|| (ts(logits) * 4, ts(sel) * 4)),
                    });
                    (vec![rng(x, h), rng(logits, rows)], vec![rng(logits, rows), rng(sel, 2 * k)])
                }
                Op::Experts(ref ex) => {
                    if mt && ex.edits.is_some() {
                        return None;
                    }
                    // Experts that stream: on-demand parts of the op's unit, one per expert.
                    let unit = g.units.iter().take_while(|u| u.0 <= i).last().map(|u| u.1);
                    let streamed_unit = match (&g.gate, unit) {
                        (Some(gt), Some(u)) if gt.part_count(u) > 0 => {
                            if gt.part_count(u) != ex.gate.len() {
                                return None;
                            }
                            Some(u)
                        }
                        _ => None,
                    };
                    let mut addrs = Vec::new();
                    let (tabs, fused, g, d, pf) = self.expert_tables(
                        ex,
                        &mut owned,
                        &mut shared,
                        &mut indirect,
                        streamed_unit.is_some().then_some(&mut addrs),
                    )?;
                    if let Some(u) = streamed_unit {
                        sxp = Some(StreamedExperts {
                            mop: 0,
                            unit: u,
                            tabs,
                            addrs,
                            sel: byte(ex.sel),
                            sel_stride: ts(ex.sel) * 4,
                            k: ex.k,
                        });
                    }
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
                        t: mt.then(|| [ts(ex.x) * 4, ts(ex.sel) * 4, ts(ex.act) * 4, ts(ex.y) * 4]),
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
                        t: if mt {
                            let o = next.map_or(x, |(o, _)| o);
                            Some((
                                self.named_pipeline("dec_add_norm_t")?,
                                [ts(y) as u32, ts(x) as u32, ts(o) as u32, 0],
                            ))
                        } else {
                            None
                        },
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
                        Ew::SiluScale(s) => [3.0, s, 0.0, 0.0],
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
                        t: if mt {
                            Some((
                                self.named_pipeline("dec_ew_t")?,
                                [ts(a) as u32, ts(b.unwrap_or(a)) as u32, ts(out) as u32, 0],
                            ))
                        } else {
                            None
                        },
                    });
                    let mut r = vec![rng(a, n)];
                    r.extend(b.map(|b| rng(b, n)));
                    (r, vec![rng(out, n)])
                }
                Op::HcPre { gate, xn, out, n, hc } => {
                    emit.push(MOp::Ew {
                        pipe: self.named_pipeline("dec_ew")?,
                        a: byte(gate),
                        b: byte(xn),
                        out: byte(out),
                        n: n as u32,
                        p: [4.0, hc as f32, 0.0, 1.0 / hc as f32],
                        t: if mt {
                            Some((
                                self.named_pipeline("dec_ew_t")?,
                                [ts(gate) as u32, ts(xn) as u32, ts(out) as u32, 0],
                            ))
                        } else {
                            None
                        },
                    });
                    (vec![rng(gate, hc * n), rng(xn, hc * n)], vec![rng(out, n)])
                }
                Op::HcPost { x, y, inj, n, hc } => {
                    emit.push(MOp::Ew {
                        pipe: self.named_pipeline("dec_ew")?,
                        a: byte(y),
                        b: byte(inj),
                        out: byte(x),
                        n: (hc * n) as u32,
                        p: [5.0, n as f32, 1.0 / hc as f32, 0.0],
                        t: if mt {
                            Some((self.named_pipeline("dec_ew_t")?, [ts(y) as u32, ts(inj) as u32, ts(x) as u32, 0]))
                        } else {
                            None
                        },
                    });
                    (vec![rng(y, n), rng(inj, hc), rng(x, hc * n)], vec![rng(x, hc * n)])
                }
                Op::PleGate { k, q, v, out, n, hc } => {
                    emit.push(MOp::PleGate {
                        pipe: self.named_pipeline("dec_ple_gate")?,
                        k: byte(k),
                        q: byte(q),
                        v: byte(v),
                        out: byte(out),
                        n: n as u32,
                        hc: hc as u32,
                        t: if mt {
                            Some((
                                self.named_pipeline("dec_ple_gate_t")?,
                                [ts(k) as u32, ts(q) as u32, ts(v) as u32, ts(out) as u32],
                            ))
                        } else {
                            None
                        },
                    });
                    (vec![rng(k, hc * n), rng(q, hc * n), rng(v, n)], vec![rng(out, hc * n)])
                }
                Op::Glu { g: gv, u, out, n, gelu } => {
                    emit.push(MOp::Glu {
                        pipe: self.named_pipeline("dec_glu")?,
                        g: byte(gv),
                        u: byte(u),
                        out: byte(out),
                        n: n as u32,
                        gelu: gelu as u32,
                        t: if mt {
                            Some((self.named_pipeline("dec_glu_t")?, [ts(gv) as u32, ts(u) as u32, ts(out) as u32, 0]))
                        } else {
                            None
                        },
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
                    if mt || n_items > crate::intervene::GPU_MAX_ITEMS {
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
                    if mt {
                        return None;
                    }
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
                        t: if mt {
                            Some((self.named_pipeline("dec_gdn_conv_t")?, ts(x) as u32, ts(out) as u32))
                        } else {
                            None
                        },
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
                    if mt && (r.z.buf != r.b.buf || r.a.buf != r.b.buf) {
                        return None; // one stride for z, b and a
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
                            tiled: r.tiled as u32 | (r.sigmoid_gate as u32) << 1,
                            eps: r.eps,
                            qscale: (r.hdk as f32).powf(-0.5),
                            hdk: r.hdk as u32,
                            hdv: r.hdv as u32,
                        },
                        t: if mt {
                            Some((
                                self.named_pipeline("dec_gdn_rule_t")?,
                                ts(r.c) as u32,
                                ts(r.b) as u32,
                                ts(r.out) as u32,
                            ))
                        } else {
                            None
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
                    block,
                } => {
                    if block.is_some() && !mt {
                        return None;
                    }
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
                        t: mt.then(|| (ts(q) * 4, ts(out) * 4, gate.map_or(0, |(v, _)| ts(v) * 4))),
                        block,
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
            if let Some((w, w2)) = smv {
                streamed.push(StreamedMv { mop: ops.len() - 1, w, w2 });
            }
            if let Some(mut x) = sxp {
                x.mop = ops.len() - 1;
                experts.push(x);
            }
            rd.extend(r);
            wr.extend(w);
        }
        if head_from == usize::MAX {
            head_from = ops.len();
        }
        // Streaming units: every streamed matvec must lie in a unit that is not resident.
        let mut units: Vec<PlanUnit> = Vec::new();
        if !streamed.is_empty() || !experts.is_empty() {
            let gate = g.gate.as_ref()?;
            for &(op, u) in &g.units {
                let start = op_mop.get(op).copied().unwrap_or(ops.len());
                if units.last().is_some_and(|l| l.start > start) {
                    return None;
                }
                units.push(PlanUnit {
                    start,
                    unit: u,
                    spans: (!gate.resident(u)).then(|| gate.spans(u)),
                });
            }
            for sm in &streamed {
                let k = units.partition_point(|u| u.start <= sm.mop);
                let spans = k.checked_sub(1).and_then(|k| units[k].spans.as_ref())?;
                for a in std::iter::once(&sm.w).chain(&sm.w2).flat_map(|a| a.iter()).filter(|&&a| a != 0) {
                    if !spans.iter().any(|&(s, l)| *a >= s && *a < s + l) {
                        return None;
                    }
                }
            }
        }
        // Split-attention partials after the activations (a region per token).
        let part = total;
        let part_stride = (part_floats * 4).next_multiple_of(256);
        total += part_stride * tokens;
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
            shared,
            indirect,
            ops,
            head_from,
            input: byte(g.input),
            n_input: if g.n_input > 0 { g.n_input } else { g.n_hidden },
            hidden: (byte(g.hidden), g.n_hidden),
            logits: g.logits.map(|(v, n)| (byte(v), n)),
            tokens,
            strides: [
                ts(g.input) * 4,
                ts(g.hidden) * 4,
                g.logits.map_or(0, |(v, _)| ts(v) * 4),
                g.input2.map_or(0, |(v, _)| ts(v) * 4),
                g.capture.map_or(0, |(v, _)| ts(v) * 4),
            ],
            rope_off,
            part_stride,
            capture: g.capture.map(|(v, n)| (byte(v), n)),
            kv: g.kv.clone(),
            rope_halves: g.rope_halves.clone(),
            input2: g.input2.map(|(v, n)| (byte(v), n)),
            states: g.states.clone(),
            params: g.params.map(|((h, hn), (c, cn))| ((byte(h), hn), (byte(c), cn))),
            params_key: AtomicU64::new(0),
            lock: Mutex::new(()),
            id: layout_id(g),
            gate: if streamed.is_empty() && experts.is_empty() {
                None
            } else {
                // The GPU maps the streamed pages itself as it wires them.
                if !streamed.is_empty()
                    && let Some(gt) = &g.gate
                {
                    gt.device_reads(true);
                }
                g.gate.clone()
            },
            streamed,
            experts,
            units,
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
        shared: &mut Vec<Id>,
        indirect: &mut Vec<Id>,
        stream: Option<&mut Vec<ExpertAddrs>>,
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
        if let Some(addrs) = stream {
            // Streamed experts: empty tables, filled per step with the chosen experts' wraps.
            let new_len: MsgNewBufLen = self.o.msg_send();
            let mut tabs = [std::ptr::null_mut(); 3];
            for (t, v) in tabs.iter_mut().zip(&infos) {
                if v.is_empty() {
                    continue;
                }
                *t = unsafe { new_len(self.device, self.sels.new_buffer_len, v.len() * 32, 0) };
                if t.is_null() {
                    return None;
                }
                owned.push(*t);
            }
            if fused {
                tabs[1] = tabs[0];
            }
            addrs.clear();
            for e in 0..n {
                let mut a = [[0usize; 3]; 3];
                for (j, v) in infos.iter().enumerate() {
                    if let Some(i) = v.get(e) {
                        for (q, r) in i.2.iter().enumerate() {
                            a[j][q] = r.map_or(0, |r| r.0);
                        }
                    }
                }
                addrs.push(a);
            }
            return Some((tabs, fused, infos[0][0].1, infos[2][0].1, (infos[0][0].0, infos[2][0].0)));
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
            let (id, a) = self.wrap_shared(s as *const u8, e - s)?;
            addr.push(a);
            shared.push(id);
            indirect.push(id);
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
    /// room for `t` more positions.
    fn sync_kv(&self, slot: &mut MetalSlot, kv: &mut [&mut crate::model::LayerCache], pos: usize, t: usize) -> bool {
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
            if m.cap < pos + t {
                let cap = (pos + t).next_power_of_two().max(256);
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

impl MetalPlan<'_> {
    /// Encodes one op of a multi-token plan for `t` tokens at positions `pos..`.
    #[allow(clippy::too_many_arguments)]
    fn encode_multi(&self, e: &FastEnc, op: &MOp, arena: Id, consts: Id, kvb: &[(Id, Id)], stb: &[(Id, usize)], pos: usize, t: usize) {
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
                t: tk,
            } => {
                let (sx, so, native) = tk.expect("multi-token norm");
                e.pipe(*pipe);
                e.buf(consts, w.unwrap_or(0), 2);
                e.bytes(p, 3);
                if native {
                    e.buf(arena, *x, 0);
                    e.buf(arena, *out, 1);
                    e.dispatch((t, 1), *threads);
                } else {
                    for j in 0..t {
                        e.buf(arena, *x + j * sx * 4, 0);
                        e.buf(arena, *out + j * so * 4, 1);
                        e.dispatch((*rows, 1), *threads);
                    }
                }
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
                t: None,
                each,
            } => {
                for j in 0..t {
                    let (xo, yo) = (*x + j * each.0, *y + j * each.1);
                    encode_mv(e, *pipe, *nr, *ks, w, w2.as_ref(), (arena, xo), (arena, yo), *mode, (arena, *gate));
                }
            }
            MOp::Topk {
                pipe,
                logits,
                sel,
                n,
                k,
                scale,
                t: tk,
            } => {
                let (sl, ss) = tk.expect("multi-token top-k");
                e.pipe(*pipe);
                e.bytes(&[*n, *k, scale.is_some() as u32, 0], 2);
                e.buf(consts, scale.unwrap_or(0), 3);
                for j in 0..t {
                    e.buf(arena, *logits + j * sl, 0);
                    e.buf(arena, *sel + j * ss, 1);
                    e.dispatch((1, 1), (*n as usize).next_multiple_of(32));
                }
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
                edit: None,
                t: tk,
            } => {
                // Every token's gate / up pairs, then every token's down projections.
                let [sx, ssel, sact, sy] = tk.expect("multi-token experts");
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
                let am = if *gelu { MvMode::GeluPair } else { MvMode::SiluPair };
                e.bytes(&p(*inter, *h, am, *g, if *fused { *inter } else { 0 }), 4);
                for j in 0..t {
                    e.buf(arena, *x + j * sx, 2);
                    e.buf(arena, *act + j * sact, 3);
                    e.buf(arena, *sel + j * ssel, 5);
                    e.dispatch((*inter, *k), 32);
                }
                e.barrier();
                e.pipe(*sum);
                e.buf(tabs[2], 0, 0);
                e.buf(tabs[2], 0, 1);
                e.bytes(&p(*h, *inter, if *add { MvMode::Add } else { MvMode::Store }, *d, 0), 4);
                for j in 0..t {
                    e.buf(arena, *act + j * sact, 2);
                    e.buf(arena, *y + j * sy, 3);
                    e.buf(arena, *sel + j * ssel, 5);
                    e.dispatch((*h, 1), 32);
                }
            }
            MOp::Mv {
                w,
                w2,
                x,
                y,
                mode,
                gate,
                t: Some(tk),
                ..
            } => {
                let (pipe, sx, sy, sgate, nr, threads) = *tk;
                let w2 = w2.as_ref().unwrap_or(w);
                let p = MvkParams {
                    m: MvParams {
                        n_out: w.n_out as u32,
                        n_in: w.n_in as u32,
                        mode: *mode as u32,
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
                    },
                    ntok: t as u32,
                    sx,
                    sy,
                    sgate,
                };
                e.pipe(pipe);
                e.buf(w.data, 0, 0);
                e.buf(w.scales, 0, 1);
                e.buf(arena, *x, 2);
                e.buf(arena, *y, 3);
                e.bytes(&p, 4);
                e.buf(w2.data, 0, 5);
                e.buf(w2.scales, 0, 6);
                e.buf(w.offs, 0, 7);
                e.buf(w2.offs, 0, 8);
                e.buf(arena, *gate, 9);
                e.dispatch((w.n_out.div_ceil(nr), 1), threads);
            }
            MOp::QkRope {
                qkv,
                qo,
                kv,
                qn,
                kn,
                p,
                t: tk,
                ..
            } => {
                let (pipe, sq, so, cs) = tk.expect("multi-token rope");
                let mut p = *p;
                p.pos = pos as u32;
                e.pipe(pipe);
                e.buf(arena, *qkv, 0);
                e.buf(arena, *qo, 1);
                e.buf(kvb[*kv].0, 0, 2);
                e.buf(kvb[*kv].1, 0, 3);
                e.buf(consts, qn.unwrap_or(0), 4);
                e.buf(consts, kn.unwrap_or(0), 5);
                e.bytes(&p, 6);
                e.buf(arena, cs, 7);
                e.bytes(&[sq, so, 2 * p.half_rot, 0u32], 8);
                e.dispatch(((p.nh + 2 * p.nkv) as usize, t), p.hd as usize);
            }
            MOp::AddNorm {
                y,
                x,
                out,
                wy,
                wn,
                p,
                threads,
                t: tk,
                ..
            } => {
                let (pipe, ts) = tk.expect("multi-token add-norm");
                e.pipe(pipe);
                e.buf(arena, *y, 0);
                e.buf(arena, *x, 1);
                e.buf(arena, *out, 2);
                e.buf(consts, *wy, 3);
                e.buf(consts, *wn, 4);
                e.bytes(p, 5);
                e.bytes(&ts, 6);
                e.dispatch((1, t), *threads);
            }
            MOp::Ew {
                a, b, out, n, p, t: tk, ..
            } => {
                let (pipe, ts) = tk.expect("multi-token elementwise");
                e.pipe(pipe);
                e.buf(arena, *a, 0);
                e.buf(arena, *b, 1);
                e.buf(arena, *out, 2);
                e.bytes(p, 3);
                e.bytes(n, 4);
                e.bytes(&ts, 5);
                e.dispatch(((*n as usize).div_ceil(256), t), 256);
            }
            MOp::Glu {
                g, u, out, n, gelu, t: tk, ..
            } => {
                let (pipe, ts) = tk.expect("multi-token glu");
                e.pipe(pipe);
                e.buf(arena, *g, 0);
                e.buf(arena, *u, 1);
                e.buf(arena, *out, 2);
                e.bytes(&[*n, *gelu], 3);
                e.bytes(&ts, 4);
                e.dispatch(((*n as usize).div_ceil(256), t), 256);
            }
            MOp::GdnConv {
                x,
                out,
                state,
                w,
                dim,
                kc,
                t: tk,
                ..
            } => {
                let (pipe, sx, so) = tk.expect("multi-token conv");
                e.pipe(pipe);
                e.buf(arena, *x, 0);
                e.buf(arena, *out, 1);
                e.buf(stb[*state].0, stb[*state].1, 2);
                e.buf(consts, *w, 3);
                e.bytes(&[*dim, *kc], 4);
                e.bytes(&[t as u32, sx, so, 0], 5);
                e.dispatch(((*dim as usize).div_ceil(256), 1), 256);
            }
            MOp::GdnRule {
                c,
                z,
                b,
                a,
                out,
                state,
                cst,
                p,
                t: tk,
                ..
            } => {
                let (pipe, sc, sp, so) = tk.expect("multi-token delta rule");
                e.pipe(pipe);
                e.buf(arena, *c, 0);
                e.buf(arena, *z, 1);
                e.buf(arena, *b, 2);
                e.buf(arena, *a, 3);
                e.buf(arena, *out, 4);
                e.buf(stb[*state].0, stb[*state].1, 5);
                e.buf(consts, *cst, 6);
                e.bytes(p, 7);
                e.bytes(&[t as u32, sc, sp, so], 8);
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
                t: tk,
                block,
            } => {
                let (sq, so, sg) = tk.expect("multi-token attention");
                for j in 0..t {
                    // Token j attends to the positions up to its own, split as a single-token
                    // step at that position would be; a draft block's tokens to the context
                    // before the block and to the block.
                    let (len, lo) = match block {
                        None => {
                            let len = pos + j + 1;
                            (len, window.map_or(0, |w| len.saturating_sub(w)))
                        }
                        Some((causal, w)) => (
                            if *causal { pos + j + 1 } else { pos + t },
                            if *w > 0 { pos.saturating_sub(*w) } else { 0 },
                        ),
                    };
                    let mut p = *p;
                    p.len = len as u32;
                    p.lo = lo as u32;
                    let n = (p.len - p.lo) as usize;
                    let chunk = n.div_ceil(ATTN_MAX_SPLIT).next_multiple_of(256).max(256);
                    p.nsplit = n.div_ceil(chunk) as u32;
                    p.chunk = chunk as u32;
                    let part = self.part + j * self.part_stride;
                    let gt = gate.map_or(0, |g| g + j * sg);
                    e.pipe(*pipe);
                    e.buf(arena, *q + j * sq, 0);
                    e.buf(kvb[*kv].0, 0, 1);
                    e.buf(kvb[*kv].1, 0, 2);
                    e.buf(arena, *out + j * so, 3);
                    e.buf(arena, gt, 4);
                    e.bytes(&p, 5);
                    e.buf(arena, part, 6);
                    e.dispatch((p.nh as usize / grp, p.nsplit as usize), *threads);
                    if p.nsplit > 1 {
                        e.barrier();
                        e.pipe(*combine);
                        e.buf(arena, part, 0);
                        e.buf(arena, *out + j * so, 1);
                        e.buf(arena, gt, 2);
                        e.bytes(&p, 3);
                        e.bytes(hd, 4);
                        e.dispatch((p.nh as usize, 1), 256);
                    }
                }
            }
            MOp::PleGate {
                k,
                q,
                v,
                out,
                n,
                hc,
                t: tk,
                ..
            } => {
                let (pipe, ts) = tk.expect("multi-token n-gram gate");
                e.pipe(pipe);
                e.buf(arena, *k, 0);
                e.buf(arena, *q, 1);
                e.buf(arena, *v, 2);
                e.buf(arena, *out, 3);
                e.bytes(&[*n, *hc], 4);
                e.bytes(&ts, 5);
                e.dispatch((*hc as usize, t), 1024);
            }
            MOp::Experts { .. } | MOp::Steer { .. } | MOp::NeuronEdit { .. } => {
                unreachable!("single-token op in a multi-token plan")
            }
        }
    }
}

/// Encodes ops of a step (see `MetalPlan::step`): the range, and replacements for some of its ops
/// (streamed weights bound to this step's wraps).
type EncodeOps<'e> = dyn Fn(&FastEnc, std::ops::Range<usize>, &[(usize, MOp)]) + 'e;

/// Wraps of a streamer's cached parts (routed experts) kept between steps, shared by the plans of
/// the model (`UnitGate::device_state`): unit -> part -> `(span start, span end, buffer, GPU
/// address of the span start)` per span (in the queue's residency set when there is one). A part's wraps go once the streamer no longer
/// caches it (a live wrap the device used keeps the pages wired).
/// A wrapped span of a part: (start, end, buffer, GPU address of the start).
type PartSpan = (usize, usize, Id, u64);

struct PartWraps {
    map: std::collections::HashMap<usize, std::collections::HashMap<usize, Vec<PartSpan>>>,
    release: (MsgVoid, Sel),
    residency: Option<std::sync::Arc<crate::metal::Residency>>,
}
unsafe impl Send for PartWraps {}
unsafe impl Sync for PartWraps {}

impl Drop for PartWraps {
    fn drop(&mut self) {
        let ids: Vec<Id> = self.map.values().flat_map(|l| l.values()).flatten().map(|b| b.2).collect();
        if let Some(r) = &self.residency
            && !ids.is_empty()
        {
            r.update(&ids, false);
        }
        for b in ids {
            unsafe { (self.release.0)(b, self.release.1) };
        }
    }
}

/// A command buffer of a streamed step in flight: the units to leave once it completed (with
/// the wraps of their memory to release), and whether it holds a streamed unit.
struct Flight {
    cb: Id,
    ends: Vec<(usize, Vec<Id>)>,
    streamed: bool,
}

/// Where a streamed step changes command buffers: a unit starts, streamed experts run.
enum Ev {
    Unit(usize),
    Experts(usize),
    End,
}

/// The command buffers of a streamed step (see [`MetalPlan::run_streamed`]).
struct StreamRun<'p, 'b> {
    plan: &'p MetalPlan<'b>,
    gate: &'p dyn crate::gpu::decode::UnitGate,
    flights: std::collections::VecDeque<Flight>,
    /// The open command buffer and encoder, and whether it holds a streamed unit.
    cur: Option<(Id, Id)>,
    streamed: bool,
    /// Units entered and not yet handed to a flight: (unit, end of its ops, its wraps).
    active: Vec<(usize, usize, Vec<Id>)>,
    /// Ops encoded so far.
    pos: usize,
    ok: bool,
    /// OXINFER_METAL_TRACE: GPU seconds of the command buffers, and how many.
    gpu: (f64, usize),
}

impl StreamRun<'_, '_> {
    /// Streamed units in flight at most: each keeps its pages wired until it is left.
    const DEPTH: usize = 2;

    fn enc(&mut self) -> Id {
        if let Some((_, e)) = self.cur {
            return e;
        }
        let f = &self.plan.f;
        let cb = unsafe { (f.msg0)(self.plan.be.queue, f.s_cb) };
        let enc = unsafe { (f.enc_type)(cb, f.s_enc_type, 1) }; // MTLDispatchTypeConcurrent
        self.cur = Some((cb, enc));
        enc
    }

    fn finish(&mut self, fl: Flight) {
        let f = &self.plan.f;
        unsafe {
            // Spin first: the routing of streamed experts waits for one layer's work at a time,
            // and a sleeping wait wakes up late.
            let t = Instant::now();
            while (f.status)(fl.cb, f.s_status) < 4 && t.elapsed() < std::time::Duration::from_millis(20) {
                std::hint::spin_loop();
            }
            (f.void)(fl.cb, f.s_wait);
            self.ok &= (f.status)(fl.cb, f.s_status) == 4;
            if trace() {
                let b = self.plan.be;
                let time: unsafe extern "C" fn(Id, Sel) -> f64 = b.o.msg_send();
                self.gpu.0 += time(fl.cb, b.o.sel(c"GPUEndTime")) - time(fl.cb, b.o.sel(c"GPUStartTime"));
                self.gpu.1 += 1;
            }
        }
        for (u, bufs) in fl.ends {
            for b in bufs {
                unsafe { (f.void)(b, f.s_release) };
            }
            self.gate.end(u);
        }
    }

    /// Commits the open command buffer; leaves what the device finished, and waits while too
    /// many streamed units are in flight.
    fn cut(&mut self) {
        let f = &self.plan.f;
        if let Some((cb, enc)) = self.cur.take() {
            unsafe {
                (f.void)(enc, f.s_end);
                (f.void)(cb, f.s_commit);
            }
            let pos = self.pos;
            let mut ends = Vec::new();
            for a in std::mem::take(&mut self.active) {
                if a.1 <= pos {
                    ends.push((a.0, a.2));
                } else {
                    self.active.push(a);
                }
            }
            self.flights.push_back(Flight {
                cb,
                ends,
                streamed: self.streamed,
            });
            self.streamed = false;
        }
        while let Some(front) = self.flights.front() {
            let busy = self.flights.iter().filter(|fl| fl.streamed).count();
            if busy <= Self::DEPTH && unsafe { (f.status)(front.cb, f.s_status) } < 4 {
                break;
            }
            let fl = self.flights.pop_front().unwrap();
            self.finish(fl);
        }
    }

    fn drain(&mut self) {
        while let Some(fl) = self.flights.pop_front() {
            self.finish(fl);
        }
    }
}

impl MetalPlan<'_> {
    /// Runs ops `..end` of a `t`-token step whose weights partly stream (see `Graph::units`).
    /// Each unit is entered before its ops are encoded and left once the device finished them. A
    /// unit that is not resident gets command buffers of its own and its memory is wrapped for
    /// this step only: the wraps are released when the device is done, so the evictor can drop
    /// the pages, while the device already runs the next units. Streamed experts end the command
    /// buffer before them: once the routing is read back, the chosen experts are read (or found
    /// in the part cache), wrapped (wraps kept while the part cache keeps them) and written to
    /// the op's address tables.
    fn run_streamed(&self, end: usize, t: usize, encode: &EncodeOps) -> bool {
        let Some(gate) = self.gate.as_deref() else { return false };
        let mut events: Vec<(usize, u8, Ev)> = Vec::new();
        for (k, u) in self.units.iter().enumerate() {
            if u.start < end {
                events.push((u.start, 0, Ev::Unit(k)));
            }
        }
        for (j, x) in self.experts.iter().enumerate() {
            if x.mop < end {
                events.push((x.mop, 1, Ev::Experts(j)));
            }
        }
        events.sort_by_key(|e| (e.0, e.1));
        events.push((end, 2, Ev::End));
        let unit_end = |k: usize| self.units.get(k + 1).map_or(end, |n| n.start.min(end));
        let base = self.f.contents(self.arena);
        let pool = unsafe { (self.be.sels.pool_push)() };
        let mut r = StreamRun {
            plan: self,
            gate,
            flights: Default::default(),
            cur: None,
            streamed: false,
            active: Vec::new(),
            pos: 0,
            ok: true,
            gpu: (0.0, 0),
        };
        let mut bound: Vec<(usize, MOp)> = Vec::new();
        // OXINFER_METAL_TRACE: seconds waiting for routing, reading experts, wrapping them; units.
        let mut tm = [0f64; 4];
        let t_run = Instant::now();
        for (p, _, ev) in events {
            if !r.ok {
                break;
            }
            if p > r.pos {
                let enc = r.enc();
                encode(&FastEnc { enc, f: &self.f }, r.pos..p, &bound);
                r.pos = p;
            }
            match ev {
                Ev::End => {}
                Ev::Unit(k) => {
                    let u = &self.units[k];
                    if u.spans.is_some() || r.streamed {
                        r.cut();
                        bound.clear();
                    }
                    let tb = Instant::now();
                    gate.begin(u.unit);
                    tm[3] += tb.elapsed().as_secs_f64();
                    r.active.push((u.unit, unit_end(k), Vec::new()));
                    let Some(spans) = &u.spans else { continue };
                    r.streamed = true;
                    let mut bufs: Vec<(usize, usize, Id, usize)> = Vec::new();
                    for &(a, l) in spans {
                        match self.be.wrap_raw(a as *const u8, l) {
                            Some((id, off)) => bufs.push((a, a + l, id, off)),
                            None => r.ok = false,
                        }
                    }
                    r.active.last_mut().unwrap().2.extend(bufs.iter().map(|b| b.2));
                    let ue = unit_end(k);
                    let lo = self.streamed.partition_point(|m| m.mop < u.start);
                    for sm in self.streamed[lo..].iter().take_while(|m| m.mop < ue) {
                        let mut op = self.ops[sm.mop].clone();
                        if let MOp::Mv { w, w2, .. } = &mut op {
                            r.ok &= w.bind(&sm.w, &bufs);
                            if let (Some(w2), Some(a2)) = (w2.as_mut(), sm.w2.as_ref()) {
                                r.ok &= w2.bind(a2, &bufs);
                            }
                        }
                        bound.push((sm.mop, op));
                    }
                }
                Ev::Experts(j) => {
                    let x = &self.experts[j];
                    let t0 = Instant::now();
                    r.cut();
                    r.drain();
                    tm[0] += t0.elapsed().as_secs_f64();
                    if !r.ok {
                        break;
                    }
                    let mut ids: Vec<usize> = Vec::new();
                    for row in 0..t {
                        let sel = unsafe { base.add(x.sel + row * x.sel_stride) } as *const u32;
                        for q in 0..x.k {
                            let e = unsafe { *sel.add(q) } as usize;
                            if e < x.addrs.len() && !ids.contains(&e) {
                                ids.push(e);
                            }
                        }
                    }
                    let t1 = Instant::now();
                    gate.need_parts(x.unit, &ids);
                    let t2 = Instant::now();
                    let res = self.bind_experts(gate, x, &ids);
                    tm[1] += (t2 - t1).as_secs_f64();
                    tm[2] += t2.elapsed().as_secs_f64();
                    let enc = r.enc();
                    match res {
                        Some(res) => FastEnc { enc, f: &self.f }.use_resources(&res),
                        None => r.ok = false,
                    }
                    // The rest of a streamed unit (its wraps stay with it until it is left).
                    r.streamed = r.active.iter().any(|a| !a.2.is_empty());
                }
            }
        }
        r.cut();
        r.drain();
        for (u, _, bufs) in std::mem::take(&mut r.active) {
            for b in bufs {
                unsafe { (self.f.void)(b, self.f.s_release) };
            }
            gate.end(u);
        }
        unsafe { (self.be.sels.pool_pop)(pool) };
        if trace() {
            eprintln!(
                "[metal streamed step: {:.1} ms, {} command buffers, gpu {:.1} ms; routing waits {:.1}, expert reads {:.1}, expert wraps {:.1}, unit waits {:.1} ms]",
                t_run.elapsed().as_secs_f64() * 1e3,
                r.gpu.1,
                r.gpu.0 * 1e3,
                tm[0] * 1e3,
                tm[1] * 1e3,
                tm[2] * 1e3,
                tm[3] * 1e3
            );
        }
        r.ok
    }

    /// Wraps the experts `ids` of `x` (or finds their wraps kept from earlier steps), writes
    /// their entries of the op's address tables, and drops the kept wraps of parts the streamer
    /// no longer caches. Returns the buffers the op reads through the tables.
    fn bind_experts(&self, gate: &dyn crate::gpu::decode::UnitGate, x: &StreamedExperts, ids: &[usize]) -> Option<Vec<Id>> {
        let slot = gate.device_state()?;
        let mut g = slot.lock().unwrap();
        if g.as_ref().is_none_or(|b| !b.is::<PartWraps>()) {
            *g = Some(Box::new(PartWraps {
                map: Default::default(),
                release: (self.f.void, self.f.s_release),
                residency: self.be.residency.clone(),
            }));
        }
        let pw = g.as_mut().unwrap().downcast_mut::<PartWraps>().unwrap();
        let resid = self.be.residency.as_ref();
        let layer = pw.map.entry(x.unit).or_default();
        // Parts of this unit the streamer let go of (their pages may be dropped once unwired).
        let gone: Vec<usize> = layer
            .keys()
            .copied()
            .filter(|&p| !ids.contains(&p) && !gate.part_cached(x.unit, p))
            .collect();
        let mut dropped: Vec<Id> = Vec::new();
        for p in gone {
            dropped.extend(layer.remove(&p).unwrap().iter().map(|b| b.2));
        }
        if let Some(r) = resid
            && !dropped.is_empty()
        {
            r.update(&dropped, false);
        }
        for b in dropped {
            unsafe { (self.f.void)(b, self.f.s_release) };
        }
        let mut res = Vec::new();
        let mut added = Vec::new();
        for &e in ids {
            if let std::collections::hash_map::Entry::Vacant(slot) = layer.entry(e) {
                let mut v = Vec::new();
                for (a, l) in gate.part_spans(x.unit, e) {
                    let (id, off) = self.be.wrap_raw(a as *const u8, l)?;
                    v.push((a, a + l, id, self.be.buffer_gpu_address(id) + off as u64));
                    added.push(id);
                }
                slot.insert(v);
            }
            let v = &layer[&e];
            res.extend(v.iter().map(|b| b.2));
            let at = |a: usize| -> Option<u64> {
                let b = v.iter().find(|b| a >= b.0 && a < b.1)?;
                Some(b.3 + (a - b.0) as u64)
            };
            for (j, &tab) in x.tabs.iter().enumerate() {
                if tab.is_null() || (j == 1 && tab == x.tabs[0]) {
                    continue;
                }
                let r = x.addrs[e][j];
                let d = at(r[0])?;
                let entry = [d, if r[1] != 0 { at(r[1])? } else { d }, if r[2] != 0 { at(r[2])? } else { d }, 0];
                unsafe { std::ptr::copy_nonoverlapping(entry.as_ptr(), (self.f.contents(tab) as *mut u64).add(4 * e), 4) };
            }
        }
        // With a residency set the kept wraps stay resident there (no per-submission residency).
        if let Some(r) = resid {
            if !added.is_empty() {
                r.update(&added, true);
            }
            res.clear();
        }
        Some(res)
    }
}

impl DecodePlan for MetalPlan<'_> {
    fn tokens(&self) -> usize {
        self.tokens
    }

    fn replay_recurrent(&self, slot: &mut Slot, states: &mut [&mut [f32]], keep: usize) -> bool {
        if self.tokens < 2 || keep == 0 || keep > self.tokens {
            return false;
        }
        let _g = self.lock.lock().unwrap();
        let Some(ms) = slot
            .0
            .as_mut()
            .and_then(|b| b.downcast_mut::<MetalSlot>())
            .filter(|m| m.plan == self.id)
        else {
            return false;
        };
        if states.len() != self.states.len() || !self.wrap_states(ms, states) {
            return false;
        }
        let stb: Vec<(Id, usize)> = ms
            .states
            .iter()
            .map(|(_, _, b)| b.as_ref().map_or((std::ptr::null_mut(), 0), |b| (b.id, b.offset_bytes)))
            .collect();
        let ops: Vec<&MOp> = self
            .ops
            .iter()
            .filter(|o| matches!(o, MOp::GdnConv { .. } | MOp::GdnRule { .. }))
            .collect();
        let indirect = if self.be.residency.is_some() { &[][..] } else { &self.indirect[..] };
        let (ok, _) = run_concurrent(self.be, &self.f, &[std::ops::Range { start: 0, end: 1 }], |e, _| {
            e.use_resources(indirect);
            for op in &ops {
                self.encode_multi(e, op, self.arena, self.consts, &[], &stb, 0, keep);
                e.barrier();
            }
        });
        ok
    }

    fn step(&self, slot: &mut Slot, s: Step) -> bool {
        let t_start = Instant::now();
        let _g = self.lock.lock().unwrap();
        let t = s.t;
        if t == 0
            || t > self.tokens
            || s.kv.len() != self.kv.len()
            || s.rope_cs.len() != 2 * t * self.rope_halves.iter().sum::<usize>()
            || s.x.len() != t * self.n_input
            || self.input2.is_some_and(|(_, n)| s.x2.len() != t * n)
            || s.hidden.len() < t * self.hidden.1
            || s.capture
                .as_ref()
                .is_some_and(|c| self.capture.is_none_or(|(_, n)| c.len() < t * n))
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
        if !self.sync_kv(ms, s.kv, pos, t) || !self.wrap_states(ms, s.states) {
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
        let (h, ni) = (self.hidden.1, self.n_input);
        for j in 0..t {
            unsafe { std::ptr::copy_nonoverlapping(s.x[j * ni..].as_ptr(), base.add(self.input + j * self.strides[0]) as *mut f32, ni) };
            if let Some((off, n)) = self.input2 {
                unsafe { std::ptr::copy_nonoverlapping(s.x2[j * n..].as_ptr(), base.add(off + j * self.strides[3]) as *mut f32, n) };
            }
        }
        if self.tokens > 1 {
            // The per-token RoPE rows, table by table (`tokens` rows reserved per table).
            let (mut src, mut dst) = (0, self.rope_off);
            for &half in &self.rope_halves {
                let n = t * 2 * half;
                unsafe { std::ptr::copy_nonoverlapping(s.rope_cs[src..].as_ptr(), base.add(dst) as *mut f32, n) };
                src += n;
                dst += self.tokens * 2 * half * 4;
            }
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
        // Encodes the ops of `range`, those listed in `bound` replaced (streamed weights bound to
        // this step's wraps).
        let encode = |e: &FastEnc, range: std::ops::Range<usize>, bound: &[(usize, MOp)]| {
            e.use_resources(indirect);
            // OXINFER_DEC_ONLY=barrier,norm,mv,qk,attn (debugging): encode only these ops.
            static ONLY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
            let only = ONLY.get_or_init(|| std::env::var("OXINFER_DEC_ONLY").ok());
            for i in range {
                let op = bound.iter().find(|b| b.0 == i).map_or(&self.ops[i], |b| &b.1);
                if let Some(o) = only {
                    let k = match op {
                        MOp::Barrier => "barrier",
                        MOp::Norm { .. } => "norm",
                        MOp::Mv { .. } => "mv",
                        MOp::Topk { .. } | MOp::Experts { .. } => "moe",
                        MOp::QkRope { .. } => "qk",
                        MOp::GdnConv { .. } | MOp::GdnRule { .. } => "gdn",
                        MOp::Glu { .. } | MOp::Ew { .. } | MOp::PleGate { .. } => "glu",
                        MOp::Steer { .. } | MOp::NeuronEdit { .. } => "iv",
                        MOp::AddNorm { .. } => "norm",
                        MOp::Attn { .. } => "attn",
                    };
                    if !o.split(',').any(|x| x == k) {
                        continue;
                    }
                }
                if self.tokens > 1 {
                    self.encode_multi(e, op, arena, consts, &kvb, &stb, pos, t);
                    continue;
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
                        ..
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
                        ..
                    } => encode_mv(e, *pipe, *nr, *ks, w, w2.as_ref(), (arena, *x), (arena, *y), *mode, (arena, *gate)),
                    MOp::Topk {
                        pipe,
                        logits,
                        sel,
                        n,
                        k,
                        scale,
                        ..
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
                        ..
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
                        ..
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
                        ..
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
                    MOp::Ew { pipe, a, b, out, n, p, .. } => {
                        e.pipe(*pipe);
                        e.buf(arena, *a, 0);
                        e.buf(arena, *b, 1);
                        e.buf(arena, *out, 2);
                        e.bytes(p, 3);
                        e.bytes(n, 4);
                        e.dispatch(((*n as usize).div_ceil(256), 1), 256);
                    }
                    MOp::PleGate {
                        pipe, k, q, v, out, n, hc, ..
                    } => {
                        e.pipe(*pipe);
                        e.buf(arena, *k, 0);
                        e.buf(arena, *q, 1);
                        e.buf(arena, *v, 2);
                        e.buf(arena, *out, 3);
                        e.bytes(&[*n, *hc], 4);
                        e.dispatch((*hc as usize, 1), 1024);
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
                    MOp::Glu {
                        pipe, g, u, out, n, gelu, ..
                    } => {
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
                        ..
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
                        ..
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
                        ..
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
        };
        let (ok, gpu) = if self.gate.is_some() {
            (self.run_streamed(end, t, &encode), 0.0)
        } else {
            run_concurrent(self.be, &self.f, &parts, |e, r| encode(e, r, &[]))
        };
        if !ok {
            return false;
        }
        let t_ran = Instant::now();
        let _ = gpu;
        unsafe {
            for j in 0..t {
                std::ptr::copy_nonoverlapping(
                    base.add(self.hidden.0 + j * self.strides[1]) as *const f32,
                    s.hidden[j * h..].as_mut_ptr(),
                    h,
                );
            }
            if let (Some(l), Some((off, n))) = (s.logits, self.logits)
                && want_logits
            {
                let n = n.min(l.len() / t);
                for j in 0..t {
                    super::par_copy(l[j * n..].as_mut_ptr(), base.add(off + j * self.strides[2]) as *const f32, n);
                }
            }
            if let (Some(c), Some((off, n))) = (s.capture, self.capture) {
                for j in 0..t {
                    std::ptr::copy_nonoverlapping(base.add(off + j * self.strides[4]) as *const f32, c[j * n..].as_mut_ptr(), n);
                }
            }
        }
        // The new keys and values go back to the host caches (scratch ones stay on the device,
        // to be overwritten).
        if s.scratch {
            for m in ms.kv.iter_mut() {
                m.synced = pos;
            }
        }
        for (i, (m, c)) in ms.kv.iter_mut().zip(s.kv.iter_mut()).enumerate().filter(|_| !s.scratch) {
            let row = self.kv[i].0 * self.kv[i].1;
            unsafe {
                let k = std::slice::from_raw_parts((self.f.contents(m.k) as *const f32).add(pos * row), t * row);
                let v = std::slice::from_raw_parts((self.f.contents(m.v) as *const f32).add(pos * row), t * row);
                c.push(k, v, t);
            }
            m.synced = pos + t;
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

    /// The few-rows matvec (`Backend::matvec_rows`) against an exact (f64) product for 1 to 4
    /// rows, plain and SiLU pairs, over short and long rows (every variant `mv_variant` picks)
    /// and a GGUF format on the generic kernel.
    #[test]
    fn matvec_rows_matches_reference() {
        use crate::gpu::{Backend, RowsJob};
        let Ok(be) = MetalBackend::new() else { return };
        let mut seed = 5u64;
        let fmts = [GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::Q4_K, GgmlType::F16, GgmlType::IQ4_XS];
        for kind in fmts {
            for (n_out, n_in) in [(37, 512), (70, 2048), (21, 4096), (9, 8192)] {
                let mats: Vec<(GgufMatrix, Vec<f32>)> = (0..2)
                    .map(|_| {
                        let raw = crate::ggml::random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
                        let m = GgufMatrix::from_bytes(kind, &raw, n_out, n_in).unwrap();
                        let mut deq = vec![0f32; n_out * n_in];
                        m.dequantize_rows(0, n_out, &mut deq);
                        (m, deq)
                    })
                    .collect();
                let devs: Vec<_> = mats
                    .iter()
                    .map(|(m, _)| be.upload(&WeightRef::Gguf(m), n_out, n_in).unwrap())
                    .collect();
                for t in 1..=4 {
                    let x: Vec<f32> = (0..t * n_in).map(|_| lcg(&mut seed)).collect();
                    // The exact product and the sum of the terms' magnitudes (the scale of f32
                    // rounding: random blocks can have large scales that cancel).
                    let dot = |w: &[f32], ti: usize, r: usize| {
                        w[r * n_in..(r + 1) * n_in]
                            .iter()
                            .zip(&x[ti * n_in..(ti + 1) * n_in])
                            .map(|(&a, &b)| (a as f64 * b as f64, (a as f64 * b as f64).abs()))
                            .fold((0.0, 0.0), |s, v| (s.0 + v.0, s.1 + v.1))
                    };
                    let (mut y, mut yp) = (vec![0f32; t * n_out], vec![0f32; t * n_out]);
                    let mut jobs = [
                        RowsJob {
                            w: WeightRef::Gguf(&mats[0].0),
                            dev: &devs[0],
                            up: None,
                            n_out,
                            y: &mut y,
                        },
                        RowsJob {
                            w: WeightRef::Gguf(&mats[0].0),
                            dev: &devs[0],
                            up: Some((WeightRef::Gguf(&mats[1].0), &devs[1])),
                            n_out,
                            y: &mut yp,
                        },
                    ];
                    assert!(be.matvec_rows(&x, t, n_in, &mut jobs), "{kind:?}");
                    for ti in 0..t {
                        for r in 0..n_out {
                            let (a, sa) = dot(&mats[0].1, ti, r);
                            let (u, su) = dot(&mats[1].1, ti, r);
                            let sig = 1.0 / (1.0 + (-a).exp());
                            let p = a * sig * u;
                            // Bounds from the magnitudes (the pair: both factors' errors).
                            let (ea, ep) = (sa, sa * u.abs() + su * (a * sig).abs());
                            for (got, want, scale) in [(y[ti * n_out + r], a, ea), (yp[ti * n_out + r], p, ep)] {
                                let err = (got as f64 - want).abs() / (scale + 1e-3);
                                assert!(
                                    err < 2e-6,
                                    "{kind:?} {n_out}x{n_in} t {t} token {ti} row {r}: {got} vs {want} ({err:e})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The multi-token matvecs give every token's rows bit for bit as the single-token kernel does
    /// (speculative verification on the GPU reproduces one-token decoding), in every mode.
    #[test]
    fn decode_mvk_matches_single_token() {
        let Ok(be) = MetalBackend::new() else { return };
        let f = Fns::new(&be);
        let (n_out, n_in) = (37, 512);
        let mut seed = 5u64;
        for ntok in [5, 11] {
            let fmts = [
                MvFmt::Bf16,
                MvFmt::Q8,
                MvFmt::Q4,
                MvFmt::Gguf(GgmlType::F16),
                MvFmt::Gguf(GgmlType::Q4_0),
                MvFmt::Gguf(GgmlType::Q8_0),
                MvFmt::Gguf(GgmlType::Q4_K),
                MvFmt::Gguf(GgmlType::Q6_K),
                MvFmt::Gguf(GgmlType::IQ4_XS),
            ];
            for fmt in fmts {
                let (d1, s1, _) = matrix(fmt, n_out, n_in, &mut seed);
                let (d2, s2, _) = matrix(fmt, n_out, n_in, &mut seed);
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
                // Token rows of x 528 floats apart, of y 40 apart (gates: one per token, at y's end).
                let (sx, sy) = (n_in + 16, n_out + 3);
                let x: Vec<f32> = (0..ntok * sx).map(|_| lcg(&mut seed)).collect();
                let y0: Vec<f32> = (0..ntok * sy).map(|_| lcg(&mut seed)).collect();
                let mut st = be.staging.lock().unwrap();
                let st = &mut *st;
                assert!(be.ensure_staging(st, ntok * sx * 4, ntok * sy * 4));
                let variants: Vec<(usize, usize)> = if fmt.specialized() { MV_VARIANTS.to_vec() } else { vec![(2, 4)] };
                for (nr, ks) in variants {
                    for mode in [MvMode::Store, MvMode::Add, MvMode::SiluPair, MvMode::AddGated] {
                        let up = wt(1);
                        let pair = (mode == MvMode::SiluPair).then_some(&up);
                        let reset = || unsafe {
                            std::ptr::copy_nonoverlapping(x.as_ptr(), f.contents(st.x) as *mut f32, x.len());
                            std::ptr::copy_nonoverlapping(y0.as_ptr(), f.contents(st.y) as *mut f32, y0.len());
                        };
                        reset();
                        let pipe = be.mv_pipe(fmt, nr, ks).unwrap();
                        let (ok, _) = run_concurrent(&be, &f, &[std::ops::Range { start: 0, end: 1 }], |e, _| {
                            for j in 0..ntok {
                                encode_mv(
                                    e,
                                    pipe,
                                    nr,
                                    ks,
                                    &wt(0),
                                    pair,
                                    (st.x, j * sx * 4),
                                    (st.y, j * sy * 4),
                                    mode,
                                    (st.y, (j * sy + n_out) * 4),
                                );
                            }
                        });
                        assert!(ok);
                        let want: Vec<f32> = unsafe { std::slice::from_raw_parts(f.contents(st.y) as *const f32, y0.len()) }.to_vec();
                        reset();
                        // Up to 8 tokens through the 8-token kernels, more through the 16-token ones.
                        let tt = if ntok <= 8 { 8 } else { 16 };
                        let tnr = mvk_rows(tt);
                        let tpipe = be.mvk_pipe(fmt, tnr, ks, tt).unwrap();
                        let (ok, _) = run_concurrent(&be, &f, &[std::ops::Range { start: 0, end: 1 }], |e, _| {
                            let w = wt(0);
                            let w2 = pair.unwrap_or(&w);
                            let p = MvkParams {
                                m: MvParams {
                                    n_out: n_out as u32,
                                    n_in: n_in as u32,
                                    mode: mode as u32,
                                    rows_per_tg: tnr as u32,
                                    w_off: w.data_off,
                                    s_off: w.scales_off,
                                    w2_off: w2.data_off,
                                    s2_off: w2.scales_off,
                                    gpr: w.gpr,
                                    ..Default::default()
                                },
                                ntok: ntok as u32,
                                sx: sx as u32,
                                sy: sy as u32,
                                sgate: sy as u32,
                            };
                            e.pipe(tpipe);
                            e.buf(w.data, 0, 0);
                            e.buf(w.scales, 0, 1);
                            e.buf(st.x, 0, 2);
                            e.buf(st.y, 0, 3);
                            e.bytes(&p, 4);
                            e.buf(w2.data, 0, 5);
                            e.buf(w2.scales, 0, 6);
                            e.buf(w.offs, 0, 7);
                            e.buf(w2.offs, 0, 8);
                            e.buf(st.y, n_out * 4, 9);
                            e.dispatch((n_out.div_ceil(tnr), 1), 32 * ks);
                        });
                        assert!(ok);
                        let got = unsafe { std::slice::from_raw_parts(f.contents(st.y) as *const f32, y0.len()) };
                        for j in 0..ntok {
                            for r in 0..n_out {
                                let (a, b) = (got[j * sy + r], want[j * sy + r]);
                                assert_eq!(a.to_bits(), b.to_bits(), "{fmt:?} {nr}x{ks} {mode:?} token {j} row {r}: {a} vs {b}");
                            }
                        }
                    }
                }
                for b in [db.id, sb.id] {
                    unsafe { (f.void)(b, f.s_release) };
                }
            }
        }
    }

    /// Encodes a fast verification matmul (`fv_*`) of `ntok` tokens (x / y rows `sx` / `sy`
    /// floats apart, the gates at `gate`, `sy` apart).
    #[allow(clippy::too_many_arguments)]
    fn encode_fv(
        e: &FastEnc,
        (pipe, rows): (Id, usize),
        w: &MvWeight,
        w2: Option<&MvWeight>,
        mode: MvMode,
        ntok: usize,
        x: (Id, usize),
        y: (Id, usize),
        gate: (Id, usize),
        sx: usize,
        sy: usize,
    ) {
        let w2 = w2.unwrap_or(w);
        let p = MvkParams {
            m: MvParams {
                n_out: w.n_out as u32,
                n_in: w.n_in as u32,
                mode: mode as u32,
                rows_per_tg: rows as u32,
                w_off: w.data_off,
                s_off: w.scales_off,
                w2_off: w2.data_off,
                s2_off: w2.scales_off,
                o_off: w.offs_off,
                o2_off: w2.offs_off,
                gpr: w.gpr,
                gk: w.gk,
                ..Default::default()
            },
            ntok: ntok as u32,
            sx: sx as u32,
            sy: sy as u32,
            sgate: sy as u32,
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
        e.dispatch((w.n_out.div_ceil(rows), 1), 128);
    }

    /// The fast verification matmuls agree with the single-token kernels up to rounding (they
    /// sum in another order), in every mode, for token counts within and across 8-token tiles.
    #[test]
    fn decode_fv_matches_single_token() {
        const FV_TOL: f32 = 1e-4;
        let Ok(be) = MetalBackend::new() else { return };
        let f = Fns::new(&be);
        let (n_out, n_in) = (37, 512);
        let mut seed = 11u64;
        for ntok in [2usize, 8, 13] {
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
                MvFmt::Gguf(GgmlType::IQ4_XS),
            ];
            for fmt in fmts {
                let (d1, s1, _) = matrix(fmt, n_out, n_in, &mut seed);
                let (d2, s2, _) = matrix(fmt, n_out, n_in, &mut seed);
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
                // x rows for every token of the 8-token tiles the kernel reads.
                let (sx, sy) = (n_in + 16, n_out + 3);
                let x: Vec<f32> = (0..ntok.next_multiple_of(8) * sx).map(|_| lcg(&mut seed)).collect();
                let y0: Vec<f32> = (0..ntok * sy).map(|_| lcg(&mut seed)).collect();
                let mut st = be.staging.lock().unwrap();
                let st = &mut *st;
                assert!(be.ensure_staging(st, x.len() * 4, ntok * sy * 4));
                let tt = if ntok <= 8 { 8 } else { 16 };
                let Some(fpipe) = be.fv_pipe(&wt(0), tt) else {
                    assert!(!FV_FORMATS.contains(&fmt), "{fmt:?}: no fast kernel");
                    continue;
                };
                let (nr, ks) = mv_variant(fmt, n_in);
                let pipe = be.mv_pipe(fmt, nr, ks).unwrap();
                for mode in [MvMode::Store, MvMode::Add, MvMode::SiluPair, MvMode::GeluPair, MvMode::AddGated] {
                    let up = wt(1);
                    let pair = matches!(mode, MvMode::SiluPair | MvMode::GeluPair).then_some(&up);
                    let reset = || unsafe {
                        std::ptr::copy_nonoverlapping(x.as_ptr(), f.contents(st.x) as *mut f32, x.len());
                        std::ptr::copy_nonoverlapping(y0.as_ptr(), f.contents(st.y) as *mut f32, y0.len());
                    };
                    reset();
                    let (ok, _) = run_concurrent(&be, &f, &[std::ops::Range { start: 0, end: 1 }], |e, _| {
                        for j in 0..ntok {
                            encode_mv(
                                e,
                                pipe,
                                nr,
                                ks,
                                &wt(0),
                                pair,
                                (st.x, j * sx * 4),
                                (st.y, j * sy * 4),
                                mode,
                                (st.y, (j * sy + n_out) * 4),
                            );
                        }
                    });
                    assert!(ok);
                    let want: Vec<f32> = unsafe { std::slice::from_raw_parts(f.contents(st.y) as *const f32, y0.len()) }.to_vec();
                    reset();
                    let (ok, _) = run_concurrent(&be, &f, &[std::ops::Range { start: 0, end: 1 }], |e, _| {
                        encode_fv(e, fpipe, &wt(0), pair, mode, ntok, (st.x, 0), (st.y, 0), (st.y, n_out * 4), sx, sy);
                    });
                    assert!(ok);
                    let got = unsafe { std::slice::from_raw_parts(f.contents(st.y) as *const f32, y0.len()) };
                    for j in 0..ntok {
                        let scale = want[j * sy..j * sy + n_out].iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
                        for r in 0..n_out {
                            let (a, b) = (got[j * sy + r], want[j * sy + r]);
                            assert!(
                                (a - b).abs() <= FV_TOL * scale,
                                "{fmt:?} {mode:?} {ntok} tokens: token {j} row {r}: {a} vs {b}"
                            );
                        }
                        // The gate slots after the rows are left alone.
                        assert_eq!(got[j * sy + n_out].to_bits(), y0[j * sy + n_out].to_bits());
                    }
                }
                for b in [db.id, sb.id] {
                    unsafe { (f.void)(b, f.s_release) };
                }
            }
        }
    }

    /// Fast verification matmul times against the single-token kernel
    /// (`cargo test --release --lib decode_fv_bench -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn decode_fv_bench() {
        let be = MetalBackend::new().unwrap();
        let mut seed = 9u64;
        let fmts: Vec<MvFmt> = match std::env::var("OXINFER_MV_FMTS") {
            Ok(v) => v
                .split(',')
                .map(|k| match k {
                    "bf" => MvFmt::Bf16,
                    "q8" => MvFmt::Q8,
                    "q4" => MvFmt::Q4,
                    k => MvFmt::Gguf(GGUF_GPU_TYPES.iter().copied().find(|t| t.name() == k).unwrap()),
                })
                .collect(),
            Err(_) => vec![
                MvFmt::Bf16,
                MvFmt::Q8,
                MvFmt::Q4,
                MvFmt::Gguf(GgmlType::Q8_0),
                MvFmt::Gguf(GgmlType::Q4_K),
            ],
        };
        for fmt in fmts {
            for (n_out, n_in) in [(4096, 4096), (12288, 4096), (4096, 12288), (17408, 5120), (151936, 4096)] {
                let (d, sc, _) = matrix(fmt, n_out.min(17408), n_in, &mut seed);
                // The LM head shape repeats its first rows (the timing only needs the bytes).
                let rep = n_out.div_ceil(17408);
                let one = d.len();
                let d: Vec<u8> = d.iter().copied().cycle().take(one * rep).collect();
                let sc: Vec<u8> = sc.iter().copied().cycle().take(sc.len() * rep).collect();
                let copies = (512usize << 20).div_ceil(d.len()).clamp(1, 32);
                let (dl, sl) = (d.len().next_multiple_of(4096), sc.len().next_multiple_of(4096));
                let mut big = PageVec::<u8>::zeroed(copies * dl);
                let mut bigs = PageVec::<u8>::zeroed(copies * sl);
                for c in 0..copies {
                    big[c * dl..c * dl + d.len()].copy_from_slice(&d);
                    bigs[c * sl..c * sl + sc.len()].copy_from_slice(&sc);
                }
                let wb = be.wrap(big.as_ptr(), big.len()).unwrap();
                let sbuf = be.wrap(bigs.as_ptr(), bigs.len()).unwrap();
                let mut st = be.staging.lock().unwrap();
                let st = &mut *st;
                assert!(be.ensure_staging(st, 16 * n_in * 4, 16 * n_out * 4));
                let wt = |c: usize| MvWeight {
                    fmt,
                    n_out,
                    n_in,
                    data: wb.id,
                    data_off: (wb.offset_bytes + c * dl) as u64,
                    scales: sbuf.id,
                    scales_off: (sbuf.offset_bytes + c * sl) as u64,
                    offs: wb.id,
                    offs_off: 0,
                    gpr: match fmt {
                        MvFmt::Gguf(k) => k.row_bytes(n_in) as u32,
                        MvFmt::Bf16 => (n_in * 2) as u32,
                        _ => 0,
                    },
                    gk: 0,
                };
                let reps = 16;
                let best = |enc: &dyn Fn(&FastEnc, usize)| {
                    let mut b = f64::MAX;
                    for _ in 0..4 {
                        b = b.min(timed(&be, |e| {
                            for r in 0..reps {
                                enc(e, r % copies);
                            }
                        }));
                    }
                    b / reps as f64 * 1e6
                };
                let (nr, ks) = mv_variant(fmt, n_in);
                let pipe = be.mv_pipe(fmt, nr, ks).unwrap();
                let one = best(&|e, c| encode_mv(e, pipe, nr, ks, &wt(c), None, (st.x, 0), (st.y, 0), MvMode::Store, (st.y, 0)));
                let pair = best(&|e, c| {
                    encode_mv(
                        e,
                        pipe,
                        nr,
                        ks,
                        &wt(c),
                        Some(&wt((c + 1) % copies)),
                        (st.x, 0),
                        (st.y, 0),
                        MvMode::SiluPair,
                        (st.y, 0),
                    )
                });
                let mut line = format!("{:>5} {n_out}x{n_in}: t1 {one:.0}us (pair {pair:.0})", fmt.name());
                for ntok in [4, 8, 12, 16] {
                    let tt = if ntok <= 8 { 8 } else { 16 };
                    let fp = be.fv_pipe(&wt(0), tt).unwrap();
                    let t = best(&|e, c| {
                        encode_fv(
                            e,
                            fp,
                            &wt(c),
                            None,
                            MvMode::Store,
                            ntok,
                            (st.x, 0),
                            (st.y, 0),
                            (st.y, 0),
                            n_in,
                            n_out,
                        )
                    });
                    let tp = best(&|e, c| {
                        encode_fv(
                            e,
                            fp,
                            &wt(c),
                            Some(&wt((c + 1) % copies)),
                            MvMode::SiluPair,
                            ntok,
                            (st.x, 0),
                            (st.y, 0),
                            (st.y, 0),
                            n_in,
                            n_out,
                        )
                    });
                    let _ = write!(line, " | t{ntok} {t:.0}us (pair {tp:.0})");
                }
                println!("{line}");
            }
        }
    }

    /// Multi-token matvec times by rows per threadgroup, SIMD-group split and token count
    /// (`OXINFER_MVK_ALL=1 cargo test --release --lib decode_mvk_bench -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn decode_mvk_bench() {
        let be = MetalBackend::new().unwrap();
        let f = Fns::new(&be);
        let mut seed = 9u64;
        for fmt in [MvFmt::Bf16, MvFmt::Q8, MvFmt::Q4, MvFmt::Gguf(GgmlType::Q4_K)] {
            for (n_out, n_in) in [(5120, 5120), (17408, 5120), (5120, 17408)] {
                let (d, sc, _) = matrix(fmt, n_out, n_in, &mut seed);
                let copies = (512usize << 20).div_ceil(d.len()).clamp(1, 32);
                let (dl, sl) = (d.len().next_multiple_of(4096), sc.len().next_multiple_of(4096));
                let mut big = PageVec::<u8>::zeroed(copies * dl);
                let mut bigs = PageVec::<u8>::zeroed(copies * sl);
                for c in 0..copies {
                    big[c * dl..c * dl + d.len()].copy_from_slice(&d);
                    bigs[c * sl..c * sl + sc.len()].copy_from_slice(&sc);
                }
                let wb = be.wrap(big.as_ptr(), big.len()).unwrap();
                let sbuf = be.wrap(bigs.as_ptr(), bigs.len()).unwrap();
                let mut st = be.staging.lock().unwrap();
                let st = &mut *st;
                assert!(be.ensure_staging(st, 8 * n_in * 4, 8 * n_out * 4));
                let ks_list: Vec<usize> = if fmt.specialized() { vec![1, 2, 4] } else { vec![4] };
                let mut line = format!("{:>5} {n_out}x{n_in}:", fmt.name());
                for ntok in [1, 4, 8] {
                    for nr in [1, 2, 4] {
                        for &ks in &ks_list {
                            let Some(pipe) = be.mvk_pipe(fmt, nr, ks, 8) else { continue };
                            let p = |c: usize| MvkParams {
                                m: MvParams {
                                    n_out: n_out as u32,
                                    n_in: n_in as u32,
                                    rows_per_tg: nr as u32,
                                    w_off: (wb.offset_bytes + c * dl) as u64,
                                    s_off: (sbuf.offset_bytes + c * sl) as u64,
                                    w2_off: (wb.offset_bytes + c * dl) as u64,
                                    s2_off: (sbuf.offset_bytes + c * sl) as u64,
                                    gpr: match fmt {
                                        MvFmt::Gguf(k) => k.row_bytes(n_in) as u32,
                                        MvFmt::Bf16 => (n_in * 2) as u32,
                                        _ => 0,
                                    },
                                    ..Default::default()
                                },
                                ntok: ntok as u32,
                                sx: n_in as u32,
                                sy: n_out as u32,
                                sgate: 0,
                            };
                            let reps = 32;
                            let mut best = f64::MAX;
                            for _ in 0..4 {
                                best = best.min(timed(&be, |e| {
                                    for r in 0..reps {
                                        e.pipe(pipe);
                                        e.buf(wb.id, 0, 0);
                                        e.buf(sbuf.id, 0, 1);
                                        e.buf(st.x, 0, 2);
                                        e.buf(st.y, 0, 3);
                                        e.bytes(&p(r % copies), 4);
                                        e.buf(wb.id, 0, 5);
                                        e.buf(sbuf.id, 0, 6);
                                        e.buf(wb.id, 0, 7);
                                        e.buf(wb.id, 0, 8);
                                        e.buf(st.y, 0, 9);
                                        e.dispatch((n_out.div_ceil(nr), 1), 32 * ks);
                                    }
                                }));
                            }
                            let _ = write!(line, " t{ntok} {nr}x{ks}:{:.0}us", best / reps as f64 * 1e6);
                        }
                    }
                }
                println!("{line}");
                let _ = f.contents(st.x);
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
