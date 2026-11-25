//! Whole Whisper encoder passes on Metal ([`crate::gpu::whisper`]): the convolutional stem and
//! every layer of every window of a batch in a few command buffers, the activations in device
//! buffers that live as long as the session. The matmuls are the matrix-unit GEMMs
//! (`MMA_SHADER_SRC`, half inputs with a power-of-two scale per row), attention the fused
//! `wh_flash64` kernels, the row work the `wh_*` kernels of `metal_whisper.metal`. Without matrix
//! units there is no session (the per-op path runs instead).
//!
//! Each window takes its own rows (its 3000 mel frames padded to 3072, its 1500 positions to
//! 1536), and every kernel computes a row from that row alone (attention from its own window), so
//! a window gets the same bits stacked with others as alone.

use super::decode::{FastEnc, Fns, MvFmt, MvParams, MvWeight, run_concurrent};
use super::*;
use crate::gpu::whisper::{ChainRow, CrossLin, Decoder, DevMem, Encoder, Greedy, Lin, Row, Session};

pub(crate) const SRC: &str = include_str!("metal_whisper.metal");

/// Layers per command buffer (the GPU starts on the first while the rest is encoded).
const LAYERS_PER_CB: usize = 8;

/// Profiling (`OXINFER_WHISPER_SKIP=gemm,att,rows`): ops left out of the passes.
fn skip_op(op: &str) -> bool {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("OXINFER_WHISPER_SKIP").unwrap_or_default())
        .split(',')
        .any(|x| x == op)
}

/// Profiling (`OXINFER_WHISPER_REP=op:n`): op `op` of the decoder layers (qkv, self, o, cq,
/// cross, merge, co, fc1, fc2) encoded n times (its marginal cost).
fn reps(op: &str) -> usize {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let s = S.get_or_init(|| std::env::var("OXINFER_WHISPER_REP").unwrap_or_default());
    s.split_once(':')
        .filter(|(o, _)| *o == op)
        .and_then(|(_, n)| n.parse().ok())
        .unwrap_or(1)
}

/// Profiling (`OXINFER_WHISPER_PROF`): decoder submissions, steps, GPU and wall nanoseconds.
static PROF: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
fn prof_on() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var_os("OXINFER_WHISPER_PROF").is_some())
}

/// The encoder of a decoder pass: barriers left out with `OXINFER_WHISPER_SKIP=dbar`
/// (profiling); with `ind`, dispatches read their sizes from a chain's indirect arguments: the
/// arguments buffer, the host's copy of this step's sizes (`end` u32s at most, `next` used) and
/// the u32 index of the live copy the dispatches read.
struct DEnc<'a, 'f> {
    e: &'a FastEnc<'f>,
    nobar: bool,
    ind: Option<(Id, *mut u32, usize)>,
    next: std::cell::Cell<usize>,
    end: usize,
}
impl<'f> std::ops::Deref for DEnc<'_, 'f> {
    type Target = FastEnc<'f>;
    fn deref(&self) -> &FastEnc<'f> {
        self.e
    }
}
impl DEnc<'_, '_> {
    fn barrier(&self) {
        if !self.nobar {
            self.e.barrier();
        }
    }
    fn dispatch(&self, groups: (usize, usize), threads: usize) {
        if let Some((buf, host, live)) = self.ind {
            let i = self.next.get();
            if i + 3 <= self.end {
                // SAFETY: the host copy holds `end` u32s.
                unsafe { std::ptr::copy_nonoverlapping([groups.0 as u32, groups.1 as u32, 1].as_ptr(), host.add(i), 3) };
                self.next.set(i + 3);
                self.e.dispatch_indirect(buf, (live + i) * 4, threads);
                return;
            }
        }
        self.e.dispatch(groups, threads);
    }
}

/// Threads a threadgroup of the decoder matvecs (two rows with one SIMD-group a row: fastest
/// on an M5 Max, against 1 to 16 rows a threadgroup).
fn mv_threads(ks: usize) -> usize {
    64.max(32 * ks)
}

/// u32s of a chain step's indirect dispatch sizes, for `t` rows of `nl` layers (three per
/// dispatch: matvecs by groups of 4 rows, every row's attention parts and merges, the key and
/// value stores, the LayerNorms, the logits, the input and the choice). A step with more
/// dispatches than that dispatches the rest directly.
fn step_cap(nl: usize, t: usize) -> usize {
    3 * (nl * (8 * t.div_ceil(4) + 6 * t + 3) + t + 8)
}

/// A decoder matvec: the weight, its format's index in `Pipes::mv` and its SIMD-groups per row
/// (as `mvt_pipe` splits the row's chunks), when each lane takes at most `WDM_MAXC` chunks.
fn dec_mv(l: &Lin) -> Option<(MvWeight, usize, usize)> {
    let w = MvWeight::new(&l.w, l.dev, l.n_out, l.n_in)?;
    let (fi, chunk) = match w.fmt {
        MvFmt::Bf16 => (0, 16),
        MvFmt::Q8 => (1, 16),
        MvFmt::Q4 => (2, 32),
        MvFmt::Gguf(_) => return None,
    };
    let nch = l.n_in / chunk;
    let ks = (nch / 64).clamp(1, 4).next_power_of_two().min(4);
    (nch.div_ceil(ks * 32) <= WDM_MAXC).then_some((w, fi, ks))
}

/// A device buffer that grows on demand (released with the session).
struct DBuf {
    id: Id,
    cap: usize,
}

struct Pipes {
    cols1: Id,
    cols2: Id,
    res_ln: Id,
    gelu_half: Id,
    bias: Id,
    flash: Id,
    flash_f32: Id,
    /// `whd_mv_{bf,q8,q4}`.
    mv: [Id; 3],
    gate: Id,
    part_f32: Id,
    part_h: Id,
    merge: Id,
    embed: Id,
    select: Id,
    ln: Id,
}

pub(crate) struct MetalWhisper {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    bufs: Vec<DBuf>,
    /// The decoder's constants (biases, norms) in `B::DConsts`: the address of the decoder's
    /// first norm weight they were made from, and their offsets.
    dconsts: Option<(usize, DecConsts)>,
    /// The mask and position table in `B::Cmask` / `B::Cpos` (a copy of the mask, the address of
    /// the table).
    chain_consts: Option<(Vec<u8>, usize)>,
}

/// Byte offsets in `B::DConsts` (after zeros, the biases a layer lacks): per layer the biases
/// (q, k, v, o, cross q, cross o, fc1, fc2) and norms (ln1, ln_cross, ln2: weight then bias),
/// then the final norm.
struct DecConsts {
    layers: Vec<([usize; 8], [usize; 6])>,
    ln: [usize; 2],
}

// SAFETY: the session is used by one thread at a time (`&mut self`); the Objective-C objects
// it holds are thread-safe Metal buffers and pipelines.
unsafe impl Send for MetalWhisper {}

/// Buffer slots of a session.
#[derive(Clone, Copy)]
enum B {
    Mel = 0,
    C1h,
    Y1,
    C2h,
    X,
    Xh,
    Rs,
    Q,
    K,
    V,
    Qh,
    Kh,
    Vh,
    Att,
    F,
    Out,
    Consts,
    // Decoder steps.
    Dx,
    Dh,
    Dq,
    Dk,
    Dv,
    Datt,
    Df,
    Dlog,
    Dpart,
    DConsts,
    // Chains.
    Cmask,
    Cpos,
    Cstate,
    Cout,
    Cargs,
}
const NBUF: usize = 32;

/// Weight chunks per lane of the decoder matvecs (`whd_mv_*`) at most, and values of a row of
/// the decoder's LayerNorm kernel (`whd_ln`, 256 threads of 2 float4s).
const WDM_MAXC: usize = 4;
const WDM_MAXD: usize = 2048;

/// Positions per threadgroup of the decoder's attention (`whd_part_*`), and chunks at most.
const DEC_CHUNK: usize = 128;
const DEC_CHUNKS: usize = 1536 / DEC_CHUNK;

#[repr(C)]
struct U4([u32; 4]);
#[repr(C)]
struct U8([u32; 8]);

impl MetalBackend {
    /// Why the matrix-unit library did not compile, if it did not (a broken kernel there leaves
    /// every model on the slower kernels).
    #[cfg(test)]
    pub(crate) fn library_error(&self) -> Option<&str> {
        self.mma_error.as_deref()
    }

    /// [`crate::gpu::Backend::whisper_session`].
    pub(crate) fn whisper_session_impl(&'static self) -> Option<Box<dyn Session>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            cols1: pipe("wh_cols1")?,
            cols2: pipe("wh_cols2")?,
            res_ln: pipe("wh_res_ln")?,
            gelu_half: pipe("wh_gelu_half")?,
            bias: pipe("wh_bias")?,
            flash: pipe("wh_flash64")?,
            flash_f32: pipe("wh_flash64_f32")?,
            mv: [pipe("whd_mv_bf")?, pipe("whd_mv_q8")?, pipe("whd_mv_q4")?],
            gate: pipe("whc_gate")?,
            part_f32: pipe("whd_part_f32")?,
            part_h: pipe("whd_part_h")?,
            merge: pipe("whd_attn_merge")?,
            embed: pipe("whc_embed")?,
            select: pipe("whc_select")?,
            ln: pipe("whd_ln")?,
        };
        Some(Box::new(MetalWhisper {
            be: self,
            f: Fns::new(self),
            p,
            bufs: (0..NBUF)
                .map(|_| DBuf {
                    id: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
            dconsts: None,
            chain_consts: None,
        }))
    }

    /// A new shared buffer of `bytes` (zero-filled), or None.
    fn whisper_buffer(&self, bytes: usize) -> Option<Id> {
        let new_len: MsgNewBufLen = self.o.msg_send();
        let id = unsafe { new_len(self.device, self.sels.new_buffer_len, bytes.max(256).div_ceil(PAGE) * PAGE, 0) };
        (!id.is_null()).then_some(id)
    }
}

/// Device memory of [`crate::gpu::whisper::DevMem`]: `blocks` blocks of `cap` rows of `d` values
/// of `elem` bytes (cross-attention keys and values: per decoder layer the keys then the values
/// of the window's positions, half; a self-attention cache: the same of `cap` positions, f32,
/// grown on demand up to `max`).
pub(crate) struct WhMem {
    be: &'static MetalBackend,
    blocks: usize,
    d: usize,
    elem: usize,
    max: usize,
    /// The buffer and its rows per block.
    buf: std::sync::Mutex<(Id, usize)>,
}

// SAFETY: the buffer is a thread-safe Metal object, its handle behind a mutex.
unsafe impl Send for WhMem {}
unsafe impl Sync for WhMem {}

impl WhMem {
    fn new(be: &'static MetalBackend, blocks: usize, d: usize, elem: usize, cap: usize, max: usize) -> Option<Self> {
        let id = be.whisper_buffer(blocks * cap * d * elem)?;
        Some(WhMem {
            be,
            blocks,
            d,
            elem,
            max,
            buf: std::sync::Mutex::new((id, cap)),
        })
    }

    fn get(&self) -> (Id, usize) {
        *self.buf.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies the first `rows` rows of every block of `src` (`src_cap` rows per block) into
    /// `dst` (`dst_cap` rows per block).
    fn copy_rows(&self, src: Id, src_cap: usize, dst: Id, dst_cap: usize, rows: usize) {
        let f: MsgPtr = self.be.o.msg_send();
        let row = self.d * self.elem;
        // SAFETY: both buffers are shared and hold `blocks` blocks of their capacity.
        unsafe {
            let s = f(src, self.be.sels.contents) as *const u8;
            let t = f(dst, self.be.sels.contents) as *mut u8;
            for b in 0..self.blocks {
                std::ptr::copy_nonoverlapping(s.add(b * src_cap * row), t.add(b * dst_cap * row), rows * row);
            }
        }
    }

    /// Makes room for `rows` rows per block (the rows held are kept).
    fn reserve(&self, rows: usize) -> bool {
        let mut g = self.buf.lock().unwrap_or_else(|e| e.into_inner());
        let (id, cap) = *g;
        if rows <= cap {
            return true;
        }
        if rows > self.max {
            return false;
        }
        let new_cap = rows.max(2 * cap).min(self.max);
        let Some(nid) = self.be.whisper_buffer(self.blocks * new_cap * self.d * self.elem) else {
            return false;
        };
        self.copy_rows(id, cap, nid, new_cap, cap);
        let void: MsgVoid = self.be.o.msg_send();
        unsafe { void(id, self.be.sels.release) };
        *g = (nid, new_cap);
        true
    }
}

impl Drop for WhMem {
    fn drop(&mut self) {
        let void: MsgVoid = self.be.o.msg_send();
        let (id, _) = self.get();
        unsafe { void(id, self.be.sels.release) };
    }
}

impl DevMem for WhMem {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn duplicate(&self, positions: usize) -> Option<Box<dyn DevMem>> {
        let (id, cap) = self.get();
        let m = WhMem::new(self.be, self.blocks, self.d, self.elem, cap, self.max)?;
        let (nid, _) = m.get();
        self.copy_rows(id, cap, nid, cap, positions.min(cap));
        Some(Box::new(m))
    }
}

impl Drop for MetalWhisper {
    fn drop(&mut self) {
        if prof_on() {
            let v: Vec<u64> = PROF.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect();
            let st = v[1].max(1) as f64;
            eprintln!(
                "whisper decoder: {} submissions, {} steps, GPU {:.3} ms/step, wall {:.3} ms/step",
                v[0],
                v[1],
                v[2] as f64 / st / 1e6,
                v[3] as f64 / st / 1e6
            );
        }
        let void: MsgVoid = self.be.o.msg_send();
        for b in &self.bufs {
            if !b.id.is_null() {
                unsafe { void(b.id, self.be.sels.release) };
            }
        }
    }
}

impl MetalWhisper {
    /// Makes buffer `b` hold at least `bytes` (new buffers are zero-filled; contents are lost when
    /// one grows).
    fn ensure(&mut self, b: B, bytes: usize) -> bool {
        let s = &mut self.bufs[b as usize];
        if s.cap >= bytes && !s.id.is_null() {
            return true;
        }
        let new_len: MsgNewBufLen = self.be.o.msg_send();
        let void: MsgVoid = self.be.o.msg_send();
        let cap = bytes.max(256).div_ceil(PAGE) * PAGE;
        let id = unsafe { new_len(self.be.device, self.be.sels.new_buffer_len, cap, 0) };
        if id.is_null() {
            return false;
        }
        if !s.id.is_null() {
            unsafe { void(s.id, self.be.sels.release) };
        }
        s.id = id;
        s.cap = cap;
        true
    }

    fn host(&self, b: B) -> *mut f32 {
        self.f.contents(self.bufs[b as usize].id) as *mut f32
    }
}

/// Whether the device can run layer `l` (a matrix-unit GEMM for it).
fn runnable(be: &MetalBackend, l: &Lin) -> bool {
    let Some(mw) = l.dev.downcast_ref::<MetalWeight>() else {
        return false;
    };
    l.n_out.is_multiple_of(4) && be.mma_pipe(&l.w, mw.offsets.is_some(), l.n_out, l.n_in, Tile::Dense).is_some()
}

/// `y = W x` (or `y +=` with `add`) over `rows` rows (a multiple of 128) of scaled half rows `xh`
/// (inverse scales `rs`) into rows `ldy` floats apart, then a barrier.
#[allow(clippy::too_many_arguments)]
fn gemm(e: &FastEnc, be: &MetalBackend, l: &Lin, xh: Id, rs: Id, rows: usize, y: Id, ldy: usize, add: bool) {
    gemm_nb(e, be, l, xh, rs, rows, y, ldy, add);
    e.barrier();
}

impl Session for MetalWhisper {
    fn encode(&mut self, m: &Encoder, mels: &[&[f32]], out: &mut [Vec<f32>], cross: &[CrossLin]) -> Option<Vec<Box<dyn DevMem>>> {
        let be = self.be;
        let (d, f, nh, nm, n) = (m.d, m.ffn, m.heads, m.n_mels, m.n);
        let nb = mels.len();
        let frames = 2 * n;
        if nb == 0 || out.len() != nb || nh == 0 || d != nh * 64 || !d.is_multiple_of(64) || !f.is_multiple_of(4) {
            return None;
        }
        if mels.iter().any(|x| x.len() != frames * nm) || m.pos.len() != n * d {
            return None;
        }
        let k1 = m.conv1.n_in;
        if k1 < 3 * nm || m.conv1.n_out != d || m.conv2.n_in != 3 * d || m.conv2.n_out != d {
            return None;
        }
        let lins = m
            .layers
            .iter()
            .flat_map(|l| [&l.q, &l.k, &l.v, &l.o, &l.fc1, &l.fc2])
            .chain([&m.conv1, &m.conv2]);
        for l in lins.chain(cross.iter().flat_map(|c| [&c.k, &c.v])) {
            if !runnable(be, l) {
                return None;
            }
        }
        if cross
            .iter()
            .any(|c| c.k.n_in != d || c.v.n_in != d || c.k.n_out != d || c.v.n_out != d)
        {
            return None;
        }
        // The cross-attention keys and values of each window: per decoder layer n rows of keys
        // then n of values, half.
        let dc = cross.first().map_or(0, |c| c.k.n_out);
        let mut cmem = Vec::with_capacity(if cross.is_empty() { 0 } else { nb });
        if !cross.is_empty() {
            for _ in 0..nb {
                cmem.push(WhMem::new(be, 2 * cross.len(), dc, 2, n, n)?);
            }
        }
        if m.layers
            .iter()
            .any(|l| l.q.n_out != d || l.k.n_out != d || l.v.n_out != d || l.o.n_in != d || l.fc1.n_out != f || l.fc2.n_in != f)
        {
            return None;
        }
        // Rows per window: frames and positions padded to whole GEMM tiles.
        let w1 = frames.next_multiple_of(MMA_TM);
        let w2 = n.next_multiple_of(MMA_TM);
        let (r1, r) = (nb * w1, nb * w2);
        let ok = self.ensure(B::Mel, nb * frames * nm * 4)
            && self.ensure(B::C1h, r1 * k1 * 2)
            && self.ensure(B::Y1, r1 * d * 4)
            && self.ensure(B::C2h, r * 3 * d * 2)
            && self.ensure(B::X, r * d * 4)
            && self.ensure(B::Xh, r * f.max(d).max(k1) * 2)
            && self.ensure(B::Rs, r1.max(r) * 4)
            && self.ensure(B::Q, r * d * 4)
            && self.ensure(B::K, r * d * 4)
            && self.ensure(B::V, r * d * 4)
            && self.ensure(B::Qh, r * d * 4)
            && self.ensure(B::Kh, r * d * 4)
            && self.ensure(B::Vh, r * d * 4)
            && self.ensure(B::Att, r * d * 4)
            && self.ensure(B::F, r * f * 4)
            && self.ensure(B::Out, r * d * 4);
        if !ok {
            return None;
        }
        // Constants: positions, biases, norms (64-float aligned), and zeros for absent biases.
        let mut consts: Vec<f32> = Vec::new();
        let mut put = |v: &[f32]| -> usize {
            let o = consts.len();
            consts.extend_from_slice(v);
            consts.resize(consts.len().next_multiple_of(64), 0.0);
            o * 4
        };
        let zeros = put(&vec![0f32; f.max(d)]);
        let mut bias = |l: &Lin| l.bias.map_or(zeros, &mut put);
        let b_conv1 = bias(&m.conv1);
        let b_conv2 = bias(&m.conv2);
        let lb: Vec<[usize; 6]> = m
            .layers
            .iter()
            .map(|l| [bias(&l.q), bias(&l.k), bias(&l.v), bias(&l.o), bias(&l.fc1), bias(&l.fc2)])
            .collect();
        let pos = put(m.pos);
        let ln: Vec<[usize; 4]> = m
            .layers
            .iter()
            .map(|l| [put(l.ln1.w), put(l.ln1.b), put(l.ln2.w), put(l.ln2.b)])
            .collect();
        let ln_post = [put(m.ln_post.w), put(m.ln_post.b)];
        let cb: Vec<[usize; 2]> = cross
            .iter()
            .map(|c| [c.k.bias.map_or(zeros, &mut put), c.v.bias.map_or(zeros, &mut put)])
            .collect();
        if !self.ensure(B::Consts, consts.len() * 4) {
            return None;
        }
        for l in m
            .layers
            .iter()
            .flat_map(|l| [&l.q, &l.k, &l.v, &l.o, &l.fc1, &l.fc2])
            .chain([&m.conv1, &m.conv2])
        {
            if let Some(mw) = l.dev.downcast_ref::<MetalWeight>() {
                mw.keep_resident(&be.residency);
            }
        }
        // SAFETY: every buffer was sized above; the device is idle (the last pass was waited for).
        unsafe {
            par_copy(self.host(B::Consts), consts.as_ptr(), consts.len());
            let mel = self.host(B::Mel);
            for (b, x) in mels.iter().enumerate() {
                par_copy(mel.add(b * frames * nm), x.as_ptr(), frames * nm);
            }
        }
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let cst = id(B::Consts);
        let p = &self.p;
        let mma = be.mma.as_ref().unwrap();
        let half_att = m.half_attention;
        let nl = m.layers.len();
        // The residual step of rows of X then a LayerNorm into Xh (half) or Out (f32).
        let res_ln = |e: &FastEnc, b: usize, mode: u32, norm: Option<[usize; 2]>, to_out: bool| {
            e.pipe(p.res_ln);
            e.buf(id(B::X), 0, 0);
            e.buf(cst, b, 1);
            e.buf(cst, pos, 2);
            let [nw, nbias] = norm.unwrap_or([zeros, zeros]);
            e.buf(cst, nw, 3);
            e.buf(cst, nbias, 4);
            e.buf(if to_out { id(B::Out) } else { id(B::Xh) }, 0, 5);
            e.buf(id(B::Rs), 0, 6);
            e.bytes(
                &U8([
                    d as u32,
                    mode,
                    norm.is_some() as u32,
                    !to_out as u32,
                    w2 as u32,
                    n as u32,
                    m.eps.to_bits(),
                    0,
                ]),
                7,
            );
            e.dispatch((r, 1), 256);
            e.barrier();
        };
        let parts: Vec<std::ops::Range<usize>> = (0..nl.div_ceil(LAYERS_PER_CB).max(1))
            .map(|c| c * LAYERS_PER_CB..((c + 1) * LAYERS_PER_CB).min(nl))
            .collect();
        let eb = if half_att { 2 } else { 4 };
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            if range.start == 0 {
                // The stem: conv1 (+ bias, GELU in conv2's unfolding), conv2, GELU and positions.
                e.pipe(p.cols1);
                e.buf(id(B::Mel), 0, 0);
                e.buf(id(B::C1h), 0, 1);
                e.buf(id(B::Rs), 0, 2);
                e.bytes(&U4([frames as u32, nm as u32, k1 as u32, w1 as u32]), 3);
                e.dispatch((r1, 1), 256);
                e.barrier();
                gemm(e, be, &m.conv1, id(B::C1h), id(B::Rs), r1, id(B::Y1), d, false);
                e.pipe(p.cols2);
                e.buf(id(B::Y1), 0, 0);
                e.buf(cst, b_conv1, 1);
                e.buf(id(B::C2h), 0, 2);
                e.buf(id(B::Rs), 0, 3);
                e.bytes(&U8([d as u32, frames as u32, n as u32, w1 as u32, w2 as u32, 0, 0, 0]), 4);
                e.dispatch((r, 1), 256);
                e.barrier();
                gemm(e, be, &m.conv2, id(B::C2h), id(B::Rs), r, id(B::X), d, false);
                res_ln(e, b_conv2, 1, Some([ln[0][0], ln[0][1]]), false);
            }
            for li in range.clone() {
                let l = &m.layers[li];
                let [bq, bk, bv, bo, bf1, bf2] = lb[li];
                // Attention.
                gemm_nb(e, be, &l.q, id(B::Xh), id(B::Rs), r, id(B::Q), d, false);
                gemm_nb(e, be, &l.k, id(B::Xh), id(B::Rs), r, id(B::K), d, false);
                gemm_nb(e, be, &l.v, id(B::Xh), id(B::Rs), r, id(B::V), d, false);
                e.barrier();
                for (src, dst, b) in [(B::Q, B::Qh, bq), (B::K, B::Kh, bk), (B::V, B::Vh, bv)] {
                    e.pipe(p.bias);
                    e.buf(id(src), 0, 0);
                    e.buf(cst, b, 1);
                    e.buf(id(dst), 0, 2);
                    e.bytes(&U4([d as u32, 1, half_att as u32, 0]), 3);
                    e.dispatch(((d / 4).div_ceil(256), r), 256);
                }
                e.barrier();
                e.pipe(if half_att { p.flash } else { p.flash_f32 });
                for b in (0..nb).filter(|_| !skip_op("att")) {
                    let off = b * w2 * d;
                    e.buf(id(B::Qh), off * eb, 0);
                    e.buf(id(B::Kh), off * eb, 1);
                    e.buf(id(B::Vh), off * eb, 2);
                    e.buf(id(B::Att), off * 4, 3);
                    e.bytes(&U4([n as u32, n as u32, d as u32, (1.0 / 8.0f32).to_bits()]), 4);
                    e.dispatch((n.div_ceil(64), nh), 128);
                }
                e.barrier();
                e.pipe(mma.x_half);
                e.buf(id(B::Att), 0, 0);
                e.buf(id(B::Xh), 0, 1);
                e.buf(id(B::Rs), 0, 2);
                e.bytes(&U4([d as u32, 0, d as u32, 0]), 3);
                e.buf(id(B::Att), 0, 4);
                e.dispatch((r, 1), 256);
                e.barrier();
                gemm(e, be, &l.o, id(B::Xh), id(B::Rs), r, id(B::X), d, true);
                res_ln(e, bo, 0, Some([ln[li][2], ln[li][3]]), false);
                // Feed-forward.
                gemm(e, be, &l.fc1, id(B::Xh), id(B::Rs), r, id(B::F), f, false);
                e.pipe(p.gelu_half);
                e.buf(id(B::F), 0, 0);
                e.buf(cst, bf1, 1);
                e.buf(id(B::Xh), 0, 2);
                e.buf(id(B::Rs), 0, 3);
                e.bytes(&U4([f as u32, 0, 0, 0]), 4);
                e.dispatch((r, 1), 256);
                e.barrier();
                gemm(e, be, &l.fc2, id(B::Xh), id(B::Rs), r, id(B::X), d, true);
                if li + 1 < nl {
                    res_ln(e, bf2, 0, Some([ln[li + 1][0], ln[li + 1][1]]), false);
                } else {
                    res_ln(e, bf2, 0, Some(ln_post), true);
                }
            }
            if range.end == nl && !cross.is_empty() {
                // Each decoder layer's keys and values of the encoder output.
                e.pipe(mma.x_half);
                e.buf(id(B::Out), 0, 0);
                e.buf(id(B::Xh), 0, 1);
                e.buf(id(B::Rs), 0, 2);
                e.bytes(&U4([d as u32, 0, d as u32, 0]), 3);
                e.buf(id(B::Out), 0, 4);
                e.dispatch((r, 1), 256);
                e.barrier();
                for (li, c) in cross.iter().enumerate() {
                    gemm_nb(e, be, &c.k, id(B::Xh), id(B::Rs), r, id(B::K), d, false);
                    gemm_nb(e, be, &c.v, id(B::Xh), id(B::Rs), r, id(B::V), d, false);
                    e.barrier();
                    for (b, mem) in cmem.iter().enumerate() {
                        let (cid, _) = mem.get();
                        for (j, src) in [B::K, B::V].into_iter().enumerate() {
                            e.pipe(p.bias);
                            e.buf(id(src), b * w2 * d * 4, 0);
                            e.buf(cst, cb[li][j], 1);
                            e.buf(cid, (2 * li + j) * n * d * 2, 2);
                            e.bytes(&U4([d as u32, 1, 1, 0]), 3);
                            e.dispatch(((d / 4).div_ceil(256), n), 256);
                        }
                    }
                    e.barrier();
                }
            }
        });
        if !ok {
            return None;
        }
        let o = self.host(B::Out) as *const f32;
        for (b, y) in out.iter_mut().enumerate() {
            y.resize(n * d, 0.0);
            // SAFETY: Out holds r rows of d floats, window b's from row b * w2.
            unsafe { par_copy(y.as_mut_ptr(), o.add(b * w2 * d), n * d) };
        }
        Some(cmem.into_iter().map(|m| Box::new(m) as Box<dyn DevMem>).collect())
    }

    fn decodes(&self, dec: &Decoder) -> bool {
        let d = dec.d;
        if dec.heads == 0
            || d != dec.heads * 64
            || d > WDM_MAXD
            || dec.max_pos > 1536
            || !dec.ffn.is_multiple_of(4)
            || dec.head.n_out != dec.vocab
        {
            return false;
        }
        let lins = dec
            .layers
            .iter()
            .flat_map(|l| [&l.q, &l.k, &l.v, &l.o, &l.cq, &l.co, &l.fc1, &l.fc2])
            .chain([&dec.head]);
        if lins.into_iter().any(|l| dec_mv(l).is_none()) {
            return false;
        }
        dec.layers.iter().all(|l| {
            [&l.q, &l.k, &l.v, &l.o, &l.cq, &l.co].iter().all(|x| x.n_out == d && x.n_in == d)
                && l.fc1.n_out == dec.ffn
                && l.fc1.n_in == d
                && l.fc2.n_in == dec.ffn
                && l.fc2.n_out == d
        })
    }

    fn new_kv(&mut self, dec: &Decoder) -> Option<Box<dyn DevMem>> {
        Some(Box::new(WhMem::new(
            self.be,
            2 * dec.layers.len(),
            dec.d,
            4,
            32.min(dec.max_pos),
            dec.max_pos,
        )?))
    }

    fn decode(&mut self, dec: &Decoder, rows: &[Row], want: &[usize], logits: &mut [f32]) -> bool {
        let (d, v) = (dec.d, dec.vocab);
        let t = rows.len();
        if t == 0 || logits.len() < want.len() * v || want.iter().any(|&w| w >= t) || rows.iter().any(|r| r.x.len() != d) {
            return false;
        }
        let Some(mems) = rows
            .iter()
            .map(|r| Some((r.kv.as_any().downcast_ref::<WhMem>()?, r.cross.as_any().downcast_ref::<WhMem>()?)))
            .collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        let pos: Vec<usize> = rows.iter().map(|r| r.pos).collect();
        if !self.prepare(dec, &mems, &pos, 1, want.len()) {
            return false;
        }
        // SAFETY: Dx holds t rows of d floats.
        unsafe {
            let x = self.host(B::Dx);
            for (i, r) in rows.iter().enumerate() {
                std::ptr::copy_nonoverlapping(r.x.as_ptr(), x.add(i * d), d);
            }
        }
        if !self.steps(dec, &mems, &pos, 1, Some(want)) {
            return false;
        }
        // SAFETY: Dlog holds want.len() rows of the vocabulary.
        unsafe { par_copy(logits.as_mut_ptr(), self.host(B::Dlog), want.len() * v) };
        true
    }

    fn chains(&self, dec: &Decoder) -> bool {
        self.decodes(dec) && matches!(dec.head.w, WeightRef::Bf16(_)) && dec.head.n_in == dec.d
    }

    fn greedy(&mut self, dec: &Decoder, g: &Greedy, rows: &[ChainRow], steps: usize, out: &mut [(u32, f32)]) -> bool {
        let (d, v) = (dec.d, dec.vocab);
        let t = rows.len();
        if t == 0 || steps == 0 || out.len() < t * steps || g.mask.len() != v || g.pos.len() < dec.max_pos * d || !self.chains(dec) {
            return false;
        }
        if rows
            .iter()
            .any(|r| r.tokens.is_empty() || r.tokens.iter().any(|&x| x as usize >= v))
        {
            return false;
        }
        let Some(mems) = rows
            .iter()
            .map(|r| Some((r.kv.as_any().downcast_ref::<WhMem>()?, r.cross.as_any().downcast_ref::<WhMem>()?)))
            .collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        // Each row its own sequence.
        if (1..t).any(|i| (0..i).any(|j| std::ptr::eq(mems[i].0, mems[j].0))) {
            return false;
        }
        let pos: Vec<usize> = rows.iter().map(|r| r.pos).collect();
        if !self.prepare(dec, &mems, &pos, steps, t) {
            return false;
        }
        let ok = self.ensure(B::Cmask, v)
            && self.ensure(B::Cpos, dec.max_pos * d * 4)
            && self.ensure(B::Cstate, t * 32)
            && self.ensure(B::Cout, t * steps * 8)
            && self.ensure(B::Cargs, 2 * steps * step_cap(dec.layers.len(), t) * 4);
        if !ok {
            return false;
        }
        // The mask and positions, uploaded when they change.
        let key = g.pos.as_ptr() as usize;
        if self.chain_consts.as_ref().is_none_or(|(m, k)| m[..] != g.mask[..] || *k != key) {
            // SAFETY: sized above; the device is idle.
            unsafe {
                std::ptr::copy_nonoverlapping(g.mask.as_ptr(), self.host(B::Cmask) as *mut u8, v);
                par_copy(self.host(B::Cpos), g.pos.as_ptr(), dec.max_pos * d);
            }
            self.chain_consts = Some((g.mask.to_vec(), key));
        }
        // SAFETY: Cstate holds t states of 8 u32s.
        unsafe {
            let st = self.host(B::Cstate) as *mut u32;
            for (i, r) in rows.iter().enumerate() {
                let n = r.tokens.len();
                let last_ts = r
                    .tokens
                    .iter()
                    .rev()
                    .find(|&&x| x >= g.timestamp_begin)
                    .copied()
                    .unwrap_or(u32::MAX);
                let s = [
                    r.tokens[n - 1],
                    if n >= 2 { r.tokens[n - 2] } else { 0 },
                    last_ts,
                    n as u32,
                    r.timestamps as u32,
                    0,
                    0,
                    0,
                ];
                std::ptr::copy_nonoverlapping(s.as_ptr(), st.add(i * 8), 8);
            }
        }
        let chain = ChainSpec { g, steps };
        if !self.steps_chain(dec, &mems, &pos, &chain) {
            return false;
        }
        // SAFETY: Cout holds t * steps (token, log-probability) pairs.
        unsafe {
            let o = self.host(B::Cout) as *const u32;
            for (k, x) in out.iter_mut().take(t * steps).enumerate() {
                *x = (*o.add(2 * k), f32::from_bits(*o.add(2 * k + 1)));
            }
        }
        true
    }
}

/// What a chain's steps need besides the decoder pass.
struct ChainSpec<'a> {
    g: &'a Greedy<'a>,
    steps: usize,
}

impl MetalWhisper {
    /// Sizes the step buffers for `t` rows (`nlog` rows of logits), checks the caches and makes
    /// room in them for `steps` positions from `pos`, makes the decoder's constants.
    fn prepare(&mut self, dec: &Decoder, mems: &[(&WhMem, &WhMem)], pos: &[usize], steps: usize, nlog: usize) -> bool {
        let (d, ffn, nh, v, nl) = (dec.d, dec.ffn, dec.heads, dec.vocab, dec.layers.len());
        let t = pos.len();
        for (i, (kv, c)) in mems.iter().enumerate() {
            let end = pos[i] + steps;
            if kv.blocks != 2 * nl || kv.d != d || c.blocks != 2 * nl || c.d != d || end > dec.max_pos {
                return false;
            }
        }
        // The furthest position of each cache.
        for (i, (kv, _)) in mems.iter().enumerate() {
            let end = mems
                .iter()
                .zip(pos)
                .filter(|(m, _)| std::ptr::eq(m.0, *kv))
                .map(|(_, &p)| p + steps)
                .max()
                .unwrap_or(pos[i] + steps);
            if !kv.reserve(end) {
                return false;
            }
        }
        let ok = self.ensure(B::Dx, t * d * 4)
            && self.ensure(B::Dh, t * d * 4)
            && self.ensure(B::Dq, t * d * 4)
            && self.ensure(B::Dk, t * d * 4)
            && self.ensure(B::Dv, t * d * 4)
            && self.ensure(B::Datt, t * d * 4)
            && self.ensure(B::Df, t * ffn * 4)
            && self.ensure(B::Dlog, nlog.max(1) * v * 4)
            && self.ensure(B::Dpart, t * nh * DEC_CHUNKS * 68 * 4);
        if !ok {
            return false;
        }
        // The constants, made once per decoder.
        let key = dec.ln.w.as_ptr() as usize;
        if self.dconsts.as_ref().is_none_or(|(k, _)| *k != key) {
            let mut consts: Vec<f32> = Vec::new();
            let mut put = |x: &[f32]| -> usize {
                let o = consts.len();
                consts.extend_from_slice(x);
                consts.resize(consts.len().next_multiple_of(64), 0.0);
                o * 4
            };
            let zeros = put(&vec![0f32; d]);
            let mut bias = |l: &Lin| l.bias.map_or(zeros, &mut put);
            let biases: Vec<[usize; 8]> = dec
                .layers
                .iter()
                .map(|l| {
                    [
                        bias(&l.q),
                        bias(&l.k),
                        bias(&l.v),
                        bias(&l.o),
                        bias(&l.cq),
                        bias(&l.co),
                        bias(&l.fc1),
                        bias(&l.fc2),
                    ]
                })
                .collect();
            let layers: Vec<([usize; 8], [usize; 6])> = biases
                .into_iter()
                .zip(&dec.layers)
                .map(|(b, l)| {
                    (
                        b,
                        [
                            put(l.ln1.w),
                            put(l.ln1.b),
                            put(l.ln_cross.w),
                            put(l.ln_cross.b),
                            put(l.ln2.w),
                            put(l.ln2.b),
                        ],
                    )
                })
                .collect();
            let ln = [put(dec.ln.w), put(dec.ln.b)];
            if !self.ensure(B::DConsts, consts.len() * 4) {
                return false;
            }
            // SAFETY: sized above; the device is idle.
            unsafe { par_copy(self.host(B::DConsts), consts.as_ptr(), consts.len()) };
            self.dconsts = Some((key, DecConsts { layers, ln }));
        }
        true
    }

    /// One decoder pass over the rows in Dx, the logits of rows `want` into Dlog.
    fn steps(&self, dec: &Decoder, mems: &[(&WhMem, &WhMem)], pos: &[usize], steps: usize, want: Option<&[usize]>) -> bool {
        self.run_steps(dec, mems, pos, steps, want, None)
    }

    /// `chain.steps` chained greedy steps.
    fn steps_chain(&self, dec: &Decoder, mems: &[(&WhMem, &WhMem)], pos: &[usize], chain: &ChainSpec) -> bool {
        self.run_steps(dec, mems, pos, chain.steps, None, Some(chain))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_steps(
        &self,
        dec: &Decoder,
        mems: &[(&WhMem, &WhMem)],
        pos: &[usize],
        steps: usize,
        want: Option<&[usize]>,
        chain: Option<&ChainSpec>,
    ) -> bool {
        let be = self.be;
        let (d, nh, v, nl) = (dec.d, dec.heads, dec.vocab, dec.layers.len());
        let t = pos.len();
        // The rows' sequences: runs of rows sharing a cache, at consecutive positions.
        let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
        for i in 0..t {
            match runs.last_mut() {
                Some(r) if std::ptr::eq(mems[r.start].0, mems[i].0) && pos[i] == pos[i - 1] + 1 => r.end = i + 1,
                _ => runs.push(i..i + 1),
            }
        }
        let mut ws = Vec::with_capacity(nl * 8 + 1);
        for l in dec
            .layers
            .iter()
            .flat_map(|l| [&l.q, &l.k, &l.v, &l.o, &l.cq, &l.co, &l.fc1, &l.fc2])
            .chain([&dec.head])
        {
            let Some((w, fi, ks)) = dec_mv(l) else {
                return false;
            };
            if let Some(mw) = l.dev.downcast_ref::<MetalWeight>() {
                mw.keep_resident(&be.residency);
            }
            ws.push((w, fi, ks));
        }
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let p = &self.p;
        let dcs = &self.dconsts.as_ref().unwrap().1;
        let cst = id(B::DConsts);
        let scale = (64f32).powf(-0.5);
        let cross_ids: Vec<(Id, usize)> = mems.iter().map(|(_, c)| c.get()).collect();
        let kv_ids: Vec<(Id, usize)> = mems.iter().map(|(k, _)| k.get()).collect();
        // Rows `r0..r0 + n` of x (`n_in` apart) through matvec `wi` into y (`n_out` apart), by
        // `mode`: 0 store, 1 residual add then bias `bias`, 2 GELU of the sum with the bias.
        let mv = |e: &DEnc, wi: usize, x: Id, r0: usize, n: usize, y: Id, y0: usize, mode: u32, bias: usize| {
            if (wi == nl * 8 && skip_op("dhead")) || (wi < nl * 8 && skip_op("dmv")) {
                return;
            }
            let (w, fi, ks) = &ws[wi];
            let thr = mv_threads(*ks);
            let rpt = thr / 32 / ks;
            for g in (0..n).step_by(4) {
                let k = (n - g).min(4);
                e.pipe(p.mv[*fi]);
                e.buf(w.data, 0, 0);
                e.buf(w.scales, 0, 1);
                e.buf(x, (r0 + g) * w.n_in * 4, 2);
                e.buf(y, (y0 + g) * w.n_out * 4, 3);
                e.bytes(
                    &MvParams {
                        n_out: w.n_out as u32,
                        n_in: w.n_in as u32,
                        mode,
                        rows_per_tg: rpt as u32,
                        w_off: w.data_off,
                        s_off: w.scales_off,
                        w2_off: w.data_off,
                        s2_off: w.scales_off,
                        o_off: w.offs_off,
                        o2_off: w.offs_off,
                        gpr: w.gpr,
                        gk: w.gk,
                        ldy: k as u32,
                        _p: 0,
                    },
                    4,
                );
                e.bytes(
                    &U8([w.n_out as u32, w.n_in as u32, mode, *ks as u32, k as u32, thr as u32, 0, 0]),
                    5,
                );
                e.buf(w.offs, 0, 6);
                e.buf(cst, bias, 7);
                e.dispatch((w.n_out.div_ceil(rpt), 1), thr);
            }
        };
        // The LayerNorms of Dx's rows into Dh.
        let norm = |e: &DEnc, ln: [usize; 2]| {
            if skip_op("dln") {
                return;
            }
            e.pipe(p.ln);
            e.buf(id(B::Dx), 0, 0);
            e.buf(cst, ln[0], 1);
            e.buf(cst, ln[1], 2);
            e.buf(id(B::Dh), 0, 3);
            e.bytes(&U4([d as u32, dec.eps.to_bits(), 0, 0]), 4);
            e.dispatch((t, 1), 256);
            e.barrier();
        };
        // Attention of row i's query (Dq + bias) over `n` rows of keys and values: straight into
        // Datt from one chunk, else the chunks into Dpart, then (`merge`, after a barrier) into
        // Datt. With `fuse` (key and value biases), the row's own key and value (position n - 1)
        // come from Dk / Dv and go to the cache there.
        let part =
            |e: &DEnc, pipe: Id, i: usize, qb: usize, kv: (Id, usize, usize), n: usize, elem: usize, fuse: Option<(usize, usize)>| {
                if skip_op(if elem == 4 { "dself" } else { "dcross" }) {
                    return;
                }
                let (buf, k_off, v_off) = kv;
                e.pipe(pipe);
                e.buf(id(B::Dq), i * d * 4, 0);
                e.buf(cst, qb, 1);
                e.buf(buf, k_off * elem, 2);
                e.buf(buf, v_off * elem, 3);
                e.buf(id(B::Dpart), i * nh * DEC_CHUNKS * 68 * 4, 4);
                let chunks = n.div_ceil(DEC_CHUNK);
                e.bytes(
                    &U8([n as u32, d as u32, scale.to_bits(), chunks as u32, fuse.is_some() as u32, 0, 0, 0]),
                    5,
                );
                if let Some((bk, bv)) = fuse {
                    e.buf(id(B::Dk), i * d * 4, 6);
                    e.buf(id(B::Dv), i * d * 4, 7);
                    e.buf(cst, bk, 8);
                    e.buf(cst, bv, 9);
                } else if elem == 4 {
                    for j in 6..10 {
                        e.buf(id(B::Dq), 0, j);
                    }
                }
                e.buf(id(B::Datt), i * d * 4, 10);
                e.dispatch((chunks, nh), DEC_CHUNK);
            };
        let merge = |e: &DEnc, i: usize, n: usize| {
            if skip_op("dmerge") {
                return;
            }
            e.pipe(p.merge);
            e.buf(id(B::Dpart), i * nh * DEC_CHUNKS * 68 * 4, 0);
            e.buf(id(B::Datt), i * d * 4, 1);
            e.bytes(&U4([n as u32, d as u32, 0, n.div_ceil(DEC_CHUNK) as u32]), 2);
            e.dispatch((nh, 1), 64);
        };
        let all: Vec<usize> = (0..t).collect();
        // A chain's steps after the first dispatch indirectly, from sizes a gate at the start of
        // the step copies from the ones written here, or zeros once every row has ended.
        let step_cap = step_cap(nl, t);
        let args = chain
            .filter(|_| !skip_op("dstop"))
            .map(|_| (id(B::Cargs), self.host(B::Cargs) as *mut u32));
        // One command buffer per step of a chain (the GPU starts on a step while the next ones
        // are encoded).
        let parts: Vec<std::ops::Range<usize>> = match chain {
            Some(_) => (0..steps).map(|s| s..s + 1).collect(),
            None => std::iter::once(0..steps).collect(),
        };
        let t0 = std::time::Instant::now();
        let (ok, gpu) = run_concurrent(be, &self.f, &parts, |e0, r| {
            let mut e = DEnc {
                e: e0,
                nobar: skip_op("dbar"),
                ind: None,
                next: std::cell::Cell::new(0),
                end: 0,
            };
            for step in r {
                if let Some(c) = chain {
                    if let Some((abuf, host)) = args.filter(|_| step > 0) {
                        // The gate: this step's sizes into their live copies.
                        let tmpl = 2 * step * step_cap;
                        e.pipe(p.gate);
                        e.buf(id(B::Cstate), 0, 0);
                        e.buf(abuf, 0, 1);
                        e.bytes(&U4([t as u32, c.g.eot, tmpl as u32, step_cap as u32]), 2);
                        e.e.dispatch((1, 1), 256);
                        e.barrier();
                        // SAFETY: Cargs holds 2 * steps * step_cap u32s.
                        e.ind = Some((abuf, unsafe { host.add(tmpl) }, tmpl + step_cap));
                        e.next.set(0);
                        e.end = step_cap;
                    }
                    // The rows' inputs from their last tokens.
                    let rp: Vec<f32> = pos.iter().map(|&x| f32::from_bits((x + step) as u32)).collect();
                    e.pipe(p.embed);
                    e.buf(ws[nl * 8].0.data, ws[nl * 8].0.data_off as usize, 0);
                    e.buf(id(B::Cpos), 0, 1);
                    e.buf(id(B::Cstate), 0, 2);
                    e.buf(id(B::Dx), 0, 3);
                    e.bytes(&U4([d as u32, c.g.embed_scale.to_bits(), 0, 0]), 4);
                    e.bytes_slice(&rp, 5);
                    e.dispatch((d.div_ceil(256), t), 256);
                    e.barrier();
                }
                let at = |i: usize| pos[i] + step;
                for (li, (lb, ln)) in dcs.layers.iter().enumerate() {
                    let w0 = li * 8;
                    // Self-attention.
                    let ln1 = [ln[0], ln[1]];
                    for _ in 0..reps("qkv") {
                        norm(&e, ln1);
                        mv(&e, w0, id(B::Dh), 0, t, id(B::Dq), 0, 0, 0);
                        mv(&e, w0 + 1, id(B::Dh), 0, t, id(B::Dk), 0, 0, 0);
                        mv(&e, w0 + 2, id(B::Dh), 0, t, id(B::Dv), 0, 0, 0);
                        e.barrier();
                    }
                    // A row alone in its sequence stores its key and value in the attention.
                    let fused = runs.iter().all(|r| r.len() == 1);
                    if !fused {
                        for r in &runs {
                            let (kid, cap) = kv_ids[r.start];
                            let p0 = at(r.start);
                            for (j, (src, b)) in [(B::Dk, lb[1]), (B::Dv, lb[2])].into_iter().enumerate() {
                                e.pipe(p.bias);
                                e.buf(id(src), r.start * d * 4, 0);
                                e.buf(cst, b, 1);
                                e.buf(kid, ((2 * li + j) * cap + p0) * d * 4, 2);
                                e.bytes(&U4([d as u32, 1, 0, 0]), 3);
                                e.dispatch(((d / 4).div_ceil(256), r.len()), 256);
                            }
                        }
                        e.barrier();
                    }
                    for _ in 0..reps("self") {
                        for i in 0..t {
                            let (kid, cap) = kv_ids[i];
                            let kv = (kid, 2 * li * cap * d, (2 * li + 1) * cap * d);
                            part(&e, p.part_f32, i, lb[0], kv, at(i) + 1, 4, fused.then_some((lb[1], lb[2])));
                        }
                        e.barrier();
                    }
                    if (0..t).any(|i| at(i) + 1 > DEC_CHUNK) {
                        for i in (0..t).filter(|&i| at(i) + 1 > DEC_CHUNK) {
                            merge(&e, i, at(i) + 1);
                        }
                        e.barrier();
                    }
                    for _ in 0..reps("o") {
                        mv(&e, w0 + 3, id(B::Datt), 0, t, id(B::Dx), 0, 1, lb[3]);
                        e.barrier();
                    }
                    // Cross-attention.
                    for _ in 0..reps("cq") {
                        norm(&e, [ln[2], ln[3]]);
                        mv(&e, w0 + 4, id(B::Dh), 0, t, id(B::Dq), 0, 0, 0);
                        e.barrier();
                    }
                    for _ in 0..reps("cross") {
                        for (i, &(cid, n)) in cross_ids.iter().enumerate() {
                            part(&e, p.part_h, i, lb[4], (cid, 2 * li * n * d, (2 * li + 1) * n * d), n, 2, None);
                        }
                        e.barrier();
                    }
                    for _ in 0..reps("merge") {
                        for (i, &(_, n)) in cross_ids.iter().enumerate() {
                            merge(&e, i, n);
                        }
                        e.barrier();
                    }
                    for _ in 0..reps("co") {
                        mv(&e, w0 + 5, id(B::Datt), 0, t, id(B::Dx), 0, 1, lb[5]);
                        e.barrier();
                    }
                    // Feed-forward.
                    for _ in 0..reps("fc1") {
                        norm(&e, [ln[4], ln[5]]);
                        mv(&e, w0 + 6, id(B::Dh), 0, t, id(B::Df), 0, 2, lb[6]);
                        e.barrier();
                    }
                    for _ in 0..reps("fc2") {
                        mv(&e, w0 + 7, id(B::Df), 0, t, id(B::Dx), 0, 1, lb[7]);
                        e.barrier();
                    }
                }
                // The logits of the rows asked for (every row in a chain), consecutive ones
                // together, after the final LayerNorm.
                norm(&e, dcs.ln);
                let want = want.unwrap_or(&all[..]);
                let mut k = 0;
                while k < want.len() {
                    let mut n = 1;
                    while k + n < want.len() && want[k + n] == want[k] + n {
                        n += 1;
                    }
                    mv(&e, nl * 8, id(B::Dh), want[k], n, id(B::Dlog), k, 0, 0);
                    k += n;
                }
                if let Some(c) = chain {
                    e.barrier();
                    let g = c.g;
                    e.pipe(p.select);
                    e.buf(id(B::Dlog), 0, 0);
                    e.buf(id(B::Cmask), 0, 1);
                    e.buf(id(B::Cstate), 0, 2);
                    e.buf(id(B::Cout), 0, 3);
                    e.bytes(
                        &U8([
                            v as u32,
                            g.timestamp_begin,
                            g.eot,
                            g.no_timestamps,
                            g.max_initial.unwrap_or(u32::MAX),
                            step as u32,
                            steps as u32,
                            0,
                        ]),
                        4,
                    );
                    e.dispatch((t, 1), 1024);
                    e.barrier();
                }
            }
        });
        if prof_on() {
            use std::sync::atomic::Ordering::Relaxed;
            PROF[0].fetch_add(1, Relaxed);
            PROF[1].fetch_add(steps as u64, Relaxed);
            PROF[2].fetch_add((gpu * 1e9) as u64, Relaxed);
            PROF[3].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        }
        ok
    }
}

/// [`gemm`] without the barrier after it (GEMMs of one input into different outputs).
#[allow(clippy::too_many_arguments)]
fn gemm_nb(e: &FastEnc, be: &MetalBackend, l: &Lin, xh: Id, rs: Id, rows: usize, y: Id, ldy: usize, add: bool) {
    if skip_op("gemm") {
        return;
    }
    let mw = l.dev.downcast_ref::<MetalWeight>().unwrap();
    let (pipe, tn) = be.mma_pipe(&l.w, mw.offsets.is_some(), l.n_out, l.n_in, Tile::Dense).unwrap();
    let (_, elem_bytes, scale_bytes) = format_index(&l.w);
    let s_off = match &l.w {
        WeightRef::Gguf(m) => gguf_s_off(m, mw),
        _ => mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0),
    };
    e.pipe(pipe);
    e.buf(mw.data.id, 0, 0);
    e.buf(mw.scales.as_ref().map(|s| s.id).unwrap_or(mw.data.id), 0, 1);
    e.buf(xh, 0, 2);
    e.buf(y, 0, 3);
    e.bytes(
        &MmaParams {
            n_out: l.n_out as u32,
            n_in: l.n_in as u32,
            w_off: (mw.data.offset_bytes / elem_bytes) as u32,
            s_off: s_off as u32,
            rows: rows as u32,
            // Rows as wide as the output and no add: the tensor store (faster).
            _p: [if ldy == l.n_out && !add { 0 } else { ldy as u32 }, 0, add as u32],
        },
        4,
    );
    e.buf(rs, 0, 5);
    if let Some((ob, gpr, gk)) = &mw.offsets {
        e.buf(ob.id, 0, 6);
        e.bytes(
            &OffsetParams {
                o_off: (ob.offset_bytes / 2) as u32,
                gpr: *gpr,
                gk: *gk,
                _p: 0,
            },
            7,
        );
    }
    e.dispatch((l.n_out / tn, rows / MMA_TM), 128);
}
