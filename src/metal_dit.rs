//! Whole diffusion-transformer passes on Metal ([`crate::gpu::dit`]): every block of every
//! sequence of a batch in a few command buffers, the activations in device buffers that live as
//! long as the session. The matmuls are the matrix-unit GEMMs (`MMA_SHADER_SRC`, half inputs
//! with a power-of-two scale per row), attention the fused `flash_attn128_mma` kernels, the row
//! work the `dit_*` kernels of `metal_dit.metal`. Without matrix units there is no session (the
//! per-op path runs instead).

use super::decode::{FastEnc, Fns, run_concurrent};
use super::*;
use crate::gpu::dit::{FluxInput, FluxModel, FluxSession, Lin, Model, Seq, Session};

#[path = "metal_diffvae.rs"]
mod diffvae;
#[path = "metal_ltx.rs"]
mod ltx;
#[path = "metal_t5.rs"]
mod t5;

pub(crate) const SRC: &str = include_str!("metal_dit.metal");
/// Kernels on the matrix units (appended after `MMA_SHADER_SRC`).
pub(crate) const MMA_SRC: &str = include_str!("metal_dit_mma.metal");
/// LTX-2.5's row kernels (appended after `SRC`).
pub(crate) const LTX_SRC: &str = include_str!("metal_ltx.metal");

/// Blocks per command buffer (the GPU starts on the first while the rest is encoded).
const BLOCKS_PER_CB: usize = 6;

/// A layer whose LoRA side path runs: (layer, input, rows, output, output row stride).
type SideJob<'l, 'a> = (&'l Lin<'a>, Xh, usize, (Id, usize), usize);

/// LoRA side paths a FLUX / Qwen-Image pass runs side by side (a double block's q, k, v of both
/// streams).
const SIDE_SLOTS: usize = 6;

/// Profiling (`OXINFER_DIT_SKIP=qkv,att,o,up,down`): ops left out of the passes.
fn skip_op(op: &str) -> bool {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("OXINFER_DIT_SKIP").unwrap_or_default())
        .split(',')
        .any(|x| x == op)
}

/// A device buffer that grows on demand (released with the session).
struct DBuf {
    id: Id,
    cap: usize,
}

struct Pipes {
    rms: Id,
    situ: Id,
    heads: Id,
    gated: Id,
    bias: Id,
    rot: Id,
}

pub(crate) struct MetalDit {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    bufs: Vec<DBuf>,
}

// SAFETY: the session is used by one thread at a time (`&mut self`); the Objective-C objects
// it holds are thread-safe Metal buffers and pipelines.
unsafe impl Send for MetalDit {}

/// Buffer slots of a session.
#[derive(Clone, Copy)]
enum B {
    H = 0,
    Xh,
    Rs,
    Qkv,
    Q,
    K,
    V,
    Att,
    Tmp,
    Hg,
    U,
    Uh,
    Urs,
    Consts,
    Cos,
    Sin,
    Xin,
    Xf,
}
const NBUF: usize = 18;

#[repr(C)]
struct Drm {
    n: u32,
    ld: u32,
    has_mod: u32,
    out_half: u32,
    eps: f32,
    _p: [u32; 3],
}
#[repr(C)]
struct Dhd {
    ld_src: u32,
    norm: u32,
    rope: u32,
    out_half: u32,
    eps: f32,
    heads: u32,
    _p: [u32; 2],
}
#[repr(C)]
struct U4([u32; 4]);

impl MetalBackend {
    /// [`crate::gpu::Backend::dit_session`].
    pub(crate) fn dit_session_impl(&'static self) -> Option<Box<dyn Session>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            rms: pipe("dit_rms_mod")?,
            situ: pipe("dit_situ")?,
            heads: pipe("dit_heads")?,
            gated: pipe("dit_gated_add")?,
            bias: pipe("dit_bias_rows")?,
            rot: pipe("dit_rot_half")?,
        };
        Some(Box::new(MetalDit {
            be: self,
            f: Fns::new(self),
            p,
            bufs: (0..NBUF)
                .map(|_| DBuf {
                    id: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
        }))
    }
}

impl Drop for MetalDit {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

/// Where a GEMM's input comes from: scaled half rows and their inverse scales.
#[derive(Clone, Copy)]
struct Xh {
    xh: (Id, usize),
    rs: (Id, usize),
}

impl MetalDit {
    /// Makes buffer `b` hold at least `bytes` (new buffers are zero-filled; contents are lost when
    /// one grows).
    fn ensure(&mut self, b: B, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn id(&self, b: B) -> Id {
        self.bufs[b as usize].id
    }

    fn host(&self, b: B) -> *mut f32 {
        self.f.contents(self.id(b)) as *mut f32
    }
}

/// Makes `s` hold at least `bytes` (new buffers are zero-filled; contents are lost when one
/// grows).
fn grow(be: &MetalBackend, s: &mut DBuf, bytes: usize) -> bool {
    if s.cap >= bytes && !s.id.is_null() {
        return true;
    }
    let new_len: MsgNewBufLen = be.o.msg_send();
    let void: MsgVoid = be.o.msg_send();
    let cap = bytes.max(256).div_ceil(PAGE) * PAGE;
    let id = unsafe { new_len(be.device, be.sels.new_buffer_len, cap, 0) };
    if id.is_null() {
        return false;
    }
    if !s.id.is_null() {
        unsafe { void(s.id, be.sels.release) };
    }
    s.id = id;
    s.cap = cap;
    true
}

fn release_all(be: &MetalBackend, bufs: &[DBuf]) {
    let void: MsgVoid = be.o.msg_send();
    for b in bufs {
        if !b.id.is_null() {
            unsafe { void(b.id, be.sels.release) };
        }
    }
}

/// Whether the device can run layer `l` (a matrix-unit GEMM for it and its side path).
fn runnable(be: &MetalBackend, l: &Lin) -> bool {
    let Some(mw) = l.dev.downcast_ref::<MetalWeight>() else {
        return false;
    };
    if be.mma_pipe(&l.w, mw.offsets.is_some(), l.n_out, l.n_in, Tile::Dense).is_none() {
        return false;
    }
    if l.rot.is_some_and(|g| g != 256 || !l.n_in.is_multiple_of(256)) {
        return false;
    }
    match &l.side {
        Some((d, u)) => d.side.is_none() && u.side.is_none() && d.rot.is_none() && u.rot.is_none() && runnable(be, d) && runnable(be, u),
        None => true,
    }
}

struct Enc<'e, 'f> {
    e: &'e FastEnc<'f>,
    be: &'static MetalBackend,
}

impl Enc<'_, '_> {
    /// `y = W x` (or `y +=` with `add`) over `rows` rows (a multiple of 128) into rows `ldy`
    /// floats apart.
    fn gemm(&self, l: &Lin, x: Xh, rows: usize, y: (Id, usize), ldy: usize, add: bool) {
        self.gemm_nb(l, x, rows, y, ldy, add);
        self.e.barrier();
    }

    /// [`Self::gemm`] without the barrier after it (GEMMs that may run together).
    fn gemm_nb(&self, l: &Lin, x: Xh, rows: usize, y: (Id, usize), ldy: usize, add: bool) {
        let mw = l.dev.downcast_ref::<MetalWeight>().unwrap();
        let (pipe, tn) = self.be.mma_pipe(&l.w, mw.offsets.is_some(), l.n_out, l.n_in, Tile::Dense).unwrap();
        let (_, elem_bytes, scale_bytes) = format_index(&l.w);
        let s_off = match &l.w {
            WeightRef::Gguf(m) => gguf_s_off(m, mw),
            _ => mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0),
        };
        let e = self.e;
        e.pipe(pipe);
        e.buf(mw.data.id, 0, 0);
        e.buf(mw.scales.as_ref().map(|s| s.id).unwrap_or(mw.data.id), 0, 1);
        e.buf(x.xh.0, x.xh.1, 2);
        e.buf(y.0, y.1, 3);
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
        e.buf(x.rs.0, x.rs.1, 5);
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

    /// f32 rows (`ld` apart) to scaled half rows.
    fn to_half(&self, x: (Id, usize), ld: usize, n: usize, rows: usize, out: Xh) {
        self.to_half_nb(x, ld, n, rows, out);
        self.e.barrier();
    }

    /// [`Self::to_half`] without the barrier after it.
    fn to_half_nb(&self, x: (Id, usize), ld: usize, n: usize, rows: usize, out: Xh) {
        let m = self.be.mma.as_ref().unwrap();
        let e = self.e;
        e.pipe(m.x_half);
        e.buf(x.0, x.1, 0);
        e.buf(out.xh.0, out.xh.1, 1);
        e.buf(out.rs.0, out.rs.1, 2);
        e.bytes(&U4([n as u32, 0, ld as u32, 0]), 3);
        e.buf(x.0, x.1, 4);
        e.dispatch((rows, 1), 256);
    }
}

impl Session for MetalDit {
    fn forward(&mut self, m: &Model, x: &[f32], image: usize, seqs: &[Seq], modv: &[f32], head: &[f32], out: &mut [Vec<f32>]) -> bool {
        let be = self.be;
        let (d, f, hd, heads, lat) = (m.dim, m.ffn, m.head_dim, m.heads, m.latent);
        if hd != 128 || heads * hd != d || !d.is_multiple_of(256) || !f.is_multiple_of(256) || seqs.is_empty() || out.len() != seqs.len() {
            return false;
        }
        if x.len() != image * lat || modv.len() != 6 * d || head.len() != 2 * d {
            return false;
        }
        let lins = m
            .blocks
            .iter()
            .flat_map(|b| [&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down])
            .chain([&m.video, &m.final_linear]);
        let mut rmax = 0;
        for l in lins {
            if !runnable(be, l) {
                return false;
            }
            if let Some((dn, _)) = &l.side {
                rmax = rmax.max(dn.n_out);
            }
        }
        let blk = &m.blocks;
        if blk
            .iter()
            .any(|b| b.q.bias.is_some() || b.k.bias.is_some() || b.v.bias.is_some() || b.o.bias.is_some())
            || blk.iter().any(|b| b.up_gate.bias.is_some() || b.down.bias.is_some())
            || blk.iter().any(|b| b.up_gate.n_out != 2 * f || b.down.n_in != f || b.q.n_out != d)
        {
            return false;
        }
        // Rows: each sequence (image then text) padded to 64; the total to 128 more rows than the
        // last whole tile (GEMM tiles of the final head read past the image rows).
        let mut r0 = Vec::with_capacity(seqs.len());
        let mut rows = 0;
        for s in seqs {
            let n = image + s.text_len;
            if s.text.len() != s.text_len * d || s.cos.len() != n * hd / 2 || s.sin.len() != n * hd / 2 {
                return false;
            }
            r0.push(rows);
            rows += n.next_multiple_of(64);
        }
        let rg = rows.next_multiple_of(MMA_TM) + MMA_TM;
        let ri = image.next_multiple_of(MMA_TM);
        let wide = (2 * f).max(3 * d);
        let ok = self.ensure(B::H, rg * d * 4)
            && self.ensure(B::Xh, rg * f.max(d) * 2)
            && self.ensure(B::Rs, rg * 4)
            && self.ensure(B::Qkv, rg * 3 * d * 4)
            && self.ensure(B::Q, rg * d * 4)
            && self.ensure(B::K, rg * d * 4)
            && self.ensure(B::V, rg * d * 4)
            && self.ensure(B::Att, rg * d * 4)
            && self.ensure(B::Tmp, rg * d.max(lat) * 4)
            && self.ensure(B::Hg, rg * wide * 4)
            && self.ensure(B::U, rg * rmax.max(64) * 4)
            && self.ensure(B::Uh, rg * rmax.max(64) * 2)
            && self.ensure(B::Urs, rg * 4)
            && self.ensure(B::Cos, rg * hd / 2 * 4)
            && self.ensure(B::Sin, rg * hd / 2 * 4)
            && self.ensure(B::Xin, ri * lat * 4)
            && self.ensure(B::Xf, rg * f.max(d) * 4);
        if !ok {
            return false;
        }
        // Constants: the modulations, the norm weights, the biases.
        let mut consts: Vec<f32> = Vec::with_capacity(8 * d + m.blocks.len() * (2 * d + 2 * hd) + 4 * d);
        let mut put = |v: &[f32]| -> usize {
            let o = consts.len();
            consts.extend_from_slice(v);
            consts.resize(consts.len().next_multiple_of(64), 0.0);
            o * 4
        };
        let mo: Vec<usize> = modv.chunks(d).map(&mut put).collect();
        let (hshift, hscale) = (put(&head[..d]), put(&head[d..]));
        let fnorm = put(m.final_norm);
        let vbias = m.video.bias.map(&mut put);
        let norms: Vec<[usize; 4]> = m
            .blocks
            .iter()
            .map(|b| [put(b.norm1), put(b.norm2), put(b.q_norm), put(b.k_norm)])
            .collect();
        if !self.ensure(B::Consts, consts.len() * 4) {
            return false;
        }
        // SAFETY: every buffer was sized above; the device is idle (the last pass was waited for).
        unsafe {
            par_copy(self.host(B::Consts), consts.as_ptr(), consts.len());
            let xin = self.host(B::Xin);
            par_copy(xin, x.as_ptr(), image * lat);
            std::ptr::write_bytes(xin.add(image * lat), 0, (ri - image) * lat);
            let (h, cs, sn) = (self.host(B::H), self.host(B::Cos), self.host(B::Sin));
            for (s, &r) in seqs.iter().zip(&r0) {
                let n = image + s.text_len;
                par_copy(h.add((r + image) * d), s.text.as_ptr(), s.text_len * d);
                std::ptr::write_bytes(h.add((r + n) * d), 0, (n.next_multiple_of(64) - n) * d);
                par_copy(cs.add(r * hd / 2), s.cos.as_ptr(), n * hd / 2);
                par_copy(sn.add(r * hd / 2), s.sin.as_ptr(), n * hd / 2);
            }
            std::ptr::write_bytes(h.add(rows * d), 0, (rg - rows) * d);
        }
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let at = |b: B, bytes: usize| (id(b), bytes);
        let main_x = Xh {
            xh: at(B::Xh, 0),
            rs: at(B::Rs, 0),
        };
        let p = &self.p;
        let mma = be.mma.as_ref().unwrap();
        let eps = m.eps;
        let cst = id(B::Consts);
        // The input of layer l from f32 rows `src` (`ld` apart, n wide): rotated first when l is
        // stored rotated, else converted (or, when `fused` already wrote main_x, used as is).
        let input = |en: &Enc, l: &Lin, src: (Id, usize), ld: usize, n: usize, fused: bool| -> Xh {
            if l.rot.is_some() {
                // Rotation works in place on a contiguous copy.
                let e = en.e;
                e.pipe(p.bias);
                e.buf(src.0, src.1, 0);
                e.buf(id(B::Xf), 0, 1);
                e.buf(src.0, src.1, 2);
                e.bytes(&U4([n as u32, ld as u32, n as u32, 0]), 3);
                e.dispatch((n.div_ceil(256), rg), 256);
                e.barrier();
                e.pipe(p.rot);
                e.buf(id(B::Xf), 0, 0);
                e.buf(id(B::Xh), 0, 1);
                e.buf(id(B::Rs), 0, 2);
                e.bytes(&U4([n as u32, 0, 0, 0]), 3);
                e.dispatch((rg, 1), 256);
                e.barrier();
            } else if !fused {
                en.to_half(src, ld, n, rg, main_x);
            }
            main_x
        };
        // y = W x (+ the side path), x from `input`.
        let linear = |en: &Enc, l: &Lin, x: Xh, y: (Id, usize), ldy: usize| {
            en.gemm(l, x, rg, y, ldy, false);
            if let Some((dn, up)) = &l.side {
                let r = dn.n_out;
                en.gemm(dn, x, rg, at(B::U, 0), r, false);
                en.to_half(
                    at(B::U, 0),
                    r,
                    r,
                    rg,
                    Xh {
                        xh: at(B::Uh, 0),
                        rs: at(B::Urs, 0),
                    },
                );
                en.gemm(
                    up,
                    Xh {
                        xh: at(B::Uh, 0),
                        rs: at(B::Urs, 0),
                    },
                    rg,
                    y,
                    ldy,
                    true,
                );
            }
        };
        let rms = |en: &Enc, w: usize, sc_sh: Option<(usize, usize)>, any_rot: bool| {
            let e = en.e;
            e.pipe(p.rms);
            e.buf(id(B::H), 0, 0);
            e.buf(cst, w, 1);
            let (sc, sh) = sc_sh.unwrap_or((w, w));
            e.buf(cst, sc, 2);
            e.buf(cst, sh, 3);
            e.buf(if any_rot { id(B::Xf) } else { id(B::Xh) }, 0, 4);
            e.buf(id(B::Rs), 0, 5);
            e.bytes(
                &Drm {
                    n: d as u32,
                    ld: d as u32,
                    has_mod: sc_sh.is_some() as u32,
                    out_half: !any_rot as u32,
                    eps,
                    _p: [0; 3],
                },
                6,
            );
            e.dispatch((rg, 1), 256);
            e.barrier();
        };
        let gated = |en: &Enc, gate: usize, active: &[(usize, usize)]| {
            let e = en.e;
            for &(r, n) in active {
                e.pipe(p.gated);
                e.buf(id(B::H), r * d * 4, 0);
                e.buf(id(B::Tmp), r * d * 4, 1);
                e.buf(cst, gate, 2);
                e.bytes(&U4([d as u32, d as u32, 0, 0]), 3);
                e.dispatch(((d / 4).div_ceil(256), n), 256);
            }
            e.barrier();
        };
        let half_att = m.half_attention;
        let eb = if half_att { 2 } else { 4 };
        let heads_op = |en: &Enc, col: usize, dst: B, norm: Option<usize>, rope: bool| {
            let e = en.e;
            e.pipe(p.heads);
            e.buf(id(B::Qkv), col * rg * 4, 0);
            e.buf(id(dst), 0, 1);
            e.buf(cst, norm.unwrap_or(0), 2);
            e.buf(id(B::Cos), 0, 3);
            e.buf(id(B::Sin), 0, 4);
            e.bytes(
                &Dhd {
                    ld_src: d as u32,
                    norm: norm.is_some() as u32,
                    rope: rope as u32,
                    out_half: half_att as u32,
                    eps,
                    heads: heads as u32,
                    _p: [0; 2],
                },
                5,
            );
            e.dispatch((rows, heads), 32);
        };
        let nb = m.blocks.len();
        let parts: Vec<std::ops::Range<usize>> = (0..nb.div_ceil(BLOCKS_PER_CB).max(1))
            .map(|c| c * BLOCKS_PER_CB..((c + 1) * BLOCKS_PER_CB).min(nb))
            .collect();
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            let en = Enc { e, be };
            if range.start == 0 {
                // Image tokens: the latent embedding into each sequence's first rows.
                en.to_half(at(B::Xin, 0), lat, lat, ri, main_x);
                en.gemm(&m.video, main_x, ri, at(B::Tmp, 0), d, false);
                for &r in &r0 {
                    e.pipe(p.bias);
                    e.buf(id(B::Tmp), 0, 0);
                    e.buf(id(B::H), r * d * 4, 1);
                    e.buf(cst, vbias.unwrap_or(0), 2);
                    e.bytes(&U4([d as u32, d as u32, d as u32, vbias.is_some() as u32]), 3);
                    e.dispatch((d.div_ceil(256), image), 256);
                }
                e.barrier();
            }
            for i in range.clone() {
                let b = &m.blocks[i];
                let active: Vec<(usize, usize)> = seqs
                    .iter()
                    .zip(&r0)
                    .filter(|(s, _)| !s.skip.contains(&i))
                    .map(|(s, &r)| (r, image + s.text_len))
                    .collect();
                if active.is_empty() {
                    continue;
                }
                let [n1, n2, qn, kn] = norms[i];
                // Attention.
                let rot = b.q.rot.is_some();
                rms(&en, n1, Some((mo[1], mo[0])), rot);
                let xh = input(&en, &b.q, at(B::Xf, 0), d, d, !rot);
                for (j, l) in [&b.q, &b.k, &b.v].into_iter().enumerate() {
                    if !skip_op("qkv") {
                        linear(&en, l, xh, at(B::Qkv, j * rg * d * 4), d);
                    }
                }
                heads_op(&en, 0, B::Q, Some(qn), true);
                heads_op(&en, d, B::K, Some(kn), true);
                heads_op(&en, 2 * d, B::V, None, false);
                e.barrier();
                let fpipe = if half_att { mma.flash128 } else { mma.flash128_f32 };
                for &(r, n) in &active {
                    if skip_op("att") {
                        continue;
                    }
                    e.pipe(fpipe);
                    e.buf(id(B::Q), r * d * eb, 0);
                    e.buf(id(B::K), r * d * eb, 1);
                    e.buf(id(B::V), r * d * eb, 2);
                    e.buf(id(B::Att), r * d * 4, 3);
                    e.bytes(&U4([n as u32, n as u32, d as u32, (1.0 / (hd as f32).sqrt()).to_bits()]), 4);
                    e.dispatch((n.div_ceil(64), heads), 128);
                }
                e.barrier();
                let xh = input(&en, &b.o, at(B::Att, 0), d, d, false);
                if !skip_op("o") {
                    linear(&en, &b.o, xh, at(B::Tmp, 0), d);
                }
                gated(&en, mo[2], &active);
                // Feed-forward.
                let rot = b.up_gate.rot.is_some();
                rms(&en, n2, Some((mo[4], mo[3])), rot);
                let xh = input(&en, &b.up_gate, at(B::Xf, 0), d, d, !rot);
                if !skip_op("up") {
                    linear(&en, &b.up_gate, xh, at(B::Hg, 0), 2 * f);
                }
                let rot = b.down.rot.is_some();
                e.pipe(p.situ);
                e.buf(id(B::Hg), 0, 0);
                e.buf(if rot { id(B::Xf) } else { id(B::Xh) }, 0, 1);
                e.buf(id(B::Rs), 0, 2);
                e.bytes(&U4([f as u32, !rot as u32, 0, 0]), 3);
                e.dispatch((rg, 1), 256);
                e.barrier();
                let xh = input(&en, &b.down, at(B::Xf, 0), f, f, !rot);
                if !skip_op("down") {
                    linear(&en, &b.down, xh, at(B::Tmp, 0), d);
                }
                gated(&en, mo[5], &active);
            }
            if range.end == nb {
                // The head: final norm with the head modulation, the last linear.
                let rot = m.final_linear.rot.is_some();
                e.pipe(p.rms);
                e.buf(id(B::H), 0, 0);
                e.buf(cst, fnorm, 1);
                e.buf(cst, hscale, 2);
                e.buf(cst, hshift, 3);
                e.buf(if rot { id(B::Xf) } else { id(B::Xh) }, 0, 4);
                e.buf(id(B::Rs), 0, 5);
                e.bytes(
                    &Drm {
                        n: d as u32,
                        ld: d as u32,
                        has_mod: 1,
                        out_half: !rot as u32,
                        eps,
                        _p: [0; 3],
                    },
                    6,
                );
                e.dispatch((rg, 1), 256);
                e.barrier();
                let xh = input(&en, &m.final_linear, at(B::Xf, 0), d, d, !rot);
                linear(&en, &m.final_linear, xh, at(B::Tmp, 0), lat);
            }
        });
        if !ok {
            return false;
        }
        let tmp = self.host(B::Tmp) as *const f32;
        for ((o, &r), _) in out.iter_mut().zip(&r0).zip(seqs) {
            o.resize(image * lat, 0.0);
            // SAFETY: Tmp holds rg rows of lat floats.
            unsafe { par_copy(o.as_mut_ptr(), tmp.add(r * lat), image * lat) };
            if let Some(bias) = m.final_linear.bias {
                for row in o.chunks_mut(lat) {
                    for (v, &bb) in row.iter_mut().zip(bias) {
                        *v += bb;
                    }
                }
            }
        }
        true
    }
}

// ---- FLUX.1 ----

struct FluxPipes {
    ln: Id,
    heads: Id,
    gelu: Id,
    cat: Id,
    gated: Id,
    bias: Id,
    /// Split-precision attention (`dit_flash128_hilo`), when the library has it.
    hilo: Option<Id>,
}

/// FLUX.1 and Qwen-Image passes ([`FluxSession`]): the joint sequence `[text; image]` in one
/// set of device buffers, the text rows first, padded to a multiple of 128 so both streams
/// start on a GEMM tile; the attention operands are compact (`flux_heads` skips the padding).
pub(crate) struct MetalFlux {
    be: &'static MetalBackend,
    f: Fns,
    p: FluxPipes,
    bufs: Vec<DBuf>,
}

// SAFETY: as MetalDit.
unsafe impl Send for MetalFlux {}

#[derive(Clone, Copy, PartialEq)]
enum F {
    H = 0,
    Xh,
    Rs,
    Qkv,
    Q,
    K,
    V,
    Att,
    Tmp,
    Hg,
    Consts,
    Cos,
    Sin,
    Img,
    Txt,
    U,
    Uh,
    Urs,
    Ql,
    Kl,
}
const NFBUF: usize = 20;

#[repr(C)]
struct Fln {
    n: u32,
    eps: f32,
    _p: [u32; 2],
}
#[repr(C)]
struct Fhd {
    ld_src: u32,
    norm: u32,
    rope: u32,
    out_half: u32,
    eps: f32,
    heads: u32,
    split: u32,
    has_bias: u32,
    gap: u32,
    _p: [u32; 3],
}

impl MetalBackend {
    /// [`crate::gpu::Backend::flux_session`].
    pub(crate) fn flux_session_impl(&'static self) -> Option<Box<dyn FluxSession>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = FluxPipes {
            ln: pipe("flux_ln_mod")?,
            heads: pipe("flux_heads")?,
            gelu: pipe("flux_gelu_half")?,
            cat: pipe("flux_cat_half")?,
            gated: pipe("flux_gated_add")?,
            bias: pipe("dit_bias_rows")?,
            hilo: pipe("dit_flash128_hilo"),
        };
        Some(Box::new(MetalFlux {
            be: self,
            f: Fns::new(self),
            p,
            bufs: (0..NFBUF)
                .map(|_| DBuf {
                    id: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
        }))
    }
}

impl Drop for MetalFlux {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

impl MetalFlux {
    fn ensure(&mut self, b: F, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn host(&self, b: F) -> *mut f32 {
        self.f.contents(self.bufs[b as usize].id) as *mut f32
    }
}

/// Bias offsets (into the constants) of a double block's layers.
struct DoubleBias {
    iq: [Option<usize>; 3],
    tq: [Option<usize>; 3],
    io: Option<usize>,
    to: Option<usize>,
    im: [Option<usize>; 2],
    tm: [Option<usize>; 2],
}

impl FluxSession for MetalFlux {
    fn forward(&mut self, m: &FluxModel, inp: &FluxInput, out: &mut Vec<f32>) -> bool {
        let be = self.be;
        let (d, heads, hd) = (m.dim, m.heads, m.head_dim);
        let (ni, nx) = (inp.n_img, inp.n_txt);
        let (nd, ns) = (m.doubles.len(), m.singles.len());
        // Text rows padded to a GEMM tile; single blocks see the joint rows as one stream, so
        // they need none.
        let rt = nx.next_multiple_of(MMA_TM);
        let gap = rt - nx;
        if hd != 128 || heads * hd != d || !d.is_multiple_of(256) || ni == 0 || nd + ns == 0 || (ns > 0 && gap != 0) {
            return false;
        }
        let (split, no) = (inp.img_split, inp.n_out);
        if split > ni || no == 0 || no > ni || inp.mods2.len() != if split < ni { 6 * nd * d } else { 0 } {
            return false;
        }
        let n = nx + ni;
        let (cin, cout) = (m.x_emb.n_in, m.proj_out.n_out);
        let cj = m.ctx_emb.as_ref().map_or(d, |l| l.n_in);
        if inp.img.len() != ni * cin
            || inp.txt.len() != nx * cj
            || inp.cos.len() != n * hd / 2
            || inp.sin.len() != n * hd / 2
            || inp.mods.len() != (12 * nd + 3 * ns + 2) * d
        {
            return false;
        }
        let mut lins: Vec<&Lin> = vec![&m.x_emb, &m.proj_out];
        lins.extend(&m.ctx_emb);
        let mut fm = 0;
        for b in &m.doubles {
            lins.extend(b.img_qkv.iter().chain(&b.txt_qkv).chain(&b.img_mlp).chain(&b.txt_mlp));
            lins.extend([&b.img_out, &b.txt_out]);
            let f = b.img_mlp[0].n_out;
            if b.txt_mlp[0].n_out != f || b.img_mlp[1].n_in != f || b.txt_mlp[1].n_in != f {
                return false;
            }
            fm = fm.max(f);
        }
        let mut wide = cj.max(cin).max(fm).max(d);
        for b in &m.singles {
            if b.qkv.iter().chain([&b.mlp, &b.out]).any(|l| l.side.is_some()) {
                return false;
            }
            lins.extend(b.qkv.iter().chain([&b.mlp, &b.out]));
            if b.out.n_in != d + b.mlp.n_out || !b.mlp.n_out.is_multiple_of(4) {
                return false;
            }
            fm = fm.max(b.mlp.n_out);
            wide = wide.max(b.out.n_in);
        }
        if lins.iter().any(|l| l.rot.is_some() || !runnable(be, l))
            || lins.iter().any(|l| l.bias.is_some_and(|b| b.len() != l.n_out))
            || m.doubles.iter().any(|b| {
                b.img_qkv
                    .iter()
                    .chain(&b.txt_qkv)
                    .chain([&b.img_out, &b.txt_out])
                    .any(|l| l.n_out != d)
                    || b.img_mlp[1].n_out != d
                    || b.txt_mlp[1].n_out != d
            })
            || m.singles.iter().any(|b| b.qkv.iter().any(|l| l.n_out != d) || b.out.n_out != d)
            || m.x_emb.n_out != d
            || m.ctx_emb.as_ref().is_some_and(|l| l.n_out != d)
        {
            return false;
        }
        // The widest LoRA rank.
        let rmax = lins
            .iter()
            .filter_map(|l| l.side.as_ref().map(|(dn, _)| dn.n_out))
            .max()
            .unwrap_or(0);
        let ri = ni.next_multiple_of(MMA_TM);
        let rg = rt + ri;
        // Attention operands: 1 half, 2 hi / lo halves, 0 f32.
        let hilo = !m.half_attention && m.split_attention && self.p.hilo.is_some();
        let att_mode = if m.half_attention {
            1
        } else if hilo {
            2
        } else {
            0
        };
        let eb = if att_mode == 0 { 4 } else { 2 };
        let ok = self.ensure(F::H, rg * d * 4)
            && (!hilo || self.ensure(F::Ql, rg * d * 2) && self.ensure(F::Kl, rg * d * 2))
            && self.ensure(F::Xh, rg * wide * 2)
            && self.ensure(F::Rs, rg * 4)
            && self.ensure(F::Qkv, 3 * rg * d * 4)
            && self.ensure(F::Q, rg * d * eb)
            && self.ensure(F::K, rg * d * eb)
            && self.ensure(F::V, rg * d * eb)
            && self.ensure(F::Att, rg * d * 4)
            && self.ensure(F::Tmp, rg * d.max(cout) * 4)
            && self.ensure(F::Hg, rg * fm * 4)
            && self.ensure(F::Cos, rg * hd / 2 * 4)
            && self.ensure(F::Sin, rg * hd / 2 * 4)
            && self.ensure(F::Img, ri * cin * 4)
            && self.ensure(F::Txt, rt * cj * 4)
            && (rmax == 0
                || self.ensure(F::U, SIDE_SLOTS * ri.max(rt) * rmax * 4)
                    && self.ensure(F::Uh, SIDE_SLOTS * ri.max(rt) * rmax * 2)
                    && self.ensure(F::Urs, SIDE_SLOTS * ri.max(rt) * 4));
        if !ok {
            return false;
        }
        // Constants: the modulations, the biases, the norm weights.
        let mut consts: Vec<f32> = Vec::with_capacity(inp.mods.len() + inp.mods2.len() + (nd + ns) * 24 * d);
        let mut put = |v: &[f32]| -> usize {
            let o = consts.len();
            consts.extend_from_slice(v);
            consts.resize(consts.len().next_multiple_of(64), 0.0);
            o * 4
        };
        let mods = put(inp.mods);
        let mods2 = put(inp.mods2);
        let x_b = m.x_emb.bias.map(&mut put);
        let c_b = m.ctx_emb.as_ref().and_then(|l| l.bias).map(&mut put);
        let dbs: Vec<DoubleBias> = m
            .doubles
            .iter()
            .map(|b| {
                let mut bias = |l: &Lin| l.bias.map(&mut put);
                DoubleBias {
                    iq: [bias(&b.img_qkv[0]), bias(&b.img_qkv[1]), bias(&b.img_qkv[2])],
                    tq: [bias(&b.txt_qkv[0]), bias(&b.txt_qkv[1]), bias(&b.txt_qkv[2])],
                    io: bias(&b.img_out),
                    to: bias(&b.txt_out),
                    im: [bias(&b.img_mlp[0]), bias(&b.img_mlp[1])],
                    tm: [bias(&b.txt_mlp[0]), bias(&b.txt_mlp[1])],
                }
            })
            .collect();
        type SingleBias = ([Option<usize>; 3], Option<usize>, Option<usize>);
        let sbs: Vec<SingleBias> = m
            .singles
            .iter()
            .map(|b| {
                let mut bias = |l: &Lin| l.bias.map(&mut put);
                ([bias(&b.qkv[0]), bias(&b.qkv[1]), bias(&b.qkv[2])], bias(&b.mlp), bias(&b.out))
            })
            .collect();
        let dnorms: Vec<[usize; 4]> = m
            .doubles
            .iter()
            .map(|b| [put(b.norms[0]), put(b.norms[1]), put(b.norms[2]), put(b.norms[3])])
            .collect();
        let snorms: Vec<[usize; 2]> = m.singles.iter().map(|b| [put(b.norms[0]), put(b.norms[1])]).collect();
        if !self.ensure(F::Consts, consts.len() * 4) {
            return false;
        }
        // SAFETY: every buffer was sized above; the device is idle (the last pass was waited for).
        unsafe {
            par_copy(self.host(F::Consts), consts.as_ptr(), consts.len());
            let img = self.host(F::Img);
            par_copy(img, inp.img.as_ptr(), ni * cin);
            std::ptr::write_bytes(img.add(ni * cin), 0, (ri - ni) * cin);
            let h = self.host(F::H);
            if m.ctx_emb.is_some() {
                let txt = self.host(F::Txt);
                par_copy(txt, inp.txt.as_ptr(), nx * cj);
                std::ptr::write_bytes(txt.add(nx * cj), 0, gap * cj);
            } else {
                par_copy(h, inp.txt.as_ptr(), nx * d);
            }
            par_copy(self.host(F::Cos), inp.cos.as_ptr(), n * hd / 2);
            par_copy(self.host(F::Sin), inp.sin.as_ptr(), n * hd / 2);
            std::ptr::write_bytes(h.add(nx * d), 0, gap * d);
            std::ptr::write_bytes(h.add((rt + ni) * d), 0, (ri - ni) * d);
        }
        let ids: [Id; NFBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: F| ids[b as usize];
        let at = |b: F, bytes: usize| (id(b), bytes);
        let mo = |k: usize| mods + k * d * 4;
        let mo2 = |k: usize| mods2 + k * d * 4;
        let cst = id(F::Consts);
        let p = &self.p;
        let mma = be.mma.as_ref().unwrap();
        const EPS: f32 = 1e-6;
        // Scaled half rows of width w from row r0 on.
        let xh = |r0: usize, w: usize| Xh {
            xh: at(F::Xh, r0 * w * 2),
            rs: at(F::Rs, r0 * 4),
        };
        // Layer norm + modulation of `rows` rows of H from r0 into xh(r0, d).
        let ln = |e: &FastEnc, r0: usize, rows: usize, scale: usize, shift: usize| {
            if rows == 0 {
                return;
            }
            e.pipe(p.ln);
            e.buf(id(F::H), r0 * d * 4, 0);
            e.buf(cst, scale, 1);
            e.buf(cst, shift, 2);
            e.buf(id(F::Xh), r0 * d * 2, 3);
            e.buf(id(F::Rs), r0 * 4, 4);
            e.bytes(
                &Fln {
                    n: d as u32,
                    eps: EPS,
                    _p: [0; 2],
                },
                5,
            );
            e.dispatch((rows, 1), 256);
        };
        // H rows from r0 += gate * (y + bias), y rows of d in Tmp at the same rows.
        let gated = |e: &FastEnc, r0: usize, rows: usize, gate: usize, b: Option<usize>| {
            if rows == 0 {
                return;
            }
            e.pipe(p.gated);
            e.buf(id(F::H), r0 * d * 4, 0);
            e.buf(id(F::Tmp), r0 * d * 4, 1);
            e.buf(cst, gate, 2);
            e.buf(cst, b.unwrap_or(0), 3);
            e.bytes(&U4([d as u32, b.is_some() as u32, 0, 0]), 4);
            e.dispatch(((d / 4).div_ceil(256), rows), 256);
        };
        // The image rows under block i's image modulation k (scale, shift / gate offsets k):
        // the tokens from `split` on under the second one.
        let ln_img = |e: &FastEnc, i: usize, ks: usize, kh: usize| {
            ln(e, rt, split, mo(12 * i + ks), mo(12 * i + kh));
            ln(e, rt + split, ni - split, mo2(6 * i + ks), mo2(6 * i + kh));
        };
        let gated_img = |e: &FastEnc, i: usize, k: usize, b: Option<usize>| {
            gated(e, rt, split, mo(12 * i + k), b);
            gated(e, rt + split, ni - split, mo2(6 * i + k), b);
        };
        // Plane j of Qkv (+ biases), normed and rotated with `norm`, into compact rows of dst;
        // rows below `split` take the A weights and bias, the others come from `gap` rows on.
        let heads_op = |e: &FastEnc, j: usize, dst: F, norm: Option<(usize, usize)>, b: (Option<usize>, Option<usize>), split: usize| {
            e.pipe(p.heads);
            e.buf(id(F::Qkv), j * rg * d * 4, 0);
            e.buf(id(dst), 0, 1);
            let (wa, wb) = norm.unwrap_or((0, 0));
            e.buf(cst, wa, 2);
            e.buf(cst, wb, 3);
            e.buf(cst, b.0.unwrap_or(0), 4);
            e.buf(cst, b.1.unwrap_or(0), 5);
            e.buf(id(F::Cos), 0, 6);
            e.buf(id(F::Sin), 0, 7);
            let lo = match dst {
                F::Q if att_mode == 2 => F::Ql,
                F::K if att_mode == 2 => F::Kl,
                _ => dst,
            };
            e.buf(id(lo), 0, 9);
            e.bytes(
                &Fhd {
                    ld_src: d as u32,
                    norm: norm.is_some() as u32,
                    rope: norm.is_some() as u32,
                    // Values stay plain half with hi / lo queries and keys.
                    out_half: if att_mode == 2 && dst == F::V { 1 } else { att_mode },
                    eps: EPS,
                    heads: heads as u32,
                    split: split as u32,
                    has_bias: (b.0.is_some() || b.1.is_some()) as u32,
                    gap: gap as u32,
                    _p: [0; 3],
                },
                8,
            );
            e.dispatch((n, heads), 32);
        };
        let attention = |e: &FastEnc| {
            if skip_op("att") {
                return;
            }
            e.pipe(match att_mode {
                1 => mma.flash128,
                2 => p.hilo.unwrap(),
                _ => mma.flash128_f32,
            });
            e.buf(id(F::Q), 0, 0);
            e.buf(id(F::K), 0, 1);
            e.buf(id(F::V), 0, 2);
            e.buf(id(F::Att), 0, 3);
            e.bytes(&U4([n as u32, n as u32, d as u32, (1.0 / (hd as f32).sqrt()).to_bits()]), 4);
            if att_mode == 2 {
                e.buf(id(F::Ql), 0, 5);
                e.buf(id(F::Kl), 0, 6);
            }
            e.dispatch((n.div_ceil(64), heads), 128);
            e.barrier();
        };
        // The LoRA side paths of layers that ran together (after their main GEMMs and a
        // barrier), side by side: y += up (down x) for each (layer, x, rows, y, ldy).
        let su = ri.max(rt);
        let sides = |en: &Enc, jobs: &[SideJob]| {
            let jobs: Vec<_> = jobs.iter().filter_map(|j| j.0.side.as_ref().map(|s| (s, j))).collect();
            if jobs.is_empty() {
                return;
            }
            debug_assert!(jobs.len() <= SIDE_SLOTS);
            let u = |k: usize| at(F::U, k * su * rmax * 4);
            let uh = |k: usize| Xh {
                xh: at(F::Uh, k * su * rmax * 2),
                rs: at(F::Urs, k * su * 4),
            };
            for (k, ((dn, _), j)) in jobs.iter().enumerate() {
                en.gemm_nb(dn, j.1, j.2, u(k), dn.n_out, false);
            }
            en.e.barrier();
            for (k, ((dn, _), j)) in jobs.iter().enumerate() {
                en.to_half_nb(u(k), dn.n_out, dn.n_out, j.2, uh(k));
            }
            en.e.barrier();
            for (k, ((_, up), j)) in jobs.iter().enumerate() {
                en.gemm_nb(up, uh(k), j.2, j.3, j.4, true);
            }
            en.e.barrier();
        };
        let side = |en: &Enc, l: &Lin, x: Xh, rows: usize, y: (Id, usize), ldy: usize| sides(en, &[(l, x, rows, y, ldy)]);
        let total = nd + ns;
        let parts: Vec<std::ops::Range<usize>> = (0..total.div_ceil(BLOCKS_PER_CB))
            .map(|c| c * BLOCKS_PER_CB..((c + 1) * BLOCKS_PER_CB).min(total))
            .collect();
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            let en = Enc { e, be };
            if range.start == 0 {
                // The embeddings: image tokens into rows rt.., text tokens into rows ..nx.
                let emb = |l: &Lin, src: F, rows: usize, valid: usize, r0: usize, b: Option<usize>| {
                    en.to_half(at(src, 0), l.n_in, l.n_in, rows, xh(0, l.n_in));
                    en.gemm(l, xh(0, l.n_in), rows, at(F::Tmp, 0), d, false);
                    side(&en, l, xh(0, l.n_in), rows, at(F::Tmp, 0), d);
                    e.pipe(p.bias);
                    e.buf(id(F::Tmp), 0, 0);
                    e.buf(id(F::H), r0 * d * 4, 1);
                    e.buf(cst, b.unwrap_or(0), 2);
                    e.bytes(&U4([d as u32, d as u32, d as u32, b.is_some() as u32]), 3);
                    e.dispatch((d.div_ceil(256), valid), 256);
                    e.barrier();
                };
                emb(&m.x_emb, F::Img, ri, ni, rt, x_b);
                if let Some(c) = &m.ctx_emb {
                    emb(c, F::Txt, rt, nx, 0, c_b);
                }
            }
            for i in range.clone() {
                if i < nd {
                    let b = &m.doubles[i];
                    let (bi, nr) = (&dbs[i], &dnorms[i]);
                    // Modulations: image k at mo(12 i + k), text at mo(12 i + 6 + k).
                    let tm = |k: usize| mo(12 * i + 6 + k);
                    ln_img(e, i, 1, 0);
                    ln(e, 0, nx, tm(1), tm(0));
                    e.barrier();
                    let qkv_at = |j: usize, r0: usize| at(F::Qkv, (j * rg + r0) * d * 4);
                    if !skip_op("qkv") {
                        for j in 0..3 {
                            en.gemm_nb(&b.img_qkv[j], xh(rt, d), ri, qkv_at(j, rt), d, false);
                            en.gemm_nb(&b.txt_qkv[j], xh(0, d), rt, qkv_at(j, 0), d, false);
                        }
                        e.barrier();
                        let jobs: Vec<_> = (0..3)
                            .flat_map(|j| {
                                [
                                    (&b.img_qkv[j], xh(rt, d), ri, qkv_at(j, rt), d),
                                    (&b.txt_qkv[j], xh(0, d), rt, qkv_at(j, 0), d),
                                ]
                            })
                            .collect();
                        sides(&en, &jobs);
                    }
                    heads_op(e, 0, F::Q, Some((nr[2], nr[0])), (bi.tq[0], bi.iq[0]), nx);
                    heads_op(e, 1, F::K, Some((nr[3], nr[1])), (bi.tq[1], bi.iq[1]), nx);
                    heads_op(e, 2, F::V, None, (bi.tq[2], bi.iq[2]), nx);
                    e.barrier();
                    attention(e);
                    if gap == 0 {
                        en.to_half(at(F::Att, 0), d, d, rg, xh(0, d));
                    } else {
                        en.to_half(at(F::Att, 0), d, d, nx, xh(0, d));
                        en.to_half(at(F::Att, nx * d * 4), d, d, ni, xh(rt, d));
                    }
                    let tmp_at = |r0: usize| at(F::Tmp, r0 * d * 4);
                    if !skip_op("o") {
                        en.gemm_nb(&b.img_out, xh(rt, d), ri, tmp_at(rt), d, false);
                        en.gemm_nb(&b.txt_out, xh(0, d), rt, tmp_at(0), d, false);
                        e.barrier();
                        sides(
                            &en,
                            &[(&b.img_out, xh(rt, d), ri, tmp_at(rt), d), (&b.txt_out, xh(0, d), rt, tmp_at(0), d)],
                        );
                    }
                    gated_img(e, i, 2, bi.io);
                    gated(e, 0, nx, tm(2), bi.to);
                    e.barrier();
                    // The MLPs.
                    ln_img(e, i, 4, 3);
                    ln(e, 0, nx, tm(4), tm(3));
                    e.barrier();
                    let f = b.img_mlp[0].n_out;
                    let hg_at = |r0: usize| at(F::Hg, r0 * f * 4);
                    if !skip_op("up") {
                        en.gemm_nb(&b.img_mlp[0], xh(rt, d), ri, hg_at(rt), f, false);
                        en.gemm_nb(&b.txt_mlp[0], xh(0, d), rt, hg_at(0), f, false);
                        e.barrier();
                        sides(
                            &en,
                            &[
                                (&b.img_mlp[0], xh(rt, d), ri, hg_at(rt), f),
                                (&b.txt_mlp[0], xh(0, d), rt, hg_at(0), f),
                            ],
                        );
                    }
                    for (r0, rows, bb) in [(rt, ni, bi.im[0]), (0, nx, bi.tm[0])] {
                        e.pipe(p.gelu);
                        e.buf(id(F::Hg), r0 * f * 4, 0);
                        e.buf(cst, bb.unwrap_or(0), 1);
                        e.buf(id(F::Xh), r0 * f * 2, 2);
                        e.buf(id(F::Rs), r0 * 4, 3);
                        e.bytes(&U4([f as u32, bb.is_some() as u32, 0, 0]), 4);
                        e.dispatch((rows, 1), 256);
                    }
                    e.barrier();
                    if !skip_op("down") {
                        en.gemm_nb(&b.img_mlp[1], xh(rt, f), ri, tmp_at(rt), d, false);
                        en.gemm_nb(&b.txt_mlp[1], xh(0, f), rt, tmp_at(0), d, false);
                        e.barrier();
                        sides(
                            &en,
                            &[
                                (&b.img_mlp[1], xh(rt, f), ri, tmp_at(rt), d),
                                (&b.txt_mlp[1], xh(0, f), rt, tmp_at(0), d),
                            ],
                        );
                    }
                    gated_img(e, i, 5, bi.im[1]);
                    gated(e, 0, nx, tm(5), bi.tm[1]);
                    e.barrier();
                } else {
                    let k = i - nd;
                    let b = &m.singles[k];
                    let (qb, mb, ob) = sbs[k];
                    let nr = snorms[k];
                    let sm = |j: usize| mo(12 * nd + 3 * k + j);
                    ln(e, 0, n, sm(1), sm(0));
                    e.barrier();
                    let f = b.mlp.n_out;
                    if !skip_op("qkv") {
                        for j in 0..3 {
                            en.gemm_nb(&b.qkv[j], xh(0, d), rg, at(F::Qkv, j * rg * d * 4), d, false);
                        }
                    }
                    if !skip_op("up") {
                        en.gemm_nb(&b.mlp, xh(0, d), rg, at(F::Hg, 0), f, false);
                    }
                    e.barrier();
                    heads_op(e, 0, F::Q, Some((nr[0], nr[0])), (qb[0], qb[0]), n);
                    heads_op(e, 1, F::K, Some((nr[1], nr[1])), (qb[1], qb[1]), n);
                    heads_op(e, 2, F::V, None, (qb[2], qb[2]), n);
                    e.barrier();
                    attention(e);
                    e.pipe(p.cat);
                    e.buf(id(F::Att), 0, 0);
                    e.buf(id(F::Hg), 0, 1);
                    e.buf(cst, mb.unwrap_or(0), 2);
                    e.buf(id(F::Xh), 0, 3);
                    e.buf(id(F::Rs), 0, 4);
                    e.bytes(&U4([d as u32, f as u32, mb.is_some() as u32, 0]), 5);
                    e.dispatch((n, 1), 256);
                    e.barrier();
                    if !skip_op("down") {
                        en.gemm(&b.out, xh(0, d + f), rg, at(F::Tmp, 0), d, false);
                    }
                    gated(e, 0, n, sm(2), ob);
                    e.barrier();
                }
            }
            if range.end == total {
                // The head: the final norm under its modulation, the output projection, over the
                // leading `no` image tokens.
                let hb = 12 * nd + 3 * ns;
                let ro = no.next_multiple_of(MMA_TM);
                ln(e, rt, no, mo(hb), mo(hb + 1));
                e.barrier();
                en.gemm(&m.proj_out, xh(rt, d), ro, at(F::Tmp, 0), cout, false);
                side(&en, &m.proj_out, xh(rt, d), ro, at(F::Tmp, 0), cout);
            }
        });
        if !ok {
            return false;
        }
        out.resize(no * cout, 0.0);
        // SAFETY: Tmp holds ro >= no rows of cout floats.
        unsafe { par_copy(out.as_mut_ptr(), self.host(F::Tmp), no * cout) };
        if let Some(b) = m.proj_out.bias {
            for row in out.chunks_mut(cout) {
                for (v, &bb) in row.iter_mut().zip(b) {
                    *v += bb;
                }
            }
        }
        true
    }
}
