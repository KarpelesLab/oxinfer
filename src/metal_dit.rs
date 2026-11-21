//! Whole diffusion-transformer passes on Metal ([`crate::gpu::dit`]): every block of every
//! sequence of a batch in a few command buffers, the activations in device buffers that live as
//! long as the session. The matmuls are the matrix-unit GEMMs (`MMA_SHADER_SRC`, half inputs
//! with a power-of-two scale per row), attention the fused `flash_attn128_mma` kernels, the row
//! work the `dit_*` kernels of `metal_dit.metal`. Without matrix units there is no session (the
//! per-op path runs instead).

use super::decode::{FastEnc, Fns, run_concurrent};
use super::*;
use crate::gpu::dit::{Lin, Model, Seq, Session};

pub(crate) const SRC: &str = include_str!("metal_dit.metal");

/// Blocks per command buffer (the GPU starts on the first while the rest is encoded).
const BLOCKS_PER_CB: usize = 6;

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
        self.mma()?;
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
        let void: MsgVoid = self.be.o.msg_send();
        for b in &self.bufs {
            if !b.id.is_null() {
                unsafe { void(b.id, self.be.sels.release) };
            }
        }
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

    fn id(&self, b: B) -> Id {
        self.bufs[b as usize].id
    }

    fn host(&self, b: B) -> *mut f32 {
        self.f.contents(self.id(b)) as *mut f32
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
        e.barrier();
    }

    /// f32 rows (`ld` apart) to scaled half rows.
    fn to_half(&self, x: (Id, usize), ld: usize, n: usize, rows: usize, out: Xh) {
        let m = self.be.mma.as_ref().unwrap();
        let e = self.e;
        e.pipe(m.x_half);
        e.buf(x.0, x.1, 0);
        e.buf(out.xh.0, out.xh.1, 1);
        e.buf(out.rs.0, out.rs.1, 2);
        e.bytes(&U4([n as u32, 0, ld as u32, 0]), 3);
        e.buf(x.0, x.1, 4);
        e.dispatch((rows, 1), 256);
        e.barrier();
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
