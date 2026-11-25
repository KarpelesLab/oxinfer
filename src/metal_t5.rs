//! Whole text-encoder passes on Metal: T5 v1.1 ([`T5Session`], `crate::flux::t5`) and CLIP
//! ([`ClipSession`], `crate::sd::clip`). The residual stream stays in f32 device rows for all
//! blocks; the projections run on the matrix-unit GEMMs of the diffusion-transformer plans
//! (scaled half inputs; any weight format they take: bf16, q8 / q4, GGUF blocks); attention runs
//! on f32 operands with an additive bias over key - query (`t5_flash64` in `metal_dit_mma.metal`:
//! T5's relative-position bias, CLIP's causal mask); the norms, activations and residual adds are
//! row kernels. One submission per encode, read back once.

use super::super::decode::{FastEnc, Fns, run_concurrent};
use super::super::*;
use super::{BLOCKS_PER_CB, DBuf, Drm, Enc, U4, Xh, grow, release_all, runnable};
use crate::gpu::dit::{ClipModel, ClipSession, Lin, T5Model, T5Session};

struct Pipes {
    rms: Id,
    flash: Id,
    ln: Id,
    act: Id,
    res: Id,
    bias: Id,
}

/// A text-encoder session (one per encoder: T5 or CLIP).
pub(crate) struct MetalText {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    bufs: Vec<DBuf>,
}

// SAFETY: as MetalDit.
unsafe impl Send for MetalText {}

#[derive(Clone, Copy)]
enum T {
    H = 0,
    Xh,
    Rs,
    Q,
    K,
    V,
    Att,
    G,
    Out,
    Pen,
    Consts,
}
const NTBUF: usize = 11;

/// `t5_flash64`'s parameters.
#[repr(C)]
struct Tfa {
    tq: u32,
    tk: u32,
    d: u32,
    /// Bias row stride per head (0: one row for all heads).
    hs: u32,
    scale: f32,
    _p: [u32; 3],
}

/// `clip_ln`'s parameters.
#[repr(C)]
struct Cln {
    n: u32,
    eps: f32,
    out_f32: u32,
    _p: u32,
}

impl MetalBackend {
    fn text_session(&'static self) -> Option<MetalText> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        Some(MetalText {
            be: self,
            f: Fns::new(self),
            p: Pipes {
                rms: pipe("dit_rms_mod")?,
                flash: pipe("t5_flash64")?,
                ln: pipe("clip_ln")?,
                act: pipe("clip_act_half")?,
                res: pipe("clip_res_add")?,
                bias: pipe("dit_bias_rows")?,
            },
            bufs: (0..NTBUF)
                .map(|_| DBuf {
                    id: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
        })
    }

    /// [`crate::gpu::Backend::t5_session`].
    pub(crate) fn t5_session_impl(&'static self) -> Option<Box<dyn T5Session>> {
        Some(Box::new(self.text_session()?))
    }

    /// [`crate::gpu::Backend::clip_session`].
    pub(crate) fn clip_session_impl(&'static self) -> Option<Box<dyn ClipSession>> {
        Some(Box::new(self.text_session()?))
    }
}

impl Drop for MetalText {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

impl MetalText {
    fn ensure(&mut self, b: T, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn host(&self, b: T) -> *mut f32 {
        self.f.contents(self.bufs[b as usize].id) as *mut f32
    }

    /// Sizes the buffers for `rows` rows of width `d` (feed-forward `f`, `g` gate / up planes),
    /// uploads the constants and the embeddings `x` (`t` rows) and clears the padding rows.
    #[allow(clippy::too_many_arguments)]
    fn prepare(&mut self, rows: usize, d: usize, f: usize, g: usize, consts: &[f32], x: &[f32], t: usize, pen: bool) -> bool {
        let wide = f.max(d);
        let ok = self.ensure(T::H, rows * d * 4)
            && self.ensure(T::Xh, rows * wide * 2)
            && self.ensure(T::Rs, rows * 4)
            && self.ensure(T::Q, rows * d * 4)
            && self.ensure(T::K, rows * d * 4)
            && self.ensure(T::V, rows * d * 4)
            && self.ensure(T::Att, rows * d * 4)
            && self.ensure(T::G, g * rows * f * 4)
            && self.ensure(T::Out, rows * d * 4)
            && (!pen || self.ensure(T::Pen, rows * d * 4))
            && self.ensure(T::Consts, consts.len() * 4);
        if !ok {
            return false;
        }
        // SAFETY: every buffer was sized above; the device is idle (the last pass was waited for).
        unsafe {
            par_copy(self.host(T::Consts), consts.as_ptr(), consts.len());
            let h = self.host(T::H);
            par_copy(h, x.as_ptr(), t * d);
            std::ptr::write_bytes(h.add(t * d), 0, (rows - t) * d);
            // The GEMMs read whole tiles: rows past t stay finite (zero, never written).
            std::ptr::write_bytes(self.host(T::Xh) as *mut u8, 0, rows * wide * 2);
            let rs = self.host(T::Rs);
            for r in 0..rows {
                *rs.add(r) = 1.0;
            }
        }
        true
    }

    fn ids(&self) -> [Id; NTBUF] {
        std::array::from_fn(|i| self.bufs[i].id)
    }
}

/// Every layer runs on the device with tile `tile`.
fn all_runnable<'l, 'a: 'l>(be: &MetalBackend, mut lins: impl Iterator<Item = &'l Lin<'a>>, tile: Tile) -> bool {
    lins.all(|l| {
        l.side.is_none()
            && l.rot.is_none()
            && runnable(be, l)
            && l.dev
                .downcast_ref::<MetalWeight>()
                .is_some_and(|mw| be.mma_pipe(&l.w, mw.offsets.is_some(), l.n_out, l.n_in, tile).is_some())
    })
}

/// Appends `v` to `consts` (64-float aligned), returning its byte offset.
fn put(consts: &mut Vec<f32>, v: &[f32]) -> usize {
    let o = consts.len();
    consts.extend_from_slice(v);
    consts.resize(consts.len().next_multiple_of(64), 0.0);
    o * 4
}

/// Attention over the t rows of Q, K, V (heads of 64) into Att.
#[allow(clippy::too_many_arguments)]
fn attend(e: &FastEnc, pipe: Id, ids: &[Id; NTBUF], t: usize, d: usize, bias: (Id, usize), hs: usize, scale: f32) {
    let id = |b: T| ids[b as usize];
    e.pipe(pipe);
    e.buf(id(T::Q), 0, 0);
    e.buf(id(T::K), 0, 1);
    e.buf(id(T::V), 0, 2);
    e.buf(id(T::Att), 0, 3);
    e.bytes(
        &Tfa {
            tq: t as u32,
            tk: t as u32,
            d: d as u32,
            hs: hs as u32,
            scale,
            _p: [0; 3],
        },
        4,
    );
    e.buf(bias.0, bias.1, 5);
    if !super::skip_op("att") {
        e.dispatch((t.div_ceil(64), d / 64), 128);
    }
    e.barrier();
}

/// GEMM tiles of the encoder's projections: `OXINFER_T5_TILE=dense|mid|small` (128-, 64-, 32-token
/// tiles; default by row count).
fn tile_for(rows: usize) -> (Tile, usize) {
    match std::env::var("OXINFER_T5_TILE").as_deref() {
        Ok("dense") => (Tile::Dense, MMA_TM),
        Ok("mid") => (Tile::Mid, MMA_TM_S * 2),
        Ok("small") => (Tile::Small, MMA_TM_S),
        _ if rows <= 512 => (Tile::Mid, MMA_TM_S * 2),
        _ => (Tile::Dense, MMA_TM),
    }
}

/// `y = W x` (`y +=` with `add`) over `rows` rows (a multiple of the tile's) with `tile`'s kernel,
/// rows of y `ldy` floats apart; no barrier.
#[allow(clippy::too_many_arguments)]
fn gemm(e: &FastEnc, be: &MetalBackend, l: &Lin, x: Xh, rows: usize, y: (Id, usize), ldy: usize, add: bool, tile: (Tile, usize)) {
    let mw = l.dev.downcast_ref::<MetalWeight>().unwrap();
    let (pipe, tn) = be.mma_pipe(&l.w, mw.offsets.is_some(), l.n_out, l.n_in, tile.0).unwrap();
    let (_, elem_bytes, scale_bytes) = format_index(&l.w);
    let s_off = match &l.w {
        WeightRef::Gguf(m) => gguf_s_off(m, mw),
        _ => mw.scales.as_ref().map(|s| s.offset_bytes / scale_bytes).unwrap_or(0),
    };
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
    e.dispatch((l.n_out / tn, rows / tile.1), 128);
}

impl T5Session for MetalText {
    fn forward(&mut self, m: &T5Model, x: &[f32], t: usize, bias: &[f32], out: &mut Vec<f32>) -> bool {
        let be = self.be;
        let (d, nh, dk) = (m.dim, m.heads, m.d_kv);
        if dk != 64 || nh * dk != d || !d.is_multiple_of(256) || t == 0 || x.len() != t * d || bias.len() != nh * (2 * t - 1) {
            return false;
        }
        let Some(f) = m.blocks.first().map(|b| b.wi[0].n_out) else {
            return false;
        };
        let shapes_ok = m.blocks.iter().all(|b| {
            [&b.q, &b.k, &b.v, &b.o]
                .iter()
                .all(|l| l.n_out == d && l.n_in == d && l.bias.is_none())
                && b.wi.iter().all(|l| l.n_out == f && l.n_in == d && l.bias.is_none())
                && b.wo.n_out == d
                && b.wo.n_in == f
                && b.wo.bias.is_none()
                && b.ln1.len() == d
                && b.ln2.len() == d
        });
        let tile = tile_for(t);
        let lins = m.blocks.iter().flat_map(|b| [&b.q, &b.k, &b.v, &b.o, &b.wi[0], &b.wi[1], &b.wo]);
        if !shapes_ok || m.final_ln.len() != d || !f.is_multiple_of(4) || !all_runnable(be, lins, tile.0) {
            return false;
        }
        // Whole tiles of the largest kind (the attention and the row kernels read t rows only).
        let rows = t.next_multiple_of(MMA_TM);
        let mut consts: Vec<f32> = Vec::with_capacity((2 * m.blocks.len() + 1) * d + bias.len() + 64 * (2 * m.blocks.len() + 2));
        let lns: Vec<[usize; 2]> = m
            .blocks
            .iter()
            .map(|b| [put(&mut consts, b.ln1), put(&mut consts, b.ln2)])
            .collect();
        let fin = put(&mut consts, m.final_ln);
        let bo = put(&mut consts, bias);
        if !self.prepare(rows, d, f, 2, &consts, x, t, false) {
            return false;
        }
        let ids = self.ids();
        let id = |b: T| ids[b as usize];
        let xh = Xh {
            xh: (id(T::Xh), 0),
            rs: (id(T::Rs), 0),
        };
        let cst = id(T::Consts);
        let mma = be.mma.as_ref().unwrap();
        let (p, eps) = (&self.p, m.eps);
        // RMS norm of the t rows of H with weight `w`, into scaled half rows (else f32 rows of Out).
        let rms = |e: &FastEnc, w: usize, half: bool| {
            e.pipe(p.rms);
            e.buf(id(T::H), 0, 0);
            e.buf(cst, w, 1);
            e.buf(cst, w, 2);
            e.buf(cst, w, 3);
            e.buf(if half { id(T::Xh) } else { id(T::Out) }, 0, 4);
            e.buf(id(T::Rs), 0, 5);
            e.bytes(
                &Drm {
                    n: d as u32,
                    ld: d as u32,
                    has_mod: 0,
                    out_half: half as u32,
                    eps,
                    _p: [0; 3],
                },
                6,
            );
            e.dispatch((t, 1), 256);
            e.barrier();
        };
        let nb = m.blocks.len();
        let parts: Vec<std::ops::Range<usize>> = (0..nb.div_ceil(BLOCKS_PER_CB))
            .map(|c| c * BLOCKS_PER_CB..((c + 1) * BLOCKS_PER_CB).min(nb))
            .collect();
        let gm = |e: &FastEnc, l: &Lin, y: (Id, usize), ldy: usize, add: bool| gemm(e, be, l, xh, rows, y, ldy, add, tile);
        let skip = super::skip_op;
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            let en = Enc { e, be };
            for i in range.clone() {
                let b = &m.blocks[i];
                rms(e, lns[i][0], true);
                if !skip("qkv") {
                    gm(e, &b.q, (id(T::Q), 0), d, false);
                    gm(e, &b.k, (id(T::K), 0), d, false);
                    gm(e, &b.v, (id(T::V), 0), d, false);
                }
                e.barrier();
                // T5: unscaled scores.
                attend(e, p.flash, &ids, t, d, (cst, bo), 2 * t - 1, 1.0);
                en.to_half((id(T::Att), 0), d, d, t, xh);
                if !skip("o") {
                    gm(e, &b.o, (id(T::H), 0), d, true);
                }
                e.barrier();
                rms(e, lns[i][1], true);
                if !skip("up") {
                    gm(e, &b.wi[0], (id(T::G), 0), f, false);
                    gm(e, &b.wi[1], (id(T::G), rows * f * 4), f, false);
                }
                e.barrier();
                // gelu_tanh(gate) * up into scaled half rows (gate row r at r f, up at (rows + r) f).
                e.pipe(mma.silu_half);
                e.buf(id(T::G), 0, 0);
                e.buf(id(T::Xh), 0, 1);
                e.buf(id(T::Rs), 0, 2);
                e.bytes(&U4([f as u32, 1, rows as u32, 0]), 3);
                e.dispatch((t, 1), 256);
                e.barrier();
                if !skip("down") {
                    gm(e, &b.wo, (id(T::H), 0), d, true);
                }
                e.barrier();
            }
            if range.end == nb {
                rms(e, fin, false);
            }
        });
        if !ok {
            return false;
        }
        out.resize(t * d, 0.0);
        // SAFETY: Out holds t rows of d floats.
        unsafe { par_copy(out.as_mut_ptr(), self.host(T::Out), t * d) };
        true
    }
}

impl ClipSession for MetalText {
    fn forward(&mut self, m: &ClipModel, x: &[f32], t: usize, penultimate: Option<&mut Vec<f32>>, out: &mut Vec<f32>) -> bool {
        let be = self.be;
        let (d, nh) = (m.dim, m.heads);
        let nb = m.blocks.len();
        if nh * 64 != d || !d.is_multiple_of(256) || t == 0 || x.len() != t * d || nb == 0 {
            return false;
        }
        let f = m.blocks[0].fc1.n_out;
        let has = |l: &Lin, n_out: usize, n_in: usize| l.n_out == n_out && l.n_in == n_in && l.bias.is_some_and(|b| b.len() == n_out);
        let shapes_ok = m.blocks.iter().all(|b| {
            [&b.q, &b.k, &b.v, &b.o].iter().all(|l| has(l, d, d))
                && has(&b.fc1, f, d)
                && has(&b.fc2, d, f)
                && b.ln1.iter().chain(&b.ln2).all(|v| v.len() == d)
        });
        let tile = tile_for(t);
        let lins = m.blocks.iter().flat_map(|b| [&b.q, &b.k, &b.v, &b.o, &b.fc1, &b.fc2]);
        if !shapes_ok || m.final_ln.iter().any(|v| v.len() != d) || !f.is_multiple_of(4) || !all_runnable(be, lins, tile.0) {
            return false;
        }
        let rows = t.next_multiple_of(MMA_TM);
        let mut consts: Vec<f32> = Vec::new();
        // Per block: ln1 w, b, ln2 w, b, then the biases of q, k, v, o, fc1, fc2.
        let offs: Vec<[usize; 10]> = m
            .blocks
            .iter()
            .map(|b| {
                let mut c = |v: &[f32]| put(&mut consts, v);
                [
                    c(b.ln1[0]),
                    c(b.ln1[1]),
                    c(b.ln2[0]),
                    c(b.ln2[1]),
                    c(b.q.bias.unwrap()),
                    c(b.k.bias.unwrap()),
                    c(b.v.bias.unwrap()),
                    c(b.o.bias.unwrap()),
                    c(b.fc1.bias.unwrap()),
                    c(b.fc2.bias.unwrap()),
                ]
            })
            .collect();
        let fin = [put(&mut consts, m.final_ln[0]), put(&mut consts, m.final_ln[1])];
        // The causal mask over key - query (one row for all heads).
        let mask: Vec<f32> = (0..2 * t - 1).map(|r| if r < t { 0.0 } else { f32::NEG_INFINITY }).collect();
        let mo = put(&mut consts, &mask);
        let pen = penultimate.is_some();
        if !self.prepare(rows, d, f, 1, &consts, x, t, pen) {
            return false;
        }
        let ids = self.ids();
        let id = |b: T| ids[b as usize];
        let xh = Xh {
            xh: (id(T::Xh), 0),
            rs: (id(T::Rs), 0),
        };
        let cst = id(T::Consts);
        let (p, eps) = (&self.p, m.eps);
        // Layer norm of the t rows of H (weight, bias offsets) into scaled half rows (or f32 rows of Out).
        let ln = |e: &FastEnc, w: usize, b: usize, half: bool| {
            e.pipe(p.ln);
            e.buf(id(T::H), 0, 0);
            e.buf(cst, w, 1);
            e.buf(cst, b, 2);
            e.buf(if half { id(T::Xh) } else { id(T::Out) }, 0, 3);
            e.buf(id(T::Rs), 0, 4);
            e.bytes(
                &Cln {
                    n: d as u32,
                    eps,
                    out_f32: (!half) as u32,
                    _p: 0,
                },
                5,
            );
            e.dispatch((t, 1), 256);
            e.barrier();
        };
        // dst[r] (+)= src[r] + bias over t rows of n: `res` adds into dst (H), else in place.
        let add_bias = |e: &FastEnc, buf: T, n: usize, b: usize| {
            e.pipe(p.bias);
            e.buf(id(buf), 0, 0);
            e.buf(id(buf), 0, 1);
            e.buf(cst, b, 2);
            e.bytes(&U4([n as u32, n as u32, n as u32, 1]), 3);
            e.dispatch((n.div_ceil(256), t), 256);
        };
        let res = |e: &FastEnc, src: T, b: usize| {
            e.pipe(p.res);
            e.buf(id(T::H), 0, 0);
            e.buf(id(src), 0, 1);
            e.buf(cst, b, 2);
            e.bytes(&U4([d as u32, 0, 0, 0]), 3);
            e.dispatch(((d / 4).div_ceil(256), t), 256);
            e.barrier();
        };
        let gm = |e: &FastEnc, l: &Lin, y: (Id, usize), ldy: usize| gemm(e, be, l, xh, rows, y, ldy, false, tile);
        let scale = 1.0 / 8.0;
        // One command buffer (CLIP is small).
        let whole = std::iter::once(0..nb).collect::<Vec<_>>();
        let (ok, _) = run_concurrent(be, &self.f, &whole, |e, _| {
            let en = Enc { e, be };
            for (i, (b, o)) in m.blocks.iter().zip(&offs).enumerate() {
                if pen && i + 1 == nb {
                    // The penultimate hidden state: H before the last layer.
                    e.pipe(p.bias);
                    e.buf(id(T::H), 0, 0);
                    e.buf(id(T::Pen), 0, 1);
                    e.buf(cst, 0, 2);
                    e.bytes(&U4([d as u32, d as u32, d as u32, 0]), 3);
                    e.dispatch((d.div_ceil(256), t), 256);
                }
                ln(e, o[0], o[1], true);
                gm(e, &b.q, (id(T::Q), 0), d);
                gm(e, &b.k, (id(T::K), 0), d);
                gm(e, &b.v, (id(T::V), 0), d);
                e.barrier();
                add_bias(e, T::Q, d, o[4]);
                add_bias(e, T::K, d, o[5]);
                add_bias(e, T::V, d, o[6]);
                e.barrier();
                attend(e, p.flash, &ids, t, d, (cst, mo), 0, scale);
                en.to_half((id(T::Att), 0), d, d, t, xh);
                gm(e, &b.o, (id(T::Att), 0), d);
                e.barrier();
                res(e, T::Att, o[7]);
                ln(e, o[2], o[3], true);
                gm(e, &b.fc1, (id(T::G), 0), f);
                e.barrier();
                e.pipe(p.act);
                e.buf(id(T::G), 0, 0);
                e.buf(cst, o[8], 1);
                e.buf(id(T::Xh), 0, 2);
                e.buf(id(T::Rs), 0, 3);
                e.bytes(&U4([f as u32, if m.gelu { 2 } else { 1 }, 0, 0]), 4);
                e.dispatch((t, 1), 256);
                e.barrier();
                gm(e, &b.fc2, (id(T::Att), 0), d);
                e.barrier();
                res(e, T::Att, o[9]);
            }
            ln(e, fin[0], fin[1], false);
        });
        if !ok {
            return false;
        }
        out.resize(t * d, 0.0);
        // SAFETY: Out (and Pen) hold t rows of d floats.
        unsafe { par_copy(out.as_mut_ptr(), self.host(T::Out), t * d) };
        if let Some(pv) = penultimate {
            pv.resize(t * d, 0.0);
            unsafe { par_copy(pv.as_mut_ptr(), self.host(T::Pen), t * d) };
        }
        true
    }
}
