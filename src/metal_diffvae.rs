//! LTX-2.5 DiffVAE blocks on Metal ([`NaSession`], `crate::ltx::diffvae`): the rows of a run of
//! neighborhood-attention blocks stay in an f32 device buffer; the projections run on the
//! matrix-unit GEMMs (scaled half inputs), q / k / v are normed, rotated and stored as half by
//! `dv_qkv_heads`, the attention is `dv_na3d_k*` (half operands, float32 arithmetic), the norms
//! `dit_rms_mod`, SwiGLU's middle `dv_silu_mul`. Projections and the MLP run over row chunks, so
//! the large f32 intermediates stay bounded; one command buffer per block.

use super::super::decode::{Fns, run_concurrent};
use super::super::*;
use super::{DBuf, Enc, Xh, grow, release_all, runnable};
use crate::gpu::dit::{Lin, NaBlock, NaSession};

struct Pipes {
    rms: Id,
    heads: Id,
    silu: Id,
    bias: Id,
    na: [(usize, Id); 4],
}

pub(crate) struct MetalNa {
    be: &'static MetalBackend,
    f: Fns,
    p: Pipes,
    bufs: Vec<DBuf>,
}

// SAFETY: as MetalDit.
unsafe impl Send for MetalNa {}

#[derive(Clone, Copy)]
enum B {
    H = 0,
    Y,
    Rs,
    Qkv,
    Q,
    K,
    V,
    Att,
    Ones,
    Ctx,
    CtxRs,
    G,
    U,
    Gh,
    GRs,
    Consts,
}
const NBUF: usize = 16;

/// Rows per chunk of the projections and the MLP (a multiple of the GEMM's row tile).
const CHUNK: usize = 1 << 16;

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
struct Dqh {
    dim: u32,
    heads: u32,
    qh: u32,
    qw: u32,
    kw: u32,
    halo: u32,
    r0: u32,
    pt: u32,
    ph: u32,
    ot: u32,
    oh: u32,
    ow: u32,
    scale: f32,
    eps: f32,
    nt: u32,
    nh: u32,
    nw: u32,
    _p: [u32; 3],
}

#[repr(C)]
struct Dna {
    qt: u32,
    qh: u32,
    qw: u32,
    kt: u32,
    kh: u32,
    kw: u32,
    ow: u32,
    kt_: u32,
    kh_: u32,
    heads: u32,
    dim: u32,
    hblocks: u32,
    row0: u32,
    wtiles: u32,
    _p: [u32; 2],
}

impl MetalBackend {
    /// [`crate::gpu::Backend::na_session`].
    pub(crate) fn na_session_impl(&'static self) -> Option<Box<dyn NaSession>> {
        self.mma().filter(|m| m.units)?;
        let pipe = |n: &str| self.named_pipeline(n);
        let p = Pipes {
            rms: pipe("dit_rms_mod")?,
            heads: pipe("dv_qkv_heads")?,
            silu: pipe("dv_silu_mul")?,
            bias: pipe("dv_add_bias")?,
            na: [
                (3, pipe("dv_na3d_k3")?),
                (5, pipe("dv_na3d_k5")?),
                (7, pipe("dv_na3d_k7")?),
                (11, pipe("dv_na3d_k11")?),
            ],
        };
        Some(Box::new(MetalNa {
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

impl Drop for MetalNa {
    fn drop(&mut self) {
        release_all(self.be, &self.bufs);
    }
}

/// `default_rope_dim_split(64)` as pairs: time, height (the rest: width).
const PAIRS: (usize, usize) = (8, 12);

/// cos then sin of `pos * 1 / 10000^(2 i / d)` per position (d = 2 pairs).
fn rope_table(pos: impl Iterator<Item = f32> + Clone, pairs: usize) -> Vec<f32> {
    let inv: Vec<f32> = (0..pairs)
        .map(|i| (1.0 / 10000f64.powf((2 * i) as f64 / (2 * pairs) as f64)) as f32)
        .collect();
    let mut c = Vec::new();
    let mut s = Vec::new();
    for p in pos {
        for &f in &inv {
            let a = p * f;
            c.push(a.cos());
            s.push(a.sin());
        }
    }
    c.extend(s);
    c
}

impl MetalNa {
    fn ensure(&mut self, b: B, bytes: usize) -> bool {
        grow(self.be, &mut self.bufs[b as usize], bytes)
    }

    fn host(&self, b: B) -> *mut u8 {
        self.f.contents(self.bufs[b as usize].id)
    }
}

/// Every layer of a block, for the checks.
fn layers<'l, 'a>(b: &'l NaBlock<'a>) -> Vec<&'l Lin<'a>> {
    let mut v = vec![&b.qkv, &b.proj, &b.gate, &b.up, &b.down];
    if let Some(c) = &b.ctx {
        v.push(c);
    }
    v
}

impl NaSession for MetalNa {
    fn blocks(&mut self, blocks: &[NaBlock], x: &mut [f32], ctx: Option<&[f32]>, d: [usize; 3]) -> bool {
        let be = self.be;
        let Some(b0) = blocks.first() else { return true };
        let dim = b0.dim;
        let n = d[0] * d[1] * d[2];
        if n == 0 || x.len() != n * dim || !dim.is_multiple_of(64) {
            return false;
        }
        let heads = dim / 64;
        let mut hidden = 0;
        for b in blocks {
            let k = b.kernel;
            if b.dim != dim || (0..3).any(|a| k[a] > d[a] && !(a == 2 && b.replicate_w)) || !self.p.na.iter().any(|(w, _)| *w == k[2]) {
                return false;
            }
            if layers(b).iter().any(|l| !runnable(be, l) || l.side.is_some() || l.rot.is_some()) {
                return false;
            }
            if b.qkv.n_out != 3 * dim || b.qkv.bias.is_none() || b.proj.n_out != dim || b.down.n_out != dim || b.gate.n_out != b.up.n_out {
                return false;
            }
            if b.ctx.is_some() != ctx.is_some() || b.ctx.as_ref().is_some_and(|c| c.n_out != dim) {
                return false;
            }
            hidden = hidden.max(b.gate.n_out);
        }
        let cdim = match (ctx, &b0.ctx) {
            (Some(c), Some(l)) => {
                if c.len() != n * l.n_in {
                    return false;
                }
                l.n_in
            }
            _ => 0,
        };
        let halo = blocks
            .iter()
            .map(|b| if b.replicate_w { b.kernel[2] / 2 } else { 0 })
            .max()
            .unwrap_or(0);
        if blocks.iter().any(|b| b.replicate_w && b.kernel[2] / 2 != halo) {
            return false;
        }
        let kw = d[2] + 2 * halo;
        let nk = d[0] * d[1] * kw;
        let np = n.next_multiple_of(MMA_TM);
        let ch = CHUNK.min(np);
        let ok = self.ensure(B::H, np * dim * 4)
            && self.ensure(B::Y, np * dim * 2)
            && self.ensure(B::Rs, np * 4)
            && self.ensure(B::Qkv, ch * 3 * dim * 4)
            && self.ensure(B::Q, n * dim * 2)
            && self.ensure(B::K, nk * dim * 2)
            && self.ensure(B::V, nk * dim * 2)
            && self.ensure(B::Att, np * dim * 2)
            && self.ensure(B::Ones, np * 4)
            && self.ensure(B::G, ch * hidden * 4)
            && self.ensure(B::U, ch * hidden * 4)
            && self.ensure(B::Gh, ch * hidden * 2)
            && self.ensure(B::GRs, ch * 4)
            && (cdim == 0 || (self.ensure(B::Ctx, np * cdim * 2) && self.ensure(B::CtxRs, np * 4)));
        if !ok {
            return false;
        }
        // Constants: per block the norms, the q / k / v biases, the q / k norms, the projection
        // biases and the modulations; then the RoPE tables.
        let mut c: Vec<f32> = Vec::new();
        let mut put = |v: &[f32]| -> usize {
            let o = c.len();
            c.extend_from_slice(v);
            c.resize(c.len().next_multiple_of(64), 0.0);
            o * 4
        };
        struct Off {
            n1: usize,
            n2: usize,
            qkv_b: usize,
            qn: usize,
            kn: usize,
            proj_b: Option<usize>,
            ctx_b: Option<usize>,
            mods: Option<[usize; 4]>,
        }
        let offs: Vec<Off> = blocks
            .iter()
            .map(|b| Off {
                n1: put(b.norm1),
                n2: put(b.norm2),
                qkv_b: put(b.qkv.bias.unwrap()),
                qn: put(b.q_norm),
                kn: put(b.k_norm),
                proj_b: b.proj.bias.map(&mut put),
                ctx_b: b.ctx.as_ref().and_then(|l| l.bias).map(&mut put),
                mods: b.mods.map(|m| [put(m[0]), put(m[1]), put(m[2]), put(m[3])]),
            })
            .collect();
        let (pt, ph) = PAIRS;
        let pw = 32 - pt - ph;
        let ot = put(&rope_table((0..d[0]).map(|i| i as f32), pt));
        let oh = put(&rope_table((0..d[1]).map(|i| i as f32), ph));
        let ow = put(&rope_table((0..kw).map(|i| i as f32 - halo as f32), pw));
        let zero = put(&vec![0.0; dim.max(hidden)]);
        if !self.ensure(B::Consts, c.len() * 4) {
            return false;
        }
        // SAFETY: every buffer holds at least what was ensured above; the device is idle.
        unsafe {
            par_copy(self.host(B::Consts) as *mut f32, c.as_ptr(), c.len());
            let h = self.host(B::H) as *mut f32;
            par_copy(h, x.as_ptr(), n * dim);
            std::ptr::write_bytes(h.add(n * dim), 0, (np - n) * dim);
            let ones = std::slice::from_raw_parts_mut(self.host(B::Ones) as *mut f32, np);
            ones.fill(1.0);
            if let Some(cx) = ctx {
                // Context rows as scaled half rows (the GEMM's input), on the host.
                let xh = std::slice::from_raw_parts_mut(self.host(B::Ctx) as *mut u16, np * cdim);
                let rs = std::slice::from_raw_parts_mut(self.host(B::CtxRs) as *mut f32, np);
                to_half_rows(cx, cdim, xh, rs);
            }
        }
        let ids: [Id; NBUF] = std::array::from_fn(|i| self.bufs[i].id);
        let id = |b: B| ids[b as usize];
        let at = |b: B, o: usize| (id(b), o);
        let p = &self.p;
        let stat = id(B::Consts);
        let parts: Vec<std::ops::Range<usize>> = (0..blocks.len()).map(|i| i..i + 1).collect();
        let (ok, _) = run_concurrent(be, &self.f, &parts, |e, range| {
            let en = Enc { e, be };
            let rms = |src: B, w: usize, m: Option<(usize, usize)>| {
                e.pipe(p.rms);
                e.buf(id(src), 0, 0);
                e.buf(stat, w, 1);
                e.buf(stat, m.map_or(zero, |m| m.0), 2);
                e.buf(stat, m.map_or(zero, |m| m.1), 3);
                e.buf(id(B::Y), 0, 4);
                e.buf(id(B::Rs), 0, 5);
                e.bytes(
                    &Drm {
                        n: dim as u32,
                        ld: dim as u32,
                        has_mod: m.is_some() as u32,
                        out_half: 1,
                        eps: 1e-6,
                        _p: [0; 3],
                    },
                    6,
                );
                e.dispatch((np, 1), 256);
                e.barrier();
            };
            let bias = |b: usize, rows: usize| {
                e.pipe(p.bias);
                e.buf(id(B::H), 0, 0);
                e.buf(stat, b, 1);
                e.bytes(&(dim as u32), 2);
                e.dispatch((dim.div_ceil(256), rows), 256);
                e.barrier();
            };
            for i in range {
                let b = &blocks[i];
                let o = &offs[i];
                if let Some(cl) = &b.ctx {
                    let xc = Xh {
                        xh: at(B::Ctx, 0),
                        rs: at(B::CtxRs, 0),
                    };
                    en.gemm(cl, xc, np, at(B::H, 0), dim, true);
                    if let Some(cb) = o.ctx_b {
                        bias(cb, n);
                    }
                }
                // Attention.
                rms(B::H, o.n1, o.mods.map(|m| (m[0], m[1])));
                let mut r0 = 0;
                while r0 < n {
                    let rows = ch.min(np - r0);
                    let xc = Xh {
                        xh: at(B::Y, r0 * dim * 2),
                        rs: at(B::Rs, r0 * 4),
                    };
                    en.gemm(&b.qkv, xc, rows, at(B::Qkv, 0), 3 * dim, false);
                    e.pipe(p.heads);
                    e.buf(id(B::Qkv), 0, 0);
                    e.buf(stat, o.qkv_b, 1);
                    e.buf(stat, o.qn, 2);
                    e.buf(stat, o.kn, 3);
                    e.buf(stat, 0, 4);
                    e.buf(id(B::Q), 0, 5);
                    e.buf(id(B::K), 0, 6);
                    e.buf(id(B::V), 0, 7);
                    let bh = if b.replicate_w { halo } else { 0 };
                    e.bytes(
                        &Dqh {
                            dim: dim as u32,
                            heads: heads as u32,
                            qh: d[1] as u32,
                            qw: d[2] as u32,
                            kw: (d[2] + 2 * bh) as u32,
                            halo: bh as u32,
                            r0: r0 as u32,
                            pt: pt as u32,
                            ph: ph as u32,
                            ot: (ot / 4) as u32,
                            oh: (oh / 4) as u32,
                            ow: ((ow / 4) + (halo - bh) * pw) as u32,
                            scale: 0.125,
                            eps: 1e-6,
                            nt: d[0] as u32,
                            nh: d[1] as u32,
                            nw: kw as u32,
                            _p: [0; 3],
                        },
                        8,
                    );
                    e.dispatch(((n - r0).min(rows), 1), 256);
                    e.barrier();
                    r0 += rows;
                }
                let bh = if b.replicate_w { halo } else { 0 };
                let k = b.kernel;
                let kd = [d[0], d[1], d[2] + 2 * bh];
                let ker = [k[0].min(kd[0]), k[1].min(kd[1]), k[2]];
                let hblocks = d[1].div_ceil(4);
                let wtiles = d[2].div_ceil(32);
                e.pipe(p.na.iter().find(|(w, _)| *w == k[2]).unwrap().1);
                e.buf(id(B::Q), 0, 0);
                e.buf(id(B::K), 0, 1);
                e.buf(id(B::V), 0, 2);
                e.buf(id(B::Att), 0, 3);
                e.bytes(
                    &Dna {
                        qt: d[0] as u32,
                        qh: d[1] as u32,
                        qw: d[2] as u32,
                        kt: kd[0] as u32,
                        kh: kd[1] as u32,
                        kw: kd[2] as u32,
                        ow: bh as u32,
                        kt_: ker[0] as u32,
                        kh_: ker[1] as u32,
                        heads: heads as u32,
                        dim: dim as u32,
                        hblocks: hblocks as u32,
                        row0: 0,
                        wtiles: wtiles as u32,
                        _p: [0; 2],
                    },
                    4,
                );
                e.dispatch((wtiles * heads, d[0] * hblocks), 128);
                e.barrier();
                let xa = Xh {
                    xh: at(B::Att, 0),
                    rs: at(B::Ones, 0),
                };
                en.gemm(&b.proj, xa, np, at(B::H, 0), dim, true);
                if let Some(pb) = o.proj_b {
                    bias(pb, n);
                }
                // SwiGLU.
                rms(B::H, o.n2, o.mods.map(|m| (m[2], m[3])));
                let f = b.gate.n_out;
                let mut r0 = 0;
                while r0 < n {
                    let rows = ch.min(np - r0);
                    let xc = Xh {
                        xh: at(B::Y, r0 * dim * 2),
                        rs: at(B::Rs, r0 * 4),
                    };
                    en.gemm_nb(&b.gate, xc, rows, at(B::G, 0), f, false);
                    en.gemm(&b.up, xc, rows, at(B::U, 0), f, false);
                    e.pipe(p.silu);
                    e.buf(id(B::G), 0, 0);
                    e.buf(id(B::U), 0, 1);
                    e.buf(id(B::Gh), 0, 2);
                    e.buf(id(B::GRs), 0, 3);
                    e.bytes(&(f as u32), 4);
                    e.dispatch((rows, 1), 256);
                    e.barrier();
                    let xg = Xh {
                        xh: at(B::Gh, 0),
                        rs: at(B::GRs, 0),
                    };
                    en.gemm(&b.down, xg, rows, at(B::H, r0 * dim * 4), dim, true);
                    r0 += rows;
                }
            }
        });
        if !ok {
            return false;
        }
        // SAFETY: H holds at least n rows of dim floats.
        unsafe { par_copy(x.as_mut_ptr(), self.host(B::H) as *const f32, n * dim) };
        true
    }
}

/// f32 rows of `n` to half rows scaled by a power of two per row (`xh = x * s`, `rs = 1 / s`,
/// the largest magnitude near 2^14, as `dit_row_scale`); rows past `x`'s are zeros.
fn to_half_rows(x: &[f32], n: usize, xh: &mut [u16], rs: &mut [f32]) {
    let rows = x.len() / n;
    let per = 1024;
    // Addresses, so the closure is Sync.
    let (xp, rp) = (xh.as_mut_ptr() as usize, rs.as_mut_ptr() as usize);
    crate::pool::run(rows.div_ceil(per), &|ci| {
        for r in ci * per..((ci + 1) * per).min(rows) {
            let row = &x[r * n..(r + 1) * n];
            let m = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let e = if m > 0.0 && m.is_finite() {
                (14 - (libm_frexp(m))).clamp(-100, 100)
            } else {
                0
            };
            let s = (2f32).powi(e);
            // SAFETY: row r of xh / rs belongs to this task alone.
            unsafe {
                let o = std::slice::from_raw_parts_mut((xp as *mut u16).add(r * n), n);
                for (d, &v) in o.iter_mut().zip(row) {
                    *d = half::f16::from_f32(v * s).to_bits();
                }
                *(rp as *mut f32).add(r) = 1.0 / s;
            }
        }
    });
    xh[rows * n..].fill(0);
    rs[rows..].fill(1.0);
}

/// The exponent `e` of `m = f 2^e` with `f` in `[0.5, 1)` (C's `frexp`).
fn libm_frexp(m: f32) -> i32 {
    let bits = m.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    if exp == 0 {
        // Subnormal: normalize first.
        return libm_frexp(m * (1u64 << 32) as f32) - 32;
    }
    exp - 126
}
