//! Int8 prefill GEMM (aarch64 SDOT).
//!
//! The quantized matrices' rows are unpacked, four at a time, into int8 tiles with the rows
//! interleaved by groups of four values (`[block of 32][group of 4][row][4]`), plus their f32
//! scales per 32 (and integer scales per 16, mins per 32, as the format needs). The activations
//! are the per-32 int8 blocks the decode kernels use (`QAct`). A micro-kernel then broadcasts four
//! activation bytes of one token per `SDOT` (by element) against four rows, so each instruction
//! adds sixteen products into one int32 lane per row, without horizontal reductions; integer sums
//! per block of 32 are scaled into f32 accumulators.
//!
//! On Linux (Arm's own cores) the GEMM is cache-blocked by columns, and where the CPU has i8mm it runs
//! on `SMMLA` instead: weight rows and tokens two by two, 8 bytes of each per register, 32
//! products per instruction into a 2 x 2 tile of int32 sums ([`gemm_rows_mmla`]). On a
//! Cortex-X925 that instruction issues at SDOT's rate. Same integer sums, same f32 operations in
//! the same order: bit-identical results.
//!
//! Rounding the activations to int8 is the decode kernels' approximation (and llama.cpp's for
//! prefill); the f32 sums run in another order than the decode kernels', so rows are close to
//! but not bit-identical to them.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::kernels::{I8Layout, QMat};
use crate::quant::QBLOCK;
use std::arch::aarch64::*;

/// One row unpacked by [`QMat::unpack_row_i8`]: `q` (`n_in`), `ds` and `mn` (`n_in / 32`), `sc`
/// (`n_in / 16`).
pub struct URow {
    pub q: Vec<i8>,
    pub ds: Vec<f32>,
    pub mn: Vec<f32>,
    pub sc: Vec<i32>,
}

impl URow {
    const fn new() -> Self {
        Self {
            q: Vec::new(),
            ds: Vec::new(),
            mn: Vec::new(),
            sc: Vec::new(),
        }
    }
    fn fit(&mut self, n_in: usize) {
        if self.q.len() < n_in {
            self.q.resize(n_in, 0);
            self.ds.resize(n_in / QBLOCK, 0.0);
            self.mn.resize(n_in / QBLOCK, 0.0);
            self.sc.resize(n_in / 16, 0);
        }
    }
}

/// Four rows interleaved: `q[b * 128 + 16 * g + 4 * r + e]` is element `32 b + 4 g + e` of row
/// `r`; `ds[4 b + r]`, `mn[4 b + r]`; `sc[8 b + 4 h + r]` scales half `h` of block `b`.
struct Tile {
    q: Vec<i8>,
    ds: Vec<f32>,
    mn: Vec<f32>,
    sc: Vec<i32>,
}

impl Tile {
    const fn new() -> Self {
        Self {
            q: Vec::new(),
            ds: Vec::new(),
            mn: Vec::new(),
            sc: Vec::new(),
        }
    }
    fn fit(&mut self, n_in: usize) {
        let nb = n_in / QBLOCK;
        if self.q.len() < nb * 128 {
            self.q.resize(nb * 128, 0);
            self.ds.resize(nb * 4, 0.0);
            self.mn.resize(nb * 4, 0.0);
            self.sc.resize(nb * 8, 0);
        }
    }

    /// Rows `r..r + 4` of `m` (rows past `n_out` as zeros).
    unsafe fn fill<M: QMat + ?Sized>(&mut self, m: &M, lay: I8Layout, r: usize, rows: &mut [URow; 4]) {
        let n_in = m.n_in();
        let nb = n_in / QBLOCK;
        self.fit(n_in);
        for (i, u) in rows.iter_mut().enumerate() {
            u.fit(n_in);
            if r + i < m.n_out() {
                m.unpack_row_i8(r + i, u.q.as_mut_ptr(), u.ds.as_mut_ptr(), u.mn.as_mut_ptr(), u.sc.as_mut_ptr());
            } else {
                u.q[..n_in].fill(0);
                u.ds[..nb].fill(0.0);
                u.mn[..nb].fill(0.0);
                u.sc[..n_in / 16].fill(0);
            }
        }
        let q = self.q.as_mut_ptr();
        for b in 0..nb {
            for h in 0..2 {
                let o = 32 * b + 16 * h;
                let v = rows.each_ref().map(|u| vreinterpretq_u32_s8(vld1q_s8(u.q.as_ptr().add(o))));
                let (t0, t1) = (vtrn1q_u32(v[0], v[1]), vtrn2q_u32(v[0], v[1]));
                let (t2, t3) = (vtrn1q_u32(v[2], v[3]), vtrn2q_u32(v[2], v[3]));
                let (t0, t1, t2, t3) = (
                    vreinterpretq_u64_u32(t0),
                    vreinterpretq_u64_u32(t1),
                    vreinterpretq_u64_u32(t2),
                    vreinterpretq_u64_u32(t3),
                );
                let w = [vtrn1q_u64(t0, t2), vtrn1q_u64(t1, t3), vtrn2q_u64(t0, t2), vtrn2q_u64(t1, t3)];
                for (g, wg) in w.iter().enumerate() {
                    vst1q_s8(q.add(b * 128 + 64 * h + 16 * g), vreinterpretq_s8_u64(*wg));
                }
            }
            for (i, u) in rows.iter().enumerate() {
                self.ds[4 * b + i] = u.ds[b];
                if lay.min32 {
                    self.mn[4 * b + i] = u.mn[b];
                }
                if lay.sc16 {
                    self.sc[8 * b + i] = u.sc[2 * b];
                    self.sc[8 * b + 4 + i] = u.sc[2 * b + 1];
                }
            }
        }
    }
}

/// `acc` (int32 per row) += sixteen products: rows' bytes `w` (4 per row) times bytes
/// `4L..4L + 4` of `x`, broadcast.
#[inline(always)]
unsafe fn sdot_lane<const L: i32>(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    let mut r = acc;
    std::arch::asm!(
        ".arch_extension dotprod",
        "sdot {r:v}.4s, {w:v}.16b, {x:v}.4b[{l}]",
        r = inout(vreg) r,
        w = in(vreg) w,
        x = in(vreg) x,
        l = const L,
        options(pure, nomem, nostack)
    );
    r
}

/// Activations of the tokens of one micro-kernel call (`I8Act` from a token on).
#[derive(Clone, Copy)]
struct Xs {
    q: *const i8,
    s: *const f32,
    f: *const f32,
    n_in: usize,
}

/// Activations `[t, n_in]` for the GEMM: per block of 32, int8 values `q1` with scale
/// `s1 = amax / 127`, and with `two` planes also the rounding residual in int8 at a step of
/// `s1 / 254`: `x ~ s1 / 254 * (254 q1 + q2)` (errors below `amax / 64516`, about 16-bit
/// precision; one plane gives the decode kernels' `amax / 254`). A token's row holds, per block
/// and half of 16, `q1` (then `q2`) vectors. `s` is the scale of the integer sums (`s1`, or
/// `s1 / 254` with two planes); `f` the exact f32 block sums.
///
/// The residual is needed in every matrix: on Llama 3.2 1B, two planes for only the blocks with
/// the largest steps (95% of the squared rounding error) or for only the down projections left
/// the KL against f32 at 3e-4 / 5e-4, against 9.5e-4 for one plane and 2e-6 for two everywhere.
///
/// With `pairs` (one plane, for the SMMLA GEMM) tokens go two by two: pair `p` (tokens `2p`,
/// `2p + 1`; a missing last one is zeros) holds, per 8 inputs, 8 bytes of each token
/// (`q[2 p n_in + 16 i + 8 k + e]` is input `8 i + e` of token `2 p + k`), and `s` / `f` hold
/// per pair and block the four values `[t0, t1, t0, t1]`, the order of an SMMLA's 2 x 2 sums.
pub(crate) struct I8Act {
    q: Vec<i8>,
    s: Vec<f32>,
    f: Vec<f32>,
    t: usize,
    n_in: usize,
    two: bool,
    pairs: bool,
}

impl I8Act {
    /// The activations in the layout of the GEMM that [`gemm_rows`] runs on this CPU.
    pub(crate) fn new(x: &[f32], t: usize, n_in: usize, two: bool) -> Self {
        Self::with_layout(x, t, n_in, two, !two && mmla())
    }

    fn with_layout(x: &[f32], t: usize, n_in: usize, two: bool, pairs: bool) -> Self {
        assert_eq!(x.len(), t * n_in);
        assert!(n_in.is_multiple_of(QBLOCK) && !(two && pairs));
        let nb = n_in / QBLOCK;
        let planes = if two { 2 } else { 1 };
        let np = t.div_ceil(2);
        let mut a = Self {
            q: vec![0; if pairs { np * 2 * n_in } else { t * n_in * planes }],
            s: vec![0.0; if pairs { np * nb * 4 } else { t * nb }],
            f: vec![0.0; if pairs { np * nb * 4 } else { t * nb }],
            t,
            n_in,
            two,
            pairs,
        };
        struct P(*mut i8, *mut f32, *mut f32);
        unsafe impl Sync for P {}
        let p = P(a.q.as_mut_ptr(), a.s.as_mut_ptr(), a.f.as_mut_ptr());
        let run = |c: usize| unsafe {
            let p = &p;
            for b in 0..nb {
                let src = x.as_ptr().add(c * n_in + 32 * b);
                let v: [float32x4_t; 8] = std::array::from_fn(|i| vld1q_f32(src.add(4 * i)));
                let mut m = vdupq_n_f32(0.0);
                let mut sum = vdupq_n_f32(0.0);
                for x in v {
                    m = vmaxnmq_f32(m, vabsq_f32(x));
                    sum = vaddq_f32(sum, x);
                }
                let amax = vmaxnmvq_f32(m);
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                let s1 = amax / 127.0;
                let o = p.0.add((c * n_in + 32 * b) * planes);
                // Pair layout: this token's 8 bytes of each 16 of its pair.
                let (pr, k) = (c / 2, c % 2);
                let op = p.0.wrapping_add(pr * 2 * n_in + 64 * b + 8 * k);
                for h in 0..2 {
                    let mut q1 = [vdupq_n_s32(0); 4];
                    let mut q2 = [vdupq_n_s32(0); 4];
                    for i in 0..4 {
                        let xv = v[4 * h + i];
                        q1[i] = vcvtaq_s32_f32(vmulq_n_f32(xv, inv));
                        if two {
                            let e = vfmsq_n_f32(xv, vcvtq_f32_s32(q1[i]), s1);
                            q2[i] = vcvtaq_s32_f32(vmulq_n_f32(e, inv * 254.0));
                        }
                    }
                    let n8 = |q: &[int32x4_t; 4]| {
                        vcombine_s8(
                            vqmovn_s16(vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]))),
                            vqmovn_s16(vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3]))),
                        )
                    };
                    if pairs {
                        let v = n8(&q1);
                        vst1_s8(op.add(32 * h), vget_low_s8(v));
                        vst1_s8(op.add(32 * h + 16), vget_high_s8(v));
                        continue;
                    }
                    vst1q_s8(o.add(16 * planes * h), n8(&q1));
                    if two {
                        vst1q_s8(o.add(32 * h + 16), n8(&q2));
                    }
                }
                if pairs {
                    let (ps, pf) = (p.1.add((pr * nb + b) * 4), p.2.add((pr * nb + b) * 4));
                    let fs = vaddvq_f32(sum);
                    for j in [k, 2 + k] {
                        *ps.add(j) = s1;
                        *pf.add(j) = fs;
                    }
                    continue;
                }
                *p.1.add(c * nb + b) = if two { s1 / 254.0 } else { s1 };
                *p.2.add(c * nb + b) = vaddvq_f32(sum);
            }
        };
        if t == 1 {
            run(0);
        } else {
            crate::pool::run(t, &|c| run(c));
        }
        a
    }

    fn at(&self, c: usize) -> Xs {
        let nb = self.n_in / QBLOCK;
        debug_assert!(!self.pairs);
        Xs {
            q: unsafe { self.q.as_ptr().add(c * self.n_in * if self.two { 2 } else { 1 }) },
            s: unsafe { self.s.as_ptr().add(c * nb) },
            f: unsafe { self.f.as_ptr().add(c * nb) },
            n_in: self.n_in,
        }
    }

    /// The pair layout from pair `p` on.
    fn pair_at(&self, p: usize) -> Xs {
        let nb = self.n_in / QBLOCK;
        debug_assert!(self.pairs);
        Xs {
            q: unsafe { self.q.as_ptr().add(p * 2 * self.n_in) },
            s: unsafe { self.s.as_ptr().add(p * nb * 4) },
            f: unsafe { self.f.as_ptr().add(p * nb * 4) },
            n_in: self.n_in,
        }
    }
}

/// Whether the GEMM is the SMMLA one ([`gemm_rows_mmla`]): the CPU has i8mm and the tuning asks
/// for it (see [`tuning`]).
pub(crate) fn mmla() -> bool {
    tuning().4 && crate::kernels::cpu_has_i8mm()
}

/// `acc` (2 x 2 int32: rows of `a` by rows of `b`) += the 2 x 8 bytes of `a` times the 2 x 8
/// bytes of `b` transposed: 32 products.
#[inline(always)]
unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    std::arch::asm!(
        ".arch_extension i8mm",
        "smmla {r:v}.4s, {a:v}.16b, {b:v}.16b",
        r = inout(vreg) r,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack)
    );
    r
}

/// Row pairs per [`MTile`] (8 rows).
const MRP: usize = 4;

/// Token pairs per kernel call at most (the partial sums of a tile and call are `MRP * MTP`
/// vectors).
const MTP: usize = 4;

/// Eight rows for the SMMLA GEMM, by pairs: `q[(i * MRP + p) * 16 + 8 k + e]` is input
/// `8 i + e` of row `2 p + k`; `ds` and `mn` hold per block of 32 and pair `[r0, r0, r1, r1]`
/// (`[(b * MRP + p) * 4]`, the order of an SMMLA's 2 x 2 sums), `sc` the same per half block
/// (`[((2 b + h) * MRP + p) * 4]`).
struct MTile {
    q: Vec<i8>,
    ds: Vec<f32>,
    mn: Vec<f32>,
    sc: Vec<i32>,
}

impl MTile {
    const fn new() -> Self {
        Self {
            q: Vec::new(),
            ds: Vec::new(),
            mn: Vec::new(),
            sc: Vec::new(),
        }
    }

    /// Rows `r..r + 8` of `m` (rows from `r1` on as zeros).
    unsafe fn fill<M: QMat + ?Sized>(&mut self, m: &M, lay: I8Layout, r: usize, r1: usize, rows: &mut [URow; 4]) {
        let n_in = m.n_in();
        let nb = n_in / QBLOCK;
        if self.q.len() < n_in * 2 * MRP {
            self.q.resize(n_in * 2 * MRP, 0);
            self.ds.resize(nb * 4 * MRP, 0.0);
            self.mn.resize(nb * 4 * MRP, 0.0);
            self.sc.resize(nb * 8 * MRP, 0);
        }
        let q = self.q.as_mut_ptr();
        for p in 0..MRP {
            for (k, u) in rows.iter_mut().enumerate().take(2) {
                u.fit(n_in);
                if r + 2 * p + k < r1 {
                    m.unpack_row_i8(
                        r + 2 * p + k,
                        u.q.as_mut_ptr(),
                        u.ds.as_mut_ptr(),
                        u.mn.as_mut_ptr(),
                        u.sc.as_mut_ptr(),
                    );
                } else {
                    u.q[..n_in].fill(0);
                    u.ds[..nb].fill(0.0);
                    u.mn[..nb].fill(0.0);
                    u.sc[..n_in / 16].fill(0);
                }
            }
            let (u0, u1) = (&rows[0], &rows[1]);
            for o in (0..n_in).step_by(16) {
                let a = vreinterpretq_s64_s8(vld1q_s8(u0.q.as_ptr().add(o)));
                let b = vreinterpretq_s64_s8(vld1q_s8(u1.q.as_ptr().add(o)));
                let d = q.add((o / 8 * MRP + p) * 16);
                vst1q_s8(d, vreinterpretq_s8_s64(vzip1q_s64(a, b)));
                vst1q_s8(d.add(16 * MRP), vreinterpretq_s8_s64(vzip2q_s64(a, b)));
            }
            for b in 0..nb {
                let o = (b * MRP + p) * 4;
                self.ds[o..o + 4].copy_from_slice(&[u0.ds[b], u0.ds[b], u1.ds[b], u1.ds[b]]);
                if lay.min32 {
                    self.mn[o..o + 4].copy_from_slice(&[u0.mn[b], u0.mn[b], u1.mn[b], u1.mn[b]]);
                }
                if lay.sc16 {
                    for h in 0..2 {
                        let o = ((2 * b + h) * MRP + p) * 4;
                        let (s0, s1) = (u0.sc[2 * b + h], u1.sc[2 * b + h]);
                        self.sc[o..o + 4].copy_from_slice(&[s0, s0, s1, s1]);
                    }
                }
            }
        }
    }
}

/// `RP` row pairs of a tile (from pair `p0`) against `TP` token pairs over the blocks of 32
/// `b0..b1`, added to the partial sums at `f` (`[(p * MTP + t) * 4]`: rows by tokens of pair
/// `p` and token pair `t`). Every sum takes the operations of [`block`] in the same order
/// (the integer sum of a block of 32, or of its halves times their scales, converted and
/// multiplied by the row's and the token's scales with one fused multiply-add, then the
/// offset), so the results are those of the SDOT GEMM bit for bit: an SMMLA makes the same
/// integer sums 32 products at a time, for two rows and two tokens.
#[inline(always)]
unsafe fn mkern<const RP: usize, const TP: usize, const SC16: bool, const MIN: bool>(
    tile: &MTile,
    p0: usize,
    x: Xs,
    b0: usize,
    b1: usize,
    f: *mut f32,
) {
    let n_in = x.n_in;
    let nb = n_in / QBLOCK;
    let zero = vdupq_n_s32(0);
    let (tq, tds, tmn, tsc) = (tile.q.as_ptr(), tile.ds.as_ptr(), tile.mn.as_ptr(), tile.sc.as_ptr());
    let mut fv: [[float32x4_t; TP]; RP] = std::array::from_fn(|p| std::array::from_fn(|t| vld1q_f32(f.add(((p0 + p) * MTP + t) * 4))));
    for b in b0..b1 {
        // [half (SC16) or 0][row pair][token pair]
        let mut a = [[[zero; TP]; RP]; 2];
        for h in 0..2 {
            let hs = if SC16 { h } else { 0 };
            for g in 0..2 {
                let i = 4 * b + 2 * h + g;
                let w: [int8x16_t; RP] = std::array::from_fn(|p| vld1q_s8(tq.add((i * MRP + p0 + p) * 16)));
                let xv: [int8x16_t; TP] = std::array::from_fn(|t| vld1q_s8(x.q.add(t * 2 * n_in + 16 * i)));
                for p in 0..RP {
                    for t in 0..TP {
                        a[hs][p][t] = smmla(a[hs][p][t], w[p], xv[t]);
                    }
                }
            }
        }
        let xs: [float32x4_t; TP] = std::array::from_fn(|t| vld1q_f32(x.s.add((t * nb + b) * 4)));
        for p in 0..RP {
            let ds = vld1q_f32(tds.add((b * MRP + p0 + p) * 4));
            for t in 0..TP {
                let sum = if SC16 {
                    let s0 = vld1q_s32(tsc.add((2 * b * MRP + p0 + p) * 4));
                    let s1 = vld1q_s32(tsc.add(((2 * b + 1) * MRP + p0 + p) * 4));
                    vmlaq_s32(vmulq_s32(a[0][p][t], s0), a[1][p][t], s1)
                } else {
                    a[0][p][t]
                };
                let mut v = vfmaq_f32(fv[p][t], vcvtq_f32_s32(sum), vmulq_f32(ds, xs[t]));
                if MIN {
                    let mn = vld1q_f32(tmn.add((b * MRP + p0 + p) * 4));
                    v = vfmsq_f32(v, mn, vld1q_f32(x.f.add((t * nb + b) * 4)));
                }
                fv[p][t] = v;
            }
        }
    }
    for p in 0..RP {
        for t in 0..TP {
            vst1q_f32(f.add(((p0 + p) * MTP + t) * 4), fv[p][t]);
        }
    }
}

/// One block `b` of [`kern`] added into `f`: `PL` planes in the activation layout, `P2` whether
/// this block multiplies the residual one.
#[inline(always)]
unsafe fn block<const G: usize, const T: usize, const SC16: bool, const MIN: bool, const PL: usize, const P2: bool>(
    tiles: &[&Tile; G],
    x: &Xs,
    b: usize,
    f: &mut [[float32x4_t; T]; G],
) {
    let n_in = x.n_in;
    let nb = n_in / QBLOCK;
    let np = if P2 { 2 } else { 1 };
    let zero = vdupq_n_s32(0);
    // [plane][half (SC16) or 0][g][t]
    let mut a = [[[[zero; T]; G]; 2]; 2];
    for h in 0..2 {
        let w: [[int8x16_t; 4]; G] =
            std::array::from_fn(|g| std::array::from_fn(|j| vld1q_s8(tiles[g].q.as_ptr().add(b * 128 + 64 * h + 16 * j))));
        let hs = if SC16 { h } else { 0 };
        for t in 0..T {
            for p in 0..np {
                let xv = vld1q_s8(x.q.add(PL * (t * n_in + 32 * b + 16 * h) + 16 * p));
                for g in 0..G {
                    let mut s = a[p][hs][g][t];
                    s = sdot_lane::<0>(s, w[g][0], xv);
                    s = sdot_lane::<1>(s, w[g][1], xv);
                    s = sdot_lane::<2>(s, w[g][2], xv);
                    s = sdot_lane::<3>(s, w[g][3], xv);
                    a[p][hs][g][t] = s;
                }
            }
        }
    }
    for g in 0..G {
        let ds = vld1q_f32(tiles[g].ds.as_ptr().add(4 * b));
        let (s0, s1) = if SC16 {
            let p = tiles[g].sc.as_ptr().add(8 * b);
            (vld1q_s32(p), vld1q_s32(p.add(4)))
        } else {
            (zero, zero)
        };
        let mn = if MIN {
            vld1q_f32(tiles[g].mn.as_ptr().add(4 * b))
        } else {
            vdupq_n_f32(0.0)
        };
        for t in 0..T {
            let sum = |p: usize| {
                if SC16 {
                    vmlaq_s32(vmulq_s32(a[p][0][g][t], s0), a[p][1][g][t], s1)
                } else {
                    a[p][0][g][t]
                }
            };
            let iv = if !P2 {
                vcvtq_f32_s32(sum(0))
            } else if SC16 {
                vfmaq_n_f32(vcvtq_f32_s32(sum(1)), vcvtq_f32_s32(sum(0)), 254.0)
            } else {
                vcvtq_f32_s32(vmlaq_n_s32(a[1][0][g][t], a[0][0][g][t], 254))
            };
            let xs = *x.s.add(t * nb + b);
            let mut v = vfmaq_f32(f[g][t], iv, vmulq_n_f32(ds, xs));
            if MIN {
                v = vfmsq_f32(v, mn, vdupq_n_f32(*x.f.add(t * nb + b)));
            }
            f[g][t] = v;
        }
    }
}

/// `G` tiles (4 rows each) against `T` tokens over all of `n_in`: `out[g][t]` = 4 rows. `TWO`:
/// activations with a residual plane (see [`I8Act`]).
#[inline(always)]
unsafe fn kern<const G: usize, const T: usize, const SC16: bool, const MIN: bool, const TWO: bool>(
    tiles: [&Tile; G],
    x: Xs,
) -> [[float32x4_t; T]; G] {
    let nb = x.n_in / QBLOCK;
    let mut f = [[vdupq_n_f32(0.0); T]; G];
    for b in 0..nb {
        if TWO {
            block::<G, T, SC16, MIN, 2, true>(&tiles, &x, b, &mut f);
        } else {
            block::<G, T, SC16, MIN, 1, false>(&tiles, &x, b, &mut f);
        }
    }
    f
}

/// `G` tiles (4 rows each) against `T` tokens over the blocks of 32 `b0..b1` only, added to the
/// partial sums at `f[g]` (`T` vectors of 4 rows). A row's sum over all blocks, taken in order
/// from zero through any split into ranges, is the same chain of f32 operations as [`kern`]'s.
#[inline(always)]
unsafe fn kern_range<const G: usize, const T: usize, const SC16: bool, const MIN: bool, const TWO: bool>(
    tiles: [&Tile; G],
    x: Xs,
    b0: usize,
    b1: usize,
    f: [*mut f32; G],
) {
    let mut v: [[float32x4_t; T]; G] = std::array::from_fn(|g| std::array::from_fn(|t| vld1q_f32(f[g].add(4 * t))));
    for b in b0..b1 {
        if TWO {
            block::<G, T, SC16, MIN, 2, true>(&tiles, &x, b, &mut v);
        } else {
            block::<G, T, SC16, MIN, 1, false>(&tiles, &x, b, &mut v);
        }
    }
    for g in 0..G {
        for (t, s) in v[g].iter().enumerate() {
            vst1q_f32(f[g].add(4 * t), *s);
        }
    }
}

/// Row tiles (of 4) per micro-kernel call: one (two measured no faster, three spill).
const G: usize = 1;

/// Cache blocking of the GEMM (none of it changes a result): columns per block
/// (`OXINFER_I8_KB`, a multiple of 32; 0: whole rows, no blocking) and rows per group
/// (`OXINFER_I8_ROWS`, a multiple of 4).
///
/// Without blocking, each tile of 4 rows passes over every token of the job, so every 4 rows
/// read the job's activations again (`tokens * n_in` bytes: megabytes), one byte per four
/// multiply-adds. That is free where they stay in a large shared L2 (Apple Silicon, which keeps
/// that path) and bound by L3 or memory where a core has 0.5-2 MiB of L2 (Cortex-X925 / A725).
/// Blocked, a group of rows is unpacked once and the columns go in blocks: the group's weights
/// of one column block stay in L1 (32 rows x 512 columns: 16 KiB) while the tokens' activations
/// of that block (256 tokens x 512: 128 KiB, in L2) pass once.
///
/// Then the rows per pool job of a prefill matmul on this GEMM (`OXINFER_I8_JOB_ROWS`, a
/// multiple of 4; 0: the driver's blocks, which are sized by weight bytes for the decode
/// kernels, 128 KiB at least: ~100 rows of a 2048-wide Q4_K matrix, 20 jobs for the whole
/// matrix). A job is one row group here, so that cores of unequal speed share a matrix in
/// small pieces: jobs are claimed one at a time, and one taken by a slow core near the end is
/// what the whole matmul waits for. Whether an SDOT kernel call takes two tiles
/// (`OXINFER_I8_PAIR`). Whether the GEMM is the SMMLA one where the
/// CPU has i8mm (`OXINFER_I8_MMLA`), and its row pairs and token pairs per kernel call
/// (`OXINFER_I8_MMLA_RP`: 2 or 4, `OXINFER_I8_MMLA_TP`: 1 to 4).
///
/// These are the defaults on Linux, where they were measured (Cortex-X925 / A725). Elsewhere
/// (Apple's cores, Android, Windows) the whole-row SDOT GEMM stays: on Apple's cores SMMLA
/// issues at half SDOT's rate for twice the products, and nothing of this was measured there.
#[allow(clippy::type_complexity)]
pub(crate) fn tuning() -> (usize, usize, usize, bool, bool, usize, usize) {
    static V: std::sync::OnceLock<(usize, usize, usize, bool, bool, usize, usize)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let e = |k: &str, d: usize, q: usize| {
            let v: usize = std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
            v / q * q
        };
        // Measured on a DGX Spark (20 threads, 256 tokens, Q4_K, TOP/s): 5120 x 17408 at 3.9 with
        // 512 columns and 32-row jobs, 3.6 with 1024 columns, 2.6 with 16-row jobs (a group's
        // activations pass once per group); the 2048-wide shapes within the noise of each other.
        let (kb, job, pair) = if cfg!(target_os = "linux") { (512, 32, 1) } else { (0, 0, 0) };
        (
            e("OXINFER_I8_KB", kb, QBLOCK),
            e("OXINFER_I8_ROWS", 32, 4).max(4),
            e("OXINFER_I8_JOB_ROWS", job, 4),
            e("OXINFER_I8_PAIR", pair, 1) != 0,
            e("OXINFER_I8_MMLA", pair, 1) != 0,
            if e("OXINFER_I8_MMLA_RP", 4, 1) == 2 { 2 } else { 4 },
            e("OXINFER_I8_MMLA_TP", 2, 1).clamp(1, MTP),
        )
    })
}

/// Rows per pool job of a prefill matmul on this GEMM, or None for the driver's blocks (see
/// [`tuning`]).
pub(crate) fn job_rows() -> Option<usize> {
    Some(tuning().2).filter(|&r| r > 0)
}

/// Tokens per micro-kernel call: 4 with one activation plane, 2 with two (more spill, and the
/// per-16 scale formats spill earlier: Q6_K at 4 tokens with two planes ran at 0.65x).
fn tokens_per_call(two: bool) -> usize {
    if two { 2 } else { 4 }
}

/// Per-thread unpacking buffers.
struct Scratch {
    tiles: [Tile; G],
    rows: [URow; 4],
    /// The blocked GEMM's row group, unpacked, and its partial sums `[tile][token group][4][4]`.
    group: Vec<Tile>,
    acc: Vec<f32>,
    /// The SMMLA GEMM's row group.
    mgroup: Vec<MTile>,
}

// Already `const`; clippy misreads the emulated-TLS expansion on Android.
#[allow(clippy::missing_const_for_thread_local)]
mod scratch {
    use std::cell::RefCell;
    thread_local! {
        pub(super) static SCRATCH: RefCell<super::Scratch> = const {
            RefCell::new(super::Scratch {
                tiles: [const { super::Tile::new() }; super::G],
                rows: [const { super::URow::new() }; 4],
                group: Vec::new(),
                acc: Vec::new(),
                mgroup: Vec::new(),
            })
        };
    }
}

/// `y[c * n_out + r]` for tokens `c0..c1` and rows `r0..r1` (`r0` a multiple of 4) of `m`, which
/// must have an [`QMat::i8_layout`].
pub(crate) unsafe fn gemm_rows<M: QMat + ?Sized>(m: &M, xa: &I8Act, c0: usize, c1: usize, r0: usize, r1: usize, y: *mut f32) {
    let (kb, rg, _, pair, _, rp, tp) = tuning();
    if xa.pairs {
        gemm_rows_mmla(m, xa, c0, c1, r0, r1, y, kb, rg, rp, tp)
    } else if kb == 0 {
        gemm_rows_whole(m, xa, c0, c1, r0, r1, y)
    } else {
        gemm_rows_blocked(m, xa, c0, c1, r0, r1, y, kb, rg, pair)
    }
}

/// [`gemm_rows`] by column blocks of `kb` inputs and groups of `rg` rows, with two tiles per
/// kernel call if `pair` (see [`tuning`]).
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_rows_blocked<M: QMat + ?Sized>(
    m: &M,
    xa: &I8Act,
    c0: usize,
    c1: usize,
    r0: usize,
    r1: usize,
    y: *mut f32,
    kb: usize,
    rg: usize,
    pair: bool,
) {
    let lay = m.i8_layout().expect("no int8 GEMM layout");
    let (n_in, n_out) = (m.n_in(), m.n_out());
    debug_assert!(xa.n_in == n_in && c1 <= xa.t);
    if c0 >= c1 || r0 >= r1 {
        return;
    }
    let nb = n_in / QBLOCK;
    let kbb = (kb / QBLOCK).max(1);
    let tmax = tokens_per_call(xa.two);
    let ntg = (c1 - c0).div_ceil(tmax);
    // Two tiles per call, 8 chains of SDOTs instead of 4, where their sums fit the registers
    // (one activation plane, one integer sum per block of 32): +10% on a Cortex-X925, +4% on
    // an A725 (Q4_K, Q8_0); with a sum per 16 inputs (Q6_K, Q3_K) it spills, at 0.65x.
    let pair = pair && !xa.two && !lay.sc16;
    scratch::SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let Scratch { rows, group, acc, .. } = &mut *sc;
        let nt_max = rg.div_ceil(4);
        if group.len() < nt_max {
            group.resize_with(nt_max, Tile::new);
        }
        let mut r = r0;
        while r < r1 {
            let nt = (r1 - r).div_ceil(4).min(nt_max);
            for (i, tile) in group.iter_mut().enumerate().take(nt) {
                tile.fill(m, lay, r + 4 * i, rows);
            }
            acc.clear();
            acc.resize(nt * ntg * 16, 0.0);
            let ap = acc.as_mut_ptr();
            let mut b0 = 0;
            while b0 < nb {
                let b1 = (b0 + kbb).min(nb);
                for tg in 0..ntg {
                    let c = c0 + tg * tmax;
                    let tc = (c1 - c).min(tmax);
                    let x = xa.at(c);
                    let fp = |i: usize| ap.add((i * ntg + tg) * 16);
                    macro_rules! go {
                        ($G:literal, $T:literal, $tiles:expr, $f:expr) => {
                            match (lay.sc16, lay.min32, xa.two) {
                                (false, false, false) => kern_range::<$G, $T, false, false, false>($tiles, x, b0, b1, $f),
                                (false, true, false) => kern_range::<$G, $T, false, true, false>($tiles, x, b0, b1, $f),
                                (true, false, false) => kern_range::<$G, $T, true, false, false>($tiles, x, b0, b1, $f),
                                (true, true, false) => kern_range::<$G, $T, true, true, false>($tiles, x, b0, b1, $f),
                                (false, false, true) => kern_range::<$G, $T, false, false, true>($tiles, x, b0, b1, $f),
                                (false, true, true) => kern_range::<$G, $T, false, true, true>($tiles, x, b0, b1, $f),
                                (true, false, true) => kern_range::<$G, $T, true, false, true>($tiles, x, b0, b1, $f),
                                (true, true, true) => kern_range::<$G, $T, true, true, true>($tiles, x, b0, b1, $f),
                            }
                        };
                    }
                    let mut i = 0;
                    while pair && tc == 4 && i + 2 <= nt {
                        go!(2, 4, [&group[i], &group[i + 1]], [fp(i), fp(i + 1)]);
                        i += 2;
                    }
                    while i < nt {
                        let (tiles, f) = ([&group[i]], [fp(i)]);
                        match tc {
                            1 => go!(1, 1, tiles, f),
                            2 => go!(1, 2, tiles, f),
                            3 => go!(1, 3, tiles, f),
                            4 => go!(1, 4, tiles, f),
                            _ => unreachable!(),
                        }
                        i += 1;
                    }
                }
                b0 = b1;
            }
            for tg in 0..ntg {
                let c = c0 + tg * tmax;
                for t in 0..(c1 - c).min(tmax) {
                    let o = y.add((c + t) * n_out + r);
                    for i in 0..nt {
                        let a = acc.as_ptr().add((i * ntg + tg) * 16 + 4 * t);
                        if r + 4 * i + 4 <= r1 {
                            vst1q_f32(o.add(4 * i), vld1q_f32(a));
                        } else {
                            for j in 0..r1 - (r + 4 * i) {
                                *o.add(4 * i + j) = *a.add(j);
                            }
                        }
                    }
                }
            }
            r += 4 * nt;
        }
    });
}

/// [`gemm_rows`] on SMMLA (activations in the pair layout), by column blocks of `kb` inputs
/// (0: whole rows) and groups of `rg` rows: tiles of 8 rows, `rp` row pairs and up to `tp`
/// token pairs per kernel call. A pair that straddles `c0` or `c1` is computed whole and its
/// outside token dropped.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_rows_mmla<M: QMat + ?Sized>(
    m: &M,
    xa: &I8Act,
    c0: usize,
    c1: usize,
    r0: usize,
    r1: usize,
    y: *mut f32,
    kb: usize,
    rg: usize,
    rp: usize,
    tp: usize,
) {
    let lay = m.i8_layout().expect("no int8 GEMM layout");
    let (n_in, n_out) = (m.n_in(), m.n_out());
    debug_assert!(xa.n_in == n_in && c1 <= xa.t && xa.pairs);
    if c0 >= c1 || r0 >= r1 {
        return;
    }
    let nb = n_in / QBLOCK;
    let kbb = if kb == 0 { nb } else { (kb / QBLOCK).max(1) };
    let (p0, p1) = (c0 / 2, c1.div_ceil(2));
    let ntg = (p1 - p0).div_ceil(tp);
    const AV: usize = MRP * MTP * 4;
    scratch::SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let Scratch { rows, mgroup, acc, .. } = &mut *sc;
        let nt_max = rg.div_ceil(2 * MRP);
        if mgroup.len() < nt_max {
            mgroup.resize_with(nt_max, MTile::new);
        }
        let mut r = r0;
        while r < r1 {
            let nt = (r1 - r).div_ceil(2 * MRP).min(nt_max);
            for (i, tile) in mgroup.iter_mut().enumerate().take(nt) {
                tile.fill(m, lay, r + 2 * MRP * i, r1, rows);
            }
            acc.clear();
            acc.resize(nt * ntg * AV, 0.0);
            let ap = acc.as_mut_ptr();
            let mut b0 = 0;
            while b0 < nb {
                let b1 = (b0 + kbb).min(nb);
                for tg in 0..ntg {
                    let pa = p0 + tg * tp;
                    let tc = (p1 - pa).min(tp);
                    let x = xa.pair_at(pa);
                    for (i, tile) in mgroup.iter().enumerate().take(nt) {
                        let f = ap.add((i * ntg + tg) * AV);
                        macro_rules! go {
                            ($RP:literal, $TP:literal, $q:expr) => {
                                match (lay.sc16, lay.min32) {
                                    (false, false) => mkern::<$RP, $TP, false, false>(tile, $q, x, b0, b1, f),
                                    (false, true) => mkern::<$RP, $TP, false, true>(tile, $q, x, b0, b1, f),
                                    (true, false) => mkern::<$RP, $TP, true, false>(tile, $q, x, b0, b1, f),
                                    (true, true) => mkern::<$RP, $TP, true, true>(tile, $q, x, b0, b1, f),
                                }
                            };
                        }
                        macro_rules! tokens {
                            ($RP:literal, $q:expr) => {
                                match tc {
                                    1 => go!($RP, 1, $q),
                                    2 => go!($RP, 2, $q),
                                    3 => go!($RP, 3, $q),
                                    4 => go!($RP, 4, $q),
                                    _ => unreachable!(),
                                }
                            };
                        }
                        // Row pairs past `r1` are zeros: skipped.
                        let live = (r1 - (r + 2 * MRP * i)).div_ceil(2).min(MRP);
                        if rp == 4 && live > 2 {
                            tokens!(4, 0);
                        } else {
                            tokens!(2, 0);
                            if live > 2 {
                                tokens!(2, 2);
                            }
                        }
                    }
                }
                b0 = b1;
            }
            for tg in 0..ntg {
                let pa = p0 + tg * tp;
                for t in 0..(p1 - pa).min(tp) {
                    for i in 0..nt {
                        for p in 0..MRP {
                            let a = acc.as_ptr().add((i * ntg + tg) * AV + (p * MTP + t) * 4);
                            for j in 0..4 {
                                let (row, c) = (r + 2 * MRP * i + 2 * p + j / 2, 2 * (pa + t) + j % 2);
                                if row < r1 && c >= c0 && c < c1 {
                                    *y.add(c * n_out + row) = *a.add(j);
                                }
                            }
                        }
                    }
                }
            }
            r += 2 * MRP * nt;
        }
    });
}

/// [`gemm_rows`] with each tile of 4 rows over every token in turn (whole rows, no blocking).
unsafe fn gemm_rows_whole<M: QMat + ?Sized>(m: &M, xa: &I8Act, c0: usize, c1: usize, r0: usize, r1: usize, y: *mut f32) {
    let lay = m.i8_layout().expect("no int8 GEMM layout");
    let n_in = m.n_in();
    let n_out = m.n_out();
    debug_assert!(xa.n_in == n_in && c1 <= xa.t);
    if c0 >= c1 {
        return;
    }
    scratch::SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let Scratch { tiles: tl, rows, .. } = &mut *sc;
        let tmax = tokens_per_call(xa.two);
        let mut r = r0;
        while r < r1 {
            let groups = (r1 - r).div_ceil(4).min(G);
            for (g, tile) in tl.iter_mut().enumerate().take(groups) {
                tile.fill(m, lay, r + 4 * g, rows);
            }
            let mut c = c0;
            while c < c1 {
                let tc = (c1 - c).min(tmax);
                let x = xa.at(c);
                let store = |g: usize, t: usize, v: float32x4_t| {
                    let rr = r + 4 * g;
                    let o = y.add((c + t) * n_out + rr);
                    if rr + 4 <= r1 {
                        vst1q_f32(o, v);
                    } else {
                        let a: [f32; 4] = std::mem::transmute(v);
                        for (i, av) in a.iter().enumerate().take(r1.saturating_sub(rr)) {
                            *o.add(i) = *av;
                        }
                    }
                };
                macro_rules! go {
                    ($G:literal, $T:literal) => {{
                        let tiles: [&Tile; $G] = std::array::from_fn(|g| &tl[g]);
                        let f = match (lay.sc16, lay.min32, xa.two) {
                            (false, false, false) => kern::<$G, $T, false, false, false>(tiles, x),
                            (false, true, false) => kern::<$G, $T, false, true, false>(tiles, x),
                            (true, false, false) => kern::<$G, $T, true, false, false>(tiles, x),
                            (true, true, false) => kern::<$G, $T, true, true, false>(tiles, x),
                            (false, false, true) => kern::<$G, $T, false, false, true>(tiles, x),
                            (false, true, true) => kern::<$G, $T, false, true, true>(tiles, x),
                            (true, false, true) => kern::<$G, $T, true, false, true>(tiles, x),
                            (true, true, true) => kern::<$G, $T, true, true, true>(tiles, x),
                        };
                        for (g, fg) in f.iter().enumerate() {
                            for (t, v) in fg.iter().enumerate() {
                                store(g, t, *v);
                            }
                        }
                    }};
                }
                match (groups, tc) {
                    (1, 1) => go!(1, 1),
                    (1, 2) => go!(1, 2),
                    (1, 3) => go!(1, 3),
                    (1, 4) => go!(1, 4),
                    _ => unreachable!(),
                }
                c += tc;
            }
            r += 4 * groups;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
    use crate::quant::quantize_act;

    /// The GEMM over row and token counts that leave partial tiles: with one plane against the
    /// one-token decode kernels (same int8 activations; only the f32 summation order differs),
    /// with two against the dequantized weights times the f32 activations.
    fn check<M: QMat>(m: &M, name: &str) {
        m.i8_layout().expect("layout");
        let (n_out, n_in, t) = (m.n_out(), m.n_in(), 11);
        let x: Vec<f32> = (0..t * n_in)
            .map(|i| (((i * 7919) % 2001) as f32 / 1000.0 - 1.0) * if i % 61 == 0 { 9.0 } else { 1.0 })
            .collect();
        let xa = quantize_act(&x, t, n_in);
        let mut dec = vec![0f32; t * n_out];
        for c in 0..t {
            unsafe { m.rows_t1(&xa, c, dec.as_mut_ptr().add(c * n_out), 0, n_out) };
        }
        let mut w = vec![0f32; n_out * n_in];
        m.dequantize_rows(0, n_out, &mut w);
        let exact: Vec<f32> = (0..t * n_out)
            .map(|i| {
                let (c, r) = (i / n_out, i % n_out);
                (0..n_in).map(|k| w[r * n_in + k] as f64 * x[c * n_in + k] as f64).sum::<f64>() as f32
            })
            .collect();
        for (two, want, tol) in [(false, &dec, 2e-5), (true, &exact, 1e-4)] {
            let ga = I8Act::with_layout(&x, t, n_in, two, false);
            let mut got = vec![f32::NAN; t * n_out];
            // Rows 0..8 for tokens 0..5 then 5..11, the other rows in one call (row starts are
            // multiples of 4).
            unsafe {
                gemm_rows_whole(m, &ga, 0, 5, 0, 8, got.as_mut_ptr());
                gemm_rows_whole(m, &ga, 5, t, 0, 8, got.as_mut_ptr());
                gemm_rows_whole(m, &ga, 0, t, 8, n_out, got.as_mut_ptr());
            }
            // The SMMLA GEMM gives the SDOT GEMM's bits, whatever its blocks and tiles and
            // through token ranges that cut pairs.
            if !two && crate::kernels::cpu_has_i8mm() {
                let gp = I8Act::with_layout(&x, t, n_in, false, true);
                for (kb, rg, rp, tp) in [(0, 8, 4, 2), (32, 8, 2, 1), (96, 16, 4, 4), (1024, 32, 2, 3), (64, 24, 4, 3)] {
                    let mut mm = vec![f32::NAN; t * n_out];
                    unsafe {
                        gemm_rows_mmla(m, &gp, 0, 5, 0, 8, mm.as_mut_ptr(), kb, rg, rp, tp);
                        gemm_rows_mmla(m, &gp, 5, t, 0, 8, mm.as_mut_ptr(), kb, rg, rp, tp);
                        gemm_rows_mmla(m, &gp, 0, 3, 8, n_out, mm.as_mut_ptr(), kb, rg, rp, tp);
                        gemm_rows_mmla(m, &gp, 3, t, 8, n_out, mm.as_mut_ptr(), kb, rg, rp, tp);
                    }
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&mm), bits(&got), "{name} SMMLA kb={kb} rg={rg} rp={rp} tp={tp}");
                }
            }
            // Column blocks and row groups do not change a bit.
            for (kb, rg, pair) in [(32, 4, false), (96, 8, true), (1024, 32, true), (64, 12, true)] {
                let mut blocked = vec![f32::NAN; t * n_out];
                let mut whole = vec![f32::NAN; t * n_out];
                unsafe {
                    gemm_rows_blocked(m, &ga, 0, 5, 0, 8, blocked.as_mut_ptr(), kb, rg, pair);
                    gemm_rows_blocked(m, &ga, 5, t, 0, 8, blocked.as_mut_ptr(), kb, rg, pair);
                    gemm_rows_blocked(m, &ga, 0, t, 8, n_out, blocked.as_mut_ptr(), kb, rg, pair);
                    gemm_rows_whole(m, &ga, 0, t, 0, n_out, whole.as_mut_ptr());
                }
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&blocked), bits(&whole), "{name} two={two} kb={kb} rg={rg} pair={pair}");
            }
            let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol * scale,
                    "{name} two={two} [{}, {}]: {a} vs {b} (scale {scale})",
                    i / n_out,
                    i % n_out
                );
            }
        }
    }

    #[test]
    fn gguf_matches_decode_kernels() {
        let (n_out, n_in) = (15, 512);
        let mut n = 0;
        for kind in GgmlType::ALL {
            let mut seed = 3u64;
            let bytes = random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
            let m = GgufMatrix::from_bytes(kind, &bytes, n_out, n_in).unwrap();
            if m.i8_layout().is_some() {
                check(&m, kind.name());
                n += 1;
            }
        }
        assert_eq!(n, 20, "formats");
    }

    #[test]
    fn oxinfer_formats_match_decode_kernels() {
        let (n_out, n_in) = (13, 256);
        let w: Vec<f32> = (0..n_out * n_in).map(|i| ((i * 104729) % 1999) as f32 / 999.0 - 1.0).collect();
        check(&crate::quant::quantize_f32(&w, n_out, n_in), "Q8");
        let mut q4 = crate::quant::quantize_q4_f32(&w, n_out, n_in);
        check(&q4, "Q4");
        let group = 64;
        let mut v = crate::quant::PageVec::<u16>::zeroed(n_out * n_in / group);
        for (i, o) in v.iter_mut().enumerate() {
            *o = ((((i * 37) % 11) as f32 * 0.01 - 0.05).to_bits() >> 16) as u16;
        }
        q4.offsets = Some(crate::quant::Offsets { v: v.into(), group });
        check(&q4, "Q4 affine");
    }
}
