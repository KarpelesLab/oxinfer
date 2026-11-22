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
pub(crate) struct I8Act {
    q: Vec<i8>,
    s: Vec<f32>,
    f: Vec<f32>,
    t: usize,
    n_in: usize,
    two: bool,
}

impl I8Act {
    pub(crate) fn new(x: &[f32], t: usize, n_in: usize, two: bool) -> Self {
        assert_eq!(x.len(), t * n_in);
        assert!(n_in.is_multiple_of(QBLOCK));
        let nb = n_in / QBLOCK;
        let planes = if two { 2 } else { 1 };
        let mut a = Self {
            q: vec![0; t * n_in * planes],
            s: vec![0.0; t * nb],
            f: vec![0.0; t * nb],
            t,
            n_in,
            two,
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
                    vst1q_s8(o.add(16 * planes * h), n8(&q1));
                    if two {
                        vst1q_s8(o.add(32 * h + 16), n8(&q2));
                    }
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
        Xs {
            q: unsafe { self.q.as_ptr().add(c * self.n_in * if self.two { 2 } else { 1 }) },
            s: unsafe { self.s.as_ptr().add(c * nb) },
            f: unsafe { self.f.as_ptr().add(c * nb) },
            n_in: self.n_in,
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

/// Row tiles (of 4) per micro-kernel call: one (two measured no faster, three spill).
const G: usize = 1;

/// Tokens per micro-kernel call: 4 with one activation plane, 2 with two (more spill, and the
/// per-16 scale formats spill earlier: Q6_K at 4 tokens with two planes ran at 0.65x).
fn tokens_per_call(two: bool) -> usize {
    if two { 2 } else { 4 }
}

/// Per-thread unpacking buffers.
struct Scratch {
    tiles: [Tile; G],
    rows: [URow; 4],
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
            })
        };
    }
}

/// `y[c * n_out + r]` for tokens `c0..c1` and rows `r0..r1` (`r0` a multiple of 4) of `m`, which
/// must have an [`QMat::i8_layout`].
pub(crate) unsafe fn gemm_rows<M: QMat + ?Sized>(m: &M, xa: &I8Act, c0: usize, c1: usize, r0: usize, r1: usize, y: *mut f32) {
    let lay = m.i8_layout().expect("no int8 GEMM layout");
    let n_in = m.n_in();
    let n_out = m.n_out();
    debug_assert!(xa.n_in == n_in && c1 <= xa.t);
    if c0 >= c1 {
        return;
    }
    scratch::SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let Scratch { tiles: tl, rows } = &mut *sc;
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
            let ga = I8Act::new(&x, t, n_in, two);
            let mut got = vec![f32::NAN; t * n_out];
            // Rows 0..8 for tokens 0..5 then 5..11, the other rows in one call (row starts are
            // multiples of 4).
            unsafe {
                gemm_rows(m, &ga, 0, 5, 0, 8, got.as_mut_ptr());
                gemm_rows(m, &ga, 5, t, 0, 8, got.as_mut_ptr());
                gemm_rows(m, &ga, 0, t, 8, n_out, got.as_mut_ptr());
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
