//! CPU compute kernels: bf16 weights, f32 activations, multi-threaded via the spin pool.
//!
//! # Safety
//! The `unsafe fn` kernels here and in the `simd` modules share one contract: every pointer
//! argument must be valid for the extent implied by the dimensions passed alongside it (rows times
//! `n_in` weights, `t` or `tc` activation rows of `n_in`, outputs indexed as documented on each
//! function), and output ranges written concurrently by different threads must not overlap. They
//! are documented once here rather than on every tile variant.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use crate::pool;
use crate::prof::{self, Cat};
use crate::quant::{PairAct, Q4Matrix, QAct, QBLOCK, QMatrix, dequantize_rows, dequantize_rows_q4, quantize_act};

#[inline(always)]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub fn bf16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = bf16_to_f32(s);
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct JobRef<'a> {
    w: &'a [u16],
    n_out: usize,
    y: SendPtr,
}

// Already `const`; clippy misreads the emulated-TLS expansion on Android.
#[allow(clippy::missing_const_for_thread_local)]
mod scratch {
    use std::cell::RefCell;
    thread_local! {
        pub(super) static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }
}
use scratch::SCRATCH;

/// y[t][o] = sum_i w[o][i] * x[t][i]
///
/// `w` is `[n_out, n_in]` bf16 row-major (PyTorch Linear layout),
/// `x` is `[t, n_in]` f32, `y` is `[t, n_out]` f32.
#[allow(dead_code)]
pub fn matmul(w: &[u16], n_out: usize, n_in: usize, x: &[f32], t: usize, y: &mut [f32]) {
    matmul_multi(&mut [(w, n_out, y)], n_in, x, t);
}

/// Several matmuls sharing the same input `x`, executed as one parallel region.
/// Each job is `(w [n_out, n_in], n_out, y [t, n_out])`.
pub fn matmul_multi(jobs: &mut [(&[u16], usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
    assert_eq!(x.len(), t * n_in, "input shape mismatch");
    if t == 0 {
        return;
    }
    let mut total_bytes = 0usize;
    let refs: Vec<JobRef> = jobs
        .iter_mut()
        .map(|(w, n_out, y)| {
            assert_eq!(w.len(), *n_out * n_in, "weight shape mismatch");
            assert_eq!(y.len(), t * *n_out, "output shape mismatch");
            total_bytes += w.len() * 2;
            JobRef {
                w,
                n_out: *n_out,
                y: SendPtr(y.as_mut_ptr()),
            }
        })
        .collect();
    let _timer = prof::start(Cat::Matmul);
    prof::add_bytes(total_bytes);

    // Block size: ~total/(8*threads) bytes of weights, but at least 128 KiB, in multiples of 4 rows.
    let nthreads = pool::threads();
    let target = (total_bytes / (nthreads * 8)).max(128 * 1024);
    let rows_per_block = (target / (n_in * 2)).max(16) / 16 * 16;
    let mut blocks: Vec<(u32, u32, u32)> = Vec::new();
    for (j, job) in refs.iter().enumerate() {
        let mut r = 0;
        while r < job.n_out {
            let r1 = (r + rows_per_block).min(job.n_out);
            blocks.push((j as u32, r as u32, r1 as u32));
            r = r1;
        }
    }
    // Prefill: pack activations as [tile][k][8 tokens] so the GEMM kernel can broadcast weights by lane.
    let xt = if t >= 8 { Some(pack_xt(x, t, n_in)) } else { None };
    let (tchunks, tc) = token_split(blocks.len(), t, nthreads);
    pool::run(blocks.len() * tchunks, &|b| {
        let (j, r0, r1) = blocks[b / tchunks];
        let c0 = (b % tchunks) * tc;
        if c0 >= t {
            return;
        }
        let c1 = (c0 + tc).min(t);
        let job = refs[j as usize];
        // A token range is a contiguous slice of x, whole tiles of xt, and an offset into y.
        let xs = &x[c0 * n_in..c1 * n_in];
        let xts = xt.as_deref().map(|xt| &xt[c0 / TN * n_in * TN..]);
        unsafe {
            matmul_block(
                job.w,
                job.n_out,
                n_in,
                xs,
                xts,
                c1 - c0,
                r0 as usize,
                r1 as usize,
                job.y.get().add(c0 * job.n_out),
            )
        }
    });
}

/// Work items of a grouped matmul: `(job, row0, row1, token0, token1)`, row blocks of about
/// `total_bytes / (8 * threads)` bytes and token chunks of at most 64 (whole TN tiles).
fn grouped_blocks(segs: &[(usize, usize)], n_out: usize, row_bytes: usize) -> Vec<(u32, u32, u32, u32, u32)> {
    let total = segs.iter().filter(|s| s.1 > 0).count() * n_out * row_bytes;
    let target = (total / (pool::threads() * 8)).max(64 * 1024);
    let rows_per_block = (target / row_bytes).max(16) / 16 * 16;
    let mut blocks = Vec::new();
    for (j, &(s, len)) in segs.iter().enumerate() {
        debug_assert!(s.is_multiple_of(TN) && len.is_multiple_of(TN));
        let mut c = s;
        while c < s + len {
            let c1 = (c + 64).min(s + len);
            let mut r = 0;
            while r < n_out {
                let r1 = (r + rows_per_block).min(n_out);
                blocks.push((j as u32, r as u32, r1 as u32, c as u32, c1 as u32));
                r = r1;
            }
            c = c1;
        }
    }
    blocks
}

/// Grouped matmuls over row segments of one input (MoE experts, each with the tokens routed to
/// it): job `j` multiplies rows `segs[j].0 .. + segs[j].1` of `x` (`[_, n_in]`) by `ws[j]`
/// (`[n_out, n_in]`) into the same rows of `y` (`[_, n_out]`), all in one parallel region.
/// Segments start and end on multiples of TN tokens (pad them with zero rows).
pub fn matmul_grouped(ws: &[&[u16]], segs: &[(usize, usize)], n_out: usize, n_in: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(ws.len(), segs.len());
    let rows = x.len() / n_in;
    assert!(y.len() == rows * n_out && rows.is_multiple_of(TN));
    let _timer = prof::start(Cat::Matmul);
    prof::add_bytes(ws.iter().zip(segs).filter(|(_, s)| s.1 > 0).map(|(w, _)| w.len() * 2).sum());
    let blocks = grouped_blocks(segs, n_out, n_in * 2);
    let xt = pack_xt(x, rows, n_in);
    let yp = SendPtr(y.as_mut_ptr());
    pool::run(blocks.len(), &|b| {
        let (j, r0, r1, c0, c1) = blocks[b];
        let (c0, c1) = (c0 as usize, c1 as usize);
        unsafe {
            matmul_block(
                ws[j as usize],
                n_out,
                n_in,
                &x[c0 * n_in..c1 * n_in],
                Some(&xt[c0 / TN * n_in * TN..]),
                c1 - c0,
                r0 as usize,
                r1 as usize,
                yp.get().add(c0 * n_out),
            )
        }
    });
}

/// [`matmul_grouped`] with quantized weights.
pub fn matmul_quant_grouped<M: QMat>(ms: &[&M], segs: &[(usize, usize)], x: &[f32], y: &mut [f32]) {
    assert_eq!(ms.len(), segs.len());
    let (n_out, n_in) = (ms[0].n_out(), ms[0].n_in());
    let rows = x.len() / n_in;
    assert!(y.len() == rows * n_out && rows.is_multiple_of(TN));
    let _timer = prof::start(Cat::Matmul);
    prof::add_bytes(ms.iter().zip(segs).filter(|(_, s)| s.1 > 0).map(|(m, _)| m.bytes()).sum());
    let blocks = grouped_blocks(segs, n_out, (ms[0].bytes() / n_out).max(1));
    let xt = pack_xt(x, rows, n_in);
    let yp = SendPtr(y.as_mut_ptr());
    pool::run(blocks.len(), &|b| {
        let (j, r0, r1, c0, c1) = blocks[b];
        let (c0, c1) = (c0 as usize, c1 as usize);
        unsafe {
            matmul_quant_block_prefill(
                ms[j as usize],
                &x[c0 * n_in..c1 * n_in],
                &xt[c0 / TN * n_in * TN..],
                c1 - c0,
                r0 as usize,
                r1 as usize,
                yp.get().add(c0 * n_out),
            )
        }
    });
}

/// When the weights split into too few row blocks to occupy the pool (small matrices over many
/// tokens: convolutions, attention products), also split the tokens, in whole TN-token tiles.
/// Returns the number of token chunks and their length.
fn token_split(row_blocks: usize, t: usize, nthreads: usize) -> (usize, usize) {
    if t < 64 || row_blocks >= nthreads * 2 {
        return (1, t);
    }
    let chunks = (nthreads * 3).div_ceil(row_blocks).min(t / 32).max(1);
    let len = t.div_ceil(chunks).next_multiple_of(TN);
    (t.div_ceil(len), len)
}

/// Tokens per prefill GEMM tile (8 rows x 8 tokens = 16 accumulators; 12 tokens measured slower, register pressure).
pub const TN: usize = 8;

/// Packs `x` ([t, n_in]) into token tiles of TN: `xt[tile][k][c] = x[tile*TN + c][k]`, zero padded.
fn pack_xt(x: &[f32], t: usize, n_in: usize) -> Vec<f32> {
    let tiles = t.div_ceil(TN);
    let mut xt = vec![0f32; tiles * n_in * TN];
    let p = SendPtr(xt.as_mut_ptr());
    pool::run(tiles, &|tile| {
        // SAFETY: each tile writes a disjoint range of `xt`.
        let dst = unsafe { std::slice::from_raw_parts_mut(p.get().add(tile * n_in * TN), n_in * TN) };
        for c in 0..TN {
            let tok = tile * TN + c;
            if tok >= t {
                break;
            }
            let src = &x[tok * n_in..(tok + 1) * n_in];
            for k in 0..n_in {
                dst[k * TN + c] = src[k];
            }
        }
    });
    xt
}

#[cfg(target_arch = "x86_64")]
#[path = "kernels_x86.rs"]
pub mod simd;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[path = "kernels_wasm.rs"]
pub mod simd;

#[cfg(target_arch = "aarch64")]
pub mod simd {
    use std::arch::aarch64::*;

    pub trait Load8: Copy {
        unsafe fn load8(p: *const Self) -> (float32x4_t, float32x4_t);
    }
    impl Load8 for u16 {
        #[inline(always)]
        unsafe fn load8(p: *const u16) -> (float32x4_t, float32x4_t) {
            let w = vld1q_u16(p);
            let z = vdupq_n_u16(0);
            (vreinterpretq_f32_u16(vzip1q_u16(z, w)), vreinterpretq_f32_u16(vzip2q_u16(z, w)))
        }
    }
    impl Load8 for f32 {
        #[inline(always)]
        unsafe fn load8(p: *const f32) -> (float32x4_t, float32x4_t) {
            (vld1q_f32(p), vld1q_f32(p.add(4)))
        }
    }

    /// Dot products of R weight rows against TC activation rows.
    /// Each iteration consumes 8*G elements per row; A independent accumulators per output
    /// hide FMA latency. `n` must be a multiple of 8*G.
    #[inline(always)]
    pub unsafe fn kernel<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
        w: *const W,
        ws: usize,
        x: *const f32,
        xs: usize,
        n: usize,
    ) -> [[f32; TC]; R] {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [[[zero; A]; TC]; R];
        let mut i = 0;
        while i < n {
            let mut wv = [[(zero, zero); G]; R];
            for r in 0..R {
                for g in 0..G {
                    wv[r][g] = W::load8(w.add(r * ws + i + 8 * g));
                }
            }
            for c in 0..TC {
                for g in 0..G {
                    let xl = vld1q_f32(x.add(c * xs + i + 8 * g));
                    let xh = vld1q_f32(x.add(c * xs + i + 8 * g + 4));
                    for r in 0..R {
                        acc[r][c][(2 * g) % A] = vfmaq_f32(acc[r][c][(2 * g) % A], wv[r][g].0, xl);
                        acc[r][c][(2 * g + 1) % A] = vfmaq_f32(acc[r][c][(2 * g + 1) % A], wv[r][g].1, xh);
                    }
                }
            }
            i += 8 * G;
        }
        let mut out = [[0f32; TC]; R];
        for r in 0..R {
            for c in 0..TC {
                let mut s = acc[r][c][0];
                for a in 1..A {
                    s = vaddq_f32(s, acc[r][c][a]);
                }
                out[r][c] = vaddvq_f32(s);
            }
        }
        out
    }

    /// Runs the kernel over every R-row group in `r0..r1` (the count must be a multiple of R)
    /// for tokens `c0..c0+TC`, storing to `y[(c0+c)*n_out + r]`. Kept out of line so each tile
    /// shape gets its own register allocation.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_tiles<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
        w: *const W,
        n_in: usize,
        x: *const f32,
        y: *mut f32,
        n_out: usize,
        r0: usize,
        r1: usize,
        c0: usize,
    ) {
        let mut r = r0;
        while r + R <= r1 {
            let o = kernel::<W, R, TC, G, A>(w.add(r * n_in), n_in, x.add(c0 * n_in), n_in, n_in);
            for rr in 0..R {
                for c in 0..TC {
                    *y.add((c0 + c) * n_out + r + rr) = o[rr][c];
                }
            }
            r += R;
        }
    }

    /// Loads 4 consecutive weights as f32.
    pub trait LoadW4: Copy {
        unsafe fn load4(p: *const Self) -> float32x4_t;
    }
    impl LoadW4 for u16 {
        #[inline(always)]
        unsafe fn load4(p: *const u16) -> float32x4_t {
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(p)))
        }
    }
    impl LoadW4 for f32 {
        #[inline(always)]
        unsafe fn load4(p: *const f32) -> float32x4_t {
            vld1q_f32(p)
        }
    }

    /// Prefill GEMM micro-kernel: 8 weight rows (row-major, starting at `w`) times TN=8 tokens
    /// from a packed panel `xt` ([n_in][8]). Weights are loaded 4 at a time and broadcast by lane;
    /// accumulators are vectors over tokens. Stores rows `rb..rb+8` for tokens `c0..c0+c_valid`.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_8x8<W: LoadW4>(
        w: *const W,
        n_in: usize,
        xt: *const f32,
        y: *mut f32,
        n_out: usize,
        rb: usize,
        c0: usize,
        c_valid: usize,
    ) {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [[zero; 2]; 8];
        let mut k = 0;
        while k < n_in {
            let xp = xt.add(k * 8);
            let x0a = vld1q_f32(xp);
            let x0b = vld1q_f32(xp.add(4));
            let x1a = vld1q_f32(xp.add(8));
            let x1b = vld1q_f32(xp.add(12));
            let x2a = vld1q_f32(xp.add(16));
            let x2b = vld1q_f32(xp.add(20));
            let x3a = vld1q_f32(xp.add(24));
            let x3b = vld1q_f32(xp.add(28));
            for r in 0..8 {
                let wq = W::load4(w.add(r * n_in + k));
                acc[r][0] = vfmaq_laneq_f32::<0>(acc[r][0], x0a, wq);
                acc[r][1] = vfmaq_laneq_f32::<0>(acc[r][1], x0b, wq);
                acc[r][0] = vfmaq_laneq_f32::<1>(acc[r][0], x1a, wq);
                acc[r][1] = vfmaq_laneq_f32::<1>(acc[r][1], x1b, wq);
                acc[r][0] = vfmaq_laneq_f32::<2>(acc[r][0], x2a, wq);
                acc[r][1] = vfmaq_laneq_f32::<2>(acc[r][1], x2b, wq);
                acc[r][0] = vfmaq_laneq_f32::<3>(acc[r][0], x3a, wq);
                acc[r][1] = vfmaq_laneq_f32::<3>(acc[r][1], x3b, wq);
            }
            k += 4;
        }
        let mut tmp = [0f32; 8];
        for r in 0..8 {
            vst1q_f32(tmp.as_mut_ptr(), acc[r][0]);
            vst1q_f32(tmp.as_mut_ptr().add(4), acc[r][1]);
            for c in 0..c_valid {
                *y.add((c0 + c) * n_out + rb + r) = tmp[c];
            }
        }
    }

    /// Q8 dot products of R quantized rows (int8 + per-32 scales) with one quantized activation row.
    /// Four blocks (128 weights) per step: SDOT into int32, pairwise-add into one vector of block
    /// sums, convert, and FMA with the product of weight and activation scales.
    #[inline(never)]
    pub unsafe fn q8_dot_rows<const R: usize>(q: *const i8, s: *const f32, n_in: usize, xq: *const i8, xs: *const f32) -> [f32; R] {
        let bpr = n_in / 32;
        let zero = vdupq_n_s32(0);
        let mut accf = [vdupq_n_f32(0.0); R];
        let mut b = 0;
        while b + 4 <= bpr {
            let k = b * 32;
            let x0 = vld1q_s8(xq.add(k));
            let x1 = vld1q_s8(xq.add(k + 16));
            let x2 = vld1q_s8(xq.add(k + 32));
            let x3 = vld1q_s8(xq.add(k + 48));
            let x4 = vld1q_s8(xq.add(k + 64));
            let x5 = vld1q_s8(xq.add(k + 80));
            let x6 = vld1q_s8(xq.add(k + 96));
            let x7 = vld1q_s8(xq.add(k + 112));
            let sxv = vld1q_f32(xs.add(b));
            for r in 0..R {
                let wp = q.add(r * n_in + k);
                let a0 = sdot(sdot(zero, vld1q_s8(wp), x0), vld1q_s8(wp.add(16)), x1);
                let a1 = sdot(sdot(zero, vld1q_s8(wp.add(32)), x2), vld1q_s8(wp.add(48)), x3);
                let a2 = sdot(sdot(zero, vld1q_s8(wp.add(64)), x4), vld1q_s8(wp.add(80)), x5);
                let a3 = sdot(sdot(zero, vld1q_s8(wp.add(96)), x6), vld1q_s8(wp.add(112)), x7);
                let sums = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
                let swv = vld1q_f32(s.add(r * bpr + b));
                accf[r] = vfmaq_f32(accf[r], vcvtq_f32_s32(sums), vmulq_f32(swv, sxv));
            }
            b += 4;
        }
        let mut out = [0f32; R];
        for r in 0..R {
            out[r] = vaddvq_f32(accf[r]);
        }
        while b < bpr {
            let k = b * 32;
            let x0 = vld1q_s8(xq.add(k));
            let x1 = vld1q_s8(xq.add(k + 16));
            let sx = *xs.add(b);
            for r in 0..R {
                let wp = q.add(r * n_in + k);
                let a = sdot(sdot(zero, vld1q_s8(wp), x0), vld1q_s8(wp.add(16)), x1);
                out[r] += vaddvq_s32(a) as f32 * *s.add(r * bpr + b) * sx;
            }
            b += 1;
        }
        out
    }

    /// Q8 decode for one token: rows `r0..r1` of `m` against activation row (`xq`, `xs`), into `y[r]`.
    pub unsafe fn q8_rows_t1(m: &super::QMatrix, xq: *const i8, xs: *const f32, y: *mut f32, r0: usize, r1: usize) {
        let n_in = m.n_in;
        let bpr = n_in / 32;
        let mut r = r0;
        while r + 2 <= r1 {
            let o = q8_dot_rows::<2>(m.q.as_ptr().add(r * n_in), m.scales.as_ptr().add(r * bpr), n_in, xq, xs);
            *y.add(r) = o[0];
            *y.add(r + 1) = o[1];
            r += 2;
        }
        if r < r1 {
            let o = q8_dot_rows::<1>(m.q.as_ptr().add(r * n_in), m.scales.as_ptr().add(r * bpr), n_in, xq, xs);
            *y.add(r) = o[0];
        }
    }

    /// SDOT: each 32-bit lane of `acc` gains the dot product of four int8 pairs from `a` and `b`.
    /// Inline asm rather than `vdotq_s32`: the intrinsic is stable only from Rust 1.98 (MSRV is
    /// 1.95), and it is not inlined into code built without the `dotprod` target feature (the
    /// default for the iOS and Android targets), where every call would go out of line.
    #[inline(always)]
    pub unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut r = acc;
        std::arch::asm!(
            ".arch_extension dotprod",
            "sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
            r = inout(vreg) r,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
        r
    }

    /// Four f16 values -> f32 (FCVTL; the intrinsic is not stable yet).
    #[inline(always)]
    pub unsafe fn f16x4_to_f32(p: *const u16) -> float32x4_t {
        let h = vld1_u16(p);
        let r: float32x4_t;
        std::arch::asm!("fcvtl {r:v}.4s, {h:v}.4h", r = out(vreg) r, h = in(vreg) h, options(pure, nomem, nostack));
        r
    }

    /// Q4_0 dot products of R rows with one quantized activation row. Nibbles are used
    /// unsigned (0..15) in SDOT and the +8 offset is folded in via the block sums of x.
    #[inline(never)]
    pub unsafe fn q4_dot_rows<const R: usize>(
        q: *const u8,
        s: *const u16,
        n_in: usize,
        xq: *const i8,
        xs: *const f32,
        xsums: *const i32,
    ) -> [f32; R] {
        let bpr = n_in / 32;
        let half_row = n_in / 2;
        let zero = vdupq_n_s32(0);
        let m0f = vdupq_n_u8(0x0F);
        let mut accf = [vdupq_n_f32(0.0); R];
        let mut b = 0;
        while b + 4 <= bpr {
            let k = b * 32;
            let x0 = vld1q_s8(xq.add(k));
            let x1 = vld1q_s8(xq.add(k + 16));
            let x2 = vld1q_s8(xq.add(k + 32));
            let x3 = vld1q_s8(xq.add(k + 48));
            let x4 = vld1q_s8(xq.add(k + 64));
            let x5 = vld1q_s8(xq.add(k + 80));
            let x6 = vld1q_s8(xq.add(k + 96));
            let x7 = vld1q_s8(xq.add(k + 112));
            let sxv = vld1q_f32(xs.add(b));
            let xoff = vshlq_n_s32::<3>(vld1q_s32(xsums.add(b)));
            for r in 0..R {
                let wp = q.add(r * half_row + b * 16);
                macro_rules! blk {
                    ($i:expr, $xa:expr, $xb:expr) => {{
                        let v = vld1q_u8(wp.add(16 * $i));
                        let lo = vreinterpretq_s8_u8(vandq_u8(v, m0f));
                        let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
                        sdot(sdot(zero, lo, $xa), hi, $xb)
                    }};
                }
                let a0 = blk!(0, x0, x1);
                let a1 = blk!(1, x2, x3);
                let a2 = blk!(2, x4, x5);
                let a3 = blk!(3, x6, x7);
                let sums = vsubq_s32(vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)), xoff);
                let swv = f16x4_to_f32(s.add(r * bpr + b));
                accf[r] = vfmaq_f32(accf[r], vcvtq_f32_s32(sums), vmulq_f32(swv, sxv));
            }
            b += 4;
        }
        let mut out = [0f32; R];
        for r in 0..R {
            out[r] = vaddvq_f32(accf[r]);
        }
        while b < bpr {
            let k = b * 32;
            let x0 = vld1q_s8(xq.add(k));
            let x1 = vld1q_s8(xq.add(k + 16));
            let sx = *xs.add(b);
            let xoff = 8 * *xsums.add(b);
            for r in 0..R {
                let v = vld1q_u8(q.add(r * half_row + b * 16));
                let lo = vreinterpretq_s8_u8(vandq_u8(v, m0f));
                let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
                let a = sdot(sdot(zero, lo, x0), hi, x1);
                let sum = vaddvq_s32(a) - xoff;
                out[r] += sum as f32 * crate::quant::f16_to_f32(*s.add(r * bpr + b)) * sx;
            }
            b += 1;
        }
        out
    }

    /// Q4 decode for one token: rows `r0..r1` of `m` against activation row, into `y[r]`.
    pub unsafe fn q4_rows_t1(m: &super::Q4Matrix, xq: *const i8, xs: *const f32, xsums: *const i32, y: *mut f32, r0: usize, r1: usize) {
        let n_in = m.n_in;
        let bpr = n_in / 32;
        let mut r = r0;
        while r + 2 <= r1 {
            let o = q4_dot_rows::<2>(m.q.as_ptr().add(r * n_in / 2), m.scales.as_ptr().add(r * bpr), n_in, xq, xs, xsums);
            *y.add(r) = o[0];
            *y.add(r + 1) = o[1];
            r += 2;
        }
        if r < r1 {
            let o = q4_dot_rows::<1>(m.q.as_ptr().add(r * n_in / 2), m.scales.as_ptr().add(r * bpr), n_in, xq, xs, xsums);
            *y.add(r) = o[0];
        }
    }

    /// Q8: R weight rows against TC quantized activation rows (token stride `n_in`), reading each
    /// weight block once and each activation vector once per R rows. Used for short multi-token
    /// passes (speculative verification).
    #[inline(never)]
    pub unsafe fn q8_dot_rows_tc<const R: usize, const TC: usize>(
        q: *const i8,
        s: *const f32,
        n_in: usize,
        xq: *const i8,
        xs: *const f32,
    ) -> [[f32; TC]; R] {
        let bpr = n_in / 32;
        let zero = vdupq_n_s32(0);
        let mut accf = [[vdupq_n_f32(0.0); TC]; R];
        let mut b = 0;
        while b + 4 <= bpr {
            let k = b * 32;
            let mut xv = [[vdupq_n_s8(0); 8]; TC];
            for c in 0..TC {
                let xp = xq.add(c * n_in + k);
                for i in 0..8 {
                    xv[c][i] = vld1q_s8(xp.add(16 * i));
                }
            }
            for r in 0..R {
                let wp = q.add(r * n_in + k);
                let w0 = vld1q_s8(wp);
                let w1 = vld1q_s8(wp.add(16));
                let w2 = vld1q_s8(wp.add(32));
                let w3 = vld1q_s8(wp.add(48));
                let w4 = vld1q_s8(wp.add(64));
                let w5 = vld1q_s8(wp.add(80));
                let w6 = vld1q_s8(wp.add(96));
                let w7 = vld1q_s8(wp.add(112));
                let swv = vld1q_f32(s.add(r * bpr + b));
                for c in 0..TC {
                    let x = &xv[c];
                    let a0 = sdot(sdot(zero, w0, x[0]), w1, x[1]);
                    let a1 = sdot(sdot(zero, w2, x[2]), w3, x[3]);
                    let a2 = sdot(sdot(zero, w4, x[4]), w5, x[5]);
                    let a3 = sdot(sdot(zero, w6, x[6]), w7, x[7]);
                    let sums = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
                    let sxv = vld1q_f32(xs.add(c * bpr + b));
                    accf[r][c] = vfmaq_f32(accf[r][c], vcvtq_f32_s32(sums), vmulq_f32(swv, sxv));
                }
            }
            b += 4;
        }
        let mut out = [[0f32; TC]; R];
        for r in 0..R {
            for c in 0..TC {
                out[r][c] = vaddvq_f32(accf[r][c]);
            }
        }
        while b < bpr {
            let k = b * 32;
            for r in 0..R {
                let w0 = vld1q_s8(q.add(r * n_in + k));
                let w1 = vld1q_s8(q.add(r * n_in + k + 16));
                let sw = *s.add(r * bpr + b);
                for c in 0..TC {
                    let xp = xq.add(c * n_in + k);
                    let a = sdot(sdot(zero, w0, vld1q_s8(xp)), w1, vld1q_s8(xp.add(16)));
                    out[r][c] += vaddvq_s32(a) as f32 * sw * *xs.add(c * bpr + b);
                }
            }
            b += 1;
        }
        out
    }

    /// Q4_0: R weight rows against TC quantized activation rows, reading each block once.
    #[inline(never)]
    pub unsafe fn q4_dot_rows_tc<const R: usize, const TC: usize>(
        q: *const u8,
        s: *const u16,
        n_in: usize,
        xq: *const i8,
        xs: *const f32,
        xsums: *const i32,
    ) -> [[f32; TC]; R] {
        let bpr = n_in / 32;
        let half_row = n_in / 2;
        let zero = vdupq_n_s32(0);
        let m0f = vdupq_n_u8(0x0F);
        let mut accf = [[vdupq_n_f32(0.0); TC]; R];
        let mut b = 0;
        while b + 4 <= bpr {
            let k = b * 32;
            let mut xv = [[vdupq_n_s8(0); 8]; TC];
            let mut xoff = [vdupq_n_s32(0); TC];
            for c in 0..TC {
                let xp = xq.add(c * n_in + k);
                for i in 0..8 {
                    xv[c][i] = vld1q_s8(xp.add(16 * i));
                }
                xoff[c] = vshlq_n_s32::<3>(vld1q_s32(xsums.add(c * bpr + b)));
            }
            for r in 0..R {
                let wp = q.add(r * half_row + b * 16);
                macro_rules! unpack {
                    ($i:expr) => {{
                        let v = vld1q_u8(wp.add(16 * $i));
                        (
                            vreinterpretq_s8_u8(vandq_u8(v, m0f)),
                            vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)),
                        )
                    }};
                }
                let (l0, h0) = unpack!(0);
                let (l1, h1) = unpack!(1);
                let (l2, h2) = unpack!(2);
                let (l3, h3) = unpack!(3);
                let swv = f16x4_to_f32(s.add(r * bpr + b));
                for c in 0..TC {
                    let x = &xv[c];
                    let a0 = sdot(sdot(zero, l0, x[0]), h0, x[1]);
                    let a1 = sdot(sdot(zero, l1, x[2]), h1, x[3]);
                    let a2 = sdot(sdot(zero, l2, x[4]), h2, x[5]);
                    let a3 = sdot(sdot(zero, l3, x[6]), h3, x[7]);
                    let sums = vsubq_s32(vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)), xoff[c]);
                    let sxv = vld1q_f32(xs.add(c * bpr + b));
                    accf[r][c] = vfmaq_f32(accf[r][c], vcvtq_f32_s32(sums), vmulq_f32(swv, sxv));
                }
            }
            b += 4;
        }
        let mut out = [[0f32; TC]; R];
        for r in 0..R {
            for c in 0..TC {
                out[r][c] = vaddvq_f32(accf[r][c]);
            }
        }
        while b < bpr {
            let k = b * 32;
            for r in 0..R {
                let v = vld1q_u8(q.add(r * half_row + b * 16));
                let lo = vreinterpretq_s8_u8(vandq_u8(v, m0f));
                let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
                let sw = crate::quant::f16_to_f32(*s.add(r * bpr + b));
                for c in 0..TC {
                    let xp = xq.add(c * n_in + k);
                    let a = sdot(sdot(zero, lo, vld1q_s8(xp)), hi, vld1q_s8(xp.add(16)));
                    let sum = vaddvq_s32(a) - 8 * *xsums.add(c * bpr + b);
                    out[r][c] += sum as f32 * sw * *xs.add(c * bpr + b);
                }
            }
            b += 1;
        }
        out
    }

    /// Rows `r0..r1` (any count) for `tc` tokens (1..=4) starting at `c0`. Requires `n_in % 32 == 0`.
    ///
    /// Decode is memory-bound, and the prefetcher likes few long streams per core, so tiles
    /// use at most 2 weight rows with many accumulators (measured ~2x faster than 4+ rows).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block_tc<W: Load8>(
        w: *const W,
        n_in: usize,
        x: *const f32,
        y: *mut f32,
        n_out: usize,
        r0: usize,
        r1: usize,
        c0: usize,
        tc: usize,
    ) {
        macro_rules! go {
            ($TC:expr, $G:expr, $A:expr) => {{
                let m = r0 + (r1 - r0) / 2 * 2;
                run_tiles::<W, 2, $TC, $G, $A>(w, n_in, x, y, n_out, r0, m, c0);
                run_tiles::<W, 1, $TC, $G, $A>(w, n_in, x, y, n_out, m, r1, c0);
            }};
        }
        // Token pairs use the single-token accumulator layout (G=4, A=8; one row at a time to fit
        // the registers), so each output is bit-identical whatever the token count: speculative
        // verification over a few tokens then reproduces plain decoding exactly.
        let pair = |c: usize| run_tiles::<W, 1, 2, 4, 8>(w, n_in, x, y, n_out, r0, r1, c);
        match tc {
            1 => go!(1, 4, 8),
            2 => pair(c0),
            3 => {
                pair(c0);
                let m = r0 + (r1 - r0) / 2 * 2;
                run_tiles::<W, 2, 1, 4, 8>(w, n_in, x, y, n_out, r0, m, c0 + 2);
                run_tiles::<W, 1, 1, 4, 8>(w, n_in, x, y, n_out, m, r1, c0 + 2);
            }
            4 => {
                pair(c0);
                pair(c0 + 2)
            }
            _ => unreachable!("bad tc {tc}"),
        }
    }
}

/// Computes rows r0..r1 of the output for all t tokens.
#[cfg(has_simd)]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_block(w: &[u16], n_out: usize, n_in: usize, x: &[f32], xt: Option<&[f32]>, t: usize, r0: usize, r1: usize, y: *mut f32) {
    use simd::{block_tc, gemm_8x8};
    if !n_in.is_multiple_of(32) {
        return matmul_block_scalar(w, n_out, n_in, x, t, r0, r1, y);
    }
    let wp = w.as_ptr();
    let xp = x.as_ptr();
    let mut r = r0;
    if let Some(xt) = xt {
        // Prefill: 8x8 outer-product tiles; weights stay in L2 across token tiles. When the packed
        // activations are larger than L2 (convolutions over thousands of pixels), walk the tokens
        // in groups of tiles instead, so each group stays cached while every row passes over it.
        let tiles = t.div_ceil(TN);
        let group = if t * n_in * 4 > 8 << 20 {
            (256 << 10) / (n_in * TN * 4)
        } else {
            tiles
        }
        .max(1);
        let full = r0 + (r1 - r0) / 8 * 8;
        let mut g0 = 0;
        while g0 < tiles {
            let g1 = (g0 + group).min(tiles);
            let mut rr = r0;
            while rr < full {
                for tile in g0..g1 {
                    let c0 = tile * TN;
                    gemm_8x8::<u16>(
                        wp.add(rr * n_in),
                        n_in,
                        xt.as_ptr().add(tile * n_in * TN),
                        y,
                        n_out,
                        rr,
                        c0,
                        (t - c0).min(TN),
                    );
                }
                rr += 8;
            }
            g0 = g1;
        }
        r = full;
    }
    // Decode (and leftover rows): streaming dot-product tiles over up to 4 tokens at a time.
    if r < r1 {
        let mut c = 0;
        while c < t {
            let tc = (t - c).min(4);
            block_tc::<u16>(wp, n_in, xp, y, n_out, r, r1, c, tc);
            c += tc;
        }
    }
}

#[cfg(not(has_simd))]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_block(w: &[u16], n_out: usize, n_in: usize, x: &[f32], _xt: Option<&[f32]>, t: usize, r0: usize, r1: usize, y: *mut f32) {
    matmul_block_scalar(w, n_out, n_in, x, t, r0, r1, y)
}

#[allow(clippy::too_many_arguments)]
unsafe fn matmul_block_scalar(w: &[u16], n_out: usize, n_in: usize, x: &[f32], t: usize, r0: usize, r1: usize, y: *mut f32) {
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        if s.len() < n_in {
            s.resize(n_in, 0.0);
        }
        for r in r0..r1 {
            bf16_to_f32_slice(&w[r * n_in..(r + 1) * n_in], &mut s[..n_in]);
            for c in 0..t {
                let v = dot(&s[..n_in], &x[c * n_in..(c + 1) * n_in]);
                *y.add(c * n_out + r) = v;
            }
        }
    });
}

/// A quantized weight matrix usable by the shared quantized-matmul driver.
pub trait QMat: Sync {
    fn n_out(&self) -> usize;
    fn n_in(&self) -> usize;
    fn bytes(&self) -> usize;
    fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]);
    /// Float storage (GGUF F32/F16/BF16): decode also takes the dequantize + f32 GEMM path, so
    /// the activations are not rounded to int8.
    fn dense(&self) -> bool {
        false
    }
    /// Decode: rows `r0..r1` against quantized activation row `c` of `xa`, into `y[r]`.
    unsafe fn rows_t1(&self, xa: &QAct, c: usize, y: *mut f32, r0: usize, r1: usize);
    /// Decode: rows `r0..r1` against activation rows `c0..c0+tc` (tc in 2..=4), into `y[(c0+c)*n_out + r]`.
    unsafe fn rows_tc(&self, xa: &QAct, c0: usize, tc: usize, y: *mut f32, r0: usize, r1: usize);
    /// Whether `rows_multi` handles this matrix for `t` tokens (`Some`), and if so whether it
    /// wants the pair-interleaved activations for SMMLA (`Some(true)`).
    fn multi(&self, _t: usize) -> Option<bool> {
        None
    }
    /// Decode over all `t` (2..=16) activation rows at once (each weight block unpacked once),
    /// rows `r0..r1` into `y[c*n_out + r]`, with SMMLA when `pairs` is given; false if not
    /// supported (the driver then runs token pairs). Rows must equal `rows_t1` bit for bit.
    unsafe fn rows_multi(&self, _xa: &QAct, _pairs: Option<&PairAct>, _t: usize, _y: *mut f32, _r0: usize, _r1: usize) -> bool {
        false
    }
}

impl QMat for QMatrix {
    fn n_out(&self) -> usize {
        self.n_out
    }
    fn n_in(&self) -> usize {
        self.n_in
    }
    fn bytes(&self) -> usize {
        QMatrix::bytes(self)
    }
    fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        dequantize_rows(self, r0, r1, out)
    }
    unsafe fn rows_t1(&self, xa: &QAct, c: usize, y: *mut f32, r0: usize, r1: usize) {
        let bpr = self.n_in / QBLOCK;
        let xq = xa.q.as_ptr().add(c * self.n_in);
        let xs = xa.scales.as_ptr().add(c * bpr);
        #[cfg(has_simd)]
        {
            simd::q8_rows_t1(self, xq, xs, y, r0, r1);
        }
        #[cfg(not(has_simd))]
        {
            for r in r0..r1 {
                let mut acc = 0f32;
                for b in 0..bpr {
                    let mut s = 0i32;
                    for j in 0..QBLOCK {
                        s += self.q[r * self.n_in + b * QBLOCK + j] as i32 * *xq.add(b * QBLOCK + j) as i32;
                    }
                    acc += s as f32 * self.scales[r * bpr + b] * *xs.add(b);
                }
                *y.add(r) = acc;
            }
        }
    }
    unsafe fn rows_tc(&self, xa: &QAct, c0: usize, tc: usize, y: *mut f32, r0: usize, r1: usize) {
        self.rows_tc_impl(xa, c0, tc, y, r0, r1)
    }
}

impl QMatrix {
    unsafe fn rows_tc_impl(&self, xa: &QAct, c0: usize, tc: usize, y: *mut f32, r0: usize, r1: usize) {
        let n_in = self.n_in;
        let bpr = n_in / QBLOCK;
        let xq = xa.q.as_ptr().add(c0 * n_in);
        let xs = xa.scales.as_ptr().add(c0 * bpr);
        #[cfg(has_simd)]
        {
            macro_rules! go {
                ($TC:expr) => {{
                    let mut r = r0;
                    while r + 2 <= r1 {
                        let o =
                            simd::q8_dot_rows_tc::<2, $TC>(self.q.as_ptr().add(r * n_in), self.scales.as_ptr().add(r * bpr), n_in, xq, xs);
                        for rr in 0..2 {
                            for c in 0..$TC {
                                *y.add((c0 + c) * self.n_out + r + rr) = o[rr][c];
                            }
                        }
                        r += 2;
                    }
                    if r < r1 {
                        let o =
                            simd::q8_dot_rows_tc::<1, $TC>(self.q.as_ptr().add(r * n_in), self.scales.as_ptr().add(r * bpr), n_in, xq, xs);
                        for c in 0..$TC {
                            *y.add((c0 + c) * self.n_out + r) = o[0][c];
                        }
                    }
                }};
            }
            match tc {
                2 => go!(2),
                3 => go!(3),
                4 => go!(4),
                _ => unreachable!("bad tc {tc}"),
            }
        }
        #[cfg(not(has_simd))]
        {
            let _ = (xq, xs);
            for c in 0..tc {
                self.rows_t1(xa, c0 + c, y.add((c0 + c) * self.n_out), r0, r1);
            }
        }
    }
}

impl Q4Matrix {
    unsafe fn rows_tc_impl(&self, xa: &QAct, c0: usize, tc: usize, y: *mut f32, r0: usize, r1: usize) {
        let n_in = self.n_in;
        let bpr = n_in / QBLOCK;
        let xq = xa.q.as_ptr().add(c0 * n_in);
        let xs = xa.scales.as_ptr().add(c0 * bpr);
        let xsums = xa.sums.as_ptr().add(c0 * bpr);
        #[cfg(has_simd)]
        {
            macro_rules! go {
                ($TC:expr) => {{
                    let mut r = r0;
                    while r + 2 <= r1 {
                        let o = simd::q4_dot_rows_tc::<2, $TC>(
                            self.q.as_ptr().add(r * n_in / 2),
                            self.scales.as_ptr().add(r * bpr),
                            n_in,
                            xq,
                            xs,
                            xsums,
                        );
                        for rr in 0..2 {
                            for c in 0..$TC {
                                *y.add((c0 + c) * self.n_out + r + rr) = o[rr][c];
                            }
                        }
                        r += 2;
                    }
                    if r < r1 {
                        let o = simd::q4_dot_rows_tc::<1, $TC>(
                            self.q.as_ptr().add(r * n_in / 2),
                            self.scales.as_ptr().add(r * bpr),
                            n_in,
                            xq,
                            xs,
                            xsums,
                        );
                        for c in 0..$TC {
                            *y.add((c0 + c) * self.n_out + r) = o[0][c];
                        }
                    }
                }};
            }
            match tc {
                2 => go!(2),
                3 => go!(3),
                4 => go!(4),
                _ => unreachable!("bad tc {tc}"),
            }
        }
        #[cfg(has_simd)]
        for c in c0..c0 + tc {
            self.add_offsets(xa, c, y.add(c * self.n_out), r0, r1);
        }
        #[cfg(not(has_simd))]
        {
            let _ = (xq, xs, xsums);
            for c in 0..tc {
                self.rows_t1(xa, c0 + c, y.add((c0 + c) * self.n_out), r0, r1);
            }
        }
    }

    /// Adds the per-group offsets of an affine matrix: `y[r] += sum_g offset[r][g] * sum(x in g)`
    /// for activation row `c` (the SIMD dot products cover only the `d * (q - 8)` part).
    #[cfg(has_simd)]
    unsafe fn add_offsets(&self, xa: &QAct, c: usize, y: *mut f32, r0: usize, r1: usize) {
        let Some(o) = &self.offsets else { return };
        let bpr = self.n_in / QBLOCK;
        let k = o.group / QBLOCK;
        let gpr = self.n_in / o.group;
        let fs = &xa.fsums[c * bpr..(c + 1) * bpr];
        let gs: Vec<f32> = fs.chunks(k).map(|ch| ch.iter().sum()).collect();
        for r in r0..r1 {
            *y.add(r) += dot_bf16(&o.v[r * gpr..(r + 1) * gpr], &gs);
        }
    }
}

impl QMat for Q4Matrix {
    fn n_out(&self) -> usize {
        self.n_out
    }
    fn n_in(&self) -> usize {
        self.n_in
    }
    fn bytes(&self) -> usize {
        Q4Matrix::bytes(self)
    }
    fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        dequantize_rows_q4(self, r0, r1, out)
    }
    unsafe fn rows_t1(&self, xa: &QAct, c: usize, y: *mut f32, r0: usize, r1: usize) {
        let bpr = self.n_in / QBLOCK;
        let xq = xa.q.as_ptr().add(c * self.n_in);
        let xs = xa.scales.as_ptr().add(c * bpr);
        let xsums = xa.sums.as_ptr().add(c * bpr);
        #[cfg(has_simd)]
        {
            simd::q4_rows_t1(self, xq, xs, xsums, y, r0, r1);
            self.add_offsets(xa, c, y, r0, r1);
        }
        #[cfg(not(has_simd))]
        {
            let _ = xsums;
            let mut row = vec![0f32; self.n_in];
            for r in r0..r1 {
                dequantize_rows_q4(self, r, r + 1, &mut row);
                let mut acc = 0f32;
                for b in 0..bpr {
                    let mut s = 0f32;
                    for j in 0..QBLOCK {
                        s += row[b * QBLOCK + j] * *xq.add(b * QBLOCK + j) as f32;
                    }
                    acc += s * *xs.add(b);
                }
                *y.add(r) = acc;
            }
        }
    }
    unsafe fn rows_tc(&self, xa: &QAct, c0: usize, tc: usize, y: *mut f32, r0: usize, r1: usize) {
        self.rows_tc_impl(xa, c0, tc, y, r0, r1)
    }
}

struct QJobRef<'a, M: QMat> {
    m: &'a M,
    y: SendPtr,
}
impl<M: QMat> Clone for QJobRef<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M: QMat> Copy for QJobRef<'_, M> {}

/// Most tokens a quantized matmul runs with the decode kernels (int8 dot products, token pairs),
/// whose rows are bit-identical whatever the token count; more use the prefill GEMM.
static DECODE_MAX_TOKENS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(7);

/// See [`set_decode_max_tokens`].
pub fn decode_max_tokens() -> usize {
    DECODE_MAX_TOKENS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runs quantized matmuls of up to `n` tokens (clamped to 7..=15) with the batch-invariant
/// decode kernels instead of the prefill GEMM (default 7). Speculative verification of `n`
/// tokens then reproduces one-token decoding bit for bit; the decode kernels are slower than the
/// GEMM when compute-bound, which matters little when the weights stream from storage.
pub fn set_decode_max_tokens(n: usize) {
    DECODE_MAX_TOKENS.store(n.clamp(7, 15), std::sync::atomic::Ordering::Relaxed);
}

/// Kernels for quantized decode passes of 2 or more tokens (speculative verification). All three
/// give bit-identical rows whatever the token count; they differ in speed only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiTokenKernels {
    /// SMMLA (i8mm) when the CPU has it (Apple A15 / M2 and later, Armv8.6 cores), else SDOT.
    Auto,
    /// One weight row at a time against all tokens with SDOT (no i8mm).
    Dot,
    /// The older token-pair loop (each weight block unpacked once per pair of tokens).
    Pairs,
}

static MULTI_KERNELS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Selects the multi-token decode kernels (default `Auto`; also `OXINFER_MULTI=auto|dot|pairs`).
pub fn set_multi_token_kernels(k: MultiTokenKernels) {
    let v = match k {
        MultiTokenKernels::Auto => 1,
        MultiTokenKernels::Dot => 2,
        MultiTokenKernels::Pairs => 3,
    };
    MULTI_KERNELS.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// The multi-token decode kernels in use.
pub fn multi_token_kernels() -> MultiTokenKernels {
    let mut v = MULTI_KERNELS.load(std::sync::atomic::Ordering::Relaxed);
    if v == 0 {
        set_multi_token_kernels(match std::env::var("OXINFER_MULTI").as_deref() {
            Ok("dot") => MultiTokenKernels::Dot,
            Ok("pairs") => MultiTokenKernels::Pairs,
            _ => MultiTokenKernels::Auto,
        });
        v = MULTI_KERNELS.load(std::sync::atomic::Ordering::Relaxed);
    }
    match v {
        2 => MultiTokenKernels::Dot,
        3 => MultiTokenKernels::Pairs,
        _ => MultiTokenKernels::Auto,
    }
}

/// Whether this CPU has the int8 matrix-multiply instructions (Arm i8mm: SMMLA).
pub fn cpu_has_i8mm() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("i8mm")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Most tokens the multi-token decode kernels take in one pass.
const MULTI_MAX: usize = 16;

/// Several quantized matmuls sharing the same f32 input `x` ([t, n_in]); each job is `(matrix, y [t, n_out])`.
pub fn matmul_quant_multi<M: QMat>(jobs: &mut [(&M, &mut [f32])], x: &[f32], t: usize) {
    matmul_quant_multi_at(jobs, x, t, decode_max_tokens())
}

/// [`matmul_quant_multi`] with the decode kernels up to `decode_max` tokens.
pub(crate) fn matmul_quant_multi_at<M: QMat>(jobs: &mut [(&M, &mut [f32])], x: &[f32], t: usize, decode_max: usize) {
    if t == 0 || jobs.is_empty() {
        return;
    }
    let n_in = jobs[0].0.n_in();
    assert_eq!(x.len(), t * n_in, "input shape mismatch");
    let mut total_bytes = 0usize;
    let refs: Vec<QJobRef<M>> = jobs
        .iter_mut()
        .map(|(m, y)| {
            assert_eq!(m.n_in(), n_in, "all jobs must share n_in");
            assert_eq!(y.len(), t * m.n_out(), "output shape mismatch");
            total_bytes += m.bytes();
            QJobRef {
                m: *m,
                y: SendPtr(y.as_mut_ptr()),
            }
        })
        .collect();
    let _timer = prof::start(Cat::Matmul);
    prof::add_bytes(total_bytes);

    let nthreads = pool::threads();
    let bytes_per_row = (total_bytes / refs.iter().map(|j| j.m.n_out()).sum::<usize>()).max(1);
    let target = (total_bytes / (nthreads * 8)).max(128 * 1024);
    let rows_per_block = (target / bytes_per_row).max(16) / 16 * 16;
    let mut blocks: Vec<(u32, u32, u32)> = Vec::new();
    for (j, job) in refs.iter().enumerate() {
        let mut r = 0;
        while r < job.m.n_out() {
            let r1 = (r + rows_per_block).min(job.m.n_out());
            blocks.push((j as u32, r as u32, r1 as u32));
            r = r1;
        }
    }
    // Prefill (and float-stored weights at any t): dequantize + f32 GEMM. Decode: int8 dot products.
    let gemm = t > decode_max;
    let dense = |m: &M| gemm || m.dense();
    let xt = refs.iter().any(|j| dense(j.m)).then(|| pack_xt(x, t, n_in));
    let xa = refs.iter().any(|j| !dense(j.m)).then(|| quantize_act(x, t, n_in));
    let (tchunks, tc) = if gemm { token_split(blocks.len(), t, nthreads) } else { (1, t) };
    let mode = multi_token_kernels();
    // Two tokens: the pair kernels already unpack each block once, and keep both accumulators in
    // registers.
    let multi = !gemm && (3..=MULTI_MAX).contains(&t) && mode != MultiTokenKernels::Pairs;
    let pairs =
        (multi && mode == MultiTokenKernels::Auto && cpu_has_i8mm() && refs.iter().any(|j| !dense(j.m) && j.m.multi(t) == Some(true)))
            .then(|| PairAct::new(xa.as_ref().unwrap()));
    pool::run(blocks.len() * tchunks, &|b| {
        let (j, r0, r1) = blocks[b / tchunks];
        let job = refs[j as usize];
        let (r0, r1) = (r0 as usize, r1 as usize);
        if dense(job.m) {
            let c0 = (b % tchunks) * tc;
            if c0 >= t {
                return;
            }
            let c1 = (c0 + tc).min(t);
            let xs = &x[c0 * n_in..c1 * n_in];
            let xts = &xt.as_deref().unwrap()[c0 / TN * n_in * TN..];
            unsafe { matmul_quant_block_prefill(job.m, xs, xts, c1 - c0, r0, r1, job.y.get().add(c0 * job.m.n_out())) }
            return;
        }
        let xa = xa.as_ref().unwrap();
        if multi && unsafe { job.m.rows_multi(xa, pairs.as_ref(), t, job.y.get(), r0, r1) } {
            return;
        }
        let mut c = 0;
        while c < t {
            // Token pairs: the 3- and 4-token tiles keep 24-32 activation vectors live and
            // spill; re-reading a row block from cache per pair is cheaper (27B Q4, 4 tokens:
            // 1.7x faster). The row block stays in L2 between pairs.
            let tc = (t - c).min(2);
            unsafe {
                if tc == 1 {
                    job.m.rows_t1(xa, c, job.y.get().add(c * job.m.n_out()), r0, r1)
                } else {
                    job.m.rows_tc(xa, c, tc, job.y.get(), r0, r1)
                }
            }
            c += tc;
        }
    });
}

/// Prefill with quantized weights: dequantize 8 rows at a time into a thread-local f32 scratch
/// and run the f32 GEMM tiles over all tokens.
unsafe fn matmul_quant_block_prefill<M: QMat>(m: &M, x: &[f32], xt: &[f32], t: usize, r0: usize, r1: usize, y: *mut f32) {
    let n_in = m.n_in();
    let n_out = m.n_out();
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        if s.len() < 8 * n_in {
            s.resize(8 * n_in, 0.0);
        }
        let mut r = r0;
        while r < r1 {
            let rows = (r1 - r).min(8);
            m.dequantize_rows(r, r + rows, &mut s[..rows * n_in]);
            #[cfg(has_simd)]
            {
                let sp = s.as_ptr();
                if rows == 8 {
                    let tiles = t.div_ceil(TN);
                    for tile in 0..tiles {
                        let c0 = tile * TN;
                        simd::gemm_8x8::<f32>(sp, n_in, xt.as_ptr().add(tile * n_in * TN), y, n_out, r, c0, (t - c0).min(TN));
                    }
                } else if n_in.is_multiple_of(32) {
                    let mut c = 0;
                    while c < t {
                        let tc = (t - c).min(4);
                        simd::block_tc::<f32>(sp, n_in, x.as_ptr(), y.add(r), n_out, 0, rows, c, tc);
                        c += tc;
                    }
                } else {
                    scalar_rows_f32(&s[..rows * n_in], n_in, x, t, y.add(r), n_out, rows);
                }
            }
            #[cfg(not(has_simd))]
            {
                let _ = xt;
                scalar_rows_f32(&s[..rows * n_in], n_in, x, t, y.add(r), n_out, rows);
            }
            r += rows;
        }
    });
}

/// Reference path: f32 rows `w` ([rows, n_in]) times all tokens, stored at `y[c*n_out + i]`.
unsafe fn scalar_rows_f32(w: &[f32], n_in: usize, x: &[f32], t: usize, y: *mut f32, n_out: usize, rows: usize) {
    for i in 0..rows {
        for c in 0..t {
            *y.add(c * n_out + i) = dot(&w[i * n_in..(i + 1) * n_in], &x[c * n_in..(c + 1) * n_in]);
        }
    }
}

/// Dot product of a bf16 row with an f32 vector (NEON path requires `len % 32 == 0`).
#[inline]
pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    #[cfg(has_simd)]
    {
        if w.len().is_multiple_of(32) {
            let o = unsafe { simd::kernel::<u16, 1, 1, 4, 8>(w.as_ptr(), 0, x.as_ptr(), 0, w.len()) };
            return o[0][0];
        }
    }
    // 8 independent lanes so it vectorizes.
    let mut acc = [0f32; 8];
    let (wc, wr) = w.as_chunks::<8>();
    let (xc, xr) = x.as_chunks::<8>();
    for (a, b) in wc.iter().zip(xc) {
        for i in 0..8 {
            acc[i] += bf16_to_f32(a[i]) * b[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (&a, &b) in wr.iter().zip(xr) {
        s += bf16_to_f32(a) * b;
    }
    s
}

/// f32 dot product with 8 independent accumulators (auto-vectorizes).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for i in 0..chunks {
        let a8 = &a[i * 8..i * 8 + 8];
        let b8 = &b[i * 8..i * 8 + 8];
        for j in 0..8 {
            acc[j] += a8[j] * b8[j];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// y += alpha * x
#[inline]
pub fn axpy(y: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// RMSNorm: out = x * rsqrt(mean(x^2) + eps) * w   (w optional, bf16).
pub fn rmsnorm(x: &[f32], w: Option<&[u16]>, eps: f32, out: &mut [f32]) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let r = ss.powf(-0.5);
    match w {
        Some(w) => {
            for i in 0..n {
                out[i] = x[i] * r * bf16_to_f32(w[i]);
            }
        }
        None => {
            for i in 0..n {
                out[i] = x[i] * r;
            }
        }
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], w: Option<&[u16]>, eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let r = ss.powf(-0.5);
    match w {
        Some(w) => {
            for i in 0..n {
                x[i] *= r * bf16_to_f32(w[i]);
            }
        }
        None => {
            for v in x.iter_mut() {
                *v *= r;
            }
        }
    }
}

/// Row-wise RMSNorm over a `[t, n]` matrix, in parallel when there are many rows.
pub fn rmsnorm_rows(x: &[f32], n: usize, w: Option<&[u16]>, eps: f32, out: &mut [f32]) {
    let _timer = prof::start(Cat::Norm);
    let par = x.len() > 4 * n;
    for_chunks(out, n, par, |i, o| rmsnorm(&x[i * n..(i + 1) * n], w, eps, o));
}

pub fn rmsnorm_rows_inplace(x: &mut [f32], n: usize, w: Option<&[u16]>, eps: f32) {
    let _timer = prof::start(Cat::Norm);
    let par = x.len() > 4 * n;
    for_chunks(x, n, par, |_, xi| rmsnorm_inplace(xi, w, eps));
}

/// RMSNorm with precomputed f32 weights (e.g. zero-centered `1 + w`): out = x * rsqrt(mean(x^2)+eps) * w.
pub fn rmsnorm_f32w(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let r = ss.powf(-0.5);
    for i in 0..n {
        out[i] = x[i] * r * w[i];
    }
}

pub fn rmsnorm_f32w_rows(x: &[f32], n: usize, w: &[f32], eps: f32, out: &mut [f32]) {
    let _timer = prof::start(Cat::Norm);
    let par = x.len() > 4 * n;
    for_chunks(out, n, par, |i, o| rmsnorm_f32w(&x[i * n..(i + 1) * n], w, eps, o));
}

pub fn rmsnorm_f32w_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let r = ss.powf(-0.5);
    for i in 0..n {
        x[i] *= r * w[i];
    }
}

pub fn rmsnorm_f32w_rows_inplace(x: &mut [f32], n: usize, w: &[f32], eps: f32) {
    let _timer = prof::start(Cat::Norm);
    let par = x.len() > 4 * n;
    for_chunks(x, n, par, |_, xi| rmsnorm_f32w_inplace(xi, w, eps));
}

/// a[i] = silu(a[i]) * b[i]
pub fn silu_mul(a: &mut [f32], b: &[f32]) {
    let _timer = prof::start(Cat::Elementwise);
    let par = a.len() >= PAR_MIN;
    let chunk = if par { par_chunk(a.len()) } else { a.len() };
    for_chunks(a, chunk, par, |i, c| {
        for (ai, &bi) in c.iter_mut().zip(&b[i * chunk..]) {
            let x = *ai;
            *ai = x / (1.0 + fast_exp(-x)) * bi;
        }
    });
}

/// Runs `f(index, chunk)` over fixed-size chunks, in parallel only when asked.
pub fn for_chunks<F>(x: &mut [f32], size: usize, par: bool, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let n = x.len();
    let nchunks = n.div_ceil(size.max(1));
    if !par || nchunks <= 1 {
        for (i, c) in x.chunks_mut(size).enumerate() {
            f(i, c);
        }
        return;
    }
    let p = SendPtr(x.as_mut_ptr());
    pool::run(nchunks, &|i| {
        let start = i * size;
        let len = size.min(n - start);
        // SAFETY: chunks are disjoint and `x` outlives the call.
        let c = unsafe { std::slice::from_raw_parts_mut(p.get().add(start), len) };
        f(i, c);
    });
}

/// Fast vectorizable exp (relative error ~1e-7).
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let fx = (x * std::f32::consts::LOG2_E).round();
    let r = (x - fx * 0.693_359_4) - fx * -2.121_944_4e-4;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_668 + r * (0.008_333_334 + r * 0.001_388_888_9)))));
    f32::from_bits(((fx as i32 + 127) << 23) as u32) * p
}

/// GELU with tanh approximation (`gelu_pytorch_tanh`): x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3)).
#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    const K2: f32 = 2.0 * 0.797_884_6;
    let u = K2 * (x + 0.044_715 * x * x * x);
    x / (1.0 + fast_exp(-u))
}

const PAR_MIN: usize = 8192;

fn par_chunk(len: usize) -> usize {
    (len.div_ceil(pool::threads())).max(1024).next_multiple_of(16)
}

/// a[i] = gelu(a[i]) * b[i]
pub fn gelu_mul(a: &mut [f32], b: &[f32]) {
    let _timer = prof::start(Cat::Elementwise);
    let par = a.len() >= PAR_MIN;
    let chunk = if par { par_chunk(a.len()) } else { a.len() };
    for_chunks(a, chunk, par, |i, c| {
        for (ai, &bi) in c.iter_mut().zip(&b[i * chunk..]) {
            *ai = gelu_tanh(*ai) * bi;
        }
    });
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    let _timer = prof::start(Cat::Elementwise);
    let par = a.len() >= PAR_MIN;
    let chunk = if par { par_chunk(a.len()) } else { a.len() };
    for_chunks(a, chunk, par, |i, c| {
        for (ai, &bi) in c.iter_mut().zip(&b[i * chunk..]) {
            *ai += bi;
        }
    });
}

pub fn scale_inplace(a: &mut [f32], s: f32) {
    for v in a.iter_mut() {
        *v *= s;
    }
}

/// Inverse frequencies for RoPE on a head of `dim`, rotating `proportion` of it
/// (zeros for the un-rotated part, matching HF "proportional" RoPE).
pub fn rope_inv_freq(dim: usize, theta: f64, proportion: f64) -> Vec<f32> {
    let half = dim / 2;
    let rot = ((proportion * dim as f64) as usize) / 2;
    (0..half)
        .map(|i| {
            if i < rot {
                (1.0 / theta.powf((2 * i) as f64 / dim as f64)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// cos/sin tables `[positions.len(), dim/2]` for the given inverse frequencies.
pub fn rope_tables(positions: &[usize], inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0f32; positions.len() * half];
    let mut sin = vec![0f32; positions.len() * half];
    for (pi, &p) in positions.iter().enumerate() {
        for (i, &f) in inv_freq.iter().enumerate() {
            // Match the reference: angle computed in f32.
            let a = p as f32 * f;
            cos[pi * half + i] = a.cos();
            sin[pi * half + i] = a.sin();
        }
    }
    (cos, sin)
}

/// Applies rotate-half RoPE to one head vector `v` of length `2*cos.len()`.
#[inline]
pub fn apply_rope(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    debug_assert_eq!(v.len(), 2 * half);
    let (a, b) = v.split_at_mut(half);
    for i in 0..half {
        let x1 = a[i];
        let x2 = b[i];
        a[i] = x1 * cos[i] - x2 * sin[i];
        b[i] = x2 * cos[i] + x1 * sin[i];
    }
}

/// GPT-J style RoPE: rotates pairs of adjacent dimensions of `v` (length `2*cos.len()`).
#[inline]
pub fn apply_rope_interleaved(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    debug_assert_eq!(v.len(), 2 * cos.len());
    for (i, p) in v.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let (x1, x2) = (p[0], p[1]);
        p[0] = x1 * cos[i] - x2 * sin[i];
        p[1] = x2 * cos[i] + x1 * sin[i];
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = fast_exp(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequantize_rows, dequantize_rows_q4, quantize_act, quantize_bf16, quantize_q4};

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    fn reference(wf: &[f32], n_out: usize, n_in: usize, xa_f: &[f32], t: usize) -> Vec<f32> {
        // Reference uses the *quantized* activations dequantized, so only weight-side rounding differs.
        let mut y = vec![0f32; t * n_out];
        for c in 0..t {
            for r in 0..n_out {
                y[c * n_out + r] = dot(&wf[r * n_in..(r + 1) * n_in], &xa_f[c * n_in..(c + 1) * n_in]);
            }
        }
        y
    }

    fn dequant_act(xa: &crate::quant::QAct) -> Vec<f32> {
        let bpr = xa.n_in / QBLOCK;
        let mut out = vec![0f32; xa.t * xa.n_in];
        for ti in 0..xa.t {
            for i in 0..xa.n_in {
                out[ti * xa.n_in + i] = xa.q[ti * xa.n_in + i] as f32 * xa.scales[ti * bpr + i / QBLOCK];
            }
        }
        out
    }

    fn check(name: &str, y: &[f32], yr: &[f32]) {
        let mut max = 0f32;
        let mut scale = 0f32;
        for (a, b) in y.iter().zip(yr) {
            max = max.max((a - b).abs());
            scale = scale.max(b.abs());
        }
        assert!(max <= 1e-4 * scale.max(1.0), "{name}: max abs diff {max} (scale {scale})");
    }

    #[test]
    fn grouped_matmuls_match_per_segment() {
        crate::pool::init(4);
        let (n_out, n_in) = (48usize, 128usize);
        let mut seed = 3u64;
        // Segments of 8, 0, 16 and 72 rows (the last split into token chunks of 64).
        let segs = [(0usize, 8usize), (8, 0), (8, 16), (24, 72)];
        let rows = 96;
        let x: Vec<f32> = (0..rows * n_in).map(|_| lcg(&mut seed)).collect();
        let wb: Vec<Vec<u16>> = (0..segs.len())
            .map(|_| (0..n_out * n_in).map(|_| half::bf16::from_f32(lcg(&mut seed)).to_bits()).collect())
            .collect();
        let wf: Vec<Vec<f32>> = wb.iter().map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect()).collect();
        let want = |wf: &[Vec<f32>]| {
            let mut y = vec![0f32; rows * n_out];
            for (j, &(s, len)) in segs.iter().enumerate() {
                let r = reference(&wf[j], n_out, n_in, &x[s * n_in..(s + len) * n_in], len);
                y[s * n_out..(s + len) * n_out].copy_from_slice(&r);
            }
            y
        };
        let mut y = vec![0f32; rows * n_out];
        matmul_grouped(&wb.iter().map(|w| &w[..]).collect::<Vec<_>>(), &segs, n_out, n_in, &x, &mut y);
        check("grouped bf16", &y, &want(&wf));
        let q: Vec<_> = wb.iter().map(|w| quantize_q4(w, n_out, n_in)).collect();
        let qf: Vec<Vec<f32>> = q
            .iter()
            .map(|m| {
                let mut f = vec![0f32; n_out * n_in];
                dequantize_rows_q4(m, 0, n_out, &mut f);
                f
            })
            .collect();
        let mut y = vec![0f32; rows * n_out];
        matmul_quant_grouped(&q.iter().collect::<Vec<_>>(), &segs, &x, &mut y);
        check("grouped q4", &y, &want(&qf));
    }

    #[test]
    fn mlx_affine_q4_matches_reference() {
        crate::pool::init(2);
        let (n_out, n_in, group) = (37usize, 4096 + 64, 64usize);
        let gpr = n_in / group;
        let mut seed = 7u64;
        let words: Vec<u32> = (0..n_out * n_in / 8)
            .map(|_| lcg(&mut seed).to_bits() ^ (lcg(&mut seed).to_bits() << 7))
            .collect();
        let bf = |v: f32| half::bf16::from_f32(v).to_bits();
        let scales: Vec<u16> = (0..n_out * gpr).map(|_| bf(0.002 + 0.01 * lcg(&mut seed).abs())).collect();
        let biases: Vec<u16> = (0..n_out * gpr)
            .map(|i| bf(-8.0 * crate::kernels::bf16_to_f32(scales[i]) + 0.01 * lcg(&mut seed)))
            .collect();
        // Exact MLX values.
        let mut exact = vec![0f32; n_out * n_in];
        for r in 0..n_out {
            crate::quant::mlx_dequant_row(
                &words[r * n_in / 8..(r + 1) * n_in / 8],
                &scales[r * gpr..(r + 1) * gpr],
                &biases[r * gpr..(r + 1) * gpr],
                4,
                group,
                &mut exact[r * n_in..(r + 1) * n_in],
            );
        }
        let m = crate::quant::from_mlx_q4(&words, &scales, &biases, n_out, n_in, group);
        let mut conv = vec![0f32; n_out * n_in];
        dequantize_rows_q4(&m, 0, n_out, &mut conv);
        let err = exact.iter().zip(&conv).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 2e-4, "repacked values differ by {err}");
        for t in [1usize, 2, 3, 5, 9] {
            let x: Vec<f32> = (0..t * n_in).map(|_| lcg(&mut seed)).collect();
            let yr = reference(&exact, n_out, n_in, &x, t);
            let mut y = vec![0f32; t * n_out];
            matmul_quant_multi(&mut [(&m, &mut y[..])], &x, t);
            let (mut max, mut scale) = (0f32, 0f32);
            for (a, b) in y.iter().zip(&yr) {
                max = max.max((a - b).abs());
                scale = scale.max(b.abs());
            }
            // int8 activations: ~1% of the output range.
            assert!(max <= 1e-2 * scale.max(1.0), "t={t}: max abs diff {max} (scale {scale})");
        }
    }

    #[test]
    fn quantized_kernels_match_reference() {
        crate::pool::init(2);
        let (n_out, n_in) = (37usize, 4096 + 32); // odd rows, blocks not a multiple of 4
        let mut seed = 42u64;
        let w: Vec<u16> = (0..n_out * n_in).map(|_| (lcg(&mut seed).to_bits() >> 16) as u16).collect();
        for t in [1usize, 2, 3, 4, 5, 7] {
            let x: Vec<f32> = (0..t * n_in).map(|_| lcg(&mut seed) * 3.0).collect();
            let xa = quantize_act(&x, t, n_in);
            let xf = dequant_act(&xa);
            // Q8
            let q8 = quantize_bf16(&w, n_out, n_in);
            let mut w8 = vec![0f32; n_out * n_in];
            dequantize_rows(&q8, 0, n_out, &mut w8);
            let yr = reference(&w8, n_out, n_in, &xf, t);
            let mut y = vec![0f32; t * n_out];
            matmul_quant_multi(&mut [(&q8, &mut y[..])], &x, t);
            check(&format!("q8 t={t}"), &y, &yr);
            // Q4
            let q4 = quantize_q4(&w, n_out, n_in);
            let mut w4 = vec![0f32; n_out * n_in];
            dequantize_rows_q4(&q4, 0, n_out, &mut w4);
            let yr = reference(&w4, n_out, n_in, &xf, t);
            let mut y = vec![0f32; t * n_out];
            matmul_quant_multi(&mut [(&q4, &mut y[..])], &x, t);
            check(&format!("q4 t={t}"), &y, &yr);
        }
    }

    #[test]
    fn bf16_matmul_matches_reference() {
        crate::pool::init(2);
        let (n_out, n_in) = (37usize, 4096 + 32);
        let mut seed = 7u64;
        let w: Vec<u16> = (0..n_out * n_in).map(|_| (lcg(&mut seed).to_bits() >> 16) as u16).collect();
        let wf: Vec<f32> = w.iter().map(|&v| bf16_to_f32(v)).collect();
        for t in [1usize, 2, 3, 4, 7, 8, 9, 20] {
            let x: Vec<f32> = (0..t * n_in).map(|_| lcg(&mut seed)).collect();
            let yr = reference(&wf, n_out, n_in, &x, t);
            let mut y = vec![0f32; t * n_out];
            matmul(&w, n_out, n_in, &x, t, &mut y);
            check(&format!("bf16 t={t}"), &y, &yr);
        }
    }
}
