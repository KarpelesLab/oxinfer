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

/// A zeroed `Vec<f32>` of `n` elements, filled across the pool when large. A prompt's per-layer
/// temporaries are megabytes each and fully written by the matmuls that follow: zeroing them on
/// the calling thread alone (what `vec![0f32; n]` does once the allocator reuses freed blocks)
/// was most of the prefill time outside the kernels.
pub fn zeros(n: usize) -> Vec<f32> {
    /// Below this (1 MiB), the parallel region costs more than it saves.
    const MIN: usize = 1 << 18;
    if n < MIN || pool::threads() == 1 {
        return vec![0f32; n];
    }
    let mut v: Vec<f32> = Vec::with_capacity(n);
    let p = SendPtr(v.as_mut_ptr());
    let chunk = (n / (pool::threads() * 2)).max(MIN / 4);
    pool::run(n.div_ceil(chunk), &|i| {
        let (a, b) = (i * chunk, ((i + 1) * chunk).min(n));
        // SAFETY: the chunks are disjoint and inside the allocation.
        unsafe { std::ptr::write_bytes(p.get().add(a), 0, b - a) };
    });
    // SAFETY: all `n` elements were written above (zero bytes are 0.0).
    unsafe { v.set_len(n) };
    v
}

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
    #[cfg(target_arch = "aarch64")]
    if t > decode_max_tokens() && prefill_kernel() == PrefillGemm::Sme && sme::fits(n_in) {
        // Prefill on the SME tiles, through the quantized driver (bf16 rows widen exactly).
        let mats: Vec<Bf16Mat> = jobs.iter().map(|(w, n_out, _)| Bf16Mat { w, n_out: *n_out, n_in }).collect();
        let mut qj: Vec<(&Bf16Mat, &mut [f32])> = mats.iter().zip(jobs.iter_mut()).map(|(m, (_, _, y))| (m, &mut **y)).collect();
        matmul_quant_multi(&mut qj, x, t);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if t >= TN && n_in.is_multiple_of(32) && simd::avx512() {
        // Prefill on the cache-blocked AVX-512 GEMM, through the quantized driver (bf16 rows
        // widen exactly, and the GEMM below takes these shapes too: the same bits).
        let mats: Vec<Bf16Mat> = jobs.iter().map(|(w, n_out, _)| Bf16Mat { w, n_out: *n_out, n_in }).collect();
        let mut qj: Vec<(&Bf16Mat, &mut [f32])> = mats.iter().zip(jobs.iter_mut()).map(|(m, (_, _, y))| (m, &mut **y)).collect();
        matmul_quant_multi_at(&mut qj, x, t, TN - 1);
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
fn grouped_blocks(segs: &[(usize, usize)], n_out: usize, row_bytes: usize, rq: usize) -> Vec<(u32, u32, u32, u32, u32)> {
    let total = segs.iter().filter(|s| s.1 > 0).count() * n_out * row_bytes;
    let target = (total / (pool::threads() * 8)).max(64 * 1024);
    let rows_per_block = (target / row_bytes).max(rq) / rq * rq;
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
    let blocks = grouped_blocks(segs, n_out, n_in * 2, 16);
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
    #[cfg(target_arch = "aarch64")]
    if prefill_kernel() == PrefillGemm::Sme && sme::fits(n_in) {
        // `Hybrid` / `Fast`: the int8 GEMM on the NEON units takes a share of the blocks (all
        // of them with `grouped_sme_share` 0), as in `matmul_quant_multi`.
        let selected = prefill_effective();
        let int8 = matches!(selected, PrefillGemm::Hybrid | PrefillGemm::Fast) && ms.iter().all(|m| m.i8_layout().is_some());
        let share = if int8 {
            grouped_sme_share(selected == PrefillGemm::Fast)
        } else {
            16
        };
        let blocks = grouped_blocks(segs, n_out, (ms[0].bytes() / n_out).max(1), 32);
        let on_sme = |b: usize| share == 16 || b * share / 16 != (b + 1) * share / 16;
        // Each segment's tokens as their own 32-token panels (chunks of 64 tokens from the
        // segment start, so whole panels).
        let mut base = Vec::with_capacity(segs.len() + 1);
        let mut tiles = 0;
        for &(_, len) in segs {
            base.push(tiles);
            tiles += len.div_ceil(32);
        }
        base.push(tiles);
        let xt = if share > 0 {
            let mut xt = vec![0f32; tiles * 32 * n_in];
            let xp = SendPtr(xt.as_mut_ptr());
            pool::run(tiles, &|i| {
                let j = base.partition_point(|&b| b <= i) - 1;
                let (s, len) = segs[j];
                let k = i - base[j];
                // SAFETY: each panel is written by one task; rows past the segment are not read.
                unsafe {
                    sme::pack32(
                        x.as_ptr().add((s + 32 * k) * n_in),
                        n_in,
                        (len - 32 * k).min(32),
                        n_in,
                        xp.get().add(i * 32 * n_in),
                    )
                }
            });
            xt
        } else {
            Vec::new()
        };
        let xa = int8.then(|| i8gemm::I8Act::new(x, rows, n_in, selected == PrefillGemm::Hybrid));
        // The int8 blocks skip a segment's trailing zero rows (its padding: their outputs are
        // zero whatever the weights) and store zeros there.
        let ends: Vec<usize> = segs
            .iter()
            .map(|&(s, len)| {
                let mut e = s + len;
                while e > s && x[(e - 1) * n_in..e * n_in].iter().all(|&v| v == 0.0) {
                    e -= 1;
                }
                e
            })
            .collect();
        let yp = SendPtr(y.as_mut_ptr());
        let run = |b: usize| {
            let (j, r0, r1, c0, c1) = blocks[b];
            let (s, _) = segs[j as usize];
            let m = ms[j as usize];
            let (r0, r1, c0, c1) = (r0 as usize, r1 as usize, c0 as usize, c1 as usize);
            if on_sme(b) {
                let xs = &xt[base[j as usize] * 32 * n_in..base[j as usize + 1] * 32 * n_in];
                unsafe { sme::gemm_rows(m, xs, c0 - s, c1 - s, r0, r1, yp.get().add(s * n_out)) }
            } else {
                let e = ends[j as usize].clamp(c0, c1);
                unsafe { i8gemm::gemm_rows(m, xa.as_ref().unwrap(), c0, e, r0, r1, yp.get()) }
                for c in e..c1 {
                    // SAFETY: this block owns rows r0..r1 of these tokens.
                    unsafe { std::ptr::write_bytes(yp.get().add(c * n_out + r0), 0, r1 - r0) }
                }
            }
        };
        if share == 16 || share == 0 {
            pool::run(blocks.len(), &run);
        } else {
            let (sq, nq): (Vec<usize>, Vec<usize>) = (0..blocks.len()).partition(|&b| on_sme(b));
            two_queues(&sq, &nq, &run);
        }
        return;
    }
    let blocks = grouped_blocks(segs, n_out, (ms[0].bytes() / n_out).max(1), 16);
    #[cfg(target_arch = "aarch64")]
    if let Some(two) = i8_planes().filter(|_| ms.iter().all(|m| m.i8_layout().is_some())) {
        let xa = i8gemm::I8Act::new(x, rows, n_in, two);
        let yp = SendPtr(y.as_mut_ptr());
        pool::run(blocks.len(), &|b| {
            let (j, r0, r1, c0, c1) = blocks[b];
            let m = ms[j as usize];
            unsafe { i8gemm::gemm_rows(m, &xa, c0 as usize, c1 as usize, r0 as usize, r1 as usize, yp.get()) }
        });
        return;
    }
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

    /// Loads 4 consecutive weights (or one) as f32.
    pub trait LoadW4: Copy {
        unsafe fn load4(p: *const Self) -> float32x4_t;
        unsafe fn load1(p: *const Self) -> f32;
    }
    impl LoadW4 for u16 {
        #[inline(always)]
        unsafe fn load4(p: *const u16) -> float32x4_t {
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(p)))
        }
        #[inline(always)]
        unsafe fn load1(p: *const u16) -> f32 {
            super::bf16_to_f32(*p)
        }
    }
    impl LoadW4 for f32 {
        #[inline(always)]
        unsafe fn load4(p: *const f32) -> float32x4_t {
            vld1q_f32(p)
        }
        #[inline(always)]
        unsafe fn load1(p: *const f32) -> f32 {
            *p
        }
    }

    /// Prefill GEMM micro-kernel: 8 weight rows (row-major, starting at `w`) times TN=8 tokens
    /// from a packed panel `xt` ([n_in][8]). Weights are loaded 4 at a time and broadcast by lane
    /// (the last `n_in % 4` one at a time); accumulators are vectors over tokens. Stores rows
    /// `rb..rb+8` for tokens `c0..c0+c_valid`.
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
        let n4 = n_in / 4 * 4;
        while k < n4 {
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
        // A width that is not a multiple of 4 (small float-stored matrices): a 4-wide step would
        // read the next row's weights and the next tile's activations.
        while k < n_in {
            let xa = vld1q_f32(xt.add(k * 8));
            let xb = vld1q_f32(xt.add(k * 8 + 4));
            for r in 0..8 {
                let wv = W::load1(w.add(r * n_in + k));
                acc[r][0] = vfmaq_n_f32(acc[r][0], xa, wv);
                acc[r][1] = vfmaq_n_f32(acc[r][1], xb, wv);
            }
            k += 1;
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

    /// Q4_0: one weight row against TC quantized activation rows (token stride `n_in`) with the
    /// row's four blocks per step unpacked once into registers and each token's activations
    /// streamed from L1 (`q4_dot_rows_tc` holds all tokens' activations instead and spills past
    /// two tokens). Every token's operations are `q4_dot_rows`'s, so results are bit-identical
    /// to it. (The int8 format gains nothing from this: its blocks need no unpacking.)
    #[inline(always)]
    pub unsafe fn q4_dot_cols<const TC: usize>(
        q: *const u8,
        s: *const u16,
        n_in: usize,
        xq: *const i8,
        xs: *const f32,
        xsums: *const i32,
    ) -> [f32; TC] {
        let bpr = n_in / 32;
        let zero = vdupq_n_s32(0);
        let m0f = vdupq_n_u8(0x0F);
        let mut accf = [vdupq_n_f32(0.0); TC];
        let mut b = 0;
        while b + 4 <= bpr {
            let k = b * 32;
            let mut w = [vdupq_n_s8(0); 8];
            for i in 0..4 {
                let v = vld1q_u8(q.add(b * 16 + 16 * i));
                w[2 * i] = vreinterpretq_s8_u8(vandq_u8(v, m0f));
                w[2 * i + 1] = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
            }
            let swv = f16x4_to_f32(s.add(b));
            for (c, acc) in accf.iter_mut().enumerate() {
                let xp = xq.add(c * n_in + k);
                let a0 = sdot(sdot(zero, w[0], vld1q_s8(xp)), w[1], vld1q_s8(xp.add(16)));
                let a1 = sdot(sdot(zero, w[2], vld1q_s8(xp.add(32))), w[3], vld1q_s8(xp.add(48)));
                let a2 = sdot(sdot(zero, w[4], vld1q_s8(xp.add(64))), w[5], vld1q_s8(xp.add(80)));
                let a3 = sdot(sdot(zero, w[6], vld1q_s8(xp.add(96))), w[7], vld1q_s8(xp.add(112)));
                let xoff = vshlq_n_s32::<3>(vld1q_s32(xsums.add(c * bpr + b)));
                let sums = vsubq_s32(vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)), xoff);
                *acc = vfmaq_f32(*acc, vcvtq_f32_s32(sums), vmulq_f32(swv, vld1q_f32(xs.add(c * bpr + b))));
            }
            b += 4;
        }
        let mut out = [0f32; TC];
        for c in 0..TC {
            out[c] = vaddvq_f32(accf[c]);
        }
        while b < bpr {
            let k = b * 32;
            let v = vld1q_u8(q.add(b * 16));
            let lo = vreinterpretq_s8_u8(vandq_u8(v, m0f));
            let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
            let sw = crate::quant::f16_to_f32(*s.add(b));
            for (c, o) in out.iter_mut().enumerate() {
                let xp = xq.add(c * n_in + k);
                let a = sdot(sdot(zero, lo, vld1q_s8(xp)), hi, vld1q_s8(xp.add(16)));
                let sum = vaddvq_s32(a) - 8 * *xsums.add(c * bpr + b);
                *o += sum as f32 * sw * *xs.add(c * bpr + b);
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
            #[cfg(target_arch = "x86_64")]
            if simd::avx512() {
                // Two panels and 16 rows per step, the same bits.
                while rr < full {
                    let rows = (full - rr).min(16);
                    simd::gemm_tiles_avx512::<u16>(wp.add(rr * n_in), rows, n_in, xt.as_ptr(), g0, g1, t, y, n_out, rr);
                    rr += rows;
                }
            }
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

/// bf16 weights as a float-stored matrix of the quantized driver (its SME prefill path).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct Bf16Mat<'a> {
    w: &'a [u16],
    n_out: usize,
    n_in: usize,
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl QMat for Bf16Mat<'_> {
    fn n_out(&self) -> usize {
        self.n_out
    }
    fn n_in(&self) -> usize {
        self.n_in
    }
    fn bytes(&self) -> usize {
        self.w.len() * 2
    }
    fn dense(&self) -> bool {
        true
    }
    fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        bf16_to_f32_slice(&self.w[r0 * self.n_in..r1 * self.n_in], &mut out[..(r1 - r0) * self.n_in]);
    }
    fn dequantize_block(&self, r0: usize, r1: usize, k0: usize, k1: usize, out: &mut [f32]) -> bool {
        let n = k1 - k0;
        for (i, r) in (r0..r1).enumerate() {
            bf16_to_f32_slice(&self.w[r * self.n_in + k0..r * self.n_in + k1], &mut out[i * n..(i + 1) * n]);
        }
        true
    }
    unsafe fn rows_t1(&self, _: &QAct, _: usize, _: *mut f32, _: usize, _: usize) {
        unreachable!("bf16 rows run on the GEMM (dense)")
    }
    unsafe fn rows_tc(&self, _: &QAct, _: usize, _: usize, _: *mut f32, _: usize, _: usize) {
        unreachable!("bf16 rows run on the GEMM (dense)")
    }
}

/// A quantized weight matrix usable by the shared quantized-matmul driver.
pub trait QMat: Sync {
    fn n_out(&self) -> usize;
    fn n_in(&self) -> usize;
    fn bytes(&self) -> usize;
    fn dequantize_rows(&self, r0: usize, r1: usize, out: &mut [f32]);
    /// Columns `k0..k1` of rows `r0..r1` into `out` (`[r1 - r0][k1 - k0]`, row-major): the
    /// values `dequantize_rows` gives there. False (nothing written) when the format cannot be
    /// cut at these columns; the cache-blocked prefill GEMM then leaves the matrix to the
    /// whole-row path.
    fn dequantize_block(&self, _r0: usize, _r1: usize, _k0: usize, _k1: usize, _out: &mut [f32]) -> bool {
        false
    }
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
    /// How rows unpack for the int8 prefill GEMM, if they do (aarch64, and x86-64 with AVX-512
    /// VNNI).
    fn i8_layout(&self) -> Option<I8Layout> {
        None
    }
    /// x86-64: columns `k0..k1` (whole blocks of the format, multiples of 32) of row `r` for the
    /// int8 prefill GEMM: `q` gets the `k1 - k0` int8 values, `s` one f32 scale per 16 inputs
    /// (equal within a block of 32 unless `sc16`) and, with `min32`, `m` one offset per 16
    /// inputs (equal within a block of 32): input `i` is `s[i / 16] * q[i] + m[i / 16]`.
    unsafe fn unpack_cols_i8(&self, _r: usize, _k0: usize, _k1: usize, _q: *mut i8, _s: *mut f32, _m: *mut f32) {
        unreachable!("no int8 GEMM layout")
    }
    /// Unpacks row `r` for the int8 prefill GEMM (see [`I8Layout`]): `q` gets `n_in` values,
    /// `ds` and (`min32`) `mn` `n_in / 32`, (`sc16`) `sc` `n_in / 16`.
    unsafe fn unpack_row_i8(&self, _r: usize, _q: *mut i8, _ds: *mut f32, _mn: *mut f32, _sc: *mut i32) {
        unreachable!("no int8 GEMM layout")
    }
}

/// How a matrix's rows unpack for the int8 prefill GEMM. Row element `i` (in block `b = i / 32`)
/// is `ds[b] * sc[i / 16] * q[i]` (`sc = 1` unless `sc16`), and with `min32` the dot product
/// also subtracts `mn[b]` times the exact f32 sum of the activations of block `b`
/// ([`QMat::unpack_row_i8`], aarch64). On x86-64 ([`QMat::unpack_cols_i8`]) `sc16` says that the
/// f32 scales differ within a block of 32 and `min32` that the rows have offsets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct I8Layout {
    pub sc16: bool,
    pub min32: bool,
}

#[cfg(target_arch = "aarch64")]
#[path = "kernels_i8gemm.rs"]
pub(crate) mod i8gemm;

#[cfg(target_arch = "x86_64")]
#[path = "kernels_i8gemm_x86.rs"]
pub(crate) mod i8gemm;

#[cfg(target_arch = "aarch64")]
#[path = "kernels_sme.rs"]
pub(crate) mod sme;

impl QMat for QMatrix {
    fn n_out(&self) -> usize {
        self.n_out
    }
    #[cfg(target_arch = "aarch64")]
    fn i8_layout(&self) -> Option<I8Layout> {
        Some(I8Layout { sc16: false, min32: false })
    }
    #[cfg(target_arch = "aarch64")]
    unsafe fn unpack_row_i8(&self, r: usize, q: *mut i8, ds: *mut f32, _mn: *mut f32, _sc: *mut i32) {
        let (n, bpr) = (self.n_in, self.n_in / QBLOCK);
        std::ptr::copy_nonoverlapping(self.q.as_ptr().add(r * n), q, n);
        std::ptr::copy_nonoverlapping(self.scales.as_ptr().add(r * bpr), ds, bpr);
    }
    #[cfg(target_arch = "x86_64")]
    fn i8_layout(&self) -> Option<I8Layout> {
        Some(I8Layout { sc16: false, min32: false })
    }
    #[cfg(target_arch = "x86_64")]
    unsafe fn unpack_cols_i8(&self, r: usize, k0: usize, k1: usize, q: *mut i8, s: *mut f32, _m: *mut f32) {
        let (n, bpr) = (self.n_in, self.n_in / QBLOCK);
        std::ptr::copy_nonoverlapping(self.q.as_ptr().add(r * n + k0), q, k1 - k0);
        for b in k0 / QBLOCK..k1 / QBLOCK {
            let d = self.scales[r * bpr + b];
            *s.add(2 * (b - k0 / QBLOCK)) = d;
            *s.add(2 * (b - k0 / QBLOCK) + 1) = d;
        }
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
    fn dequantize_block(&self, r0: usize, r1: usize, k0: usize, k1: usize, out: &mut [f32]) -> bool {
        if !k0.is_multiple_of(QBLOCK) || !k1.is_multiple_of(QBLOCK) || k0 >= k1 || k1 > self.n_in {
            return false;
        }
        crate::quant::dequantize_block(self, r0, r1, k0, k1, out);
        true
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
    #[cfg(target_arch = "aarch64")]
    fn i8_layout(&self) -> Option<I8Layout> {
        Some(I8Layout {
            sc16: false,
            min32: self.offsets.is_some(),
        })
    }
    #[cfg(target_arch = "aarch64")]
    unsafe fn unpack_row_i8(&self, r: usize, q: *mut i8, ds: *mut f32, mn: *mut f32, _sc: *mut i32) {
        use std::arch::aarch64::*;
        let (n, bpr) = (self.n_in, self.n_in / QBLOCK);
        let src = self.q.as_ptr().add(r * n / 2);
        let (m4, eight) = (vdupq_n_u8(0x0F), vdupq_n_s8(8));
        for b in 0..bpr {
            let v = vld1q_u8(src.add(16 * b));
            vst1q_s8(q.add(32 * b), vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), eight));
            vst1q_s8(q.add(32 * b + 16), vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), eight));
            *ds.add(b) = crate::quant::f16_to_f32(self.scales[r * bpr + b]);
        }
        if let Some(o) = &self.offsets {
            let gpr = n / o.group;
            for b in 0..bpr {
                *mn.add(b) = -bf16_to_f32(o.v[r * gpr + b * QBLOCK / o.group]);
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    fn i8_layout(&self) -> Option<I8Layout> {
        // Offsets per group of whole blocks only.
        if self.offsets.as_ref().is_some_and(|o| !o.group.is_multiple_of(QBLOCK)) {
            return None;
        }
        Some(I8Layout {
            sc16: false,
            min32: self.offsets.is_some(),
        })
    }
    #[cfg(target_arch = "x86_64")]
    unsafe fn unpack_cols_i8(&self, r: usize, k0: usize, k1: usize, q: *mut i8, s: *mut f32, m: *mut f32) {
        simd::q4_unpack_cols(self, r, k0, k1, q, s, m)
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
    fn dequantize_block(&self, r0: usize, r1: usize, k0: usize, k1: usize, out: &mut [f32]) -> bool {
        if !k0.is_multiple_of(QBLOCK) || !k1.is_multiple_of(QBLOCK) || k0 >= k1 || k1 > self.n_in {
            return false;
        }
        crate::quant::dequantize_block_q4(self, r0, r1, k0, k1, out);
        true
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
    #[cfg(target_arch = "aarch64")]
    fn multi(&self, _t: usize) -> Option<bool> {
        Some(false)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe fn rows_multi(&self, xa: &QAct, _pairs: Option<&PairAct>, t: usize, y: *mut f32, r0: usize, r1: usize) -> bool {
        let (n_in, n_out) = (self.n_in, self.n_out);
        let bpr = n_in / QBLOCK;
        col_tiles(t, n_in, r0, r1, n_in / 2, |c0, c1, ta, tb| {
            let xq = xa.q.as_ptr().add(c0 * n_in);
            let xs = xa.scales.as_ptr().add(c0 * bpr);
            let xsums = xa.sums.as_ptr().add(c0 * bpr);
            let yg = y.add(c0 * n_out);
            macro_rules! go {
                ($TC:literal) => {
                    for r in ta..tb {
                        let o = simd::q4_dot_cols::<$TC>(
                            self.q.as_ptr().add(r * n_in / 2),
                            self.scales.as_ptr().add(r * bpr),
                            n_in,
                            xq,
                            xs,
                            xsums,
                        );
                        for (c, v) in o.iter().enumerate() {
                            *yg.add(c * n_out + r) = *v;
                        }
                    }
                };
            }
            with_cols!(c1 - c0, go)
        });
        for c in 0..t {
            self.add_offsets(xa, c, y.add(c * n_out), r0, r1);
        }
        true
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

/// Kernels for quantized (and, with SME, float-stored) prefill matmuls on the CPU (more than
/// [`decode_max_tokens`] tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillGemm {
    /// The default: `Fast` (llama.cpp's int8 activation rounding on most rows, SME beside it
    /// where present; `F32` on x86-64 without AVX-512 VNNI and elsewhere) in a language model's
    /// prompt processing (inside
    /// [`lm_prefill`]), `Exact` everywhere else (vision and audio towers, diffusion models,
    /// embeddings, text encoders). [`PrefillGemm::Exact`] opts out everywhere.
    Auto,
    /// The exact products: `Sme` where the CPU has SME (Apple M4 and later), else `F32`. Logits
    /// as with the f32 GEMM (bit-identical on Llama 3.2 and Qwen3).
    Exact,
    /// The f32 GEMM on the Scalable Matrix Extension's outer-product tiles (Apple M4 and later):
    /// f32 activations, each product accumulated in input order like the NEON FMA GEMM (logits
    /// bit-identical to `F32` on Llama 3.2 and Qwen3, KL 4e-7 on Gemma 4 12B); `F32` without SME.
    Sme,
    /// The SME GEMM on part of the row blocks (7/16, `OXINFER_SME_SHARE`) and `Int8x2` on the
    /// cores' NEON units for the rest, at the same time (~10-20% faster than `Sme` alone); which
    /// kernel computes a block does not depend on timing. `Int8x2` without SME.
    Hybrid,
    /// `Hybrid` with `Int8` on the NEON units (5/16 of the blocks on SME): the fastest, with
    /// llama.cpp's activation rounding on most rows (KL ~1e-4..1e-3 against `F32`). `Int8`
    /// without SME, also on x86-64 with AVX-512 VNNI.
    Fast,
    /// The int8 SDOT GEMM (aarch64) with the activations as two int8 planes per block of 32 (a
    /// value and its rounding residual: about 16-bit precision); `F32` elsewhere.
    Int8x2,
    /// The int8 GEMM with one int8 plane, the decode kernels' rounding: faster than `Int8x2`,
    /// less precise (KL ~1e-3 against `F32` after a 2.5k-token prompt). aarch64 (`SDOT`) and
    /// x86-64 with AVX-512 VNNI (`vpdpbusd`); `F32` elsewhere.
    Int8,
    /// Dequantized weights times f32 activations on the NEON / AVX2 FMA GEMM.
    F32,
}

impl std::str::FromStr for PrefillGemm {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "auto" | "default" | "" => Self::Auto,
            "exact" | "precise" => Self::Exact,
            "sme" => Self::Sme,
            "hybrid" => Self::Hybrid,
            "fast" => Self::Fast,
            "int8x2" => Self::Int8x2,
            "int8" => Self::Int8,
            "f32" | "float" | "fma" | "off" | "0" => Self::F32,
            _ => anyhow::bail!("prefill GEMM must be auto, exact, sme, hybrid, fast, int8x2, int8 or f32, not {s:?}"),
        })
    }
}

static PREFILL_GEMM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Selects the CPU prefill GEMM (default `Auto`, i.e. `Fast`; also
/// `OXINFER_PREFILL_GEMM=auto|exact|sme|hybrid|fast|int8x2|int8|f32`). Decoding (up to [`decode_max_tokens`]
/// tokens) is not affected.
pub fn set_prefill_gemm(g: PrefillGemm) {
    let v = match g {
        PrefillGemm::Auto => 1,
        PrefillGemm::Sme => 2,
        PrefillGemm::Hybrid => 3,
        PrefillGemm::Fast => 4,
        PrefillGemm::Int8x2 => 5,
        PrefillGemm::Int8 => 6,
        PrefillGemm::F32 => 7,
        PrefillGemm::Exact => 8,
    };
    PREFILL_GEMM.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// The CPU prefill GEMM selected (see [`prefill_kernel`] for the one that runs).
pub fn prefill_gemm() -> PrefillGemm {
    let mut v = PREFILL_GEMM.load(std::sync::atomic::Ordering::Relaxed);
    if v == 0 {
        let g = std::env::var("OXINFER_PREFILL_GEMM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PrefillGemm::Auto);
        set_prefill_gemm(g);
        v = PREFILL_GEMM.load(std::sync::atomic::Ordering::Relaxed);
    }
    match v {
        1 => PrefillGemm::Auto,
        2 => PrefillGemm::Sme,
        3 => PrefillGemm::Hybrid,
        4 => PrefillGemm::Fast,
        5 => PrefillGemm::Int8x2,
        6 => PrefillGemm::Int8,
        8 => PrefillGemm::Exact,
        _ => PrefillGemm::F32,
    }
}

thread_local! {
    /// Set while this thread (or the pool job it runs for such a thread) processes a language
    /// model's prompt: `PrefillGemm::Auto` means `Fast` there.
    // Already const; clippy (1.100) still flags it on Android's thread-local implementation.
    #[allow(clippy::missing_const_for_thread_local)]
    static LM_PREFILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` as a language model's prompt processing: the default prefill GEMM ([`PrefillGemm::Auto`])
/// takes llama.cpp's int8 activation rounding ([`PrefillGemm::Fast`]) for the matmuls it runs,
/// also those the thread pool runs for it.
pub fn lm_prefill<R>(f: impl FnOnce() -> R) -> R {
    let prev = LM_PREFILL.with(|c| c.replace(true));
    let r = f();
    LM_PREFILL.with(|c| c.set(prev));
    r
}

/// Whether this thread is inside [`lm_prefill`].
pub(crate) fn in_lm_prefill() -> bool {
    LM_PREFILL.with(|c| c.get())
}

/// Sets the [`lm_prefill`] flag of a pool worker to its caller's for one job.
pub(crate) fn with_lm_prefill<R>(on: bool, f: impl FnOnce() -> R) -> R {
    if !on {
        return f();
    }
    lm_prefill(f)
}

/// The selected prefill GEMM with `Auto` and `Exact` resolved (`Fast` and `Sme`).
fn prefill_effective() -> PrefillGemm {
    match prefill_gemm() {
        PrefillGemm::Auto if in_lm_prefill() => PrefillGemm::Fast,
        PrefillGemm::Auto => PrefillGemm::Sme,
        PrefillGemm::Exact => PrefillGemm::Sme,
        g => g,
    }
}

/// The CPU prefill GEMM that runs on this machine for the selected [`prefill_gemm`]: `Sme`
/// (also for `Hybrid` and `Fast` with SME), `Int8x2` (aarch64), `Int8` (aarch64, x86-64 with
/// AVX-512 VNNI) or `F32`.
pub fn prefill_kernel() -> PrefillGemm {
    let sme = sme_available();
    let arm = cfg!(target_arch = "aarch64");
    // One int8 plane: aarch64, and x86-64 with AVX-512 VNNI.
    #[cfg(target_arch = "x86_64")]
    let int8 = i8gemm::available();
    #[cfg(not(target_arch = "x86_64"))]
    let int8 = arm;
    match prefill_effective() {
        PrefillGemm::Sme | PrefillGemm::Hybrid | PrefillGemm::Fast if sme => PrefillGemm::Sme,
        PrefillGemm::Hybrid if arm => PrefillGemm::Int8x2,
        PrefillGemm::Fast if int8 => PrefillGemm::Int8,
        PrefillGemm::Int8x2 if arm => PrefillGemm::Int8x2,
        PrefillGemm::Int8 if int8 => PrefillGemm::Int8,
        _ => PrefillGemm::F32,
    }
}

/// Whether the CPU has SME in the shape the SME GEMM uses (512-bit streaming vectors).
pub fn sme_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        sme::available()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Whether this pass's int8 GEMM takes two activation planes (None: not the int8 GEMM).
fn i8_planes() -> Option<bool> {
    match prefill_kernel() {
        PrefillGemm::Int8x2 => Some(true),
        PrefillGemm::Int8 => Some(false),
        _ => None,
    }
}

static MOE_UNBATCHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// For benchmarks: `false` runs the routed experts of MoE decode and verification passes one
/// matmul at a time (the older path) instead of one batch per stage. Results are identical.
#[doc(hidden)]
pub fn set_moe_batched(on: bool) {
    MOE_UNBATCHED.store(!on, std::sync::atomic::Ordering::Relaxed);
}

/// See [`set_moe_batched`].
pub(crate) fn moe_batched() -> bool {
    !MOE_UNBATCHED.load(std::sync::atomic::Ordering::Relaxed)
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

/// The widest column (token) group of the multi-token kernels that instantiate a one-row kernel
/// per token count (the blocks-of-32 formats: GGUF Q8_0, Q4_0, ..., and Q4).
pub(crate) const COLS_MAX: usize = 8;

/// Weight bytes per row tile of those kernels: each tile is re-read (from cache) once per column
/// group while the group's activations stay in L1.
const COLS_TILE_BYTES: usize = 64 * 1024;

/// Int8 activation bytes a column group may span (`n_in` per token), so that they stay in L1
/// while a row tile streams past.
const COLS_ACT_BYTES: usize = 64 * 1024;

/// Runs `f(c0, c1, ta, tb)` over row tiles `ta..tb` of `r0..r1` (rows of `row_bytes`) and, within
/// each tile, over column groups `c0..c1` of `t` tokens of `n_in` int8 activations: groups of at
/// most `COLS_MAX` tokens, and of at most `COLS_ACT_BYTES` of activations (but 2 tokens at
/// least), as few and as even as possible (9 = 5 + 4, 16 = 8 + 8).
pub(crate) fn col_tiles(t: usize, n_in: usize, r0: usize, r1: usize, row_bytes: usize, mut f: impl FnMut(usize, usize, usize, usize)) {
    let wmax = (COLS_ACT_BYTES / n_in.max(1)).clamp(2, COLS_MAX);
    let n = t.div_ceil(wmax);
    let tile = (COLS_TILE_BYTES / row_bytes.max(1)).max(4);
    let mut ta = r0;
    while ta < r1 {
        let tb = (ta + tile).min(r1);
        for g in 0..n {
            f(g * t / n, (g + 1) * t / n, ta, tb);
        }
        ta = tb;
    }
}

/// `$go!(W)` with the literal `W` equal to the column-group width `$w` (1..=`COLS_MAX`).
macro_rules! with_cols {
    ($w:expr, $go:ident) => {
        match $w {
            1 => $go!(1),
            2 => $go!(2),
            3 => $go!(3),
            4 => $go!(4),
            5 => $go!(5),
            6 => $go!(6),
            7 => $go!(7),
            8 => $go!(8),
            w => unreachable!("column group of {w}"),
        }
    };
}
pub(crate) use with_cols;

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
    // Multiples of 32 rows (whole SME panels), 16 is enough elsewhere.
    let rq = if t > decode_max && prefill_kernel() == PrefillGemm::Sme {
        32
    } else {
        16
    };
    let rows_per_block = (target / bytes_per_row).max(rq) / rq * rq;
    // Decode: int8 dot products (float-stored weights: dequantize + f32 GEMM). Prefill: the
    // selected `PrefillGemm` (see `Path`).
    let gemm = t > decode_max;
    // The cache-blocked AVX-512 GEMM: enough rows per job to reuse each block of activations,
    // few enough tokens for a block to stay in L2 (see `gemm_tuning`).
    #[cfg(target_arch = "x86_64")]
    let blocked = (gemm && simd::avx512()).then(gemm_tuning);
    #[cfg(not(target_arch = "x86_64"))]
    let blocked: Option<(usize, usize, usize)> = None;
    let rows_per_block = blocked.map_or(rows_per_block, |(_, rows, _)| rows_per_block.clamp(rows, 2 * rows));
    // The int8 GEMM of x86-64 (when any matrix takes it): its own rows and tokens per job (each
    // row is unpacked once per job, so a job takes as many tokens as its cache blocks allow).
    #[cfg(target_arch = "x86_64")]
    let i8_job = (gemm && i8_planes() == Some(false) && refs.iter().any(|j| j.m.i8_layout().is_some())).then(i8gemm::tuning);
    #[cfg(not(target_arch = "x86_64"))]
    let i8_job: Option<(usize, usize, usize, usize)> = None;
    let rows_per_block = i8_job.map_or(rows_per_block, |(rows, ..)| rows);
    let mut blocks: Vec<(u32, u32, u32)> = Vec::new();
    for (j, job) in refs.iter().enumerate() {
        let mut r = 0;
        while r < job.m.n_out() {
            let r1 = (r + rows_per_block).min(job.m.n_out());
            blocks.push((j as u32, r as u32, r1 as u32));
            r = r1;
        }
    }
    #[cfg(target_arch = "aarch64")]
    let sme_on = gemm && prefill_kernel() == PrefillGemm::Sme && sme::fits(n_in);
    #[cfg(not(target_arch = "aarch64"))]
    let sme_on = false;
    // `Hybrid` / `Fast` with SME: the SME units take a share of the row blocks and the cores'
    // NEON the rest (int8x2 / int8), at the same time.
    let selected = prefill_effective();
    let hybrid = sme_on && matches!(selected, PrefillGemm::Hybrid | PrefillGemm::Fast);
    let planes = if !gemm {
        None
    } else if hybrid {
        Some(selected == PrefillGemm::Hybrid)
    } else {
        i8_planes()
    };
    let (tchunks, tc) = if gemm { token_split(blocks.len(), t, nthreads) } else { (1, t) };
    let (tchunks, tc) = match blocked {
        Some((_, _, tokens)) if tc > tokens => (t.div_ceil(tokens), tokens),
        _ => (tchunks, tc),
    };
    let (tchunks, tc) = match i8_job {
        Some((_, tokens, ..)) => (t.div_ceil(tokens), tokens),
        None => (tchunks, tc),
    };
    // SME: whole 32-token tiles per chunk.
    let (tchunks, tc) = if sme_on {
        (t.div_ceil(tc.next_multiple_of(32)), tc.next_multiple_of(32))
    } else {
        (tchunks, tc)
    };
    let (share, _) = hybrid_tuning(selected == PrefillGemm::Fast);
    let path = |b: usize| -> Path {
        let m = refs[blocks[b / tchunks].0 as usize].m;
        let int8 = planes.is_some() && m.i8_layout().is_some();
        if !gemm {
            if m.dense() { Path::F32 } else { Path::Decode }
        } else if sme_on && !(int8 && hybrid && (b * share / 16 == (b + 1) * share / 16)) {
            Path::Sme
        } else if int8 {
            Path::I8
        } else {
            Path::F32
        }
    };
    let n_jobs = blocks.len() * tchunks;
    let mut uses = [false; 4];
    for b in 0..n_jobs {
        uses[path(b) as usize] = true;
    }
    let xt = uses[Path::F32 as usize].then(|| pack_xt(x, t, n_in));
    #[cfg(target_arch = "aarch64")]
    let xs32 = uses[Path::Sme as usize].then(|| sme::pack_x(x, t, n_in));
    let xa = uses[Path::Decode as usize].then(|| quantize_act(x, t, n_in));
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let xg = uses[Path::I8 as usize].then(|| i8gemm::I8Act::new(x, t, n_in, planes.unwrap()));
    let mode = multi_token_kernels();
    // Two tokens: the pair kernels already unpack each block once, and keep both accumulators in
    // registers.
    let multi = !gemm && (3..=MULTI_MAX).contains(&t) && mode != MultiTokenKernels::Pairs;
    let pairs =
        (multi && mode == MultiTokenKernels::Auto && cpu_has_i8mm() && refs.iter().any(|j| !j.m.dense() && j.m.multi(t) == Some(true)))
            .then(|| PairAct::new(xa.as_ref().unwrap()));
    let run = |b: usize| {
        let (j, r0, r1) = blocks[b / tchunks];
        let job = refs[j as usize];
        let (r0, r1) = (r0 as usize, r1 as usize);
        let c0 = (b % tchunks) * tc;
        if c0 >= t {
            return;
        }
        let c1 = (c0 + tc).min(t);
        match path(b) {
            #[cfg(target_arch = "aarch64")]
            Path::Sme => unsafe { sme::gemm_rows(job.m, xs32.as_ref().unwrap(), c0, c1, r0, r1, job.y.get()) },
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            Path::I8 => unsafe { i8gemm::gemm_rows(job.m, xg.as_ref().unwrap(), c0, c1, r0, r1, job.y.get()) },
            Path::F32 => {
                let xs = &x[c0 * n_in..c1 * n_in];
                let xts = &xt.as_deref().unwrap()[c0 / TN * n_in * TN..];
                unsafe { matmul_quant_block_prefill(job.m, xs, xts, c1 - c0, r0, r1, job.y.get().add(c0 * job.m.n_out())) }
            }
            Path::Decode => unsafe { decode_rows(job.m, xa.as_ref().unwrap(), pairs.as_ref(), multi, job.y.get(), r0, r1) },
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    };
    if !hybrid {
        pool::run(n_jobs, &run);
        return;
    }
    let (sq, nq): (Vec<usize>, Vec<usize>) = (0..n_jobs).partition(|&b| path(b) == Path::Sme);
    two_queues(&sq, &nq, &run);
}

/// Runs the blocks of two queues (`sq` on the SME units, `nq` on the NEON units) at the same
/// time: the first `sme_threads` tasks start on `sq`, the others on `nq`; each moves to the
/// other queue when its own is empty. Which kernel computes a block depends on the block only,
/// so results do not depend on the timing.
fn two_queues(sq: &[usize], nq: &[usize], run: &(dyn Fn(usize) + Sync)) {
    let (si, ni) = (std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0));
    let (_, sme_threads) = hybrid_tuning(false);
    pool::run(pool::threads(), &|w| {
        let order = if w < sme_threads {
            [(sq, &si), (nq, &ni)]
        } else {
            [(nq, &ni), (sq, &si)]
        };
        for (q, i) in order {
            loop {
                let k = i.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if k >= q.len() {
                    break;
                }
                run(q[k]);
            }
        }
    });
}

/// How the driver computes one block of a quantized matmul.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Sme,
    I8,
    F32,
    Decode,
}

/// The hybrid prefill's SME share of the row blocks (in sixteenths; `fast`: against int8 rows)
/// and the number of tasks that start on the SME blocks (`OXINFER_SME_SHARE`,
/// `OXINFER_SME_THREADS`). M5 Max, Qwen3-8B Q8_0 pp512: int8x2 rows do best at 6-8 of 16 on
/// SME, int8 rows at 3-6.
fn hybrid_tuning(fast: bool) -> (usize, usize) {
    static T: std::sync::OnceLock<(Option<usize>, usize)> = std::sync::OnceLock::new();
    let (share, threads) = *T.get_or_init(|| {
        let e = |k: &str| std::env::var(k).ok().and_then(|v| v.parse().ok());
        (
            e("OXINFER_SME_SHARE").map(|s: usize| s.min(16)),
            e("OXINFER_SME_THREADS").unwrap_or(4),
        )
    });
    (share.unwrap_or(if fast { 5 } else { 7 }), threads)
}

/// The SME share (in sixteenths) of the grouped matmuls' blocks (mixture-of-experts prefill) in
/// `hybrid` / `fast` (`OXINFER_GROUPED_SME_SHARE`).
#[cfg(target_arch = "aarch64")]
fn grouped_sme_share(fast: bool) -> usize {
    static S: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let s = *S.get_or_init(|| {
        std::env::var("OXINFER_GROUPED_SME_SHARE")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(|s: usize| s.min(16))
    });
    s.unwrap_or(if fast { 0 } else { 7 })
}

/// Decode kernels: rows `r0..r1` of `m` against all `xa.t` activation rows into `y[c*n_out + r]`,
/// with the multi-token kernels when `multi` (and the matrix has them), else token pairs.
unsafe fn decode_rows<M: QMat>(m: &M, xa: &QAct, pairs: Option<&PairAct>, multi: bool, y: *mut f32, r0: usize, r1: usize) {
    let t = xa.t;
    if multi && unsafe { m.rows_multi(xa, pairs, t, y, r0, r1) } {
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
                m.rows_t1(xa, c, y.add(c * m.n_out()), r0, r1)
            } else {
                m.rows_tc(xa, c, tc, y, r0, r1)
            }
        }
        c += tc;
    }
}

/// Quantized matmuls with their own (already quantized) inputs, in one parallel region: job
/// `(m, i, y)` is `y [xs[i].t, n_out] = xs[i] m^T` on the decode kernels, so each row is
/// bit-identical to [`matmul_quant_multi`] of that job alone on the f32 rows `xs[i]` was
/// quantized from (activations are quantized per row). For many small matmuls that would each
/// pay a parallel region (a mixture-of-experts layer's experts). Every input must have at most
/// [`decode_max_tokens`] rows (beyond that `matmul_quant_multi` takes the f32 GEMM), and no
/// matrix may be float-stored.
pub fn matmul_quant_batch<M: QMat>(xs: &[QAct], jobs: &mut [(&M, usize, &mut [f32])]) {
    if jobs.is_empty() {
        return;
    }
    let decode_max = decode_max_tokens();
    let mut total_bytes = 0usize;
    let refs: Vec<(QJobRef<M>, usize)> = jobs
        .iter_mut()
        .map(|(m, i, y)| {
            let xa = &xs[*i];
            assert_eq!(m.n_in(), xa.n_in, "input shape mismatch");
            assert_eq!(y.len(), xa.t * m.n_out(), "output shape mismatch");
            assert!(!m.dense() && xa.t <= decode_max, "batched matmuls run on the decode kernels only");
            total_bytes += m.bytes();
            (
                QJobRef {
                    m: *m,
                    y: SendPtr(y.as_mut_ptr()),
                },
                *i,
            )
        })
        .collect();
    let _timer = prof::start(Cat::Matmul);
    prof::add_bytes(total_bytes);
    let nthreads = pool::threads();
    let bytes_per_row = (total_bytes / refs.iter().map(|j| j.0.m.n_out()).sum::<usize>()).max(1);
    let target = (total_bytes / (nthreads * 8)).max(128 * 1024);
    let rows_per_block = (target / bytes_per_row).max(16) / 16 * 16;
    let mut blocks: Vec<(u32, u32, u32)> = Vec::new();
    for (j, (job, i)) in refs.iter().enumerate() {
        if xs[*i].t == 0 {
            continue;
        }
        let mut r = 0;
        while r < job.m.n_out() {
            let r1 = (r + rows_per_block).min(job.m.n_out());
            blocks.push((j as u32, r as u32, r1 as u32));
            r = r1;
        }
    }
    let mode = multi_token_kernels();
    let multi = |t: usize| (3..=MULTI_MAX).contains(&t) && mode != MultiTokenKernels::Pairs;
    let i8mm = mode == MultiTokenKernels::Auto && cpu_has_i8mm();
    let pairs: Vec<Option<PairAct>> = xs
        .iter()
        .enumerate()
        .map(|(i, xa)| {
            (i8mm && multi(xa.t) && refs.iter().any(|(j, ji)| *ji == i && j.m.multi(xa.t) == Some(true))).then(|| PairAct::new(xa))
        })
        .collect();
    pool::run(blocks.len(), &|b| {
        let (j, r0, r1) = blocks[b];
        let (job, i) = refs[j as usize];
        let xa = &xs[i];
        unsafe { decode_rows(job.m, xa, pairs[i].as_ref(), multi(xa.t), job.y.get(), r0 as usize, r1 as usize) }
    });
}

/// Tuning of the cache-blocked AVX-512 prefill GEMM ([`gemm_blocked`]): columns per block
/// (`OXINFER_GEMM_KB`, a multiple of 256), rows per pool job (`OXINFER_GEMM_ROWS`, a multiple of
/// 16) and tokens per pool job (`OXINFER_GEMM_TOKENS`, a multiple of 16). None of them changes a
/// result. Defaults for 1 MiB of L2 per core: a job's activations for one column block
/// (`tokens * kb` floats, 512 KiB) and its partial sums (`rows * tokens` floats) stay in L2
/// while the job's rows pass over them, and 16 dequantized rows of a column block (32 KiB) in L1.
#[cfg(target_arch = "x86_64")]
fn gemm_tuning() -> (usize, usize, usize) {
    static T: std::sync::OnceLock<(usize, usize, usize)> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let e = |k: &str, d: usize, q: usize| {
            let v: usize = std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
            (v / q * q).max(q)
        };
        (
            e("OXINFER_GEMM_KB", 512, 256),
            e("OXINFER_GEMM_ROWS", 64, 16),
            e("OXINFER_GEMM_TOKENS", 256, 16),
        )
    })
}

/// The prefill GEMM on AVX-512, cache-blocked: rows `r0..` of `m` (whole groups of 8) times the
/// `t` tokens of `xt` (8-token panels from the first token on), into `y[c * n_out + r]`. The
/// columns go in blocks of `kb`: each block of 16 rows is dequantized for those columns only and
/// multiplied with every token before the next one, the partial sums kept between blocks. Every
/// output is still one fused multiply-add chain over the columns in order from zero, so the
/// results are bit-identical to the 8x8 GEMM's; what changes is that a row block no longer
/// streams the whole of `xt` (megabytes) from L3 or memory once per 8 rows.
///
/// Returns the first row it did not compute: `r0` when the format cannot be dequantized by
/// column blocks ([`QMat::dequantize_block`]), else the end of the last whole group of 8 rows.
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_blocked<M: QMat>(m: &M, xt: &[f32], t: usize, kb: usize, r0: usize, r1: usize, y: *mut f32) -> usize {
    let (n_in, n_out) = (m.n_in(), m.n_out());
    let full = r0 + (r1 - r0) / 8 * 8;
    if full == r0 || t == 0 || n_in == 0 {
        return r0;
    }
    // Panels of 16 tokens (two of `xt`'s), groups of 16 rows (the last one may have 8).
    let (tiles, np, ng) = (t.div_ceil(TN), t.div_ceil(2 * TN), (full - r0).div_ceil(16));
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let need = 16 * kb + ng * np * 256;
        if s.len() < need {
            s.resize(need, 0.0);
        }
        let (w, acc) = s.split_at_mut(16 * kb);
        let acc = &mut acc[..ng * np * 256];
        acc.fill(0.0);
        let mut k0 = 0;
        while k0 < n_in {
            let kn = (n_in - k0).min(kb);
            for g in 0..ng {
                let rg = r0 + 16 * g;
                let rows = (full - rg).min(16);
                if !m.dequantize_block(rg, rg + rows, k0, k0 + kn, &mut w[..rows * kn]) {
                    // The columns are the same for every row: only the first call can decline.
                    assert!(k0 == 0 && g == 0, "column blocks declined after the first");
                    return r0;
                }
                for p in 0..np {
                    let xa = xt.as_ptr().add((2 * p * n_in + k0) * TN);
                    // An odd last panel is paired with itself; its second copy is not stored.
                    let xb = if 2 * p + 1 < tiles { xa.add(n_in * TN) } else { xa };
                    simd::gemm_acc_avx512(w.as_ptr(), rows, kn, xa, xb, acc.as_mut_ptr().add((g * np + p) * 256));
                }
            }
            k0 += kn;
        }
        for p in 0..np {
            for c in 0..(t - 16 * p).min(16) {
                let yc = y.add((16 * p + c) * n_out + r0);
                for (i, a) in acc.as_chunks::<256>().0.iter().skip(p).step_by(np).enumerate() {
                    for r in 0..(full - r0 - 16 * i).min(16) {
                        *yc.add(16 * i + r) = a[16 * r + c];
                    }
                }
            }
        }
        full
    })
}

/// Prefill with quantized weights: dequantize 8 rows at a time into a thread-local f32 scratch
/// and run the f32 GEMM tiles over all tokens (x86-64 with AVX-512: [`gemm_blocked`]).
unsafe fn matmul_quant_block_prefill<M: QMat>(m: &M, x: &[f32], xt: &[f32], t: usize, r0: usize, r1: usize, y: *mut f32) {
    let n_in = m.n_in();
    let n_out = m.n_out();
    #[cfg(target_arch = "x86_64")]
    let (wide, r0) = if simd::avx512() {
        (true, gemm_blocked(m, xt, t, gemm_tuning().0, r0, r1, y))
    } else {
        (false, r0)
    };
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
                #[cfg(target_arch = "x86_64")]
                if wide && rows == 8 {
                    // A format without column blocks: whole rows, two panels per step.
                    simd::gemm_tiles_avx512::<f32>(sp, rows, n_in, xt.as_ptr(), 0, t.div_ceil(TN), t, y, n_out, r);
                    r += rows;
                    continue;
                }
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
    // The maximum over 8 lanes (any order gives the same maximum).
    let (c8, tail) = x.as_chunks::<8>();
    let mut m8 = [f32::NEG_INFINITY; 8];
    for c in c8 {
        for (m, &v) in m8.iter_mut().zip(c) {
            *m = m.max(v);
        }
    }
    let max = m8.iter().chain(tail).copied().fold(f32::NEG_INFINITY, f32::max);
    // The exponentials in their own loop (which vectorizes), then the sum in order: the same bits
    // as one loop doing both, several times faster.
    for v in x.iter_mut() {
        *v = fast_exp(*v - max);
    }
    let mut sum = 0f32;
    for v in x.iter() {
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

    /// The SME GEMM gives the NEON f32 GEMM's bits on whole 8-row groups (each output is the
    /// same fused-multiply-add chain over the inputs in order).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sme_gemm_matches_neon_f32_bitwise() {
        if !sme::available() {
            eprintln!("no SME: skipped");
            return;
        }
        use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
        for (kind, n_out, n_in, t) in [
            (GgmlType::Q8_0, 64, 256, 40),
            (GgmlType::Q4_K, 40, 512, 9),
            (GgmlType::BF16, 16, 48, 70),
        ] {
            let mut seed = 9u64;
            let bytes = random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
            let m = GgufMatrix::from_bytes(kind, &bytes, n_out, n_in).unwrap();
            let mut s = 3u64;
            let x: Vec<f32> = (0..t * n_in).map(|_| lcg(&mut s)).collect();
            let xt = pack_xt(&x, t, n_in);
            let mut neon = vec![0f32; t * n_out];
            unsafe { matmul_quant_block_prefill(&m, &x, &xt, t, 0, n_out, neon.as_mut_ptr()) };
            let xs = sme::pack_x(&x, t, n_in);
            let mut smey = vec![0f32; t * n_out];
            unsafe { sme::gemm_rows(&m, &xs, 0, t, 0, n_out, smey.as_mut_ptr()) };
            for (i, (a, b)) in neon.iter().zip(&smey).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{kind} [{}, {}]: {a} vs {b}", i / n_out, i % n_out);
            }
        }
    }

    /// The cache-blocked AVX-512 GEMM gives the bits of the 8x8 AVX2 GEMM over whole rows:
    /// several column blocks with a short last one, 16-row groups with an 8-row group and
    /// leftover rows, token counts around the 8- and 16-token panels, and the column slices of
    /// every kind of dequantizer. Rows it does not compute stay untouched.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn blocked_gemm_matches_8x8_bitwise() {
        if !simd::cpu_has_avx512() || !simd::avx2() {
            eprintln!("skipped: no AVX-512");
            return;
        }
        use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
        fn go<M: QMat>(name: &str, m: &M) {
            let (n_out, n_in) = (m.n_out(), m.n_in());
            let mut s = 3u64;
            for (t, kb, r0) in [
                (1usize, 512usize, 0usize),
                (9, 256, 0),
                (16, 512, 8),
                (23, 256, 16),
                (40, 512, 0),
                (40, 4096, 3),
            ] {
                let x: Vec<f32> = (0..t * n_in).map(|_| lcg(&mut s)).collect();
                let xt = pack_xt(&x, t, n_in);
                let full = r0 + (n_out - r0) / 8 * 8;
                let mut want = vec![f32::NAN; t * n_out];
                let mut w = vec![0f32; 8 * n_in];
                for r in (r0..full).step_by(8) {
                    m.dequantize_rows(r, r + 8, &mut w);
                    for tile in 0..t.div_ceil(TN) {
                        let c0 = tile * TN;
                        unsafe {
                            simd::gemm_8x8::<f32>(
                                w.as_ptr(),
                                n_in,
                                xt.as_ptr().add(tile * n_in * TN),
                                want.as_mut_ptr(),
                                n_out,
                                r,
                                c0,
                                (t - c0).min(TN),
                            )
                        };
                    }
                }
                let mut got = vec![f32::NAN; t * n_out];
                let done = unsafe { gemm_blocked(m, &xt, t, kb, r0, n_out, got.as_mut_ptr()) };
                assert_eq!(done, full, "{name} t={t} kb={kb} r0={r0}");
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{name} t={t} kb={kb} r0={r0} [{}, {}]: {a} vs {b}",
                        i / n_out,
                        i % n_out
                    );
                }
                assert!(want[(t - 1) * n_out + full - 1].is_finite() && (full == n_out || want[full].is_nan()));
            }
        }
        let (n_out, n_in) = (16 + 16 + 8 + 3, 1280 + 256);
        let mut seed = 21u64;
        let wf: Vec<f32> = (0..n_out * n_in).map(|_| lcg(&mut seed)).collect();
        go("Q8", &crate::quant::quantize_f32(&wf, n_out, n_in));
        go("Q4", &crate::quant::quantize_q4_f32(&wf, n_out, n_in));
        for kind in [
            GgmlType::Q8_0,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
            GgmlType::IQ3_S,
            GgmlType::IQ2_XS,
            GgmlType::BF16,
        ] {
            let mut bs = 9u64;
            let bytes = random_blocks(kind, n_out * n_in / kind.block_size(), &mut bs);
            go(kind.name(), &GgufMatrix::from_bytes(kind, &bytes, n_out, n_in).unwrap());
        }
    }

    /// `zeros` gives `n` zeros whatever path fills them (the pool's chunks cover every element).
    #[test]
    fn parallel_zeros_are_all_zero() {
        crate::pool::init(4);
        for n in [0usize, 1, (1 << 18) - 1, 1 << 18, (1 << 18) + 1, 3 * (1 << 18) + 12345, 1 << 22] {
            // Dirty the allocator's free blocks first, so that a missed element would show.
            drop(vec![1.5f32; n]);
            let v = zeros(n);
            assert_eq!(v.len(), n);
            assert!(v.iter().all(|x| x.to_bits() == 0), "n={n}");
        }
    }

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

    /// Grouped matmuls in a language model's prompt (`fast`: the int8 GEMM, beside SME where
    /// present) against the dequantized weights, with zero padding rows at segment ends (skipped
    /// by the int8 blocks, which must still store zeros there).
    #[test]
    fn grouped_fast_matches_reference() {
        use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
        crate::pool::init(4);
        let (n_out, n_in) = (72usize, 256usize);
        let mut seed = 9u64;
        let segs = [(0usize, 8usize), (8, 0), (8, 16), (24, 72), (96, 40)];
        let real = [5usize, 0, 16, 70, 33];
        let rows = 136;
        let mut x: Vec<f32> = (0..rows * n_in).map(|_| lcg(&mut seed)).collect();
        for (&(s, len), &r) in segs.iter().zip(&real) {
            x[(s + r) * n_in..(s + len) * n_in].fill(0.0);
        }
        for kind in [GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::Q4_K] {
            let ms: Vec<GgufMatrix> = (0..segs.len())
                .map(|_| {
                    GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed), n_out, n_in).unwrap()
                })
                .collect();
            let wf: Vec<Vec<f32>> = ms
                .iter()
                .map(|m| {
                    let mut f = vec![0f32; n_out * n_in];
                    m.dequantize_rows(0, n_out, &mut f);
                    f
                })
                .collect();
            let mut want = vec![0f32; rows * n_out];
            for (j, &(s, len)) in segs.iter().enumerate() {
                let r = reference(&wf[j], n_out, n_in, &x[s * n_in..(s + len) * n_in], len);
                want[s * n_out..(s + len) * n_out].copy_from_slice(&r);
            }
            let mut y = vec![f32::NAN; rows * n_out];
            lm_prefill(|| matmul_quant_grouped(&ms.iter().collect::<Vec<_>>(), &segs, &x, &mut y));
            let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
            for (i, (a, b)) in y.iter().zip(&want).enumerate() {
                assert!((a - b).abs() <= 2e-2 * scale, "{kind} grouped fast [{i}]: {a} vs {b}");
            }
            for (&(s, len), &r) in segs.iter().zip(&real) {
                assert!(
                    y[(s + r) * n_out..(s + len) * n_out].iter().all(|&v| v == 0.0),
                    "{kind}: padding rows"
                );
            }
        }
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

    /// Multi-token decode passes (2..=16 tokens: column-group kernels, token pairs) give every
    /// token bit-identical rows to a one-token pass, for the int8, Q4 and affine (MLX) Q4 formats.
    #[test]
    fn native_multi_token_passes_are_batch_invariant() {
        crate::pool::init(3);
        let (n_out, n_in, group) = (45usize, 4096 + 96, 32usize);
        let mut seed = 11u64;
        let w: Vec<u16> = (0..n_out * n_in).map(|_| (lcg(&mut seed).to_bits() >> 16) as u16).collect();
        let gpr = n_in / group;
        let words: Vec<u32> = (0..n_out * n_in / 8)
            .map(|_| lcg(&mut seed).to_bits() ^ (lcg(&mut seed).to_bits() << 7))
            .collect();
        let bf = |v: f32| half::bf16::from_f32(v).to_bits();
        let scales: Vec<u16> = (0..n_out * gpr).map(|_| bf(0.002 + 0.01 * lcg(&mut seed).abs())).collect();
        let biases: Vec<u16> = (0..n_out * gpr).map(|_| bf(0.05 * lcg(&mut seed))).collect();
        let q8 = quantize_bf16(&w, n_out, n_in);
        let q4 = quantize_q4(&w, n_out, n_in);
        let mlx = crate::quant::from_mlx_q4(&words, &scales, &biases, n_out, n_in, group);
        fn run<M: QMat>(m: &M, x: &[f32], t: usize) -> Vec<u32> {
            let mut y = vec![0f32; t * m.n_out()];
            matmul_quant_multi_at(&mut [(m, &mut y[..])], x, t, MULTI_MAX - 1);
            y.iter().map(|v| v.to_bits()).collect()
        }
        fn check<M: QMat>(name: &str, m: &M, x: &[f32], t: usize, n_in: usize) {
            let alone: Vec<u32> = (0..t).flat_map(|c| run(m, &x[c * n_in..(c + 1) * n_in], 1)).collect();
            assert_eq!(run(m, x, t), alone, "{name} t={t}");
        }
        for t in 2..MULTI_MAX {
            let x: Vec<f32> = (0..t * n_in).map(|_| lcg(&mut seed) * 3.0).collect();
            check("q8", &q8, &x, t, n_in);
            check("q4", &q4, &x, t, n_in);
            check("mlx q4", &mlx, &x, t, n_in);
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

    /// Float-stored matrices of any width through the prefill GEMM: the 8-row tiles of a width
    /// that is not a multiple of 4 (NEON takes 4 inputs per step), with the leftover rows and a
    /// last token tile that is not whole.
    #[test]
    fn float_prefill_gemm_takes_any_width() {
        use crate::ggml::{GgmlType, GgufMatrix};
        for (n_out, n_in) in [(24usize, 6usize), (19, 5), (16, 7), (8, 1), (27, 34), (16, 64)] {
            let mut seed = 11u64;
            let wf: Vec<f32> = (0..n_out * n_in).map(|_| lcg(&mut seed)).collect();
            let bytes: Vec<u8> = wf.iter().flat_map(|v| v.to_le_bytes()).collect();
            let m = GgufMatrix::from_bytes(GgmlType::F32, &bytes, n_out, n_in).unwrap();
            for t in [8usize, 17, 40] {
                let x: Vec<f32> = (0..t * n_in).map(|_| lcg(&mut seed)).collect();
                let yr = reference(&wf, n_out, n_in, &x, t);
                // The f32 tiles themselves (the driver takes the SME ones where there are any).
                let xt = pack_xt(&x, t, n_in);
                let mut y = vec![f32::NAN; t * n_out];
                unsafe { matmul_quant_block_prefill(&m, &x, &xt, t, 0, n_out, y.as_mut_ptr()) };
                check(&format!("f32 tiles {n_out}x{n_in} t={t}"), &y, &yr);
                let mut y = vec![f32::NAN; t * n_out];
                matmul_quant_multi(&mut [(&m, &mut y[..])], &x, t);
                check(&format!("f32 {n_out}x{n_in} t={t}"), &y, &yr);
            }
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
            // Whatever GEMM the driver took (the cache-blocked AVX-512 one through the quantized
            // driver, where there is one), the bits are those of the row-block kernel. Not the
            // SME GEMM's on a partial group of 8 rows (see `sme_gemm_matches_neon_f32_bitwise`).
            #[cfg(has_simd)]
            if !(t > decode_max_tokens() && sme_available()) {
                let xt = (t >= TN).then(|| pack_xt(&x, t, n_in));
                let mut yb = vec![0f32; t * n_out];
                unsafe { matmul_block(&w, n_out, n_in, &x, xt.as_deref(), t, 0, n_out, yb.as_mut_ptr()) };
                for (i, (a, b)) in y.iter().zip(&yb).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "bf16 t={t} [{}, {}]: {a} vs {b}", i / n_out, i % n_out);
                }
            }
        }
    }
}
