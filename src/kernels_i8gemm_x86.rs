//! Int8 prefill GEMM (x86-64, AVX-512 VNNI).
//!
//! The activations are rounded to int8 per block of 32 exactly as the decode kernels round them
//! (`quant::quantize_act`: `amax / 127` steps, halves away from zero), then stored as unsigned
//! bytes (`x + 128`) interleaved by groups of 16 tokens: `[group][block of 32][8][16 tokens][4]`,
//! so one 512-bit load holds four consecutive inputs of 16 tokens. Weight rows are unpacked, 8 at
//! a time and one block of columns at a time, to their int8 values in row order with an f32 scale
//! (and, for the formats that have them, an f32 offset) per 16 or 32 inputs. A `vpdpbusd` then
//! multiplies four weight bytes of one row, broadcast, with those of 16 tokens: 64 products per
//! instruction against 16 for an FMA. The `+ 128` is taken out again by starting each integer
//! sum at `-128 * sum(w)`.
//!
//! Columns go in blocks (`kb`), as in the f32 GEMM (`kernels::gemm_blocked`): a pool job's
//! activations for one column block stay in L2 while its rows pass over them, 8 unpacked rows of
//! a column block in L1, and the partial f32 sums are kept between blocks.
//!
//! Rounding the activations to int8 is the decode kernels' approximation (and llama.cpp's for
//! prefill); the f32 sums run in another order than the decode kernels', so rows are close to
//! but not bit-identical to them.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::kernels::QMat;
use crate::quant::QBLOCK;
use std::arch::x86_64::*;

/// Whether the CPU has what this GEMM needs (AVX-512 F / BW / VNNI next to AVX2, FMA and F16C),
/// whatever `OXINFER_SIMD` says.
pub fn cpu_has_vnni512() -> bool {
    crate::kernels::simd::cpu_has_avx512() && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")
}

/// Whether this GEMM is in use: the CPU has it and `OXINFER_SIMD` leaves the AVX-512 kernels on.
pub fn available() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    crate::kernels::simd::avx512() && *V.get_or_init(cpu_has_vnni512)
}

/// Rows per micro-kernel call (one int32 accumulator each: eight independent `vpdpbusd` chains).
const R: usize = 8;
/// Tokens per micro-kernel call (the lanes of a 512-bit register).
const T: usize = 16;

/// Activations `[t, n_in]` for the GEMM, by groups of 16 tokens (the last one padded with
/// zeros): per group and block of 32, `q` holds 8 vectors of `[16 tokens][4]` bytes `x + 128`,
/// `s` the tokens' scales (`amax / 127`) and `f` their exact f32 block sums.
pub(crate) struct I8Act {
    q: Vec<u8>,
    s: Vec<f32>,
    f: Vec<f32>,
    t: usize,
    n_in: usize,
}

struct P<X>(*mut X);
unsafe impl<X> Sync for P<X> {}

impl I8Act {
    /// `two` (a second, residual plane) is the aarch64 GEMM's; there is one plane here.
    pub(crate) fn new(x: &[f32], t: usize, n_in: usize, two: bool) -> Self {
        assert_eq!(x.len(), t * n_in);
        assert!(n_in.is_multiple_of(QBLOCK) && !two);
        let nb = n_in / QBLOCK;
        let groups = t.div_ceil(T);
        // Every byte and scale is written below (padding tokens as zeros), so the megabytes are
        // not zero-filled first (that fill, on the calling thread, cost as much as the rounding
        // on all of them).
        let mut a = Self {
            q: Vec::with_capacity(groups * n_in * T),
            s: Vec::with_capacity(groups * nb * T),
            f: Vec::with_capacity(groups * nb * T),
            t,
            n_in,
        };
        let p = (P(a.q.as_mut_ptr()), P(a.s.as_mut_ptr()), P(a.f.as_mut_ptr()));
        let run = |g: usize| {
            let p = &p;
            // SAFETY: the caller of the GEMM checked the CPU features; each group is written by
            // one task.
            unsafe {
                quantize_group(
                    x.as_ptr().add(T * g * n_in),
                    n_in,
                    (t - T * g).min(T),
                    p.0.0.add(g * n_in * T),
                    p.1.0.add(g * nb * T),
                    p.2.0.add(g * nb * T),
                )
            }
        };
        if groups == 1 {
            run(0);
        } else {
            crate::pool::run(groups, &run);
        }
        // SAFETY: `quantize_group` wrote all 16 lanes of every block of every group.
        unsafe {
            a.q.set_len(groups * n_in * T);
            a.s.set_len(groups * nb * T);
            a.f.set_len(groups * nb * T);
        }
        a
    }
}

/// One group of `tokens` (up to 16) tokens of `n_in` values each at `x` into its `q`, `s` and
/// `f` (the lanes past `tokens` as zeros). The int8 values are
/// `quant::quantize_act`'s: `(v * (127 / amax)).round()`.
#[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
unsafe fn quantize_group(x: *const f32, n_in: usize, tokens: usize, q: *mut u8, s: *mut f32, f: *mut f32) {
    let sign = _mm512_set1_ps(-0.0);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    // `[token][8]` quads of one block, then stored as `[8][token]`.
    let mut quads = [[0u32; 8]; T];
    for b in 0..n_in / QBLOCK {
        for c in 0..tokens {
            let xp = x.add(c * n_in + 32 * b);
            let v = [_mm512_loadu_ps(xp), _mm512_loadu_ps(xp.add(16))];
            let amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_andnot_ps(sign, v[0]), _mm512_andnot_ps(sign, v[1])));
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            for (h, vh) in v.iter().enumerate() {
                let y = _mm512_mul_ps(*vh, _mm512_set1_ps(inv));
                // Round half away from zero: truncate, then one more where the fraction reaches 0.5.
                let r = _mm512_roundscale_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(y);
                let d = _mm512_andnot_ps(sign, _mm512_sub_ps(y, r));
                let up = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(d, half);
                let step = _mm512_or_ps(one, _mm512_and_ps(y, sign));
                let r = _mm512_mask_add_ps(r, up, r, step);
                let i = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(r));
                _mm_storeu_si128(
                    (quads[c].as_mut_ptr() as *mut __m128i).add(h),
                    _mm_xor_si128(i, _mm_set1_epi8(-128)),
                );
            }
            *s.add(T * b + c) = amax / 127.0;
            *f.add(T * b + c) = _mm512_reduce_add_ps(_mm512_add_ps(v[0], v[1]));
        }
        for c in tokens..T {
            *s.add(T * b + c) = 0.0;
            *f.add(T * b + c) = 0.0;
        }
        let o = q.add(b * 512) as *mut u32;
        for j in 0..8 {
            for (c, qc) in quads.iter().enumerate() {
                *o.add(T * j + c) = qc[j];
            }
        }
    }
}

/// `R` rows of one column block, unpacked: `q[r * kn + i]` int8, `s` and `m` per 16 inputs
/// (`[r * kn / 16 + g]`), `n16` / `n32` the `-128 * sum(q)` of each 16 / 32 inputs.
struct Tile {
    q: Vec<i8>,
    s: Vec<f32>,
    m: Vec<f32>,
    n16: Vec<i32>,
    n32: Vec<i32>,
    /// Partial sums of a job: `[row group][token group][R][T]`.
    acc: Vec<f32>,
}

impl Tile {
    const fn new() -> Self {
        Self {
            q: Vec::new(),
            s: Vec::new(),
            m: Vec::new(),
            n16: Vec::new(),
            n32: Vec::new(),
            acc: Vec::new(),
        }
    }
    fn fit(&mut self, kb: usize) {
        if self.q.len() < R * kb {
            self.q.resize(R * kb, 0);
            self.s.resize(R * kb / 16, 0.0);
            self.m.resize(R * kb / 16, 0.0);
            self.n16.resize(R * kb / 16, 0);
            self.n32.resize(R * kb / 32, 0);
        }
    }

    /// Columns `k0..k0 + kn` of rows `r..r + R` of `m` (rows from `r1` on as zeros).
    #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
    unsafe fn fill<M: QMat + ?Sized>(&mut self, m: &M, r: usize, r1: usize, k0: usize, kn: usize) {
        let (g16, g32) = (kn / 16, kn / 32);
        for i in 0..R {
            let (q, s, mn) = (
                self.q.as_mut_ptr().add(i * kn),
                self.s.as_mut_ptr().add(i * g16),
                self.m.as_mut_ptr().add(i * g16),
            );
            let (n16, n32) = (self.n16.as_mut_ptr().add(i * g16), self.n32.as_mut_ptr().add(i * g32));
            if r + i >= r1 {
                std::ptr::write_bytes(q, 0, kn);
                std::ptr::write_bytes(s, 0, g16);
                std::ptr::write_bytes(mn, 0, g16);
                std::ptr::write_bytes(n16, 0, g16);
                std::ptr::write_bytes(n32, 0, g32);
                continue;
            }
            m.unpack_cols_i8(r + i, k0, k0 + kn, q, s, mn);
            // -128 * sum(q) per 16 and per 32: sums of `q + 128` over 8 bytes by `psadbw`.
            let bias = _mm256_set1_epi8(-128);
            for b in 0..g32 {
                let v = _mm256_xor_si256(_mm256_loadu_si256(q.add(32 * b) as *const __m256i), bias);
                let sad = _mm256_sad_epu8(v, _mm256_setzero_si256());
                let mut t = [0u64; 4];
                _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, sad);
                let lo = (t[0] + t[1]) as i32 - 128 * 16;
                let hi = (t[2] + t[3]) as i32 - 128 * 16;
                *n16.add(2 * b) = -128 * lo;
                *n16.add(2 * b + 1) = -128 * hi;
                *n32.add(b) = -128 * (lo + hi);
            }
        }
    }
}

#[inline(always)]
unsafe fn bcast4(p: *const i8) -> __m512i {
    _mm512_set1_epi32(std::ptr::read_unaligned(p as *const i32))
}

/// `f[r][c] +=` the products of `R` rows with one group of 16 tokens over `nb` blocks of 32
/// columns: the tile's rows from its start (`kn` columns each), the activations at `xq` / `xs` /
/// `xf` (the group's first block of these columns). `S16`: the rows' scales change per 16
/// inputs; `MIN`: the rows have offsets (per 32).
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
unsafe fn kern<const S16: bool, const MIN: bool>(
    tile: &Tile,
    kn: usize,
    nb: usize,
    xq: *const u8,
    xs: *const f32,
    xf: *const f32,
    f: *mut f32,
) {
    let (wq, ws, wm) = (tile.q.as_ptr(), tile.s.as_ptr(), tile.m.as_ptr());
    let (n16, n32) = (tile.n16.as_ptr(), tile.n32.as_ptr());
    let (g16, g32) = (kn / 16, kn / 32);
    let mut fa: [__m512; R] = std::array::from_fn(|r| _mm512_loadu_ps(f.add(T * r)));
    for b in 0..nb {
        let xv: [__m512i; 8] = std::array::from_fn(|j| _mm512_loadu_si512(xq.add((8 * b + j) * 64) as *const __m512i));
        let xsv = _mm512_loadu_ps(xs.add(T * b));
        if S16 {
            let mut fl = [_mm512_setzero_ps(); R];
            for h in 0..2 {
                let mut a: [__m512i; R] = std::array::from_fn(|r| _mm512_set1_epi32(*n16.add(r * g16 + 2 * b + h)));
                for j in 0..4 {
                    for r in 0..R {
                        a[r] = _mm512_dpbusd_epi32(a[r], xv[4 * h + j], bcast4(wq.add(r * kn + 32 * b + 16 * h + 4 * j)));
                    }
                }
                for r in 0..R {
                    fl[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[r]), _mm512_set1_ps(*ws.add(r * g16 + 2 * b + h)), fl[r]);
                }
            }
            for r in 0..R {
                fa[r] = _mm512_fmadd_ps(fl[r], xsv, fa[r]);
            }
        } else {
            let mut a: [__m512i; R] = std::array::from_fn(|r| _mm512_set1_epi32(*n32.add(r * g32 + b)));
            for (j, x) in xv.iter().enumerate() {
                for r in 0..R {
                    a[r] = _mm512_dpbusd_epi32(a[r], *x, bcast4(wq.add(r * kn + 32 * b + 4 * j)));
                }
            }
            for r in 0..R {
                let v = _mm512_mul_ps(_mm512_cvtepi32_ps(a[r]), _mm512_set1_ps(*ws.add(r * g16 + 2 * b)));
                fa[r] = _mm512_fmadd_ps(v, xsv, fa[r]);
            }
        }
        if MIN {
            let xfv = _mm512_loadu_ps(xf.add(T * b));
            for r in 0..R {
                fa[r] = _mm512_fmadd_ps(_mm512_set1_ps(*wm.add(r * g16 + 2 * b)), xfv, fa[r]);
            }
        }
    }
    for r in 0..R {
        _mm512_storeu_ps(f.add(T * r), fa[r]);
    }
}

// Already `const`; clippy misreads the emulated-TLS expansion on some targets.
#[allow(clippy::missing_const_for_thread_local)]
mod scratch {
    use std::cell::RefCell;
    thread_local! {
        pub(super) static TILE: RefCell<super::Tile> = const { RefCell::new(super::Tile::new()) };
    }
}

/// Tuning (none of it changes a result): rows per pool job (`OXINFER_I8_ROWS`, a multiple of 8),
/// tokens per pool job (`OXINFER_I8_TOKENS`, a multiple of 16) and the bytes of activations of
/// one column block of a job (`OXINFER_I8_L2`), from which the columns per block follow (a
/// multiple of 256, at most `OXINFER_I8_KB`). Defaults for 1 MiB of L2 and 48 KiB of L1 per core:
/// 8 rows (8 KiB) and 16 tokens (16 KiB) of a 1024-column block fit L1.
pub(crate) fn tuning() -> (usize, usize, usize, usize) {
    static V: std::sync::OnceLock<(usize, usize, usize, usize)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let e = |k: &str, d: usize, q: usize| {
            let v: usize = std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
            (v / q * q).max(q)
        };
        (
            e("OXINFER_I8_ROWS", 32, R),
            e("OXINFER_I8_TOKENS", 512, T),
            e("OXINFER_I8_L2", 512 * 1024, 4096),
            e("OXINFER_I8_KB", 1024, 256),
        )
    })
}

/// `y[c * n_out + r]` for tokens `c0..c1` (`c0` a multiple of 16) and rows `r0..r1` of `m`, which
/// must have an [`QMat::i8_layout`].
///
/// # Safety
/// [`cpu_has_vnni512`] is true; `y` covers the outputs named.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
pub(crate) unsafe fn gemm_rows<M: QMat + ?Sized>(m: &M, xa: &I8Act, c0: usize, c1: usize, r0: usize, r1: usize, y: *mut f32) {
    let lay = m.i8_layout().expect("no int8 GEMM layout");
    let (n_in, n_out) = (m.n_in(), m.n_out());
    debug_assert!(xa.n_in == n_in && c1 <= xa.t && c0.is_multiple_of(T));
    if c0 >= c1 || r0 >= r1 {
        return;
    }
    let nb = n_in / QBLOCK;
    let (g0, ng) = (c0 / T, (c1 - c0).div_ceil(T));
    let nrg = (r1 - r0).div_ceil(R);
    let (_, _, l2, kb_max) = tuning();
    let kb = (l2 / (ng * T) / 256 * 256).clamp(256, kb_max);
    scratch::TILE.with(|tl| {
        let mut tl = tl.borrow_mut();
        tl.fit(kb.min(n_in));
        let mut acc = std::mem::take(&mut tl.acc);
        acc.clear();
        acc.resize(nrg * ng * R * T, 0.0);
        let mut k0 = 0;
        while k0 < n_in {
            let kn = (n_in - k0).min(kb);
            for rg in 0..nrg {
                tl.fill(m, r0 + R * rg, r1, k0, kn);
                for g in 0..ng {
                    let xq = xa.q.as_ptr().add((g0 + g) * n_in * T + k0 * T);
                    let xs = xa.s.as_ptr().add(((g0 + g) * nb + k0 / QBLOCK) * T);
                    let xf = xa.f.as_ptr().add(((g0 + g) * nb + k0 / QBLOCK) * T);
                    let f = acc.as_mut_ptr().add((rg * ng + g) * R * T);
                    match (lay.sc16, lay.min32) {
                        (false, false) => kern::<false, false>(&tl, kn, kn / QBLOCK, xq, xs, xf, f),
                        (false, true) => kern::<false, true>(&tl, kn, kn / QBLOCK, xq, xs, xf, f),
                        (true, false) => kern::<true, false>(&tl, kn, kn / QBLOCK, xq, xs, xf, f),
                        (true, true) => kern::<true, true>(&tl, kn, kn / QBLOCK, xq, xs, xf, f),
                    }
                }
            }
            k0 += kn;
        }
        for g in 0..ng {
            for ci in 0..(c1 - c0 - T * g).min(T) {
                let yc = y.add((c0 + T * g + ci) * n_out + r0);
                for rg in 0..nrg {
                    let a = acc.as_ptr().add((rg * ng + g) * R * T + ci);
                    for i in 0..(r1 - r0 - R * rg).min(R) {
                        *yc.add(R * rg + i) = *a.add(T * i);
                    }
                }
            }
        }
        tl.acc = acc;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::{GgmlType, GgufMatrix, random_blocks};
    use crate::quant::quantize_act;

    fn input(t: usize, n_in: usize) -> Vec<f32> {
        (0..t * n_in)
            .map(|i| (((i * 7919) % 2001) as f32 / 1000.0 - 1.0) * if i % 61 == 0 { 9.0 } else { 1.0 })
            .collect()
    }

    /// The activations are the decode kernels' int8 values and scales.
    #[test]
    fn activations_match_quantize_act() {
        if !cpu_has_vnni512() {
            return;
        }
        let (t, n_in) = (19, 96);
        let mut x = input(t, n_in);
        // Exact halves and a zero block.
        x[..32].copy_from_slice(&std::array::from_fn::<f32, 32, _>(|i| (i as f32 - 16.0) / 2.0));
        x[0] = 127.0;
        x[32..64].fill(0.0);
        let want = quantize_act(&x, t, n_in);
        let got = I8Act::new(&x, t, n_in, false);
        for c in 0..t {
            for i in 0..n_in {
                let (b, j, e) = (i / 32, (i % 32) / 4, i % 4);
                let v = got.q[(c / T) * n_in * T + (8 * b + j) * 64 + 4 * (c % T) + e];
                assert_eq!((v ^ 0x80) as i8, want.q[c * n_in + i], "token {c} input {i}");
            }
            for b in 0..n_in / 32 {
                assert_eq!(got.s[((c / T) * (n_in / 32) + b) * T + c % T], want.scales[c * (n_in / 32) + b]);
            }
        }
    }

    /// The GEMM over row and token counts that leave partial tiles and several column blocks,
    /// against the one-token decode kernels (same int8 activations; only the f32 summation order
    /// differs).
    fn check<M: QMat>(m: &M, name: &str) {
        m.i8_layout().expect("layout");
        let (n_out, n_in, t) = (m.n_out(), m.n_in(), 37);
        let x = input(t, n_in);
        let xa = quantize_act(&x, t, n_in);
        let mut dec = vec![0f32; t * n_out];
        for c in 0..t {
            unsafe { m.rows_t1(&xa, c, dec.as_mut_ptr().add(c * n_out), 0, n_out) };
        }
        let ga = I8Act::new(&x, t, n_in, false);
        let mut got = vec![f32::NAN; t * n_out];
        // Rows 0..8 for tokens 0..16 then 16..37, the other rows in one call.
        unsafe {
            gemm_rows(m, &ga, 0, 16, 0, 8, got.as_mut_ptr());
            gemm_rows(m, &ga, 16, t, 0, 8, got.as_mut_ptr());
            gemm_rows(m, &ga, 0, t, 8, n_out, got.as_mut_ptr());
        }
        let scale = dec.iter().fold(0f32, |a, v| a.max(v.abs()));
        for (i, (a, b)) in got.iter().zip(dec.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 2e-5 * scale,
                "{name} [{}, {}]: {a} vs {b} (scale {scale})",
                i / n_out,
                i % n_out
            );
        }
    }

    #[test]
    fn gguf_matches_decode_kernels() {
        if !cpu_has_vnni512() {
            return;
        }
        let (n_out, n_in) = (21, 1536);
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
        assert_eq!(n, 25, "formats");
    }

    #[test]
    fn oxinfer_formats_match_decode_kernels() {
        if !cpu_has_vnni512() {
            return;
        }
        let (n_out, n_in) = (13, 1280);
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
