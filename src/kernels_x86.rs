//! x86-64 kernels: AVX2 + FMA (and F16C for the f16 scales), chosen at run time
//! (`is_x86_feature_detected!`), with scalar fallbacks in the same shapes so the call sites in
//! `kernels.rs` are shared with the NEON module. `OXINFER_SIMD=scalar` selects the fallbacks on
//! any CPU. bf16 weights widen with a 16-bit shift; Q8 uses the `maddubs` sign trick (|w| <= 127
//! so the int16 pair sums cannot saturate); Q4 nibbles go in unsigned with the +8 offset folded
//! through the activation block sums, like the NEON path.

use crate::quant::f16_to_f32;
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU8, Ordering};

static LEVEL: AtomicU8 = AtomicU8::new(0);

/// True when the CPU has AVX2, FMA and F16C (every AVX2 CPU has the last), whatever
/// `OXINFER_SIMD` says.
pub fn cpu_has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
}

/// An f16 as f32 with F16C: exact, as `f16_to_f32`, without its call (the `half` crate picks
/// its conversion at run time, out of line: per block of 32 weights that costs more than the
/// dot product).
///
/// # Safety
/// Inlined into `#[target_feature(enable = "f16c")]` functions only.
#[inline(always)]
pub unsafe fn f16c_to_f32(h: u16) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
}

/// True when the AVX2 + FMA kernels are in use (cached): the CPU has both and
/// `OXINFER_SIMD=scalar` does not ask for the scalar fallbacks (to compare the two paths).
#[inline]
pub fn avx2() -> bool {
    match LEVEL.load(Ordering::Relaxed) {
        0 => {
            let off = matches!(std::env::var("OXINFER_SIMD").as_deref(), Ok("scalar" | "off" | "0"));
            let ok = !off && cpu_has_avx2();
            LEVEL.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
            ok
        }
        l => l == 2,
    }
}

/// The kernel set in use, for reports: `avx2+fma` or `scalar`.
pub fn level_name() -> &'static str {
    if avx2() { "avx2+fma" } else { "scalar" }
}

pub trait Load8: Copy {
    /// 8 consecutive weights as f32 lanes.
    unsafe fn load8(p: *const Self) -> __m256;
    fn to_f32(self) -> f32;
}
impl Load8 for u16 {
    #[inline(always)]
    unsafe fn load8(p: *const u16) -> __m256 {
        let h = _mm_loadu_si128(p as *const __m128i);
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)))
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from_bits((self as u32) << 16)
    }
}
impl Load8 for f32 {
    #[inline(always)]
    unsafe fn load8(p: *const f32) -> __m256 {
        _mm256_loadu_ps(p)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

/// Loads one weight broadcast to all lanes (prefill GEMM).
pub trait LoadW4: Copy + Load8 {
    unsafe fn bcast(p: *const Self) -> __m256;
}
impl LoadW4 for u16 {
    #[inline(always)]
    unsafe fn bcast(p: *const u16) -> __m256 {
        _mm256_set1_ps(f32::from_bits((*p as u32) << 16))
    }
}
impl LoadW4 for f32 {
    #[inline(always)]
    unsafe fn bcast(p: *const f32) -> __m256 {
        _mm256_broadcast_ss(&*p)
    }
}

#[inline(always)]
unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
    _mm_cvtss_f32(s)
}

// ---- f32/bf16 dot-product tiles ------------------------------------------------------------------

/// Dot products of R weight rows against TC activation rows; each step consumes 8*G elements per
/// row with A independent accumulators per output. `n` must be a multiple of 8*G.
#[inline(always)]
pub unsafe fn kernel<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
    w: *const W,
    ws: usize,
    x: *const f32,
    xs: usize,
    n: usize,
) -> [[f32; TC]; R] {
    if avx2() {
        kernel_avx2::<W, R, TC, G, A>(w, ws, x, xs, n)
    } else {
        kernel_scalar::<W, R, TC>(w, ws, x, xs, n)
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
    w: *const W,
    ws: usize,
    x: *const f32,
    xs: usize,
    n: usize,
) -> [[f32; TC]; R] {
    let zero = _mm256_setzero_ps();
    let mut acc = [[[zero; A]; TC]; R];
    let mut i = 0;
    while i < n {
        let mut wv = [[zero; G]; R];
        for r in 0..R {
            for g in 0..G {
                wv[r][g] = W::load8(w.add(r * ws + i + 8 * g));
            }
        }
        for c in 0..TC {
            for g in 0..G {
                let xv = _mm256_loadu_ps(x.add(c * xs + i + 8 * g));
                for r in 0..R {
                    acc[r][c][g % A] = _mm256_fmadd_ps(wv[r][g], xv, acc[r][c][g % A]);
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
                s = _mm256_add_ps(s, acc[r][c][a]);
            }
            out[r][c] = hsum(s);
        }
    }
    out
}

/// Scalar fallback of [`kernel`]: 8 lanes per output, a multiply then an add.
unsafe fn kernel_scalar<W: Load8, const R: usize, const TC: usize>(
    w: *const W,
    ws: usize,
    x: *const f32,
    xs: usize,
    n: usize,
) -> [[f32; TC]; R] {
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            let mut acc = [0f32; 8];
            for i in 0..n {
                acc[i % 8] += (*w.add(r * ws + i)).to_f32() * *x.add(c * xs + i);
            }
            out[r][c] = acc.iter().sum();
        }
    }
    out
}

/// Runs the kernel over every R-row group in `r0..r1` for tokens `c0..c0+TC`, storing to
/// `y[(c0+c)*n_out + r]`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn run_tiles<W: Load8, const R: usize, const TC: usize, const G: usize, const A: usize>(
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

/// Rows `r0..r1` (any count) for `tc` tokens (1..=4) starting at `c0`. Requires `n_in % 32 == 0`.
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
    // 16 ymm registers: keep accumulators + weight vectors under that.
    macro_rules! go {
        ($TC:expr, $G:expr, $A:expr, $c:expr) => {{
            let m = r0 + (r1 - r0) / 2 * 2;
            run_tiles::<W, 2, $TC, $G, $A>(w, n_in, x, y, n_out, r0, m, $c);
            run_tiles::<W, 1, $TC, $G, $A>(w, n_in, x, y, n_out, m, r1, $c);
        }};
    }
    // One and two tokens share an accumulator layout (A = 2) and three or four tokens run as
    // those, so each output is bit-identical whatever the token count: speculative verification
    // over a few tokens then reproduces plain decoding exactly.
    match tc {
        1 => go!(1, 4, 2, c0),
        2 => go!(2, 2, 2, c0),
        3 => {
            go!(2, 2, 2, c0);
            go!(1, 4, 2, c0 + 2)
        }
        4 => {
            go!(2, 2, 2, c0);
            go!(2, 2, 2, c0 + 2)
        }
        _ => unreachable!("bad tc {tc}"),
    }
}

/// Prefill GEMM micro-kernel: 8 weight rows times 8 tokens from a packed panel `xt` ([n_in][8]);
/// weights are broadcast, accumulators are vectors over tokens.
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
    if avx2() {
        gemm_8x8_avx2(w, n_in, xt, y, n_out, rb, c0, c_valid)
    } else {
        gemm_8x8_scalar(w, n_in, xt, y, n_out, rb, c0, c_valid)
    }
}

/// Scalar fallback of [`gemm_8x8`].
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_8x8_scalar<W: LoadW4>(
    w: *const W,
    n_in: usize,
    xt: *const f32,
    y: *mut f32,
    n_out: usize,
    rb: usize,
    c0: usize,
    c_valid: usize,
) {
    for r in 0..8 {
        for c in 0..c_valid {
            let mut acc = 0f32;
            for k in 0..n_in {
                acc += (*w.add(r * n_in + k)).to_f32() * *xt.add(k * 8 + c);
            }
            *y.add((c0 + c) * n_out + rb + r) = acc;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_8x8_avx2<W: LoadW4>(
    w: *const W,
    n_in: usize,
    xt: *const f32,
    y: *mut f32,
    n_out: usize,
    rb: usize,
    c0: usize,
    c_valid: usize,
) {
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut k = 0;
    while k < n_in {
        let xv = _mm256_loadu_ps(xt.add(k * 8));
        for r in 0..8 {
            acc[r] = _mm256_fmadd_ps(W::bcast(w.add(r * n_in + k)), xv, acc[r]);
        }
        k += 1;
    }
    let mut tmp = [0f32; 8];
    for r in 0..8 {
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc[r]);
        for c in 0..c_valid {
            *y.add((c0 + c) * n_out + rb + r) = tmp[c];
        }
    }
}

// ---- Q8 -----------------------------------------------------------------------------------------

/// Q8: R weight rows against TC quantized activation rows (token stride `n_in`).
#[inline(never)]
pub unsafe fn q8_dot_rows_tc<const R: usize, const TC: usize>(
    q: *const i8,
    s: *const f32,
    n_in: usize,
    xq: *const i8,
    xs: *const f32,
) -> [[f32; TC]; R] {
    if avx2() {
        q8_dot_rows_tc_avx2::<R, TC>(q, s, n_in, xq, xs)
    } else {
        q8_dot_rows_tc_scalar::<R, TC>(q, s, n_in, xq, xs)
    }
}

/// Scalar fallback of [`q8_dot_rows_tc`]: one integer sum per block of 32, scaled in f32.
unsafe fn q8_dot_rows_tc_scalar<const R: usize, const TC: usize>(
    q: *const i8,
    s: *const f32,
    n_in: usize,
    xq: *const i8,
    xs: *const f32,
) -> [[f32; TC]; R] {
    let bpr = n_in / 32;
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            let mut acc = 0f32;
            for b in 0..bpr {
                let mut sum = 0i32;
                for j in 0..32 {
                    sum += *q.add(r * n_in + b * 32 + j) as i32 * *xq.add(c * n_in + b * 32 + j) as i32;
                }
                acc += sum as f32 * *s.add(r * bpr + b) * *xs.add(c * bpr + b);
            }
            out[r][c] = acc;
        }
    }
    out
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q8_dot_rows_tc_avx2<const R: usize, const TC: usize>(
    q: *const i8,
    s: *const f32,
    n_in: usize,
    xq: *const i8,
    xs: *const f32,
) -> [[f32; TC]; R] {
    let bpr = n_in / 32;
    let ones = _mm256_set1_epi16(1);
    let mut accf = [[_mm256_setzero_ps(); TC]; R];
    for b in 0..bpr {
        let k = b * 32;
        let mut xv = [_mm256_setzero_si256(); TC];
        for c in 0..TC {
            xv[c] = _mm256_loadu_si256(xq.add(c * n_in + k) as *const __m256i);
        }
        for r in 0..R {
            let wv = _mm256_loadu_si256(q.add(r * n_in + k) as *const __m256i);
            let wabs = _mm256_sign_epi8(wv, wv);
            let d = *s.add(r * bpr + b);
            for c in 0..TC {
                let xsgn = _mm256_sign_epi8(xv[c], wv);
                let p16 = _mm256_maddubs_epi16(wabs, xsgn);
                let p32 = _mm256_madd_epi16(p16, ones);
                let scale = _mm256_set1_ps(d * *xs.add(c * bpr + b));
                accf[r][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scale, accf[r][c]);
            }
        }
    }
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            out[r][c] = hsum(accf[r][c]);
        }
    }
    out
}

/// Q8 decode for one token: rows `r0..r1` of `m` against activation row (`xq`, `xs`), into `y[r]`.
pub unsafe fn q8_rows_t1(m: &crate::quant::QMatrix, xq: *const i8, xs: *const f32, y: *mut f32, r0: usize, r1: usize) {
    let n_in = m.n_in;
    let bpr = n_in / 32;
    let mut r = r0;
    while r + 2 <= r1 {
        let o = q8_dot_rows_tc::<2, 1>(m.q.as_ptr().add(r * n_in), m.scales.as_ptr().add(r * bpr), n_in, xq, xs);
        *y.add(r) = o[0][0];
        *y.add(r + 1) = o[1][0];
        r += 2;
    }
    if r < r1 {
        let o = q8_dot_rows_tc::<1, 1>(m.q.as_ptr().add(r * n_in), m.scales.as_ptr().add(r * bpr), n_in, xq, xs);
        *y.add(r) = o[0][0];
    }
}

// ---- Q4_0 ---------------------------------------------------------------------------------------

/// Q4_0: R weight rows against TC quantized activation rows. Nibbles are used unsigned (0..15) and
/// the +8 offset is folded in via the block sums of x.
#[inline(never)]
pub unsafe fn q4_dot_rows_tc<const R: usize, const TC: usize>(
    q: *const u8,
    s: *const u16,
    n_in: usize,
    xq: *const i8,
    xs: *const f32,
    xsums: *const i32,
) -> [[f32; TC]; R] {
    if avx2() {
        q4_dot_rows_tc_avx2::<R, TC>(q, s, n_in, xq, xs, xsums)
    } else {
        q4_dot_rows_tc_scalar::<R, TC>(q, s, n_in, xq, xs)
    }
}

/// Scalar fallback of [`q4_dot_rows_tc`]: signed nibbles (`q - 8`), one integer sum per block.
unsafe fn q4_dot_rows_tc_scalar<const R: usize, const TC: usize>(
    q: *const u8,
    s: *const u16,
    n_in: usize,
    xq: *const i8,
    xs: *const f32,
) -> [[f32; TC]; R] {
    let bpr = n_in / 32;
    let half_row = n_in / 2;
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            let mut acc = 0f32;
            for b in 0..bpr {
                let blk = q.add(r * half_row + b * 16);
                let mut sum = 0i32;
                for j in 0..16 {
                    let v = *blk.add(j);
                    sum += ((v & 15) as i32 - 8) * *xq.add(c * n_in + b * 32 + j) as i32;
                    sum += ((v >> 4) as i32 - 8) * *xq.add(c * n_in + b * 32 + j + 16) as i32;
                }
                acc += sum as f32 * f16_to_f32(*s.add(r * bpr + b)) * *xs.add(c * bpr + b);
            }
            out[r][c] = acc;
        }
    }
    out
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4_dot_rows_tc_avx2<const R: usize, const TC: usize>(
    q: *const u8,
    s: *const u16,
    n_in: usize,
    xq: *const i8,
    xs: *const f32,
    xsums: *const i32,
) -> [[f32; TC]; R] {
    let bpr = n_in / 32;
    let half_row = n_in / 2;
    let ones = _mm256_set1_epi16(1);
    let m0f = _mm_set1_epi8(0x0F);
    let mut accf = [[_mm256_setzero_ps(); TC]; R];
    let mut corr = [[0f32; TC]; R];
    for b in 0..bpr {
        let k = b * 32;
        let mut xv = [_mm256_setzero_si256(); TC];
        for c in 0..TC {
            xv[c] = _mm256_loadu_si256(xq.add(c * n_in + k) as *const __m256i);
        }
        for r in 0..R {
            let bytes = _mm_loadu_si128(q.add(r * half_row + b * 16) as *const __m128i);
            let lo = _mm_and_si128(bytes, m0f);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(bytes), m0f);
            let wv = _mm256_set_m128i(hi, lo); // elements 0..15 = low nibbles, 16..31 = high nibbles
            let d = f16c_to_f32(*s.add(r * bpr + b));
            for c in 0..TC {
                let p16 = _mm256_maddubs_epi16(wv, xv[c]);
                let p32 = _mm256_madd_epi16(p16, ones);
                let sc = d * *xs.add(c * bpr + b);
                accf[r][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(sc), accf[r][c]);
                corr[r][c] += sc * *xsums.add(c * bpr + b) as f32;
            }
        }
    }
    let mut out = [[0f32; TC]; R];
    for r in 0..R {
        for c in 0..TC {
            out[r][c] = hsum(accf[r][c]) - 8.0 * corr[r][c];
        }
    }
    out
}

/// Q4 decode for one token: rows `r0..r1` of `m` against activation row, into `y[r]`.
pub unsafe fn q4_rows_t1(m: &crate::quant::Q4Matrix, xq: *const i8, xs: *const f32, xsums: *const i32, y: *mut f32, r0: usize, r1: usize) {
    let n_in = m.n_in;
    let bpr = n_in / 32;
    let mut r = r0;
    while r + 2 <= r1 {
        let o = q4_dot_rows_tc::<2, 1>(m.q.as_ptr().add(r * n_in / 2), m.scales.as_ptr().add(r * bpr), n_in, xq, xs, xsums);
        *y.add(r) = o[0][0];
        *y.add(r + 1) = o[1][0];
        r += 2;
    }
    if r < r1 {
        let o = q4_dot_rows_tc::<1, 1>(m.q.as_ptr().add(r * n_in / 2), m.scales.as_ptr().add(r * bpr), n_in, xq, xs, xsums);
        *y.add(r) = o[0][0];
    }
}

/// The AVX2 kernels against their scalar fallbacks, called directly (the dispatch picks one per
/// process). They skip on CPUs without AVX2 + FMA.
#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }
    fn unit(seed: &mut u64) -> f32 {
        lcg(seed) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    }
    fn skip() -> bool {
        if !cpu_has_avx2() {
            eprintln!("skipped: this CPU has no AVX2 + FMA");
        }
        !cpu_has_avx2()
    }
    fn bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }
    /// |a - b| <= tol * mag, with `mag` the sum of the absolute products behind the value.
    fn close(name: &str, a: f32, b: f32, mag: f32, tol: f32) {
        assert!((a - b).abs() <= tol * mag.max(1e-30), "{name}: {a} vs {b} (magnitude {mag})");
    }

    /// Weights of each float type with the same values.
    trait Wt: Load8 + LoadW4 {
        const NAME: &'static str;
        fn of(v: f32) -> Self;
    }
    impl Wt for u16 {
        const NAME: &'static str = "bf16";
        fn of(v: f32) -> u16 {
            bf16(v)
        }
    }
    impl Wt for f32 {
        const NAME: &'static str = "f32";
        fn of(v: f32) -> f32 {
            v
        }
    }

    #[test]
    fn avx2_is_detected_and_selected() {
        // The dispatch follows the CPU unless OXINFER_SIMD=scalar turns the AVX2 kernels off.
        let off = matches!(std::env::var("OXINFER_SIMD").as_deref(), Ok("scalar" | "off" | "0"));
        assert_eq!(avx2(), cpu_has_avx2() && !off);
        assert_eq!(level_name(), if avx2() { "avx2+fma" } else { "scalar" });
        eprintln!("x86-64 kernels in use: {}", level_name());
    }

    /// F16C conversion of the f16 scales equals the software one for all 65536 values.
    #[test]
    fn f16c_matches_software_conversion() {
        if skip() {
            return;
        }
        #[target_feature(enable = "f16c")]
        unsafe fn conv(h: u16) -> f32 {
            f16c_to_f32(h)
        }
        for h in 0..=u16::MAX {
            let (a, b) = (unsafe { conv(h) }, half::f16::from_bits(h).to_f32_const());
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{h:#06x}: {a} vs {b}");
        }
    }

    fn dot_tiles<W: Wt, const R: usize, const TC: usize, const G: usize, const A: usize>(seed: &mut u64) {
        for n in [8 * G, 16 * G, 96, 4128].into_iter().filter(|n| n % (8 * G) == 0) {
            // Strides larger than the row, so a kernel reading past `n` picks up other values.
            let (ws, xs) = (n + 8, n + 16);
            let w: Vec<W> = (0..R * ws).map(|_| W::of(unit(seed))).collect();
            let x: Vec<f32> = (0..TC * xs).map(|_| unit(seed) * 3.0).collect();
            let (a, s) = unsafe {
                (
                    kernel_avx2::<W, R, TC, G, A>(w.as_ptr(), ws, x.as_ptr(), xs, n),
                    kernel_scalar::<W, R, TC>(w.as_ptr(), ws, x.as_ptr(), xs, n),
                )
            };
            for r in 0..R {
                for c in 0..TC {
                    let (mut exact, mut mag) = (0f64, 0f32);
                    for i in 0..n {
                        let p = w[r * ws + i].to_f32() as f64 * x[c * xs + i] as f64;
                        exact += p;
                        mag += p.abs() as f32;
                    }
                    let name = format!("{} R={R} TC={TC} G={G} A={A} n={n} [{r}][{c}]", W::NAME);
                    close(&name, a[r][c], s[r][c], mag, 1e-6);
                    close(&format!("{name} vs f64"), a[r][c], exact as f32, mag, 1e-6);
                }
            }
        }
    }

    /// Every tile shape the drivers instantiate (`block_tc`, `dot_bf16`), bf16 and f32 weights.
    #[test]
    fn avx2_dot_tiles_match_scalar() {
        if skip() {
            return;
        }
        let mut seed = 1u64;
        macro_rules! go {
            ($W:ty) => {
                dot_tiles::<$W, 2, 1, 4, 2>(&mut seed);
                dot_tiles::<$W, 1, 1, 4, 2>(&mut seed);
                dot_tiles::<$W, 2, 2, 2, 2>(&mut seed);
                dot_tiles::<$W, 1, 2, 2, 2>(&mut seed);
                dot_tiles::<$W, 1, 1, 4, 8>(&mut seed);
            };
        }
        go!(u16);
        go!(f32);
    }

    /// `block_tc` over 1..=4 tokens and odd row ranges: every output equals the scalar kernel's
    /// within rounding, and is bit-identical to the same token run alone (the property that makes
    /// speculative verification reproduce plain decoding).
    #[test]
    fn avx2_block_tc_is_batch_invariant() {
        if skip() {
            return;
        }
        let mut seed = 2u64;
        let (n_out, n_in) = (13usize, 4096 + 32);
        let w: Vec<u16> = (0..n_out * n_in).map(|_| bf16(unit(&mut seed))).collect();
        let wf: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
        let x: Vec<f32> = (0..4 * n_in).map(|_| unit(&mut seed)).collect();
        #[allow(clippy::too_many_arguments)]
        fn run<W: Load8>(w: &[W], x: &[f32], n_out: usize, n_in: usize, r0: usize, r1: usize, c0: usize, tc: usize) -> Vec<u32> {
            let mut y = vec![0f32; 4 * n_out];
            unsafe { block_tc::<W>(w.as_ptr(), n_in, x.as_ptr(), y.as_mut_ptr(), n_out, r0, r1, c0, tc) };
            y.iter().map(|v| v.to_bits()).collect()
        }
        if !avx2() {
            eprintln!("skipped: OXINFER_SIMD=scalar");
            return;
        }
        for (r0, r1) in [(0usize, n_out), (3, 8), (4, 5)] {
            let mut alone = vec![0u32; 4 * n_out];
            for c in 0..4 {
                let one = run(&w, &x, n_out, n_in, r0, r1, c, 1);
                alone[c * n_out..(c + 1) * n_out].copy_from_slice(&one[c * n_out..(c + 1) * n_out]);
            }
            for tc in 2..=4usize {
                for c0 in 0..=4 - tc {
                    let got = run(&w, &x, n_out, n_in, r0, r1, c0, tc);
                    for c in c0..c0 + tc {
                        assert_eq!(
                            got[c * n_out..(c + 1) * n_out],
                            alone[c * n_out..(c + 1) * n_out],
                            "rows {r0}..{r1} tc={tc} c0={c0} token {c}"
                        );
                    }
                }
            }
            // f32 weights (dequantized rows in prefill tails) give the same bits as bf16 ones.
            assert_eq!(run(&wf, &x, n_out, n_in, r0, r1, 0, 4), run(&w, &x, n_out, n_in, r0, r1, 0, 4));
            for c in 0..4 {
                for r in r0..r1 {
                    let s =
                        unsafe { kernel_scalar::<u16, 1, 1>(w.as_ptr().add(r * n_in), n_in, x.as_ptr().add(c * n_in), n_in, n_in) }[0][0];
                    let mag: f32 = (0..n_in).map(|i| (wf[r * n_in + i] * x[c * n_in + i]).abs()).sum();
                    close(&format!("block_tc [{r}][{c}]"), f32::from_bits(alone[c * n_out + r]), s, mag, 1e-6);
                }
            }
        }
    }

    fn gemm<W: Wt>(seed: &mut u64) {
        // Any n_in (the GEMM has no multiple-of-32 requirement), 1..=8 valid tokens of the tile.
        for n_in in [1usize, 7, 37, 96, 4128] {
            let (n_out, rb, c0) = (11usize, 2usize, 3usize);
            let w: Vec<W> = (0..8 * n_in).map(|_| W::of(unit(seed))).collect();
            let xt: Vec<f32> = (0..n_in * 8).map(|_| unit(seed) * 2.0).collect();
            for c_valid in 1..=8usize {
                let mut ya = vec![f32::NAN; (c0 + 8) * n_out];
                let mut ys = ya.clone();
                unsafe {
                    gemm_8x8_avx2::<W>(w.as_ptr(), n_in, xt.as_ptr(), ya.as_mut_ptr(), n_out, rb, c0, c_valid);
                    gemm_8x8_scalar::<W>(w.as_ptr(), n_in, xt.as_ptr(), ys.as_mut_ptr(), n_out, rb, c0, c_valid);
                }
                for (i, (a, s)) in ya.iter().zip(&ys).enumerate() {
                    let (c, r) = (i / n_out, i % n_out);
                    let written = (c0..c0 + c_valid).contains(&c) && (rb..rb + 8).contains(&r);
                    // Both write exactly the valid tokens of the 8 rows and nothing else.
                    assert_eq!(a.is_nan(), !written, "{} n_in={n_in} c_valid={c_valid}: y[{c}][{r}]", W::NAME);
                    assert_eq!(s.is_nan(), !written);
                    if written {
                        let mag: f32 = (0..n_in)
                            .map(|k| (w[(r - rb) * n_in + k].to_f32() * xt[k * 8 + c - c0]).abs())
                            .sum();
                        close(
                            &format!("{} gemm n_in={n_in} c_valid={c_valid} [{r}][{c}]", W::NAME),
                            *a,
                            *s,
                            mag,
                            1e-6,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn avx2_gemm_matches_scalar() {
        if skip() {
            return;
        }
        let mut seed = 3u64;
        gemm::<u16>(&mut seed);
        gemm::<f32>(&mut seed);
    }

    /// Q8 rows and activations for the tests: random, or every value at +-127 (the largest
    /// `maddubs` pair sums), with float or power-of-two scales.
    struct QFix {
        q: Vec<i8>,
        s: Vec<f32>,
        xq: Vec<i8>,
        xs: Vec<f32>,
        xsums: Vec<i32>,
    }
    fn qfix(rows: usize, tokens: usize, n_in: usize, extreme: Option<(i8, i8)>, pow2: bool, seed: &mut u64) -> QFix {
        let bpr = n_in / 32;
        let val = |seed: &mut u64, e: Option<i8>| e.unwrap_or_else(|| ((lcg(seed) % 255) as i32 - 127) as i8);
        let scale = |seed: &mut u64| {
            if pow2 {
                [0.5f32, 1.0, 2.0][(lcg(seed) % 3) as usize]
            } else {
                0.001 + unit(seed).abs() * 0.01
            }
        };
        let q: Vec<i8> = (0..rows * n_in).map(|_| val(seed, extreme.map(|e| e.0))).collect();
        let xq: Vec<i8> = (0..tokens * n_in).map(|_| val(seed, extreme.map(|e| e.1))).collect();
        let xsums = xq.chunks(32).map(|b| b.iter().map(|&v| v as i32).sum()).collect();
        QFix {
            q,
            s: (0..rows * bpr).map(|_| scale(seed)).collect(),
            xq,
            xs: (0..tokens * bpr).map(|_| scale(seed)).collect(),
            xsums,
        }
    }

    fn q8_case<const R: usize, const TC: usize>(n_in: usize, extreme: Option<(i8, i8)>, pow2: bool, seed: &mut u64) {
        let f = qfix(R, TC, n_in, extreme, pow2, seed);
        let (a, s) = unsafe {
            (
                q8_dot_rows_tc_avx2::<R, TC>(f.q.as_ptr(), f.s.as_ptr(), n_in, f.xq.as_ptr(), f.xs.as_ptr()),
                q8_dot_rows_tc_scalar::<R, TC>(f.q.as_ptr(), f.s.as_ptr(), n_in, f.xq.as_ptr(), f.xs.as_ptr()),
            )
        };
        let bpr = n_in / 32;
        for r in 0..R {
            for c in 0..TC {
                let name = format!("q8 R={R} TC={TC} n_in={n_in} extreme={extreme:?} pow2={pow2} [{r}][{c}]");
                if pow2 {
                    // Integer sums times powers of two below 2^24: every f32 operation is exact,
                    // so the integer arithmetic (signs, saturation) is checked bit for bit.
                    assert_eq!(a[r][c].to_bits(), s[r][c].to_bits(), "{name}: {} vs {}", a[r][c], s[r][c]);
                } else {
                    let mag: f32 = (0..n_in)
                        .map(|i| {
                            (f.q[r * n_in + i] as f32 * f.xq[c * n_in + i] as f32 * f.s[r * bpr + i / 32] * f.xs[c * bpr + i / 32]).abs()
                        })
                        .sum();
                    close(&name, a[r][c], s[r][c], mag, 1e-6);
                }
            }
        }
    }

    /// Q8 (`maddubs` sign trick) for every tile shape of the drivers: 1 or 2 rows, 1..=4 tokens.
    #[test]
    fn avx2_q8_matches_scalar() {
        if skip() {
            return;
        }
        let mut seed = 4u64;
        macro_rules! go {
            ($R:literal, $TC:literal) => {
                for n_in in [32usize, 96, 4128] {
                    q8_case::<$R, $TC>(n_in, None, false, &mut seed);
                }
                // 96 inputs: sums stay below 2^24 even with every product at 127 * 127.
                q8_case::<$R, $TC>(96, None, true, &mut seed);
                for e in [(127i8, 127i8), (-127, 127), (127, -127), (-127, -127), (0, -127)] {
                    q8_case::<$R, $TC>(96, Some(e), true, &mut seed);
                }
            };
        }
        go!(1, 1);
        go!(2, 1);
        go!(1, 2);
        go!(2, 2);
        go!(1, 3);
        go!(2, 3);
        go!(1, 4);
        go!(2, 4);
    }

    fn q4_case<const R: usize, const TC: usize>(n_in: usize, nibble: Option<u8>, xval: Option<i8>, pow2: bool, seed: &mut u64) {
        let f = qfix(R, TC, n_in, xval.map(|v| (0, v)), pow2, seed);
        let bpr = n_in / 32;
        let q: Vec<u8> = (0..R * n_in / 2)
            .map(|_| nibble.map_or(lcg(seed) as u8, |v| v | (v << 4)))
            .collect();
        let s: Vec<u16> =
            f.s.iter()
                .map(|&v| half::f16::from_f32(if lcg(seed) & 1 == 1 { -v } else { v }).to_bits())
                .collect();
        let (a, sc) = unsafe {
            (
                q4_dot_rows_tc_avx2::<R, TC>(q.as_ptr(), s.as_ptr(), n_in, f.xq.as_ptr(), f.xs.as_ptr(), f.xsums.as_ptr()),
                q4_dot_rows_tc_scalar::<R, TC>(q.as_ptr(), s.as_ptr(), n_in, f.xq.as_ptr(), f.xs.as_ptr()),
            )
        };
        for r in 0..R {
            for c in 0..TC {
                let name = format!("q4 R={R} TC={TC} n_in={n_in} nibble={nibble:?} x={xval:?} pow2={pow2} [{r}][{c}]");
                if pow2 {
                    assert_eq!(a[r][c].to_bits(), sc[r][c].to_bits(), "{name}: {} vs {}", a[r][c], sc[r][c]);
                } else {
                    // The AVX2 kernel sums unsigned nibbles and subtracts 8 * sum(x) per block:
                    // its terms are up to 15 * |x| + 8 * |sum x|, not |q - 8| * |x|.
                    let mut mag = 0f32;
                    for b in 0..bpr {
                        let d = (f16_to_f32(s[r * bpr + b]) * f.xs[c * bpr + b]).abs();
                        let ax: f32 = f.xq[c * n_in + b * 32..][..32].iter().map(|&v| (v as f32).abs()).sum();
                        mag += d * (15.0 * ax + 8.0 * (f.xsums[c * bpr + b] as f32).abs());
                    }
                    close(&name, a[r][c], sc[r][c], mag, 1e-6);
                }
            }
        }
    }

    /// Q4 (unsigned nibbles, the +8 folded through the activation block sums) for every tile shape.
    #[test]
    fn avx2_q4_matches_scalar() {
        if skip() {
            return;
        }
        let mut seed = 5u64;
        macro_rules! go {
            ($R:literal, $TC:literal) => {
                for n_in in [32usize, 96, 4128] {
                    q4_case::<$R, $TC>(n_in, None, None, false, &mut seed);
                }
                q4_case::<$R, $TC>(96, None, None, true, &mut seed);
                for (nib, x) in [(15u8, 127i8), (15, -127), (0, 127), (0, -127), (8, 127)] {
                    q4_case::<$R, $TC>(96, Some(nib), Some(x), true, &mut seed);
                }
            };
        }
        go!(1, 1);
        go!(2, 1);
        go!(1, 2);
        go!(2, 2);
        go!(1, 3);
        go!(2, 3);
        go!(1, 4);
        go!(2, 4);
    }
}
