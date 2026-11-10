//! AVX2 decode kernels for the GGUF formats. Each block is unpacked to the int8 values and
//! per-16 scales / offsets of `decode_i8` (in registers for the block-of-32 formats, the K-quants
//! and IQ4_XS; through `decode_i8` itself for the grid-table i-quants), then multiplied with the
//! int8 activations by `maddubs` / `madd`, 32 elements per instruction. The integer sums per 16
//! elements are exact and the f32 operations are those of the portable kernel
//! (`dot_row_generic`) in the same order, so every result is bit-identical to it: the AVX2 path
//! changes speed only, and multi-token passes stay bit-identical to one-token ones.
//!
//! Dequantization (prefill) reuses the unpacking for the formats whose reference formula is
//! `scale * q` (and Q4_K / Q5_K with their mins), bit-identical to `dequantize_row`.
#![allow(unsafe_op_in_unsafe_fn)]

use super::{Act, GgmlType, GgufMatrix, QK_K, decode_i8, e8m0_half, iq4xs_scale, q3k_scales, scale_min_k4, u32at};
use crate::ggml::tables::{KVALUES_IQ4NL, KVALUES_MXFP4};
use crate::kernels::simd::f16c_to_f32;
use crate::quant::QBLOCK;
use std::arch::x86_64::*;

/// The f16 at byte `i` of a block (`super::f16at` with F16C: the same value, inlined).
#[inline(always)]
unsafe fn f16at(b: &[u8], i: usize) -> f32 {
    f16c_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// One block's int8 values, aligned for whole-register loads.
#[repr(align(32))]
struct Q([i8; QK_K]);

/// The 32 nibbles of 16 bytes: elements 0..16 are the low nibbles, 16..32 the high ones.
#[inline(always)]
unsafe fn nibbles(p: *const u8) -> __m256i {
    let bytes = _mm_loadu_si128(p as *const __m128i);
    let m = _mm_set1_epi8(0x0F);
    _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16::<4>(bytes), m), _mm_and_si128(bytes, m))
}

/// 16 where bit `i` of `bits` is set, for the 32 elements `i`.
#[inline(always)]
unsafe fn high_bits(bits: u32) -> __m256i {
    let v = _mm256_set1_epi32(bits as i32);
    // Byte i / 8 of `bits` in element i (the shuffle is per 128-bit lane: bytes 0, 1 then 2, 3).
    let sh = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    );
    let bit = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    let m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(v, sh), bit), bit);
    _mm256_and_si256(m, _mm256_set1_epi8(16))
}

/// 16 signed table entries looked up by 32 nibbles.
#[inline(always)]
unsafe fn lookup(table: &[i8; 16], idx: __m256i) -> __m256i {
    let t = _mm_loadu_si128(table.as_ptr() as *const __m128i);
    _mm256_shuffle_epi8(_mm256_set_m128i(t, t), idx)
}

#[inline(always)]
unsafe fn st(q: *mut i8, at: usize, v: __m256i) {
    _mm256_storeu_si256(q.add(at) as *mut __m256i, v)
}

/// `v >> n` per byte (logical), then `& mask`.
#[inline(always)]
unsafe fn shr_and(v: __m256i, n: i32, mask: i8) -> __m256i {
    _mm256_and_si256(_mm256_srl_epi16(v, _mm_cvtsi32_si128(n)), _mm256_set1_epi8(mask))
}

/// `decode_i8` with the int8 values unpacked in registers: the same `q`, `s`, `m` and return
/// value, bit for bit. Formats without a vector unpacker go through `decode_i8`.
#[inline(always)]
unsafe fn unpack(kind: GgmlType, b: &[u8], q: &mut [i8; QK_K], s: &mut [f32; 16], m: &mut [f32; 16]) -> bool {
    use GgmlType::*;
    let (p, q) = (b.as_ptr(), q.as_mut_ptr());
    match kind {
        Q8_0 => {
            debug_assert!(b.len() >= 34);
            st(q, 0, _mm256_loadu_si256(p.add(2) as *const __m256i));
            s[..2].fill(f16at(b, 0));
            false
        }
        Q4_0 => {
            debug_assert!(b.len() >= 18);
            st(q, 0, _mm256_sub_epi8(nibbles(p.add(2)), _mm256_set1_epi8(8)));
            s[..2].fill(f16at(b, 0));
            false
        }
        IQ4_NL => {
            debug_assert!(b.len() >= 18);
            st(q, 0, lookup(&KVALUES_IQ4NL, nibbles(p.add(2))));
            s[..2].fill(f16at(b, 0));
            false
        }
        MXFP4 => {
            debug_assert!(b.len() >= 17);
            st(q, 0, lookup(&KVALUES_MXFP4, nibbles(p.add(1))));
            s[..2].fill(e8m0_half(b[0]));
            false
        }
        Q4_1 => {
            debug_assert!(b.len() >= 20);
            let d = f16at(b, 0);
            st(q, 0, _mm256_sub_epi8(nibbles(p.add(4)), _mm256_set1_epi8(8)));
            s[..2].fill(d);
            m[..2].fill(f16at(b, 2) + d * 8.0);
            true
        }
        Q5_0 => {
            debug_assert!(b.len() >= 22);
            let v = _mm256_or_si256(nibbles(p.add(6)), high_bits(u32at(b, 2)));
            st(q, 0, _mm256_sub_epi8(v, _mm256_set1_epi8(16)));
            s[..2].fill(f16at(b, 0));
            false
        }
        Q5_1 => {
            debug_assert!(b.len() >= 24);
            let d = f16at(b, 0);
            let v = _mm256_or_si256(nibbles(p.add(8)), high_bits(u32at(b, 4)));
            st(q, 0, _mm256_sub_epi8(v, _mm256_set1_epi8(16)));
            s[..2].fill(d);
            m[..2].fill(f16at(b, 2) + d * 16.0);
            true
        }
        Q2_K => {
            debug_assert!(b.len() >= 84);
            let (d, min) = (f16at(b, 80), f16at(b, 82));
            for g in 0..16 {
                s[g] = d * (b[g] & 0xF) as f32;
                m[g] = 2.0 * s[g] - min * (b[g] >> 4) as f32;
            }
            for n in 0..2 {
                let v = _mm256_loadu_si256(p.add(16 + 32 * n) as *const __m256i);
                for j in 0..4 {
                    st(
                        q,
                        128 * n + 32 * j,
                        _mm256_sub_epi8(shr_and(v, 2 * j as i32, 3), _mm256_set1_epi8(2)),
                    );
                }
            }
            true
        }
        Q3_K => {
            debug_assert!(b.len() >= 110);
            let d = f16at(b, 108);
            let sc = q3k_scales(&b[96..108]);
            for g in 0..16 {
                s[g] = d * (sc[g] as i32 - 32) as f32;
            }
            let hm = _mm256_loadu_si256(p as *const __m256i);
            for n in 0..2 {
                let v = _mm256_loadu_si256(p.add(32 + 32 * n) as *const __m256i);
                for j in 0..4 {
                    let bit = _mm256_set1_epi8((1u8 << (4 * n + j)) as i8);
                    let set = _mm256_cmpeq_epi8(_mm256_and_si256(hm, bit), bit);
                    let sub = _mm256_andnot_si256(set, _mm256_set1_epi8(4));
                    st(q, 128 * n + 32 * j, _mm256_sub_epi8(shr_and(v, 2 * j as i32, 3), sub));
                }
            }
            false
        }
        Q4_K | Q5_K => {
            let five = kind == Q5_K;
            debug_assert!(b.len() >= if five { 176 } else { 144 });
            let (d, min) = (f16at(b, 0), f16at(b, 2));
            let off: i8 = if five { 16 } else { 8 };
            let (qh, qs) = if five {
                (_mm256_loadu_si256(p.add(16) as *const __m256i), p.add(48))
            } else {
                (_mm256_setzero_si256(), p.add(16))
            };
            for j in 0..4 {
                let v = _mm256_loadu_si256(qs.add(32 * j) as *const __m256i);
                for h in 0..2 {
                    let (sc, mn) = scale_min_k4(2 * j + h, &b[4..16]);
                    let g = 2 * (2 * j + h);
                    s[g] = d * sc as f32;
                    s[g + 1] = s[g];
                    m[g] = off as f32 * s[g] - min * mn as f32;
                    m[g + 1] = m[g];
                    let mut w = shr_and(v, 4 * h as i32, 0x0F);
                    if five {
                        let bit = _mm256_set1_epi8((1u8 << (2 * j + h)) as i8);
                        let set = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit);
                        w = _mm256_or_si256(w, _mm256_and_si256(set, _mm256_set1_epi8(16)));
                    }
                    st(q, 64 * j + 32 * h, _mm256_sub_epi8(w, _mm256_set1_epi8(off)));
                }
            }
            true
        }
        Q6_K => {
            debug_assert!(b.len() >= 210);
            let d = f16at(b, 208);
            for g in 0..16 {
                s[g] = d * (b[192 + g] as i8) as f32;
            }
            let c32 = _mm256_set1_epi8(32);
            for n in 0..2 {
                let ql = [
                    _mm256_loadu_si256(p.add(64 * n) as *const __m256i),
                    _mm256_loadu_si256(p.add(64 * n + 32) as *const __m256i),
                ];
                let qh = _mm256_loadu_si256(p.add(128 + 32 * n) as *const __m256i);
                for k in 0..4 {
                    // Elements 32 k ..: nibble k / 2 of ql[k % 2], with bits 2 k, 2 k + 1 of qh on top.
                    let lo = shr_and(ql[k % 2], 4 * (k as i32 / 2), 0x0F);
                    let hi = _mm256_slli_epi16::<4>(shr_and(qh, 2 * k as i32, 3));
                    st(q, 128 * n + 32 * k, _mm256_sub_epi8(_mm256_or_si256(lo, hi), c32));
                }
            }
            false
        }
        IQ4_XS => {
            debug_assert!(b.len() >= 136);
            let d = f16at(b, 0);
            for ib in 0..8 {
                s[2 * ib] = d * iq4xs_scale(b, ib) as f32;
                s[2 * ib + 1] = s[2 * ib];
                st(q, 32 * ib, lookup(&KVALUES_IQ4NL, nibbles(p.add(8 + 16 * ib))));
            }
            false
        }
        _ => {
            let bs = kind.block_size();
            decode_i8(
                kind,
                b,
                &mut std::slice::from_raw_parts_mut(q, QK_K)[..bs],
                &mut s[..bs / 16],
                &mut m[..bs / 16],
            )
        }
    }
}

/// Integer dot products of 32 int8 pairs, as 8 partial sums: lanes 0..4 add up to elements
/// 0..16, lanes 4..8 to elements 16..32. `|w| * |x|` pairs stay below 2^15 (the activations are
/// within +-127), so `maddubs` cannot saturate.
#[inline(always)]
unsafe fn dot32(w: *const i8, x: *const i8) -> __m256i {
    let wv = _mm256_load_si256(w as *const __m256i);
    let xv = _mm256_loadu_si256(x as *const __m256i);
    let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
    _mm256_madd_epi16(p16, _mm256_set1_epi16(1))
}

/// `p[g] = s[g] * xs[g / 2] * (sum of q * xq over the 16 elements of group g)` for the `bs / 16`
/// groups of one block: the terms the portable kernel adds up, bit for bit.
#[inline(always)]
unsafe fn products(q: *const i8, s: &[f32; 16], xq: *const i8, xs: *const f32, bs: usize, p: &mut [f32; 16]) {
    if bs == QBLOCK {
        let v = dot32(q, xq);
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        // [lo01, lo23, hi01, hi23] -> [lo, hi, ..]
        let t = _mm_hadd_epi32(lo, hi);
        let t = _mm_hadd_epi32(t, t);
        p[0] = s[0] * *xs * _mm_cvtsi128_si32(t) as f32;
        p[1] = s[1] * *xs * _mm_extract_epi32::<1>(t) as f32;
        return;
    }
    debug_assert_eq!(bs, QK_K);
    // Natural group order from [a_lo, b_lo, c_lo, d_lo | a_hi, b_hi, c_hi, d_hi].
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let twice = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    for h in 0..2 {
        let (qh, xh) = (q.add(128 * h), xq.add(128 * h));
        let t0 = _mm256_hadd_epi32(dot32(qh, xh), dot32(qh.add(32), xh.add(32)));
        let t1 = _mm256_hadd_epi32(dot32(qh.add(64), xh.add(64)), dot32(qh.add(96), xh.add(96)));
        let isum = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(t0, t1), order);
        let sv = _mm256_loadu_ps(s.as_ptr().add(8 * h));
        let xsv = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(_mm_loadu_ps(xs.add(4 * h))), twice);
        let pv = _mm256_mul_ps(_mm256_mul_ps(sv, xsv), _mm256_cvtepi32_ps(isum));
        _mm256_storeu_ps(p.as_mut_ptr().add(8 * h), pv);
    }
}

/// One row against TC activation rows: `dot_row_generic` with the block unpacked and the integer
/// dot products formed by AVX2.
#[inline(always)]
unsafe fn dot_row<const TC: usize>(kind: GgmlType, row: &[u8], x: &Act) -> [f32; TC] {
    let (bs, ts) = (kind.block_size(), kind.type_size());
    let mut q = Q([0i8; QK_K]);
    let mut s = [0f32; 16];
    let mut m = [0f32; 16];
    let mut p = [0f32; 16];
    let mut acc = [0f32; TC];
    for (bi, blk) in row.chunks_exact(ts).enumerate() {
        let has_min = unpack(kind, blk, &mut q.0, &mut s, &mut m);
        let e0 = bi * bs;
        for (c, a) in acc.iter_mut().enumerate() {
            let xq = x.q.add(c * x.n_in + e0);
            let xs = x.s.add(c * x.n_in / QBLOCK + e0 / QBLOCK);
            products(q.0.as_ptr(), &s, xq, xs, bs, &mut p);
            let mut sum = 0f32;
            for pg in &p[..bs / 16] {
                sum += pg;
            }
            if has_min {
                let xh = x.h.add(c * x.n_in / 16 + e0 / 16);
                for g in 0..bs / 16 {
                    sum += m[g] * *xh.add(g);
                }
            }
            *a += sum;
        }
    }
    acc
}

/// Rows `r0..r1` of `m` (a quantized type) against TC activation rows, into `y[c * ys + r]`.
///
/// # Safety
/// The CPU has AVX2, FMA and F16C (`simd::avx2()`); `x` and `y` cover TC activation rows and the outputs written.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn rows<const TC: usize>(m: &GgufMatrix, x: &Act, y: *mut f32, ys: usize, r0: usize, r1: usize) {
    for r in r0..r1 {
        let o = dot_row::<TC>(m.kind, m.row(r), x);
        for c in 0..TC {
            *y.add(c * ys + r) = o[c];
        }
    }
}

/// Stores `q[i] as f32 * s` for 16 int8 values.
#[inline(always)]
unsafe fn st16(o: *mut f32, q: *const i8, s: __m256) {
    for k in 0..2 {
        let v = _mm256_cvtepi8_epi32(_mm_loadl_epi64(q.add(8 * k) as *const __m128i));
        _mm256_storeu_ps(o.add(8 * k), _mm256_mul_ps(s, _mm256_cvtepi32_ps(v)));
    }
}

/// Whether [`dequant_row`] handles `kind`.
fn dequant_kind(kind: GgmlType) -> bool {
    use GgmlType::*;
    matches!(kind, Q4_0 | Q5_0 | Q8_0 | IQ4_NL | MXFP4 | Q3_K | Q4_K | Q5_K | Q6_K | IQ4_XS)
}

/// Dequantizes one row of `n` elements into `out`, bit-identical to `dequantize_row`; false if
/// `kind` has no AVX2 version (nothing is written).
///
/// # Safety
/// The CPU has AVX2, FMA and F16C (`simd::avx2()`); `row` holds `n` elements of `kind` and `out` has room for them.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn dequant_row(kind: GgmlType, row: &[u8], n: usize, out: *mut f32) -> bool {
    if !dequant_kind(kind) {
        return false;
    }
    let (bs, ts) = (kind.block_size(), kind.type_size());
    let mut q = Q([0i8; QK_K]);
    let mut s = [0f32; 16];
    let mut m = [0f32; 16];
    debug_assert_eq!(row.len(), n / bs * ts);
    for (bi, blk) in row.chunks_exact(ts).enumerate() {
        unpack(kind, blk, &mut q.0, &mut s, &mut m);
        let o = out.add(bi * bs);
        if matches!(kind, GgmlType::Q4_K | GgmlType::Q5_K) {
            // The reference formula is `d * sc * v - min * mn` on the unsigned value `v`.
            let min = f16at(blk, 2);
            let off = _mm256_set1_epi32(if kind == GgmlType::Q5_K { 16 } else { 8 });
            for g in 0..8 {
                let (d1, m1) = (
                    _mm256_set1_ps(s[2 * g]),
                    _mm256_set1_ps(min * scale_min_k4(g, &blk[4..16]).1 as f32),
                );
                for k in 0..4 {
                    let at = 32 * g + 8 * k;
                    let v = _mm256_cvtepi8_epi32(_mm_loadl_epi64(q.0.as_ptr().add(at) as *const __m128i));
                    let f = _mm256_cvtepi32_ps(_mm256_add_epi32(v, off));
                    _mm256_storeu_ps(o.add(at), _mm256_sub_ps(_mm256_mul_ps(d1, f), m1));
                }
            }
        } else {
            for g in 0..bs / 16 {
                st16(o.add(16 * g), q.0.as_ptr().add(16 * g), _mm256_set1_ps(s[g]));
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::{dequantize_row, dot_row_generic, random_blocks};
    use super::*;
    use crate::kernels::simd::cpu_has_avx2;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    fn skip() -> bool {
        if !cpu_has_avx2() {
            eprintln!("skipped: this CPU has no AVX2 + FMA");
        }
        !cpu_has_avx2()
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn unpack_avx2(kind: GgmlType, b: &[u8], q: &mut [i8; QK_K], s: &mut [f32; 16], m: &mut [f32; 16]) -> bool {
        unpack(kind, b, q, s, m)
    }

    /// Random blocks plus blocks of all-zero and all-one payload bits (every nibble, 2-bit field
    /// and high bit at its extremes), with the fields `random_blocks` keeps sane left alone.
    fn blocks(kind: GgmlType, seed: &mut u64) -> Vec<u8> {
        let ts = kind.type_size();
        let mut v = random_blocks(kind, 24, seed);
        for (i, fill) in [(0usize, 0x00u8), (1, 0xFF), (2, 0x0F), (3, 0xF0), (4, 0x55), (5, 0xAA)] {
            let sane = random_blocks(kind, 1, seed);
            let b = &mut v[i * ts..(i + 1) * ts];
            b.fill(fill);
            // Restore the f16 / exponent scale fields from a sane block.
            let r = match kind {
                GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => 0..4,
                GgmlType::Q2_K => 80..84,
                GgmlType::Q3_K => 108..110,
                GgmlType::Q6_K => 208..210,
                GgmlType::IQ1_M => 48..56,
                GgmlType::MXFP4 => 0..1,
                _ => 0..2,
            };
            b[r.clone()].copy_from_slice(&sane[r]);
        }
        v
    }

    /// The vector unpackers give `decode_i8`'s int8 values, scales and offsets bit for bit.
    #[test]
    fn avx2_unpack_matches_decode_i8() {
        if skip() {
            return;
        }
        for kind in GgmlType::ALL.into_iter().filter(|k| !k.is_float()) {
            let mut seed = 400 + kind.id() as u64;
            let bs = kind.block_size();
            for (bi, blk) in blocks(kind, &mut seed).chunks_exact(kind.type_size()).enumerate() {
                let (mut q, mut s, mut m) = ([0i8; QK_K], [0f32; 16], [0f32; 16]);
                let (mut qa, mut sa, mut ma) = ([0i8; QK_K], [0f32; 16], [0f32; 16]);
                let want = decode_i8(kind, blk, &mut q[..bs], &mut s[..bs / 16], &mut m[..bs / 16]);
                let got = unsafe { unpack_avx2(kind, blk, &mut qa, &mut sa, &mut ma) };
                assert_eq!(got, want, "{kind}");
                assert_eq!(qa[..bs], q[..bs], "{kind} block {bi}: values");
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&sa[..bs / 16]), bits(&s[..bs / 16]), "{kind} block {bi}: scales");
                if want {
                    assert_eq!(bits(&ma[..bs / 16]), bits(&m[..bs / 16]), "{kind} block {bi}: offsets");
                }
                // Nothing beyond the block is touched.
                assert!(
                    qa[bs..].iter().all(|&v| v == 0) && sa[bs / 16..].iter().all(|&v| v == 0.0),
                    "{kind}"
                );
            }
        }
    }

    /// Every quantized type, 1..=8 tokens (the column groups of multi-token passes), odd row
    /// ranges and block counts: the AVX2 rows are bit-identical to the portable kernel's.
    #[test]
    fn avx2_rows_match_portable_bit_for_bit() {
        if skip() {
            return;
        }
        let n_out = 9usize;
        for kind in GgmlType::ALL.into_iter().filter(|k| !k.is_float()) {
            // Block-of-32 formats get a block count that is not a multiple of 4 or 8.
            let n_in = if kind.block_size() == QK_K { 3 * QK_K } else { 2 * QK_K + 96 };
            let mut seed = 500 + kind.id() as u64;
            let mut raw = random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
            let ext = blocks(kind, &mut seed);
            raw[..ext.len()].copy_from_slice(&ext);
            let m = GgufMatrix::from_bytes(kind, &raw, n_out, n_in).unwrap();
            let t = 8usize;
            // Activations with whole blocks at +-full scale (int8 +-127: the largest products).
            let x: Vec<f32> = (0..t * n_in)
                .map(|i| match (i / 32) % 5 {
                    0 => 1.0,
                    1 => -1.0,
                    _ => (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0,
                })
                .collect();
            let xa = crate::quant::quantize_act(&x, t, n_in);
            macro_rules! go {
                ($($TC:literal),*) => {$(
                    for c0 in [0usize, t - $TC] {
                        let act = Act::new(&xa, c0);
                        let mut want = vec![0f32; $TC * n_out];
                        for r in 0..n_out {
                            let o = unsafe { dot_row_generic::<$TC>(kind, m.row(r), &act) };
                            for c in 0..$TC {
                                want[c * n_out + r] = o[c];
                            }
                        }
                        let mut got = vec![0f32; $TC * n_out];
                        unsafe {
                            rows::<$TC>(&m, &act, got.as_mut_ptr(), n_out, 0, 3);
                            rows::<$TC>(&m, &act, got.as_mut_ptr(), n_out, 3, n_out);
                        }
                        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(&got), bits(&want), "{kind} TC={} c0={c0}", $TC);
                        assert!(want.iter().any(|v| *v != 0.0), "{kind}: all-zero fixture");
                    }
                )*};
            }
            go!(1, 2, 3, 4, 5, 6, 7, 8);
        }
    }

    /// AVX2 dequantization is bit-identical to `dequantize_row` for the types it covers, and
    /// declines the others without writing.
    #[test]
    fn avx2_dequantization_matches_reference() {
        if skip() {
            return;
        }
        for kind in GgmlType::ALL.into_iter().filter(|k| !k.is_float()) {
            let mut seed = 600 + kind.id() as u64;
            let raw = blocks(kind, &mut seed);
            let n = raw.len() / kind.type_size() * kind.block_size();
            let mut got = vec![f32::NAN; n];
            let done = unsafe { dequant_row(kind, &raw, n, got.as_mut_ptr()) };
            assert_eq!(done, dequant_kind(kind), "{kind}");
            if !done {
                assert!(got.iter().all(|v| v.is_nan()), "{kind}: written without being handled");
                continue;
            }
            let mut want = vec![0f32; n];
            dequantize_row(kind, &raw, &mut want);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{kind} element {i}: {a} vs {b}");
            }
        }
    }
}
