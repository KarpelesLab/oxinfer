//! NEON decode kernels for the hot GGUF formats: each unpacks a block's weights to int8 vectors in
//! registers once, then runs SDOT against TC int8 activation rows. Integer sums are formed per
//! 32-element activation block (sub-blocks of 16 are scaled in int32 first) and scaled in f32.
//! Formats with mins subtract `min * sum(x)` using the exact activation sums.
//!
//! Multi-token passes (speculative verification, 2-16 tokens) have their own kernels that unpack
//! each super-block once for all tokens: SDOT one row at a time, or SMMLA (i8mm) on 2x2 tiles of
//! rows and tokens for the formats with integer scales per 16. Both repeat the one-token kernels'
//! f32 operations in the same order, so results are bit-identical whatever the token count. The
//! blocks-of-32 formats (Q8_0, Q4_0, ...) instead run their one-token kernels instantiated for up
//! to 8 tokens on column groups (9 = 5 + 4), each block unpacked once per group. On
//! Apple's 2025 cores SMMLA issues at half SDOT's rate for twice the multiply-adds and does not
//! overlap other vector work, so it only pays where SDOT needs many extra instructions to apply
//! per-16 scales.
#![allow(unsafe_op_in_unsafe_fn)]

use super::{
    Act, GgmlType, GgufMatrix, QK_K, q3k_scales,
    tables::{KVALUES_IQ4NL, KVALUES_MXFP4},
};
use crate::kernels::simd::{f16x4_to_f32, sdot};
use crate::quant::PairAct;
use crate::quant::f16_to_f32;
use std::arch::aarch64::*;

/// `for $i in 0..8 { $body }`, unrolled (keeps the unpacked vectors in registers).
macro_rules! unroll8 {
    ($i:ident, $body:block) => {
        unroll8!(@ $i $body 0 1 2 3 4 5 6 7)
    };
    (@ $i:ident $body:block $($n:literal)*) => {
        $({
            let $i: usize = $n;
            $body
        })*
    };
}

/// `for $i in 0..4 { $body }` with `$i` offset by the constant `$off`, unrolled.
macro_rules! unroll4 {
    ($i:ident, $off:expr, $body:block) => {
        unroll4!(@ $i $off, $body 0 1 2 3)
    };
    (@ $i:ident $off:expr, $body:block $($n:literal)*) => {
        $({
            let $i: usize = $off + $n;
            $body
        })*
    };
}

type RowFn<const TC: usize> = unsafe fn(*const u8, usize, &Act) -> [f32; TC];

/// Rows `r0..r1` of `m` against TC activation rows into `y[c * ys + r]`; false if `m.kind` has
/// no NEON kernel (the caller then uses the portable path).
pub(super) unsafe fn rows<const TC: usize>(m: &GgufMatrix, x: &Act, y: *mut f32, ys: usize, r0: usize, r1: usize) -> bool {
    if m.x4 {
        rows_x4::<TC>(m, x, y, ys, r0, r1);
        return true;
    }
    let f: RowFn<TC> = match m.kind {
        k if x.k && q8k_kind(k) => k_kern::<TC>(k),
        GgmlType::Q4_K => q4k::<TC, false>,
        GgmlType::Q5_K => q4k::<TC, true>,
        GgmlType::Q6_K => sb16::<TC, Q6K>,
        GgmlType::Q3_K => sb16::<TC, Q3K>,
        GgmlType::IQ2_XXS => sb16::<TC, Iq2Xxs>,
        GgmlType::IQ2_XS => sb16::<TC, Iq2Xs>,
        GgmlType::IQ2_S => sb16::<TC, Iq2S>,
        GgmlType::IQ3_XXS => sb16::<TC, Iq3Xxs>,
        GgmlType::IQ3_S => sb16::<TC, Iq3S>,
        GgmlType::IQ1_S => iq1s::<TC>,
        GgmlType::IQ1_M => sb16::<TC, Iq1M>,
        GgmlType::Q2_K => q2k::<TC>,
        GgmlType::IQ4_XS => iq4xs::<TC>,
        GgmlType::Q8_0 => q8_0::<TC>,
        GgmlType::Q4_0 => q32::<TC, Q4_0>,
        GgmlType::IQ4_NL => q32::<TC, IQ4_NL>,
        GgmlType::MXFP4 => q32::<TC, MXFP4>,
        GgmlType::Q4_1 => q32::<TC, Q4_1>,
        GgmlType::Q5_0 => q32::<TC, Q5_0>,
        GgmlType::Q5_1 => q32::<TC, Q5_1>,
        GgmlType::NVFP4 => nvfp4::<TC>,
        GgmlType::Q2_0 => q2_0::<TC>,
        GgmlType::PQ2_0 => pq2_0::<TC>,
        GgmlType::PTQ1_0 => ptq1_0::<TC>,
        _ => return false,
    };
    let rb = m.row_bytes();
    let base = m.data.as_ptr();
    if TC == 1 && rows_t1(m, x, y, r0, r1) {
        return true;
    }
    for r in r0..r1 {
        let o = f(base.add(r * rb), m.n_in, x);
        for c in 0..TC {
            *y.add(c * ys + r) = o[c];
        }
    }
    true
}

/// One-token rows of the formats with kernels that take several rows at a time (bit for bit the
/// one-row kernels'); false for the others.
unsafe fn rows_t1(m: &GgufMatrix, x: &Act, y: *mut f32, r0: usize, r1: usize) -> bool {
    let rb = m.row_bytes();
    let base = m.data.as_ptr();
    let (n, p) = (m.n_in, |r: usize| base.add(r * rb));
    macro_rules! go {
        ($f:ident) => {{
            let mut r = r0;
            while r + 2 <= r1 {
                let o = $f::<2>([p(r), p(r + 1)], n, x);
                *y.add(r) = o[0];
                *y.add(r + 1) = o[1];
                r += 2;
            }
            if r < r1 {
                *y.add(r) = $f::<1>([p(r)], n, x)[0];
            }
        }};
    }
    match m.kind {
        GgmlType::IQ1_S => go!(iq1s_rows),
        GgmlType::IQ1_M => go!(iq1m_rows),
        GgmlType::IQ2_XXS => go!(iq2xxs_rows),
        _ => return false,
    }
    true
}

#[inline(always)]
unsafe fn f16p(p: *const u8) -> f32 {
    f16_to_f32(std::ptr::read_unaligned(p as *const u16))
}

/// u8 lanes 0..4 / 4..8 of `v` as f32.
#[inline(always)]
unsafe fn u8x8_to_f32(v: uint8x8_t) -> (float32x4_t, float32x4_t) {
    let w = vmovl_u8(v);
    (vcvtq_f32_u32(vmovl_u16(vget_low_u16(w))), vcvtq_f32_u32(vmovl_high_u16(w)))
}

/// 16 signed bytes widened to four int32x4.
#[inline(always)]
unsafe fn s8x16_to_s32(v: int8x16_t) -> [int32x4_t; 4] {
    let lo = vmovl_s8(vget_low_s8(v));
    let hi = vmovl_high_s8(v);
    [
        vmovl_s16(vget_low_s16(lo)),
        vmovl_high_s16(lo),
        vmovl_s16(vget_low_s16(hi)),
        vmovl_high_s16(hi),
    ]
}

/// Sums of `acc` lanes per token.
#[inline(always)]
unsafe fn finish<const TC: usize>(acc: [float32x4_t; TC]) -> [f32; TC] {
    let mut out = [0f32; TC];
    for c in 0..TC {
        out[c] = vaddvq_f32(acc[c]);
    }
    out
}

/// Per-32 integer sums of eight sub-blocks of 32 (`w[2k], w[2k+1]` against x at `32k`), as two
/// vectors of four.
#[inline(always)]
unsafe fn sums32(w: &[int8x16_t; 16], xq: *const i8) -> (int32x4_t, int32x4_t) {
    let zero = vdupq_n_s32(0);
    let mut a = [zero; 8];
    for k in 0..8 {
        a[k] = sdot(
            sdot(zero, w[2 * k], vld1q_s8(xq.add(32 * k))),
            w[2 * k + 1],
            vld1q_s8(xq.add(32 * k + 16)),
        );
    }
    (
        vpaddq_s32(vpaddq_s32(a[0], a[1]), vpaddq_s32(a[2], a[3])),
        vpaddq_s32(vpaddq_s32(a[4], a[5]), vpaddq_s32(a[6], a[7])),
    )
}

/// Per-32 integer sums of sixteen sub-blocks of 16 (`w[k]` against x at `16k`), each sub-block
/// weighted by its integer scale `sc[k]` (`sc` as four int32x4).
#[inline(always)]
unsafe fn sums16(w: &[int8x16_t; 16], sc: &[int32x4_t; 4], xq: *const i8) -> (int32x4_t, int32x4_t) {
    let zero = vdupq_n_s32(0);
    let mut m = [zero; 4];
    for g in 0..4 {
        let p0 = sdot(zero, w[4 * g], vld1q_s8(xq.add(64 * g)));
        let p1 = sdot(zero, w[4 * g + 1], vld1q_s8(xq.add(64 * g + 16)));
        let p2 = sdot(zero, w[4 * g + 2], vld1q_s8(xq.add(64 * g + 32)));
        let p3 = sdot(zero, w[4 * g + 3], vld1q_s8(xq.add(64 * g + 48)));
        m[g] = vmulq_s32(vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)), sc[g]);
    }
    (vpaddq_s32(m[0], m[1]), vpaddq_s32(m[2], m[3]))
}

/// Q4_K (and Q5_K with `FIVE`): 8 sub-blocks of 32 with 6-bit scales and mins.
unsafe fn q4k<const TC: usize, const FIVE: bool>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let bpr = n_in / 32;
    let tsz = if FIVE { 176 } else { 144 };
    let m4 = vdupq_n_u8(0x0F);
    let mut acc = [vdupq_n_f32(0.0); TC];
    for sb in 0..n_in / QK_K {
        let b = row.add(sb * tsz);
        let (d, dmin) = (f16p(b), f16p(b.add(2)));
        // Scales and mins (get_scale_min_k4), unpacked four at a time.
        const KM1: u32 = 0x3f3f_3f3f;
        const KM2: u32 = 0x0f0f_0f0f;
        const KM3: u32 = 0x0303_0303;
        let u = [
            std::ptr::read_unaligned(b.add(4) as *const u32),
            std::ptr::read_unaligned(b.add(8) as *const u32),
            std::ptr::read_unaligned(b.add(12) as *const u32),
        ];
        let sc = (u[0] & KM1) as u64 | ((((u[2] & KM2) | (((u[0] >> 6) & KM3) << 4)) as u64) << 32);
        let mn = (u[1] & KM1) as u64 | (((((u[2] >> 4) & KM2) | (((u[1] >> 6) & KM3) << 4)) as u64) << 32);
        let (s0, s1) = u8x8_to_f32(vcreate_u8(sc));
        let (n0, n1) = u8x8_to_f32(vcreate_u8(mn));
        let (ds0, ds1) = (vmulq_n_f32(s0, d), vmulq_n_f32(s1, d));
        // Weights are centered (q - 8, or q - 16 for Q5_K: the int8 rounding of the activations then
        // multiplies small values), as integer sums minus `off * sum(xq)`; the offset moves into
        // the min term with the exact activation sums.
        let off = if FIVE { 16.0 } else { 8.0 };
        let dm0 = vsubq_f32(vmulq_n_f32(n0, dmin), vmulq_n_f32(ds0, off));
        let dm1 = vsubq_f32(vmulq_n_f32(n1, dmin), vmulq_n_f32(ds1, off));
        let qs = b.add(if FIVE { 48 } else { 16 });
        let mut w = [vdupq_n_s8(0); 16];
        let (h0, h1) = if FIVE {
            (vld1q_u8(b.add(16)), vld1q_u8(b.add(32)))
        } else {
            (vdupq_n_u8(0), vdupq_n_u8(0))
        };
        let b16 = vdupq_n_u8(16);
        for j in 0..4 {
            let v0 = vld1q_u8(qs.add(32 * j));
            let v1 = vld1q_u8(qs.add(32 * j + 16));
            let mut q = [vandq_u8(v0, m4), vandq_u8(v1, m4), vshrq_n_u8::<4>(v0), vshrq_n_u8::<4>(v1)];
            if FIVE {
                let (u1, u2) = (vdupq_n_u8(1 << (2 * j)), vdupq_n_u8(2 << (2 * j)));
                q[0] = vorrq_u8(q[0], vandq_u8(vtstq_u8(h0, u1), b16));
                q[1] = vorrq_u8(q[1], vandq_u8(vtstq_u8(h1, u1), b16));
                q[2] = vorrq_u8(q[2], vandq_u8(vtstq_u8(h0, u2), b16));
                q[3] = vorrq_u8(q[3], vandq_u8(vtstq_u8(h1, u2), b16));
            }
            for i in 0..4 {
                w[4 * j + i] = vreinterpretq_s8_u8(q[i]);
            }
        }
        for c in 0..TC {
            let (a0, a1) = sums32(&w, x.q.add(c * n_in + sb * QK_K));
            let xi = x.i.add(c * bpr + sb * 8);
            let (a0, a1) = if FIVE {
                (
                    vsubq_s32(a0, vshlq_n_s32::<4>(vld1q_s32(xi))),
                    vsubq_s32(a1, vshlq_n_s32::<4>(vld1q_s32(xi.add(4)))),
                )
            } else {
                (
                    vsubq_s32(a0, vshlq_n_s32::<3>(vld1q_s32(xi))),
                    vsubq_s32(a1, vshlq_n_s32::<3>(vld1q_s32(xi.add(4)))),
                )
            };
            let xs = x.s.add(c * bpr + sb * 8);
            let xf = x.f.add(c * bpr + sb * 8);
            let mut v = vfmaq_f32(acc[c], vcvtq_f32_s32(a0), vmulq_f32(ds0, vld1q_f32(xs)));
            v = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(ds1, vld1q_f32(xs.add(4))));
            v = vfmsq_f32(v, dm0, vld1q_f32(xf));
            acc[c] = vfmsq_f32(v, dm1, vld1q_f32(xf.add(4)));
        }
    }
    finish(acc)
}

/// A super-block format unpacked to 16 vectors of 16 int8 values (vector k = elements 16k..16k+16),
/// one integer scale per vector and a float factor: element = factor * sc[k] * w[k][i].
trait Sb16 {
    const TSZ: usize;
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32;
    /// The integer scales and the factor alone (as `unpack` returns them).
    #[inline(always)]
    unsafe fn scales(b: *const u8, sc: &mut [i8; 16]) -> f32 {
        let mut w = [vdupq_n_s8(0); 16];
        Self::unpack(b, &mut w, sc)
    }
    /// Vectors `8 H .. 8 H + 8` of `unpack` (the hot formats compute only those).
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        let mut all = [vdupq_n_s8(0); 16];
        let mut sc = [0i8; 16];
        Self::unpack(b, &mut all, &mut sc);
        w.copy_from_slice(&all[8 * H..8 * H + 8]);
    }
}

unsafe fn sb16<const TC: usize, U: Sb16>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let bpr = n_in / 32;
    let mut acc = [vdupq_n_f32(0.0); TC];
    for sb in 0..n_in / QK_K {
        let mut w = [vdupq_n_s8(0); 16];
        let mut scb = [0i8; 16];
        let d = vdupq_n_f32(U::unpack(row.add(sb * U::TSZ), &mut w, &mut scb));
        let sc = s8x16_to_s32(vld1q_s8(scb.as_ptr()));
        for c in 0..TC {
            let (a0, a1) = sums16(&w, &sc, x.q.add(c * n_in + sb * QK_K));
            let xs = x.s.add(c * bpr + sb * 8);
            let v = vfmaq_f32(acc[c], vcvtq_f32_s32(a0), vmulq_f32(d, vld1q_f32(xs)));
            acc[c] = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(d, vld1q_f32(xs.add(4))));
        }
    }
    finish(acc)
}

#[inline(always)]
unsafe fn rd32(p: *const u8) -> u32 {
    std::ptr::read_unaligned(p as *const u32)
}
#[inline(always)]
unsafe fn rd64(p: *const u8) -> u64 {
    std::ptr::read_unaligned(p as *const u64)
}
#[inline(always)]
unsafe fn rd16(p: *const u8) -> u16 {
    std::ptr::read_unaligned(p as *const u16)
}

/// Two 8-byte grid entries as one vector.
#[inline(always)]
unsafe fn grid2(a: u64, b: u64) -> uint8x16_t {
    vcombine_u8(vcreate_u8(a), vcreate_u8(b))
}
/// Four 4-byte grid entries as one vector.
#[inline(always)]
unsafe fn grid4(a: u32, b: u32, c: u32, d: u32) -> uint8x16_t {
    grid2(a as u64 | (b as u64) << 32, c as u64 | (d as u64) << 32)
}
/// `v` negated where `m` is all ones (`m` lanes are 0 or 0xFF).
#[inline(always)]
unsafe fn apply_signs(v: uint8x16_t, m: uint8x16_t) -> int8x16_t {
    vreinterpretq_s8_u8(vsubq_u8(veorq_u8(v, m), m))
}
/// Sign masks for 32 elements from four sign bytes (bit j of byte l: element 8l + j negative).
#[inline(always)]
unsafe fn sign_masks(bytes: u32) -> (uint8x16_t, uint8x16_t) {
    const IDX: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    ];
    const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let v = vreinterpretq_u8_u32(vdupq_n_u32(bytes));
    let bits = vld1q_u8(BITS.as_ptr());
    (
        vtstq_u8(vqtbl1q_u8(v, vld1q_u8(IDX.as_ptr())), bits),
        vtstq_u8(vqtbl1q_u8(v, vld1q_u8(IDX.as_ptr().add(16))), bits),
    )
}

/// `KSIGNS_IQ2XS` expanded to byte masks (entry i: 7 sign bits plus an even-parity eighth).
static KSIGNS64: [u64; 128] = {
    let mut t = [0u64; 128];
    let mut i = 0usize;
    while i < 128 {
        let s = i | ((i.count_ones() as usize & 1) << 7);
        let mut j = 0;
        while j < 8 {
            if s & (1 << j) != 0 {
                t[i] |= 0xFF << (8 * j);
            }
            j += 1;
        }
        i += 1;
    }
    t
};

/// Q6_K: 16 sub-blocks of 16 with int8 scales; 4 low bits in `ql`, 2 high bits in `qh`.
struct Q6K;
impl Sb16 for Q6K {
    const TSZ: usize = 210;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        let m4 = vdupq_n_u8(0x0F);
        let m30 = vdupq_n_u8(0x30);
        let b32 = vdupq_n_s8(32);
        std::ptr::copy_nonoverlapping(b.add(192) as *const i8, sc.as_mut_ptr(), 16);
        for n in 0..2 {
            let ql = b.add(64 * n);
            let qh = b.add(128 + 32 * n);
            let (l0, l1, l2, l3) = (vld1q_u8(ql), vld1q_u8(ql.add(16)), vld1q_u8(ql.add(32)), vld1q_u8(ql.add(48)));
            let (h0, h1) = (vld1q_u8(qh), vld1q_u8(qh.add(16)));
            let q = [
                vorrq_u8(vandq_u8(l0, m4), vandq_u8(vshlq_n_u8::<4>(h0), m30)),
                vorrq_u8(vandq_u8(l1, m4), vandq_u8(vshlq_n_u8::<4>(h1), m30)),
                vorrq_u8(vandq_u8(l2, m4), vandq_u8(vshlq_n_u8::<2>(h0), m30)),
                vorrq_u8(vandq_u8(l3, m4), vandq_u8(vshlq_n_u8::<2>(h1), m30)),
                vorrq_u8(vshrq_n_u8::<4>(l0), vandq_u8(h0, m30)),
                vorrq_u8(vshrq_n_u8::<4>(l1), vandq_u8(h1, m30)),
                vorrq_u8(vshrq_n_u8::<4>(l2), vandq_u8(vshrq_n_u8::<2>(h0), m30)),
                vorrq_u8(vshrq_n_u8::<4>(l3), vandq_u8(vshrq_n_u8::<2>(h1), m30)),
            ];
            for i in 0..8 {
                w[8 * n + i] = vsubq_s8(vreinterpretq_s8_u8(q[i]), b32);
            }
        }
        f16p(b.add(208))
    }
    #[inline(always)]
    unsafe fn scales(b: *const u8, sc: &mut [i8; 16]) -> f32 {
        std::ptr::copy_nonoverlapping(b.add(192) as *const i8, sc.as_mut_ptr(), 16);
        f16p(b.add(208))
    }
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        let m4 = vdupq_n_u8(0x0F);
        let m30 = vdupq_n_u8(0x30);
        let b32 = vdupq_n_s8(32);
        let ql = b.add(64 * H);
        let qh = b.add(128 + 32 * H);
        let (l0, l1, l2, l3) = (vld1q_u8(ql), vld1q_u8(ql.add(16)), vld1q_u8(ql.add(32)), vld1q_u8(ql.add(48)));
        let (h0, h1) = (vld1q_u8(qh), vld1q_u8(qh.add(16)));
        let q = [
            vorrq_u8(vandq_u8(l0, m4), vandq_u8(vshlq_n_u8::<4>(h0), m30)),
            vorrq_u8(vandq_u8(l1, m4), vandq_u8(vshlq_n_u8::<4>(h1), m30)),
            vorrq_u8(vandq_u8(l2, m4), vandq_u8(vshlq_n_u8::<2>(h0), m30)),
            vorrq_u8(vandq_u8(l3, m4), vandq_u8(vshlq_n_u8::<2>(h1), m30)),
            vorrq_u8(vshrq_n_u8::<4>(l0), vandq_u8(h0, m30)),
            vorrq_u8(vshrq_n_u8::<4>(l1), vandq_u8(h1, m30)),
            vorrq_u8(vshrq_n_u8::<4>(l2), vandq_u8(vshrq_n_u8::<2>(h0), m30)),
            vorrq_u8(vshrq_n_u8::<4>(l3), vandq_u8(vshrq_n_u8::<2>(h1), m30)),
        ];
        for i in 0..8 {
            w[i] = vsubq_s8(vreinterpretq_s8_u8(q[i]), b32);
        }
    }
}

/// Q3_K: 16 sub-blocks of 16 with 6-bit scales; 2 low bits in `qs`, the third in `hmask`
/// (clear = subtract 4).
struct Q3K;
impl Sb16 for Q3K {
    const TSZ: usize = 110;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        let m3 = vdupq_n_u8(3);
        let b4 = vdupq_n_u8(4);
        let s4 = vdupq_n_s8(4);
        let scb = q3k_scales(std::slice::from_raw_parts(b.add(96), 12));
        vst1q_s8(sc.as_mut_ptr(), vsubq_s8(vld1q_s8(scb.as_ptr()), vdupq_n_s8(32)));
        let (hm0, hm1) = (vld1q_u8(b), vld1q_u8(b.add(16)));
        for n in 0..2 {
            let (q0, q1) = (vld1q_u8(b.add(32 + 32 * n)), vld1q_u8(b.add(48 + 32 * n)));
            for j in 0..4 {
                let sh = vdupq_n_s8(-2 * j as i8);
                let bit = vdupq_n_u8(1 << (4 * n + j));
                let wa = vorrq_u8(vandq_u8(vshlq_u8(q0, sh), m3), vandq_u8(vtstq_u8(hm0, bit), b4));
                let wb = vorrq_u8(vandq_u8(vshlq_u8(q1, sh), m3), vandq_u8(vtstq_u8(hm1, bit), b4));
                w[8 * n + 2 * j] = vsubq_s8(vreinterpretq_s8_u8(wa), s4);
                w[8 * n + 2 * j + 1] = vsubq_s8(vreinterpretq_s8_u8(wb), s4);
            }
        }
        f16p(b.add(108))
    }
    #[inline(always)]
    unsafe fn scales(b: *const u8, sc: &mut [i8; 16]) -> f32 {
        let scb = q3k_scales(std::slice::from_raw_parts(b.add(96), 12));
        vst1q_s8(sc.as_mut_ptr(), vsubq_s8(vld1q_s8(scb.as_ptr()), vdupq_n_s8(32)));
        f16p(b.add(108))
    }
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        let m3 = vdupq_n_u8(3);
        let b4 = vdupq_n_u8(4);
        let s4 = vdupq_n_s8(4);
        let (hm0, hm1) = (vld1q_u8(b), vld1q_u8(b.add(16)));
        let (q0, q1) = (vld1q_u8(b.add(32 + 32 * H)), vld1q_u8(b.add(48 + 32 * H)));
        for j in 0..4 {
            let sh = vdupq_n_s8(-2 * j as i8);
            let bit = vdupq_n_u8(1 << (4 * H + j));
            let wa = vorrq_u8(vandq_u8(vshlq_u8(q0, sh), m3), vandq_u8(vtstq_u8(hm0, bit), b4));
            let wb = vorrq_u8(vandq_u8(vshlq_u8(q1, sh), m3), vandq_u8(vtstq_u8(hm1, bit), b4));
            w[2 * j] = vsubq_s8(vreinterpretq_s8_u8(wa), s4);
            w[2 * j + 1] = vsubq_s8(vreinterpretq_s8_u8(wb), s4);
        }
    }
}

/// IQ2_XXS: 8-value grid entries (256) with 7-bit sign patterns, a 4-bit scale per 32.
struct Iq2Xxs;
impl Sb16 for Iq2Xxs {
    const TSZ: usize = 66;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        use super::tables::IQ2XXS_GRID as G;
        unroll8!(ib, {
            let (a0, a1) = (rd32(b.add(2 + 8 * ib)), rd32(b.add(6 + 8 * ib)));
            let s = (2 * (a1 >> 28) + 1) as i8;
            sc[2 * ib] = s;
            sc[2 * ib + 1] = s;
            for h in 0..2 {
                let g = grid2(
                    *G.get_unchecked(((a0 >> (16 * h)) & 255) as usize),
                    *G.get_unchecked(((a0 >> (16 * h + 8)) & 255) as usize),
                );
                let m = grid2(
                    *KSIGNS64.get_unchecked(((a1 >> (14 * h)) & 127) as usize),
                    *KSIGNS64.get_unchecked(((a1 >> (14 * h + 7)) & 127) as usize),
                );
                w[2 * ib + h] = apply_signs(g, m);
            }
        });
        f16p(b) * 0.125
    }
}

/// IQ2_XS: 9-bit grid indices (512 entries) with 7-bit sign patterns, 4-bit scales per 16.
struct Iq2Xs;
impl Sb16 for Iq2Xs {
    const TSZ: usize = 74;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        use super::tables::IQ2XS_GRID as G;
        unroll8!(ib, {
            let s = *b.add(66 + ib);
            sc[2 * ib] = (2 * (s & 0xF) + 1) as i8;
            sc[2 * ib + 1] = (2 * (s >> 4) + 1) as i8;
            for h in 0..2 {
                let q0 = rd16(b.add(2 + 2 * (4 * ib + 2 * h)));
                let q1 = rd16(b.add(4 + 2 * (4 * ib + 2 * h)));
                let g = grid2(*G.get_unchecked((q0 & 511) as usize), *G.get_unchecked((q1 & 511) as usize));
                let m = grid2(
                    *KSIGNS64.get_unchecked((q0 >> 9) as usize),
                    *KSIGNS64.get_unchecked((q1 >> 9) as usize),
                );
                w[2 * ib + h] = apply_signs(g, m);
            }
        });
        f16p(b) * 0.125
    }
}

/// IQ2_S: 10-bit grid indices (1024 entries), explicit sign bytes, 4-bit scales per 16.
struct Iq2S;
impl Sb16 for Iq2S {
    const TSZ: usize = 82;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        use super::tables::IQ2S_GRID as G;
        unroll8!(ib, {
            let s = *b.add(74 + ib);
            sc[2 * ib] = (2 * (s & 0xF) + 1) as i8;
            sc[2 * ib + 1] = (2 * (s >> 4) + 1) as i8;
            let qh = *b.add(66 + ib) as usize;
            let q4 = rd32(b.add(2 + 4 * ib)) as usize;
            let idx = |l: usize| ((q4 >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300);
            let (m0, m1) = sign_masks(rd32(b.add(34 + 4 * ib)));
            w[2 * ib] = apply_signs(grid2(*G.get_unchecked(idx(0)), *G.get_unchecked(idx(1))), m0);
            w[2 * ib + 1] = apply_signs(grid2(*G.get_unchecked(idx(2)), *G.get_unchecked(idx(3))), m1);
        });
        f16p(b) * 0.125
    }
}

/// IQ3_XXS: 4-value grid entries (256) with 7-bit sign patterns, a 4-bit scale per 32.
struct Iq3Xxs;
impl Sb16 for Iq3Xxs {
    const TSZ: usize = 98;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        use super::tables::IQ3XXS_GRID as G;
        unroll8!(ib, {
            let aux = rd32(b.add(66 + 4 * ib));
            let s = (2 * (aux >> 28) + 1) as i8;
            sc[2 * ib] = s;
            sc[2 * ib + 1] = s;
            let q8 = rd64(b.add(2 + 8 * ib)) as usize;
            let g = |i: usize| *G.get_unchecked((q8 >> (8 * i)) & 255);
            for h in 0..2 {
                let v = grid4(g(4 * h), g(4 * h + 1), g(4 * h + 2), g(4 * h + 3));
                let m = grid2(
                    *KSIGNS64.get_unchecked(((aux >> (14 * h)) & 127) as usize),
                    *KSIGNS64.get_unchecked(((aux >> (14 * h + 7)) & 127) as usize),
                );
                w[2 * ib + h] = apply_signs(v, m);
            }
        });
        f16p(b) * 0.25
    }
    #[inline(always)]
    unsafe fn scales(b: *const u8, sc: &mut [i8; 16]) -> f32 {
        unroll8!(ib, {
            let s = (2 * (rd32(b.add(66 + 4 * ib)) >> 28) + 1) as i8;
            sc[2 * ib] = s;
            sc[2 * ib + 1] = s;
        });
        f16p(b) * 0.25
    }
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        use super::tables::IQ3XXS_GRID as G;
        unroll4!(ib, 4 * H, {
            let aux = rd32(b.add(66 + 4 * ib));
            let q8 = rd64(b.add(2 + 8 * ib)) as usize;
            let g = |i: usize| *G.get_unchecked((q8 >> (8 * i)) & 255);
            for h in 0..2 {
                let v = grid4(g(4 * h), g(4 * h + 1), g(4 * h + 2), g(4 * h + 3));
                let m = grid2(
                    *KSIGNS64.get_unchecked(((aux >> (14 * h)) & 127) as usize),
                    *KSIGNS64.get_unchecked(((aux >> (14 * h + 7)) & 127) as usize),
                );
                w[2 * (ib - 4 * H) + h] = apply_signs(v, m);
            }
        });
    }
}

/// IQ3_S: 9-bit indices into 4-value grid entries (512), explicit sign bytes, 4-bit scales per 32.
struct Iq3S;
impl Sb16 for Iq3S {
    const TSZ: usize = 110;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        use super::tables::IQ3S_GRID as G;
        unroll8!(ib, {
            let s = (1 + 2 * ((*b.add(106 + ib / 2) >> (4 * (ib % 2))) & 0xF)) as i8;
            sc[2 * ib] = s;
            sc[2 * ib + 1] = s;
            let qh = *b.add(66 + ib) as usize;
            let q8 = rd64(b.add(2 + 8 * ib)) as usize;
            let g = |l: usize, k: usize| *G.get_unchecked(((q8 >> (16 * l + 8 * k)) & 255) | ((qh << (8 - 2 * l - k)) & 256));
            let (m0, m1) = sign_masks(rd32(b.add(74 + 4 * ib)));
            w[2 * ib] = apply_signs(grid4(g(0, 0), g(0, 1), g(1, 0), g(1, 1)), m0);
            w[2 * ib + 1] = apply_signs(grid4(g(2, 0), g(2, 1), g(3, 0), g(3, 1)), m1);
        });
        f16p(b)
    }
    #[inline(always)]
    unsafe fn scales(b: *const u8, sc: &mut [i8; 16]) -> f32 {
        unroll8!(ib, {
            let s = (1 + 2 * ((*b.add(106 + ib / 2) >> (4 * (ib % 2))) & 0xF)) as i8;
            sc[2 * ib] = s;
            sc[2 * ib + 1] = s;
        });
        f16p(b)
    }
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        use super::tables::IQ3S_GRID as G;
        unroll4!(ib, 4 * H, {
            let qh = *b.add(66 + ib) as usize;
            let q8 = rd64(b.add(2 + 8 * ib)) as usize;
            let g = |l: usize, k: usize| *G.get_unchecked(((q8 >> (16 * l + 8 * k)) & 255) | ((qh << (8 - 2 * l - k)) & 256));
            let (m0, m1) = sign_masks(rd32(b.add(74 + 4 * ib)));
            w[2 * (ib - 4 * H)] = apply_signs(grid4(g(0, 0), g(0, 1), g(1, 0), g(1, 1)), m0);
            w[2 * (ib - 4 * H) + 1] = apply_signs(grid4(g(2, 0), g(2, 1), g(3, 0), g(3, 1)), m1);
        });
    }
}

/// IQ1_S: 11-bit indices into ternary grid entries, `q = 8 * grid +- 1` (the +-1/8 delta), 3-bit
/// scales per 32.
struct Iq1S;
impl Sb16 for Iq1S {
    const TSZ: usize = 50;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        use super::tables::IQ1S_GRID as G;
        unroll8!(ib, {
            let qh = rd16(b.add(34 + 2 * ib)) as usize;
            let s = (2 * ((qh >> 12) & 7) + 1) as i8;
            sc[2 * ib] = s;
            sc[2 * ib + 1] = s;
            let delta = vdupq_n_s8(if qh & 0x8000 != 0 { -1 } else { 1 });
            let q4 = rd32(b.add(2 + 4 * ib)) as usize;
            let g = |l: usize| *G.get_unchecked(((q4 >> (8 * l)) & 255) | (((qh >> (3 * l)) & 7) << 8));
            for h in 0..2 {
                let v = vreinterpretq_s8_u8(grid2(g(2 * h), g(2 * h + 1)));
                w[2 * ib + h] = vaddq_s8(vshlq_n_s8::<3>(v), delta);
            }
        });
        f16p(b) * 0.125
    }
}

/// IQ1_S rows for the one-token kernels: `sb16::<TC, Iq1S>` with the `+-1` delta out of the
/// weights. Per 32 values `sum((8 g + delta) x) = 8 sum(g x) + delta sum(x)`, the sum of `x` being
/// the activations' integer block sum, so the ternary grid entries go straight into the dot
/// products (no shift and add per vector) and each 32 takes its scale once. The integer sums
/// equal `sb16`'s, and the float steps are the same: the results are bit for bit `sb16`'s.
unsafe fn iq1s<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    use super::tables::IQ1S_GRID as G;
    let bpr = n_in / 32;
    let mut acc = [vdupq_n_f32(0.0); TC];
    let gp = G.as_ptr();
    for sb in 0..n_in / QK_K {
        let b = row.add(sb * 50);
        let mut w = [vdupq_n_s8(0); 16];
        let (mut sc, mut dl) = ([0i32; 8], [0i32; 8]);
        unroll8!(ib, {
            let qh = rd16(b.add(34 + 2 * ib)) as usize;
            sc[ib] = (2 * ((qh >> 12) & 7) + 1) as i32;
            dl[ib] = if qh & 0x8000 != 0 { -1 } else { 1 };
            let q4 = rd32(b.add(2 + 4 * ib)) as usize;
            let g = |l: usize| gp.add(((q4 >> (8 * l)) & 255) | (((qh >> (3 * l)) & 7) << 8));
            for h in 0..2 {
                w[2 * ib + h] = vreinterpretq_s8_u64(vcombine_u64(vld1_u64(g(2 * h)), vld1_u64(g(2 * h + 1))));
            }
        });
        let d = vdupq_n_f32(f16p(b) * 0.125);
        let (sc0, sc1) = (vld1q_s32(sc.as_ptr()), vld1q_s32(sc.as_ptr().add(4)));
        let (dl0, dl1) = (vld1q_s32(dl.as_ptr()), vld1q_s32(dl.as_ptr().add(4)));
        for c in 0..TC {
            let (g0, g1) = sums32(&w, x.q.add(c * n_in + sb * QK_K));
            let xi = x.i.add(c * bpr + sb * 8);
            let a0 = vmulq_s32(vmlaq_n_s32(vmulq_s32(dl0, vld1q_s32(xi)), g0, 8), sc0);
            let a1 = vmulq_s32(vmlaq_n_s32(vmulq_s32(dl1, vld1q_s32(xi.add(4))), g1, 8), sc1);
            let xs = x.s.add(c * bpr + sb * 8);
            let v = vfmaq_f32(acc[c], vcvtq_f32_s32(a0), vmulq_f32(d, vld1q_f32(xs)));
            acc[c] = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(d, vld1q_f32(xs.add(4))));
        }
    }
    finish(acc)
}

/// One-token IQ1_S rows, `R` at a time: `iq1s::<1>` of each row, bit for bit (the same integer
/// sums per 32 and the same float steps), with the activation vectors loaded once for the `R`
/// rows, a block's index bits read as whole words and its scales and delta signs unpacked as
/// vectors. These kernels are bound by the instructions a grid entry costs (its index from two
/// bit fields, the load, the insert into a vector: about four for 8 weights), not by memory.
#[inline(always)]
unsafe fn iq1s_rows<const R: usize>(rows: [*const u8; R], n_in: usize, x: &Act) -> [f32; R] {
    use super::tables::IQ1S_GRID as G;
    let gp = G.as_ptr();
    let mut acc = [vdupq_n_f32(0.0); R];
    let (m7, one) = (vdupq_n_u16(7), vdupq_n_u16(1));
    for sb in 0..n_in / QK_K {
        let xq = x.q.add(sb * QK_K);
        let mut b = [std::ptr::null::<u8>(); R];
        let mut qw = [[0u64; 4]; R];
        let mut hw = [[0u64; 2]; R];
        for r in 0..R {
            b[r] = rows[r].add(sb * 50);
            for i in 0..4 {
                qw[r][i] = rd64(b[r].add(2 + 8 * i));
            }
            hw[r] = [rd64(b[r].add(34)), rd64(b[r].add(42))];
        }
        // Per-32 SDOT sums of the ternary grid values, pairwise-added as `sums32` does.
        let mut s = [[vdupq_n_s32(0); 8]; R];
        unroll8!(ib, {
            let (x0, x1) = (vld1q_s8(xq.add(32 * ib)), vld1q_s8(xq.add(32 * ib + 16)));
            for r in 0..R {
                let q4 = (qw[r][ib / 2] >> (32 * (ib % 2))) as usize;
                let qh = (hw[r][ib / 4] >> (16 * (ib % 4))) as usize;
                let g = |l: usize| gp.add(((q4 >> (8 * l)) & 255) | (((qh >> (3 * l)) & 7) << 8));
                let w0 = vreinterpretq_s8_u64(vcombine_u64(vld1_u64(g(0)), vld1_u64(g(1))));
                let w1 = vreinterpretq_s8_u64(vcombine_u64(vld1_u64(g(2)), vld1_u64(g(3))));
                s[r][ib] = sdot(sdot(vdupq_n_s32(0), w0, x0), w1, x1);
            }
        });
        let xi = x.i.add(sb * 8);
        let (xi0, xi1) = (vld1q_s32(xi), vld1q_s32(xi.add(4)));
        let xs = x.s.add(sb * 8);
        let (xs0, xs1) = (vld1q_f32(xs), vld1q_f32(xs.add(4)));
        for r in 0..R {
            let a = &s[r];
            let g0 = vpaddq_s32(vpaddq_s32(a[0], a[1]), vpaddq_s32(a[2], a[3]));
            let g1 = vpaddq_s32(vpaddq_s32(a[4], a[5]), vpaddq_s32(a[6], a[7]));
            // Scales 2 ((qh >> 12) & 7) + 1 and deltas -1 / +1 (bit 15) of the eight blocks.
            let qh = vld1q_u16(b[r].add(34) as *const u16);
            let sc = vorrq_u16(vshlq_n_u16::<1>(vandq_u16(vshrq_n_u16::<12>(qh), m7)), one);
            let dl = vorrq_s16(vshrq_n_s16::<15>(vreinterpretq_s16_u16(qh)), vdupq_n_s16(1));
            let sc = vreinterpretq_s16_u16(sc);
            let (sc0, sc1) = (vmovl_s16(vget_low_s16(sc)), vmovl_high_s16(sc));
            let (dl0, dl1) = (vmovl_s16(vget_low_s16(dl)), vmovl_high_s16(dl));
            let a0 = vmulq_s32(vmlaq_n_s32(vmulq_s32(dl0, xi0), g0, 8), sc0);
            let a1 = vmulq_s32(vmlaq_n_s32(vmulq_s32(dl1, xi1), g1, 8), sc1);
            let d = vdupq_n_f32(f16p(b[r]) * 0.125);
            let v = vfmaq_f32(acc[r], vcvtq_f32_s32(a0), vmulq_f32(d, xs0));
            acc[r] = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(d, xs1));
        }
    }
    finish(acc)
}

/// The 16 IQ1_M sub-block scales `2 s + 1` of a block (3 bits each, four per u16 word at bits 0,
/// 3, 6 and 9 of the words at `b + 48`).
#[inline(always)]
unsafe fn iq1m_scales(b: *const u8) -> int8x16_t {
    const IDX: [u8; 32] = [
        0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
    ];
    const SH: [i16; 8] = [0, -3, -6, -9, 0, -3, -6, -9];
    let v = vcombine_u8(vld1_u8(b.add(48)), vdup_n_u8(0));
    let sh = vld1q_s16(SH.as_ptr());
    let m7 = vdupq_n_u16(7);
    let lo = vandq_u16(vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(v, vld1q_u8(IDX.as_ptr()))), sh), m7);
    let hi = vandq_u16(
        vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(v, vld1q_u8(IDX.as_ptr().add(16)))), sh),
        m7,
    );
    let s = vuzp1q_u8(vreinterpretq_u8_u16(lo), vreinterpretq_u8_u16(hi));
    vreinterpretq_s8_u8(vorrq_u8(vshlq_n_u8::<1>(s), vdupq_n_u8(1)))
}

/// One-token IQ1_M rows, `R` at a time: `sb16::<1, Iq1M>` of each row, bit for bit. Per 16 values
/// `sum((8 g + delta) x)` is `8 sum(g x) + delta_lo sum8_lo(x) + delta_hi sum8_hi(x)`: the ternary
/// grid entries go straight into the dot products and the deltas weigh the activations' sums of 8
/// (formed once for the `R` rows), so the integer sums per 32 equal `sums16`'s; the float steps
/// are `sb16`'s.
#[inline(always)]
unsafe fn iq1m_rows<const R: usize>(rows: [*const u8; R], n_in: usize, x: &Act) -> [f32; R] {
    use super::tables::IQ1S_GRID as G;
    let gp = G.as_ptr();
    let mut acc = [vdupq_n_f32(0.0); R];
    let zero = vdupq_n_s32(0);
    for sb in 0..n_in / QK_K {
        let xq = x.q.add(sb * QK_K);
        let mut b = [std::ptr::null::<u8>(); R];
        let mut qw = [[0u64; 4]; R];
        let mut hw = [[0u64; 2]; R];
        let mut scd = [[vdupq_n_s16(0); 4]; R];
        let mut sc32 = [[zero; 4]; R];
        let mut d = [vdupq_n_f32(0.0); R];
        for r in 0..R {
            let br = rows[r].add(sb * 56);
            b[r] = br;
            for i in 0..4 {
                qw[r][i] = rd64(br.add(8 * i));
            }
            hw[r] = [rd64(br.add(32)), rd64(br.add(40))];
            let s = [rd16(br.add(48)), rd16(br.add(50)), rd16(br.add(52)), rd16(br.add(54))];
            let dh = f16_to_f32((s[0] >> 12) | ((s[1] >> 8) & 0x00f0) | ((s[2] >> 4) & 0x0f00) | (s[3] & 0xf000));
            d[r] = vdupq_n_f32(dh * 0.125);
            let sc = iq1m_scales(br);
            sc32[r] = s8x16_to_s32(sc);
            // Scale times delta sign per 8 values (sub-block k's halves at 2k, 2k + 1).
            let qh = vld1q_u8(br.add(32));
            let (mlo, mhi) = (vtstq_u8(qh, vdupq_n_u8(0x08)), vtstq_u8(qh, vdupq_n_u8(0x80)));
            let (m0, m1) = (vzip1q_u8(mlo, mhi), vzip2q_u8(mlo, mhi));
            let scu = vreinterpretq_u8_s8(sc);
            let (s0, s1) = (vzip1q_u8(scu, scu), vzip2q_u8(scu, scu));
            let v0 = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(s0, m0), m0));
            let v1 = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(s1, m1), m1));
            scd[r] = [
                vmovl_s8(vget_low_s8(v0)),
                vmovl_high_s8(v0),
                vmovl_s8(vget_low_s8(v1)),
                vmovl_high_s8(v1),
            ];
        }
        let mut gs = [[zero; 4]; R];
        let mut ds = [[zero; 4]; R];
        unroll4!(g, 0, {
            let xv = [
                vld1q_s8(xq.add(64 * g)),
                vld1q_s8(xq.add(64 * g + 16)),
                vld1q_s8(xq.add(64 * g + 32)),
                vld1q_s8(xq.add(64 * g + 48)),
            ];
            // Sums of 8 of these 64 activations, in order.
            let p = [vpaddlq_s8(xv[0]), vpaddlq_s8(xv[1]), vpaddlq_s8(xv[2]), vpaddlq_s8(xv[3])];
            let s8 = vpaddq_s16(vpaddq_s16(p[0], p[1]), vpaddq_s16(p[2], p[3]));
            for r in 0..R {
                let mut pr = [zero; 4];
                for j in 0..4 {
                    let k = 4 * g + j;
                    let q2 = (qw[r][k / 4] >> (16 * (k % 4))) as usize;
                    let qh = (hw[r][k / 8] >> (8 * (k % 8))) as usize;
                    let e0 = gp.add((q2 & 255) | ((qh << 8) & 0x700));
                    let e1 = gp.add(((q2 >> 8) & 255) | ((qh << 4) & 0x700));
                    let w = vreinterpretq_s8_u64(vcombine_u64(vld1_u64(e0), vld1_u64(e1)));
                    pr[j] = sdot(zero, w, xv[j]);
                }
                gs[r][g] = vmulq_s32(vpaddq_s32(vpaddq_s32(pr[0], pr[1]), vpaddq_s32(pr[2], pr[3])), sc32[r][g]);
                ds[r][g] = vpaddlq_s16(vmulq_s16(scd[r][g], s8));
            }
        });
        let xs = x.s.add(sb * 8);
        let (xs0, xs1) = (vld1q_f32(xs), vld1q_f32(xs.add(4)));
        for r in 0..R {
            let (g, e) = (&gs[r], &ds[r]);
            let a0 = vaddq_s32(vshlq_n_s32::<3>(vpaddq_s32(g[0], g[1])), vpaddq_s32(e[0], e[1]));
            let a1 = vaddq_s32(vshlq_n_s32::<3>(vpaddq_s32(g[2], g[3])), vpaddq_s32(e[2], e[3]));
            let v = vfmaq_f32(acc[r], vcvtq_f32_s32(a0), vmulq_f32(d[r], xs0));
            acc[r] = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(d[r], xs1));
        }
    }
    finish(acc)
}

/// `KSIGNS64` as bytes of +1 / -1 (a sign pattern multiplies the grid values).
static KSIGNS_PM: [u64; 128] = {
    let mut t = [0u64; 128];
    let mut i = 0usize;
    while i < 128 {
        t[i] = KSIGNS64[i] | 0x0101_0101_0101_0101;
        i += 1;
    }
    t
};

/// One-token IQ2_XXS rows, `R` at a time: `sb16::<1, Iq2Xxs>` of each row, bit for bit. The scale
/// is per 32, so each 32 takes one SDOT chain and its scale once (the integer sums equal
/// `sums16`'s with equal scales on both halves); the signs multiply the grid values (+1 / -1
/// bytes); the activation vectors are loaded once for the `R` rows.
#[inline(always)]
unsafe fn iq2xxs_rows<const R: usize>(rows: [*const u8; R], n_in: usize, x: &Act) -> [f32; R] {
    use super::tables::IQ2XXS_GRID as G;
    let gp = G.as_ptr();
    let kp = KSIGNS_PM.as_ptr();
    let mut acc = [vdupq_n_f32(0.0); R];
    let zero = vdupq_n_s32(0);
    for sb in 0..n_in / QK_K {
        let xq = x.q.add(sb * QK_K);
        let mut b = [std::ptr::null::<u8>(); R];
        for r in 0..R {
            b[r] = rows[r].add(sb * 66);
        }
        let mut s = [[zero; 8]; R];
        unroll8!(ib, {
            let (x0, x1) = (vld1q_s8(xq.add(32 * ib)), vld1q_s8(xq.add(32 * ib + 16)));
            for r in 0..R {
                let q = rd64(b[r].add(2 + 8 * ib));
                let (a0, a1) = (q as usize, (q >> 32) as usize);
                let g = |l: usize| gp.add((a0 >> (8 * l)) & 255);
                let k = |l: usize| kp.add((a1 >> (7 * l)) & 127);
                let ld = |p: *const u64, p2: *const u64| vreinterpretq_s8_u64(vcombine_u64(vld1_u64(p), vld1_u64(p2)));
                let (g0, g1) = (ld(g(0), g(1)), ld(g(2), g(3)));
                let (k0, k1) = (ld(k(0), k(1)), ld(k(2), k(3)));
                let (w0, w1) = (vmulq_s8(g0, k0), vmulq_s8(g1, k1));
                s[r][ib] = sdot(sdot(zero, w0, x0), w1, x1);
            }
        });
        let xs = x.s.add(sb * 8);
        let (xs0, xs1) = (vld1q_f32(xs), vld1q_f32(xs.add(4)));
        for r in 0..R {
            let a = &s[r];
            let g0 = vpaddq_s32(vpaddq_s32(a[0], a[1]), vpaddq_s32(a[2], a[3]));
            let g1 = vpaddq_s32(vpaddq_s32(a[4], a[5]), vpaddq_s32(a[6], a[7]));
            // Scales 2 (aux >> 28) + 1 of the eight blocks (the second word of each).
            let w = vld2q_u32_x2(b[r].add(2) as *const u32);
            let one = vdupq_n_u32(1);
            let sc0 = vreinterpretq_s32_u32(vorrq_u32(vshlq_n_u32::<1>(vshrq_n_u32::<28>(w.0.1)), one));
            let sc1 = vreinterpretq_s32_u32(vorrq_u32(vshlq_n_u32::<1>(vshrq_n_u32::<28>(w.1.1)), one));
            let (a0, a1) = (vmulq_s32(g0, sc0), vmulq_s32(g1, sc1));
            let d = vdupq_n_f32(f16p(b[r]) * 0.125);
            let v = vfmaq_f32(acc[r], vcvtq_f32_s32(a0), vmulq_f32(d, xs0));
            acc[r] = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(d, xs1));
        }
    }
    finish(acc)
}

/// Two `vld2q_u32` of 32 bytes each from `p` (the words of 8 blocks of 8 bytes, deinterleaved).
#[inline(always)]
unsafe fn vld2q_u32_x2(p: *const u32) -> ((uint32x4_t, uint32x4_t), (uint32x4_t, uint32x4_t)) {
    let a = vld2q_u32(p);
    let b = vld2q_u32(p.add(8));
    ((a.0, a.1), (b.0, b.1))
}

/// IQ1_M: like IQ1_S with the delta sign per 8 values, 3-bit scales per 16 and the f16 super-block
/// scale spread over the top nibbles of the scale words.
struct Iq1M;
impl Sb16 for Iq1M {
    const TSZ: usize = 56;
    #[inline(always)]
    unsafe fn unpack(b: *const u8, w: &mut [int8x16_t; 16], sc: &mut [i8; 16]) -> f32 {
        use super::tables::IQ1S_GRID as G;
        let s = [rd16(b.add(48)), rd16(b.add(50)), rd16(b.add(52)), rd16(b.add(54))];
        let d = f16_to_f32((s[0] >> 12) | ((s[1] >> 8) & 0x00f0) | ((s[2] >> 4) & 0x0f00) | (s[3] & 0xf000));
        unroll8!(ib, {
            let sv = s[ib / 2] >> (6 * (ib % 2));
            sc[2 * ib] = (2 * (sv & 7) + 1) as i8;
            sc[2 * ib + 1] = (2 * ((sv >> 3) & 7) + 1) as i8;
            for h in 0..2 {
                let qh = *b.add(32 + 2 * ib + h) as usize;
                let q = b.add(4 * ib + 2 * h);
                let v = vreinterpretq_s8_u8(grid2(
                    *G.get_unchecked(*q as usize | ((qh << 8) & 0x700)),
                    *G.get_unchecked(*q.add(1) as usize | ((qh << 4) & 0x700)),
                ));
                let dl = |bit: usize| if qh & bit != 0 { -1 } else { 1 };
                let delta = vcombine_s8(vdup_n_s8(dl(0x08)), vdup_n_s8(dl(0x80)));
                w[2 * ib + h] = vaddq_s8(vshlq_n_s8::<3>(v), delta);
            }
        });
        d * 0.125
    }
}

/// Q2_K: 16 sub-blocks of 16 with 4-bit scales and 4-bit mins.
unsafe fn q2k<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let bpr = n_in / 32;
    let m3 = vdupq_n_u8(3);
    let mut acc = [vdupq_n_f32(0.0); TC];
    for sb in 0..n_in / QK_K {
        let b = row.add(sb * 84);
        let d = vdupq_n_f32(f16p(b.add(80)));
        let dmin = vdupq_n_f32(f16p(b.add(82)));
        let sm = vld1q_u8(b);
        let sc = s8x16_to_s32(vreinterpretq_s8_u8(vandq_u8(sm, vdupq_n_u8(0x0F))));
        let mn = s8x16_to_s32(vreinterpretq_s8_u8(vshrq_n_u8::<4>(sm)));
        // Centered weights (q - 2): per sub-block offset dmin * min - 2 * d * scale.
        let d2 = vaddq_f32(d, d);
        let mnf: [float32x4_t; 4] =
            std::array::from_fn(|k| vsubq_f32(vmulq_f32(vcvtq_f32_s32(mn[k]), dmin), vmulq_f32(vcvtq_f32_s32(sc[k]), d2)));
        let two = vdupq_n_s8(2);
        let mut w = [vdupq_n_s8(0); 16];
        for n in 0..2 {
            let (q0, q1) = (vld1q_u8(b.add(16 + 32 * n)), vld1q_u8(b.add(32 + 32 * n)));
            for j in 0..4 {
                let sh = vdupq_n_s8(-2 * j as i8);
                w[8 * n + 2 * j] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q0, sh), m3)), two);
                w[8 * n + 2 * j + 1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q1, sh), m3)), two);
            }
        }
        for c in 0..TC {
            let (a0, a1) = sums16(&w, &sc, x.q.add(c * n_in + sb * QK_K));
            let xs = x.s.add(c * bpr + sb * 8);
            let xh = x.h.add(c * bpr * 2 + sb * 16);
            let mut v = vfmaq_f32(acc[c], vcvtq_f32_s32(a0), vmulq_f32(d, vld1q_f32(xs)));
            v = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(d, vld1q_f32(xs.add(4))));
            let mut t = vmulq_f32(mnf[0], vld1q_f32(xh));
            t = vfmaq_f32(t, mnf[1], vld1q_f32(xh.add(4)));
            t = vfmaq_f32(t, mnf[2], vld1q_f32(xh.add(8)));
            t = vfmaq_f32(t, mnf[3], vld1q_f32(xh.add(12)));
            acc[c] = vsubq_f32(v, t);
        }
    }
    finish(acc)
}

/// IQ4_XS: 8 sub-blocks of 32, non-linear 4-bit values (table lookup), 6-bit scales.
unsafe fn iq4xs<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let bpr = n_in / 32;
    let m4 = vdupq_n_u8(0x0F);
    let kv = vld1q_s8(KVALUES_IQ4NL.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); TC];
    for sb in 0..n_in / QK_K {
        let b = row.add(sb * 136);
        let d = f16p(b);
        // The eight 6-bit scales in one vector: low nibbles from scales_l, 2-bit tops from scales_h.
        const HSH: [i16; 8] = [4, 2, 0, -2, -4, -6, -8, -10];
        let sl = vcreate_u8(rd32(b.add(4)) as u64);
        let lo = vmovl_u8(vzip1_u8(vand_u8(sl, vdup_n_u8(0x0F)), vshr_n_u8::<4>(sl)));
        let hi = vandq_u16(vshlq_u16(vdupq_n_u16(rd16(b.add(2))), vld1q_s16(HSH.as_ptr())), vdupq_n_u16(0x30));
        let ls = vsubq_s16(vreinterpretq_s16_u16(vorrq_u16(lo, hi)), vdupq_n_s16(32));
        let ds0 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ls))), d);
        let ds1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(ls)), d);
        let mut w = [vdupq_n_s8(0); 16];
        unroll8!(ib, {
            let v = vld1q_u8(b.add(8 + 16 * ib));
            w[2 * ib] = vqtbl1q_s8(kv, vandq_u8(v, m4));
            w[2 * ib + 1] = vqtbl1q_s8(kv, vshrq_n_u8::<4>(v));
        });
        for c in 0..TC {
            let (a0, a1) = sums32(&w, x.q.add(c * n_in + sb * QK_K));
            let xs = x.s.add(c * bpr + sb * 8);
            let v = vfmaq_f32(acc[c], vcvtq_f32_s32(a0), vmulq_f32(ds0, vld1q_f32(xs)));
            acc[c] = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(ds1, vld1q_f32(xs.add(4))));
        }
    }
    finish(acc)
}

/// Q8_0: blocks of 32 int8 with an f16 scale, four blocks per step.
/// NVFP4 in ggml's blocks of 64 (four unsigned E4M3 scales, then per 16 values 8 bytes whose low
/// nibbles are elements 0..8 and high nibbles 8..16): E2M1 codes through MXFP4's doubled table,
/// halved scales. Without the tensor-wide factor, which the caller applies.
unsafe fn nvfp4<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let lut = super::ue4m3_half_table();
    let kv = vld1q_s8(KVALUES_MXFP4.as_ptr());
    let m4 = vdupq_n_u8(0x0F);
    let zero = vdupq_n_s32(0);
    let bpr = n_in / 32;
    let mut acc = [vdupq_n_f32(0.0); TC];
    for blk in 0..n_in / 64 {
        let b = row.add(36 * blk);
        let d = [
            lut[*b as usize],
            lut[*b.add(1) as usize],
            lut[*b.add(2) as usize],
            lut[*b.add(3) as usize],
        ];
        let dv = vld1q_f32(d.as_ptr());
        let (u0, u1) = (vld1q_u8(b.add(4)), vld1q_u8(b.add(20)));
        let (l0, h0) = (vandq_u8(u0, m4), vshrq_n_u8::<4>(u0));
        let (l1, h1) = (vandq_u8(u1, m4), vshrq_n_u8::<4>(u1));
        let w = [
            vqtbl1q_s8(kv, vcombine_u8(vget_low_u8(l0), vget_low_u8(h0))),
            vqtbl1q_s8(kv, vcombine_u8(vget_high_u8(l0), vget_high_u8(h0))),
            vqtbl1q_s8(kv, vcombine_u8(vget_low_u8(l1), vget_low_u8(h1))),
            vqtbl1q_s8(kv, vcombine_u8(vget_high_u8(l1), vget_high_u8(h1))),
        ];
        for c in 0..TC {
            let xq = x.q.add(c * n_in + 64 * blk);
            let a0 = sdot(zero, w[0], vld1q_s8(xq));
            let a1 = sdot(zero, w[1], vld1q_s8(xq.add(16)));
            let a2 = sdot(zero, w[2], vld1q_s8(xq.add(32)));
            let a3 = sdot(zero, w[3], vld1q_s8(xq.add(48)));
            let s = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
            let xs = vld1_f32(x.s.add(c * bpr + 2 * blk));
            let xsv = vcombine_f32(vdup_lane_f32::<0>(xs), vdup_lane_f32::<1>(xs));
            acc[c] = vfmaq_f32(acc[c], vcvtq_f32_s32(s), vmulq_f32(dv, xsv));
        }
    }
    finish(acc)
}

/// The 64 weights of a Q2_0 block (an f16 scale, then 16 bytes of 2-bit codes `c - 1`: element
/// `j` in bits `2 (j % 4)` of byte `j / 4`) as four vectors of 16 in element order.
#[inline(always)]
unsafe fn q2_0_unpack(b: *const u8) -> [int8x16_t; 4] {
    let v = vld1q_u8(b.add(2));
    let m3 = vdupq_n_u8(3);
    // Field k of byte i is element 4 i + k: interleave the four fields byte by byte.
    let e0 = vandq_u8(v, m3);
    let e1 = vandq_u8(vshrq_n_u8::<2>(v), m3);
    let e2 = vandq_u8(vshrq_n_u8::<4>(v), m3);
    let e3 = vshrq_n_u8::<6>(v);
    let (a_lo, a_hi) = (vreinterpretq_u16_u8(vzip1q_u8(e0, e1)), vreinterpretq_u16_u8(vzip2q_u8(e0, e1)));
    let (b_lo, b_hi) = (vreinterpretq_u16_u8(vzip1q_u8(e2, e3)), vreinterpretq_u16_u8(vzip2q_u8(e2, e3)));
    let one = vdupq_n_s8(1);
    let w = |u: uint16x8_t| vsubq_s8(vreinterpretq_s8_u16(u), one);
    [
        w(vzip1q_u16(a_lo, b_lo)),
        w(vzip2q_u16(a_lo, b_lo)),
        w(vzip1q_u16(a_hi, b_hi)),
        w(vzip2q_u16(a_hi, b_hi)),
    ]
}

/// Q2_0 (ggml type 42: ternary checkpoints, low-bit experts): blocks of 64 with one f16 scale.
unsafe fn q2_0<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let zero = vdupq_n_s32(0);
    let bpr = n_in / 32;
    let mut acc = [vdupq_n_f32(0.0); TC];
    for blk in 0..n_in / 64 {
        let b = row.add(18 * blk);
        let w = q2_0_unpack(b);
        let dv = vdupq_n_f32(f16p(b));
        for c in 0..TC {
            let xq = x.q.add(c * n_in + 64 * blk);
            let a0 = sdot(zero, w[0], vld1q_s8(xq));
            let a1 = sdot(zero, w[1], vld1q_s8(xq.add(16)));
            let a2 = sdot(zero, w[2], vld1q_s8(xq.add(32)));
            let a3 = sdot(zero, w[3], vld1q_s8(xq.add(48)));
            let s = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
            let xs = vld1_f32(x.s.add(c * bpr + 2 * blk));
            let xsv = vcombine_f32(vdup_lane_f32::<0>(xs), vdup_lane_f32::<1>(xs));
            acc[c] = vfmaq_f32(acc[c], vcvtq_f32_s32(s), vmulq_f32(dv, xsv));
        }
    }
    finish(acc)
}

/// The 128 weights of a PQ2_0 block (PrismML: Q2_0's codes, 32 bytes of them after one f16
/// scale) as eight vectors of 16 in element order: two Q2_0 halves under one scale.
#[inline(always)]
unsafe fn pq2_0_unpack(b: *const u8) -> [int8x16_t; 8] {
    let (lo, hi) = (q2_0_unpack(b), q2_0_unpack(b.add(16)));
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}

/// Base-3 digit `n` (`3^n` in `p`) of each byte of `v` minus one, as ggml reads TQ1_0 /
/// PTQ1_0 bytes: `((v * 3^n mod 256) * 3) >> 8` (see `ggml::TRITS`).
#[inline(always)]
unsafe fn trit16(v: uint8x16_t, p: u8) -> int8x16_t {
    let q = vmulq_u8(v, vdupq_n_u8(p));
    let three = vdup_n_u8(3);
    let d = vcombine_u8(
        vshrn_n_u16::<8>(vmull_u8(vget_low_u8(q), three)),
        vshrn_n_u16::<8>(vmull_high_u8(q, vdupq_n_u8(3))),
    );
    vsubq_s8(vreinterpretq_s8_u8(d), vdupq_n_s8(1))
}

/// [`trit16`] of eight bytes.
#[inline(always)]
unsafe fn trit8(v: uint8x8_t, p: uint8x8_t) -> int8x8_t {
    let d = vshrn_n_u16::<8>(vmull_u8(vmul_u8(v, p), vdup_n_u8(3)));
    vsub_s8(vreinterpret_s8_u8(d), vdup_n_s8(1))
}

/// The 128 trits of a PTQ1_0 block (PrismML: 24 bytes of five trits each in runs of 16 and 8,
/// digit-major, then 2 bytes of four trits, then an f16 scale; see `ggml::low_bit_i8`) as eight
/// vectors of 16 in element order.
#[inline(always)]
unsafe fn ptq1_0_unpack(b: *const u8) -> [int8x16_t; 8] {
    // Elements 0..80: digit n of byte m of the first run is element 16 n + m.
    let v = vld1q_u8(b);
    let r: [int8x16_t; 5] = [trit16(v, 1), trit16(v, 3), trit16(v, 9), trit16(v, 27), trit16(v, 81)];
    // Elements 80..120: digit n of byte m of the second run is element 80 + 8 n + m.
    let w = vld1_u8(b.add(16));
    let d = |p: u8| trit8(w, vdup_n_u8(p));
    // Elements 120..128: digit n of qh byte h is element 120 + 2 n + h.
    let (h0, h1) = (*b.add(24), *b.add(25));
    let hv = vld1_u8([h0, h1, h0, h1, h0, h1, h0, h1].as_ptr());
    let hp = vld1_u8([1u8, 1, 3, 3, 9, 9, 27, 27].as_ptr());
    [
        r[0],
        r[1],
        r[2],
        r[3],
        r[4],
        vcombine_s8(d(1), d(3)),
        vcombine_s8(d(9), d(27)),
        vcombine_s8(d(81), trit8(hv, hp)),
    ]
}

/// The eight vectors of a 128-value ternary block (see [`pq2_0_unpack`], [`ptq1_0_unpack`]).
type Unpack128 = unsafe fn(*const u8) -> [int8x16_t; 8];

/// A row of 128-blocks of ternary int8 values (`unpack` gives a block's eight vectors, its f16
/// scale at byte `scale_at`) against TC int8 activation rows (one scale per 32).
#[inline(always)]
unsafe fn t128<const TC: usize>(row: *const u8, n_in: usize, x: &Act, bs: usize, scale_at: usize, unpack: Unpack128) -> [f32; TC] {
    let zero = vdupq_n_s32(0);
    let bpr = n_in / 32;
    let mut acc = [vdupq_n_f32(0.0); TC];
    for blk in 0..n_in / 128 {
        let b = row.add(bs * blk);
        let w = unpack(b);
        let dv = vdupq_n_f32(f16p(b.add(scale_at)));
        for c in 0..TC {
            let xq = x.q.add(c * n_in + 128 * blk);
            let xs = vld1q_f32(x.s.add(c * bpr + 4 * blk));
            for h in 0..2 {
                let a0 = sdot(zero, w[4 * h], vld1q_s8(xq.add(64 * h)));
                let a1 = sdot(zero, w[4 * h + 1], vld1q_s8(xq.add(64 * h + 16)));
                let a2 = sdot(zero, w[4 * h + 2], vld1q_s8(xq.add(64 * h + 32)));
                let a3 = sdot(zero, w[4 * h + 3], vld1q_s8(xq.add(64 * h + 48)));
                let s = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
                // Lanes: elements 0..16, 16..32 of block 2h, then of block 2h + 1.
                let xsv = if h == 0 { vzip1q_f32(xs, xs) } else { vzip2q_f32(xs, xs) };
                acc[c] = vfmaq_f32(acc[c], vcvtq_f32_s32(s), vmulq_f32(dv, xsv));
            }
        }
    }
    finish(acc)
}

/// PQ2_0 (PrismML's type 142): Q2_0's codes in blocks of 128 with one f16 scale.
unsafe fn pq2_0<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    t128::<TC>(row, n_in, x, 34, 0, pq2_0_unpack)
}

/// PTQ1_0 (PrismML's type 143): five trits a byte in blocks of 128 with one f16 scale.
unsafe fn ptq1_0<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    t128::<TC>(row, n_in, x, 28, 26, ptq1_0_unpack)
}

unsafe fn q8_0<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let bpr = n_in / 32;
    let zero = vdupq_n_s32(0);
    let mut acc = [vdupq_n_f32(0.0); TC];
    let mut blk = 0;
    while blk + 4 <= bpr {
        let b = row.add(blk * 34);
        let mut dh = [0u16; 4];
        let mut w = [vdupq_n_s8(0); 8];
        for k in 0..4 {
            dh[k] = std::ptr::read_unaligned(b.add(34 * k) as *const u16);
            w[2 * k] = vld1q_s8(b.add(34 * k + 2) as *const i8);
            w[2 * k + 1] = vld1q_s8(b.add(34 * k + 18) as *const i8);
        }
        let dv = f16x4_to_f32(dh.as_ptr());
        for c in 0..TC {
            let xq = x.q.add(c * n_in + blk * 32);
            let mut a = [zero; 4];
            for k in 0..4 {
                a[k] = sdot(
                    sdot(zero, w[2 * k], vld1q_s8(xq.add(32 * k))),
                    w[2 * k + 1],
                    vld1q_s8(xq.add(32 * k + 16)),
                );
            }
            let s = vpaddq_s32(vpaddq_s32(a[0], a[1]), vpaddq_s32(a[2], a[3]));
            acc[c] = vfmaq_f32(acc[c], vcvtq_f32_s32(s), vmulq_f32(dv, vld1q_f32(x.s.add(c * bpr + blk))));
        }
        blk += 4;
    }
    let mut out = finish(acc);
    while blk < bpr {
        let b = row.add(blk * 34);
        let (w0, w1) = (vld1q_s8(b.add(2) as *const i8), vld1q_s8(b.add(18) as *const i8));
        let d = f16p(b);
        for c in 0..TC {
            let xq = x.q.add(c * n_in + blk * 32);
            let s = vaddvq_s32(sdot(sdot(zero, w0, vld1q_s8(xq)), w1, vld1q_s8(xq.add(16))));
            out[c] += s as f32 * d * *x.s.add(c * bpr + blk);
        }
        blk += 1;
    }
    out
}

/// SDOT by element: each lane of `acc` gains the dot product of its four int8 values of `w`
/// with the four of `a` at 32-bit lane `L`.
#[inline(always)]
unsafe fn sdot_lane<const L: i32>(acc: int32x4_t, w: int8x16_t, a: int8x16_t) -> int32x4_t {
    let mut r = acc;
    std::arch::asm!(
        ".arch_extension dotprod",
        "sdot {r:v}.4s, {w:v}.16b, {a:v}.4b[{l}]",
        r = inout(vreg) r,
        w = in(vreg) w,
        a = in(vreg) a,
        l = const L,
        options(pure, nomem, nostack)
    );
    r
}

/// Q8_0 or Q4_0 (`K`: `Q8` or `Q4_0`) interleaved four rows at a time ([`super::repack`]): the
/// four rows of the group at `g` against TC activation rows, lane `i` of each result row `i`.
/// Every row repeats the plain kernel's f32 operations (`q8_0`, `q32::<_, Q4_0>`) in its order
/// (lane `k` of its accumulator there, the block `blk + k` of each step of four, is accumulator
/// `k` here; `finish` adds them as `(0 + 1) + (2 + 3)`), so the results are bit-identical to the
/// plain rows'.
#[inline(always)]
unsafe fn x4_kern<const TC: usize, const K: u8>(g: *const u8, n_in: usize, x: &Act) -> [float32x4_t; TC] {
    let bpr = n_in / 32;
    let zero = vdupq_n_s32(0);
    let bsz = if K == Q8 { 136 } else { 72 };
    let m4 = vdupq_n_u8(0x0F);
    let mut acc = [[vdupq_n_f32(0.0); 4]; TC];
    // The integer sums of block `b` for token `c`, one row per lane (Q4_0: less 8 times the sum
    // of the activations, as `q32` takes it off).
    let isum = |w: &[int8x16_t; 8], c: usize, b: usize| -> int32x4_t {
        let xq = x.q.add(c * n_in + b * 32);
        let (a0, a1) = (vld1q_s8(xq), vld1q_s8(xq.add(16)));
        let s0 = sdot_lane::<3>(
            sdot_lane::<2>(sdot_lane::<1>(sdot_lane::<0>(zero, w[0], a0), w[1], a0), w[2], a0),
            w[3],
            a0,
        );
        let s1 = sdot_lane::<3>(
            sdot_lane::<2>(sdot_lane::<1>(sdot_lane::<0>(zero, w[4], a1), w[5], a1), w[6], a1),
            w[7],
            a1,
        );
        let s = vaddq_s32(s0, s1);
        if K == Q8 {
            s
        } else {
            vsubq_s32(s, vdupq_n_s32(*x.i.add(c * bpr + b) << 3))
        }
    };
    // Pieces `w[j]`, j < 4: values `4j..4j + 4` of each row; `w[4 + j]`: values `16 + 4j..`.
    let load = |b: usize| -> (float32x4_t, [int8x16_t; 8]) {
        let p = g.add(b * bsz);
        let mut w = [vdupq_n_s8(0); 8];
        if K == Q8 {
            for (j, v) in w.iter_mut().enumerate() {
                *v = vld1q_s8(p.add(8 + 16 * j) as *const i8);
            }
        } else {
            for j in 0..4 {
                let v = vld1q_u8(p.add(8 + 16 * j));
                w[j] = vreinterpretq_s8_u8(vandq_u8(v, m4));
                w[4 + j] = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
            }
        }
        (f16x4_to_f32(p as *const u16), w)
    };
    let mut blk = 0;
    while blk + 4 <= bpr {
        for k in 0..4 {
            let (dv, w) = load(blk + k);
            for c in 0..TC {
                let s = isum(&w, c, blk + k);
                acc[c][k] = vfmaq_f32(acc[c][k], vcvtq_f32_s32(s), vmulq_n_f32(dv, *x.s.add(c * bpr + blk + k)));
            }
        }
        blk += 4;
    }
    let mut out = [vdupq_n_f32(0.0); TC];
    for c in 0..TC {
        out[c] = vaddq_f32(vaddq_f32(acc[c][0], acc[c][1]), vaddq_f32(acc[c][2], acc[c][3]));
    }
    while blk < bpr {
        let (dv, w) = load(blk);
        for c in 0..TC {
            let s = isum(&w, c, blk);
            out[c] = vaddq_f32(out[c], vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(s), dv), *x.s.add(c * bpr + blk)));
        }
        blk += 1;
    }
    out
}

/// The `x4_kern` tag of Q8_0.
const Q8: u8 = 255;

/// The `x4_group` tag of Q4_K.
const Q4K: u8 = 254;

/// Q4_K interleaved four rows at a time ([`super::repack`]: per super-block the four `d`, the
/// four `dmin`, then the 12 scale bytes and the 128 value bytes of each row in 4-byte pieces):
/// `q4k`'s operations for each row in its order, lane `i` of each result row `i`. `q4k`'s lane
/// `l` (sub-blocks `l` and `l + 4` of every super-block) is accumulator `l` here; the scales of
/// the four rows are unpacked together, to the same values (small integers, then the same f32
/// products), one vector per sub-block.
///
/// Fewer instructions for the same results: each sub-block's integer sum starts from `-8` times
/// the activations' sum ([`Act::n8`], loaded to every lane) instead of having it subtracted, the
/// scale bytes become f32 lanes in one table lookup and one conversion each, and `dm` is one
/// fused multiply-subtract (`ds * 8` is exact, so `n * dmin - ds * 8` rounds once either way).
#[inline(always)]
unsafe fn q4k_x4<const TC: usize>(g: *const u8, n_in: usize, x: &Act) -> [float32x4_t; TC] {
    let bpr = n_in / 32;
    let m4 = vdupq_n_u8(0x0F);
    let km1 = vdupq_n_u32(0x3f3f_3f3f);
    let (km2, km3) = (vdupq_n_u32(0x0f0f_0f0f), vdupq_n_u32(0x0303_0303));
    // Byte `k` of each 32-bit lane alone, in the lane's low byte (index 0xFF reads as 0).
    const BY_K: [[u8; 16]; 4] = {
        let mut t = [[0xFFu8; 16]; 4];
        let mut k = 0;
        while k < 4 {
            let mut i = 0;
            while i < 4 {
                t[k][4 * i] = (4 * i + k) as u8;
                i += 1;
            }
            k += 1;
        }
        t
    };
    let idx = [
        vld1q_u8(BY_K[0].as_ptr()),
        vld1q_u8(BY_K[1].as_ptr()),
        vld1q_u8(BY_K[2].as_ptr()),
        vld1q_u8(BY_K[3].as_ptr()),
    ];
    let eight = vdupq_n_f32(8.0);
    let mut acc = [[vdupq_n_f32(0.0); 4]; TC];
    for sb in 0..n_in / QK_K {
        let b = g.add(sb * 576);
        // Integer sums of the eight sub-blocks (less 8 times the activations' sums) first, the
        // scales after (fewer vectors live at once).
        let mut isum = [[vdupq_n_s32(0); 8]; TC];
        let qs = b.add(64);
        unroll4!(j, 0, {
            // Chunk `j`'s pieces: low nibbles are sub-block `2j`, high ones `2j + 1`.
            let mut lo = [vdupq_n_s8(0); 8];
            let mut hi = [vdupq_n_s8(0); 8];
            for p in 0..8 {
                let v = vld1q_u8(qs.add(16 * (8 * j + p)));
                lo[p] = vreinterpretq_s8_u8(vandq_u8(v, m4));
                hi[p] = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
            }
            for c in 0..TC {
                let xq = x.q.add(c * n_in + sb * QK_K + 64 * j);
                let n8 = x.n8.add(c * bpr + sb * 8 + 2 * j);
                let dot = |init: int32x4_t, w: &[int8x16_t; 8], xq: *const i8| -> int32x4_t {
                    let (a0, a1) = (vld1q_s8(xq), vld1q_s8(xq.add(16)));
                    let s = sdot_lane::<3>(
                        sdot_lane::<2>(sdot_lane::<1>(sdot_lane::<0>(init, w[0], a0), w[1], a0), w[2], a0),
                        w[3],
                        a0,
                    );
                    sdot_lane::<3>(
                        sdot_lane::<2>(sdot_lane::<1>(sdot_lane::<0>(s, w[4], a1), w[5], a1), w[6], a1),
                        w[7],
                        a1,
                    )
                };
                isum[c][2 * j] = dot(vld1q_dup_s32(n8), &lo, xq);
                isum[c][2 * j + 1] = dot(vld1q_dup_s32(n8.add(1)), &hi, xq.add(32));
            }
        });
        let (d, dmin) = (f16x4_to_f32(b as *const u16), f16x4_to_f32(b.add(8) as *const u16));
        // The scales and mins (`get_scale_min_k4`, as `q4k` unpacks them) of the four rows at
        // once: word `w` of the twelve bytes of row `i` is lane `i` of `u[w]`.
        let u = [
            vld1q_u32(b.add(16) as *const u32),
            vld1q_u32(b.add(32) as *const u32),
            vld1q_u32(b.add(48) as *const u32),
        ];
        let hi6 = |a: uint32x4_t, c: uint32x4_t| vorrq_u32(a, vshlq_n_u32::<4>(vandq_u32(vshrq_n_u32::<6>(c), km3)));
        // Each: lane `i` holds row `i`'s four bytes (sub-blocks 0..4, or 4..8).
        let raw = [
            vandq_u32(u[0], km1),
            hi6(vandq_u32(u[2], km2), u[0]),
            vandq_u32(u[1], km1),
            hi6(vandq_u32(vshrq_n_u32::<4>(u[2]), km2), u[1]),
        ];
        // Sub-block `k` of the four rows (byte `k` of each lane) as f32 lanes.
        let by_k = |v: uint32x4_t, k: usize| vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(v), idx[k])));
        let xs: [*const f32; TC] = std::array::from_fn(|c| x.s.add(c * bpr + sb * 8));
        let xf: [*const f32; TC] = std::array::from_fn(|c| x.f.add(c * bpr + sb * 8));
        let xs0: [float32x4_t; TC] = std::array::from_fn(|c| vld1q_f32(xs[c]));
        let xs1: [float32x4_t; TC] = std::array::from_fn(|c| vld1q_f32(xs[c].add(4)));
        let xf0: [float32x4_t; TC] = std::array::from_fn(|c| vld1q_f32(xf[c]));
        let xf1: [float32x4_t; TC] = std::array::from_fn(|c| vld1q_f32(xf[c].add(4)));
        // Sub-blocks `l` and `l + 4` (accumulator `l`).
        macro_rules! sub {
            ($l:literal) => {{
                let ds_lo = vmulq_f32(by_k(raw[0], $l), d);
                let ds_hi = vmulq_f32(by_k(raw[1], $l), d);
                let dm_lo = vfmsq_f32(vmulq_f32(by_k(raw[2], $l), dmin), ds_lo, eight);
                let dm_hi = vfmsq_f32(vmulq_f32(by_k(raw[3], $l), dmin), ds_hi, eight);
                for c in 0..TC {
                    let mut v = vfmaq_f32(acc[c][$l], vcvtq_f32_s32(isum[c][$l]), vmulq_laneq_f32::<$l>(ds_lo, xs0[c]));
                    v = vfmaq_f32(v, vcvtq_f32_s32(isum[c][$l + 4]), vmulq_laneq_f32::<$l>(ds_hi, xs1[c]));
                    v = vfmsq_laneq_f32::<$l>(v, dm_lo, xf0[c]);
                    acc[c][$l] = vfmsq_laneq_f32::<$l>(v, dm_hi, xf1[c]);
                }
            }};
        }
        sub!(0);
        sub!(1);
        sub!(2);
        sub!(3);
    }
    let mut out = [vdupq_n_f32(0.0); TC];
    for c in 0..TC {
        out[c] = vaddq_f32(vaddq_f32(acc[c][0], acc[c][1]), vaddq_f32(acc[c][2], acc[c][3]));
    }
    out
}

/// The `x4_group` tag of Q6_K.
const Q6KX: u8 = 253;

/// Q6_K interleaved four rows at a time ([`super::repack`]: per super-block the four `d`, the
/// int8 scales by sub-block (the four rows' scale of sub-block `k` at `4k`), then `ql` and `qh`
/// in 4-byte pieces): `sb16::<_, Q6K>`'s operations for each row in its order, lane `i` of each
/// result row `i` (its lane `l`, blocks of 32 `l` and `l + 4`, is accumulator `l` here). Each
/// sub-block of 16's integer sum times its scale, added in pairs, is the same integer as there.
///
/// The values stay unsigned (0..64): each sub-block's sum starts from `-32` times the
/// activations' sum over its 16 values ([`Act::n32h`]), the same integer as the sum of the
/// centered values. Each `qh` piece is read once for the two `ql` segments it completes, and
/// every value takes two instructions to build (a masked insert or add of the shifted nibble).
#[inline(always)]
unsafe fn q6k_x4<const TC: usize>(g: *const u8, n_in: usize, x: &Act) -> [float32x4_t; TC] {
    let bpr = n_in / 32;
    let zero = vdupq_n_s32(0);
    let (m3, m30) = (vdupq_n_u8(0x03), vdupq_n_u8(0x30));
    let mut acc = [[vdupq_n_f32(0.0); 4]; TC];
    for sb in 0..n_in / QK_K {
        let b = g.add(sb * 840);
        let dv = f16x4_to_f32(b as *const u16);
        let (ql, qh) = (b.add(72), b.add(584));
        // Sums of the sub-blocks of 16, one lane per row.
        let mut s16 = [[zero; 16]; TC];
        for n in 0..2 {
            for s in 0..2 {
                // `qh` bytes `32n + 16s..` complete `ql` segments `s` (bits 0-1: its low
                // nibbles, sub-block `8n + s`; bits 4-5: its high ones, `8n + s + 4`) and
                // `s + 2` (bits 2-3: `8n + s + 2`; bits 6-7: `8n + s + 6`).
                let mut w = [[vdupq_n_s8(0); 4]; 4];
                for t in 0..4 {
                    let h = vld1q_u8(qh.add(16 * (8 * n + 4 * s + t)));
                    let l0 = vld1q_u8(ql.add(16 * (16 * n + 4 * s + t)));
                    let l1 = vld1q_u8(ql.add(16 * (16 * n + 4 * (s + 2) + t)));
                    let h2 = vshrq_n_u8::<2>(h);
                    w[0][t] = vreinterpretq_s8_u8(vsliq_n_u8::<4>(l0, vandq_u8(h, m3)));
                    w[1][t] = vreinterpretq_s8_u8(vsraq_n_u8::<4>(vandq_u8(h, m30), l0));
                    w[2][t] = vreinterpretq_s8_u8(vsliq_n_u8::<4>(l1, vandq_u8(h2, m3)));
                    w[3][t] = vreinterpretq_s8_u8(vsraq_n_u8::<4>(vandq_u8(h2, m30), l1));
                }
                let ks = [8 * n + s, 8 * n + s + 4, 8 * n + s + 2, 8 * n + s + 6];
                for c in 0..TC {
                    let xq = x.q.add(c * n_in + sb * QK_K);
                    let nh = x.n32h.add(c * 2 * bpr + sb * 16);
                    for (wj, &k) in w.iter().zip(&ks) {
                        let a = vld1q_s8(xq.add(16 * k));
                        s16[c][k] = sdot_lane::<3>(
                            sdot_lane::<2>(
                                sdot_lane::<1>(sdot_lane::<0>(vld1q_dup_s32(nh.add(k)), wj[0], a), wj[1], a),
                                wj[2],
                                a,
                            ),
                            wj[3],
                            a,
                        );
                    }
                }
            }
        }
        // The scales, sub-block `k`'s four rows as int32 lanes.
        let mut sc = [zero; 16];
        for q in 0..4 {
            let v = s8x16_to_s32(vld1q_s8(b.add(8 + 16 * q) as *const i8));
            sc[4 * q..4 * q + 4].copy_from_slice(&v);
        }
        for c in 0..TC {
            let xs = x.s.add(c * bpr + sb * 8);
            let (xs0, xs1) = (vld1q_f32(xs), vld1q_f32(xs.add(4)));
            let blk = |j: usize| vcvtq_f32_s32(vmlaq_s32(vmulq_s32(s16[c][2 * j], sc[2 * j]), s16[c][2 * j + 1], sc[2 * j + 1]));
            macro_rules! sub {
                ($l:literal) => {{
                    let v = vfmaq_f32(acc[c][$l], blk($l), vmulq_laneq_f32::<$l>(dv, xs0));
                    acc[c][$l] = vfmaq_f32(v, blk($l + 4), vmulq_laneq_f32::<$l>(dv, xs1));
                }};
            }
            sub!(0);
            sub!(1);
            sub!(2);
            sub!(3);
        }
    }
    let mut out = [vdupq_n_f32(0.0); TC];
    for c in 0..TC {
        out[c] = vaddq_f32(vaddq_f32(acc[c][0], acc[c][1]), vaddq_f32(acc[c][2], acc[c][3]));
    }
    out
}

// ---- K-quant kernels on activations quantized per 256 (`quant::Q8K`) ---------------------------
//
// With one f32 activation scale per super-block and integer scales per 32, every product, scale,
// sub-scale and min of a super-block adds up in integers (exact), and each row takes two f32
// operations per super-block: `acc += sumi * (d * dx)`, `acc -= summ * (dmin * dx)`, `sumi` the
// products weighted by the weights' sub-block scales and the activations' integer scales, `summ`
// the mins times the activations' scaled sums (`Q8K::bs`). The plain and the interleaved
// kernels form the same integers and the same f32 operations in the same order, so they give the
// same bits, whatever the token count.
//
// Q6_K and Q3_K keep the per-32 activations: their kernels on these (measured in round 9 of the
// CPU decode notes) were no faster, as their per-32 float steps cost about what the integer
// scales cost them (int8 scales per 16, and sums that need two i32 accumulators).

/// Whether `kind` has kernels on [`crate::quant::Q8K`] activations.
pub(super) fn q8k_kind(kind: GgmlType) -> bool {
    matches!(kind, GgmlType::Q2_K | GgmlType::Q4_K | GgmlType::Q5_K)
}

/// The integer scales of super-block `sb` of token `c` (sub-blocks of 32 0..4, 4..8).
#[inline(always)]
unsafe fn ksub(x: &Act, c: usize, sb: usize) -> (int32x4_t, int32x4_t) {
    let p = x.ks.add(c * x.n_in / 32 + sb * 8);
    (vld1q_s32(p), vld1q_s32(p.add(4)))
}

/// The scaled sums of each 32 activations of super-block `sb` of token `c` (0..4, 4..8).
#[inline(always)]
unsafe fn kbs32(x: &Act, c: usize, sb: usize) -> (int32x4_t, int32x4_t) {
    let p = x.kbs.add(c * x.n_in / 16 + sb * 16);
    (
        vpaddq_s32(vld1q_s32(p), vld1q_s32(p.add(4))),
        vpaddq_s32(vld1q_s32(p.add(8)), vld1q_s32(p.add(12))),
    )
}

/// The integer scales of super-block `sb` of token `c` for its sub-blocks of 16, 4 at a time.
#[inline(always)]
unsafe fn ksub16(x: &Act, c: usize, sb: usize) -> [int32x4_t; 4] {
    let (s0, s1) = ksub(x, c, sb);
    [vzip1q_s32(s0, s0), vzip2q_s32(s0, s0), vzip1q_s32(s1, s1), vzip2q_s32(s1, s1)]
}

/// The f32 scale of super-block `sb` of token `c`.
#[inline(always)]
unsafe fn kscale(x: &Act, c: usize, sb: usize) -> f32 {
    *x.kd.add(c * x.n_in / QK_K + sb)
}

/// Q4_K (and Q5_K with `FIVE`) on [`crate::quant::Q8K`] activations.
unsafe fn q4k_k<const TC: usize, const FIVE: bool>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let tsz = if FIVE { 176 } else { 144 };
    let m4 = vdupq_n_u8(0x0F);
    let mut acc = [0f32; TC];
    for sb in 0..n_in / QK_K {
        let b = row.add(sb * tsz);
        let (d, dmin) = (f16p(b), f16p(b.add(2)));
        const KM1: u32 = 0x3f3f_3f3f;
        const KM2: u32 = 0x0f0f_0f0f;
        const KM3: u32 = 0x0303_0303;
        let u = [rd32(b.add(4)), rd32(b.add(8)), rd32(b.add(12))];
        let sc = (u[0] & KM1) as u64 | ((((u[2] & KM2) | (((u[0] >> 6) & KM3) << 4)) as u64) << 32);
        let mn = (u[1] & KM1) as u64 | (((((u[2] >> 4) & KM2) | (((u[1] >> 6) & KM3) << 4)) as u64) << 32);
        let wide = |v: u64| {
            let w = vmovl_u8(vcreate_u8(v));
            (
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w))),
                vreinterpretq_s32_u32(vmovl_high_u16(w)),
            )
        };
        let ((sc0, sc1), (mn0, mn1)) = (wide(sc), wide(mn));
        let qs = b.add(if FIVE { 48 } else { 16 });
        let mut w = [vdupq_n_s8(0); 16];
        let (h0, h1) = if FIVE {
            (vld1q_u8(b.add(16)), vld1q_u8(b.add(32)))
        } else {
            (vdupq_n_u8(0), vdupq_n_u8(0))
        };
        let b16 = vdupq_n_u8(16);
        for j in 0..4 {
            let v0 = vld1q_u8(qs.add(32 * j));
            let v1 = vld1q_u8(qs.add(32 * j + 16));
            let mut q = [vandq_u8(v0, m4), vandq_u8(v1, m4), vshrq_n_u8::<4>(v0), vshrq_n_u8::<4>(v1)];
            if FIVE {
                let (u1, u2) = (vdupq_n_u8(1 << (2 * j)), vdupq_n_u8(2 << (2 * j)));
                q[0] = vorrq_u8(q[0], vandq_u8(vtstq_u8(h0, u1), b16));
                q[1] = vorrq_u8(q[1], vandq_u8(vtstq_u8(h1, u1), b16));
                q[2] = vorrq_u8(q[2], vandq_u8(vtstq_u8(h0, u2), b16));
                q[3] = vorrq_u8(q[3], vandq_u8(vtstq_u8(h1, u2), b16));
            }
            for i in 0..4 {
                w[4 * j + i] = vreinterpretq_s8_u8(q[i]);
            }
        }
        for c in 0..TC {
            let (a0, a1) = sums32(&w, x.kq.add(c * n_in + sb * QK_K));
            let (s0, s1) = ksub(x, c, sb);
            let si = vaddvq_s32(vmlaq_s32(vmulq_s32(a0, vmulq_s32(sc0, s0)), a1, vmulq_s32(sc1, s1)));
            let (b0, b1) = kbs32(x, c, sb);
            let sm = vaddvq_s32(vmlaq_s32(vmulq_s32(mn0, b0), mn1, b1));
            let dx = kscale(x, c, sb);
            acc[c] = (si as f32).mul_add(d * dx, acc[c]);
            acc[c] = (-(sm as f32)).mul_add(dmin * dx, acc[c]);
        }
    }
    acc
}

/// Q2_K on [`crate::quant::Q8K`] activations.
unsafe fn q2k_k<const TC: usize>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let m3 = vdupq_n_u8(3);
    let mut acc = [0f32; TC];
    for sb in 0..n_in / QK_K {
        let b = row.add(sb * 84);
        let (d, dmin) = (f16p(b.add(80)), f16p(b.add(82)));
        let sm = vld1q_u8(b);
        let sc = s8x16_to_s32(vreinterpretq_s8_u8(vandq_u8(sm, vdupq_n_u8(0x0F))));
        let mn = s8x16_to_s32(vreinterpretq_s8_u8(vshrq_n_u8::<4>(sm)));
        let mut w = [vdupq_n_s8(0); 16];
        for n in 0..2 {
            let (q0, q1) = (vld1q_u8(b.add(16 + 32 * n)), vld1q_u8(b.add(32 + 32 * n)));
            for j in 0..4 {
                let sh = vdupq_n_s8(-2 * j as i8);
                w[8 * n + 2 * j] = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q0, sh), m3));
                w[8 * n + 2 * j + 1] = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q1, sh), m3));
            }
        }
        for c in 0..TC {
            let s16 = ksub16(x, c, sb);
            let scs: [int32x4_t; 4] = std::array::from_fn(|g| vmulq_s32(sc[g], s16[g]));
            let (a0, a1) = sums16(&w, &scs, x.kq.add(c * n_in + sb * QK_K));
            let si = vaddvq_s32(vaddq_s32(a0, a1));
            let kb = x.kbs.add(c * n_in / 16 + sb * 16);
            let mut s = vmulq_s32(mn[0], vld1q_s32(kb));
            for g in 1..4 {
                s = vmlaq_s32(s, mn[g], vld1q_s32(kb.add(4 * g)));
            }
            let dx = kscale(x, c, sb);
            acc[c] = (si as f32).mul_add(d * dx, acc[c]);
            acc[c] = (-(vaddvq_s32(s) as f32)).mul_add(dmin * dx, acc[c]);
        }
    }
    acc
}

/// The kernel of a [`q8k_kind`] on [`crate::quant::Q8K`] activations.
fn k_kern<const TC: usize>(kind: GgmlType) -> RowFn<TC> {
    match kind {
        GgmlType::Q4_K => q4k_k::<TC, false>,
        GgmlType::Q5_K => q4k_k::<TC, true>,
        GgmlType::Q2_K => q2k_k::<TC>,
        k => unreachable!("no Q8_K kernel for {k}"),
    }
}

/// Sub-block `k` of the four rows of an interleaved Q4_K group's scales: bytes `k`, `4 + k`,
/// `8 + k` and `12 + k` of `v` (row `i`'s four bytes in lane `i`), in that order.
#[inline(always)]
unsafe fn q4k_by_k(v: uint32x4_t) -> uint8x16_t {
    const BY_K: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
    vqtbl1q_u8(vreinterpretq_u8_u32(v), vld1q_u8(BY_K.as_ptr()))
}

/// Four sub-blocks' bytes (`q4k_by_k`) as int32 lanes, one vector per sub-block.
#[inline(always)]
unsafe fn q4k_by_k_s32(v: uint32x4_t) -> [int32x4_t; 4] {
    let t = q4k_by_k(v);
    let (lo, hi) = (vmovl_u8(vget_low_u8(t)), vmovl_high_u8(t));
    [
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo))),
        vreinterpretq_s32_u32(vmovl_high_u16(lo)),
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi))),
        vreinterpretq_s32_u32(vmovl_high_u16(hi)),
    ]
}

/// The interleaved Q4_K group's 6-bit scales and mins of the four rows (`get_scale_min_k4`):
/// lane `i` of each vector holds row `i`'s bytes of sub-blocks 0..4 / 4..8 of the scales, then
/// of the mins.
#[inline(always)]
unsafe fn q4k_x4_scales(b: *const u8) -> [uint32x4_t; 4] {
    let km1 = vdupq_n_u32(0x3f3f_3f3f);
    let (km2, km3) = (vdupq_n_u32(0x0f0f_0f0f), vdupq_n_u32(0x0303_0303));
    let u = [
        vld1q_u32(b.add(16) as *const u32),
        vld1q_u32(b.add(32) as *const u32),
        vld1q_u32(b.add(48) as *const u32),
    ];
    let hi6 = |a: uint32x4_t, c: uint32x4_t| vorrq_u32(a, vshlq_n_u32::<4>(vandq_u32(vshrq_n_u32::<6>(c), km3)));
    [
        vandq_u32(u[0], km1),
        hi6(vandq_u32(u[2], km2), u[0]),
        vandq_u32(u[1], km1),
        hi6(vandq_u32(vshrq_n_u32::<4>(u[2]), km2), u[1]),
    ]
}

/// Four rows' dot products of 32 values: pieces `w` (`w[j]`, j < 4: values `4j..4j + 4` of each
/// row; `w[4 + j]`: values `16 + 4j..`) against `xq[0..32]`, one row per lane.
#[inline(always)]
unsafe fn dot_x4(w: &[int8x16_t; 8], xq: *const i8) -> int32x4_t {
    let zero = vdupq_n_s32(0);
    let (a0, a1) = (vld1q_s8(xq), vld1q_s8(xq.add(16)));
    let s = sdot_lane::<3>(
        sdot_lane::<2>(sdot_lane::<1>(sdot_lane::<0>(zero, w[0], a0), w[1], a0), w[2], a0),
        w[3],
        a0,
    );
    sdot_lane::<3>(
        sdot_lane::<2>(sdot_lane::<1>(sdot_lane::<0>(s, w[4], a1), w[5], a1), w[6], a1),
        w[7],
        a1,
    )
}

/// Q4_K interleaved four rows at a time on [`crate::quant::Q8K`] activations: `q4k_k`'s
/// integers and f32 operations for each row, lane `i` of each result row `i`.
#[inline(always)]
unsafe fn q4k_x4k<const TC: usize>(g: *const u8, n_in: usize, x: &Act) -> [float32x4_t; TC] {
    let zero = vdupq_n_s32(0);
    let m4 = vdupq_n_u8(0x0F);
    let mut acc = [vdupq_n_f32(0.0); TC];
    for sb in 0..n_in / QK_K {
        let b = g.add(sb * 576);
        let raw = q4k_x4_scales(b);
        // The scales and mins of sub-block `k` (four rows) as int32 lanes.
        let (s_lo, s_hi) = (q4k_by_k_s32(raw[0]), q4k_by_k_s32(raw[1]));
        let (n_lo, n_hi) = (q4k_by_k_s32(raw[2]), q4k_by_k_s32(raw[3]));
        let sc = [s_lo[0], s_lo[1], s_lo[2], s_lo[3], s_hi[0], s_hi[1], s_hi[2], s_hi[3]];
        let mut si = [[zero; 2]; TC];
        let qs = b.add(64);
        for j in 0..4 {
            // Chunk `j`'s pieces: low nibbles are sub-block `2j`, high ones `2j + 1`.
            let mut lo = [vdupq_n_s8(0); 8];
            let mut hi = [vdupq_n_s8(0); 8];
            for p in 0..8 {
                let v = vld1q_u8(qs.add(16 * (8 * j + p)));
                lo[p] = vreinterpretq_s8_u8(vandq_u8(v, m4));
                hi[p] = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
            }
            for c in 0..TC {
                let xq = x.kq.add(c * n_in + sb * QK_K + 64 * j);
                let ks = x.ks.add(c * n_in / 32 + sb * 8 + 2 * j);
                si[c][0] = vmlaq_s32(si[c][0], dot_x4(&lo, xq), vmulq_n_s32(sc[2 * j], *ks));
                si[c][1] = vmlaq_s32(si[c][1], dot_x4(&hi, xq.add(32)), vmulq_n_s32(sc[2 * j + 1], *ks.add(1)));
            }
        }
        let (d, dmin) = (f16x4_to_f32(b as *const u16), f16x4_to_f32(b.add(8) as *const u16));
        for c in 0..TC {
            let (b0, b1) = kbs32(x, c, sb);
            let mut sm = vmulq_laneq_s32::<0>(n_lo[0], b0);
            sm = vmlaq_laneq_s32::<1>(sm, n_lo[1], b0);
            sm = vmlaq_laneq_s32::<2>(sm, n_lo[2], b0);
            sm = vmlaq_laneq_s32::<3>(sm, n_lo[3], b0);
            sm = vmlaq_laneq_s32::<0>(sm, n_hi[0], b1);
            sm = vmlaq_laneq_s32::<1>(sm, n_hi[1], b1);
            sm = vmlaq_laneq_s32::<2>(sm, n_hi[2], b1);
            sm = vmlaq_laneq_s32::<3>(sm, n_hi[3], b1);
            let dx = kscale(x, c, sb);
            let v = vfmaq_f32(acc[c], vcvtq_f32_s32(vaddq_s32(si[c][0], si[c][1])), vmulq_n_f32(d, dx));
            acc[c] = vfmsq_f32(v, vcvtq_f32_s32(sm), vmulq_n_f32(dmin, dx));
        }
    }
    acc
}

/// The `x4_group` tag of the Q4_K kernel on [`crate::quant::Q8K`] activations.
const Q4KK: u8 = 252;

/// The interleaved kernel of tag `K` for TC (1 or 2) tokens.
#[inline(always)]
unsafe fn x4_kernel<const TC: usize, const K: u8>(g: *const u8, n_in: usize, x: &Act) -> [float32x4_t; TC] {
    match K {
        Q4K => q4k_x4::<TC>(g, n_in, x),
        Q6KX => q6k_x4::<TC>(g, n_in, x),
        Q4KK => q4k_x4k::<TC>(g, n_in, x),
        _ => x4_kern::<TC, K>(g, n_in, x),
    }
}

/// [`x4_kern`], [`q4k_x4`] (`K == Q4K`) and the others for any TC (1..=8) in kernels of at most
/// two tokens (more would keep too many accumulators for the registers; the group's weights stay
/// in L1 between them).
#[inline(always)]
unsafe fn x4_group<const TC: usize, const K: u8>(g: *const u8, n_in: usize, x: &Act) -> [float32x4_t; TC] {
    let mut out = [vdupq_n_f32(0.0); TC];
    let mut c = 0;
    while c < TC {
        let xc = act_at(x, c);
        if TC - c >= 2 {
            let o = x4_kernel::<2, K>(g, n_in, &xc);
            out[c] = o[0];
            out[c + 1] = o[1];
            c += 2;
        } else {
            out[c] = x4_kernel::<1, K>(g, n_in, &xc)[0];
            c += 1;
        }
    }
    out
}

/// Rows `r0..r1` of an interleaved matrix (`m.x4`) against TC activation rows into
/// `y[c * ys + r]`: whole groups of four rows at a time (a group cut by `r0` or `r1` is computed
/// whole and only its rows in range stored).
unsafe fn rows_x4<const TC: usize>(m: &GgufMatrix, x: &Act, y: *mut f32, ys: usize, r0: usize, r1: usize) {
    let gb = 4 * m.row_bytes();
    let base = m.data.as_ptr();
    let f = match m.kind {
        GgmlType::Q8_0 => x4_group::<TC, Q8>,
        GgmlType::Q4_0 => x4_group::<TC, Q4_0>,
        GgmlType::Q4_K if x.k => x4_group::<TC, Q4KK>,
        GgmlType::Q4_K => x4_group::<TC, Q4K>,
        GgmlType::Q6_K => x4_group::<TC, Q6KX>,
        k => unreachable!("no interleaved kernel for {k}"),
    };
    let mut g = r0 / 4;
    while 4 * g < r1 {
        let o = f(base.add(g * gb), m.n_in, x);
        if 4 * g >= r0 && 4 * g + 4 <= r1 {
            for c in 0..TC {
                vst1q_f32(y.add(c * ys + 4 * g), o[c]);
            }
        } else {
            for c in 0..TC {
                let mut v = [0f32; 4];
                vst1q_f32(v.as_mut_ptr(), o[c]);
                for (i, &vi) in v.iter().enumerate() {
                    let r = 4 * g + i;
                    if (r0..r1).contains(&r) {
                        *y.add(c * ys + r) = vi;
                    }
                }
            }
        }
        g += 1;
    }
}

/// Rows `r..r + 4` (one group) of an interleaved Q8_0 or Q4_0 matrix straight into the int8
/// GEMM's four-row tile (`QMat::fill_tile4`): the tile's layout is the Q8_0 groups' values, so
/// Q8_0 copies them and Q4_0 splits each piece's nibbles (less 8, as `b32_i8` centers them).
pub(super) unsafe fn fill_tile4_x4(m: &GgufMatrix, r: usize, q: *mut i8, ds: *mut f32) -> bool {
    let g = m.data.as_ptr().add(r / 4 * 4 * m.row_bytes());
    let nb = m.n_in / 32;
    match m.kind {
        GgmlType::Q8_0 => {
            for b in 0..nb {
                let p = g.add(b * 136);
                for j in 0..8 {
                    vst1q_s8(q.add(b * 128 + 16 * j), vld1q_s8(p.add(8 + 16 * j) as *const i8));
                }
                vst1q_f32(ds.add(4 * b), f16x4_to_f32(p as *const u16));
            }
        }
        GgmlType::Q4_0 => {
            let (m4, eight) = (vdupq_n_u8(0x0F), vdupq_n_s8(8));
            for b in 0..nb {
                let p = g.add(b * 72);
                for j in 0..4 {
                    let v = vld1q_u8(p.add(8 + 16 * j));
                    let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), eight);
                    let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), eight);
                    vst1q_s8(q.add(b * 128 + 16 * j), lo);
                    vst1q_s8(q.add(b * 128 + 64 + 16 * j), hi);
                }
                vst1q_f32(ds.add(4 * b), f16x4_to_f32(p as *const u16));
            }
        }
        _ => return false,
    }
    true
}

/// Rows `r..r + 4` (one group) of an interleaved Q8_0 or Q4_0 matrix as pairs `p0`, `p0 + 1` of
/// the SMMLA GEMM's tile (`QMat::fill_pairs4`): two 4-byte pieces of a row are its 8 values of
/// an SMMLA operand, so zipping consecutive pieces gives rows 0 and 1, then 2 and 3.
pub(super) unsafe fn fill_pairs4_x4(m: &GgufMatrix, r: usize, q: *mut i8, ds: *mut f32, p0: usize, mrp: usize) -> bool {
    let g = m.data.as_ptr().add(r / 4 * 4 * m.row_bytes());
    let (bsz, nib) = match m.kind {
        GgmlType::Q8_0 => (136, false),
        GgmlType::Q4_0 => (72, true),
        _ => return false,
    };
    let (m4, eight) = (vdupq_n_u8(0x0F), vdupq_n_s8(8));
    for b in 0..m.n_in / 32 {
        let p = g.add(b * bsz);
        // Values `4j..4j + 4` of the four rows.
        let mut e = [vdupq_n_u32(0); 8];
        if nib {
            for j in 0..4 {
                let v = vld1q_u8(p.add(8 + 16 * j));
                e[j] = vreinterpretq_u32_s8(vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), eight));
                e[4 + j] = vreinterpretq_u32_s8(vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), eight));
            }
        } else {
            for (j, v) in e.iter_mut().enumerate() {
                *v = vld1q_u32(p.add(8 + 16 * j) as *const u32);
            }
        }
        for c in 0..4 {
            let i = 4 * b + c;
            vst1q_u32(q.add((i * mrp + p0) * 16) as *mut u32, vzip1q_u32(e[2 * c], e[2 * c + 1]));
            vst1q_u32(q.add((i * mrp + p0 + 1) * 16) as *mut u32, vzip2q_u32(e[2 * c], e[2 * c + 1]));
        }
        let d = f16x4_to_f32(p as *const u16);
        vst1q_f32(ds.add((b * mrp + p0) * 4), vzip1q_f32(d, d));
        vst1q_f32(ds.add((b * mrp + p0 + 1) * 4), vzip2q_f32(d, d));
    }
    true
}

/// Row `r` of an interleaved Q8_0 matrix unpacked for the int8 prefill GEMM (as
/// [`unpack_row_i8`] unpacks a plain row).
pub(super) unsafe fn unpack_row_i8_x4(m: &GgufMatrix, r: usize, q: *mut i8, ds: *mut f32, mn: *mut f32, sc: *mut i32) {
    if m.kind != GgmlType::Q8_0 {
        // One block at a time back in the file's layout.
        let (rb, ts, bs) = (m.row_bytes(), m.kind.type_size(), m.kind.block_size());
        let mut blk = [0u8; 256];
        for b in 0..m.n_in / bs {
            super::repack::plain_block(m.kind, &m.data, rb, r, b, &mut blk[..ts]);
            let (q, ds, mn, sc) = (q.add(bs * b), ds.add(bs / 32 * b), mn.add(bs / 32 * b), sc.add(bs / 16 * b));
            unpack_row_i8(m.kind, blk.as_ptr(), bs, q, ds, mn, sc);
        }
        return;
    }
    let g = m.data.as_ptr().add(r / 4 * 4 * m.row_bytes());
    let i = r % 4;
    for b in 0..m.n_in / 32 {
        let p = g.add(b * 136);
        // The 32 four-byte pieces of the block: piece `4j + i` is values `4j..4j + 4` of row `i`.
        let lo = vld4q_u32(p.add(8) as *const u32);
        let hi = vld4q_u32(p.add(72) as *const u32);
        let (vlo, vhi) = match i {
            0 => (lo.0, hi.0),
            1 => (lo.1, hi.1),
            2 => (lo.2, hi.2),
            _ => (lo.3, hi.3),
        };
        vst1q_s8(q.add(32 * b), vreinterpretq_s8_u32(vlo));
        vst1q_s8(q.add(32 * b + 16), vreinterpretq_s8_u32(vhi));
        *ds.add(b) = f16p(p.add(2 * i));
    }
}

/// The blocks-of-32 nibble formats (`q32` kernel parameter).
const Q4_0: u8 = 0;
const IQ4_NL: u8 = 1;
const Q4_1: u8 = 2;
const Q5_0: u8 = 3;
const Q5_1: u8 = 4;
const MXFP4: u8 = 5;

/// The scale of the block at `b`: f16, or MXFP4's E8M0 byte (halved, its values doubled).
#[inline(always)]
unsafe fn scale32<const K: u8>(b: *const u8) -> f32 {
    if K == MXFP4 { super::e8m0_half(*b) } else { f16p(b) }
}

/// Q4_0, IQ4_NL, Q4_1, Q5_0, Q5_1 and MXFP4 (`K`): blocks of 32 with an f16 scale (and an f16 min
/// for Q4_1 / Q5_1; an E8M0 scale byte for MXFP4), nibbles plus, for Q5, a fifth bit per element in
/// a 32-bit word; four blocks per step. IQ4_NL and MXFP4 map their nibbles through a table.
unsafe fn q32<const TC: usize, const K: u8>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let (tsz, qs) = match K {
        Q4_0 | IQ4_NL => (18, 2),
        MXFP4 => (17, 1),
        Q4_1 => (20, 4),
        Q5_0 => (22, 6),
        _ => (24, 8),
    };
    let table = K == IQ4_NL || K == MXFP4;
    let has_min = K == Q4_1 || K == Q5_1;
    // Nibbles are centered by `qoff` (Q4_1 / Q5_1 like Q4_0 / Q5_0; the min absorbs `d * qoff`).
    let qoff = if K == Q5_0 || K == Q5_1 { 16.0 } else { 8.0 };
    let bpr = n_in / 32;
    let zero = vdupq_n_s32(0);
    let m4 = vdupq_n_u8(0x0F);
    let kv = vld1q_s8(if K == MXFP4 {
        KVALUES_MXFP4.as_ptr()
    } else {
        KVALUES_IQ4NL.as_ptr()
    });
    // The 32 weights of the block at `b` as two vectors (elements 0..16, 16..32).
    let unpack = |b: *const u8| -> (int8x16_t, int8x16_t) {
        let v = vld1q_u8(b.add(qs));
        let (mut lo, mut hi) = (vandq_u8(v, m4), vshrq_n_u8::<4>(v));
        if K == Q5_0 || K == Q5_1 {
            let (h0, h1) = sign_masks(rd32(b.add(qs - 4)));
            let b16 = vdupq_n_u8(16);
            lo = vorrq_u8(lo, vandq_u8(h0, b16));
            hi = vorrq_u8(hi, vandq_u8(h1, b16));
        }
        let (lo, hi) = (vreinterpretq_s8_u8(lo), vreinterpretq_s8_u8(hi));
        if table {
            (vqtbl1q_s8(kv, vreinterpretq_u8_s8(lo)), vqtbl1q_s8(kv, vreinterpretq_u8_s8(hi)))
        } else {
            // The offset (8 or 16) is subtracted from the integer sums.
            (lo, hi)
        }
    };
    let mut acc = [vdupq_n_f32(0.0); TC];
    let mut blk = 0;
    while blk + 4 <= bpr {
        let b = row.add(blk * tsz);
        let (mut dh, mut mh) = ([0u16; 4], [0u16; 4]);
        let mut w = [vdupq_n_s8(0); 8];
        for k in 0..4 {
            if K != MXFP4 {
                dh[k] = rd16(b.add(tsz * k));
                mh[k] = rd16(b.add(tsz * k + 2));
            }
            (w[2 * k], w[2 * k + 1]) = unpack(b.add(tsz * k));
        }
        let dv = if K == MXFP4 {
            let d = [
                scale32::<K>(b),
                scale32::<K>(b.add(tsz)),
                scale32::<K>(b.add(2 * tsz)),
                scale32::<K>(b.add(3 * tsz)),
            ];
            vld1q_f32(d.as_ptr())
        } else {
            f16x4_to_f32(dh.as_ptr())
        };
        let mv = if has_min {
            vaddq_f32(f16x4_to_f32(mh.as_ptr()), vmulq_n_f32(dv, qoff))
        } else {
            vdupq_n_f32(0.0)
        };
        for c in 0..TC {
            let xq = x.q.add(c * n_in + blk * 32);
            let mut a = [zero; 4];
            for k in 0..4 {
                a[k] = sdot(
                    sdot(zero, w[2 * k], vld1q_s8(xq.add(32 * k))),
                    w[2 * k + 1],
                    vld1q_s8(xq.add(32 * k + 16)),
                );
            }
            let mut s = vpaddq_s32(vpaddq_s32(a[0], a[1]), vpaddq_s32(a[2], a[3]));
            if !table {
                let xi = vld1q_s32(x.i.add(c * bpr + blk));
                s = vsubq_s32(s, if qoff == 16.0 { vshlq_n_s32::<4>(xi) } else { vshlq_n_s32::<3>(xi) });
            }
            let mut v = vfmaq_f32(acc[c], vcvtq_f32_s32(s), vmulq_f32(dv, vld1q_f32(x.s.add(c * bpr + blk))));
            if has_min {
                v = vfmaq_f32(v, mv, vld1q_f32(x.f.add(c * bpr + blk)));
            }
            acc[c] = v;
        }
        blk += 4;
    }
    let mut out = finish(acc);
    while blk < bpr {
        let b = row.add(blk * tsz);
        let (w0, w1) = unpack(b);
        let d = scale32::<K>(b);
        for c in 0..TC {
            let xq = x.q.add(c * n_in + blk * 32);
            let mut s = vaddvq_s32(sdot(sdot(zero, w0, vld1q_s8(xq)), w1, vld1q_s8(xq.add(16))));
            if !table {
                s -= qoff as i32 * *x.i.add(c * bpr + blk);
            }
            out[c] += s as f32 * d * *x.s.add(c * bpr + blk);
            if has_min {
                out[c] += (f16p(b.add(2)) + d * qoff) * *x.f.add(c * bpr + blk);
            }
        }
        blk += 1;
    }
    out
}

// ---- Multi-token decode (speculative verification: 2..=16 tokens) ------------------------------
// Each weight super-block is unpacked once for all tokens of the pass (the token-pair driver
// unpacks it once per pair). Both variants reproduce the one-token kernels above operation by
// operation in f32, and the integer sums are exact whatever the instruction, so every row's
// result is bit-identical to `rows::<1>` for every token count.

/// Most tokens a multi-token pass takes.
pub(crate) const MULTI_MAX: usize = 16;

/// SMMLA: `acc` (2x2 int32, row-major) += `a` (2x8 int8 rows) times `b` (2x8 int8 rows) transposed.
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

/// How a format corrects the scaled integer sums.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Corr {
    None,
    /// Q4_K / Q5_K: minus per-32 mins times the block sums (the weights are centered by 8 / 16 in
    /// int8, which gives the one-token kernel's integer sums minus `off * sum(xq)` directly).
    Min32,
    /// Q2_K: minus per-16 mins times the half-block sums.
    Q2K,
}

/// The scales of one super-block of one row: integer scales per 16 (`SC16` formats, as four
/// vectors of four), f32 multipliers per 32 (lanes = blocks 0..4 / 4..8) and min terms (Q4_K /
/// Q5_K: per 32 in `mn[0..2]`; Q2_K: per 16 in `mn[0..4]`).
#[derive(Clone, Copy)]
struct Meta {
    sc: [int32x4_t; 4],
    ds: [float32x4_t; 2],
    mn: [float32x4_t; 4],
}

impl Meta {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self {
            sc: [vdupq_n_s32(0); 4],
            ds: [vdupq_n_f32(0.0); 2],
            mn: [vdupq_n_f32(0.0); 4],
        }
    }
}

/// One super-block of one row unpacked: 16 vectors of 16 int8 weights (vector k = elements
/// 16k..16k+16) and its scales.
#[derive(Clone, Copy)]
struct Unpacked {
    w: [int8x16_t; 16],
    m: Meta,
}

/// The two halves of `w` (vectors 0..8 and 8..16).
#[inline(always)]
fn halves(w: &mut [int8x16_t; 16]) -> (&mut [int8x16_t; 8], &mut [int8x16_t; 8]) {
    let (a, b) = w.split_at_mut(8);
    (a.try_into().unwrap(), b.try_into().unwrap())
}

/// A super-block format for the multi-token kernels. Every method computes exactly the values of
/// the format's one-token kernel: `unpack` all of them, `meta` the scales alone and `half` the
/// weight vectors `8 H .. 8 H + 8` alone.
trait Multi {
    const TSZ: usize;
    /// Integer scales per 16 (`sums16`); otherwise per-32 sums (`sums32`) with f32 scales.
    const SC16: bool;
    const CORR: Corr;
    unsafe fn meta(b: *const u8) -> Meta;
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]);
    #[inline(always)]
    unsafe fn unpack(b: *const u8, u: &mut Unpacked) {
        u.m = Self::meta(b);
        let (lo, hi) = halves(&mut u.w);
        Self::half::<0>(b, lo);
        Self::half::<1>(b, hi);
    }
}

struct MSb16<U>(std::marker::PhantomData<U>);
impl<U: Sb16> Multi for MSb16<U> {
    const TSZ: usize = U::TSZ;
    const SC16: bool = true;
    const CORR: Corr = Corr::None;
    #[inline(always)]
    unsafe fn meta(b: *const u8) -> Meta {
        let mut scb = [0i8; 16];
        let d = vdupq_n_f32(U::scales(b, &mut scb));
        let mut m = Meta::zero();
        m.sc = s8x16_to_s32(vld1q_s8(scb.as_ptr()));
        m.ds = [d, d];
        m
    }
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        U::half::<H>(b, w)
    }
    #[inline(always)]
    unsafe fn unpack(b: *const u8, u: &mut Unpacked) {
        let mut scb = [0i8; 16];
        let d = vdupq_n_f32(U::unpack(b, &mut u.w, &mut scb));
        u.m.sc = s8x16_to_s32(vld1q_s8(scb.as_ptr()));
        u.m.ds = [d, d];
    }
}

struct MQ4K<const FIVE: bool>;
impl<const FIVE: bool> Multi for MQ4K<FIVE> {
    const TSZ: usize = if FIVE { 176 } else { 144 };
    const SC16: bool = false;
    const CORR: Corr = Corr::Min32;
    #[inline(always)]
    unsafe fn meta(b: *const u8) -> Meta {
        let (d, dmin) = (f16p(b), f16p(b.add(2)));
        const KM1: u32 = 0x3f3f_3f3f;
        const KM2: u32 = 0x0f0f_0f0f;
        const KM3: u32 = 0x0303_0303;
        let w = [rd32(b.add(4)), rd32(b.add(8)), rd32(b.add(12))];
        let sc = (w[0] & KM1) as u64 | ((((w[2] & KM2) | (((w[0] >> 6) & KM3) << 4)) as u64) << 32);
        let mn = (w[1] & KM1) as u64 | (((((w[2] >> 4) & KM2) | (((w[1] >> 6) & KM3) << 4)) as u64) << 32);
        let (s0, s1) = u8x8_to_f32(vcreate_u8(sc));
        let (n0, n1) = u8x8_to_f32(vcreate_u8(mn));
        let (ds0, ds1) = (vmulq_n_f32(s0, d), vmulq_n_f32(s1, d));
        let off = if FIVE { 16.0 } else { 8.0 };
        let mut m = Meta::zero();
        m.ds = [ds0, ds1];
        m.mn[0] = vsubq_f32(vmulq_n_f32(n0, dmin), vmulq_n_f32(ds0, off));
        m.mn[1] = vsubq_f32(vmulq_n_f32(n1, dmin), vmulq_n_f32(ds1, off));
        m
    }
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        let m4 = vdupq_n_u8(0x0F);
        let qs = b.add(if FIVE { 48 } else { 16 });
        let (h0, h1) = if FIVE {
            (vld1q_u8(b.add(16)), vld1q_u8(b.add(32)))
        } else {
            (vdupq_n_u8(0), vdupq_n_u8(0))
        };
        let b16 = vdupq_n_u8(16);
        for jj in 0..2 {
            let j = 2 * H + jj;
            let v0 = vld1q_u8(qs.add(32 * j));
            let v1 = vld1q_u8(qs.add(32 * j + 16));
            let mut q = [vandq_u8(v0, m4), vandq_u8(v1, m4), vshrq_n_u8::<4>(v0), vshrq_n_u8::<4>(v1)];
            if FIVE {
                let (u1, u2) = (vdupq_n_u8(1 << (2 * j)), vdupq_n_u8(2 << (2 * j)));
                q[0] = vorrq_u8(q[0], vandq_u8(vtstq_u8(h0, u1), b16));
                q[1] = vorrq_u8(q[1], vandq_u8(vtstq_u8(h1, u1), b16));
                q[2] = vorrq_u8(q[2], vandq_u8(vtstq_u8(h0, u2), b16));
                q[3] = vorrq_u8(q[3], vandq_u8(vtstq_u8(h1, u2), b16));
            }
            let off = vdupq_n_s8(if FIVE { 16 } else { 8 });
            for i in 0..4 {
                w[4 * jj + i] = vsubq_s8(vreinterpretq_s8_u8(q[i]), off);
            }
        }
    }
}

struct MQ2K;
impl Multi for MQ2K {
    const TSZ: usize = 84;
    const SC16: bool = true;
    const CORR: Corr = Corr::Q2K;
    #[inline(always)]
    unsafe fn meta(b: *const u8) -> Meta {
        let d = vdupq_n_f32(f16p(b.add(80)));
        let dmin = vdupq_n_f32(f16p(b.add(82)));
        let sm = vld1q_u8(b);
        let mut m = Meta::zero();
        m.sc = s8x16_to_s32(vreinterpretq_s8_u8(vandq_u8(sm, vdupq_n_u8(0x0F))));
        let mn = s8x16_to_s32(vreinterpretq_s8_u8(vshrq_n_u8::<4>(sm)));
        let d2 = vaddq_f32(d, d);
        for k in 0..4 {
            m.mn[k] = vsubq_f32(vmulq_f32(vcvtq_f32_s32(mn[k]), dmin), vmulq_f32(vcvtq_f32_s32(m.sc[k]), d2));
        }
        m.ds = [d, d];
        m
    }
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        let m3 = vdupq_n_u8(3);
        let two = vdupq_n_s8(2);
        let (q0, q1) = (vld1q_u8(b.add(16 + 32 * H)), vld1q_u8(b.add(32 + 32 * H)));
        for j in 0..4 {
            let sh = vdupq_n_s8(-2 * j as i8);
            w[2 * j] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q0, sh), m3)), two);
            w[2 * j + 1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q1, sh), m3)), two);
        }
    }
}

struct MIq4Xs;
impl Multi for MIq4Xs {
    const TSZ: usize = 136;
    const SC16: bool = false;
    const CORR: Corr = Corr::None;
    #[inline(always)]
    unsafe fn meta(b: *const u8) -> Meta {
        let d = f16p(b);
        const HSH: [i16; 8] = [4, 2, 0, -2, -4, -6, -8, -10];
        let sl = vcreate_u8(rd32(b.add(4)) as u64);
        let lo = vmovl_u8(vzip1_u8(vand_u8(sl, vdup_n_u8(0x0F)), vshr_n_u8::<4>(sl)));
        let hi = vandq_u16(vshlq_u16(vdupq_n_u16(rd16(b.add(2))), vld1q_s16(HSH.as_ptr())), vdupq_n_u16(0x30));
        let ls = vsubq_s16(vreinterpretq_s16_u16(vorrq_u16(lo, hi)), vdupq_n_s16(32));
        let mut m = Meta::zero();
        m.ds = [
            vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ls))), d),
            vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(ls)), d),
        ];
        m
    }
    #[inline(always)]
    unsafe fn half<const H: usize>(b: *const u8, w: &mut [int8x16_t; 8]) {
        let m4 = vdupq_n_u8(0x0F);
        let kv = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        unroll4!(ib, 4 * H, {
            let v = vld1q_u8(b.add(8 + 16 * ib));
            w[2 * (ib - 4 * H)] = vqtbl1q_s8(kv, vandq_u8(v, m4));
            w[2 * (ib - 4 * H) + 1] = vqtbl1q_s8(kv, vshrq_n_u8::<4>(v));
        });
    }
}

/// One token's contribution of super-block `sb` (unpacked in `u`) added to `acc`, in exactly the
/// operation order of the one-token kernels.
#[inline(always)]
unsafe fn tok_dot<F: Multi>(u: &Unpacked, x: &Act, c: usize, sb: usize, acc: float32x4_t) -> float32x4_t {
    let n_in = x.n_in;
    let bpr = n_in / 32;
    let xq = x.q.add(c * n_in + sb * QK_K);
    let (a0, a1) = if F::SC16 { sums16(&u.w, &u.m.sc, xq) } else { sums32(&u.w, xq) };
    let xb = c * bpr + sb * 8;
    let xs = x.s.add(xb);
    let mut v = vfmaq_f32(acc, vcvtq_f32_s32(a0), vmulq_f32(u.m.ds[0], vld1q_f32(xs)));
    v = vfmaq_f32(v, vcvtq_f32_s32(a1), vmulq_f32(u.m.ds[1], vld1q_f32(xs.add(4))));
    match F::CORR {
        Corr::Min32 => {
            let xf = x.f.add(xb);
            v = vfmsq_f32(v, u.m.mn[0], vld1q_f32(xf));
            v = vfmsq_f32(v, u.m.mn[1], vld1q_f32(xf.add(4)));
        }
        Corr::Q2K => {
            let xh = x.h.add(2 * xb);
            let mut t = vmulq_f32(u.m.mn[0], vld1q_f32(xh));
            t = vfmaq_f32(t, u.m.mn[1], vld1q_f32(xh.add(4)));
            t = vfmaq_f32(t, u.m.mn[2], vld1q_f32(xh.add(8)));
            t = vfmaq_f32(t, u.m.mn[3], vld1q_f32(xh.add(12)));
            v = vsubq_f32(v, t);
        }
        Corr::None => {}
    }
    v
}

/// SDOT variant: rows `r0..r1` one at a time, each super-block unpacked once for all `t` tokens.
unsafe fn multi_dot<F: Multi>(m: &GgufMatrix, x: &Act, t: usize, y: *mut f32, ys: usize, r0: usize, r1: usize) {
    let rb = m.row_bytes();
    let mut acc = [vdupq_n_f32(0.0); MULTI_MAX];
    for r in r0..r1 {
        let row = m.data.as_ptr().add(r * rb);
        acc[..t].fill(vdupq_n_f32(0.0));
        for sb in 0..m.n_in / QK_K {
            let mut u = Unpacked {
                w: [vdupq_n_s8(0); 16],
                m: Meta::zero(),
            };
            F::unpack(row.add(sb * F::TSZ), &mut u);
            for (c, a) in acc[..t].iter_mut().enumerate() {
                *a = tok_dot::<F>(&u, x, c, sb, *a);
            }
        }
        for (c, a) in acc[..t].iter().enumerate() {
            *y.add(c * ys + r) = vaddvq_f32(*a);
        }
    }
}

/// Lanes `[a_k, a_k, b_k, b_k]` for k = 0..16 from per-16 values of two rows (four vectors of four each).
#[inline(always)]
unsafe fn pair16_s32(a: &[int32x4_t; 4], b: &[int32x4_t; 4]) -> [int32x4_t; 16] {
    let mut o = [vdupq_n_s32(0); 16];
    let z = |p: int32x4_t, q: int32x4_t| {
        let (p, q) = (vreinterpretq_s64_s32(p), vreinterpretq_s64_s32(q));
        (vreinterpretq_s32_s64(vzip1q_s64(p, q)), vreinterpretq_s32_s64(vzip2q_s64(p, q)))
    };
    for g in 0..4 {
        let (al, ah) = (vzip1q_s32(a[g], a[g]), vzip2q_s32(a[g], a[g]));
        let (bl, bh) = (vzip1q_s32(b[g], b[g]), vzip2q_s32(b[g], b[g]));
        (o[4 * g], o[4 * g + 1]) = z(al, bl);
        (o[4 * g + 2], o[4 * g + 3]) = z(ah, bh);
    }
    o
}

/// The f32 version of `pair16_s32` for the first `N` vectors of `a` and `b` (into `4 N` of `O`).
#[inline(always)]
unsafe fn pair_f32<const N: usize, const O: usize>(a: &[float32x4_t], b: &[float32x4_t]) -> [float32x4_t; O] {
    let mut o = [vdupq_n_f32(0.0); O];
    let z = |p: float32x4_t, q: float32x4_t| {
        let (p, q) = (vreinterpretq_f64_f32(p), vreinterpretq_f64_f32(q));
        (vreinterpretq_f32_f64(vzip1q_f64(p, q)), vreinterpretq_f32_f64(vzip2q_f64(p, q)))
    };
    for g in 0..N {
        let (al, ah) = (vzip1q_f32(a[g], a[g]), vzip2q_f32(a[g], a[g]));
        let (bl, bh) = (vzip1q_f32(b[g], b[g]), vzip2q_f32(b[g], b[g]));
        (o[4 * g], o[4 * g + 1]) = z(al, bl);
        (o[4 * g + 2], o[4 * g + 3]) = z(ah, bh);
    }
    o
}

/// Half `H` (blocks `4 H .. 4 H + 4`) of super-block `sb` of two rows (`a`, `b`) against every
/// token pair: the halves are unpacked, interleaved per 8 values (`[a x8, b x8]`) and kept in
/// registers while the token pairs stream past. Formats with integer scales per 16 only.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn mm_half<F: Multi, const H: usize>(
    a: *const u8,
    b: *const u8,
    sc: &[int32x4_t; 16],
    ds: &[float32x4_t; 8],
    p: &PairAct,
    np: usize,
    sb: usize,
    n_in: usize,
    acc: &mut [[float32x4_t; 4]; MULTI_MAX / 2],
) {
    let (mut wa, mut wb) = ([vdupq_n_s8(0); 8], [vdupq_n_s8(0); 8]);
    F::half::<H>(a, &mut wa);
    F::half::<H>(b, &mut wb);
    let mut w = [vdupq_n_s8(0); 16];
    for v in 0..8 {
        let (x, y) = (vreinterpretq_s64_s8(wa[v]), vreinterpretq_s64_s8(wb[v]));
        w[2 * v] = vreinterpretq_s8_s64(vzip1q_s64(x, y));
        w[2 * v + 1] = vreinterpretq_s8_s64(vzip2q_s64(x, y));
    }
    let bpr = n_in / 32;
    let zero = vdupq_n_s32(0);
    for (pi, acc) in acc[..np].iter_mut().enumerate() {
        let xq = p.q.as_ptr().add(pi * 2 * n_in + sb * 2 * QK_K + H * QK_K);
        let xv = |j: usize| vld1q_s8(xq.add(16 * j));
        let xb = (pi * bpr + sb * 8 + 4 * H) * 4;
        unroll4!(bl, 0, {
            let bk = 4 * H + bl;
            let p0 = smmla(smmla(zero, w[4 * bl], xv(4 * bl)), w[4 * bl + 1], xv(4 * bl + 1));
            let p1 = smmla(smmla(zero, w[4 * bl + 2], xv(4 * bl + 2)), w[4 * bl + 3], xv(4 * bl + 3));
            let iv = vmlaq_s32(vmulq_s32(p0, sc[2 * bk]), p1, sc[2 * bk + 1]);
            let xs = vld1q_f32(p.s.as_ptr().add(xb + 4 * bl));
            acc[bl] = vfmaq_f32(acc[bl], vcvtq_f32_s32(iv), vmulq_f32(ds[bk], xs));
        });
    }
}

/// Rows `r` and `r + 1` against all tokens with SMMLA.
unsafe fn mm_pair<F: Multi>(m: &GgufMatrix, p: &PairAct, t: usize, y: *mut f32, ys: usize, r: usize) {
    let (rb, n_in) = (m.row_bytes(), m.n_in);
    let bpr = n_in / 32;
    let np = t.div_ceil(2);
    debug_assert!(F::SC16, "SMMLA tiles take formats with integer scales per 16");
    let mut acc = [[vdupq_n_f32(0.0); 4]; MULTI_MAX / 2];
    let (ra, rbp) = (m.data.as_ptr().add(r * rb), m.data.as_ptr().add((r + 1) * rb));
    for sb in 0..n_in / QK_K {
        let (a, b) = (ra.add(sb * F::TSZ), rbp.add(sb * F::TSZ));
        let (ma, mb) = (F::meta(a), F::meta(b));
        let sc = pair16_s32(&ma.sc, &mb.sc);
        let ds = pair_f32::<2, 8>(&ma.ds, &mb.ds);
        mm_half::<F, 0>(a, b, &sc, &ds, p, np, sb, n_in, &mut acc);
        mm_half::<F, 1>(a, b, &sc, &ds, p, np, sb, n_in, &mut acc);
        // The min terms come after both halves, as in the one-token kernels.
        match F::CORR {
            Corr::Q2K => {
                let mn = pair_f32::<4, 16>(&ma.mn, &mb.mn);
                for (pi, a) in acc[..np].iter_mut().enumerate() {
                    let xh = p.h.as_ptr().add((pi * bpr + sb * 8) * 8);
                    for j in 0..4 {
                        let mut tt = vmulq_f32(mn[j], vld1q_f32(xh.add(4 * j)));
                        tt = vfmaq_f32(tt, mn[j + 4], vld1q_f32(xh.add(4 * j + 16)));
                        tt = vfmaq_f32(tt, mn[j + 8], vld1q_f32(xh.add(4 * j + 32)));
                        tt = vfmaq_f32(tt, mn[j + 12], vld1q_f32(xh.add(4 * j + 48)));
                        a[j] = vsubq_f32(a[j], tt);
                    }
                }
            }
            Corr::None | Corr::Min32 => {}
        }
    }
    for (pi, a) in acc[..np].iter().enumerate() {
        let o = vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3]));
        let (c0, c1) = (2 * pi, 2 * pi + 1);
        *y.add(c0 * ys + r) = vgetq_lane_f32::<0>(o);
        *y.add(c0 * ys + r + 1) = vgetq_lane_f32::<2>(o);
        if c1 < t {
            *y.add(c1 * ys + r) = vgetq_lane_f32::<1>(o);
            *y.add(c1 * ys + r + 1) = vgetq_lane_f32::<3>(o);
        }
    }
}

/// SMMLA variant: rows in pairs against tokens in pairs (`p`: the activations interleaved per
/// token pair). Each 2x2 block of (row, token) results is one vector `[r0 t0, r0 t1, r1 t0, r1 t1]`;
/// its four f32 accumulators are the lanes of the one-token kernel's accumulator (blocks j mod 4),
/// so the final `(a0 + a1) + (a2 + a3)` matches that kernel's `vaddvq_f32`.
#[allow(clippy::too_many_arguments)]
unsafe fn multi_mm<F: Multi>(m: &GgufMatrix, x: &Act, p: &PairAct, t: usize, y: *mut f32, ys: usize, r0: usize, r1: usize) {
    let mut r = r0;
    while r + 2 <= r1 {
        mm_pair::<F>(m, p, t, y, ys, r);
        r += 2;
    }
    if r < r1 {
        multi_dot::<F>(m, x, t, y, ys, r, r1);
    }
}

/// Fewest tokens for which the SMMLA variant beats the SDOT one (formats with integer scales per
/// 16 only: with per-32 f32 scales SDOT needs few enough extra instructions that SMMLA, at half
/// SDOT's issue rate on Apple's cores for twice the work, does not pay off).
const SMMLA_MIN_TOKENS: usize = 4;

/// Whether `kind` has multi-token kernels (`Some`), and whether they use SMMLA for `t` tokens
/// with the pair-interleaved activations (`Some(true)`).
pub(super) fn multi_kind(kind: GgmlType, t: usize) -> Option<bool> {
    if cols_kind(kind) {
        return Some(false);
    }
    let sc16 = !matches!(kind, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ4_XS);
    has_multi(kind).then_some(sc16 && t >= SMMLA_MIN_TOKENS)
}

/// The blocks-of-32 formats, whose one-token kernels take a compile-time token count: their
/// multi-token passes run those kernels on column groups (`cols_multi`).
fn cols_kind(kind: GgmlType) -> bool {
    matches!(
        kind,
        GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::IQ4_NL
            | GgmlType::MXFP4
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::NVFP4
            | GgmlType::Q2_0
            | GgmlType::PQ2_0
            | GgmlType::PTQ1_0
    )
}

/// `x` from token `c0` on.
#[inline(always)]
unsafe fn act_at(x: &Act, c0: usize) -> Act {
    x.at(c0)
}

/// Multi-token pass of a blocks-of-32 format (or of a K-quant on per-256 activations, or of an
/// interleaved matrix): rows in tiles, each tile against every column
/// group with the one-token kernel instantiated for that group's width (each weight block is
/// unpacked once per group). Every token's arithmetic is the one-token kernel's, so the rows are
/// bit-identical to `rows::<1>`.
unsafe fn cols_multi(m: &GgufMatrix, x: &Act, t: usize, y: *mut f32, ys: usize, r0: usize, r1: usize) {
    if m.x4 {
        // Tiles of whole groups of four rows (`col_tiles` over group indices).
        crate::kernels::col_tiles(t, m.n_in, r0 / 4, r1.div_ceil(4), 4 * m.row_bytes(), |c0, c1, ga, gb| {
            let (ta, tb) = ((4 * ga).max(r0), (4 * gb).min(r1));
            let xg = act_at(x, c0);
            let yg = y.add(c0 * ys);
            macro_rules! go {
                ($TC:literal) => {
                    rows_x4::<$TC>(m, &xg, yg, ys, ta, tb)
                };
            }
            crate::kernels::with_cols!(c1 - c0, go)
        });
        return;
    }
    macro_rules! kern {
        ($TC:literal) => {
            match m.kind {
                k if x.k && q8k_kind(k) => k_kern::<$TC>(k),
                GgmlType::Q8_0 => q8_0::<$TC> as RowFn<$TC>,
                GgmlType::Q4_0 => q32::<$TC, Q4_0>,
                GgmlType::IQ4_NL => q32::<$TC, IQ4_NL>,
                GgmlType::MXFP4 => q32::<$TC, MXFP4>,
                GgmlType::Q4_1 => q32::<$TC, Q4_1>,
                GgmlType::Q5_0 => q32::<$TC, Q5_0>,
                GgmlType::NVFP4 => nvfp4::<$TC>,
                GgmlType::Q2_0 => q2_0::<$TC>,
                GgmlType::PQ2_0 => pq2_0::<$TC>,
                GgmlType::PTQ1_0 => ptq1_0::<$TC>,
                _ => q32::<$TC, Q5_1>,
            }
        };
    }
    let rb = m.row_bytes();
    let base = m.data.as_ptr();
    crate::kernels::col_tiles(t, m.n_in, r0, r1, rb, |c0, c1, ta, tb| {
        let xg = act_at(x, c0);
        let yg = y.add(c0 * ys);
        macro_rules! go {
            ($TC:literal) => {{
                let f = kern!($TC);
                for r in ta..tb {
                    let o = f(base.add(r * rb), m.n_in, &xg);
                    for (c, v) in o.iter().enumerate() {
                        *yg.add(c * ys + r) = *v;
                    }
                }
            }};
        }
        crate::kernels::with_cols!(c1 - c0, go)
    });
}

/// Whether `kind` has multi-token kernels.
fn has_multi(kind: GgmlType) -> bool {
    matches!(
        kind,
        GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q3_K
            | GgmlType::Q2_K
            | GgmlType::IQ4_XS
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::IQ1_S
            | GgmlType::IQ1_M
    )
}

/// Rows `r0..r1` of `m` against all `t` (2..=MULTI_MAX) activation rows into `y[c * ys + r]`,
/// with SMMLA when `p` (the pair-interleaved activations) is given and `multi_kind` says so;
/// false if `m.kind` has no multi-token kernel.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn rows_multi(
    m: &GgufMatrix,
    x: &Act,
    p: Option<&PairAct>,
    t: usize,
    y: *mut f32,
    ys: usize,
    r0: usize,
    r1: usize,
) -> bool {
    debug_assert!((2..=MULTI_MAX).contains(&t));
    if cols_kind(m.kind) || m.x4 || (x.k && q8k_kind(m.kind)) {
        cols_multi(m, x, t, y, ys, r0, r1);
        return true;
    }
    macro_rules! go {
        ($F:ty) => {
            match p {
                Some(p) if <$F>::SC16 && t >= SMMLA_MIN_TOKENS => multi_mm::<$F>(m, x, p, t, y, ys, r0, r1),
                _ => multi_dot::<$F>(m, x, t, y, ys, r0, r1),
            }
        };
    }
    match m.kind {
        GgmlType::Q4_K => go!(MQ4K<false>),
        GgmlType::Q5_K => go!(MQ4K<true>),
        GgmlType::Q6_K => go!(MSb16<Q6K>),
        GgmlType::Q3_K => go!(MSb16<Q3K>),
        GgmlType::Q2_K => go!(MQ2K),
        GgmlType::IQ4_XS => go!(MIq4Xs),
        GgmlType::IQ2_XXS => go!(MSb16<Iq2Xxs>),
        GgmlType::IQ2_XS => go!(MSb16<Iq2Xs>),
        GgmlType::IQ2_S => go!(MSb16<Iq2S>),
        GgmlType::IQ3_XXS => go!(MSb16<Iq3Xxs>),
        GgmlType::IQ3_S => go!(MSb16<Iq3S>),
        GgmlType::IQ1_S => go!(MSb16<Iq1S>),
        GgmlType::IQ1_M => go!(MSb16<Iq1M>),
        _ => return false,
    }
    true
}

// ---- Int8 prefill GEMM rows (`kernels::i8gemm`) ------------------------------------------------

/// `(sc16, min32)` of `kind`'s rows for the int8 prefill GEMM (see `kernels::I8Layout`); None if
/// `kind` has no row unpacker there.
pub(super) fn i8_layout(kind: GgmlType) -> Option<(bool, bool)> {
    use GgmlType::*;
    Some(match kind {
        Q8_0 | Q4_0 | IQ4_NL | MXFP4 | Q5_0 | IQ4_XS | Q2_0 | PQ2_0 | PTQ1_0 => (false, false),
        Q4_1 | Q5_1 | Q4_K | Q5_K => (false, true),
        Q6_K | Q3_K | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ1_S | IQ1_M => (true, false),
        _ => return None,
    })
}

/// One row of `kind` (`n_in` elements at `row`) unpacked for the int8 prefill GEMM: centered
/// int8 values `q`, f32 scales per 32 `ds`, and per `i8_layout` mins per 32 `mn` (subtracted
/// times the activations' block sums) or integer scales per 16 `sc`.
pub(super) unsafe fn unpack_row_i8(kind: GgmlType, row: *const u8, n_in: usize, q: *mut i8, ds: *mut f32, mn: *mut f32, sc: *mut i32) {
    match kind {
        GgmlType::Q8_0 => {
            for b in 0..n_in / 32 {
                let p = row.add(34 * b);
                vst1q_s8(q.add(32 * b), vld1q_s8(p.add(2) as *const i8));
                vst1q_s8(q.add(32 * b + 16), vld1q_s8(p.add(18) as *const i8));
                *ds.add(b) = f16p(p);
            }
        }
        GgmlType::Q4_0 => b32_i8::<Q4_0>(row, n_in, q, ds, mn),
        GgmlType::IQ4_NL => b32_i8::<IQ4_NL>(row, n_in, q, ds, mn),
        GgmlType::MXFP4 => b32_i8::<MXFP4>(row, n_in, q, ds, mn),
        GgmlType::Q4_1 => b32_i8::<Q4_1>(row, n_in, q, ds, mn),
        GgmlType::Q5_0 => b32_i8::<Q5_0>(row, n_in, q, ds, mn),
        GgmlType::Q5_1 => b32_i8::<Q5_1>(row, n_in, q, ds, mn),
        GgmlType::Q2_0 => {
            for b in 0..n_in / 64 {
                let p = row.add(18 * b);
                for (k, w) in q2_0_unpack(p).iter().enumerate() {
                    vst1q_s8(q.add(64 * b + 16 * k), *w);
                }
                let d = f16p(p);
                *ds.add(2 * b) = d;
                *ds.add(2 * b + 1) = d;
            }
        }
        GgmlType::PQ2_0 | GgmlType::PTQ1_0 => {
            let (bs, sa, un): (usize, usize, Unpack128) = if kind == GgmlType::PQ2_0 {
                (34, 0, pq2_0_unpack)
            } else {
                (28, 26, ptq1_0_unpack)
            };
            for b in 0..n_in / 128 {
                let p = row.add(bs * b);
                for (k, w) in un(p).iter().enumerate() {
                    vst1q_s8(q.add(128 * b + 16 * k), *w);
                }
                let d = f16p(p.add(sa));
                for j in 0..4 {
                    *ds.add(4 * b + j) = d;
                }
            }
        }
        GgmlType::Q4_K => sb_i8::<MQ4K<false>>(row, n_in, q, ds, mn, sc),
        GgmlType::Q5_K => sb_i8::<MQ4K<true>>(row, n_in, q, ds, mn, sc),
        GgmlType::Q6_K => sb_i8::<MSb16<Q6K>>(row, n_in, q, ds, mn, sc),
        GgmlType::Q3_K => sb_i8::<MSb16<Q3K>>(row, n_in, q, ds, mn, sc),
        GgmlType::IQ4_XS => sb_i8::<MIq4Xs>(row, n_in, q, ds, mn, sc),
        GgmlType::IQ2_XXS => sb_i8::<MSb16<Iq2Xxs>>(row, n_in, q, ds, mn, sc),
        GgmlType::IQ2_XS => sb_i8::<MSb16<Iq2Xs>>(row, n_in, q, ds, mn, sc),
        GgmlType::IQ2_S => sb_i8::<MSb16<Iq2S>>(row, n_in, q, ds, mn, sc),
        GgmlType::IQ3_XXS => sb_i8::<MSb16<Iq3Xxs>>(row, n_in, q, ds, mn, sc),
        GgmlType::IQ3_S => sb_i8::<MSb16<Iq3S>>(row, n_in, q, ds, mn, sc),
        GgmlType::IQ1_S => sb_i8::<MSb16<Iq1S>>(row, n_in, q, ds, mn, sc),
        GgmlType::IQ1_M => sb_i8::<MSb16<Iq1M>>(row, n_in, q, ds, mn, sc),
        _ => unreachable!("no int8 GEMM rows for {kind}"),
    }
}

/// [`unpack_row_i8`] for the blocks-of-32 nibble formats (`K` as in `q32`), nibbles centered.
unsafe fn b32_i8<const K: u8>(row: *const u8, n_in: usize, q: *mut i8, ds: *mut f32, mn: *mut f32) {
    let (tsz, qs) = match K {
        Q4_0 | IQ4_NL => (18, 2),
        MXFP4 => (17, 1),
        Q4_1 => (20, 4),
        Q5_0 => (22, 6),
        _ => (24, 8),
    };
    let table = K == IQ4_NL || K == MXFP4;
    let has_min = K == Q4_1 || K == Q5_1;
    let five = K == Q5_0 || K == Q5_1;
    let qoff: i8 = if five { 16 } else { 8 };
    let m4 = vdupq_n_u8(0x0F);
    let kv = vld1q_s8(if K == MXFP4 {
        KVALUES_MXFP4.as_ptr()
    } else {
        KVALUES_IQ4NL.as_ptr()
    });
    for b in 0..n_in / 32 {
        let p = row.add(b * tsz);
        let v = vld1q_u8(p.add(qs));
        let (mut lo, mut hi) = (vandq_u8(v, m4), vshrq_n_u8::<4>(v));
        if five {
            let (h0, h1) = sign_masks(rd32(p.add(qs - 4)));
            let b16 = vdupq_n_u8(16);
            lo = vorrq_u8(lo, vandq_u8(h0, b16));
            hi = vorrq_u8(hi, vandq_u8(h1, b16));
        }
        let (lo, hi) = if table {
            (vqtbl1q_s8(kv, lo), vqtbl1q_s8(kv, hi))
        } else {
            let o = vdupq_n_s8(qoff);
            (vsubq_s8(vreinterpretq_s8_u8(lo), o), vsubq_s8(vreinterpretq_s8_u8(hi), o))
        };
        vst1q_s8(q.add(32 * b), lo);
        vst1q_s8(q.add(32 * b + 16), hi);
        let d = scale32::<K>(p);
        *ds.add(b) = d;
        if has_min {
            *mn.add(b) = -(f16p(p.add(2)) + d * qoff as f32);
        }
    }
}

/// [`unpack_row_i8`] for the super-block formats, through their multi-token unpackers.
unsafe fn sb_i8<F: Multi>(row: *const u8, n_in: usize, q: *mut i8, ds: *mut f32, mn: *mut f32, sc: *mut i32) {
    for s in 0..n_in / QK_K {
        let mut u = Unpacked {
            w: [vdupq_n_s8(0); 16],
            m: Meta::zero(),
        };
        F::unpack(row.add(s * F::TSZ), &mut u);
        for (k, w) in u.w.iter().enumerate() {
            vst1q_s8(q.add(QK_K * s + 16 * k), *w);
        }
        vst1q_f32(ds.add(8 * s), u.m.ds[0]);
        vst1q_f32(ds.add(8 * s + 4), u.m.ds[1]);
        if F::CORR == Corr::Min32 {
            vst1q_f32(mn.add(8 * s), u.m.mn[0]);
            vst1q_f32(mn.add(8 * s + 4), u.m.mn[1]);
        }
        if F::SC16 {
            for k in 0..4 {
                vst1q_s32(sc.add(16 * s + 4 * k), u.m.sc[k]);
            }
        }
    }
}

// ---- Dequantization (prefill and tables) -------------------------------------------------------
// The same unpacking, then f32 scaling in the operation order of ggml's reference code, so the
// values are bit-identical to `super::dequantize_row`.

/// Dequantizes one row of `n` elements into `out`; false if `kind` has no NEON version.
pub(super) unsafe fn dequant_row(kind: GgmlType, row: *const u8, n: usize, out: *mut f32) -> bool {
    match kind {
        GgmlType::Q4_K => deq_q4k::<false>(row, n, out),
        GgmlType::Q5_K => deq_q4k::<true>(row, n, out),
        GgmlType::Q6_K => deq_sb16::<Q6K>(row, n, out),
        GgmlType::Q3_K => deq_sb16::<Q3K>(row, n, out),
        GgmlType::IQ2_XXS => deq_sb16::<Iq2Xxs>(row, n, out),
        GgmlType::IQ2_XS => deq_sb16::<Iq2Xs>(row, n, out),
        GgmlType::IQ2_S => deq_sb16::<Iq2S>(row, n, out),
        GgmlType::IQ3_XXS => deq_sb16::<Iq3Xxs>(row, n, out),
        GgmlType::IQ3_S => deq_sb16::<Iq3S>(row, n, out),
        GgmlType::IQ4_XS => deq_iq4xs(row, n, out),
        GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::IQ4_NL => deq_b32(kind, row, n, out),
        _ => return false,
    }
    true
}

/// The four rows of the group at `g` of an interleaved Q8_0 or Q4_0 matrix ([`super::repack`])
/// dequantized into `out` (`[4][n]`, rows `n` apart): `deq_b32`'s values (each row's pieces
/// gathered by a de-interleaving load, then the same products).
pub(super) unsafe fn dequant_group_x4(kind: GgmlType, g: *const u8, n: usize, out: *mut f32) -> bool {
    let (bsz, eight) = match kind {
        GgmlType::Q8_0 => (136, None),
        GgmlType::Q4_0 => (72, Some((vdupq_n_u8(0x0F), vdupq_n_s8(8)))),
        _ => return false,
    };
    for b in 0..n / 32 {
        let p = g.add(b * bsz);
        let mut d = [0f32; 4];
        vst1q_f32(d.as_mut_ptr(), f16x4_to_f32(p as *const u16));
        // Row `i`'s values 0..16 and 16..32.
        let mut v = [[vdupq_n_s8(0); 2]; 4];
        match eight {
            None => {
                let (lo, hi) = (vld4q_u32(p.add(8) as *const u32), vld4q_u32(p.add(72) as *const u32));
                v = [[lo.0, hi.0], [lo.1, hi.1], [lo.2, hi.2], [lo.3, hi.3]].map(|r| r.map(|x| vreinterpretq_s8_u32(x)));
            }
            Some((m4, e)) => {
                let a = vld4q_u32(p.add(8) as *const u32);
                for (i, r) in [a.0, a.1, a.2, a.3].into_iter().enumerate() {
                    let u = vreinterpretq_u8_u32(r);
                    v[i] = [
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u, m4)), e),
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u)), e),
                    ];
                }
            }
        }
        for i in 0..4 {
            let o = out.add(i * n + 32 * b);
            st16(o, v[i][0], d[i]);
            st16(o.add(16), v[i][1], d[i]);
        }
    }
    true
}

/// Q8_0, Q4_0, Q5_0 and IQ4_NL: `q * d` per block of 32, as ggml computes it (one f32 product
/// of the centered integer or table value).
unsafe fn deq_b32(kind: GgmlType, row: *const u8, n: usize, out: *mut f32) {
    let mut q = [0i8; 32];
    let mut d = 0f32;
    let mut m = 0f32;
    let tsz = kind.type_size();
    for b in 0..n / 32 {
        let p = row.add(b * tsz);
        match kind {
            GgmlType::Q8_0 => unpack_row_i8(kind, p, 32, q.as_mut_ptr(), &mut d, &mut m, std::ptr::null_mut()),
            GgmlType::Q4_0 => b32_i8::<Q4_0>(p, 32, q.as_mut_ptr(), &mut d, &mut m),
            GgmlType::Q5_0 => b32_i8::<Q5_0>(p, 32, q.as_mut_ptr(), &mut d, &mut m),
            _ => b32_i8::<IQ4_NL>(p, 32, q.as_mut_ptr(), &mut d, &mut m),
        }
        st16(out.add(32 * b), vld1q_s8(q.as_ptr()), d);
        st16(out.add(32 * b + 16), vld1q_s8(q.as_ptr().add(16)), d);
    }
}

/// Stores `v * s` (16 int8 values times an f32 scale) at `o`.
#[inline(always)]
unsafe fn st16(o: *mut f32, v: int8x16_t, s: f32) {
    let lo = vmovl_s8(vget_low_s8(v));
    let hi = vmovl_high_s8(v);
    vst1q_f32(o, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), s));
    vst1q_f32(o.add(4), vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(lo)), s));
    vst1q_f32(o.add(8), vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), s));
    vst1q_f32(o.add(12), vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(hi)), s));
}

/// The per-16 f32 scales of a super-block exactly as ggml's dequantization computes them.
trait Sb16Exact: Sb16 {
    unsafe fn exact_scales(b: *const u8, s: &mut [f32; 16]);
}
impl Sb16Exact for Q6K {
    unsafe fn exact_scales(b: *const u8, s: &mut [f32; 16]) {
        let d = f16p(b.add(208));
        for k in 0..16 {
            s[k] = d * (*b.add(192 + k) as i8) as f32;
        }
    }
}
impl Sb16Exact for Q3K {
    unsafe fn exact_scales(b: *const u8, s: &mut [f32; 16]) {
        let d = f16p(b.add(108));
        let sc = q3k_scales(std::slice::from_raw_parts(b.add(96), 12));
        for k in 0..16 {
            s[k] = d * (sc[k] as i32 - 32) as f32;
        }
    }
}
impl Sb16Exact for Iq2Xxs {
    unsafe fn exact_scales(b: *const u8, s: &mut [f32; 16]) {
        let d = f16p(b);
        for ib in 0..8 {
            let db = d * (0.5 + (rd32(b.add(6 + 8 * ib)) >> 28) as f32) * 0.25;
            s[2 * ib] = db;
            s[2 * ib + 1] = db;
        }
    }
}
impl Sb16Exact for Iq2Xs {
    unsafe fn exact_scales(b: *const u8, s: &mut [f32; 16]) {
        let d = f16p(b);
        for ib in 0..8 {
            let v = *b.add(66 + ib);
            s[2 * ib] = d * (0.5 + (v & 0xF) as f32) * 0.25;
            s[2 * ib + 1] = d * (0.5 + (v >> 4) as f32) * 0.25;
        }
    }
}
impl Sb16Exact for Iq2S {
    unsafe fn exact_scales(b: *const u8, s: &mut [f32; 16]) {
        let d = f16p(b);
        for ib in 0..8 {
            let v = *b.add(74 + ib);
            s[2 * ib] = d * (0.5 + (v & 0xF) as f32) * 0.25;
            s[2 * ib + 1] = d * (0.5 + (v >> 4) as f32) * 0.25;
        }
    }
}
impl Sb16Exact for Iq3Xxs {
    unsafe fn exact_scales(b: *const u8, s: &mut [f32; 16]) {
        let d = f16p(b);
        for ib in 0..8 {
            let db = d * (0.5 + (rd32(b.add(66 + 4 * ib)) >> 28) as f32) * 0.5;
            s[2 * ib] = db;
            s[2 * ib + 1] = db;
        }
    }
}
impl Sb16Exact for Iq3S {
    unsafe fn exact_scales(b: *const u8, s: &mut [f32; 16]) {
        let d = f16p(b);
        for ib in 0..8 {
            let db = d * (1 + 2 * ((*b.add(106 + ib / 2) >> (4 * (ib % 2))) & 0xF) as i32) as f32;
            s[2 * ib] = db;
            s[2 * ib + 1] = db;
        }
    }
}

unsafe fn deq_sb16<U: Sb16Exact>(row: *const u8, n: usize, out: *mut f32) {
    for sb in 0..n / QK_K {
        let b = row.add(sb * U::TSZ);
        let mut w = [vdupq_n_s8(0); 16];
        let mut sci = [0i8; 16];
        U::unpack(b, &mut w, &mut sci);
        let mut s = [0f32; 16];
        U::exact_scales(b, &mut s);
        for k in 0..16 {
            st16(out.add(sb * QK_K + 16 * k), w[k], s[k]);
        }
    }
}

unsafe fn deq_q4k<const FIVE: bool>(row: *const u8, n: usize, out: *mut f32) {
    let tsz = if FIVE { 176 } else { 144 };
    let m4 = vdupq_n_u8(0x0F);
    let b16 = vdupq_n_u8(16);
    for sb in 0..n / QK_K {
        let b = row.add(sb * tsz);
        let (d, min) = (f16p(b), f16p(b.add(2)));
        let scales = std::slice::from_raw_parts(b.add(4), 12);
        let qs = b.add(if FIVE { 48 } else { 16 });
        let (h0, h1) = if FIVE {
            (vld1q_u8(b.add(16)), vld1q_u8(b.add(32)))
        } else {
            (vdupq_n_u8(0), vdupq_n_u8(0))
        };
        for j in 0..4 {
            let v0 = vld1q_u8(qs.add(32 * j));
            let v1 = vld1q_u8(qs.add(32 * j + 16));
            for h in 0..2 {
                let (sc, mn) = super::scale_min_k4(2 * j + h, scales);
                let (d1, m1) = (vdupq_n_f32(d * sc as f32), vdupq_n_f32(min * mn as f32));
                let mut q = if h == 0 {
                    [vandq_u8(v0, m4), vandq_u8(v1, m4)]
                } else {
                    [vshrq_n_u8::<4>(v0), vshrq_n_u8::<4>(v1)]
                };
                if FIVE {
                    let u = vdupq_n_u8(1 << (2 * j + h));
                    q[0] = vorrq_u8(q[0], vandq_u8(vtstq_u8(h0, u), b16));
                    q[1] = vorrq_u8(q[1], vandq_u8(vtstq_u8(h1, u), b16));
                }
                let o = out.add(sb * QK_K + 64 * j + 32 * h);
                for (k, qv) in q.into_iter().enumerate() {
                    let lo = vmovl_u8(vget_low_u8(qv));
                    let hi = vmovl_high_u8(qv);
                    let f = [
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
                        vcvtq_f32_u32(vmovl_high_u16(lo)),
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
                        vcvtq_f32_u32(vmovl_high_u16(hi)),
                    ];
                    for (i, fv) in f.into_iter().enumerate() {
                        vst1q_f32(o.add(16 * k + 4 * i), vsubq_f32(vmulq_f32(d1, fv), m1));
                    }
                }
            }
        }
    }
}

unsafe fn deq_iq4xs(row: *const u8, n: usize, out: *mut f32) {
    let m4 = vdupq_n_u8(0x0F);
    let kv = vld1q_s8(KVALUES_IQ4NL.as_ptr());
    for sb in 0..n / QK_K {
        let b = row.add(sb * 136);
        let blk = std::slice::from_raw_parts(b, 136);
        let d = f16p(b);
        for ib in 0..8 {
            let dl = d * super::iq4xs_scale(blk, ib) as f32;
            let v = vld1q_u8(b.add(8 + 16 * ib));
            let o = out.add(sb * QK_K + 32 * ib);
            st16(o, vqtbl1q_s8(kv, vandq_u8(v, m4)), dl);
            st16(o.add(16), vqtbl1q_s8(kv, vshrq_n_u8::<4>(v)), dl);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::random_blocks;

    /// The Q2_K, Q4_K and Q5_K kernels on per-256 activations (plain rows of 1-3 tokens, the
    /// interleaved Q4_K groups, the multi-token column groups) against the scalar definition in f64:
    /// `sum(s * q * x) + m * sum(x)` per sub-block of 16 (`decode_i8`'s scales and offsets) with
    /// the activations' values `x = d * s32 * xq`, from activations that have only that form.
    #[test]
    fn q8k_kernels_match_scalar_reference() {
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let kinds = [GgmlType::Q2_K, GgmlType::Q4_K, GgmlType::Q5_K];
        for (kind, worst) in kinds.into_iter().flat_map(|k| [(k, false), (k, true)]) {
            let (n_out, n_in, t) = (12usize, 3 * QK_K, 3usize);
            let mut seed = 7 + kind.id() as u64;
            let mut raw = random_blocks(kind, n_out * n_in / QK_K, &mut seed);
            let x: Vec<f32> = if worst {
                // The largest integer sums: every weight value, scale and min at its largest
                // with f16 scales of 1, every activation at its largest.
                let ts = kind.type_size();
                for b in raw.chunks_exact_mut(ts) {
                    b.fill(0xFF);
                    let one = 0x3C00u16.to_le_bytes();
                    let d = match kind {
                        GgmlType::Q2_K => vec![80, 82],
                        _ => vec![0, 2],
                    };
                    for o in d {
                        b[o..o + 2].copy_from_slice(&one);
                    }
                }
                vec![1.0; t * n_in]
            } else {
                // Activations with a wide range inside each block of 256 (and an all-zero block).
                let mut x: Vec<f32> = (0..t * n_in)
                    .map(|i| (((i * 7919 + 13) % 2001) as f32 / 1000.0 - 1.0) * [0.01, 1.0, 30.0][(i / 64) % 3])
                    .collect();
                x[QK_K..2 * QK_K].fill(0.0);
                x
            };
            let m = GgufMatrix::from_bytes(kind, &raw, n_out, n_in).unwrap();
            let xa = crate::quant::quantize_act_with(&x, t, n_in, false, true);
            let k = xa.k.as_ref().unwrap();
            let act = Act::new(&xa, 0);
            let (mut q, mut s, mut mn) = ([0i8; QK_K], [0f32; 16], [0f32; 16]);
            let mut want = vec![0f32; t * n_out];
            let mut mag = vec![0f64; t * n_out];
            for r in 0..n_out {
                for c in 0..t {
                    let (mut acc, mut ab) = (0f64, 0f64);
                    for (bi, blk) in m.row(r).chunks_exact(kind.type_size()).enumerate() {
                        let has_min = crate::ggml::decode_i8(kind, blk, &mut q, &mut s, &mut mn);
                        let d = k.d[c * n_in / QK_K + bi] as f64;
                        for g in 0..16 {
                            let e = c * n_in + bi * QK_K + 16 * g;
                            // The activations' values: `d * s32 * q`.
                            let xs = d * k.s[e / 32] as f64;
                            for i in 0..16 {
                                let v = s[g] as f64 * q[16 * g + i] as f64 * k.q[e + i] as f64 * xs;
                                acc += v;
                                ab += v.abs();
                            }
                            if has_min {
                                let xsum: f64 = (0..16).map(|i| k.q[e + i] as f64 * xs).sum();
                                assert_eq!(k.bs[e / 16] as f64 * d, xsum);
                                let v = mn[g] as f64 * xsum;
                                acc += v;
                                ab += v.abs();
                            }
                        }
                    }
                    want[c * n_out + r] = acc as f32;
                    mag[c * n_out + r] = ab;
                }
            }
            let check = |y: &[f32], what: &str| {
                for (i, (&a, &b)) in y.iter().zip(&want).enumerate() {
                    assert!(((a - b).abs() as f64) <= 1e-5 * (mag[i] + 1e-6), "{kind} {what} [{i}]: {a} vs {b}");
                }
            };
            // One token at a time, then three at once (bit for bit the same).
            let mut y1 = vec![0f32; t * n_out];
            for c in 0..t {
                unsafe { assert!(rows::<1>(&m, &act.at(c), y1.as_mut_ptr().add(c * n_out), 0, 0, n_out)) };
            }
            check(&y1, "one token");
            let mut y3 = vec![0f32; t * n_out];
            unsafe { assert!(rows::<3>(&m, &act, y3.as_mut_ptr(), n_out, 0, n_out)) };
            assert_eq!(bits(&y3), bits(&y1), "{kind} three tokens");
            let mut ym = vec![0f32; t * n_out];
            unsafe { assert!(rows_multi(&m, &act, None, t, ym.as_mut_ptr(), n_out, 0, n_out)) };
            assert_eq!(bits(&ym), bits(&y1), "{kind} multi-token");
            if crate::ggml::can_interleave(kind, n_out, n_in) {
                let x4 = m.to_interleaved().unwrap();
                let mut y4 = vec![0f32; t * n_out];
                for c in 0..t {
                    // Ranges that cut groups of four.
                    for (r0, r1) in [(0, 3), (3, 9), (9, n_out)] {
                        unsafe { assert!(rows::<1>(&x4, &act.at(c), y4.as_mut_ptr().add(c * n_out), 0, r0, r1)) };
                    }
                }
                assert_eq!(bits(&y4), bits(&y1), "{kind} interleaved");
                let mut y5 = vec![0f32; t * n_out];
                unsafe { assert!(rows::<3>(&x4, &act, y5.as_mut_ptr(), n_out, 0, n_out)) };
                assert_eq!(bits(&y5), bits(&y1), "{kind} interleaved, three tokens");
            }
        }
    }

    /// The one-token kernels that take several rows at a time give each row bit for bit the
    /// one-row kernel's result (which the multi-token kernels also reproduce), on row ranges of
    /// every parity, and agree with the portable scalar dot product.
    #[test]
    fn row_group_kernels_match_one_row_kernels() {
        for kind in [GgmlType::IQ1_S, GgmlType::IQ1_M, GgmlType::IQ2_XXS] {
            let (n_out, n_in) = (11usize, 3 * QK_K);
            let mut seed = 101 + kind.id() as u64;
            let m = GgufMatrix::from_bytes(kind, &random_blocks(kind, n_out * n_in / QK_K, &mut seed), n_out, n_in).unwrap();
            let x: Vec<f32> = (0..n_in).map(|i| ((i * 7919 + 13) % 2001) as f32 / 1000.0 - 1.0).collect();
            let xa = crate::quant::quantize_act(&x, 1, n_in);
            let act = Act::new(&xa, 0);
            let one: RowFn<1> = match kind {
                GgmlType::IQ1_S => iq1s::<1>,
                GgmlType::IQ1_M => sb16::<1, Iq1M>,
                _ => sb16::<1, Iq2Xxs>,
            };
            let want: Vec<u32> = (0..n_out)
                .map(|r| unsafe { one(m.row(r).as_ptr(), n_in, &act)[0].to_bits() })
                .collect();
            for split in [0, 1, 4, 7, n_out] {
                let mut y = vec![f32::NAN; n_out];
                unsafe {
                    assert!(rows_t1(&m, &act, y.as_mut_ptr(), 0, split));
                    assert!(rows_t1(&m, &act, y.as_mut_ptr(), split, n_out));
                }
                let got: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
                assert_eq!(got, want, "{kind} split {split}");
                for (r, &v) in y.iter().enumerate() {
                    let p = unsafe { super::super::dot_row_generic::<1>(kind, m.row(r), &act)[0] };
                    assert!((v - p).abs() <= 1e-4 * (1.0 + p.abs()), "{kind} row {r}: {v} vs portable {p}");
                }
            }
        }
    }
}
