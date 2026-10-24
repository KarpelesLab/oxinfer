//! NEON decode kernels for the hot GGUF formats: each unpacks a block's weights to int8 vectors in
//! registers once, then runs SDOT against TC int8 activation rows. Integer sums are formed per
//! 32-element activation block (sub-blocks of 16 are scaled in int32 first) and scaled in f32.
//! Formats with mins subtract `min * sum(x)` using the exact activation sums.
#![allow(unsafe_op_in_unsafe_fn)]

use super::{Act, GgmlType, GgufMatrix, QK_K, q3k_scales, tables::KVALUES_IQ4NL};
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
    let f: RowFn<TC> = match m.kind {
        GgmlType::Q4_K => q4k::<TC, false>,
        GgmlType::Q5_K => q4k::<TC, true>,
        GgmlType::Q6_K => sb16::<TC, Q6K>,
        GgmlType::Q3_K => sb16::<TC, Q3K>,
        GgmlType::IQ2_XXS => sb16::<TC, Iq2Xxs>,
        GgmlType::IQ2_XS => sb16::<TC, Iq2Xs>,
        GgmlType::IQ2_S => sb16::<TC, Iq2S>,
        GgmlType::IQ3_XXS => sb16::<TC, Iq3Xxs>,
        GgmlType::IQ3_S => sb16::<TC, Iq3S>,
        GgmlType::IQ1_S => sb16::<TC, Iq1S>,
        GgmlType::IQ1_M => sb16::<TC, Iq1M>,
        GgmlType::Q2_K => q2k::<TC>,
        GgmlType::IQ4_XS => iq4xs::<TC>,
        GgmlType::Q8_0 => q8_0::<TC>,
        GgmlType::Q4_0 => q32::<TC, Q4_0>,
        GgmlType::IQ4_NL => q32::<TC, IQ4_NL>,
        GgmlType::Q4_1 => q32::<TC, Q4_1>,
        GgmlType::Q5_0 => q32::<TC, Q5_0>,
        GgmlType::Q5_1 => q32::<TC, Q5_1>,
        _ => return false,
    };
    let rb = m.row_bytes();
    let base = m.data.as_ptr();
    for r in r0..r1 {
        let o = f(base.add(r * rb), m.n_in, x);
        for c in 0..TC {
            *y.add(c * ys + r) = o[c];
        }
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

/// The blocks-of-32 nibble formats (`q32` kernel parameter).
const Q4_0: u8 = 0;
const IQ4_NL: u8 = 1;
const Q4_1: u8 = 2;
const Q5_0: u8 = 3;
const Q5_1: u8 = 4;

/// Q4_0, IQ4_NL, Q4_1, Q5_0 and Q5_1 (`K`): blocks of 32 with an f16 scale (and an f16 min for
/// Q4_1 / Q5_1), nibbles plus, for Q5, a fifth bit per element in a 32-bit word; four blocks per step.
unsafe fn q32<const TC: usize, const K: u8>(row: *const u8, n_in: usize, x: &Act) -> [f32; TC] {
    let (tsz, qs) = match K {
        Q4_0 | IQ4_NL => (18, 2),
        Q4_1 => (20, 4),
        Q5_0 => (22, 6),
        _ => (24, 8),
    };
    let has_min = K == Q4_1 || K == Q5_1;
    // Nibbles are centered by `qoff` (Q4_1 / Q5_1 like Q4_0 / Q5_0; the min absorbs `d * qoff`).
    let qoff = if K == Q5_0 || K == Q5_1 { 16.0 } else { 8.0 };
    let bpr = n_in / 32;
    let zero = vdupq_n_s32(0);
    let m4 = vdupq_n_u8(0x0F);
    let kv = vld1q_s8(KVALUES_IQ4NL.as_ptr());
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
        match K {
            IQ4_NL => (vqtbl1q_s8(kv, vreinterpretq_u8_s8(lo)), vqtbl1q_s8(kv, vreinterpretq_u8_s8(hi))),
            // The offset (8 or 16) is subtracted from the integer sums.
            _ => (lo, hi),
        }
    };
    let mut acc = [vdupq_n_f32(0.0); TC];
    let mut blk = 0;
    while blk + 4 <= bpr {
        let b = row.add(blk * tsz);
        let (mut dh, mut mh) = ([0u16; 4], [0u16; 4]);
        let mut w = [vdupq_n_s8(0); 8];
        for k in 0..4 {
            dh[k] = rd16(b.add(tsz * k));
            mh[k] = rd16(b.add(tsz * k + 2));
            (w[2 * k], w[2 * k + 1]) = unpack(b.add(tsz * k));
        }
        let dv = f16x4_to_f32(dh.as_ptr());
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
            if K != IQ4_NL {
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
        let d = f16p(b);
        for c in 0..TC {
            let xq = x.q.add(c * n_in + blk * 32);
            let mut s = vaddvq_s32(sdot(sdot(zero, w0, vld1q_s8(xq)), w1, vld1q_s8(xq.add(16))));
            if K != IQ4_NL {
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
/// registers while the token pairs stream past.
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
            let iv = if F::SC16 {
                let p0 = smmla(smmla(zero, w[4 * bl], xv(4 * bl)), w[4 * bl + 1], xv(4 * bl + 1));
                let p1 = smmla(smmla(zero, w[4 * bl + 2], xv(4 * bl + 2)), w[4 * bl + 3], xv(4 * bl + 3));
                vmlaq_s32(vmulq_s32(p0, sc[2 * bk]), p1, sc[2 * bk + 1])
            } else {
                let mut s = smmla(zero, w[4 * bl], xv(4 * bl));
                s = smmla(s, w[4 * bl + 1], xv(4 * bl + 1));
                s = smmla(s, w[4 * bl + 2], xv(4 * bl + 2));
                smmla(s, w[4 * bl + 3], xv(4 * bl + 3))
            };
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
    let mut acc = [[vdupq_n_f32(0.0); 4]; MULTI_MAX / 2];
    let (ra, rbp) = (m.data.as_ptr().add(r * rb), m.data.as_ptr().add((r + 1) * rb));
    for sb in 0..n_in / QK_K {
        let (a, b) = (ra.add(sb * F::TSZ), rbp.add(sb * F::TSZ));
        let (ma, mb) = (F::meta(a), F::meta(b));
        let sc = if F::SC16 {
            pair16_s32(&ma.sc, &mb.sc)
        } else {
            [vdupq_n_s32(0); 16]
        };
        let ds = pair_f32::<2, 8>(&ma.ds, &mb.ds);
        mm_half::<F, 0>(a, b, &sc, &ds, p, np, sb, n_in, &mut acc);
        mm_half::<F, 1>(a, b, &sc, &ds, p, np, sb, n_in, &mut acc);
        // The min terms come after both halves, as in the one-token kernels.
        match F::CORR {
            Corr::Min32 => {
                let mn = pair_f32::<2, 8>(&ma.mn, &mb.mn);
                for (pi, a) in acc[..np].iter_mut().enumerate() {
                    let xf = p.f.as_ptr().add((pi * bpr + sb * 8) * 4);
                    for j in 0..4 {
                        let v = vfmsq_f32(a[j], mn[j], vld1q_f32(xf.add(4 * j)));
                        a[j] = vfmsq_f32(v, mn[j + 4], vld1q_f32(xf.add(4 * j + 16)));
                    }
                }
            }
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
            Corr::None => {}
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
    let sc16 = !matches!(kind, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ4_XS);
    has_multi(kind).then_some(sc16 && t >= SMMLA_MIN_TOKENS)
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
        _ => return false,
    }
    true
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
