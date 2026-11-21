//! ggml block quantizers: f32 rows to the GGUF block formats, ported from llama.cpp's
//! `ggml/src/ggml-quants.c` (the inverse of [`crate::ggml::dequantize_row`]).
//!
//! [`quantize_row`] dispatches as `ggml_quantize_chunk` does: without an importance matrix the
//! deterministic `quantize_row_*_ref` paths (or, for the types that have no reference path, the
//! `quantize_row_*_impl(.., NULL)` ones, which weight by `x^2`), with one the weighted `_impl`
//! paths. The arithmetic follows the C code operation by operation in IEEE single precision, so
//! the bytes equal those of a llama.cpp build that does not contract `a * b + c` into fused
//! multiply-adds (`-ffp-contract=off`); builds that do (clang's default on arm64) round a few
//! intermediate values differently, which flips a small share of near-tie decisions.
//!
//! Differences from llama.cpp, on purpose:
//! - IQ2_XXS and IQ2_XS need an importance matrix there (`ggml_quantize_requires_imatrix`; the
//!   quantizer aborts without one). Here a missing one is taken as all ones, which weights each
//!   value by `sqrt(sigma^2 + x^2)`, the other types' weighting with a flat matrix.
//! - The IQ2 nearest-neighbour search runs exhaustively over the candidate list; llama.cpp
//!   caches per-pattern coefficients but falls back to the same exhaustive search whenever the
//!   cache could disagree with it, so the result is the same.
//! - IQ3_S: an all-zero group of 32 values advances the grid-index and sign pointers here; in
//!   llama.cpp it does not, which shifts the following groups' codes into its slot (an encoder
//!   bug that only shows on rows with zero groups).
//!
//! [`Fit::Best`] departs from llama.cpp on purpose where its quantizer only rounds against the
//! absolute maximum (Q2_0), with a least-squares fit; LoRA merges ([`crate::model::Lin::merge_lora`])
//! use it, since re-quantizing a file made by a better quantizer with the crude one costs far more
//! precision than the update adds.

use crate::ggml::tables::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KVALUES_IQ4NL, KVALUES_MXFP4};
use crate::ggml::{GgmlType, QK_K};
use std::sync::OnceLock;

const GROUP_MAX_EPS: f32 = 1e-15;
const GROUP_MAX_EPS_IQ3_XXS: f32 = 1e-8;
const GROUP_MAX_EPS_IQ2_S: f32 = 1e-8;

/// Whether [`quantize_row`] can write `kind`.
pub fn can_quantize(kind: GgmlType) -> bool {
    use GgmlType::*;
    matches!(
        kind,
        F32 | F16
            | BF16
            | Q4_0
            | Q4_1
            | Q5_0
            | Q5_1
            | Q8_0
            | Q2_K
            | Q3_K
            | Q4_K
            | Q5_K
            | Q6_K
            | IQ2_XXS
            | IQ2_XS
            | IQ2_S
            | IQ3_XXS
            | IQ3_S
            | IQ4_NL
            | IQ4_XS
            | TQ1_0
            | TQ2_0
            | Q1_0
            | Q2_0
            | MXFP4
    )
}

/// Whether llama.cpp requires an importance matrix for `kind` (here a missing one is all ones).
pub fn wants_imatrix(kind: GgmlType) -> bool {
    matches!(kind, GgmlType::IQ2_XXS | GgmlType::IQ2_XS)
}

/// How closely [`quantize_row_with`] fits where llama.cpp's quantizer is crude.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Fit {
    /// llama.cpp's bytes (see the module docs).
    #[default]
    Llama,
    /// A least-squares fit where llama.cpp only rounds against the absolute maximum: Q2_0
    /// (whose reference quantizer never uses its `+2d` level and leaves most values at 0). Files
    /// made with better Q2_0 quantizers (AtomicChat's Inkling-2 IQ2_M experts use the `+2d` level
    /// for 6% of the values) keep their precision when re-quantized this way, as LoRA merges do.
    Best,
}

/// Quantizes the row `src` into `dst` (`kind.row_bytes(src.len())` bytes) as `kind`, with the
/// importance matrix `imatrix` (one weight per column, as llama.cpp's `quant_weights`) when given:
/// llama.cpp's bytes ([`Fit::Llama`]). Panics on a type [`can_quantize`] refuses or on mismatched
/// lengths.
pub fn quantize_row(kind: GgmlType, src: &[f32], dst: &mut [u8], imatrix: Option<&[f32]>) {
    quantize_row_with(kind, src, dst, imatrix, Fit::Llama)
}

/// [`quantize_row`] with the choice of [`Fit`].
pub fn quantize_row_with(kind: GgmlType, src: &[f32], dst: &mut [u8], imatrix: Option<&[f32]>, fit: Fit) {
    use GgmlType::*;
    let n = src.len();
    assert!(can_quantize(kind), "no quantizer for {kind}");
    assert!(n.is_multiple_of(kind.block_size()), "{kind}: row of {n} is not whole blocks");
    assert_eq!(dst.len(), kind.row_bytes(n), "{kind}: output size");
    if let Some(q) = imatrix {
        assert_eq!(q.len(), n, "{kind}: importance matrix length");
    }
    let mut blocks = |bs: usize, f: &dyn Fn(&[f32], &mut [u8])| {
        for (x, y) in src.chunks_exact(bs).zip(dst.chunks_exact_mut(kind.type_size())) {
            f(x, y);
        }
    };
    match (kind, imatrix) {
        (F32, _) => {
            for (o, v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                o.copy_from_slice(&v.to_le_bytes());
            }
        }
        (F16, _) => {
            for (o, v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                o.copy_from_slice(&f16b(*v));
            }
        }
        (BF16, _) => {
            for (o, v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                o.copy_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
            }
        }
        (Q4_0, None) => blocks(32, &q4_0_ref),
        (Q4_0, Some(q)) => q4_0_impl(src, dst, q),
        (Q4_1, None) => blocks(32, &q4_1_ref),
        (Q4_1, Some(q)) => q4_1_impl(src, dst, q),
        (Q5_0, None) => blocks(32, &q5_0_ref),
        (Q5_0, Some(q)) => q5_0_impl(src, dst, q),
        (Q5_1, None) => blocks(32, &q5_1_ref),
        (Q5_1, Some(q)) => q5_1_impl(src, dst, q),
        (Q8_0, _) => blocks(32, &q8_0_ref),
        (Q2_K, None) => blocks(QK_K, &q2_k_ref),
        (Q3_K, None) => blocks(QK_K, &q3_k_ref),
        (Q4_K, None) => blocks(QK_K, &q4_k_ref),
        (Q5_K, None) => blocks(QK_K, &q5_k_ref),
        (Q6_K, None) => blocks(QK_K, &|x, y| q6_k_impl(x, y, None)),
        (Q2_K | Q3_K | Q4_K | Q5_K | Q6_K, Some(q)) => {
            for ((x, y), qw) in src
                .as_chunks::<QK_K>()
                .0
                .iter()
                .zip(dst.chunks_exact_mut(kind.type_size()))
                .zip(q.as_chunks::<QK_K>().0.iter())
            {
                match kind {
                    Q2_K => q2_k_impl(x, y, qw),
                    Q3_K => q3_k_impl(x, y, qw),
                    Q4_K => q45_k_impl(x, y, qw, false),
                    Q5_K => q45_k_impl(x, y, qw, true),
                    _ => q6_k_impl(x, y, Some(qw)),
                }
            }
        }
        (TQ1_0, _) => blocks(QK_K, &tq1_0_ref),
        (TQ2_0, _) => blocks(QK_K, &tq2_0_ref),
        (Q1_0, _) => blocks(128, &q1_0_ref),
        (Q2_0, _) if fit == Fit::Best => {
            for (i, (x, y)) in src
                .as_chunks::<64>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<18>().0.iter_mut())
                .enumerate()
            {
                q2_0_best(x, y, imatrix.map(|q| &q[64 * i..64 * (i + 1)]));
            }
        }
        (Q2_0, _) => blocks(64, &q2_0_ref),
        (MXFP4, _) => blocks(32, &mxfp4_ref),
        (IQ4_NL | IQ4_XS, _) => {
            let (sb, bt) = if kind == IQ4_NL { (32, 18) } else { (QK_K, 136) };
            for (i, (x, y)) in src.chunks_exact(sb).zip(dst.chunks_exact_mut(bt)).enumerate() {
                iq4_impl(x, y, imatrix.map(|q| &q[i * sb..(i + 1) * sb]), 7);
            }
        }
        (IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S, _) => {
            let ones;
            let qw = match imatrix {
                Some(q) => Some(q),
                None if wants_imatrix(kind) => {
                    ones = vec![1f32; n];
                    Some(&ones[..])
                }
                None => None,
            };
            for (i, (x, y)) in src
                .as_chunks::<QK_K>()
                .0
                .iter()
                .zip(dst.chunks_exact_mut(kind.type_size()))
                .enumerate()
            {
                let qw = qw.map(|q| &q[i * QK_K..(i + 1) * QK_K]);
                match kind {
                    IQ2_XXS => iq2_xxs_block(x, y, qw.unwrap()),
                    IQ2_XS => iq2_xs_block(x, y, qw.unwrap()),
                    IQ2_S => iq2_s_block(x, y, qw),
                    IQ3_XXS => iq3_xxs_block(x, y, qw),
                    _ => iq3_s_block(x, y, qw),
                }
            }
        }
        _ => unreachable!(),
    }
}

/// Quantizes the `rows x n_per_row` matrix `src` (row-major) as `kind`, rows in parallel on the
/// thread pool. `imatrix`, when given, has one weight per column.
pub fn quantize_matrix(kind: GgmlType, src: &[f32], n_per_row: usize, imatrix: Option<&[f32]>) -> Vec<u8> {
    assert!(n_per_row > 0 && src.len().is_multiple_of(n_per_row), "matrix shape");
    let mut out = vec![0u8; src.len() / n_per_row * kind.row_bytes(n_per_row)];
    quantize_matrix_into(kind, src, n_per_row, imatrix, Fit::Llama, &mut out);
    out
}

/// [`quantize_matrix`] into `out` (`rows * kind.row_bytes(n_per_row)` bytes), with the choice of
/// [`Fit`].
pub fn quantize_matrix_into(kind: GgmlType, src: &[f32], n_per_row: usize, imatrix: Option<&[f32]>, fit: Fit, out: &mut [u8]) {
    assert!(n_per_row > 0 && src.len().is_multiple_of(n_per_row), "matrix shape");
    let rows = src.len() / n_per_row;
    let rb = kind.row_bytes(n_per_row);
    assert_eq!(out.len(), rows * rb, "{kind}: output size");
    let op = SendPtr(out.as_mut_ptr());
    const ROWS: usize = 16;
    crate::pool::run(rows.div_ceil(ROWS), &|j| {
        for r in j * ROWS..((j + 1) * ROWS).min(rows) {
            // SAFETY: each row of `out` is written by one item.
            let y = unsafe { std::slice::from_raw_parts_mut(op.get().add(r * rb), rb) };
            quantize_row_with(kind, &src[r * n_per_row..(r + 1) * n_per_row], y, imatrix, fit);
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut u8 {
        self.0
    }
}

// ===================================================================== helpers

#[inline(always)]
fn f16b(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_le_bytes()
}

/// `x` rounded to f16 and back (what the C code reads back from a block's `d`).
#[inline(always)]
fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// ggml's `nearest_int`: round half to even through the 1.5 * 2^23 trick (|x| < 2^22).
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    (val.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

fn make_qx_quants(n: usize, nmax: i32, x: &[f32], l_out: &mut [i8], rmse_type: i32, qw: Option<&[f32]>) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for &v in &x[..n] {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l_out[..n].fill(0);
        return 0.0;
    }
    let nmaxf = nmax as f32;
    let mut iscale = -nmaxf / max;
    if rmse_type == 0 {
        for i in 0..n {
            let l = nearest_int(iscale * x[i]);
            l_out[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rmse_type = rmse_type;
    let mut return_early = false;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    let wt = |i: usize| -> f32 {
        match qw {
            Some(q) => q[i],
            None => match rmse_type {
                1 => x[i] * x[i],
                2 => 1.0,
                3 => x[i].abs(),
                _ => x[i].abs().sqrt(),
            },
        }
    };
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l_out[i] = (l + nmax) as i8;
        let w = wt(i);
        let lf = l as f32;
        sumlx += w * x[i] * lf;
        suml2 += w * lf * lf;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 { 0.5 * (scale + 1.0 / iscale) } else { 1.0 / iscale };
    }
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmaxf + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1) as f32;
            let w = wt(i);
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let l = nearest_int(iscale * x[i]);
                l_out[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

fn make_q3_quants(n: usize, nmax: i32, x: &[f32], l_out: &mut [i8], do_rmse: bool) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for &v in &x[..n] {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l_out[..n].fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0f32;
        let mut suml2 = 0f32;
        for i in 0..n {
            let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            l_out[i] = l as i8;
            let w = x[i] * x[i];
            let lf = l as f32;
            sumlx += w * x[i] * lf;
            suml2 += w * lf * lf;
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let li = l_out[i] as f32;
                let mut slx = sumlx - w * x[i] * li;
                if slx > 0.0 {
                    let mut sl2 = suml2 - w * li * li;
                    let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                    if new_l != l_out[i] as i32 {
                        let nl = new_l as f32;
                        slx += w * x[i] * nl;
                        sl2 += w * nl * nl;
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l_out[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for v in &mut l_out[..n] {
            *v += nmax as i8;
        }
        return if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    }
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l_out[i] = (l + nmax) as i8;
    }
    1.0 / iscale
}

#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l_out: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l_out[..n].fill(0);
        *the_min = -min;
        return 0.0;
    }
    let nmaxf = nmax as f32;
    let mut iscale = nmaxf / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        l_out[i] = l.clamp(0, nmax) as u8;
        let mut diff = scale * l_out[i] as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmaxf) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = l as u8;
            let w = weights[i];
            let lf = l as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                l_out[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

#[allow(clippy::too_many_arguments)]
fn make_qkx3_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: Option<&[f32]>,
    l_out: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let wt = |i: usize| weights.map_or(x[i] * x[i], |w| w[i]);
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = wt(0);
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = wt(i);
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        l_out[..n].fill(0);
        *the_min = -min;
        return 0.0;
    }
    let nmaxf = nmax as f32;
    let mut iscale = nmaxf / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        l_out[i] = l.clamp(0, nmax) as u8;
        let mut diff = scale * l_out[i] as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_mad += wt(i) * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmaxf) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let v = iscale * (x[i] - min);
            let l = if v > 0.0 {
                nearest_int(if v < nmaxf { v } else { nmaxf })
            } else {
                0
            };
            laux[i] = l as u8;
            let w = wt(i);
            let lf = l as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                mad += wt(i) * diff;
            }
            if mad < best_mad {
                l_out[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l_out: &mut [u8], qw: &[f32]) -> f32 {
    let mut max = 0f32;
    for &v in &x[..n] {
        max = if max > v { max } else { v };
    }
    if max < GROUP_MAX_EPS {
        l_out[..n].fill(0);
        return 0.0;
    }
    let nmaxf = nmax as f32;
    let mut iscale = nmaxf / max;
    for i in 0..n {
        l_out[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0f32;
    for i in 0..n {
        let diff = x[i] - scale * l_out[i] as f32;
        best_mse += qw[i] * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmaxf) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0f32;
        for i in 0..n {
            let l = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * l as f32;
            mse += qw[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).min(nmax);
        l_out[i] = l as u8;
        // (The sums take the signed level; the refinement below reads the stored byte.)
        let lf = l as f32;
        sumlx += qw[i] * x[i] * lf;
        suml2 += qw[i] * lf * lf;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = qw[i];
            let li = l_out[i] as f32;
            let mut slx = sumlx - w * x[i] * li;
            let mut sl2 = suml2 - w * li * li;
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l != l_out[i] as i32 {
                    let nl = new_l as f32;
                    slx += w * x[i] * nl;
                    sl2 += w * nl * nl;
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l_out[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

/// `sum x^2 / n` over a row, in order.
fn mean_sq(x: &[f32]) -> f32 {
    let mut s = 0f32;
    for &v in x {
        s += v * v;
    }
    s / x.len() as f32
}

// ===================================================================== 32-element blocks

fn absmax_signed(x: &[f32]) -> f32 {
    let (mut amax, mut max) = (0f32, 0f32);
    for &v in x {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    max
}

fn q4_0_ref(x: &[f32], y: &mut [u8]) {
    let d = absmax_signed(x) / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    y[..2].copy_from_slice(&f16b(d));
    for j in 0..16 {
        let x0 = x[j] * id;
        let x1 = x[16 + j] * id;
        let xi0 = ((x0 + 8.5) as i8).min(15) as u8;
        let xi1 = ((x1 + 8.5) as i8).min(15) as u8;
        y[2 + j] = xi0 | (xi1 << 4);
    }
}

fn min_max(x: &[f32]) -> (f32, f32) {
    let (mut min, mut max) = (f32::MAX, -f32::MAX);
    for &v in x {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

fn q4_1_ref(x: &[f32], y: &mut [u8]) {
    let (min, max) = min_max(x);
    let d = (max - min) / 15.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    y[..2].copy_from_slice(&f16b(d));
    y[2..4].copy_from_slice(&f16b(min));
    for j in 0..16 {
        let x0 = (x[j] - min) * id;
        let x1 = (x[16 + j] - min) * id;
        let xi0 = ((x0 + 0.5) as i8).min(15) as u8;
        let xi1 = ((x1 + 0.5) as i8).min(15) as u8;
        y[4 + j] = xi0 | (xi1 << 4);
    }
}

/// Stores 5-bit codes `l` of a Q5_0 / Q5_1 block: low nibbles into `qs`, fifth bits into `qh`.
fn put_q5(l: impl Fn(usize) -> u8, qh: &mut [u8], qs: &mut [u8]) {
    let mut h = 0u32;
    for j in 0..16 {
        let (xi0, xi1) = (l(j), l(j + 16));
        qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        h |= (((xi0 as u32) & 0x10) >> 4) << j;
        h |= (((xi1 as u32) & 0x10) >> 4) << (j + 16);
    }
    qh.copy_from_slice(&h.to_le_bytes());
}

fn q5_0_ref(x: &[f32], y: &mut [u8]) {
    let d = absmax_signed(x) / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    y[..2].copy_from_slice(&f16b(d));
    let (qh, qs) = y[2..].split_at_mut(4);
    put_q5(|j| ((x[j] * id + 16.5) as i8).min(31) as u8, qh, qs);
}

fn q5_1_ref(x: &[f32], y: &mut [u8]) {
    let (min, max) = min_max(x);
    let d = (max - min) / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    y[..2].copy_from_slice(&f16b(d));
    y[2..4].copy_from_slice(&f16b(min));
    let (qh, qs) = y[4..].split_at_mut(4);
    put_q5(|j| ((x[j] - min) * id + 0.5) as u8, qh, qs);
}

fn q8_0_ref(x: &[f32], y: &mut [u8]) {
    let mut amax = 0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    y[..2].copy_from_slice(&f16b(d));
    for j in 0..32 {
        y[2 + j] = (x[j] * id).round() as i8 as u8;
    }
}

/// The per-value weights of the imatrix paths of the 32-blocks: `qw * sqrt(sigma2 + x^2)`.
fn block_weights(xb: &[f32], qw: &[f32], sigma2: f32, w: &mut [f32]) {
    for j in 0..xb.len() {
        w[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
    }
}

fn q4_0_impl(x: &[f32], y: &mut [u8], qw: &[f32]) {
    let sigma2 = mean_sq(x);
    let (mut weight, mut l) = ([0f32; 32], [0i8; 32]);
    for (ib, yb) in y.as_chunks_mut::<18>().0.iter_mut().enumerate() {
        let xb = &x[32 * ib..32 * ib + 32];
        block_weights(xb, &qw[32 * ib..], sigma2, &mut weight);
        let d = make_qx_quants(32, 8, xb, &mut l, 1, Some(&weight));
        yb[..2].copy_from_slice(&f16b(d));
        for j in 0..16 {
            yb[2 + j] = (l[j] as u8) | ((l[j + 16] as u8) << 4);
        }
    }
}

fn q4_1_impl(x: &[f32], y: &mut [u8], qw: &[f32]) {
    let sigma2 = mean_sq(x);
    let (mut weight, mut l, mut laux) = ([0f32; 32], [0u8; 32], [0u8; 32]);
    for (ib, yb) in y.as_chunks_mut::<20>().0.iter_mut().enumerate() {
        let xb = &x[32 * ib..32 * ib + 32];
        block_weights(xb, &qw[32 * ib..], sigma2, &mut weight);
        let mut min = 0f32;
        let d = make_qkx3_quants(32, 15, xb, Some(&weight), &mut l, &mut min, &mut laux, -0.9, 0.05, 36, false);
        yb[..2].copy_from_slice(&f16b(d));
        yb[2..4].copy_from_slice(&f16b(-min));
        for j in 0..16 {
            yb[4 + j] = l[j] | (l[j + 16] << 4);
        }
    }
}

fn q5_0_impl(x: &[f32], y: &mut [u8], qw: &[f32]) {
    let sigma2 = mean_sq(x);
    let (mut weight, mut l) = ([0f32; 32], [0i8; 32]);
    for (ib, yb) in y.as_chunks_mut::<22>().0.iter_mut().enumerate() {
        let xb = &x[32 * ib..32 * ib + 32];
        block_weights(xb, &qw[32 * ib..], sigma2, &mut weight);
        let d = make_qx_quants(32, 16, xb, &mut l, 1, Some(&weight));
        yb[..2].copy_from_slice(&f16b(d));
        let (qh, qs) = yb[2..].split_at_mut(4);
        put_q5(|j| l[j] as u8, qh, qs);
    }
}

fn q5_1_impl(x: &[f32], y: &mut [u8], qw: &[f32]) {
    let sigma2 = mean_sq(x);
    let (mut weight, mut l, mut laux) = ([0f32; 32], [0u8; 32], [0u8; 32]);
    for (ib, yb) in y.as_chunks_mut::<24>().0.iter_mut().enumerate() {
        let xb = &x[32 * ib..32 * ib + 32];
        block_weights(xb, &qw[32 * ib..], sigma2, &mut weight);
        let mut min = 0f32;
        let d = make_qkx3_quants(32, 31, xb, Some(&weight), &mut l, &mut min, &mut laux, -0.9, 0.05, 36, false);
        yb[..2].copy_from_slice(&f16b(d));
        yb[2..4].copy_from_slice(&f16b(-min));
        let (qh, qs) = yb[4..].split_at_mut(4);
        put_q5(|j| l[j], qh, qs);
    }
}

// ===================================================================== k-quants

/// Packs 2-bit codes `l` (256) into Q2_K / Q3_K `qs` (64 bytes).
fn pack_2bit(l: &[u8; QK_K], qs: &mut [u8]) {
    for j in (0..QK_K).step_by(128) {
        for k in 0..32 {
            qs[j / 4 + k] = l[j + k] | (l[j + k + 32] << 2) | (l[j + k + 64] << 4) | (l[j + k + 96] << 6);
        }
    }
}

fn q2_k_ref(x: &[f32], y: &mut [u8]) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 16];
    let mut weights = [0f32; 16];
    let mut mins = [0f32; 16];
    let mut scales = [0f32; 16];
    let q4scale = 15f32;
    let (mut max_scale, mut max_min) = (0f32, 0f32);
    for j in 0..16 {
        for k in 0..16 {
            weights[k] = x[16 * j + k].abs();
        }
        scales[j] = make_qkx2_quants(
            16,
            3,
            &x[16 * j..],
            &weights,
            &mut l[16 * j..],
            &mut mins[j],
            &mut laux,
            -0.5,
            0.1,
            15,
            true,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }
    let (sc, rest) = y.split_at_mut(16);
    let (qs, dd) = rest.split_at_mut(64);
    if max_scale > 0.0 {
        let iscale = q4scale / max_scale;
        for j in 0..16 {
            sc[j] = nearest_int(iscale * scales[j]) as u8;
        }
        dd[..2].copy_from_slice(&f16b(max_scale / q4scale));
    } else {
        sc.fill(0);
        dd[..2].copy_from_slice(&f16b(0.0));
    }
    if max_min > 0.0 {
        let iscale = q4scale / max_min;
        for j in 0..16 {
            let m = nearest_int(iscale * mins[j]);
            sc[j] |= (m << 4) as u8;
        }
        dd[2..4].copy_from_slice(&f16b(max_min / q4scale));
    } else {
        dd[2..4].copy_from_slice(&f16b(0.0));
    }
    let d0 = half::f16::from_le_bytes([dd[0], dd[1]]).to_f32();
    let m0 = half::f16::from_le_bytes([dd[2], dd[3]]).to_f32();
    for j in 0..16 {
        let d = d0 * (sc[j] & 0xF) as f32;
        if d == 0.0 {
            continue;
        }
        let dm = m0 * (sc[j] >> 4) as f32;
        for ii in 0..16 {
            l[16 * j + ii] = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3) as u8;
        }
    }
    pack_2bit(&l, qs);
}

fn q2_k_impl(x: &[f32], y: &mut [u8], qw: &[f32]) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 16];
    let mut mins = [0f32; 16];
    let mut scales = [0f32; 16];
    let mut sw = [0f32; 16];
    let mut weight = [0f32; 16];
    let (mut ls, mut lm) = ([0u8; 16], [0u8; 16]);
    let sigma2 = mean_sq(&x[..QK_K]);
    for j in 0..16 {
        for k in 0..16 {
            weight[k] = qw[16 * j + k] * (sigma2 + x[16 * j + k] * x[16 * j + k]).sqrt();
        }
        for &w in &weight {
            sw[j] += w;
        }
        scales[j] = make_qkx3_quants(
            16,
            3,
            &x[16 * j..],
            Some(&weight),
            &mut l[16 * j..],
            &mut mins[j],
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
    }
    let dm = make_qp_quants(16, 15, &scales, &mut ls, &sw);
    let mm = make_qp_quants(16, 15, &mins, &mut lm, &sw);
    let (sc, rest) = y.split_at_mut(16);
    let (qs, dd) = rest.split_at_mut(64);
    dd[..2].copy_from_slice(&f16b(dm));
    dd[2..4].copy_from_slice(&f16b(mm));
    let (dm, mm) = (f16r(dm), f16r(mm));
    for j in 0..16 {
        sc[j] = ls[j] | (lm[j] << 4);
    }
    for j in 0..16 {
        let d = dm * (sc[j] & 0xF) as f32;
        if d == 0.0 {
            continue;
        }
        let m = mm * (sc[j] >> 4) as f32;
        for ii in 0..16 {
            l[16 * j + ii] = nearest_int((x[16 * j + ii] + m) / d).clamp(0, 3) as u8;
        }
    }
    pack_2bit(&l, qs);
}

/// Q3_K: stores the 16 6-bit scale codes `ls` (0..63) into the 12 scale bytes.
fn put_q3k_scales(ls: impl Fn(usize) -> i32, s: &mut [u8]) {
    s[..12].fill(0);
    for j in 0..16 {
        let l = ls(j);
        if j < 8 {
            s[j] = (l & 0xF) as u8;
        } else {
            s[j - 8] |= ((l & 0xF) << 4) as u8;
        }
        let h = l >> 4;
        s[j % 4 + 8] |= (h << (2 * (j / 4))) as u8;
    }
}

/// Q3_K: the 16 scales re-read from the stored bytes, then the 3-bit codes and their packing.
fn q3k_finish(x: &[f32], l: &mut [i8; QK_K], y: &mut [u8]) {
    let (hmask, rest) = y.split_at_mut(32);
    let (qs, rest) = rest.split_at_mut(64);
    let (s, dd) = rest.split_at_mut(12);
    let d_all = half::f16::from_le_bytes([dd[0], dd[1]]).to_f32();
    for j in 0..16 {
        let sc = if j < 8 { s[j] & 0xF } else { s[j - 8] >> 4 } as i32;
        let sc = (sc | (((s[8 + j % 4] >> (2 * (j / 4))) & 3) as i32) << 4) - 32;
        let d = d_all * sc as f32;
        if d == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let v = nearest_int(x[16 * j + ii] / d).clamp(-4, 3);
            l[16 * j + ii] = (v + 4) as i8;
        }
    }
    hmask.fill(0);
    let (mut m, mut hm) = (0usize, 1u8);
    for j in 0..QK_K {
        if l[j] > 3 {
            hmask[m] |= hm;
            l[j] -= 4;
        }
        m += 1;
        if m == QK_K / 8 {
            m = 0;
            hm <<= 1;
        }
    }
    let lu: [u8; QK_K] = std::array::from_fn(|i| l[i] as u8);
    pack_2bit(&lu, qs);
}

fn q3_k_ref(x: &[f32], y: &mut [u8]) {
    let mut l = [0i8; QK_K];
    let mut scales = [0f32; 16];
    let (mut max_scale, mut amax) = (0f32, 0f32);
    for j in 0..16 {
        scales[j] = make_q3_quants(16, 4, &x[16 * j..], &mut l[16 * j..], true);
        let scale = scales[j].abs();
        if scale > amax {
            amax = scale;
            max_scale = scales[j];
        }
    }
    if max_scale != 0.0 {
        let iscale = -32.0 / max_scale;
        put_q3k_scales(
            |j| (nearest_int(iscale * scales[j]) as i8 as i32).clamp(-32, 31) + 32,
            &mut y[96..108],
        );
        y[108..110].copy_from_slice(&f16b(1.0 / iscale));
    } else {
        y[96..108].fill(0);
        y[108..110].copy_from_slice(&f16b(0.0));
    }
    q3k_finish(x, &mut l, y);
}

fn q3_k_impl(x: &[f32], y: &mut [u8], qw: &[f32]) {
    let mut l = [0i8; QK_K];
    let mut scales = [0f32; 16];
    let mut weight = [0f32; 16];
    let mut sw = [0f32; 16];
    let mut ls = [0i8; 16];
    let mut sumx2 = 0f32;
    for &v in &x[..QK_K] {
        sumx2 += v * v;
    }
    let sigma2 = 2.0 * sumx2 / QK_K as f32;
    for j in 0..16 {
        for k in 0..16 {
            weight[k] = qw[16 * j + k] * (sigma2 + x[16 * j + k] * x[16 * j + k]).sqrt();
        }
        let mut sumw = 0f32;
        for &w in &weight {
            sumw += w;
        }
        sw[j] = sumw;
        scales[j] = make_qx_quants(16, 4, &x[16 * j..], &mut l[16 * j..], 1, Some(&weight));
    }
    let d_block = make_qx_quants(16, 32, &scales, &mut ls, 1, Some(&sw));
    put_q3k_scales(|j| ls[j] as i32, &mut y[96..108]);
    y[108..110].copy_from_slice(&f16b(d_block));
    q3k_finish(x, &mut l, y);
}

/// Q4_K / Q5_K: stores the eight 6-bit scales and mins.
fn put_k4_scales(ls: &[u8; 8], lm: &[u8; 8], s: &mut [u8]) {
    for j in 0..8 {
        let (a, m) = (ls[j], lm[j]);
        if j < 4 {
            s[j] = a;
            s[j + 4] = m;
        } else {
            s[j + 4] = (a & 0xF) | ((m & 0xF) << 4);
            s[j - 4] |= (a >> 4) << 6;
            s[j] |= (m >> 4) << 6;
        }
    }
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Q4_K / Q5_K after the scales are stored: the final codes and their packing.
fn k45_finish(x: &[f32], l: &mut [u8; QK_K], y: &mut [u8], five: bool) {
    let d0 = half::f16::from_le_bytes([y[0], y[1]]).to_f32();
    let m0 = half::f16::from_le_bytes([y[2], y[3]]).to_f32();
    let nmax = if five { 31 } else { 15 };
    for j in 0..8 {
        let (sc, m) = scale_min_k4(j, &y[4..16]);
        let d = d0 * sc as f32;
        if d == 0.0 {
            continue;
        }
        let dm = m0 * m as f32;
        for ii in 0..32 {
            l[32 * j + ii] = nearest_int((x[32 * j + ii] + dm) / d).clamp(0, nmax) as u8;
        }
    }
    if five {
        let (qh, ql) = y[16..].split_at_mut(32);
        qh.fill(0);
        let (mut m1, mut m2) = (1u8, 2u8);
        for (c, n) in (0..QK_K).step_by(64).enumerate() {
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[32 * c + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
    } else {
        let q = &mut y[16..];
        for (c, j) in (0..QK_K).step_by(64).enumerate() {
            for k in 0..32 {
                q[32 * c + k] = l[j + k] | (l[j + k + 32] << 4);
            }
        }
    }
}

fn q45_k_ref(x: &[f32], y: &mut [u8], five: bool) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    let mut mins = [0f32; 8];
    let mut scales = [0f32; 8];
    let (mut max_scale, mut max_min) = (0f32, 0f32);
    let (nmax, rmin, nstep) = if five { (31, -0.5, 15) } else { (15, -1.0, 20) };
    for j in 0..8 {
        let mut sum_x2 = 0f32;
        for k in 0..32 {
            sum_x2 += x[32 * j + k] * x[32 * j + k];
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for k in 0..32 {
            weights[k] = av_x + x[32 * j + k].abs();
        }
        scales[j] = make_qkx2_quants(
            32,
            nmax,
            &x[32 * j..],
            &weights,
            &mut l[32 * j..],
            &mut mins[j],
            &mut laux,
            rmin,
            0.1,
            nstep,
            false,
        );
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let ls: [u8; 8] = std::array::from_fn(|j| (nearest_int(inv_scale * scales[j]) as u8).min(63));
    let lm: [u8; 8] = std::array::from_fn(|j| (nearest_int(inv_min * mins[j]) as u8).min(63));
    put_k4_scales(&ls, &lm, &mut y[4..16]);
    y[..2].copy_from_slice(&f16b(max_scale / 63.0));
    y[2..4].copy_from_slice(&f16b(max_min / 63.0));
    k45_finish(x, &mut l, y, five);
}

fn q4_k_ref(x: &[f32], y: &mut [u8]) {
    q45_k_ref(x, y, false)
}

fn q5_k_ref(x: &[f32], y: &mut [u8]) {
    q45_k_ref(x, y, true)
}

fn q45_k_impl(x: &[f32], y: &mut [u8], qw: &[f32], five: bool) {
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let (mut ls, mut lm) = ([0u8; 8], [0u8; 8]);
    let mut weights = [0f32; 32];
    let mut sw = [0f32; 8];
    let mut mins = [0f32; 8];
    let mut scales = [0f32; 8];
    let mut sum_x2 = 0f32;
    for &v in &x[..QK_K] {
        sum_x2 += v * v;
    }
    let sigma2 = 2.0 * sum_x2 / QK_K as f32;
    let nmax = if five { 31 } else { 15 };
    for j in 0..8 {
        for k in 0..32 {
            weights[k] = qw[32 * j + k] * (sigma2 + x[32 * j + k] * x[32 * j + k]).sqrt();
        }
        let mut sumw = 0f32;
        for &w in &weights {
            sumw += w;
        }
        sw[j] = sumw;
        scales[j] = make_qkx3_quants(
            32,
            nmax,
            &x[32 * j..],
            Some(&weights),
            &mut l[32 * j..],
            &mut mins[j],
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
    }
    let d_block = make_qp_quants(8, 63, &scales, &mut ls, &sw);
    let m_block = make_qp_quants(8, 63, &mins, &mut lm, &sw);
    if five {
        for v in ls.iter_mut().chain(lm.iter_mut()) {
            *v = (*v).min(63);
        }
    }
    put_k4_scales(&ls, &lm, &mut y[4..16]);
    y[..2].copy_from_slice(&f16b(d_block));
    y[2..4].copy_from_slice(&f16b(m_block));
    k45_finish(x, &mut l, y, five);
}

fn q6_k_impl(x: &[f32], y: &mut [u8], qw: Option<&[f32]>) {
    let mut l = [0i8; QK_K];
    let mut scales = [0f32; 16];
    let (mut max_scale, mut max_abs_scale) = (0f32, 0f32);
    for ib in 0..16 {
        let scale = make_qx_quants(16, 32, &x[16 * ib..], &mut l[16 * ib..], 1, qw.map(|q| &q[16 * ib..16 * ib + 16]));
        scales[ib] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }
    if max_abs_scale < GROUP_MAX_EPS {
        y.fill(0);
        return;
    }
    let iscale = -128.0 / max_scale;
    y[208..210].copy_from_slice(&f16b(1.0 / iscale));
    let d0 = f16r(1.0 / iscale);
    for ib in 0..16 {
        y[192 + ib] = nearest_int(iscale * scales[ib]).min(127) as i8 as u8;
    }
    for j in 0..16 {
        let d = d0 * y[192 + j] as i8 as f32;
        if d == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let v = nearest_int(x[16 * j + ii] / d).clamp(-32, 31);
            l[16 * j + ii] = (v + 32) as i8;
        }
    }
    let (ql, rest) = y.split_at_mut(128);
    let qh = &mut rest[..64];
    for (c, j) in (0..QK_K).step_by(128).enumerate() {
        for k in 0..32 {
            let q = |o: usize| l[j + k + o] as u8;
            ql[64 * c + k] = (q(0) & 0xF) | ((q(64) & 0xF) << 4);
            ql[64 * c + k + 32] = (q(32) & 0xF) | ((q(96) & 0xF) << 4);
            qh[32 * c + k] = (q(0) >> 4) | ((q(32) >> 4) << 2) | ((q(64) >> 4) << 4) | ((q(96) >> 4) << 6);
        }
    }
}

// ===================================================================== ternary, 1-bit, 2-bit, FP4

fn absmax(x: &[f32]) -> f32 {
    let mut amax = 0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    amax
}

fn tq1_0_ref(x: &[f32], y: &mut [u8]) {
    let d = absmax(x);
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let trit = |v: f32| ((v * id).round() as i32 + 1) as u8;
    let ceil243 = |q: u8| (q as u16 * 256).div_ceil(243) as u8;
    let mut x = x;
    for m in 0..32 {
        let mut q = 0u8;
        for n in 0..5 {
            q = q.wrapping_mul(3).wrapping_add(trit(x[m + n * 32]));
        }
        y[m] = ceil243(q);
    }
    x = &x[160..];
    for m in 0..16 {
        let mut q = 0u8;
        for n in 0..5 {
            q = q.wrapping_mul(3).wrapping_add(trit(x[m + n * 16]));
        }
        y[32 + m] = ceil243(q);
    }
    x = &x[80..];
    for j in 0..4 {
        let mut q = 0u8;
        for m in 0..4 {
            q = q.wrapping_mul(3).wrapping_add(trit(x[j + m * 4]));
        }
        q = q.wrapping_mul(3);
        y[48 + j] = ceil243(q);
    }
    y[52..54].copy_from_slice(&f16b(d));
}

fn tq2_0_ref(x: &[f32], y: &mut [u8]) {
    let d = absmax(x);
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for j in (0..64).step_by(32) {
        let xb = &x[j * 4..];
        for m in 0..32 {
            let mut q = 0u8;
            for n in 0..4 {
                let xi = (xb[m + n * 32] * id).round() as i32 + 1;
                q = q.wrapping_add(((xi & 3) << (2 * n)) as u8);
            }
            y[j + m] = q;
        }
    }
    y[64..66].copy_from_slice(&f16b(d));
}

fn q1_0_ref(x: &[f32], y: &mut [u8]) {
    let mut sum_abs = 0f32;
    for &v in x {
        sum_abs += v.abs();
    }
    y[..2].copy_from_slice(&f16b(sum_abs / 128.0));
    y[2..18].fill(0);
    for (j, &v) in x.iter().enumerate() {
        if v >= 0.0 {
            y[2 + j / 8] |= 1 << (j % 8);
        }
    }
}

fn q2_0_ref(x: &[f32], y: &mut [u8]) {
    let mut amax = 0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
        }
    }
    let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };
    y[..2].copy_from_slice(&f16b(amax));
    y[2..18].fill(0);
    for (j, &w) in x.iter().enumerate() {
        let q = ((w * id).round() as i32 + 1).clamp(0, 3);
        y[2 + j / 4] |= (q as u8) << ((j % 4) * 2);
    }
}

/// Q2_0 by weighted least squares ([`Fit::Best`]): levels `{-1, 0, 1, 2} * d` with `d` of either
/// sign (the `2d` level on the side that needs it), the scale searched over a grid of 41 steps
/// from a quarter of the absolute maximum to 1.25 times it, each candidate's codes refitted to
/// their optimal scale, the best one refined twice more.
fn q2_0_best(x: &[f32], y: &mut [u8], qw: Option<&[f32]>) {
    let w = |i: usize| qw.map_or(1.0, |q| q[i]);
    let amax = absmax(x);
    y.fill(0);
    if amax < GROUP_MAX_EPS {
        return;
    }
    // (sign, scale) -> (error, refitted scale), codes in -1..=2 of `s * x / d`.
    let fit = |s: f32, d: f32| -> (f32, f32) {
        let (mut sxq, mut sq2) = (0f32, 0f32);
        for (i, &v) in x.iter().enumerate() {
            let q = (s * v / d).round().clamp(-1.0, 2.0);
            sxq += w(i) * s * v * q;
            sq2 += w(i) * q * q;
        }
        let d2 = if sq2 > 0.0 && sxq > 0.0 { sxq / sq2 } else { d };
        let mut err = 0f32;
        for (i, &v) in x.iter().enumerate() {
            let q = (s * v / d2).round().clamp(-1.0, 2.0);
            let e = s * v - d2 * q;
            err += w(i) * e * e;
        }
        (err, d2)
    };
    let (mut best, mut bs, mut bd) = (f32::MAX, 1f32, amax);
    for s in [1f32, -1.0] {
        for t in 0..=40 {
            let (e, d) = fit(s, amax * (0.25 + 0.025 * t as f32));
            if e < best {
                (best, bs, bd) = (e, s, d);
            }
        }
    }
    for _ in 0..2 {
        let (e, d) = fit(bs, bd);
        if e < best {
            (best, bd) = (e, d);
        }
    }
    let d = f16r(bs * bd);
    y[..2].copy_from_slice(&f16b(bs * bd));
    if d == 0.0 {
        return;
    }
    for (j, &v) in x.iter().enumerate() {
        let q = ((v / d).round() as i32 + 1).clamp(0, 3);
        y[2 + j / 4] |= (q as u8) << ((j % 4) * 2);
    }
}

fn mxfp4_ref(x: &[f32], y: &mut [u8]) {
    let amax = absmax(x);
    let e = if amax > 0.0 { (amax.log2().floor() - 2.0 + 127.0) as u8 } else { 0 };
    let d = crate::ggml::e8m0_half(e);
    y[0] = e;
    let best = |v: f32| -> u8 {
        let mut bi = 0;
        let mut be = (KVALUES_MXFP4[0] as f32 * d - v).abs();
        for (i, &k) in KVALUES_MXFP4.iter().enumerate().skip(1) {
            let err = (k as f32 * d - v).abs();
            if err < be {
                bi = i;
                be = err;
            }
        }
        bi as u8
    };
    for j in 0..16 {
        y[1 + j] = best(x[j]) | (best(x[16 + j]) << 4);
    }
}

// ===================================================================== IQ4_NL / IQ4_XS

#[inline(always)]
fn best_index_iq4nl(mid: &[f32; 15], x: f32) -> usize {
    let mut l = 15usize;
    for &m in mid {
        l -= (x < m) as usize;
    }
    l
}

/// `quantize_row_iq4_nl_impl` for one super-block: IQ4_NL (32 values, `y` 18 bytes) or IQ4_XS
/// (256 values, `y` 136 bytes), with `ntry` scale tries (7 in `ggml_quantize_chunk`).
fn iq4_impl(x: &[f32], y: &mut [u8], qw: Option<&[f32]>, ntry: i32) {
    let sbs = x.len();
    let nb = sbs / 32;
    let values = &KVALUES_IQ4NL;
    let mut sigma2 = 0f32;
    for &v in x {
        sigma2 += v * v;
    }
    sigma2 *= 2.0 / sbs as f32;
    let mut l = [0u8; QK_K];
    let mut scales = [0f32; 8];
    let mut weight = [0f32; 32];
    let mid: [f32; 15] = std::array::from_fn(|k| 0.5 * (values[k] as i32 + values[k + 1] as i32) as f32);
    let fval: [f32; 16] = std::array::from_fn(|k| values[k] as f32);
    let v0 = values[0] as i32;
    let (mut max_scale, mut amax_scale) = (0f32, 0f32);
    for ib in 0..nb {
        let xb = &x[32 * ib..32 * ib + 32];
        match qw {
            Some(q) => block_weights(xb, &q[32 * ib..], sigma2, &mut weight),
            None => {
                for j in 0..32 {
                    weight[j] = xb[j] * xb[j];
                }
            }
        }
        let max = absmax_signed(xb);
        if max.abs() < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let lb = &mut l[32 * ib..32 * ib + 32];
        let mut d = if ntry > 0 { -max / v0 as f32 } else { max / v0 as f32 };
        let nt = if ntry >= 0 { (2 * ntry + 2) as usize } else { 1 };
        let mut idt = [0f32; 16];
        let (mut sqx, mut sq2) = ([0f32; 16], [0f32; 16]);
        idt[0] = 1.0 / d;
        for (t, v) in idt.iter_mut().enumerate().take(nt).skip(1) {
            *v = (t as i32 - 1 - ntry + v0) as f32 / max;
        }
        for j in 0..32 {
            lb[j] = best_index_iq4nl(&mid, idt[0] * xb[j]) as u8;
        }
        for j in 0..32 {
            let w = weight[j];
            for t in 0..nt {
                let q = fval[best_index_iq4nl(&mid, idt[t] * xb[j])];
                sqx[t] += w * q * xb[j];
                sq2[t] += w * q * q;
            }
        }
        d = if sq2[0] > 0.0 { sqx[0] / sq2[0] } else { 0.0 };
        let mut best = d * sqx[0];
        for t in 1..nt {
            if sq2[t] > 0.0 && sqx[t] * sqx[t] > best * sq2[t] {
                d = sqx[t] / sq2[t];
                best = d * sqx[t];
            }
        }
        scales[ib] = d;
        if d.abs() > amax_scale {
            amax_scale = d.abs();
            max_scale = d;
        }
    }
    let q4 = if nb > 1 {
        let d = -max_scale / 32.0;
        y[..2].copy_from_slice(&f16b(d));
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let (sh, rest) = y[2..].split_at_mut(2);
        let (sl, q4) = rest.split_at_mut(4);
        let mut scales_h = 0u16;
        for ib in 0..nb {
            let li = nearest_int(id * scales[ib]).clamp(-32, 31);
            let dl = d * li as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            for j in 0..32 {
                l[32 * ib + j] = best_index_iq4nl(&mid, idl * x[32 * ib + j]) as u8;
            }
            let li = (li + 32) as u8;
            let (l_l, l_h) = (li & 0xf, li >> 4);
            if ib % 2 == 0 {
                sl[ib / 2] = l_l;
            } else {
                sl[ib / 2] |= l_l << 4;
            }
            scales_h |= (l_h as u16) << (2 * (ib % 8));
        }
        sh.copy_from_slice(&scales_h.to_le_bytes());
        q4
    } else {
        y[..2].copy_from_slice(&f16b(scales[0]));
        if ntry > 0 {
            let id = if scales[0] != 0.0 { 1.0 / scales[0] } else { 0.0 };
            for j in 0..sbs {
                l[j] = best_index_iq4nl(&mid, id * x[j]) as u8;
            }
        }
        &mut y[2..]
    };
    for i in 0..sbs / 32 {
        for j in 0..16 {
            q4[16 * i + j] = l[32 * i + j] | (l[32 * i + 16 + j] << 4);
        }
    }
}

// ===================================================================== IQ2 / IQ3 grids

/// A lattice codebook prepared for quantization (llama.cpp's `iq2xs_init_impl` /
/// `iq3xs_init_impl`): the grid points as odd levels `2l + 1`, the map from a point's packed
/// level code to its grid index (negative: `-(offset + 1)` into `neighbours`, the list of the
/// nearest grid points of an off-grid code, preceded by its length).
struct Lattice {
    grid: Vec<[i8; 8]>,
    map: Vec<i32>,
    neighbours: Vec<u16>,
}

impl Lattice {
    /// `levels[k]` the level codes of grid point `k` (`dim` of them), `bits` per level in the map
    /// key, `nwant` the number of distance shells kept for off-grid codes.
    fn build(levels: Vec<[u8; 8]>, dim: usize, bits: usize, map_size: usize, nwant: usize) -> Self {
        let grid: Vec<[i8; 8]> = levels.iter().map(|lv| std::array::from_fn(|i| 2 * lv[i] as i8 + 1)).collect();
        let mut map = vec![-1i32; map_size];
        for (k, lv) in levels.iter().enumerate() {
            let mut index = 0usize;
            for (i, &q) in lv.iter().enumerate().take(dim) {
                index |= (q as usize) << (bits * i);
            }
            map[index] = k as i32;
        }
        let mask = (1usize << bits) - 1;
        let mut neighbours = Vec::new();
        let mut dist2: Vec<(i32, usize)> = Vec::with_capacity(grid.len());
        for i in 0..map_size {
            if map[i] >= 0 {
                continue;
            }
            let pos: [i32; 8] = std::array::from_fn(|k| 2 * ((i >> (bits * k)) & mask) as i32 + 1);
            dist2.clear();
            for (j, g) in grid.iter().enumerate() {
                let mut d2 = 0;
                for k in 0..dim {
                    let d = g[k] as i32 - pos[k];
                    d2 += d * d;
                }
                dist2.push((d2, j));
            }
            dist2.sort_unstable();
            map[i] = -(neighbours.len() as i32 + 1);
            let start = neighbours.len();
            neighbours.push(0);
            let (mut d2, mut nhave, mut n) = (dist2[0].0, 1, 0u16);
            for &(dj, j) in &dist2 {
                if dj > d2 {
                    if nhave == nwant {
                        break;
                    }
                    d2 = dj;
                    nhave += 1;
                }
                neighbours.push(j as u16);
                n += 1;
            }
            neighbours[start] = n;
        }
        Self { grid, map, neighbours }
    }

    /// The neighbour list of off-grid code `u`, its length first.
    fn neighbours_of(&self, u: usize) -> &[u16] {
        let off = (-self.map[u] - 1) as usize;
        &self.neighbours[off..off + 1 + self.neighbours[off] as usize]
    }

    /// `iq2_find_best_neighbour` / `iq3_find_best_neighbour`: the grid point of `nb` (a
    /// neighbour list) nearest to `xval` under `weight` at `scale`; its levels go to `l`.
    fn best_neighbour(&self, nb: &[u16], xval: &[f32], weight: &[f32], scale: f32, l: &mut [i8]) -> usize {
        let dim = l.len();
        let mut best_d2 = f32::MAX;
        let mut grid_index = usize::MAX;
        for &j in &nb[1..] {
            let pg = &self.grid[j as usize];
            let mut d2 = 0f32;
            for i in 0..dim {
                let diff = scale * pg[i] as f32 - xval[i];
                d2 += weight[i] * diff * diff;
            }
            if d2 < best_d2 {
                best_d2 = d2;
                grid_index = j as usize;
            }
        }
        assert!(grid_index != usize::MAX, "no grid neighbour");
        let pg = &self.grid[grid_index];
        for i in 0..dim {
            l[i] = (pg[i] - 1) / 2;
        }
        grid_index
    }

    /// The map key of the levels `l` (2 bits each for IQ2, 3 for IQ3).
    fn index(&self, l: &[i8], bits: usize) -> usize {
        let mut u = 0usize;
        for (i, &v) in l.iter().enumerate() {
            u |= (v as usize) << (bits * i);
        }
        u
    }
}

/// The level codes of each point of a dequantization grid (bytes, `dim` per point): each byte's
/// rank among the grid's distinct byte values (ggml's `kgrid_*` tables, which these grids were
/// generated from).
fn grid_levels(points: &[u64], dim: usize) -> Vec<[u8; 8]> {
    let mut vals: Vec<u8> = points.iter().flat_map(|&p| (0..dim).map(move |i| (p >> (8 * i)) as u8)).collect();
    vals.sort_unstable();
    vals.dedup();
    points
        .iter()
        .map(|&p| {
            std::array::from_fn(|i| {
                if i < dim {
                    vals.binary_search(&((p >> (8 * i)) as u8)).unwrap() as u8
                } else {
                    0
                }
            })
        })
        .collect()
}

fn iq2_lattice(kind: GgmlType) -> &'static Lattice {
    static L: [OnceLock<Lattice>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let (i, grid, nwant): (usize, &[u64], usize) = match kind {
        GgmlType::IQ2_XXS => (0, &IQ2XXS_GRID, 2),
        GgmlType::IQ2_XS => (1, &IQ2XS_GRID, 2),
        _ => (2, &IQ2S_GRID, 1),
    };
    L[i].get_or_init(|| Lattice::build(grid_levels(grid, 8), 8, 2, 43692, nwant))
}

fn iq3_lattice(kind: GgmlType) -> &'static Lattice {
    static L: [OnceLock<Lattice>; 2] = [OnceLock::new(), OnceLock::new()];
    let (i, grid, nwant): (usize, &[u32], usize) = match kind {
        GgmlType::IQ3_XXS => (0, &IQ3XXS_GRID, 2),
        _ => (1, &IQ3S_GRID, 3),
    };
    L[i].get_or_init(|| {
        let g: Vec<u64> = grid.iter().map(|&v| v as u64).collect();
        Lattice::build(grid_levels(&g, 4), 4, 3, 4096, nwant)
    })
}

/// Signs of 8 values with even parity (IQ2_XXS / IQ2_XS / IQ3_XXS): the absolute values go to
/// `xval`; with an odd number of negatives the one of least `w x^2` is flipped back.
fn even_signs(xb: &[f32], weight: &[f32], xval: &mut [f32]) -> u8 {
    let (mut nflip, mut s) = (0, 0u8);
    for i in 0..8 {
        if xb[i] >= 0.0 {
            xval[i] = xb[i];
        } else {
            xval[i] = -xb[i];
            nflip += 1;
            s |= 1 << i;
        }
    }
    if nflip % 2 == 1 {
        let mut imin = 0;
        let mut min = weight[0] * xb[0] * xb[0];
        for i in 1..8 {
            let ax = weight[i] * xb[i] * xb[i];
            if ax < min {
                min = ax;
                imin = i;
            }
        }
        xval[imin] = -xval[imin];
        s ^= 1 << imin;
    }
    s & 127
}

/// `sum w x q`, `sum w q^2` with `q = 2 l + 1`.
fn sums_q(weight: &[f32], xval: &[f32], l: &[i8]) -> (f32, f32) {
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for i in 0..l.len() {
        let w = weight[i];
        let q = (2 * l[i] as i32 + 1) as f32;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
    }
    (sumqx, sumq2)
}

/// The levels `l` of `xval` at inverse scale `id`, clamped to `0..kmax`.
fn levels(xval: &[f32], id: f32, kmax: i32, l: &mut [i8]) {
    for i in 0..l.len() {
        l[i] = nearest_int(0.5 * (id * xval[i] - 1.0)).clamp(0, kmax - 1) as i8;
    }
}

/// Snaps the levels `l` of group `xval` to the lattice (the nearest neighbour when off the grid);
/// whether they were on it.
fn snap(lat: &Lattice, bits: usize, xval: &[f32], waux: &[f32], scale: f32, l: &mut [i8]) -> bool {
    let u = lat.index(l, bits);
    if lat.map[u] >= 0 {
        return true;
    }
    lat.best_neighbour(lat.neighbours_of(u), xval, waux, scale, l);
    false
}

fn grid_index_of(lat: &Lattice, bits: usize, l: &[i8]) -> usize {
    let g = lat.map[lat.index(l, bits)];
    assert!(g >= 0, "point not on the grid");
    g as usize
}

fn iq2_xxs_block(x: &[f32], y: &mut [u8], qw: &[f32]) {
    let lat = iq2_lattice(GgmlType::IQ2_XXS);
    const KMAXQ: i32 = 3;
    let mut scales = [0f32; 8];
    let mut weight = [0f32; 32];
    let mut xval = [0f32; 32];
    let mut l = [0i8; 32];
    let mut laux = [0i8; 32];
    let mut waux = [0f32; 32];
    let mut block_signs = [0u8; 4];
    let mut q2 = [0u32; 16];
    let mut laux_prev = [0i8; 32];
    y[..2].copy_from_slice(&f16b(0.0));
    let mut max_scale = 0f32;
    let sigma2 = mean_sq(x);
    for ib in 0..8 {
        let xb = &x[32 * ib..32 * ib + 32];
        block_weights(xb, &qw[32 * ib..], sigma2, &mut weight);
        for i in 0..32 {
            waux[i] = weight[i].sqrt();
        }
        for k in 0..4 {
            block_signs[k] = even_signs(&xb[8 * k..], &weight[8 * k..], &mut xval[8 * k..]);
        }
        let mut max = xval[0];
        for &v in &xval[1..] {
            max = if max > v { max } else { v };
        }
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            l.fill(0);
            continue;
        }
        let mut lu = [0u8; 32];
        let mut scale = make_qp_quants(32, KMAXQ + 1, &xval, &mut lu, &weight);
        for i in 0..32 {
            l[i] = lu[i] as i8;
        }
        let eff_max = scale * KMAXQ as f32;
        if eff_max <= 0.0 {
            scales[ib] = 0.0;
            l.fill(0);
            continue;
        }
        let mut best = 0f32;
        for is in -6..=6 {
            let id = ((2 * KMAXQ - 1) as f32 + is as f32 * 0.1) / eff_max;
            let this_scale = 1.0 / id;
            for k in 0..4 {
                let g = 8 * k..8 * k + 8;
                levels(&xval[g.clone()], id, KMAXQ, &mut laux[g.clone()]);
                snap(lat, 2, &xval[g.clone()], &waux[g.clone()], this_scale, &mut laux[g]);
            }
            if is > -6 && laux == laux_prev {
                continue;
            }
            laux_prev = laux;
            let (sumqx, sumq2) = sums_q(&weight, &xval, &laux);
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l = laux;
            }
        }
        if scale > 0.0 {
            let id = 1.0 / scale;
            for k in 0..4 {
                let g = 8 * k..8 * k + 8;
                let mut lk = [0i8; 8];
                levels(&xval[g.clone()], id, KMAXQ, &mut lk);
                let u = lat.index(&lk, 2);
                let gi = if lat.map[u] < 0 {
                    lat.best_neighbour(lat.neighbours_of(u), &xval[g.clone()], &waux[g.clone()], scale, &mut l[g.clone()])
                } else {
                    lat.map[u] as usize
                };
                let pg = &lat.grid[gi];
                for i in 0..8 {
                    l[8 * k + i] = (pg[i] - 1) / 2;
                }
            }
            let (sumqx, sumq2) = sums_q(&weight, &xval, &l);
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in &mut block_signs {
                *s = !*s & 127;
            }
        }
        for k in 0..4 {
            let gi = grid_index_of(lat, 2, &l[8 * k..8 * k + 8]);
            q2[2 * ib] |= (gi as u32) << (8 * k);
            q2[2 * ib + 1] |= (block_signs[k] as u32) << (7 * k);
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        y[2..66].fill(0);
        return;
    }
    let d = max_scale / 31.0;
    y[..2].copy_from_slice(&f16b(d));
    let id = 1.0 / d;
    for ib in 0..8 {
        let li = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
        q2[2 * ib + 1] |= (li as u32) << 28;
    }
    for (o, v) in y[2..66].as_chunks_mut::<4>().0.iter_mut().zip(q2) {
        o.copy_from_slice(&v.to_le_bytes());
    }
}

/// The shared search of IQ2_XS and IQ2_S over one group of 16 (`xval`, `weight`, `waux`): the
/// scale and levels `l` (zeros when the group is empty).
fn iq2_search16(lat: &Lattice, xval: &[f32], weight: &[f32], waux: &[f32], l: &mut [i8; 16], eps: f32) -> f32 {
    const KMAXQ: i32 = 3;
    let mut max = xval[0];
    for &v in &xval[1..16] {
        max = if max > v { max } else { v };
    }
    l.fill(0);
    if max < eps {
        return 0.0;
    }
    let mut best = 0f32;
    let mut scale = max / (2 * KMAXQ - 1) as f32;
    let mut on_grid = [true; 2];
    let mut laux = [0i8; 16];
    let mut laux_prev = [0i8; 16];
    let mut on_prev = [false; 2];
    for is in -9..=9 {
        let id = ((2 * KMAXQ - 1) as f32 + is as f32 * 0.1) / max;
        let this_scale = 1.0 / id;
        let mut on_aux = [true; 2];
        for k in 0..2 {
            let g = 8 * k..8 * k + 8;
            levels(&xval[g.clone()], id, KMAXQ, &mut laux[g.clone()]);
            on_aux[k] = snap(lat, 2, &xval[g.clone()], &waux[g.clone()], this_scale, &mut laux[g]);
        }
        if is > -9 && laux == laux_prev && on_aux == on_prev {
            continue;
        }
        laux_prev = laux;
        on_prev = on_aux;
        let (sumqx, sumq2) = sums_q(weight, xval, &laux);
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            scale = sumqx / sumq2;
            best = scale * sumqx;
            *l = laux;
            on_grid = on_aux;
        }
    }
    if on_grid.iter().any(|&o| !o) && scale > 0.0 {
        let id = 1.0 / scale;
        for k in 0..2 {
            if on_grid[k] {
                continue;
            }
            let g = 8 * k..8 * k + 8;
            levels(&xval[g.clone()], id, KMAXQ, &mut l[g.clone()]);
            snap(lat, 2, &xval[g.clone()], &waux[g.clone()], scale, &mut l[g]);
        }
        let (sumqx, sumq2) = sums_q(weight, xval, l);
        if sumq2 > 0.0 {
            scale = sumqx / sumq2;
        }
    }
    scale
}

fn iq2_xs_block(x: &[f32], y: &mut [u8], qw: &[f32]) {
    let lat = iq2_lattice(GgmlType::IQ2_XS);
    let mut scales = [0f32; 16];
    let mut weight = [0f32; 16];
    let mut xval = [0f32; 16];
    let mut waux = [0f32; 16];
    let mut l = [0i8; 16];
    let mut q2 = [0u16; 32];
    y.fill(0);
    let mut max_scale = 0f32;
    let sigma2 = mean_sq(x);
    for ib in 0..16 {
        let xb = &x[16 * ib..16 * ib + 16];
        block_weights(xb, &qw[16 * ib..], sigma2, &mut weight);
        for i in 0..16 {
            waux[i] = weight[i].sqrt();
        }
        let mut signs = [0u8; 2];
        for k in 0..2 {
            signs[k] = even_signs(&xb[8 * k..], &weight[8 * k..], &mut xval[8 * k..]);
        }
        let max = xval.iter().fold(xval[0], |m, &v| if m > v { m } else { v });
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let mut scale = iq2_search16(lat, &xval, &weight, &waux, &mut l, GROUP_MAX_EPS);
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = !*s & 127;
            }
        }
        for k in 0..2 {
            let gi = grid_index_of(lat, 2, &l[8 * k..8 * k + 8]);
            q2[2 * ib + k] = gi as u16 | ((signs[k] as u16) << 9);
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / 31.0;
    y[..2].copy_from_slice(&f16b(d));
    let id = 1.0 / d;
    for ib in 0..16 {
        let li = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
        if ib % 2 == 0 {
            y[66 + ib / 2] = li;
        } else {
            y[66 + ib / 2] |= li << 4;
        }
    }
    for (o, v) in y[2..66].as_chunks_mut::<2>().0.iter_mut().zip(q2) {
        o.copy_from_slice(&v.to_le_bytes());
    }
}

fn iq2_s_block(x: &[f32], y: &mut [u8], qw: Option<&[f32]>) {
    let lat = iq2_lattice(GgmlType::IQ2_S);
    let mut scales = [0f32; 16];
    let mut weight = [0f32; 16];
    let mut xval = [0f32; 16];
    let mut waux = [0f32; 16];
    let mut l = [0i8; 16];
    y.fill(0);
    let mut max_scale = 0f32;
    let mut sumx2 = 0f32;
    for &v in &x[..QK_K] {
        sumx2 += v * v;
    }
    let sigma2 = 2.0 * sumx2 / QK_K as f32;
    for ib in 0..16 {
        let xb = &x[16 * ib..16 * ib + 16];
        match qw {
            Some(q) => block_weights(xb, &q[16 * ib..], sigma2, &mut weight),
            None => {
                for i in 0..16 {
                    weight[i] = 0.25 * sigma2 + xb[i] * xb[i];
                }
            }
        }
        for i in 0..16 {
            waux[i] = weight[i].sqrt();
        }
        let mut signs = [0u8; 2];
        for k in 0..2 {
            let mut s = 0u8;
            for i in 0..8 {
                let v = xb[8 * k + i];
                if v >= 0.0 {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    s |= 1 << i;
                }
            }
            signs[k] = s;
        }
        let max = xval.iter().fold(xval[0], |m, &v| if m > v { m } else { v });
        if max < GROUP_MAX_EPS_IQ2_S {
            scales[ib] = 0.0;
            continue;
        }
        let mut scale = iq2_search16(lat, &xval, &weight, &waux, &mut l, GROUP_MAX_EPS_IQ2_S);
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = !*s;
            }
        }
        for k in 0..2 {
            let gi = grid_index_of(lat, 2, &l[8 * k..8 * k + 8]);
            let i8 = 2 * ib + k;
            y[2 + i8] = (gi & 255) as u8;
            y[66 + i8 / 4] |= ((gi >> 8) << (2 * (i8 % 4))) as u8;
            y[2 + 32 + i8] = signs[k];
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / 31.0;
    y[..2].copy_from_slice(&f16b(d * 0.9875));
    let id = 1.0 / d;
    for ib in 0..16 {
        let li = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
        if ib % 2 == 0 {
            y[74 + ib / 2] = li;
        } else {
            y[74 + ib / 2] |= li << 4;
        }
    }
}

/// The shared search of IQ3_XXS and IQ3_S over one group of 32: the scale and levels `l`, with
/// `ns` scale steps of 0.2 either side, the on-grid flags' handling of the type (IQ3_S snaps
/// every group again, IQ3_XXS only those that were off the grid).
#[allow(clippy::too_many_arguments)]
fn iq3_search32(lat: &Lattice, xval: &[f32], weight: &[f32], waux: &[f32], l: &mut [i8; 32], ns: i32, start_on: bool, all: bool) -> f32 {
    const KMAXQ: i32 = 8;
    let max = xval[..32].iter().fold(xval[0], |m, &v| if m > v { m } else { v });
    let mut best = 0f32;
    let mut scale = max / (2 * KMAXQ - 1) as f32;
    let mut on_grid = [start_on; 8];
    let mut laux = [0i8; 32];
    for is in -ns..=ns {
        let id = ((2 * KMAXQ - 1) as f32 + is as f32 * 0.2) / max;
        let this_scale = 1.0 / id;
        let mut on_aux = [true; 8];
        for k in 0..8 {
            let g = 4 * k..4 * k + 4;
            levels(&xval[g.clone()], id, KMAXQ, &mut laux[g.clone()]);
            on_aux[k] = snap(lat, 3, &xval[g.clone()], &waux[g.clone()], this_scale, &mut laux[g]);
        }
        let (sumqx, sumq2) = sums_q(weight, xval, &laux);
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            scale = sumqx / sumq2;
            best = scale * sumqx;
            *l = laux;
            on_grid = on_aux;
        }
    }
    if on_grid.iter().any(|&o| !o) && scale > 0.0 {
        let id = 1.0 / scale;
        for k in 0..8 {
            if on_grid[k] && !all {
                continue;
            }
            let g = 4 * k..4 * k + 4;
            let mut lk = [0i8; 4];
            levels(&xval[g.clone()], id, KMAXQ, &mut lk);
            let u = lat.index(&lk, 3);
            let gi = if lat.map[u] < 0 {
                lat.best_neighbour(lat.neighbours_of(u), &xval[g.clone()], &waux[g.clone()], scale, &mut l[g.clone()])
            } else {
                lat.map[u] as usize
            };
            let pg = &lat.grid[gi];
            for i in 0..4 {
                l[4 * k + i] = (pg[i] - 1) / 2;
            }
        }
        let (sumqx, sumq2) = sums_q(weight, xval, l);
        if sumq2 > 0.0 {
            scale = sumqx / sumq2;
        }
    }
    scale
}

/// The IQ3 weights of a group of 32: `qw * sqrt(sigma2 + x^2)`, or `x^2` without an imatrix.
fn iq3_weights(xb: &[f32], qw: Option<&[f32]>, sigma2: f32, weight: &mut [f32; 32], waux: &mut [f32; 32]) {
    match qw {
        Some(q) => block_weights(xb, q, sigma2, weight),
        None => {
            for i in 0..32 {
                weight[i] = xb[i] * xb[i];
            }
        }
    }
    for i in 0..32 {
        waux[i] = weight[i].sqrt();
    }
}

fn iq3_xxs_block(x: &[f32], y: &mut [u8], qw: Option<&[f32]>) {
    let lat = iq3_lattice(GgmlType::IQ3_XXS);
    let mut scales = [0f32; 8];
    let mut weight = [0f32; 32];
    let mut waux = [0f32; 32];
    let mut xval = [0f32; 32];
    let mut l = [0i8; 32];
    let mut q3 = [0u8; 96];
    y.fill(0);
    let mut max_scale = 0f32;
    let mut sumx2 = 0f32;
    for &v in &x[..QK_K] {
        sumx2 += v * v;
    }
    let sigma2 = 2.0 * sumx2 / QK_K as f32;
    for ib in 0..8 {
        let xb = &x[32 * ib..32 * ib + 32];
        iq3_weights(xb, qw.map(|q| &q[32 * ib..]), sigma2, &mut weight, &mut waux);
        let mut signs = [0u8; 4];
        for k in 0..4 {
            signs[k] = even_signs(&xb[8 * k..], &weight[8 * k..], &mut xval[8 * k..]);
        }
        let max = xval.iter().fold(xval[0], |m, &v| if m > v { m } else { v });
        l.fill(0);
        if max < GROUP_MAX_EPS_IQ3_XXS {
            scales[ib] = 0.0;
            continue;
        }
        let mut scale = iq3_search32(lat, &xval, &weight, &waux, &mut l, 15, true, false);
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = !*s & 127;
            }
        }
        for k in 0..8 {
            q3[8 * ib + k] = grid_index_of(lat, 3, &l[4 * k..4 * k + 4]) as u8;
        }
        let ss = signs[0] as u32 | (signs[1] as u32) << 7 | (signs[2] as u32) << 14 | (signs[3] as u32) << 21;
        q3[64 + 4 * ib..68 + 4 * ib].copy_from_slice(&ss.to_le_bytes());
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / 31.0;
    y[..2].copy_from_slice(&f16b(d * 1.0125));
    let id = 1.0 / d;
    for ib in 0..8 {
        let li = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
        let o = 64 + 4 * ib;
        let ss = u32::from_le_bytes([q3[o], q3[o + 1], q3[o + 2], q3[o + 3]]) | (li as u32) << 28;
        q3[o..o + 4].copy_from_slice(&ss.to_le_bytes());
    }
    y[2..98].copy_from_slice(&q3);
}

fn iq3_s_block(x: &[f32], y: &mut [u8], qw: Option<&[f32]>) {
    let lat = iq3_lattice(GgmlType::IQ3_S);
    let mut scales = [0f32; 8];
    let mut weight = [0f32; 32];
    let mut waux = [0f32; 32];
    let mut xval = [0f32; 32];
    let mut l = [0i8; 32];
    y.fill(0);
    let mut max_scale = 0f32;
    let mut sumx2 = 0f32;
    for &v in &x[..QK_K] {
        sumx2 += v * v;
    }
    let sigma2 = 2.0 * sumx2 / QK_K as f32;
    // d, qs[64], qh[8], signs[32], scales[4]
    for ib in 0..8 {
        let xb = &x[32 * ib..32 * ib + 32];
        iq3_weights(xb, qw.map(|q| &q[32 * ib..]), sigma2, &mut weight, &mut waux);
        let mut signs = [0u8; 4];
        for k in 0..4 {
            let mut s = 0u8;
            for i in 0..8 {
                let v = xb[8 * k + i];
                if v >= 0.0 {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    s |= 1 << i;
                }
            }
            signs[k] = s;
        }
        let max = xval.iter().fold(xval[0], |m, &v| if m > v { m } else { v });
        l.fill(0);
        if max == 0.0 {
            scales[ib] = 0.0;
            continue;
        }
        let mut scale = iq3_search32(lat, &xval, &weight, &waux, &mut l, 9, false, true);
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = !*s;
            }
        }
        for k in 0..8 {
            let gi = grid_index_of(lat, 3, &l[4 * k..4 * k + 4]);
            y[2 + 8 * ib + k] = (gi & 255) as u8;
            let j = ib * 8 + k;
            y[66 + j / 8] |= ((gi >> 8) << (j % 8)) as u8;
        }
        y[74 + 4 * ib..78 + 4 * ib].copy_from_slice(&signs);
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / 31.0;
    y[..2].copy_from_slice(&f16b(d * 1.033));
    let id = 1.0 / d;
    for ib in (0..8).step_by(2) {
        let l1 = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
        let l2 = nearest_int(0.5 * (id * scales[ib + 1] - 1.0)).clamp(0, 15);
        y[106 + ib / 2] = (l1 | (l2 << 4)) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::dequantize_row;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Gaussian-ish rows with a few outliers.
    fn row(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let g = (0..4).map(|_| lcg(seed)).sum::<f32>() * 0.02;
                if i % 97 == 5 { g * 6.0 } else { g }
            })
            .collect()
    }

    fn rel_rmse(a: &[f32], b: &[f32]) -> f32 {
        let e: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let s: f32 = a.iter().map(|x| x * x).sum();
        (e / s).sqrt()
    }

    /// Each quantizer decodes to a row close to its input (the bound is the format's usual
    /// error on Gaussian data with outliers), with and without an importance matrix, and
    /// re-quantizing the decoded row moves it far less than the first quantization did (most
    /// types reproduce it exactly). Bit-exactness against llama.cpp is checked by
    /// `examples/ggml_quant_check.rs`.
    #[test]
    fn round_trip_every_type() {
        use GgmlType::*;
        let n = 2 * QK_K;
        let mut seed = 7;
        let x = row(n, &mut seed);
        let imat: Vec<f32> = (0..n).map(|_| lcg(&mut seed).abs() + 0.1).collect();
        for (kind, bound) in [
            (Q4_0, 0.17),
            (Q4_1, 0.11),
            (Q5_0, 0.075),
            (Q5_1, 0.055),
            (Q8_0, 0.01),
            (Q2_K, 0.38),
            (Q3_K, 0.22),
            (Q4_K, 0.1),
            (Q5_K, 0.052),
            (Q6_K, 0.03),
            (IQ2_XXS, 0.56),
            (IQ2_XS, 0.45),
            (IQ2_S, 0.42),
            (IQ3_XXS, 0.29),
            (IQ3_S, 0.22),
            (IQ4_NL, 0.125),
            (IQ4_XS, 0.125),
            (MXFP4, 0.18),
            (Q2_0, 0.75),
            (TQ2_0, 0.9),
            (TQ1_0, 0.9),
            (Q1_0, 0.8),
        ] {
            for im in [None, Some(&imat[..])] {
                let mut q = vec![0u8; kind.row_bytes(n)];
                quantize_row(kind, &x, &mut q, im);
                let mut d = vec![0f32; n];
                dequantize_row(kind, &q, &mut d);
                let e = rel_rmse(&x, &d);
                assert!(e < bound, "{kind} (imatrix {}): rel rmse {e}", im.is_some());
                let mut q2 = vec![0u8; q.len()];
                quantize_row(kind, &d, &mut q2, im);
                let mut d2 = vec![0f32; n];
                dequantize_row(kind, &q2, &mut d2);
                let e2 = rel_rmse(&d, &d2);
                assert!(e2 < 0.5 * e, "{kind} (imatrix {}): requantized drift {e2} vs {e}", im.is_some());
            }
        }
        // The least-squares Q2_0 fits far better than llama.cpp's absmax rounding.
        let mut q = vec![0u8; GgmlType::Q2_0.row_bytes(n)];
        let mut d = vec![0f32; n];
        quantize_row_with(GgmlType::Q2_0, &x, &mut q, None, Fit::Best);
        dequantize_row(GgmlType::Q2_0, &q, &mut d);
        let best = rel_rmse(&x, &d);
        quantize_row(GgmlType::Q2_0, &x, &mut q, None);
        dequantize_row(GgmlType::Q2_0, &q, &mut d);
        let plain = rel_rmse(&x, &d);
        assert!(best < 0.7 * plain, "Q2_0 least squares {best} vs absmax {plain}");
    }

    /// `Lin::merge_lora` on a GGUF block weight stores the merged weight in the same type (the
    /// same bytes), as the float merge quantized again.
    #[test]
    fn merge_keeps_block_type() {
        use crate::model::Lin;
        let (n_out, n_in, r) = (8, 2 * QK_K, 2);
        let mut seed = 11;
        let w: Vec<f32> = (0..n_out).flat_map(|_| row(n_in, &mut seed)).collect();
        let a: Vec<f32> = (0..r * n_in).map(|_| lcg(&mut seed) * 0.05).collect();
        let b: Vec<f32> = (0..n_out * r).map(|_| lcg(&mut seed) * 0.05).collect();
        for kind in [GgmlType::Q4_K, GgmlType::IQ4_XS, GgmlType::Q2_0, GgmlType::Q6_K] {
            let q = quantize_matrix(kind, &w, n_in, None);
            let mut lin = Lin::from_gguf(kind, &q, n_out, n_in).unwrap();
            let before = lin.bytes();
            let rows: Vec<u32> = (0..n_out as u32).collect();
            let mut want = vec![0f32; n_out * n_in];
            assert!(lin.rows(&rows, n_in, &mut want));
            let d = crate::lora::LoraDelta::fixed(a.clone(), &b, r, n_in, n_out);
            d.merge_into(&mut want);
            assert!(lin.merge_lora(&d).unwrap(), "{kind}");
            assert_eq!(lin.bytes(), before, "{kind}: storage size");
            let mut got = vec![0f32; n_out * n_in];
            assert!(lin.rows(&rows, n_in, &mut got));
            let rb = kind.row_bytes(n_in);
            let mut requant = vec![0u8; rb];
            let mut dq = vec![0f32; n_in];
            for i in 0..n_out {
                quantize_row_with(kind, &want[i * n_in..(i + 1) * n_in], &mut requant, None, Fit::Best);
                dequantize_row(kind, &requant, &mut dq);
                assert_eq!(dq, got[i * n_in..(i + 1) * n_in], "{kind}: row {i}");
            }
        }
    }

    /// Zero rows quantize to zero rows; the plain float types round-trip.
    #[test]
    fn zeros_and_floats() {
        for kind in GgmlType::ALL.into_iter().filter(|&k| can_quantize(k)) {
            let n = 2 * QK_K;
            let mut q = vec![0xAAu8; kind.row_bytes(n)];
            quantize_row(kind, &vec![0f32; n], &mut q, None);
            let mut d = vec![1f32; n];
            dequantize_row(kind, &q, &mut d);
            assert!(d.iter().all(|&v| v == 0.0), "{kind}: zero row");
        }
        let x = [1.5f32, -2.25, 0.0, 3.0e-3];
        for kind in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            let mut q = vec![0u8; kind.row_bytes(4)];
            quantize_row(kind, &x, &mut q, None);
            let mut d = [0f32; 4];
            dequantize_row(kind, &q, &mut d);
            assert!(x.iter().zip(&d).all(|(a, b)| (a - b).abs() <= 1e-5), "{kind}");
        }
    }
}
