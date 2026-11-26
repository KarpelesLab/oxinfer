//! f32 prefill GEMM on the Arm Scalable Matrix Extension (SME: Apple M4 and later), for vector
//! length 512 bits.
//!
//! Each `FMOPA` adds the outer product of 16 weights (one input column of 16 rows) and 16
//! activations (16 tokens) into one of the four 16 x 16 f32 tiles of the ZA array, so a 32-row by
//! 32-token block takes four per input element. The weights are dequantized to f32 as for the
//! NEON GEMM, 32 rows at a time, and both operands are transposed to `[k][32]` panels through ZA
//! (rows loaded as horizontal slices, stored as vertical ones). Products and sums are plain f32
//! (no activation rounding), in another order than the NEON GEMM's.
//!
//! The SME units are shared by the cores of a cluster; on an M5 Max one thread reaches about
//! 1.8 TFLOP/s and the machine about 3, against about 1.3 for the NEON FMA GEMM on all cores.
//! Code between `smstart` and `smstop` runs in streaming mode, where most NEON instructions are
//! not allowed, so each routine is one `asm!` block that enters and leaves it.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::kernels::QMat;

/// `asm!` for a routine that enters and leaves streaming mode: `smstart` / `smstop` zero every
/// vector and predicate register (callee-saved `d8`-`d15` included), so all are clobbers.
macro_rules! sme_asm {
    ($($t:tt)*) => {
        std::arch::asm!(
            $($t)*,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            out("ffr") _,
            options(nostack)
        )
    };
}

/// Whether the CPU has SME with a 512-bit streaming vector length (the kernels' tile shapes).
pub(crate) fn available() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| detect() && unsafe { svl_bytes() } == 64)
}

#[cfg(target_vendor = "apple")]
fn detect() -> bool {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(name: *const c_char, oldp: *mut c_void, oldlenp: *mut usize, newp: *const c_void, newlen: usize) -> c_int;
    }
    let mut v: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let rc = unsafe {
        sysctlbyname(
            c"hw.optional.arm.FEAT_SME".as_ptr(),
            &mut v as *mut i32 as *mut c_void,
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    rc == 0 && v != 0
}

#[cfg(all(not(target_vendor = "apple"), any(target_os = "linux", target_os = "android")))]
fn detect() -> bool {
    unsafe extern "C" {
        fn getauxval(t: std::ffi::c_ulong) -> std::ffi::c_ulong;
    }
    const AT_HWCAP2: std::ffi::c_ulong = 26;
    const HWCAP2_SME: std::ffi::c_ulong = 1 << 23;
    unsafe { getauxval(AT_HWCAP2) & HWCAP2_SME != 0 }
}

#[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "android")))]
fn detect() -> bool {
    false
}

/// The streaming vector length in bytes (`RDSVL`, legal outside streaming mode). Only call with
/// SME present.
unsafe fn svl_bytes() -> usize {
    let n: usize;
    std::arch::asm!(".arch_extension sme", "rdsvl {n}, #1", n = out(reg) n, options(pure, nomem, nostack));
    n
}

/// Transposes rows `0..rows` (at most 32; the others count as zeros) of `k` f32 each (`k` a
/// multiple of 16, row stride `ld` floats) at `src` into the panel `dst[k][32]`.
pub(crate) unsafe fn pack32(src: *const f32, ld: usize, rows: usize, k: usize, dst: *mut f32) {
    debug_assert!(rows <= 32 && k.is_multiple_of(16) && k > 0);
    for h in 0..2 {
        let n = rows.saturating_sub(16 * h).min(16);
        // Rows of this strip as ZA horizontal slices, stored back as vertical slices (the 16
        // rows of one input element), 16 input elements per round.
        sme_asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.s",
            "1:",
            "zero {{za0.s}}",
            "mov w12, #0",
            "mov {s}, {src}",
            "cbz {n}, 3f",
            "2:",
            "ld1w {{za0h.s[w12, 0]}}, p0/z, [{s}]",
            "add {s}, {s}, {ldb}",
            "add w12, w12, #1",
            "cmp w12, {n:w}",
            "b.ne 2b",
            "3:",
            "mov w12, #0",
            "mov {d}, {dst}",
            "4:",
            "st1w {{za0v.s[w12, 0]}}, p0, [{d}]",
            "add {d}, {d}, #128",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 4b",
            "add {src}, {src}, #64",
            "add {dst}, {dst}, #2048",
            "subs {kb}, {kb}, #1",
            "b.ne 1b",
            "smstop",
            src = inout(reg) src.add(16 * h * ld) => _,
            dst = inout(reg) dst.add(16 * h) => _,
            kb = inout(reg) k / 16 => _,
            n = in(reg) n,
            ldb = in(reg) ld * 4,
            s = out(reg) _,
            d = out(reg) _,
            out("w12") _
        );
    }
}

/// `c[t][r] = sum_k a[k][r] * b[k][t]` for the panels `a`, `b` (`[k][32]`, `k > 0`), `c`
/// `[32 tokens][32 rows]`.
pub(crate) unsafe fn kern32(a: *const f32, b: *const f32, k: usize, c: *mut f32) {
    debug_assert!(k > 0);
    // za0: rows 0..16 x tokens 0..16, za1: rows 0..16 x tokens 16..32, za2 / za3: rows 16..32.
    sme_asm!(
        ".arch_extension sme",
        "smstart",
        "ptrue p0.s",
        "zero {{za}}",
        "1:",
        "ld1w {{z0.s}}, p0/z, [{a}]",
        "ld1w {{z1.s}}, p0/z, [{a}, #1, mul vl]",
        "ld1w {{z2.s}}, p0/z, [{b}]",
        "ld1w {{z3.s}}, p0/z, [{b}, #1, mul vl]",
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
        "add {a}, {a}, #128",
        "add {b}, {b}, #128",
        "subs {k}, {k}, #1",
        "b.ne 1b",
        // Vertical slice j of a tile: its 16 rows at token j.
        "mov w12, #0",
        "2:",
        "st1w {{za0v.s[w12, 0]}}, p0, [{c}]",
        "add {c2}, {c}, #64",
        "st1w {{za2v.s[w12, 0]}}, p0, [{c2}]",
        "add {c2}, {c}, #2048",
        "st1w {{za1v.s[w12, 0]}}, p0, [{c2}]",
        "add {c2}, {c}, #2112",
        "st1w {{za3v.s[w12, 0]}}, p0, [{c2}]",
        "add {c}, {c}, #128",
        "add w12, w12, #1",
        "cmp w12, #16",
        "b.ne 2b",
        "smstop",
        a = inout(reg) a => _,
        b = inout(reg) b => _,
        k = inout(reg) k => _,
        c = inout(reg) c => _,
        c2 = out(reg) _,
        out("w12") _
    );
}

/// [`kern32`] for `n` panels `b` in a row (`b + i * 32 * k`, `n > 0`) against one panel `a`, into
/// `c + i * 32 * 32`, in one streaming-mode session.
pub(crate) unsafe fn kern32_seq(a: *const f32, b: *const f32, k: usize, n: usize, c: *mut f32) {
    debug_assert!(k > 0 && n > 0);
    sme_asm!(
        ".arch_extension sme",
        "smstart",
        "ptrue p0.s",
        "5:",
        "zero {{za}}",
        "mov {ap}, {a}",
        "mov {kk}, {k}",
        "1:",
        "ld1w {{z0.s}}, p0/z, [{ap}]",
        "ld1w {{z1.s}}, p0/z, [{ap}, #1, mul vl]",
        "ld1w {{z2.s}}, p0/z, [{b}]",
        "ld1w {{z3.s}}, p0/z, [{b}, #1, mul vl]",
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
        "add {ap}, {ap}, #128",
        "add {b}, {b}, #128",
        "subs {kk}, {kk}, #1",
        "b.ne 1b",
        "mov w12, #0",
        "2:",
        "st1w {{za0v.s[w12, 0]}}, p0, [{c}]",
        "add {c2}, {c}, #64",
        "st1w {{za2v.s[w12, 0]}}, p0, [{c2}]",
        "add {c2}, {c}, #2048",
        "st1w {{za1v.s[w12, 0]}}, p0, [{c2}]",
        "add {c2}, {c}, #2112",
        "st1w {{za3v.s[w12, 0]}}, p0, [{c2}]",
        "add {c}, {c}, #128",
        "add w12, w12, #1",
        "cmp w12, #16",
        "b.ne 2b",
        "add {c}, {c}, #2048",
        "subs {n}, {n}, #1",
        "b.ne 5b",
        "smstop",
        a = in(reg) a,
        b = inout(reg) b => _,
        k = in(reg) k,
        n = inout(reg) n => _,
        c = inout(reg) c => _,
        ap = out(reg) _,
        kk = out(reg) _,
        c2 = out(reg) _,
        out("w12") _
    );
}

/// The activations `x` (`[t, n_in]`) as panels of 32 tokens (`[tile][n_in][32]`, zero padded).
pub(crate) fn pack_x(x: &[f32], t: usize, n_in: usize) -> Vec<f32> {
    let tiles = t.div_ceil(32);
    let mut xt = vec![0f32; tiles * n_in * 32];
    struct P(*mut f32);
    unsafe impl Sync for P {}
    let p = P(xt.as_mut_ptr());
    crate::pool::run(tiles, &|tile| {
        let p = &p;
        // SAFETY: each tile writes its own panel; rows past `t` are not read.
        unsafe {
            pack32(
                x.as_ptr().add(tile * 32 * n_in),
                n_in,
                (t - 32 * tile).min(32),
                n_in,
                p.0.add(tile * n_in * 32),
            )
        }
    });
    xt
}

/// Whether [`gemm_rows`] takes a matrix of `n_in` inputs.
pub(crate) fn fits(n_in: usize) -> bool {
    n_in.is_multiple_of(16) && n_in > 0
}

/// Per-thread buffers: dequantized rows and their panel.
struct Scratch {
    rows: Vec<f32>,
    panel: Vec<f32>,
    c: Vec<f32>,
}

// Already `const`; clippy misreads the emulated-TLS expansion on Android.
#[allow(clippy::missing_const_for_thread_local)]
mod scratch {
    use std::cell::RefCell;
    thread_local! {
        pub(super) static SCRATCH: RefCell<super::Scratch> =
            const { RefCell::new(super::Scratch { rows: Vec::new(), panel: Vec::new(), c: Vec::new() }) };
    }
}

/// Bytes of weight panels [`gemm_rows`] keeps at once (each activation panel is then read once
/// per this many weights; the SME units read from L2).
const PANELS_BYTES: usize = 2 << 20;

/// `y[c * n_out + r]` for tokens `c0..c1` and rows `r0..r1` of `m`, with `xt` the activations
/// packed by [`pack_x`]; `c0` should be a multiple of 32 (a tile shared with another call is
/// computed twice).
pub(crate) unsafe fn gemm_rows<M: QMat + ?Sized>(m: &M, xt: &[f32], c0: usize, c1: usize, r0: usize, r1: usize, y: *mut f32) {
    let (n_in, n_out) = (m.n_in(), m.n_out());
    debug_assert!(fits(n_in) && xt.len() >= c1.div_ceil(32) * 32 * n_in);
    let pmax = (PANELS_BYTES / (128 * n_in)).clamp(1, 8);
    scratch::SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let Scratch { rows, panel, c } = &mut *s;
        if rows.len() < 32 * n_in {
            rows.resize(32 * n_in, 0.0);
        }
        if panel.len() < pmax * 32 * n_in {
            panel.resize(pmax * 32 * n_in, 0.0);
        }
        c.resize(32 * 32, 0.0);
        let mut r = r0;
        while r < r1 {
            let np = (r1 - r).div_ceil(32).min(pmax);
            for p in 0..np {
                let (ra, nr) = (r + 32 * p, (r1 - r - 32 * p).min(32));
                m.dequantize_rows(ra, ra + nr, &mut rows[..nr * n_in]);
                pack32(rows.as_ptr(), n_in, nr, n_in, panel.as_mut_ptr().add(p * 32 * n_in));
            }
            for tile in c0 / 32..c1.div_ceil(32) {
                for p in 0..np {
                    let (ra, nr) = (r + 32 * p, (r1 - r - 32 * p).min(32));
                    kern32(
                        panel.as_ptr().add(p * 32 * n_in),
                        xt.as_ptr().add(tile * n_in * 32),
                        n_in,
                        c.as_mut_ptr(),
                    );
                    for tok in (32 * tile).max(c0)..(32 * tile + 32).min(c1) {
                        std::ptr::copy_nonoverlapping(c.as_ptr().add((tok - 32 * tile) * 32), y.add(tok * n_out + ra), nr);
                    }
                }
            }
            r = (r + 32 * np).min(r1);
        }
    });
}

/// Per-thread buffers of [`attention`]: the query panel, the scores, a 32 x 32 tile.
struct AttnScratch {
    qp: Vec<f32>,
    s: Vec<f32>,
    c: Vec<f32>,
}

#[allow(clippy::missing_const_for_thread_local)]
mod attn_scratch {
    use std::cell::RefCell;
    thread_local! {
        pub(super) static SCRATCH: RefCell<super::AttnScratch> =
            const { RefCell::new(super::AttnScratch { qp: Vec::new(), s: Vec::new(), c: Vec::new() }) };
    }
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// Whether [`attention`] takes heads of `dh` values.
pub(crate) fn attention_fits(dh: usize) -> bool {
    dh > 0 && dh.is_multiple_of(32)
}

/// Bidirectional attention: `out` (`[n, nh * dh]`) = softmax(scale q k^T) v per head, for `q`,
/// `k`, `v` (`[n, nh * dh]`), with `dh` a multiple of 32. One task per head and block of 32
/// queries: the scores of every key as 32 x 32 tiles on the SME array (queries scaled first,
/// keys as `[dh][32]` panels, all tiles in one streaming session), laid out key by key
/// (`[key][32 queries]`, the panel the values' product takes); the softmax of each query over
/// all keys on the NEON units (the maximum, the exponentials and their sums run along the keys
/// for the 32 queries at once); then the values, `[key][32 dims]` panels, times those weights
/// on the SME array over all the keys, and each output divided by its query's sum.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, dh: usize, scale: f32, out: &mut [f32]) {
    let d = nh * dh;
    assert!(attention_fits(dh) && q.len() == n * d && k.len() == n * d && v.len() == n * d && out.len() == n * d);
    if n == 0 {
        return;
    }
    let tiles = n.div_ceil(32);
    let np = tiles * 32;
    let ng = dh / 32;
    // Keys: `[head][tile][dh][32]`.
    let mut kp = vec![0f32; nh * tiles * dh * 32];
    let kpp = Ptr(kp.as_mut_ptr());
    crate::pool::run(nh * tiles, &|i| {
        let kpp = &kpp;
        let (h, j) = (i / tiles, i % tiles);
        // SAFETY: each task writes its own panel; rows past `n` are not read.
        unsafe {
            pack32(
                k.as_ptr().add(32 * j * d + h * dh),
                d,
                (n - 32 * j).min(32),
                dh,
                kpp.0.add(i * dh * 32),
            )
        }
    });
    // Values: `[head][group of 32 dims][np keys][32]` (zero rows past `n`).
    let mut vp = vec![0f32; nh * ng * np * 32];
    let vpp = Ptr(vp.as_mut_ptr());
    crate::pool::run(nh * ng, &|i| {
        let vpp = &vpp;
        let (h, g) = (i / ng, i % ng);
        for key in 0..n {
            // SAFETY: each task writes its own panel.
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr().add(key * d + h * dh + 32 * g), vpp.0.add((i * np + key) * 32), 32);
            }
        }
    });
    let op = Ptr(out.as_mut_ptr());
    crate::pool::run(nh * tiles, &|task| {
        let op = &op;
        let (h, qt) = (task / tiles, task % tiles);
        let rows = (n - 32 * qt).min(32);
        attn_scratch::SCRATCH.with(|sc| {
            let mut sc = sc.borrow_mut();
            let AttnScratch { qp, s, c } = &mut *sc;
            if qp.len() < dh * 32 {
                qp.resize(dh * 32, 0.0);
            }
            if s.len() < np * 32 {
                s.resize(np * 32, 0.0);
            }
            c.resize(32 * 32, 0.0);
            // SAFETY: the panels hold whole tiles.
            unsafe {
                pack32(q.as_ptr().add(32 * qt * d + h * dh), d, rows, dh, qp.as_mut_ptr());
                qp[..dh * 32].iter_mut().for_each(|x| *x *= scale);
                kern32_seq(qp.as_ptr(), kp.as_ptr().add(h * tiles * dh * 32), dh, tiles, s.as_mut_ptr());
            }
            let (live, pad) = s[..np * 32].split_at_mut(n * 32);
            let (rows_s, _) = live.as_chunks_mut::<32>();
            let mut m = [f32::NEG_INFINITY; 32];
            for r in rows_s.iter() {
                for (a, &x) in m.iter_mut().zip(r) {
                    *a = a.max(x);
                }
            }
            let mut l = [0f32; 32];
            for r in rows_s.iter_mut() {
                for i in 0..32 {
                    let e = crate::kernels::fast_exp(r[i] - m[i]);
                    r[i] = e;
                    l[i] += e;
                }
            }
            pad.fill(0.0);
            for g in 0..ng {
                // SAFETY: as above; the task writes its own rows of this head's outputs.
                unsafe {
                    kern32(vp.as_ptr().add((h * ng + g) * np * 32), s.as_ptr(), np, c.as_mut_ptr());
                    for qi in 0..rows {
                        let inv = 1.0 / l[qi];
                        let o = op.0.add((32 * qt + qi) * d + h * dh + 32 * g);
                        for r in 0..32 {
                            *o.add(r) = c[qi * 32 + r] * inv;
                        }
                    }
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::{GgmlType, GgufMatrix, random_blocks};

    #[test]
    fn attention_matches_f64_reference() {
        if !available() {
            eprintln!("no SME: skipped");
            return;
        }
        // Partial last tiles of queries and keys, one and two groups of 32 dimensions, heads.
        for (n, nh, dh) in [(70usize, 3usize, 64usize), (32, 1, 32), (5, 2, 32)] {
            let d = nh * dh;
            let mut s = 1u64;
            let mut r = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) % 2001) as f32 / 500.0 - 2.0
            };
            let q: Vec<f32> = (0..n * d).map(|_| r()).collect();
            let k: Vec<f32> = (0..n * d).map(|_| r()).collect();
            let v: Vec<f32> = (0..n * d).map(|_| r()).collect();
            let scale = 1.0 / (dh as f32).sqrt();
            let mut out = vec![f32::NAN; n * d];
            attention(&q, &k, &v, n, nh, dh, scale, &mut out);
            for h in 0..nh {
                for i in 0..n {
                    let sc: Vec<f64> = (0..n)
                        .map(|j| {
                            (0..dh)
                                .map(|c| q[i * d + h * dh + c] as f64 * k[j * d + h * dh + c] as f64)
                                .sum::<f64>()
                                * scale as f64
                        })
                        .collect();
                    let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                    let l: f64 = e.iter().sum();
                    for c in 0..dh {
                        let want: f64 = (0..n).map(|j| e[j] * v[j * d + h * dh + c] as f64).sum::<f64>() / l;
                        let got = out[i * d + h * dh + c] as f64;
                        assert!(
                            (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                            "n={n} head {h} query {i} dim {c}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn matches_f64_reference() {
        if !available() {
            eprintln!("no SME: skipped");
            return;
        }
        // Several shapes in one thread (the scratch buffers regrow), rows past one call's panels,
        // partial panels and token tiles; Q4_K and BF16 (float-stored) rows too.
        for (kind, n_out, n_in, t) in [
            (GgmlType::Q8_0, 45, 96, 37),
            (GgmlType::Q4_K, 70, 2048, 33),
            (GgmlType::Q8_0, 33, 160, 64),
            (GgmlType::BF16, 40, 48, 5),
        ] {
            let mut seed = 5u64;
            let bytes = random_blocks(kind, n_out * n_in / kind.block_size(), &mut seed);
            let m = GgufMatrix::from_bytes(kind, &bytes, n_out, n_in).unwrap();
            let x: Vec<f32> = (0..t * n_in).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect();
            let mut w = vec![0f32; n_out * n_in];
            m.dequantize_rows(0, n_out, &mut w);
            let xt = pack_x(&x, t, n_in);
            let mut y = vec![f32::NAN; t * n_out];
            // Rows 0..32 in two token ranges (0..32, 32..t), the rest in one call.
            unsafe {
                gemm_rows(&m, &xt, 0, t.min(32), 0, 32, y.as_mut_ptr());
                gemm_rows(&m, &xt, t.min(32), t, 0, 32, y.as_mut_ptr());
                gemm_rows(&m, &xt, 0, t, 32, n_out, y.as_mut_ptr());
            }
            for c in 0..t {
                for r in 0..n_out {
                    let want: f64 = (0..n_in).map(|k| w[r * n_in + k] as f64 * x[c * n_in + k] as f64).sum();
                    let got = y[c * n_out + r] as f64;
                    let scale: f64 = (0..n_in).map(|k| (w[r * n_in + k] as f64 * x[c * n_in + k] as f64).abs()).sum();
                    assert!((got - want).abs() <= 1e-5 * scale.max(1e-30), "{kind} [{c}, {r}]: {got} vs {want}");
                }
            }
        }
    }
}
