//! Causal attention over a prompt chunk on aarch64's NEON units (`model::attention_prefill`).
//!
//! The score and value GEMMs this replaces ran on the f32 prefill GEMM: per block of up to 256
//! queries and KV head, four parallel regions (scores, softmax, values, copy-out), each GEMM
//! with the keys or the transposed values as its weights. With the SME GEMM (Apple M4 and
//! later) every thread queued on the few shared SME units with an inner dimension of 64-256,
//! where the panel packing and copies cost about as much as the products: a third of Llama 3.2
//! 1B's CPU prefill at 2k tokens.
//!
//! Here one parallel region takes a task per KV head and tile of queries (the query heads of a
//! few tokens, about 32 rows), on every core. Scores: register tiles of 4 rows by 16 keys, each
//! tile of keys (transposed once per call, 16 keys by the head dimension, contiguous) passing
//! every group of rows while it is in L1. Then the same softmax over each row's keys. Values: 4
//! rows by 16 columns, read straight from the cache, by blocks of keys that pass every group of
//! rows while in L1 (the partial sums stored in the output between blocks). A tile passes only
//! the keys its rows see (to 16), instead of every key of the 256-query block.
//!
//! The arithmetic is the GEMMs': each score is one fused multiply-add chain over the head
//! dimension in order from zero, and each output one chain over the keys in order, so the
//! results are bit-identical (the GEMMs' extra terms are exact zeros: padding, and the weights
//! of keys a row does not see).
#![allow(unsafe_op_in_unsafe_fn)]

use crate::kernels::{Scratch, softmax_inplace};
use crate::model::LayerCache;
use std::arch::aarch64::*;

/// Keys per score tile.
const KT: usize = 16;

/// Keys per block of the values (16 columns of them stay in L1 for every group of rows).
const KV: usize = 128;

/// Query rows (query heads x tokens) per task, about.
const ROWS: usize = 32;

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// Whether [`attention`] takes heads of `hd` values (whole vectors) rather than the GEMMs. With
/// SME, heads wider than 256 stay on its GEMM, whose long inner dimension then pays for the
/// packing (M5 Max, 2k keys, 256 queries: 1.5 TFLOP/s at 512 dimensions against 1.2 here; at
/// 256, 0.95 against 1.0; at 64-128, 0.5-0.8 against 0.9-1.0).
pub(crate) fn supported(hd: usize) -> bool {
    static GEMM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // `OXINFER_ATTN_PREFILL_GEMM=1`: the GEMMs (for comparisons).
    hd > 0
        && hd.is_multiple_of(4)
        && (hd <= 256 || !crate::kernels::sme_available())
        && !*GEMM.get_or_init(|| std::env::var_os("OXINFER_ATTN_PREFILL_GEMM").is_some())
}

/// `out` (`[t, nh, hd]`) = softmax(q k^T) v for the queries `q` (`[t, nh, hd]`, already scaled)
/// against the cached keys and values, query `ti` seeing the keys `lo_of(ti)..hi_of(ti)`
/// (`lo_of` non-decreasing in `ti`).
pub(crate) fn attention(
    q: &[f32],
    cache: &LayerCache,
    nh: usize,
    t: usize,
    lo_of: &(dyn Fn(usize) -> usize + Sync),
    hi_of: &(dyn Fn(usize) -> usize + Sync),
    out: &mut [f32],
) {
    let (hd, n_kv) = (cache.hd, cache.n_kv);
    debug_assert!(hd.is_multiple_of(4) && nh.is_multiple_of(n_kv) && q.len() == t * nh * hd);
    let rep = nh / n_kv;
    let ka = lo_of(0);
    let kb = (0..t).map(hi_of).max().unwrap_or(ka).max(ka);
    debug_assert!(kb <= cache.len);
    let nall = kb - ka;
    let nblk = nall.div_ceil(KT);
    // Keys transposed by tiles of 16, per KV head: `kt[((g * nblk + b) * hd + d) * 16 + l]` is
    // dimension `d` of key `ka + 16 b + l` (zeros past `kb`).
    let mut kt = Scratch::new(n_kv * nblk * hd * KT);
    {
        let p = Ptr(kt.as_mut_ptr());
        let per = 4;
        crate::pool::run(n_kv * nblk.div_ceil(per), &|i| {
            let p = &p;
            let (g, b0) = (i % n_kv, i / n_kv * per);
            for b in b0..(b0 + per).min(nblk) {
                // SAFETY: each task writes its own tiles.
                unsafe {
                    let tile = p.0.add((g * nblk + b) * hd * KT);
                    for l in 0..KT {
                        let j = KT * b + l;
                        if j < nall {
                            for (d, &x) in cache.k_at(ka + j, g).iter().enumerate() {
                                *tile.add(d * KT + l) = x;
                            }
                        } else {
                            for d in 0..hd {
                                *tile.add(d * KT + l) = 0.0;
                            }
                        }
                    }
                }
            }
        });
    }
    let qt = (ROWS / rep).clamp(1, 32);
    let ntiles = t.div_ceil(qt);
    let ctx = Ctx {
        q,
        cache,
        nh,
        rep,
        ka,
        nblk,
        kt: &kt,
        lo_of,
        hi_of,
        out: Ptr(out.as_mut_ptr()),
    };
    // The last tiles (the most keys) first.
    crate::pool::run(ntiles * n_kv, &|task| {
        let (tile, g) = (ntiles - 1 - task / n_kv, task % n_kv);
        // SAFETY: the task writes its own rows of `out`.
        unsafe { ctx.task(g, tile * qt, ((tile + 1) * qt).min(t)) }
    });
}

thread_local! {
    // Already const; clippy (1.100) still flags it on Android's thread-local implementation.
    #[allow(clippy::missing_const_for_thread_local)]
    static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct Ctx<'a> {
    q: &'a [f32],
    cache: &'a LayerCache,
    nh: usize,
    rep: usize,
    ka: usize,
    nblk: usize,
    kt: &'a [f32],
    lo_of: &'a (dyn Fn(usize) -> usize + Sync),
    hi_of: &'a (dyn Fn(usize) -> usize + Sync),
    out: Ptr,
}

impl Ctx<'_> {
    /// The query heads of KV head `g` for tokens `t0..t1`.
    unsafe fn task(&self, g: usize, t0: usize, t1: usize) {
        let (hd, n_kv, nh, rep, ka) = (self.cache.hd, self.cache.n_kv, self.nh, self.rep, self.ka);
        let rows = (t1 - t0) * rep;
        let ngroups = rows.div_ceil(4);
        // Keys of the tile relative to `ka`: from `w0` (a multiple of 16), `w` of them.
        let w0 = ((self.lo_of)(t0) - ka) / KT * KT;
        let w1 = ((t0..t1).map(self.hi_of).max().unwrap() - ka)
            .next_multiple_of(KT)
            .min(self.nblk * KT);
        let w = w1.saturating_sub(w0);
        // Row `r`: token `t0 + r / rep`, head `g * rep + r % rep`; its keys `a..b` of the tile.
        let qrow = |r: usize| self.q.as_ptr().add(((t0 + r / rep) * nh + g * rep + r % rep) * hd);
        let orow = |r: usize| self.out.0.add(((t0 + r / rep) * nh + g * rep + r % rep) * hd);
        let span = |r: usize| {
            let ti = t0 + r / rep;
            let (lo, hi) = ((self.lo_of)(ti) - ka - w0, (self.hi_of)(ti) - ka - w0);
            (lo, hi.max(lo))
        };
        // Each group of 4 rows: its rows and keys `a..b` (the union of theirs).
        let groups: Vec<(usize, usize, usize, usize)> = (0..ngroups)
            .map(|gi| {
                let (r, nr) = (4 * gi, (rows - 4 * gi).min(4));
                let a = (r..r + nr).map(|i| span(i).0).min().unwrap();
                let b = (r..r + nr).map(|i| span(i).1).max().unwrap();
                (r, nr, a, b)
            })
            .collect();
        SCORES.with(|s| {
            let mut s = s.borrow_mut();
            if s.len() < rows * w {
                s.resize(rows * w, 0.0);
            }
            let sp = s.as_mut_ptr();
            // Scores: each tile of 16 keys against every group that sees any of them (the others
            // are zeroed below).
            for j in (0..w).step_by(KT) {
                let kp = self.kt.as_ptr().add((g * self.nblk + (w0 + j) / KT) * hd * KT);
                for &(r, nr, a, b) in &groups {
                    if j + KT <= a || j >= b {
                        continue;
                    }
                    macro_rules! go {
                        ($n:literal) => {
                            scores::<$n>(
                                std::array::from_fn(|i| qrow(r + i)),
                                kp,
                                hd,
                                std::array::from_fn(|i| sp.add((r + i) * w + j)),
                            )
                        };
                    }
                    match nr {
                        1 => go!(1),
                        2 => go!(2),
                        3 => go!(3),
                        _ => go!(4),
                    }
                }
            }
            for i in 0..rows {
                let row = std::slice::from_raw_parts_mut(sp.add(i * w), w);
                let (lo, hi) = span(i);
                row[..lo].fill(0.0);
                row[hi..].fill(0.0);
                softmax_inplace(&mut row[lo..hi]);
            }
            // Values: blocks of keys, each passing every group (whose keys start in the block or
            // before), 16 columns at a time; a group's sums start from zero in the block of its
            // first key and continue from the output in later ones (the same chain).
            let amin = groups.iter().map(|x| x.2).min().unwrap();
            let bmax = groups.iter().map(|x| x.3).max().unwrap();
            for &(r, nr, a, b) in &groups {
                if a == b {
                    for i in r..r + nr {
                        std::ptr::write_bytes(orow(i), 0, hd);
                    }
                }
            }
            let mut k0 = amin;
            while k0 < bmax {
                let k1 = (k0 + KV).min(bmax);
                let mut c = 0;
                while c < hd {
                    let nv = ((hd - c) / 4).min(4);
                    for &(r, nr, a, b) in &groups {
                        let (j0, j1) = (a.max(k0), b.min(k1));
                        if j0 >= j1 {
                            continue;
                        }
                        let first = a >= k0;
                        let v = self.cache.v_from(ka + w0 + j0, g).as_ptr().add(c);
                        macro_rules! go {
                            ($r:literal, $v:literal) => {
                                values::<$r, $v>(
                                    std::array::from_fn(|i| sp.add((r + i) * w + j0) as *const f32),
                                    v,
                                    n_kv * hd,
                                    j1 - j0,
                                    first,
                                    std::array::from_fn(|i| orow(r + i).add(c)),
                                )
                            };
                        }
                        macro_rules! rows {
                            ($v:literal) => {
                                match nr {
                                    1 => go!(1, $v),
                                    2 => go!(2, $v),
                                    3 => go!(3, $v),
                                    _ => go!(4, $v),
                                }
                            };
                        }
                        match nv {
                            1 => rows!(1),
                            2 => rows!(2),
                            3 => rows!(3),
                            _ => rows!(4),
                        }
                    }
                    c += 4 * nv;
                }
                k0 = k1;
            }
        });
    }
}

/// `s[i][0..16] = q_i . k_j` for a tile of 16 keys (`kt[d * 16 + l]`): fused multiply-adds over
/// the dimensions in order from zero.
#[inline(always)]
unsafe fn scores<const NR: usize>(qr: [*const f32; NR], kt: *const f32, hd: usize, s: [*mut f32; NR]) {
    let mut acc = [[vdupq_n_f32(0.0); 4]; NR];
    let mut d = 0;
    while d < hd {
        let qv: [float32x4_t; NR] = std::array::from_fn(|i| vld1q_f32(qr[i].add(d)));
        macro_rules! lane {
            ($l:literal) => {{
                let kp = kt.add((d + $l) * KT);
                let k = [
                    vld1q_f32(kp),
                    vld1q_f32(kp.add(4)),
                    vld1q_f32(kp.add(8)),
                    vld1q_f32(kp.add(12)),
                ];
                for i in 0..NR {
                    for c in 0..4 {
                        acc[i][c] = vfmaq_laneq_f32::<$l>(acc[i][c], k[c], qv[i]);
                    }
                }
            }};
        }
        lane!(0);
        lane!(1);
        lane!(2);
        lane!(3);
        d += 4;
    }
    for i in 0..NR {
        for c in 0..4 {
            vst1q_f32(s[i].add(4 * c), acc[i][c]);
        }
    }
}

/// `o_i[0..4 NV] (+)= sum_j p_i[j] v_j` over `n` value rows from `v` (`stride` apart), from zero
/// if `first`, else from the sums in `o`: fused multiply-adds over the keys in order.
#[inline(always)]
unsafe fn values<const NR: usize, const NV: usize>(
    p: [*const f32; NR],
    v: *const f32,
    stride: usize,
    n: usize,
    first: bool,
    o: [*mut f32; NR],
) {
    let mut acc: [[float32x4_t; NV]; NR] = if first {
        [[vdupq_n_f32(0.0); NV]; NR]
    } else {
        std::array::from_fn(|i| std::array::from_fn(|c| vld1q_f32(o[i].add(4 * c))))
    };
    let mut j = 0;
    while j + 4 <= n {
        let pv: [float32x4_t; NR] = std::array::from_fn(|i| vld1q_f32(p[i].add(j)));
        macro_rules! lane {
            ($l:literal) => {{
                let vp = v.add((j + $l) * stride);
                let vv: [float32x4_t; NV] = std::array::from_fn(|c| vld1q_f32(vp.add(4 * c)));
                for i in 0..NR {
                    for c in 0..NV {
                        acc[i][c] = vfmaq_laneq_f32::<$l>(acc[i][c], vv[c], pv[i]);
                    }
                }
            }};
        }
        lane!(0);
        lane!(1);
        lane!(2);
        lane!(3);
        j += 4;
    }
    while j < n {
        let vp = v.add(j * stride);
        let vv: [float32x4_t; NV] = std::array::from_fn(|c| vld1q_f32(vp.add(4 * c)));
        for i in 0..NR {
            let pj = *p[i].add(j);
            for c in 0..NV {
                acc[i][c] = vfmaq_n_f32(acc[i][c], vv[c], pj);
            }
        }
        j += 1;
    }
    for i in 0..NR {
        for c in 0..NV {
            vst1q_f32(o[i].add(4 * c), acc[i][c]);
        }
    }
}
