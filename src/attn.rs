//! Decode attention for query heads that share their key / value rows (grouped-query attention's
//! groups, the single key-value head of multi-query attention, an MLA latent cache).
//!
//! The per-head kernels this replaces gave each (query, head) its own task, so a key / value row
//! shared by `g` heads was read `g` times, and a decode step with few key-value heads had few
//! tasks. Here one step runs in three parallel phases, each reading every shared row once:
//!
//! 1. scores: tasks over blocks of keys, each scoring its keys against every head that shares them
//!    ([`dots`]: one load of a key serves up to 8 heads);
//! 2. weights: per (unit, head) the caller's softmax over the whole score row;
//! 3. values: tasks over column slices of the value width (and blocks of heads), each
//!    accumulating its heads' output slices over all keys in order ([`weighted_rows`]: a value
//!    slice loaded once serves 4 heads).
//!
//! The arithmetic of every (query, head) is unchanged from the per-head kernels: the same dot
//! products (8 lanes, summed in the same order), the same softmax, and outputs accumulated `o +=
//! w_j v_j` element by element in key order (a column slice does not reorder a sum). So results
//! are bit-identical to them, for any number of queries in the pass: a speculative verification
//! pass gives each token the bits of a one-token step. (A single-pass online softmax, as in
//! flash-decoding kernels, would rescale partial sums and change the bits.)

#[cfg(test)]
use crate::kernels::axpy;
use crate::kernels::dot;
use crate::pool;
use std::ops::Range;

/// `out[g] = dot(q[g], k)` for every row of `q`, bit for bit [`dot`], with `k` loaded once per 8
/// rows.
#[inline]
pub(crate) fn dots(q: &[&[f32]], k: &[f32], out: &mut [f32]) {
    debug_assert_eq!(q.len(), out.len());
    let mut g = 0;
    while g + 8 <= q.len() {
        out[g..g + 8].copy_from_slice(&dot_n::<8>(q[g..g + 8].try_into().unwrap(), k));
        g += 8;
    }
    if g + 4 <= q.len() {
        out[g..g + 4].copy_from_slice(&dot_n::<4>(q[g..g + 4].try_into().unwrap(), k));
        g += 4;
    }
    if g + 2 <= q.len() {
        out[g..g + 2].copy_from_slice(&dot_n::<2>(q[g..g + 2].try_into().unwrap(), k));
        g += 2;
    }
    if g < q.len() {
        out[g] = dot(q[g], k);
    }
}

/// `G` dot products against one `k`, each with [`dot`]'s 8 lanes and summation order.
#[inline(always)]
fn dot_n<const G: usize>(q: &[&[f32]; G], k: &[f32]) -> [f32; G] {
    let n = k.len();
    let (kc, _) = k.as_chunks::<8>();
    let qp: [*const [f32; 8]; G] = std::array::from_fn(|g| {
        assert_eq!(q[g].len(), n);
        q[g].as_ptr() as *const [f32; 8]
    });
    let acc = lanes::<G>(&qp, kc);
    std::array::from_fn(|g| {
        let mut s = acc[g].iter().sum::<f32>();
        for i in kc.len() * 8..n {
            s += q[g][i] * k[i];
        }
        s
    })
}

/// The 8 lane sums `acc[g][j] = sum_i q[g][8 i + j] k[8 i + j]` (in order of `i`, a multiply
/// then an add, as [`dot`]); every `qp[g]` points to `kc.len()` chunks.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn lanes<const G: usize>(qp: &[*const [f32; 8]; G], kc: &[[f32; 8]]) -> [[f32; 8]; G] {
    use std::arch::aarch64::*;
    // SAFETY: NEON is baseline on aarch64; the pointers cover `kc.len()` chunks (caller).
    unsafe {
        let z = vdupq_n_f32(0.0);
        let mut lo = [z; G];
        let mut hi = [z; G];
        for (i, k8) in kc.iter().enumerate() {
            let k0 = vld1q_f32(k8.as_ptr());
            let k1 = vld1q_f32(k8.as_ptr().add(4));
            for g in 0..G {
                let qg = qp[g].add(i) as *const f32;
                lo[g] = vaddq_f32(lo[g], vmulq_f32(vld1q_f32(qg), k0));
                hi[g] = vaddq_f32(hi[g], vmulq_f32(vld1q_f32(qg.add(4)), k1));
            }
        }
        std::array::from_fn(|g| {
            let mut a = [0f32; 8];
            vst1q_f32(a.as_mut_ptr(), lo[g]);
            vst1q_f32(a.as_mut_ptr().add(4), hi[g]);
            a
        })
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn lanes<const G: usize>(qp: &[*const [f32; 8]; G], kc: &[[f32; 8]]) -> [[f32; 8]; G] {
    let mut acc = [[0f32; 8]; G];
    for (i, k8) in kc.iter().enumerate() {
        for g in 0..G {
            // SAFETY: every `qp[g]` covers `kc.len()` chunks (caller).
            let q8 = unsafe { &*qp[g].add(i) };
            for j in 0..8 {
                acc[g][j] += q8[j] * k8[j];
            }
        }
    }
    acc
}

/// Scores of the query rows `q` against `key(j)` for `j` in `keys`, into `out` (`[q.len()][
/// keys.len()]`), each times `scale` when given: per block of 8 heads the block's keys are
/// walked, so both stay in the nearest cache.
pub(crate) fn score_rows<'k>(q: &[&[f32]], keys: Range<usize>, key: &dyn Fn(usize) -> &'k [f32], scale: Option<f32>, out: &mut [f32]) {
    let len = keys.len();
    let mut s = [0f32; 8];
    for (b, qb) in q.chunks(8).enumerate() {
        for (jj, j) in keys.clone().enumerate() {
            let k = key(j);
            dots(qb, k, &mut s[..qb.len()]);
            for (h, &v) in s[..qb.len()].iter().enumerate() {
                out[(b * 8 + h) * len + jj] = match scale {
                    Some(f) => v * f,
                    None => v,
                };
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}
impl Ptr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// A value row's address.
#[derive(Clone, Copy)]
struct Row(*const f32);
unsafe impl Send for Row {}
unsafe impl Sync for Row {}

#[derive(Clone, Copy)]
struct RowPtr(*mut Row);
unsafe impl Send for RowPtr {}
unsafe impl Sync for RowPtr {}
impl RowPtr {
    #[inline(always)]
    fn get(self) -> *mut Row {
        self.0
    }
}

/// `scores(unit, keys, out)` of [`attend_shared`].
pub(crate) type ScoreFn<'a> = dyn Fn(usize, Range<usize>, &mut [f32]) + Sync + 'a;

/// One shared-row attention step (see the module docs) over `units` (a query and the group of
/// `heads` query heads that share one key / value sequence), unit `u` attending to its `n_of(u)`
/// keys:
///
/// - `scores(u, keys, out)` writes the raw scores of keys `keys` (`[heads][keys.len()]`);
/// - `weights(u, h, row)` turns head `h`'s whole score row into the attention weights in place;
/// - `value(u, j)` is the value row (at least `vw` wide) of key `j`;
/// - `out_row(u, h)` is where head `h`'s output row (`vw`) starts in `out`, which is written as
///   `sum_j w_j value(u, j)`, accumulated from zero in key order, element by element.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_shared<'v>(
    units: usize,
    heads: usize,
    vw: usize,
    n_of: &(dyn Fn(usize) -> usize + Sync),
    scores: &ScoreFn,
    weights: &(dyn Fn(usize, usize, &mut [f32]) + Sync),
    value: &(dyn Fn(usize, usize) -> &'v [f32] + Sync),
    out_row: &(dyn Fn(usize, usize) -> usize + Sync),
    out: &mut [f32],
) {
    let threads = pool::threads();
    let ns: Vec<usize> = (0..units).map(n_of).collect();
    // Per unit: where its keys start in `vrows` (times `heads`: its score rows in `sc`).
    let mut koffs = Vec::with_capacity(units + 1);
    koffs.push(0);
    for &n in &ns {
        koffs.push(koffs.last().unwrap() + n);
    }
    let total = koffs[units];
    let mut sc = vec![0f32; heads * total];
    let sp = Ptr(sc.as_mut_ptr());
    let mut vrows: Vec<Row> = vec![Row(std::ptr::null()); total];
    let vp = RowPtr(vrows.as_mut_ptr());

    // 1. Scores (and the value rows' addresses), over blocks of keys: enough blocks to keep every
    // thread busy, at least 16 keys each so the queries' loads are amortized; key-block-major, so
    // the tasks running together read neighbouring rows of every unit (the units of one query
    // are usually interleaved in memory, as KV heads are).
    let kc = total.div_ceil(3 * threads).max(16);
    let nmax = ns.iter().copied().max().unwrap_or(0);
    let nsr = &ns;
    let tasks: Vec<(usize, usize)> = (0..nmax)
        .step_by(kc)
        .flat_map(|j| (0..units).filter(move |&u| j < nsr[u]).map(move |u| (u, j)))
        .collect();
    pool::run(tasks.len(), &|ti| {
        let (u, j0) = tasks[ti];
        let (n, j1) = (ns[u], (j0 + kc).min(ns[u]));
        let len = j1 - j0;
        let mut buf = vec![0f32; heads * len];
        scores(u, j0..j1, &mut buf);
        let base = heads * koffs[u];
        for h in 0..heads {
            // SAFETY: tasks write disjoint (unit, head, key) entries of `sc` and disjoint (unit,
            // key) entries of `vrows`, both outliving the call.
            let dst = unsafe { std::slice::from_raw_parts_mut(sp.get().add(base + h * n + j0), len) };
            dst.copy_from_slice(&buf[h * len..(h + 1) * len]);
        }
        for j in j0..j1 {
            let v = value(u, j);
            assert!(v.len() >= vw);
            // SAFETY: as above.
            unsafe { *vp.get().add(koffs[u] + j) = Row(v.as_ptr()) };
        }
    });

    // 2. Weights, per (unit, head).
    pool::run(units * heads, &|ti| {
        let (u, h) = (ti / heads, ti % heads);
        let n = ns[u];
        // SAFETY: each task owns one row of `sc`.
        let row = unsafe { std::slice::from_raw_parts_mut(sp.get().add(heads * koffs[u] + h * n), n) };
        weights(u, h, row);
    });

    // 3. Values, over column slices of the output rows and blocks of heads (a value row's slice
    // is read once per 4 heads, from a near cache).
    let (mut cw, mut hb) = (vw.min(64), heads);
    while units * vw.div_ceil(cw) * heads.div_ceil(hb) < 2 * threads {
        if cw > 16 {
            cw /= 2;
        } else if hb > 4 {
            hb = hb.div_ceil(2).next_multiple_of(4);
        } else {
            break;
        }
    }
    let (slices, hblocks) = (vw.div_ceil(cw), heads.div_ceil(hb));
    let op = Ptr(out.as_mut_ptr());
    let out_len = out.len();
    let (sc, vrows) = (&sc, &vrows);
    pool::run(units * hblocks * slices, &|ti| {
        let (u, rest) = (ti / (hblocks * slices), ti % (hblocks * slices));
        let (h0, s) = ((rest / slices) * hb, rest % slices);
        let h1 = (h0 + hb).min(heads);
        let (c0, c1) = (s * cw, ((s + 1) * cw).min(vw));
        let n = ns[u];
        let w = &sc[heads * koffs[u]..heads * koffs[u + 1]];
        let rows = &vrows[koffs[u]..koffs[u + 1]];
        let dst = |h: usize| -> *mut f32 {
            let o = out_row(u, h) + c0;
            assert!(o + (c1 - c0) <= out_len);
            // SAFETY: in bounds (checked); (unit, head, column slice) ranges of `out` are
            // disjoint across tasks.
            unsafe { op.get().add(o) }
        };
        let mut h = h0;
        while h < h1 {
            let r = (h1 - h).min(4);
            let ws: Vec<&[f32]> = (h..h + r).map(|hh| &w[hh * n..(hh + 1) * n]).collect();
            let ds: Vec<*mut f32> = (h..h + r).map(dst).collect();
            // SAFETY: `rows` hold value rows at least `vw >= c1` wide; `ds` point to `c1 - c0`
            // writable floats each.
            unsafe {
                match r {
                    4 => weighted_rows::<4>(&[ws[0], ws[1], ws[2], ws[3]], rows, c0, c1, &[ds[0], ds[1], ds[2], ds[3]]),
                    3 => weighted_rows::<3>(&[ws[0], ws[1], ws[2]], rows, c0, c1, &[ds[0], ds[1], ds[2]]),
                    2 => weighted_rows::<2>(&[ws[0], ws[1]], rows, c0, c1, &[ds[0], ds[1]]),
                    _ => weighted_rows::<1>(&[ws[0]], rows, c0, c1, &[ds[0]]),
                }
            }
            h += r;
        }
    });
}

/// `out[r][c - c0] = sum_j w[r][j] rows[j][c]` for `c` in `c0..c1`, from zero in order of `j`
/// (a multiply then an add, as an `axpy` into a zeroed row).
///
/// # Safety
/// Every `rows[j]` points to at least `c1` floats, every `out[r]` to `c1 - c0` writable ones,
/// every `w[r]` has `rows.len()` entries.
#[inline(always)]
unsafe fn weighted_rows<const R: usize>(w: &[&[f32]; R], rows: &[Row], c0: usize, c1: usize, out: &[*mut f32; R]) {
    let mut c = c0;
    #[cfg(target_arch = "aarch64")]
    while c + 16 <= c1 {
        use std::arch::aarch64::*;
        // SAFETY: NEON is baseline on aarch64; bounds per the contract.
        unsafe {
            let z = vdupq_n_f32(0.0);
            let mut acc = [[z; 4]; R];
            for (j, p) in rows.iter().enumerate() {
                let p = p.0.add(c);
                let v = [vld1q_f32(p), vld1q_f32(p.add(4)), vld1q_f32(p.add(8)), vld1q_f32(p.add(12))];
                for r in 0..R {
                    let x = *w[r].get_unchecked(j);
                    for q in 0..4 {
                        acc[r][q] = vaddq_f32(acc[r][q], vmulq_n_f32(v[q], x));
                    }
                }
            }
            for r in 0..R {
                for q in 0..4 {
                    vst1q_f32(out[r].add(c - c0 + 4 * q), acc[r][q]);
                }
            }
        }
        c += 16;
    }
    while c < c1 {
        let cn = (c + 8).min(c1);
        let mut acc = [[0f32; 8]; R];
        for (j, p) in rows.iter().enumerate() {
            let p = p.0;
            for r in 0..R {
                let x = w[r][j];
                for (i, a) in acc[r][..cn - c].iter_mut().enumerate() {
                    // SAFETY: per the contract.
                    *a += x * unsafe { *p.add(c + i) };
                }
            }
        }
        for r in 0..R {
            // SAFETY: per the contract.
            unsafe { std::ptr::copy_nonoverlapping(acc[r].as_ptr(), out[r].add(c - c0), cn - c) };
        }
        c = cn;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::softmax_inplace;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    #[test]
    fn dots_match_dot_bitwise() {
        let mut seed = 7;
        for &d in &[1usize, 7, 8, 64, 100, 128, 256, 512, 513] {
            for g in 1..=19 {
                let q: Vec<Vec<f32>> = (0..g).map(|_| (0..d).map(|_| lcg(&mut seed)).collect()).collect();
                let k: Vec<f32> = (0..d).map(|_| lcg(&mut seed)).collect();
                let qr: Vec<&[f32]> = q.iter().map(|v| &v[..]).collect();
                let mut out = vec![0f32; g];
                dots(&qr, &k, &mut out);
                for (o, qv) in out.iter().zip(&q) {
                    assert_eq!(o.to_bits(), dot(qv, &k).to_bits(), "d {d} g {g}");
                }
            }
        }
    }

    /// The shared-row kernel against the per-head loop it replaces (GQA, causal), bit for bit.
    #[test]
    fn shared_matches_per_head() {
        // (KV heads, heads per KV head, head width, queries, cached positions before them):
        // GQA with a column remainder, MQA as DeepSeek V4's, Qwen3-like, a short one.
        for cfg in [
            (2, 6, 72, 3, 200),
            (1, 64, 512, 1, 300),
            (8, 4, 128, 2, 1000),
            (1, 5, 40, 1, 17),
            (2, 12, 256, 1, 700),
        ] {
            shared_case(cfg);
        }
    }

    fn shared_case((nkv, rep, hd, t, start): (usize, usize, usize, usize, usize)) {
        let mut seed = 11 + hd as u64;
        let nh = nkv * rep;
        let len = start + t;
        let k: Vec<f32> = (0..len * nkv * hd).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = (0..len * nkv * hd).map(|_| lcg(&mut seed)).collect();
        let q: Vec<f32> = (0..t * nh * hd).map(|_| lcg(&mut seed) * 0.5).collect();
        let kat = |p: usize, g: usize| &k[(p * nkv + g) * hd..(p * nkv + g + 1) * hd];
        let vat = |p: usize, g: usize| &v[(p * nkv + g) * hd..(p * nkv + g + 1) * hd];
        let mut want = vec![0f32; t * nh * hd];
        for ti in 0..t {
            for h in 0..nh {
                let g = h / rep;
                let n = start + ti + 1;
                let qh = &q[(ti * nh + h) * hd..(ti * nh + h + 1) * hd];
                let mut s: Vec<f32> = (0..n).map(|j| dot(qh, kat(j, g))).collect();
                softmax_inplace(&mut s);
                let o = &mut want[(ti * nh + h) * hd..(ti * nh + h + 1) * hd];
                for (j, &p) in s.iter().enumerate() {
                    axpy(o, p, vat(j, g));
                }
            }
        }
        let mut got = vec![0f32; t * nh * hd];
        attend_shared(
            t * nkv,
            rep,
            hd,
            &|u| start + u / nkv + 1,
            &|u, keys, out| {
                let (ti, g) = (u / nkv, u % nkv);
                let qs: Vec<&[f32]> = (0..rep)
                    .map(|r| &q[(ti * nh + g * rep + r) * hd..(ti * nh + g * rep + r + 1) * hd])
                    .collect();
                score_rows(&qs, keys, &|j| kat(j, g), None, out);
            },
            &|_, _, row| softmax_inplace(row),
            &|u, j| vat(j, u % nkv),
            &|u, r| ((u / nkv) * nh + (u % nkv) * rep + r) * hd,
            &mut got,
        );
        for (a, b) in got.iter().zip(&want) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::kernels::softmax_inplace;

    /// `cargo test --release --lib attn::bench -- --ignored --nocapture` (ATTN_N, ATTN_KV,
    /// ATTN_REP, ATTN_HD): one decode step, per-head tasks against the shared-row kernel.
    #[test]
    #[ignore]
    fn shared_vs_per_head() {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (n, nkv, rep, hd) = (env("ATTN_N", 16384), env("ATTN_KV", 8), env("ATTN_REP", 4), env("ATTN_HD", 128));
        let nh = nkv * rep;
        let k: Vec<f32> = (0..n * nkv * hd).map(|i| ((i * 7919) % 1000) as f32 * 1e-4).collect();
        let v = k.clone();
        let q: Vec<f32> = (0..nh * hd).map(|i| ((i * 31) % 100) as f32 * 1e-3).collect();
        let kat = |p: usize, g: usize| &k[(p * nkv + g) * hd..(p * nkv + g + 1) * hd];
        let vat = |p: usize, g: usize| &v[(p * nkv + g) * hd..(p * nkv + g + 1) * hd];
        let mut out = vec![0f32; nh * hd];
        let mut out2 = vec![0f32; nh * hd];
        let op = Ptr(out.as_mut_ptr());
        for round in 0..6 {
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                pool::run(nh, &|h| {
                    let g = h / rep;
                    let qh = &q[h * hd..(h + 1) * hd];
                    let mut s: Vec<f32> = (0..n).map(|j| dot(qh, kat(j, g))).collect();
                    softmax_inplace(&mut s);
                    // SAFETY: one row per task.
                    let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(h * hd), hd) };
                    o.fill(0.0);
                    for (j, &p) in s.iter().enumerate() {
                        axpy(o, p, vat(j, g));
                    }
                });
            }
            let a = t0.elapsed().as_secs_f64() / 10.0;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                attend_shared(
                    nkv,
                    rep,
                    hd,
                    &|_| n,
                    &|g, keys, sc| {
                        let qs: Vec<&[f32]> = (0..rep).map(|r| &q[(g * rep + r) * hd..(g * rep + r + 1) * hd]).collect();
                        score_rows(&qs, keys, &|j| kat(j, g), None, sc);
                    },
                    &|_, _, row| softmax_inplace(row),
                    &|g, j| vat(j, g),
                    &|g, r| (g * rep + r) * hd,
                    &mut out2,
                );
            }
            let b = t0.elapsed().as_secs_f64() / 10.0;
            assert!(out.iter().zip(&out2).all(|(x, y)| x.to_bits() == y.to_bits()));
            let mb = (2 * n * nkv * hd * 4) as f64 / 1e6;
            println!(
                "round {round}: per-head {:.3} ms, shared {:.3} ms ({mb:.0} MB of K+V: {:.0} / {:.0} GB/s)",
                a * 1e3,
                b * 1e3,
                mb / a / 1e3,
                mb / b / 1e3
            );
        }
    }
}
