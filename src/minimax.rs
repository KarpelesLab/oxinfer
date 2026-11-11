//! MiniMax models on the Qwen code path (`crate::qwen`): M2 / M2.1 / M2.5 / M2.7 need only
//! configuration (query / key norms over all heads, sigmoid routing with a selection bias, see
//! `crate::config::QwenConfig`); M3 / M3.1 adds the MiniMax Sparse Attention indexer here.
//!
//! M3's sparse layers (all but the leading dense ones; the checkpoint holds indexer weights for
//! exactly those) score every cached position with a small "lightning indexer": one index head
//! per KV head, `q . k` of RMS-normed, partially rotated projections of the attention input. The
//! scores are max-pooled into blocks of `block` positions, the `local` blocks ending at the query
//! are forced in, and the query's KV group attends only to the keys of its `topk` best blocks
//! (causally). A query that can see at most `topk` blocks attends to everything, so below
//! `topk * block` positions (2048 for the releases) the layers are exactly dense; past that the
//! selection runs per (token, KV head) and attention reads only those keys
//! ([`crate::qwen4exp::attention_lists`]). The index keys of each layer are cached beside its
//! keys and values (`LayerCache::aux_push`), so rewinds and snapshots keep them in step.

use crate::config::IndexerConfig;
use crate::kernels::*;
use crate::model::{LayerCache, Lin, linear_multi};
use crate::pool;
use crate::qwen::Loader;
use anyhow::{Result, bail};

/// The lightning indexer of one sparse attention layer.
pub(crate) struct Indexer<'a> {
    pub(crate) q: Lin<'a>, // [heads * dim, H]
    pub(crate) k: Lin<'a>, // [dim, H]
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) cfg: IndexerConfig,
}

impl<'a> Indexer<'a> {
    /// The indexer under `ap` (`...self_attn.`), if the layer has one: `index_q_proj`, ...
    /// (MiniMax's release) or `indexer.q_proj`, ... (transformers, GGUF).
    pub(crate) fn load(ld: &Loader<'a, '_>, ap: &str) -> Result<Option<Self>> {
        let p = if ld.has(&format!("{ap}index_q_proj.weight")) {
            format!("{ap}index_")
        } else if ld.has(&format!("{ap}indexer.q_proj.weight")) {
            format!("{ap}indexer.")
        } else {
            return Ok(None);
        };
        let Some(c) = ld.cfg.indexer() else {
            bail!("{ap}: indexer weights, but no sparse attention settings in the config");
        };
        if c.heads != ld.cfg.num_key_value_heads || c.block == 0 || c.topk == 0 {
            bail!(
                "{ap}: indexer of {} heads for {} KV heads (one per KV head expected)",
                c.heads,
                ld.cfg.num_key_value_heads
            );
        }
        if c.dim < ld.cfg.rotary_dim() {
            bail!("{ap}: indexer heads of {} narrower than the rotary dimensions", c.dim);
        }
        let h = ld.cfg.hidden_size;
        Ok(Some(Self {
            q: ld.lin(&format!("{p}q_proj.weight"), c.heads * c.dim, h)?,
            k: ld.lin(&format!("{p}k_proj.weight"), c.dim, h)?,
            q_norm: ld.norm(&format!("{p}q_norm.weight"), c.dim)?,
            k_norm: ld.norm(&format!("{p}k_norm.weight"), c.dim)?,
            cfg: c,
        }))
    }

    pub(crate) fn lins(&self) -> [&Lin<'a>; 2] {
        [&self.q, &self.k]
    }

    /// Appends the index keys of `t` tokens (`hn`: the attention input, at `start`) to the cache
    /// (after their keys and values were pushed), then returns the positions each (token, KV
    /// head) attends (`[t * n_kv]` sorted lists), or None when every one sees all its blocks
    /// (dense attention). `cos` / `sin`: the rotary rows of the `t` tokens (`half` each).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select(
        &self,
        cache: &mut LayerCache,
        hn: &[f32],
        t: usize,
        h: usize,
        start: usize,
        cos: &[f32],
        sin: &[f32],
        half: usize,
        eps: f32,
    ) -> Option<Vec<Vec<u32>>> {
        let IndexerConfig {
            heads,
            dim,
            block,
            topk,
            local,
        } = self.cfg;
        let mut q = vec![0f32; t * heads * dim];
        let mut k = vec![0f32; t * dim];
        linear_multi(&mut [(&self.q, heads * dim, &mut q[..]), (&self.k, dim, &mut k[..])], h, hn, t);
        let par = t > 1;
        let rope = |r: &mut [f32], ti: usize| {
            apply_rope(
                &mut r[..2 * half],
                &cos[ti * half..(ti + 1) * half],
                &sin[ti * half..(ti + 1) * half],
            )
        };
        for_chunks(&mut q, dim, par, |i, r| {
            rmsnorm_f32w_inplace(r, &self.q_norm, eps);
            rope(r, i / heads);
        });
        for_chunks(&mut k, dim, par, |i, r| {
            rmsnorm_f32w_inplace(r, &self.k_norm, eps);
            rope(r, i);
        });
        cache.aux_push(&k, dim);
        // The last query sees the most blocks: when even it sees at most `topk`, all are dense.
        if (start + t - 1) / block < topk {
            return None;
        }
        let keys = cache.aux();
        let _timer = crate::prof::start(crate::prof::Cat::Attention);
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); t * heads];
        let lp = SendLists(lists.as_mut_ptr());
        pool::run(t * heads, &|task| {
            let ti = task / heads;
            let qpos = start + ti;
            let nb = qpos / block + 1;
            let sel: Vec<u32> = if nb <= topk {
                (0..=qpos as u32).collect()
            } else {
                let qr = &q[task * dim..(task + 1) * dim];
                // Max-pooled scores of the blocks up to the query's (causal within its own).
                let mut bs: Vec<f32> = (0..nb)
                    .map(|b| {
                        (b * block..((b + 1) * block).min(qpos + 1))
                            .map(|j| dot(qr, &keys[j * dim..(j + 1) * dim]))
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .collect();
                for l in 0..local {
                    bs[(nb - 1).saturating_sub(l)] = f32::INFINITY;
                }
                let mut idx: Vec<usize> = (0..nb).collect();
                let ord = |a: &usize, b: &usize| bs[*b].total_cmp(&bs[*a]).then(a.cmp(b));
                idx.select_nth_unstable_by(topk, ord);
                idx.truncate(topk);
                idx.sort_unstable();
                idx.iter()
                    .flat_map(|&b| (b * block) as u32..(((b + 1) * block).min(qpos + 1)) as u32)
                    .collect()
            };
            // SAFETY: each task writes its own list.
            unsafe { *lp.get().add(task) = sel };
        });
        Some(lists)
    }
}

#[derive(Clone, Copy)]
struct SendLists(*mut Vec<u32>);
unsafe impl Send for SendLists {}
unsafe impl Sync for SendLists {}
impl SendLists {
    fn get(self) -> *mut Vec<u32> {
        self.0
    }
}

/// MiniMax-01's lightning attention (linear attention with a fixed decay per head), the mixer
/// of the layers between its softmax attention ones. With `q, k, v = silu(qkv_proj(x))` split per
/// head, each head keeps a `[hd, hd]` state: `S = exp(-slope) * S + k^T v`, `o = q S` (the
/// blockwise form of the reference computes the same sums); the heads' outputs are RMS-normed
/// together, gated by `sigmoid(output_gate(x))` and projected by `out_proj`.
pub(crate) struct Lightning<'a> {
    pub(crate) qkv: Lin<'a>,  // [3 * nh * hd, H], per head [q | k | v]
    pub(crate) gate: Lin<'a>, // [nh * hd, H]
    pub(crate) out: Lin<'a>,  // [H, nh * hd]
    pub(crate) norm: Vec<f32>,
    /// `exp(-slope)` per head.
    pub(crate) decay: Vec<f32>,
}

/// What a recording forward keeps of a lightning layer to rewind into its span: the state
/// before it and every token's keys and values.
#[derive(Clone, Default)]
pub(crate) struct LightningRec {
    pub(crate) s0: Vec<f32>,
    pub(crate) kv: Vec<f32>, // [t, nh, 2 * hd]: k then v
    pub(crate) decay: Vec<f32>,
    pub(crate) hd: usize,
}

impl LightningRec {
    /// Replays the first `n` tokens of the span from the state before it into `s`.
    pub(crate) fn replay(&self, s: &mut [f32], n: usize) {
        s.copy_from_slice(&self.s0);
        let (nh, hd) = (self.decay.len(), self.hd);
        for ti in 0..n {
            for h in 0..nh {
                let row = &self.kv[(ti * nh + h) * 2 * hd..(ti * nh + h + 1) * 2 * hd];
                step(&mut s[h * hd * hd..(h + 1) * hd * hd], &row[..hd], &row[hd..], self.decay[h]);
            }
        }
    }
}

/// `S = d * S + k^T v` for one head.
#[inline]
fn step(s: &mut [f32], k: &[f32], v: &[f32], d: f32) {
    let hd = k.len();
    for (r, &kr) in k.iter().enumerate() {
        let row = &mut s[r * hd..(r + 1) * hd];
        for (sv, &vv) in row.iter_mut().zip(v) {
            *sv = d * *sv + kr * vv;
        }
    }
}

/// The slopes of MiniMax-01's lightning heads (ALiBi's, as its code builds them): for a power
/// of two `n`, `2^(-8/n)` to the powers 1..=n; otherwise those of the nearest lower power of two
/// then every other one of the next.
pub(crate) fn slopes(n: usize) -> Vec<f64> {
    fn pow2(n: usize) -> Vec<f64> {
        let start = 2f64.powf(-(2f64.powf(-((n as f64).log2() - 3.0))));
        (0..n).map(|i| start * start.powi(i as i32)).collect()
    }
    if (n as f64).log2().fract() == 0.0 {
        return pow2(n);
    }
    let c = 1usize << (n as f64).log2().floor() as u32;
    let mut v = pow2(c);
    v.extend(slopes(2 * c).into_iter().step_by(2).take(n - c));
    v
}

impl<'a> Lightning<'a> {
    /// The layer under `ap` (`...self_attn.`) of layer `l` of `n_layers`.
    pub(crate) fn load(ld: &Loader<'a, '_>, ap: &str, l: usize, n_layers: usize) -> Result<Self> {
        let c = ld.cfg;
        let (h, nh, hd) = (c.hidden_size, c.num_attention_heads, c.head_dim);
        // Shallower layers decay faster: slope * (1 - l / (L - 1) + 1e-5).
        let f = 1.0 - l as f64 / (n_layers.max(2) - 1) as f64 + 1e-5;
        Ok(Self {
            qkv: ld.lin(&format!("{ap}qkv_proj.weight"), 3 * nh * hd, h)?,
            gate: ld.lin(&format!("{ap}output_gate.weight"), nh * hd, h)?,
            out: ld.lin(&format!("{ap}out_proj.weight"), h, nh * hd)?,
            norm: ld.getf(&format!("{ap}norm.weight"), &[nh * hd])?,
            decay: slopes(nh).iter().map(|s| (-s * f).exp() as f32).collect(),
        })
    }

    pub(crate) fn lins(&self) -> Vec<&Lin<'a>> {
        vec![&self.qkv, &self.gate, &self.out]
    }

    /// `t` tokens of `hn` through the layer into `y`, updating the state `s` (`[nh, hd, hd]`);
    /// `rec`: keeps what a rewind into these tokens needs.
    pub(crate) fn forward(&self, s: &mut [f32], rec: Option<&mut LightningRec>, hn: &[f32], t: usize, h: usize, y: &mut [f32]) {
        let (nh, hd) = (self.decay.len(), self.norm.len() / self.decay.len());
        let w = nh * hd;
        let mut qkv = vec![0f32; t * 3 * w];
        let mut g = vec![0f32; t * w];
        linear_multi(&mut [(&self.qkv, 3 * w, &mut qkv[..]), (&self.gate, w, &mut g[..])], h, hn, t);
        for v in qkv.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
        if let Some(r) = rec {
            r.s0.clear();
            r.s0.extend_from_slice(s);
            r.kv.clear();
            for ti in 0..t {
                for hh in 0..nh {
                    let b = (ti * nh + hh) * 3 * hd;
                    r.kv.extend_from_slice(&qkv[b + hd..b + 3 * hd]);
                }
            }
            r.decay.clone_from(&self.decay);
            r.hd = hd;
        }
        let mut o = vec![0f32; t * w];
        {
            let _timer = crate::prof::start(crate::prof::Cat::Attention);
            let (sp, op) = (SendF(s.as_mut_ptr()), SendF(o.as_mut_ptr()));
            pool::run(nh, &|hh| {
                // SAFETY: head `hh` owns its state block and its columns of `o`.
                let st = unsafe { std::slice::from_raw_parts_mut(sp.get().add(hh * hd * hd), hd * hd) };
                for ti in 0..t {
                    let b = (ti * nh + hh) * 3 * hd;
                    let (q, k, v) = (&qkv[b..b + hd], &qkv[b + hd..b + 2 * hd], &qkv[b + 2 * hd..b + 3 * hd]);
                    step(st, k, v, self.decay[hh]);
                    let out = unsafe { std::slice::from_raw_parts_mut(op.get().add(ti * w + hh * hd), hd) };
                    out.fill(0.0);
                    for (r, &qr) in q.iter().enumerate() {
                        for (ov, &sv) in out.iter_mut().zip(&st[r * hd..(r + 1) * hd]) {
                            *ov += qr * sv;
                        }
                    }
                }
            });
        }
        // The heads' outputs normed together (the reference's default eps), then gated.
        for_chunks(&mut o, w, t > 1, |ti, r| {
            rmsnorm_f32w_inplace(r, &self.norm, 1e-6);
            for (v, &gv) in r.iter_mut().zip(&g[ti * w..(ti + 1) * w]) {
                *v *= 1.0 / (1.0 + (-gv).exp());
            }
        });
        crate::model::linear(&self.out, h, w, &o, t, y);
    }
}

#[derive(Clone, Copy)]
struct SendF(*mut f32);
unsafe impl Send for SendF {}
unsafe impl Sync for SendF {}
impl SendF {
    fn get(self) -> *mut f32 {
        self.0
    }
}
