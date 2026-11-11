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
