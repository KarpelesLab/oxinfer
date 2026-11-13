//! DeepSeek V4.1 (`deepseek_v41`: DeepSeek-V4.1-Flash and -1017, 40 layers of 5120) on the
//! DeepSeek V4 model of the parent module, as DeepSeek's own inference code (`inference/model.py`
//! and `engram.py` of the checkpoint). What changes from V4:
//!
//! - Hyper-connections in one pass: every sublayer computes its `pre` / `post` / `comb` weights
//!   from the copies entering it, but collapses them with the `pre` of the sublayer before it (the
//!   first attention with a one-hot on copy 0), and the final norm takes the copies collapsed with
//!   the last MoE's `pre`: there is no `hc_head`.
//! - Attention: no norm on the query heads. Compressed keys are shared: a few "key source" layers
//!   (`kv_source_layer_ids`) compress their own input (rate 1: a projection and an RMS norm per
//!   token; above 1: a softmax over learned gates pools each group of `rate` tokens, without
//!   position bias or overlap) and every layer after one attends to its keys. The lightning
//!   indexers are shared as well: an "index source" layer (`index_source_layer_ids`) scores the
//!   compressed keys (keys `k_norm(wk(latent))` made by the key source, queries from the layer's
//!   own `wq_b`) and the layers after it reuse its choice. With `candidate_source_layer_id`, that
//!   index layer also keeps the `candidate_topk_blocks` best blocks of `candidate_block_size`
//!   compressed positions per query (a block scores as its best position; the newest block is
//!   always kept), and the index layers after it choose inside those blocks only.
//! - Engram: before a few layers (`engram_layer_ids`), each token's 2-grams to
//!   `engram_max_ngram_size`-grams (over compressed token ids, a table that folds case, accents
//!   and spacing variants) are hashed by `engram_n_heads` heads each into a table; the rows,
//!   concatenated and projected (`wkv`), give one key per residual copy and a value, and each copy
//!   gets the value times `sigmoid(sign(d) sqrt(|d|))`, `d` the normalized product of copy and key.
//! - MoE: no hash-routed layers; the tokens of an image span are selected with their own bias
//!   (`gate.bias_vl`).
//!
//! As for V4, the reference's quantization simulations (FP8 of the window keys and of the
//! attention output, FP4 of the compressed keys and of the indexer) are left out.
//!
//! Checkpoints: DeepSeek's naming (float tensors; the engram hash inputs, which the reference
//! derives from the tokenizer and a NumPy generator, read from `engram.token_map` /
//! `engram.multipliers` when the config does not give them) and GGUF files (`deepseek41`), in
//! llama.cpp's naming (the tables of all engram layers in one `per_layer_token_embd`, the hash
//! inputs in the header) or with the reference's buffers as tensors (`engram_token_map`, ...).

use super::*;
use crate::gguf::Value;

/// A key source's compressor.
pub(super) struct Comp41<'a> {
    wkv: Lin<'a>,           // [hd, H]
    wgate: Option<Lin<'a>>, // [hd, H], rates above 1
    norm: Vec<f32>,         // [hd]
}

impl Comp41<'_> {
    /// Width of the rows kept while a group pools (none at rate 1).
    pub(super) fn width(&self, hd: usize) -> usize {
        if self.wgate.is_some() { hd } else { 0 }
    }
}

/// An index source's indexer; `wk` / `k_norm` on the layers that are key sources too.
pub(super) struct Idx41<'a> {
    wq: Lin<'a>,   // [heads * dim, q_lora]
    proj: Lin<'a>, // [heads, H]
    wk: Option<Lin<'a>>,
    k_norm: Vec<f32>,
}

/// An engram layer: its table (`base`: the first row of this layer in a table shared by all
/// layers), the projection of the looked-up rows and the gate's per-copy weights (`q_weight *
/// k_weight`).
pub(super) struct Engram<'a> {
    table: Table<'a>,
    base: usize,
    wkv: Lin<'a>,     // [(hc + 1) * H, cols * head_dim]
    weight: Vec<f32>, // [hc, H]
    /// Index among the engram layers.
    li: usize,
}

pub(super) struct Layer41<'a> {
    comp: Option<Comp41<'a>>,
    idx: Option<Idx41<'a>>,
    engram: Option<Engram<'a>>,
    /// The layer whose compressed keys this layer attends to (its choice of them comes from the
    /// last index source before it).
    kv_src: Option<usize>,
}

impl<'a> Layer41<'a> {
    pub(super) fn comp(&self) -> Option<&Comp41<'a>> {
        self.comp.as_ref()
    }
    pub(super) fn lins(&self) -> Vec<&Lin<'a>> {
        let mut v = Vec::new();
        if let Some(cp) = &self.comp {
            v.push(&cp.wkv);
            v.extend(cp.wgate.as_ref());
        }
        if let Some(ix) = &self.idx {
            v.extend([&ix.wq, &ix.proj]);
            v.extend(ix.wk.as_ref());
        }
        if let Some(e) = &self.engram {
            v.push(&e.wkv);
        }
        v
    }
}

/// The engram hash: per engram layer the multiplier of each look-back, and per (n-gram size,
/// head) the prime bucket count and its offset in the layer's table.
pub(super) struct Hash {
    token_map: Vec<i32>,
    pad: i64,
    mult: Vec<Vec<i64>>,
    primes: Vec<Vec<i64>>,
    offsets: Vec<Vec<i64>>,
    ngram: usize,
    heads: usize,
}

fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// An integer tensor (I32 / I64, also as GGUF types 26 / 27) as i64.
fn ints(files: &ModelFiles, name: &str) -> Result<Vec<i64>> {
    let info = files.info(name).with_context(|| format!("missing tensor {name}"))?;
    let b = files.bytes(name)?;
    let dtype = match info.ggml {
        Some(26) => "I32",
        Some(27) => "I64",
        _ => info.dtype.as_str(),
    };
    Ok(match dtype {
        "I32" | "U32" => b.as_chunks::<4>().0.iter().map(|x| i32::from_le_bytes(*x) as i64).collect(),
        "I64" | "U64" => b.as_chunks::<8>().0.iter().map(|x| i64::from_le_bytes(*x)).collect(),
        d => bail!("tensor {name}: dtype {d}, expected integers"),
    })
}

impl Hash {
    pub(super) fn load(ld: &Loader) -> Result<Option<Self>> {
        let c = ld.cfg;
        let nl = c.engram_layer_ids.len();
        if nl == 0 {
            return Ok(None);
        }
        let (ngram, heads) = (c.engram_max_ngram_size, c.engram_n_heads);
        let cols = (ngram - 1) * heads;
        let table = |given: &Option<Vec<Vec<i64>>>, name: &str, w: usize| -> Result<Option<Vec<Vec<i64>>>> {
            if let Some(v) = given {
                return Ok(Some(v.clone()));
            }
            if !ld.files.has(name) {
                return Ok(None);
            }
            let v = ints(ld.files, name)?;
            if v.len() != nl * w {
                bail!("tensor {name}: {} values, expected {nl} x {w}", v.len());
            }
            Ok(Some(v.chunks(w).map(<[i64]>::to_vec).collect()))
        };
        let mult = table(&c.engram_multipliers, "engram.multipliers", ngram)?
            .context("DeepSeek V4.1: the engram hash multipliers are neither in the config nor in the checkpoint (engram.multipliers)")?;
        let primes = match table(&c.engram_primes, "engram.primes", cols)? {
            Some(p) => p,
            // As the reference: per layer and n-gram size, the next unused primes above
            // `engram_vocab_size - 1`.
            None => {
                let mut seen = std::collections::BTreeSet::new();
                (0..nl)
                    .map(|_| {
                        let mut row = Vec::with_capacity(cols);
                        for _ in 0..ngram - 1 {
                            let mut cur = c.engram_vocab_size as i64 - 1;
                            for _ in 0..heads {
                                cur += 1;
                                while !is_prime(cur) || seen.contains(&cur) {
                                    cur += 1;
                                }
                                seen.insert(cur);
                                row.push(cur);
                            }
                        }
                        row
                    })
                    .collect()
            }
        };
        let offsets = match table(&c.engram_offsets, "engram.offsets", cols)? {
            Some(o) => o,
            None => primes
                .iter()
                .map(|p| {
                    let mut acc = 0;
                    p.iter()
                        .map(|&x| {
                            acc += x;
                            acc - x
                        })
                        .collect()
                })
                .collect(),
        };
        if primes.iter().flatten().any(|&p| p <= 0) {
            bail!("DeepSeek V4.1: bad engram bucket counts");
        }
        let token_map: Vec<i32> = match &c.engram_token_map {
            Some(m) => m.clone(),
            None => ints(ld.files, "engram.token_map")
                .context("DeepSeek V4.1: the engram token map is neither in the config nor in the checkpoint")?
                .into_iter()
                .map(|v| v as i32)
                .collect(),
        };
        if token_map.len() != c.vocab_size {
            bail!(
                "DeepSeek V4.1: engram token map of {} tokens for a vocabulary of {}",
                token_map.len(),
                c.vocab_size
            );
        }
        let pad = *token_map
            .get(c.engram_pad_token_id as usize)
            .context("DeepSeek V4.1: engram pad token out of range")? as i64;
        Ok(Some(Self {
            token_map,
            pad,
            mult,
            primes,
            offsets,
            ngram,
            heads,
        }))
    }

    /// Rows of engram layer `li`'s table.
    fn rows(&self, li: usize) -> usize {
        self.primes[li].iter().sum::<i64>() as usize
    }

    /// The table rows (`(ngram - 1) * heads`, relative to the layer's table) of position `p` of
    /// `hist` (the engram ids of all positions so far): the hash of each n-gram ending there.
    /// Look-back stops at the start of the sequence and at an image position.
    fn ids(&self, li: usize, hist: &[i32], p: usize, out: &mut [usize]) {
        let (mut blocked, mut rolling) = (false, 0i64);
        for s in 0..self.ngram {
            blocked |= p < s || hist[p - s.min(p)] < 0;
            let tok = if blocked { self.pad } else { hist[p - s] as i64 };
            let prod = tok.wrapping_mul(self.mult[li][s]);
            if s == 0 {
                rolling = prod;
                continue;
            }
            rolling ^= prod;
            for hh in 0..self.heads {
                let col = (s - 1) * self.heads + hh;
                out[col] = (rolling.rem_euclid(self.primes[li][col]) + self.offsets[li][col]) as usize;
            }
        }
    }
}

/// The V4.1 parts of layer `l`.
pub(super) fn load_layer<'a>(ld: &Loader<'a, '_>, l: usize, hash: Option<&Hash>) -> Result<Layer41<'a>> {
    let c = ld.cfg;
    let (h, hd, ql, n) = (c.hidden_size, c.head_dim, c.q_lora_rank, c.hc_mult);
    let ratio = c.ratio(l);
    let ap = format!("layers.{l}.attn.");
    let is_kv = ratio > 0 && c.kv_source_layer_ids.contains(&l);
    let comp = if is_kv {
        let p = format!("{ap}compressor.");
        Some(Comp41 {
            wkv: ld.lin(&format!("{p}wkv.weight"), hd, h)?,
            wgate: if ratio > 1 {
                Some(ld.lin(&format!("{p}wgate.weight"), hd, h)?)
            } else {
                None
            },
            norm: ld.getf(&format!("{p}norm.weight"), &[hd])?,
        })
    } else {
        None
    };
    let idx = if ratio > 0 && c.index_source_layer_ids.contains(&l) {
        let (ni, di) = (c.index_n_heads, c.index_head_dim);
        let p = format!("{ap}indexer.");
        Some(Idx41 {
            wq: ld.lin(&format!("{p}wq_b.weight"), ni * di, ql)?,
            proj: ld.lin(&format!("{p}weights_proj.weight"), ni, h)?,
            wk: if is_kv {
                Some(ld.lin(&format!("{p}wk.weight"), di, hd)?)
            } else {
                None
            },
            k_norm: if is_kv {
                ld.getf(&format!("{p}k_norm.weight"), &[di])?
            } else {
                Vec::new()
            },
        })
    } else {
        None
    };
    if is_kv && idx.is_none() {
        bail!("deepseek_v41: key source layer {l} is not an index source (nothing would make its index keys)");
    }
    let engram = match (c.engram_layer_ids.iter().position(|&e| e == l), hash) {
        (Some(li), Some(hash)) => {
            let ed = c.engram_head_dim;
            let cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads;
            let rows = |i: usize| c.engram_num_embeddings.get(i).copied().unwrap_or_else(|| hash.rows(i));
            if rows(li) < hash.rows(li) {
                bail!("deepseek_v41: engram table of {} rows for {} buckets", rows(li), hash.rows(li));
            }
            let p = format!("layers.{l}.engram.");
            let own = format!("{p}embed.weight");
            let missing = "DeepSeek V4.1: the engram tables are missing (a GGUF conversion keeps them in a file of their own: put it beside the model)";
            let (table, base) = if ld.files.has(&own) {
                (Table::load(ld.files, &own, rows(li), ed, ld.quant)?, 0)
            } else {
                // One table for all the engram layers.
                let total: usize = (0..c.engram_layer_ids.len()).map(rows).sum();
                let t = Table::load(ld.files, "engram.embed.weight", total, ed, ld.quant).context(missing)?;
                (t, (0..li).map(rows).sum())
            };
            let weight = if ld.files.has(&format!("{p}qk_weight")) {
                ld.getf(&format!("{p}qk_weight"), &[n, h])?
            } else {
                let q = ld.getf(&format!("{p}q_weight"), &[n, h])?;
                let k = ld.getf(&format!("{p}k_weight"), &[n, h])?;
                q.iter().zip(&k).map(|(a, b)| a * b).collect()
            };
            Some(Engram {
                table,
                base,
                wkv: ld.lin(&format!("{p}wkv.weight"), (n + 1) * h, cols * ed)?,
                weight,
                li,
            })
        }
        _ => None,
    };
    Ok(Layer41 {
        comp,
        idx,
        engram,
        kv_src: if ratio > 0 { c.kv_source(l) } else { None },
    })
}

/// The mixing weights of every token's copies `x` (`[t, hc * H]`): `pre` `[t, hc]`, `post`
/// `[t, hc]`, `comb` `[t, hc * hc]`.
fn mixes(c: &DeepSeek4Config, w: &Hc, x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (n, h) = (c.hc_mult, c.hidden_size);
    let mut pre = vec![0f32; t * n];
    let mut post = vec![0f32; t * n];
    let mut comb = vec![0f32; t * n * n];
    let (pp, cp) = (SendPtr(post.as_mut_ptr()), SendPtr(comb.as_mut_ptr()));
    for_chunks(&mut pre, n, t > 1, |ti, pr| {
        // SAFETY: each token writes its own rows.
        let (po, co) = unsafe {
            (
                std::slice::from_raw_parts_mut(pp.get().add(ti * n), n),
                std::slice::from_raw_parts_mut(cp.get().add(ti * n * n), n * n),
            )
        };
        hc_mix(c, w, &x[ti * n * h..(ti + 1) * n * h], pr, po, co);
    });
    (pre, post, comb)
}

/// Every token's copies collapsed with `pre` (`[t, hc]`) and RMS-normed with `norm`.
fn collapse(c: &DeepSeek4Config, pre: &[f32], x: &[f32], norm: &[f32], t: usize) -> Vec<f32> {
    let (n, h) = (c.hc_mult, c.hidden_size);
    let mut y = vec![0f32; t * h];
    for_chunks(&mut y, h, t > 1, |ti, yr| {
        hc_collapse(&pre[ti * n..(ti + 1) * n], &x[ti * n * h..(ti + 1) * n * h], yr);
        rmsnorm_f32w_inplace(yr, norm, c.rms_norm_eps as f32);
    });
    y
}

/// What the layers hand down the stack within a pass: the last index source's choice per query
/// (None: every visible compressed key) and the candidate source's blocks per query (None: all).
#[derive(Default)]
struct Shared {
    sel: Vec<Option<Vec<u32>>>,
    cand: Vec<Option<Vec<u32>>>,
}

/// The blocks (sorted) kept for a query with scores `s` over its visible compressed positions:
/// the `topk` blocks of `bs` positions with the best position, the newest block always among
/// them. None when every block is kept.
fn candidate_blocks(s: &[f32], topk: usize, bs: usize) -> Option<Vec<u32>> {
    let nb = s.len().div_ceil(bs);
    if nb <= topk {
        return None;
    }
    let mut score: Vec<f32> = s.chunks(bs).map(|b| b.iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
    score[nb - 1] = f32::INFINITY;
    Some(top_positions(&score, topk))
}

/// Adds engram layer `e`'s lookups to the copies `x` (`[t, hc * H]`) of the tokens at `start..`.
fn engram(m: &DeepSeek4Model, e: &Engram, hist: &[i32], start: usize, t: usize, x: &mut [f32]) {
    let c = &m.cfg;
    let hash = m.hash.as_ref().expect("engram layer without a hash");
    let (h, n, ed) = (c.hidden_size, c.hc_mult, c.engram_head_dim);
    let cols = (hash.ngram - 1) * hash.heads;
    let mut ids = vec![0usize; t * cols];
    for (ti, row) in ids.chunks_mut(cols).enumerate() {
        hash.ids(e.li, hist, start + ti, row);
        row.iter_mut().for_each(|r| *r += e.base);
    }
    if let Some(s) = &m.stream {
        let rows: Vec<(usize, usize)> = ids.iter().flat_map(|&r| e.table.row_ranges(r, ed)).collect();
        s.need_ranges(&rows);
    }
    let mut emb = vec![0f32; t * cols * ed];
    for_chunks(&mut emb, ed, t > 1, |i, row| e.table.row(ids[i], row));
    let mut kv = vec![0f32; t * (n + 1) * h];
    linear(&e.wkv, (n + 1) * h, cols * ed, &emb, t, &mut kv);
    let eps = c.rms_norm_eps as f32;
    for_chunks(x, n * h, t > 1, |ti, xr| {
        // An image position takes no lookup.
        if hist[start + ti] < 0 {
            return;
        }
        let kr = &kv[ti * (n + 1) * h..(ti + 1) * (n + 1) * h];
        let value = &kr[n * h..];
        for k in 0..n {
            let (xc, key, w) = (&mut xr[k * h..(k + 1) * h], &kr[k * h..(k + 1) * h], &e.weight[k * h..(k + 1) * h]);
            let mut d = 0f32;
            for i in 0..h {
                d += xc[i] * w[i] * key[i];
            }
            let rstd = 1.0 / ((dot(xc, xc) / h as f32 + eps).sqrt() * (dot(key, key) / h as f32 + eps).sqrt());
            let d = d * rstd * (h as f32).powf(-0.5);
            let gate = sigmoid(d.abs().max(1e-6).sqrt().copysign(d));
            axpy(xc, gate, value);
        }
    });
}

/// Runs `tokens` from `sess.pos` (`mm`: input embeddings replaced, the positions of an image
/// span); see [`DeepSeek4Model::forward_capture_mm`].
pub(super) fn forward(
    m: &DeepSeek4Model,
    sess: &mut DeepSeek4Session,
    tokens: &[u32],
    mm: &[(usize, &[f32])],
    layers: &[usize],
    feats: &mut Vec<f32>,
) -> Vec<f32> {
    let t = tokens.len();
    assert!(t > 0);
    let c = &m.cfg;
    let (h, n) = (c.hidden_size, c.hc_mult);
    let start = sess.pos;
    let timer = prof::start(Cat::Embed);
    let mut image = vec![false; t];
    for &(i, row) in mm {
        assert!(i < t && row.len() == h, "bad embedding override");
        image[i] = true;
    }
    let any_image = !mm.is_empty();
    if let Some(s) = &m.stream {
        let rows: Vec<(usize, usize)> = tokens
            .iter()
            .zip(&image)
            .filter(|x| !x.1)
            .flat_map(|(&tk, _)| m.embed.row_ranges(tk as usize, h))
            .collect();
        s.need_ranges(&rows);
    }
    let mut x = vec![0f32; t * n * h];
    for_chunks(&mut x, n * h, t > 1, |ti, row| {
        match mm.iter().find(|o| o.0 == ti) {
            Some(o) => row[..h].copy_from_slice(o.1),
            None => m.embed.row(tokens[ti] as usize, &mut row[..h]),
        }
        for k in 1..n {
            row.copy_within(..h, k * h);
        }
    });
    if let Some(hash) = &m.hash {
        debug_assert_eq!(sess.eng.len(), start);
        sess.eng.extend(
            tokens
                .iter()
                .zip(&image)
                .map(|(&tk, &im)| if im { -1 } else { hash.token_map[tk as usize] }),
        );
    }
    let positions: Vec<usize> = (start..start + t).collect();
    let ropes = [rope_tables(&positions, &m.inv_main), rope_tables(&positions, &m.inv_comp)];
    drop(timer);
    let nf = layers.len();
    feats.clear();
    feats.resize(t * nf * h, 0.0);
    let iv = sess.iv.active().then(|| sess.iv.clone());
    let point = |p: usize, x: &mut [f32]| {
        if let Some(iv) = &iv {
            iv.point_copies(p, x, h, n, start);
        }
    };
    point(0, &mut x);
    let mut capture = |l: usize, x: &[f32]| {
        for (k, _) in layers.iter().enumerate().filter(|e| *e.1 == l) {
            for ti in 0..t {
                let o = &mut feats[(ti * nf + k) * h..(ti * nf + k + 1) * h];
                hc_collapse(&vec![1.0 / n as f32; n], &x[ti * n * h..(ti + 1) * n * h], o);
            }
        }
    };
    // The first attention reads copy 0.
    let mut pre = vec![0f32; t * n];
    pre.chunks_mut(n).for_each(|p| p[0] = 1.0);
    let mut shared = Shared::default();
    for (l, layer) in m.layers.iter().enumerate() {
        let lx = &m.x41[l];
        let _unit = crate::stream::enter(&m.stream, l);
        if let Some(e) = &lx.engram {
            engram(m, e, &sess.eng, start, t, &mut x);
        }
        // (A DSpark draft reads the stream entering its target layers, after the engram.)
        capture(l, &x);
        let (apre, post, comb) = mixes(c, &layer.hc_attn, &x, t);
        let xa = collapse(c, &pre, &x, &layer.attn_norm, t);
        let rope = &ropes[usize::from(layer.ratio > 0)];
        let mut o = vec![0f32; t * h];
        attn(m, l, &xa, t, start, &mut sess.layers, (&rope.0, &rope.1), &mut shared, &mut o);
        hc_post(c, &mut x, &o, &post, &comb, t);
        point(2 * l + 1, &mut x);
        let (fpre, post, comb) = mixes(c, &layer.hc_ffn, &x, t);
        let xf = collapse(c, &apre, &x, &layer.ffn_norm, t);
        o.fill(0.0);
        let need = |ids: &[usize]| {
            if let Some(s) = &m.stream {
                s.need_parts(l, ids)
            }
        };
        let need: NeedExperts = m.stream.as_ref().map(|_| &need as &dyn Fn(&[usize]));
        let hook = iv.as_ref().and_then(|s| s.hook(l, start));
        let im: &[bool] = if any_image { &image } else { &[] };
        moe_forward(c, &layer.moe, &xf, tokens, im, t, &mut o, need, hook.as_ref());
        hc_post(c, &mut x, &o, &post, &comb, t);
        point(2 * l + 2, &mut x);
        pre = fpre;
    }
    capture(m.layers.len(), &x);
    let out = collapse(c, &pre, &x, &m.norm, t);
    sess.pos += t;
    sess.trim();
    out
}

/// Layer `l`'s attention over `xa` (`[t, H]`, the normed block input) into `out`; `states`: every
/// layer's caches.
#[allow(clippy::too_many_arguments)]
fn attn(
    m: &DeepSeek4Model,
    l: usize,
    xa: &[f32],
    t: usize,
    start: usize,
    states: &mut [LayerState],
    rope: (&[f32], &[f32]),
    shared: &mut Shared,
    out: &mut [f32],
) {
    let c = &m.cfg;
    let (layer, lx) = (&m.layers[l], &m.x41[l]);
    let a = &layer.attn;
    let (h, nh, hd, ql) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank);
    let (ni, di, topk) = (c.index_n_heads, c.index_head_dim, c.index_topk);
    let eps = c.rms_norm_eps as f32;
    let (cos, sin) = rope;
    let r = layer.ratio;
    // Projections of the block input.
    let mut qa = vec![0f32; t * ql];
    let mut kv = vec![0f32; t * hd];
    let cw = if lx.comp.is_some() { hd } else { 0 };
    let gw = lx.comp.as_ref().map_or(0, |cp| cp.width(hd));
    let (mut ckv, mut cg) = (vec![0f32; t * cw], vec![0f32; t * gw]);
    let iw_n = if lx.idx.is_some() { ni } else { 0 };
    let mut iw = vec![0f32; t * iw_n];
    {
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.wq_a, ql, &mut qa[..]), (&a.wkv, hd, &mut kv[..])];
        if let Some(cp) = &lx.comp {
            jobs.push((&cp.wkv, hd, &mut ckv[..]));
            if let Some(g) = &cp.wgate {
                jobs.push((g, hd, &mut cg[..]));
            }
        }
        if let Some(ix) = &lx.idx {
            jobs.push((&ix.proj, ni, &mut iw[..]));
        }
        linear_multi(&mut jobs, h, xa, t);
    }
    rmsnorm_f32w_rows_inplace(&mut qa, ql, &a.q_norm, eps);
    let mut q = vec![0f32; t * nh * hd];
    linear(&a.wq_b, nh * hd, ql, &qa, t, &mut q);
    let half = cos.len() / t;
    for_chunks(&mut q, hd, t * nh > 64, |i, qh| {
        let ti = i / nh;
        rope_tail(qh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], false);
    });
    norm_rope_rows(&mut kv, hd, Some(&a.kv_norm), eps, cos, sin);
    debug_assert_eq!(states[l].raw_start + states[l].raw.len() / hd, start);
    states[l].raw.extend_from_slice(&kv);
    // A key source: the latents of the groups this pass completes, then the indexers' keys made
    // from them (before the rotation) and the compressed keys (after it).
    if let Some(cp) = &lx.comp {
        let st = &mut states[l];
        let mut lat: Vec<f32> = Vec::new();
        let mut gpos: Vec<usize> = Vec::new();
        if cp.wgate.is_none() {
            lat = ckv;
            gpos = (start..start + t).collect();
        } else {
            let cs = st.comp.as_mut().unwrap();
            debug_assert_eq!(cs.rows_start + cs.rows(), start);
            cs.kv.extend_from_slice(&ckv);
            cs.sc.extend_from_slice(&cg);
            for p in start..start + t {
                if !(p + 1).is_multiple_of(r) {
                    continue;
                }
                let w0 = p + 1 - r;
                let o = (w0 - cs.rows_start) * hd;
                let (rows, sc) = (&cs.kv[o..o + r * hd], &cs.sc[o..o + r * hd]);
                lat.extend((0..hd).map(|ch| {
                    let mx = (0..r).map(|j| sc[j * hd + ch]).fold(f32::NEG_INFINITY, f32::max);
                    let (mut tot, mut acc) = (0f32, 0f32);
                    for j in 0..r {
                        let e = (sc[j * hd + ch] - mx).exp();
                        tot += e;
                        acc += e * rows[j * hd + ch];
                    }
                    acc / tot
                }));
                gpos.push(w0);
            }
        }
        let g = gpos.len();
        if g > 0 {
            rmsnorm_f32w_rows_inplace(&mut lat, hd, &cp.norm, eps);
            // A latent stands for the first token of its group.
            let (gc, gs) = rope_tables(&gpos, &m.inv_comp);
            let gh = m.inv_comp.len();
            if let Some(ix) = &lx.idx
                && let Some(wk) = &ix.wk
            {
                let mut ik = vec![0f32; g * di];
                linear(wk, di, hd, &lat, g, &mut ik);
                norm_rope_rows(&mut ik, di, Some(&ix.k_norm), eps, &gc, &gs);
                let is = st.idx.as_mut().unwrap();
                is.keys.extend_from_slice(&ik);
                is.n += g;
            }
            for_chunks(&mut lat, hd, g > 64, |i, row| {
                rope_tail(row, &gc[i * gh..(i + 1) * gh], &gs[i * gh..(i + 1) * gh], false);
            });
            let cs = st.comp.as_mut().unwrap();
            debug_assert_eq!(cs.n, gpos[0] / r);
            cs.keys.extend_from_slice(&lat);
            cs.n += g;
        }
    }
    // An index source chooses for the layers down to the next one. The indexer only matters for
    // a query that sees more than `index_topk` compressed keys (before that every key is taken).
    if let Some(ix) = &lx.idx {
        let src = lx.kv_src.unwrap();
        let choose = (start + t) / r > topk;
        let cand_src = c.candidate_source();
        let (makes, uses) = (cand_src == Some(l), cand_src.is_some_and(|s| s < l));
        if !uses {
            shared.cand = vec![None; t];
        }
        shared.sel = vec![None; t];
        if choose {
            let is = states[src].idx.as_ref().unwrap();
            let mut iq = vec![0f32; t * ni * di];
            linear(&ix.wq, ni * di, ql, &qa, t, &mut iq);
            for_chunks(&mut iq, di, t * ni > 64, |i, qh| {
                let ti = i / ni;
                rope_tail(qh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], false);
            });
            let ws = (di as f32).powf(-0.5) * (ni as f32).powf(-0.5);
            iw.iter_mut().for_each(|w| *w *= ws);
            let (bs, tb) = (c.candidate_block_size, c.candidate_topk_blocks);
            let cand = &shared.cand;
            let res = select_with(c, &iq, &iw, is, t, start, &|i, s: &mut [f32]| {
                let made = if makes { candidate_blocks(s, tb, bs) } else { None };
                if uses && let Some(blocks) = &cand[i] {
                    for (j, v) in s.iter_mut().enumerate() {
                        if blocks.binary_search(&((j / bs) as u32)).is_err() {
                            *v = f32::NEG_INFINITY;
                        }
                    }
                }
                (top_positions(s, topk), made)
            });
            let mut cands = Vec::with_capacity(if makes { t } else { 0 });
            for (i, v) in res.into_iter().enumerate() {
                let (s, made) = v.unzip();
                shared.sel[i] = s;
                if makes {
                    cands.push(made.flatten());
                }
            }
            if makes {
                shared.cand = cands;
            }
        }
    }
    let mut att = vec![0f32; t * nh * hd];
    let none: Vec<Option<Vec<u32>>>;
    let (comp, sel) = match lx.kv_src {
        Some(src) => {
            debug_assert_eq!(shared.sel.len(), t);
            (states[src].comp.as_ref(), &shared.sel)
        }
        None => {
            none = vec![None; t];
            (None, &none)
        }
    };
    attend(c, &a.sink, &q, &states[l], comp, sel, r, t, start, &mut att);
    // Undo the rotation of the values' tails at each query's position.
    for_chunks(&mut att, hd, t * nh > 64, |i, oh| {
        let ti = i / nh;
        rope_tail(oh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half], true);
    });
    grouped_out(c, a, &att, t, out);
}

/// The DeepSeek-naming name and PyTorch-order shape of a `deepseek41` GGUF tensor (`ne`: ggml's
/// shape), in llama.cpp's naming or with the reference's engram buffers as tensors.
pub(crate) fn gguf_name(name: &str, ne: &[usize]) -> Option<(String, Vec<usize>)> {
    let shape: Vec<usize> = ne.iter().rev().copied().collect();
    let top = match name {
        // The tables of all the engram layers, one after the other.
        "per_layer_token_embd.weight" => Some("engram.embed.weight"),
        "engram_token_map.weight" => Some("engram.token_map"),
        "engram_primes.weight" => Some("engram.primes"),
        "engram_offsets.weight" => Some("engram.offsets"),
        "engram_multipliers.weight" => Some("engram.multipliers"),
        _ => None,
    };
    if let Some(t) = top {
        return Some((t.to_string(), shape));
    }
    if let Some((l, rest)) = name.strip_prefix("blk.").and_then(|r| r.split_once('.'))
        && let Some((base, _)) = rest.rsplit_once('.')
    {
        let n = match base {
            "indexer.attn_k" => Some("attn.indexer.wk.weight"),
            "indexer.k_norm" => Some("attn.indexer.k_norm.weight"),
            "exp_probs_b_vl" => Some("ffn.gate.bias_vl"),
            "engram_kv" | "engram_wkv" => Some("engram.wkv.weight"),
            // The product of the gate's two weights, or each of them.
            "engram_gate" => Some("engram.qk_weight"),
            "engram_q" => Some("engram.q_weight"),
            "engram_k" => Some("engram.k_weight"),
            _ => None,
        };
        if let Some(n) = n {
            l.parse::<usize>().ok()?;
            return Some((format!("layers.{l}.{n}"), shape));
        }
    }
    super::gguf_name(name, ne)
}

/// The V4.1 keys of a `deepseek41` GGUF header added to `base` (the keys V4 shares, from
/// `gguf::Header::deepseek4_config`): llama.cpp's metadata names, or the config's own.
pub(crate) fn gguf_config(hd: &crate::gguf::Header, mut base: serde_json::Value) -> Result<serde_json::Value> {
    let arch = hd.arch().to_string();
    let get = |keys: &[&str]| keys.iter().find_map(|k| hd.get(&format!("{arch}.{k}")));
    let int = |v: &Value| match v {
        Value::U(u) => Some(*u as i64),
        Value::I(i) => Some(*i),
        _ => None,
    };
    let ints = |keys: &[&str]| -> Option<Vec<i64>> { get(keys)?.as_array()?.iter().map(int).collect() };
    let one = |keys: &[&str]| get(keys).and_then(int);
    let o = base.as_object_mut().context("deepseek41 config")?;
    o.insert("model_type".into(), "deepseek_v41".into());
    let mut put = |k: &str, v: Option<serde_json::Value>| {
        if let Some(v) = v {
            o.insert(k.into(), v);
        }
    };
    put(
        "kv_source_layer_ids",
        ints(&["attention.kv_source_layers", "kv_source_layer_ids"]).map(Into::into),
    );
    put(
        "index_source_layer_ids",
        ints(&["attention.indexer.source_layers", "index_source_layer_ids"]).map(Into::into),
    );
    put(
        "candidate_source_layer_id",
        one(&["attention.indexer.candidate_source_layer", "attention.candidate_source_layer_id"]).map(Into::into),
    );
    put(
        "candidate_topk_blocks",
        one(&["attention.indexer.candidate_top_k_blocks", "attention.candidate_topk_blocks"]).map(Into::into),
    );
    put(
        "candidate_block_size",
        one(&["attention.indexer.candidate_block_size", "attention.candidate_block_size"]).map(Into::into),
    );
    let layers = ints(&["ple.layers", "engram.layer_ids"]).unwrap_or_default();
    let ngram = one(&["ple.ngram_size", "engram_max_ngram_size"]).unwrap_or(0) as usize;
    let heads = one(&["ple.heads_per_ngram", "engram_n_heads"]).unwrap_or(0) as usize;
    if !layers.is_empty() {
        if ngram < 2 || heads == 0 {
            bail!("GGUF deepseek41: engram layers without n-gram settings");
        }
        let (nl, cols) = (layers.len(), (ngram - 1) * heads);
        // A flat array as one row per engram layer.
        let table = |keys: &[&str], w: usize| -> Result<Option<Vec<Vec<i64>>>> {
            match ints(keys) {
                Some(v) if v.len() == nl * w => Ok(Some(v.chunks(w).map(<[i64]>::to_vec).collect())),
                Some(v) => bail!("GGUF deepseek41: {} has {} values, expected {nl} x {w}", keys[0], v.len()),
                None => Ok(None),
            }
        };
        put("engram_layer_ids", Some(layers.into()));
        put("engram_max_ngram_size", Some(ngram.into()));
        put("engram_n_heads", Some(heads.into()));
        put(
            "engram_head_dim",
            one(&["embedding_length_per_layer_input", "engram_head_dim"]).map(Into::into),
        );
        put(
            "engram_vocab_size",
            one(&["engram.vocab_size", "engram_vocab_size"]).map(Into::into),
        );
        put(
            "engram_pad_token_id",
            one(&["engram.pad_token_id", "engram_pad_token_id"]).map(Into::into),
        );
        put("engram_num_embeddings", ints(&["engram.num_embeddings"]).map(Into::into));
        put("engram_multipliers", table(&["ple.layer_multipliers"], ngram)?.map(Into::into));
        let primes = table(&["ple.head_vocab_sizes"], cols)?;
        // llama.cpp's offsets count from the start of the table of all layers.
        if let Some(mut offs) = table(&["ple.head_offsets"], cols)? {
            for row in &mut offs {
                let first = row[0];
                row.iter_mut().for_each(|v| *v -= first);
            }
            put("engram_offsets", Some(offs.into()));
        }
        if let (Some(p), None) = (&primes, ints(&["engram.num_embeddings"])) {
            let rows: Vec<i64> = p.iter().map(|r| r.iter().sum()).collect();
            put("engram_num_embeddings", Some(rows.into()));
        }
        put("engram_primes", primes.map(Into::into));
        put("engram_token_map", ints(&["engram.token_map"]).map(Into::into));
    }
    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let n = |name: &str, ne: &[usize]| gguf_name(name, ne);
        assert_eq!(
            n("per_layer_token_embd.weight", &[256, 1000]),
            Some(("engram.embed.weight".into(), vec![1000, 256]))
        );
        assert_eq!(
            n("blk.14.engram_kv.weight", &[6144, 25600]),
            Some(("layers.14.engram.wkv.weight".into(), vec![25600, 6144]))
        );
        assert_eq!(
            n("blk.1.engram_gate.weight", &[5120, 4]),
            Some(("layers.1.engram.qk_weight".into(), vec![4, 5120]))
        );
        assert_eq!(
            n("blk.2.indexer.attn_k.weight", &[512, 128]).map(|x| x.0).as_deref(),
            Some("layers.2.attn.indexer.wk.weight")
        );
        assert_eq!(
            n("blk.2.exp_probs_b_vl.bias", &[384]).map(|x| x.0).as_deref(),
            Some("layers.2.ffn.gate.bias_vl")
        );
        // The names V4 has too.
        assert_eq!(
            n("blk.20.attn_compressor_kv.weight", &[5120, 512]),
            Some(("layers.20.attn.compressor.wkv.weight".into(), vec![512, 5120]))
        );
        assert_eq!(
            n("blk.5.attn_output_a.weight", &[4096, 8192]),
            Some(("layers.5.attn.wo_a.weight".into(), vec![8192, 4096]))
        );
        assert_eq!(n("engram_primes.weight", &[24, 2]).map(|x| x.1), Some(vec![2, 24]));
    }

    /// The bucket counts are the primes the reference draws (the first ones match the header of
    /// the released conversions), and a position's n-grams stop at the sequence start.
    #[test]
    fn hash() {
        let primes = [16000057i64, 16000079, 16000081, 16000097];
        assert!(primes.iter().all(|&p| is_prime(p)));
        assert!((16000000..16000057).all(|p| !is_prime(p)));
        let hsh = Hash {
            token_map: vec![0, 1, 2, 3],
            pad: 2,
            mult: vec![vec![700, 1100, 1300]],
            primes: vec![vec![101, 103]],
            offsets: vec![vec![0, 101]],
            ngram: 3,
            heads: 1,
        };
        let hist = [3i32, 1, -1, 3, 0];
        let mut out = [0usize; 2];
        // The rows of a 2-gram hash and of a 3-gram hash.
        let rows = |two: i64, three: i64| [(two % 101) as usize, 101 + (three % 103) as usize];
        // Position 1: the 2-gram (1, 3), then the 3-gram padded past the start.
        hsh.ids(0, &hist, 1, &mut out);
        assert_eq!(out, rows(700 ^ 3300, 700 ^ 3300 ^ 2600));
        // Position 3: the image position before it blocks every look-back.
        hsh.ids(0, &hist, 3, &mut out);
        assert_eq!(out, rows(2100 ^ 2200, 2100 ^ 2200 ^ 2600));
        // Position 4: one real token back, then blocked.
        hsh.ids(0, &hist, 4, &mut out);
        assert_eq!(out, rows(3300, 3300 ^ 2600));
        assert_eq!(candidate_blocks(&[0.0; 8], 2, 4), None);
        assert_eq!(candidate_blocks(&[9.0, 0.0, 1.0, 5.0, 0.5, 0.2], 2, 2), Some(vec![0, 2]));
    }

    /// The config of the tiny model: key sources at rates 2 and 1, an index layer on another
    /// layer's keys, candidate blocks from layer 3 used by layer 4, engram lookups before two
    /// layers. `over`: keys replaced.
    fn tiny_config(over: serde_json::Value) -> DeepSeek4Config {
        let mut c = serde_json::json!({
            "model_type": "deepseek_v41",
            "text_config": {
                "hidden_size": 16, "num_hidden_layers": 6, "num_attention_heads": 2, "head_dim": 8,
                "qk_rope_head_dim": 4, "q_lora_rank": 8, "o_groups": 2, "o_lora_rank": 4, "vocab_size": 32,
                "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
                "routed_scaling_factor": 1.5, "swiglu_limit": 1.5, "sliding_window": 4, "rms_norm_eps": 1e-20,
                "compress_ratios": [0, 2, 2, 1, 1, 0], "kv_source_layer_ids": [1, 3],
                "index_source_layer_ids": [1, 2, 3, 4], "index_n_heads": 8, "index_head_dim": 8, "index_topk": 2,
                "candidate_source_layer_id": 3, "candidate_topk_blocks": 2, "candidate_block_size": 2,
                "engram_layer_ids": [1, 3], "engram_max_ngram_size": 3, "engram_vocab_size": 20,
                "engram_n_heads": 2, "engram_head_dim": 4, "engram_pad_token_id": 2,
                "engram_multipliers": [[7, 11, 13], [17, 19, 23]],
                "engram_token_map": (0..32).map(|i| i % 20).collect::<Vec<i32>>(),
            }
        });
        for (k, v) in over.as_object().unwrap() {
            c["text_config"][k] = v.clone();
        }
        DeepSeek4Config::from_json(c).unwrap()
    }

    /// Writes a tiny random checkpoint of [`tiny_config`] in DeepSeek's naming (float32).
    fn tiny_checkpoint(dir: &std::path::Path) {
        let c = tiny_config(serde_json::json!({}));
        let (h, nh, hd, ql) = (c.hidden_size, c.num_attention_heads, c.head_dim, c.q_lora_rank);
        let (g, r, hc) = (c.o_groups, c.o_lora_rank, c.hc_mult);
        let (ne, mi, v) = (c.n_routed_experts, c.moe_intermediate_size, c.vocab_size);
        let (ni, di, ed) = (c.index_n_heads, c.index_head_dim, c.engram_head_dim);
        let cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads;
        let mut seed = 0x5eed41u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        // `scale` times uniform noise, plus `bias`.
        let mut add = |name: String, shape: &[usize], scale: f32, bias: f32| {
            let n: usize = shape.iter().product();
            let bytes = (0..n).flat_map(|_| (bias + scale * rng()).to_le_bytes()).collect();
            tensors.push((name, shape.to_vec(), bytes));
        };
        let mat = |fan_in: usize, gain: f32| gain * (3.0 / fan_in as f32).sqrt();
        add("embed.weight".into(), &[v, h], 1.7, 0.0);
        add("head.weight".into(), &[v, h], mat(h, 4.0), 0.0);
        add("norm.weight".into(), &[h], 0.2, 1.0);
        // The primes above 19, two heads per n-gram size and layer: 23 + 29 + 31 + 37, then
        // 41 + 43 + 47 + 53.
        let rows = [120usize, 184];
        for l in 0..c.num_hidden_layers {
            let p = format!("layers.{l}.");
            let ratio = c.ratio(l);
            let mix = (2 + hc) * hc;
            for part in ["attn", "ffn"] {
                add(format!("{p}{part}_norm.weight"), &[h], 0.2, 1.0);
                add(format!("{p}hc_{part}_fn"), &[mix, hc * h], mat(hc * h, 4.0), 0.0);
                add(format!("{p}hc_{part}_base"), &[mix], 0.5, 0.0);
                add(format!("{p}hc_{part}_scale"), &[3], 0.25, 0.75);
            }
            add(format!("{p}attn.wq_a.weight"), &[ql, h], mat(h, 2.0), 0.0);
            add(format!("{p}attn.q_norm.weight"), &[ql], 0.2, 1.0);
            add(format!("{p}attn.wq_b.weight"), &[nh * hd, ql], mat(ql, 2.0), 0.0);
            add(format!("{p}attn.wkv.weight"), &[hd, h], mat(h, 2.0), 0.0);
            add(format!("{p}attn.kv_norm.weight"), &[hd], 0.2, 1.0);
            add(format!("{p}attn.wo_a.weight"), &[g * r, nh * hd / g], mat(nh * hd / g, 1.0), 0.0);
            add(format!("{p}attn.wo_b.weight"), &[h, g * r], mat(g * r, 1.0), 0.0);
            add(format!("{p}attn.attn_sink"), &[nh], 1.0, 0.0);
            let is_kv = c.kv_source_layer_ids.contains(&l);
            if is_kv {
                add(format!("{p}attn.compressor.wkv.weight"), &[hd, h], mat(h, 2.0), 0.0);
                if ratio > 1 {
                    add(format!("{p}attn.compressor.wgate.weight"), &[hd, h], mat(h, 3.0), 0.0);
                }
                add(format!("{p}attn.compressor.norm.weight"), &[hd], 0.2, 1.0);
                add(format!("{p}attn.indexer.wk.weight"), &[di, hd], mat(hd, 2.0), 0.0);
                add(format!("{p}attn.indexer.k_norm.weight"), &[di], 0.2, 1.0);
            }
            if c.index_source_layer_ids.contains(&l) {
                add(format!("{p}attn.indexer.wq_b.weight"), &[ni * di, ql], mat(ql, 2.0), 0.0);
                add(format!("{p}attn.indexer.weights_proj.weight"), &[ni, h], mat(h, 1.0), 0.0);
            }
            if let Some(li) = c.engram_layer_ids.iter().position(|&e| e == l) {
                add(format!("{p}engram.embed.weight"), &[rows[li], ed], 1.0, 0.0);
                add(
                    format!("{p}engram.wkv.weight"),
                    &[(hc + 1) * h, cols * ed],
                    mat(cols * ed, 3.0),
                    0.0,
                );
                add(format!("{p}engram.q_weight"), &[hc, h], 0.3, 1.0);
                add(format!("{p}engram.k_weight"), &[hc, h], 0.3, 1.0);
            }
            add(format!("{p}ffn.gate.weight"), &[ne, h], mat(h, 3.0), 0.0);
            add(format!("{p}ffn.gate.bias"), &[ne], 0.3, 0.0);
            add(format!("{p}ffn.gate.bias_vl"), &[ne], 2.0, 0.0);
            for e in (0..ne).map(|e| format!("experts.{e}")).chain(["shared_experts".to_string()]) {
                add(format!("{p}ffn.{e}.w1.weight"), &[mi, h], mat(h, 2.0), 0.0);
                add(format!("{p}ffn.{e}.w3.weight"), &[mi, h], mat(h, 2.0), 0.0);
                add(format!("{p}ffn.{e}.w2.weight"), &[h, mi], mat(mi, 1.0), 0.0);
            }
        }
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, bytes) in tensors {
            let entry = serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]});
            head.insert(name, entry);
            data.extend(bytes);
        }
        let mut head = serde_json::Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
    }

    /// A tiny random V4.1 model, without a reference (tools/ref_deepseek41.py measures the
    /// agreement with DeepSeek's code): the logits do not depend on how the tokens are split into
    /// passes (prefill blocks, short passes, one-token steps), a rollback leaves no trace, and
    /// each V4.1 mechanism (the indexer's choice, the candidate blocks, the engram lookups, the
    /// image tokens' routing bias and their break of the n-grams) changes the logits.
    #[test]
    fn tiny_model() {
        let dir = std::env::temp_dir().join(format!("oxinfer-ds41-test-{}", std::process::id()));
        tiny_checkpoint(&dir);
        let files = ModelFiles::open(&dir, true).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let load = |over: serde_json::Value| DeepSeek4Model::load(tiny_config(over), &files, Quant::Bf16).unwrap();
        let m = load(serde_json::json!({}));
        let (h, v) = (m.cfg.hidden_size, m.cfg.vocab_size);
        let ids: Vec<u32> = (0..70).map(|i| (i * 7 + 3) % v as u32).collect();
        let more = [5u32, 17, 2, 9];
        // The logits after the prompt (in passes of `chunk`) and after each of `more`.
        let run = |m: &DeepSeek4Model, chunk: usize| -> Vec<f32> {
            let mut sess = m.new_session();
            let hs = crate::generate::prefill(m, &mut sess, &ids, chunk, &[]);
            let mut out = m.logits(&hs[hs.len() - h..], 1);
            for &t in &more {
                out.extend(m.forward(&mut sess, &[t]));
            }
            out
        };
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let base = run(&m, 70);
        assert!(base.iter().all(|x| x.is_finite()));
        let scale = base.iter().fold(0f32, |a, b| a.max(b.abs()));
        let tol = 1e-3 * scale;
        for chunk in [33, 7, 1] {
            let d = diff(&run(&m, chunk), &base);
            assert!(d <= tol, "passes of {chunk} differ from one pass by {d} (logits up to {scale})");
        }
        // A rollback: the tokens after it run as if the dropped ones never had.
        let mut sess = m.new_session();
        m.forward(&mut sess, &ids);
        m.forward(&mut sess, &[30, 31, 29]);
        sess.truncate(ids.len());
        let again: Vec<f32> = more.iter().flat_map(|&t| m.forward(&mut sess, &[t])).collect();
        let d = diff(&again, &base[v..]);
        assert!(d <= tol, "after a rollback the logits differ by {d}");
        // Each mechanism matters.
        for (what, over) in [
            ("the indexer's choice", serde_json::json!({"index_topk": 64})),
            ("the candidate blocks", serde_json::json!({"candidate_source_layer_id": -1})),
            ("the engram lookups", serde_json::json!({"engram_layer_ids": []})),
        ] {
            let d = diff(&run(&load(over), 70), &base);
            assert!(d > 20.0 * tol, "{what}: the logits moved by {d} only");
        }
        // An image position: its own embedding and routing bias, no engram lookup, and the
        // n-grams of the tokens after it stop there.
        let row = vec![0.5f32; h];
        let with = |mm: &[(usize, &[f32])]| {
            let mut sess = m.new_session();
            let hs = m.forward_capture_mm(&mut sess, &ids, mm, &[], &mut Vec::new());
            (m.logits(&hs[hs.len() - h..], 1), sess.eng.clone())
        };
        let (img, eng) = with(&[(67, &row)]);
        assert_eq!(eng[67], -1);
        assert!(eng[66] >= 0 && eng[68] >= 0);
        assert!(diff(&img, &base[..v]) > 20.0 * tol);
    }
}
