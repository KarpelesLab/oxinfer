//! Qwen4-Exp (Qwen3.8-Flash-Next, transformers `qwen4_exp`): the Qwen3.5 MoE hybrid (Gated
//! DeltaNet with a sigmoid output gate, gated full attention, 512 routed + 1 shared expert) with
//! three additions, run by [`QwenModel`] when its config has `hc_count`:
//!
//! - **Hyper-connections** ("gated residual") in place of every layer norm: the residual stream
//!   is `hc_count` (4) parallel streams. Before each block, the streams are RMS-normed per stream
//!   (`hc_norm`), a low-rank bottleneck (`down` -> SiLU of `/ hc` -> `up` -> sigmoid) gives an
//!   element-wise read gate, and the block input is the mean over the streams of the gated normed
//!   streams; a per-stream write weight `2 sigmoid(inject . normed / hc)` scales the block output
//!   added to each stream. The embedding starts as `hc` copies; a final mixer (no write weight)
//!   gives the hidden state the LM head reads (there is no output norm).
//! - **N-gram hash embedding** (`ple`, 51B parameters, read from storage a few rows per token):
//!   before layer `ple_layer_ids - 1`, each token hashes its bigram and trigram (previous tokens,
//!   reset at EOS) into 8 heads each (`(t0 m0 ^ t1 m1 [^ t2 m2]) mod p_h + offset_h`), gathers
//!   16 rows of 160 and concatenates them (2560). A value projection gated per stream by
//!   `sigmoid(signed_sqrt(norm(key_s) . norm(stream_s) / sqrt(h)))` plus a dilated depthwise
//!   causal conv (kernel 4, dilation 3) of its normed copy, SiLU, is added to every stream.
//! - **QSA**: the full-attention layers attend to the top `indexer_budget / ratio` (512) blocks of
//!   4 cached tokens plus the incomplete tail, blocks scored by a small indexer (4 query heads of
//!   128 against the block's mean raw key, normed and rotated to the block's first position;
//!   `sum_h relu(q_h . k) / sqrt(128)`). Up to 2051 visible tokens every block is picked, so short
//!   contexts attend densely.

use crate::config::QwenConfig;
use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, linear, linear_multi};
use crate::pool;
use crate::qwen::{Loader, QwenModel, QwenSession};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One hyper-connection module (`Qwen4ExpTextGatedResidual`).
pub(crate) struct Hc<'a> {
    /// `[hc * h]`, as applied (1 + w for zero-centered checkpoints).
    norm: Vec<f32>,
    down: Lin<'a>, // [lowrank, hc * h]
    up: Lin<'a>,   // [hc * h, lowrank]
    /// `[hc, hc * h]` write weights (none for the final mixer).
    inject: Option<Lin<'a>>,
}

impl<'a> Hc<'a> {
    fn load(ld: &Loader<'a, '_>, p: &str, combine: bool) -> Result<Self> {
        let cfg = ld.cfg;
        let (h, hc, lr) = (cfg.hidden_size, cfg.hc_count, cfg.hc_lowrank);
        Ok(Hc {
            norm: ld.norm(&format!("{p}hc_norm.weight"), hc * h)?,
            down: ld.lin(&format!("{p}input_mix_weight_down.weight"), lr, hc * h)?,
            up: ld.lin(&format!("{p}input_mix_weight_up.weight"), hc * h, lr)?,
            inject: if combine {
                Some(ld.lin(&format!("{p}block_inject_weight.weight"), hc, hc * h)?)
            } else {
                None
            },
        })
    }

    pub(crate) fn lins(&self) -> Vec<&Lin<'a>> {
        let mut v = vec![&self.down, &self.up];
        v.extend(&self.inject);
        v
    }

    /// The block input `mixed` (`[t, h]`) of the streams `x` (`[t, hc * h]`), and with a write
    /// weight, the per-stream weights `inj` (`[t, hc]`, already `2 sigmoid(. / hc)`).
    pub(crate) fn mix(&self, cfg: &QwenConfig, x: &[f32], t: usize, mixed: &mut [f32], inj: &mut [f32]) {
        let (h, hc, lr) = (cfg.hidden_size, cfg.hc_count, cfg.hc_lowrank);
        let eps = cfg.rms_norm_eps as f32;
        let w = hc * h;
        let mut xn = vec![0f32; t * w];
        grouped_norm(x, &self.norm, h, eps, &mut xn);
        let mut lo = vec![0f32; t * lr];
        linear(&self.down, lr, w, &xn, t, &mut lo);
        let inv = 1.0 / hc as f32;
        for v in lo.iter_mut() {
            let a = *v * inv;
            *v = a * sigmoid(a);
        }
        let mut gate = vec![0f32; t * w];
        linear(&self.up, w, lr, &lo, t, &mut gate);
        for_chunks(mixed, h, t > 1, |ti, m| {
            m.fill(0.0);
            for s in 0..hc {
                let o = ti * w + s * h;
                for ((mv, &g), &xv) in m.iter_mut().zip(&gate[o..o + h]).zip(&xn[o..o + h]) {
                    *mv += sigmoid(g) * xv;
                }
            }
            for v in m.iter_mut() {
                *v *= inv;
            }
        });
        if let Some(iw) = &self.inject {
            linear(iw, hc, w, &xn, t, inj);
            for v in inj.iter_mut() {
                *v = 2.0 * sigmoid(*v * inv);
            }
        }
    }
}

/// `x[s] += y * inj[s]` for each stream `s` of each token.
pub(crate) fn combine(x: &mut [f32], y: &[f32], inj: &[f32], t: usize, h: usize, hc: usize) {
    for_chunks(x, hc * h, t > 1, |ti, row| {
        let yr = &y[ti * h..(ti + 1) * h];
        for s in 0..hc {
            axpy(&mut row[s * h..(s + 1) * h], inj[ti * hc + s], yr);
        }
    });
}

/// RMSNorm of every `h`-wide group of `x` with the weights of its stream (`w`: `[hc * h]`).
pub(crate) fn grouped_norm(x: &[f32], w: &[f32], h: usize, eps: f32, out: &mut [f32]) {
    let hc = w.len() / h;
    for_chunks(out, h, x.len() > 4 * h * hc, |i, o| {
        let s = i % hc;
        rmsnorm_f32w(&x[i * h..(i + 1) * h], &w[s * h..(s + 1) * h], eps, o);
    });
}

/// The n-gram embedding table: one GGUF tensor, or the shards of a safetensors checkpoint
/// (`(first row, table)`), read a few rows per token from the mapped files.
struct NgramTable<'a> {
    parts: Vec<(usize, Table<'a>)>,
    dim: usize,
}

impl NgramTable<'_> {
    fn part(&self, r: usize) -> (&Table<'_>, usize) {
        let i = self.parts.partition_point(|p| p.0 <= r) - 1;
        (&self.parts[i].1, r - self.parts[i].0)
    }
    fn row(&self, r: usize, out: &mut [f32]) {
        let (t, r) = self.part(r);
        t.row(r, out);
    }
    fn row_ranges(&self, r: usize) -> Vec<(usize, usize)> {
        let (t, r) = self.part(r);
        t.row_ranges(r, self.dim)
    }
}

/// The n-gram hash embedding block (`Qwen4ExpTextPLELayer`), before layer `layer`.
pub(crate) struct Ple<'a> {
    pub layer: usize,
    key: Lin<'a>,   // [hc * h, e]
    value: Lin<'a>, // [h, e]
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    /// `[hc * h, kernel]`.
    conv_w: Vec<f32>,
    /// The same as one undilated kernel of `hist + 1` taps (GPU decoding's conv).
    conv_dense: Vec<f32>,
    kernel: usize,
    dilation: usize,
    table: NgramTable<'a>,
    mult: Vec<u64>,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    eos: u32,
    per_ngram: usize,
    n: usize,
    e: usize,
}

impl<'a> Ple<'a> {
    /// Rows of the conv state (`(kernel - 1) * dilation`).
    pub(crate) fn hist(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }

    pub(crate) fn lins(&self) -> [&Lin<'a>; 2] {
        [&self.key, &self.value]
    }

    /// The table rows of the token at `i` of `toks` (all tokens so far): `(n - 1) * per_ngram`
    /// heads. Earlier tokens past an EOS (or before the start) read as EOS.
    fn rows(&self, toks: &[u32], i: usize, out: &mut Vec<usize>) {
        let eos = self.eos;
        let mut ctx = vec![eos as u64; self.n];
        ctx[0] = toks[i] as u64;
        let mut cut = false;
        for (s, c) in ctx.iter_mut().enumerate().skip(1) {
            let t = if cut || i < s { eos } else { toks[i - s] };
            cut = cut || i < s || t == eos;
            *c = if cut { eos as u64 } else { t as u64 };
        }
        for g in 2..=self.n {
            let mut mixed = ctx[0].wrapping_mul(self.mult[0]);
            for j in 1..g {
                mixed ^= ctx[j].wrapping_mul(self.mult[j]);
            }
            let base = (g - 2) * self.per_ngram;
            for hh in base..base + self.per_ngram {
                out.push((mixed % self.sizes[hh] + self.offsets[hh]) as usize);
            }
        }
    }

    /// Adds the block's output to the streams `x` (`[t, hc * h]`) for the last `t` tokens of
    /// `toks`, with conv state `conv` (`[hist, hc * h]`, oldest first; updated). `rec`: keeps the
    /// conv inputs of the pass (`[t, hc * h]`) for rewinds.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        cfg: &QwenConfig,
        stream: Option<&crate::stream::StreamHook>,
        toks: &[u32],
        x: &mut [f32],
        t: usize,
        conv: &mut [f32],
        rec: Option<&mut Vec<f32>>,
    ) {
        let (h, hc) = (cfg.hidden_size, cfg.hc_count);
        let eps = cfg.rms_norm_eps as f32;
        let w = hc * h;
        let first = toks.len() - t;
        let mut rows = Vec::with_capacity(t * 16);
        for i in first..toks.len() {
            self.rows(toks, i, &mut rows);
        }
        if let Some(s) = stream {
            let r: Vec<(usize, usize)> = rows.iter().flat_map(|&r| self.table.row_ranges(r)).collect();
            s.need_ranges(&r);
        }
        let d = self.table.dim;
        let mut emb = vec![0f32; t * self.e];
        for_chunks(&mut emb, d, t * rows.len() / t.max(1) > 64, |i, o| self.table.row(rows[i], o));
        let mut key = vec![0f32; t * w];
        let mut value = vec![0f32; t * h];
        linear_multi(
            &mut [(&self.key, w, &mut key[..]), (&self.value, h, &mut value[..])],
            self.e,
            &emb,
            t,
        );
        let mut kn = vec![0f32; t * w];
        grouped_norm(&key, &self.norm_key, h, eps, &mut kn);
        let mut qn = vec![0f32; t * w];
        grouped_norm(x, &self.norm_query, h, eps, &mut qn);
        // gated[t, s] = sigmoid(signed sqrt(kn_s . qn_s / sqrt(h))) * value
        let mut gated = vec![0f32; t * w];
        let scale = 1.0 / (h as f32).sqrt();
        for_chunks(&mut gated, w, t > 1, |ti, g| {
            for s in 0..hc {
                let o = ti * w + s * h;
                let a = dot(&kn[o..o + h], &qn[o..o + h]) * scale;
                let a = if a == 0.0 { 0.0 } else { a.abs().max(1e-6).sqrt() * a.signum() };
                let sg = sigmoid(a);
                for (gv, &v) in g[s * h..(s + 1) * h].iter_mut().zip(&value[ti * h..(ti + 1) * h]) {
                    *gv = sg * v;
                }
            }
        });
        let mut gn = vec![0f32; t * w];
        grouped_norm(&gated, &self.norm_conv, h, eps, &mut gn);
        if let Some(r) = rec {
            r.clone_from(&gn);
        }
        // out = gated + silu(sum_k w[c][k] * gn[t - (K - 1 - k) * dil][c]), history from `conv`.
        let (kk, dil, hist) = (self.kernel, self.dilation, self.hist());
        {
            let src = |i: isize, c: usize| -> f32 {
                if i >= 0 {
                    gn[i as usize * w + c]
                } else {
                    conv[(hist as isize + i) as usize * w + c]
                }
            };
            let cw = &self.conv_w;
            for_chunks(x, w, t > 1, |ti, row| {
                for c in 0..w {
                    let mut acc = 0f32;
                    for k in 0..kk {
                        acc += cw[c * kk + k] * src(ti as isize - ((kk - 1 - k) * dil) as isize, c);
                    }
                    row[c] += gated[ti * w + c] + acc * sigmoid(acc);
                }
            });
        }
        shift_state(conv, &gn, t, w);
    }
}

/// Shifts `t` new rows (`[t, w]`) into a state of the last rows (`[n, w]`, oldest first).
fn shift_state(state: &mut [f32], new: &[f32], t: usize, w: usize) {
    let n = state.len() / w;
    if t >= n {
        state.copy_from_slice(&new[(t - n) * w..t * w]);
    } else {
        state.copy_within(t * w.., 0);
        state[(n - t) * w..].copy_from_slice(&new[..t * w]);
    }
}

/// The QSA block selector of one full-attention layer (`Qwen4ExpTextQSAIndexer`).
pub(crate) struct Indexer<'a> {
    /// `[n_heads * d, h]`, or `[(n_heads + 1) * d, h]` with the key rows after the queries (the
    /// fused projection of the safetensors checkpoint) when `k` is None.
    q: Lin<'a>,
    k: Option<Lin<'a>>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl<'a> Indexer<'a> {
    pub(crate) fn lins(&self) -> Vec<&Lin<'a>> {
        let mut v = vec![&self.q];
        v.extend(&self.k);
        v
    }
}

/// Per-layer Qwen4-Exp weights.
pub(crate) struct ExpLayer<'a> {
    pub attn_hc: Hc<'a>,
    pub mlp_hc: Hc<'a>,
    pub indexer: Option<Indexer<'a>>,
}

/// The Qwen4-Exp parts of a [`QwenModel`].
pub(crate) struct Exp<'a> {
    pub layers: Vec<ExpLayer<'a>>,
    pub mixer: Hc<'a>,
    pub ple: Option<Ple<'a>>,
}

/// Indexer keys of one full-attention layer: the raw key of every cached position and the pooled
/// (normed, rotated) key of every complete block.
#[derive(Clone, Default)]
pub struct IdxCache {
    /// `[len, d]`.
    raw: Vec<f32>,
    /// `[blocks, d]`.
    pooled: Vec<f32>,
    d: usize,
}

impl IdxCache {
    fn truncate(&mut self, pos: usize, ratio: usize) {
        self.raw.truncate(pos * self.d);
        self.pooled.truncate(pos / ratio * self.d);
    }
}

/// Session state of the Qwen4-Exp parts.
#[derive(Clone)]
pub struct ExpSession {
    /// Every token fed (n-gram contexts).
    toks: Vec<u32>,
    /// The n-gram block's conv state (`[hist, hc * h]`; page-aligned: GPU decoding updates it in
    /// place).
    ple_conv: crate::quant::PageVec<f32>,
    /// The conv state before the last recorded pass and that pass's conv inputs.
    ple_conv0: Vec<f32>,
    ple_rec: Vec<f32>,
    idx: Vec<Option<IdxCache>>,
    ratio: usize,
    w: usize,
    /// The layer the n-gram block precedes (its conv state comes before that layer's states).
    ple_layer: Option<usize>,
    /// The residual streams (`[t, hc * h]`) after the last layer of the last forward (what the MTP
    /// head reads).
    pub(crate) last_wide: Vec<f32>,
}

impl ExpSession {
    /// Rewinds to `pos` tokens; `rec`: the span the last forward recorded.
    /// Drops the n-gram contexts and indexer keys of positions from `pos` on.
    pub(crate) fn rewind_keys(&mut self, pos: usize) {
        self.toks.truncate(pos);
        for c in self.idx.iter_mut().flatten() {
            c.truncate(pos, self.ratio);
        }
    }

    pub(crate) fn rewind(&mut self, pos: usize, rec: Option<(usize, usize)>) -> Result<()> {
        if pos >= self.toks.len() {
            return Ok(());
        }
        let w = self.w;
        if !self.ple_conv.is_empty() {
            let Some((start, len)) = rec else {
                bail!("cannot rewind the n-gram conv state to {pos}: the last forward did not record");
            };
            if pos < start || pos > start + len || self.ple_rec.len() != len * w {
                bail!(
                    "cannot rewind the n-gram conv state to {pos}: recorded span is {start}..{}",
                    start + len
                );
            }
            self.ple_conv.copy_from_slice(&self.ple_conv0);
            shift_state(&mut self.ple_conv, &self.ple_rec[..(pos - start) * w], pos - start, w);
            // A later rewind inside the shorter span stays possible.
            self.ple_rec.truncate((pos - start) * w);
        }
        self.toks.truncate(pos);
        for c in self.idx.iter_mut().flatten() {
            c.truncate(pos, self.ratio);
        }
        Ok(())
    }
}

impl<'a> Exp<'a> {
    pub(crate) fn load(ld: &Loader<'a, '_>, prefix: &str) -> Result<Self> {
        let cfg = ld.cfg;
        let files = ld.files;
        let h = cfg.hidden_size;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}layers.{l}.");
            if ld.has(&format!("{lp}attn_hyper_connection.down_inject.weight")) || ld.has(&format!("{lp}linear_attn.in_proj_qkvz.weight")) {
                bail!("GGUF files with merged Qwen4-Exp weights (hc_*_down_inject, attn_qkv_gate) are not supported");
            }
            let indexer = if cfg.is_full(l) {
                Some(load_indexer(ld, &format!("{lp}self_attn.indexer."))?)
            } else {
                None
            };
            layers.push(ExpLayer {
                attn_hc: Hc::load(ld, &format!("{lp}attn_hyper_connection."), true)?,
                mlp_hc: Hc::load(ld, &format!("{lp}mlp_hyper_connection."), true)?,
                indexer,
            });
        }
        let mixer = Hc::load(ld, &format!("{prefix}hyper_connection_mixer."), false)?;
        let ple = match cfg.ple_layer_ids.as_slice() {
            [] => None,
            [one] if *one >= 1 && *one <= cfg.num_hidden_layers => Some(load_ple(ld, files, prefix, one - 1, h)?),
            other => bail!("n-gram embedding on layers {other:?}: one layer is supported"),
        };
        Ok(Exp { layers, mixer, ple })
    }

    /// The extra weights of layer `l` (streaming units).
    pub(crate) fn layer_lins(&self, l: usize) -> Vec<&Lin<'a>> {
        let el = &self.layers[l];
        let mut v: Vec<&Lin> = el.attn_hc.lins().into_iter().chain(el.mlp_hc.lins()).collect();
        if let Some(ix) = &el.indexer {
            v.extend(ix.lins());
        }
        if let Some(p) = &self.ple
            && p.layer == l
        {
            v.extend(p.lins());
        }
        v
    }

    pub(crate) fn new_session(&self, cfg: &QwenConfig) -> ExpSession {
        let w = cfg.hc_count * cfg.hidden_size;
        ExpSession {
            toks: Vec::new(),
            ple_conv: crate::quant::PageVec::zeroed(self.ple.as_ref().map_or(0, |p| p.hist() * w)),
            ple_conv0: Vec::new(),
            ple_rec: Vec::new(),
            idx: self
                .layers
                .iter()
                .map(|l| {
                    l.indexer.as_ref().map(|_| IdxCache {
                        d: cfg.indexer_head_dim,
                        ..Default::default()
                    })
                })
                .collect(),
            ratio: cfg.indexer_compress_ratio.max(1),
            w,
            ple_layer: self.ple.as_ref().map(|p| p.layer),
            last_wide: Vec::new(),
        }
    }
}

fn load_indexer<'a>(ld: &Loader<'a, '_>, p: &str) -> Result<Indexer<'a>> {
    let cfg = ld.cfg;
    let (h, nh, d) = (cfg.hidden_size, cfg.indexer_n_heads, cfg.indexer_head_dim);
    let (q, k) = if ld.has(&format!("{p}index_qk_proj.weight")) {
        (ld.lin(&format!("{p}index_qk_proj.weight"), (nh + 1) * d, h)?, None)
    } else {
        (
            ld.lin(&format!("{p}q_proj.weight"), nh * d, h)?,
            Some(ld.lin(&format!("{p}k_proj.weight"), d, h)?),
        )
    };
    Ok(Indexer {
        q,
        k,
        q_norm: ld.norm(&format!("{p}q_layernorm.weight"), d)?,
        k_norm: ld.norm(&format!("{p}k_layernorm.weight"), d)?,
    })
}

/// Little-endian int64 buffer of a safetensors checkpoint.
fn i64s(files: &ModelFiles, name: &str) -> Result<Vec<u64>> {
    let b = files.bytes(name)?;
    Ok(b.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c) as u64).collect())
}

fn load_ple<'a>(ld: &Loader<'a, '_>, files: &'a ModelFiles, prefix: &str, layer: usize, h: usize) -> Result<Ple<'a>> {
    let cfg = ld.cfg;
    let hc = cfg.hc_count;
    let w = hc * h;
    let e = cfg.ple_embed_dim;
    let n = cfg.ngram_size;
    let per = cfg.heads_per_ngram;
    let heads = (n - 1) * per;
    let p = format!("{prefix}layers.{layer}.ple.");
    let ep = format!("{p}ple_embedding.");
    let (mult, offsets, sizes) = match (&cfg.ple_layer_multipliers, &cfg.ple_head_offsets, &cfg.ple_head_vocab_sizes) {
        (Some(m), Some(o), Some(s)) => (m.clone(), o.clone(), s.clone()),
        _ => (
            i64s(files, &format!("{ep}layer_multipliers"))?,
            i64s(files, &format!("{ep}ngram_heads_offsets"))?,
            i64s(files, &format!("{ep}ngram_heads_vocab_sizes"))?,
        ),
    };
    if mult.len() < n || offsets.len() < heads || sizes.len() < heads || sizes.contains(&0) {
        bail!(
            "n-gram hash constants: {} multipliers, {} offsets, {} sizes for {heads} heads",
            mult.len(),
            offsets.len(),
            sizes.len()
        );
    }
    let dim = e / heads;
    // The table: one GGUF tensor, or the checkpoint's row shards.
    let mut parts = Vec::new();
    if let Some(info) = files.info("model.ngram_embedding.weight") {
        let rows = info.shape[0];
        parts.push((0, Table::load(files, "model.ngram_embedding.weight", rows, dim, Quant::Bf16)?));
    } else {
        let mut row = 0;
        for i in 0.. {
            let name = format!("{ep}ngram_embedding.shard_{i}.weight");
            let Some(info) = files.info(&name) else { break };
            let rows = info.shape[0];
            parts.push((row, Table::load(files, &name, rows, dim, Quant::Bf16)?));
            row += rows;
        }
    }
    if parts.is_empty() {
        bail!("missing the n-gram embedding table");
    }
    let total: usize = files
        .info("model.ngram_embedding.weight")
        .map(|i| i.shape[0])
        .unwrap_or_else(|| parts.iter().map(|(r, t)| r + table_rows(t, dim)).max().unwrap_or(0));
    let need = (0..heads).map(|i| offsets[i] + sizes[i]).max().unwrap_or(0) as usize;
    if total < need {
        bail!("the n-gram table has {total} rows, the hash needs {need}");
    }
    let kernel = cfg.ple_conv_kernel_size;
    let conv_w = ld.getf(&format!("{p}conv1d.weight"), &[w, 1, kernel])?;
    Ok(Ple {
        layer,
        key: ld.lin(&format!("{p}key_proj.weight"), w, e)?,
        value: ld.lin(&format!("{p}value_proj.weight"), h, e)?,
        norm_key: ld.norm(&format!("{p}norm_key.weight"), w)?,
        norm_query: ld.norm(&format!("{p}norm_query.weight"), w)?,
        norm_conv: ld.norm(&format!("{p}norm_conv.weight"), w)?,
        conv_dense: dense_conv(&conv_w, w, kernel, n),
        conv_w,
        kernel,
        dilation: n,
        table: NgramTable { parts, dim },
        mult,
        offsets,
        sizes,
        eos: cfg.eos_id().context("the n-gram embedding needs eos_token_id")?,
        per_ngram: per,
        n,
        e,
    })
}

/// A dilated depthwise kernel (`[c, k]`, dilation `dil`) as an undilated one of
/// `(k - 1) * dil + 1` taps, zero between the dilated ones.
fn dense_conv(w: &[f32], c: usize, k: usize, dil: usize) -> Vec<f32> {
    let kd = (k - 1) * dil + 1;
    let mut d = vec![0f32; c * kd];
    for ch in 0..c {
        for j in 0..k {
            d[ch * kd + j * dil] = w[ch * k + j];
        }
    }
    d
}

fn table_rows(t: &Table, dim: usize) -> usize {
    match t {
        Table::Bf16(w) => w.len() / dim,
        _ => 0,
    }
}

/// The blocks a query at `qpos` (absolute) attends to under QSA, as token positions, or None
/// when every visible token is selected.
fn qsa_select(cfg: &QwenConfig, scores: &[f32], qpos: usize) -> Option<Vec<u32>> {
    let r = cfg.indexer_compress_ratio;
    let top = cfg.indexer_budget / r;
    let nb = (qpos + 1) / r;
    if nb <= top {
        return None;
    }
    let mut idx: Vec<u32> = (0..nb as u32).collect();
    // Highest scores first, ties to the lower block (torch.topk is unspecified there).
    idx.select_nth_unstable_by(top - 1, |&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut blocks = idx[..top].to_vec();
    blocks.sort_unstable();
    let mut sel: Vec<u32> = blocks.iter().flat_map(|&b| (b * r as u32)..(b + 1) * r as u32).collect();
    sel.extend((nb * r) as u32..=qpos as u32);
    Some(sel)
}

/// Attention of the queries `q` (`[t, nh, hd]`, scaled) over the cached positions `sel[ti]` of
/// each token (or all up to its own when None).
pub(crate) fn attention_sel(q: &[f32], cache: &LayerCache, nh: usize, t: usize, start: usize, sel: &[Option<Vec<u32>>], out: &mut [f32]) {
    let hd = cache.hd;
    let n_kv = cache.n_kv;
    let rep = nh / n_kv;
    let op = SendF(out.as_mut_ptr());
    pool::run(t * n_kv, &|task| {
        let (ti, kvh) = (task / n_kv, task % n_kv);
        let qpos = start + ti;
        let all: Vec<u32>;
        let pos: &[u32] = match &sel[ti] {
            Some(s) => s,
            None => {
                all = (0..=qpos as u32).collect();
                &all
            }
        };
        let n = pos.len();
        let mut scores = vec![0f32; n];
        for hh in 0..rep {
            let h0 = kvh * rep + hh;
            let qr = &q[(ti * nh + h0) * hd..(ti * nh + h0 + 1) * hd];
            for (s, &p) in scores.iter_mut().zip(pos) {
                *s = dot(qr, cache.k_at(p as usize, kvh));
            }
            softmax_inplace(&mut scores);
            // SAFETY: each (token, head) row of `out` is written by one task.
            let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((ti * nh + h0) * hd), hd) };
            o.fill(0.0);
            for (&s, &p) in scores.iter().zip(pos) {
                axpy(o, s, cache.v_at(p as usize, kvh));
            }
        }
    });
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

#[derive(Clone, Copy)]
struct SendSel(*mut Option<Vec<u32>>);
unsafe impl Send for SendSel {}
unsafe impl Sync for SendSel {}
impl SendSel {
    fn get(self) -> *mut Option<Vec<u32>> {
        self.0
    }
}

impl<'a> QwenModel<'a> {
    /// The Qwen4-Exp forward (see the module docs): the mixed hidden states `[t, h]` the LM head
    /// reads.
    pub(crate) fn exp_forward(
        &self,
        ex: &Exp<'a>,
        sess: &mut QwenSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        record: bool,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let t = tokens.len();
        let (h, hc) = (cfg.hidden_size, cfg.hc_count);
        let w = hc * h;
        let start = sess.pos;

        let mut x0 = vec![0f32; t * h];
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        for_chunks(&mut x0, h, t > 1, |ti, row| self.embed.row(tokens[ti] as usize, row));
        for &(i, emb) in mm {
            x0[i * h..(i + 1) * h].copy_from_slice(emb);
        }
        let (cos, sin) = match &sess.positions {
            Some(p) if p.len() >= start + t => self.rope_cs_at(&p[start..start + t]),
            _ => self.rope_cs(&sess.spans, start, t),
        };
        // The n-gram ids of image positions are the image token (as the reference hashes them).
        {
            let es = sess.exp.as_mut().expect("Qwen4-Exp session");
            es.toks.truncate(start);
            for (i, &tk) in tokens.iter().enumerate() {
                let img = mm.iter().any(|m| m.0 == i);
                es.toks.push(if img { cfg.image_token_id.unwrap_or(tk) } else { tk });
            }
        }
        let mut x = vec![0f32; t * w];
        for ti in 0..t {
            for s in 0..hc {
                x[ti * w + s * h..ti * w + (s + 1) * h].copy_from_slice(&x0[ti * h..(ti + 1) * h]);
            }
        }
        drop(x0);

        let mut hn = vec![0f32; t * h];
        let mut y = vec![0f32; t * h];
        let mut inj = vec![0f32; t * hc];
        for (l, layer) in self.layers.iter().enumerate() {
            let _unit = crate::stream::enter(&self.stream, l);
            let el = &ex.layers[l];
            if let Some(p) = &ex.ple
                && p.layer == l
            {
                let es = sess.exp.as_mut().unwrap();
                let mut rec = Vec::new();
                if record {
                    es.ple_conv0 = es.ple_conv.to_vec();
                }
                p.forward(
                    cfg,
                    self.stream.as_ref(),
                    &es.toks,
                    &mut x,
                    t,
                    &mut es.ple_conv,
                    record.then_some(&mut rec),
                );
                if record {
                    es.ple_rec = rec;
                }
            }
            el.attn_hc.mix(cfg, &x, t, &mut hn, &mut inj);
            match &layer.mixer {
                crate::qwen::Mixer::Gdn(gd) => {
                    let rec = if record { Some(&mut sess.rec[l]) } else { None };
                    sess.gpu_states_kept = false;
                    self.gdn_forward(gd, sess.gdn[l].as_mut().unwrap(), rec, &hn, t, &mut y);
                }
                crate::qwen::Mixer::Attn(at) => {
                    let ix = el.indexer.as_ref().expect("QSA indexer");
                    let sel = self.qsa_select_all(ix, sess, l, &hn, t, start);
                    let cache = sess.kv[l].as_mut().unwrap();
                    self.attn_forward_sel(at, cache, &hn, t, start, &cos, &sin, sel.as_deref(), &mut y);
                }
            }
            combine(&mut x, &y, &inj, t, h, hc);
            el.mlp_hc.mix(cfg, &x, t, &mut hn, &mut inj);
            let need = |ids: &[usize]| {
                if let Some(s) = &self.stream {
                    s.need_parts(l, ids)
                }
            };
            layer
                .ffn
                .forward(&hn, t, h, &mut y, self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize])), None);
            combine(&mut x, &y, &inj, t, h, hc);
            if sess.capture.contains(&l) {
                sess.captured.push((l, x.clone()));
            }
        }
        let mut out = vec![0f32; t * h];
        {
            let _unit = crate::stream::enter(&self.stream, self.layers.len());
            ex.mixer.mix(cfg, &x, t, &mut out, &mut []);
        }
        sess.exp.as_mut().unwrap().last_wide = x;
        sess.rec_span = if record { Some((start, t)) } else { None };
        sess.pos += t;
        out
    }

    /// The QSA indexer of full-attention layer `l` for `t` tokens at `start` (block input `hn`):
    /// caches their raw keys and the pooled keys of the blocks they complete, then picks the
    /// positions each token attends to (None for all: every visible block is in the budget).
    fn qsa_select_all(
        &self,
        ix: &Indexer,
        sess: &mut QwenSession,
        l: usize,
        hn: &[f32],
        t: usize,
        start: usize,
    ) -> Option<Vec<Option<Vec<u32>>>> {
        let QwenSession { exp, spans, positions, .. } = sess;
        let ic = exp.as_mut().unwrap().idx[l].as_mut().expect("indexer cache");
        let (spans, positions) = (&*spans, &*positions);
        let rope_at = |p: usize, n: usize| match positions {
            Some(ps) if ps.len() >= p + n => self.rope_cs_at(&ps[p..p + n]),
            _ => self.rope_cs(spans, p, n),
        };
        self.qsa_select(ix, ic, &rope_at, hn, t, start, l)
    }

    /// [`Self::qsa_select_all`] over an indexer cache, with `rope_at(p, n)` the rotary tables of
    /// positions `p..p + n`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qsa_select(
        &self,
        ix: &Indexer,
        ic: &mut IdxCache,
        rope_at: &dyn Fn(usize, usize) -> (Vec<f32>, Vec<f32>),
        hn: &[f32],
        t: usize,
        start: usize,
        l: usize,
    ) -> Option<Vec<Option<Vec<u32>>>> {
        let cfg = &self.cfg;
        let (nh, d, h) = (cfg.indexer_n_heads, cfg.indexer_head_dim, cfg.hidden_size);
        let r = cfg.indexer_compress_ratio;
        let eps = cfg.rms_norm_eps as f32;
        let half = self.inv_freq.len();
        let mut q = vec![0f32; t * nh * d];
        let mut k = vec![0f32; t * d];
        match &ix.k {
            Some(kl) => linear_multi(&mut [(&ix.q, nh * d, &mut q[..]), (kl, d, &mut k[..])], h, hn, t),
            None => {
                let mut qk = vec![0f32; t * (nh + 1) * d];
                linear(&ix.q, (nh + 1) * d, h, hn, t, &mut qk);
                for ti in 0..t {
                    let row = &qk[ti * (nh + 1) * d..(ti + 1) * (nh + 1) * d];
                    q[ti * nh * d..(ti + 1) * nh * d].copy_from_slice(&row[..nh * d]);
                    k[ti * d..(ti + 1) * d].copy_from_slice(&row[nh * d..]);
                }
            }
        }
        let top = cfg.indexer_budget / r;
        let last_nb = (start + t) / r;
        let have = {
            ic.raw.truncate(start * d);
            ic.raw.extend_from_slice(&k);
            ic.pooled.truncate(start / r * d);
            ic.pooled.len() / d
        };
        if last_nb <= top {
            // Every block is selected: the pooled keys wait until some token needs them.
            return None;
        }
        let mut newp = Vec::with_capacity(last_nb.saturating_sub(have) * d);
        if last_nb > have {
            let mut cos = Vec::new();
            let mut sin = Vec::new();
            for b in have..last_nb {
                let (cb, sb) = rope_at(b * r, 1);
                cos.extend(cb);
                sin.extend(sb);
            }
            let raw = &ic.raw;
            let mut m = vec![0f32; d];
            let mut kn = vec![0f32; d];
            for (i, b) in (have..last_nb).enumerate() {
                m.fill(0.0);
                for p in b * r..(b + 1) * r {
                    add_inplace(&mut m, &raw[p * d..(p + 1) * d]);
                }
                scale_inplace(&mut m, 1.0 / r as f32);
                rmsnorm_f32w(&m, &ix.k_norm, eps, &mut kn);
                apply_rope(&mut kn[..2 * half], &cos[i * half..(i + 1) * half], &sin[i * half..(i + 1) * half]);
                newp.extend_from_slice(&kn);
            }
        }
        let (cos, sin) = rope_at(start, t);
        ic.pooled.extend_from_slice(&newp);
        let pooled = &ic.pooled;
        let scale = 1.0 / (d as f32).sqrt();
        let mut sel: Vec<Option<Vec<u32>>> = vec![None; t];
        let sp = SendSel(sel.as_mut_ptr());
        pool::run(t, &|ti| {
            let qpos = start + ti;
            let nb = (qpos + 1) / r;
            if nb <= top {
                return;
            }
            let mut qn = vec![0f32; nh * d];
            for hh in 0..nh {
                let o = &mut qn[hh * d..(hh + 1) * d];
                rmsnorm_f32w(&q[(ti * nh + hh) * d..(ti * nh + hh + 1) * d], &ix.q_norm, eps, o);
                apply_rope(
                    &mut o[..2 * half],
                    &cos[ti * half..(ti + 1) * half],
                    &sin[ti * half..(ti + 1) * half],
                );
            }
            let scores: Vec<f32> = (0..nb)
                .map(|b| {
                    let kb = &pooled[b * d..(b + 1) * d];
                    (0..nh).map(|hh| dot(&qn[hh * d..(hh + 1) * d], kb).max(0.0)).sum::<f32>() * scale
                })
                .collect();
            // SAFETY: each token's entry is written by one task.
            unsafe { *sp.get().add(ti) = qsa_select(cfg, &scores, qpos) };
        });
        if std::env::var_os("OXINFER_QSA_DEBUG").is_some() {
            for (i, s) in sel.iter().enumerate() {
                eprintln!(
                    "L {l} {} {:?}",
                    start + i,
                    s.clone().unwrap_or_else(|| (0..=(start + i) as u32).collect())
                );
            }
        }
        Some(sel)
    }
}

/// Most positions the GPU decode plan attends to densely: beyond `indexer_budget +
/// indexer_compress_ratio - 1` visible tokens QSA drops blocks, which only the CPU path does.
fn dense_limit(cfg: &QwenConfig) -> usize {
    cfg.indexer_budget + cfg.indexer_compress_ratio.max(1) - 1
}

impl<'a> QwenModel<'a> {
    /// One token's Qwen4-Exp forward as a device graph (`crate::gpu::decode`): the hyper-connection
    /// mixes and writes, the n-gram block (its embedding rows gathered on the host: the second
    /// input), Gated DeltaNet, the attention layers densely (valid up to [`dense_limit`] tokens),
    /// the MoE blocks. The indexer keys of the attention layers are captured for the host cache.
    pub(crate) fn exp_gpu_graph<'s>(&'s self, ex: &'s Exp<'s>, multi: bool) -> Option<crate::gpu::decode::Graph<'s>> {
        use crate::gpu::decode::{Ew, Experts, Graph, LinMode, Op};
        use crate::qwen::{Ffn, Mixer};
        let cfg = &self.cfg;
        let (h, hc, lr) = (cfg.hidden_size, cfg.hc_count, cfg.hc_lowrank);
        let w = hc * h;
        let eps = cfg.rms_norm_eps as f32;
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let qw = 2 * nh * hd;
        let d = cfg.indexer_head_dim;
        let dev = |l: &'s Lin, n_out: usize, n_in: usize| -> Option<&'s crate::gpu::DeviceWeight> {
            if l.has_lora() || l.is_transient() {
                return None;
            }
            l.device_weight(n_out, n_in)
        };
        let lin = |l: &'s Lin, n_out: usize, n_in: usize, x, y, mode| -> Option<Op<'s>> {
            Some(Op::Linear {
                w: l.as_ref(),
                dev: dev(l, n_out, n_in)?,
                w2: None,
                n_out,
                n_in,
                x,
                y,
                mode,
            })
        };
        let mut g = Graph::new();
        let x0 = g.buf(h);
        let xw = g.buf(w);
        let xn = g.buf(w);
        let lo = g.buf(lr);
        let gate = g.buf(w);
        let hn = g.buf(h);
        let inj = g.buf(hc);
        let y = g.buf(h);
        let qkv = g.buf(qw + 2 * nkv * hd);
        let qo = g.buf(nh * hd);
        let att = g.buf(nh * hd);
        let n_attn = self.layers.iter().filter(|l| matches!(l.mixer, Mixer::Attn(_))).count();
        // Captured: the indexer keys of the attention layers, then the residual streams after the
        // last layer (what the MTP head reads).
        let cap = g.buf(n_attn * d + w);
        let itmp = g.buf((cfg.indexer_n_heads + 1) * d);
        let (mut n_exp, mut top_k, mut e_inter, mut s_inter) = (0, 0, 0, 0);
        for l in &self.layers {
            match &l.ffn {
                Ffn::Moe(m) => {
                    n_exp = n_exp.max(m.experts.down.len());
                    top_k = top_k.max(m.experts.top_k);
                    e_inter = e_inter.max(m.experts.inter);
                    s_inter = s_inter.max(m.shared.inter);
                }
                Ffn::Dense(_) => return None,
            }
        }
        let act = g.buf(s_inter.max(1));
        let act2 = g.buf(s_inter.max(1));
        let rlogits = g.buf(n_exp + 1);
        let sel = g.buf((2 * top_k).max(1));
        let eact = g.buf((top_k * e_inter).max(1));
        let (nk, nv, hdk, hdv) = (
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        );
        let (key_dim, value_dim) = (nk * hdk, nv * hdv);
        let conv_dim = 2 * key_dim + value_dim;
        let gproj = g.buf(conv_dim + value_dim + 2 * nv);
        let gconv = g.buf(conv_dim);
        let gout = g.buf(value_dim);
        let hidden = g.buf(h);
        let logits = g.buf(cfg.vocab_size);
        g.input = x0;
        g.hidden = hidden;
        g.n_hidden = h;
        g.rope_halves = vec![self.inv_freq.len()];
        g.capture = Some((cap, n_attn * d + w));
        // The n-gram block's buffers.
        let ple = ex.ple.as_ref();
        let e = ple.map_or(1, |p| p.e);
        let emb = g.buf(e);
        let (kb, kn, qn, vb, gated, gn, cv) = if ple.is_some() {
            (g.buf(w), g.buf(w), g.buf(w), g.buf(h), g.buf(w), g.buf(w), g.buf(w))
        } else {
            (emb, emb, emb, emb, emb, emb, emb)
        };
        if let Some(p) = ple {
            g.input2 = Some((emb, p.e));
        }
        // Per-stream RMSNorms of `src` into `dst` with the stream's weights.
        let gnorm = |g: &mut Graph<'s>, src: crate::gpu::decode::View, dst: crate::gpu::decode::View, wt: &'s [f32]| {
            for s in 0..hc {
                g.ops.push(Op::norm(src.at(s * h), dst.at(s * h), &wt[s * h..(s + 1) * h], h, eps));
            }
        };
        // A hyper-connection read of xw into `out` (and the write weights into `inj`).
        let mix = |g: &mut Graph<'s>, c: &'s Hc<'s>, out: crate::gpu::decode::View| -> Option<()> {
            gnorm(g, xw, xn, &c.norm);
            g.ops.push(lin(&c.down, lr, w, xn, lo, LinMode::Store)?);
            g.ops.push(Op::Ew {
                kind: Ew::SiluScale(1.0 / hc as f32),
                a: lo,
                b: None,
                out: lo,
                n: lr,
            });
            g.ops.push(lin(&c.up, w, lr, lo, gate, LinMode::Store)?);
            g.ops.push(Op::HcPre { gate, xn, out, n: h, hc });
            if let Some(iw) = &c.inject {
                g.ops.push(lin(iw, hc, w, xn, inj, LinMode::Store)?);
            }
            Some(())
        };
        for s in 0..hc {
            g.ops.push(Op::Ew {
                kind: Ew::Scale(1.0),
                a: x0,
                b: None,
                out: xw.at(s * h),
                n: h,
            });
        }
        let mut ai = 0;
        for (li, layer) in self.layers.iter().enumerate() {
            let el = &ex.layers[li];
            if let Some(p) = ple
                && p.layer == li
            {
                let st = g.states.len();
                let hist = p.hist();
                g.states.push(hist * w);
                g.ops.push(lin(&p.key, w, p.e, emb, kb, LinMode::Store)?);
                g.ops.push(lin(&p.value, h, p.e, emb, vb, LinMode::Store)?);
                gnorm(&mut g, kb, kn, &p.norm_key);
                gnorm(&mut g, xw, qn, &p.norm_query);
                g.ops.push(Op::PleGate {
                    k: kn,
                    q: qn,
                    v: vb,
                    out: gated,
                    n: h,
                    hc,
                });
                gnorm(&mut g, gated, gn, &p.norm_conv);
                g.ops.push(Op::GdnConv {
                    x: gn,
                    out: cv,
                    state: st,
                    w: &p.conv_dense,
                    dim: w,
                    kc: hist + 1,
                });
                for src in [gated, cv] {
                    g.ops.push(Op::Ew {
                        kind: Ew::Combine { sa: 1.0, sb: 1.0, s: 1.0 },
                        a: xw,
                        b: Some(src),
                        out: xw,
                        n: w,
                    });
                }
            }
            mix(&mut g, &el.attn_hc, hn)?;
            match &layer.mixer {
                Mixer::Gdn(gd) => {
                    // Multi-token plans keep each recurrent layer's inputs in its own buffers (replays).
                    let (gproj, gconv) = if multi {
                        (g.buf(conv_dim + value_dim + 2 * nv), g.buf(conv_dim))
                    } else {
                        (gproj, gconv)
                    };
                    let st = g.states.len();
                    g.states.push(nv * hdk * hdv);
                    g.states.push((cfg.linear_conv_kernel_dim - 1) * conv_dim);
                    g.ops.push(lin(&gd.qkv, conv_dim, h, hn, gproj, LinMode::Store)?);
                    g.ops.push(lin(&gd.z, value_dim, h, hn, gproj.at(conv_dim), LinMode::Store)?);
                    g.ops.push(lin(&gd.b, nv, h, hn, gproj.at(conv_dim + value_dim), LinMode::Store)?);
                    g.ops
                        .push(lin(&gd.a, nv, h, hn, gproj.at(conv_dim + value_dim + nv), LinMode::Store)?);
                    g.ops.push(Op::GdnConv {
                        x: gproj,
                        out: gconv,
                        state: st + 1,
                        w: &gd.conv_w,
                        dim: conv_dim,
                        kc: cfg.linear_conv_kernel_dim,
                    });
                    g.ops.push(Op::GdnRule(crate::gpu::decode::GdnRule {
                        c: gconv,
                        z: gproj.at(conv_dim),
                        b: gproj.at(conv_dim + value_dim),
                        a: gproj.at(conv_dim + value_dim + nv),
                        out: gout,
                        state: st,
                        nk,
                        nv,
                        hdk,
                        hdv,
                        tiled: cfg.v_heads_tiled,
                        decay_coef: &gd.decay_coef,
                        dt_bias: &gd.dt_bias,
                        norm_w: &gd.norm_w,
                        eps,
                        sigmoid_gate: true,
                    }));
                    g.ops.push(lin(&gd.out, h, value_dim, gout, y, LinMode::Store)?);
                }
                Mixer::Attn(at) => {
                    if at.bias.is_some() {
                        return None;
                    }
                    let ix = el.indexer.as_ref()?;
                    match &ix.k {
                        Some(k) => g.ops.push(lin(k, d, h, hn, cap.at(ai * d), LinMode::Store)?),
                        None => {
                            let nq = cfg.indexer_n_heads * d;
                            g.ops.push(lin(&ix.q, nq + d, h, hn, itmp, LinMode::Store)?);
                            g.ops.push(Op::Ew {
                                kind: Ew::Scale(1.0),
                                a: itmp.at(nq),
                                b: None,
                                out: cap.at(ai * d),
                                n: d,
                            });
                        }
                    }
                    ai += 1;
                    let slot = g.kv.len();
                    g.kv.push((nkv, hd));
                    g.ops.push(lin(&at.q, qw, h, hn, qkv, LinMode::Store)?);
                    g.ops.push(lin(&at.k, nkv * hd, h, hn, qkv.at(qw), LinMode::Store)?);
                    g.ops.push(lin(&at.v, nkv * hd, h, hn, qkv.at(qw + nkv * hd), LinMode::Store)?);
                    g.ops.push(Op::QkRope {
                        qkv,
                        q_out: qo,
                        kv: slot,
                        nh,
                        nkv,
                        hd,
                        half_rot: self.inv_freq.len(),
                        q_stride: 2 * hd,
                        k_off: qw,
                        v_off: qw + nkv * hd,
                        interleaved: false,
                        q_norm: at.q_norm.as_deref(),
                        k_norm: at.k_norm.as_deref(),
                        eps,
                        rope: 0,
                        v_norm: false,
                        v_from_k: false,
                    });
                    g.ops.push(Op::Attend {
                        q: qo,
                        out: att,
                        kv: slot,
                        nh,
                        nkv,
                        hd,
                        scale: (hd as f32).powf(-0.5),
                        window: None,
                        gate: Some((qkv.at(hd), 2 * hd)),
                        block: None,
                    });
                    g.ops.push(lin(&at.o, h, nh * hd, att, y, LinMode::Store)?);
                }
            }
            g.ops.push(Op::HcPost { x: xw, y, inj, n: h, hc });
            mix(&mut g, &el.mlp_hc, hn)?;
            let Ffn::Moe(m) = &layer.ffn else { return None };
            if m.experts.act != crate::moe::Act::Silu {
                return None;
            }
            let ne = m.experts.down.len();
            g.ops.push(Op::Route {
                x: hn,
                router: &m.router,
                gate: Some(&m.shared_gate),
                n_exp: ne,
                h,
                k: m.experts.top_k,
                logits: rlogits,
                sel,
                expert_scale: None,
            });
            let mut gw = Vec::with_capacity(ne);
            let mut uw = Vec::with_capacity(ne);
            for gu in &m.experts.gate_up {
                match gu {
                    crate::moe::GateUp::Fused(l) if !l.has_lora() && !l.is_transient() => gw.push(l.as_ref()),
                    crate::moe::GateUp::Split(a, b) if !a.has_lora() && !b.has_lora() && !a.is_transient() && !b.is_transient() => {
                        gw.push(a.as_ref());
                        uw.push(b.as_ref());
                    }
                    _ => return None,
                }
            }
            if m.experts.down.iter().any(|l| l.has_lora() || l.is_transient()) {
                return None;
            }
            g.ops.push(Op::Experts(Experts {
                gate: gw,
                up: (!uw.is_empty()).then_some(uw),
                down: m.experts.down.iter().map(|l| l.as_ref()).collect(),
                h,
                inter: m.experts.inter,
                k: m.experts.top_k,
                x: hn,
                sel,
                act: eact,
                y,
                gelu: false,
                add: false,
                edits: None,
            }));
            // The shared expert, sigmoid-gated, added to the routed experts' sum.
            let sh = &m.shared;
            let (sg, su) = (sh.gate.as_ref(), sh.up.as_ref());
            if crate::gpu::decode::pairable(&sg, &su) {
                g.ops.push(Op::Linear {
                    w: sg,
                    dev: dev(&sh.gate, sh.inter, h)?,
                    w2: Some((su, dev(&sh.up, sh.inter, h)?)),
                    n_out: sh.inter,
                    n_in: h,
                    x: hn,
                    y: act,
                    mode: LinMode::SiluPair,
                });
            } else {
                g.ops.push(lin(&sh.gate, sh.inter, h, hn, act2, LinMode::Store)?);
                g.ops.push(lin(&sh.up, sh.inter, h, hn, act, LinMode::Store)?);
                g.ops.push(Op::Glu {
                    g: act2,
                    u: act,
                    out: act,
                    n: sh.inter,
                    gelu: false,
                });
            }
            g.ops.push(lin(&sh.down, h, sh.inter, act, y, LinMode::AddGated(rlogits.at(ne)))?);
            g.ops.push(Op::HcPost { x: xw, y, inj, n: h, hc });
        }
        g.ops.push(Op::Ew {
            kind: Ew::Scale(1.0),
            a: xw,
            b: None,
            out: cap.at(n_attn * d),
            n: w,
        });
        mix(&mut g, &ex.mixer, hidden)?;
        g.head_from = g.ops.len();
        g.ops.push(lin(&self.lm_head, cfg.vocab_size, h, hidden, logits, LinMode::Store)?);
        g.logits = Some((logits, cfg.vocab_size));
        Some(g)
    }

    /// Runs token `tok` through the Qwen4-Exp GPU decode plan (see [`Self::exp_gpu_graph`]): the
    /// mixed hidden state and, if wanted, the logits. None (nothing changed) when unavailable,
    /// or past the dense-attention limit.
    pub(crate) fn exp_gpu_step(
        &self,
        ex: &Exp<'a>,
        sess: &mut QwenSession,
        tok: u32,
        want_logits: bool,
    ) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
        use crate::qwen::Mixer;
        let cfg = &self.cfg;
        if sess.pos + 1 > dense_limit(cfg) || sess.iv.active() || !sess.spans.is_empty() || sess.positions.is_some() {
            return None;
        }
        let Some(plan) = self.gpu_plan() else {
            if std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                eprintln!("[no gpu decode plan]");
            }
            return None;
        };
        let h = cfg.hidden_size;
        let d = cfg.indexer_head_dim;
        let mut x = vec![0f32; h];
        self.embed.row(tok as usize, &mut x);
        let (mut cs, sin) = self.rope_cs(&sess.spans, sess.pos, 1);
        cs.extend_from_slice(&sin);
        let pos = sess.pos;
        let es = sess.exp.as_mut()?;
        es.toks.truncate(pos);
        es.toks.push(tok);
        let mut x2 = Vec::new();
        if let Some(p) = &ex.ple {
            let mut rows = Vec::new();
            p.rows(&es.toks, pos, &mut rows);
            x2 = vec![0f32; p.e];
            for (i, &r) in rows.iter().enumerate() {
                p.table.row(r, &mut x2[i * p.table.dim..(i + 1) * p.table.dim]);
            }
        }
        let n_attn = self.layers.iter().filter(|l| matches!(l.mixer, Mixer::Attn(_))).count();
        let w = cfg.hc_count * h;
        let mut cap = vec![0f32; n_attn * d + w];
        let mut hidden = vec![0f32; h];
        let mut logits = want_logits.then(|| vec![0f32; cfg.vocab_size]);
        let ok = {
            let _timer = crate::prof::start(crate::prof::Cat::Matmul);
            let es = sess.exp.as_mut().unwrap();
            let ple_layer = ex.ple.as_ref().map(|p| p.layer);
            let mut states: Vec<&mut [f32]> = Vec::new();
            let mut ple_conv = Some(&mut es.ple_conv[..]);
            for (l, g) in sess.gdn.iter_mut().enumerate() {
                if ple_layer == Some(l) {
                    states.push(ple_conv.take()?);
                }
                if let Some(g) = g {
                    states.push(&mut g.s[..]);
                    states.push(&mut g.conv[..]);
                }
            }
            let mut kv: Vec<&mut LayerCache> = sess.kv.iter_mut().flatten().collect();
            plan.step(
                &mut sess.gpu,
                crate::gpu::decode::Step {
                    pos,
                    t: 1,
                    rope_cs: &cs,
                    x: &x,
                    x2: &x2,
                    kv: &mut kv,
                    states: &mut states,
                    hidden: &mut hidden,
                    logits: logits.as_deref_mut(),
                    params: None,
                    capture: Some(&mut cap),
                    scratch: false,
                    states_kept: false,
                },
            )
        };
        let es = sess.exp.as_mut().unwrap();
        if !ok {
            if std::env::var_os("OXINFER_GPU_PLAN_DEBUG").is_some() {
                eprintln!("[gpu decode step failed at {pos}]");
            }
            es.toks.truncate(pos);
            return None;
        }
        for (ai, c) in es.idx.iter_mut().flatten().enumerate() {
            c.raw.truncate(pos * d);
            c.raw.extend_from_slice(&cap[ai * d..(ai + 1) * d]);
            c.pooled.truncate(pos / es.ratio * d);
        }
        es.last_wide = cap[n_attn * d..].to_vec();
        sess.rec_span = None;
        sess.pos += 1;
        Some((hidden, logits))
    }
}

/// The Qwen4-Exp parts of the MTP layer: per stream `s`, `x_s = fc([enorm(e) ; hnorm_s(h_s)])` from
/// the trunk's residual streams `h` (pre-mixer) and the next token's embedding `e`, then one
/// QSA-attention + MoE layer with hyper-connections; the next draft step reads its streams, its
/// own final mixer gives the hidden state of the (shared) LM head.
pub struct MtpExp<'a> {
    /// `fc_hidden` (`[H, H]`) when the checkpoint splits the projection (then the layer's `fc` is
    /// `fc_embedding`), None for GGUF's `[H, 2H]` `eh_proj`.
    pub(crate) fc_hidden: Option<Lin<'a>>,
    pub(crate) attn_hc: Hc<'a>,
    pub(crate) mlp_hc: Hc<'a>,
    pub(crate) mixer: Hc<'a>,
    pub(crate) indexer: Indexer<'a>,
}

impl<'a> MtpExp<'a> {
    /// Loads the parts under `p` (`mtp.`); `split`: the checkpoint has `fc_embedding` and
    /// `fc_hidden` rather than one `fc`.
    pub(crate) fn load(ld: &Loader<'a, '_>, p: &str, split: bool) -> Result<Self> {
        let h = ld.cfg.hidden_size;
        let lp = format!("{p}layers.0.");
        Ok(MtpExp {
            fc_hidden: if split {
                Some(ld.lin(&format!("{p}fc_hidden.weight"), h, h)?)
            } else {
                None
            },
            attn_hc: Hc::load(ld, &format!("{lp}attn_hyper_connection."), true)?,
            mlp_hc: Hc::load(ld, &format!("{lp}mlp_hyper_connection."), true)?,
            mixer: Hc::load(ld, &format!("{p}hyper_connection_mixer."), false)?,
            indexer: load_indexer(ld, &format!("{lp}self_attn.indexer."))?,
        })
    }

    pub(crate) fn lins(&self) -> Vec<&Lin<'a>> {
        let mut v: Vec<&Lin> = self.fc_hidden.iter().collect();
        v.extend(self.attn_hc.lins());
        v.extend(self.mlp_hc.lins());
        v.extend(self.mixer.lins());
        v.extend(self.indexer.lins());
        v
    }

    /// The LM head's input (`[t, H]`) for MTP output streams `x` (`[t, hc * H]`).
    pub(crate) fn mix_out(&self, cfg: &QwenConfig, x: &[f32], t: usize) -> Vec<f32> {
        let mut out = vec![0f32; t * cfg.hidden_size];
        self.mixer.mix(cfg, x, t, &mut out, &mut []);
        out
    }
}

/// The MTP layer's indexer cache (Qwen4-Exp).
pub fn new_idx_cache(cfg: &QwenConfig) -> IdxCache {
    IdxCache {
        d: cfg.indexer_head_dim,
        ..Default::default()
    }
}

impl IdxCache {
    /// Drops the keys of positions from `len` on.
    pub fn truncate_to(&mut self, len: usize, ratio: usize) {
        self.truncate(len, ratio.max(1));
    }
}

impl QwenModel<'_> {
    /// Floats per position of the hidden state the MTP head reads: the residual streams for
    /// Qwen4-Exp (`hc * H`), else the final-normed hidden state (`H`).
    pub fn mtp_width(&self) -> usize {
        if self.cfg.is_exp() {
            self.cfg.hc_count * self.cfg.hidden_size
        } else {
            self.cfg.hidden_size
        }
    }

    /// The MTP head's inputs for the positions of the last forward: `hs` (its hidden states), or
    /// for Qwen4-Exp the residual streams it kept.
    pub fn mtp_hidden<'s>(&self, sess: &'s QwenSession, hs: &'s [f32]) -> &'s [f32] {
        match &sess.exp {
            Some(e) if self.cfg.is_exp() => &e.last_wide,
            _ => hs,
        }
    }
}

/// Bytes of the n-gram embedding table in the checkpoint at `path` (a GGUF file or the first part
/// of a split one; 0 when there is none or it cannot be told): read a few rows per token, it never
/// needs to fit in memory.
pub fn ngram_table_bytes(path: &std::path::Path) -> u64 {
    if !crate::gguf::is_gguf(path) {
        return 0;
    }
    let Ok(parts) = crate::gguf::split_files(path) else { return 0 };
    parts
        .iter()
        .filter_map(|p| crate::gguf::read_header(p).ok())
        .flat_map(|h| h.tensors.into_iter())
        .filter(|t| t.name == "per_layer_token_embd.weight")
        .map(|t| (t.end - t.start) as u64)
        .sum()
}

/// Whether a Qwen4-Exp checkpoint is too big to hold in memory: its weights other than the n-gram
/// table over 75% of RAM (the table is read from storage as used).
pub fn too_big(path: &std::path::Path) -> bool {
    let bytes = crate::safetensors::checkpoint_bytes(path).saturating_sub(ngram_table_bytes(path));
    crate::platform::total_memory().is_some_and(|m| bytes as f64 > 0.75 * m as f64)
}

impl QwenSession {
    /// The recurrent states in GPU plan order: per layer, the n-gram block's conv state (Qwen4-Exp,
    /// before its layer) then the Gated DeltaNet rule and conv states.
    pub(crate) fn recurrent_states(&mut self) -> Vec<&mut [f32]> {
        let QwenSession { gdn, exp, .. } = self;
        recurrent_states_of(gdn, exp.as_deref_mut())
    }
}

pub(crate) fn recurrent_states_of<'s>(gdn: &'s mut [Option<crate::qwen::GdnState>], exp: Option<&'s mut ExpSession>) -> Vec<&'s mut [f32]> {
    {
        let ple = exp.as_ref().and_then(|e| e.ple_layer);
        let mut ple_conv = exp.filter(|_| ple.is_some()).map(|e| &mut e.ple_conv[..]);
        let mut v: Vec<&mut [f32]> = Vec::new();
        for (l, g) in gdn.iter_mut().enumerate() {
            if ple == Some(l)
                && let Some(p) = ple_conv.take()
            {
                v.push(p);
            }
            if let Some(g) = g {
                v.push(&mut g.s[..]);
                v.push(&mut g.conv[..]);
            }
        }
        v
    }
}

impl<'a> QwenModel<'a> {
    /// The n-gram embedding rows (`[t, e]`) of the last `t` tokens of `toks`.
    fn ngram_rows(&self, ex: &Exp<'a>, toks: &[u32], t: usize) -> Vec<f32> {
        let Some(p) = &ex.ple else { return Vec::new() };
        let mut out = vec![0f32; t * p.e];
        let mut rows = Vec::new();
        for j in 0..t {
            rows.clear();
            p.rows(toks, toks.len() - t + j, &mut rows);
            for (i, &r) in rows.iter().enumerate() {
                let o = j * p.e + i * p.table.dim;
                p.table.row(r, &mut out[o..o + p.table.dim]);
            }
        }
        out
    }

    /// A speculative verification of `tokens` on the multi-token GPU plan (see
    /// `QwenModel::gpu_verify_hidden`): the mixed hidden states and the logits, the session
    /// advanced (indexer keys, n-gram contexts and the MTP head's residual streams included) and
    /// what a rewind inside the pass needs recorded.
    pub(crate) fn exp_gpu_pass(&self, ex: &Exp<'a>, sess: &mut QwenSession, tokens: &[u32], fast: bool) -> Option<(Vec<f32>, Vec<f32>)> {
        use crate::qwen::Mixer;
        let cfg = &self.cfg;
        let t = tokens.len();
        if sess.pos + t > dense_limit(cfg) || !sess.spans.is_empty() {
            return None;
        }
        let plan = self.gpu_multi_plan_pub(t, fast)?;
        let (h, d) = (cfg.hidden_size, cfg.indexer_head_dim);
        let w = cfg.hc_count * h;
        let pos = sess.pos;
        let mut x = vec![0f32; t * h];
        for (j, &tok) in tokens.iter().enumerate() {
            self.embed.row(tok as usize, &mut x[j * h..(j + 1) * h]);
        }
        let half = self.inv_freq.len();
        let (cos, sin) = self.rope_cs(&sess.spans, pos, t);
        let mut cs = Vec::with_capacity(2 * t * half);
        for j in 0..t {
            cs.extend_from_slice(&cos[j * half..(j + 1) * half]);
            cs.extend_from_slice(&sin[j * half..(j + 1) * half]);
        }
        let x2 = {
            let es = sess.exp.as_mut()?;
            es.toks.truncate(pos);
            es.toks.extend_from_slice(tokens);
            self.ngram_rows(ex, &es.toks, t)
        };
        let n_attn = self.layers.iter().filter(|l| matches!(l.mixer, Mixer::Attn(_))).count();
        let nc = n_attn * d + w;
        let mut captured = vec![0f32; t * nc];
        let mut hidden = vec![0f32; t * h];
        let mut logits = vec![0f32; t * cfg.vocab_size];
        let mut snap = sess.gpu_rec.take().map(|r| r.states).unwrap_or_default();
        if t > 1 {
            let src = sess.recurrent_states();
            snap.resize_with(src.len(), Vec::new);
            for (dst, s) in snap.iter_mut().zip(&src) {
                dst.clear();
                dst.extend_from_slice(s);
            }
        }
        let ok = {
            let _timer = crate::prof::start(crate::prof::Cat::Matmul);
            let QwenSession { kv, gpu, gdn, exp, .. } = sess;
            let mut kv: Vec<&mut LayerCache> = kv.iter_mut().flatten().collect();
            let mut states = recurrent_states_of(gdn, exp.as_deref_mut());
            plan.step(
                gpu,
                crate::gpu::decode::Step {
                    pos,
                    t,
                    rope_cs: &cs,
                    x: &x,
                    x2: &x2,
                    kv: &mut kv,
                    states: &mut states,
                    hidden: &mut hidden,
                    logits: Some(&mut logits),
                    params: None,
                    capture: Some(&mut captured),
                    scratch: false,
                    states_kept: false,
                },
            )
        };
        let es = sess.exp.as_mut()?;
        if !ok {
            es.toks.truncate(pos);
            return None;
        }
        for (ai, c) in es.idx.iter_mut().flatten().enumerate() {
            c.raw.truncate(pos * d);
            for j in 0..t {
                c.raw.extend_from_slice(&captured[j * nc + ai * d..j * nc + (ai + 1) * d]);
            }
            c.pooled.truncate(pos / es.ratio * d);
        }
        es.last_wide.clear();
        for j in 0..t {
            es.last_wide.extend_from_slice(&captured[j * nc + n_attn * d..(j + 1) * nc]);
        }
        sess.rec_span = None;
        sess.gpu_rec = (t > 1).then_some(crate::qwen::GpuRec {
            plan,
            start: pos,
            len: t,
            states: snap,
        });
        sess.pos += t;
        Some((hidden, logits))
    }
}
