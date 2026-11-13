//! The language model of Audio8 TTS (`ArkttsModel`, a port of `modeling_arktts.py` of
//! Edge0/Audio8-TTS-Realistic-Air), a "DualAR" pair of transformers:
//!
//! - the slow one reads one row per position (a text token, or an audio frame: its semantic
//!   token plus the embeddings of the frame's ten codes) and predicts the next frame's semantic
//!   token, whose logits are the tied embedding rows of the semantic tokens and of the end token;
//! - the fast one runs once per frame over at most ten positions: the slow model's normalized
//!   hidden state, then the embedding of each code of the frame, predicting the next code.
//!
//! Both are pre-norm blocks with RMS norms, one matrix for queries, keys and values (fewer key /
//! value heads than query heads), RoPE on pairs of neighbours with tables rounded to bf16 as the
//! release computes them, and SwiGLU. Weights stay in the checkpoint's bf16; the arithmetic is
//! f32.

use super::codec::{apply_rope, rope_table};
use crate::kernels::{bf16_to_f32, dot, dot_bf16, matmul, matmul_multi};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;

fn d_true() -> bool {
    true
}
fn d_eps() -> f32 {
    1e-6
}
fn d_ras_window() -> usize {
    10
}
fn d_ras_temp() -> f32 {
    1.0
}
fn d_ras_top_p() -> f32 {
    0.9
}
fn d_codec_file() -> String {
    "codec.pth".into()
}
fn d_rate() -> u32 {
    44100
}
fn d_frame() -> usize {
    2048
}

/// `config.json` of an `arktts` model.
#[derive(Clone, Debug, Deserialize)]
pub struct LmConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    #[serde(default = "d_eps")]
    pub norm_eps: f32,
    #[serde(default = "d_true")]
    pub attention_qkv_bias: bool,
    #[serde(default)]
    pub attention_qk_norm: bool,
    #[serde(default)]
    pub attention_o_bias: bool,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub semantic_begin_id: u32,
    pub semantic_end_id: u32,
    pub n_fast_layer: usize,
    pub fast_dim: usize,
    pub fast_n_head: usize,
    pub fast_n_local_heads: usize,
    pub fast_head_dim: usize,
    pub fast_intermediate_size: usize,
    #[serde(default)]
    pub fast_attention_qkv_bias: bool,
    #[serde(default)]
    pub fast_attention_qk_norm: bool,
    #[serde(default)]
    pub fast_attention_o_bias: bool,
    #[serde(default = "d_true")]
    pub norm_fastlayer_input: bool,
    pub eos_token_id: u32,
    #[serde(default = "d_codec_file")]
    pub codec_filename: String,
    #[serde(default = "d_rate")]
    pub codec_sample_rate: u32,
    #[serde(default = "d_frame")]
    pub codec_frame_size: usize,
    #[serde(default = "d_ras_window")]
    pub ras_window_size: usize,
    #[serde(default = "d_ras_temp")]
    pub ras_temperature: f32,
    #[serde(default = "d_ras_top_p")]
    pub ras_top_p: f32,
}

struct Block {
    attn_norm: Vec<f32>,
    wqkv: Vec<u16>,
    qkv_bias: Option<Vec<f32>>,
    wo: Vec<u16>,
    ffn_norm: Vec<f32>,
    w1: Vec<u16>,
    w3: Vec<u16>,
    w2: Vec<u16>,
}

/// The shape of one of the two transformers.
#[derive(Clone, Copy)]
struct Shape {
    dim: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
}

pub struct Lm {
    pub cfg: LmConfig,
    pub weight_bytes: usize,
    emb: Vec<u16>,
    code_emb: Vec<u16>,
    layers: Vec<Block>,
    norm: Vec<f32>,
    fast_emb: Vec<u16>,
    fast_layers: Vec<Block>,
    fast_norm: Vec<f32>,
    fast_out: Vec<u16>,
    rope: Vec<f32>,
    fast_rope: Vec<f32>,
}

/// The keys and values of every layer of one transformer (`[positions, kv_heads * head_dim]`).
#[derive(Clone, Default)]
pub struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
}

impl Cache {
    fn new(layers: usize) -> Self {
        Self {
            k: vec![Vec::new(); layers],
            v: vec![Vec::new(); layers],
            len: 0,
        }
    }
    fn clear(&mut self) {
        self.k.iter_mut().for_each(Vec::clear);
        self.v.iter_mut().for_each(Vec::clear);
        self.len = 0;
    }
}

fn rms(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps;
    let s = 1.0 / ss.sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(w) {
        *o = v * s * w;
    }
}

fn rms_rows(x: &[f32], n: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    crate::kernels::for_chunks(&mut out, n, x.len() > 8 * n, |i, o| rms(&x[i * n..(i + 1) * n], w, eps, o));
    out
}

impl Lm {
    pub fn load(files: &ModelFiles, cfg: LmConfig) -> Result<Self> {
        let c = &cfg;
        if c.attention_qk_norm || c.fast_attention_qk_norm {
            bail!("arktts: normalized queries and keys (attention_qk_norm) are not supported");
        }
        if c.fast_dim != c.dim {
            bail!(
                "arktts: a fast transformer of another width ({} / {}) is not supported",
                c.fast_dim,
                c.dim
            );
        }
        if c.attention_o_bias || c.fast_attention_o_bias {
            bail!("arktts: a bias on the attention output is not supported");
        }
        if c.n_head * c.head_dim != c.dim
            || c.fast_n_head * c.fast_head_dim != c.fast_dim
            || !c.n_head.is_multiple_of(c.n_local_heads.max(1))
            || !c.fast_n_head.is_multiple_of(c.fast_n_local_heads.max(1))
            || c.n_local_heads == 0
            || c.fast_n_local_heads == 0
            || !c.head_dim.is_multiple_of(2)
            || !c.fast_head_dim.is_multiple_of(2)
        {
            bail!("arktts: inconsistent attention shape in config.json");
        }
        if c.semantic_end_id + 1 - c.semantic_begin_id != c.codebook_size as u32 || c.num_codebooks == 0 {
            bail!("arktts: the semantic tokens do not span one codebook");
        }
        let mut bytes = 0usize;
        let mut mat = |name: &str, rows: usize, cols: usize| -> Result<Vec<u16>> {
            let info = files.info(name).with_context(|| format!("missing tensor {name}"))?;
            if info.shape[..] != [rows, cols] {
                bail!("tensor {name} has shape {:?}, expected [{rows}, {cols}]", info.shape);
            }
            bytes += rows * cols * 2;
            Ok(files.bf16(name)?.to_vec())
        };
        let vec = |name: &str, n: usize| -> Result<Vec<f32>> {
            let v = files.f32(name)?;
            if v.len() != n {
                bail!("tensor {name} has {} values, expected {n}", v.len());
            }
            Ok(v)
        };
        let mut block = |p: &str, s: Shape, bias: bool| -> Result<Block> {
            let qkv = (s.heads + 2 * s.kv_heads) * s.head_dim;
            Ok(Block {
                attn_norm: vec(&format!("{p}.attention_norm.weight"), s.dim)?,
                wqkv: mat(&format!("{p}.attention.wqkv.weight"), qkv, s.dim)?,
                qkv_bias: if bias {
                    Some(vec(&format!("{p}.attention.wqkv.bias"), qkv)?)
                } else {
                    None
                },
                wo: mat(&format!("{p}.attention.wo.weight"), s.dim, s.heads * s.head_dim)?,
                ffn_norm: vec(&format!("{p}.ffn_norm.weight"), s.dim)?,
                w1: mat(&format!("{p}.feed_forward.w1.weight"), s.inter, s.dim)?,
                w3: mat(&format!("{p}.feed_forward.w3.weight"), s.inter, s.dim)?,
                w2: mat(&format!("{p}.feed_forward.w2.weight"), s.dim, s.inter)?,
            })
        };
        let (slow, fast) = (Self::shape(c, false), Self::shape(c, true));
        let layers = (0..c.n_layer)
            .map(|i| block(&format!("layers.{i}"), slow, c.attention_qkv_bias))
            .collect::<Result<Vec<_>>>()?;
        let fast_layers = (0..c.n_fast_layer)
            .map(|i| block(&format!("fast_layers.{i}"), fast, c.fast_attention_qkv_bias))
            .collect::<Result<Vec<_>>>()?;
        let emb = mat("embeddings.weight", c.vocab_size, c.dim)?;
        let code_emb = mat("codebook_embeddings.weight", c.codebook_size * c.num_codebooks, c.dim)?;
        let fast_emb = mat("fast_embeddings.weight", c.codebook_size, c.fast_dim)?;
        let fast_out = mat("fast_output.weight", c.codebook_size, c.fast_dim)?;
        Ok(Self {
            norm: vec("norm.weight", c.dim)?,
            fast_norm: vec("fast_norm.weight", c.fast_dim)?,
            rope: rope_table(c.max_seq_len, c.head_dim, c.rope_base),
            fast_rope: rope_table(c.num_codebooks, c.fast_head_dim, c.rope_base),
            weight_bytes: bytes,
            emb,
            code_emb,
            layers,
            fast_emb,
            fast_layers,
            fast_out,
            cfg,
        })
    }

    fn shape(c: &LmConfig, fast: bool) -> Shape {
        if fast {
            Shape {
                dim: c.fast_dim,
                heads: c.fast_n_head,
                kv_heads: c.fast_n_local_heads,
                head_dim: c.fast_head_dim,
                inter: c.fast_intermediate_size,
                eps: c.norm_eps,
            }
        } else {
            Shape {
                dim: c.dim,
                heads: c.n_head,
                kv_heads: c.n_local_heads,
                head_dim: c.head_dim,
                inter: c.intermediate_size,
                eps: c.norm_eps,
            }
        }
    }

    /// The RoPE table of the slow transformer (`[max_seq_len, head_dim / 2, 2]`).
    pub fn rope(&self) -> &[f32] {
        &self.rope
    }

    pub fn new_cache(&self) -> Cache {
        Cache::new(self.layers.len())
    }

    pub fn new_fast_cache(&self) -> Cache {
        Cache::new(self.fast_layers.len())
    }

    /// One block on `t` new positions of `x` (`[t, dim]`), which follow the `cache.len` cached.
    #[allow(clippy::too_many_arguments)]
    fn block(&self, b: &Block, s: Shape, rope: &[f32], x: &mut [f32], t: usize, pos0: usize, k: &mut Vec<f32>, v: &mut Vec<f32>) {
        let (d, hd) = (s.dim, s.head_dim);
        let (nq, nkv) = (s.heads * hd, s.kv_heads * hd);
        let tot = nq + 2 * nkv;
        let h = rms_rows(x, d, &b.attn_norm, s.eps);
        let mut qkv = vec![0f32; t * tot];
        matmul(&b.wqkv, tot, d, &h, t, &mut qkv);
        for (i, row) in qkv.chunks_mut(tot).enumerate() {
            if let Some(bias) = &b.qkv_bias {
                for (v, &bb) in row.iter_mut().zip(bias) {
                    *v += bb;
                }
            }
            apply_rope(&mut row[..nq + nkv], &rope[(pos0 + i) * hd..(pos0 + i + 1) * hd], hd);
            k.extend_from_slice(&row[nq..nq + nkv]);
            v.extend_from_slice(&row[nq + nkv..]);
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let rep = s.heads / s.kv_heads;
        let mut att = vec![0f32; t * nq];
        {
            let (k, v, qkv) = (&*k, &*v, &qkv);
            crate::kernels::for_chunks(&mut att, nq, t > 4, |i, out| {
                let n = pos0 + i + 1;
                let mut p = vec![0f32; n];
                for (hq, o) in out.chunks_mut(hd).enumerate() {
                    let hk = hq / rep;
                    let q = &qkv[i * tot + hq * hd..i * tot + (hq + 1) * hd];
                    let mut max = f32::NEG_INFINITY;
                    for (j, pv) in p.iter_mut().enumerate() {
                        *pv = dot(q, &k[j * nkv + hk * hd..j * nkv + (hk + 1) * hd]) * scale;
                        max = max.max(*pv);
                    }
                    let mut sum = 0f32;
                    for pv in p.iter_mut() {
                        *pv = (*pv - max).exp();
                        sum += *pv;
                    }
                    for (j, &pv) in p.iter().enumerate() {
                        crate::kernels::axpy(o, pv / sum, &v[j * nkv + hk * hd..j * nkv + (hk + 1) * hd]);
                    }
                }
            });
        }
        let mut o = vec![0f32; t * d];
        matmul(&b.wo, d, nq, &att, t, &mut o);
        for (x, o) in x.iter_mut().zip(&o) {
            *x += o;
        }
        let h = rms_rows(x, d, &b.ffn_norm, s.eps);
        let n = s.inter;
        let (mut a, mut g) = (vec![0f32; t * n], vec![0f32; t * n]);
        matmul_multi(&mut [(&b.w1[..], n, &mut a[..]), (&b.w3[..], n, &mut g[..])], d, &h, t);
        for (a, &g) in a.iter_mut().zip(&g) {
            *a = *a / (1.0 + (-*a).exp()) * g;
        }
        matmul(&b.w2, d, n, &a, t, &mut o);
        for (x, o) in x.iter_mut().zip(&o) {
            *x += o;
        }
    }

    /// The embeddings of `t` rows of `1 + num_codebooks` ids (the token, then the frame's codes,
    /// which count only under a semantic token).
    fn embed(&self, rows: &[u32], t: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (d, w) = (c.dim, c.num_codebooks + 1);
        let mut x = vec![0f32; t * d];
        for (row, out) in rows.chunks(w).zip(x.chunks_mut(d)) {
            let tok = row[0] as usize;
            if tok >= c.vocab_size {
                bail!("token id {tok} outside the vocabulary of {}", c.vocab_size);
            }
            if (c.semantic_begin_id..=c.semantic_end_id).contains(&row[0]) {
                // The codes' embeddings are summed first, then added to the token's.
                for (j, &code) in row[1..].iter().enumerate() {
                    if code as usize >= c.codebook_size {
                        bail!("code {code} outside the codebook of {}", c.codebook_size);
                    }
                    let e = &self.code_emb[(j * c.codebook_size + code as usize) * d..][..d];
                    for (o, &v) in out.iter_mut().zip(e) {
                        *o += bf16_to_f32(v);
                    }
                }
            }
            for (o, &v) in out.iter_mut().zip(&self.emb[tok * d..(tok + 1) * d]) {
                *o += bf16_to_f32(v);
            }
        }
        Ok(x)
    }

    /// Feeds `t` rows (see [`Lm::embed`]) to the slow transformer. Returns the logits of the next
    /// semantic token (`codebook_size` values, then the end token's) and the normalized hidden
    /// state of the last row, which the fast transformer starts from.
    pub fn slow(&self, cache: &mut Cache, rows: &[u32], t: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.cfg;
        if t == 0 || rows.len() != t * (c.num_codebooks + 1) {
            bail!("{} ids are not {t} rows of {}", rows.len(), c.num_codebooks + 1);
        }
        if cache.len + t > c.max_seq_len {
            bail!("{} positions exceed the model context of {}", cache.len + t, c.max_seq_len);
        }
        let s = Self::shape(c, false);
        let mut x = self.embed(rows, t)?;
        for (i, b) in self.layers.iter().enumerate() {
            self.block(b, s, &self.rope, &mut x, t, cache.len, &mut cache.k[i], &mut cache.v[i]);
        }
        cache.len += t;
        let d = c.dim;
        let mut hn = vec![0f32; d];
        rms(&x[(t - 1) * d..], &self.norm, c.norm_eps, &mut hn);
        let cs = c.codebook_size;
        let mut logits = vec![0f32; cs + 1];
        let begin = c.semantic_begin_id as usize;
        matmul(&self.emb[begin * d..(begin + cs) * d], cs, d, &hn, 1, &mut logits[..cs]);
        let eos = c.eos_token_id as usize;
        logits[cs] = dot_bf16(&self.emb[eos * d..(eos + 1) * d], &hn);
        if !c.norm_fastlayer_input {
            hn.copy_from_slice(&x[(t - 1) * d..]);
        }
        Ok((logits, hn))
    }

    /// Starts a frame in the fast transformer: position 0 is the slow model's state.
    pub fn fast_begin(&self, cache: &mut Cache, hidden: &[f32]) {
        cache.clear();
        let mut x = hidden.to_vec();
        self.fast_blocks(cache, &mut x);
    }

    /// Feeds one code of the frame (codebook `cache.len - 1`) and returns the logits of the next
    /// codebook's code (`codebook_size` values).
    pub fn fast_step(&self, cache: &mut Cache, code: u32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if cache.len == 0 || cache.len >= c.num_codebooks {
            bail!("the fast transformer is at position {} of {}", cache.len, c.num_codebooks);
        }
        if code as usize >= c.codebook_size {
            bail!("code {code} outside the codebook of {}", c.codebook_size);
        }
        let d = c.fast_dim;
        let mut x: Vec<f32> = self.fast_emb[code as usize * d..(code as usize + 1) * d]
            .iter()
            .map(|&v| bf16_to_f32(v))
            .collect();
        self.fast_blocks(cache, &mut x);
        let mut hn = vec![0f32; d];
        rms(&x, &self.fast_norm, c.norm_eps, &mut hn);
        let mut logits = vec![0f32; c.codebook_size];
        matmul(&self.fast_out, c.codebook_size, d, &hn, 1, &mut logits);
        Ok(logits)
    }

    fn fast_blocks(&self, cache: &mut Cache, x: &mut [f32]) {
        let s = Self::shape(&self.cfg, true);
        for (i, b) in self.fast_layers.iter().enumerate() {
            self.block(b, s, &self.fast_rope, x, 1, cache.len, &mut cache.k[i], &mut cache.v[i]);
        }
        cache.len += 1;
    }
}
