//! The T5 v1.1 encoder (transformers `T5EncoderModel`), FLUX's main text encoder (T5-XXL:
//! 24 blocks, width 4096, 64 heads of 64, gated-GELU feed-forward of 10240): RMS norms, no
//! attention scaling, and a learned relative-position bias (32 buckets, max distance 128)
//! computed once from the first block and shared by all. The tokenizer is the SentencePiece
//! `tokenizer.json` through `crate::tokenizer`; prompts are padded (`<pad>`, attended like
//! any token, as diffusers runs it) to a fixed length.

use super::weights::{Dense, Loader, forward_many};
use crate::kernels::{add_inplace, for_chunks, gelu_tanh};
use crate::model::Quant;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

struct Block<'a> {
    ln1: Vec<f32>,
    qkv: Vec<Dense<'a>>,
    o: Dense<'a>,
    ln2: Vec<f32>,
    /// wi_0 (GELU gate), wi_1.
    wi: Vec<Dense<'a>>,
    wo: Dense<'a>,
}

pub struct T5Encoder {
    pub d_model: usize,
    pub heads: usize,
    pub d_kv: usize,
    eps: f32,
    num_buckets: usize,
    max_distance: usize,
    /// `[num_buckets, heads]`.
    rel_bias: Vec<f32>,
    final_ln: Vec<f32>,
    blocks: Vec<Block<'static>>,
    tokenizer: crate::tokenizer::Tokenizer,
    pub pad_id: u32,
    pub eos_id: u32,
    embed: &'static str,
    embed_dtype: String,
    pub weight_bytes: usize,
    // Declared last: dropped after every layer borrowing it.
    files: Box<ModelFiles>,
}

impl T5Encoder {
    /// `dir` is the `text_encoder_2/` folder (config + weights) or a llama.cpp `t5encoder` GGUF
    /// file (e.g. city96/t5-v1_1-xxl-encoder-gguf), `tok_dir` the tokenizer folder (`tokenizer.json`).
    pub fn load(dir: &Path, tok_dir: &Path, quant: Quant) -> Result<Self> {
        let cfg: serde_json::Value = if crate::gguf::is_gguf(dir) {
            gguf_config(dir)?
        } else {
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?
        };
        let u = |k: &str| cfg[k].as_u64().map(|v| v as usize).with_context(|| format!("T5 config: {k}"));
        let (d_model, heads, d_kv, n_layers) = (u("d_model")?, u("num_heads")?, u("d_kv")?, u("num_layers")?);
        if cfg["feed_forward_proj"].as_str() != Some("gated-gelu") {
            bail!("T5 feed_forward_proj {} is not supported (gated-gelu is)", cfg["feed_forward_proj"]);
        }
        let files = Box::new(ModelFiles::open(dir, cfg!(target_arch = "wasm32"))?);
        // SAFETY: as in FluxTransformer::load (field order drops the layers first).
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let ld = Loader { files: fs, quant };
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("encoder.block.{i}.layer");
            blocks.push(Block {
                ln1: ld.f32(&format!("{p}.0.layer_norm.weight"))?,
                qkv: ["q", "k", "v"]
                    .iter()
                    .map(|x| ld.dense(&format!("{p}.0.SelfAttention.{x}"), true))
                    .collect::<Result<_>>()?,
                o: ld.dense(&format!("{p}.0.SelfAttention.o"), true)?,
                ln2: ld.f32(&format!("{p}.1.layer_norm.weight"))?,
                wi: ["wi_0", "wi_1"]
                    .iter()
                    .map(|x| ld.dense(&format!("{p}.1.DenseReluDense.{x}"), true))
                    .collect::<Result<_>>()?,
                wo: ld.dense(&format!("{p}.1.DenseReluDense.wo"), true)?,
            });
        }
        let embed = ["shared.weight", "encoder.embed_tokens.weight"]
            .into_iter()
            .find(|n| fs.has(n))
            .context("T5: no token embedding (shared.weight)")?;
        let embed_dtype = fs.info(embed).unwrap().dtype.clone();
        let tokenizer =
            crate::tokenizer::Tokenizer::from_file(tok_dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("T5 tokenizer: {e}"))?;
        let pad_id = cfg["pad_token_id"].as_u64().unwrap_or(0) as u32;
        let eos_id = cfg["eos_token_id"].as_u64().unwrap_or(1) as u32;
        let weight_bytes = blocks
            .iter()
            .flat_map(|b| b.qkv.iter().chain(&b.wi).chain([&b.o, &b.wo]))
            .map(|l| l.bytes())
            .sum();
        Ok(Self {
            d_model,
            heads,
            d_kv,
            eps: cfg["layer_norm_epsilon"].as_f64().unwrap_or(1e-6) as f32,
            num_buckets: cfg["relative_attention_num_buckets"].as_u64().unwrap_or(32) as usize,
            max_distance: cfg["relative_attention_max_distance"].as_u64().unwrap_or(128) as usize,
            rel_bias: ld.f32("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?,
            final_ln: ld.f32("encoder.final_layer_norm.weight")?,
            blocks,
            tokenizer,
            pad_id,
            eos_id,
            embed,
            embed_dtype,
            weight_bytes,
            files,
        })
    }

    /// Token ids of `text`: `</s>`-terminated, truncated and padded with `<pad>` to `len`
    /// (diffusers `padding="max_length", truncation=True`).
    pub fn tokenize(&self, text: &str, len: usize) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("T5 tokenizer: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        if ids.len() > len {
            ids.truncate(len - 1);
            ids.push(self.eos_id);
        }
        ids.resize(len, self.pad_id);
        Ok(ids)
    }

    fn embed_row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        let d = self.d_model;
        let b = self.files.bytes(self.embed)?;
        let id = id as usize;
        if let Some(g) = self.files.ggml(self.embed) {
            let (kind, b) = g?;
            let rb = kind.row_bytes(d);
            crate::ggml::dequantize_row(kind, &b[id * rb..(id + 1) * rb], out);
            return Ok(());
        }
        match self.embed_dtype.as_str() {
            "BF16" => {
                for (o, c) in out.iter_mut().zip(b[id * d * 2..(id + 1) * d * 2].as_chunks::<2>().0) {
                    *o = f32::from_bits((u16::from_le_bytes(*c) as u32) << 16);
                }
            }
            "F16" => {
                for (o, c) in out.iter_mut().zip(b[id * d * 2..(id + 1) * d * 2].as_chunks::<2>().0) {
                    *o = half::f16::from_le_bytes(*c).to_f32();
                }
            }
            "F32" => {
                for (o, c) in out.iter_mut().zip(b[id * d * 4..(id + 1) * d * 4].as_chunks::<4>().0) {
                    *o = f32::from_le_bytes(*c);
                }
            }
            t => bail!("T5 embedding dtype {t} is not supported"),
        }
        Ok(())
    }

    fn bucket(&self, rel: i64) -> usize {
        relative_bucket(rel, self.num_buckets, self.max_distance)
    }

    /// The last hidden state (after the final norm) for `ids`: `[ids.len(), d_model]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let (t, d) = (ids.len(), self.d_model);
        let mut h = vec![0f32; t * d];
        for (i, &id) in ids.iter().enumerate() {
            self.embed_row(id, &mut h[i * d..(i + 1) * d])?;
        }
        // bias[head][q][k].
        let (nh, dk) = (self.heads, self.d_kv);
        let mut bias = vec![0f32; nh * t * t];
        let buckets: Vec<usize> = (0..t * t).map(|p| self.bucket((p % t) as i64 - (p / t) as i64)).collect();
        for_chunks(&mut bias, t * t, true, |hh, b| {
            for (o, &bk) in b.iter_mut().zip(&buckets) {
                *o = self.rel_bias[bk * nh + hh];
            }
        });
        for blk in &self.blocks {
            let x = rms(&h, &blk.ln1, d, self.eps);
            let qkv = forward_many(&blk.qkv.iter().collect::<Vec<_>>(), &x, t);
            let a = attention_bias(&qkv[0], &qkv[1], &qkv[2], t, nh, dk, &bias);
            add_inplace(&mut h, &blk.o.forward(&a, t));
            let x = rms(&h, &blk.ln2, d, self.eps);
            let mut g = forward_many(&blk.wi.iter().collect::<Vec<_>>(), &x, t);
            let up = g.pop().unwrap();
            let mut gate = g.pop().unwrap();
            for_chunks(&mut gate, 1 << 16, true, |i, c| {
                for (v, &u) in c.iter_mut().zip(&up[i << 16..]) {
                    *v = gelu_tanh(*v) * u;
                }
            });
            add_inplace(&mut h, &blk.wo.forward(&gate, t));
        }
        Ok(rms(&h, &self.final_ln, d, self.eps))
    }

    /// Tokenizes and encodes `text` padded to `len` tokens: `[len, d_model]`.
    pub fn encode(&self, text: &str, len: usize) -> Result<Vec<f32>> {
        let ids = self.tokenize(text, len)?;
        self.forward(&ids)
    }
}

/// The `config.json` equivalent of a llama.cpp `t5encoder` GGUF file's metadata.
fn gguf_config(path: &Path) -> Result<serde_json::Value> {
    let h = crate::gguf::header(path)?;
    if h.arch() != "t5encoder" {
        bail!("{}: GGUF architecture {} is not a T5 encoder", path.display(), h.arch());
    }
    let u = |k: &str| {
        h.get(&format!("t5encoder.{k}"))
            .and_then(|v| v.as_u64())
            .with_context(|| format!("GGUF t5encoder.{k}"))
    };
    let eps = h
        .get("t5encoder.attention.layer_norm_rms_epsilon")
        .or_else(|| h.get("t5encoder.attention.layer_norm_epsilon"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6);
    Ok(serde_json::json!({
        "d_model": u("embedding_length")?,
        "num_heads": u("attention.head_count")?,
        "d_kv": u("attention.key_length")?,
        "num_layers": u("block_count")?,
        "feed_forward_proj": "gated-gelu",
        // Stored as f32: round to the f32 value transformers configs carry (1e-6).
        "layer_norm_epsilon": (eps as f32) as f64,
        "relative_attention_num_buckets": u("attention.relative_buckets_count").unwrap_or(32),
        "relative_attention_max_distance": 128,
        "pad_token_id": h.get("tokenizer.ggml.padding_token_id").and_then(|v| v.as_u64()).unwrap_or(0),
        "eos_token_id": h.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).unwrap_or(1),
    }))
}

/// transformers' `T5Attention._relative_position_bucket(bidirectional=True)` for
/// `rel = key - query`.
pub fn relative_bucket(rel: i64, num_buckets: usize, max_distance: usize) -> usize {
    let nb = num_buckets / 2;
    let mut ret = if rel > 0 { nb } else { 0 };
    let n = rel.unsigned_abs() as usize;
    let max_exact = nb / 2;
    if n < max_exact {
        ret += n;
    } else {
        let v = (n as f32 / max_exact as f32).ln() / (max_distance as f32 / max_exact as f32).ln() * (nb - max_exact) as f32;
        ret += (max_exact + v as usize).min(nb - 1);
    }
    ret
}

/// T5 RMS norm: `x / sqrt(mean(x^2) + eps) * w` per row.
fn rms(x: &[f32], w: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    for_chunks(&mut y, d, x.len() > d * 8, |_, r| {
        let ms = r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
        let inv = (1.0 / (ms + eps as f64).sqrt()) as f32;
        for (v, &g) in r.iter_mut().zip(w) {
            *v = *v * inv * g;
        }
    });
    y
}

/// Unscaled multi-head attention with an additive bias `[heads, t, t]` (T5).
fn attention_bias(q: &[f32], k: &[f32], v: &[f32], t: usize, heads: usize, dh: usize, bias: &[f32]) -> Vec<f32> {
    use crate::sd::ops::matmul_f32;
    let d = heads * dh;
    let dhp = dh.next_multiple_of(32);
    let tkp = t.next_multiple_of(32);
    let mut out = vec![0f32; t * d];
    let mut qh = vec![0f32; t * dhp];
    let mut kh = vec![0f32; t * dhp];
    let mut vt = vec![0f32; dh * tkp];
    for h in 0..heads {
        for_chunks(&mut qh, dhp, true, |i, r| {
            r[..dh].copy_from_slice(&q[i * d + h * dh..i * d + (h + 1) * dh])
        });
        for_chunks(&mut kh, dhp, true, |j, r| {
            r[..dh].copy_from_slice(&k[j * d + h * dh..j * d + (h + 1) * dh])
        });
        for_chunks(&mut vt, tkp, true, |c, r| {
            for j in 0..t {
                r[j] = v[j * d + h * dh + c];
            }
        });
        let s = matmul_f32(&kh, t, dhp, &qh, t);
        let bh = &bias[h * t * t..(h + 1) * t * t];
        let mut p = vec![0f32; t * tkp];
        for_chunks(&mut p, tkp, true, |i, r| {
            let row = &s[i * t..(i + 1) * t];
            let br = &bh[i * t..(i + 1) * t];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..t {
                mx = mx.max(row[j] + br[j]);
            }
            let mut sum = 0f32;
            for j in 0..t {
                let e = (row[j] + br[j] - mx).exp();
                r[j] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for x in &mut r[..t] {
                *x *= inv;
            }
        });
        let o = matmul_f32(&vt, dh, tkp, &p, t);
        for_chunks(&mut out, d, true, |i, r| {
            r[h * dh..(h + 1) * dh].copy_from_slice(&o[i * dh..(i + 1) * dh])
        });
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn buckets() {
        // transformers 5.22 T5Attention._relative_position_bucket(rel, True, 32, 128).
        let want: [(i64, usize); 14] = [
            (0, 0),
            (1, 17),
            (-1, 1),
            (7, 23),
            (8, 24),
            (-8, 8),
            (12, 25),
            (-50, 13),
            (127, 31),
            (-511, 15),
            (20, 26),
            (-20, 10),
            (64, 30),
            (200, 31),
        ];
        for (r, w) in want {
            assert_eq!(super::relative_bucket(r, 32, 128), w, "rel {r}");
        }
    }
}
