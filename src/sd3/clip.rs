//! The CLIP text encoders of Stable Diffusion 3 (transformers `CLIPTextModelWithProjection`):
//! OpenAI CLIP ViT-L/14 (`text_encoder/`, 768 wide, quick_gelu) and OpenCLIP ViT-bigG/14
//! (`text_encoder_2/`, 1280 wide, gelu). SD 3 reads two outputs of each: the hidden states
//! before the last block (`hidden_states[-2]`, no final layer norm) as per-token conditioning,
//! and the projected pooled embedding (`text_embeds`: the final-layer-norm state at the
//! end-of-text token through `text_projection`).
//!
//! The tokenizer is [`crate::sd::clip::ClipTokenizer`]; the weights go through
//! [`crate::flux::weights`], which runs fp16 / f32 matrices in place from the mapped file (no
//! rounding to bf16).

use crate::flux::weights::{Dense, Loader, forward_many};
use crate::kernels::{add_inplace, for_chunks};
use crate::model::Quant;
use crate::safetensors::ModelFiles;
use crate::sd::ops::{attention, gelu_erf, quick_gelu};
use anyhow::{Context, Result, bail};
use std::path::Path;

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(ld: &Loader, p: &str) -> Result<Self> {
        Ok(Self {
            w: ld.f32(&format!("{p}.weight"))?,
            b: ld.f32(&format!("{p}.bias"))?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize, eps: f32) -> Vec<f32> {
        let d = self.w.len();
        let mut y = vec![0f32; rows * d];
        for_chunks(&mut y, d, rows > 16, |r, out| {
            let row = &x[r * d..(r + 1) * d];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = (1.0 / (var + eps as f64).sqrt()) as f32;
            let m = mean as f32;
            for i in 0..d {
                out[i] = (row[i] - m) * inv * self.w[i] + self.b[i];
            }
        });
        y
    }
}

struct Layer<'a> {
    ln1: Norm,
    qkv: Vec<Dense<'a>>,
    o: Dense<'a>,
    ln2: Norm,
    fc1: Dense<'a>,
    fc2: Dense<'a>,
}

/// What SD 3 takes from one CLIP encoder.
pub struct ClipOutput {
    /// `hidden_states[-2]`: `[tokens, hidden]`.
    pub penultimate: Vec<f32>,
    /// `text_embeds`: `[projection_dim]`.
    pub pooled: Vec<f32>,
}

pub struct ClipTextProj {
    /// Token embedding rows as f32.
    tok: Vec<f32>,
    pos: Vec<f32>,
    layers: Vec<Layer<'static>>,
    final_ln: Norm,
    projection: Dense<'static>,
    quick_gelu: bool,
    heads: usize,
    eps: f32,
    pub hidden: usize,
    pub projection_dim: usize,
    // Declared last: dropped after every layer borrowing it.
    _files: Box<ModelFiles>,
}

impl ClipTextProj {
    /// `dir` holds `config.json` and `model.safetensors`.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let u = |k: &str| cfg[k].as_u64().map(|v| v as usize).with_context(|| format!("CLIP config: {k}"));
        let (hidden, heads, n_layers) = (u("hidden_size")?, u("num_attention_heads")?, u("num_hidden_layers")?);
        let quick_gelu = match cfg["hidden_act"].as_str().unwrap_or("quick_gelu") {
            "quick_gelu" => true,
            "gelu" => false,
            other => bail!("text encoder activation {other} is not supported"),
        };
        if n_layers < 2 {
            bail!("a CLIP text encoder with {n_layers} layers has no penultimate state");
        }
        let files = Box::new(ModelFiles::open(dir, cfg!(target_arch = "wasm32"))?);
        // SAFETY: the box's heap block does not move, and every layer borrowing it is dropped
        // before `_files` (field order).
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let ld = Loader {
            files: fs,
            quant: Quant::Bf16,
        };
        if !ld.has("text_projection.weight") {
            bail!("{}: no text_projection (SD 3 needs CLIPTextModelWithProjection)", dir.display());
        }
        let p = "text_model.";
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lp = format!("{p}encoder.layers.{l}.");
            layers.push(Layer {
                ln1: Norm::load(&ld, &format!("{lp}layer_norm1"))?,
                qkv: ["q_proj", "k_proj", "v_proj"]
                    .iter()
                    .map(|n| ld.dense(&format!("{lp}self_attn.{n}"), false))
                    .collect::<Result<_>>()?,
                o: ld.dense(&format!("{lp}self_attn.out_proj"), false)?,
                ln2: Norm::load(&ld, &format!("{lp}layer_norm2"))?,
                fc1: ld.dense(&format!("{lp}mlp.fc1"), false)?,
                fc2: ld.dense(&format!("{lp}mlp.fc2"), false)?,
            });
        }
        let projection = ld.dense("text_projection", false)?;
        Ok(Self {
            tok: ld.f32(&format!("{p}embeddings.token_embedding.weight"))?,
            pos: ld.f32(&format!("{p}embeddings.position_embedding.weight"))?,
            layers,
            final_ln: Norm::load(&ld, &format!("{p}final_layer_norm"))?,
            projection_dim: projection.n_out,
            projection,
            quick_gelu,
            heads,
            eps: cfg["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            hidden,
            _files: files,
        })
    }

    /// Runs the encoder on `ids` (framed and padded by the tokenizer); `eos` is the index of the
    /// first end-of-text token, where the pooled state is read.
    pub fn forward(&self, ids: &[u32], eos: usize) -> ClipOutput {
        let (t, h) = (ids.len(), self.hidden);
        let mut x = vec![0f32; t * h];
        for (i, r) in x.chunks_mut(h).enumerate() {
            let id = ids[i] as usize;
            for ((o, &e), &p) in r.iter_mut().zip(&self.tok[id * h..(id + 1) * h]).zip(&self.pos[i * h..(i + 1) * h]) {
                *o = e + p;
            }
        }
        let mut penultimate = Vec::new();
        let last = self.layers.len() - 1;
        for (n, l) in self.layers.iter().enumerate() {
            if n == last {
                penultimate = x.clone();
            }
            let y = l.ln1.forward(&x, t, self.eps);
            let qkv = forward_many(&l.qkv.iter().collect::<Vec<_>>(), &y, t);
            let a = attention(&qkv[0], &qkv[1], &qkv[2], t, t, self.heads, h / self.heads, true);
            add_inplace(&mut x, &l.o.forward(&a, t));
            let y = l.ln2.forward(&x, t, self.eps);
            let mut m = l.fc1.forward(&y, t);
            if self.quick_gelu {
                m.iter_mut().for_each(|v| *v = quick_gelu(*v));
            } else {
                m.iter_mut().for_each(|v| *v = gelu_erf(*v));
            }
            add_inplace(&mut x, &l.fc2.forward(&m, t));
        }
        let e = eos.min(t - 1);
        let pooled = self.final_ln.forward(&x[e * h..(e + 1) * h], 1, self.eps);
        ClipOutput {
            penultimate,
            pooled: self.projection.forward_vec(&pooled),
        }
    }
}
