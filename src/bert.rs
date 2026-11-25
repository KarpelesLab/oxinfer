//! BERT-style encoders (BERT, RoBERTa, XLM-RoBERTa: BGE-M3 and friends): learned absolute
//! positions, post-LayerNorm blocks, biased projections, exact (erf) GELU, bidirectional attention.

use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, attention_masked, linear, linear_multi, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d_eps() -> f64 {
    1e-12
}

#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub type_vocab_size: usize,
    #[serde(default = "d_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub pad_token_id: usize,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

impl BertConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: BertConfig = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse BERT config")?;
        if let Some(a) = &cfg.hidden_act
            && a != "gelu"
        {
            bail!("hidden_act {a} is not supported (only gelu)");
        }
        Ok(cfg)
    }
    /// RoBERTa numbers positions from `pad_token_id + 1`.
    fn position_offset(&self) -> usize {
        if self.model_type.contains("roberta") {
            self.pad_token_id + 1
        } else {
            0
        }
    }
}

/// Whether `model_type` is a BERT-family encoder.
pub fn is_bert(model_type: &str) -> bool {
    matches!(model_type, "bert" | "roberta" | "xlm-roberta" | "camembert")
}

pub(crate) struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    pub(crate) fn load(files: &ModelFiles, p: &str, h: usize) -> Result<Self> {
        let (w, b) = (files.f32(&format!("{p}.weight"))?, files.f32(&format!("{p}.bias"))?);
        if w.len() != h || b.len() != h {
            bail!("{p}: expected {h} values");
        }
        Ok(Self { w, b })
    }
    pub(crate) fn weight(&self) -> &[f32] {
        &self.w
    }
    pub(crate) fn bias(&self) -> &[f32] {
        &self.b
    }
    pub(crate) fn apply_rows(&self, x: &mut [f32], eps: f32) {
        let h = self.w.len();
        for_chunks(x, h, x.len() > 4 * h, |_, r| {
            let mean = r.iter().sum::<f32>() / h as f32;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
            let inv = (var + eps).sqrt().recip();
            for ((v, w), b) in r.iter_mut().zip(&self.w).zip(&self.b) {
                *v = (*v - mean) * inv * w + b;
            }
        });
    }
}

/// A linear layer with bias.
pub(crate) struct Dense<'a> {
    pub(crate) w: Lin<'a>,
    b: Vec<f32>,
    n_out: usize,
    n_in: usize,
}

impl<'a> Dense<'a> {
    pub(crate) fn load(files: &'a ModelFiles, p: &str, n_out: usize, n_in: usize, q: Quant) -> Result<Self> {
        Ok(Self {
            w: load_lin(files, &format!("{p}.weight"), n_out, n_in, q)?,
            b: files.f32(&format!("{p}.bias"))?,
            n_out,
            n_in,
        })
    }
    pub(crate) fn add_bias(&self, y: &mut [f32]) {
        for_chunks(y, self.n_out, y.len() > 4 * self.n_out, |_, r| add_inplace(r, &self.b));
    }
    pub(crate) fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        linear(&self.w, self.n_out, self.n_in, x, t, y);
        self.add_bias(y);
    }
}

struct BertLayer<'a> {
    q: Dense<'a>,
    k: Dense<'a>,
    v: Dense<'a>,
    o: Dense<'a>,
    ln_attn: LayerNorm,
    up: Dense<'a>,
    down: Dense<'a>,
    ln_out: LayerNorm,
}

pub struct BertModel<'a> {
    pub cfg: BertConfig,
    pub quant: Quant,
    word: Table<'a>,
    pos: Table<'a>,
    token_type: Vec<f32>,
    ln_embed: LayerNorm,
    layers: Vec<BertLayer<'a>>,
}

/// Exact GELU, `x * Phi(x)`, with erf from Abramowitz & Stegun 7.1.26 (error < 1.5e-7).
#[inline]
pub(crate) fn gelu_erf(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let poly = t * (0.254_829_6 + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152_1 + t * 1.061_405_4))));
    let erf = 1.0 - poly * (-a * a).exp();
    0.5 * x * (1.0 + erf.copysign(z))
}

impl<'a> BertModel<'a> {
    pub fn load(cfg: BertConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let prefix = ["", "roberta.", "bert.", "model."]
            .into_iter()
            .find(|p| files.has(&format!("{p}embeddings.word_embeddings.weight")))
            .context("cannot find embeddings.word_embeddings in checkpoint")?;
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let n = |s: &str| format!("{prefix}{s}");
        let word = Table::load(files, &n("embeddings.word_embeddings.weight"), cfg.vocab_size, h, Quant::Bf16)?;
        let pos = Table::load(
            files,
            &n("embeddings.position_embeddings.weight"),
            cfg.max_position_embeddings,
            h,
            Quant::Bf16,
        )?;
        let token_type = if files.has(&n("embeddings.token_type_embeddings.weight")) {
            files.f32(&n("embeddings.token_type_embeddings.weight"))?[..h].to_vec()
        } else {
            vec![0.0; h]
        };
        let ln_embed = LayerNorm::load(files, &n("embeddings.LayerNorm"), h)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = |s: &str| n(&format!("encoder.layer.{l}.{s}"));
            let d = |s: &str, o: usize, i: usize| Dense::load(files, &p(s), o, i, quant);
            layers.push(BertLayer {
                q: d("attention.self.query", h, h)?,
                k: d("attention.self.key", h, h)?,
                v: d("attention.self.value", h, h)?,
                o: d("attention.output.dense", h, h)?,
                ln_attn: LayerNorm::load(files, &p("attention.output.LayerNorm"), h)?,
                up: d("intermediate.dense", inter, h)?,
                down: d("output.dense", h, inter)?,
                ln_out: LayerNorm::load(files, &p("output.LayerNorm"), h)?,
            });
        }
        Ok(Self {
            cfg,
            quant,
            word,
            pos,
            token_type,
            ln_embed,
            layers,
        })
    }

    /// Last hidden states `[t, hidden]` for one sequence (special tokens included).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, nh) = (cfg.hidden_size, cfg.num_attention_heads);
        let hd = h / nh;
        let t = tokens.len().min(cfg.max_position_embeddings - cfg.position_offset());
        let tokens = &tokens[..t];
        let eps = cfg.layer_norm_eps as f32;
        let off = cfg.position_offset();
        let mut x = vec![0f32; t * h];
        let mut pe = vec![0f32; h];
        for (i, row) in x.chunks_mut(h).enumerate() {
            self.word.row(tokens[i] as usize, row);
            self.pos.row(off + i, &mut pe);
            add_inplace(row, &pe);
            add_inplace(row, &self.token_type);
        }
        self.ln_embed.apply_rows(&mut x, eps);
        let scale = (hd as f32).powf(-0.5);
        let (mut q, mut k, mut v) = (vec![0f32; t * h], vec![0f32; t * h], vec![0f32; t * h]);
        let mut att = vec![0f32; t * h];
        let mut o = vec![0f32; t * h];
        let mut u = vec![0f32; t * cfg.intermediate_size];
        for layer in &self.layers {
            linear_multi(
                &mut [
                    (&layer.q.w, h, &mut q[..]),
                    (&layer.k.w, h, &mut k[..]),
                    (&layer.v.w, h, &mut v[..]),
                ],
                h,
                &x,
                t,
            );
            layer.q.add_bias(&mut q);
            layer.k.add_bias(&mut k);
            layer.v.add_bias(&mut v);
            scale_inplace(&mut q, scale);
            let mut cache = LayerCache::new(nh, hd);
            cache.push(&k, &v, t);
            attention_masked(&q, &cache, nh, t, 0, None, false, &mut att);
            layer.o.forward(&att, t, &mut o);
            add_inplace(&mut x, &o);
            layer.ln_attn.apply_rows(&mut x, eps);
            layer.up.forward(&x, t, &mut u);
            let par = u.len() > 8192;
            for_chunks(&mut u, 4096, par, |_, c| {
                for e in c.iter_mut() {
                    *e = gelu_erf(*e);
                }
            });
            layer.down.forward(&u, t, &mut o);
            add_inplace(&mut x, &o);
            layer.ln_out.apply_rows(&mut x, eps);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gelu_erf_matches_known_values() {
        // x * Phi(x) for a few x (Phi from erf tables).
        for (x, want) in [
            (0.0f32, 0.0f32),
            (1.0, 0.841_344_7),
            (-1.0, -0.158_655_3),
            (2.5, 2.484_475_8),
            (-3.0, -0.004_049_7),
        ] {
            assert!(
                (super::gelu_erf(x) - want).abs() < 2e-6,
                "gelu({x}) = {} want {want}",
                super::gelu_erf(x)
            );
        }
    }
}
