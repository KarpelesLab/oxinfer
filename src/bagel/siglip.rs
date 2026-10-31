//! BAGEL's understanding encoder: SigLIP so400m/14 run NaViT-style (any image size whose sides
//! are multiples of 14, one token per patch, no padding to a square), with the learned position
//! embedding of a 70x70 grid indexed by each patch's `row * 70 + col`; 26 of its 27 layers
//! (the last is dropped, as the reference does), the final LayerNorm, then BAGEL's connector
//! (Linear, tanh-GELU, Linear) into the language model's width plus a fixed 2-D sin-cos position
//! embedding of the same grid.

use super::llm::{Proj, project};
use crate::kernels::{add_inplace, for_chunks, gelu_tanh};
use crate::model::Quant;
use crate::safetensors::ModelFiles;
use anyhow::{Result, bail};

/// `vit_config.json` plus BAGEL's settings.
#[derive(Clone, Debug)]
pub struct VitConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Layers run (the config's count minus one).
    pub layers: usize,
    pub heads: usize,
    pub patch: usize,
    /// Patches per side of the position-embedding grid (`vit_max_num_patch_per_side`, 70).
    pub max_patches: usize,
    pub eps: f32,
}

impl VitConfig {
    pub fn from_json(v: &serde_json::Value, max_patches: usize) -> Self {
        let get = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        Self {
            hidden_size: get("hidden_size", 1152),
            intermediate_size: get("intermediate_size", 4304),
            layers: get("num_hidden_layers", 27).saturating_sub(1),
            heads: get("num_attention_heads", 16),
            patch: get("patch_size", 14),
            max_patches,
            eps: v.get("layer_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6) as f32,
        }
    }
}

pub(crate) struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn load(files: &ModelFiles, p: &str, n: usize, eps: f32) -> Result<Self> {
        let (w, b) = (files.f32(&format!("{p}.weight"))?, files.f32(&format!("{p}.bias"))?);
        if w.len() != n || b.len() != n {
            bail!("{p}: expected {n} channels");
        }
        Ok(Self { w, b, eps })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = self.w.len();
        let mut y = vec![0f32; x.len()];
        for_chunks(&mut y, n, x.len() > 4 * n, |i, r| {
            let xr = &x[i * n..(i + 1) * n];
            let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let var = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
            let inv = 1.0 / (var + self.eps as f64).sqrt();
            for j in 0..n {
                r[j] = ((xr[j] as f64 - mean) * inv) as f32 * self.w[j] + self.b[j];
            }
        });
        y
    }
}

pub(crate) fn gelu_tanh_inplace(x: &mut [f32]) {
    for_chunks(x, 16384, x.len() > 65536, |_, c| c.iter_mut().for_each(|v| *v = gelu_tanh(*v)));
}

struct Layer<'a> {
    ln1: LayerNorm,
    q: Proj<'a>,
    k: Proj<'a>,
    v: Proj<'a>,
    o: Proj<'a>,
    ln2: LayerNorm,
    fc1: Proj<'a>,
    fc2: Proj<'a>,
}

pub struct Vit<'a> {
    pub cfg: VitConfig,
    patch_embed: Proj<'a>,
    /// `[max_patches^2, hidden]`.
    pos_embed: Vec<f32>,
    layers: Vec<Layer<'a>>,
    post_ln: LayerNorm,
    fc1: Proj<'a>,
    fc2: Proj<'a>,
    /// The connector output's position embedding `[max_patches^2, llm hidden]` (bf16 rows).
    llm_pos: &'a [u16],
    pub llm_hidden: usize,
}

impl<'a> Vit<'a> {
    pub fn load(cfg: VitConfig, files: &'a ModelFiles, llm_hidden: usize, quant: Quant) -> Result<Self> {
        let (d, inter, p) = (cfg.hidden_size, cfg.intermediate_size, cfg.patch);
        let pre = "vit_model.vision_model";
        let lin = |n: &str, o: usize, i: usize| Proj::load(files, n, o, i, quant, true);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = format!("{pre}.encoder.layers.{i}");
            layers.push(Layer {
                ln1: LayerNorm::load(files, &format!("{l}.layer_norm1"), d, cfg.eps)?,
                q: lin(&format!("{l}.self_attn.q_proj"), d, d)?,
                k: lin(&format!("{l}.self_attn.k_proj"), d, d)?,
                v: lin(&format!("{l}.self_attn.v_proj"), d, d)?,
                o: lin(&format!("{l}.self_attn.out_proj"), d, d)?,
                ln2: LayerNorm::load(files, &format!("{l}.layer_norm2"), d, cfg.eps)?,
                fc1: lin(&format!("{l}.mlp.fc1"), inter, d)?,
                fc2: lin(&format!("{l}.mlp.fc2"), d, inter)?,
            });
        }
        let grid = cfg.max_patches * cfg.max_patches;
        let pos_embed = files.f32(&format!("{pre}.embeddings.position_embedding.weight"))?;
        if pos_embed.len() != grid * d {
            bail!("SigLIP position embedding: {} values, expected {grid}x{d}", pos_embed.len());
        }
        let llm_pos = files.bf16("vit_pos_embed.pos_embed")?;
        if llm_pos.len() != grid * llm_hidden {
            bail!("vit_pos_embed: {} values, expected {grid}x{llm_hidden}", llm_pos.len());
        }
        // The checkpoint stores the patch embedding as a linear layer over (row, col, channel).
        let patch_embed = Proj::load(files, &format!("{pre}.embeddings.patch_embedding"), d, 3 * p * p, Quant::Bf16, true)?;
        Ok(Self {
            patch_embed,
            pos_embed,
            layers,
            post_ln: LayerNorm::load(files, &format!("{pre}.post_layernorm"), d, cfg.eps)?,
            fc1: Proj::load(files, "connector.fc1", llm_hidden, d, quant, true)?,
            fc2: Proj::load(files, "connector.fc2", llm_hidden, llm_hidden, quant, true)?,
            llm_pos,
            llm_hidden,
            cfg,
        })
    }

    /// The patch grid's position ids (`get_flattened_position_ids_extrapolate`).
    pub fn position_ids(&self, gh: usize, gw: usize) -> Vec<usize> {
        let m = self.cfg.max_patches;
        (0..gh).flat_map(|r| (0..gw).map(move |c| r * m + c)).collect()
    }

    /// Encodes pixels (`[h*w, 3]` channels-last, normalized to [-1, 1]; `h`, `w` multiples of
    /// the patch size, at most `max_patches` patches per side) into language-model embeddings
    /// `[patches, llm_hidden]` (connector + position embedding). With `features`, also the
    /// SigLIP output before the connector.
    pub fn encode(&self, px: &[f32], h: usize, w: usize, features: Option<&mut Vec<f32>>) -> Result<Vec<f32>> {
        let (p, d) = (self.cfg.patch, self.cfg.hidden_size);
        if !h.is_multiple_of(p) || !w.is_multiple_of(p) || px.len() != h * w * 3 {
            bail!("SigLIP input {w}x{h} is not a multiple of the {p}-pixel patch");
        }
        let (gh, gw) = (h / p, w / p);
        if gh > self.cfg.max_patches || gw > self.cfg.max_patches {
            bail!("SigLIP input {w}x{h} exceeds {} patches per side", self.cfg.max_patches);
        }
        let n = gh * gw;
        // patchify: rows of (y, x, channel) within the patch.
        let mut patches = vec![0f32; n * 3 * p * p];
        for_chunks(&mut patches, 3 * p * p, true, |i, r| {
            let (gy, gx) = (i / gw, i % gw);
            for y in 0..p {
                let src = ((gy * p + y) * w + gx * p) * 3;
                r[y * p * 3..(y + 1) * p * 3].copy_from_slice(&px[src..src + p * 3]);
            }
        });
        let mut x = project(&[&self.patch_embed], 3 * p * p, &patches, n).swap_remove(0);
        drop(patches);
        let ids = self.position_ids(gh, gw);
        for_chunks(&mut x, d, n > 64, |i, r| {
            add_inplace(r, &self.pos_embed[ids[i] * d..(ids[i] + 1) * d]);
        });
        let heads = self.cfg.heads;
        let hd = d / heads;
        for l in &self.layers {
            let hn = l.ln1.forward(&x);
            let mut ys = project(&[&l.q, &l.k, &l.v], d, &hn, n);
            drop(hn);
            let v = ys.pop().unwrap();
            let k = ys.pop().unwrap();
            let q = ys.pop().unwrap();
            let a = crate::sd::ops::attention(&q, &k, &v, n, n, heads, hd, false);
            drop((q, k, v));
            add_inplace(&mut x, &project(&[&l.o], d, &a, n).swap_remove(0));
            let hn = l.ln2.forward(&x);
            let mut m = project(&[&l.fc1], d, &hn, n).swap_remove(0);
            gelu_tanh_inplace(&mut m);
            add_inplace(&mut x, &project(&[&l.fc2], self.cfg.intermediate_size, &m, n).swap_remove(0));
        }
        let x = self.post_ln.forward(&x);
        let mut y = project(&[&self.fc1], d, &x, n).swap_remove(0);
        if let Some(f) = features {
            *f = x;
        }
        gelu_tanh_inplace(&mut y);
        let mut y = project(&[&self.fc2], self.llm_hidden, &y, n).swap_remove(0);
        let lh = self.llm_hidden;
        for_chunks(&mut y, lh, n > 64, |i, r| {
            for (o, &b) in r.iter_mut().zip(&self.llm_pos[ids[i] * lh..(ids[i] + 1) * lh]) {
                *o += crate::kernels::bf16_to_f32(b);
            }
        });
        Ok(y)
    }
}
