//! The image-token decoder of Janus (LlamaGen's VQ-16, `janus/models/vq_model.py`): codebook
//! lookup (16384 l2-normalized 8-dim codes), a 1x1 `post_quant_conv`, then a taming-style
//! convolutional decoder with 16x upsampling (24x24 tokens -> 384x384 RGB). Built from the Stable
//! Diffusion blocks (`crate::sd::ops`), so its convolutions run on the GPU backends too.

use crate::sd::ops::{Conv, GroupNorm, Linear, Pre, Tensors, add_inplace, attention, upsample2x};
use crate::sd::unet::Resnet;
use anyhow::{Result, bail};

/// A single-head spatial self-attention block (`AttnBlock`) with its residual.
struct AttnBlock {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    proj: Linear,
}

impl AttnBlock {
    fn load(t: &Tensors, p: &str) -> Result<Self> {
        Ok(Self {
            norm: GroupNorm::load(t, &format!("{p}.norm"), 32, 1e-6)?,
            q: Linear::load(t, &format!("{p}.q"))?,
            k: Linear::load(t, &format!("{p}.k"))?,
            v: Linear::load(t, &format!("{p}.v"))?,
            proj: Linear::load(t, &format!("{p}.proj_out"))?,
        })
    }

    fn forward(&self, x: &mut [f32], n: usize) {
        let pre = Pre::Group(&self.norm, false);
        let (q, k, v) = if self.q.on_device(n) {
            (
                self.q.forward_pre(x, n, pre),
                self.k.forward_pre(x, n, pre),
                self.v.forward_pre(x, n, pre),
            )
        } else {
            let y = pre.apply(x, n);
            (self.q.forward(&y, n), self.k.forward(&y, n), self.v.forward(&y, n))
        };
        let a = attention(&q, &k, &v, n, n, 1, self.q.n_out, false);
        add_inplace(x, &self.proj.forward(&a, n));
    }
}

struct Level {
    res: Vec<Resnet>,
    /// One per resnet at the lowest resolution, none elsewhere.
    attn: Vec<AttnBlock>,
    up: Option<Conv>,
}

pub struct VqDecoder {
    /// l2-normalized codebook `[codes, dim]`.
    codebook: Vec<f32>,
    pub codes: usize,
    pub dim: usize,
    post_quant: Linear,
    conv_in: Conv,
    mid: (Resnet, AttnBlock, Resnet),
    levels: Vec<Level>,
    norm_out: GroupNorm,
    conv_out: Conv,
}

impl VqDecoder {
    /// The decoder under `p` (`gen_vision_model`) of `t`.
    pub fn load(t: &Tensors, p: &str) -> Result<Self> {
        let cb_name = format!("{p}.quantize.embedding.weight");
        let shape = t.shape(&cb_name)?;
        let [codes, dim] = shape[..] else {
            bail!("{cb_name}: expected a matrix, got {shape:?}")
        };
        let mut codebook = t.f32(&cb_name)?;
        // F.normalize(p=2, dim=-1): x / max(|x|, 1e-12).
        for r in codebook.chunks_mut(dim) {
            let n = r.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            r.iter_mut().for_each(|v| *v /= n);
        }
        let d = format!("{p}.decoder");
        let mut levels = Vec::new();
        while t.has(&format!("{d}.conv_blocks.{}.res.0.conv1.weight", levels.len())) {
            let b = format!("{d}.conv_blocks.{}", levels.len());
            let mut res = Vec::new();
            while t.has(&format!("{b}.res.{}.conv1.weight", res.len())) {
                res.push(Resnet::load(t, &format!("{b}.res.{}", res.len()), 1e-6)?);
            }
            let mut attn = Vec::new();
            while t.has(&format!("{b}.attn.{}.q.weight", attn.len())) {
                attn.push(AttnBlock::load(t, &format!("{b}.attn.{}", attn.len()))?);
            }
            if !attn.is_empty() && attn.len() != res.len() {
                bail!("{b}: {} attention blocks for {} resnets", attn.len(), res.len());
            }
            let up = if t.has(&format!("{b}.upsample.conv.weight")) {
                Some(Conv::load(t, &format!("{b}.upsample.conv"))?)
            } else {
                None
            };
            levels.push(Level { res, attn, up });
        }
        if levels.is_empty() {
            bail!("{d}: no conv_blocks");
        }
        let post_quant = Linear::load(t, &format!("{p}.post_quant_conv"))?;
        if post_quant.n_in != dim {
            bail!("{p}.post_quant_conv takes {} channels, the codebook has {dim}", post_quant.n_in);
        }
        Ok(Self {
            codebook,
            codes,
            dim,
            post_quant,
            conv_in: Conv::load(t, &format!("{d}.conv_in"))?,
            mid: (
                Resnet::load(t, &format!("{d}.mid.0"), 1e-6)?,
                AttnBlock::load(t, &format!("{d}.mid.1"))?,
                Resnet::load(t, &format!("{d}.mid.2"), 1e-6)?,
            ),
            levels,
            norm_out: GroupNorm::load(t, &format!("{d}.norm_out"), 32, 1e-6)?,
            conv_out: Conv::load(t, &format!("{d}.conv_out"))?,
        })
    }

    /// The upsampling factor (16 for VQ-16).
    pub fn scale(&self) -> usize {
        1 << self.levels.iter().filter(|l| l.up.is_some()).count()
    }

    /// Image tokens (row-major `gh` x `gw` grid) to RGB in [-1, 1], channels-last
    /// (`[height * width, 3]`); returns `(pixels, height, width)`.
    pub fn decode(&self, tokens: &[u32], gh: usize, gw: usize) -> Result<(Vec<f32>, usize, usize)> {
        if tokens.len() != gh * gw {
            bail!("{} image tokens for a {gh}x{gw} grid", tokens.len());
        }
        let dim = self.dim;
        let mut z = vec![0f32; tokens.len() * dim];
        for (r, &tok) in z.chunks_mut(dim).zip(tokens) {
            let tok = tok as usize;
            if tok >= self.codes {
                bail!("image token {tok} is outside the codebook ({} codes)", self.codes);
            }
            r.copy_from_slice(&self.codebook[tok * dim..(tok + 1) * dim]);
        }
        let z = self.post_quant.forward(&z, gh * gw);
        let (mut x, mut h, mut w) = self.conv_in.forward(&z, gh, gw, 1);
        x = self.mid.0.forward(&x, h, w, None);
        self.mid.1.forward(&mut x, h * w);
        x = self.mid.2.forward(&x, h, w, None);
        for l in &self.levels {
            for (i, r) in l.res.iter().enumerate() {
                x = r.forward(&x, h, w, None);
                if let Some(a) = l.attn.get(i) {
                    a.forward(&mut x, h * w);
                }
            }
            if let Some(u) = &l.up {
                let c = x.len() / (h * w);
                let up = upsample2x(&x, h, w, c);
                (h, w) = (h * 2, w * 2);
                x = u.forward(&up, h, w, 1).0;
            }
        }
        let (y, _, _) = self.conv_out.forward_pre(&x, h, w, 1, Pre::Group(&self.norm_out, true));
        Ok((y, h, w))
    }
}

/// RGB in [-1, 1] to bytes as the reference does: `clip((x + 1) / 2 * 255, 0, 255)` stored
/// into uint8 (truncation).
pub fn to_rgb8(x: &[f32]) -> Vec<u8> {
    x.iter().map(|&v| ((v + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0) as u8).collect()
}
