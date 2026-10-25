//! Janus's understanding encoder: a SigLIP ViT-L/16 at 384x384 (timm `VisionTransformer`, 24
//! pre-norm blocks, no class token, final LayerNorm, pooling head unused) and the `MlpProjector`
//! aligner (Linear, GELU, Linear) into the language model's embedding space: 576 tokens per image.

use crate::model::{Lin, linear, linear_multi};
use crate::safetensors::ModelFiles;
use crate::sd::ops::{attention, gelu_erf};
use anyhow::{Result, bail};

/// A linear layer with an f32 bias; the weight is borrowed bf16.
pub(crate) struct Dense<'a> {
    pub w: Lin<'a>,
    pub b: Vec<f32>,
    pub n_out: usize,
    pub n_in: usize,
}

impl<'a> Dense<'a> {
    /// `{p}.weight` (`[n_out, n_in]`, or any shape with that many values when `n_in` is given
    /// explicitly, e.g. a patch-embedding kernel) and `{p}.bias`.
    pub fn load(files: &'a ModelFiles, p: &str, n_out: usize, n_in: usize) -> Result<Self> {
        let w = files.bf16(&format!("{p}.weight"))?;
        let b = files.f32(&format!("{p}.bias"))?;
        if w.len() != n_out * n_in || b.len() != n_out {
            bail!("{p}: {} weights / {} biases, expected {n_out}x{n_in}", w.len(), b.len());
        }
        Ok(Self {
            w: Lin::bf16(w),
            b,
            n_out,
            n_in,
        })
    }

    /// As [`Dense::load`], with the weight quantized to `quant` (a big head read every step).
    pub fn load_quant(files: &'a ModelFiles, p: &str, n_out: usize, n_in: usize, quant: crate::model::Quant) -> Result<Self> {
        let b = files.f32(&format!("{p}.bias"))?;
        if b.len() != n_out {
            bail!("{p}: {} biases, expected {n_out}", b.len());
        }
        Ok(Self {
            w: crate::model::load_lin(files, &format!("{p}.weight"), n_out, n_in, quant)?,
            b,
            n_out,
            n_in,
        })
    }

    /// Rows `r0..r1` of a fused weight (e.g. the query part of `qkv`) as a layer of their own.
    fn slice(files: &'a ModelFiles, p: &str, n_in: usize, r0: usize, r1: usize) -> Result<Self> {
        let w = files.bf16(&format!("{p}.weight"))?;
        let b = files.f32(&format!("{p}.bias"))?;
        if w.len() < r1 * n_in || b.len() < r1 {
            bail!("{p}: too small for rows {r0}..{r1}");
        }
        Ok(Self {
            w: Lin::bf16(&w[r0 * n_in..r1 * n_in]),
            b: b[r0..r1].to_vec(),
            n_out: r1 - r0,
            n_in,
        })
    }

    pub fn add_bias(&self, y: &mut [f32]) {
        crate::kernels::for_chunks(y, self.n_out, y.len() > 4 * self.n_out, |_, r| {
            for (v, &b) in r.iter_mut().zip(&self.b) {
                *v += b;
            }
        });
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.n_out];
        linear(&self.w, self.n_out, self.n_in, x, rows, &mut y);
        self.add_bias(&mut y);
        y
    }
}

/// Exact (erf) GELU in place, as `torch.nn.GELU()`.
pub(crate) fn gelu_inplace(x: &mut [f32]) {
    crate::kernels::for_chunks(x, 4096, x.len() > 65536, |_, c| c.iter_mut().for_each(|v| *v = gelu_erf(*v)));
}

/// The two-layer `mlp_gelu` projector (`aligner`, `gen_aligner`).
pub(crate) struct Mlp<'a> {
    pub l0: Dense<'a>,
    pub l1: Dense<'a>,
}

impl<'a> Mlp<'a> {
    pub fn load(files: &'a ModelFiles, p: &str, n_in: usize, n_embed: usize) -> Result<Self> {
        Ok(Self {
            l0: Dense::load(files, &format!("{p}.layers.0"), n_embed, n_in)?,
            l1: Dense::load(files, &format!("{p}.layers.2"), n_embed, n_embed)?,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = self.l0.forward(x, rows);
        gelu_inplace(&mut y);
        self.l1.forward(&y, rows)
    }
}

struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn load(files: &ModelFiles, p: &str, n: usize) -> Result<Self> {
        let (w, b) = (files.f32(&format!("{p}.weight"))?, files.f32(&format!("{p}.bias"))?);
        if w.len() != n || b.len() != n {
            bail!("{p}: expected {n} channels");
        }
        Ok(Self { w, b })
    }

    /// timm's LayerNorm (eps 1e-6) over rows of `x`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = self.w.len();
        let mut y = vec![0f32; x.len()];
        crate::kernels::for_chunks(&mut y, n, x.len() > 4 * n, |i, r| {
            let xr = &x[i * n..(i + 1) * n];
            let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let var = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for j in 0..n {
                r[j] = ((xr[j] as f64 - mean) * inv) as f32 * self.w[j] + self.b[j];
            }
        });
        y
    }
}

struct Block<'a> {
    norm1: LayerNorm,
    q: Dense<'a>,
    k: Dense<'a>,
    v: Dense<'a>,
    proj: Dense<'a>,
    norm2: LayerNorm,
    fc1: Dense<'a>,
    fc2: Dense<'a>,
}

pub struct VisionEncoder<'a> {
    pub image_size: usize,
    pub patch: usize,
    width: usize,
    heads: usize,
    patch_embed: Dense<'a>,
    pos_embed: Vec<f32>,
    blocks: Vec<Block<'a>>,
    norm: LayerNorm,
    aligner: Mlp<'a>,
    /// Pixel normalization (`preprocessor_config.json`).
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Padding colour of non-square images.
    pub background: [u8; 3],
}

impl<'a> VisionEncoder<'a> {
    /// `siglip_large_patch16_384` under `vision_model.vision_tower` plus the aligner, which
    /// projects to `n_embed`; `layers` is the depth kept (`select_layer`).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        files: &'a ModelFiles,
        image_size: usize,
        layers: usize,
        n_embed: usize,
        mean: [f32; 3],
        std: [f32; 3],
        background: [u8; 3],
    ) -> Result<Self> {
        let (width, heads, patch) = (1024, 16, 16);
        let p = "vision_model.vision_tower";
        let grid = image_size / patch;
        let pos_embed = files.f32(&format!("{p}.pos_embed"))?;
        if pos_embed.len() != grid * grid * width {
            bail!(
                "{p}.pos_embed: {} values for a {grid}x{grid} grid of width {width}",
                pos_embed.len()
            );
        }
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let b = format!("{p}.blocks.{i}");
            let qkv = format!("{b}.attn.qkv");
            blocks.push(Block {
                norm1: LayerNorm::load(files, &format!("{b}.norm1"), width)?,
                q: Dense::slice(files, &qkv, width, 0, width)?,
                k: Dense::slice(files, &qkv, width, width, 2 * width)?,
                v: Dense::slice(files, &qkv, width, 2 * width, 3 * width)?,
                proj: Dense::load(files, &format!("{b}.attn.proj"), width, width)?,
                norm2: LayerNorm::load(files, &format!("{b}.norm2"), width)?,
                fc1: Dense::load(files, &format!("{b}.mlp.fc1"), 4 * width, width)?,
                fc2: Dense::load(files, &format!("{b}.mlp.fc2"), width, 4 * width)?,
            });
        }
        Ok(Self {
            image_size,
            patch,
            width,
            heads,
            patch_embed: Dense::load(files, &format!("{p}.patch_embed.proj"), width, 3 * patch * patch)?,
            pos_embed,
            blocks,
            norm: LayerNorm::load(files, &format!("{p}.norm"), width)?,
            aligner: Mlp::load(files, "aligner", width, n_embed)?,
            mean,
            std,
            background,
        })
    }

    /// The processor's pixels for an RGB8 image: resized (Pillow bicubic) so that its longer side
    /// is `image_size`, padded to a square with the background colour (centered), rescaled and
    /// normalized; planar `[3, size, size]`.
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        if w == 0 || h == 0 || rgb.len() != w * h * 3 {
            bail!("image: {} bytes for {w}x{h} RGB", rgb.len());
        }
        let s = self.image_size;
        let m = w.max(h);
        // size = [max(int(h / m * s), 14), max(int(w / m * s), 14)] (min_size 14).
        let th = ((h as f64 / m as f64 * s as f64) as usize).max(14);
        let tw = ((w as f64 / m as f64 * s as f64) as usize).max(14);
        let img = crate::resample::resize_bicubic_rgb8(rgb, w, h, tw, th);
        // expand2square: centered on the longer side.
        let side = tw.max(th);
        let (ox, oy) = ((side - tw) / 2, (side - th) / 2);
        let mut sq = Vec::with_capacity(side * side * 3);
        for y in 0..side {
            for x in 0..side {
                if x >= ox && x < ox + tw && y >= oy && y < oy + th {
                    let o = ((y - oy) * tw + (x - ox)) * 3;
                    sq.extend_from_slice(&img[o..o + 3]);
                } else {
                    sq.extend_from_slice(&self.background);
                }
            }
        }
        if side != s {
            bail!("image: resized to {side}x{side}, expected {s}x{s}");
        }
        let mut px = vec![0f32; 3 * s * s];
        for c in 0..3 {
            for i in 0..s * s {
                let v = (sq[i * 3 + c] as f64 * (1.0 / 255.0)) as f32;
                px[c * s * s + i] = (v - self.mean[c]) / self.std[c];
            }
        }
        Ok(px)
    }

    /// The image's language-model embeddings `[576, n_embed]` from preprocessed pixels; with
    /// `features`, also the encoder output before the aligner (`[576, 1024]`).
    pub fn encode(&self, px: &[f32], features: Option<&mut Vec<f32>>) -> Vec<f32> {
        let (s, p, d) = (self.image_size, self.patch, self.width);
        let g = s / p;
        let n = g * g;
        // Patches as rows of (channel, y, x), the kernel's layout.
        let mut patches = vec![0f32; n * 3 * p * p];
        crate::kernels::for_chunks(&mut patches, 3 * p * p, true, |i, r| {
            let (gy, gx) = (i / g, i % g);
            for c in 0..3 {
                for y in 0..p {
                    let src = c * s * s + (gy * p + y) * s + gx * p;
                    r[(c * p + y) * p..(c * p + y + 1) * p].copy_from_slice(&px[src..src + p]);
                }
            }
        });
        let mut x = self.patch_embed.forward(&patches, n);
        crate::kernels::add_inplace(&mut x, &self.pos_embed);
        let hd = d / self.heads;
        for b in &self.blocks {
            let hn = b.norm1.forward(&x);
            let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
            linear_multi(
                &mut [(&b.q.w, d, &mut q[..]), (&b.k.w, d, &mut k[..]), (&b.v.w, d, &mut v[..])],
                d,
                &hn,
                n,
            );
            b.q.add_bias(&mut q);
            b.k.add_bias(&mut k);
            b.v.add_bias(&mut v);
            let a = attention(&q, &k, &v, n, n, self.heads, hd, false);
            crate::kernels::add_inplace(&mut x, &b.proj.forward(&a, n));
            let hn = b.norm2.forward(&x);
            let mut m = b.fc1.forward(&hn, n);
            gelu_inplace(&mut m);
            crate::kernels::add_inplace(&mut x, &b.fc2.forward(&m, n));
        }
        let x = self.norm.forward(&x);
        if let Some(f) = features {
            f.clone_from(&x);
        }
        self.aligner.forward(&x, n)
    }

    /// Image tokens per image.
    pub fn tokens(&self) -> usize {
        (self.image_size / self.patch).pow(2)
    }
}
