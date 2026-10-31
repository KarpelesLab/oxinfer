//! `AutoencoderKLQwenImage`: the Wan-style causal 3D-convolution VAE of Qwen-Image (16 latent
//! channels, 8x spatial compression), run on single images. For one frame every causal
//! convolution reduces to a 2D convolution with the last temporal slice of its kernel (the
//! earlier slices meet the causal zero padding), and the temporal down / upsampling layers are
//! skipped (the first chunk only primes their caches), so encoder and decoder are 2D networks:
//! RMS norms over channels (`F.normalize * sqrt(C) * gamma`), SiLU residual blocks, one
//! single-head attention in the middle block, stride-2 downsampling after a `(0, 1, 0, 1)` zero
//! pad, nearest 2x upsampling.

use crate::kernels::for_chunks;
use crate::sd::ops::{Conv, Tensors, silu};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d96() -> usize {
    96
}
fn d16() -> usize {
    16
}
fn d2() -> usize {
    2
}
fn dmult() -> Vec<usize> {
    vec![1, 2, 4, 4]
}
fn dtd() -> Vec<bool> {
    vec![false, true, true]
}

/// `vae/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    #[serde(default = "d96")]
    pub base_dim: usize,
    #[serde(default = "d16")]
    pub z_dim: usize,
    #[serde(default = "dmult")]
    pub dim_mult: Vec<usize>,
    #[serde(default = "d2")]
    pub num_res_blocks: usize,
    #[serde(default)]
    pub attn_scales: Vec<f64>,
    #[serde(default = "dtd")]
    pub temperal_downsample: Vec<bool>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

/// Channel RMS norm: `x / max(|x|, 1e-12) * sqrt(C) * gamma` per pixel.
struct Rms {
    g: Vec<f32>,
}

impl Rms {
    fn load(t: &Tensors, name: &str, c: usize) -> Result<Self> {
        let g = t.f32(&format!("{name}.gamma"))?;
        if g.len() != c {
            bail!("{name}.gamma has {} values, expected {c}", g.len());
        }
        let s = (c as f32).sqrt();
        Ok(Self {
            g: g.into_iter().map(|v| v * s).collect(),
        })
    }

    /// The normalized copy of `[n, C]` channels-last `x`, optionally followed by SiLU.
    fn apply(&self, x: &[f32], silu_after: bool) -> Vec<f32> {
        let c = self.g.len();
        let mut y = vec![0f32; x.len()];
        for_chunks(&mut y, c * 64, true, |i, r| {
            for (p, out) in r.chunks_mut(c).enumerate() {
                let src = &x[(i * 64 + p) * c..(i * 64 + p + 1) * c];
                let n = src.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt().max(1e-12);
                let inv = (1.0 / n) as f32;
                for ((o, &v), &g) in out.iter_mut().zip(src).zip(&self.g) {
                    let a = v * inv * g;
                    *o = if silu_after { silu(a) } else { a };
                }
            }
        });
        y
    }
}

/// A causal 3D convolution `[cout, cin, kt, k, k]` applied to one frame: its last temporal slice.
fn conv(t: &Tensors, name: &str) -> Result<Conv> {
    let wn = format!("{name}.weight");
    let shape = t.shape(&wn)?;
    let w = t.f32(&wn)?;
    let (cout, cin, kt, k) = match shape[..] {
        [o, i, kt, kh, kw] if kh == kw => (o, i, kt, kh),
        [o, i, kh, kw] if kh == kw => (o, i, 1, kh),
        _ => bail!("{wn}: unexpected kernel shape {shape:?}"),
    };
    let sl = k * k;
    let mut w2 = vec![0f32; cout * cin * sl];
    for oc in 0..cout {
        for ic in 0..cin {
            let src = ((oc * cin + ic) * kt + kt - 1) * sl;
            w2[(oc * cin + ic) * sl..(oc * cin + ic + 1) * sl].copy_from_slice(&w[src..src + sl]);
        }
    }
    let bias = t.f32(&format!("{name}.bias")).ok();
    Conv::from_kernel(&w2, cout, cin, k, bias).with_context(|| name.to_string())
}

struct Res {
    norm1: Rms,
    conv1: Conv,
    norm2: Rms,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl Res {
    fn load(t: &Tensors, p: &str, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self {
            norm1: Rms::load(t, &format!("{p}.norm1"), cin)?,
            conv1: conv(t, &format!("{p}.conv1"))?,
            norm2: Rms::load(t, &format!("{p}.norm2"), cout)?,
            conv2: conv(t, &format!("{p}.conv2"))?,
            shortcut: if cin != cout {
                Some(conv(t, &format!("{p}.conv_shortcut"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let sc = match &self.shortcut {
            Some(c) => c.forward(x, h, w, 1).0,
            None => x.to_vec(),
        };
        let a = self.norm1.apply(x, true);
        let (a, _, _) = self.conv1.forward(&a, h, w, 1);
        let a = self.norm2.apply(&a, true);
        let (mut a, _, _) = self.conv2.forward(&a, h, w, 1);
        crate::kernels::add_inplace(&mut a, &sc);
        a
    }
}

struct Attn {
    norm: Rms,
    qkv: Conv,
    proj: Conv,
    c: usize,
}

impl Attn {
    fn load(t: &Tensors, p: &str, c: usize) -> Result<Self> {
        Ok(Self {
            norm: Rms::load(t, &format!("{p}.norm"), c)?,
            qkv: conv(t, &format!("{p}.to_qkv"))?,
            proj: conv(t, &format!("{p}.proj"))?,
            c,
        })
    }

    fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let c = self.c;
        let n = h * w;
        let xn = self.norm.apply(x, false);
        let (qkv, _, _) = self.qkv.forward(&xn, h, w, 1);
        let (mut q, mut k, mut v) = (vec![0f32; n * c], vec![0f32; n * c], vec![0f32; n * c]);
        for (buf, o) in [(&mut q, 0), (&mut k, c), (&mut v, 2 * c)] {
            for_chunks(buf, c, true, |i, r| r.copy_from_slice(&qkv[i * 3 * c + o..i * 3 * c + o + c]));
        }
        let a = crate::qwen_vision::attention_chunked(&q, &k, &v, n, 1, c);
        let (mut y, _, _) = self.proj.forward(&a, h, w, 1);
        crate::kernels::add_inplace(&mut y, x);
        y
    }
}

enum Layer {
    Res(Box<Res>),
    Attn(Box<Attn>),
    /// Stride-2 convolution after a right / bottom zero pad.
    Down(Conv),
    /// Nearest 2x upsampling, then a 3x3 convolution.
    Up(Conv),
}

impl Layer {
    fn forward(&self, x: Vec<f32>, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        match self {
            Layer::Res(r) => (r.forward(&x, h, w), h, w),
            Layer::Attn(a) => (a.forward(&x, h, w), h, w),
            Layer::Down(c) => c.forward(&x, h, w, 2),
            Layer::Up(c) => {
                let u = crate::sd::ops::upsample2x(&x, h, w, c.cin);
                let (y, _, _) = c.forward(&u, 2 * h, 2 * w, 1);
                (y, 2 * h, 2 * w)
            }
        }
    }
}

struct Stack {
    conv_in: Conv,
    layers: Vec<Layer>,
    norm_out: Rms,
    conv_out: Conv,
}

impl Stack {
    fn forward(&self, x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let (mut x, mut h, mut w) = self.conv_in.forward(x, h, w, 1);
        for l in &self.layers {
            (x, h, w) = l.forward(x, h, w);
        }
        let a = self.norm_out.apply(&x, true);
        let (y, _, _) = self.conv_out.forward(&a, h, w, 1);
        (y, h, w)
    }
}

/// The VAE: encoder (with `quant_conv`) and decoder (with `post_quant_conv`).
pub struct Vae {
    pub cfg: VaeConfig,
    enc: Option<Stack>,
    quant: Option<Conv>,
    dec: Stack,
    post_quant: Conv,
}

impl Vae {
    /// Loads `vae/` (config and weights); `encoder` false skips the encoder (text to image).
    pub fn load(dir: &Path, encoder: bool) -> Result<Self> {
        let cfg: VaeConfig = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("vae/config.json")?;
        if cfg.latents_mean.len() != cfg.z_dim || cfg.latents_std.len() != cfg.z_dim {
            bail!("vae config: latents_mean / latents_std do not have z_dim values");
        }
        if !cfg.attn_scales.is_empty() {
            bail!("vae attn_scales are not supported");
        }
        let file = ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| crate::fsx::exists(p))
            .with_context(|| format!("no diffusion_pytorch_model.safetensors in {}", dir.display()))?;
        let t = Tensors::open(&file)?;
        let b = cfg.base_dim;
        let mult = &cfg.dim_mult;
        let nm = mult.len();
        let dims: Vec<usize> = std::iter::once(b).chain(mult.iter().map(|m| b * m)).collect();
        let mid = |p: &str, c: usize| -> Result<Vec<Layer>> {
            Ok(vec![
                Layer::Res(Box::new(Res::load(&t, &format!("{p}.mid_block.resnets.0"), c, c)?)),
                Layer::Attn(Box::new(Attn::load(&t, &format!("{p}.mid_block.attentions.0"), c)?)),
                Layer::Res(Box::new(Res::load(&t, &format!("{p}.mid_block.resnets.1"), c, c)?)),
            ])
        };
        let enc = if encoder {
            let mut layers = Vec::new();
            let mut idx = 0;
            for i in 0..nm {
                let (mut cin, cout) = (dims[i], dims[i + 1]);
                for _ in 0..cfg.num_res_blocks {
                    layers.push(Layer::Res(Box::new(Res::load(
                        &t,
                        &format!("encoder.down_blocks.{idx}"),
                        cin,
                        cout,
                    )?)));
                    idx += 1;
                    cin = cout;
                }
                if i != nm - 1 {
                    layers.push(Layer::Down(
                        conv(&t, &format!("encoder.down_blocks.{idx}.resample.1"))?.with_padding(0, 1),
                    ));
                    idx += 1;
                }
            }
            let top = *dims.last().unwrap();
            layers.extend(mid("encoder", top)?);
            Some(Stack {
                conv_in: conv(&t, "encoder.conv_in")?,
                layers,
                norm_out: Rms::load(&t, "encoder.norm_out", top)?,
                conv_out: conv(&t, "encoder.conv_out")?,
            })
        } else {
            None
        };
        // Decoder: dims reversed, starting from the top width.
        let ddims: Vec<usize> = std::iter::once(b * mult[nm - 1]).chain(mult.iter().rev().map(|m| b * m)).collect();
        let mut layers = mid("decoder", ddims[0])?;
        for i in 0..nm {
            let mut cin = if i > 0 { ddims[i] / 2 } else { ddims[i] };
            let cout = ddims[i + 1];
            for j in 0..=cfg.num_res_blocks {
                layers.push(Layer::Res(Box::new(Res::load(
                    &t,
                    &format!("decoder.up_blocks.{i}.resnets.{j}"),
                    cin,
                    cout,
                )?)));
                cin = cout;
            }
            if i != nm - 1 {
                layers.push(Layer::Up(conv(&t, &format!("decoder.up_blocks.{i}.upsamplers.0.resample.1"))?));
            }
        }
        let dec = Stack {
            conv_in: conv(&t, "decoder.conv_in")?,
            layers,
            norm_out: Rms::load(&t, "decoder.norm_out", *ddims.last().unwrap())?,
            conv_out: conv(&t, "decoder.conv_out")?,
        };
        Ok(Self {
            enc,
            quant: if encoder { Some(conv(&t, "quant_conv")?) } else { None },
            post_quant: conv(&t, "post_quant_conv")?,
            dec,
            cfg,
        })
    }

    /// Spatial compression factor (8).
    pub fn scale(&self) -> usize {
        1 << (self.cfg.dim_mult.len() - 1)
    }

    /// Encodes an image (`[h*w, 3]` channels-last, values in [-1, 1]) into its normalized latent
    /// mean (`(mode - latents_mean) / latents_std`): `([lh*lw, z], lh, lw)`.
    pub fn encode(&self, px: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        let (Some(enc), Some(q)) = (&self.enc, &self.quant) else {
            bail!("the VAE was loaded without its encoder");
        };
        let (y, lh, lw) = enc.forward(px, h, w);
        let (m, _, _) = q.forward(&y, lh, lw, 1);
        let z = self.cfg.z_dim;
        let mut out = vec![0f32; lh * lw * z];
        let (mean, std) = (&self.cfg.latents_mean, &self.cfg.latents_std);
        for_chunks(&mut out, z, true, |p, r| {
            for c in 0..z {
                r[c] = (m[p * 2 * z + c] - mean[c]) / std[c];
            }
        });
        Ok((out, lh, lw))
    }

    /// Decodes normalized latents (`[lh*lw, z]`) into an image `[h*w, 3]` in [-1, 1].
    pub fn decode(&self, lat: &[f32], lh: usize, lw: usize) -> (Vec<f32>, usize, usize) {
        let z = self.cfg.z_dim;
        let (mean, std) = (&self.cfg.latents_mean, &self.cfg.latents_std);
        let mut x = vec![0f32; lat.len()];
        for_chunks(&mut x, z, true, |p, r| {
            for c in 0..z {
                r[c] = lat[p * z + c] * std[c] + mean[c];
            }
        });
        let (x, _, _) = self.post_quant.forward(&x, lh, lw, 1);
        let (mut y, h, w) = self.dec.forward(&x, lh, lw);
        y.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
        (y, h, w)
    }
}
