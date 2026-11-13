//! `AutoencoderKLQwenImage`: the Wan-style causal 3D-convolution VAE of Qwen-Image (16 latent
//! channels, 8x spatial compression), run on single images. For one frame every causal
//! convolution reduces to a 2D convolution with the last temporal slice of its kernel (the
//! earlier slices meet the causal zero padding), and the temporal down / upsampling layers are
//! skipped (the first chunk only primes their caches), so encoder and decoder are 2D networks:
//! RMS norms over channels (`F.normalize * sqrt(C) * gamma`), SiLU residual blocks, one
//! single-head attention in the middle block, stride-2 downsampling after a `(0, 1, 0, 1)` zero
//! pad, nearest 2x upsampling. Qwen-Image-Layered's VAE is the same network with four image
//! channels (RGBA).
//!
//! `AutoencoderKLQwenImage21` (Qwen-Image-2.1: 64 latent channels, 16x compression, RGBA, a wider
//! decoder) is the same family with `is_residual`: every level is a group of residual blocks and
//! its resampling layer, with a parameter-free shortcut around the whole group added to its
//! output: the mean over channel groups of the space-to-depth input going down (`AvgDown3D`), the
//! channel-repeated depth-to-space input going up (`DupUp3D`). On one frame the temporal factor
//! of a shortcut meets the zero frame the causal padding puts first: going down, the half of the
//! folded channels that would come from it is zero; going up, the second temporal copy is kept.

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
    /// Width of the decoder when it differs from the encoder's `base_dim` (Qwen-Image-2.1: 144).
    #[serde(default)]
    pub decoder_base_dim: Option<usize>,
    /// Residual groups with shortcuts (`AutoencoderKLQwenImage21`).
    #[serde(default)]
    pub is_residual: bool,
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

/// The shortcut around a residual group (`is_residual`), on one frame.
#[derive(Clone, Copy)]
enum Shortcut {
    /// `AvgDown3D`: space-to-depth by `fs` (and a temporal factor `ft`), then the mean over
    /// channel groups: `cin -> cout`.
    AvgDown { cin: usize, cout: usize, ft: usize, fs: usize },
    /// `DupUp3D`: channels repeated, then depth-to-space by 2 (and the temporal factor `ft`).
    DupUp { cin: usize, cout: usize, ft: usize },
}

impl Shortcut {
    /// The channel factor of the fold (temporal x spatial).
    fn factor(&self) -> usize {
        match *self {
            Shortcut::AvgDown { ft, fs, .. } => ft * fs * fs,
            Shortcut::DupUp { ft, .. } => ft * 4,
        }
    }

    /// `x` `[h*w, cin]` to `[h2*w2, cout]`.
    fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let factor = self.factor();
        match *self {
            Shortcut::AvgDown { cin, cout, ft, fs } => {
                let gs = cin * factor / cout;
                let (h2, w2) = (h / fs, w / fs);
                let mut y = vec![0f32; h2 * w2 * cout];
                let inv = 1.0 / gs as f32;
                for_chunks(&mut y, cout, true, |p, r| {
                    let (py, px) = (p / w2, p % w2);
                    for (o, out) in r.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for e in o * gs..(o + 1) * gs {
                            let (c, rem) = (e / factor, e % factor);
                            // Every temporal slot but the last is a zero frame of the causal padding.
                            if rem / (fs * fs) != ft - 1 {
                                continue;
                            }
                            let (b, d) = ((rem / fs) % fs, rem % fs);
                            acc += x[((py * fs + b) * w + px * fs + d) * cin + c];
                        }
                        *out = acc * inv;
                    }
                });
                y
            }
            Shortcut::DupUp { cin, cout, ft } => {
                let rep = cout * factor / cin;
                let w2 = 2 * w;
                let mut y = vec![0f32; 4 * h * w * cout];
                for_chunks(&mut y, cout, true, |p, r| {
                    let (py, px) = (p / w2, p % w2);
                    let at = (py / 2) * w + px / 2;
                    let src = &x[at * cin..(at + 1) * cin];
                    let sub = (py % 2) * 2 + px % 2;
                    for (o, out) in r.iter_mut().enumerate() {
                        *out = src[((o * ft + ft - 1) * 4 + sub) / rep];
                    }
                });
                y
            }
        }
    }
}

enum Layer {
    /// A residual group: its layers, plus the shortcut of the group's input.
    Group(Vec<Layer>, Shortcut),
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
            Layer::Group(layers, sc) => {
                let s = sc.forward(&x, h, w);
                let (mut y, mut h, mut w) = (x, h, w);
                for l in layers {
                    (y, h, w) = l.forward(y, h, w);
                }
                crate::kernels::add_inplace(&mut y, &s);
                (y, h, w)
            }
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
        let res = |p: String, cin: usize, cout: usize| -> Result<Layer> { Ok(Layer::Res(Box::new(Res::load(&t, &p, cin, cout)?))) };
        let td = |i: usize| cfg.temperal_downsample.get(i).copied().unwrap_or(false);
        let enc = if encoder {
            let mut layers = Vec::new();
            let mut idx = 0;
            for i in 0..nm {
                let (mut cin, cout) = (dims[i], dims[i + 1]);
                let down = i != nm - 1;
                if cfg.is_residual {
                    let sc = Shortcut::AvgDown {
                        cin,
                        cout,
                        ft: if down && td(i) { 2 } else { 1 },
                        fs: if down { 2 } else { 1 },
                    };
                    if !(cin * sc.factor()).is_multiple_of(cout) {
                        bail!("vae: encoder level {i} cannot fold {cin} channels into {cout}");
                    }
                    let mut group = Vec::new();
                    for j in 0..cfg.num_res_blocks {
                        group.push(res(format!("encoder.down_blocks.{i}.resnets.{j}"), cin, cout)?);
                        cin = cout;
                    }
                    if down {
                        group.push(Layer::Down(
                            conv(&t, &format!("encoder.down_blocks.{i}.downsampler.resample.1"))?.with_padding(0, 1),
                        ));
                    }
                    layers.push(Layer::Group(group, sc));
                    continue;
                }
                for _ in 0..cfg.num_res_blocks {
                    layers.push(res(format!("encoder.down_blocks.{idx}"), cin, cout)?);
                    idx += 1;
                    cin = cout;
                }
                if down {
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
        let db = cfg.decoder_base_dim.unwrap_or(b);
        let ddims: Vec<usize> = std::iter::once(db * mult[nm - 1])
            .chain(mult.iter().rev().map(|m| db * m))
            .collect();
        let mut layers = mid("decoder", ddims[0])?;
        for i in 0..nm {
            let mut cin = if i > 0 && !cfg.is_residual { ddims[i] / 2 } else { ddims[i] };
            let cout = ddims[i + 1];
            let up = i != nm - 1;
            let sc = Shortcut::DupUp {
                cin,
                cout,
                // `temperal_upsample` is `temperal_downsample` reversed.
                ft: if up && td(nm.saturating_sub(2 + i)) { 2 } else { 1 },
            };
            let mut group = Vec::new();
            for j in 0..=cfg.num_res_blocks {
                group.push(res(format!("decoder.up_blocks.{i}.resnets.{j}"), cin, cout)?);
                cin = cout;
            }
            if up {
                let name = if cfg.is_residual { "upsampler" } else { "upsamplers.0" };
                group.push(Layer::Up(conv(&t, &format!("decoder.up_blocks.{i}.{name}.resample.1"))?));
            }
            if cfg.is_residual && up {
                if !(cout * sc.factor()).is_multiple_of(ddims[i]) {
                    bail!("vae: decoder level {i} cannot spread {} channels over {cout}", ddims[i]);
                }
                layers.push(Layer::Group(group, sc));
            } else {
                layers.extend(group);
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

    /// Image channels the encoder takes (3, or 4 for the RGBA VAE of Qwen-Image-Layered).
    pub fn image_channels(&self) -> usize {
        self.dec.conv_out.cout
    }

    /// Encodes an image (`[h*w, 3]` channels-last, values in [-1, 1]) into its normalized latent
    /// mean (`(mode - latents_mean) / latents_std`): `([lh*lw, z], lh, lw)`.
    pub fn encode(&self, px: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        self.encode_sample(px, h, w, None)
    }

    /// [`Vae::encode`] with a sample of the posterior instead of its mode when `eps` is given:
    /// `mean + exp(logvar / 2) * eps(lh, lw)`, the noise `[lh*lw, z]` channels-last (diffusers'
    /// `latent_dist.sample()`, log-variance clamped to [-30, 20]), then normalized.
    pub fn encode_sample(
        &self,
        px: &[f32],
        h: usize,
        w: usize,
        eps: Option<&mut dyn FnMut(usize, usize) -> Result<Vec<f32>>>,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let (Some(enc), Some(q)) = (&self.enc, &self.quant) else {
            bail!("the VAE was loaded without its encoder");
        };
        if px.len() != h * w * enc.conv_in.cin {
            bail!("VAE input: {} values for {w}x{h} with {} channels", px.len(), enc.conv_in.cin);
        }
        let (y, lh, lw) = enc.forward(px, h, w);
        let (m, _, _) = q.forward(&y, lh, lw, 1);
        let z = self.cfg.z_dim;
        let eps = match eps {
            Some(f) => {
                let e = f(lh, lw)?;
                if e.len() != lh * lw * z {
                    bail!("posterior noise: {} values for a {lw}x{lh} latent", e.len());
                }
                Some(e)
            }
            None => None,
        };
        let mut out = vec![0f32; lh * lw * z];
        let (mean, std) = (&self.cfg.latents_mean, &self.cfg.latents_std);
        for_chunks(&mut out, z, true, |p, r| {
            for c in 0..z {
                let mut v = m[p * 2 * z + c];
                if let Some(e) = &eps {
                    v += (0.5 * m[p * 2 * z + z + c].clamp(-30.0, 20.0)).exp() * e[p * z + c];
                }
                r[c] = (v - mean[c]) / std[c];
            }
        });
        Ok((out, lh, lw))
    }

    /// Decodes normalized latents (`[lh*lw, z]`) into an image `[h*w, 3]` (`[h*w, 4]` for an
    /// RGBA VAE) in [-1, 1].
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
