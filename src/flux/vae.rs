//! The FLUX VAE (diffusers `AutoencoderKL` with 16 latent channels, `block_out_channels`
//! 128/256/512/512, no quant convs, `scaling_factor` 0.3611, `shift_factor` 0.1159), shared by
//! FLUX.1, FLUX.1 Kontext, Step1X-Edit, OmniGen2 and BAGEL: images to latents (`encode`, the
//! posterior mean as diffusers' `sample_mode="argmax"`) and back (`decode`), plus the 2x2
//! patch packing FLUX-style transformers read latents in.
//!
//! Activations are channels-last (`[pixels, channels]`) like the rest of `crate::sd`, whose
//! convolution / group-norm / attention ops (and their GPU paths) this reuses; the decoder is
//! [`crate::sd::vae::VaeDecoder`].

use crate::kernels::{add_inplace, for_chunks};
use crate::sd::ops::{Conv, GroupNorm, Linear, Pre, Tensors};
use crate::sd::unet::Resnet;
use crate::sd::vae::VaeDecoder;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// Single-head self-attention of the middle blocks.
struct MidAttn {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl MidAttn {
    fn load(t: &Tensors, a: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            norm: GroupNorm::load(t, &format!("{a}.group_norm"), 32, eps)?,
            q: Linear::load(t, &format!("{a}.to_q"))?,
            k: Linear::load(t, &format!("{a}.to_k"))?,
            v: Linear::load(t, &format!("{a}.to_v"))?,
            o: Linear::load(t, &format!("{a}.to_out.0"))?,
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
        // In query chunks: one head over every latent pixel (16384 at 1024x1024) would hold a
        // 1 GB score matrix (twice, with the probabilities) at once.
        let a = crate::qwen_vision::attention_chunked(&q, &k, &v, n, 1, self.q.n_out);
        add_inplace(x, &self.o.forward(&a, n));
    }
}

struct Down {
    resnets: Vec<Resnet>,
    down: Option<Conv>,
}

/// The encoder half (diffusers `Encoder`, `double_z`): RGB in [-1, 1] to the posterior's mean
/// and log-variance, `2 * latent_channels` per latent pixel.
pub struct VaeEncoder {
    conv_in: Conv,
    downs: Vec<Down>,
    mid1: Resnet,
    attn: Option<MidAttn>,
    mid2: Resnet,
    norm_out: GroupNorm,
    conv_out: Conv,
    quant: Option<Linear>,
    pub latent_channels: usize,
}

impl VaeEncoder {
    pub fn load(t: &Tensors) -> Result<Self> {
        let eps = 1e-6;
        let p = "encoder";
        let mut downs = Vec::new();
        while t.has(&format!("{p}.down_blocks.{}.resnets.0.conv1.weight", downs.len())) {
            let b = format!("{p}.down_blocks.{}", downs.len());
            let mut resnets = Vec::new();
            while t.has(&format!("{b}.resnets.{}.conv1.weight", resnets.len())) {
                resnets.push(Resnet::load(t, &format!("{b}.resnets.{}", resnets.len()), eps)?);
            }
            let down = if t.has(&format!("{b}.downsamplers.0.conv.weight")) {
                Some(Conv::load(t, &format!("{b}.downsamplers.0.conv"))?)
            } else {
                None
            };
            downs.push(Down { resnets, down });
        }
        if downs.is_empty() {
            bail!("no VAE encoder weights (encoder.down_blocks.*)");
        }
        let a = format!("{p}.mid_block.attentions.0");
        let conv_out = Conv::load(t, &format!("{p}.conv_out"))?;
        let quant = if t.has("quant_conv.weight") {
            Some(Linear::load(t, "quant_conv")?)
        } else {
            None
        };
        Ok(Self {
            conv_in: Conv::load(t, &format!("{p}.conv_in"))?,
            downs,
            mid1: Resnet::load(t, &format!("{p}.mid_block.resnets.0"), eps)?,
            attn: if t.has(&format!("{a}.to_q.weight")) {
                Some(MidAttn::load(t, &a, eps)?)
            } else {
                None
            },
            mid2: Resnet::load(t, &format!("{p}.mid_block.resnets.1"), eps)?,
            norm_out: GroupNorm::load(t, &format!("{p}.conv_norm_out"), 32, eps)?,
            latent_channels: conv_out.cout / 2,
            conv_out,
            quant,
        })
    }

    /// `x` is `[h*w, 3]` RGB in [-1, 1]; returns the posterior moments `([h/f * w/f, 2c], h/f,
    /// w/f)` (mean channels first), `f` = 2^(number of downsamplers).
    pub fn moments(&self, x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let (mut x, mut h, mut w) = self.conv_in.forward(x, h, w, 1);
        for b in &self.downs {
            for r in &b.resnets {
                x = r.forward(&x, h, w, None);
            }
            if let Some(c) = &b.down {
                (x, h, w) = downsample(c, &x, h, w);
            }
        }
        x = self.mid1.forward(&x, h, w, None);
        if let Some(a) = &self.attn {
            a.forward(&mut x, h * w);
        }
        x = self.mid2.forward(&x, h, w, None);
        let (mut y, _, _) = self.conv_out.forward_pre(&x, h, w, 1, Pre::Group(&self.norm_out, true));
        if let Some(q) = &self.quant {
            y = q.forward(&y, h * w);
        }
        (y, h, w)
    }
}

/// diffusers `Downsample2D(padding=0)`: pad one zero row / column at the bottom / right, then a
/// stride-2 3x3 convolution. Run as the regular padding-1 convolution over the input framed by
/// one more zero pixel, dropping the first output row and column (the same sums).
fn downsample(c: &Conv, x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let ch = c.cin;
    let (ph, pw) = (h + 2, w + 2);
    let mut p = vec![0f32; ph * pw * ch];
    for_chunks(&mut p, pw * ch, true, |y, row| {
        if y >= 1 && y <= h {
            row[ch..(w + 1) * ch].copy_from_slice(&x[(y - 1) * w * ch..y * w * ch]);
        }
    });
    let (o, oh, ow) = c.forward(&p, ph, pw, 2);
    let (nh, nw, co) = (oh - 1, ow - 1, c.cout);
    let mut y = vec![0f32; nh * nw * co];
    for_chunks(&mut y, nw * co, true, |r, row| {
        let s = ((r + 1) * ow + 1) * co;
        row.copy_from_slice(&o[s..s + nw * co]);
    });
    (y, nh, nw)
}

/// The autoencoder: encoder, decoder and the latent normalization.
pub struct Vae {
    pub encoder: Option<VaeEncoder>,
    pub decoder: VaeDecoder,
    pub scaling_factor: f32,
    pub shift_factor: f32,
    pub latent_channels: usize,
    /// Pixels per latent pixel along each side (8).
    pub factor: usize,
}

impl Vae {
    /// Loads a diffusers `vae/` folder (`config.json` + weights). Without `encoder` only the
    /// decoder is read (text to image).
    pub fn load(dir: &Path, encoder: bool) -> Result<Self> {
        let wp = ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors"]
            .iter()
            .map(|n| dir.join(n))
            .find(|p| crate::fsx::exists(p))
            .with_context(|| format!("no VAE weights in {}", dir.display()))?;
        let decoder = VaeDecoder::load(dir, &wp)?;
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let factor = 1 << cfg["block_out_channels"].as_array().map_or(3, |a| a.len().saturating_sub(1));
        let encoder = if encoder {
            // f32 checkpoints (Step1X-Edit's) keep their precision: the encoder's latents move
            // by 1.5% with the weights rounded to bf16.
            let mut t = Tensors::open(&wp)?;
            t.set_split_f32(true);
            Some(VaeEncoder::load(&t)?)
        } else {
            None
        };
        Ok(Self {
            scaling_factor: decoder.scaling_factor,
            shift_factor: decoder.shift_factor,
            latent_channels: decoder.latent_channels,
            factor,
            encoder,
            decoder,
        })
    }

    /// Loads the autoencoder in Black Forest Labs' original layout (`ae.safetensors` of FLUX.1,
    /// BAGEL, Step1X-Edit: `encoder.down.N.block.M`, `mid.attn_1`, ...), with FLUX.1's latent
    /// normalization (scale 0.3611, shift 0.1159). Without `encoder` only the decoder is read.
    pub fn load_bfl(path: &Path, encoder: bool) -> Result<Self> {
        let t = Tensors::open_renamed(path, bfl_to_diffusers)?;
        let levels = t
            .names()
            .filter_map(|n| n.strip_prefix("decoder.up.")?.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |m| m + 1);
        if levels == 0 || !t.has("decoder.conv_in.weight") {
            bail!("{}: not a BFL autoencoder (no decoder.up.*)", path.display());
        }
        // The decoder's levels are stored lowest resolution first (`up.0`); diffusers counts
        // from the top. Rename again now that the depth is known.
        let mut t = Tensors::open_renamed(path, |n| {
            let n = bfl_to_diffusers(n)?;
            match n.strip_prefix("decoder.up_blocks.") {
                Some(rest) => {
                    let (i, tail) = rest.split_once('.')?;
                    Some(format!("decoder.up_blocks.{}.{tail}", levels - 1 - i.parse::<usize>().ok()?))
                }
                None => Some(n),
            }
        })?;
        let (scaling_factor, shift_factor) = (0.3611, 0.1159);
        let decoder = VaeDecoder::from_tensors(&t, scaling_factor, shift_factor)?;
        // f32 checkpoints keep their encoder's precision (see `load`).
        t.set_split_f32(true);
        let encoder = if encoder { Some(VaeEncoder::load(&t)?) } else { None };
        Ok(Self {
            scaling_factor,
            shift_factor,
            latent_channels: decoder.latent_channels,
            factor: 1 << (levels - 1),
            encoder,
            decoder,
        })
    }

    /// Normalized latents (`(mean - shift) * scale`, `[h/8 * w/8, c]`) of `x` (`[h*w, 3]` RGB in
    /// [-1, 1]); returns `(latents, lh, lw)`.
    pub fn encode(&self, x: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        let enc = self.encoder.as_ref().context("the VAE was loaded without its encoder")?;
        let (m, lh, lw) = enc.moments(x, h, w);
        let c = enc.latent_channels;
        let mut z = vec![0f32; lh * lw * c];
        let (s, sh) = (self.scaling_factor, self.shift_factor);
        for_chunks(&mut z, c, true, |p, r| {
            for (i, v) in r.iter_mut().enumerate() {
                *v = (m[p * 2 * c + i] - sh) * s;
            }
        });
        Ok((z, lh, lw))
    }

    /// RGB in [-1, 1] (`[8lh * 8lw, 3]`) from normalized latents `[lh * lw, c]`.
    pub fn decode(&self, z: &[f32], lh: usize, lw: usize) -> (Vec<f32>, usize, usize) {
        self.decoder.decode(z, lh, lw)
    }
}

/// A BFL autoencoder tensor name in diffusers' `AutoencoderKL` naming (decoder levels keep the
/// BFL index; [`Vae::load_bfl`] reverses them).
fn bfl_to_diffusers(n: &str) -> Option<String> {
    let (part, rest) = n.split_once('.')?;
    if part != "encoder" && part != "decoder" {
        return None;
    }
    let block = |s: &str| {
        s.replace("nin_shortcut", "conv_shortcut")
            .replacen("block.", "resnets.", 1)
            .replacen("downsample.", "downsamplers.0.", 1)
            .replacen("upsample.", "upsamplers.0.", 1)
    };
    let mapped = if let Some(r) = rest.strip_prefix("down.") {
        format!("down_blocks.{}", block(r))
    } else if let Some(r) = rest.strip_prefix("up.") {
        format!("up_blocks.{}", block(r))
    } else if let Some(r) = rest.strip_prefix("mid.block_1.") {
        format!("mid_block.resnets.0.{r}")
    } else if let Some(r) = rest.strip_prefix("mid.block_2.") {
        format!("mid_block.resnets.1.{r}")
    } else if let Some(r) = rest.strip_prefix("mid.attn_1.") {
        let (l, tail) = r.split_once('.')?;
        let l = match l {
            "norm" => "group_norm",
            "q" => "to_q",
            "k" => "to_k",
            "v" => "to_v",
            "proj_out" => "to_out.0",
            _ => return None,
        };
        format!("mid_block.attentions.0.{l}.{tail}")
    } else if let Some(r) = rest.strip_prefix("norm_out.") {
        format!("conv_norm_out.{r}")
    } else {
        rest.to_string()
    };
    Some(format!("{part}.{mapped}"))
}

/// RGB8 to `[h*w, 3]` floats in [-1, 1] (diffusers `VaeImageProcessor`: `x / 255 * 2 - 1`).
pub fn rgb8_to_unit(rgb: &[u8]) -> Vec<f32> {
    rgb.iter().map(|&v| v as f32 / 255.0 * 2.0 - 1.0).collect()
}

/// `[-1, 1]` floats to RGB8 (diffusers `postprocess`: `(x / 2 + 0.5).clamp(0, 1)`, then
/// `(x * 255).round()` as numpy does).
pub fn unit_to_rgb8(x: &[f32]) -> Vec<u8> {
    x.iter().map(|&v| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8).collect()
}

/// 2x2 patches of a channels-last latent `[h*w, c]` as tokens `[(h/2)*(w/2), 4c]`, feature
/// `ch * 4 + dy * 2 + dx` (diffusers `_pack_latents`).
pub fn pack(z: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let (h2, w2) = (h / 2, w / 2);
    let mut t = vec![0f32; h2 * w2 * 4 * c];
    for_chunks(&mut t, 4 * c, true, |i, r| {
        let (y, x) = (i / w2, i % w2);
        for dy in 0..2 {
            for dx in 0..2 {
                let src = &z[((2 * y + dy) * w + 2 * x + dx) * c..][..c];
                for ch in 0..c {
                    r[ch * 4 + dy * 2 + dx] = src[ch];
                }
            }
        }
    });
    t
}

/// The inverse of [`pack`]: tokens `[(h/2)*(w/2), 4c]` back to `[h*w, c]`.
pub fn unpack(t: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let w2 = w / 2;
    let mut z = vec![0f32; h * w * c];
    for_chunks(&mut z, c, true, |p, r| {
        let (y, x) = (p / w, p % w);
        let tok = &t[((y / 2) * w2 + x / 2) * 4 * c..][..4 * c];
        let (dy, dx) = (y % 2, x % 2);
        for ch in 0..c {
            r[ch] = tok[ch * 4 + dy * 2 + dx];
        }
    });
    z
}

/// `[c, h, w]` (PyTorch layout) to channels-last `[h*w, c]`.
pub fn chw_to_hwc(v: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut o = vec![0f32; v.len()];
    for ch in 0..c {
        for p in 0..h * w {
            o[p * c + ch] = v[ch * h * w + p];
        }
    }
    o
}

/// Channels-last `[h*w, c]` to `[c, h, w]`.
pub fn hwc_to_chw(v: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut o = vec![0f32; v.len()];
    for ch in 0..c {
        for p in 0..h * w {
            o[ch * h * w + p] = v[p * c + ch];
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfl_names() {
        let m = |n: &str| bfl_to_diffusers(n).unwrap();
        assert_eq!(
            m("encoder.down.1.block.0.nin_shortcut.weight"),
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight"
        );
        assert_eq!(
            m("encoder.down.2.downsample.conv.bias"),
            "encoder.down_blocks.2.downsamplers.0.conv.bias"
        );
        assert_eq!(
            m("decoder.up.3.upsample.conv.weight"),
            "decoder.up_blocks.3.upsamplers.0.conv.weight"
        );
        assert_eq!(m("decoder.mid.block_2.norm1.weight"), "decoder.mid_block.resnets.1.norm1.weight");
        assert_eq!(
            m("encoder.mid.attn_1.proj_out.weight"),
            "encoder.mid_block.attentions.0.to_out.0.weight"
        );
        assert_eq!(m("decoder.mid.attn_1.norm.bias"), "decoder.mid_block.attentions.0.group_norm.bias");
        assert_eq!(m("decoder.norm_out.weight"), "decoder.conv_norm_out.weight");
        assert_eq!(m("encoder.conv_in.weight"), "encoder.conv_in.weight");
        assert!(bfl_to_diffusers("vae2llm.weight").is_none());
    }

    #[test]
    fn pack_roundtrip() {
        let (h, w, c) = (4, 6, 3);
        let z: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        let t = pack(&z, h, w, c);
        // Token 0, channel 1: pixels (0,0), (0,1), (1,0), (1,1) of channel 1.
        assert_eq!(&t[4..8], &[1.0, 4.0, 19.0, 22.0]);
        assert_eq!(unpack(&t, h, w, c), z);
        assert_eq!(chw_to_hwc(&hwc_to_chw(&z, c, h, w), c, h, w), z);
    }
}
