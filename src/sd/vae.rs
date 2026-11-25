//! The VAE (diffusers `AutoencoderKL`): the decoder turns latents back into RGB at 8x the
//! resolution; the encoder (for image-to-image, inpainting and instruction edits) maps RGB to the
//! latent distribution.

use super::ops::{Conv, GroupNorm, Linear, Pre, Rng, Tensors, upsample2x};
use super::unet::Resnet;
use crate::kernels::add_inplace;
use anyhow::Result;
use std::path::Path;

/// The single-head self-attention in the decoder's middle block.
struct MidAttn {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl MidAttn {
    fn load(t: &Tensors, a: &str) -> Result<Self> {
        Ok(Self {
            norm: GroupNorm::load(t, &format!("{a}.group_norm"), 32, 1e-6)?,
            q: Linear::load(t, &format!("{a}.to_q"))?,
            k: Linear::load(t, &format!("{a}.to_k"))?,
            v: Linear::load(t, &format!("{a}.to_v"))?,
            o: Linear::load(t, &format!("{a}.to_out.0"))?,
        })
    }

    /// `x += attn(norm(x))` over the `n` pixels.
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
        let c = self.q.n_out;
        // In query chunks (a 1 GB score matrix at 1024x1024 otherwise).
        let a = crate::qwen_vision::attention_chunked(&q, &k, &v, n, 1, c);
        add_inplace(x, &self.o.forward(&a, n));
    }
}

struct Up {
    resnets: Vec<Resnet>,
    up: Option<Conv>,
}

pub struct VaeDecoder {
    post_quant: Option<Linear>,
    conv_in: Conv,
    mid1: Resnet,
    attn: MidAttn,
    mid2: Resnet,
    ups: Vec<Up>,
    norm_out: GroupNorm,
    conv_out: Conv,
    pub scaling_factor: f32,
    pub shift_factor: f32,
    pub latent_channels: usize,
    /// The whole-pass device session ([`crate::gpu::unet`]), once made.
    gpu: std::sync::Mutex<Option<Box<dyn crate::gpu::unet::Session>>>,
    /// Whether [`VaeDecoder::decode`] tries the device plan (default on, or
    /// `OXINFER_SD_GPU_PASS=0` for off).
    gpu_pass: std::sync::atomic::AtomicBool,
}

impl VaeDecoder {
    pub fn load(dir: &Path, weights: &Path) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        // Older configs (SD 1.x) omit it; diffusers' default is the SD 1/2 VAE's 0.18215.
        Self::from_tensors(
            &Tensors::open(weights)?,
            cfg["scaling_factor"].as_f64().unwrap_or(0.18215) as f32,
            cfg["shift_factor"].as_f64().unwrap_or(0.0) as f32,
        )
    }

    /// The decoder from diffusers-named weights (`decoder.*`, `post_quant_conv.*`).
    pub fn from_tensors(t: &Tensors, scaling_factor: f32, shift_factor: f32) -> Result<Self> {
        let eps = 1e-6;
        let p = "decoder";
        let mut ups = Vec::new();
        while t.has(&format!("{p}.up_blocks.{}.resnets.0.conv1.weight", ups.len())) {
            let b = format!("{p}.up_blocks.{}", ups.len());
            let mut resnets = Vec::new();
            while t.has(&format!("{b}.resnets.{}.conv1.weight", resnets.len())) {
                resnets.push(Resnet::load(t, &format!("{b}.resnets.{}", resnets.len()), eps)?);
            }
            let up = if t.has(&format!("{b}.upsamplers.0.conv.weight")) {
                Some(Conv::load(t, &format!("{b}.upsamplers.0.conv"))?)
            } else {
                None
            };
            ups.push(Up { resnets, up });
        }
        let conv_in = Conv::load(t, &format!("{p}.conv_in"))?;
        Ok(Self {
            post_quant: if t.has("post_quant_conv.weight") {
                Some(Linear::load(t, "post_quant_conv")?)
            } else {
                None
            },
            latent_channels: conv_in.cin,
            conv_in,
            mid1: Resnet::load(t, &format!("{p}.mid_block.resnets.0"), eps)?,
            attn: MidAttn::load(t, &format!("{p}.mid_block.attentions.0"))?,
            mid2: Resnet::load(t, &format!("{p}.mid_block.resnets.1"), eps)?,
            ups,
            norm_out: GroupNorm::load(t, &format!("{p}.conv_norm_out"), 32, eps)?,
            conv_out: Conv::load(t, &format!("{p}.conv_out"))?,
            scaling_factor,
            shift_factor,
            gpu: std::sync::Mutex::new(None),
            gpu_pass: std::sync::atomic::AtomicBool::new(std::env::var("OXINFER_SD_GPU_PASS").map_or(true, |v| v != "0")),
        })
    }

    /// Whether decoding runs as one device plan when the accelerator has one (default on).
    pub fn set_gpu_pass(&self, on: bool) {
        self.gpu_pass.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// `latent` (`[h*w, latent_channels]`, as produced by the denoiser) to RGB in [-1, 1]
    /// (`[8h * 8w, 3]`); returns `(pixels, height, width)`.
    pub fn decode(&self, latent: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let mut z: Vec<f32> = latent.iter().map(|&v| v / self.scaling_factor + self.shift_factor).collect();
        if let Some(pq) = &self.post_quant {
            z = pq.forward(&z, h * w);
        }
        if self.gpu_pass.load(std::sync::atomic::Ordering::Relaxed)
            && let Some(r) = self.decode_device(&z, h, w)
        {
            return r;
        }
        let (mut x, mut h, mut w) = self.conv_in.forward(&z, h, w, 1);
        x = self.mid1.forward(&x, h, w, None);
        self.attn.forward(&mut x, h * w);
        x = self.mid2.forward(&x, h, w, None);
        let mut c = self.conv_in.cout;
        for b in &self.ups {
            for r in &b.resnets {
                x = r.forward(&x, h, w, None);
                c = r.cout;
            }
            if let Some(u) = &b.up {
                let up = upsample2x(&x, h, w, c);
                (h, w) = (h * 2, w * 2);
                x = u.forward(&up, h, w, 1).0;
            }
        }
        (self.conv_out.forward_pre(&x, h, w, 1, Pre::Group(&self.norm_out, true)).0, h, w)
    }
}

impl VaeDecoder {
    /// [`VaeDecoder::decode`] of the (scaled) latents `z` as one device plan.
    fn decode_device(&self, z: &[f32], h: usize, w: usize) -> Option<(Vec<f32>, usize, usize)> {
        use super::unet::{conv_op, new_tensor};
        use crate::gpu::unet::{Graph, Item, Op};
        let be = crate::gpu::active()?;
        if super::ops::gpu_check::on() || crate::gpu::exact() {
            return None;
        }
        let _t = super::ops::timing::start(super::ops::timing::PASS);
        let mut g = Graph {
            tensors: vec![[h, w, self.latent_channels]],
            ops: Vec::new(),
            input: 0,
            output: 0,
            ctx_dim: 0,
        };
        let (mut h, mut w) = (h, w);
        let mut x = new_tensor(&mut g, [h, w, self.conv_in.cout]);
        g.ops.push(conv_op(&self.conv_in, 0, x, None, None, false)?);
        x = self.mid1.plan(&mut g, x, &[], 1)?;
        let a = &self.attn;
        let s = g.tensors[x];
        let y = new_tensor(&mut g, s);
        g.ops.push(Op::Attention {
            x,
            out: y,
            norm: a.norm.plan_norm(),
            a: crate::gpu::unet::Attn {
                q: a.q.plan_lin()?,
                k: a.k.plan_lin()?,
                v: a.v.plan_lin()?,
                o: a.o.plan_lin()?,
                heads: 1,
            },
        });
        x = self.mid2.plan(&mut g, y, &[], 1)?;
        for b in &self.ups {
            for r in &b.resnets {
                x = r.plan(&mut g, x, &[], 1)?;
            }
            if let Some(u) = &b.up {
                let c = g.tensors[x][2];
                (h, w) = (h * 2, w * 2);
                let y = new_tensor(&mut g, [h, w, c]);
                g.ops.push(conv_op(u, x, y, None, None, true)?);
                x = y;
            }
        }
        let y = new_tensor(&mut g, [h, w, self.conv_out.cout]);
        g.ops
            .push(conv_op(&self.conv_out, x, y, Some((self.norm_out.plan_norm(), true)), None, false)?);
        g.output = y;
        let mut sess = self.gpu.lock().unwrap_or_else(|e| e.into_inner());
        if sess.is_none() {
            *sess = Some(be.unet_session()?);
        }
        let item = Item {
            latent: z,
            ctx: &[],
            ctx_len: 0,
        };
        let mut out = vec![Vec::new()];
        sess.as_mut()?
            .forward(&g, std::slice::from_ref(&item), &mut out)
            .then(|| (out.pop().unwrap(), h, w))
    }
}

struct Down {
    resnets: Vec<Resnet>,
    down: Option<Conv>,
}

/// The VAE encoder (diffusers `AutoencoderKL.encode`): RGB in [-1, 1] to the diagonal Gaussian
/// over latents (mean and log-variance) at 1/8 of the resolution.
pub struct VaeEncoder {
    conv_in: Conv,
    downs: Vec<Down>,
    mid1: Resnet,
    attn: MidAttn,
    mid2: Resnet,
    norm_out: GroupNorm,
    conv_out: Conv,
    quant: Option<Linear>,
    pub latent_channels: usize,
}

/// The encoder's output: the latent distribution's mean and standard deviation, channels-last
/// `[h*w, latent_channels]` (unscaled: multiply by the VAE's `scaling_factor` for the denoiser).
pub struct LatentDist {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

impl LatentDist {
    /// `mean + std * noise` with `noise` standard normal of the same layout
    /// (`latent_dist.sample()`).
    pub fn sample_with(&self, noise: &[f32]) -> Vec<f32> {
        self.mean.iter().zip(&self.std).zip(noise).map(|((&m, &s), &e)| m + s * e).collect()
    }

    /// A sample drawn with `rng`.
    pub fn sample(&self, rng: &mut Rng) -> Vec<f32> {
        self.sample_with(&rng.normals(self.mean.len()))
    }
}

impl VaeEncoder {
    /// `None` when the weight file has no encoder (decoder-only exports).
    pub fn load(weights: &Path) -> Result<Option<Self>> {
        let t = &Tensors::open(weights)?;
        let p = "encoder";
        if !t.has(&format!("{p}.conv_in.weight")) {
            return Ok(None);
        }
        let eps = 1e-6;
        let mut downs = Vec::new();
        while t.has(&format!("{p}.down_blocks.{}.resnets.0.conv1.weight", downs.len())) {
            let b = format!("{p}.down_blocks.{}", downs.len());
            let mut resnets = Vec::new();
            while t.has(&format!("{b}.resnets.{}.conv1.weight", resnets.len())) {
                resnets.push(Resnet::load(t, &format!("{b}.resnets.{}", resnets.len()), eps)?);
            }
            // `Downsample2D(padding=0)`: one zero row and column after the input, then stride 2.
            let down = if t.has(&format!("{b}.downsamplers.0.conv.weight")) {
                Some(Conv::load_padded(t, &format!("{b}.downsamplers.0.conv"), 0, 1)?)
            } else {
                None
            };
            downs.push(Down { resnets, down });
        }
        let conv_out = Conv::load(t, &format!("{p}.conv_out"))?;
        Ok(Some(Self {
            conv_in: Conv::load(t, &format!("{p}.conv_in"))?,
            downs,
            mid1: Resnet::load(t, &format!("{p}.mid_block.resnets.0"), eps)?,
            attn: MidAttn::load(t, &format!("{p}.mid_block.attentions.0"))?,
            mid2: Resnet::load(t, &format!("{p}.mid_block.resnets.1"), eps)?,
            norm_out: GroupNorm::load(t, &format!("{p}.conv_norm_out"), 32, eps)?,
            latent_channels: conv_out.cout / 2,
            conv_out,
            quant: if t.has("quant_conv.weight") {
                Some(Linear::load(t, "quant_conv")?)
            } else {
                None
            },
        }))
    }

    /// `rgb` (`[h*w, 3]` in [-1, 1]; `h`, `w` multiples of 8) to the latent distribution.
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize) -> LatentDist {
        let (mut x, mut h, mut w) = self.conv_in.forward(rgb, h, w, 1);
        for b in &self.downs {
            for r in &b.resnets {
                x = r.forward(&x, h, w, None);
            }
            if let Some(d) = &b.down {
                (x, h, w) = d.forward(&x, h, w, 2);
            }
        }
        x = self.mid1.forward(&x, h, w, None);
        self.attn.forward(&mut x, h * w);
        x = self.mid2.forward(&x, h, w, None);
        let (mut m, _, _) = self.conv_out.forward_pre(&x, h, w, 1, Pre::Group(&self.norm_out, true));
        if let Some(q) = &self.quant {
            m = q.forward(&m, h * w);
        }
        let c = self.latent_channels;
        let mut mean = Vec::with_capacity(h * w * c);
        let mut std = Vec::with_capacity(h * w * c);
        for px in m.chunks_exact(2 * c) {
            mean.extend_from_slice(&px[..c]);
            std.extend(px[c..].iter().map(|&lv| (0.5 * lv.clamp(-30.0, 20.0)).exp()));
        }
        LatentDist { mean, std, h, w }
    }
}
