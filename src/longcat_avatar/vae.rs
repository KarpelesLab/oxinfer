//! `AutoencoderKLWan` (Wan 2.1's video VAE, the one LongCat-Video ships in `vae/`): causal 3D
//! convolutions, 8x in space and 4x in time (`1 + 4 n` frames to `1 + n` latent frames of 16
//! channels), run on whole clips the way diffusers runs it: the encoder takes the first frame,
//! then four frames at a time; the decoder one latent frame at a time; every causal convolution
//! keeps the last two frames of its input from one chunk to the next instead of seeing the
//! whole clip (`feat_cache`). The temporal resampling layers skip their time convolution on the
//! first chunk, which is what makes the first frame a latent frame of its own.
//!
//! A causal convolution `[cout, cin, kt, k, k]` runs here as one 2D convolution over the `kt`
//! input frames laid side by side in the channels (`[cout, kt * cin, k, k]`), so it goes through
//! [`crate::sd::ops::Conv`] (and to the accelerator when one is active). Frames are
//! channels-last `[h * w, c]`. On CUDA the same passes run with the frames kept on the device
//! (`vae/dev.rs`), the code below being what they are checked against and the fallback.
//!
//! [`crate::qwen_image::vae`] is the same network collapsed to a single frame.

use crate::kernels::for_chunks;
use crate::qwen_image::vae::VaeConfig;
use crate::sd::ops::{Conv, Tensors, silu};
use anyhow::{Context, Result, bail};
use std::path::Path;

#[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
mod dev;

type Frame = Vec<f32>;

/// The device engine of a VAE: decided at the first pass.
#[derive(Default)]
struct Device {
    #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
    engine: std::sync::OnceLock<Option<dev::Engine>>,
}

/// A bias or a norm gain, rounded to bfloat16 as the kernels are: the reference loads the whole
/// VAE in bfloat16, and with these left as stored the released VAE's latents sit 1.7e-3 (mean)
/// away from the reference's float32 run on its rounded weights, instead of 7.8e-7.
pub(crate) fn rounded(t: &Tensors, name: &str) -> Result<Vec<f32>> {
    let mut v = t.f32(name)?;
    for x in &mut v {
        *x = f32::from_bits((crate::sd::ops::f32_to_bf16(*x) as u32) << 16);
    }
    Ok(v)
}

/// Channel RMS norm: `x / max(|x|, 1e-12) * sqrt(C) * gamma` per pixel.
struct Rms {
    g: Vec<f32>,
}

impl Rms {
    fn load(t: &Tensors, name: &str, c: usize) -> Result<Self> {
        let g = rounded(t, &format!("{name}.gamma"))?;
        if g.len() != c {
            bail!("{name}.gamma has {} values, expected {c}", g.len());
        }
        let s = (c as f32).sqrt();
        Ok(Self {
            g: g.into_iter().map(|v| v * s).collect(),
        })
    }

    fn apply(&self, x: &[f32], silu_after: bool) -> Frame {
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

/// What a causal convolution keeps of its previous input.
#[derive(Default)]
enum Cache {
    #[default]
    None,
    /// The first chunk went through an upsampling layer without its time convolution.
    Rep,
    Frames(Vec<Frame>),
}

/// `WanCausalConv3d`.
struct CConv {
    conv: Conv,
    kt: usize,
    cin: usize,
    /// Zero frames in front of the input (twice the layer's temporal padding).
    pad_t: usize,
    stride_t: usize,
}

impl CConv {
    /// `{name}.weight` (`[cout, cin, kt, k, k]`) and `.bias`.
    fn load(t: &Tensors, name: &str, pad_t: usize, stride_t: usize) -> Result<Self> {
        let wn = format!("{name}.weight");
        let shape = t.shape(&wn)?;
        let [cout, cin, kt, kh, kw] = shape[..] else {
            bail!("{wn}: expected a 3D kernel, got {shape:?}");
        };
        if kh != kw {
            bail!("{wn}: kernel {kh}x{kw}");
        }
        let w = t.f32(&wn)?;
        let sl = kh * kw;
        let mut w2 = vec![0f32; cout * kt * cin * sl];
        for oc in 0..cout {
            for ic in 0..cin {
                for j in 0..kt {
                    let src = ((oc * cin + ic) * kt + j) * sl;
                    let dst = (oc * kt * cin + j * cin + ic) * sl;
                    w2[dst..dst + sl].copy_from_slice(&w[src..src + sl]);
                }
            }
        }
        let bias = rounded(t, &format!("{name}.bias")).ok();
        Ok(Self {
            conv: Conv::from_kernel(&w2, cout, kt * cin, kh, bias).with_context(|| name.to_string())?,
            kt,
            cin,
            pad_t,
            stride_t,
        })
    }

    /// The convolution of `[zero frames] ++ cache ++ x`.
    fn forward(&self, x: &[Frame], cache: Option<&[Frame]>, h: usize, w: usize) -> Vec<Frame> {
        let cin = self.cin;
        if self.kt == 1 {
            return x.iter().map(|f| self.conv.forward(f, h, w, 1).0).collect();
        }
        let cached = if self.pad_t > 0 { cache.unwrap_or(&[]) } else { &[] };
        let pad = self.pad_t.saturating_sub(cached.len());
        let seq: Vec<Option<&Frame>> = (0..pad)
            .map(|_| None)
            .chain(cached.iter().map(Some))
            .chain(x.iter().map(Some))
            .collect();
        let n_out = if seq.len() >= self.kt {
            (seq.len() - self.kt) / self.stride_t + 1
        } else {
            0
        };
        let width = self.kt * cin;
        let mut out = Vec::with_capacity(n_out);
        let mut cat = vec![0f32; h * w * width];
        for o in 0..n_out {
            let taps = &seq[o * self.stride_t..o * self.stride_t + self.kt];
            for_chunks(&mut cat, width * 64, true, |i, r| {
                for (p, px) in r.chunks_mut(width).enumerate() {
                    let at = (i * 64 + p) * cin;
                    for (j, tap) in taps.iter().enumerate() {
                        match tap {
                            Some(f) => px[j * cin..(j + 1) * cin].copy_from_slice(&f[at..at + cin]),
                            None => px[j * cin..(j + 1) * cin].fill(0.0),
                        }
                    }
                }
            });
            out.push(self.conv.forward(&cat, h, w, 1).0);
        }
        out
    }

    /// [`Self::forward`] with the layer's cache: the last two frames of the input (one of them
    /// from the cache when the chunk has a single frame) replace it.
    fn cached(&self, x: &[Frame], cache: &mut Cache, h: usize, w: usize) -> Vec<Frame> {
        let prev = match &*cache {
            Cache::Frames(f) => Some(&f[..]),
            _ => None,
        };
        let y = self.forward(x, prev, h, w);
        let keep = tail2(x, prev);
        *cache = Cache::Frames(keep);
        y
    }
}

/// The last two frames of `x`, with the last cached frame in front when `x` has only one.
fn tail2(x: &[Frame], prev: Option<&[Frame]>) -> Vec<Frame> {
    let mut keep: Vec<Frame> = x[x.len().saturating_sub(2)..].to_vec();
    if keep.len() < 2
        && let Some(last) = prev.and_then(|p| p.last())
    {
        keep.insert(0, last.clone());
    }
    keep
}

struct Res {
    norm1: Rms,
    conv1: CConv,
    norm2: Rms,
    conv2: CConv,
    shortcut: Option<CConv>,
}

impl Res {
    fn load(t: &Tensors, p: &str, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self {
            norm1: Rms::load(t, &format!("{p}.norm1"), cin)?,
            conv1: CConv::load(t, &format!("{p}.conv1"), 2, 1)?,
            norm2: Rms::load(t, &format!("{p}.norm2"), cout)?,
            conv2: CConv::load(t, &format!("{p}.conv2"), 2, 1)?,
            shortcut: if cin != cout {
                Some(CConv::load(t, &format!("{p}.conv_shortcut"), 0, 1)?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: Vec<Frame>, st: &mut State, h: usize, w: usize) -> Vec<Frame> {
        let sc: Option<Vec<Frame>> = self.shortcut.as_ref().map(|c| c.forward(&x, None, h, w));
        let a: Vec<Frame> = x.iter().map(|f| self.norm1.apply(f, true)).collect();
        let a = self.conv1.cached(&a, st.next(), h, w);
        let a: Vec<Frame> = a.iter().map(|f| self.norm2.apply(f, true)).collect();
        let mut a = self.conv2.cached(&a, st.next(), h, w);
        let sc = sc.as_ref().unwrap_or(&x);
        for (f, s) in a.iter_mut().zip(sc) {
            crate::kernels::add_inplace(f, s);
        }
        a
    }
}

/// Single-head attention over the pixels of each frame.
struct Attn {
    norm: Rms,
    qkv: Conv,
    proj: Conv,
    c: usize,
}

fn conv2d(t: &Tensors, name: &str) -> Result<Conv> {
    let wn = format!("{name}.weight");
    let shape = t.shape(&wn)?;
    let [cout, cin, kh, kw] = shape[..] else {
        bail!("{wn}: expected a 2D kernel, got {shape:?}");
    };
    if kh != kw {
        bail!("{wn}: kernel {kh}x{kw}");
    }
    Conv::from_kernel(&t.f32(&wn)?, cout, cin, kh, rounded(t, &format!("{name}.bias")).ok()).with_context(|| name.to_string())
}

impl Attn {
    fn load(t: &Tensors, p: &str, c: usize) -> Result<Self> {
        Ok(Self {
            norm: Rms::load(t, &format!("{p}.norm"), c)?,
            qkv: conv2d(t, &format!("{p}.to_qkv"))?,
            proj: conv2d(t, &format!("{p}.proj"))?,
            c,
        })
    }

    fn forward(&self, x: &[f32], h: usize, w: usize) -> Frame {
        let c = self.c;
        let n = h * w;
        let xn = self.norm.apply(x, false);
        let (qkv, _, _) = self.qkv.forward(&xn, h, w, 1);
        let (mut q, mut k, mut v) = (vec![0f32; n * c], vec![0f32; n * c], vec![0f32; n * c]);
        for (buf, o) in [(&mut q, 0), (&mut k, c), (&mut v, 2 * c)] {
            for_chunks(buf, c, true, |i, r| r.copy_from_slice(&qkv[i * 3 * c + o..i * 3 * c + o + c]));
        }
        let a = crate::sd::ops::attention(&q, &k, &v, n, n, 1, c, false);
        let (mut y, _, _) = self.proj.forward(&a, h, w, 1);
        crate::kernels::add_inplace(&mut y, x);
        y
    }
}

enum Layer {
    Res(Box<Res>),
    Attn(Box<Attn>),
    /// Stride-2 convolution after a right / bottom zero pad, then (3D) a stride-2 time convolution
    /// over the frames.
    Down(Conv, Option<CConv>),
    /// (3D) a time convolution to twice the channels, its halves interleaved as frames; then
    /// nearest 2x upsampling and a 3x3 convolution.
    Up(Conv, Option<CConv>),
}

/// The caches of one pass over a clip, in the order the layers use them.
struct State {
    caches: Vec<Cache>,
    at: usize,
}

impl State {
    fn next(&mut self) -> &mut Cache {
        if self.at == self.caches.len() {
            self.caches.push(Cache::None);
        }
        self.at += 1;
        &mut self.caches[self.at - 1]
    }
}

impl Layer {
    fn forward(&self, x: Vec<Frame>, st: &mut State, h: usize, w: usize) -> (Vec<Frame>, usize, usize) {
        match self {
            Layer::Res(r) => (r.forward(x, st, h, w), h, w),
            Layer::Attn(a) => (x.iter().map(|f| a.forward(f, h, w)).collect(), h, w),
            Layer::Down(c, time) => {
                let mut oh = h;
                let mut ow = w;
                let y: Vec<Frame> = x
                    .iter()
                    .map(|f| {
                        let (y, a, b) = c.forward(f, h, w, 2);
                        (oh, ow) = (a, b);
                        y
                    })
                    .collect();
                let Some(tc) = time else { return (y, oh, ow) };
                let cache = st.next();
                match cache {
                    Cache::Frames(prev) => {
                        let mut seq = vec![prev.last().unwrap().clone()];
                        seq.extend(y);
                        let keep = seq.last().unwrap().clone();
                        let out = tc.forward(&seq, None, oh, ow);
                        *cache = Cache::Frames(vec![keep]);
                        (out, oh, ow)
                    }
                    _ => {
                        *cache = Cache::Frames(y.clone());
                        (y, oh, ow)
                    }
                }
            }
            Layer::Up(c, time) => {
                let mut x = x;
                if let Some(tc) = time {
                    let cache = st.next();
                    match cache {
                        Cache::None => *cache = Cache::Rep,
                        _ => {
                            let prev = match &*cache {
                                Cache::Frames(f) => Some(&f[..]),
                                _ => None,
                            };
                            let mut keep = tail2(&x, prev);
                            if keep.len() < 2 && prev.is_none() {
                                keep.insert(0, vec![0f32; keep[0].len()]);
                            }
                            let y = tc.forward(&x, prev, h, w);
                            *cache = Cache::Frames(keep);
                            // Channels `[0, C)` are the first of each pair of frames, `[C, 2C)` the second.
                            let ch = tc.cin;
                            let mut out = Vec::with_capacity(2 * y.len());
                            for f in &y {
                                for half in 0..2 {
                                    let mut g = vec![0f32; h * w * ch];
                                    for_chunks(&mut g, ch, true, |p, r| {
                                        r.copy_from_slice(&f[p * 2 * ch + half * ch..p * 2 * ch + (half + 1) * ch])
                                    });
                                    out.push(g);
                                }
                            }
                            x = out;
                        }
                    }
                }
                let y = x
                    .iter()
                    .map(|f| {
                        let u = crate::sd::ops::upsample2x(f, h, w, c.cin);
                        c.forward(&u, 2 * h, 2 * w, 1).0
                    })
                    .collect();
                (y, 2 * h, 2 * w)
            }
        }
    }
}

struct Stack {
    conv_in: CConv,
    layers: Vec<Layer>,
    norm_out: Rms,
    conv_out: CConv,
}

impl Stack {
    /// One chunk of frames through the stack.
    fn forward(&self, x: &[Frame], st: &mut State, h: usize, w: usize) -> (Vec<Frame>, usize, usize) {
        st.at = 0;
        let (mut x, mut h, mut w) = (self.conv_in.cached(x, st.next(), h, w), h, w);
        for l in &self.layers {
            (x, h, w) = l.forward(x, st, h, w);
        }
        let a: Vec<Frame> = x.iter().map(|f| self.norm_out.apply(f, true)).collect();
        (self.conv_out.cached(&a, st.next(), h, w), h, w)
    }
}

/// The video VAE.
pub struct VideoVae {
    pub cfg: VaeConfig,
    enc: Option<Stack>,
    quant: Option<Conv>,
    dec: Stack,
    post_quant: Conv,
    // (Read by the CUDA passes only.)
    #[cfg_attr(
        not(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32")))),
        allow(dead_code)
    )]
    device: Device,
}

/// A 1x1x1 convolution as a 2D one.
fn conv111(t: &Tensors, name: &str) -> Result<Conv> {
    let wn = format!("{name}.weight");
    let shape = t.shape(&wn)?;
    let (cout, cin) = (shape[0], shape[1]);
    Conv::from_kernel(&t.f32(&wn)?, cout, cin, 1, rounded(t, &format!("{name}.bias")).ok()).with_context(|| name.to_string())
}

impl VideoVae {
    /// Loads `vae/` (config and `diffusion_pytorch_model.safetensors`); `encoder` false skips the
    /// encoder (text to video). Every weight (kernels, biases, norm gains) is rounded to bfloat16,
    /// which is what the reference runs this VAE at (activations included; here they stay float32).
    pub fn load(dir: &Path, encoder: bool) -> Result<Self> {
        let cfg: VaeConfig = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("vae/config.json")?;
        if cfg.latents_mean.len() != cfg.z_dim || cfg.latents_std.len() != cfg.z_dim {
            bail!("vae config: latents_mean / latents_std do not have z_dim values");
        }
        if !cfg.attn_scales.is_empty() || cfg.is_residual || cfg.decoder_base_dim.is_some() {
            bail!("this video VAE layout (attn_scales, is_residual or decoder_base_dim) is not supported");
        }
        if cfg.temperal_downsample.len() + 1 != cfg.dim_mult.len() {
            bail!("vae config: temperal_downsample does not match dim_mult");
        }
        let t = Tensors::open(&dir.join("diffusion_pytorch_model.safetensors"))?;
        let levels = cfg.dim_mult.len();
        let enc = if encoder {
            let dims: Vec<usize> = std::iter::once(cfg.base_dim)
                .chain(cfg.dim_mult.iter().map(|m| cfg.base_dim * m))
                .collect();
            let mut layers = Vec::new();
            let mut idx = 0;
            for i in 0..levels {
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
                if i != levels - 1 {
                    let p = format!("encoder.down_blocks.{idx}");
                    let conv = conv2d(&t, &format!("{p}.resample.1"))?.with_padding(0, 1);
                    let time = if cfg.temperal_downsample[i] {
                        Some(CConv::load(&t, &format!("{p}.time_conv"), 0, 2)?)
                    } else {
                        None
                    };
                    layers.push(Layer::Down(conv, time));
                    idx += 1;
                }
            }
            let top = dims[levels];
            layers.push(Layer::Res(Box::new(Res::load(&t, "encoder.mid_block.resnets.0", top, top)?)));
            layers.push(Layer::Attn(Box::new(Attn::load(&t, "encoder.mid_block.attentions.0", top)?)));
            layers.push(Layer::Res(Box::new(Res::load(&t, "encoder.mid_block.resnets.1", top, top)?)));
            Some(Stack {
                conv_in: CConv::load(&t, "encoder.conv_in", 2, 1)?,
                layers,
                norm_out: Rms::load(&t, "encoder.norm_out", top)?,
                conv_out: CConv::load(&t, "encoder.conv_out", 2, 1)?,
            })
        } else {
            None
        };
        let quant = if encoder { Some(conv111(&t, "quant_conv")?) } else { None };

        // Decoder: dims [top, top, ..., base], the temporal upsampling in the encoder's order reversed.
        let rev: Vec<usize> = cfg.dim_mult.iter().rev().map(|m| cfg.base_dim * m).collect();
        let dims: Vec<usize> = std::iter::once(rev[0]).chain(rev.iter().copied()).collect();
        let t_up: Vec<bool> = cfg.temperal_downsample.iter().rev().copied().collect();
        let top = dims[0];
        let mut layers = vec![
            Layer::Res(Box::new(Res::load(&t, "decoder.mid_block.resnets.0", top, top)?)),
            Layer::Attn(Box::new(Attn::load(&t, "decoder.mid_block.attentions.0", top)?)),
            Layer::Res(Box::new(Res::load(&t, "decoder.mid_block.resnets.1", top, top)?)),
        ];
        let mut last = top;
        for i in 0..levels {
            let (mut cin, cout) = (dims[i], dims[i + 1]);
            if i > 0 {
                cin /= 2;
            }
            for j in 0..cfg.num_res_blocks + 1 {
                layers.push(Layer::Res(Box::new(Res::load(
                    &t,
                    &format!("decoder.up_blocks.{i}.resnets.{j}"),
                    cin,
                    cout,
                )?)));
                cin = cout;
            }
            if i != levels - 1 {
                let p = format!("decoder.up_blocks.{i}.upsamplers.0");
                let conv = conv2d(&t, &format!("{p}.resample.1"))?;
                let time = if t_up[i] {
                    Some(CConv::load(&t, &format!("{p}.time_conv"), 2, 1)?)
                } else {
                    None
                };
                layers.push(Layer::Up(conv, time));
            }
            last = cout;
        }
        Ok(Self {
            enc,
            quant,
            dec: Stack {
                conv_in: CConv::load(&t, "decoder.conv_in", 2, 1)?,
                layers,
                norm_out: Rms::load(&t, "decoder.norm_out", last)?,
                conv_out: CConv::load(&t, "decoder.conv_out", 2, 1)?,
            },
            post_quant: conv111(&t, "post_quant_conv")?,
            cfg,
            device: Device::default(),
        })
    }

    /// Spatial compression.
    pub fn scale(&self) -> usize {
        1 << (self.cfg.dim_mult.len() - 1)
    }

    /// Temporal compression.
    pub fn scale_t(&self) -> usize {
        1 << self.cfg.temperal_downsample.iter().filter(|&&d| d).count()
    }

    /// The posterior mean of a clip: `frames` are RGB in `[-1, 1]`, channels-last `[h * w, 3]`,
    /// `1 + 4 n` of them (a single image is a clip of one frame). Returns the latents
    /// `[z, 1 + n, h / 8, w / 8]`, channel-first, not normalized.
    pub fn encode(&self, frames: &[Frame], h: usize, w: usize) -> Result<Vec<f32>> {
        let (Some(enc), Some(quant)) = (&self.enc, &self.quant) else {
            bail!("the VAE was loaded without its encoder");
        };
        let st_t = self.scale_t();
        if frames.is_empty()
            || !(frames.len() - 1).is_multiple_of(st_t)
            || !h.is_multiple_of(self.scale())
            || !w.is_multiple_of(self.scale())
        {
            bail!(
                "the VAE encodes 1 + {st_t} n frames with sides multiples of {} (got {} frames of {w}x{h})",
                self.scale(),
                frames.len()
            );
        }
        let mut out: Vec<Frame> = Vec::new();
        let (mut lh, mut lw) = (0, 0);
        #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
        if let Some(e) = self.device.engine.get_or_init(|| dev::Engine::new(dev::OFF))
            && let Some((y, a, b)) = dev::encode(e, self, frames, h, w)
        {
            (out, lh, lw) = (y, a, b);
        }
        if out.is_empty() {
            let mut st = State { caches: Vec::new(), at: 0 };
            let mut at = 0;
            while at < frames.len() {
                let n = if at == 0 { 1 } else { st_t };
                let (y, a, b) = enc.forward(&frames[at..at + n], &mut st, h, w);
                (lh, lw) = (a, b);
                out.extend(y);
                at += n;
            }
        }
        let z = self.cfg.z_dim;
        let t = out.len();
        let mut lat = vec![0f32; z * t * lh * lw];
        for (f, fr) in out.iter().enumerate() {
            let (m, _, _) = quant.forward(fr, lh, lw, 1);
            // The first z channels are the mean.
            for p in 0..lh * lw {
                for c in 0..z {
                    lat[(c * t + f) * lh * lw + p] = m[p * 2 * z + c];
                }
            }
        }
        Ok(lat)
    }

    /// Decodes latents `[z, t, h, w]` (channel-first, not normalized) to `1 + 4 (t - 1)` frames,
    /// RGB in `[-1, 1]`, channels-last; `progress(done, total)` after every latent frame.
    pub fn decode(&self, z: &[f32], t: usize, h: usize, w: usize, progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<Frame>> {
        let zd = self.cfg.z_dim;
        if z.len() != zd * t * h * w || t == 0 {
            bail!("latents: {} values for {zd} x {t} x {h} x {w}", z.len());
        }
        let latent = |f: usize| -> Frame {
            let mut fr = vec![0f32; h * w * zd];
            for p in 0..h * w {
                for c in 0..zd {
                    fr[p * zd + c] = z[(c * t + f) * h * w + p];
                }
            }
            self.post_quant.forward(&fr, h, w, 1).0
        };
        #[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
        if let Some(e) = self.device.engine.get_or_init(|| dev::Engine::new(dev::OFF)) {
            let x: Vec<Frame> = (0..t).map(latent).collect();
            if let Some(mut out) = dev::decode(e, self, &x, h, w, progress) {
                for g in &mut out {
                    g.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
                }
                return Ok(out);
            }
        }
        let mut st = State { caches: Vec::new(), at: 0 };
        let mut out = Vec::new();
        for f in 0..t {
            let x = latent(f);
            let (y, _, _) = self.dec.forward(&[x], &mut st, h, w);
            for mut g in y {
                g.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
                out.push(g);
            }
            progress(f + 1, t);
        }
        Ok(out)
    }

    /// `(latents - mean) / std` per channel (`[z, ...]` channel-first), in place.
    pub fn normalize(&self, lat: &mut [f32]) {
        let per = lat.len() / self.cfg.z_dim;
        for (c, ch) in lat.chunks_mut(per).enumerate() {
            let (m, s) = (self.cfg.latents_mean[c], 1.0 / self.cfg.latents_std[c]);
            ch.iter_mut().for_each(|v| *v = (*v - m) * s);
        }
    }

    /// The inverse of [`Self::normalize`].
    pub fn denormalize(&self, lat: &mut [f32]) {
        let per = lat.len() / self.cfg.z_dim;
        for (c, ch) in lat.chunks_mut(per).enumerate() {
            let (m, s) = (self.cfg.latents_mean[c], 1.0 / self.cfg.latents_std[c]);
            ch.iter_mut().for_each(|v| *v = *v / s + m);
        }
    }
}
