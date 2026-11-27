//! LTX-2.5's convolutional video VAE (`vae/ltx-2.5-video-vae-conv-bf16.safetensors`,
//! `ltx_core.model.video_vae`: `ConvVideoDecoder` and `VideoEncoder`): 128 latent channels for a
//! 32 x 32 patch of 8 frames (the first latent frame stands for the first frame alone).
//!
//! Every convolution is a 3 x 3 x 3 `CausalConv3d`: zero padding in space, the first (and last)
//! frame repeated in time: two frames in front (causal, the encoder) or one on each side (the
//! decoder, `causal_decoder: false`). It runs as a 3 x 3 convolution over the three input frames'
//! channels side by side ([`crate::sd::ops::Conv`]: im2col and the shared matmul, on the
//! accelerator when one is active). Norms are `PixelNorm` (RMS over the channels, eps 1e-8), the
//! activation SiLU.
//!
//! - decoder: latents un-normalized by the per-channel statistics, `conv_in` to 1024 channels,
//!   the reversed `decoder_blocks` (`res_x` residual blocks, `compress_*` depth-to-space
//!   upsamplers whose doubled first frame is dropped), norm, SiLU, `conv_out` to 48 channels,
//!   4 x 4 depth-to-space;
//! - encoder: 4 x 4 space-to-depth, `conv_in`, the `encoder_blocks` (residual blocks and
//!   space-to-depth downsamplers that add the mean of their channel groups), norm, SiLU,
//!   `conv_out` (the means and one log-variance channel), the means normalized.
//!
//! Activations are frames of `[h * w, c]` rows.

use crate::safetensors::ModelFiles;
use crate::sd::ops::Conv;
use anyhow::{Context, Result, bail};

/// Frames of `[h * w, c]` rows.
#[derive(Clone)]
pub struct Vid {
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<f32>,
}

impl Vid {
    pub fn new(t: usize, h: usize, w: usize, c: usize) -> Self {
        Self {
            t,
            h,
            w,
            c,
            data: vec![0f32; t * h * w * c],
        }
    }

    fn frame(&self, i: usize) -> &[f32] {
        let n = self.h * self.w * self.c;
        &self.data[i * n..(i + 1) * n]
    }

    /// From `[c, t, h, w]` (PyTorch's layout without the batch).
    pub fn from_cthw(x: &[f32], c: usize, t: usize, h: usize, w: usize) -> Self {
        let mut v = Self::new(t, h, w, c);
        for ch in 0..c {
            for f in 0..t {
                for p in 0..h * w {
                    v.data[(f * h * w + p) * c + ch] = x[((ch * t + f) * h * w) + p];
                }
            }
        }
        v
    }

    /// To `[c, t, h, w]`.
    pub fn to_cthw(&self) -> Vec<f32> {
        let (t, hw, c) = (self.t, self.h * self.w, self.c);
        let mut out = vec![0f32; self.data.len()];
        for f in 0..t {
            for p in 0..hw {
                for ch in 0..c {
                    out[(ch * t + f) * hw + p] = self.data[(f * hw + p) * c + ch];
                }
            }
        }
        out
    }
}

/// `PixelNorm` then SiLU (or not), over every row.
fn norm_act(x: &mut [f32], c: usize, eps: f32, silu: bool) {
    crate::kernels::for_chunks(x, c * 64, x.len() > c * 64, |_, rows| {
        for r in rows.chunks_mut(c) {
            let ms = r.iter().map(|v| v * v).sum::<f32>() / c as f32;
            let s = 1.0 / (ms + eps).sqrt();
            for v in r.iter_mut() {
                let y = *v * s;
                *v = if silu { y / (1.0 + (-y).exp()) } else { y };
            }
        }
    });
}

/// How a 3 x 3 x 3 convolution pads in time.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TPad {
    /// The first frame twice in front (`CausalConv3d` with `causal=True`).
    Causal,
    /// The first frame in front and the last one behind (`causal=False`).
    Replicate,
    /// A zero frame on each side (`torch.nn.Conv3d(padding=1)`).
    Zero,
}

/// A 3 x 3 x 3 convolution (stride 1, zero padding in space).
pub(crate) struct Conv3 {
    conv: Conv,
    pub(crate) cin: usize,
    pub(crate) cout: usize,
}

impl Conv3 {
    pub(crate) fn load(f: &ModelFiles, p: &str) -> Result<Self> {
        let wn = format!("{p}.weight");
        let i = f.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        let [cout, cin, 3, 3, 3] = i.shape[..] else {
            bail!("{wn}: {:?}, expected a 3x3x3 kernel", i.shape);
        };
        let w = f.f32(&wn)?;
        let b = f.f32(&format!("{p}.bias"))?;
        Self::new(&w, cout, cin, b)
    }

    /// From a `[cout, cin, 3, 3, 3]` kernel.
    fn new(w: &[f32], cout: usize, cin: usize, b: Vec<f32>) -> Result<Self> {
        // As a 3 x 3 kernel over `[frame t-2.., c]` stacked channels: `[cout, 3 cin, 3, 3]`.
        let mut w2 = vec![0f32; w.len()];
        for o in 0..cout {
            for c in 0..cin {
                for kt in 0..3 {
                    for k in 0..9 {
                        w2[((o * 3 * cin) + kt * cin + c) * 9 + k] = w[((o * cin + c) * 3 + kt) * 9 + k];
                    }
                }
            }
        }
        Ok(Self {
            conv: Conv::from_kernel(&w2, cout, 3 * cin, 3, Some(b))?,
            cin,
            cout,
        })
    }

    fn forward(&self, x: &Vid, causal: bool) -> Vid {
        self.forward_pad(x, if causal { TPad::Causal } else { TPad::Replicate })
    }

    pub(crate) fn forward_pad(&self, x: &Vid, pad: TPad) -> Vid {
        debug_assert_eq!(x.c, self.cin);
        let (t, h, w, cin) = (x.t, x.h, x.w, self.cin);
        let mut out = Vid::new(t, h, w, self.cout);
        let n = h * w * self.cout;
        let mut stacked = vec![0f32; h * w * 3 * cin];
        let zero = vec![0f32; h * w * cin];
        for o in 0..t {
            let frames: [&[f32]; 3] = match pad {
                TPad::Causal => [x.frame(o.saturating_sub(2)), x.frame(o.saturating_sub(1)), x.frame(o)],
                TPad::Replicate => [x.frame(o.saturating_sub(1)), x.frame(o), x.frame((o + 1).min(t - 1))],
                TPad::Zero => [
                    if o == 0 { &zero } else { x.frame(o - 1) },
                    x.frame(o),
                    if o + 1 == t { &zero } else { x.frame(o + 1) },
                ],
            };
            crate::kernels::for_chunks(&mut stacked, 3 * cin * 64, true, |i, chunk| {
                for (j, row) in chunk.chunks_mut(3 * cin).enumerate() {
                    let p = i * 64 + j;
                    for (k, fr) in frames.iter().enumerate() {
                        row[k * cin..(k + 1) * cin].copy_from_slice(&fr[p * cin..(p + 1) * cin]);
                    }
                }
            });
            let (y, _, _) = self.conv.forward(&stacked, h, w, 1);
            out.data[o * n..(o + 1) * n].copy_from_slice(&y);
        }
        out
    }
}

struct Res {
    c1: Conv3,
    c2: Conv3,
}

impl Res {
    fn forward(&self, x: &Vid, causal: bool) -> Vid {
        let mut h = x.clone();
        norm_act(&mut h.data, x.c, 1e-8, true);
        let mut h = self.c1.forward(&h, causal);
        norm_act(&mut h.data, h.c, 1e-8, true);
        let mut h = self.c2.forward(&h, causal);
        h.data.iter_mut().zip(&x.data).for_each(|(a, &b)| *a += b);
        h
    }
}

/// `(c p1 p2 p3) d h w -> c (d p1) (h p2) (w p3)`.
pub(crate) fn depth_to_space(x: &Vid, st: [usize; 3]) -> Vid {
    let [p1, p2, p3] = st;
    let c = x.c / (p1 * p2 * p3);
    let mut out = Vid::new(x.t * p1, x.h * p2, x.w * p3, c);
    for f in 0..x.t {
        for y in 0..x.h {
            for xx in 0..x.w {
                let src = &x.data[((f * x.h + y) * x.w + xx) * x.c..][..x.c];
                for ch in 0..c {
                    for i1 in 0..p1 {
                        for i2 in 0..p2 {
                            for i3 in 0..p3 {
                                let v = src[((ch * p1 + i1) * p2 + i2) * p3 + i3];
                                let (of, oy, ox) = (f * p1 + i1, y * p2 + i2, xx * p3 + i3);
                                out.data[((of * out.h + oy) * out.w + ox) * c + ch] = v;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// `c (d p1) (h p2) (w p3) -> (c p1 p2 p3) d h w`.
fn space_to_depth(x: &Vid, st: [usize; 3]) -> Vid {
    let [p1, p2, p3] = st;
    let mut out = Vid::new(x.t / p1, x.h / p2, x.w / p3, x.c * p1 * p2 * p3);
    for f in 0..out.t {
        for y in 0..out.h {
            for xx in 0..out.w {
                let dst = ((f * out.h + y) * out.w + xx) * out.c;
                for ch in 0..x.c {
                    for i1 in 0..p1 {
                        for i2 in 0..p2 {
                            for i3 in 0..p3 {
                                let (sf, sy, sx) = (f * p1 + i1, y * p2 + i2, xx * p3 + i3);
                                out.data[dst + ((ch * p1 + i1) * p2 + i2) * p3 + i3] = x.data[((sf * x.h + sy) * x.w + sx) * x.c + ch];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub(crate) fn drop_first_frame(x: &mut Vid) {
    let n = x.h * x.w * x.c;
    x.data.drain(..n);
    x.t -= 1;
}

enum Up {
    Res(Vec<Res>),
    /// `DepthToSpaceUpsample`: the convolution, then depth-to-space by the stride.
    D2s(Box<Conv3>, [usize; 3]),
}

enum Down {
    Res(Vec<Res>),
    /// `SpaceToDepthDownsample`: space-to-depth of the convolution plus the channel-group means
    /// of the input's.
    S2d(Box<Conv3>, [usize; 3], usize),
}

fn stride_of(name: &str) -> Option<[usize; 3]> {
    Some(match name {
        "compress_time" | "compress_time_res" => [2, 1, 1],
        "compress_space" | "compress_space_res" => [1, 2, 2],
        "compress_all" | "compress_all_res" => [2, 2, 2],
        _ => return None,
    })
}

/// The `vae` object of the file's `config` metadata.
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub latent_channels: usize,
    pub patch: usize,
    pub causal_decoder: bool,
    pub encoder_blocks: Vec<(String, usize, usize)>,
    pub decoder_blocks: Vec<(String, usize, usize)>,
    pub base_channels: usize,
}

impl VaeConfig {
    pub fn from_files(f: &ModelFiles) -> Result<Self> {
        let s = f.metadata("config").context("LTX video VAE: no `config` metadata")?;
        let c: serde_json::Value = serde_json::from_str(s)?;
        let v = c.get("vae").unwrap_or(&c);
        if v.get("timestep_conditioning").and_then(|x| x.as_bool()) == Some(true) {
            bail!("LTX video VAE: timestep-conditioned decoders are not supported");
        }
        if v.get("norm_layer").and_then(|x| x.as_str()).is_some_and(|n| n != "pixel_norm") {
            bail!("LTX video VAE: only pixel_norm is supported");
        }
        // The diffusion VAE's file (`CausalDiffusionVAE`) nests its encoder's blocks and has no
        // convolutional decoder.
        let enc_blocks = v.get("encoder").and_then(|e| e.get("blocks"));
        let blocks = |k: &str| -> Result<Vec<(String, usize, usize)>> {
            let b = match k {
                "encoder_blocks" => v.get(k).or(enc_blocks),
                _ if enc_blocks.is_some() => v.get(k).or(Some(&serde_json::Value::Null)),
                _ => v.get(k),
            };
            if b.is_some_and(|b| b.is_null()) {
                return Ok(Vec::new());
            }
            b.and_then(|b| b.as_array())
                .with_context(|| format!("LTX video VAE: no {k}"))?
                .iter()
                .map(|b| {
                    let name = b.get(0).and_then(|n| n.as_str()).context("block name")?.to_string();
                    let p = b.get(1);
                    let layers = p
                        .and_then(|p| p.as_u64().or_else(|| p.get("num_layers").and_then(|n| n.as_u64())))
                        .unwrap_or(1) as usize;
                    let mult = p.and_then(|p| p.get("multiplier")).and_then(|m| m.as_u64()).unwrap_or(1) as usize;
                    Ok((name, layers, mult))
                })
                .collect()
        };
        Ok(Self {
            latent_channels: v.get("latent_channels").and_then(|x| x.as_u64()).unwrap_or(128) as usize,
            patch: v.get("patch_size").and_then(|x| x.as_u64()).unwrap_or(4) as usize,
            causal_decoder: v.get("causal_decoder").and_then(|x| x.as_bool()).unwrap_or(false),
            encoder_blocks: blocks("encoder_blocks")?,
            decoder_blocks: blocks("decoder_blocks")?,
            base_channels: v.get("decoder_base_channels").and_then(|x| x.as_u64()).unwrap_or(128) as usize,
        })
    }
}

fn stats(f: &ModelFiles, pfx: &str, c: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let m = f.f32(&format!("{pfx}per_channel_statistics.mean-of-means"))?;
    let s = f.f32(&format!("{pfx}per_channel_statistics.std-of-means"))?;
    if m.len() != c || s.len() != c {
        bail!(
            "LTX video VAE: per-channel statistics of {} / {} values, expected {c}",
            m.len(),
            s.len()
        );
    }
    Ok((m, s))
}

/// The latent statistics `(mean, std)` of a VAE file (the upsamplers' normalization).
pub fn stats_of(f: &ModelFiles) -> Result<(Vec<f32>, Vec<f32>)> {
    let cfg = VaeConfig::from_files(f)?;
    let pfx = if f.has("per_channel_statistics.mean-of-means") {
        ""
    } else {
        "vae."
    };
    stats(f, pfx, cfg.latent_channels)
}

fn res_blocks(f: &ModelFiles, p: &str, n: usize) -> Result<Vec<Res>> {
    (0..n)
        .map(|i| {
            Ok(Res {
                c1: Conv3::load(f, &format!("{p}.res_blocks.{i}.conv1.conv"))?,
                c2: Conv3::load(f, &format!("{p}.res_blocks.{i}.conv2.conv"))?,
            })
        })
        .collect()
}

pub struct VideoDecoder {
    pub cfg: VaeConfig,
    mean: Vec<f32>,
    std: Vec<f32>,
    conv_in: Conv3,
    ups: Vec<Up>,
    conv_out: Conv3,
}

impl VideoDecoder {
    pub fn load(f: &ModelFiles) -> Result<Self> {
        let cfg = VaeConfig::from_files(f)?;
        let pfx = if f.has("decoder.conv_in.conv.weight") { "" } else { "vae." };
        let (mean, std) = stats(f, pfx, cfg.latent_channels)?;
        let mut ups = Vec::new();
        for (i, (name, layers, _)) in cfg.decoder_blocks.iter().rev().enumerate() {
            let p = format!("{pfx}decoder.up_blocks.{i}");
            ups.push(match name.as_str() {
                "res_x" => Up::Res(res_blocks(f, &p, *layers)?),
                n => match stride_of(n) {
                    Some(st) if !n.ends_with("_res") => Up::D2s(Box::new(Conv3::load(f, &format!("{p}.conv.conv"))?), st),
                    _ => bail!("LTX video VAE: decoder block {n} is not supported"),
                },
            });
        }
        Ok(Self {
            conv_in: Conv3::load(f, &format!("{pfx}decoder.conv_in.conv"))?,
            conv_out: Conv3::load(f, &format!("{pfx}decoder.conv_out.conv"))?,
            mean,
            std,
            ups,
            cfg,
        })
    }

    /// Frames in `[-1, 1]` (`[t, h, w, 3]` rows: `(t - 1) * 8 + 1` frames of 32 times the latent's
    /// size) from latents `[c, t, h, w]`.
    pub fn decode(&self, lat: &[f32], t: usize, h: usize, w: usize, progress: &mut dyn FnMut(usize, usize)) -> Vid {
        let c = self.cfg.latent_channels;
        let mut x = Vid::from_cthw(lat, c, t, h, w);
        for row in x.data.chunks_mut(c) {
            for (ch, v) in row.iter_mut().enumerate() {
                *v = *v * self.std[ch] + self.mean[ch];
            }
        }
        let causal = self.cfg.causal_decoder;
        let mut x = self.conv_in.forward(&x, causal);
        let total = self.ups.len() + 1;
        for (i, u) in self.ups.iter().enumerate() {
            x = match u {
                Up::Res(rs) => {
                    let mut x = x;
                    for r in rs {
                        x = r.forward(&x, causal);
                    }
                    x
                }
                Up::D2s(conv, st) => {
                    let y = conv.forward(&x, causal);
                    let mut y = depth_to_space(&y, *st);
                    if st[0] == 2 {
                        drop_first_frame(&mut y);
                    }
                    y
                }
            };
            progress(i + 1, total);
        }
        norm_act(&mut x.data, x.c, 1e-8, true);
        let y = self.conv_out.forward(&x, causal);
        progress(total, total);
        // 4 x 4 depth-to-space with the patch's channels ordered (c, x, y).
        let p = self.cfg.patch;
        let oc = y.c / (p * p);
        let mut out = Vid::new(y.t, y.h * p, y.w * p, oc);
        for f in 0..y.t {
            for yy in 0..y.h {
                for xx in 0..y.w {
                    let src = &y.data[((f * y.h + yy) * y.w + xx) * y.c..][..y.c];
                    for ch in 0..oc {
                        for r in 0..p {
                            for q in 0..p {
                                let (oy, ox) = (yy * p + q, xx * p + r);
                                out.data[((f * out.h + oy) * out.w + ox) * oc + ch] = src[(ch * p + r) * p + q];
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

pub struct VideoEncoder {
    pub cfg: VaeConfig,
    mean: Vec<f32>,
    std: Vec<f32>,
    conv_in: Conv3,
    downs: Vec<Down>,
    conv_out: Conv3,
}

impl VideoEncoder {
    pub fn load(f: &ModelFiles) -> Result<Self> {
        let cfg = VaeConfig::from_files(f)?;
        let pfx = if f.has("encoder.conv_in.conv.weight") { "" } else { "vae." };
        let (mean, std) = stats(f, pfx, cfg.latent_channels)?;
        let mut downs = Vec::new();
        let mut ch = cfg.latent_channels;
        for (i, (name, layers, mult)) in cfg.encoder_blocks.iter().enumerate() {
            let p = format!("{pfx}encoder.down_blocks.{i}");
            downs.push(match name.as_str() {
                "res_x" => Down::Res(res_blocks(f, &p, *layers)?),
                n => match stride_of(n) {
                    Some(st) if n.ends_with("_res") => {
                        let out = ch * mult;
                        let group = ch * st.iter().product::<usize>() / out;
                        ch = out;
                        Down::S2d(Box::new(Conv3::load(f, &format!("{p}.conv.conv"))?), st, group)
                    }
                    _ => bail!("LTX video VAE: encoder block {n} is not supported"),
                },
            });
        }
        Ok(Self {
            conv_in: Conv3::load(f, &format!("{pfx}encoder.conv_in.conv"))?,
            conv_out: Conv3::load(f, &format!("{pfx}encoder.conv_out.conv"))?,
            mean,
            std,
            downs,
            cfg,
        })
    }

    /// Normalized latent means `[c, t', h / 32, w / 32]` of frames `[t, h, w, 3]` in `[-1, 1]`
    /// (`t = 8 n + 1`).
    pub fn encode(&self, x: &Vid) -> Vid {
        let p = self.cfg.patch;
        // 4 x 4 space-to-depth with the patch's channels ordered (c, x, y).
        let mut v = Vid::new(x.t, x.h / p, x.w / p, x.c * p * p);
        for f in 0..x.t {
            for yy in 0..v.h {
                for xx in 0..v.w {
                    let dst = ((f * v.h + yy) * v.w + xx) * v.c;
                    for ch in 0..x.c {
                        for r in 0..p {
                            for q in 0..p {
                                let (sy, sx) = (yy * p + q, xx * p + r);
                                v.data[dst + (ch * p + r) * p + q] = x.data[((f * x.h + sy) * x.w + sx) * x.c + ch];
                            }
                        }
                    }
                }
            }
        }
        let mut x = self.conv_in.forward(&v, true);
        for d in &self.downs {
            x = match d {
                Down::Res(rs) => {
                    let mut x = x;
                    for r in rs {
                        x = r.forward(&x, true);
                    }
                    x
                }
                Down::S2d(conv, st, group) => {
                    let mut x = x;
                    if st[0] == 2 {
                        let n = x.h * x.w * x.c;
                        let first = x.data[..n].to_vec();
                        x.data.splice(0..0, first);
                        x.t += 1;
                    }
                    let mut skip = space_to_depth(&x, *st);
                    let y = space_to_depth(&conv.forward(&x, true), *st);
                    // The input's channel groups, averaged: `(c g) -> c`.
                    let oc = y.c;
                    let mut out = y;
                    for (orow, srow) in out.data.chunks_mut(oc).zip(skip.data.chunks(oc * group)) {
                        for (c, o) in orow.iter_mut().enumerate() {
                            let m = srow[c * group..(c + 1) * group].iter().sum::<f32>() / *group as f32;
                            *o += m;
                        }
                    }
                    skip.data.clear();
                    out
                }
            };
        }
        norm_act(&mut x.data, x.c, 1e-8, true);
        let y = self.conv_out.forward(&x, true);
        let c = self.cfg.latent_channels;
        let mut out = Vid::new(y.t, y.h, y.w, c);
        for (o, s) in out.data.chunks_mut(c).zip(y.data.chunks(y.c)) {
            for ch in 0..c {
                o[ch] = (s[ch] - self.mean[ch]) / self.std[ch];
            }
        }
        out
    }
}
