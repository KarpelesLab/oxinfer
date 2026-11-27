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
    /// The kernel as bfloat16 bits `[cout][kt][ky][kx][cin]` (the device passes' layout: per time
    /// tap, nine spatial taps of `cin` inputs), the rows padded with zeros to a multiple of 64.
    pub(crate) w: Vec<u16>,
    pub(crate) bias: Vec<f32>,
    /// The CPU path's 3 x 3 convolution over stacked frames, made on first use.
    conv: std::sync::OnceLock<Conv>,
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

    /// From a `[cout, cin, 3, 3, 3]` kernel (values bfloat16 holds: the files' are bf16).
    fn new(w: &[f32], cout: usize, cin: usize, b: Vec<f32>) -> Result<Self> {
        if w.len() != cout * cin * 27 || b.len() != cout {
            bail!("3x3x3 convolution: {} weights and {} biases for {cout}x{cin}", w.len(), b.len());
        }
        let rows = cout.next_multiple_of(64);
        let mut wb = vec![0u16; rows * 27 * cin];
        let per = cout.div_ceil(16).max(1);
        std::thread::scope(|s| {
            for (j, part) in wb[..cout * 27 * cin].chunks_mut(per * 27 * cin).enumerate() {
                s.spawn(move || {
                    for (i, row) in part.chunks_mut(27 * cin).enumerate() {
                        let o = j * per + i;
                        for c in 0..cin {
                            for k in 0..27 {
                                row[k * cin + c] = half::bf16::from_f32(w[(o * cin + c) * 27 + k]).to_bits();
                            }
                        }
                    }
                });
            }
        });
        let mut bias = b;
        bias.resize(rows, 0.0);
        Ok(Self {
            w: wb,
            bias,
            conv: std::sync::OnceLock::new(),
            cin,
            cout,
        })
    }

    /// This convolution with its inputs zero-padded to `cin` channels.
    fn padded_cin(&self, cin: usize) -> Self {
        let rows = self.cout.next_multiple_of(64);
        let mut w = vec![0u16; rows * 27 * cin];
        for (o, src) in w.chunks_mut(27 * cin).zip(self.w.chunks(27 * self.cin)) {
            for k in 0..27 {
                o[k * cin..k * cin + self.cin].copy_from_slice(&src[k * self.cin..(k + 1) * self.cin]);
            }
        }
        Self {
            w,
            bias: self.bias.clone(),
            conv: std::sync::OnceLock::new(),
            cin,
            cout: self.cout,
        }
    }

    /// The CPU convolution: a 3 x 3 kernel over `[frame t-2.., c]` stacked channels
    /// (`[cout, 3 cin, 3, 3]`).
    fn cpu(&self) -> &Conv {
        self.conv.get_or_init(|| {
            let (cin, cout) = (self.cin, self.cout);
            let mut w2 = vec![0f32; cout * 27 * cin];
            for o in 0..cout {
                for kt in 0..3 {
                    for c in 0..cin {
                        for k in 0..9 {
                            let v = self.w[(o * 27 + kt * 9 + k) * cin + c];
                            w2[((o * 3 * cin) + kt * cin + c) * 9 + k] = half::bf16::from_bits(v).to_f32();
                        }
                    }
                }
            }
            Conv::from_kernel(&w2, cout, 3 * cin, 3, Some(self.bias[..cout].to_vec())).expect("a 3x3 kernel")
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
            let (y, _, _) = self.cpu().forward(&stacked, h, w, 1);
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
        if let Some(v) = self.decode_device(&x.data, t, h, w, progress) {
            return v;
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

impl VideoDecoder {
    /// The whole decode on the accelerator's session ([`crate::gpu::dit::LtxVaeSession`]) from
    /// un-normalized latent rows `[t, h, w, c]`; None when there is none or it declined
    /// (`OXINFER_LTX_VAE_GPU=0`: the CPU path).
    fn decode_device(&self, lat: &[f32], t: usize, h: usize, w: usize, progress: &mut dyn FnMut(usize, usize)) -> Option<Vid> {
        use crate::gpu::dit::{LtxVaeConv, LtxVaeDecoder, LtxVaeOp};
        if std::env::var("OXINFER_LTX_VAE_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        let mut s = crate::gpu::active()?.ltx_vae_session()?;
        fn conv(c: &Conv3) -> LtxVaeConv<'_> {
            LtxVaeConv {
                w: &c.w,
                bias: &c.bias,
                cin: c.cin,
                cout: c.cout,
            }
        }
        let (mut oh, mut ow) = (h * self.cfg.patch, w * self.cfg.patch);
        let mut ops = Vec::new();
        for u in &self.ups {
            match u {
                Up::Res(rs) => ops.extend(rs.iter().map(|r| LtxVaeOp::Res(conv(&r.c1), conv(&r.c2)))),
                Up::D2s(c, st) => {
                    oh *= st[1];
                    ow *= st[2];
                    ops.push(LtxVaeOp::D2s(conv(c), *st));
                }
            }
        }
        let m = LtxVaeDecoder {
            conv_in: conv(&self.conv_in),
            ops,
            conv_out: conv(&self.conv_out),
            patch: self.cfg.patch,
            causal: self.cfg.causal_decoder,
        };
        let mut out = Vec::new();
        if !s.decode(&m, lat, t, h, w, &mut out, progress) {
            return None;
        }
        let oc = self.conv_out.cout / (self.cfg.patch * self.cfg.patch);
        Some(Vid {
            t: out.len() / (oh * ow * oc),
            h: oh,
            w: ow,
            c: oc,
            data: out,
        })
    }
}

pub struct VideoEncoder {
    pub cfg: VaeConfig,
    mean: Vec<f32>,
    std: Vec<f32>,
    conv_in: Conv3,
    downs: Vec<Down>,
    conv_out: Conv3,
    /// `conv_in` with its inputs zero-padded to a multiple of 32 (the device's GEMMs), made on
    /// the first device pass.
    conv_in_dev: std::sync::OnceLock<Conv3>,
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
            conv_in_dev: std::sync::OnceLock::new(),
        })
    }

    /// The whole encode on the accelerator's session ([`crate::gpu::dit::LtxVaeSession::encode`])
    /// from the patchified frames; None when there is none or it declined
    /// (`OXINFER_LTX_VAE_GPU=0`: the CPU path).
    fn encode_device(&self, v: &Vid) -> Option<Vid> {
        use crate::gpu::dit::{LtxVaeConv, LtxVaeEncOp, LtxVaeEncoder};
        if std::env::var("OXINFER_LTX_VAE_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        let mut s = crate::gpu::active()?.ltx_vae_session()?;
        fn conv(c: &Conv3) -> LtxVaeConv<'_> {
            LtxVaeConv {
                w: &c.w,
                bias: &c.bias,
                cin: c.cin,
                cout: c.cout,
            }
        }
        let cin = self.conv_in.cin.next_multiple_of(32);
        let cin_dev = self.conv_in_dev.get_or_init(|| self.conv_in.padded_cin(cin));
        let mut ops = Vec::new();
        let (mut t, mut h, mut w) = (v.t, v.h, v.w);
        for d in &self.downs {
            match d {
                Down::Res(rs) => ops.extend(rs.iter().map(|r| LtxVaeEncOp::Res(conv(&r.c1), conv(&r.c2)))),
                Down::S2d(c, st, group) => {
                    (t, h, w) = ((t + (st[0] == 2) as usize) / st[0], h / st[1], w / st[2]);
                    ops.push(LtxVaeEncOp::S2d(conv(c), *st, *group));
                }
            }
        }
        let m = LtxVaeEncoder {
            conv_in: conv(cin_dev),
            ops,
            conv_out: conv(&self.conv_out),
            latent: self.cfg.latent_channels,
        };
        let mut x = vec![0f32; v.t * v.h * v.w * cin];
        for (o, i) in x.chunks_mut(cin).zip(v.data.chunks(v.c)) {
            o[..v.c].copy_from_slice(i);
        }
        let mut out = Vec::new();
        if !s.encode(&m, &x, v.t, v.h, v.w, &mut out, &mut |_, _| {}) {
            return None;
        }
        let c = self.cfg.latent_channels;
        if out.len() != t * h * w * c {
            return None;
        }
        let mut z = Vid::new(t, h, w, c);
        for (o, s) in z.data.chunks_mut(c).zip(out.chunks(c)) {
            for ch in 0..c {
                o[ch] = (s[ch] - self.mean[ch]) / self.std[ch];
            }
        }
        Some(z)
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
        if let Some(z) = self.encode_device(&v) {
            return z;
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

/// One interval of a tiled axis (latent units): `[start, end)` and its blend ramps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub start: usize,
    pub end: usize,
    pub left: usize,
    pub right: usize,
}

/// `split_by_size` (and `split_temporal_causal` with `causal`): tiles of `size` latent cells
/// sharing `overlap`, the last one shorter; causally, each tile after the first starts a cell
/// earlier with a ramp a cell longer.
pub fn split_by_size(dim: usize, size: usize, overlap: usize, causal: bool) -> Vec<Interval> {
    if dim <= size {
        return vec![Interval {
            start: 0,
            end: dim,
            left: 0,
            right: 0,
        }];
    }
    let step = size - overlap;
    let amount = (dim + size - 2 * overlap - 1) / step;
    let mut v = vec![Interval {
        start: 0,
        end: size,
        left: 0,
        right: overlap,
    }];
    for i in 1..amount - 1 {
        v.push(Interval {
            start: i * step,
            end: i * step + size,
            left: overlap,
            right: overlap,
        });
    }
    v.push(Interval {
        start: (amount - 1) * step,
        end: dim,
        left: overlap,
        right: 0,
    });
    if causal {
        for iv in v.iter_mut().skip(1) {
            iv.start -= 1;
            iv.left += 1;
        }
    }
    v
}

/// `torch.linspace(a, b, n)` in float32 (its two-sided formula).
fn linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f32;
    let half = n / 2;
    (0..n)
        .map(|i| {
            if i < half {
                a + step * i as f32
            } else {
                b - step * (n - 1 - i) as f32
            }
        })
        .collect()
}

/// `compute_trapezoidal_mask_1d`.
pub fn trapezoid(len: usize, left: usize, right: usize, from_zero: bool) -> Vec<f32> {
    let (l, r) = (left.min(len), right.min(len));
    let mut m = vec![1f32; len];
    if l > 0 {
        let mut f = linspace(0.0, 1.0, if from_zero { l + 1 } else { l + 2 });
        f.pop();
        if !from_zero {
            f.remove(0);
        }
        m[..l].iter_mut().zip(&f).for_each(|(a, b)| *a *= b);
    }
    if r > 0 {
        let f = linspace(1.0, 0.0, r + 2);
        m[len - r..].iter_mut().zip(&f[1..r + 1]).for_each(|(a, b)| *a *= b);
    }
    m.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
    m
}

/// Whether the masks of an axis' tiles sum to one everywhere (`masks_are_complementary`).
fn complementary(tiles: &[(Interval, Vec<f32>)], len: usize) -> bool {
    let mut acc = vec![0f32; len];
    for (iv, m) in tiles {
        acc[iv.start..iv.end].iter_mut().zip(m).for_each(|(a, b)| *a += b);
    }
    acc.iter().all(|v| (v - 1.0).abs() <= 1e-5)
}

impl VideoEncoder {
    /// `tiled_encode` with the release's `TileSizeConfig.default()` (the clip encoder of
    /// `video_latent_from_file`): tiles of 80 frames sharing 24 (causal: each after the first
    /// starts a latent frame earlier) and of 768 pixels sharing 64 in height and width, each
    /// encoded alone and blended with trapezoidal ramps (in latent units); frames past `8 k + 1`
    /// are cropped.
    pub fn encode_tiled(&self, x: &Vid, progress: &mut dyn FnMut(usize, usize)) -> Vid {
        let (tile_t, ov_t, tile_s, ov_s) = (80 / 8, 24 / 8, 768 / 32, 64 / 32);
        let frames = (x.t - 1) / 8 * 8 + 1;
        let (lt, lh, lw) = ((frames - 1) / 8 + 1, x.h / 32, x.w / 32);
        let axis = |dim: usize, size: usize, ov: usize, causal: bool, from_zero: bool| -> Vec<(Interval, Vec<f32>)> {
            split_by_size(dim, size.max(ov + 1).max(2), ov, causal)
                .into_iter()
                .map(|iv| (iv, trapezoid(iv.end - iv.start, iv.left, iv.right, from_zero)))
                .collect()
        };
        let (ts, hs, ws) = (
            axis(lt, tile_t, ov_t, true, true),
            axis(lh, tile_s, ov_s, false, false),
            axis(lw, tile_s, ov_s, false, false),
        );
        let c = self.cfg.latent_channels;
        let mut buf = Vid::new(lt, lh, lw, c);
        let mut wts = vec![0f32; lt * lh * lw];
        let total = ts.len() * hs.len() * ws.len();
        let mut done = 0;
        for (ti, tm) in &ts {
            let (f0, f1) = (ti.start * 8, 1 + (ti.end - 1) * 8);
            for (hi, hm) in &hs {
                let (y0, y1) = (hi.start * 32, hi.end * 32);
                for (wi, wm) in &ws {
                    let (x0, x1) = (wi.start * 32, wi.end * 32);
                    let mut tile = Vid::new(f1 - f0, y1 - y0, x1 - x0, x.c);
                    for f in f0..f1 {
                        for yy in y0..y1 {
                            let src = ((f * x.h + yy) * x.w + x0) * x.c;
                            let dst = (((f - f0) * tile.h + yy - y0) * tile.w) * x.c;
                            tile.data[dst..dst + (x1 - x0) * x.c].copy_from_slice(&x.data[src..src + (x1 - x0) * x.c]);
                        }
                    }
                    let z = self.encode(&tile);
                    for (a, f) in (ti.start..ti.end).enumerate() {
                        for (b, yy) in (hi.start..hi.end).enumerate() {
                            for (d, xx) in (wi.start..wi.end).enumerate() {
                                let m = tm[a] * hm[b] * wm[d];
                                let o = (f * lh + yy) * lw + xx;
                                let s = ((a * z.h + b) * z.w + d) * c;
                                buf.data[o * c..(o + 1) * c]
                                    .iter_mut()
                                    .zip(&z.data[s..s + c])
                                    .for_each(|(p, q)| *p += q * m);
                                wts[o] += m;
                            }
                        }
                    }
                    done += 1;
                    progress(done, total);
                }
            }
        }
        if !(complementary(&ts, lt) && complementary(&hs, lh) && complementary(&ws, lw)) {
            for (row, &w) in buf.data.chunks_mut(c).zip(&wts) {
                let w = w.max(1e-8);
                row.iter_mut().for_each(|v| *v /= w);
            }
        }
        buf
    }
}

#[cfg(test)]
mod tiling_tests {
    use super::*;

    #[test]
    fn splits_like_the_reference() {
        // 121 frames: 16 latent frames, tiles of 10 sharing 3, causal.
        let t = split_by_size(16, 10, 3, true);
        assert_eq!(
            t,
            vec![
                Interval {
                    start: 0,
                    end: 10,
                    left: 0,
                    right: 3
                },
                Interval {
                    start: 6,
                    end: 16,
                    left: 4,
                    right: 0
                }
            ]
        );
        assert_eq!(split_by_size(24, 24, 2, false).len(), 1);
        let s = split_by_size(40, 24, 2, false);
        assert_eq!(
            s[1],
            Interval {
                start: 22,
                end: 40,
                left: 2,
                right: 0
            }
        );
        let close = |a: Vec<f32>, b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        assert!(close(trapezoid(5, 2, 1, false), &[1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 0.5]));
        assert!(close(trapezoid(4, 2, 0, true), &[0.0, 0.5, 1.0, 1.0]));
    }
}
