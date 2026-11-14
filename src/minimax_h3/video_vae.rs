//! MiniMax-H3's video VAE encoder (diffusers' `AutoencoderKLMiniMaxH3.encode`): a causal 3D CNN
//! (`f16t4d24`: 16x spatially, 4x in time, 24 latent channels) whose decoder is the ViT of
//! [`super::vae`].
//!
//! - every convolution pads space symmetrically by reflection and time causally (two zero frames
//!   before, none after), so a single frame goes through the spatial path alone;
//! - group normalization per frame; residual blocks of two 3x3x3 convolutions; a strided 3x3x3
//!   convolution after a level (space: one reflected row / column appended, stride 2);
//! - a video is encoded in clips of `clip_length` (17) frames, the last frame repeated to fill a
//!   clip, and the `token_drop` (3) trailing latent frames dropped: `17 n + 5` frames give
//!   `5 n + 2` latent frames; one frame gives one;
//! - 256-pixel tiles overlapping by at least 64, encoded on their own and blended in latent space
//!   (the decoder's tiling).
//!
//! The output is the posterior's moments `[2C, t, h, w]` (mean, then log-variance).

use super::vae::{VaeConfig, stitch};
use crate::kernels::for_chunks;
use crate::safetensors::ModelFiles;
use crate::sd::ops::{Conv, silu};
use anyhow::{Context, Result, bail};

/// A 3D convolution as one 2D convolution per temporal tap over channels-last frames.
struct Conv3 {
    taps: Vec<Conv>,
    bias: Vec<f32>,
    cout: usize,
    /// Reflected rows / columns on every side.
    spad: usize,
    /// Zero frames before the clip.
    tpad: usize,
    stride_t: usize,
    stride_s: usize,
}

/// `x` (`[h * w, c]`) padded by reflection: `(top, bottom, left, right)`.
fn pad_reflect(x: &[f32], h: usize, w: usize, c: usize, pad: (usize, usize, usize, usize)) -> (Vec<f32>, usize, usize) {
    let (top, bottom, left, right) = pad;
    let (oh, ow) = (h + top + bottom, w + left + right);
    let src = |i: usize, before: usize, n: usize| -> usize {
        let p = i as isize - before as isize;
        let p = if p < 0 { -p } else { p };
        let p = p as usize;
        if p >= n { (2 * (n - 1)).saturating_sub(p).min(n - 1) } else { p }
    };
    let mut out = vec![0f32; oh * ow * c];
    for_chunks(&mut out, ow * c, oh > 8, |y, row| {
        let sy = src(y, top, h);
        for x0 in 0..ow {
            let sx = src(x0, left, w);
            row[x0 * c..(x0 + 1) * c].copy_from_slice(&x[(sy * w + sx) * c..(sy * w + sx + 1) * c]);
        }
    });
    (out, oh, ow)
}

impl Conv3 {
    fn load(files: &ModelFiles, name: &str, spad: usize, tpad: usize, stride_t: usize, stride_s: usize) -> Result<Self> {
        let wn = format!("{name}.weight");
        let shape = files.info(&wn).with_context(|| format!("missing tensor {wn}"))?.shape.clone();
        let [cout, cin, kt, kh, kw] = shape[..] else {
            bail!("{wn}: expected a 5-D kernel, got {shape:?}")
        };
        if kh != kw {
            bail!("{wn}: {kh}x{kw} kernels are not supported");
        }
        let w = files.f32(&wn)?;
        let bias = files.f32(&format!("{name}.bias"))?;
        let mut taps = Vec::with_capacity(kt);
        for j in 0..kt {
            let mut k2 = vec![0f32; cout * cin * kh * kw];
            for oc in 0..cout * cin {
                k2[oc * kh * kw..(oc + 1) * kh * kw].copy_from_slice(&w[(oc * kt + j) * kh * kw..(oc * kt + j + 1) * kh * kw]);
            }
            taps.push(Conv::from_kernel(&k2, cout, cin, kh, None)?.with_padding(0, 0));
        }
        Ok(Self {
            taps,
            bias,
            cout,
            spad,
            tpad,
            stride_t,
            stride_s,
        })
    }

    fn forward(&self, frames: &[Vec<f32>], h: usize, w: usize) -> (Vec<Vec<f32>>, usize, usize) {
        let cin = frames[0].len() / (h * w);
        let padded: Vec<(Vec<f32>, usize, usize)> = if self.spad > 0 {
            let p = self.spad;
            frames.iter().map(|f| pad_reflect(f, h, w, cin, (p, p, p, p))).collect()
        } else {
            Vec::new()
        };
        let src = |i: usize| -> (&[f32], usize, usize) {
            if self.spad > 0 {
                (&padded[i].0, padded[i].1, padded[i].2)
            } else {
                (&frames[i], h, w)
            }
        };
        let kt = self.taps.len();
        let t_out = (frames.len() + self.tpad - kt) / self.stride_t + 1;
        let (mut oh, mut ow) = (0, 0);
        let mut out = Vec::with_capacity(t_out);
        for to in 0..t_out {
            let mut acc: Option<Vec<f32>> = None;
            for (j, tap) in self.taps.iter().enumerate() {
                // Taps over the causal zero padding contribute nothing.
                let Some(si) = (to * self.stride_t + j).checked_sub(self.tpad) else {
                    continue;
                };
                let (x, ph, pw) = src(si);
                let (y, a, b) = tap.forward(x, ph, pw, self.stride_s);
                (oh, ow) = (a, b);
                match &mut acc {
                    Some(s) => s.iter_mut().zip(&y).for_each(|(s, v)| *s += v),
                    None => acc = Some(y),
                }
            }
            let mut y = acc.expect("a causal convolution reads the current frame");
            for_chunks(&mut y, self.cout, true, |_, px| {
                px.iter_mut().zip(&self.bias).for_each(|(v, b)| *v += b)
            });
            out.push(y);
        }
        (out, oh, ow)
    }
}

/// Group normalization of one channels-last frame, then SiLU.
struct Gn {
    w: Vec<f32>,
    b: Vec<f32>,
    groups: usize,
    eps: f32,
}

impl Gn {
    fn load(files: &ModelFiles, name: &str, groups: usize, eps: f32) -> Result<Self> {
        let w = files.f32(&format!("{name}.weight"))?;
        let b = files.f32(&format!("{name}.bias"))?;
        if w.len() % groups != 0 || b.len() != w.len() {
            bail!("{name}: {} channels do not split into {groups} groups", w.len());
        }
        Ok(Self { w, b, groups, eps })
    }

    fn silu(&self, x: &mut [f32]) {
        let c = self.w.len();
        let cpg = c / self.groups;
        let n = x.len() / c;
        let mut sum = vec![0f64; self.groups];
        let mut sq = vec![0f64; self.groups];
        for px in x.chunks(c) {
            for (g, ch) in px.chunks(cpg).enumerate() {
                let (mut s, mut s2) = (0f32, 0f32);
                for &v in ch {
                    s += v;
                    s2 += v * v;
                }
                sum[g] += s as f64;
                sq[g] += s2 as f64;
            }
        }
        let cnt = (n * cpg) as f64;
        let stats: Vec<(f32, f32)> = (0..self.groups)
            .map(|g| {
                let m = sum[g] / cnt;
                let var = (sq[g] / cnt - m * m).max(0.0);
                (m as f32, (1.0 / (var + self.eps as f64).sqrt()) as f32)
            })
            .collect();
        for_chunks(x, c, n > 64, |_, px| {
            for (ch, v) in px.iter_mut().enumerate() {
                let (m, inv) = stats[ch / cpg];
                *v = silu((*v - m) * inv * self.w[ch] + self.b[ch]);
            }
        });
    }
}

struct Resnet {
    norm1: Gn,
    conv1: Conv3,
    norm2: Gn,
    conv2: Conv3,
    shortcut: Option<Conv3>,
}

impl Resnet {
    fn forward(&self, frames: Vec<Vec<f32>>, h: usize, w: usize) -> Vec<Vec<f32>> {
        let mut x = frames.clone();
        x.iter_mut().for_each(|f| self.norm1.silu(f));
        let (mut x, _, _) = self.conv1.forward(&x, h, w);
        x.iter_mut().for_each(|f| self.norm2.silu(f));
        let (mut x, _, _) = self.conv2.forward(&x, h, w);
        let res = match &self.shortcut {
            Some(s) => s.forward(&frames, h, w).0,
            None => frames,
        };
        for (a, b) in x.iter_mut().zip(&res) {
            a.iter_mut().zip(b).for_each(|(v, r)| *v += r);
        }
        x
    }
}

struct Level {
    resnets: Vec<Resnet>,
    /// The strided convolution and whether it halves space.
    down: Option<(Conv3, bool)>,
}

/// The encoder (see the module docs).
pub struct VideoEncoder {
    conv_in: Conv3,
    levels: Vec<Level>,
    norm_out: Gn,
    conv_out: Conv3,
    quant: Conv3,
    /// Moment channels (`2 * latent_channels`).
    pub channels: usize,
    ratio: usize,
    clip_length: usize,
    token_drop: usize,
    tile: usize,
    overlap: usize,
}

impl VideoEncoder {
    /// Loads the encoder of diffusers' layout (`vae/`: `encoder.*`, `quant_conv`).
    pub fn load(dir: &std::path::Path, cfg: &VaeConfig, files: &ModelFiles) -> Result<Self> {
        let c: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let list = |k: &str| -> Result<Vec<usize>> {
            Ok(c.get(k)
                .and_then(|v| v.as_array())
                .with_context(|| format!("video VAE config has no {k}"))?
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect())
        };
        let channels = list("block_out_channels")?;
        let (sd, td) = (list("spatial_downsample_factors")?, list("temporal_downsample_factors")?);
        if sd.len() != channels.len() || td.len() != channels.len() {
            bail!("video VAE config: one downsample factor per level is expected");
        }
        let layers = c.get("layers_per_block").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let groups = c.get("norm_num_groups").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let eps = c.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
        if c.get("spatial_padding_mode").and_then(|v| v.as_str()).unwrap_or("reflect") != "reflect" {
            bail!("video VAE: only reflect spatial padding is supported");
        }
        if !files.has("encoder.conv_in.weight") {
            bail!("the video VAE has no encoder (encoder.conv_in.weight)");
        }
        let mut levels = Vec::new();
        for (i, &ch) in channels.iter().enumerate() {
            let cin = if i == 0 { channels[0] } else { channels[i - 1] };
            let mut resnets = Vec::new();
            for j in 0..layers {
                let p = format!("encoder.down_blocks.{i}.resnets.{j}.");
                resnets.push(Resnet {
                    norm1: Gn::load(files, &format!("{p}norm1"), groups, eps)?,
                    conv1: Conv3::load(files, &format!("{p}conv1"), 1, 2, 1, 1)?,
                    norm2: Gn::load(files, &format!("{p}norm2"), groups, eps)?,
                    conv2: Conv3::load(files, &format!("{p}conv2"), 1, 2, 1, 1)?,
                    shortcut: (j == 0 && cin != ch)
                        .then(|| Conv3::load(files, &format!("{p}conv_shortcut"), 0, 0, 1, 1))
                        .transpose()?,
                });
            }
            let down = (td[i] * sd[i] > 1)
                .then(|| Conv3::load(files, &format!("encoder.down_blocks.{i}.downsamplers.0.conv"), 0, 2, td[i], sd[i]))
                .transpose()?
                .map(|conv| (conv, sd[i] == 2));
            if sd[i] > 2 || (sd[i] == 1 && td[i] > 1) {
                bail!(
                    "video VAE: level {i} downsamples by {}x{} (space x time), which is not supported",
                    sd[i],
                    td[i]
                );
            }
            levels.push(Level { resnets, down });
        }
        let last = *channels.last().context("video VAE: no levels")?;
        Ok(Self {
            conv_in: Conv3::load(files, "encoder.conv_in", 1, 2, 1, 1)?,
            levels,
            norm_out: Gn::load(files, "encoder.norm_out", groups, eps)?,
            conv_out: Conv3::load(files, "encoder.conv_out", 1, 2, 1, 1)?,
            quant: Conv3::load(files, "quant_conv", 0, 0, 1, 1)?,
            channels: 2 * cfg.latent_channels,
            ratio: sd.iter().product(),
            clip_length: cfg.clip_length,
            token_drop: cfg.token_drop,
            tile: cfg.tile,
            overlap: cfg.overlap,
        })
        .and_then(|e| if last == 0 { bail!("video VAE: empty level") } else { Ok(e) })
    }

    /// One tile of one clip: channels-last frames in, `([2C, t, h, w], t, h, w)` out.
    fn encode_tile(&self, frames: &[Vec<f32>], h: usize, w: usize) -> (Vec<f32>, usize, usize, usize) {
        let (mut x, mut h, mut w) = self.conv_in.forward(frames, h, w);
        for level in &self.levels {
            for r in &level.resnets {
                x = r.forward(x, h, w);
            }
            if let Some((conv, space)) = &level.down {
                if *space {
                    let c = x[0].len() / (h * w);
                    let padded: Vec<Vec<f32>> = x.iter().map(|f| pad_reflect(f, h, w, c, (0, 1, 0, 1)).0).collect();
                    (x, h, w) = conv.forward(&padded, h + 1, w + 1);
                } else {
                    (x, h, w) = conv.forward(&x, h, w);
                }
            }
        }
        x.iter_mut().for_each(|f| self.norm_out.silu(f));
        let (x, _, _) = self.conv_out.forward(&x, h, w);
        let (x, _, _) = self.quant.forward(&x, h, w);
        let (t, c, plane) = (x.len(), self.channels, h * w);
        let mut out = vec![0f32; c * t * plane];
        for (f, fr) in x.iter().enumerate() {
            for (i, px) in fr.chunks(c).enumerate() {
                for (ch, &v) in px.iter().enumerate() {
                    out[(ch * t + f) * plane + i] = v;
                }
            }
        }
        (out, t, h, w)
    }

    /// `_split_tiles` (the decoder's).
    fn split(&self, length: usize) -> (Vec<usize>, usize, Vec<usize>) {
        let (tile, min_ov) = (self.tile, self.overlap);
        if tile >= length {
            return (vec![0], length, Vec::new());
        }
        let mut n = length.div_ceil(tile);
        while tile * n < min_ov * (n - 1) + length {
            n += 1;
        }
        let mut ov = vec![min_ov; n - 1];
        let remaining = tile * n - ov.iter().sum::<usize>() - length;
        for i in 0..remaining / self.ratio {
            ov[i % (n - 1)] += self.ratio;
        }
        let mut starts = vec![0];
        for i in 0..n - 1 {
            starts.push(starts[i] + tile - ov[i]);
        }
        (starts, tile, ov)
    }

    /// `_encode_clip`: one clip of channels-last frames (`[h * w, 3]` each), spatially tiled.
    fn encode_clip(&self, frames: &[&[f32]], h: usize, w: usize) -> (Vec<f32>, usize, usize, usize) {
        let (ys, th, yov) = self.split(h);
        let (xs, tw, xov) = self.split(w);
        let mut lt = 0;
        let tiles: Vec<Vec<Vec<f32>>> = ys
            .iter()
            .map(|&y0| {
                xs.iter()
                    .map(|&x0| {
                        let cut: Vec<Vec<f32>> = frames
                            .iter()
                            .map(|f| {
                                let mut o = vec![0f32; th * tw * 3];
                                for y in 0..th {
                                    o[y * tw * 3..(y + 1) * tw * 3]
                                        .copy_from_slice(&f[((y0 + y) * w + x0) * 3..((y0 + y) * w + x0 + tw) * 3]);
                                }
                                o
                            })
                            .collect();
                        let (z, t, _, _) = self.encode_tile(&cut, th, tw);
                        lt = t;
                        z
                    })
                    .collect()
            })
            .collect();
        let r = self.ratio;
        let lov = |v: &[usize]| v.iter().map(|o| o / r).collect::<Vec<_>>();
        let (lh, lw) = (h.div_ceil(r), w.div_ceil(r));
        let out = stitch(
            &tiles,
            self.channels * lt,
            th.div_ceil(r),
            tw.div_ceil(r),
            &lov(&yov),
            &lov(&xov),
            lh,
            lw,
        );
        (out, lt, lh, lw)
    }

    /// The posterior moments `([2C, t, h/16, w/16], t, h/16, w/16)` of `frames` RGB frames
    /// (`[3, frames, h, w]`, ImageNet-normalized).
    pub fn encode(&self, video: &[f32], frames: usize, h: usize, w: usize) -> Result<(Vec<f32>, usize, usize, usize)> {
        let plane = h * w;
        if frames == 0 || video.len() != 3 * frames * plane {
            bail!("video encode: {} values for [3, {frames}, {h}, {w}]", video.len());
        }
        // Channels-last frames.
        let fr: Vec<Vec<f32>> = (0..frames)
            .map(|f| {
                let mut o = vec![0f32; plane * 3];
                for ch in 0..3 {
                    let src = &video[(ch * frames + f) * plane..(ch * frames + f + 1) * plane];
                    o.iter_mut().skip(ch).step_by(3).zip(src).for_each(|(d, &s)| *d = s);
                }
                o
            })
            .collect();
        if frames == 1 {
            return Ok(self.encode_clip(&[&fr[0]], h, w));
        }
        let clip = self.clip_length;
        let total = frames.div_ceil(clip) * clip;
        let at = |i: usize| -> &[f32] { &fr[i.min(frames - 1)] };
        let mut parts = Vec::new();
        let (mut lt, mut lh, mut lw) = (0, 0, 0);
        for c0 in (0..total).step_by(clip) {
            let refs: Vec<&[f32]> = (c0..c0 + clip).map(at).collect();
            let (z, t, a, b) = self.encode_clip(&refs, h, w);
            (lt, lh, lw) = (t, a, b);
            parts.push(z);
        }
        // Concatenated along time, the trailing latent frames dropped.
        let all = parts.len() * lt;
        let keep = all
            .checked_sub(self.token_drop)
            .filter(|&k| k > 0)
            .context("video encode: too few frames")?;
        let lp = lh * lw;
        let mut out = vec![0f32; self.channels * keep * lp];
        for ch in 0..self.channels {
            for f in 0..keep {
                let (pi, pf) = (f / lt, f % lt);
                out[(ch * keep + f) * lp..(ch * keep + f + 1) * lp]
                    .copy_from_slice(&parts[pi][(ch * lt + pf) * lp..(ch * lt + pf + 1) * lp]);
            }
        }
        Ok((out, keep, lh, lw))
    }
}
