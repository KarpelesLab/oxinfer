//! LTX-2.5's latent upsamplers (`latent_upscale_models/`, `ltx_core.model.upsampler.LatentUpsampler`):
//! the spatial x2 one of the two-stage pipelines (latents doubled in height and width between the
//! two denoising stages) and the temporal x2 one (`8 (t - 1) + 1` frames to twice that, minus one
//! latent frame).
//!
//! Over un-normalized latents (the video VAE's per-channel statistics): a 3 x 3 x 3 convolution
//! (zero padding, in time too) to the middle width, group norm (32 groups over all frames), SiLU,
//! residual blocks (convolution, group norm, SiLU, convolution, group norm, SiLU of the sum), the
//! upsampler (spatial: a 3 x 3 convolution per frame to 4x the channels and a 2 x 2 pixel
//! shuffle; temporal: a 3 x 3 x 3 convolution to 2x and a time shuffle, its first frame dropped),
//! as many residual blocks again, a convolution back to the latent channels; normalized again.
//!
//! With an accelerator's VAE session ([`crate::gpu::dit::LtxVaeSession::upsample`]) the whole
//! upsampler runs there, the group norms' statistics reduced on the device
//! (`OXINFER_LTX_UP_GPU=0`: the CPU path).

use super::vae::{Conv3, TPad, Vid, depth_to_space, drop_first_frame};
use crate::safetensors::ModelFiles;
use crate::sd::ops::Conv;
use anyhow::{Context, Result, bail};

struct GroupNorm {
    w: Vec<f32>,
    b: Vec<f32>,
    groups: usize,
}

impl GroupNorm {
    fn load(f: &ModelFiles, p: &str) -> Result<Self> {
        Ok(Self {
            w: f.f32(&format!("{p}.weight"))?,
            b: f.f32(&format!("{p}.bias"))?,
            groups: 32,
        })
    }

    /// Normalizes `x` in place over each group's channels of every frame and position, then the
    /// affine transform and (with `silu`) SiLU.
    fn apply(&self, x: &mut Vid, silu: bool) {
        let c = x.c;
        let g = c / self.groups;
        let rows = x.data.len() / c;
        let mut sums = vec![(0f64, 0f64); self.groups];
        for r in 0..rows {
            let row = &x.data[r * c..(r + 1) * c];
            for (gi, s) in sums.iter_mut().enumerate() {
                for &v in &row[gi * g..(gi + 1) * g] {
                    s.0 += v as f64;
                    s.1 += (v as f64) * (v as f64);
                }
            }
        }
        let n = (rows * g) as f64;
        let stats: Vec<(f32, f32)> = sums
            .iter()
            .map(|&(s, ss)| {
                let mean = s / n;
                let var = (ss / n - mean * mean).max(0.0);
                (mean as f32, (1.0 / (var + 1e-5).sqrt()) as f32)
            })
            .collect();
        crate::kernels::for_chunks(&mut x.data, c * 64, rows > 64, |_, chunk| {
            for row in chunk.chunks_mut(c) {
                for (ch, v) in row.iter_mut().enumerate() {
                    let (m, inv) = stats[ch / g];
                    let y = (*v - m) * inv * self.w[ch] + self.b[ch];
                    *v = if silu { y / (1.0 + (-y).exp()) } else { y };
                }
            }
        });
    }
}

struct Res {
    c1: Conv3,
    n1: GroupNorm,
    c2: Conv3,
    n2: GroupNorm,
}

impl Res {
    fn forward(&self, x: &Vid) -> Vid {
        let mut h = self.c1.forward_pad(x, TPad::Zero);
        self.n1.apply(&mut h, true);
        let mut h = self.c2.forward_pad(&h, TPad::Zero);
        self.n2.apply(&mut h, false);
        for (a, &b) in h.data.iter_mut().zip(&x.data) {
            let y = *a + b;
            *a = y / (1.0 + (-y).exp());
        }
        h
    }
}

enum Upsampler {
    /// A 3 x 3 convolution per frame to 4x the channels, 2 x 2 pixel shuffle; its kernel also as
    /// bfloat16 `[n_out][ky][kx][cin]` rows and the bias (`n_out` the outputs rounded up to 64)
    /// for the device.
    Spatial(Conv, Vec<u16>, Vec<f32>),
    /// A 3 x 3 x 3 convolution to 2x the channels, time shuffle, first frame dropped.
    Temporal(Conv3),
}

pub struct LatentUpsampler {
    conv_in: Conv3,
    norm_in: GroupNorm,
    pre: Vec<Res>,
    up: Upsampler,
    post: Vec<Res>,
    conv_out: Conv3,
    /// The scale of each axis `(time, height, width)`.
    pub scale: [usize; 3],
}

fn res_blocks(f: &ModelFiles, p: &str, n: usize) -> Result<Vec<Res>> {
    (0..n)
        .map(|i| {
            Ok(Res {
                c1: Conv3::load(f, &format!("{p}.{i}.conv1"))?,
                n1: GroupNorm::load(f, &format!("{p}.{i}.norm1"))?,
                c2: Conv3::load(f, &format!("{p}.{i}.conv2"))?,
                n2: GroupNorm::load(f, &format!("{p}.{i}.norm2"))?,
            })
        })
        .collect()
}

impl LatentUpsampler {
    pub fn load(f: &ModelFiles) -> Result<Self> {
        let c: serde_json::Value = serde_json::from_str(f.metadata("config").context("LTX latent upsampler: no `config` metadata")?)?;
        let flag = |k: &str| c.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        let n = c.get("num_blocks_per_stage").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
        if c.get("dims").and_then(|v| v.as_u64()).unwrap_or(3) != 3 {
            bail!("LTX latent upsampler: only 3-D upsamplers are supported");
        }
        let (spatial, temporal) = (flag("spatial_upsample"), flag("temporal_upsample"));
        let (up, scale) = match (spatial, temporal) {
            (true, false) => {
                if flag("rational_resampler") {
                    bail!("LTX latent upsampler: the rational spatial resampler is not supported");
                }
                let wn = "upsampler.0.weight";
                let i = f.info(wn).with_context(|| format!("missing tensor {wn}"))?;
                let [cout, cin, 3, 3] = i.shape[..] else {
                    bail!("{wn}: {:?}", i.shape);
                };
                let (k, b) = (f.f32(wn)?, f.f32("upsampler.0.bias")?);
                let rows = cout.next_multiple_of(64);
                let mut wd = vec![0u16; rows * 9 * cin];
                for (o, row) in wd.chunks_mut(9 * cin).take(cout).enumerate() {
                    for c in 0..cin {
                        for t in 0..9 {
                            row[t * cin + c] = half::bf16::from_f32(k[(o * cin + c) * 9 + t]).to_bits();
                        }
                    }
                }
                let mut bd = b.clone();
                bd.resize(rows, 0.0);
                let conv = Conv::from_kernel(&k, cout, cin, 3, Some(b))?;
                (Upsampler::Spatial(conv, wd, bd), [1, 2, 2])
            }
            (false, true) => (Upsampler::Temporal(Conv3::load(f, "upsampler.0")?), [2, 1, 1]),
            _ => bail!("LTX latent upsampler: spatial and temporal at once is not supported"),
        };
        Ok(Self {
            conv_in: Conv3::load(f, "initial_conv")?,
            norm_in: GroupNorm::load(f, "initial_norm")?,
            pre: res_blocks(f, "res_blocks", n)?,
            up,
            post: res_blocks(f, "post_upsample_res_blocks", n)?,
            conv_out: Conv3::load(f, "final_conv")?,
            scale,
        })
    }

    /// Upsamples un-normalized latents (frames of `[h * w, c]` rows): on the accelerator when
    /// there is one, else on the CPU.
    pub fn forward(&self, x: &Vid) -> Vid {
        if std::env::var("OXINFER_LTX_UP_GPU").is_ok_and(|v| v == "0") {
            return self.forward_cpu(x);
        }
        self.forward_device(x).unwrap_or_else(|| self.forward_cpu(x))
    }

    /// The whole upsampler on the accelerator's VAE session; None when there is none or it
    /// declined.
    pub fn forward_device(&self, x: &Vid) -> Option<Vid> {
        use crate::gpu::dit::{LtxGroupNorm, LtxUpRes, LtxUpsampler, LtxVaeConv};
        let mut s = crate::gpu::active()?.ltx_vae_session()?;
        fn conv(c: &Conv3) -> LtxVaeConv<'_> {
            LtxVaeConv {
                w: &c.w,
                bias: &c.bias,
                cin: c.cin,
                cout: c.cout,
            }
        }
        fn norm(n: &GroupNorm) -> LtxGroupNorm<'_> {
            LtxGroupNorm {
                w: &n.w,
                b: &n.b,
                groups: n.groups,
            }
        }
        fn res(rs: &[Res]) -> Vec<LtxUpRes<'_>> {
            rs.iter()
                .map(|r| LtxUpRes {
                    c1: conv(&r.c1),
                    n1: norm(&r.n1),
                    c2: conv(&r.c2),
                    n2: norm(&r.n2),
                })
                .collect()
        }
        let (up, spatial) = match &self.up {
            Upsampler::Spatial(c, w, b) => (
                LtxVaeConv {
                    w,
                    bias: b,
                    cin: c.cin,
                    cout: c.cout,
                },
                true,
            ),
            Upsampler::Temporal(c) => (conv(c), false),
        };
        let m = LtxUpsampler {
            conv_in: conv(&self.conv_in),
            norm_in: norm(&self.norm_in),
            pre: res(&self.pre),
            up,
            spatial,
            post: res(&self.post),
            conv_out: conv(&self.conv_out),
        };
        let mut out = Vec::new();
        if !s.upsample(&m, &x.data, x.t, x.h, x.w, &mut out, &mut |_, _| {}) {
            return None;
        }
        let (t, h, w) = if spatial {
            (x.t, 2 * x.h, 2 * x.w)
        } else {
            (2 * x.t - 1, x.h, x.w)
        };
        let c = self.conv_out.cout;
        (out.len() == t * h * w * c).then_some(Vid { t, h, w, c, data: out })
    }

    /// The upsampler on the CPU kernels.
    pub fn forward_cpu(&self, x: &Vid) -> Vid {
        let mut h = self.conv_in.forward_pad(x, TPad::Zero);
        self.norm_in.apply(&mut h, true);
        for r in &self.pre {
            h = r.forward(&h);
        }
        h = match &self.up {
            Upsampler::Spatial(conv, ..) => {
                let mut out = Vid::new(h.t, 2 * h.h, 2 * h.w, h.c);
                let n = out.h * out.w * out.c;
                for f in 0..h.t {
                    let fr = &h.data[f * h.h * h.w * h.c..(f + 1) * h.h * h.w * h.c];
                    let (y, _, _) = conv.forward(fr, h.h, h.w, 1);
                    let y = Vid {
                        t: 1,
                        h: h.h,
                        w: h.w,
                        c: conv.cout,
                        data: y,
                    };
                    let s = depth_to_space(&y, [1, 2, 2]);
                    out.data[f * n..(f + 1) * n].copy_from_slice(&s.data);
                }
                out
            }
            Upsampler::Temporal(conv) => {
                let y = conv.forward_pad(&h, TPad::Zero);
                let mut y = depth_to_space(&y, [2, 1, 1]);
                drop_first_frame(&mut y);
                y
            }
        };
        for r in &self.post {
            h = r.forward(&h);
        }
        self.conv_out.forward_pad(&h, TPad::Zero)
    }
}
