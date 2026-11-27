//! Apple Depth Pro 2 (`apple/DepthPro2-S`, `-B`, `-L`, `-H`): metric monocular depth at the
//! input's full resolution, with the shared focal length estimator (`apple/DepthPro2-focal`),
//! written from the reference package `depth_pro_2` (`ml-depth-pro-2`: `model.py`,
//! `network/encoder.py`, `network/decoder.py`, `network/fov.py`) and timm's DINOv3 ViT.
//!
//! The network:
//! - the image, ImageNet-normalized, is padded (edge replicated) to a multiple of 32 on the
//!   right and bottom;
//! - a DINOv3 ViT (S/16, B/16, L/16 or H+/16; a class and four register tokens, 2D rotary
//!   positions on the patch tokens normalized by the longer side, layer scale) runs over the
//!   16-pixel patches; for one frame the "cross-frame" blocks are plain blocks;
//! - four blocks' patch tokens are reassembled into a pyramid at 1/4, 1/8, 1/16 and 1/32 (a 1x1
//!   conv, then a 4x4 / 2x2 transposed conv, nothing, or a strided 3x3 conv);
//! - a DPT decoder fuses it from coarse to fine (3x3 convs, residual units, 2x2 transposed
//!   convs) to 1/2 resolution, and the head (conv, 2x2 transposed conv, conv, ReLU, 1x1 conv,
//!   ReLU) gives the focal-normalized inverse depth at full resolution;
//! - depth in meters = `f_px / (width * inverse_depth)`, the focal length `f_px` given (EXIF)
//!   or estimated: a DINOv3 ViT-S/16 over the image resized (antialiased bilinear) to a short
//!   side of at most 512 (centre-cropped beyond 16:9), its tokens of four blocks through an
//!   attention head that regresses the horizontal and vertical field of view.
//!
//! Everything runs in f32 as the reference does on the CPU; the matrices are f32 (exact) and run
//! through the shared matmul driver and convolution (GPU backend for large inputs).

use crate::ggml::{GgmlType, GgufMatrix};
use crate::kernels::for_chunks;
use crate::model::{Lin, linear};
use crate::platform::DevicePref;
use crate::sd::ops::{Conv, Tensors};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Inputs are padded to a multiple of this.
pub const INPUT_MULTIPLE: usize = 32;
/// The focal length estimator's repository.
pub const FOCAL_REPO: &str = "apple/DepthPro2-focal";

/// `config.json` of a Depth Pro 2 size.
#[derive(Clone, Debug, Deserialize)]
pub struct DepthConfig {
    pub backbone_preset: String,
    pub reassembled_blocks: Vec<usize>,
    #[serde(default)]
    pub cross_frame_blocks: Vec<usize>,
    pub dims_reassemble: Vec<usize>,
    pub dim_decoder: usize,
}

/// A DINOv3 ViT's shape (timm presets).
#[derive(Clone, Copy, Debug)]
struct VitShape {
    dim: usize,
    depth: usize,
    heads: usize,
}

fn vit_shape(preset: &str) -> Result<VitShape> {
    let p = preset.split('.').next().unwrap_or(preset);
    Ok(match p {
        "vit_small_patch16_dinov3" => VitShape {
            dim: 384,
            depth: 12,
            heads: 6,
        },
        "vit_small_plus_patch16_dinov3" => VitShape {
            dim: 384,
            depth: 12,
            heads: 6,
        },
        "vit_base_patch16_dinov3" => VitShape {
            dim: 768,
            depth: 12,
            heads: 12,
        },
        "vit_large_patch16_dinov3" => VitShape {
            dim: 1024,
            depth: 24,
            heads: 16,
        },
        "vit_huge_plus_patch16_dinov3" => VitShape {
            dim: 1280,
            depth: 32,
            heads: 20,
        },
        _ => bail!("unknown backbone preset {preset:?}"),
    })
}

/// Loader options. Keys for [`DepthOptions::set`]: `focal` (the focal length estimator: a
/// folder, a `.safetensors` file, a repo id, or `none`), `device`, `threads`, `cache_dir`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DepthOptions {
    /// The focal length estimator (default [`FOCAL_REPO`]; None: depth needs a given focal
    /// length, inverse depth is always there).
    pub focal: Option<String>,
    pub device: DevicePref,
    pub threads: Option<usize>,
    pub cache_dir: Option<PathBuf>,
}

impl Default for DepthOptions {
    fn default() -> Self {
        Self {
            focal: Some(FOCAL_REPO.to_string()),
            device: DevicePref::Auto,
            threads: None,
            cache_dir: None,
        }
    }
}

impl DepthOptions {
    pub const KEYS: &'static [&'static str] = &["focal", "device", "threads", "cache_dir"];

    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "focal" | "fov" => {
                self.focal = match value {
                    "" | "none" | "off" | "0" | "false" => None,
                    "auto" | "default" | "1" | "true" | "on" => Some(FOCAL_REPO.to_string()),
                    v => Some(v.to_string()),
                }
            }
            "device" => self.device = DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?}"))?,
            "threads" => self.threads = Some(value.parse().with_context(|| format!("threads: {value:?}"))?),
            "cache_dir" => self.cache_dir = Some(PathBuf::from(value)),
            _ => bail!("unknown depth option {key:?} (known: {})", Self::KEYS.join(", ")),
        }
        Ok(())
    }
}

/// One prediction.
#[derive(Clone, Debug)]
pub struct DepthMap {
    pub width: usize,
    pub height: usize,
    /// Focal-normalized inverse depth (`f_px / (width * depth)`), row-major.
    pub inverse_depth: Vec<f32>,
    /// The focal length in pixels used (given or estimated), when there is one.
    pub focal_px: Option<f32>,
    /// Whether `focal_px` was estimated.
    pub focal_estimated: bool,
    /// The estimated horizontal and vertical field of view in degrees.
    pub fov_deg: Option<[f32; 2]>,
}

impl DepthMap {
    /// Depth in meters (infinite where infinitely far), when there is a focal length.
    pub fn depth(&self) -> Option<Vec<f32>> {
        let f = self.focal_px?;
        let w = self.width as f32;
        Some(self.inverse_depth.iter().map(|&d| f / (w * d)).collect())
    }

    /// Metric inverse depth `1 / depth` in 1/m (zero where infinitely far), when there is a
    /// focal length.
    pub fn inverse_metric(&self) -> Option<Vec<f32>> {
        let f = self.focal_px?;
        let s = self.width as f32 / f;
        Some(self.inverse_depth.iter().map(|&d| d * s).collect())
    }

    /// The reference CLI's color scale: the largest metric inverse depth plus 20% headroom.
    pub fn color_scale(inv: &[f32]) -> f32 {
        let high = inv.iter().copied().fold(0f32, f32::max);
        if high > 0.0 { high * 1.2 } else { 1.0 }
    }

    /// The reference CLI's preview: metric inverse depth (or the focal-normalized one without a
    /// focal length) through the turbo color map, from 0 (infinitely far) to the scale. RGB8.
    pub fn colorize(&self) -> Vec<u8> {
        let inv = self.inverse_metric().unwrap_or_else(|| self.inverse_depth.clone());
        let scale = Self::color_scale(&inv);
        let mut out = Vec::with_capacity(inv.len() * 3);
        for &v in &inv {
            let i = (v * (256.0 / scale)).clamp(0.0, 255.0) as usize;
            out.extend_from_slice(&TURBO[i * 3..i * 3 + 3]);
        }
        out
    }

    /// The reference CLI's 16-bit PNG values: inverse depth over the color scale times 65535
    /// (0 is infinitely far), and the factor `k`: depth in meters = `k / value`.
    pub fn gray16(&self) -> (Vec<u16>, f32) {
        let inv = self.inverse_metric().unwrap_or_else(|| self.inverse_depth.clone());
        let scale = Self::color_scale(&inv);
        let v = inv
            .iter()
            .map(|&d| ((d / scale).clamp(0.0, 1.0) * 65535.0).round() as u16)
            .collect();
        (v, 65535.0 / scale)
    }

    /// The 5th and 95th percentiles of the finite depth (meters, on a strided sample) and the
    /// fraction of pixels infinitely far, as the reference CLI's summary.
    pub fn depth_range(&self) -> Option<(f32, f32, f32)> {
        let depth = self.depth()?;
        let mut finite = Vec::new();
        let mut total = 0usize;
        for y in (0..self.height).step_by(8) {
            for x in (0..self.width).step_by(8) {
                total += 1;
                let d = depth[y * self.width + x];
                if d.is_finite() && d > 0.0 {
                    finite.push(d);
                }
            }
        }
        let far = 1.0 - finite.len() as f32 / total.max(1) as f32;
        if finite.is_empty() {
            return Some((f32::INFINITY, f32::INFINITY, far));
        }
        finite.sort_by(f32::total_cmp);
        let at = |q: f32| finite[((finite.len() - 1) as f32 * q).round() as usize];
        Some((at(0.05), at(0.95), far))
    }
}

// ---------------------------------------------------------------------------------------------
// Layers.

/// A linear layer with its bias, f32 weights.
struct Dense {
    w: Lin<'static>,
    b: Option<Vec<f32>>,
    n_out: usize,
    n_in: usize,
}

impl Dense {
    fn new(w: &[f32], b: Option<Vec<f32>>, n_out: usize, n_in: usize) -> Result<Self> {
        if w.len() != n_out * n_in {
            bail!("weight of {} values, expected {n_out} x {n_in}", w.len());
        }
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(Self {
            w: Lin::gguf(GgufMatrix::from_bytes(GgmlType::F32, &bytes, n_out, n_in)?),
            b,
            n_out,
            n_in,
        })
    }

    fn load(t: &Tensors, p: &str, bias: bool) -> Result<Self> {
        let shape = t.shape(&format!("{p}.weight"))?;
        let (n_out, n_in) = match shape.as_slice() {
            [o, i] | [o, i, 1, 1] => (*o, *i),
            s => bail!("{p}.weight: expected a matrix, got {s:?}"),
        };
        let w = t.f32(&format!("{p}.weight"))?;
        let b = if bias && t.has(&format!("{p}.bias")) {
            Some(t.f32(&format!("{p}.bias"))?)
        } else {
            None
        };
        Self::new(&w, b, n_out, n_in).with_context(|| p.to_string())
    }

    fn fwd(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * self.n_out];
        if t == 0 {
            return y;
        }
        linear(&self.w, self.n_out, self.n_in, &x[..t * self.n_in], t, &mut y);
        if let Some(b) = &self.b {
            let n = self.n_out;
            for_chunks(&mut y, n * 64, t > 64, |_, c| {
                for r in c.chunks_mut(n) {
                    for (v, &bb) in r.iter_mut().zip(b) {
                        *v += bb;
                    }
                }
            });
        }
        y
    }
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl Norm {
    fn load(t: &Tensors, p: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            w: t.f32(&format!("{p}.weight"))?,
            b: t.f32(&format!("{p}.bias"))?,
            eps,
        })
    }

    fn fwd(&self, x: &[f32]) -> Vec<f32> {
        let c = self.w.len();
        let mut y = x.to_vec();
        let rows = (8192 / c).max(1);
        for_chunks(&mut y, rows * c, x.len() > rows * c, |_, blk| {
            for r in blk.chunks_mut(c) {
                let n = c as f32;
                let mean = r.iter().sum::<f32>() / n;
                let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
                let inv = 1.0 / (var + self.eps).sqrt();
                for ((v, &w), &b) in r.iter_mut().zip(&self.w).zip(&self.b) {
                    *v = (*v - mean) * inv * w + b;
                }
            }
        });
        y
    }
}

fn gelu_inplace(x: &mut [f32]) {
    for_chunks(x, 16384, x.len() > 32768, |_, c| {
        c.iter_mut().for_each(|v| *v = crate::sd::ops::gelu_erf(*v))
    });
}

fn relu(x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    relu_inplace(&mut y);
    y
}

fn relu_inplace(x: &mut [f32]) {
    for_chunks(x, 16384, x.len() > 32768, |_, c| c.iter_mut().for_each(|v| *v = v.max(0.0)));
}

fn add_into(a: &mut [f32], b: &[f32]) {
    for_chunks(a, 16384, a.len() > 32768, |i, c| {
        for (v, &w) in c.iter_mut().zip(&b[i * 16384..]) {
            *v += w;
        }
    });
}

/// A transposed convolution whose stride equals its kernel (no overlap): a matmul from each
/// input pixel to its `k x k` output block. Weights `[cin, cout, k, k]`.
struct Deconv {
    lin: Dense,
    k: usize,
    cout: usize,
}

impl Deconv {
    fn load(t: &Tensors, p: &str) -> Result<Self> {
        let shape = t.shape(&format!("{p}.weight"))?;
        let [cin, cout, kh, kw] = shape[..] else {
            bail!("{p}.weight: expected a 4-D kernel, got {shape:?}")
        };
        if kh != kw {
            bail!("{p}: square kernels only");
        }
        let k = kh;
        let w = t.f32(&format!("{p}.weight"))?;
        // Rows (ky, kx, co), columns ci.
        let mut m = vec![0f32; k * k * cout * cin];
        for ci in 0..cin {
            for co in 0..cout {
                for ky in 0..k {
                    for kx in 0..k {
                        m[((ky * k + kx) * cout + co) * cin + ci] = w[((ci * cout + co) * k + ky) * k + kx];
                    }
                }
            }
        }
        let b = if t.has(&format!("{p}.bias")) {
            let b = t.f32(&format!("{p}.bias"))?;
            Some((0..k * k).flat_map(|_| b.iter().copied()).collect())
        } else {
            None
        };
        Ok(Self {
            lin: Dense::new(&m, b, k * k * cout, cin)?,
            k,
            cout,
        })
    }

    /// `x` `[h*w, cin]` -> `[(k h) * (k w), cout]`.
    fn fwd(&self, x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let (k, c) = (self.k, self.cout);
        let y = self.lin.fwd(x, h * w);
        let (oh, ow) = (h * k, w * k);
        let mut out = vec![0f32; oh * ow * c];
        for_chunks(&mut out, ow * c, oh > 1, |oy, row| {
            let (iy, ky) = (oy / k, oy % k);
            for ox in 0..ow {
                let (ix, kx) = (ox / k, ox % k);
                let src = &y[(iy * w + ix) * k * k * c + (ky * k + kx) * c..][..c];
                row[ox * c..(ox + 1) * c].copy_from_slice(src);
            }
        });
        (out, oh, ow)
    }
}

fn conv(t: &Tensors, p: &str) -> Result<Conv> {
    let shape = t.shape(&format!("{p}.weight"))?;
    let [cout, cin, k, _] = shape[..] else {
        bail!("{p}.weight: expected a 4-D kernel, got {shape:?}")
    };
    let w = t.f32(&format!("{p}.weight"))?;
    let b = if t.has(&format!("{p}.bias")) {
        Some(t.f32(&format!("{p}.bias"))?)
    } else {
        None
    };
    Conv::from_kernel_exact(&w, cout, cin, k, b, true).with_context(|| p.to_string())
}

// ---------------------------------------------------------------------------------------------
// The DINOv3 ViT.

enum Mlp {
    Gelu(Box<Dense>, Box<Dense>),
    /// timm `SwiGLU`: `fc2(silu(fc1_g x) * fc1_x x)`.
    SwiGlu(Box<Dense>, Box<Dense>, Box<Dense>),
}

struct VitBlock {
    norm1: Norm,
    qkv: Dense,
    proj: Dense,
    gamma1: Option<Vec<f32>>,
    norm2: Norm,
    mlp: Mlp,
    gamma2: Option<Vec<f32>>,
}

impl VitBlock {
    fn load(t: &Tensors, p: &str, eps: f32) -> Result<Self> {
        let g = |n: &str| -> Result<Option<Vec<f32>>> {
            let k = format!("{p}.{n}");
            Ok(if t.has(&k) { Some(t.f32(&k)?) } else { None })
        };
        let mlp = if t.has(&format!("{p}.mlp.fc1_g.weight")) {
            Mlp::SwiGlu(
                Box::new(Dense::load(t, &format!("{p}.mlp.fc1_g"), true)?),
                Box::new(Dense::load(t, &format!("{p}.mlp.fc1_x"), true)?),
                Box::new(Dense::load(t, &format!("{p}.mlp.fc2"), true)?),
            )
        } else {
            Mlp::Gelu(
                Box::new(Dense::load(t, &format!("{p}.mlp.fc1"), true)?),
                Box::new(Dense::load(t, &format!("{p}.mlp.fc2"), true)?),
            )
        };
        Ok(Self {
            norm1: Norm::load(t, &format!("{p}.norm1"), eps)?,
            qkv: Dense::load(t, &format!("{p}.attn.qkv"), true)?,
            proj: Dense::load(t, &format!("{p}.attn.proj"), true)?,
            gamma1: g("gamma_1")?,
            norm2: Norm::load(t, &format!("{p}.norm2"), eps)?,
            mlp,
            gamma2: g("gamma_2")?,
        })
    }

    /// One block over `x` (`[n, dim]`), rotary positions `rope` on the tokens from `prefix` on.
    fn fwd(&self, x: &mut [f32], n: usize, heads: usize, prefix: usize, rope: Option<&Rope>) {
        let dim = self.norm1.w.len();
        let hd = dim / heads;
        let h = self.norm1.fwd(x);
        let qkv = self.qkv.fwd(&h, n);
        let mut q = vec![0f32; n * dim];
        let mut k = vec![0f32; n * dim];
        let mut v = vec![0f32; n * dim];
        for i in 0..n {
            q[i * dim..(i + 1) * dim].copy_from_slice(&qkv[i * 3 * dim..i * 3 * dim + dim]);
            k[i * dim..(i + 1) * dim].copy_from_slice(&qkv[i * 3 * dim + dim..i * 3 * dim + 2 * dim]);
            v[i * dim..(i + 1) * dim].copy_from_slice(&qkv[i * 3 * dim + 2 * dim..(i + 1) * 3 * dim]);
        }
        if let Some(r) = rope {
            r.apply(&mut q[prefix * dim..], heads, hd);
            r.apply(&mut k[prefix * dim..], heads, hd);
        }
        let o = crate::sd::ops::attention(&q, &k, &v, n, n, heads, hd, false);
        let mut a = self.proj.fwd(&o, n);
        if let Some(g) = &self.gamma1 {
            scale_rows(&mut a, g);
        }
        add_into(x, &a);
        let h = self.norm2.fwd(x);
        let mut m = match &self.mlp {
            Mlp::Gelu(f1, f2) => {
                let mut u = f1.fwd(&h, n);
                gelu_inplace(&mut u);
                f2.fwd(&u, n)
            }
            Mlp::SwiGlu(g, xl, f2) => {
                let mut a = g.fwd(&h, n);
                let b = xl.fwd(&h, n);
                for_chunks(&mut a, 16384, true, |i, c| {
                    for (v, &w) in c.iter_mut().zip(&b[i * 16384..]) {
                        *v = crate::sd::ops::silu(*v) * w;
                    }
                });
                f2.fwd(&a, n)
            }
        };
        if let Some(g) = &self.gamma2 {
            scale_rows(&mut m, g);
        }
        add_into(x, &m);
    }
}

fn scale_rows(x: &mut [f32], g: &[f32]) {
    let c = g.len();
    for_chunks(x, c * 64, x.len() > c * 64, |_, blk| {
        for r in blk.chunks_mut(c) {
            for (v, &s) in r.iter_mut().zip(g) {
                *v *= s;
            }
        }
    });
}

/// DINOv3's 2D rotary positions of a `gh x gw` grid (`RotaryEmbeddingDinoV3`, temperature 100,
/// coordinates 0.5-centred, normalized by the longer side, in [-1, 1]; halves rotated).
struct Rope {
    /// Per patch, `head_dim / 2` angles (16 along y, then 16 along x for 64 dimensions).
    sin: Vec<f32>,
    cos: Vec<f32>,
    half: usize,
}

impl Rope {
    fn new(gh: usize, gw: usize, head_dim: usize) -> Self {
        let q = head_dim / 4;
        let periods: Vec<f32> = (0..q).map(|i| 100f32.powf(2.0 * i as f32 / (head_dim / 2) as f32)).collect();
        let d = gh.max(gw) as f32;
        let half = 2 * q;
        let mut sin = Vec::with_capacity(gh * gw * half);
        let mut cos = Vec::with_capacity(gh * gw * half);
        for y in 0..gh {
            for x in 0..gw {
                let cy = 2.0 * ((y as f32 + 0.5) / d) - 1.0;
                let cx = 2.0 * ((x as f32 + 0.5) / d) - 1.0;
                for c in [cy, cx] {
                    for &p in &periods {
                        let a = 2.0 * std::f32::consts::PI * c / p;
                        sin.push(a.sin());
                        cos.push(a.cos());
                    }
                }
            }
        }
        Self { sin, cos, half }
    }

    /// Rotates the patch rows `x` (`[patches, heads * head_dim]`).
    fn apply(&self, x: &mut [f32], heads: usize, hd: usize) {
        let half = self.half;
        let dim = heads * hd;
        for_chunks(x, dim * 16, x.len() > dim * 64, |b, blk| {
            for (j, row) in blk.chunks_mut(dim).enumerate() {
                let p = b * 16 + j;
                let (s, c) = (&self.sin[p * half..(p + 1) * half], &self.cos[p * half..(p + 1) * half]);
                for h in 0..heads {
                    let v = &mut row[h * hd..(h + 1) * hd];
                    for i in 0..half {
                        let (a, b2) = (v[i], v[i + half]);
                        v[i] = a * c[i] - b2 * s[i];
                        v[i + half] = b2 * c[i] + a * s[i];
                    }
                }
            }
        });
    }
}

struct Vit {
    shape: VitShape,
    patch: Dense, // [dim][3 * 16 * 16]
    cls: Vec<f32>,
    reg: Vec<f32>,
    blocks: Vec<VitBlock>,
}

const PATCH: usize = 16;

impl Vit {
    /// The ViT under `p`, its shape from the preset when known (else from the weights: the
    /// width of the class token, the blocks present, DINOv3's 64-wide heads).
    fn load(t: &Tensors, p: &str, preset: &str) -> Result<Self> {
        let cls = t.f32(&format!("{p}.cls_token"))?;
        let mut depth = 0;
        while t.has(&format!("{p}.blocks.{depth}.norm1.weight")) {
            depth += 1;
        }
        let shape = match vit_shape(preset) {
            Ok(s) => s,
            Err(_) if cls.len() >= 64 && depth > 0 => VitShape {
                dim: cls.len(),
                depth,
                heads: cls.len() / 64,
            },
            Err(e) => return Err(e),
        };
        if cls.len() != shape.dim || depth != shape.depth {
            bail!(
                "{p}: {} blocks of width {}, expected {} of {} ({preset})",
                depth,
                cls.len(),
                shape.depth,
                shape.dim
            );
        }
        let pw = t.f32(&format!("{p}.patch_embed.proj.weight"))?;
        let pb = t.f32(&format!("{p}.patch_embed.proj.bias"))?;
        let patch = Dense::new(&pw, Some(pb), shape.dim, 3 * PATCH * PATCH)?;
        let blocks = (0..shape.depth)
            .map(|i| VitBlock::load(t, &format!("{p}.blocks.{i}"), 1e-5))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shape,
            patch,
            cls,
            reg: t.f32(&format!("{p}.reg_token"))?,
            blocks,
        })
    }

    fn prefix(&self) -> usize {
        1 + self.reg.len() / self.shape.dim
    }

    /// Tokens `[prefix + gh*gw, dim]` of a normalized image `[3][h][w]` (h, w multiples of 16).
    fn embed(&self, img: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (gh, gw) = (h / PATCH, w / PATCH);
        let pp = PATCH * PATCH;
        let mut cols = vec![0f32; gh * gw * 3 * pp];
        for_chunks(&mut cols, 3 * pp, gh * gw > 1, |i, r| {
            let (py, px) = (i / gw, i % gw);
            for c in 0..3 {
                for ky in 0..PATCH {
                    let src = &img[(c * h + py * PATCH + ky) * w + px * PATCH..][..PATCH];
                    r[c * pp + ky * PATCH..c * pp + (ky + 1) * PATCH].copy_from_slice(src);
                }
            }
        });
        let pt = self.patch.fwd(&cols, gh * gw);
        let mut x = Vec::with_capacity((self.prefix() + gh * gw) * self.shape.dim);
        x.extend_from_slice(&self.cls);
        x.extend_from_slice(&self.reg);
        x.extend_from_slice(&pt);
        x
    }
}

// ---------------------------------------------------------------------------------------------
// Depth.

struct Fusion {
    resnet1: Option<(Conv, Conv)>,
    resnet2: (Conv, Conv),
    deconv: Deconv,
    out_conv: Dense,
}

fn residual(r: &(Conv, Conv), x: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (a, _, _) = r.0.forward(&relu(x), h, w, 1);
    let (mut b, _, _) = r.1.forward(&relu(&a), h, w, 1);
    add_into(&mut b, x);
    b
}

impl Fusion {
    fn fwd(&self, x: &[f32], skip: Option<&[f32]>, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let mut x = x.to_vec();
        if let (Some(r), Some(s)) = (&self.resnet1, skip) {
            let y = residual(r, s, h, w);
            add_into(&mut x, &y);
        }
        let x = residual(&self.resnet2, &x, h, w);
        let (x, oh, ow) = self.deconv.fwd(&x, h, w);
        (self.out_conv.fwd(&x, oh * ow), oh, ow)
    }
}

enum Reassemble {
    Up(Box<Dense>, Box<Deconv>),
    Same(Box<Dense>),
    Down(Box<Dense>, Box<Conv>),
}

/// A loaded Depth Pro 2 size, with its focal length estimator when given.
pub struct DepthPro {
    cfg: DepthConfig,
    vit: Vit,
    reassemble: Vec<Reassemble>,
    convs: Vec<Conv>,
    fusions: Vec<Fusion>,
    head: (Conv, Deconv, Conv, Dense),
    fov: Option<FovEstimator>,
    info: DepthInfo,
}

/// What a loaded model is.
#[derive(Clone, Debug, serde::Serialize)]
pub struct DepthInfo {
    pub path: PathBuf,
    pub backbone: String,
    pub params: usize,
    pub focal_estimator: Option<PathBuf>,
    pub device: String,
    pub load_secs: f64,
}

/// Whether `dir` holds a Depth Pro 2 checkpoint (`config.json` with a DINOv3 `backbone_preset`
/// and `dims_reassemble`).
pub fn is_depth_pro(dir: &Path) -> bool {
    let Ok(b) = std::fs::read(dir.join("config.json")) else {
        return false;
    };
    serde_json::from_slice::<DepthConfig>(&b).is_ok_and(|c| c.backbone_preset.contains("dinov3"))
}

/// Whether `spec` is one of the Depth Pro 2 repository ids.
pub fn is_repo_id(spec: &str) -> bool {
    let s = spec.to_ascii_lowercase();
    s.starts_with("apple/depthpro2-") && !s.ends_with("-focal")
}

fn resolve_dir(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    if p.is_file() {
        return Ok(p.parent().unwrap_or(Path::new(".")).to_path_buf());
    }
    #[cfg(feature = "hf")]
    {
        crate::hf::resolve_files(spec, &["config.json", "model.safetensors"], cache_dir)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = cache_dir;
        bail!("{spec}: not a local folder (built without the `hf` feature, so repo ids cannot be fetched)")
    }
}

fn model_file(dir: &Path) -> Result<PathBuf> {
    let f = dir.join("model.safetensors");
    if f.is_file() {
        return Ok(f);
    }
    bail!("{}: no model.safetensors", dir.display())
}

impl DepthPro {
    /// Loads a size (a folder with `config.json` and `model.safetensors`, or a repo id such as
    /// `apple/DepthPro2-S`) and, per `opts.focal`, the focal length estimator.
    pub fn load(spec: &str, opts: &DepthOptions) -> Result<Self> {
        let t0 = std::time::Instant::now();
        if let Some(n) = opts.threads {
            let mut o = crate::pool::options();
            o.threads = n;
            crate::pool::configure_with(o);
        }
        let gpu = crate::gpu::ensure(opts.device)?;
        let dir = resolve_dir(spec, opts.cache_dir.as_deref())?;
        let cfg: DepthConfig = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)
            .with_context(|| format!("{}: config.json", dir.display()))?;
        if cfg.reassembled_blocks.len() != 4 || cfg.dims_reassemble.len() != 4 {
            bail!("Depth Pro 2 has four reassembled levels");
        }
        let path = model_file(&dir)?;
        let t = Tensors::open(&path)?;
        let vit = Vit::load(&t, "encoder", &cfg.backbone_preset)?;
        if cfg.reassembled_blocks.iter().any(|&b| b >= vit.shape.depth) {
            bail!("reassembled block beyond the backbone's {} blocks", vit.shape.depth);
        }
        let mut reassemble = Vec::new();
        for i in 0..4 {
            let p = format!("encoder.reassemble.{i}");
            let first = Dense::load(&t, &format!("{p}.0"), true)?;
            reassemble.push(match i {
                0 | 1 => Reassemble::Up(Box::new(first), Box::new(Deconv::load(&t, &format!("{p}.1"))?)),
                2 => Reassemble::Same(Box::new(first)),
                _ => Reassemble::Down(Box::new(first), Box::new(conv(&t, &format!("{p}.1"))?)),
            });
        }
        let convs = (0..4)
            .map(|i| conv(&t, &format!("decoder.convs.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let mut fusions = Vec::new();
        for i in 0..4 {
            let p = format!("decoder.fusions.{i}");
            let rb = |n: &str| -> Result<(Conv, Conv)> {
                Ok((conv(&t, &format!("{p}.{n}.residual.1"))?, conv(&t, &format!("{p}.{n}.residual.3"))?))
            };
            fusions.push(Fusion {
                resnet1: if i < 3 { Some(rb("resnet1")?) } else { None },
                resnet2: rb("resnet2")?,
                deconv: Deconv::load(&t, &format!("{p}.deconv"))?,
                out_conv: Dense::load(&t, &format!("{p}.out_conv"), true)?,
            });
        }
        let head = (
            conv(&t, "head.0")?,
            Deconv::load(&t, "head.1")?,
            conv(&t, "head.2")?,
            Dense::load(&t, "head.4", true)?,
        );
        let params = t.names().filter_map(|n| t.shape(n).ok()).map(|s| s.iter().product::<usize>()).sum();
        let (fov, fov_path) = match &opts.focal {
            None => (None, None),
            Some(f) => {
                let fd = resolve_dir(f, opts.cache_dir.as_deref())?;
                let fp = if Path::new(f).is_file() {
                    PathBuf::from(f)
                } else {
                    model_file(&fd)?
                };
                (Some(FovEstimator::load(&fp)?), Some(fp))
            }
        };
        let info = DepthInfo {
            path,
            backbone: cfg.backbone_preset.clone(),
            params,
            focal_estimator: fov_path,
            device: gpu.map_or("cpu".to_string(), |b| b.name().to_string()),
            load_secs: t0.elapsed().as_secs_f64(),
        };
        Ok(Self {
            cfg,
            vit,
            reassemble,
            convs,
            fusions,
            head,
            fov,
            info,
        })
    }

    pub fn info(&self) -> &DepthInfo {
        &self.info
    }

    pub fn config(&self) -> &DepthConfig {
        &self.cfg
    }

    /// Whether the focal length estimator is loaded.
    pub fn has_focal_estimator(&self) -> bool {
        self.fov.is_some()
    }

    /// Predicts an RGB8 image (`width * height * 3` bytes): inverse depth, and with `f_px` (or
    /// the estimator) the focal length that gives metric depth.
    pub fn infer_rgb(&self, rgb: &[u8], width: usize, height: usize, f_px: Option<f32>) -> Result<DepthMap> {
        if rgb.len() != width * height * 3 || width == 0 || height == 0 {
            bail!("RGB image of {} bytes for {width} x {height}", rgb.len());
        }
        let img = normalize(rgb, width, height);
        let inverse_depth = self.inverse_depth(&img, height, width);
        let (focal_px, fov_deg, est) = match (f_px, &self.fov) {
            (Some(f), _) => (Some(f), None, false),
            (None, Some(fov)) => {
                let _p = crate::prof::phase(10, "depth focal");
                let a = fov.fov_deg(&img, height, width);
                (Some(focal_from_fov(a, height, width)), Some(a), true)
            }
            (None, None) => (None, None, false),
        };
        Ok(DepthMap {
            width,
            height,
            inverse_depth,
            focal_px,
            focal_estimated: est,
            fov_deg,
        })
    }

    /// The estimated horizontal and vertical field of view (degrees) of an RGB8 image.
    pub fn estimate_fov(&self, rgb: &[u8], width: usize, height: usize) -> Result<[f32; 2]> {
        let fov = self.fov.as_ref().context("no focal length estimator loaded")?;
        Ok(fov.fov_deg(&normalize(rgb, width, height), height, width))
    }

    /// Focal-normalized inverse depth `[h][w]` of a normalized image `[3][h][w]`.
    fn inverse_depth(&self, img: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (hp, wp) = (h.next_multiple_of(INPUT_MULTIPLE), w.next_multiple_of(INPUT_MULTIPLE));
        let padded;
        let x = if (hp, wp) != (h, w) {
            padded = pad_replicate(img, h, w, hp, wp);
            &padded[..]
        } else {
            img
        };
        let (gh, gw) = (hp / PATCH, wp / PATCH);
        let vit = &self.vit;
        let dim = vit.shape.dim;
        let np = vit.prefix();
        let mut tokens = vit.embed(x, hp, wp);
        let n = np + gh * gw;
        let rope = Rope::new(gh, gw, dim / vit.shape.heads);
        let mut taps = Vec::new();
        let pv = crate::prof::phase(8, "depth backbone");
        for (i, b) in vit.blocks.iter().enumerate() {
            // A cross-frame block attends over every frame's patches with the frames' mean class
            // and register tokens, and hands each frame its own prefix back: for one frame, the
            // block with its prefix tokens left as they came in.
            let prefix = self.cfg.cross_frame_blocks.contains(&i).then(|| tokens[..np * dim].to_vec());
            b.fwd(&mut tokens, n, vit.shape.heads, np, Some(&rope));
            if let Some(p) = prefix {
                tokens[..np * dim].copy_from_slice(&p);
            }
            if self.cfg.reassembled_blocks.contains(&i) {
                taps.push(tokens[np * dim..].to_vec());
            }
        }
        drop(pv);
        let _pd = crate::prof::phase(9, "depth decoder");
        // The pyramid, HWC: 1/4, 1/8, 1/16, 1/32.
        let mut levels: Vec<(Vec<f32>, usize, usize)> = Vec::new();
        for (r, f) in self.reassemble.iter().zip(&taps) {
            levels.push(match r {
                Reassemble::Up(a, d) => {
                    let y = a.fwd(f, gh * gw);
                    d.fwd(&y, gh, gw)
                }
                Reassemble::Same(a) => (a.fwd(f, gh * gw), gh, gw),
                Reassemble::Down(a, c) => {
                    let y = a.fwd(f, gh * gw);
                    c.forward(&y, gh, gw, 2)
                }
            });
        }
        let enc: Vec<(Vec<f32>, usize, usize)> = levels
            .iter()
            .zip(&self.convs)
            .map(|((x, lh, lw), c)| c.forward(x, *lh, *lw, 1))
            .collect();
        let (e3, h3, w3) = &enc[3];
        let (mut f, mut fh, mut fw) = self.fusions[3].fwd(e3, None, *h3, *w3);
        for i in (0..3).rev() {
            let (e, eh, ew) = &enc[i];
            debug_assert_eq!((fh, fw), (*eh, *ew));
            (f, fh, fw) = self.fusions[i].fwd(&f, Some(e), *eh, *ew);
        }
        let (c0, d1, c2, c4) = &self.head;
        let (y, _, _) = c0.forward(&f, fh, fw, 1);
        let (y, oh, ow) = d1.fwd(&y, fh, fw);
        let (mut y, _, _) = c2.forward(&y, oh, ow, 1);
        relu_inplace(&mut y);
        let mut y = c4.fwd(&y, oh * ow);
        relu_inplace(&mut y);
        debug_assert_eq!((oh, ow), (hp, wp));
        let mut out = Vec::with_capacity(h * w);
        for r in 0..h {
            out.extend_from_slice(&y[r * ow..r * ow + w]);
        }
        out
    }
}

/// An RGB8 image ImageNet-normalized, `[3][h][w]`.
pub fn normalize(rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * h * w];
    for c in 0..3 {
        let (m, s) = (IMAGENET_MEAN[c], IMAGENET_STD[c]);
        for i in 0..h * w {
            out[c * h * w + i] = (rgb[i * 3 + c] as f32 / 255.0 - m) / s;
        }
    }
    out
}

/// `[3][h][w]` padded to `[3][hp][wp]` by replicating the last row and column.
fn pad_replicate(x: &[f32], h: usize, w: usize, hp: usize, wp: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * hp * wp];
    for c in 0..3 {
        for y in 0..hp {
            let sy = y.min(h - 1);
            for xx in 0..wp {
                out[(c * hp + y) * wp + xx] = x[(c * h + sy) * w + xx.min(w - 1)];
            }
        }
    }
    out
}

/// The focal length in pixels of the shorter side's field of view (`median_focal_length_px`,
/// one frame).
pub fn focal_from_fov(fov: [f32; 2], h: usize, w: usize) -> f32 {
    if w < h {
        0.5 * w as f32 / (fov[0].to_radians() / 2.0).tan()
    } else {
        0.5 * h as f32 / (fov[1].to_radians() / 2.0).tan()
    }
}

/// torch's antialiased bilinear resize (`F.interpolate(..., mode="bilinear", antialias=True)`,
/// `align_corners=False`) of `[c][h][w]` to `[c][oh][ow]`.
pub fn resize_bilinear_aa(x: &[f32], c: usize, h: usize, w: usize, oh: usize, ow: usize) -> Vec<f32> {
    fn weights(n_in: usize, n_out: usize) -> Vec<(usize, Vec<f32>)> {
        let scale = n_in as f64 / n_out as f64;
        let support = if scale >= 1.0 { scale } else { 1.0 };
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        (0..n_out)
            .map(|i| {
                let center = scale * (i as f64 + 0.5);
                let xmin = ((center - support + 0.5) as i64).max(0) as usize;
                let xmax = ((center + support + 0.5) as i64).min(n_in as i64) as usize;
                let mut ws: Vec<f64> = (xmin..xmax)
                    .map(|j| {
                        let t = ((j as f64 - center + 0.5) * invscale).abs();
                        if t < 1.0 { 1.0 - t } else { 0.0 }
                    })
                    .collect();
                let total: f64 = ws.iter().sum();
                if total != 0.0 {
                    ws.iter_mut().for_each(|v| *v /= total);
                }
                (xmin, ws.into_iter().map(|v| v as f32).collect())
            })
            .collect()
    }
    let wx = weights(w, ow);
    let wy = weights(h, oh);
    // Horizontal, then vertical (as torch's separable passes).
    let mut tmp = vec![0f32; c * h * ow];
    for_chunks(&mut tmp, ow, true, |r, row| {
        let src = &x[r * w..(r + 1) * w];
        for (o, (x0, ws)) in row.iter_mut().zip(&wx) {
            *o = ws.iter().enumerate().map(|(j, &k)| k * src[x0 + j]).sum();
        }
    });
    let mut out = vec![0f32; c * oh * ow];
    for_chunks(&mut out, ow, true, |r, row| {
        let (ci, oy) = (r / oh, r % oh);
        let (y0, ws) = &wy[oy];
        for (ox, o) in row.iter_mut().enumerate() {
            *o = ws.iter().enumerate().map(|(j, &k)| k * tmp[(ci * h + y0 + j) * ow + ox]).sum();
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------
// The focal length estimator.

/// A timm `Block` (LayerNorm eps 1e-6, qkv bias, GELU MLP, no layer scale).
struct FovBlock(VitBlock);

struct FovEstimator {
    vit: Vit,
    tap_norms: Vec<Norm>,
    proj: Dense,
    query: Vec<f32>,
    prefix_embed: Vec<f32>,
    blocks: Vec<FovBlock>,
    norm: Norm,
    out: Dense,
    heads: usize,
}

const FOV_TAPS: [usize; 4] = [2, 5, 8, 11];
const FOV_CROSS: [usize; 4] = [5, 7, 9, 11];
const MAX_ASPECT: f32 = 1.7778;

impl FovEstimator {
    fn load(path: &Path) -> Result<Self> {
        let t = Tensors::open(path)?;
        let vit = Vit::load(
            &t,
            "trunk",
            if t.f32("trunk.cls_token")?.len() == 384 {
                "vit_small_patch16_dinov3"
            } else {
                ""
            },
        )?;
        let tap_norms = (0..4)
            .map(|i| Norm::load(&t, &format!("head.tap_norms.{i}"), 1e-5))
            .collect::<Result<Vec<_>>>()?;
        let mut blocks = Vec::new();
        let mut i = 0;
        while t.has(&format!("head.blocks.{i}.norm1.weight")) {
            blocks.push(FovBlock(VitBlock::load(&t, &format!("head.blocks.{i}"), 1e-6)?));
            i += 1;
        }
        Ok(Self {
            vit,
            tap_norms,
            proj: Dense::load(&t, "head.proj", true)?,
            query: t.f32("head.query")?,
            prefix_embed: t.f32("head.prefix_embed")?,
            blocks,
            norm: Norm::load(&t, "head.norm", 1e-5)?,
            out: Dense::load(&t, "head.out", true)?,
            heads: 4,
        })
    }

    /// `input_size`: the resize target, never upsampled.
    fn input_size(h: usize, w: usize) -> (usize, usize) {
        let short = 32usize.max(512.min(h.min(w) / 32 * 32));
        let scale = short as f64 / h.min(w) as f64;
        let r = |v: usize| 32usize.max(((v as f64 * scale / 32.0).round_ties_even() as usize) * 32);
        (r(h), r(w))
    }

    /// The horizontal and vertical field of view in degrees of a normalized image `[3][h][w]`.
    fn fov_deg(&self, img: &[f32], h: usize, w: usize) -> [f32; 2] {
        let (nh, nw) = Self::input_size(h, w);
        let y = resize_bilinear_aa(img, 3, h, w, nh, nw);
        let (lo, hi) = (nh.min(nw), nh.max(nw));
        let (y, ch, cw) = if (hi as f32) <= MAX_ASPECT * lo as f32 {
            (y, nh, nw)
        } else {
            let keep = 32usize.max(((MAX_ASPECT * lo as f32) as usize) / 32 * 32);
            let off = (hi - keep) / 2;
            let (ch, cw) = if nw > nh { (nh, keep) } else { (keep, nw) };
            let mut c = vec![0f32; 3 * ch * cw];
            for ci in 0..3 {
                for r in 0..ch {
                    for col in 0..cw {
                        let (sr, sc) = if nw > nh { (r, col + off) } else { (r + off, col) };
                        c[(ci * ch + r) * cw + col] = y[(ci * nh + sr) * nw + sc];
                    }
                }
            }
            (c, ch, cw)
        };
        let fov = self.forward(&y, ch, cw);
        if (ch, cw) == (nh, nw) {
            return fov;
        }
        // `crop_angles_to_full`.
        let (crop_h, crop_w) = (ch as f32 * h as f32 / nh as f32, cw as f32 * w as f32 / nw as f32);
        let fx = (crop_w / 2.0) / (fov[0].to_radians() / 2.0).tan();
        let fy = (crop_h / 2.0) / (fov[1].to_radians() / 2.0).tan();
        [
            (2.0 * ((w as f32 / 2.0) / fx).atan()).to_degrees(),
            (2.0 * ((h as f32 / 2.0) / fy).atan()).to_degrees(),
        ]
    }

    fn forward(&self, img: &[f32], h: usize, w: usize) -> [f32; 2] {
        let vit = &self.vit;
        let dim = vit.shape.dim;
        let np = vit.prefix();
        let (gh, gw) = (h / PATCH, w / PATCH);
        let n = np + gh * gw;
        let mut tokens = vit.embed(img, h, w);
        let rope = Rope::new(gh, gw, dim / vit.shape.heads);
        let mut taps = Vec::new();
        for (i, b) in vit.blocks.iter().enumerate() {
            let before = tokens[..np * dim].to_vec();
            b.fwd(&mut tokens, n, vit.shape.heads, np, Some(&rope));
            if FOV_CROSS.contains(&i) {
                tokens[..np * dim].copy_from_slice(&before);
            }
            if FOV_TAPS.contains(&i) {
                taps.push(tokens.clone());
            }
        }
        // The taps normed and concatenated per token, projected.
        let mut cat = vec![0f32; n * 4 * dim];
        for (k, (t, norm)) in taps.iter().zip(&self.tap_norms).enumerate() {
            let tn = norm.fwd(t);
            for r in 0..n {
                cat[r * 4 * dim + k * dim..r * 4 * dim + (k + 1) * dim].copy_from_slice(&tn[r * dim..(r + 1) * dim]);
            }
        }
        let x = self.proj.fwd(&cat, n);
        let d = self.proj.n_out;
        // Average pool 2 (ceil mode) over the patch grid, plus the sin-cos code.
        let (ph, pw) = (gh.div_ceil(2), gw.div_ceil(2));
        let code = sincos_2d(ph, pw, d);
        let mut seq = Vec::with_capacity((1 + np + ph * pw) * d);
        seq.extend_from_slice(&self.query);
        for p in 0..np {
            for j in 0..d {
                seq.push(x[p * d + j] + self.prefix_embed[p * d + j]);
            }
        }
        for py in 0..ph {
            for px in 0..pw {
                let mut acc = vec![0f32; d];
                let mut cnt = 0f32;
                for y in 2 * py..(2 * py + 2).min(gh) {
                    for xx in 2 * px..(2 * px + 2).min(gw) {
                        let r = np + y * gw + xx;
                        for (a, &v) in acc.iter_mut().zip(&x[r * d..(r + 1) * d]) {
                            *a += v;
                        }
                        cnt += 1.0;
                    }
                }
                let c = &code[(py * pw + px) * d..(py * pw + px + 1) * d];
                for (a, &cv) in acc.iter().zip(c) {
                    seq.push(a / cnt + cv);
                }
            }
        }
        let len = 1 + np + ph * pw;
        for b in &self.blocks {
            b.0.fwd(&mut seq, len, self.heads, len, None);
        }
        let q = self.norm.fwd(&seq[..d]);
        let logits = self.out.fwd(&q, 1);
        let s = |v: f32| 1.0 + (1.0 / (1.0 + (-v).exp())) * 178.0;
        [s(logits[0]), s(logits[1])]
    }
}

/// `sincos_2d`: `[gh*gw, dim]`, both axes scaled by the longer side.
fn sincos_2d(gh: usize, gw: usize, dim: usize) -> Vec<f32> {
    let g = gh.max(gw) as f32;
    let nf = dim / 4;
    let freq: Vec<f32> = (0..nf)
        .map(|i| {
            let t = if nf > 1 { 7.0 * i as f32 / (nf - 1) as f32 } else { 0.0 };
            std::f32::consts::PI * 2f32.powf(t)
        })
        .collect();
    let mut out = Vec::with_capacity(gh * gw * dim);
    for y in 0..gh {
        for x in 0..gw {
            let ys = (y as f32 + 0.5 - gh as f32 / 2.0) / g;
            let xs = (x as f32 + 0.5 - gw as f32 / 2.0) / g;
            for c in [ys, xs] {
                for &f in &freq {
                    out.push((c * f).sin());
                }
                for &f in &freq {
                    out.push((c * f).cos());
                }
            }
        }
    }
    out
}

/// matplotlib's `turbo`, 256 entries (the reference CLI's `COLOR_LUT`).
pub static TURBO: [u8; 768] = [
    48, 18, 59, 49, 21, 66, 50, 24, 74, 52, 27, 81, 53, 30, 88, 54, 33, 95, 55, 35, 101, 56, 38, 108, 57, 41, 114, 58, 44, 121, 59, 47,
    127, 60, 50, 133, 60, 53, 139, 61, 55, 145, 62, 58, 150, 63, 61, 156, 64, 64, 161, 64, 67, 166, 65, 69, 171, 65, 72, 176, 66, 75, 181,
    67, 78, 186, 67, 80, 190, 67, 83, 194, 68, 86, 199, 68, 88, 203, 69, 91, 206, 69, 94, 210, 69, 96, 214, 69, 99, 217, 70, 102, 221, 70,
    104, 224, 70, 107, 227, 70, 109, 230, 70, 112, 232, 70, 115, 235, 70, 117, 237, 70, 120, 240, 70, 122, 242, 70, 125, 244, 70, 127, 246,
    70, 130, 248, 69, 132, 249, 69, 135, 251, 69, 137, 252, 68, 140, 253, 67, 142, 253, 66, 145, 254, 65, 147, 254, 64, 150, 254, 63, 152,
    254, 62, 155, 254, 60, 157, 253, 59, 160, 252, 57, 162, 252, 56, 165, 251, 54, 168, 249, 52, 170, 248, 51, 172, 246, 49, 175, 245, 47,
    177, 243, 45, 180, 241, 43, 182, 239, 42, 185, 237, 40, 187, 235, 38, 189, 233, 37, 192, 230, 35, 194, 228, 33, 196, 225, 32, 198, 223,
    30, 201, 220, 29, 203, 218, 28, 205, 215, 27, 207, 212, 26, 209, 210, 25, 211, 207, 24, 213, 204, 24, 215, 202, 23, 217, 199, 23, 218,
    196, 23, 220, 194, 23, 222, 191, 24, 224, 189, 24, 225, 186, 25, 227, 184, 26, 228, 182, 27, 229, 180, 29, 231, 177, 30, 232, 175, 32,
    233, 172, 34, 235, 169, 36, 236, 166, 39, 237, 163, 41, 238, 160, 44, 239, 157, 47, 240, 154, 50, 241, 151, 53, 243, 148, 56, 244, 145,
    59, 244, 141, 63, 245, 138, 66, 246, 135, 70, 247, 131, 74, 248, 128, 77, 249, 124, 81, 249, 121, 85, 250, 118, 89, 251, 114, 93, 251,
    111, 97, 252, 108, 101, 252, 104, 105, 253, 101, 109, 253, 98, 113, 253, 95, 116, 254, 92, 120, 254, 89, 124, 254, 86, 128, 254, 83,
    132, 254, 80, 135, 254, 77, 139, 254, 75, 142, 254, 72, 146, 254, 70, 149, 254, 68, 152, 254, 66, 155, 253, 64, 158, 253, 62, 161, 252,
    61, 164, 252, 59, 166, 251, 58, 169, 251, 57, 172, 250, 55, 174, 249, 55, 177, 248, 54, 179, 248, 53, 182, 247, 53, 185, 245, 52, 187,
    244, 52, 190, 243, 52, 192, 242, 51, 195, 241, 51, 197, 239, 51, 200, 238, 51, 202, 237, 51, 205, 235, 52, 207, 234, 52, 209, 232, 52,
    212, 231, 53, 214, 229, 53, 216, 227, 53, 218, 226, 54, 221, 224, 54, 223, 222, 54, 225, 220, 55, 227, 218, 55, 229, 216, 56, 231, 215,
    56, 232, 213, 56, 234, 211, 57, 236, 209, 57, 237, 207, 57, 239, 205, 57, 240, 203, 58, 242, 200, 58, 243, 198, 58, 244, 196, 58, 246,
    194, 58, 247, 192, 57, 248, 190, 57, 249, 188, 57, 249, 186, 56, 250, 183, 55, 251, 181, 55, 251, 179, 54, 252, 176, 53, 252, 174, 52,
    253, 171, 51, 253, 169, 50, 253, 166, 49, 253, 163, 48, 254, 161, 47, 254, 158, 46, 254, 155, 45, 254, 152, 44, 253, 149, 43, 253, 146,
    41, 253, 143, 40, 253, 140, 39, 252, 137, 38, 252, 134, 36, 251, 131, 35, 251, 128, 34, 250, 125, 32, 250, 122, 31, 249, 119, 30, 248,
    116, 28, 247, 113, 27, 247, 110, 26, 246, 107, 24, 245, 104, 23, 244, 101, 22, 243, 99, 21, 242, 96, 20, 241, 93, 19, 239, 90, 17, 238,
    88, 16, 237, 85, 15, 236, 82, 14, 234, 80, 13, 233, 77, 13, 232, 75, 12, 230, 73, 11, 229, 70, 10, 227, 68, 10, 226, 66, 9, 224, 64, 8,
    222, 62, 8, 221, 60, 7, 219, 58, 7, 217, 56, 6, 215, 54, 6, 214, 52, 5, 212, 50, 5, 210, 48, 5, 208, 47, 4, 206, 45, 4, 203, 43, 3,
    201, 41, 3, 199, 40, 3, 197, 38, 2, 195, 36, 2, 192, 35, 2, 190, 33, 2, 187, 31, 1, 185, 30, 1, 182, 28, 1, 180, 27, 1, 177, 25, 1,
    174, 24, 1, 172, 22, 1, 169, 21, 1, 166, 20, 1, 163, 18, 1, 160, 17, 1, 157, 16, 1, 154, 14, 1, 151, 13, 1, 148, 12, 1, 145, 11, 1,
    142, 10, 1, 139, 9, 1, 135, 8, 1, 132, 7, 1, 129, 6, 2, 125, 5, 2, 122, 4, 2,
];

// ---------------------------------------------------------------------------------------------
// Images.

/// A decoded image, upright, with the focal length its EXIF gives.
#[derive(Clone, Debug)]
pub struct LoadedImage {
    pub width: usize,
    pub height: usize,
    /// RGB8, row-major.
    pub rgb: Vec<u8>,
    /// From the EXIF 35 mm equivalent focal length (`FocalLengthIn35mmFilm`), in pixels of the
    /// upright image, as the reference's `load_rgb` derives it.
    pub focal_px: Option<f32>,
}

/// What a photo's EXIF says: orientation (1-8) and the 35 mm equivalent focal length.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Exif {
    pub orientation: Option<u16>,
    pub focal_35mm: Option<f32>,
}

/// The EXIF of a JPEG (APP1 `Exif`) or PNG (`eXIf` chunk) file.
pub fn read_exif(bytes: &[u8]) -> Option<Exif> {
    let tiff = if bytes.starts_with(&[0xFF, 0xD8]) {
        let mut i = 2;
        let mut found = None;
        while i + 4 <= bytes.len() && bytes[i] == 0xFF {
            let marker = bytes[i + 1];
            if marker == 0xD9 || marker == 0xDA {
                break;
            }
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            let seg = bytes.get(i + 4..i + 2 + len)?;
            if marker == 0xE1 && seg.starts_with(b"Exif\0\0") {
                found = Some(&seg[6..]);
                break;
            }
            i += 2 + len;
        }
        found?
    } else if bytes.starts_with(b"\x89PNG") {
        let mut i = 8;
        let mut found = None;
        while i + 8 <= bytes.len() {
            let len = u32::from_be_bytes(bytes[i..i + 4].try_into().ok()?) as usize;
            let kind = &bytes[i + 4..i + 8];
            if kind == b"eXIf" {
                found = Some(bytes.get(i + 8..i + 8 + len)?);
                break;
            }
            if kind == b"IDAT" {
                break;
            }
            i += 12 + len;
        }
        found?
    } else {
        return None;
    };
    let le = match tiff.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b: [u8; 2] = tiff.get(o..o + 2)?.try_into().ok()?;
        Some(if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b: [u8; 4] = tiff.get(o..o + 4)?.try_into().ok()?;
        Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    };
    // Entries of an IFD: (tag, type, value field offset).
    let entries = |ifd: usize| -> Vec<(u16, u16, usize)> {
        let Some(n) = u16_at(ifd) else { return Vec::new() };
        (0..n as usize)
            .filter_map(|k| {
                let e = ifd + 2 + 12 * k;
                Some((u16_at(e)?, u16_at(e + 2)?, e + 8))
            })
            .collect()
    };
    let short_or_long = |ty: u16, at: usize| -> Option<u32> {
        match ty {
            3 => u16_at(at).map(u32::from),
            4 => u32_at(at),
            _ => None,
        }
    };
    let mut ex = Exif::default();
    let ifd0 = u32_at(4)? as usize;
    for (tag, ty, at) in entries(ifd0) {
        match tag {
            0x0112 => ex.orientation = short_or_long(ty, at).map(|v| v as u16),
            0x8769 => {
                if let Some(sub) = u32_at(at) {
                    for (t2, ty2, at2) in entries(sub as usize) {
                        if t2 == 0xA405 {
                            ex.focal_35mm = short_or_long(ty2, at2).map(|v| v as f32).filter(|&f| f > 0.0);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Some(ex)
}

/// `rgb` (`w x h`) turned upright per an EXIF orientation (PIL's `exif_transpose`).
pub fn apply_orientation(rgb: &[u8], w: usize, h: usize, orientation: u16) -> (Vec<u8>, usize, usize) {
    if !(2..=8).contains(&orientation) {
        return (rgb.to_vec(), w, h);
    }
    let (ow, oh) = if orientation >= 5 { (h, w) } else { (w, h) };
    let mut out = vec![0u8; rgb.len()];
    for y in 0..oh {
        for x in 0..ow {
            // The source pixel of output (x, y).
            let (sx, sy) = match orientation {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                _ => (w - 1 - y, x),
            };
            out[(y * ow + x) * 3..(y * ow + x) * 3 + 3].copy_from_slice(&rgb[(sy * w + sx) * 3..(sy * w + sx) * 3 + 3]);
        }
    }
    (out, ow, oh)
}

/// The focal length in pixels of a 35 mm equivalent focal length (`fpx_from_f35`).
pub fn fpx_from_f35(w: usize, h: usize, f_mm: f32) -> f32 {
    f_mm * ((w * w + h * h) as f32).sqrt() / (36f32 * 36.0 + 24.0 * 24.0).sqrt()
}

/// Decodes an encoded image (PNG, JPEG, WebP, GIF, BMP; the `media` feature), upright per its
/// EXIF orientation, with the focal length of its EXIF.
pub fn decode_image(bytes: &[u8]) -> Result<LoadedImage> {
    let img = crate::mm::decode_images(&[bytes.to_vec()])?.pop().context("no image")?;
    let ex = read_exif(bytes).unwrap_or_default();
    let (rgb, width, height) = apply_orientation(&img.data, img.width, img.height, ex.orientation.unwrap_or(1));
    Ok(LoadedImage {
        focal_px: ex.focal_35mm.map(|f| fpx_from_f35(width, height, f)),
        width,
        height,
        rgb,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    type T = Vec<(String, Vec<usize>, Vec<f32>)>;

    fn write_st(path: &Path, t: &T) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, vals) in t {
            assert_eq!(vals.len(), shape.iter().product::<usize>(), "{name}");
            let start = data.len();
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.clone(),
                serde_json::json!({ "dtype": "F32", "shape": shape, "data_offsets": [start, data.len()] }),
            );
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&json);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    struct Gen(u64);
    impl Gen {
        fn put(&mut self, t: &mut T, name: &str, shape: &[usize], scale: f32) {
            let n: usize = shape.iter().product();
            let v = (0..n).map(|_| lcg(&mut self.0) * scale).collect();
            t.push((name.to_string(), shape.to_vec(), v));
        }
        fn ln(&mut self, t: &mut T, p: &str, n: usize) {
            t.push((format!("{p}.weight"), vec![n], vec![1.0; n]));
            t.push((format!("{p}.bias"), vec![n], vec![0.0; n]));
        }
        fn vit(&mut self, t: &mut T, p: &str, dim: usize, depth: usize) {
            self.put(t, &format!("{p}.patch_embed.proj.weight"), &[dim, 3, 16, 16], 0.05);
            self.put(t, &format!("{p}.patch_embed.proj.bias"), &[dim], 0.1);
            self.put(t, &format!("{p}.cls_token"), &[1, 1, dim], 0.5);
            self.put(t, &format!("{p}.reg_token"), &[1, 4, dim], 0.5);
            for i in 0..depth {
                let b = format!("{p}.blocks.{i}");
                self.ln(t, &format!("{b}.norm1"), dim);
                self.ln(t, &format!("{b}.norm2"), dim);
                self.put(t, &format!("{b}.attn.qkv.weight"), &[3 * dim, dim], 0.2);
                self.put(t, &format!("{b}.attn.proj.weight"), &[dim, dim], 0.2);
                self.put(t, &format!("{b}.attn.proj.bias"), &[dim], 0.1);
                self.put(t, &format!("{b}.mlp.fc1.weight"), &[4 * dim, dim], 0.2);
                self.put(t, &format!("{b}.mlp.fc1.bias"), &[4 * dim], 0.1);
                self.put(t, &format!("{b}.mlp.fc2.weight"), &[dim, 4 * dim], 0.1);
                self.put(t, &format!("{b}.mlp.fc2.bias"), &[dim], 0.1);
                self.put(t, &format!("{b}.gamma_1"), &[dim], 0.5);
                self.put(t, &format!("{b}.gamma_2"), &[dim], 0.5);
            }
        }
        fn conv(&mut self, t: &mut T, p: &str, o: usize, i: usize, k: usize, bias: bool) {
            self.put(t, &format!("{p}.weight"), &[o, i, k, k], 0.3);
            if bias {
                self.put(t, &format!("{p}.bias"), &[o], 0.1);
            }
        }
    }

    /// A tiny random Depth Pro 2 (a 64-wide 4-block ViT) and focal estimator in a new folder.
    pub(crate) fn tiny(tag: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("oxinfer-dp-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (dim, dims, dd) = (64usize, [8usize, 8, 16, 16], 8usize);
        let cfg = serde_json::json!({
            "backbone_preset": "tiny_dinov3", "reassembled_blocks": [0, 1, 2, 3], "cross_frame_blocks": [2, 3],
            "dims_reassemble": dims, "dim_decoder": dd, "fov": false,
        });
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let mut g = Gen(11);
        let mut t = T::new();
        g.vit(&mut t, "encoder", dim, 4);
        g.conv(&mut t, "encoder.reassemble.0.0", dims[0], dim, 1, true);
        g.put(&mut t, "encoder.reassemble.0.1.weight", &[dims[0], dims[0], 4, 4], 0.3);
        g.put(&mut t, "encoder.reassemble.0.1.bias", &[dims[0]], 0.1);
        g.conv(&mut t, "encoder.reassemble.1.0", dims[1], dim, 1, true);
        g.put(&mut t, "encoder.reassemble.1.1.weight", &[dims[1], dims[1], 2, 2], 0.3);
        g.put(&mut t, "encoder.reassemble.1.1.bias", &[dims[1]], 0.1);
        g.conv(&mut t, "encoder.reassemble.2.0", dims[2], dim, 1, true);
        g.conv(&mut t, "encoder.reassemble.3.0", dims[3], dim, 1, true);
        g.conv(&mut t, "encoder.reassemble.3.1", dims[3], dims[3], 3, true);
        for (i, &d) in dims.iter().enumerate() {
            g.conv(&mut t, &format!("decoder.convs.{i}"), dd, d, 3, false);
            let p = format!("decoder.fusions.{i}");
            let rbs: &[&str] = if i < 3 { &["resnet1", "resnet2"] } else { &["resnet2"] };
            for r in rbs {
                g.conv(&mut t, &format!("{p}.{r}.residual.1"), dd, dd, 3, true);
                g.conv(&mut t, &format!("{p}.{r}.residual.3"), dd, dd, 3, true);
            }
            g.put(&mut t, &format!("{p}.deconv.weight"), &[dd, dd, 2, 2], 0.3);
            g.conv(&mut t, &format!("{p}.out_conv"), dd, dd, 1, true);
        }
        g.conv(&mut t, "head.0", dd / 2, dd, 3, true);
        g.put(&mut t, "head.1.weight", &[dd / 2, dd / 2, 2, 2], 0.3);
        g.put(&mut t, "head.1.bias", &[dd / 2], 0.1);
        g.conv(&mut t, "head.2", 32, dd / 2, 3, true);
        g.conv(&mut t, "head.4", 1, 32, 1, true);
        // A positive output somewhere (ReLU heads).
        t.iter_mut().find(|e| e.0 == "head.4.bias").unwrap().2[0] = 0.5;
        write_st(&dir.join("model.safetensors"), &t);
        // The focal estimator: a 12-block trunk and its head (width 16, 4 heads).
        let fdir = dir.join("focal");
        std::fs::create_dir_all(&fdir).unwrap();
        let mut t = T::new();
        g.vit(&mut t, "trunk", dim, 12);
        let hd = 16usize;
        for i in 0..4 {
            g.ln(&mut t, &format!("head.tap_norms.{i}"), dim);
        }
        g.put(&mut t, "head.proj.weight", &[hd, 4 * dim], 0.1);
        g.put(&mut t, "head.proj.bias", &[hd], 0.1);
        g.put(&mut t, "head.query", &[1, 1, hd], 0.5);
        g.put(&mut t, "head.prefix_embed", &[1, 5, hd], 0.5);
        for i in 0..2 {
            let b = format!("head.blocks.{i}");
            g.ln(&mut t, &format!("{b}.norm1"), hd);
            g.ln(&mut t, &format!("{b}.norm2"), hd);
            g.put(&mut t, &format!("{b}.attn.qkv.weight"), &[3 * hd, hd], 0.3);
            g.put(&mut t, &format!("{b}.attn.qkv.bias"), &[3 * hd], 0.1);
            g.put(&mut t, &format!("{b}.attn.proj.weight"), &[hd, hd], 0.3);
            g.put(&mut t, &format!("{b}.attn.proj.bias"), &[hd], 0.1);
            g.put(&mut t, &format!("{b}.mlp.fc1.weight"), &[4 * hd, hd], 0.3);
            g.put(&mut t, &format!("{b}.mlp.fc1.bias"), &[4 * hd], 0.1);
            g.put(&mut t, &format!("{b}.mlp.fc2.weight"), &[hd, 4 * hd], 0.3);
            g.put(&mut t, &format!("{b}.mlp.fc2.bias"), &[hd], 0.1);
        }
        g.ln(&mut t, "head.norm", hd);
        g.put(&mut t, "head.out.weight", &[2, hd], 0.3);
        g.put(&mut t, "head.out.bias", &[2], 0.1);
        write_st(&fdir.join("model.safetensors"), &t);
        std::fs::write(fdir.join("config.json"), "{}").unwrap();
        (dir, fdir)
    }

    pub(crate) fn image(w: usize, h: usize) -> Vec<u8> {
        (0..w * h * 3).map(|i| ((i * 37 + i / 7) % 251) as u8).collect()
    }

    #[test]
    fn tiny_model_runs() {
        let (dir, fdir) = tiny("run");
        let o = DepthOptions {
            focal: Some(fdir.to_string_lossy().into_owned()),
            ..Default::default()
        };
        let m = DepthPro::load(&dir.to_string_lossy(), &o).unwrap();
        assert!(is_depth_pro(&dir));
        // Not a multiple of 32: padded, then cropped back.
        let (w, h) = (50, 40);
        let d = m.infer_rgb(&image(w, h), w, h, None).unwrap();
        assert_eq!((d.width, d.height, d.inverse_depth.len()), (w, h, w * h));
        assert!(d.inverse_depth.iter().all(|v| v.is_finite() && *v >= 0.0));
        let f = d.focal_px.unwrap();
        assert!(d.focal_estimated && f.is_finite() && f > 0.0);
        let fov = d.fov_deg.unwrap();
        assert!(fov.iter().all(|a| (1.0..=179.0).contains(a)));
        assert_eq!(focal_from_fov(fov, h, w), f);
        // A given focal length is used as is, and depth follows from it.
        let g = m.infer_rgb(&image(w, h), w, h, Some(100.0)).unwrap();
        assert_eq!(g.inverse_depth, d.inverse_depth);
        assert_eq!((g.focal_px, g.focal_estimated), (Some(100.0), false));
        let depth = g.depth().unwrap();
        for (z, i) in depth.iter().zip(&g.inverse_depth) {
            assert!((z * w as f32 * i - 100.0).abs() < 1e-3 || *i == 0.0);
        }
        let (px, k) = g.gray16();
        assert_eq!(px.len(), w * h);
        assert!(k > 0.0);
        assert_eq!(g.colorize().len(), w * h * 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn resize_matches_torch_antialias_weights() {
        // Downscaling by 2 with antialias: a triangle of support 2, normalized (edges clipped).
        let x: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let y = resize_bilinear_aa(&x, 1, 1, 8, 1, 4);
        // torch: F.interpolate(arange(8).view(1,1,1,8), size=(1,4), mode="bilinear", antialias=True)
        let want = [0.714_285_7f32, 2.5, 4.5, 6.285_714_6];
        for (a, b) in y.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{y:?}");
        }
        // Same size: identity.
        assert_eq!(resize_bilinear_aa(&x, 1, 2, 4, 2, 4), x);
    }

    #[test]
    fn exif_orientation_and_focal() {
        // A PNG eXIf chunk: little-endian TIFF, orientation 6, an Exif IFD with 28 mm.
        let mut tiff = b"II\x2a\x00\x08\x00\x00\x00".to_vec();
        tiff.extend_from_slice(&2u16.to_le_bytes());
        tiff.extend_from_slice(&[0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0]);
        tiff.extend_from_slice(&[0x69, 0x87, 4, 0, 1, 0, 0, 0, 38, 0, 0, 0]);
        tiff.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(tiff.len(), 38);
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&[0x05, 0xA4, 3, 0, 1, 0, 0, 0, 28, 0, 0, 0]);
        tiff.extend_from_slice(&0u32.to_le_bytes());
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&(tiff.len() as u32).to_be_bytes());
        png.extend_from_slice(b"eXIf");
        png.extend_from_slice(&tiff);
        png.extend_from_slice(&[0; 4]);
        let ex = read_exif(&png).unwrap();
        assert_eq!(
            ex,
            Exif {
                orientation: Some(6),
                focal_35mm: Some(28.0)
            }
        );
        // Orientation 6 (rotate 90 clockwise to view): pixel (0, 0) goes to the top right.
        let rgb: Vec<u8> = (0..6u8).flat_map(|i| [i, 0, 0]).collect(); // 3 wide, 2 high
        let (o, w, h) = apply_orientation(&rgb, 3, 2, 6);
        assert_eq!((w, h), (2, 3));
        let px: Vec<u8> = o.chunks(3).map(|c| c[0]).collect();
        assert_eq!(px, [3, 0, 4, 1, 5, 2]);
        let (o8, ..) = apply_orientation(&rgb, 3, 2, 8);
        assert_eq!(o8.chunks(3).map(|c| c[0]).collect::<Vec<_>>(), [2, 5, 1, 4, 0, 3]);
        assert!((fpx_from_f35(1920, 1440, 28.0) - 28.0 * 2400.0 / 43.266_615).abs() < 1e-2);
    }
}
