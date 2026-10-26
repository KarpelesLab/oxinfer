//! SNAC (Multi-Scale Neural Audio Codec, `hubertsiuzdak/snac_24khz`, `snac_32khz`, `snac_44khz`):
//! a waveform to codes of three or four codebooks at different rates (12, 23 and 47 Hz for the
//! 24 kHz model) and back. Orpheus TTS writes its audio tokens as 24 kHz codes (see
//! [`crate::tts`]) and reads reference clips (voice cloning) through the encoder.
//!
//! The decoder, as the `snac` Python package builds it (weight-normalized convolutions, folded
//! at load time):
//! - each codebook's entries are projected to the latent (`out_proj`, a 1x1 conv; precomputed
//!   per entry here) and repeated to the finest rate (`vq_strides`), the codebooks summed;
//! - a depthwise k7 conv and a 1x1 conv to `decoder_dim` channels;
//! - the 32 / 44 kHz models only: local windowed attention ([`LocalAttention`]);
//! - per decoder rate: Snake, a transposed conv (kernel 2 * stride) halving the channels, a
//!   noise block (`x + randn(t) * W x`) and three residual units (Snake, depthwise k7 conv with
//!   dilation 1 / 3 / 9, Snake, 1x1 conv);
//! - Snake, a k7 conv to one channel, tanh.
//!
//! The encoder ([`Snac::encode`]) mirrors it: a k7 conv from one channel to `encoder_dim`, per
//! encoder rate three residual units and a strided conv (kernel 2 * stride) doubling the
//! channels, local attention (32 / 44 kHz), a depthwise k7 conv; then residual vector
//! quantization: per codebook an average pool by its stride, a 1x1 projection to the 8-dim code
//! space, the nearest L2-normalized codebook entry (cosine), and the residual minus that entry's
//! `out_proj`. The input is right-padded with zeros to a multiple of the hop times
//! `lcm(vq_strides[0], attn_window_size)`, as the reference's `preprocess` does.
//!
//! Activations are `[time, channels]` row-major, so 1x1 convs and transposed convs are matmuls
//! through the shared driver (`model::linear`: SIMD kernels, GPU prefill backend for bf16).
//! Weights are kept in f32 (exact) or bf16 ([`SnacOptions::f32_weights`]).
//!
//! The noise of the noise blocks is drawn per (seed, layer, absolute time step) rather than from
//! a running generator, so decoding overlapping windows (streaming) gives every sample the same
//! noise as a decode of the whole clip.

use crate::ggml::{GgmlType, GgufMatrix};
use crate::kernels::for_chunks;
use crate::model::{Lin, linear};
use crate::quant::PageVec;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d_true() -> bool {
    true
}

/// `f(row index, row)` over the `c`-wide rows of `y`, in parallel blocks of rows (one pool item
/// per row costs more than the work on short rows).
fn for_rows(y: &mut [f32], c: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = (16384 / c.max(1)).max(1);
    let par = y.len() > rows * c;
    for_chunks(y, rows * c, par, |b, block| {
        for (j, row) in block.chunks_mut(c).enumerate() {
            f(b * rows + j, row);
        }
    });
}

/// `config.json` of a SNAC checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct SnacConfig {
    pub sampling_rate: u32,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    #[serde(default)]
    pub latent_dim: Option<usize>,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    #[serde(default)]
    pub attn_window_size: Option<usize>,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vq_strides: Vec<usize>,
    #[serde(default = "d_true")]
    pub noise: bool,
    #[serde(default = "d_true")]
    pub depthwise: bool,
}

impl SnacConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: SnacConfig = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse SNAC config")?;
        if cfg.attn_window_size == Some(0) {
            bail!("SNAC config with attn_window_size 0");
        }
        if !cfg.depthwise {
            bail!("SNAC models without depthwise convolutions are not supported");
        }
        if cfg.vq_strides.is_empty() || cfg.decoder_rates.is_empty() {
            bail!("SNAC config without codebooks or decoder rates");
        }
        Ok(cfg)
    }
    pub fn latent_dim(&self) -> usize {
        self.latent_dim.unwrap_or(self.encoder_dim << self.encoder_rates.len())
    }
    /// Input samples per latent step (the product of the encoder rates: 512 at 24 kHz).
    pub fn hop(&self) -> usize {
        self.encoder_rates.iter().product()
    }
    /// What [`Snac::encode`] pads its input to a multiple of: the hop times
    /// `lcm(vq_strides[0], attn_window_size)`.
    pub fn pad_multiple(&self) -> usize {
        let a = self.vq_strides[0];
        let b = self.attn_window_size.unwrap_or(1);
        let gcd = |mut x: usize, mut y: usize| {
            while y != 0 {
                (x, y) = (y, x % y);
            }
            x
        };
        self.hop() * (a / gcd(a, b) * b)
    }
}

/// Whether `dir` holds a SNAC checkpoint (`config.json` with SNAC's fields).
pub fn is_snac(dir: &Path) -> bool {
    crate::fsx::read(&dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v.get("vq_strides").is_some() && v.get("decoder_rates").is_some())
}

/// How to load a SNAC decoder.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct SnacOptions {
    /// Keep the matmul weights in f32 (exact; default) rather than bf16 (faster, GPU-capable).
    pub f32_weights: bool,
}

impl Default for SnacOptions {
    fn default() -> Self {
        Self { f32_weights: true }
    }
}

/// The noise blocks' noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Noise {
    /// No noise (`x` unchanged), e.g. to compare against a reference with the noise disabled.
    Off,
    /// Gaussian noise keyed by this seed, the layer and the absolute time step.
    Seeded(u64),
}

/// A matrix of a 1x1 / transposed conv, with its bias.
struct Mat {
    w: Lin<'static>,
    b: Option<Vec<f32>>,
    n_out: usize,
    n_in: usize,
}

impl Mat {
    fn new(w: &[f32], n_out: usize, n_in: usize, b: Option<Vec<f32>>, f32w: bool) -> Result<Self> {
        debug_assert_eq!(w.len(), n_out * n_in);
        let w = if f32w {
            let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
            Lin::gguf(GgufMatrix::from_bytes(GgmlType::F32, &bytes, n_out, n_in)?)
        } else {
            let mut v = PageVec::<u16>::zeroed(w.len());
            for (o, &x) in v.iter_mut().zip(w) {
                *o = half::bf16::from_f32(x).to_bits();
            }
            Lin::bf16_owned(v)
        };
        Ok(Self { w, b, n_out, n_in })
    }

    /// `x` `[t, n_in]` -> `[t, n_out]` (bias added).
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * self.n_out];
        linear(&self.w, self.n_out, self.n_in, x, t, &mut y);
        if let Some(b) = &self.b {
            for_rows(&mut y, self.n_out, |_, r| {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            });
        }
        y
    }
}

/// A depthwise k-tap conv (`groups == channels`, same padding), taps stored `[k][c]`.
struct Depthwise {
    w: Vec<f32>,
    b: Vec<f32>,
    c: usize,
    k: usize,
    dilation: usize,
}

impl Depthwise {
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (c, k, d) = (self.c, self.k, self.dilation);
        let half = (k - 1) / 2 * d;
        let mut y = vec![0f32; t * c];
        for_rows(&mut y, c, |ti, out| {
            out.copy_from_slice(&self.b);
            for j in 0..k {
                let src = ti as isize + (j * d) as isize - half as isize;
                if src < 0 || src >= t as isize {
                    continue;
                }
                let xs = &x[src as usize * c..(src as usize + 1) * c];
                let ws = &self.w[j * c..(j + 1) * c];
                for ((o, &xv), &wv) in out.iter_mut().zip(xs).zip(ws) {
                    *o += xv * wv;
                }
            }
        });
        y
    }
}

/// Snake activation `x + sin(alpha x)^2 / (alpha + 1e-9)`, per channel.
struct Snake {
    alpha: Vec<f32>,
    inv: Vec<f32>,
}

impl Snake {
    fn new(alpha: Vec<f32>) -> Self {
        let inv = alpha.iter().map(|&a| 1.0 / (a + 1e-9)).collect();
        Self { alpha, inv }
    }
    /// `y = snake(x)`.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let c = self.alpha.len();
        let mut y = x.to_vec();
        let par = x.len() / c > 1;
        // Rows in groups, so small layers do not pay a pool round per row.
        let rows = (8192 / c).max(1);
        for_chunks(&mut y, rows * c, par, |_, block| {
            for row in block.chunks_mut(c) {
                for ((v, &a), &inv) in row.iter_mut().zip(&self.alpha).zip(&self.inv) {
                    let s = sin_f32(a * *v);
                    *v += inv * s * s;
                }
            }
        });
        y
    }
}

/// `sin(x)` to within ~1 ulp of libm over the range Snake sees, branch-free so it vectorizes:
/// reduction by pi (three-part Cody-Waite) and an odd degree-11 polynomial on [-pi/2, pi/2].
#[inline(always)]
fn sin_f32(x: f32) -> f32 {
    const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
    const P1: f32 = 3.140625;
    const P2: f32 = 9.670_257_6e-4;
    const P3: f32 = 6.277_115e-7;
    let k = (x * INV_PI).round();
    let r = ((x - k * P1) - k * P2) - k * P3;
    let r2 = r * r;
    let p = -2.505_210_8e-8f32;
    let p = p * r2 + 2.755_731_4e-6;
    let p = p * r2 - 1.984_127e-4;
    let p = p * r2 + 8.333_333e-3;
    let p = p * r2 - 1.666_666_7e-1;
    let s = r + r * r2 * p;
    // (-1)^k
    let odd = (k as i32) & 1;
    f32::from_bits(s.to_bits() ^ ((odd as u32) << 31))
}

struct ResUnit {
    snake1: Snake,
    conv: Depthwise,
    snake2: Snake,
    pw: Mat,
}

impl ResUnit {
    fn forward(&self, x: &mut [f32], t: usize) {
        let h = self.snake1.apply(x);
        let h = self.conv.forward(&h, t);
        let h = self.snake2.apply(&h);
        let h = self.pw.forward(&h, t);
        crate::kernels::add_inplace(x, &h);
    }
}

struct Block {
    snake: Snake,
    /// The transposed conv as a matmul: `[kernel * c_out, c_in]`, rows ordered (tap, out channel).
    up: Mat,
    up_bias: Vec<f32>,
    stride: usize,
    kernel: usize,
    pad: usize,
    out_pad: usize,
    c_out: usize,
    noise: Option<Mat>,
    res: Vec<ResUnit>,
}

impl Block {
    /// `x` `[t, c_in]` -> `[t_out, c_out]`; `t0` is the absolute index of `x`'s first step at
    /// this block's input rate (for the noise).
    fn forward(&self, x: &[f32], t: usize, t0: usize, layer: usize, noise: Noise) -> (Vec<f32>, usize) {
        let h = self.snake.apply(x);
        let taps = self.up.forward(&h, t); // [t, kernel * c_out]
        let (s, k, p, co) = (self.stride, self.kernel, self.pad, self.c_out);
        let t_out = (t - 1) * s + k + self.out_pad - 2 * p;
        let mut y = vec![0f32; t_out * co];
        // Gather: output step o takes tap (o + p - i * s) of every input step i it overlaps.
        for_rows(&mut y, co, |o, out| {
            out.copy_from_slice(&self.up_bias);
            let q = o + p;
            let i_hi = (q / s).min(t - 1);
            let i_lo = (q + 1).saturating_sub(k).div_ceil(s);
            for i in i_lo..=i_hi {
                let tap = q - i * s;
                if tap >= k {
                    continue;
                }
                let src = &taps[i * k * co + tap * co..i * k * co + (tap + 1) * co];
                for (v, &a) in out.iter_mut().zip(src) {
                    *v += a;
                }
            }
        });
        if let (Some(nm), Noise::Seeded(seed)) = (&self.noise, noise) {
            let lin = nm.forward(&y, t_out);
            let base = t0 * s;
            for_rows(&mut y, co, |o, out| {
                let n = gaussian(seed, layer as u64, (base + o) as u64);
                for (v, &l) in out.iter_mut().zip(&lin[o * co..(o + 1) * co]) {
                    *v += n * l;
                }
            });
        }
        for r in &self.res {
            r.forward(&mut y, t_out);
        }
        (y, t_out)
    }
}

/// A standard normal draw keyed by `(seed, layer, step)` (SplitMix64 hashing, Box-Muller).
fn gaussian(seed: u64, layer: u64, step: u64) -> f32 {
    let mix = |mut z: u64| {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let h = mix(seed
        .wrapping_add(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(mix(layer.wrapping_mul(0x1000_0000_01b3) ^ step)));
    let u1 = ((h >> 40) as f64 + 0.5) / (1u64 << 24) as f64;
    let u2 = ((h & 0xff_ffff) as f64) / (1u64 << 24) as f64;
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Local multi-head self-attention (`snac.attention.LocalMHA`), residual: LayerNorm, attention
/// within non-overlapping windows of `window` steps (64-dim heads, rotary positions restarting at
/// 0 in every window, no mask), output projection, plus the input. The reference needs the length
/// to be a multiple of the window; here a shorter last window attends within itself.
struct LocalAttention {
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    qkv: Mat,
    out: Mat,
    /// Rotary inverse frequencies (`rel_pos.inv_freq`, half the head size).
    inv_freq: Vec<f32>,
    window: usize,
    c: usize,
}

const HEAD_DIM: usize = 64;

impl LocalAttention {
    fn load(files: &ModelFiles, p: &str, c: usize, window: usize, f32w: bool) -> Result<Self> {
        let get = |n: &str, len: usize| -> Result<Vec<f32>> {
            let v = files.f32(&format!("{p}.{n}"))?;
            if v.len() != len {
                bail!("{p}.{n}: {} values, expected {len}", v.len());
            }
            Ok(v)
        };
        if !c.is_multiple_of(HEAD_DIM) {
            bail!("{p}: {c} channels are not a multiple of the {HEAD_DIM}-dim heads");
        }
        let inv_freq = if files.has(&format!("{p}.rel_pos.inv_freq")) {
            get("rel_pos.inv_freq", HEAD_DIM / 2)?
        } else {
            (0..HEAD_DIM / 2)
                .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / HEAD_DIM as f32))
                .collect()
        };
        Ok(Self {
            norm_w: get("norm.weight", c)?,
            norm_b: get("norm.bias", c)?,
            qkv: Mat::new(&get("to_qkv.weight", 3 * c * c)?, 3 * c, c, None, f32w)?,
            out: Mat::new(&get("to_out.weight", c * c)?, c, c, None, f32w)?,
            inv_freq,
            window,
            c,
        })
    }

    /// `x` `[t, c]` in place.
    fn forward(&self, x: &mut [f32], t: usize) {
        let c = self.c;
        let mut h = x.to_vec();
        for_rows(&mut h, c, |_, r| {
            let mean = r.iter().map(|&v| v as f64).sum::<f64>() / c as f64;
            let var = r.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / c as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for ((v, &w), &b) in r.iter_mut().zip(&self.norm_w).zip(&self.norm_b) {
                *v = ((*v as f64 - mean) * inv) as f32 * w + b;
            }
        });
        let mut qkv = self.qkv.forward(&h, t); // [t, 3c]: q heads, k heads, v heads
        let w = self.window;
        let heads = c / HEAD_DIM;
        let half = HEAD_DIM / 2;
        // Rotary embedding of q and k by the position within the window (rotate_half layout).
        for_rows(&mut qkv, 3 * c, |ti, row| {
            let pos = (ti % w) as f32;
            for hd in 0..2 * heads {
                let v = &mut row[hd * HEAD_DIM..(hd + 1) * HEAD_DIM];
                for i in 0..half {
                    let (s, co) = (pos * self.inv_freq[i]).sin_cos();
                    let (a, b) = (v[i], v[i + half]);
                    v[i] = a * co - b * s;
                    v[i + half] = b * co + a * s;
                }
            }
        });
        // Per (window, head): `[w, 64]` outputs.
        let windows = t.div_ceil(w);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut blocks = vec![0f32; windows * heads * w * HEAD_DIM];
        for_chunks(&mut blocks, w * HEAD_DIM, true, |job, out| {
            let (wi, hd) = (job / heads, job % heads);
            let (t0, t1) = (wi * w, ((wi + 1) * w).min(t));
            let col = |j: usize, part: usize| &qkv[j * 3 * c + part * c + hd * HEAD_DIM..j * 3 * c + part * c + (hd + 1) * HEAD_DIM];
            let mut s = vec![0f32; t1 - t0];
            for (i, o) in (t0..t1).zip(out.chunks_mut(HEAD_DIM)) {
                let q = col(i, 0);
                let mut mx = f32::NEG_INFINITY;
                for (j, sj) in (t0..t1).zip(s.iter_mut()) {
                    *sj = crate::kernels::dot(q, col(j, 1)) * scale;
                    mx = mx.max(*sj);
                }
                let mut sum = 0f32;
                for v in s.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                for (j, &pj) in (t0..t1).zip(&s) {
                    for (a, &b) in o.iter_mut().zip(col(j, 2)) {
                        *a += pj * b;
                    }
                }
                let inv = 1.0 / sum;
                for a in o.iter_mut() {
                    *a *= inv;
                }
            }
        });
        let mut att = vec![0f32; t * c];
        for_rows(&mut att, c, |ti, row| {
            let (wi, i) = (ti / w, ti % w);
            for hd in 0..heads {
                let b = ((wi * heads + hd) * w + i) * HEAD_DIM;
                row[hd * HEAD_DIM..(hd + 1) * HEAD_DIM].copy_from_slice(&blocks[b..b + HEAD_DIM]);
            }
        });
        let y = self.out.forward(&att, t);
        crate::kernels::add_inplace(x, &y);
    }
}

/// A strided conv (`kernel = 2 * stride`, padding `ceil(stride / 2)`) as a matmul over gathered
/// input windows: weights `[c_out, kernel * c_in]`, columns ordered (tap, in channel).
struct StridedConv {
    w: Mat,
    stride: usize,
    kernel: usize,
    pad: usize,
    c_in: usize,
}

impl StridedConv {
    fn load(files: &ModelFiles, p: &str, stride: usize) -> Result<Self> {
        let (w, shape) = wn(files, p)?;
        let [co, ci, k] = shape[..] else {
            bail!("{p}: expected a conv kernel [out, in, k], got {shape:?}")
        };
        if k != 2 * stride {
            bail!("{p}: kernel {k} for stride {stride}");
        }
        let mut r = vec![0f32; w.len()];
        for o in 0..co {
            for i in 0..ci {
                for j in 0..k {
                    r[o * k * ci + j * ci + i] = w[(o * ci + i) * k + j];
                }
            }
        }
        Ok(Self {
            w: Mat::new(&r, co, k * ci, Some(bias(files, p, co)?), true)?,
            stride,
            kernel: k,
            pad: stride.div_ceil(2),
            c_in: ci,
        })
    }

    /// `x` `[t, c_in]` -> `[t_out, c_out]`.
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        let (s, k, p, ci) = (self.stride, self.kernel, self.pad, self.c_in);
        let t_out = (t + 2 * p - k) / s + 1;
        let mut cols = vec![0f32; t_out * k * ci];
        for_rows(&mut cols, k * ci, |o, row| {
            for j in 0..k {
                let src = (o * s + j) as isize - p as isize;
                if src >= 0 && (src as usize) < t {
                    row[j * ci..(j + 1) * ci].copy_from_slice(&x[src as usize * ci..(src as usize + 1) * ci]);
                }
            }
        });
        (self.w.forward(&cols, t_out), t_out)
    }
}

/// One encoder rate: three residual units, Snake, the strided conv.
struct EncBlock {
    res: Vec<ResUnit>,
    snake: Snake,
    down: StridedConv,
}

/// The encoder half of a SNAC checkpoint and the quantizers' input side (f32 weights).
struct Encoder {
    /// First conv, 1 -> `encoder_dim` channels, taps `[k][c]`.
    conv_in: Vec<f32>,
    conv_in_b: Vec<f32>,
    conv_in_k: usize,
    blocks: Vec<EncBlock>,
    attn: Option<LocalAttention>,
    conv_out: Depthwise,
    /// Per codebook: `in_proj` (latent -> code dim) and the L2-normalized codebook.
    in_proj: Vec<Mat>,
    codebooks: Vec<Vec<f32>>,
}

impl Encoder {
    fn load(files: &ModelFiles, cfg: &SnacConfig) -> Result<Self> {
        let (w, shape) = wn(files, "encoder.block.0")?;
        let [c0, 1, k] = shape[..] else {
            bail!("encoder input conv: shape {shape:?}")
        };
        let mut conv_in = vec![0f32; k * c0];
        for ch in 0..c0 {
            for j in 0..k {
                conv_in[j * c0 + ch] = w[ch * k + j];
            }
        }
        let conv_in_b = bias(files, "encoder.block.0", c0)?;
        let mut c = c0;
        let mut blocks = Vec::new();
        for (bi, &s) in cfg.encoder_rates.iter().enumerate() {
            let p = format!("encoder.block.{}.block", bi + 1);
            let mut res = Vec::new();
            for (j, dil) in [1, 3, 9].into_iter().enumerate() {
                let rp = format!("{p}.{j}.block");
                res.push(ResUnit {
                    snake1: snake(files, &format!("{rp}.0"), c)?,
                    conv: depthwise(files, &format!("{rp}.1"), dil)?,
                    snake2: snake(files, &format!("{rp}.2"), c)?,
                    pw: pointwise(files, &format!("{rp}.3"), true, true)?,
                });
            }
            let snake = snake(files, &format!("{p}.3"), c)?;
            let down = StridedConv::load(files, &format!("{p}.4"), s)?;
            if down.c_in != c {
                bail!("{p}.4: {} input channels, expected {c}", down.c_in);
            }
            c = down.w.n_out;
            blocks.push(EncBlock { res, snake, down });
        }
        if c != cfg.latent_dim() {
            bail!("encoder: {c} output channels, latent_dim {}", cfg.latent_dim());
        }
        let mut next = cfg.encoder_rates.len() + 1;
        let attn = match cfg.attn_window_size {
            Some(w) => {
                next += 1;
                Some(LocalAttention::load(files, &format!("encoder.block.{}", next - 1), c, w, true)?)
            }
            None => None,
        };
        let conv_out = depthwise(files, &format!("encoder.block.{next}"), 1)?;
        let mut in_proj = Vec::new();
        let mut codebooks = Vec::new();
        for i in 0..cfg.vq_strides.len() {
            let p = format!("quantizer.quantizers.{i}");
            in_proj.push(pointwise(files, &format!("{p}.in_proj"), true, true)?);
            let mut cb = files.f32(&format!("{p}.codebook.weight"))?;
            for e in cb.chunks_mut(cfg.codebook_dim) {
                let n = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                for v in e {
                    *v /= n;
                }
            }
            codebooks.push(cb);
        }
        Ok(Self {
            conv_in,
            conv_in_b,
            conv_in_k: k,
            blocks,
            attn,
            conv_out,
            in_proj,
            codebooks,
        })
    }

    /// The latent `[t, latent]` of `audio` (its length a multiple of the hop).
    fn latent(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (c, k) = (self.conv_in_b.len(), self.conv_in_k);
        let half = (k - 1) / 2;
        let mut t = audio.len();
        let mut x = vec![0f32; t * c];
        for_rows(&mut x, c, |ti, out| {
            out.copy_from_slice(&self.conv_in_b);
            for j in 0..k {
                let src = ti as isize + j as isize - half as isize;
                if src < 0 || src >= t as isize {
                    continue;
                }
                let a = audio[src as usize];
                for (o, &w) in out.iter_mut().zip(&self.conv_in[j * c..(j + 1) * c]) {
                    *o += a * w;
                }
            }
        });
        for b in &self.blocks {
            for r in &b.res {
                r.forward(&mut x, t);
            }
            let h = b.snake.apply(&x);
            (x, t) = b.down.forward(&h, t);
        }
        if let Some(a) = &self.attn {
            a.forward(&mut x, t);
        }
        (self.conv_out.forward(&x, t), t)
    }
}

/// A loaded SNAC codec (decoder, and the encoder when the checkpoint has it).
pub struct Snac {
    pub cfg: SnacConfig,
    /// Per codebook: `[codebook_size, latent]`, each entry through `out_proj` (bias included).
    tables: Vec<Vec<f32>>,
    conv_in: Depthwise,
    proj_in: Mat,
    attn: Option<LocalAttention>,
    blocks: Vec<Block>,
    snake_out: Snake,
    /// Final k7 conv to one channel, taps `[k][c]`.
    conv_out: Vec<f32>,
    conv_out_b: f32,
    conv_out_k: usize,
    encoder: Option<Encoder>,
}

/// Weight-norm parametrization: `g * v / ||v||`, the norm over every dim but the first.
fn wn(files: &ModelFiles, p: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let (gn, vn) = if files.has(&format!("{p}.parametrizations.weight.original0")) {
        (
            format!("{p}.parametrizations.weight.original0"),
            format!("{p}.parametrizations.weight.original1"),
        )
    } else if files.has(&format!("{p}.weight_g")) {
        (format!("{p}.weight_g"), format!("{p}.weight_v"))
    } else {
        // Already folded (a converted checkpoint).
        let n = format!("{p}.weight");
        let shape = files.info(&n).with_context(|| format!("missing tensor {n}"))?.shape.clone();
        return Ok((files.f32(&n)?, shape));
    };
    let shape = files.info(&vn).with_context(|| format!("missing tensor {vn}"))?.shape.clone();
    let g = files.f32(&gn)?;
    let mut v = files.f32(&vn)?;
    let rows = shape[0];
    if g.len() != rows || v.len() % rows != 0 {
        bail!("{p}: weight norm shapes {:?} / {} do not match", shape, g.len());
    }
    let n = v.len() / rows;
    for (r, row) in v.chunks_mut(n).enumerate() {
        let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() as f32;
        let s = g[r] / norm;
        for x in row {
            *x *= s;
        }
    }
    Ok((v, shape))
}

fn bias(files: &ModelFiles, p: &str, n: usize) -> Result<Vec<f32>> {
    let b = files.f32(&format!("{p}.bias"))?;
    if b.len() != n {
        bail!("{p}.bias: {} values, expected {n}", b.len());
    }
    Ok(b)
}

fn depthwise(files: &ModelFiles, p: &str, dilation: usize) -> Result<Depthwise> {
    let (w, shape) = wn(files, p)?;
    let [c, 1, k] = shape[..] else {
        bail!("{p}: expected a depthwise kernel [c, 1, k], got {shape:?}")
    };
    let mut wt = vec![0f32; k * c];
    for ch in 0..c {
        for j in 0..k {
            wt[j * c + ch] = w[ch * k + j];
        }
    }
    Ok(Depthwise {
        w: wt,
        b: bias(files, p, c)?,
        c,
        k,
        dilation,
    })
}

/// A 1x1 conv `[out, in, 1]`.
fn pointwise(files: &ModelFiles, p: &str, with_bias: bool, f32w: bool) -> Result<Mat> {
    let (w, shape) = wn(files, p)?;
    let [o, i, 1] = shape[..] else {
        bail!("{p}: expected a 1x1 kernel [out, in, 1], got {shape:?}")
    };
    let b = if with_bias { Some(bias(files, p, o)?) } else { None };
    Mat::new(&w, o, i, b, f32w)
}

fn snake(files: &ModelFiles, p: &str, c: usize) -> Result<Snake> {
    let a = files.f32(&format!("{p}.alpha"))?;
    if a.len() != c {
        bail!("{p}.alpha: {} values, expected {c}", a.len());
    }
    Ok(Snake::new(a))
}

impl Snac {
    /// Loads a SNAC checkpoint directory (`config.json` + `pytorch_model.bin` or safetensors).
    pub fn load(dir: &Path, opts: &SnacOptions) -> Result<Self> {
        let cfg = SnacConfig::load(dir)?;
        let files = ModelFiles::open(dir, true)?;
        let f32w = opts.f32_weights;
        let latent = cfg.latent_dim();
        let mut tables = Vec::with_capacity(cfg.vq_strides.len());
        for i in 0..cfg.vq_strides.len() {
            let p = format!("quantizer.quantizers.{i}");
            let cb = files.f32(&format!("{p}.codebook.weight"))?;
            let (w, shape) = wn(&files, &format!("{p}.out_proj"))?;
            if shape[..] != [latent, cfg.codebook_dim, 1] || cb.len() != cfg.codebook_size * cfg.codebook_dim {
                bail!("{p}: unexpected codebook / out_proj shapes {shape:?}");
            }
            let b = bias(&files, &format!("{p}.out_proj"), latent)?;
            let d = cfg.codebook_dim;
            let mut t = vec![0f32; cfg.codebook_size * latent];
            for (e, row) in t.chunks_mut(latent).enumerate() {
                let code = &cb[e * d..(e + 1) * d];
                for (o, v) in row.iter_mut().enumerate() {
                    *v = b[o] + w[o * d..(o + 1) * d].iter().zip(code).map(|(a, b)| a * b).sum::<f32>();
                }
            }
            tables.push(t);
        }
        let conv_in = depthwise(&files, "decoder.model.0", 1)?;
        let proj_in = pointwise(&files, "decoder.model.1", true, f32w)?;
        if proj_in.n_out != cfg.decoder_dim || proj_in.n_in != latent {
            bail!(
                "decoder.model.1: shape [{}, {}], expected [{}, {latent}]",
                proj_in.n_out,
                proj_in.n_in,
                cfg.decoder_dim
            );
        }
        // With local attention, it is layer 2 and the blocks follow.
        let first = if cfg.attn_window_size.is_some() { 3 } else { 2 };
        let attn = match cfg.attn_window_size {
            Some(w) => Some(LocalAttention::load(&files, "decoder.model.2", cfg.decoder_dim, w, f32w)?),
            None => None,
        };
        let mut blocks = Vec::new();
        let mut c = cfg.decoder_dim;
        for (bi, &s) in cfg.decoder_rates.iter().enumerate() {
            let p = format!("decoder.model.{}.block", bi + first);
            let co = c / 2;
            let (w, shape) = wn(&files, &format!("{p}.1"))?;
            let k = 2 * s;
            if shape[..] != [c, co, k] {
                bail!("{p}.1: transposed conv shape {shape:?}, expected [{c}, {co}, {k}]");
            }
            // PyTorch [in][out][tap] -> rows (tap, out) of `in` values.
            let mut r = vec![0f32; w.len()];
            for i in 0..c {
                for o in 0..co {
                    for j in 0..k {
                        r[(j * co + o) * c + i] = w[(i * co + o) * k + j];
                    }
                }
            }
            let up = Mat::new(&r, k * co, c, None, f32w)?;
            let mut next = 2;
            let noise = if cfg.noise {
                next = 3;
                Some(pointwise(&files, &format!("{p}.2.linear"), false, f32w)?)
            } else {
                None
            };
            let mut res = Vec::new();
            for (j, dil) in [1, 3, 9].into_iter().enumerate() {
                let rp = format!("{p}.{}.block", next + j);
                res.push(ResUnit {
                    snake1: snake(&files, &format!("{rp}.0"), co)?,
                    conv: depthwise(&files, &format!("{rp}.1"), dil)?,
                    snake2: snake(&files, &format!("{rp}.2"), co)?,
                    pw: pointwise(&files, &format!("{rp}.3"), true, f32w)?,
                });
            }
            blocks.push(Block {
                snake: snake(&files, &format!("{p}.0"), c)?,
                up,
                up_bias: bias(&files, &format!("{p}.1"), co)?,
                stride: s,
                kernel: k,
                pad: s.div_ceil(2),
                out_pad: s % 2,
                c_out: co,
                noise,
                res,
            });
            c = co;
        }
        let n = cfg.decoder_rates.len() + first;
        let snake_out = snake(&files, &format!("decoder.model.{n}"), c)?;
        let (w, shape) = wn(&files, &format!("decoder.model.{}", n + 1))?;
        let [1, ci, k] = shape[..] else {
            bail!("decoder output conv: shape {shape:?}")
        };
        if ci != c {
            bail!("decoder output conv: {ci} input channels, expected {c}");
        }
        let mut conv_out = vec![0f32; k * c];
        for ch in 0..c {
            for j in 0..k {
                conv_out[j * c + ch] = w[ch * k + j];
            }
        }
        let conv_out_b = bias(&files, &format!("decoder.model.{}", n + 1), 1)?[0];
        let encoder = if files.has("encoder.block.0.bias") {
            Some(Encoder::load(&files, &cfg).context("SNAC encoder")?)
        } else {
            None
        };
        Ok(Self {
            encoder,
            cfg,
            tables,
            conv_in,
            proj_in,
            attn,
            blocks,
            snake_out,
            conv_out,
            conv_out_b,
            conv_out_k: k,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.cfg.sampling_rate
    }

    /// Output samples per latent step (the product of the decoder rates: 512 at 24 kHz).
    pub fn hop(&self) -> usize {
        self.cfg.decoder_rates.iter().product()
    }

    /// Output samples per code of the coarsest codebook (2048 at 24 kHz).
    pub fn samples_per_frame(&self) -> usize {
        self.hop() * self.cfg.vq_strides[0]
    }

    /// Number of codebooks.
    pub fn codebooks(&self) -> usize {
        self.cfg.vq_strides.len()
    }

    /// Whether the checkpoint has the encoder ([`Snac::encode`]).
    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    /// Encodes mono samples at [`Snac::sample_rate`] (in [-1, 1]) to codes, as the reference's
    /// `SNAC.encode`: the input is right-padded with zeros to a multiple of
    /// [`SnacConfig::pad_multiple`], then codebook `i` holds `n * vq_strides[0] / vq_strides[i]`
    /// codes for `n` coarse frames (`n = padded / samples_per_frame()`). Empty input: no codes.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<u32>>> {
        let enc = self.encoder.as_ref().context("this SNAC checkpoint has no encoder weights")?;
        let cfg = &self.cfg;
        if samples.is_empty() {
            return Ok(vec![Vec::new(); cfg.vq_strides.len()]);
        }
        let m = cfg.pad_multiple();
        let mut audio = samples.to_vec();
        audio.resize(samples.len().div_ceil(m) * m, 0.0);
        let (z, t) = enc.latent(&audio);
        let latent = cfg.latent_dim();
        let d = cfg.codebook_dim;
        let mut residual = z;
        let mut codes = Vec::with_capacity(cfg.vq_strides.len());
        for (i, &s) in cfg.vq_strides.iter().enumerate() {
            let n = t / s;
            // Average pool by the stride.
            let mut pooled = vec![0f32; n * latent];
            for_rows(&mut pooled, latent, |j, row| {
                for k in 0..s {
                    for (v, &r) in row.iter_mut().zip(&residual[(j * s + k) * latent..(j * s + k + 1) * latent]) {
                        *v += r;
                    }
                }
                for v in row.iter_mut() {
                    *v /= s as f32;
                }
            });
            let e = enc.in_proj[i].forward(&pooled, n); // [n, d]
            let cb = &enc.codebooks[i];
            // Nearest entry by the reference's distance `|e|^2 - 2 e.c + |c|^2` over normalized
            // vectors (the first minimum wins); stored as f32 bits to go through `for_rows`.
            let mut best = vec![0f32; n];
            for_rows(&mut best, 1, |j, b| {
                let mut v = e[j * d..(j + 1) * d].to_vec();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in &mut v {
                    *x /= norm;
                }
                let e2: f32 = v.iter().map(|x| x * x).sum();
                let (mut bi, mut bd) = (0usize, f32::INFINITY);
                for (k, c) in cb.chunks_exact(d).enumerate() {
                    let dot: f32 = v.iter().zip(c).map(|(a, b)| a * b).sum();
                    let c2: f32 = c.iter().map(|x| x * x).sum();
                    let dist = e2 - 2.0 * dot + c2;
                    if dist < bd {
                        (bi, bd) = (k, dist);
                    }
                }
                b[0] = f32::from_bits(bi as u32);
            });
            let idx: Vec<u32> = best.iter().map(|b| b.to_bits()).collect();
            // The residual minus the chosen entries (through `out_proj`), repeated by the stride.
            let table = &self.tables[i];
            for_rows(&mut residual, latent, |ti, row| {
                let e = idx[ti / s] as usize;
                for (v, &a) in row.iter_mut().zip(&table[e * latent..(e + 1) * latent]) {
                    *v -= a;
                }
            });
            codes.push(idx);
        }
        Ok(codes)
    }

    /// Decodes `n` coarse frames: `codes[i]` holds `n * vq_strides[0] / vq_strides[i]` codes of
    /// codebook `i`. Returns `n * samples_per_frame()` samples in [-1, 1].
    pub fn decode(&self, codes: &[&[u32]], noise: Noise) -> Result<Vec<f32>> {
        self.decode_at(codes, 0, noise)
    }

    /// [`Snac::decode`] for a window whose first coarse frame is frame `frame0` of the clip
    /// (the noise depends on absolute positions, see [`Noise`]).
    pub fn decode_at(&self, codes: &[&[u32]], frame0: usize, noise: Noise) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if codes.len() != cfg.vq_strides.len() {
            bail!("SNAC: {} codebooks given, the model has {}", codes.len(), cfg.vq_strides.len());
        }
        let s0 = cfg.vq_strides[0];
        let n = codes[0].len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let t = n * s0;
        for (i, (c, &s)) in codes.iter().zip(&cfg.vq_strides).enumerate() {
            if c.len() * s != t {
                bail!("SNAC: codebook {i} has {} codes, expected {}", c.len(), t / s);
            }
            if let Some(&bad) = c.iter().find(|&&v| v as usize >= cfg.codebook_size) {
                bail!("SNAC: code {bad} of codebook {i} is out of range (0..{})", cfg.codebook_size);
            }
        }
        let latent = cfg.latent_dim();
        let mut z = vec![0f32; t * latent];
        for_rows(&mut z, latent, |ti, row| {
            for ((c, &s), table) in codes.iter().zip(&cfg.vq_strides).zip(&self.tables) {
                let e = c[ti / s] as usize;
                for (v, &a) in row.iter_mut().zip(&table[e * latent..(e + 1) * latent]) {
                    *v += a;
                }
            }
        });
        let h = self.conv_in.forward(&z, t);
        let mut x = self.proj_in.forward(&h, t);
        if let Some(a) = &self.attn {
            a.forward(&mut x, t);
        }
        let (mut len, mut t0) = (t, frame0 * s0);
        for (l, b) in self.blocks.iter().enumerate() {
            let (y, ny) = b.forward(&x, len, t0, l, noise);
            t0 *= b.stride;
            x = y;
            len = ny;
        }
        let h = self.snake_out.apply(&x);
        let (c, k) = (self.snake_out.alpha.len(), self.conv_out_k);
        let half = (k - 1) / 2;
        let mut out = vec![0f32; len];
        let rows = 4096;
        for_chunks(&mut out, rows, true, |ci, block| {
            for (j, o) in block.iter_mut().enumerate() {
                let ti = ci * rows + j;
                let mut acc = self.conv_out_b;
                for tap in 0..k {
                    let src = ti as isize + tap as isize - half as isize;
                    if src < 0 || src >= len as isize {
                        continue;
                    }
                    let xs = &h[src as usize * c..(src as usize + 1) * c];
                    acc += xs
                        .iter()
                        .zip(&self.conv_out[tap * c..(tap + 1) * c])
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                }
                *o = acc.tanh();
            }
        });
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_matches_libm() {
        let mut worst = 0f32;
        let mut x = -300.0f32;
        while x < 300.0 {
            worst = worst.max((sin_f32(x) - x.sin()).abs());
            x += 0.000_731;
        }
        assert!(worst < 1e-6, "worst sin error {worst}");
    }

    #[test]
    fn pad_multiples() {
        let cfg24: SnacConfig = serde_json::from_str(
            r#"{"sampling_rate": 24000, "encoder_dim": 48, "encoder_rates": [2, 4, 8, 8], "decoder_dim": 1024,
                "decoder_rates": [8, 8, 4, 2], "attn_window_size": null, "codebook_size": 4096, "codebook_dim": 8,
                "vq_strides": [4, 2, 1]}"#,
        )
        .unwrap();
        assert_eq!((cfg24.hop(), cfg24.pad_multiple(), cfg24.latent_dim()), (512, 2048, 768));
        let cfg44: SnacConfig = serde_json::from_str(
            r#"{"sampling_rate": 44100, "encoder_dim": 64, "encoder_rates": [2, 3, 8, 8], "decoder_dim": 1536,
                "decoder_rates": [8, 8, 3, 2], "attn_window_size": 32, "codebook_size": 4096, "codebook_dim": 8,
                "vq_strides": [8, 4, 2, 1]}"#,
        )
        .unwrap();
        assert_eq!((cfg44.hop(), cfg44.pad_multiple()), (384, 384 * 32));
    }

    #[test]
    fn gaussian_moments() {
        let n = 200_000;
        let v: Vec<f64> = (0..n).map(|i| gaussian(7, 2, i) as f64).collect();
        let mean = v.iter().sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.02, "mean {mean} var {var}");
        assert_eq!(gaussian(7, 2, 5), gaussian(7, 2, 5));
        assert_ne!(gaussian(7, 2, 5), gaussian(7, 3, 5));
    }
}
