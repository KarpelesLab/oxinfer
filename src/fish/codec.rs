//! The audio codec of Drama 3 (`codec.pth`: Fish Audio's causal DAC variant, `DAC` in
//! fish-speech's `fish_speech/models/dac/modded_dac.py` with the quantizer of `rvq.py` and the
//! settings of `configs/modded_dac_vq.yaml`): ten codes per frame of 2048 samples at 44.1 kHz
//! (21.5 frames per second), one of a semantic codebook of 4096 entries and nine of residual
//! codebooks of 1024.
//!
//! Decoding ([`Codec::decode`]):
//! 1. the frame's latent (1024 wide): the sum over the ten codebooks of each code's 8-d entry
//!    through that codebook's output projection (one table per codebook here);
//! 2. eight causal transformer layers over the frames (RMSNorm, attention over the last 128
//!    frames with full rotary positions, SiLU-gated MLP, LayerScale) and a final norm;
//! 3. twice: a transposed convolution doubling the rate and a ConvNeXt block (causal depthwise
//!    k7, LayerNorm, a GELU MLP);
//! 4. the DAC decoder: a k7 convolution to 1536 channels, four blocks (Snake, a transposed
//!    convolution of stride 8 / 8 / 4 / 2 halving the channels, three residual units of a dilated
//!    k7 and a 1x1 convolution), Snake, a k7 convolution to one channel, tanh.
//!
//! Encoding ([`Codec::encode`], for reference clips) is the mirror: a k7 convolution, four
//! blocks (three residual units, Snake, a strided convolution of 2 / 4 / 8 / 8 doubling the
//! channels; the last followed by four transformer layers over 512 frames), Snake and a k3
//! convolution; two strided convolutions with ConvNeXt blocks; eight transformer layers; then
//! per codebook the nearest entry (cosine) of the latent projected to 8 dimensions, the next
//! codebook taking what is left.
//!
//! Every convolution is causal (padded on the left only), weight-normalized ones folded at load.
//! Activations are `[time, channels]` row-major, the convolutions matmuls over gathered taps;
//! weights are kept in f32.

use crate::ggml::{GgmlType, GgufMatrix};
use crate::kernels::*;
use crate::model::{LayerCache, Lin, attention, linear, linear_multi};
use crate::safetensors::ModelFiles;
use crate::snac::{sin_f32, wn};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

pub const SAMPLE_RATE: u32 = 44100;
/// Samples per frame of codes.
pub const HOP: usize = 2048;
const LATENT: usize = 1024;
const RMS_EPS: f32 = 1e-5;
/// Floats gathered per matmul of a convolution (bounds the scratch memory).
const COL_FLOATS: usize = 1 << 24;

fn mat(w: &[f32], n_out: usize, n_in: usize) -> Result<Lin<'static>> {
    if w.len() != n_out * n_in {
        bail!("matrix of {} values, expected [{n_out}, {n_in}]", w.len());
    }
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(Lin::gguf(GgufMatrix::from_bytes(GgmlType::F32, &bytes, n_out, n_in)?))
}

fn vector(files: &ModelFiles, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = files.f32(name)?;
    if v.len() != n {
        bail!("{name}: {} values, expected {n}", v.len());
    }
    Ok(v)
}

fn add_bias(y: &mut [f32], b: &[f32]) {
    let c = b.len();
    let rows = (16384 / c).max(1);
    for_chunks(y, rows * c, y.len() > rows * c, |_, block| {
        for r in block.chunks_mut(c) {
            add_inplace(r, b);
        }
    });
}

/// A causal convolution (`CausalConvNet`): `stride` rows in for one out, the input padded on the
/// left by the kernel's reach minus the stride and on the right to a whole number of strides.
struct Conv {
    /// `[c_out, k * c_in]`, tap-major.
    w: Lin<'static>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    k: usize,
    dilation: usize,
    stride: usize,
}

impl Conv {
    /// The convolution under `p` (`[out, in, k]`, weight-normalized or plain).
    fn load(files: &ModelFiles, p: &str, dilation: usize, stride: usize) -> Result<Self> {
        let (w, shape) = wn(files, p)?;
        let [c_out, _, _] = shape[..] else {
            bail!("{p}: expected a kernel [out, in, k], got {shape:?}")
        };
        Self::new(&w, &shape, vector(files, &format!("{p}.bias"), c_out)?, dilation, stride).with_context(|| p.to_string())
    }

    /// From a torch kernel `w` of `shape` (`[out, in, k]`) and its bias.
    fn new(w: &[f32], shape: &[usize], b: Vec<f32>, dilation: usize, stride: usize) -> Result<Self> {
        let [c_out, c_in, k] = shape[..] else {
            bail!("expected a kernel [out, in, k], got {shape:?}")
        };
        if w.len() != c_out * c_in * k || b.len() != c_out || (k - 1) * dilation + 1 < stride {
            bail!("kernel {shape:?} with a bias of {} and stride {stride}", b.len());
        }
        let mut m = vec![0f32; w.len()];
        for o in 0..c_out {
            for c in 0..c_in {
                for i in 0..k {
                    m[(o * k + i) * c_in + c] = w[(o * c_in + c) * k + i];
                }
            }
        }
        Ok(Self {
            w: mat(&m, c_out, k * c_in)?,
            b,
            c_in,
            c_out,
            k,
            dilation,
            stride,
        })
    }

    fn bytes(&self) -> usize {
        self.w.bytes()
    }

    /// `x` (`[t, c_in]`) to `[ceil(t / stride), c_out]`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (ci, co, k, d, s) = (self.c_in, self.c_out, self.k, self.dilation, self.stride);
        let t = x.len() / ci;
        let t_out = t.div_ceil(s);
        let mut y = vec![0f32; t_out * co];
        if k == 1 && s == 1 {
            linear(&self.w, co, ci, x, t, &mut y);
        } else {
            let pad = ((k - 1) * d + 1 - s) as isize;
            let w = k * ci;
            let rows = (COL_FLOATS / w).max(1);
            let mut col = vec![0f32; rows.min(t_out) * w];
            let mut j0 = 0;
            while j0 < t_out {
                let n = rows.min(t_out - j0);
                for_chunks(&mut col[..n * w], w, n > 1, |j, r| {
                    for i in 0..k {
                        let src = ((j0 + j) * s + i * d) as isize - pad;
                        let o = &mut r[i * ci..(i + 1) * ci];
                        if src >= 0 && (src as usize) < t {
                            o.copy_from_slice(&x[src as usize * ci..(src as usize + 1) * ci]);
                        } else {
                            o.fill(0.0);
                        }
                    }
                });
                linear(&self.w, co, w, &col[..n * w], n, &mut y[j0 * co..(j0 + n) * co]);
                j0 += n;
            }
        }
        add_bias(&mut y, &self.b);
        y
    }
}

/// A causal transposed convolution (`CausalTransConvNet`) of kernel `stride` or `2 * stride`:
/// every input row gives `stride` output rows, each from the row itself and (for the longer
/// kernel) the one before it.
struct TransConv {
    /// `[stride * c_out, reach * c_in]`: the current row's taps, then the previous row's.
    w: Lin<'static>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    stride: usize,
    reach: usize,
}

impl TransConv {
    /// The transposed convolution under `p` (`[in, out, k]`).
    fn load(files: &ModelFiles, p: &str, stride: usize) -> Result<Self> {
        let (w, shape) = wn(files, p)?;
        let [_, c_out, _] = shape[..] else {
            bail!("{p}: expected a kernel [in, out, k], got {shape:?}")
        };
        Self::new(&w, &shape, vector(files, &format!("{p}.bias"), c_out)?, stride).with_context(|| p.to_string())
    }

    /// From a torch kernel `w` of `shape` (`[in, out, k]`) and its bias.
    fn new(w: &[f32], shape: &[usize], bias: Vec<f32>, stride: usize) -> Result<Self> {
        let [c_in, c_out, k] = shape[..] else {
            bail!("expected a kernel [in, out, k], got {shape:?}")
        };
        if (k != stride && k != 2 * stride) || w.len() != c_in * c_out * k || bias.len() != c_out {
            bail!("kernel {shape:?} with stride {stride} is not supported");
        }
        let reach = k / stride;
        let mut m = vec![0f32; stride * c_out * reach * c_in];
        for r in 0..stride {
            for o in 0..c_out {
                let row = &mut m[(r * c_out + o) * reach * c_in..(r * c_out + o + 1) * reach * c_in];
                for back in 0..reach {
                    for c in 0..c_in {
                        row[back * c_in + c] = w[(c * c_out + o) * k + r + back * stride];
                    }
                }
            }
        }
        Ok(Self {
            w: mat(&m, stride * c_out, reach * c_in)?,
            b: bias,
            c_in,
            c_out,
            stride,
            reach,
        })
    }

    fn bytes(&self) -> usize {
        self.w.bytes()
    }

    /// `x` (`[t, c_in]`) to `[t * stride, c_out]`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (ci, co, s) = (self.c_in, self.c_out, self.stride);
        let t = x.len() / ci;
        let mut y = vec![0f32; t * s * co];
        if self.reach == 1 {
            linear(&self.w, s * co, ci, x, t, &mut y);
        } else {
            let w = 2 * ci;
            let rows = (COL_FLOATS / w).max(1);
            let mut col = vec![0f32; rows.min(t) * w];
            let mut j0 = 0;
            while j0 < t {
                let n = rows.min(t - j0);
                for_chunks(&mut col[..n * w], w, n > 1, |j, r| {
                    let q = j0 + j;
                    r[..ci].copy_from_slice(&x[q * ci..(q + 1) * ci]);
                    if q > 0 {
                        r[ci..].copy_from_slice(&x[(q - 1) * ci..q * ci]);
                    } else {
                        r[ci..].fill(0.0);
                    }
                });
                linear(&self.w, s * co, w, &col[..n * w], n, &mut y[j0 * s * co..(j0 + n) * s * co]);
                j0 += n;
            }
        }
        add_bias(&mut y, &self.b);
        y
    }
}

/// Snake, `x + sin(alpha x)^2 / (alpha + 1e-9)` per channel, in place.
struct Snake {
    alpha: Vec<f32>,
    inv: Vec<f32>,
}

impl Snake {
    fn load(files: &ModelFiles, name: &str, c: usize) -> Result<Self> {
        let alpha = vector(files, name, c)?;
        let inv = alpha.iter().map(|&a| 1.0 / (a + 1e-9)).collect();
        Ok(Self { alpha, inv })
    }

    fn apply(&self, x: &mut [f32]) {
        let c = self.alpha.len();
        let rows = (8192 / c).max(1);
        for_chunks(x, rows * c, x.len() > rows * c, |_, block| {
            for row in block.chunks_mut(c) {
                for ((v, &a), &inv) in row.iter_mut().zip(&self.alpha).zip(&self.inv) {
                    let s = sin_f32(a * *v);
                    *v += inv * s * s;
                }
            }
        });
    }
}

/// `x + conv1(snake(conv7(snake(x))))`.
struct ResUnit {
    snake1: Snake,
    conv7: Conv,
    snake2: Snake,
    conv1: Conv,
}

impl ResUnit {
    fn load(files: &ModelFiles, p: &str, c: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            snake1: Snake::load(files, &format!("{p}.block.0.alpha"), c)?,
            conv7: Conv::load(files, &format!("{p}.block.1.conv"), dilation, 1)?,
            snake2: Snake::load(files, &format!("{p}.block.2.alpha"), c)?,
            conv1: Conv::load(files, &format!("{p}.block.3.conv"), 1, 1)?,
        })
    }

    fn bytes(&self) -> usize {
        self.conv7.bytes() + self.conv1.bytes()
    }

    fn forward(&self, x: &mut [f32]) {
        let mut h = x.to_vec();
        self.snake1.apply(&mut h);
        let mut h = self.conv7.forward(&h);
        self.snake2.apply(&mut h);
        let h = self.conv1.forward(&h);
        add_inplace(x, &h);
    }
}

fn res_units(files: &ModelFiles, p: &str, first: usize, c: usize) -> Result<Vec<ResUnit>> {
    [1, 3, 9]
        .iter()
        .enumerate()
        .map(|(i, &d)| ResUnit::load(files, &format!("{p}.block.{}", first + i), c, d))
        .collect()
}

/// A ConvNeXt block: causal depthwise k7, LayerNorm, a GELU MLP, a scale per channel, residual.
struct ConvNext {
    /// Taps `[k][c]`.
    dw: Vec<f32>,
    dw_b: Vec<f32>,
    k: usize,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    pw1: Lin<'static>,
    pw1_b: Vec<f32>,
    pw2: Lin<'static>,
    pw2_b: Vec<f32>,
    gamma: Vec<f32>,
}

impl ConvNext {
    fn load(files: &ModelFiles, p: &str, c: usize) -> Result<Self> {
        let name = format!("{p}.dwconv.conv.weight");
        let shape = files.info(&name).with_context(|| format!("missing tensor {name}"))?.shape.clone();
        let [cc, 1, k] = shape[..] else {
            bail!("{name}: expected a depthwise kernel [c, 1, k], got {shape:?}")
        };
        if cc != c {
            bail!("{name}: {cc} channels, expected {c}");
        }
        let w = files.f32(&name)?;
        let mut dw = vec![0f32; k * c];
        for ch in 0..c {
            for j in 0..k {
                dw[j * c + ch] = w[ch * k + j];
            }
        }
        let hidden = files
            .info(&format!("{p}.pwconv1.bias"))
            .with_context(|| format!("missing tensor {p}.pwconv1.bias"))?
            .shape[0];
        Ok(Self {
            dw,
            dw_b: vector(files, &format!("{p}.dwconv.conv.bias"), c)?,
            k,
            ln_w: vector(files, &format!("{p}.norm.weight"), c)?,
            ln_b: vector(files, &format!("{p}.norm.bias"), c)?,
            pw1: mat(&files.f32(&format!("{p}.pwconv1.weight"))?, hidden, c)?,
            pw1_b: vector(files, &format!("{p}.pwconv1.bias"), hidden)?,
            pw2: mat(&files.f32(&format!("{p}.pwconv2.weight"))?, c, hidden)?,
            pw2_b: vector(files, &format!("{p}.pwconv2.bias"), c)?,
            gamma: vector(files, &format!("{p}.gamma"), c)?,
        })
    }

    fn bytes(&self) -> usize {
        self.pw1.bytes() + self.pw2.bytes()
    }

    fn forward(&self, x: &mut [f32]) {
        let c = self.dw_b.len();
        let t = x.len() / c;
        let k = self.k;
        let mut h = vec![0f32; t * c];
        {
            let x = &*x;
            for_chunks(&mut h, c, t > 1, |ti, out| {
                out.copy_from_slice(&self.dw_b);
                for j in 0..k {
                    // Tap j reads k - 1 - j rows back.
                    if ti + j + 1 < k {
                        continue;
                    }
                    let src = ti + j + 1 - k;
                    let xs = &x[src * c..(src + 1) * c];
                    let ws = &self.dw[j * c..(j + 1) * c];
                    for ((o, &xv), &wv) in out.iter_mut().zip(xs).zip(ws) {
                        *o += xv * wv;
                    }
                }
                let mean = out.iter().sum::<f32>() / c as f32;
                let var = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
                let s = 1.0 / (var + 1e-6).sqrt();
                for ((v, w), b) in out.iter_mut().zip(&self.ln_w).zip(&self.ln_b) {
                    *v = (*v - mean) * s * w + b;
                }
            });
        }
        let hidden = self.pw1_b.len();
        let mut m = vec![0f32; t * hidden];
        linear(&self.pw1, hidden, c, &h, t, &mut m);
        for_chunks(&mut m, hidden, t > 1, |_, r| {
            for (v, b) in r.iter_mut().zip(&self.pw1_b) {
                *v = crate::sd::ops::gelu_erf(*v + b);
            }
        });
        linear(&self.pw2, c, hidden, &m, t, &mut h);
        for_chunks(x, c, t > 1, |ti, r| {
            let hr = &h[ti * c..(ti + 1) * c];
            for (((v, &y), &b), &g) in r.iter_mut().zip(hr).zip(&self.pw2_b).zip(&self.gamma) {
                *v += (y + b) * g;
            }
        });
    }
}

struct TfLayer {
    attn_norm: Vec<f32>,
    /// Rows `[query | key | value]`.
    wqkv: Lin<'static>,
    wo: Lin<'static>,
    attn_scale: Vec<f32>,
    ffn_norm: Vec<f32>,
    w1: Lin<'static>,
    w3: Lin<'static>,
    w2: Lin<'static>,
    ffn_scale: Vec<f32>,
}

/// `WindowLimitedTransformer`: causal layers over the whole sequence, each row attending the
/// last `window` rows.
struct Transformer {
    layers: Vec<TfLayer>,
    norm: Vec<f32>,
    dim: usize,
    heads: usize,
    window: usize,
    /// `[positions, head_dim / 2]` cosines and sines as stored (bf16 values).
    cos: Vec<f32>,
    sin: Vec<f32>,
    inter: usize,
}

impl Transformer {
    fn load(files: &ModelFiles, p: &str, dim: usize, window: usize) -> Result<Self> {
        let hd = 64;
        let heads = dim / hd;
        let mut layers = Vec::new();
        let mut inter = 0;
        while files.has(&format!("{p}.layers.{}.attention.wqkv.weight", layers.len())) {
            let lp = format!("{p}.layers.{}", layers.len());
            let w1 = files.f32(&format!("{lp}.feed_forward.w1.weight"))?;
            inter = w1.len() / dim;
            let m =
                |n: &str, n_out: usize, n_in: usize| -> Result<Lin<'static>> { mat(&files.f32(&format!("{lp}.{n}.weight"))?, n_out, n_in) };
            layers.push(TfLayer {
                attn_norm: vector(files, &format!("{lp}.attention_norm.weight"), dim)?,
                wqkv: m("attention.wqkv", 3 * dim, dim)?,
                wo: m("attention.wo", dim, dim)?,
                attn_scale: vector(files, &format!("{lp}.attention_layer_scale.gamma"), dim)?,
                ffn_norm: vector(files, &format!("{lp}.ffn_norm.weight"), dim)?,
                w1: mat(&w1, inter, dim)?,
                w3: m("feed_forward.w3", inter, dim)?,
                w2: m("feed_forward.w2", dim, inter)?,
                ffn_scale: vector(files, &format!("{lp}.ffn_layer_scale.gamma"), dim)?,
            });
        }
        if layers.is_empty() {
            bail!("{p}: no transformer layers");
        }
        // The rotary table as the checkpoint keeps it (bf16); positions beyond it are computed
        // the way the reference builds it.
        let half = hd / 2;
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        let name = format!("{p}.freqs_cis");
        if files.info(&name).is_some_and(|i| i.shape.len() == 3 && i.shape[1..] == [half, 2]) {
            for pair in files.f32(&name)?.chunks(2) {
                cos.push(pair[0]);
                sin.push(pair[1]);
            }
        }
        Ok(Self {
            layers,
            norm: vector(files, &format!("{p}.norm.weight"), dim)?,
            dim,
            heads,
            window,
            cos,
            sin,
            inter,
        })
    }

    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.wqkv.bytes() + l.wo.bytes() + l.w1.bytes() + l.w2.bytes() + l.w3.bytes())
            .sum()
    }

    /// The table rows of positions `0..t` (`precompute_freqs_cis`: float32 angles, bf16 values).
    fn tables(&self, t: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.dim / self.heads / 2;
        let (mut cos, mut sin) = (self.cos.clone(), self.sin.clone());
        cos.truncate(t * half);
        sin.truncate(t * half);
        let round = |v: f32| half::bf16::from_f32(v).to_f32();
        for p in cos.len() / half..t {
            for i in 0..half {
                let f = 1.0 / 10000f32.powf((2 * i) as f32 / (2 * half) as f32);
                let a = p as f32 * f;
                cos.push(round(a.cos()));
                sin.push(round(a.sin()));
            }
        }
        (cos, sin)
    }

    fn forward(&self, x: &mut [f32]) {
        let (d, nh) = (self.dim, self.heads);
        let hd = d / nh;
        let half = hd / 2;
        let t = x.len() / d;
        let par = t > 1;
        let (cos, sin) = self.tables(t);
        let scale = (hd as f32).powf(-0.5);
        let mut xn = vec![0f32; t * d];
        let mut y = vec![0f32; t * d];
        for l in &self.layers {
            rmsnorm_f32w_rows(x, d, &l.attn_norm, RMS_EPS, &mut xn);
            let mut qkv = vec![0f32; t * 3 * d];
            linear(&l.wqkv, 3 * d, d, &xn, t, &mut qkv);
            let mut q = vec![0f32; t * d];
            let mut k = vec![0f32; t * d];
            let mut v = vec![0f32; t * d];
            for_chunks(&mut q, hd, par, |i, o| {
                let (ti, h) = (i / nh, i % nh);
                o.copy_from_slice(&qkv[ti * 3 * d + h * hd..ti * 3 * d + (h + 1) * hd]);
                apply_rope_interleaved(o, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                o.iter_mut().for_each(|v| *v *= scale);
            });
            for_chunks(&mut k, hd, par, |i, o| {
                let (ti, h) = (i / nh, i % nh);
                o.copy_from_slice(&qkv[ti * 3 * d + d + h * hd..ti * 3 * d + d + (h + 1) * hd]);
                apply_rope_interleaved(o, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            });
            for (ti, r) in v.chunks_mut(d).enumerate() {
                r.copy_from_slice(&qkv[ti * 3 * d + 2 * d..(ti + 1) * 3 * d]);
            }
            drop(qkv);
            let mut cache = LayerCache::new(nh, hd);
            cache.push(&k, &v, t);
            let mut att = vec![0f32; t * d];
            attention(&q, &cache, nh, t, 0, Some(self.window), &mut att);
            linear(&l.wo, d, d, &att, t, &mut y);
            for_chunks(x, d, par, |ti, r| {
                for ((v, &a), &g) in r.iter_mut().zip(&y[ti * d..(ti + 1) * d]).zip(&l.attn_scale) {
                    *v += a * g;
                }
            });
            rmsnorm_f32w_rows(x, d, &l.ffn_norm, RMS_EPS, &mut xn);
            let mi = self.inter;
            let mut g = vec![0f32; t * mi];
            let mut u = vec![0f32; t * mi];
            linear_multi(&mut [(&l.w1, mi, &mut g[..]), (&l.w3, mi, &mut u[..])], d, &xn, t);
            silu_mul(&mut g, &u);
            linear(&l.w2, d, mi, &g, t, &mut y);
            for_chunks(x, d, par, |ti, r| {
                for ((v, &a), &g) in r.iter_mut().zip(&y[ti * d..(ti + 1) * d]).zip(&l.ffn_scale) {
                    *v += a * g;
                }
            });
        }
        rmsnorm_f32w_rows_inplace(x, d, &self.norm, RMS_EPS);
    }
}

struct DecBlock {
    snake: Snake,
    up: TransConv,
    res: Vec<ResUnit>,
}

struct EncBlock {
    res: Vec<ResUnit>,
    snake: Snake,
    down: Conv,
    transformer: Option<Transformer>,
}

/// One codebook: its entries (8-d), their output projections and the input projection.
struct Codebook {
    size: usize,
    /// `[size, 1024]`: `out_proj` of every entry.
    table: Vec<f32>,
    /// `[size, 8]`, L2-normalized (encoding).
    entries: Vec<f32>,
    /// `[8, 1024]` and its bias.
    in_proj: Vec<f32>,
    in_bias: Vec<f32>,
}

impl Codebook {
    fn load(files: &ModelFiles, p: &str) -> Result<Self> {
        let cb = files.f32(&format!("{p}.codebook.weight"))?;
        let dim = files
            .info(&format!("{p}.codebook.weight"))
            .with_context(|| format!("missing tensor {p}.codebook.weight"))?
            .shape[1];
        let size = cb.len() / dim;
        let (wo, so) = wn(files, &format!("{p}.out_proj"))?;
        let (wi, si) = wn(files, &format!("{p}.in_proj"))?;
        if so != [LATENT, dim, 1] || si != [dim, LATENT, 1] {
            bail!("{p}: projections {so:?} / {si:?}, expected [{LATENT}, {dim}, 1] and back");
        }
        let bo = vector(files, &format!("{p}.out_proj.bias"), LATENT)?;
        let mut table = vec![0f32; size * LATENT];
        for_chunks(&mut table, LATENT, true, |e, row| {
            let v = &cb[e * dim..(e + 1) * dim];
            for (o, r) in row.iter_mut().enumerate() {
                *r = bo[o] + wo[o * dim..(o + 1) * dim].iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
            }
        });
        // `F.normalize`: over max(norm, 1e-12).
        let mut entries = cb;
        for e in entries.chunks_mut(dim) {
            let n = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            e.iter_mut().for_each(|v| *v /= n);
        }
        Ok(Self {
            size,
            table,
            entries,
            in_proj: wi,
            in_bias: vector(files, &format!("{p}.in_proj.bias"), dim)?,
        })
    }

    /// The nearest entry to every row of `z` (`[t, 1024]`), subtracted from it.
    fn quantize(&self, z: &mut [f32]) -> Vec<u32> {
        let dim = self.in_bias.len();
        let t = z.len() / LATENT;
        let mut codes = vec![0f32; t];
        {
            let z = &*z;
            for_chunks(&mut codes, 1, t > 1, |ti, out| {
                let row = &z[ti * LATENT..(ti + 1) * LATENT];
                let mut e: Vec<f32> = (0..dim)
                    .map(|j| self.in_bias[j] + dot(&self.in_proj[j * LATENT..(j + 1) * LATENT], row))
                    .collect();
                let n = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                e.iter_mut().for_each(|v| *v /= n);
                let e2: f32 = e.iter().map(|v| v * v).sum();
                // The reference's distance `|e|^2 - 2 e.c + |c|^2` over the normalized vectors,
                // the first entry on ties.
                let mut best = (f32::INFINITY, 0usize);
                for (i, c) in self.entries.chunks(dim).enumerate() {
                    let c2: f32 = c.iter().map(|v| v * v).sum();
                    let dist = e2 - 2.0 * e.iter().zip(c).map(|(a, b)| a * b).sum::<f32>() + c2;
                    if dist < best.0 {
                        best = (dist, i);
                    }
                }
                out[0] = best.1 as f32;
            });
        }
        let codes: Vec<u32> = codes.iter().map(|&c| c as u32).collect();
        for_chunks(z, LATENT, t > 1, |ti, row| {
            let e = codes[ti] as usize;
            for (v, q) in row.iter_mut().zip(&self.table[e * LATENT..(e + 1) * LATENT]) {
                *v -= q;
            }
        });
        codes
    }
}

struct Encoder {
    conv_in: Conv,
    blocks: Vec<EncBlock>,
    snake: Snake,
    conv_out: Conv,
    down: Vec<(Conv, ConvNext)>,
    pre: Transformer,
}

/// The codec: the decoder, and the encoder once a clip is encoded.
pub struct Codec {
    path: PathBuf,
    codebooks: Vec<Codebook>,
    post: Transformer,
    up: Vec<(TransConv, ConvNext)>,
    conv_in: Conv,
    blocks: Vec<DecBlock>,
    snake: Snake,
    conv_out: Conv,
    encoder: std::sync::OnceLock<Encoder>,
    pub weight_bytes: usize,
}

/// The strides of the DAC encoder and decoder (`modded_dac_vq.yaml`).
const ENCODER_RATES: [usize; 4] = [2, 4, 8, 8];
const DECODER_RATES: [usize; 4] = [8, 8, 4, 2];
/// The quantizer's own down / up sampling.
const QUANTIZER_RATES: [usize; 2] = [2, 2];

impl Codec {
    /// Loads the decoder from `codec.pth` (a `torch.save` state dict, or the same tensors as
    /// safetensors).
    pub fn load(path: &Path) -> Result<Self> {
        let files = ModelFiles::open_file(path, false).with_context(|| format!("open {}", path.display()))?;
        let f = &files;
        let mut codebooks = vec![Codebook::load(f, "quantizer.semantic_quantizer.quantizers.0")?];
        let mut i = 0;
        while f.has(&format!("quantizer.quantizer.quantizers.{i}.codebook.weight")) {
            codebooks.push(Codebook::load(f, &format!("quantizer.quantizer.quantizers.{i}"))?);
            i += 1;
        }
        let post = Transformer::load(f, "quantizer.post_module", LATENT, 128)?;
        let up = (0..QUANTIZER_RATES.len())
            .map(|i| {
                Ok((
                    TransConv::load(
                        f,
                        &format!("quantizer.upsample.{i}.0.conv"),
                        QUANTIZER_RATES[QUANTIZER_RATES.len() - 1 - i],
                    )?,
                    ConvNext::load(f, &format!("quantizer.upsample.{i}.1"), LATENT)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_in = Conv::load(f, "decoder.model.0.conv", 1, 1)?;
        if conv_in.c_in != LATENT {
            bail!("{}: decoder input of {} channels, expected {LATENT}", path.display(), conv_in.c_in);
        }
        let mut c = conv_in.c_out;
        let mut blocks = Vec::new();
        for (i, &s) in DECODER_RATES.iter().enumerate() {
            let p = format!("decoder.model.{}", i + 1);
            blocks.push(DecBlock {
                snake: Snake::load(f, &format!("{p}.block.0.alpha"), c)?,
                up: TransConv::load(f, &format!("{p}.block.1.conv"), s)?,
                res: res_units(f, &p, 2, c / 2)?,
            });
            c /= 2;
        }
        let n = DECODER_RATES.len();
        let snake = Snake::load(f, &format!("decoder.model.{}.alpha", n + 1), c)?;
        let conv_out = Conv::load(f, &format!("decoder.model.{}.conv", n + 2), 1, 1)?;
        let weight_bytes = codebooks.iter().map(|c| c.table.len() * 4).sum::<usize>()
            + post.bytes()
            + up.iter().map(|(t, n)| t.bytes() + n.bytes()).sum::<usize>()
            + conv_in.bytes()
            + conv_out.bytes()
            + blocks
                .iter()
                .map(|b| b.up.bytes() + b.res.iter().map(|r| r.bytes()).sum::<usize>())
                .sum::<usize>();
        Ok(Self {
            path: path.to_path_buf(),
            codebooks,
            post,
            up,
            conv_in,
            blocks,
            snake,
            conv_out,
            encoder: std::sync::OnceLock::new(),
            weight_bytes,
        })
    }

    pub fn n_codebooks(&self) -> usize {
        self.codebooks.len()
    }

    /// Entries of codebook `i`.
    pub fn codebook_size(&self, i: usize) -> usize {
        self.codebooks[i].size
    }

    /// The latent the decoder reads (`[4 * frames, 1024]`) of `codes` (`[frames, codebooks]`):
    /// steps 1 to 3 of the module documentation.
    fn latent(&self, codes: &[u32]) -> Result<Vec<f32>> {
        let n_cb = self.codebooks.len();
        if codes.is_empty() || !codes.len().is_multiple_of(n_cb) {
            bail!("codes: {} values are not frames of {n_cb} codes", codes.len());
        }
        let frames = codes.len() / n_cb;
        let mut z = vec![0f32; frames * LATENT];
        for_chunks(&mut z, LATENT, frames > 1, |f, row| {
            for (i, cb) in self.codebooks.iter().enumerate() {
                // Out-of-range codes clamp, as the reference's `decode` does.
                let e = (codes[f * n_cb + i] as usize).min(cb.size - 1);
                add_inplace(row, &cb.table[e * LATENT..(e + 1) * LATENT]);
            }
        });
        self.post.forward(&mut z);
        for (up, next) in &self.up {
            z = up.forward(&z);
            next.forward(&mut z);
        }
        Ok(z)
    }

    /// `codes` (`[frames, codebooks]`, row-major) to `frames * 2048` samples at 44.1 kHz.
    pub fn decode(&self, codes: &[u32]) -> Result<Vec<f32>> {
        let mut x = self.conv_in.forward(&self.latent(codes)?);
        for b in &self.blocks {
            b.snake.apply(&mut x);
            x = b.up.forward(&x);
            for r in &b.res {
                r.forward(&mut x);
            }
        }
        self.snake.apply(&mut x);
        let mut y = self.conv_out.forward(&x);
        y.iter_mut().for_each(|v| *v = v.tanh());
        Ok(y)
    }

    fn encoder(&self) -> Result<&Encoder> {
        if self.encoder.get().is_none() {
            let files = ModelFiles::open_file(&self.path, false).with_context(|| format!("open {}", self.path.display()))?;
            let f = &files;
            let conv_in = Conv::load(f, "encoder.block.0.conv", 1, 1)?;
            let mut c = conv_in.c_out;
            let mut blocks = Vec::new();
            for (i, &s) in ENCODER_RATES.iter().enumerate() {
                let p = format!("encoder.block.{}", i + 1);
                let tp = format!("{p}.block.5");
                blocks.push(EncBlock {
                    res: res_units(f, &p, 0, c)?,
                    snake: Snake::load(f, &format!("{p}.block.3.alpha"), c)?,
                    down: Conv::load(f, &format!("{p}.block.4.conv"), 1, s)?,
                    transformer: if f.has(&format!("{tp}.norm.weight")) {
                        Some(Transformer::load(f, &tp, 2 * c, 512)?)
                    } else {
                        None
                    },
                });
                c *= 2;
            }
            let n = ENCODER_RATES.len();
            let enc = Encoder {
                conv_in,
                blocks,
                snake: Snake::load(f, &format!("encoder.block.{}.alpha", n + 1), c)?,
                conv_out: Conv::load(f, &format!("encoder.block.{}.conv", n + 2), 1, 1)?,
                down: (0..QUANTIZER_RATES.len())
                    .map(|i| {
                        Ok((
                            Conv::load(f, &format!("quantizer.downsample.{i}.0.conv"), 1, QUANTIZER_RATES[i])?,
                            ConvNext::load(f, &format!("quantizer.downsample.{i}.1"), LATENT)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
                pre: Transformer::load(f, "quantizer.pre_module", LATENT, 128)?,
            };
            let _ = self.encoder.set(enc);
        }
        Ok(self.encoder.get().expect("just set"))
    }

    /// Mono samples at 44.1 kHz to codes (`[frames, codebooks]`, row-major), one frame per 2048
    /// samples (the last one padded with zeros).
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<u32>> {
        if samples.is_empty() {
            bail!("encode: no samples");
        }
        let e = self.encoder()?;
        let mut pcm = samples.to_vec();
        pcm.resize(samples.len().div_ceil(HOP) * HOP, 0.0);
        let mut x = e.conv_in.forward(&pcm);
        drop(pcm);
        for b in &e.blocks {
            for r in &b.res {
                r.forward(&mut x);
            }
            b.snake.apply(&mut x);
            x = b.down.forward(&x);
            if let Some(t) = &b.transformer {
                t.forward(&mut x);
            }
        }
        e.snake.apply(&mut x);
        let mut z = e.conv_out.forward(&x);
        for (down, next) in &e.down {
            z = down.forward(&z);
            next.forward(&mut z);
        }
        e.pre.forward(&mut z);
        let frames = z.len() / LATENT;
        let n_cb = self.codebooks.len();
        let mut codes = vec![0u32; frames * n_cb];
        // The semantic codebook reads the latent, the residual ones what it leaves.
        for (i, cb) in self.codebooks.iter().enumerate() {
            for (f, c) in cb.quantize(&mut z).into_iter().enumerate() {
                codes[f * n_cb + i] = c;
            }
        }
        Ok(codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < 1e-5, "value {i}: {x} vs {y}");
        }
    }

    /// `CausalConvNet` computed tap by tap: left padding of the kernel's reach minus the stride,
    /// zeros on the right up to a whole number of strides.
    #[test]
    fn causal_convolutions_match_the_definition() {
        let mut seed = 3;
        for (ci, co, k, d, s, t) in [
            (3, 4, 7, 1, 1, 11),
            (2, 3, 7, 9, 1, 30),
            (3, 5, 4, 1, 2, 9),
            (2, 2, 16, 1, 8, 21),
            (4, 4, 2, 1, 2, 7),
        ] {
            let w = noise(co * ci * k, &mut seed);
            let b = noise(co, &mut seed);
            let x = noise(t * ci, &mut seed);
            let conv = Conv::new(&w, &[co, ci, k], b.clone(), d, s).unwrap();
            let y = conv.forward(&x);
            let pad = (k - 1) * d + 1 - s;
            let t_out = t.div_ceil(s);
            let mut want = vec![0f32; t_out * co];
            for j in 0..t_out {
                for o in 0..co {
                    let mut acc = b[o];
                    for c in 0..ci {
                        for i in 0..k {
                            let p = j * s + i * d;
                            if p >= pad && p - pad < t {
                                acc += w[(o * ci + c) * k + i] * x[(p - pad) * ci + c];
                            }
                        }
                    }
                    want[j * co + o] = acc;
                }
            }
            close(&y, &want);
        }
    }

    /// `CausalTransConvNet`: a transposed convolution with its last `k - stride` rows dropped.
    #[test]
    fn causal_transposed_convolutions_match_the_definition() {
        let mut seed = 5;
        for (ci, co, k, s, t) in [(3, 2, 4, 2, 6), (2, 3, 16, 8, 5), (4, 4, 2, 2, 7)] {
            let w = noise(ci * co * k, &mut seed);
            let b = noise(co, &mut seed);
            let x = noise(t * ci, &mut seed);
            let y = TransConv::new(&w, &[ci, co, k], b.clone(), s).unwrap().forward(&x);
            let mut full = vec![0f32; ((t - 1) * s + k) * co];
            for j in 0..t {
                for c in 0..ci {
                    for o in 0..co {
                        for i in 0..k {
                            full[(j * s + i) * co + o] += x[j * ci + c] * w[(c * co + o) * k + i];
                        }
                    }
                }
            }
            let want: Vec<f32> = full[..t * s * co]
                .chunks(co)
                .flat_map(|r| r.iter().zip(&b).map(|(v, b)| v + b))
                .collect();
            close(&y, &want);
        }
    }
}
