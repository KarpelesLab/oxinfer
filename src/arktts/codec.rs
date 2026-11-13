//! The audio codec of Audio8 TTS (`codec.pth`, a port of `modeling_arktts_codec.py` of
//! Edge0/Audio8-TTS-Realistic-Air; the design is Fish Audio's DAC variant): ten codes per frame
//! of 2048 samples at 44.1 kHz, one semantic code of 4096 values and nine residual codes of 1024.
//!
//! Decoding (codes to audio):
//! 1. Each code's vector (8 wide) through its output projection (to 1024), summed.
//! 2. `post_module`: eight causal transformer layers at the frame rate (RMS norms, RoPE with
//!    tables rounded to bf16 as the release computes them, attention over the last 128 frames,
//!    SwiGLU, LayerScale).
//! 3. Two upsampling stages (a transposed convolution by 2 and a causal ConvNeXt block).
//! 4. The DAC decoder: a causal convolution, four blocks (Snake, a transposed convolution by 8,
//!    8, 4 and 2, three residual units with dilations 1, 3 and 9), Snake, a convolution to one
//!    channel and tanh.
//!
//! Encoding (audio to codes, for reference clips) is the mirror image: the DAC encoder (strides
//! 2, 4, 8 and 8, the last block followed by four transformer layers over 512 steps), two
//! downsampling stages, `pre_module` (eight layers), then the semantic quantizer and the nine
//! residual quantizers, each choosing the nearest codebook entry by cosine similarity.
//!
//! Every convolution is causal (padded on the left), so the decoder's convolutions are run over
//! windows of the latent with a few steps of left context, which bounds the memory of long
//! utterances. Activations are `[time, channels]`.

use crate::kernels::for_chunks;
use crate::safetensors::ModelFiles;
use crate::sd::ops::gelu_erf;
use crate::snac::sin_f32;
use crate::zonos::file::{Mat, mul_multi};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

const HEAD_DIM: usize = 64;
const ROPE_BASE: f32 = 10000.0;
const NORM_EPS: f32 = 1e-5;
/// Latent steps (four per frame) the decoder's convolutions take per window, and the steps of
/// left context before each window (their receptive field is under 20 steps).
const WINDOW_STEPS: usize = 256;
const CONTEXT_STEPS: usize = 32;
/// Frames before the first one wanted that are upsampled with it.
const CONTEXT_FRAMES: usize = 16;

/// `f(row index, row)` over the rows of `x` (`width` values each), in parallel over groups of
/// rows.
fn rows_par(x: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let group = (16384 / width.max(1)).max(1);
    let par = x.len() > 1 << 15;
    for_chunks(x, group * width, par, |g, block| {
        for (i, row) in block.chunks_mut(width).enumerate() {
            f(g * group + i, row);
        }
    });
}

/// The cos / sin pairs of `positions` positions for heads of `head_dim`, computed in f32 and
/// rounded to bf16 as the release's code does (`_rope`): `[positions, head_dim / 2, 2]`.
pub(crate) fn rope_table(positions: usize, head_dim: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let freqs: Vec<f32> = (0..half).map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32)).collect();
    let round = |v: f32| half::bf16::from_f32(v).to_f32();
    let mut out = Vec::with_capacity(positions * half * 2);
    for p in 0..positions {
        for &f in &freqs {
            let a = p as f32 * f;
            out.push(round(a.cos()));
            out.push(round(a.sin()));
        }
    }
    out
}

/// Rotates the pairs of every head of `x` (`heads` of `head_dim`) by one position's table.
pub(crate) fn apply_rope(x: &mut [f32], table: &[f32], head_dim: usize) {
    for head in x.chunks_mut(head_dim) {
        for (pair, cs) in head.as_chunks_mut::<2>().0.iter_mut().zip(table.as_chunks::<2>().0) {
            let (a, b) = (pair[0], pair[1]);
            pair[0] = a * cs[0] - b * cs[1];
            pair[1] = b * cs[0] + a * cs[1];
        }
    }
}

/// A weight tensor and its shape, with PyTorch's weight normalization folded in (`g * v / |v|`,
/// the norm over every dimension but the first), in either naming.
fn weight(f: &ModelFiles, p: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let (gn, vn) = if f.has(&format!("{p}.parametrizations.weight.original0")) {
        (
            format!("{p}.parametrizations.weight.original0"),
            format!("{p}.parametrizations.weight.original1"),
        )
    } else if f.has(&format!("{p}.weight_g")) {
        (format!("{p}.weight_g"), format!("{p}.weight_v"))
    } else {
        let n = format!("{p}.weight");
        let shape = f.info(&n).with_context(|| format!("missing tensor {n}"))?.shape.clone();
        return Ok((f.f32(&n)?, shape));
    };
    let shape = f.info(&vn).with_context(|| format!("missing tensor {vn}"))?.shape.clone();
    let g = f.f32(&gn)?;
    let mut v = f.f32(&vn)?;
    let rows = shape[0];
    if g.len() != rows || rows == 0 || v.len() % rows != 0 {
        bail!("{p}: weight norm shapes {shape:?} / {} do not match", g.len());
    }
    let n = v.len() / rows;
    for (r, row) in v.chunks_mut(n).enumerate() {
        let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let s = (g[r] as f64 / norm) as f32;
        row.iter_mut().for_each(|x| *x *= s);
    }
    Ok((v, shape))
}

fn vector(f: &ModelFiles, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = f.f32(name)?;
    if v.len() != n {
        bail!("{name}: {} values, expected {n}", v.len());
    }
    Ok(v)
}

fn matrix(f: &ModelFiles, name: &str) -> Result<Mat> {
    let shape = f.info(name).with_context(|| format!("missing tensor {name}"))?.shape.clone();
    let [o, i] = shape[..] else {
        bail!("{name}: expected a matrix, got {shape:?}")
    };
    Mat::from_f32(&f.f32(name)?, o, i)
}

/// Activation buffers reused within one thread, so that an utterance's hundreds of large
/// buffers do not each come fresh from the system.
mod bufs {
    use std::cell::RefCell;

    thread_local!(static FREE: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) });

    /// A buffer of `n` values whose contents are unspecified.
    pub fn take(n: usize) -> Vec<f32> {
        let mut v = FREE.with(|f| {
            let mut f = f.borrow_mut();
            // The smallest one that is large enough, else the largest.
            let fits = f
                .iter()
                .enumerate()
                .filter(|(_, b)| b.capacity() >= n)
                .min_by_key(|(_, b)| b.capacity());
            let pick = fits
                .or_else(|| f.iter().enumerate().max_by_key(|(_, b)| b.capacity()))
                .map(|(i, _)| i);
            pick.map(|i| f.swap_remove(i)).unwrap_or_default()
        });
        if v.len() >= n {
            v.truncate(n);
        } else {
            v.clear();
            v.resize(n, 0.0);
        }
        v
    }

    pub fn give(v: Vec<f32>) {
        FREE.with(|f| {
            let mut f = f.borrow_mut();
            if f.len() < 8 {
                f.push(v);
            }
        });
    }

    pub fn clear() {
        FREE.with(|f| f.borrow_mut().clear());
    }
}

/// Time steps a convolution computes per task.
const TILE: usize = 32;

/// `y[r][o] = sum_i w[o][i] * xs[r * step + i]` for `nt` windows of `k` values that start `step`
/// apart in `xs`: eight weight rows at a time against one window, in eight lanes.
fn gemm_portable(w: &[f32], co: usize, k: usize, xs: &[f32], step: usize, nt: usize, y: &mut [f32]) {
    let k8 = k / 8 * 8;
    let sum = |a: &[f32; 8]| ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
    let mut o = 0;
    while o + 8 <= co {
        let wr = &w[o * k..(o + 8) * k];
        for r in 0..nt {
            let x = &xs[r * step..r * step + k];
            let mut acc = [[0f32; 8]; 8];
            let mut i = 0;
            while i < k8 {
                let xv: &[f32; 8] = x[i..i + 8].try_into().unwrap();
                for (q, a) in acc.iter_mut().enumerate() {
                    let wv: &[f32; 8] = wr[q * k + i..q * k + i + 8].try_into().unwrap();
                    for l in 0..8 {
                        a[l] += wv[l] * xv[l];
                    }
                }
                i += 8;
            }
            for (q, a) in acc.iter().enumerate() {
                let mut s = sum(a);
                for i in k8..k {
                    s += wr[q * k + i] * x[i];
                }
                y[r * co + o + q] = s;
            }
        }
        o += 8;
    }
    while o < co {
        let wr = &w[o * k..(o + 1) * k];
        for r in 0..nt {
            let x = &xs[r * step..r * step + k];
            let mut a = [0f32; 8];
            let mut i = 0;
            while i < k8 {
                let xv: &[f32; 8] = x[i..i + 8].try_into().unwrap();
                let wv: &[f32; 8] = wr[i..i + 8].try_into().unwrap();
                for l in 0..8 {
                    a[l] += wv[l] * xv[l];
                }
                i += 8;
            }
            let mut s = sum(&a);
            for i in k8..k {
                s += wr[i] * x[i];
            }
            y[r * co + o] = s;
        }
        o += 1;
    }
}

/// [`gemm_portable`] with AVX2 and FMA: the eight sums of a group of rows stay in registers.
///
/// # Safety
/// The CPU must have AVX2 and FMA; the slices must hold `co * k` weights, `(nt - 1) * step + k`
/// inputs and `nt * co` outputs (checked by [`gemm`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_avx2(w: &[f32], co: usize, k: usize, xs: &[f32], step: usize, nt: usize, y: &mut [f32]) {
    use std::arch::x86_64::*;
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
    }
    unsafe {
        let k8 = k / 8 * 8;
        let (wp, xp, yp) = (w.as_ptr(), xs.as_ptr(), y.as_mut_ptr());
        let mut o = 0;
        while o + 8 <= co {
            let wr = wp.add(o * k);
            for r in 0..nt {
                let x = xp.add(r * step);
                let mut acc = [_mm256_setzero_ps(); 8];
                let mut i = 0;
                while i < k8 {
                    let xv = _mm256_loadu_ps(x.add(i));
                    for (q, a) in acc.iter_mut().enumerate() {
                        *a = _mm256_fmadd_ps(_mm256_loadu_ps(wr.add(q * k + i)), xv, *a);
                    }
                    i += 8;
                }
                for (q, a) in acc.iter().enumerate() {
                    let mut s = hsum(*a);
                    for i in k8..k {
                        s += *wr.add(q * k + i) * *x.add(i);
                    }
                    *yp.add(r * co + o + q) = s;
                }
            }
            o += 8;
        }
        while o < co {
            let wr = wp.add(o * k);
            for r in 0..nt {
                let x = xp.add(r * step);
                let (mut a, mut b) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                let mut i = 0;
                while i + 16 <= k8 {
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(wr.add(i)), _mm256_loadu_ps(x.add(i)), a);
                    b = _mm256_fmadd_ps(_mm256_loadu_ps(wr.add(i + 8)), _mm256_loadu_ps(x.add(i + 8)), b);
                    i += 16;
                }
                let mut s = hsum(_mm256_add_ps(a, b));
                for i in i..k {
                    s += *wr.add(i) * *x.add(i);
                }
                *yp.add(r * co + o) = s;
            }
            o += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn gemm(w: &[f32], co: usize, k: usize, xs: &[f32], step: usize, nt: usize, y: &mut [f32]) {
    if nt == 0 {
        return;
    }
    assert!(w.len() >= co * k && xs.len() >= (nt - 1) * step + k && y.len() >= nt * co);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: the features were just detected and the lengths checked.
        return unsafe { gemm_avx2(w, co, k, xs, step, nt, y) };
    }
    gemm_portable(w, co, k, xs, step, nt, y)
}

/// A causal convolution (`ArkttsCausalConv1d`): `kernel - stride` steps of left padding (in
/// dilated taps), the input padded on the right to whole strides.
struct Conv {
    /// `[c_out, k * c_in]`, tap by tap.
    w: Vec<f32>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    dilation: usize,
}

impl Conv {
    fn load(f: &ModelFiles, p: &str, stride: usize, dilation: usize) -> Result<Self> {
        let (w, shape) = weight(f, p)?;
        let [co, ci, k] = shape[..] else {
            bail!("{p}: expected a kernel [out, in, k], got {shape:?}")
        };
        if (k - 1) * dilation + 1 < stride {
            bail!("{p}: kernel {k} shorter than its stride {stride}");
        }
        Ok(Self::new(&w, [co, ci, k], vector(f, &format!("{p}.bias"), co)?, stride, dilation))
    }

    /// From a kernel in PyTorch's layout (`[out, in, k]`).
    fn new(w: &[f32], [co, ci, k]: [usize; 3], b: Vec<f32>, stride: usize, dilation: usize) -> Self {
        let mut m = vec![0f32; co * k * ci];
        for o in 0..co {
            for i in 0..ci {
                for j in 0..k {
                    m[(o * k + j) * ci + i] = w[(o * ci + i) * k + j];
                }
            }
        }
        Self {
            w: m,
            b,
            c_in: ci,
            c_out: co,
            k,
            stride,
            dilation,
        }
    }

    /// `x` (`[t, c_in]`) to `[ceil(t / stride), c_out]`.
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        let (ci, co, k, stride, dil) = (self.c_in, self.c_out, self.k, self.stride, self.dilation);
        let t_out = t.div_ceil(stride);
        let kk = k * ci;
        let mut y = bufs::take(t_out * co);
        // The input with its padding, so that every window is a run of rows (taps `dil` apart).
        let pad = (k - 1) * dil + 1 - stride;
        let mut padded = Vec::new();
        let xp: &[f32] = if k == 1 && stride == 1 {
            x
        } else {
            let len = (t_out - 1) * stride + (k - 1) * dil + 1;
            padded = bufs::take(len * ci);
            padded[..pad * ci].fill(0.0);
            padded[pad * ci..(pad + t) * ci].copy_from_slice(x);
            padded[(pad + t) * ci..].fill(0.0);
            &padded
        };
        for_chunks(&mut y, TILE * co, t_out > TILE, |i, yt| {
            let (a, nt) = (i * TILE, yt.len() / co);
            if dil == 1 {
                gemm(&self.w, co, kk, &xp[a * stride * ci..], stride * ci, nt, yt);
            } else {
                let mut col = vec![0f32; nt * kk];
                for (r, row) in col.chunks_mut(kk).enumerate() {
                    for (j, d) in row.chunks_mut(ci).enumerate() {
                        let src = (a + r) * stride + j * dil;
                        d.copy_from_slice(&xp[src * ci..(src + 1) * ci]);
                    }
                }
                gemm(&self.w, co, kk, &col, kk, nt, yt);
            }
            for row in yt.chunks_mut(co) {
                for (v, &b) in row.iter_mut().zip(&self.b) {
                    *v += b;
                }
            }
        });
        bufs::give(padded);
        (y, t_out)
    }
}

/// A causal transposed convolution (`ArkttsCausalConvTranspose1d`) whose kernel is `taps`
/// strides long: output step `t * stride + r` reads the inputs `t - taps + 1 ..= t`.
struct ConvT {
    /// `[stride * c_out, taps * c_in]`, the oldest input first.
    w: Vec<f32>,
    /// The bias, once per phase.
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    stride: usize,
    taps: usize,
}

impl ConvT {
    fn load(f: &ModelFiles, p: &str, stride: usize) -> Result<Self> {
        let (w, shape) = weight(f, p)?;
        let [ci, co, k] = shape[..] else {
            bail!("{p}: expected a kernel [in, out, k], got {shape:?}")
        };
        if k % stride != 0 {
            bail!("{p}: kernel {k} is not a multiple of its stride {stride}");
        }
        Ok(Self::new(&w, [ci, co, k], &vector(f, &format!("{p}.bias"), co)?, stride))
    }

    /// From a kernel in PyTorch's layout (`[in, out, k]`, `k` a multiple of `stride`).
    fn new(w: &[f32], [ci, co, k]: [usize; 3], b: &[f32], stride: usize) -> Self {
        let taps = k / stride;
        let mut m = vec![0f32; stride * co * taps * ci];
        for r in 0..stride {
            for o in 0..co {
                for q in 0..taps {
                    // Input `t - q` meets tap `r + q * stride`.
                    let block = taps - 1 - q;
                    for i in 0..ci {
                        m[((r * co + o) * taps + block) * ci + i] = w[(i * co + o) * k + r + q * stride];
                    }
                }
            }
        }
        Self {
            w: m,
            b: (0..stride).flat_map(|_| b.iter().copied()).collect(),
            c_in: ci,
            c_out: co,
            stride,
            taps,
        }
    }

    /// `x` (`[t, c_in]`) to `[t * stride, c_out]`.
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        let (ci, n) = (self.c_in, self.stride * self.c_out);
        let mut y = bufs::take(t * n);
        let mut padded = Vec::new();
        let xp: &[f32] = if self.taps == 1 {
            x
        } else {
            padded = bufs::take((t + self.taps - 1) * ci);
            padded[..(self.taps - 1) * ci].fill(0.0);
            padded[(self.taps - 1) * ci..].copy_from_slice(x);
            &padded
        };
        for_chunks(&mut y, TILE * n, t > TILE, |i, yt| {
            gemm(&self.w, n, self.taps * ci, &xp[i * TILE * ci..], ci, yt.len() / n, yt);
            for row in yt.chunks_mut(n) {
                for (v, &b) in row.iter_mut().zip(&self.b) {
                    *v += b;
                }
            }
        });
        bufs::give(padded);
        (y, t * self.stride)
    }
}

/// Snake: `x + sin(alpha x)^2 / (alpha + 1e-9)`, per channel.
struct Snake {
    alpha: Vec<f32>,
    inv: Vec<f32>,
}

#[inline(always)]
fn snake_body(rows: &mut [f32], alpha: &[f32], inv: &[f32]) {
    for row in rows.chunks_mut(alpha.len()) {
        for ((v, &a), &inv) in row.iter_mut().zip(alpha).zip(inv) {
            let s = sin_f32(a * *v);
            *v += inv * s * s;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn snake_avx2(rows: &mut [f32], alpha: &[f32], inv: &[f32]) {
    snake_body(rows, alpha, inv)
}

impl Snake {
    fn load(f: &ModelFiles, name: &str) -> Result<Self> {
        let alpha = f.f32(name)?;
        let inv = alpha.iter().map(|&a| 1.0 / (a + 1e-9)).collect();
        Ok(Self { alpha, inv })
    }

    fn apply(&self, x: &mut [f32]) {
        let c = self.alpha.len();
        let group = (16384 / c).max(1) * c;
        for_chunks(x, group, x.len() > 1 << 15, |_, rows| {
            #[cfg(target_arch = "x86_64")]
            if has_avx2() {
                // SAFETY: the features were just detected.
                return unsafe { snake_avx2(rows, &self.alpha, &self.inv) };
            }
            snake_body(rows, &self.alpha, &self.inv)
        });
    }
}

/// `x + conv1(snake(conv7(snake(x))))`.
struct ResUnit {
    s1: Snake,
    c1: Conv,
    s2: Snake,
    c2: Conv,
}

impl ResUnit {
    fn load(f: &ModelFiles, p: &str, dilation: usize) -> Result<Self> {
        Ok(Self {
            s1: Snake::load(f, &format!("{p}.block.0.alpha"))?,
            c1: Conv::load(f, &format!("{p}.block.1.conv"), 1, dilation)?,
            s2: Snake::load(f, &format!("{p}.block.2.alpha"))?,
            c2: Conv::load(f, &format!("{p}.block.3.conv"), 1, 1)?,
        })
    }

    fn forward(&self, x: &mut [f32], t: usize) {
        let mut a = bufs::take(x.len());
        a.copy_from_slice(x);
        self.s1.apply(&mut a);
        let (mut h, _) = self.c1.forward(&a, t);
        bufs::give(a);
        self.s2.apply(&mut h);
        let (o, _) = self.c2.forward(&h, t);
        bufs::give(h);
        let c = self.c2.c_out;
        rows_par(x, c, |r, row| {
            for (v, &o) in row.iter_mut().zip(&o[r * c..(r + 1) * c]) {
                *v += o;
            }
        });
        bufs::give(o);
    }
}

/// A causal ConvNeXt block: a depthwise convolution over the last 7 steps, LayerNorm, a linear
/// layer to four times the width, GELU, a linear layer back, a scale per channel.
struct ConvNext {
    c: usize,
    /// `[c, 7]`, the last tap on the current step.
    dw: Vec<f32>,
    dw_b: Vec<f32>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    pw1: Mat,
    pw1_b: Vec<f32>,
    pw2: Mat,
    pw2_b: Vec<f32>,
    gamma: Vec<f32>,
}

impl ConvNext {
    fn load(f: &ModelFiles, p: &str) -> Result<Self> {
        let name = format!("{p}.dwconv.conv.weight");
        let shape = f.info(&name).with_context(|| format!("missing tensor {name}"))?.shape.clone();
        let [c, 1, 7] = shape[..] else {
            bail!("{name}: expected a depthwise kernel [c, 1, 7], got {shape:?}")
        };
        Ok(Self {
            c,
            dw: f.f32(&name)?,
            dw_b: vector(f, &format!("{p}.dwconv.conv.bias"), c)?,
            norm_w: vector(f, &format!("{p}.norm.weight"), c)?,
            norm_b: vector(f, &format!("{p}.norm.bias"), c)?,
            pw1: matrix(f, &format!("{p}.pwconv1.weight"))?,
            pw1_b: vector(f, &format!("{p}.pwconv1.bias"), 4 * c)?,
            pw2: matrix(f, &format!("{p}.pwconv2.weight"))?,
            pw2_b: vector(f, &format!("{p}.pwconv2.bias"), c)?,
            gamma: vector(f, &format!("{p}.gamma"), c)?,
        })
    }

    fn forward(&self, x: &mut [f32], t: usize) {
        let c = self.c;
        let mut h = vec![0f32; t * c];
        {
            let x = &*x;
            rows_par(&mut h, c, |r, row| {
                row.copy_from_slice(&self.dw_b);
                for j in 0..7 {
                    if r + j < 6 {
                        continue;
                    }
                    let src = &x[(r + j - 6) * c..(r + j - 5) * c];
                    for (ch, (o, &v)) in row.iter_mut().zip(src).enumerate() {
                        *o += v * self.dw[ch * 7 + j];
                    }
                }
                // LayerNorm (eps 1e-6).
                let mean = row.iter().sum::<f32>() / c as f32;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
                let s = 1.0 / (var + 1e-6).sqrt();
                for ((v, w), b) in row.iter_mut().zip(&self.norm_w).zip(&self.norm_b) {
                    *v = (*v - mean) * s * w + b;
                }
            });
        }
        let mut a = self.pw1.mul(&h, t, true);
        let b1 = &self.pw1_b;
        rows_par(&mut a, 4 * c, |_, row| {
            for (v, &b) in row.iter_mut().zip(b1) {
                *v = gelu_erf(*v + b);
            }
        });
        let o = self.pw2.mul(&a, t, true);
        rows_par(x, c, |r, row| {
            let o = &o[r * c..(r + 1) * c];
            for (i, v) in row.iter_mut().enumerate() {
                *v += self.gamma[i] * (o[i] + self.pw2_b[i]);
            }
        });
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    wqkv: Mat,
    wo: Mat,
    attn_scale: Vec<f32>,
    ffn_norm: Vec<f32>,
    w1: Mat,
    w3: Mat,
    w2: Mat,
    ffn_scale: Vec<f32>,
}

/// `ArkttsCodecWindowTransformer`: causal layers whose attention reads the last `window` steps.
struct Transformer {
    dim: usize,
    heads: usize,
    kv_heads: usize,
    window: usize,
    layers: Vec<Layer>,
    norm: Vec<f32>,
}

fn rms_rows(x: &[f32], n: usize, w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    rows_par(&mut out, n, |r, o| {
        let xr = &x[r * n..(r + 1) * n];
        let ss = xr.iter().map(|v| v * v).sum::<f32>() / n as f32 + NORM_EPS;
        let s = 1.0 / ss.sqrt();
        for ((o, &v), &w) in o.iter_mut().zip(xr).zip(w) {
            *o = v * s * w;
        }
    });
    out
}

impl Transformer {
    fn load(f: &ModelFiles, p: &str, window: usize) -> Result<Self> {
        let norm = f.f32(&format!("{p}.norm.weight"))?;
        let dim = norm.len();
        let heads = dim / HEAD_DIM;
        let mut layers = Vec::new();
        let mut kv_heads = 0;
        while f.has(&format!("{p}.layers.{}.attention.wqkv.weight", layers.len())) {
            let l = format!("{p}.layers.{}", layers.len());
            let wqkv = matrix(f, &format!("{l}.attention.wqkv.weight"))?;
            let kv = (wqkv.n_out / HEAD_DIM).saturating_sub(heads) / 2;
            if wqkv.n_in != dim || kv == 0 || (heads + 2 * kv) * HEAD_DIM != wqkv.n_out || !heads.is_multiple_of(kv) {
                bail!("{l}: attention of {} rows over width {dim}", wqkv.n_out);
            }
            kv_heads = kv;
            layers.push(Layer {
                attn_norm: vector(f, &format!("{l}.attention_norm.weight"), dim)?,
                wqkv,
                wo: matrix(f, &format!("{l}.attention.wo.weight"))?,
                attn_scale: vector(f, &format!("{l}.attention_layer_scale.gamma"), dim)?,
                ffn_norm: vector(f, &format!("{l}.ffn_norm.weight"), dim)?,
                w1: matrix(f, &format!("{l}.feed_forward.w1.weight"))?,
                w3: matrix(f, &format!("{l}.feed_forward.w3.weight"))?,
                w2: matrix(f, &format!("{l}.feed_forward.w2.weight"))?,
                ffn_scale: vector(f, &format!("{l}.ffn_layer_scale.gamma"), dim)?,
            });
        }
        if layers.is_empty() {
            bail!("{p}: no transformer layers");
        }
        Ok(Self {
            dim,
            heads,
            kv_heads,
            window,
            layers,
            norm,
        })
    }

    /// `x` (`[t, dim]`) through the layers and the final norm.
    fn forward(&self, x: &mut Vec<f32>, t: usize) {
        let (d, nh, nk) = (self.dim, self.heads, self.kv_heads);
        let (nq, nkv) = (nh * HEAD_DIM, nk * HEAD_DIM);
        let tot = nq + 2 * nkv;
        let rope = rope_table(t, HEAD_DIM, ROPE_BASE);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        for l in &self.layers {
            let h = rms_rows(x, d, &l.attn_norm);
            let mut qkv = l.wqkv.mul(&h, t, true);
            rows_par(&mut qkv, tot, |r, row| {
                let table = &rope[r * HEAD_DIM..(r + 1) * HEAD_DIM];
                apply_rope(&mut row[..nq + nkv], table, HEAD_DIM);
            });
            let mut att = vec![0f32; t * nq];
            {
                let qkv = &qkv;
                rows_par(&mut att, nq, |r, out| {
                    let lo = (r + 1).saturating_sub(self.window);
                    let mut p = vec![0f32; r + 1 - lo];
                    for (hq, o) in out.chunks_mut(HEAD_DIM).enumerate() {
                        let hk = hq / (nh / nk);
                        let q = &qkv[r * tot + hq * HEAD_DIM..r * tot + (hq + 1) * HEAD_DIM];
                        let mut max = f32::NEG_INFINITY;
                        for (s, pv) in (lo..=r).zip(p.iter_mut()) {
                            let k = &qkv[s * tot + nq + hk * HEAD_DIM..s * tot + nq + (hk + 1) * HEAD_DIM];
                            *pv = crate::kernels::dot(q, k) * scale;
                            max = max.max(*pv);
                        }
                        let mut sum = 0f32;
                        for pv in p.iter_mut() {
                            *pv = (*pv - max).exp();
                            sum += *pv;
                        }
                        for (s, &pv) in (lo..=r).zip(p.iter()) {
                            let v = &qkv[s * tot + nq + nkv + hk * HEAD_DIM..s * tot + nq + nkv + (hk + 1) * HEAD_DIM];
                            crate::kernels::axpy(o, pv / sum, v);
                        }
                    }
                });
            }
            let o = l.wo.mul(&att, t, true);
            rows_par(x, d, |r, row| {
                for (i, v) in row.iter_mut().enumerate() {
                    *v += l.attn_scale[i] * o[r * d + i];
                }
            });
            let h = rms_rows(x, d, &l.ffn_norm);
            let n = l.w1.n_out;
            let (mut a, mut g) = (vec![0f32; t * n], vec![0f32; t * n]);
            mul_multi(&mut [(&l.w1, &mut a[..]), (&l.w3, &mut g[..])], &h, t, true);
            {
                let g = &g;
                rows_par(&mut a, n, |r, row| {
                    for (v, &gv) in row.iter_mut().zip(&g[r * n..(r + 1) * n]) {
                        *v = *v / (1.0 + (-*v).exp()) * gv;
                    }
                });
            }
            let o = l.w2.mul(&a, t, true);
            rows_par(x, d, |r, row| {
                for (i, v) in row.iter_mut().enumerate() {
                    *v += l.ffn_scale[i] * o[r * d + i];
                }
            });
        }
        *x = rms_rows(x, d, &self.norm);
    }
}

/// One quantizer: its codebook and both projections.
struct Vq {
    n: usize,
    /// `[8, 1024]`.
    in_w: Vec<f32>,
    in_b: Vec<f32>,
    /// `[1024, 8]`.
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    /// `[n, 8]`.
    codebook: Vec<f32>,
    /// The codebook's rows at unit length.
    unit: Vec<f32>,
    /// `[n, 1024]`: every entry through the output projection.
    table: Vec<f32>,
}

const CODE_DIM: usize = 8;

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

impl Vq {
    fn load(f: &ModelFiles, p: &str, latent: usize, encoder: bool) -> Result<Self> {
        let codebook = f.f32(&format!("{p}.codebook.weight"))?;
        let n = codebook.len() / CODE_DIM;
        let (out_w, shape) = weight(f, &format!("{p}.out_proj"))?;
        if shape[..] != [latent, CODE_DIM, 1] || n == 0 {
            bail!("{p}.out_proj: shape {shape:?}");
        }
        let out_b = vector(f, &format!("{p}.out_proj.bias"), latent)?;
        let mut table = vec![0f32; n * latent];
        rows_par(&mut table, latent, |r, row| {
            let e = &codebook[r * CODE_DIM..(r + 1) * CODE_DIM];
            for (o, v) in row.iter_mut().enumerate() {
                *v = out_b[o] + crate::kernels::dot(&out_w[o * CODE_DIM..(o + 1) * CODE_DIM], e);
            }
        });
        let (in_w, in_b, unit) = if encoder {
            let (w, shape) = weight(f, &format!("{p}.in_proj"))?;
            if shape[..] != [CODE_DIM, latent, 1] {
                bail!("{p}.in_proj: shape {shape:?}");
            }
            let mut unit = codebook.clone();
            unit.chunks_mut(CODE_DIM).for_each(normalize);
            (w, vector(f, &format!("{p}.in_proj.bias"), CODE_DIM)?, unit)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        Ok(Self {
            n,
            in_w,
            in_b,
            out_w,
            out_b,
            codebook,
            unit,
            table,
        })
    }

    /// Quantizes the rows of `z` (`[t, latent]`): the chosen codes, and `residual -= ` what the
    /// codes stand for (`ArkttsVectorQuantizer.forward`).
    fn quantize(&self, z: &mut [f32], latent: usize) -> Vec<u32> {
        let t = z.len() / latent;
        let mut codes = vec![0f32; t];
        {
            let z = &*z;
            rows_par(&mut codes, 1, |r, c| {
                c[0] = self.nearest(&self.project(&z[r * latent..(r + 1) * latent])) as f32
            });
        }
        rows_par(z, latent, |r, row| {
            let p = self.project(row);
            let e = &self.codebook[codes[r] as usize * CODE_DIM..][..CODE_DIM];
            // The straight-through value, in the reference's arithmetic.
            let mut q = [0f32; CODE_DIM];
            for i in 0..CODE_DIM {
                q[i] = p[i] + (e[i] - p[i]);
            }
            for (o, v) in row.iter_mut().enumerate() {
                *v -= self.out_b[o] + crate::kernels::dot(&self.out_w[o * CODE_DIM..(o + 1) * CODE_DIM], &q);
            }
        });
        codes.iter().map(|&c| c as u32).collect()
    }

    fn project(&self, row: &[f32]) -> [f32; CODE_DIM] {
        let mut p = [0f32; CODE_DIM];
        let n = row.len();
        for (i, v) in p.iter_mut().enumerate() {
            *v = self.in_b[i] + crate::kernels::dot(&self.in_w[i * n..(i + 1) * n], row);
        }
        p
    }

    /// The entry nearest to `p` once both are at unit length (the first of equals).
    fn nearest(&self, p: &[f32; CODE_DIM]) -> usize {
        let mut a = *p;
        normalize(&mut a);
        let aa: f32 = a.iter().map(|x| x * x).sum();
        let (mut best, mut best_d) = (0, f32::INFINITY);
        for (i, b) in self.unit.chunks(CODE_DIM).enumerate() {
            let bb: f32 = b.iter().map(|x| x * x).sum();
            let ab: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let d = aa - 2.0 * ab + bb;
            if d < best_d {
                (best, best_d) = (i, d);
            }
        }
        best
    }
}

struct DecoderBlock {
    snake: Snake,
    up: ConvT,
    res: Vec<ResUnit>,
}

struct EncoderBlock {
    res: Vec<ResUnit>,
    snake: Snake,
    down: Conv,
    transformer: Option<Transformer>,
}

/// The audio-to-codes half, loaded when a reference clip is first encoded.
struct Encoder {
    conv_in: Conv,
    blocks: Vec<EncoderBlock>,
    snake: Snake,
    conv_out: Conv,
    down: Vec<(Conv, ConvNext)>,
    pre: Transformer,
    quantizers: Vec<Vq>,
}

const STRIDES: [usize; 4] = [2, 4, 8, 8];
const DILATIONS: [usize; 3] = [1, 3, 9];

impl Encoder {
    fn load(f: &ModelFiles, latent: usize, n_codebooks: usize) -> Result<Self> {
        let mut blocks = Vec::new();
        for (i, &stride) in STRIDES.iter().enumerate() {
            let p = format!("encoder.block.{}.block", i + 1);
            let res = DILATIONS
                .iter()
                .enumerate()
                .map(|(j, &d)| ResUnit::load(f, &format!("{p}.{j}"), d))
                .collect::<Result<_>>()?;
            let tp = format!("{p}.5");
            blocks.push(EncoderBlock {
                res,
                snake: Snake::load(f, &format!("{p}.3.alpha"))?,
                down: Conv::load(f, &format!("{p}.4.conv"), stride, 1)?,
                transformer: if f.has(&format!("{tp}.norm.weight")) {
                    Some(Transformer::load(f, &tp, 512)?)
                } else {
                    None
                },
            });
        }
        let mut down = Vec::new();
        for i in 0..2 {
            let p = format!("quantizer.downsample.{i}");
            down.push((Conv::load(f, &format!("{p}.0.conv"), 2, 1)?, ConvNext::load(f, &format!("{p}.1"))?));
        }
        let mut quantizers = vec![Vq::load(f, "quantizer.semantic_quantizer.quantizers.0", latent, true)?];
        for i in 0..n_codebooks - 1 {
            quantizers.push(Vq::load(f, &format!("quantizer.quantizer.quantizers.{i}"), latent, true)?);
        }
        Ok(Self {
            conv_in: Conv::load(f, "encoder.block.0.conv", 1, 1)?,
            blocks,
            snake: Snake::load(f, "encoder.block.5.alpha")?,
            conv_out: Conv::load(f, "encoder.block.6.conv", 1, 1)?,
            down,
            pre: Transformer::load(f, "quantizer.pre_module", 128)?,
            quantizers,
        })
    }

    /// Whole frames of audio to `[frames, n_codebooks]`.
    fn encode(&self, audio: &[f32], latent: usize) -> Vec<u32> {
        let (mut x, mut t) = self.conv_in.forward(audio, audio.len());
        for b in &self.blocks {
            for r in &b.res {
                r.forward(&mut x, t);
            }
            b.snake.apply(&mut x);
            (x, t) = b.down.forward(&x, t);
            if let Some(tr) = &b.transformer {
                tr.forward(&mut x, t);
            }
        }
        self.snake.apply(&mut x);
        (x, t) = self.conv_out.forward(&x, t);
        for (conv, next) in &self.down {
            (x, t) = conv.forward(&x, t);
            next.forward(&mut x, t);
        }
        self.pre.forward(&mut x, t);
        // The semantic code on the latent itself, the others on what the earlier ones left.
        let n = self.quantizers.len();
        let mut codes = vec![0u32; t * n];
        for (j, q) in self.quantizers.iter().enumerate() {
            for (r, c) in q.quantize(&mut x, latent).into_iter().enumerate() {
                codes[r * n + j] = c;
            }
        }
        codes
    }
}

pub struct Codec {
    pub sample_rate: u32,
    /// Samples per frame.
    pub frame_size: usize,
    pub n_codebooks: usize,
    /// Entries of the semantic codebook and of each residual codebook.
    pub semantic_size: usize,
    pub residual_size: usize,
    pub weight_bytes: usize,
    path: PathBuf,
    latent: usize,
    /// The codes' vectors through their output projections.
    tables: Vec<Vq>,
    post: Transformer,
    up: Vec<(ConvT, ConvNext)>,
    conv_in: Conv,
    blocks: Vec<DecoderBlock>,
    snake: Snake,
    conv_out: Conv,
    encoder: std::sync::OnceLock<Encoder>,
}

impl Codec {
    /// Loads the decoder from `codec.pth`; the encoder is read when [`Codec::encode`] first needs
    /// it.
    pub fn load(path: &Path, n_codebooks: usize, sample_rate: u32, frame_size: usize) -> Result<Self> {
        let f = ModelFiles::open_file(path, false)?;
        if !f.has("decoder.model.0.conv.bias") {
            bail!("{}: not the codec of an Audio8 TTS model (no decoder.model.0.conv)", path.display());
        }
        let conv_in = Conv::load(&f, "decoder.model.0.conv", 1, 1)?;
        let latent = conv_in.c_in;
        let mut tables = vec![Vq::load(&f, "quantizer.semantic_quantizer.quantizers.0", latent, false)?];
        for i in 0..n_codebooks.saturating_sub(1) {
            tables.push(Vq::load(&f, &format!("quantizer.quantizer.quantizers.{i}"), latent, false)?);
        }
        let mut up = Vec::new();
        for i in 0..2 {
            let p = format!("quantizer.upsample.{i}");
            up.push((ConvT::load(&f, &format!("{p}.0.conv"), 2)?, ConvNext::load(&f, &format!("{p}.1"))?));
        }
        let mut blocks = Vec::new();
        for (i, &stride) in STRIDES.iter().rev().enumerate() {
            let p = format!("decoder.model.{}.block", i + 1);
            blocks.push(DecoderBlock {
                snake: Snake::load(&f, &format!("{p}.0.alpha"))?,
                up: ConvT::load(&f, &format!("{p}.1.conv"), stride)?,
                res: DILATIONS
                    .iter()
                    .enumerate()
                    .map(|(j, &d)| ResUnit::load(&f, &format!("{p}.{}", j + 2), d))
                    .collect::<Result<_>>()?,
            });
        }
        let hop: usize = 4 * STRIDES.iter().product::<usize>();
        if hop != frame_size {
            bail!("{}: frames of {hop} samples, the model's config says {frame_size}", path.display());
        }
        let post = Transformer::load(&f, "quantizer.post_module", 128)?;
        let snake = Snake::load(&f, "decoder.model.5.alpha")?;
        let conv_out = Conv::load(&f, "decoder.model.6.conv", 1, 1)?;
        let weight_bytes = f
            .names()
            .filter(|n| n.starts_with("decoder.") || n.starts_with("quantizer.post_module") || n.starts_with("quantizer.upsample"))
            .filter_map(|n| f.info(n))
            .map(|i| i.end - i.start)
            .sum::<usize>()
            + tables.iter().map(|t| t.table.len() * 4).sum::<usize>();
        Ok(Self {
            sample_rate,
            frame_size,
            n_codebooks,
            semantic_size: tables[0].n,
            residual_size: tables.last().map_or(0, |t| t.n),
            weight_bytes,
            path: path.to_path_buf(),
            latent,
            tables,
            post,
            up,
            conv_in,
            blocks,
            snake,
            conv_out,
            encoder: std::sync::OnceLock::new(),
        })
    }

    /// The codes' vectors through `post_module`: `[frames, latent]`.
    fn post(&self, codes: &[u32], frames: usize) -> Vec<f32> {
        let (n, d) = (self.n_codebooks, self.latent);
        let mut z = vec![0f32; frames * d];
        rows_par(&mut z, d, |r, row| {
            // The residual codes summed in order, then added to the semantic code's vector; codes
            // beyond a codebook are clamped into it, as the reference does.
            for j in 1..n {
                let t = &self.tables[j];
                let c = (codes[r * n + j] as usize).min(t.n - 1);
                for (v, &e) in row.iter_mut().zip(&t.table[c * d..(c + 1) * d]) {
                    *v += e;
                }
            }
            let t = &self.tables[0];
            let c = (codes[r * n] as usize).min(t.n - 1);
            for (v, &e) in row.iter_mut().zip(&t.table[c * d..(c + 1) * d]) {
                *v += e;
            }
        });
        self.post.forward(&mut z, frames);
        z
    }

    /// `frames` rows of [`Codec::post`] to the latent the convolutional decoder reads, four steps
    /// per frame.
    fn upsample(&self, z: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let (mut z, mut t) = (z.to_vec(), frames);
        for (conv, next) in &self.up {
            (z, t) = conv.forward(&z, t);
            next.forward(&mut z, t);
        }
        (z, t)
    }

    /// The convolutional decoder on `t` latent steps.
    fn wave(&self, z: &[f32], t: usize) -> Vec<f32> {
        let (mut x, mut t) = self.conv_in.forward(z, t);
        for b in &self.blocks {
            b.snake.apply(&mut x);
            let (up, n) = b.up.forward(&x, t);
            bufs::give(std::mem::replace(&mut x, up));
            t = n;
            for r in &b.res {
                r.forward(&mut x, t);
            }
        }
        self.snake.apply(&mut x);
        let (mut y, _) = self.conv_out.forward(&x, t);
        bufs::give(x);
        y.iter_mut().for_each(|v| *v = v.tanh());
        y
    }

    /// Frames of codes (`[frames, n_codebooks]`) to audio, `frame_size` samples per frame.
    pub fn decode(&self, codes: &[u32], frames: usize) -> Result<Vec<f32>> {
        self.decode_windows(codes, frames, 0, WINDOW_STEPS)
    }

    /// The samples of the frames `from..frames` alone (for audio decoded while it is generated):
    /// the transformer runs over every frame, the rest over the new ones and a few before them.
    pub fn decode_tail(&self, codes: &[u32], frames: usize, from: usize) -> Result<Vec<f32>> {
        self.decode_windows(codes, frames, from, WINDOW_STEPS)
    }

    /// [`Codec::decode_tail`] with the convolutions over windows of `window` latent steps (0:
    /// everything at once, without cutting any context).
    pub fn decode_windows(&self, codes: &[u32], frames: usize, from: usize, window: usize) -> Result<Vec<f32>> {
        if codes.len() != frames * self.n_codebooks {
            bail!("{} codes are not {frames} frames of {}", codes.len(), self.n_codebooks);
        }
        if from >= frames {
            return Ok(Vec::new());
        }
        let d = self.latent;
        let z = self.post(codes, frames);
        let per_step = self.frame_size / 4;
        if window == 0 {
            let (z, t) = self.upsample(&z, frames);
            let w = self.wave(&z, t)[from * self.frame_size..].to_vec();
            bufs::clear();
            return Ok(w);
        }
        // Upsampling reads under 5 frames back, the convolutions under 20 steps.
        let lo = from.saturating_sub(CONTEXT_FRAMES);
        let (z, t) = self.upsample(&z[lo * d..], frames - lo);
        let mut out = Vec::with_capacity((frames - from) * self.frame_size);
        let mut a = (from - lo) * 4;
        while a < t {
            let b = (a + window).min(t);
            let start = a.saturating_sub(CONTEXT_STEPS);
            let w = self.wave(&z[start * d..b * d], b - start);
            out.extend_from_slice(&w[(a - start) * per_step..]);
            bufs::give(w);
            a = b;
        }
        bufs::clear();
        Ok(out)
    }

    /// Mono audio at [`Codec::sample_rate`] to `[frames, n_codebooks]` codes and the frame count
    /// (`ceil(samples / frame_size)`; the last frame is padded with silence).
    pub fn encode(&self, audio: &[f32]) -> Result<(Vec<u32>, usize)> {
        if audio.is_empty() {
            bail!("the reference audio is empty");
        }
        if self.encoder.get().is_none() {
            let f = ModelFiles::open_file(&self.path, false)?;
            let e = Encoder::load(&f, self.latent, self.n_codebooks)
                .with_context(|| format!("{}: the codec's encoder", self.path.display()))?;
            let _ = self.encoder.set(e);
        }
        let enc = self.encoder.get().expect("just set");
        let frames = audio.len().div_ceil(self.frame_size);
        let mut padded = audio.to_vec();
        padded.resize(frames * self.frame_size, 0.0);
        let codes = enc.encode(&padded, self.latent);
        bufs::clear();
        if codes.len() != frames * self.n_codebooks {
            bail!("the codec's encoder gave {} codes for {frames} frames", codes.len());
        }
        Ok((codes, frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values in [-1, 1) from a linear congruential generator.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < 1e-4, "value {i}: {x} against {y}");
        }
    }

    #[test]
    fn gemm_matches_plain_sums() {
        for (co, k, step, nt) in [(1, 7, 7, 3), (8, 16, 4, 5), (11, 37, 37, 2), (19, 64, 8, 33)] {
            let w = noise(co * k, 1);
            let xs = noise((nt - 1) * step + k, 2);
            let want: Vec<f32> = (0..nt)
                .flat_map(|r| {
                    let (w, xs) = (&w, &xs);
                    (0..co).map(move |o| (0..k).map(|i| w[o * k + i] * xs[r * step + i]).sum::<f32>())
                })
                .collect();
            let mut y = vec![0f32; nt * co];
            gemm(&w, co, k, &xs, step, nt, &mut y);
            close(&y, &want);
            y.fill(0.0);
            gemm_portable(&w, co, k, &xs, step, nt, &mut y);
            close(&y, &want);
        }
    }

    /// `ArkttsCausalConv1d` written out: `kernel - stride` steps of zeros before the input, zeros
    /// after it up to whole strides.
    #[test]
    fn causal_convolutions_match_their_definition() {
        for (co, ci, k, stride, dil, t) in [
            (3, 2, 7, 1, 1, 41),
            (5, 4, 7, 1, 9, 70),
            (9, 3, 4, 2, 1, 37),
            (2, 1, 16, 8, 1, 64),
            (4, 6, 1, 1, 1, 40),
        ] {
            let w = noise(co * ci * k, 3);
            let b = noise(co, 4);
            let x = noise(t * ci, 5);
            let conv = Conv::new(&w, [co, ci, k], b.clone(), stride, dil);
            let (y, t_out) = conv.forward(&x, t);
            assert_eq!(t_out, t.div_ceil(stride));
            let pad = ((k - 1) * dil + 1 - stride) as isize;
            let mut want = Vec::new();
            for to in 0..t_out {
                for o in 0..co {
                    let mut s = b[o];
                    for i in 0..ci {
                        for j in 0..k {
                            let src = (to * stride + j * dil) as isize - pad;
                            if src >= 0 && (src as usize) < t {
                                s += w[(o * ci + i) * k + j] * x[src as usize * ci + i];
                            }
                        }
                    }
                    want.push(s);
                }
            }
            close(&y, &want);
        }
    }

    /// `ArkttsCausalConvTranspose1d` written out: the transposed convolution, its last
    /// `kernel - stride` steps cut.
    #[test]
    fn causal_transposed_convolutions_match_their_definition() {
        for (ci, co, k, stride, t) in [(3, 2, 16, 8, 40), (4, 5, 4, 2, 37), (2, 3, 2, 2, 50)] {
            let w = noise(ci * co * k, 6);
            let b = noise(co, 7);
            let x = noise(t * ci, 8);
            let (y, t_out) = ConvT::new(&w, [ci, co, k], &b, stride).forward(&x, t);
            assert_eq!(t_out, t * stride);
            let mut full = vec![0f32; ((t - 1) * stride + k) * co];
            for ti in 0..t {
                for i in 0..ci {
                    for o in 0..co {
                        for j in 0..k {
                            full[(ti * stride + j) * co + o] += x[ti * ci + i] * w[(i * co + o) * k + j];
                        }
                    }
                }
            }
            let want: Vec<f32> = full[..t_out * co].iter().enumerate().map(|(n, v)| v + b[n % co]).collect();
            close(&y, &want);
        }
    }

    #[test]
    fn rope_tables_are_rounded_to_bf16() {
        let t = rope_table(4, 8, 10000.0);
        assert_eq!(t.len(), 4 * 4 * 2);
        assert_eq!(&t[..2], &[1.0, 0.0]);
        // Position 1, pair 0: one radian, with 8 bits of mantissa.
        assert_eq!(t[8], half::bf16::from_f32(1f32.cos()).to_f32());
        assert_ne!(t[8], 1f32.cos());
        let mut v = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0];
        apply_rope(&mut v, &t[8..16], 8);
        assert!((v[0] - t[8]).abs() < 1e-7 && (v[1] - t[9]).abs() < 1e-7);
        assert!((v[2] + t[11]).abs() < 1e-7 && (v[3] - t[10]).abs() < 1e-7);
    }
}
