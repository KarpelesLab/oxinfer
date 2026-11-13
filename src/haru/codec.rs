//! The audio codec of Haru (`codec.*`, a port of `haru/models/vibevoice_codec.py` of
//! higurashi-ai/Haru; the model is VibeVoice's causal acoustic tokenizer): 24 kHz audio to and
//! from one latent of 64 values per 3200 samples.
//!
//! Both directions are stacks of causal layers. The encoder is a convolution to 32 channels,
//! then seven stages of blocks (a depthwise convolution over the last 7 steps and an MLP of four
//! times the width, each behind an RMS norm and scaled per channel) with a strided convolution
//! between stages (by 2, 2, 4, 5, 5 and 8, doubling the channels up to 2048), then a convolution
//! to the latent. The decoder mirrors it with transposed convolutions. Every layer keeps its
//! left context as a state, so a clip is encoded or decoded frame by frame with the samples a
//! whole pass gives. Weights stay in the checkpoint's bf16; activations are `[time, channels]`
//! in f32.

use crate::kernels::{bf16_to_f32, for_chunks, matmul};
use crate::safetensors::ModelFiles;
use crate::sd::ops::gelu_erf;
use anyhow::{Context, Result, bail};
use serde::Deserialize;

fn d_std() -> f32 {
    0.5
}
fn d_eps() -> f32 {
    1e-5
}

/// `codec` of a Haru `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct CodecConfig {
    pub latent_dim: usize,
    pub base_channels: usize,
    /// The decoder's upsampling factors, first stage first.
    pub ratios: Vec<usize>,
    pub encoder_depths: Vec<usize>,
    pub decoder_depths: Vec<usize>,
    #[serde(default = "d_eps")]
    pub norm_eps: f32,
    #[serde(default = "d_std")]
    pub latent_std: f32,
}

/// Rows of `width` in parallel groups.
fn rows_par(x: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let group = (8192 / width.max(1)).max(1);
    for_chunks(x, group * width, x.len() > 1 << 15, |g, block| {
        for (i, row) in block.chunks_mut(width).enumerate() {
            f(g * group + i, row);
        }
    });
}

fn vector(f: &ModelFiles, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = f.f32(name)?;
    if v.len() != n {
        bail!("{name}: {} values, expected {n}", v.len());
    }
    Ok(v)
}

fn shape(f: &ModelFiles, name: &str) -> Result<Vec<usize>> {
    Ok(f.info(name).with_context(|| format!("missing tensor {name}"))?.shape.clone())
}

/// A bf16 matrix `[n_out, n_in]` of the checkpoint.
fn matrix(f: &'static ModelFiles, name: &str, n_out: usize, n_in: usize) -> Result<&'static [u16]> {
    let s = shape(f, name)?;
    if s[..] != [n_out, n_in] {
        bail!("{name}: shape {s:?}, expected [{n_out}, {n_in}]");
    }
    f.bf16(name)
}

/// `x` (`[t, n_in]`) times `w^T`, plus `b`.
fn linear(w: &[u16], b: Option<&[f32]>, n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
    let mut y = vec![0f32; t * n_out];
    matmul(w, n_out, n_in, x, t, &mut y);
    if let Some(b) = b {
        rows_par(&mut y, n_out, |_, row| {
            for (v, &bb) in row.iter_mut().zip(b) {
                *v += bb;
            }
        });
    }
    y
}

fn rms_rows(x: &[f32], n: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    rows_par(&mut out, n, |r, o| {
        let xr = &x[r * n..(r + 1) * n];
        let s = 1.0 / (xr.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        for ((o, &v), &w) in o.iter_mut().zip(xr).zip(w) {
            *o = v * s * w;
        }
    });
    out
}

/// `CausalConv1d`: the state is the last `kernel - stride` input steps.
struct Conv {
    /// `[c_out, k * c_in]`, tap by tap.
    w: Vec<u16>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
}

impl Conv {
    fn load(f: &ModelFiles, p: &str, stride: usize) -> Result<Self> {
        let name = format!("{p}.weight");
        let s = shape(f, &name)?;
        let [co, ci, k] = s[..] else {
            bail!("{name}: expected a kernel [out, in, k], got {s:?}")
        };
        if k < stride {
            bail!("{name}: kernel {k} shorter than its stride {stride}");
        }
        let src = f.bf16(&name)?;
        let mut w = vec![0u16; src.len()];
        for o in 0..co {
            for i in 0..ci {
                for j in 0..k {
                    w[(o * k + j) * ci + i] = src[(o * ci + i) * k + j];
                }
            }
        }
        Ok(Self {
            w,
            b: vector(f, &format!("{p}.bias"), co)?,
            c_in: ci,
            c_out: co,
            k,
            stride,
        })
    }

    fn state_len(&self) -> usize {
        (self.k - self.stride) * self.c_in
    }

    /// `x` (`[t, c_in]`, padded with zeros to whole strides) to `[ceil(t / stride), c_out]`.
    fn forward(&self, x: &[f32], state: &mut Vec<f32>) -> Vec<f32> {
        let (ci, kk) = (self.c_in, self.k * self.c_in);
        let t = x.len() / ci;
        let tp = t.next_multiple_of(self.stride);
        let mut buf = std::mem::take(state);
        buf.extend_from_slice(x);
        buf.resize(buf.len() + (tp - t) * ci, 0.0);
        let t_out = tp / self.stride;
        let mut col = vec![0f32; t_out * kk];
        let step = self.stride * ci;
        rows_par(&mut col, kk, |r, row| row.copy_from_slice(&buf[r * step..r * step + kk]));
        let y = linear(&self.w, Some(&self.b), self.c_out, kk, &col, t_out);
        let keep = self.state_len();
        *state = buf[buf.len() - keep..].to_vec();
        y
    }
}

/// `CausalConvTranspose1d` (kernel of two strides): the state is what the last input step adds
/// to the next `stride` output steps.
struct ConvT {
    /// `[2 * stride * c_out, c_in]`: output tap `j` of channel `o` in row `j * c_out + o`.
    w: Vec<u16>,
    b: Vec<f32>,
    c_in: usize,
    c_out: usize,
    stride: usize,
}

impl ConvT {
    fn load(f: &ModelFiles, p: &str, stride: usize) -> Result<Self> {
        let name = format!("{p}.weight");
        let s = shape(f, &name)?;
        let [ci, co, k] = s[..] else {
            bail!("{name}: expected a kernel [in, out, k], got {s:?}")
        };
        if k != 2 * stride {
            bail!("{name}: kernel {k} is not twice its stride {stride}");
        }
        let src = f.bf16(&name)?;
        let mut w = vec![0u16; src.len()];
        for i in 0..ci {
            for o in 0..co {
                for j in 0..k {
                    w[(j * co + o) * ci + i] = src[(i * co + o) * k + j];
                }
            }
        }
        Ok(Self {
            w,
            b: vector(f, &format!("{p}.bias"), co)?,
            c_in: ci,
            c_out: co,
            stride,
        })
    }

    fn state_len(&self) -> usize {
        self.stride * self.c_out
    }

    /// `x` (`[t, c_in]`) to `[t * stride, c_out]`.
    fn forward(&self, x: &[f32], state: &mut [f32]) -> Vec<f32> {
        let t = x.len() / self.c_in;
        let (co, half) = (self.c_out, self.stride * self.c_out);
        let m = linear(&self.w, None, 2 * half, self.c_in, x, t);
        let mut y = vec![0f32; t * half];
        {
            let state = &*state;
            rows_par(&mut y, half, |r, row| {
                let now = &m[r * 2 * half..r * 2 * half + half];
                let before = if r == 0 {
                    state
                } else {
                    &m[(r - 1) * 2 * half + half..r * 2 * half]
                };
                for (i, v) in row.iter_mut().enumerate() {
                    *v = now[i] + before[i] + self.b[i % co];
                }
            });
        }
        if t > 0 {
            state.copy_from_slice(&m[(t - 1) * 2 * half + half..]);
        }
        y
    }
}

/// `Block`: a depthwise causal convolution and an MLP, each a scaled residual. The state is the
/// last 6 steps of the normalized input.
struct Block {
    c: usize,
    eps: f32,
    norm: Vec<f32>,
    /// `[c, 7]`, the last tap on the current step.
    dw: Vec<f32>,
    dw_b: Vec<f32>,
    gamma: Vec<f32>,
    ffn_norm: Vec<f32>,
    fc1: &'static [u16],
    fc1_b: Vec<f32>,
    fc2: &'static [u16],
    fc2_b: Vec<f32>,
    ffn_gamma: Vec<f32>,
}

const DW: usize = 7;

impl Block {
    fn load(f: &'static ModelFiles, p: &str, c: usize, eps: f32) -> Result<Self> {
        let name = format!("{p}.conv.weight");
        if shape(f, &name)?[..] != [c, 1, DW] {
            bail!("{name}: expected a depthwise kernel [{c}, 1, {DW}]");
        }
        Ok(Self {
            c,
            eps,
            norm: vector(f, &format!("{p}.norm.weight"), c)?,
            dw: f.bf16(&name)?.iter().map(|&v| bf16_to_f32(v)).collect(),
            dw_b: vector(f, &format!("{p}.conv.bias"), c)?,
            gamma: vector(f, &format!("{p}.gamma"), c)?,
            ffn_norm: vector(f, &format!("{p}.ffn_norm.weight"), c)?,
            fc1: matrix(f, &format!("{p}.fc1.weight"), 4 * c, c)?,
            fc1_b: vector(f, &format!("{p}.fc1.bias"), 4 * c)?,
            fc2: matrix(f, &format!("{p}.fc2.weight"), c, 4 * c)?,
            fc2_b: vector(f, &format!("{p}.fc2.bias"), c)?,
            ffn_gamma: vector(f, &format!("{p}.ffn_gamma"), c)?,
        })
    }

    fn state_len(&self) -> usize {
        (DW - 1) * self.c
    }

    fn forward(&self, x: &mut [f32], state: &mut Vec<f32>) {
        let c = self.c;
        let t = x.len() / c;
        let mut buf = std::mem::take(state);
        buf.extend_from_slice(&rms_rows(x, c, &self.norm, self.eps));
        rows_par(x, c, |r, row| {
            for (ch, v) in row.iter_mut().enumerate() {
                let mut s = self.dw_b[ch];
                for j in 0..DW {
                    s += self.dw[ch * DW + j] * buf[(r + j) * c + ch];
                }
                *v += s * self.gamma[ch];
            }
        });
        *state = buf[buf.len() - self.state_len()..].to_vec();
        let h = rms_rows(x, c, &self.ffn_norm, self.eps);
        let mut a = linear(self.fc1, Some(&self.fc1_b), 4 * c, c, &h, t);
        rows_par(&mut a, 4 * c, |_, row| row.iter_mut().for_each(|v| *v = gelu_erf(*v)));
        let o = linear(self.fc2, Some(&self.fc2_b), c, 4 * c, &a, t);
        rows_par(x, c, |r, row| {
            for (ch, v) in row.iter_mut().enumerate() {
                *v += o[r * c + ch] * self.ffn_gamma[ch];
            }
        });
    }
}

enum Layer {
    Conv(Conv),
    Up(ConvT),
    Block(Block),
}

impl Layer {
    fn state_len(&self) -> usize {
        match self {
            Layer::Conv(l) => l.state_len(),
            Layer::Up(l) => l.state_len(),
            Layer::Block(l) => l.state_len(),
        }
    }
}

/// The left context of every layer of one direction of one stream.
#[derive(Clone)]
pub struct State(Vec<Vec<f32>>);

fn run(layers: &[Layer], mut x: Vec<f32>, state: &mut State) -> Vec<f32> {
    for (l, s) in layers.iter().zip(state.0.iter_mut()) {
        match l {
            Layer::Conv(c) => x = c.forward(&x, s),
            Layer::Up(c) => x = c.forward(&x, s),
            Layer::Block(b) => b.forward(&mut x, s),
        }
    }
    x
}

pub struct Codec {
    pub latent_dim: usize,
    /// Samples per latent.
    pub frame_samples: usize,
    pub latent_std: f32,
    pub weight_bytes: usize,
    encoder: Vec<Layer>,
    decoder: Vec<Layer>,
}

/// Frames a whole-clip pass takes at a time.
const CHUNK_FRAMES: usize = 16;

impl Codec {
    pub fn load(f: &'static ModelFiles, cfg: &CodecConfig) -> Result<Self> {
        let n = cfg.ratios.len();
        if cfg.encoder_depths.len() != n + 1 || cfg.decoder_depths.len() != n + 1 {
            bail!("haru codec: {n} ratios need {} stages in each direction", n + 1);
        }
        let (c, eps) = (cfg.base_channels, cfg.norm_eps);
        let mut encoder = vec![Layer::Conv(Conv::load(f, "codec.encoder.stem", 1)?)];
        for (i, &depth) in cfg.encoder_depths.iter().enumerate() {
            if i > 0 {
                let stride = cfg.ratios[n - i];
                encoder.push(Layer::Conv(Conv::load(f, &format!("codec.encoder.down.{}", i - 1), stride)?));
            }
            for j in 0..depth {
                encoder.push(Layer::Block(Block::load(f, &format!("codec.encoder.stages.{i}.{j}"), c << i, eps)?));
            }
        }
        encoder.push(Layer::Conv(Conv::load(f, "codec.encoder.head", 1)?));
        let mut decoder = vec![Layer::Conv(Conv::load(f, "codec.decoder.stem", 1)?)];
        for (i, &depth) in cfg.decoder_depths.iter().enumerate() {
            if i > 0 {
                decoder.push(Layer::Up(ConvT::load(
                    f,
                    &format!("codec.decoder.up.{}", i - 1),
                    cfg.ratios[i - 1],
                )?));
            }
            for j in 0..depth {
                decoder.push(Layer::Block(Block::load(
                    f,
                    &format!("codec.decoder.stages.{i}.{j}"),
                    c << (n - i),
                    eps,
                )?));
            }
        }
        decoder.push(Layer::Conv(Conv::load(f, "codec.decoder.head", 1)?));
        let weight_bytes = f
            .names()
            .filter(|n| n.starts_with("codec."))
            .filter_map(|n| f.info(n))
            .map(|i| i.end - i.start)
            .sum();
        Ok(Self {
            latent_dim: cfg.latent_dim,
            frame_samples: cfg.ratios.iter().product(),
            latent_std: cfg.latent_std,
            weight_bytes,
            encoder,
            decoder,
        })
    }

    /// The state of a stream that has heard nothing yet.
    pub fn encoder_state(&self) -> State {
        State(self.encoder.iter().map(|l| vec![0f32; l.state_len()]).collect())
    }

    pub fn decoder_state(&self) -> State {
        State(self.decoder.iter().map(|l| vec![0f32; l.state_len()]).collect())
    }

    /// The next samples of a stream to latent means (`[n, latent_dim]`). Whole frames, except
    /// for the end of a clip, whose last partial frame is padded with zeros.
    pub fn encode_stream(&self, audio: &[f32], state: &mut State) -> Vec<f32> {
        run(&self.encoder, audio.to_vec(), state)
    }

    /// The next latents (`[n, latent_dim]`) of a stream to `n * frame_samples` samples.
    pub fn decode_stream(&self, latents: &[f32], state: &mut State) -> Vec<f32> {
        run(&self.decoder, latents.to_vec(), state)
    }

    /// A whole clip to latent means, `ceil(samples / frame_samples)` of them.
    pub fn encode(&self, audio: &[f32]) -> Vec<f32> {
        let mut state = self.encoder_state();
        let mut out = Vec::new();
        for chunk in audio.chunks(CHUNK_FRAMES * self.frame_samples) {
            out.extend(self.encode_stream(chunk, &mut state));
        }
        out
    }

    /// A whole sequence of latents to audio.
    pub fn decode(&self, latents: &[f32]) -> Vec<f32> {
        let mut state = self.decoder_state();
        let mut out = Vec::new();
        for chunk in latents.chunks(CHUNK_FRAMES * self.latent_dim) {
            out.extend(self.decode_stream(chunk, &mut state));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values in [-1, 1) from a linear congruential generator, exactly representable in bf16.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                half::bf16::from_f32(((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0).to_f32()
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| half::bf16::from_f32(x).to_bits()).collect()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < 1e-4, "value {i}: {x} against {y}");
        }
    }

    /// A causal convolution against its definition (zeros before the clip, zeros up to a whole
    /// stride after it), and fed in pieces against fed at once.
    #[test]
    fn convolutions_match_their_definition_whole_or_in_pieces() {
        for (co, ci, k, stride, t) in [(3, 2, 7, 1, 40), (4, 3, 4, 2, 37), (5, 2, 16, 8, 64), (2, 1, 10, 5, 23)] {
            let w = noise(co * ci * k, 1);
            let b = noise(co, 2);
            let x = noise(t * ci, 3);
            // The layer's layout is tap by tap.
            let mut tap = vec![0f32; w.len()];
            for o in 0..co {
                for i in 0..ci {
                    for j in 0..k {
                        tap[(o * k + j) * ci + i] = w[(o * ci + i) * k + j];
                    }
                }
            }
            let conv = Conv {
                w: bits(&tap),
                b: b.clone(),
                c_in: ci,
                c_out: co,
                k,
                stride,
            };
            let mut state = vec![0f32; conv.state_len()];
            let y = conv.forward(&x, &mut state);
            let t_out = t.div_ceil(stride);
            let ctx = (k - stride) as isize;
            let mut want = Vec::new();
            for to in 0..t_out {
                for o in 0..co {
                    let mut s = b[o];
                    for i in 0..ci {
                        for j in 0..k {
                            let src = (to * stride + j) as isize - ctx;
                            if src >= 0 && (src as usize) < t {
                                s += w[(o * ci + i) * k + j] * x[src as usize * ci + i];
                            }
                        }
                    }
                    want.push(s);
                }
            }
            close(&y, &want);
            let mut state = vec![0f32; conv.state_len()];
            let mut pieces = Vec::new();
            for chunk in x.chunks(3 * stride * ci) {
                pieces.extend(conv.forward(chunk, &mut state));
            }
            close(&pieces, &want);
        }
    }

    /// A causal transposed convolution against its definition (the overlap of the last input
    /// step held back), whole or in pieces.
    #[test]
    fn transposed_convolutions_match_their_definition_whole_or_in_pieces() {
        for (ci, co, stride, t) in [(3, 2, 8, 20), (4, 5, 2, 33), (2, 3, 5, 17)] {
            let k = 2 * stride;
            let w = noise(ci * co * k, 4);
            let b = noise(co, 5);
            let x = noise(t * ci, 6);
            let mut rows = vec![0f32; w.len()];
            for i in 0..ci {
                for o in 0..co {
                    for j in 0..k {
                        rows[(j * co + o) * ci + i] = w[(i * co + o) * k + j];
                    }
                }
            }
            let up = ConvT {
                w: bits(&rows),
                b: b.clone(),
                c_in: ci,
                c_out: co,
                stride,
            };
            let mut full = vec![0f32; (t + 1) * stride * co];
            for ti in 0..t {
                for i in 0..ci {
                    for o in 0..co {
                        for j in 0..k {
                            full[(ti * stride + j) * co + o] += x[ti * ci + i] * w[(i * co + o) * k + j];
                        }
                    }
                }
            }
            let want: Vec<f32> = full[..t * stride * co].iter().enumerate().map(|(n, v)| v + b[n % co]).collect();
            let mut state = vec![0f32; up.state_len()];
            close(&up.forward(&x, &mut state), &want);
            let mut state = vec![0f32; up.state_len()];
            let mut pieces = Vec::new();
            for chunk in x.chunks(4 * ci) {
                pieces.extend(up.forward(chunk, &mut state));
            }
            close(&pieces, &want);
        }
    }
}
