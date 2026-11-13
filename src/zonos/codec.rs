//! The decoder of ZAC 1, ZONOS3's audio codec (`general.architecture = mossdual_istft`): ten
//! codes per frame at 15 Hz to 44.1 kHz audio. A port of `ISTFTCodec.forward`
//! (Zyphra/ZONOS3-Mini, `model/src/zonos3/codec/istft_head_a_20261002/model.py`) on the tensors of
//! `tools/convert_gguf.py codec-decoder` (`zac-keep2.gguf`: two transformer layers per stage;
//! `zac.gguf`: the full 32 / 12 / 12 / 12 / 12).
//!
//! 1. Latent (1024 wide, 15 Hz): the semantic code's vector (768) through four causal ConvNeXt
//!    blocks and a linear layer, plus the vectors of the nine residual codes. The tables hold each
//!    codebook already multiplied by its output projection.
//! 2. Five stages of causal transformer layers (pre-LayerNorm, RoPE, attention over the last
//!    `context` frames only, GELU feed-forward, LayerScale), each followed by a "patch" that turns
//!    one frame of `2 c` channels into two frames of `c`: 15, 30, 60, 120 and 240 Hz.
//! 3. The head at 240 Hz: eight causal ConvNeXt blocks, LayerNorm, then per window size (736 and
//!    2208 samples) a linear layer to a log-magnitude and a phase per bin and an inverse STFT with
//!    a hop of 184 samples (periodic Hann, centered); the two signals are combined by a 127-tap
//!    FIR filter.
//! 4. The result is at 44160 Hz (184 * 240); a windowed-sinc resampler (735 phases of 770 taps,
//!    torchaudio's kernel, stored in the file) brings it to 44100 Hz, 2940 samples per frame.
//!
//! Everything is causal up to the lookahead of the inverse STFT, the FIR filter and the resampler
//! (25 ms + 1.4 ms + 0.4 ms), but this implementation decodes whole utterances.

use super::file::{Gg, Mat, add_bias, mul_multi};
use crate::kernels::for_chunks;
use crate::sd::ops::gelu_erf;
use anyhow::{Result, bail};

struct ConvNext {
    /// `[c, 7]`, the last tap on the current frame.
    dw: Vec<f32>,
    dw_b: Vec<f32>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    pw1: Mat,
    pw1_b: Vec<f32>,
    pw2: Mat,
    pw2_b: Vec<f32>,
}

struct MossLayer {
    n1: (Vec<f32>, Vec<f32>),
    qkv: Mat,
    wo: Mat,
    attn_scale: Vec<f32>,
    n2: (Vec<f32>, Vec<f32>),
    ffn1: Mat,
    ffn2: Mat,
    ffn_scale: Vec<f32>,
}

struct Stage {
    inp: Mat,
    out: Option<Mat>,
    layers: Vec<MossLayer>,
    heads: usize,
    context: usize,
    patch: usize,
}

/// One inverse-STFT branch of the head.
struct Branch {
    n_fft: usize,
    out: Mat,
    out_b: Vec<f32>,
    /// `[n_fft, 2 * bins (padded)]`: the inverse real FFT of (real parts, imaginary parts) times
    /// the window.
    synth: Mat,
    window: Vec<f32>,
}

pub struct Decoder {
    pub sample_rate: u32,
    /// Samples per frame at the output rate.
    pub hop: usize,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    /// The gain the reference servers apply to the output, in dB below the decoder's level.
    pub headroom_db: f32,
    pub variant: String,
    pub weight_bytes: usize,
    sem_dim: usize,
    latent: usize,
    sem_table: Vec<f32>,
    sem_bias: Vec<f32>,
    res: Vec<(Vec<f32>, Vec<f32>)>,
    sem_head: Vec<ConvNext>,
    sem_out: Mat,
    sem_out_b: Vec<f32>,
    stages: Vec<Stage>,
    head: Vec<ConvNext>,
    head_norm: (Vec<f32>, Vec<f32>),
    branches: Vec<Branch>,
    head_hop: usize,
    max_log_mag: f32,
    /// `[branches, taps]`.
    mixer: Vec<f32>,
    /// `[new, 2 * width + orig]`.
    resample: Vec<f32>,
    rs_orig: usize,
    rs_new: usize,
    rs_width: usize,
    ln_eps: f32,
    cn_eps: f32,
}

fn layer_norm(x: &mut [f32], n: usize, w: &[f32], b: &[f32], eps: f32) {
    for_chunks(x, n, x.len() > 1 << 16, |_, r| {
        let mean = r.iter().sum::<f32>() / n as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let s = 1.0 / (var + eps).sqrt();
        for ((v, w), b) in r.iter_mut().zip(w).zip(b) {
            *v = (*v - mean) * s * w + b;
        }
    });
}

fn gelu(x: &mut [f32]) {
    for_chunks(x, 4096, x.len() > 1 << 16, |_, c| c.iter_mut().for_each(|v| *v = gelu_erf(*v)));
}

impl ConvNext {
    fn load(g: &Gg, p: &str, c: usize, hidden: usize, bytes: &mut usize) -> Result<Self> {
        let pw1 = g.mat(&format!("{p}.pw1.weight"), hidden, c)?;
        let pw2 = g.mat(&format!("{p}.pw2.weight"), c, hidden)?;
        *bytes += pw1.bytes() + pw2.bytes();
        Ok(Self {
            dw: g.f32(&format!("{p}.dw.weight"), &[c, 7])?,
            dw_b: g.f32(&format!("{p}.dw.bias"), &[c])?,
            norm_w: g.f32(&format!("{p}.norm.weight"), &[c])?,
            norm_b: g.f32(&format!("{p}.norm.bias"), &[c])?,
            pw1,
            pw1_b: g.f32(&format!("{p}.pw1.bias"), &[hidden])?,
            pw2,
            pw2_b: g.f32(&format!("{p}.pw2.bias"), &[c])?,
        })
    }

    /// `x` (`[t, c]`) plus the block's output: depthwise convolution over the last 7 frames,
    /// LayerNorm, linear, GELU, linear.
    fn forward(&self, x: &mut [f32], t: usize, eps: f32) {
        let c = self.dw_b.len();
        let k = 7;
        let mut h = vec![0f32; t * c];
        let xr: &[f32] = x;
        for_chunks(&mut h, c, t * c > 1 << 16, |i, o| {
            o.copy_from_slice(&self.dw_b);
            for j in 0..k {
                if i + j + 1 < k {
                    continue;
                }
                let src = &xr[(i + j + 1 - k) * c..(i + j + 2 - k) * c];
                for (ch, (o, s)) in o.iter_mut().zip(src).enumerate() {
                    *o += self.dw[ch * k + j] * s;
                }
            }
        });
        layer_norm(&mut h, c, &self.norm_w, &self.norm_b, eps);
        let mut u = self.pw1.mul(&h, t, true);
        add_bias(&mut u, &self.pw1_b);
        gelu(&mut u);
        let mut y = self.pw2.mul(&u, t, true);
        add_bias(&mut y, &self.pw2_b);
        crate::kernels::add_inplace(x, &y);
    }
}

impl MossLayer {
    fn forward(&self, x: &mut [f32], t: usize, heads: usize, context: usize, eps: f32) {
        let d = self.attn_scale.len();
        let hd = d / heads;
        let mut h = x.to_vec();
        layer_norm(&mut h, d, &self.n1.0, &self.n1.1, eps);
        let mut qkv = self.qkv.mul(&h, t, true);
        // RoPE on adjacent pairs of q and k, by frame index.
        let freq: Vec<f32> = (0..hd / 2)
            .map(|i| (i as f32 * (-(10000f32).ln() * 2.0 / hd as f32)).exp())
            .collect();
        for_chunks(&mut qkv, 3 * d, t * d > 1 << 14, |i, r| {
            let cs: Vec<(f32, f32)> = freq.iter().map(|f| (i as f32 * f).sin_cos()).collect();
            for part in r[..2 * d].chunks_mut(hd) {
                for (p, &(s, c)) in part.as_chunks_mut::<2>().0.iter_mut().zip(&cs) {
                    let (a, b) = (p[0], p[1]);
                    p[0] = a * c - b * s;
                    p[1] = a * s + b * c;
                }
            }
        });
        let scale = 1.0 / (hd as f32).sqrt();
        let mut att = vec![0f32; t * d];
        let qkv = &qkv;
        for_chunks(&mut att, d, true, |i, o| {
            let lo = (i + 1).saturating_sub(context);
            let n = i + 1 - lo;
            let mut w = vec![0f32; n];
            for hh in 0..heads {
                let q = &qkv[i * 3 * d + hh * hd..i * 3 * d + (hh + 1) * hd];
                for (j, w) in w.iter_mut().enumerate() {
                    let k = &qkv[(lo + j) * 3 * d + d + hh * hd..(lo + j) * 3 * d + d + (hh + 1) * hd];
                    *w = crate::kernels::dot(q, k) * scale;
                }
                super::lm::softmax(&mut w);
                let oh = &mut o[hh * hd..(hh + 1) * hd];
                for (j, &p) in w.iter().enumerate() {
                    let v = &qkv[(lo + j) * 3 * d + 2 * d + hh * hd..(lo + j) * 3 * d + 2 * d + (hh + 1) * hd];
                    crate::kernels::axpy(oh, p, v);
                }
            }
        });
        let a = self.wo.mul(&att, t, true);
        for (r, a) in x.chunks_mut(d).zip(a.chunks(d)) {
            for ((v, a), s) in r.iter_mut().zip(a).zip(&self.attn_scale) {
                *v += s * a;
            }
        }
        let mut h = x.to_vec();
        layer_norm(&mut h, d, &self.n2.0, &self.n2.1, eps);
        let mut u = self.ffn1.mul(&h, t, true);
        gelu(&mut u);
        let f = self.ffn2.mul(&u, t, true);
        for (r, f) in x.chunks_mut(d).zip(f.chunks(d)) {
            for ((v, f), s) in r.iter_mut().zip(f).zip(&self.ffn_scale) {
                *v += s * f;
            }
        }
    }
}

impl Decoder {
    pub(crate) fn load(g: &Gg) -> Result<Self> {
        let mut bytes = 0usize;
        let n_cb = g.usize("n_codebooks")?;
        let size = g.usize("codebook_size")?;
        let latent = g.usize("latent_dim")?;
        let sem_dim = g.usize("semantic_dim")?;
        let hidden = g.usize("head.hidden")?;
        let sem_head = (0..g.usize("sem_head.depth")?)
            .map(|i| ConvNext::load(g, &format!("sem_head.{i}"), sem_dim, hidden, &mut bytes))
            .collect::<Result<Vec<_>>>()?;
        let (in_dim, out_dim, d_model) = (g.usizes("stage.in_dim")?, g.usizes("stage.out_dim")?, g.usizes("stage.d_model")?);
        let (n_head, n_ff, context) = (g.usizes("stage.n_head")?, g.usizes("stage.n_ff")?, g.usizes("stage.context")?);
        let (n_layer, patch) = (g.usizes("stage.n_layer")?, g.usizes("stage.patch")?);
        let mut stages = Vec::new();
        for s in 0..d_model.len() {
            let (d, p) = (d_model[s], format!("stage.{s}"));
            let mut mat = |name: &str, n_out: usize, n_in: usize| -> Result<Mat> {
                let m = g.mat(name, n_out, n_in)?;
                bytes += m.bytes();
                Ok(m)
            };
            let layers = (0..n_layer[s])
                .map(|l| {
                    let p = format!("{p}.l{l}");
                    Ok(MossLayer {
                        n1: (g.f32(&format!("{p}.norm1.weight"), &[d])?, g.f32(&format!("{p}.norm1.bias"), &[d])?),
                        qkv: mat(&format!("{p}.qkv.weight"), 3 * d, d)?,
                        wo: mat(&format!("{p}.wo.weight"), d, d)?,
                        attn_scale: g.f32(&format!("{p}.attn_scale"), &[d])?,
                        n2: (g.f32(&format!("{p}.norm2.weight"), &[d])?, g.f32(&format!("{p}.norm2.bias"), &[d])?),
                        ffn1: mat(&format!("{p}.ffn1.weight"), n_ff[s], d)?,
                        ffn2: mat(&format!("{p}.ffn2.weight"), d, n_ff[s])?,
                        ffn_scale: g.f32(&format!("{p}.ffn_scale"), &[d])?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            stages.push(Stage {
                inp: mat(&format!("{p}.in.weight"), d, in_dim[s])?,
                out: if out_dim[s] > 0 {
                    Some(mat(&format!("{p}.out.weight"), out_dim[s], d)?)
                } else {
                    None
                },
                layers,
                heads: n_head[s],
                context: context[s],
                patch: patch[s],
            });
        }
        let hd = *d_model.last().unwrap_or(&0);
        let head = (0..g.usize("head.depth")?)
            .map(|i| ConvNext::load(g, &format!("head.{i}"), hd, hidden, &mut bytes))
            .collect::<Result<Vec<_>>>()?;
        let n_ffts = g.usizes("head.n_fft")?;
        let taps = g.usize("head.fir_kernel")?;
        let mut branches = Vec::new();
        for (i, &n) in n_ffts.iter().enumerate() {
            let window = g.f32(&format!("head.window.{i}"), &[n])?;
            let bins = n / 2 + 1;
            let cols = (2 * bins).next_multiple_of(32);
            // Row m: window[m] / n times the inverse real DFT's coefficients of the real parts,
            // then of the imaginary parts (those of the first and the last bin do not count).
            let mut synth = vec![0f32; n * cols];
            for_chunks(&mut synth, cols, true, |m, r| {
                let wn = window[m] as f64 / n as f64;
                for k in 0..bins {
                    let a = 2.0 * std::f64::consts::PI * ((k * m) % n) as f64 / n as f64;
                    let edge = k == 0 || k == n / 2;
                    r[k] = (wn * if edge { 1.0 } else { 2.0 } * a.cos()) as f32;
                    r[bins + k] = if edge { 0.0 } else { (-2.0 * wn * a.sin()) as f32 };
                }
            });
            branches.push(Branch {
                n_fft: n,
                out: g.mat(&format!("head.out.{i}.weight"), n + 2, hd)?,
                out_b: g.f32(&format!("head.out.{i}.bias"), &[n + 2])?,
                synth: Mat::from_f32(&synth, n, cols)?,
                window,
            });
        }
        let (rs_orig, rs_new, rs_width) = (g.usize("resample.orig")?, g.usize("resample.new")?, g.usize("resample.width")?);
        let res = (0..n_cb - 1)
            .map(|i| {
                Ok((
                    g.f32(&format!("res.{i}.table"), &[size, latent])?,
                    g.f32(&format!("res.{i}.bias"), &[latent])?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let sem_out = g.mat("sem_head.out.weight", latent, sem_dim)?;
        let head_hop = g.usize("head.hop")?;
        let hop = g.usize("hop")?;
        let up: usize = stages.iter().map(|s| s.patch).product();
        if up * head_hop * rs_new != hop * rs_orig || stages.is_empty() {
            bail!(
                "{}: frame of {hop} samples does not match {up} head frames of {head_hop}",
                g.path.display()
            );
        }
        Ok(Self {
            sample_rate: g.usize("sample_rate")? as u32,
            hop,
            n_codebooks: n_cb,
            codebook_size: size,
            headroom_db: g.f64("headroom_db").unwrap_or(0.0) as f32,
            variant: g.str("head.variant").unwrap_or("").to_string(),
            weight_bytes: bytes + sem_out.bytes() + (res.len() * size * latent + size * sem_dim) * 4,
            sem_dim,
            latent,
            sem_table: g.f32("sem.table", &[size, sem_dim])?,
            sem_bias: g.f32("sem.bias", &[sem_dim])?,
            res,
            sem_head,
            sem_out,
            sem_out_b: g.f32("sem_head.out.bias", &[latent])?,
            stages,
            head,
            head_norm: (g.f32("head.norm.weight", &[hd])?, g.f32("head.norm.bias", &[hd])?),
            branches,
            head_hop,
            max_log_mag: g.f64("head.max_log_magnitude")? as f32,
            mixer: g.f32("head.mixer.weight", &[n_ffts.len(), taps])?,
            resample: g.f32("resample.kernel", &[rs_new, 2 * rs_width + rs_orig])?,
            rs_orig,
            rs_new,
            rs_width,
            ln_eps: g.f64("ln_eps")? as f32,
            cn_eps: g.f64("convnext_eps")? as f32,
        })
    }

    /// The 240 Hz features `[16 * frames, 768]` of `codes` (`[frames, n_codebooks]`).
    fn features(&self, codes: &[u32], frames: usize) -> Result<(Vec<f32>, usize)> {
        let n_cb = self.n_codebooks;
        if codes.len() != frames * n_cb || frames == 0 {
            bail!("zonos3 codec: {} codes for {frames} frames of {n_cb}", codes.len());
        }
        if let Some(c) = codes.iter().find(|&&c| c as usize >= self.codebook_size) {
            bail!("zonos3 codec: {c} is not a code (the frames hold padding or an end token)");
        }
        let (sd, ld) = (self.sem_dim, self.latent);
        let mut q1 = vec![0f32; frames * sd];
        for (o, f) in q1.chunks_mut(sd).zip(codes.chunks(n_cb)) {
            let r = &self.sem_table[f[0] as usize * sd..(f[0] as usize + 1) * sd];
            for ((o, r), b) in o.iter_mut().zip(r).zip(&self.sem_bias) {
                *o = r + b;
            }
        }
        for b in &self.sem_head {
            b.forward(&mut q1, frames, self.cn_eps);
        }
        let mut z = self.sem_out.mul(&q1, frames, true);
        add_bias(&mut z, &self.sem_out_b);
        for (o, f) in z.chunks_mut(ld).zip(codes.chunks(n_cb)) {
            for (i, (table, bias)) in self.res.iter().enumerate() {
                let r = &table[f[1 + i] as usize * ld..(f[1 + i] as usize + 1) * ld];
                for ((o, r), b) in o.iter_mut().zip(r).zip(bias) {
                    *o += r + b;
                }
            }
        }
        let (mut x, mut t) = (z, frames);
        for s in &self.stages {
            x = s.inp.mul(&x, t, true);
            for l in &s.layers {
                l.forward(&mut x, t, s.heads, s.context, self.ln_eps);
            }
            if let Some(o) = &s.out {
                x = o.mul(&x, t, true);
            }
            if s.patch > 1 {
                // One frame of `c * patch` channels becomes `patch` frames of `c`.
                let (p, c) = (s.patch, x.len() / t / s.patch);
                let mut y = vec![0f32; x.len()];
                for (i, r) in x.chunks(c * p).enumerate() {
                    for h in 0..p {
                        for (cc, o) in y[(i * p + h) * c..(i * p + h + 1) * c].iter_mut().enumerate() {
                            *o = r[cc * p + h];
                        }
                    }
                }
                x = y;
                t *= p;
            }
        }
        Ok((x, t))
    }

    /// Decodes `codes` (`[frames, n_codebooks]`, every value a code) to `frames * hop` samples.
    pub fn decode(&self, codes: &[u32], frames: usize) -> Result<Vec<f32>> {
        let (mut h, t) = self.features(codes, frames)?;
        for b in &self.head {
            b.forward(&mut h, t, self.cn_eps);
        }
        let hd = self.head_norm.0.len();
        layer_norm(&mut h, hd, &self.head_norm.0, &self.head_norm.1, 1e-6);
        let hop = self.head_hop;
        let len = t * hop;
        let mut ys: Vec<Vec<f32>> = self.branches.iter().map(|b| vec![0f32; t * b.out.n_out]).collect();
        {
            let mut jobs: Vec<(&Mat, &mut [f32])> = self.branches.iter().zip(ys.iter_mut()).map(|(b, y)| (&b.out, &mut y[..])).collect();
            mul_multi(&mut jobs, &h, t, true);
        }
        let mut signals = Vec::with_capacity(self.branches.len());
        for (b, mut y) in self.branches.iter().zip(ys) {
            add_bias(&mut y, &b.out_b);
            let n = b.n_fft;
            let bins = n / 2 + 1;
            let cols = b.synth.n_in;
            let mut spec = vec![0f32; t * cols];
            let y = &y;
            for_chunks(&mut spec, cols, true, |i, o| {
                let r = &y[i * (n + 2)..(i + 1) * (n + 2)];
                for k in 0..bins {
                    let mag = r[k].min(self.max_log_mag).exp();
                    let (s, c) = r[bins + k].sin_cos();
                    o[k] = mag * c;
                    o[bins + k] = mag * s;
                }
            });
            let fr = b.synth.mul(&spec, t, true);
            // Overlap-add, divided by the summed squared windows, without the first half window.
            let total = n + hop * (t - 1);
            let (mut acc, mut env) = (vec![0f32; total], vec![0f32; total]);
            for (i, f) in fr.chunks(n).enumerate() {
                crate::kernels::add_inplace(&mut acc[i * hop..i * hop + n], f);
                for (e, w) in env[i * hop..i * hop + n].iter_mut().zip(&b.window) {
                    *e += w * w;
                }
            }
            signals.push(
                acc[n / 2..n / 2 + len]
                    .iter()
                    .zip(&env[n / 2..])
                    .map(|(a, e)| a / e)
                    .collect::<Vec<f32>>(),
            );
        }
        // The FIR mixer: a centered filter per branch, zero padded.
        let taps = self.mixer.len() / self.branches.len();
        let half = (taps - 1) / 2;
        let padded: Vec<Vec<f32>> = signals
            .iter()
            .map(|s| {
                let mut p = vec![0f32; len + taps - 1];
                p[half..half + len].copy_from_slice(s);
                p
            })
            .collect();
        let mut mixed = vec![0f32; len];
        for_chunks(&mut mixed, 4096, true, |c, o| {
            for (j, o) in o.iter_mut().enumerate() {
                let s = c * 4096 + j;
                *o = padded
                    .iter()
                    .enumerate()
                    .map(|(b, p)| crate::kernels::dot(&self.mixer[b * taps..(b + 1) * taps], &p[s..s + taps]))
                    .sum();
            }
        });
        // 44160 Hz to 44100 Hz: every block of `orig` samples gives `new`, one per kernel row.
        let (orig, new, width) = (self.rs_orig, self.rs_new, self.rs_width);
        let klen = 2 * width + orig;
        let blocks = len / orig + 1;
        let mut padded = vec![0f32; (blocks - 1) * orig + klen];
        padded[width..width + len].copy_from_slice(&mixed);
        let mut out = vec![0f32; blocks * new];
        for_chunks(&mut out, new, true, |b, o| {
            let x = &padded[b * orig..b * orig + klen];
            for (p, o) in o.iter_mut().enumerate() {
                *o = crate::kernels::dot(&self.resample[p * klen..(p + 1) * klen], x);
            }
        });
        out.truncate(frames * self.hop);
        Ok(out)
    }
}
