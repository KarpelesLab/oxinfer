//! The speaker encoder of ZONOS3-Mini (`general.architecture = mect_spk`, `spk-encoder.gguf`):
//! up to 60 s of speech to the 192-d embedding the language model's speaker row takes. A port of
//! `MECTFbankSpeakerEmbedding` (Zyphra/ZONOS3-Mini, `model/src/zonos3/models/mect_speaker.py`
//! over the vendored AntSpeaker `VPR` / `MECT_B2` / `ASTP`), on the tensors and the layout
//! `tools/convert_gguf.py speaker-encoder` documents.
//!
//! 1. Front end: mono audio at 16 kHz (resampled with torchaudio's windowed-sinc kernel), Kaldi
//!    filter banks (25 ms frames every 10 ms, DC removed, pre-emphasis 0.97, Hamming window,
//!    512-point power spectrum, 80 mel bins, log), minus each bin's mean over time.
//! 2. A 3x3 convolution to 32 channels, then 14 blocks in 4 stages. A block runs residual units
//!    of grouped 3x3 and 1x1 convolutions on the (mel, time, channel) grid, flattens mel and
//!    channels to 2560 values per frame, and runs one transformer layer of width 64 on them
//!    (a convolutional position embedding, 4-head attention over the whole clip, a 4-expert
//!    feed-forward), added back to the block's input. Stages 2 to 4 halve the mel axis and double
//!    the channels; their later blocks run at half the frame rate.
//! 3. The four stages' outputs are mixed, pooled over time into an attention-weighted mean and
//!    standard deviation per value, batch-normalized and projected to 192 values (not normalized).
//!
//! Activations are stored `[mel][time][channel]`.

use super::file::{Gg, Mat, add_bias};
use crate::kernels::for_chunks;
use crate::sd::ops::gelu_erf;
use anyhow::{Result, bail};

/// A scale and a shift per channel (a folded batch norm), or a LayerNorm's weight and bias.
type Affine = (Vec<f32>, Vec<f32>);
/// A linear layer (or 1x1 convolution) and its bias.
type Dense = (Mat, Vec<f32>);

/// A grouped 3x3 convolution without bias: input channel `g * a + j` (slot `j` of group `g`)
/// feeds the output channels `[g * b, (g + 1) * b)`.
struct Conv3 {
    /// Per slot, per tap (`kh * 3 + kw`): the weight of every output channel.
    w: Vec<Vec<f32>>,
    ic: usize,
    oc: usize,
}

struct ResUnit {
    conv1: Conv3,
    pw1: (Mat, Vec<f32>),
    bn1: (Vec<f32>, Vec<f32>),
    conv2: Conv3,
    pw2: (Mat, Vec<f32>),
    bn2: (Vec<f32>, Vec<f32>),
    /// The 1x1 shortcut with its batch norm, on the first unit of a block.
    sc: Option<(Mat, Affine)>,
}

struct Block {
    cnn: Vec<ResUnit>,
    /// (mel, time) strides of the first unit.
    stride: (usize, usize),
    reduce: Option<(Dense, Affine)>,
    rproj: (Mat, Vec<f32>),
    rproj_ln: (Vec<f32>, Vec<f32>),
    pos_dw: (Vec<f32>, Vec<f32>),
    pos_bn: (Vec<f32>, Vec<f32>),
    pos_pw: (Mat, Vec<f32>),
    q: (Mat, Vec<f32>),
    k: (Mat, Vec<f32>),
    v: (Mat, Vec<f32>),
    o: (Mat, Vec<f32>),
    ln1: (Vec<f32>, Vec<f32>),
    gate: Mat,
    experts: Mat,
    ffn_out: (Mat, Vec<f32>),
    ln2: (Vec<f32>, Vec<f32>),
    eproj: (Mat, Vec<f32>),
}

pub struct SpeakerEncoder {
    pub sample_rate: u32,
    pub max_seconds: usize,
    pub dim: usize,
    n_mels: usize,
    frame_length: usize,
    frame_shift: usize,
    n_fft: usize,
    preemphasis: f32,
    scale: f32,
    log_floor: f32,
    ln_eps: f32,
    var_floor: f32,
    heads: usize,
    n_experts: usize,
    pos_k: usize,
    window: Vec<f32>,
    mel: Vec<f32>,
    head: (Vec<f32>, Vec<f32>),
    head_bn: (Vec<f32>, Vec<f32>),
    stages: Vec<Vec<Block>>,
    dense_w: Vec<f32>,
    pool1: (Mat, Vec<f32>),
    pool2: (Mat, Vec<f32>),
    bn: (Vec<f32>, Vec<f32>),
    linear: Mat,
}

/// `x` resampled from `orig` to `new` Hz as `torchaudio.transforms.Resample` does (windowed sinc,
/// Hann window, 6 zero crossings, rolloff 0.99).
pub fn resample_sinc(x: &[f32], orig: usize, new: usize) -> Vec<f32> {
    if orig == new || x.is_empty() {
        return x.to_vec();
    }
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let g = gcd(orig, new);
    let (orig, new) = (orig / g, new / g);
    let (lpw, rolloff) = (6.0f64, 0.99f64);
    let base = orig.min(new) as f64 * rolloff;
    let width = (lpw * orig as f64 / base).ceil() as usize;
    let klen = 2 * width + orig;
    let mut kernel = vec![0f32; new * klen];
    for (p, row) in kernel.chunks_mut(klen).enumerate() {
        for (j, o) in row.iter_mut().enumerate() {
            let idx = (j as f64 - width as f64) / orig as f64;
            let t = ((-(p as f64) / new as f64 + idx) * base).clamp(-lpw, lpw);
            let window = (t * std::f64::consts::PI / lpw / 2.0).cos().powi(2);
            let t = t * std::f64::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            *o = (sinc * window * base / orig as f64) as f32;
        }
    }
    let blocks = x.len() / orig + 1;
    let mut padded = vec![0f32; (blocks - 1) * orig + klen];
    padded[width..width + x.len()].copy_from_slice(x);
    let mut out = vec![0f32; blocks * new];
    for_chunks(&mut out, new, true, |b, o| {
        let s = &padded[b * orig..b * orig + klen];
        for (p, o) in o.iter_mut().enumerate() {
            *o = crate::kernels::dot(&kernel[p * klen..(p + 1) * klen], s);
        }
    });
    out.truncate((new * x.len()).div_ceil(orig));
    out
}

/// In-place radix-2 FFT of `n` complex values (`n` a power of two).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wi, wr) = (ang * k as f64).sin_cos();
                let (a, b) = (s + k, s + k + len / 2);
                let (tr, ti) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

fn affine(x: &mut [f32], sb: &(Vec<f32>, Vec<f32>), relu: bool) {
    let c = sb.0.len();
    for_chunks(x, c, x.len() > 1 << 16, |_, r| {
        for ((v, s), b) in r.iter_mut().zip(&sb.0).zip(&sb.1) {
            *v = *v * s + b;
            if relu && *v < 0.0 {
                *v = 0.0;
            }
        }
    });
}

fn layer_norm(x: &mut [f32], wb: &(Vec<f32>, Vec<f32>), eps: f32) {
    let n = wb.0.len();
    for r in x.chunks_mut(n) {
        let mean = r.iter().sum::<f32>() / n as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let s = 1.0 / (var + eps).sqrt();
        for ((v, w), b) in r.iter_mut().zip(&wb.0).zip(&wb.1) {
            *v = (*v - mean) * s * w + b;
        }
    }
}

fn new_gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn linear(m: &(Mat, Vec<f32>), x: &[f32], t: usize) -> Vec<f32> {
    let mut y = m.0.mul(x, t, true);
    add_bias(&mut y, &m.1);
    y
}

impl Conv3 {
    fn load(g: &Gg, p: &str, ic: usize, oc: usize) -> Result<Self> {
        let mut w = Vec::new();
        while g.has(&format!("{p}.w{}", w.len())) {
            let raw = g.f32(&format!("{p}.w{}", w.len()), &[oc, 1, 3, 3])?;
            let mut taps = vec![0f32; 9 * oc];
            for o in 0..oc {
                for k in 0..9 {
                    taps[k * oc + o] = raw[o * 9 + k];
                }
            }
            w.push(taps);
        }
        if w.is_empty() || !ic.is_multiple_of(w.len()) || !oc.is_multiple_of(ic / w.len()) {
            bail!("{}: {p}: {} slots for {ic} -> {oc} channels", g.path.display(), w.len());
        }
        Ok(Self { w, ic, oc })
    }

    /// `x` (`[f, t, ic]`) with padding 1 and strides `(sf, st)`: the output and its (f, t).
    fn forward(&self, x: &[f32], f: usize, t: usize, sf: usize, st: usize) -> (Vec<f32>, usize, usize) {
        let (ic, oc, a) = (self.ic, self.oc, self.w.len());
        let b = oc / (ic / a);
        let (fo, to) = ((f - 1) / sf + 1, (t - 1) / st + 1);
        let mut y = vec![0f32; fo * to * oc];
        for_chunks(&mut y, to * oc, true, |fi, row| {
            let mut gathered = vec![0f32; oc];
            for (ti, o) in row.chunks_mut(oc).enumerate() {
                for kh in 0..3 {
                    let ff = fi * sf + kh;
                    if ff < 1 || ff > f {
                        continue;
                    }
                    for kw in 0..3 {
                        let tt = ti * st + kw;
                        if tt < 1 || tt > t {
                            continue;
                        }
                        let src = &x[((ff - 1) * t + tt - 1) * ic..((ff - 1) * t + tt) * ic];
                        for (j, w) in self.w.iter().enumerate() {
                            let w = &w[(kh * 3 + kw) * oc..(kh * 3 + kw + 1) * oc];
                            if a == 1 && b == 1 {
                                for ((o, w), s) in o.iter_mut().zip(w).zip(src) {
                                    *o += w * s;
                                }
                            } else {
                                for (c, g) in gathered.iter_mut().enumerate() {
                                    *g = src[(c / b) * a + j];
                                }
                                for ((o, w), s) in o.iter_mut().zip(w).zip(&gathered) {
                                    *o += w * s;
                                }
                            }
                        }
                    }
                }
            }
        });
        (y, fo, to)
    }
}

impl ResUnit {
    fn forward(&self, x: &[f32], f: usize, t: usize, stride: (usize, usize)) -> (Vec<f32>, usize, usize) {
        let (h, fo, to) = self.conv1.forward(x, f, t, stride.0, stride.1);
        let n = fo * to;
        let mut h = linear(&self.pw1, &h, n);
        affine(&mut h, &self.bn1, true);
        let (h, _, _) = self.conv2.forward(&h, fo, to, 1, 1);
        let mut h = linear(&self.pw2, &h, n);
        affine(&mut h, &self.bn2, true);
        match &self.sc {
            Some((w, bn)) => {
                // The input at the strided positions.
                let ic = self.conv1.ic;
                let mut xs = vec![0f32; n * ic];
                for fi in 0..fo {
                    for ti in 0..to {
                        let s = ((fi * stride.0) * t + ti * stride.1) * ic;
                        xs[(fi * to + ti) * ic..(fi * to + ti + 1) * ic].copy_from_slice(&x[s..s + ic]);
                    }
                }
                let mut s = w.mul(&xs, n, true);
                affine(&mut s, bn, false);
                crate::kernels::add_inplace(&mut h, &s);
            }
            None => crate::kernels::add_inplace(&mut h, x),
        }
        (h, fo, to)
    }
}

/// `[f, t, c]` to `[t, f * c]` (value `f * c + c'` of a frame) and back.
fn flatten(x: &[f32], f: usize, t: usize, c: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for fi in 0..f {
        for ti in 0..t {
            y[ti * f * c + fi * c..ti * f * c + (fi + 1) * c].copy_from_slice(&x[(fi * t + ti) * c..(fi * t + ti + 1) * c]);
        }
    }
    y
}

fn unflatten(x: &[f32], f: usize, t: usize, c: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for fi in 0..f {
        for ti in 0..t {
            y[(fi * t + ti) * c..(fi * t + ti + 1) * c].copy_from_slice(&x[ti * f * c + fi * c..ti * f * c + (fi + 1) * c]);
        }
    }
    y
}

impl SpeakerEncoder {
    pub(crate) fn load(g: &Gg) -> Result<Self> {
        let n_mels = g.usize("n_mels")?;
        let n_fft = g.usize("n_fft")?;
        let frame_length = g.usize("frame_length")?;
        let (hidden, feat, c0) = (64usize, 2560usize, 32usize);
        if g.str("backbone")? != "MECT_B2" || n_mels * c0 != feat {
            bail!(
                "{}: speaker encoder {:?} is not supported (MECT_B2)",
                g.path.display(),
                g.str("backbone")?
            );
        }
        let n_experts = g.usize("n_experts")?;
        let pos_k = g.usize("pos_kernel")?;
        let lin = |p: &str, n_out: usize, n_in: usize| -> Result<(Mat, Vec<f32>)> {
            // 1x1 convolutions are stored with their kernel dimensions.
            let name = format!("{p}.weight");
            let w = if g.shape(&name)?.len() == 4 {
                Mat::from_f32(&g.f32(&name, &[n_out, n_in, 1, 1])?, n_out, n_in)?
            } else {
                g.mat(&name, n_out, n_in)?
            };
            let b = if g.has(&format!("{p}.bias")) {
                g.f32(&format!("{p}.bias"), &[n_out])?
            } else {
                vec![0f32; n_out]
            };
            Ok((w, b))
        };
        let pair = |p: &str, a: &str, b: &str, n: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((g.f32(&format!("{p}.{a}"), &[n])?, g.f32(&format!("{p}.{b}"), &[n])?))
        };
        // (blocks, channels, expansion) per stage; every stage but the first halves the mel
        // axis in its first block and the frame rate inside its later ones.
        let spec = [(2usize, 32usize, 4usize), (4, 64, 2), (5, 128, 2), (3, 256, 1)];
        let mut stages = Vec::new();
        let mut c_in = c0;
        for (s, &(blocks, out_c, expansion)) in spec.iter().enumerate() {
            let mut stage = Vec::new();
            for kb in 0..blocks {
                let p = format!("fe.b{}.{kb}", s + 1);
                let stride = match (s, kb) {
                    (0, _) => (1, 1),
                    (_, 0) => (2, 1),
                    _ => (1, 2),
                };
                let oc = out_c * expansion;
                let mut cnn = Vec::new();
                while g.has(&format!("{p}.cnn.{}.conv1.w0", cnn.len())) {
                    let r = cnn.len();
                    let q = format!("{p}.cnn.{r}");
                    let ic = if r == 0 { c_in } else { oc };
                    let sc = if g.has(&format!("{q}.sc.weight")) {
                        Some((
                            lin(&format!("{q}.sc"), oc, ic)?.0,
                            pair(&format!("{q}.sc.bn"), "scale", "shift", oc)?,
                        ))
                    } else {
                        None
                    };
                    cnn.push(ResUnit {
                        conv1: Conv3::load(g, &format!("{q}.conv1"), ic, oc)?,
                        pw1: lin(&format!("{q}.pw1"), oc, oc)?,
                        bn1: pair(&format!("{q}.bn1"), "scale", "shift", oc)?,
                        conv2: Conv3::load(g, &format!("{q}.conv2"), oc, oc)?,
                        pw2: lin(&format!("{q}.pw2"), oc, oc)?,
                        bn2: pair(&format!("{q}.bn2"), "scale", "shift", oc)?,
                        sc,
                    });
                }
                let c_out = c_in * stride.0;
                let reduce = if g.has(&format!("{p}.reduce.weight")) {
                    Some((
                        lin(&format!("{p}.reduce"), c_out, oc)?,
                        pair(&format!("{p}.reduce.bn"), "scale", "shift", c_out)?,
                    ))
                } else {
                    None
                };
                stage.push(Block {
                    cnn,
                    stride,
                    reduce,
                    rproj: lin(&format!("{p}.rproj"), hidden, feat)?,
                    rproj_ln: pair(&format!("{p}.rproj.ln"), "weight", "bias", hidden)?,
                    pos_dw: (
                        g.f32(&format!("{p}.pos.dw.weight"), &[hidden, pos_k])?,
                        g.f32(&format!("{p}.pos.dw.bias"), &[hidden])?,
                    ),
                    pos_bn: pair(&format!("{p}.pos.bn"), "scale", "shift", hidden)?,
                    pos_pw: lin(&format!("{p}.pos.pw"), hidden, hidden)?,
                    q: lin(&format!("{p}.attn.q"), hidden, hidden)?,
                    k: lin(&format!("{p}.attn.k"), hidden, hidden)?,
                    v: lin(&format!("{p}.attn.v"), hidden, hidden)?,
                    o: lin(&format!("{p}.attn.o"), hidden, hidden)?,
                    ln1: pair(&format!("{p}.ln1"), "weight", "bias", hidden)?,
                    gate: g.mat(&format!("{p}.moe.gate.weight"), n_experts, hidden)?,
                    experts: g.mat(&format!("{p}.moe.experts.weight"), n_experts * hidden, hidden)?,
                    ffn_out: lin(&format!("{p}.ffn_out"), hidden, hidden)?,
                    ln2: pair(&format!("{p}.ln2"), "weight", "bias", hidden)?,
                    eproj: lin(&format!("{p}.eproj"), feat, hidden)?,
                });
                c_in = c_out;
            }
            stages.push(stage);
        }
        let dim = g.usize("embedding_length")?;
        Ok(Self {
            sample_rate: g.usize("sample_rate")? as u32,
            max_seconds: g.usize("max_seconds")?,
            dim,
            n_mels,
            frame_length,
            frame_shift: g.usize("frame_shift")?,
            n_fft,
            preemphasis: g.f64("preemphasis")? as f32,
            scale: g.f64("waveform_scale")? as f32,
            log_floor: g.f64("log_floor")? as f32,
            ln_eps: g.f64("ln_eps")? as f32,
            var_floor: g.f64("pool_var_floor")? as f32,
            heads: g.usize("n_heads")?,
            n_experts,
            pos_k,
            window: g.f32("frontend.window", &[frame_length])?,
            mel: g.f32("frontend.mel_banks", &[n_mels, n_fft / 2 + 1])?,
            head: (g.f32("fe.head.conv.weight", &[c0, 1, 3, 3])?, g.f32("fe.head.conv.bias", &[c0])?),
            head_bn: pair("fe.head.bn", "scale", "shift", c0)?,
            stages,
            dense_w: g.f32("fe.dense_w", &[4, feat])?,
            pool1: lin("pool.l1", 128, feat)?,
            pool2: lin("pool.l2", feat, 128)?,
            bn: pair("bn", "scale", "shift", 2 * feat)?,
            linear: g.mat("linear.weight", dim, 2 * feat)?,
        })
    }

    /// The log mel filter-bank energies `[mel][time]` of 16 kHz audio, mean-normalized over time.
    fn fbank(&self, wav: &[f32]) -> Result<(Vec<f32>, usize)> {
        let (fl, fs, nfft, nm) = (self.frame_length, self.frame_shift, self.n_fft, self.n_mels);
        if wav.len() < fl {
            bail!(
                "the reference clip is shorter than one analysis frame ({} ms)",
                fl * 1000 / self.sample_rate as usize
            );
        }
        let t = 1 + (wav.len() - fl) / fs;
        let bins = nfft / 2 + 1;
        let mut feats = vec![0f32; t * nm];
        for_chunks(&mut feats, nm, true, |i, out| {
            let mut x: Vec<f32> = wav[i * fs..i * fs + fl].iter().map(|v| v * self.scale).collect();
            let mean = x.iter().sum::<f32>() / fl as f32;
            x.iter_mut().for_each(|v| *v -= mean);
            for j in (1..fl).rev() {
                x[j] -= self.preemphasis * x[j - 1];
            }
            x[0] -= self.preemphasis * x[0];
            let mut re = vec![0f64; nfft];
            let mut im = vec![0f64; nfft];
            for ((r, v), w) in re.iter_mut().zip(&x).zip(&self.window) {
                *r = (v * w) as f64;
            }
            fft(&mut re, &mut im);
            let power: Vec<f32> = (0..bins).map(|k| (re[k] * re[k] + im[k] * im[k]) as f32).collect();
            for (m, o) in out.iter_mut().enumerate() {
                *o = crate::kernels::dot(&self.mel[m * bins..(m + 1) * bins], &power)
                    .max(self.log_floor)
                    .ln();
            }
        });
        // [mel][time], each mel bin minus its mean over time.
        let mut x = vec![0f32; nm * t];
        for m in 0..nm {
            let mean = (0..t).map(|i| feats[i * nm + m]).sum::<f32>() / t as f32;
            for i in 0..t {
                x[m * t + i] = feats[i * nm + m] - mean;
            }
        }
        Ok((x, t))
    }

    fn block(&self, b: &Block, x: &[f32], f: usize, t: usize, c: usize) -> (Vec<f32>, usize, usize) {
        let (hd, feat) = (64usize, f * c);
        let (mut h, mut fo, mut to) = (Vec::new(), f, t);
        for (r, u) in b.cnn.iter().enumerate() {
            let (y, f2, t2) = u.forward(if r == 0 { x } else { &h }, fo, to, if r == 0 { b.stride } else { (1, 1) });
            (h, fo, to) = (y, f2, t2);
        }
        let mut co = h.len() / (fo * to);
        if let Some((w, bn)) = &b.reduce {
            h = linear(w, &h, fo * to);
            affine(&mut h, bn, false);
            co = bn.0.len();
        }
        let x2 = flatten(&h, fo, to, co);
        // The transformer layer on [frames, 64].
        let mut y = linear(&b.rproj, &x2, to);
        layer_norm(&mut y, &b.rproj_ln, self.ln_eps);
        let (k, half) = (self.pos_k, self.pos_k / 2);
        let mut pos = vec![0f32; to * hd];
        let yr = &y;
        for_chunks(&mut pos, hd, to > 64, |i, o| {
            o.copy_from_slice(&b.pos_dw.1);
            for j in 0..k {
                if i + j < half || i + j - half >= to {
                    continue;
                }
                let src = &yr[(i + j - half) * hd..(i + j - half + 1) * hd];
                for (ch, (o, s)) in o.iter_mut().zip(src).enumerate() {
                    *o += b.pos_dw.0[ch * k + j] * s;
                }
            }
        });
        affine(&mut pos, &b.pos_bn, false);
        pos.iter_mut().for_each(|v| *v = gelu_erf(*v));
        crate::kernels::add_inplace(&mut y, &linear(&b.pos_pw, &pos, to));
        let (nh, dh) = (self.heads, hd / self.heads);
        let (mut q, kk, vv) = (linear(&b.q, &y, to), linear(&b.k, &y, to), linear(&b.v, &y, to));
        let scale = (dh as f32).powf(-0.5);
        q.iter_mut().for_each(|v| *v *= scale);
        let mut att = vec![0f32; to * hd];
        for_chunks(&mut att, hd, true, |i, o| {
            let mut w = vec![0f32; to];
            for h in 0..nh {
                let qr = &q[i * hd + h * dh..i * hd + (h + 1) * dh];
                for (j, w) in w.iter_mut().enumerate() {
                    *w = crate::kernels::dot(qr, &kk[j * hd + h * dh..j * hd + (h + 1) * dh]);
                }
                super::lm::softmax(&mut w);
                let oh = &mut o[h * dh..(h + 1) * dh];
                for (j, &p) in w.iter().enumerate() {
                    crate::kernels::axpy(oh, p, &vv[j * hd + h * dh..j * hd + (h + 1) * dh]);
                }
            }
        });
        crate::kernels::add_inplace(&mut y, &linear(&b.o, &att, to));
        layer_norm(&mut y, &b.ln1, self.ln_eps);
        let gate = b.gate.mul(&y, to, true);
        let ex = b.experts.mul(&y, to, true);
        let ne = self.n_experts;
        let mut moe = vec![0f32; to * hd];
        for (i, o) in moe.chunks_mut(hd).enumerate() {
            let mut r = gate[i * ne..(i + 1) * ne].to_vec();
            super::lm::softmax(&mut r);
            for (e, &r) in r.iter().enumerate() {
                for (o, v) in o.iter_mut().zip(&ex[(i * ne + e) * hd..(i * ne + e + 1) * hd]) {
                    *o += r * new_gelu(*v);
                }
            }
            o.iter_mut().for_each(|v| *v = new_gelu(*v));
        }
        crate::kernels::add_inplace(&mut y, &linear(&b.ffn_out, &moe, to));
        layer_norm(&mut y, &b.ln2, self.ln_eps);
        let mut e = linear(&b.eproj, &y, to);
        crate::kernels::add_inplace(&mut e, &x2);
        // Back to the input's frame rate, plus the block's input.
        let xin = flatten(x, f, t, c);
        let mut out = vec![0f32; t * feat];
        for (i, o) in out.chunks_mut(feat).enumerate() {
            let src = &e[(i / b.stride.1) * feat..(i / b.stride.1 + 1) * feat];
            for ((o, s), x) in o.iter_mut().zip(src).zip(&xin[i * feat..(i + 1) * feat]) {
                *o = s + x;
            }
        }
        (unflatten(&out, fo, t, co), fo, co)
    }

    /// The speaker embedding of mono audio at `sample_rate` (at most the first 60 s are used).
    pub fn embed(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let mut wav = resample_sinc(pcm, sample_rate as usize, self.sample_rate as usize);
        wav.truncate(self.max_seconds * self.sample_rate as usize);
        let (fb, t) = self.fbank(&wav)?;
        let nm = self.n_mels;
        // The head: a 3x3 convolution of the one input channel with bias, then batch norm.
        let c0 = self.head.1.len();
        let mut x = vec![0f32; nm * t * c0];
        for_chunks(&mut x, t * c0, true, |fi, row| {
            for (ti, o) in row.chunks_mut(c0).enumerate() {
                o.copy_from_slice(&self.head.1);
                for kh in 0..3 {
                    for kw in 0..3 {
                        let (ff, tt) = (fi + kh, ti + kw);
                        if ff < 1 || ff > nm || tt < 1 || tt > t {
                            continue;
                        }
                        let s = fb[(ff - 1) * t + tt - 1];
                        for (c, o) in o.iter_mut().enumerate() {
                            *o += self.head.0[c * 9 + kh * 3 + kw] * s;
                        }
                    }
                }
            }
        });
        affine(&mut x, &self.head_bn, false);
        let (mut f, mut c) = (nm, c0);
        let feat = f * c;
        let mut mixed = vec![0f32; t * feat];
        for (s, stage) in self.stages.iter().enumerate() {
            for b in stage {
                (x, f, c) = self.block(b, &x, f, t, c);
            }
            let w = &self.dense_w[s * feat..(s + 1) * feat];
            for (o, r) in mixed.chunks_mut(feat).zip(flatten(&x, f, t, c).chunks(feat)) {
                for ((o, r), w) in o.iter_mut().zip(r).zip(w) {
                    *o += w * r;
                }
            }
        }
        // Attentive statistics pooling: per value, a softmax over time.
        let mut u = linear(&self.pool1, &mixed, t);
        u.iter_mut().for_each(|v| *v = v.tanh());
        let s = linear(&self.pool2, &u, t);
        let mut pooled = vec![0f32; 2 * feat];
        let (mean, std) = pooled.split_at_mut(feat);
        let mut col = vec![0f32; t];
        for j in 0..feat {
            for (i, c) in col.iter_mut().enumerate() {
                *c = s[i * feat + j];
            }
            super::lm::softmax(&mut col);
            let (mut m, mut sq) = (0f32, 0f32);
            for (i, &a) in col.iter().enumerate() {
                let v = mixed[i * feat + j];
                m += a * v;
                sq += a * v * v;
            }
            mean[j] = m;
            std[j] = (sq - m * m).max(self.var_floor).sqrt();
        }
        affine(&mut pooled, &self.bn, false);
        Ok(self.linear.mul(&pooled, 1, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_of_a_sine_has_one_bin() {
        let n = 64;
        let mut re: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / n as f64).cos())
            .collect();
        let mut im = vec![0f64; n];
        fft(&mut re, &mut im);
        for k in 0..n {
            let want = if k == 5 || k == n - 5 { n as f64 / 2.0 } else { 0.0 };
            assert!((re[k] - want).abs() < 1e-9 && im[k].abs() < 1e-9, "bin {k}");
        }
    }

    #[test]
    fn resampling_keeps_a_low_tone() {
        let x: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 44100.0).sin())
            .collect();
        let y = resample_sinc(&x, 44100, 16000);
        assert_eq!(y.len(), 1600);
        for (i, v) in y.iter().enumerate().skip(50).take(1500) {
            let want = (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin();
            assert!((v - want).abs() < 2e-3, "sample {i}: {v} vs {want}");
        }
    }
}
