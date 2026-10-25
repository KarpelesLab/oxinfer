//! Gemma 4 audio: log-mel feature extraction (USM style), the conformer encoder with chunked
//! local attention and relative position bias, and the projection into the text embedding space.

use crate::kernels::*;
use crate::mm_config::AudioConfig;
use crate::pool;
use crate::safetensors::ModelFiles;
use crate::vision::{ClipLin, load_clip_lin};
use anyhow::{Context, Result, bail};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_MELS: usize = 128;
const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;

/// Windowed-sinc resampler (Hann window, 32 taps per side), good enough for speech features.
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n_out = ((x.len() as f64) / ratio).floor() as usize;
    let cutoff = (to as f64 / from as f64).min(1.0) * 0.95; // fraction of the input Nyquist
    let taps = 32i64;
    let pi = std::f64::consts::PI;
    (0..n_out)
        .map(|i| {
            let center = i as f64 * ratio;
            let c0 = center.floor() as i64;
            let (mut acc, mut wsum) = (0f64, 0f64);
            for j in (c0 - taps)..=(c0 + taps) {
                if j < 0 || j >= x.len() as i64 {
                    continue;
                }
                let d = j as f64 - center;
                let sinc = if d.abs() < 1e-12 {
                    1.0
                } else {
                    (pi * d * cutoff).sin() / (pi * d * cutoff)
                };
                let w = 0.5 + 0.5 * (pi * d / (taps as f64 + 1.0)).cos();
                let k = cutoff * sinc * w;
                acc += k * x[j as usize] as f64;
                wsum += k;
            }
            if wsum.abs() > 1e-9 { (acc / wsum) as f32 } else { 0.0 }
        })
        .collect()
}

/// Number of mel frames the extractor produces for `n` samples.
pub fn num_mel_frames(n: usize) -> usize {
    let pad_left = FRAME / 2;
    let size = FRAME + 1;
    if n + pad_left < size {
        return 0;
    }
    (n + pad_left - size) / HOP + 1
}

/// Number of audio soft tokens after the two stride-2 subsampling convolutions.
#[allow(dead_code)]
pub fn num_audio_tokens(n_samples: usize) -> usize {
    let mut t = num_mel_frames(n_samples);
    for _ in 0..2 {
        t = (t + 2 - 3) / 2 + 1;
    }
    t
}

fn hertz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}
fn mel_to_hertz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// HTK triangular mel filterbank `[FFT/2+1, N_MELS]` (transformers `mel_filter_bank`, norm None).
fn mel_filters() -> Vec<f32> {
    let bins = FFT / 2 + 1;
    let (mmin, mmax) = (hertz_to_mel(0.0), hertz_to_mel(8000.0));
    let mel_freqs: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mmin + (mmax - mmin) * i as f64 / (N_MELS + 1) as f64)
        .collect();
    let filt: Vec<f64> = mel_freqs.iter().map(|&m| mel_to_hertz(m)).collect();
    let fft_freqs: Vec<f64> = (0..bins).map(|i| (SAMPLE_RATE / 2) as f64 * i as f64 / (bins - 1) as f64).collect();
    let mut out = vec![0f32; bins * N_MELS];
    for (b, &f) in fft_freqs.iter().enumerate() {
        for m in 0..N_MELS {
            let down = (f - filt[m]) / (filt[m + 1] - filt[m]);
            let up = (filt[m + 2] - f) / (filt[m + 2] - filt[m + 1]);
            out[b * N_MELS + m] = down.min(up).max(0.0) as f32;
        }
    }
    out
}

/// In-place iterative radix-2 complex FFT (re, im arrays of power-of-two length).
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0usize;
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
        let (wr, wi) = (ang.cos() as f32, ang.sin() as f32);
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (ar, ai) = (re[start + k], im[start + k]);
                let (br, bi) = (re[start + k + len / 2], im[start + k + len / 2]);
                let (tr, ti) = (br * cr - bi * ci, br * ci + bi * cr);
                re[start + k] = ar + tr;
                im[start + k] = ai + ti;
                re[start + k + len / 2] = ar - tr;
                im[start + k + len / 2] = ai - ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
        len <<= 1;
    }
}

/// Log-mel features `[T, 128]` for 16 kHz mono audio in [-1, 1].
pub fn mel_features(pcm: &[f32]) -> Vec<f32> {
    let t = num_mel_frames(pcm.len());
    let filters = mel_filters();
    let window: Vec<f32> = (0..FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FRAME as f64).cos())
        .map(|v| v as f32)
        .collect();
    let pad_left = FRAME / 2;
    let mut out = vec![0f32; t * N_MELS];
    let mut re = vec![0f32; FFT];
    let mut im = vec![0f32; FFT];
    for f in 0..t {
        re.fill(0.0);
        im.fill(0.0);
        for i in 0..FRAME {
            let idx = f * HOP + i;
            let s = if idx < pad_left {
                0.0
            } else {
                pcm.get(idx - pad_left).copied().unwrap_or(0.0)
            };
            re[i] = s * window[i];
        }
        fft(&mut re, &mut im);
        let row = &mut out[f * N_MELS..(f + 1) * N_MELS];
        for b in 0..FFT / 2 + 1 {
            let mag = (re[b] * re[b] + im[b] * im[b]).sqrt();
            if mag != 0.0 {
                for m in 0..N_MELS {
                    row[m] += mag * filters[b * N_MELS + m];
                }
            }
        }
        for v in row.iter_mut() {
            *v = (*v + 1e-3).ln();
        }
    }
    out
}

struct Ff<'a> {
    pre: &'a [u16],
    post: &'a [u16],
    l1: ClipLin<'a>,
    l2: ClipLin<'a>,
}

struct ALayer<'a> {
    ff1: Ff<'a>,
    ff2: Ff<'a>,
    norm_pre_attn: &'a [u16],
    norm_post_attn: &'a [u16],
    norm_out: &'a [u16],
    q: ClipLin<'a>,
    k: ClipLin<'a>,
    v: ClipLin<'a>,
    post: ClipLin<'a>,
    rel_k: &'a [u16], // [H, H]
    per_dim_scale: &'a [u16],
    lc_pre: &'a [u16],
    lc_start: ClipLin<'a>,
    lc_conv: &'a [u16], // [H, 1, K]
    lc_norm: &'a [u16],
    lc_end: ClipLin<'a>,
}

pub struct AudioTower<'a> {
    pub cfg: AudioConfig,
    conv0: &'a [u16],      // [C0, 1, 3, 3]
    norm0: &'a [u16],      // [C0]
    conv1: &'a [u16],      // [C1, C0, 3, 3]
    norm1: &'a [u16],      // [C1]
    input_proj: &'a [u16], // [H, (128/4)*C1]
    layers: Vec<ALayer<'a>>,
    out_w: &'a [u16],      // [out_dims, H]
    out_b: &'a [u16],      // [out_dims]
    embed_proj: &'a [u16], // [text_hidden, out_dims]
    pub text_hidden: usize,
}

impl<'a> AudioTower<'a> {
    pub fn load(cfg: AudioConfig, files: &'a ModelFiles, text_hidden: usize) -> Result<Self> {
        let pre = "model.audio_tower.";
        let h = cfg.hidden_size;
        let get = |name: &str, shape: &[usize]| -> Result<&'a [u16]> {
            let full = format!("{pre}{name}");
            let info = files.info(&full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape != shape {
                bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
            }
            files.bf16(&full)
        };
        let conv = |name: &str, shape: &[usize]| files.bf16_conv(&format!("{pre}{name}"), shape);
        let (c0, c1) = (cfg.subsampling_conv_channels[0], cfg.subsampling_conv_channels[1]);
        let clipped = cfg.use_clipped_linears;
        let mut layers = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{pre}layers.{l}.");
            let cl = |n: &str, o: usize, i: usize| load_clip_lin(files, &format!("{lp}{n}"), o, i, clipped);
            let g = |n: &str, s: &[usize]| get(&format!("layers.{l}.{n}"), s);
            let ff = |p: &str| -> Result<Ff<'a>> {
                Ok(Ff {
                    pre: g(&format!("{p}.pre_layer_norm.weight"), &[h])?,
                    post: g(&format!("{p}.post_layer_norm.weight"), &[h])?,
                    l1: cl(&format!("{p}.ffw_layer_1"), 4 * h, h)?,
                    l2: cl(&format!("{p}.ffw_layer_2"), h, 4 * h)?,
                })
            };
            layers.push(ALayer {
                ff1: ff("feed_forward1")?,
                ff2: ff("feed_forward2")?,
                norm_pre_attn: g("norm_pre_attn.weight", &[h])?,
                norm_post_attn: g("norm_post_attn.weight", &[h])?,
                norm_out: g("norm_out.weight", &[h])?,
                q: cl("self_attn.q_proj", h, h)?,
                k: cl("self_attn.k_proj", h, h)?,
                v: cl("self_attn.v_proj", h, h)?,
                post: cl("self_attn.post", h, h)?,
                rel_k: g("self_attn.relative_k_proj.weight", &[h, h])?,
                per_dim_scale: g("self_attn.per_dim_scale", &[h / cfg.num_attention_heads])?,
                lc_pre: g("lconv1d.pre_layer_norm.weight", &[h])?,
                lc_start: cl("lconv1d.linear_start", 2 * h, h)?,
                lc_conv: conv(
                    &format!("layers.{l}.lconv1d.depthwise_conv1d.weight"),
                    &[h, 1, cfg.conv_kernel_size],
                )?,
                lc_norm: g("lconv1d.conv_norm.weight", &[h])?,
                lc_end: cl("lconv1d.linear_end", h, h)?,
            });
        }
        let embed_name = "model.embed_audio.embedding_projection.weight";
        let embed_proj = {
            let info = files.info(embed_name).context("missing embed_audio projection")?;
            if info.shape != [text_hidden, cfg.output_proj_dims] {
                bail!("embed_audio has shape {:?}", info.shape);
            }
            files.bf16(embed_name)?
        };
        Ok(Self {
            conv0: conv("subsample_conv_projection.layer0.conv.weight", &[c0, 1, 3, 3])?,
            norm0: get("subsample_conv_projection.layer0.norm.weight", &[c0])?,
            conv1: conv("subsample_conv_projection.layer1.conv.weight", &[c1, c0, 3, 3])?,
            norm1: get("subsample_conv_projection.layer1.norm.weight", &[c1])?,
            input_proj: get("subsample_conv_projection.input_proj_linear.weight", &[h, (N_MELS / 4) * c1])?,
            layers,
            out_w: get("output_proj.weight", &[cfg.output_proj_dims, h])?,
            out_b: get("output_proj.bias", &[cfg.output_proj_dims])?,
            embed_proj,
            text_hidden,
            cfg,
        })
    }

    /// Encodes log-mel features `[t, 128]` into soft tokens in the text embedding space.
    pub fn encode(&self, feats: &[f32], t: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps as f32;
        let (c0, c1) = (cfg.subsampling_conv_channels[0], cfg.subsampling_conv_channels[1]);

        // Subsampling: two 3x3 stride-2 convs over (time, mel), each followed by LayerNorm over
        // channels (weight only) and ReLU.
        let (x1, t1, f1) = conv2d_s2(feats, 1, t, N_MELS, self.conv0, c0);
        let x1 = ln_relu(x1, c0, t1 * f1, self.norm0, eps);
        let (x2, t2, f2) = conv2d_s2(&x1, c0, t1, f1, self.conv1, c1);
        let x2 = ln_relu(x2, c1, t2 * f2, self.norm1, eps);
        // [C, T, F] -> [T, F*C]
        let mut flat = vec![0f32; t2 * f2 * c1];
        for tt in 0..t2 {
            for ff in 0..f2 {
                for c in 0..c1 {
                    flat[tt * (f2 * c1) + ff * c1 + c] = x2[(c * t2 + tt) * f2 + ff];
                }
            }
        }
        let n = t2;
        let mut x = vec![0f32; n * h];
        matmul(self.input_proj, h, f2 * c1, &flat, n, &mut x);

        // Relative position embeddings for distances 0..=max_past (sin..., cos...).
        let nh = cfg.num_attention_heads;
        let hd = h / nh;
        let max_past = cfg.attention_context_left - 1;
        let n_ts = h / 2;
        let log_inc = (10000f64).ln() / (n_ts - 1).max(1) as f64;
        let mut pe = vec![0f32; (max_past + 1) * h];
        for d in 0..=max_past {
            for i in 0..n_ts {
                let a = d as f64 * (-(i as f64) * log_inc).exp();
                pe[d * h + i] = a.sin() as f32;
                pe[d * h + n_ts + i] = a.cos() as f32;
            }
        }
        let q_scale = (hd as f32).powf(-0.5) / std::f32::consts::LN_2;
        let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        let cap = cfg.attention_logit_cap as f32;
        let window = max_past; // keys with 0 <= q - kv < context_left - 1

        let mut buf = vec![0f32; n * h];
        let mut q = vec![0f32; n * h];
        let mut k = vec![0f32; n * h];
        let mut v = vec![0f32; n * h];
        let mut att = vec![0f32; n * h];
        let mut relk = vec![0f32; (max_past + 1) * h];
        for layer in &self.layers {
            self.feed_forward(&layer.ff1, &mut x, n);

            // Attention block.
            let residual = x.clone();
            rmsnorm_rows(&x, h, Some(layer.norm_pre_attn), eps, &mut buf);
            layer.q.apply(&buf, n, &mut q);
            layer.k.apply(&buf, n, &mut k);
            layer.v.apply(&buf, n, &mut v);
            let pds: Vec<f32> = (0..hd).map(|i| q_scale * softplus(bf16_to_f32(layer.per_dim_scale[i]))).collect();
            for row in q.chunks_mut(hd) {
                for (a, s) in row.iter_mut().zip(&pds) {
                    *a *= s;
                }
            }
            scale_inplace(&mut k, k_scale);
            matmul(layer.rel_k, h, h, &pe, max_past + 1, &mut relk);
            local_attention(&q, &k, &v, &relk, n, nh, hd, window, cap, &mut att);
            layer.post.apply(&att, n, &mut buf);
            rmsnorm_rows_inplace(&mut buf, h, Some(layer.norm_post_attn), eps);
            for (xi, (r, b)) in x.iter_mut().zip(residual.iter().zip(&buf)) {
                *xi = r + b;
            }

            // Lightweight convolution block.
            let residual = x.clone();
            rmsnorm_rows(&x, h, Some(layer.lc_pre), eps, &mut buf);
            let mut gl = vec![0f32; n * 2 * h];
            layer.lc_start.apply(&buf, n, &mut gl);
            let mut a = vec![0f32; n * h];
            for tt in 0..n {
                for c in 0..h {
                    let g = gl[tt * 2 * h + h + c];
                    a[tt * h + c] = gl[tt * 2 * h + c] * sigmoid(g);
                }
            }
            let ks = cfg.conv_kernel_size;
            for tt in 0..n {
                for c in 0..h {
                    let mut s = 0f32;
                    for j in 0..ks {
                        let src = tt as isize - (ks as isize - 1) + j as isize;
                        if src >= 0 {
                            s += bf16_to_f32(layer.lc_conv[c * ks + j]) * a[src as usize * h + c];
                        }
                    }
                    buf[tt * h + c] = s;
                }
            }
            rmsnorm_rows_inplace(&mut buf, h, Some(layer.lc_norm), eps);
            for vv in buf.iter_mut() {
                *vv = silu(*vv);
            }
            layer.lc_end.apply(&buf, n, &mut a);
            for (xi, (r, b)) in x.iter_mut().zip(residual.iter().zip(&a)) {
                *xi = r + b;
            }

            self.feed_forward(&layer.ff2, &mut x, n);
            rmsnorm_rows_inplace(&mut x, h, Some(layer.norm_out), eps);
        }

        // Output projection (with bias), embedder norm and projection to text space.
        let od = cfg.output_proj_dims;
        let mut y = vec![0f32; n * od];
        matmul(self.out_w, od, h, &x, n, &mut y);
        for row in y.chunks_mut(od) {
            for (v, &b) in row.iter_mut().zip(self.out_b) {
                *v += bf16_to_f32(b);
            }
        }
        rmsnorm_rows_inplace(&mut y, od, None, eps);
        let mut out = vec![0f32; n * self.text_hidden];
        matmul(self.embed_proj, self.text_hidden, od, &y, n, &mut out);
        out
    }

    fn feed_forward(&self, ff: &Ff, x: &mut [f32], n: usize) {
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let mut hn = vec![0f32; n * h];
        rmsnorm_rows(x, h, Some(ff.pre), eps, &mut hn);
        let mut mid = vec![0f32; n * 4 * h];
        ff.l1.apply(&hn, n, &mut mid);
        for v in mid.iter_mut() {
            *v = silu(*v);
        }
        ff.l2.apply(&mid, n, &mut hn);
        rmsnorm_rows_inplace(&mut hn, h, Some(ff.post), eps);
        let w = self.cfg.residual_weight as f32;
        for (xi, b) in x.iter_mut().zip(&hn) {
            *xi += w * b;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// 3x3 stride-2 pad-1 conv over an input laid out `[cin, t, f]`, weights `[cout, cin, 3, 3]`.
/// Returns `([cout, t_out, f_out], t_out, f_out)`.
fn conv2d_s2(x: &[f32], cin: usize, t: usize, f: usize, w: &[u16], cout: usize) -> (Vec<f32>, usize, usize) {
    let t_out = (t + 2 - 3) / 2 + 1;
    let f_out = (f + 2 - 3) / 2 + 1;
    let wf: Vec<f32> = w.iter().map(|&v| bf16_to_f32(v)).collect();
    let mut out = vec![0f32; cout * t_out * f_out];
    let op = SendPtrF(out.as_mut_ptr());
    pool::run(cout * t_out, &|idx| {
        let (co, to) = (idx / t_out, idx % t_out);
        // SAFETY: each task writes its own output row.
        let row = unsafe { std::slice::from_raw_parts_mut(op.get().add((co * t_out + to) * f_out), f_out) };
        for fo in 0..f_out {
            let mut s = 0f32;
            for ci in 0..cin {
                for kt in 0..3 {
                    let ti = (to * 2 + kt) as isize - 1;
                    if ti < 0 || ti >= t as isize {
                        continue;
                    }
                    for kf in 0..3 {
                        let fi = (fo * 2 + kf) as isize - 1;
                        if fi < 0 || fi >= f as isize {
                            continue;
                        }
                        s += wf[((co * cin + ci) * 3 + kt) * 3 + kf] * x[(ci * t + ti as usize) * f + fi as usize];
                    }
                }
            }
            row[fo] = s;
        }
    });
    (out, t_out, f_out)
}

/// LayerNorm over the channel axis of `[c, m]` (weight only, no bias) followed by ReLU.
fn ln_relu(x: Vec<f32>, c: usize, m: usize, w: &[u16], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; c * m];
    let wf: Vec<f32> = w.iter().map(|&v| bf16_to_f32(v)).collect();
    for i in 0..m {
        let mut mean = 0f32;
        for ch in 0..c {
            mean += x[ch * m + i];
        }
        mean /= c as f32;
        let mut var = 0f32;
        for ch in 0..c {
            let d = x[ch * m + i] - mean;
            var += d * d;
        }
        var /= c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ch in 0..c {
            out[ch * m + i] = ((x[ch * m + i] - mean) * inv * wf[ch]).max(0.0);
        }
    }
    out
}

/// Chunked local attention collapsed to its equivalent: each query attends to keys with
/// `0 <= q - kv < window`, with a relative-position bias `q . relk[q - kv]`, tanh soft-capped.
#[allow(clippy::too_many_arguments)]
fn local_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    relk: &[f32],
    n: usize,
    nh: usize,
    hd: usize,
    window: usize,
    cap: f32,
    out: &mut [f32],
) {
    let h = nh * hd;
    let op = SendPtrF(out.as_mut_ptr());
    pool::run(n * nh, &|idx| {
        let (t, hh) = (idx / nh, idx % nh);
        let qv = &q[t * h + hh * hd..t * h + (hh + 1) * hd];
        let lo = (t + 1).saturating_sub(window);
        let mut scores: Vec<f32> = (lo..=t)
            .map(|s| {
                let d = t - s;
                let ac = dot(qv, &k[s * h + hh * hd..s * h + (hh + 1) * hd]);
                let bd = dot(qv, &relk[d * h + hh * hd..d * h + (hh + 1) * hd]);
                cap * ((ac + bd) / cap).tanh()
            })
            .collect();
        softmax_inplace(&mut scores);
        // SAFETY: each task owns one disjoint [hd] slice of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(t * h + hh * hd), hd) };
        o.fill(0.0);
        for (s, &p) in (lo..=t).zip(&scores) {
            axpy(o, p, &v[s * h + hh * hd..s * h + (hh + 1) * hd]);
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtrF(*mut f32);
unsafe impl Send for SendPtrF {}
unsafe impl Sync for SendPtrF {}
impl SendPtrF {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}
