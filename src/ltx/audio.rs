//! LTX-2.5's audio decoding (`vae/ltx-2.5-audio-vae-bf16.safetensors`,
//! `ltx_core.model.audio_vae`): the audio VAE decoder turns the 8 x 16 latents of 25 frames a
//! second into a stereo log-mel spectrogram (64 bins, 100 frames a second), the BigVGAN v2
//! vocoder that into 16 kHz audio, and its bandwidth extension (`VocoderWithBWE`) that into
//! 48 kHz.
//!
//! - Decoder (`AudioDecoder`): 2-D convolutions over (time, mel), causal in time (two zero rows
//!   before, none after; zero padding in mel), pixel norms (eps 1e-6) and SiLU; `conv_in`, a mid
//!   block of two residual blocks, three levels of three residual blocks with nearest x2
//!   upsampling (and the first time row dropped) between them, norm, SiLU, `conv_out` to the two
//!   channels; `4 t - 3` time rows.
//! - Vocoder: the two channels' 64 bins side by side as 128 inputs, `conv_pre`, per stage a
//!   transposed convolution and the mean of three AMP blocks (anti-aliased SnakeBeta), the
//!   activation, `conv_post` to two channels, a clamp to `[-1, 1]`.
//! - BWE: the 16 kHz signal padded to whole hops of 80, its causal log-mel spectrogram (the
//!   checkpoint's STFT basis and mel filters), a second generator over it predicting a residual,
//!   added to the signal upsampled by 3 with a Hann-windowed sinc; clamped, cut to three times
//!   the vocoder's length.
//!
//! The waveforms run on the BigVGAN pieces of MiniMax-H3's audio VAE
//! ([`crate::minimax_h3::audio_vae`]).

use crate::arktts::codec::Conv as Conv1;
use crate::minimax_h3::audio_vae::{Act, Up, add, conv as conv1};
use crate::safetensors::ModelFiles;
use crate::sd::ops::Conv;
use anyhow::{Context, Result, bail};

/// A causal (in time) 3 x 3 convolution over `[t, mel, c]` rows, or a 1 x 1 one.
pub(super) struct Conv2 {
    pub(super) conv: Conv,
    k: usize,
    pub(super) cout: usize,
}

impl Conv2 {
    pub(super) fn load(f: &ModelFiles, p: &str) -> Result<Self> {
        let wn = format!("{p}.weight");
        let i = f.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        let [cout, cin, kh, kw] = i.shape[..] else {
            bail!("{wn}: {:?}, expected a 2-D kernel", i.shape);
        };
        if kh != kw || (kh != 3 && kh != 1) {
            bail!("{wn}: {kh}x{kw} kernels are not supported");
        }
        let w = f.f32(&wn)?;
        let b = f.f32(&format!("{p}.bias"))?;
        Ok(Self {
            conv: Conv::from_kernel(&w, cout, cin, kh, Some(b))?,
            k: kh,
            cout,
        })
    }

    /// `x` `[t, m, cin]` to `[t, m, cout]`.
    pub(super) fn forward(&self, x: &[f32], t: usize, m: usize) -> Vec<f32> {
        if self.k == 1 {
            return self.conv.forward(x, t, m, 1).0;
        }
        // Two zero rows before in time, none after: one more zero row in front of the centered
        // convolution, and its last output row dropped.
        let cin = x.len() / (t * m);
        let mut xp = vec![0f32; (t + 1) * m * cin];
        xp[m * cin..].copy_from_slice(x);
        let (mut y, _, _) = self.conv.forward(&xp, t + 1, m, 1);
        y.truncate(t * m * self.cout);
        y
    }
}

pub(super) fn pixel_norm_silu(x: &[f32], c: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    let par = y.len() > c * 64;
    crate::kernels::for_chunks(&mut y, c * 64, par, |_, rows| {
        for r in rows.chunks_mut(c) {
            let ms = r.iter().map(|v| v * v).sum::<f32>() / c as f32;
            let s = 1.0 / (ms + 1e-6).sqrt();
            for v in r.iter_mut() {
                let z = *v * s;
                *v = z / (1.0 + (-z).exp());
            }
        }
    });
    y
}

pub(super) struct Res {
    c1: Conv2,
    c2: Conv2,
    nin: Option<Conv2>,
}

impl Res {
    pub(super) fn load(f: &ModelFiles, p: &str) -> Result<Self> {
        let nin = format!("{p}.nin_shortcut.conv");
        Ok(Self {
            c1: Conv2::load(f, &format!("{p}.conv1.conv"))?,
            c2: Conv2::load(f, &format!("{p}.conv2.conv"))?,
            nin: if f.has(&format!("{nin}.weight")) {
                Some(Conv2::load(f, &nin)?)
            } else {
                None
            },
        })
    }

    pub(super) fn forward(&self, x: &[f32], t: usize, m: usize) -> Vec<f32> {
        let cin = x.len() / (t * m);
        let h = self.c1.forward(&pixel_norm_silu(x, cin), t, m);
        let mut h = self.c2.forward(&pixel_norm_silu(&h, self.c1.cout), t, m);
        let skip = match &self.nin {
            Some(n) => n.forward(x, t, m),
            None => x.to_vec(),
        };
        h.iter_mut().zip(&skip).for_each(|(a, &b)| *a += b);
        h
    }
}

struct Level {
    blocks: Vec<Res>,
    up: Option<Conv2>,
}

/// The `audio_vae.decoder` of the file.
pub struct AudioDecoder {
    mean: Vec<f32>,
    std: Vec<f32>,
    conv_in: Conv2,
    mid: [Res; 2],
    /// Highest level first (the order they run in).
    levels: Vec<Level>,
    conv_out: Conv2,
    pub z_channels: usize,
    pub mel_bins: usize,
}

impl AudioDecoder {
    pub fn load(f: &ModelFiles) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_str(f.metadata("config").context("LTX audio VAE: no `config` metadata")?)?;
        let dd = cfg
            .pointer("/audio_vae/model/params/ddconfig")
            .context("LTX audio VAE: no audio_vae.model.params.ddconfig")?;
        let u = |k: &str| dd.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        let n_levels = dd.get("ch_mult").and_then(|v| v.as_array()).map_or(3, |a| a.len());
        let nres = u("num_res_blocks").unwrap_or(2);
        if dd.get("causality_axis").and_then(|v| v.as_str()) != Some("height") {
            bail!("LTX audio VAE: only causality along height (time) is supported");
        }
        let p = "audio_vae.decoder";
        let mut levels = Vec::new();
        for level in (0..n_levels).rev() {
            let lp = format!("{p}.up.{level}");
            levels.push(Level {
                blocks: (0..=nres)
                    .map(|i| Res::load(f, &format!("{lp}.block.{i}")))
                    .collect::<Result<_>>()?,
                up: if level != 0 {
                    Some(Conv2::load(f, &format!("{lp}.upsample.conv.conv"))?)
                } else {
                    None
                },
            });
        }
        let z = u("z_channels").unwrap_or(8);
        let mean = f.f32("audio_vae.per_channel_statistics.mean-of-means")?;
        let std = f.f32("audio_vae.per_channel_statistics.std-of-means")?;
        let mel_bins = u("mel_bins").unwrap_or(64);
        Ok(Self {
            conv_in: Conv2::load(f, &format!("{p}.conv_in.conv"))?,
            mid: [
                Res::load(f, &format!("{p}.mid.block_1"))?,
                Res::load(f, &format!("{p}.mid.block_2"))?,
            ],
            levels,
            conv_out: Conv2::load(f, &format!("{p}.conv_out.conv"))?,
            mean,
            std,
            z_channels: z,
            mel_bins,
        })
    }

    /// The `[2, 4 t - 3, mel_bins]` spectrogram of patchified latent rows (`[t, z * f]`, channel
    /// major as the transformer's audio rows).
    pub fn decode(&self, rows: &[f32]) -> Result<Vec<f32>> {
        let w = self.mean.len();
        if rows.is_empty() || !rows.len().is_multiple_of(w) {
            bail!("LTX audio VAE: {} values are not rows of {w}", rows.len());
        }
        let t = rows.len() / w;
        let (z, f) = (self.z_channels, w / self.z_channels);
        // Un-normalized, as `[t, f, z]` rows.
        let mut x = vec![0f32; t * f * z];
        for r in 0..t {
            for c in 0..z {
                for m in 0..f {
                    let i = c * f + m;
                    x[(r * f + m) * z + c] = rows[r * w + i] * self.std[i] + self.mean[i];
                }
            }
        }
        let (mut t, mut m) = (t, f);
        let mut h = self.conv_in.forward(&x, t, m);
        for r in &self.mid {
            h = r.forward(&h, t, m);
        }
        for l in &self.levels {
            for b in &l.blocks {
                h = b.forward(&h, t, m);
            }
            if let Some(up) = &l.up {
                let c = h.len() / (t * m);
                let mut u = vec![0f32; 4 * h.len()];
                for r in 0..2 * t {
                    for j in 0..2 * m {
                        let s = ((r / 2) * m + j / 2) * c;
                        u[(r * 2 * m + j) * c..][..c].copy_from_slice(&h[s..s + c]);
                    }
                }
                let y = up.forward(&u, 2 * t, 2 * m);
                // The first time row dropped.
                let row = 2 * m * up.cout;
                h = y[row..].to_vec();
                t = 2 * t - 1;
                m *= 2;
            }
        }
        let c = h.len() / (t * m);
        let y = self.conv_out.forward(&pixel_norm_silu(&h, c), t, m);
        let oc = self.conv_out.cout;
        // `[t, m, 2]` to `[2, t, mel_bins]` (cropped or zero-padded in mel).
        let mb = self.mel_bins;
        let mut out = vec![0f32; oc * t * mb];
        for ch in 0..oc {
            for r in 0..t {
                for j in 0..mb.min(m) {
                    out[(ch * t + r) * mb + j] = y[(r * m + j) * oc + ch];
                }
            }
        }
        Ok(out)
    }
}

/// `AMPBlock1`: per dilation, `x += conv2(act2(conv1(act1(x))))`.
struct Amp {
    acts1: Vec<Act>,
    acts2: Vec<Act>,
    convs1: Vec<Conv1>,
    convs2: Vec<Conv1>,
}

impl Amp {
    fn load(f: &ModelFiles, p: &str, dilations: &[usize]) -> Result<Self> {
        let n = dilations.len();
        Ok(Self {
            acts1: (0..n).map(|i| Act::load(f, &format!("{p}.acts1.{i}"))).collect::<Result<_>>()?,
            acts2: (0..n).map(|i| Act::load(f, &format!("{p}.acts2.{i}"))).collect::<Result<_>>()?,
            convs1: dilations
                .iter()
                .enumerate()
                .map(|(i, &d)| conv1(f, &format!("{p}.convs1.{i}"), d))
                .collect::<Result<_>>()?,
            convs2: (0..n).map(|i| conv1(f, &format!("{p}.convs2.{i}"), 1)).collect::<Result<_>>()?,
        })
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut x = x.to_vec();
        for i in 0..self.convs1.len() {
            let a = self.acts1[i].apply(&x, t);
            let (h, _) = self.convs1[i].forward(&a, t);
            let a = self.acts2[i].apply(&h, t);
            let (h, _) = self.convs2[i].forward(&a, t);
            add(&mut x, &h, self.convs2[i].c_out);
        }
        x
    }
}

/// A BigVGAN v2 generator (`Vocoder` with `resblock: AMP1`).
struct Generator {
    pre: Conv1,
    ups: Vec<Up>,
    /// Per stage, one AMP block per kernel size.
    amps: Vec<Vec<Amp>>,
    act_post: Act,
    post: Conv1,
    clamp: bool,
    /// The layers as the device sees them, made on the first device pass.
    dev: std::sync::OnceLock<DevGen>,
}

impl Generator {
    fn load(f: &ModelFiles, p: &str, c: &serde_json::Value) -> Result<Self> {
        let list = |k: &str| -> Vec<usize> {
            c.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
                .unwrap_or_default()
        };
        let rates = list("upsample_rates");
        let kernels = list("resblock_kernel_sizes");
        let dil: Vec<Vec<usize>> = c
            .get("resblock_dilation_sizes")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .map(|d| {
                        d.as_array()
                            .map(|x| x.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default();
        if c.get("resblock").and_then(|v| v.as_str()) != Some("AMP1") || dil.len() != kernels.len() {
            bail!("{p}: only BigVGAN v2 AMP1 generators are supported");
        }
        let mut ups = Vec::new();
        let mut amps = Vec::new();
        for (i, &r) in rates.iter().enumerate() {
            ups.push(Up::load(f, &format!("{p}.ups.{i}"), r)?);
            amps.push(
                (0..kernels.len())
                    .map(|j| Amp::load(f, &format!("{p}.resblocks.{}", i * kernels.len() + j), &dil[j]))
                    .collect::<Result<_>>()?,
            );
        }
        Ok(Self {
            pre: conv1(f, &format!("{p}.conv_pre"), 1)?,
            ups,
            amps,
            act_post: Act::load(f, &format!("{p}.act_post"))?,
            post: conv1(f, &format!("{p}.conv_post"), 1)?,
            clamp: c.get("apply_final_activation").and_then(|v| v.as_bool()).unwrap_or(true),
            dev: std::sync::OnceLock::new(),
        })
    }

    /// A stereo spectrogram `[2, t, mels]` to `[2, t * hop]` samples (channel major).
    fn forward(&self, mel: &[f32], t: usize) -> Vec<f32> {
        let mels = mel.len() / (2 * t);
        // `[t, 2 * mels]` rows: left bins, then right bins.
        let mut x = vec![0f32; t * 2 * mels];
        for s in 0..2 {
            for r in 0..t {
                for j in 0..mels {
                    x[r * 2 * mels + s * mels + j] = mel[(s * t + r) * mels + j];
                }
            }
        }
        let (y, t) = match self.forward_device(&x, t) {
            Some(r) => r,
            None => self.forward_cpu(&x, t),
        };
        let mut out = vec![0f32; 2 * t];
        for r in 0..t {
            for s in 0..2 {
                let v = y[r * 2 + s];
                out[s * t + r] = if self.clamp { v.clamp(-1.0, 1.0) } else { v };
            }
        }
        out
    }

    /// The whole generator on the accelerator's session ([`crate::gpu::dit::LtxVocSession`]):
    /// `conv_post`'s rows `[t * hop, 2]` from `x` (`[t, 2 mels]`); None when there is none or it
    /// declined (`OXINFER_LTX_VOC_GPU=0`: the CPU path).
    fn forward_device(&self, x: &[f32], t: usize) -> Option<(Vec<f32>, usize)> {
        if std::env::var("OXINFER_LTX_VOC_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        let mut s = crate::gpu::active()?.ltx_voc_session()?;
        let dev = self.dev.get_or_init(|| DevGen::new(self));
        let mut y = Vec::new();
        if !s.generate(&dev.view(), x, t, &mut y) {
            return None;
        }
        // conv_post's rows carry its channels padded to 32: the first two.
        let cp = dev.post.cout;
        let n = y.len() / cp;
        Some((y.chunks(cp).flat_map(|r| [r[0], r[1]]).collect(), n))
    }

    /// `conv_post`'s rows `[t * hop, 2]` from `x` (`[t, 2 mels]`) on the CPU kernels.
    fn forward_cpu(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        let (mut x, mut t) = self.pre.forward(x, t);
        for (up, amps) in self.ups.iter().zip(&self.amps) {
            let (y, nt) = up.forward(&x, t);
            t = nt;
            let mut acc: Option<Vec<f32>> = None;
            for a in amps {
                let o = a.forward(&y, t);
                match &mut acc {
                    Some(v) => v.iter_mut().zip(&o).for_each(|(p, q)| *p += q),
                    None => acc = Some(o),
                }
            }
            let mut m = acc.unwrap_or(y);
            let k = 1.0 / amps.len().max(1) as f32;
            m.iter_mut().for_each(|v| *v *= k);
            x = m;
        }
        let a = self.act_post.apply(&x, t);
        self.post.forward(&a, t)
    }
}

/// `UpSample1d(ratio, window_type="hann")`: a sinc filter in a squared-cosine window.
fn hann_upsample(x: &[f32], ratio: usize) -> Vec<f32> {
    let rolloff = 0.99f64;
    let lw = 6.0f64;
    let width = (lw / rolloff).ceil() as usize;
    let k = 2 * width * ratio + 1;
    let filt: Vec<f32> = (0..k)
        .map(|i| {
            let t = (i as f64 / ratio as f64 - width as f64) * rolloff;
            let tc = t.clamp(-lw, lw);
            let win = (tc * std::f64::consts::PI / lw / 2.0).cos().powi(2);
            let sinc = if t == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
            };
            (sinc * win * rolloff / ratio as f64) as f32
        })
        .collect();
    let n = x.len();
    let pad = width;
    let pad_left = 2 * width * ratio;
    // Replicate-padded input, transposed convolution (stride `ratio`) times `ratio`, cropped.
    let xp = |i: isize| x[(i - pad as isize).clamp(0, n as isize - 1) as usize];
    let full = (n + 2 * pad - 1) * ratio + k;
    let out_len = full - pad_left - (k - ratio);
    (0..out_len)
        .map(|o| {
            let m = o + pad_left;
            let mut s = 0f32;
            // Output m takes input i at tap j = m - i * ratio.
            let i_hi = m / ratio;
            let i_lo = m.saturating_sub(k - 1).div_ceil(ratio);
            for i in i_lo..=i_hi.min(n + 2 * pad - 1) {
                s += xp(i as isize) * filt[m - i * ratio];
            }
            s * ratio as f32
        })
        .collect()
}

/// `VocoderWithBWE`.
pub struct Vocoder {
    vocoder: Generator,
    bwe: Generator,
    /// `[2 * n_freq, n_fft]`: real then imaginary rows (windowed DFT).
    basis: Vec<f32>,
    n_fft: usize,
    /// `[mels, n_freq]`.
    mel_basis: Vec<f32>,
    hop: usize,
    pub input_rate: u32,
    pub output_rate: u32,
}

impl Vocoder {
    pub fn load(f: &ModelFiles) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_str(f.metadata("config").context("LTX audio VAE: no `config` metadata")?)?;
        let v = cfg.pointer("/vocoder/vocoder").context("LTX audio VAE: no vocoder config")?;
        let b = cfg.pointer("/vocoder/bwe").context("LTX audio VAE: no bwe config")?;
        let basis_i = f.info("vocoder.mel_stft.stft_fn.forward_basis").context("missing STFT basis")?;
        let n_fft = *basis_i.shape.last().unwrap_or(&0);
        let hop = b.get("hop_length").and_then(|x| x.as_u64()).unwrap_or(80) as usize;
        Ok(Self {
            vocoder: Generator::load(f, "vocoder.vocoder", v)?,
            bwe: Generator::load(f, "vocoder.bwe_generator", b)?,
            basis: f.f32("vocoder.mel_stft.stft_fn.forward_basis")?,
            n_fft,
            mel_basis: f.f32("vocoder.mel_stft.mel_basis")?,
            hop,
            input_rate: b.get("input_sampling_rate").and_then(|x| x.as_u64()).unwrap_or(16000) as u32,
            output_rate: b.get("output_sampling_rate").and_then(|x| x.as_u64()).unwrap_or(48000) as u32,
        })
    }

    /// The causal log-mel spectrogram `[frames, mels]` of one channel.
    fn log_mel(&self, y: &[f32]) -> (Vec<f32>, usize) {
        let n_freq = self.basis.len() / (2 * self.n_fft);
        let mels = self.mel_basis.len() / n_freq;
        let left = self.n_fft - self.hop;
        let frames = (y.len() + left - self.n_fft) / self.hop + 1;
        let mut out = vec![0f32; frames * mels];
        crate::kernels::for_chunks(&mut out, mels, frames > 8, |fi, row| {
            let mut mag = vec![0f32; n_freq];
            let start = fi * self.hop;
            for (k, m) in mag.iter_mut().enumerate() {
                let (mut re, mut im) = (0f32, 0f32);
                let (br, bi) = (
                    &self.basis[k * self.n_fft..(k + 1) * self.n_fft],
                    &self.basis[(n_freq + k) * self.n_fft..(n_freq + k + 1) * self.n_fft],
                );
                for j in 0..self.n_fft {
                    let s = (start + j) as isize - left as isize;
                    if s >= 0 {
                        let v = y[s as usize];
                        re += br[j] * v;
                        im += bi[j] * v;
                    }
                }
                *m = (re * re + im * im).sqrt();
            }
            for (i, r) in row.iter_mut().enumerate() {
                let s: f32 = self.mel_basis[i * n_freq..(i + 1) * n_freq]
                    .iter()
                    .zip(&mag)
                    .map(|(a, b)| a * b)
                    .sum();
                *r = s.max(1e-5).ln();
            }
        });
        (out, frames)
    }

    /// A stereo spectrogram (`[2, t, mels]`) to `[2][samples]` at [`Self::output_rate`].
    pub fn forward(&self, mel: &[f32], t: usize) -> Vec<Vec<f32>> {
        let x = self.vocoder.forward(mel, t);
        let len = x.len() / 2;
        let out_len = len * self.output_rate as usize / self.input_rate as usize;
        let padded = len.div_ceil(self.hop) * self.hop;
        let chans: Vec<Vec<f32>> = (0..2)
            .map(|s| {
                let mut c = x[s * len..(s + 1) * len].to_vec();
                c.resize(padded, 0.0);
                c
            })
            .collect();
        // The BWE generator's input: both channels' log-mel spectrograms.
        let mels: Vec<(Vec<f32>, usize)> = chans.iter().map(|c| self.log_mel(c)).collect();
        let frames = mels[0].1;
        let mut m = Vec::with_capacity(2 * mels[0].0.len());
        for (v, _) in &mels {
            m.extend_from_slice(v);
        }
        let residual = self.bwe.forward(&m, frames);
        let rlen = residual.len() / 2;
        let ratio = (self.output_rate / self.input_rate) as usize;
        chans
            .iter()
            .enumerate()
            .map(|(s, c)| {
                let skip = hann_upsample(c, ratio);
                (0..out_len.min(rlen).min(skip.len()))
                    .map(|i| (residual[s * rlen + i] + skip[i]).clamp(-1.0, 1.0))
                    .collect()
            })
            .collect()
    }
}

/// A generator layer as the device sees it ([`crate::gpu::dit::LtxVocLin`]).
struct DevLin {
    w: Vec<u16>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    offs: Vec<i32>,
}

impl DevLin {
    /// From f32 rows `[cout][taps cin]` (tap by tap): bf16, the channels padded with zeros to
    /// multiples of 32 (the device's signals carry them: zeros throughout), the rows to 64.
    fn new(w: &[f32], b: &[f32], cin: usize, cout: usize, offs: Vec<i32>) -> Self {
        let (ci, co) = (cin.next_multiple_of(32), cout.next_multiple_of(32));
        let rows = co.next_multiple_of(64);
        let taps = offs.len();
        let mut wb = vec![0u16; rows * taps * ci];
        for o in 0..cout {
            for j in 0..taps {
                for c in 0..cin {
                    wb[(o * taps + j) * ci + c] = half::bf16::from_f32(w[(o * taps + j) * cin + c]).to_bits();
                }
            }
        }
        let mut bias = b[..cout].to_vec();
        bias.resize(rows, 0.0);
        Self {
            w: wb,
            bias,
            cin: ci,
            cout: co,
            offs,
        }
    }

    fn conv(c: &Conv1) -> Self {
        let (w, b, cin, offs) = c.taps();
        Self::new(w, b, cin, c.c_out, offs)
    }

    fn view(&self) -> crate::gpu::dit::LtxVocLin<'_> {
        crate::gpu::dit::LtxVocLin {
            w: &self.w,
            bias: &self.bias,
            cin: self.cin,
            cout: self.cout,
            offs: &self.offs,
        }
    }
}

struct DevAct {
    ab: Vec<f32>,
    filt: Vec<f32>,
    k: usize,
}

impl DevAct {
    fn new(a: &Act) -> Self {
        let (alpha, inv, up, down) = a.parts();
        // Zero alpha and inverse beta for the padding channels: they stay zero.
        let cp = alpha.len().next_multiple_of(32);
        let pad = |v: &[f32]| {
            v.iter()
                .copied()
                .chain(std::iter::repeat_n(0.0, cp - v.len()))
                .collect::<Vec<f32>>()
        };
        Self {
            ab: pad(alpha).into_iter().chain(pad(inv)).collect(),
            filt: up.iter().chain(down).copied().collect(),
            k: up.len(),
        }
    }

    fn view(&self) -> crate::gpu::dit::LtxVocAct<'_> {
        crate::gpu::dit::LtxVocAct {
            ab: &self.ab,
            filt: &self.filt,
            k: self.k,
        }
    }
}

/// One dilation of an AMP block on the device: activation, convolution, activation, convolution.
type DevRes = (DevAct, DevLin, DevAct, DevLin);

/// A generator's layers as the device sees them (made on the first device pass).
struct DevGen {
    pre: DevLin,
    stages: Vec<(Vec<DevLin>, Vec<Vec<DevRes>>)>,
    act_post: DevAct,
    post: DevLin,
}

impl DevGen {
    fn new(g: &Generator) -> Self {
        let stages = g
            .ups
            .iter()
            .zip(&g.amps)
            .map(|(up, amps)| {
                // Output step i s + ph is the causal transposed convolution's step i s + ph + crop:
                // phase (ph + crop) % s at input step i + (ph + crop) / s, whose taps read the
                // inputs from `taps - 1` steps before it.
                let (ct, crop) = up.parts();
                let (w, b, cin, cout, s, taps) = ct.phases();
                let ups = (0..s)
                    .map(|ph| {
                        let (pp, q) = ((ph + crop) % s, (ph + crop) / s);
                        let n = taps * cin;
                        let offs = (0..taps).map(|j| (q + j) as i32 - (taps - 1) as i32).collect();
                        DevLin::new(&w[pp * cout * n..(pp + 1) * cout * n], b, cin, cout, offs)
                    })
                    .collect();
                let amps = amps
                    .iter()
                    .map(|a| {
                        (0..a.convs1.len())
                            .map(|i| {
                                (
                                    DevAct::new(&a.acts1[i]),
                                    DevLin::conv(&a.convs1[i]),
                                    DevAct::new(&a.acts2[i]),
                                    DevLin::conv(&a.convs2[i]),
                                )
                            })
                            .collect()
                    })
                    .collect();
                (ups, amps)
            })
            .collect();
        Self {
            pre: DevLin::conv(&g.pre),
            stages,
            act_post: DevAct::new(&g.act_post),
            post: DevLin::conv(&g.post),
        }
    }

    fn view(&self) -> crate::gpu::dit::LtxVocGen<'_> {
        use crate::gpu::dit::{LtxVocGen, LtxVocRes, LtxVocStage};
        LtxVocGen {
            pre: self.pre.view(),
            stages: self
                .stages
                .iter()
                .map(|(ups, amps)| LtxVocStage {
                    up: ups.iter().map(DevLin::view).collect(),
                    amps: amps
                        .iter()
                        .map(|a| {
                            a.iter()
                                .map(|(a1, l1, a2, l2)| LtxVocRes {
                                    act1: a1.view(),
                                    conv1: l1.view(),
                                    act2: a2.view(),
                                    conv2: l2.view(),
                                })
                                .collect()
                        })
                        .collect(),
                })
                .collect(),
            act_post: self.act_post.view(),
            post: self.post.view(),
        }
    }
}
