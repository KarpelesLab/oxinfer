//! LTX-2.5's audio encoding (`audio_vae.encoder` of `vae/ltx-2.5-audio-vae-bf16.safetensors`,
//! `ltx_core.model.audio_vae.encode_audio`): a waveform to the transformer's audio latent rows,
//! for audio-to-video (the soundtrack as conditioning) and retakes (the source's soundtrack).
//!
//! - Front end (`AudioProcessor.waveform_to_mel`): each channel resampled to 16 kHz as
//!   `torchaudio.functional.resample` does (Hann-windowed sinc, 6 zero crossings, rolloff 0.99),
//!   then torchaudio's `MelSpectrogram` (1024-point frames every 160 samples, centered with
//!   reflect padding, a periodic Hann window, magnitudes, 64 Slaney mel bins from 0 to 8 kHz with
//!   Slaney's normalization) and `log(max(mel, 1e-5))`.
//! - Encoder (`AudioEncoder`): causal (in time) 2-D convolutions over (time, mel) as the
//!   decoder's ([`super::audio`]): `conv_in`, per level two residual blocks and, between levels,
//!   a stride-2 3x3 convolution (two zero rows before in time, one zero column after in mel),
//!   a mid block, pixel norm, SiLU, `conv_out`; its first `z_channels` outputs (the means),
//!   patchified (`b c t f -> b t (c f)`) and normalized by the per-channel statistics.

use super::audio::{Conv2, Res, pixel_norm_silu};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// `torchaudio.functional.resample` (`sinc_interp_hann`, `lowpass_filter_width` 6, `rolloff`
/// 0.99) of one channel from `from` Hz to `to` Hz: `ceil(len to / from)` samples.
pub fn resample_sinc(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() || from == 0 || to == 0 {
        return x.to_vec();
    }
    let g = gcd(from as u64, to as u64);
    let (orig, new) = ((from as u64 / g) as usize, (to as u64 / g) as usize);
    let lw = 6.0f32;
    let base = orig.min(new) as f64 * 0.99;
    let width = (lw as f64 * orig as f64 / base).ceil() as usize;
    let k = 2 * width + orig;
    // The kernel `[new, 2 width + orig]` in float32, as torchaudio makes it for a float32 signal.
    let (base32, pi) = (base as f32, std::f32::consts::PI);
    let mut kern = vec![0f32; new * k];
    for p in 0..new {
        for j in 0..k {
            let idx = (j as f32 - width as f32) / orig as f32;
            let t = ((-(p as f32) / new as f32 + idx) * base32).clamp(-lw, lw);
            let w = (t * pi / lw / 2.0).cos().powi(2);
            let t = t * pi;
            let s = if t == 0.0 { 1.0 } else { t.sin() / t };
            kern[p * k + j] = s * (w * (base32 / orig as f32));
        }
    }
    let n = x.len();
    // Zero padding: `width` before, `width + orig` after.
    let at = |i: usize| -> f32 { if i < width || i - width >= n { 0.0 } else { x[i - width] } };
    let blocks = (n + 2 * width + orig - k) / orig + 1;
    let target = (new as u64 * n as u64).div_ceil(orig as u64) as usize;
    let mut out = vec![0f32; blocks * new];
    crate::kernels::for_chunks(&mut out, new, blocks > 64, |b, o| {
        for (p, v) in o.iter_mut().enumerate() {
            let row = &kern[p * k..(p + 1) * k];
            *v = row.iter().enumerate().map(|(j, &c)| c * at(b * orig + j)).sum();
        }
    });
    out.truncate(target);
    out
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// torchaudio's `MelSpectrogram` (centered frames with reflect padding, a periodic Hann window,
/// magnitudes, Slaney mel scale and normalization from 0 to half the rate) then
/// `log(max(mel, 1e-5))`, for one channel: `[frames, mels]` with `frames = len / hop + 1`.
pub fn log_mel_spectrogram(x: &[f32], rate: u32, n_fft: usize, hop: usize, mels: usize) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let fb = slaney_mel_filters(rate, n_fft, mels);
    let pad = (n_fft / 2) as isize;
    let last = x.len() as isize - 1;
    let at = |i: isize| -> f32 {
        let mut j = i - pad;
        // Reflection (repeated for signals shorter than the padding).
        while last > 0 && (j < 0 || j > last) {
            j = if j < 0 { -j } else { 2 * last - j };
        }
        x[j.clamp(0, last.max(0)) as usize]
    };
    let frames = x.len() / hop + 1;
    let win: Vec<f64> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
        .collect();
    let cs: Vec<(f64, f64)> = (0..n_fft)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / n_fft as f64;
            (a.cos(), a.sin())
        })
        .collect();
    let mut out = vec![0f32; frames * mels];
    crate::kernels::for_chunks(&mut out, mels, frames > 4, |fi, row| {
        let seg: Vec<f64> = (0..n_fft).map(|j| at((fi * hop + j) as isize) as f64 * win[j]).collect();
        let mag: Vec<f64> = (0..n_freq)
            .map(|kf| {
                let (mut re, mut im) = (0f64, 0f64);
                for (j, &v) in seg.iter().enumerate() {
                    let (c, s) = cs[(kf * j) % n_fft];
                    re += v * c;
                    im -= v * s;
                }
                (re * re + im * im).sqrt()
            })
            .collect();
        for (m, r) in row.iter_mut().enumerate() {
            let s: f64 = fb[m * n_freq..(m + 1) * n_freq].iter().zip(&mag).map(|(a, b)| a * b).sum();
            *r = (s as f32).max(1e-5).ln();
        }
    });
    out
}

/// `melscale_fbanks(norm="slaney", mel_scale="slaney")` from 0 Hz to half the rate:
/// `[mels, n_fft / 2 + 1]`.
fn slaney_mel_filters(rate: u32, n_fft: usize, mels: usize) -> Vec<f64> {
    let n_freq = n_fft / 2 + 1;
    let fsp = 200.0 / 3.0;
    let (min_log_hz, min_log_mel, step) = (1000.0, 1000.0 / fsp, 6.4f64.ln() / 27.0);
    let to_mel = |f: f64| {
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / step
        } else {
            f / fsp
        }
    };
    let to_hz = |m: f64| {
        if m >= min_log_mel {
            min_log_hz * (step * (m - min_log_mel)).exp()
        } else {
            fsp * m
        }
    };
    let fmax = (rate / 2) as f64;
    let freqs: Vec<f64> = (0..n_freq).map(|i| fmax * i as f64 / (n_freq - 1) as f64).collect();
    let (m0, m1) = (to_mel(0.0), to_mel(rate as f64 / 2.0));
    let pts: Vec<f64> = (0..mels + 2)
        .map(|i| to_hz(m0 + (m1 - m0) * i as f64 / (mels + 1) as f64))
        .collect();
    let mut fb = vec![0f64; mels * n_freq];
    for m in 0..mels {
        let enorm = 2.0 / (pts[m + 2] - pts[m]);
        for (k, &f) in freqs.iter().enumerate() {
            let down = (f - pts[m]) / (pts[m + 1] - pts[m]);
            let up = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
            fb[m * n_freq + k] = down.min(up).max(0.0) * enorm;
        }
    }
    fb
}

/// The `audio_vae.encoder` of the file with its front end.
pub struct AudioEncoder {
    mean: Vec<f32>,
    std: Vec<f32>,
    conv_in: Conv2,
    /// Lowest level first; the stride-2 convolution after each but the last.
    levels: Vec<(Vec<Res>, Option<Conv2>)>,
    mid: [Res; 2],
    conv_out: Conv2,
    pub z_channels: usize,
    pub mel_bins: usize,
    pub sample_rate: u32,
    pub hop: usize,
    pub n_fft: usize,
}

impl AudioEncoder {
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
        if dd.get("attn_resolutions").and_then(|v| v.as_array()).is_some_and(|a| !a.is_empty())
            || dd.get("mid_block_add_attention").and_then(|v| v.as_bool()) == Some(true)
        {
            bail!("LTX audio VAE: encoders with attention are not supported");
        }
        let p = "audio_vae.encoder";
        if !f.has(&format!("{p}.conv_in.conv.weight")) {
            bail!("the audio VAE file has no encoder (audio_vae.encoder.*)");
        }
        let levels = (0..n_levels)
            .map(|l| {
                let blocks = (0..nres)
                    .map(|i| Res::load(f, &format!("{p}.down.{l}.block.{i}")))
                    .collect::<Result<_>>()?;
                let down = if l + 1 < n_levels {
                    let mut c = Conv2::load(f, &format!("{p}.down.{l}.downsample.conv"))?;
                    c.conv = c.conv.with_padding(0, 0);
                    Some(c)
                } else {
                    None
                };
                Ok((blocks, down))
            })
            .collect::<Result<_>>()?;
        let stft = cfg.pointer("/preprocessing/stft");
        let su = |k: &str, d: usize| stft.and_then(|s| s.get(k)).and_then(|v| v.as_u64()).map_or(d, |v| v as usize);
        Ok(Self {
            mean: f.f32("audio_vae.per_channel_statistics.mean-of-means")?,
            std: f.f32("audio_vae.per_channel_statistics.std-of-means")?,
            conv_in: Conv2::load(f, &format!("{p}.conv_in.conv"))?,
            levels,
            mid: [
                Res::load(f, &format!("{p}.mid.block_1"))?,
                Res::load(f, &format!("{p}.mid.block_2"))?,
            ],
            conv_out: Conv2::load(f, &format!("{p}.conv_out.conv"))?,
            z_channels: u("z_channels").unwrap_or(8),
            mel_bins: u("mel_bins").unwrap_or(64),
            sample_rate: cfg
                .pointer("/audio_vae/model/params/sampling_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(16000) as u32,
            hop: su("hop_length", 160),
            n_fft: su("filter_length", 1024),
        })
    }

    /// The stereo log-mel spectrogram `[2, frames, mels]` of `channels` at `rate` and its frame
    /// count. A mono signal is used for both sides (the reference needs two channels); of more
    /// than two, the first two.
    pub fn mel(&self, channels: &[Vec<f32>], rate: u32) -> Result<(Vec<f32>, usize)> {
        if channels.is_empty() || channels[0].is_empty() {
            bail!("LTX audio encoder: no samples");
        }
        let two = [&channels[0], channels.get(1).unwrap_or(&channels[0])];
        let mut out = Vec::new();
        let mut frames = 0;
        for c in two {
            let r = resample_sinc(c, rate, self.sample_rate);
            let m = log_mel_spectrogram(&r, self.sample_rate, self.n_fft, self.hop, self.mel_bins);
            frames = m.len() / self.mel_bins;
            out.extend(m);
        }
        Ok((out, frames))
    }

    /// Latent rows `[t, z * mel / 4]` (the transformer's audio rows) of a stereo spectrogram
    /// `[2, frames, mels]`.
    pub fn encode_mel(&self, mel: &[f32], frames: usize) -> Vec<f32> {
        let mb = self.mel_bins;
        // `[t, m, 2]` rows.
        let mut x = vec![0f32; frames * mb * 2];
        for ch in 0..2 {
            for r in 0..frames {
                for j in 0..mb {
                    x[(r * mb + j) * 2 + ch] = mel[(ch * frames + r) * mb + j];
                }
            }
        }
        let (mut t, mut m) = (frames, mb);
        let mut h = self.conv_in.forward(&x, t, m);
        for (blocks, down) in &self.levels {
            for b in blocks {
                h = b.forward(&h, t, m);
            }
            if let Some(d) = down {
                let c = h.len() / (t * m);
                let (pt, pm) = (t + 2, m + 1);
                let mut xp = vec![0f32; pt * pm * c];
                for r in 0..t {
                    for j in 0..m {
                        xp[((r + 2) * pm + j) * c..][..c].copy_from_slice(&h[(r * m + j) * c..][..c]);
                    }
                }
                let (y, nt, nm) = d.conv.forward(&xp, pt, pm, 2);
                h = y;
                (t, m) = (nt, nm);
            }
        }
        for r in &self.mid {
            h = r.forward(&h, t, m);
        }
        let c = h.len() / (t * m);
        let y = self.conv_out.forward(&pixel_norm_silu(&h, c), t, m);
        let oc = self.conv_out.cout;
        let z = self.z_channels;
        let w = z * m;
        let mut rows = vec![0f32; t * w];
        for r in 0..t {
            for ch in 0..z {
                for j in 0..m {
                    let i = ch * m + j;
                    rows[r * w + i] = (y[(r * m + j) * oc + ch] - self.mean[i]) / self.std[i];
                }
            }
        }
        rows
    }

    /// `encode_audio`: latent rows `[t, z * mel / 4]` of a waveform (`channels` at `rate`).
    pub fn encode(&self, channels: &[Vec<f32>], rate: u32) -> Result<Vec<f32>> {
        let (mel, frames) = self.mel(channels, rate)?;
        Ok(self.encode_mel(&mel, frames))
    }
}

/// `_conform_latent_length`: `rows` (of `width`) cut or zero-padded to `n` rows.
pub fn conform_rows(mut rows: Vec<f32>, width: usize, n: usize) -> Vec<f32> {
    rows.resize(n * width, 0.0);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_lengths_and_dc() {
        let x = vec![0.5f32; 4800];
        let y = resample_sinc(&x, 48000, 16000);
        assert_eq!(y.len(), 1600);
        // Away from the borders a constant stays (the kernel's DC gain is about 1).
        assert!((y[800] - 0.5).abs() < 5e-3, "{}", y[800]);
        assert_eq!(resample_sinc(&x[..4410], 44100, 16000).len(), 1600);
    }

    #[test]
    fn mel_frames() {
        let x: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin()).collect();
        let m = log_mel_spectrogram(&x, 16000, 1024, 160, 64);
        assert_eq!(m.len(), 101 * 64);
        assert!(m.iter().all(|v| v.is_finite()));
    }
}
