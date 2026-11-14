//! The Liquid codec's decoder (`LiquidCodecModel.decode`, LiquidAI/Liquid-Codec and the
//! `audio_tokenizer/` of LFM3-Omni): 12 codebooks at 12.5 frames per second to 24 kHz audio.
//!
//! ```text
//! codes [frames, 12]  -- each code is 4 FSQ digits (levels 8, 7, 6, 6), scaled to [-1, 1]
//!   -> project_out (48 -> 256) -> from_quantizer (256 -> 512)
//!   -> each frame repeated 6 times (one row per STFT hop)
//!   -> 8 LFM3 layers (convs, and attention over the last 30 rows)
//!   -> lin (512 -> 2 * 641): log magnitude and phase of each STFT frame
//!   -> inverse real FFT of 1280, Hann window, overlap-add at hop 320, divided by the window
//!      envelope, 480 samples trimmed from each end: 1920 samples per code frame
//! ```
//!
//! Only decoding is implemented (the encoder half, audio to codes, is not).

use super::blocks::{Span, Stack, StackConfig};
use crate::kernels::*;
use crate::pool;
use crate::qwen_vision::VLin;
use crate::safetensors::ModelFiles;
use crate::whisper::mel::Fft;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// `config.json` of the codec.
#[derive(Clone, Debug)]
pub struct CodecConfig {
    pub dim: usize,
    pub quantizer_dim: usize,
    pub levels: Vec<usize>,
    pub n_q: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub frame_stack: usize,
    pub window: usize,
    pub sample_rate: usize,
}

impl CodecConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse the codec's config.json")?;
        if v.get("model_type").and_then(|x| x.as_str()) != Some("liquid_codec") {
            bail!("{}: not a Liquid codec (model_type liquid_codec)", dir.display());
        }
        // Released configs may nest the fields under `model`.
        let get = |k: &str| v.get(k).or_else(|| v.get("model").and_then(|m| m.get(k)));
        let int = |k: &str, d: usize| get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let levels: Vec<usize> = match get("fsq_levels") {
            Some(l) if !l.is_null() => serde_json::from_value(l.clone()).context("fsq_levels")?,
            _ => vec![8, 7, 6, 6],
        };
        let c = Self {
            dim: int("encoder_dim", 512),
            quantizer_dim: int("quantizer_dim", 256),
            levels,
            n_q: int("n_q", 12),
            n_fft: int("stft_n_fft", 1280),
            hop: int("stft_hop_length", 320),
            frame_stack: int("stft_frame_stack", 6),
            window: int("decoder_window", 30),
            sample_rate: int("sampling_rate", 24_000),
        };
        if int("stft_win_length", c.n_fft) != c.n_fft {
            bail!("Liquid codec: a window shorter than the FFT is not supported");
        }
        Ok(c)
    }

    pub fn codebook_size(&self) -> usize {
        self.levels.iter().product()
    }

    /// Samples per code frame.
    pub fn frame_samples(&self) -> usize {
        self.hop * self.frame_stack
    }

    /// The LFM3 backbone of either half (`LiquidCodecConfig.lfm_config`).
    fn stack(&self) -> StackConfig {
        StackConfig {
            hidden: self.dim,
            conv: (0..8).map(|i| !matches!(i, 2 | 4 | 6)).collect(),
            heads: 8,
            kv_heads: 2,
            head_dim: 64,
            inter: 2304,
            kernel: 7,
            eps: 1e-5,
            sandwich: true,
            lfm3: true,
            rope_theta: 1_000_000.0,
        }
    }
}

/// The decoder half of the codec.
pub struct CodecDecoder<'a> {
    pub cfg: CodecConfig,
    project_out: Option<VLin<'a>>,
    from_quantizer: VLin<'a>,
    lfm: Stack<'a>,
    lin: VLin<'a>,
    window: Vec<f32>,
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

impl<'a> CodecDecoder<'a> {
    pub fn load(cfg: CodecConfig, files: &'a ModelFiles) -> Result<Self> {
        let eff = cfg.levels.len() * cfg.n_q;
        let (qd, d) = (cfg.quantizer_dim, cfg.dim);
        let bins = cfg.n_fft / 2 + 1;
        let window = files.f32("decoder.istft.window")?;
        if window.len() != cfg.n_fft {
            bail!(
                "Liquid codec: the synthesis window has {} samples, expected {}",
                window.len(),
                cfg.n_fft
            );
        }
        Ok(Self {
            project_out: if qd == eff {
                None
            } else {
                Some(VLin::load(files, "quantizer.project_out", (qd, eff), (qd, eff))?)
            },
            from_quantizer: VLin::load(files, "from_quantizer", (d, qd), (d, qd))?,
            lfm: Stack::load(files, "decoder.lfm.", cfg.stack())?,
            lin: VLin::load(files, "decoder.lin", (2 * bins, d), (2 * bins, d))?,
            window,
            cfg,
        })
    }

    /// The FSQ digits of each code, scaled to [-1, 1]: `[frames, n_q * levels]`.
    fn latents(&self, codes: &[u32], frames: usize) -> Vec<f32> {
        let c = &self.cfg;
        let nl = c.levels.len();
        let mut out = vec![0f32; frames * c.n_q * nl];
        for (i, &code) in codes.iter().enumerate() {
            let mut rest = code as usize;
            for (j, &l) in c.levels.iter().enumerate() {
                let half = (l / 2) as f32;
                out[i * nl + j] = ((rest % l) as f32 - half) / half;
                rest /= l;
            }
        }
        out
    }

    /// Decodes `codes` (`[frames, n_q]`, frame-major) to mono samples at `cfg.sample_rate`,
    /// `frame_samples()` per frame.
    pub fn decode(&self, codes: &[u32], frames: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if codes.len() != frames * c.n_q {
            bail!("Liquid codec: {} codes for {frames} frames of {}", codes.len(), c.n_q);
        }
        if let Some(&bad) = codes.iter().find(|&&v| v as usize >= c.codebook_size()) {
            bail!("Liquid codec: code {bad} is outside the codebook of {}", c.codebook_size());
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        let d = c.dim;
        let mut lat = self.latents(codes, frames);
        if let Some(po) = &self.project_out {
            let mut y = vec![0f32; frames * c.quantizer_dim];
            po.forward(&lat, frames, &mut y);
            lat = y;
        }
        let mut emb = vec![0f32; frames * d];
        self.from_quantizer.forward(&lat, frames, &mut emb);
        // Nearest-neighbour upsampling: each frame becomes `frame_stack` rows.
        let t = frames * c.frame_stack;
        let mut x = vec![0f32; t * d];
        for i in 0..t {
            x[i * d..(i + 1) * d].copy_from_slice(&emb[(i / c.frame_stack) * d..][..d]);
        }
        self.lfm.forward(&mut x, t, Span::Causal(Some(c.window)));
        let bins = c.n_fft / 2 + 1;
        let mut spec = vec![0f32; t * 2 * bins];
        self.lin.forward(&x, t, &mut spec);

        // Inverse STFT: windowed frames, then overlap-add.
        let n = c.n_fft;
        let fft = Fft::new(n);
        let mut frames_td = vec![0f32; t * n];
        let fp = SendPtr(frames_td.as_mut_ptr());
        let window = &self.window;
        pool::run(t, &|i| {
            let row = &spec[i * 2 * bins..(i + 1) * 2 * bins];
            // The Hermitian spectrum, conjugated: its forward DFT is n times the inverse's
            // conjugate, which is real.
            let mut full = vec![(0f64, 0f64); n];
            for b in 0..bins {
                let (m, a) = ((row[b] as f64).exp(), row[bins + b] as f64);
                // irfft ignores the imaginary part of the DC and Nyquist bins.
                let (re, im) = (m * a.cos(), if b == 0 || b == n / 2 { 0.0 } else { m * a.sin() });
                full[b] = (re, -im);
                if b != 0 && b != n / 2 {
                    full[n - b] = (re, im);
                }
            }
            let mut out = vec![(0f64, 0f64); n];
            fft.run(&full, &mut out);
            // SAFETY: each task writes its own frame.
            let dst = unsafe { std::slice::from_raw_parts_mut(fp.get().add(i * n), n) };
            for (j, o) in dst.iter_mut().enumerate() {
                *o = (out[j].0 / n as f64) as f32 * window[j];
            }
        });
        let total = (t - 1) * c.hop + n;
        let mut audio = vec![0f32; total];
        let mut env = vec![0f32; total];
        for i in 0..t {
            let o = i * c.hop;
            add_inplace(&mut audio[o..o + n], &frames_td[i * n..(i + 1) * n]);
            for (e, w) in env[o..o + n].iter_mut().zip(window) {
                *e += w * w;
            }
        }
        let trim = (n - c.hop) / 2;
        Ok(audio[trim..total - trim]
            .iter()
            .zip(&env[trim..total - trim])
            .map(|(a, e)| a / e)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fsq_digits_scale_to_unit_range() {
        // What `latents` computes, on the released levels: code 0 is the lowest digit of every
        // level, the last code the highest.
        let levels = [8usize, 7, 6, 6];
        let digits = |code: usize| -> Vec<f32> {
            let mut rest = code;
            levels
                .iter()
                .map(|&l| {
                    let half = (l / 2) as f32;
                    let v = ((rest % l) as f32 - half) / half;
                    rest /= l;
                    v
                })
                .collect()
        };
        assert_eq!(digits(0), vec![-1.0, -1.0, -1.0, -1.0]);
        assert_eq!(digits(2015), vec![0.75, 1.0, 2.0 / 3.0, 2.0 / 3.0]);
        // 4 + 8 * 3 + 56 * 3 + 336 * 3: the middle of every level.
        assert_eq!(digits(1204), vec![0.0; 4]);
    }
}
