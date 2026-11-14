//! LFM3-Omni's audio input: the log-mel front end (`Lfm3OmniFeatureExtractor`), the stride-8
//! downsampler, the causal LFM2 encoder with a sliding attention window, and the projector into
//! the language model's embedding space. One soft token per 8 mel frames (80 ms).
//!
//! Features: 16 kHz mono, zero-padded on the right to a whole number of hops (at least one
//! window), a periodic Hann window of 400 samples, hop 160, no centering, power spectrum, the
//! Slaney mel filterbank Whisper uses (128 bins), `log10` clamped below at -12, then
//! `(x + 4) / 4`. Unlike Whisper there is no clamp relative to the maximum and no 30-s padding.

use super::blocks::{Span, Stack, StackConfig};
use crate::bert::{LayerNorm, gelu_erf};
use crate::kernels::*;
use crate::model::{Lin, linear};
use crate::pool;
use crate::qwen_vision::VLin;
use crate::safetensors::ModelFiles;
use crate::whisper::mel::{Fft, mel_filters};
use anyhow::{Context, Result, bail};

pub const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
/// Mel frames per soft token.
pub const STRIDE: usize = 8;

/// The front end's settings (`processor_config.json`, `feature_extractor`).
#[derive(Clone, Debug)]
pub struct FeatureConfig {
    pub n_mels: usize,
    pub log_floor: f32,
    pub offset: f32,
    pub divisor: f32,
}

impl FeatureConfig {
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let mut c = Self {
            n_mels: 128,
            log_floor: -12.0,
            offset: 4.0,
            divisor: 4.0,
        };
        let Ok(bytes) = crate::fsx::read(&dir.join("processor_config.json")) else {
            return Ok(c);
        };
        let v: serde_json::Value = serde_json::from_slice(&bytes).context("parse processor_config.json")?;
        let fe = &v["feature_extractor"];
        let int = |k: &str, d: usize| fe.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let num = |k: &str, d: f32| fe.get(k).and_then(|x| x.as_f64()).map_or(d, |x| x as f32);
        if (
            int("sampling_rate", SAMPLE_RATE),
            int("n_fft", N_FFT),
            int("win_length", N_FFT),
            int("hop_length", HOP),
        ) != (SAMPLE_RATE, N_FFT, N_FFT, HOP)
            || int("encoder_stride", STRIDE) != STRIDE
        {
            bail!("LFM3-Omni feature extractor: only 16 kHz, a 400-sample window, hop 160 and stride 8 are supported");
        }
        c.n_mels = int("feature_size", c.n_mels);
        c.log_floor = num("log_floor", c.log_floor);
        c.offset = num("normalization_offset", c.offset);
        c.divisor = num("normalization_divisor", c.divisor);
        Ok(c)
    }
}

/// Mel frames of a clip of `n` samples.
pub fn num_frames(n: usize) -> usize {
    let padded = if n < N_FFT { N_FFT } else { n + (HOP - (n - N_FFT) % HOP) % HOP };
    1 + (padded - N_FFT) / HOP
}

/// Soft tokens of a clip of `n` samples.
pub fn num_tokens(n: usize) -> usize {
    num_frames(n).div_ceil(STRIDE)
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

/// Log-mel features of 16 kHz mono samples, time-major: `[frames, n_mels]`.
pub fn log_mel(pcm: &[f32], fc: &FeatureConfig) -> Vec<f32> {
    let frames = num_frames(pcm.len());
    let n_mels = fc.n_mels;
    let filters = mel_filters(n_mels);
    let window: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
        .collect();
    let fft = Fft::new(N_FFT);
    let mut data = vec![0f32; frames * n_mels];
    const ROWS: usize = 64;
    let dp = SendPtr(data.as_mut_ptr());
    let floor = fc.log_floor as f64;
    pool::run(frames.div_ceil(ROWS), &|task| {
        let (t0, t1) = (task * ROWS, ((task + 1) * ROWS).min(frames));
        let mut buf = vec![(0f64, 0f64); N_FFT];
        let mut spec = vec![(0f64, 0f64); N_FFT];
        let mut power = [0f64; BINS];
        for t in t0..t1 {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (pcm.get(t * HOP + i).copied().unwrap_or(0.0) as f64 * window[i], 0.0);
            }
            fft.run(&buf, &mut spec);
            for (p, &(re, im)) in power.iter_mut().zip(&spec[..BINS]) {
                *p = re * re + im * im;
            }
            // SAFETY: each task writes its own frame rows.
            let row = unsafe { std::slice::from_raw_parts_mut(dp.get().add(t * n_mels), n_mels) };
            for (m, r) in row.iter_mut().enumerate() {
                let mut s = 0f64;
                for b in 0..BINS {
                    s += filters[b * n_mels + m] * power[b];
                }
                // log10(0) is -inf in the reference, clamped to the floor like any small value.
                let l = if s > 0.0 { s.log10().max(floor) } else { floor };
                *r = ((l + fc.offset as f64) / fc.divisor as f64) as f32;
            }
        }
    });
    data
}

/// The audio encoder's shape (`audio_encoder_config`).
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub n_mels: usize,
    pub channels: usize,
    pub hidden: usize,
    pub window: usize,
    pub(crate) stack: StackConfig,
}

impl EncoderConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let int = |v: &serde_json::Value, k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("audio_encoder_config: missing {k}"))
        };
        let t = &v["text_config"];
        if t.get("model_type").and_then(|x| x.as_str()).is_some_and(|m| m != "lfm2") {
            bail!("LFM3-Omni audio encoder: backbone {:?} is not supported (lfm2)", t["model_type"]);
        }
        if t.get("conv_bias").and_then(|x| x.as_bool()) == Some(true) {
            bail!("LFM3-Omni audio encoder: conv_bias is not supported");
        }
        let hidden = int(v, "hidden_size")?;
        if int(t, "hidden_size")? != hidden {
            bail!("LFM3-Omni audio encoder: the backbone's width differs from the encoder's");
        }
        let heads = int(t, "num_attention_heads")?;
        let types: Vec<String> = serde_json::from_value(t["layer_types"].clone()).context("audio encoder layer_types")?;
        // Lfm2MLP: two thirds of the configured width, rounded up to a multiple.
        let mut inter = int(t, "intermediate_size")?;
        if t.get("block_auto_adjust_ff_dim").and_then(|x| x.as_bool()).unwrap_or(true) {
            inter = 2 * inter / 3;
            if let Some(m) = t.get("block_ffn_dim_multiplier").and_then(|x| x.as_f64()) {
                inter = (m * inter as f64) as usize;
            }
            let mult = t.get("block_multiple_of").and_then(|x| x.as_u64()).unwrap_or(256) as usize;
            inter = mult * inter.div_ceil(mult);
        }
        let theta = t["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1_000_000.0);
        Ok(Self {
            n_mels: int(v, "num_mel_bins")?,
            channels: int(v, "conv_channels")?,
            hidden,
            window: int(v, "sliding_window")?,
            stack: StackConfig {
                hidden,
                conv: StackConfig::conv_flags(&types)?,
                heads,
                kv_heads: int(t, "num_key_value_heads")?,
                head_dim: hidden / heads,
                inter,
                kernel: int(t, "conv_L_cache")?,
                eps: t.get("norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
                sandwich: false,
                lfm3: false,
                rope_theta: theta,
            },
        })
    }
}

/// A 3x3 stride-2 convolution with padding 1 over `[h, w, cin]` rows (channels last), `cin` 1
/// (`groups` false, weights `[c, 1, 3, 3]`) or depthwise (weights `[c, 1, 3, 3]` too).
fn conv3x3_s2(x: &[f32], h: usize, w: usize, c: usize, depthwise: bool, wt: &[f32], bias: &[f32]) -> (Vec<f32>, usize, usize) {
    let cin = if depthwise { c } else { 1 };
    let (ho, wo) = (h.div_ceil(2), w.div_ceil(2));
    let mut out = vec![0f32; ho * wo * c];
    for_chunks(&mut out, wo * c, true, |yo, row| {
        for xo in 0..wo {
            let o = &mut row[xo * c..(xo + 1) * c];
            o.copy_from_slice(bias);
            for ky in 0..3 {
                let yi = (yo * 2 + ky) as isize - 1;
                if yi < 0 || yi >= h as isize {
                    continue;
                }
                for kx in 0..3 {
                    let xi = (xo * 2 + kx) as isize - 1;
                    if xi < 0 || xi >= w as isize {
                        continue;
                    }
                    let src = &x[(yi as usize * w + xi as usize) * cin..][..cin];
                    let k = ky * 3 + kx;
                    if depthwise {
                        for (ch, ov) in o.iter_mut().enumerate() {
                            *ov += wt[ch * 9 + k] * src[ch];
                        }
                    } else {
                        for (ch, ov) in o.iter_mut().enumerate() {
                            *ov += wt[ch * 9 + k] * src[0];
                        }
                    }
                }
            }
        }
    });
    (out, ho, wo)
}

fn silu_inplace(x: &mut [f32]) {
    for_chunks(x, 4096, x.len() > 1 << 16, |_, c| {
        for v in c.iter_mut() {
            *v /= 1.0 + (-*v).exp();
        }
    });
}

/// A 1x1 convolution stored as `[o, i, 1(, 1)]`, with its bias.
struct Point<'a> {
    w: Lin<'a>,
    b: Vec<f32>,
    o: usize,
    i: usize,
}

impl<'a> Point<'a> {
    fn load(files: &'a ModelFiles, name: &str, o: usize, i: usize) -> Result<Self> {
        let w = files.bf16(&format!("{name}.weight"))?;
        let b = files.f32(&format!("{name}.bias"))?;
        if w.len() != o * i || b.len() != o {
            bail!("{name}: expected a [{o}, {i}] layer");
        }
        Ok(Self { w: Lin::bf16(w), b, o, i })
    }
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * self.o];
        linear(&self.w, self.o, self.i, x, t, &mut y);
        for_chunks(&mut y, self.o, true, |_, r| add_inplace(r, &self.b));
        y
    }
}

/// The audio encoder and its projector.
pub struct AudioEncoder<'a> {
    pub cfg: EncoderConfig,
    conv_in: (Vec<f32>, Vec<f32>),
    conv_mid: (Vec<f32>, Vec<f32>),
    conv_end: (Vec<f32>, Vec<f32>),
    lin_mid: Point<'a>,
    lin_end: Point<'a>,
    lin_out: Point<'a>,
    lfm: Stack<'a>,
    proj_norm: LayerNorm,
    proj_1: VLin<'a>,
    proj_2: VLin<'a>,
    text_hidden: usize,
}

impl<'a> AudioEncoder<'a> {
    pub fn load(cfg: EncoderConfig, files: &'a ModelFiles, text_hidden: usize) -> Result<Self> {
        let p = "model.audio_encoder.";
        let c = cfg.channels;
        if !cfg.n_mels.is_multiple_of(8) {
            bail!("LFM3-Omni audio encoder: {} mel bins (a multiple of 8 is needed)", cfg.n_mels);
        }
        let conv = |n: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            let w = files.f32(&format!("{p}downsample.{n}.weight"))?;
            let b = files.f32(&format!("{p}downsample.{n}.bias"))?;
            if w.len() != c * 9 || b.len() != c {
                bail!("{p}downsample.{n}: expected a [{c}, 1, 3, 3] convolution");
            }
            Ok((w, b))
        };
        let h = cfg.hidden;
        Ok(Self {
            conv_in: conv("conv_in")?,
            conv_mid: conv("conv_mid")?,
            conv_end: conv("conv_end")?,
            lin_mid: Point::load(files, &format!("{p}downsample.lin_mid"), c, c)?,
            lin_end: Point::load(files, &format!("{p}downsample.lin_end"), c, c)?,
            lin_out: Point::load(files, &format!("{p}downsample.lin_out"), h, cfg.n_mels / 8 * c)?,
            lfm: Stack::load(files, &format!("{p}lfm."), cfg.stack.clone())?,
            proj_norm: LayerNorm::load(files, "model.audio_projector.layer_norm", h)?,
            proj_1: VLin::load(files, "model.audio_projector.linear_1", (text_hidden, h), (text_hidden, h))?,
            proj_2: VLin::load(
                files,
                "model.audio_projector.linear_2",
                (text_hidden, text_hidden),
                (text_hidden, text_hidden),
            )?,
            text_hidden,
            cfg,
        })
    }

    /// The downsampler: `[frames, n_mels]` features to `[ceil(frames / 8), hidden]`.
    fn downsample(&self, mel: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let (nm, c) = (self.cfg.n_mels, self.cfg.channels);
        // The convolutions run over the (mel, time) plane: rows are mel bins.
        let mut plane = vec![0f32; nm * frames];
        for t in 0..frames {
            for m in 0..nm {
                plane[m * frames + t] = mel[t * nm + m];
            }
        }
        let (mut x, h1, w1) = conv3x3_s2(&plane, nm, frames, c, false, &self.conv_in.0, &self.conv_in.1);
        silu_inplace(&mut x);
        let (x, h2, w2) = conv3x3_s2(&x, h1, w1, c, true, &self.conv_mid.0, &self.conv_mid.1);
        let mut x = self.lin_mid.forward(&x, h2 * w2);
        silu_inplace(&mut x);
        let (x, h3, w3) = conv3x3_s2(&x, h2, w2, c, true, &self.conv_end.0, &self.conv_end.1);
        let mut x = self.lin_end.forward(&x, h3 * w3);
        silu_inplace(&mut x);
        // `flatten(1, 2)` of `[channels, mel, time]`: feature `ch * mel_rows + row` per step.
        let mut flat = vec![0f32; w3 * c * h3];
        for_chunks(&mut flat, c * h3, true, |t, row| {
            for r in 0..h3 {
                let src = &x[(r * w3 + t) * c..(r * w3 + t + 1) * c];
                for (ch, &v) in src.iter().enumerate() {
                    row[ch * h3 + r] = v;
                }
            }
        });
        (self.lin_out.forward(&flat, w3), w3)
    }

    /// The encoder's output before the projector: `[tokens, hidden]`.
    pub fn encode_features(&self, mel: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let (mut x, t) = self.downsample(mel, frames);
        self.lfm.forward(&mut x, t, Span::Causal(Some(self.cfg.window)));
        (x, t)
    }

    /// Soft tokens `[tokens, text_hidden]` of a clip's features.
    pub fn encode(&self, mel: &[f32], frames: usize) -> Vec<f32> {
        let (mut x, t) = self.encode_features(mel, frames);
        self.proj_norm.apply_rows(&mut x, 1e-5);
        let th = self.text_hidden;
        let mut a = vec![0f32; t * th];
        self.proj_1.forward(&x, t, &mut a);
        for_chunks(&mut a, th, true, |_, r| r.iter_mut().for_each(|v| *v = gelu_erf(*v)));
        let mut out = vec![0f32; t * th];
        self.proj_2.forward(&a, t, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_follow_the_extractor() {
        // Shorter than a window: one frame. Then one more frame per started hop.
        assert_eq!(num_frames(1), 1);
        assert_eq!(num_frames(400), 1);
        assert_eq!(num_frames(401), 2);
        assert_eq!(num_frames(560), 2);
        assert_eq!(num_frames(16_000), 99);
        assert_eq!(num_tokens(16_000), 13);
        assert_eq!(num_tokens(400 + 7 * 160), 1);
        assert_eq!(num_tokens(400 + 8 * 160), 2);
    }

    #[test]
    fn silence_sits_at_the_floor() {
        let fc = FeatureConfig {
            n_mels: 128,
            log_floor: -12.0,
            offset: 4.0,
            divisor: 4.0,
        };
        let m = log_mel(&[0.0; 800], &fc);
        assert_eq!(m.len(), num_frames(800) * 128);
        assert!(m.iter().all(|&v| v == -2.0));
    }

    #[test]
    fn stride_two_conv_halves_rounding_up() {
        // One channel, all-ones kernel, zero bias: each output counts the inputs under it.
        let x = vec![1f32; 5 * 3];
        let (y, h, w) = conv3x3_s2(&x, 5, 3, 1, false, &[1.0; 9], &[0.0]);
        assert_eq!((h, w), (3, 2));
        assert_eq!(y, vec![4.0, 4.0, 6.0, 6.0, 4.0, 4.0]);
    }
}
