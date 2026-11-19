//! The decoder of MiniMax-H3's audio VAE (`audio_vae/`, diffusers' `AutoencoderKLMiniMaxH3Audio`;
//! `dac_audio_vae.py` / `dac_bigvgan.py` in the release's `FL2VA/audio_vae/`): 40 latents of 32
//! channels per second to a 32 kHz waveform. The autoencoder is mono; the pipeline carries a
//! stereo soundtrack as two independent latent sequences.
//!
//! 1. `dec_in_proj`, a 1x1 convolution from the 32 latent channels to 2048.
//! 2. BigVGAN: a convolution of 7 to 1024 channels, then seven stages, each a transposed
//!    convolution (upsampling by 5, 5, 2, 2, 2, 2 and 2, kernels 9, 9 and 4, halving the channels)
//!    followed by the mean of three AMP blocks (kernels 3, 7 and 11). An AMP block is three
//!    residual steps `x + conv(act(conv_dilated(act(x))))` with dilations 1, 3 and 5.
//! 3. An activation, a convolution of 7 to one channel without bias, a clamp to `[-1, 1]`.
//!
//! Every activation is anti-aliased SnakeBeta: the signal is upsampled by 2 with a 12-tap
//! Kaiser-windowed sinc filter (replicate padding), goes through
//! `x + sin(exp(alpha) x)^2 / (exp(beta) + 1e-9)` per channel, and is low-passed and decimated by
//! 2 with the same filter. The filters are the checkpoint's buffers.
//!
//! The convolutions are the ones of the Audio8 codec ([`crate::arktts::codec`]) with centered
//! padding. Activations are `[time, channels]`.
//!
//! The encoder ([`AudioEncoder`]: a DAC waveform encoder, a causal attention projection and the
//! `mean_proj` head; the posterior's `logs_proj` is never read) takes `ref2va`'s reference audio.

use crate::arktts::codec::{Conv, ConvT, Snake, bufs, rows_par, weight};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// `audio_vae/config.json`, or the `minimax_h3_audio_vae` header entry of ComfyUI's single file.
#[derive(Clone, Debug)]
pub struct AudioVaeConfig {
    pub latent_channels: usize,
    pub sampling_rate: u32,
    /// The upsampling factor of each BigVGAN stage.
    pub rates: Vec<usize>,
    /// AMP blocks per stage (one per kernel size) and the dilations of each.
    pub dilations: Vec<Vec<usize>>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    /// Heads of the encoder's causal attention projection (8 in the release, which does not
    /// configure it).
    pub heads: usize,
}

impl AudioVaeConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut c: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        // The release's own layout keeps the constructor's arguments in metadata.json.
        if let Ok(b) = crate::fsx::read(&dir.join("metadata.json"))
            && let Ok(m) = serde_json::from_slice::<serde_json::Value>(&b)
            && let (Some(k), Some(o)) = (m.pointer("/metadata/kwargs").and_then(|k| k.as_object()), c.as_object_mut())
        {
            for (src, dst) in [("decoder_rates", "decoder_rates"), ("sample_rate", "sampling_rate")] {
                if let Some(v) = k.get(src)
                    && !o.contains_key(dst)
                {
                    o.insert(dst.to_string(), v.clone());
                }
            }
            if k.get("decoder_type").and_then(|v| v.as_str()).is_some_and(|t| t != "bigvgan") {
                bail!("{}: only the BigVGAN audio decoder is supported", dir.display());
            }
        }
        if let Some(o) = c.as_object_mut()
            && !o.contains_key("sampling_rate")
            && let Some(v) = o.get("sample_rate").cloned()
        {
            o.insert("sampling_rate".to_string(), v);
        }
        Self::from_json(&c)
    }

    /// The configuration in the header of ComfyUI's single file
    /// (`vae/minimax_h3_audio_vae_fp32.safetensors`: the release's tensors under the same names),
    /// in diffusers' keys: `latents_mean` / `latents_std`, `sample_rate`, and MiniMax's own
    /// `kwargs` (`decoder_rates`). None when the file has no such entry.
    pub fn comfy_json(files: &ModelFiles) -> Option<Result<serde_json::Value>> {
        let meta = files.metadata("minimax_h3_audio_vae")?;
        Some((|| {
            let m: serde_json::Value = serde_json::from_str(meta).context("the minimax_h3_audio_vae header entry")?;
            let mut o = serde_json::Map::new();
            for (to, from) in [
                ("latents_mean", "latents_mean"),
                ("latents_std", "latents_std"),
                ("latent_channels", "latent_channels"),
                ("sampling_rate", "sample_rate"),
            ] {
                if let Some(v) = m.get(from) {
                    o.insert(to.into(), v.clone());
                }
            }
            if let Some(k) = m.get("kwargs") {
                match k.get("decoder_type").and_then(|v| v.as_str()) {
                    None | Some("bigvgan") => {}
                    Some(other) => bail!("minimax_h3_audio_vae: decoder_type {other}, only bigvgan is implemented"),
                }
                if let Some(v) = k.get("decoder_rates") {
                    o.insert("decoder_rates".into(), v.clone());
                }
            }
            Ok(serde_json::Value::Object(o))
        })())
    }

    /// From a configuration in diffusers' keys.
    pub fn from_json(c: &serde_json::Value) -> Result<Self> {
        let list = |v: &serde_json::Value| -> Vec<usize> {
            v.as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
                .unwrap_or_default()
        };
        let floats = |n: &str| -> Result<Vec<f32>> {
            Ok(c.get(n)
                .and_then(|v| v.as_array())
                .with_context(|| format!("audio VAE config has no {n}"))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect())
        };
        let latent_channels = c.get("latent_channels").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let rates = c.get("decoder_rates").map(list).unwrap_or_else(|| vec![5, 5, 2, 2, 2, 2, 2]);
        let dilations: Vec<Vec<usize>> = match c.get("resblock_dilation_sizes").and_then(|v| v.as_array()) {
            Some(a) => a.iter().map(list).collect(),
            None => vec![vec![1, 3, 5]; 3],
        };
        let cfg = Self {
            latent_channels,
            sampling_rate: c.get("sampling_rate").and_then(|v| v.as_u64()).unwrap_or(32000) as u32,
            rates,
            dilations,
            latents_mean: floats("latents_mean")?,
            latents_std: floats("latents_std")?,
            heads: c.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(8) as usize,
        };
        if cfg.latents_mean.len() != latent_channels || cfg.latents_std.len() != latent_channels {
            bail!("audio VAE config: latents_mean / latents_std are not {latent_channels} values");
        }
        if cfg.rates.is_empty() || cfg.rates.contains(&0) || cfg.dilations.is_empty() || cfg.dilations.iter().any(|d| d.is_empty()) {
            bail!("audio VAE config: empty decoder_rates or resblock_dilation_sizes");
        }
        Ok(cfg)
    }

    /// Samples per latent.
    pub fn hop(&self) -> usize {
        self.rates.iter().product()
    }
}

fn shape3(shape: &[usize], p: &str) -> Result<[usize; 3]> {
    match shape[..] {
        [a, b, k] => Ok([a, b, k]),
        _ => bail!("{p}: expected a 1D convolution kernel, got {shape:?}"),
    }
}

fn bias(f: &ModelFiles, p: &str, n: usize) -> Result<Vec<f32>> {
    let name = format!("{p}.bias");
    if !f.has(&name) {
        return Ok(vec![0.0; n]);
    }
    let b = f.f32(&name)?;
    if b.len() != n {
        bail!("{name}: {} values, expected {n}", b.len());
    }
    Ok(b)
}

/// A stride-1 convolution with PyTorch's `padding = (k - 1) * dilation / 2` (weight
/// normalization folded in; no bias when the checkpoint has none).
fn conv(f: &ModelFiles, p: &str, dilation: usize) -> Result<Conv> {
    let (w, shape) = weight(f, p)?;
    let [co, ci, k] = shape3(&shape, p)?;
    if k.is_multiple_of(2) {
        bail!("{p}: an even kernel ({k}) does not keep the length");
    }
    Ok(Conv::centered(&w, [co, ci, k], bias(f, p, co)?, dilation))
}

/// PyTorch's `ConvTranspose1d(kernel, stride, padding = (kernel - stride) / 2)`: `t` steps to
/// `t * stride`.
struct Up {
    conv: ConvT,
    c_in: usize,
    c_out: usize,
    /// Output steps cropped on the left, and the zero input steps appended to produce the tail.
    crop: usize,
    extra: usize,
}

impl Up {
    fn load(f: &ModelFiles, p: &str, stride: usize) -> Result<Self> {
        let (w, shape) = weight(f, p)?;
        let [ci, co, k] = shape3(&shape, p)?;
        if k < stride || !(k - stride).is_multiple_of(2) {
            bail!("{p}: kernel {k} with stride {stride} does not upsample by the stride");
        }
        Ok(Self::new(&w, [ci, co, k], &bias(f, p, co)?, stride))
    }

    /// From a kernel in PyTorch's layout (`[in, out, k]`, `k - stride` even).
    fn new(w: &[f32], [ci, co, k]: [usize; 3], b: &[f32], stride: usize) -> Self {
        // The causal transposed convolution takes kernels of whole strides: zero taps complete it.
        let kk = k.div_ceil(stride) * stride;
        let mut wp = vec![0f32; ci * co * kk];
        for (dst, src) in wp.chunks_mut(kk).zip(w.chunks(k)) {
            dst[..k].copy_from_slice(src);
        }
        let crop = (k - stride) / 2;
        Self {
            conv: ConvT::new(&wp, [ci, co, kk], b, stride),
            c_in: ci,
            c_out: co,
            crop,
            extra: crop.div_ceil(stride),
        }
    }

    /// `x` (`[t, c_in]`) to `[t * stride, c_out]`.
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        // The causal form gives the first `t * stride` steps of the full output; the centered one
        // starts `crop` steps later, so its tail comes from zero steps after the input.
        let mut xp = bufs::take((t + self.extra) * self.c_in);
        xp[..t * self.c_in].copy_from_slice(x);
        xp[t * self.c_in..].fill(0.0);
        let (mut y, n) = self.conv.forward(&xp, t + self.extra);
        bufs::give(xp);
        let stride = n / (t + self.extra);
        y.copy_within(self.crop * self.c_out..(self.crop + t * stride) * self.c_out, 0);
        y.truncate(t * stride * self.c_out);
        (y, t * stride)
    }
}

/// The anti-aliased SnakeBeta activation (`Activation1d`).
struct Act {
    snake: Snake,
    /// The upsampling filter, times the ratio (2).
    up: Vec<f32>,
    down: Vec<f32>,
}

impl Act {
    fn load(f: &ModelFiles, p: &str) -> Result<Self> {
        let alpha: Vec<f32> = f.f32(&format!("{p}.act.alpha"))?.iter().map(|a| a.exp()).collect();
        let inv: Vec<f32> = f.f32(&format!("{p}.act.beta"))?.iter().map(|b| 1.0 / (b.exp() + 1e-9)).collect();
        if alpha.len() != inv.len() || alpha.is_empty() {
            bail!("{p}: alpha / beta of {} / {} channels", alpha.len(), inv.len());
        }
        let up: Vec<f32> = f.f32(&format!("{p}.upsample.filter"))?.iter().map(|v| 2.0 * v).collect();
        let down = f.f32(&format!("{p}.downsample.lowpass.filter"))?;
        if up.len() != down.len() || !up.len().is_multiple_of(2) || up.len() < 2 {
            bail!("{p}: resampling filters of {} / {} taps", up.len(), down.len());
        }
        Ok(Self {
            snake: Snake::new(alpha, inv),
            up,
            down,
        })
    }

    /// `x` (`[t, c]`) to a new buffer of the same shape.
    fn apply(&self, x: &[f32], t: usize) -> Vec<f32> {
        let c = x.len() / t.max(1);
        let k = self.up.len();
        // `UpSample1d`: `pad` replicated steps on both sides, a transposed convolution of stride
        // 2, `pad_left` steps cropped: output `m` is the sum over the taps `j` of the parity of
        // `m + pad_left` of `f[j] x[(m + pad_left - j) / 2 - pad]`.
        let pad = k / 2 - 1;
        let pad_left = pad * 2 + (k - 2) / 2;
        let mut u = bufs::take(2 * t * c);
        rows_par(&mut u, c, |m, row| {
            row.fill(0.0);
            let n = m + pad_left;
            for j in (n % 2..k).step_by(2) {
                let src = ((n - j) / 2).saturating_sub(pad).min(t - 1);
                let f = self.up[j];
                for (v, &s) in row.iter_mut().zip(&x[src * c..(src + 1) * c]) {
                    *v += f * s;
                }
            }
        });
        self.snake.apply(&mut u);
        // `DownSample1d`: `k / 2 - 1` replicated steps on the left, `k / 2` on the right, the
        // filter at stride 2.
        let mut y = bufs::take(t * c);
        rows_par(&mut y, c, |i, row| {
            row.fill(0.0);
            for (j, &f) in self.down.iter().enumerate() {
                let src = (2 * i + j).saturating_sub(pad).min(2 * t - 1);
                for (v, &s) in row.iter_mut().zip(&u[src * c..(src + 1) * c]) {
                    *v += f * s;
                }
            }
        });
        bufs::give(u);
        y
    }
}

/// `AMPBlock1`: per dilation, `x += conv2(act(conv1(act(x))))`.
struct Amp {
    acts: Vec<Act>,
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
}

impl Amp {
    fn load(f: &ModelFiles, p: &str, dilations: &[usize]) -> Result<Self> {
        let n = dilations.len();
        Ok(Self {
            acts: (0..2 * n)
                .map(|i| Act::load(f, &format!("{p}.activations.{i}")))
                .collect::<Result<_>>()?,
            convs1: dilations
                .iter()
                .enumerate()
                .map(|(i, &d)| conv(f, &format!("{p}.convs1.{i}"), d))
                .collect::<Result<_>>()?,
            convs2: (0..n).map(|i| conv(f, &format!("{p}.convs2.{i}"), 1)).collect::<Result<_>>()?,
        })
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut x = x.to_vec();
        for (i, (c1, c2)) in self.convs1.iter().zip(&self.convs2).enumerate() {
            let a = self.acts[2 * i].apply(&x, t);
            let (h, _) = c1.forward(&a, t);
            bufs::give(a);
            let a = self.acts[2 * i + 1].apply(&h, t);
            bufs::give(h);
            let (h, _) = c2.forward(&a, t);
            bufs::give(a);
            add(&mut x, &h, c2.c_out);
            bufs::give(h);
        }
        x
    }
}

fn add(x: &mut [f32], y: &[f32], c: usize) {
    rows_par(x, c, |r, row| {
        for (v, &o) in row.iter_mut().zip(&y[r * c..(r + 1) * c]) {
            *v += o;
        }
    });
}

struct Stage {
    up: Up,
    blocks: Vec<Amp>,
}

/// The audio VAE's decoder: latents to a mono waveform.
pub struct AudioDecoder {
    pub cfg: AudioVaeConfig,
    proj_in: Conv,
    conv_pre: Conv,
    stages: Vec<Stage>,
    act_post: Act,
    conv_post: Conv,
}

impl AudioDecoder {
    /// Loads the decoder (float32) from `path`: a folder (`config.json` and the weights), or
    /// ComfyUI's single file with its configuration in the header.
    pub fn load(path: &Path) -> Result<Self> {
        let (cfg, files) = open(path)?;
        Self::from_files(cfg, &files)
    }

    pub fn from_files(cfg: AudioVaeConfig, f: &ModelFiles) -> Result<Self> {
        let (w, shape) = weight(f, "dec_in_proj")?;
        let [dim, c, k] = shape3(&shape, "dec_in_proj")?;
        if c != cfg.latent_channels || k != 1 {
            bail!("dec_in_proj: kernel {shape:?} for {} latent channels", cfg.latent_channels);
        }
        let proj_in = Conv::new(&w, [dim, c, 1], bias(f, "dec_in_proj", dim)?, 1, 1);
        let nk = cfg.dilations.len();
        let mut stages = Vec::with_capacity(cfg.rates.len());
        let mut ch = 0;
        for (i, &rate) in cfg.rates.iter().enumerate() {
            let up = Up::load(f, &format!("decoder.ups.{i}.0"), rate)?;
            if i > 0 && up.c_in != ch {
                bail!("decoder.ups.{i}.0: {} input channels after {ch}", up.c_in);
            }
            ch = up.c_out;
            let blocks = (0..nk)
                .map(|j| Amp::load(f, &format!("decoder.resblocks.{}", i * nk + j), &cfg.dilations[j]))
                .collect::<Result<_>>()?;
            stages.push(Stage { up, blocks });
        }
        let conv_post = conv(f, "decoder.conv_post", 1)?;
        if conv_post.c_out != 1 {
            bail!("decoder.conv_post: {} output channels, the decoder is mono", conv_post.c_out);
        }
        Ok(Self {
            proj_in,
            conv_pre: conv(f, "decoder.conv_pre", 1)?,
            stages,
            act_post: Act::load(f, "decoder.activation_post")?,
            conv_post,
            cfg,
        })
    }

    /// Latents in the VAE's own scale (`[t, latent_channels]`) to `t * hop` samples in `[-1, 1]`.
    pub fn decode(&self, latents: &[f32]) -> Result<Vec<f32>> {
        let c = self.cfg.latent_channels;
        if latents.is_empty() || !latents.len().is_multiple_of(c) {
            bail!("audio VAE: {} values are not rows of {c} latents", latents.len());
        }
        let t = latents.len() / c;
        let (z, _) = self.proj_in.forward(latents, t);
        let (mut x, mut t) = self.conv_pre.forward(&z, t);
        bufs::give(z);
        for s in &self.stages {
            let (up, n) = s.up.forward(&x, t);
            bufs::give(std::mem::replace(&mut x, up));
            t = n;
            let ch = s.up.c_out;
            let mut sum = s.blocks[0].forward(&x, t);
            for b in &s.blocks[1..] {
                add(&mut sum, &b.forward(&x, t), ch);
            }
            // The reference divides by the block count.
            let div = s.blocks.len() as f32;
            rows_par(&mut sum, ch, |_, row| row.iter_mut().for_each(|v| *v /= div));
            bufs::give(std::mem::replace(&mut x, sum));
        }
        let a = self.act_post.apply(&x, t);
        bufs::give(x);
        let (mut y, _) = self.conv_post.forward(&a, t);
        bufs::give(a);
        bufs::clear();
        y.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
        Ok(y)
    }

    /// The pipeline's audio rows (`[channels * n, latent_channels]`, channel-major, normalized) to
    /// one waveform per channel: `latents * latents_std + latents_mean`, each channel decoded on
    /// its own.
    pub fn decode_rows(&self, rows: &[f32], channels: usize) -> Result<Vec<Vec<f32>>> {
        let c = self.cfg.latent_channels;
        if channels == 0 || rows.is_empty() || !rows.len().is_multiple_of(channels * c) {
            bail!(
                "audio VAE: {} values are not {channels} channels of rows of {c} latents",
                rows.len()
            );
        }
        rows.chunks(rows.len() / channels)
            .map(|ch| {
                let z: Vec<f32> = ch
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * self.cfg.latents_std[i % c] + self.cfg.latents_mean[i % c])
                    .collect();
                self.decode(&z)
            })
            .collect()
    }
}

/// The configuration and weights of the audio VAE at `path`: a folder (`config.json` and the
/// weights), or ComfyUI's single file with its configuration in the header.
fn open(path: &Path) -> Result<(AudioVaeConfig, ModelFiles)> {
    if path.is_file() {
        let files = ModelFiles::open_file(path, false)?;
        let json = AudioVaeConfig::comfy_json(&files).with_context(|| {
            format!(
                "{}: not ComfyUI's single-file MiniMax-H3 audio VAE (no minimax_h3_audio_vae header entry)",
                path.display()
            )
        })??;
        return Ok((AudioVaeConfig::from_json(&json)?, files));
    }
    Ok((AudioVaeConfig::load(path)?, ModelFiles::open(path, false)?))
}

/// A linear layer `[out, in]` (with its bias) as a 1x1 convolution over `[t, in]` rows.
fn linear(f: &ModelFiles, p: &str) -> Result<Conv> {
    let n = format!("{p}.weight");
    let shape = f.info(&n).with_context(|| format!("missing tensor {n}"))?.shape.clone();
    let [o, i] = shape[..] else { bail!("{n}: expected a matrix, got {shape:?}") };
    Ok(Conv::new(&f.f32(&n)?, [o, i, 1], bias(f, p, o)?, 1, 1))
}

fn norm(f: &ModelFiles, p: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((f.f32(&format!("{p}.weight"))?, f.f32(&format!("{p}.bias"))?))
}

fn layer_norm(x: &[f32], (w, b): &(Vec<f32>, Vec<f32>)) -> Vec<f32> {
    let mut y = x.to_vec();
    crate::mistral4::layer_norm_rows(&mut y, w.len(), w, b, 1e-5);
    y
}

/// DAC's residual unit: `x + conv1(snake(conv7_dilated(snake(x))))`.
struct EncRes {
    s1: Snake,
    c1: Conv,
    s2: Snake,
    c2: Conv,
}

impl EncRes {
    fn load(f: &ModelFiles, p: &str, dilation: usize) -> Result<Self> {
        Ok(Self {
            s1: Snake::load(f, &format!("{p}.block.0.alpha"))?,
            c1: conv(f, &format!("{p}.block.1"), dilation)?,
            s2: Snake::load(f, &format!("{p}.block.2.alpha"))?,
            c2: conv(f, &format!("{p}.block.3"), 1)?,
        })
    }

    fn forward(&self, x: &mut [f32], t: usize) {
        let mut a = x.to_vec();
        self.s1.apply(&mut a);
        let (mut h, _) = self.c1.forward(&a, t);
        self.s2.apply(&mut h);
        let (o, _) = self.c2.forward(&h, t);
        bufs::give(h);
        add(x, &o, self.c2.c_out);
        bufs::give(o);
    }
}

struct EncBlock {
    res: Vec<EncRes>,
    snake: Snake,
    down: Conv,
}

/// `pre_block`: `proj(norm3(x)) + attn(norm1(x))`, then `+ mlp(norm2(.))`, narrowing the trunk
/// to the latent width. The causal attention's heads are averaged instead of concatenated and
/// the head width is average-pooled down to the latent width.
struct PreBlock {
    norm1: (Vec<f32>, Vec<f32>),
    norm2: (Vec<f32>, Vec<f32>),
    norm3: (Vec<f32>, Vec<f32>),
    qkv: Conv,
    heads: usize,
    attn_proj: Conv,
    proj: Conv,
    mlp_norm: (Vec<f32>, Vec<f32>),
    w0: Conv,
    w1: Conv,
    w2: Conv,
}

impl PreBlock {
    fn load(f: &ModelFiles, p: &str, heads: usize) -> Result<Self> {
        let n = format!("{p}.attn.qkv.weight");
        let shape = f.info(&n).with_context(|| format!("missing tensor {n}"))?.shape.clone();
        let [o, i] = shape[..] else { bail!("{n}: expected a matrix, got {shape:?}") };
        if o != 3 * i || !i.is_multiple_of(heads) {
            bail!("{n}: {shape:?} is not a fused q, k, v projection of {heads} heads");
        }
        let mut b = f.f32(&format!("{p}.attn.q_bias"))?;
        b.extend(std::iter::repeat_n(0.0, i));
        b.extend(f.f32(&format!("{p}.attn.v_bias"))?);
        let attn_proj = linear(f, &format!("{p}.attn.proj"))?;
        if !(i / heads).is_multiple_of(attn_proj.c_out) {
            bail!("{p}: heads of {} do not pool to {}", i / heads, attn_proj.c_out);
        }
        Ok(Self {
            norm1: norm(f, &format!("{p}.norm1"))?,
            norm2: norm(f, &format!("{p}.norm2"))?,
            norm3: norm(f, &format!("{p}.norm3"))?,
            qkv: Conv::new(&f.f32(&n)?, [o, i, 1], b, 1, 1),
            heads,
            attn_proj,
            proj: linear(f, &format!("{p}.proj"))?,
            mlp_norm: norm(f, &format!("{p}.mlp.norm"))?,
            w0: linear(f, &format!("{p}.mlp.w0"))?,
            w1: linear(f, &format!("{p}.mlp.w1"))?,
            w2: linear(f, &format!("{p}.mlp.w2"))?,
        })
    }

    /// `[t, trunk]` to `[t, latent]`.
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (mut y, _) = self.proj.forward(&layer_norm(x, &self.norm3), t);
        let (qkv, _) = self.qkv.forward(&layer_norm(x, &self.norm1), t);
        let d = self.qkv.c_out / 3;
        let (nh, hd, out) = (self.heads, d / self.heads, self.attn_proj.c_out);
        let pool = hd / out;
        let scale = 1.0 / (hd as f32).sqrt();
        // Per query row: the causal attention of every head, averaged over heads and pooled.
        let mut pooled = vec![0f32; t * out];
        crate::kernels::for_chunks(&mut pooled, out, t > 1, |r, o| {
            let q_row = &qkv[r * 3 * d..r * 3 * d + d];
            let mut acc = vec![0f32; hd];
            let mut s = vec![0f32; r + 1];
            for h in 0..nh {
                let q = &q_row[h * hd..(h + 1) * hd];
                for (j, sj) in s.iter_mut().enumerate() {
                    let k = &qkv[j * 3 * d + d + h * hd..j * 3 * d + d + (h + 1) * hd];
                    *sj = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0f32;
                s.iter_mut().for_each(|v| {
                    *v = (*v - m).exp();
                    z += *v;
                });
                for (j, &p) in s.iter().enumerate() {
                    let v = &qkv[j * 3 * d + 2 * d + h * hd..j * 3 * d + 2 * d + (h + 1) * hd];
                    let w = p / z;
                    acc.iter_mut().zip(v).for_each(|(a, &vv)| *a += w * vv);
                }
            }
            for (c, ov) in o.iter_mut().enumerate() {
                *ov = acc[c * pool..(c + 1) * pool].iter().sum::<f32>() / (pool * nh) as f32;
            }
        });
        bufs::give(qkv);
        let (a, _) = self.attn_proj.forward(&pooled, t);
        add(&mut y, &a, out);
        let hn = layer_norm(&layer_norm(&y, &self.norm2), &self.mlp_norm);
        let (mut g, _) = self.w0.forward(&hn, t);
        let (v, _) = self.w1.forward(&hn, t);
        for (gv, &vv) in g.iter_mut().zip(&v) {
            let x = *gv;
            let th = (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh();
            *gv = 0.5 * x * (1.0 + th) * vv;
        }
        let (m, _) = self.w2.forward(&g, t);
        add(&mut y, &m, out);
        y
    }
}

/// The audio VAE's encoder: a mono waveform to the posterior mean (`ref2va`'s reference audio).
/// DAC's waveform encoder (a convolution of 7, then per stride a block of three residual units
/// at dilations 1, 3 and 9, a Snake and a strided convolution of `2 * stride` doubling the
/// channels, then a Snake and a convolution of 3), [`PreBlock`] and the `mean_proj` head.
pub struct AudioEncoder {
    pub cfg: AudioVaeConfig,
    conv_in: Conv,
    blocks: Vec<EncBlock>,
    snake: Snake,
    conv_out: Conv,
    pre: PreBlock,
    mean: Conv,
}

impl AudioEncoder {
    /// Loads the encoder (float32), as [`AudioDecoder::load`].
    pub fn load(path: &Path) -> Result<Self> {
        let (cfg, files) = open(path)?;
        Self::from_files(cfg, &files)
    }

    pub fn from_files(cfg: AudioVaeConfig, f: &ModelFiles) -> Result<Self> {
        let mut blocks = Vec::new();
        let mut i = 1;
        while f.has(&format!("encoder.block.{i}.block.4.weight_v")) || f.has(&format!("encoder.block.{i}.block.4.weight")) {
            let p = format!("encoder.block.{i}.block");
            let (w, shape) = weight(f, &format!("{p}.4"))?;
            let [co, ci, k] = shape3(&shape, &format!("{p}.4"))?;
            if !k.is_multiple_of(2) {
                bail!("{p}.4: an odd kernel ({k}) for a strided convolution of 2 * stride");
            }
            let stride = k / 2;
            blocks.push(EncBlock {
                res: [1, 3, 9]
                    .iter()
                    .enumerate()
                    .map(|(j, &d)| EncRes::load(f, &format!("{p}.{j}"), d))
                    .collect::<Result<_>>()?,
                snake: Snake::load(f, &format!("{p}.3.alpha"))?,
                down: Conv::padded(&w, [co, ci, k], bias(f, &format!("{p}.4"), co)?, stride, stride.div_ceil(2)),
            });
            i += 1;
        }
        if blocks.is_empty() {
            bail!("audio VAE: no encoder blocks (encoder.block.1.block.4)");
        }
        let hop: usize = blocks.iter().map(|b| b.down.stride()).product();
        if hop != cfg.hop() {
            bail!("audio VAE: the encoder downsamples by {hop}, the decoder upsamples by {}", cfg.hop());
        }
        let mean = conv(f, "mean_proj", 1)?;
        if mean.c_out != cfg.latent_channels {
            bail!("mean_proj: {} channels for {} latent channels", mean.c_out, cfg.latent_channels);
        }
        Ok(Self {
            conv_in: conv(f, "encoder.block.0", 1)?,
            blocks,
            snake: Snake::load(f, &format!("encoder.block.{i}.alpha"))?,
            conv_out: conv(f, &format!("encoder.block.{}", i + 1), 1)?,
            pre: PreBlock::load(f, "pre_block", cfg.heads)?,
            mean,
            cfg,
        })
    }

    /// A mono waveform (right-padded with zeros to whole latents) to the posterior mean
    /// `[ceil(samples / hop), latent_channels]`, in the VAE's own scale.
    pub fn encode(&self, wave: &[f32]) -> Result<Vec<f32>> {
        if wave.is_empty() {
            bail!("audio VAE: nothing to encode");
        }
        let hop = self.cfg.hop();
        let n = wave.len().div_ceil(hop) * hop;
        let mut x0 = vec![0f32; n];
        x0[..wave.len()].copy_from_slice(wave);
        let (mut x, mut t) = self.conv_in.forward(&x0, n);
        for b in &self.blocks {
            for r in &b.res {
                r.forward(&mut x, t);
            }
            b.snake.apply(&mut x);
            let (y, nt) = b.down.forward(&x, t);
            bufs::give(std::mem::replace(&mut x, y));
            t = nt;
        }
        self.snake.apply(&mut x);
        let (y, _) = self.conv_out.forward(&x, t);
        bufs::give(x);
        let h = self.pre.forward(&y, t);
        bufs::give(y);
        let (m, _) = self.mean.forward(&h, t);
        bufs::clear();
        Ok(m)
    }

    /// The pipeline's rows of a soundtrack (one waveform per channel): each channel encoded on
    /// its own and normalized (`(z - latents_mean) / latents_std`), channel-major.
    pub fn encode_rows(&self, channels: &[Vec<f32>]) -> Result<Vec<f32>> {
        let c = self.cfg.latent_channels;
        let mut rows = Vec::new();
        for ch in channels {
            let z = self.encode(ch)?;
            rows.extend(z.iter().enumerate().map(|(i, &v)| (v - self.cfg.latents_mean[i % c]) / self.cfg.latents_std[i % c]));
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < 1e-5, "value {i}: {x} vs {y}");
        }
    }

    /// `ConvTranspose1d(k, stride, padding = (k - stride) / 2)` by its definition, for the
    /// release's kernels (9 by 5: not whole strides; 4 by 2).
    #[test]
    fn upsampling_matches_the_definition() {
        let mut seed = 7;
        for (k, stride) in [(9usize, 5usize), (4, 2), (7, 1), (16, 4)] {
            let (ci, co, t) = (3usize, 2usize, 11usize);
            let w = noise(ci * co * k, &mut seed);
            let b = noise(co, &mut seed);
            let x = noise(t * ci, &mut seed);
            let (y, n) = Up::new(&w, [ci, co, k], &b, stride).forward(&x, t);
            assert_eq!(n, t * stride);
            let pad = (k - stride) / 2;
            let mut want = vec![0f32; n * co];
            for (o, v) in want.iter_mut().enumerate() {
                *v = b[o % co];
            }
            for s in 0..t {
                for j in 0..k {
                    let Some(m) = (s * stride + j).checked_sub(pad).filter(|&m| m < n) else {
                        continue;
                    };
                    for i in 0..ci {
                        for o in 0..co {
                            want[m * co + o] += x[s * ci + i] * w[(i * co + o) * k + j];
                        }
                    }
                }
            }
            close(&y, &want);
        }
    }

    /// `Conv1d(k, dilation, padding = (k - 1) dilation / 2)` by its definition.
    #[test]
    fn centered_convolutions_match_the_definition() {
        let mut seed = 11;
        for (k, d) in [(7usize, 1usize), (3, 5), (11, 3), (1, 1)] {
            let (ci, co, t) = (3usize, 2usize, 40usize);
            let w = noise(co * ci * k, &mut seed);
            let b = noise(co, &mut seed);
            let x = noise(t * ci, &mut seed);
            let (y, n) = Conv::centered(&w, [co, ci, k], b.clone(), d).forward(&x, t);
            assert_eq!(n, t);
            let pad = (k - 1) * d / 2;
            let mut want = vec![0f32; t * co];
            for m in 0..t {
                for o in 0..co {
                    let mut acc = b[o];
                    for j in 0..k {
                        let Some(s) = (m + j * d).checked_sub(pad).filter(|&s| s < t) else {
                            continue;
                        };
                        for i in 0..ci {
                            acc += x[s * ci + i] * w[(o * ci + i) * k + j];
                        }
                    }
                    want[m * co + o] = acc;
                }
            }
            close(&y, &want);
        }
    }

    /// A constant goes through the resampling pair unchanged (a symmetric filter of sum 1,
    /// replicate padding), so the activation of a constant row is SnakeBeta of it.
    #[test]
    fn the_activation_keeps_constants() {
        let f: Vec<f32> = (0..12).map(|i| (1.0 + (i as f32 - 5.5).abs()).recip()).collect();
        let sum: f32 = f.iter().sum();
        let down: Vec<f32> = f.iter().map(|v| v / sum).collect();
        let (alpha, beta) = (0.7f32, 1.3f32);
        let act = Act {
            snake: Snake::new(vec![alpha; 2], vec![1.0 / beta; 2]),
            up: down.iter().map(|v| 2.0 * v).collect(),
            down,
        };
        let (t, v) = (9usize, 0.4f32);
        let y = act.apply(&vec![v; t * 2], t);
        close(&y, &vec![v + (alpha * v).sin().powi(2) / beta; t * 2]);
    }
}
