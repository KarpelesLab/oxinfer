//! Media decoding through oxideav (pure-Rust codecs): images to RGB8, audio to 16 kHz mono f32.

use anyhow::{Context, Result, anyhow, bail};
use oxideav_core::{Frame, MediaType, RuntimeContext, SampleFormat};
use std::path::Path;

pub struct Media {
    ctx: RuntimeContext,
}

pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>, // interleaved RGB, row-major
}

impl Default for Media {
    fn default() -> Self {
        Self::new()
    }
}

impl Media {
    pub fn new() -> Self {
        let mut ctx = RuntimeContext::new();
        oxideav_meta::register_all(&mut ctx);
        Self { ctx }
    }

    /// The oxideav registries (containers and codecs).
    pub fn context(&self) -> &RuntimeContext {
        &self.ctx
    }

    pub fn load_image(&self, path: &Path) -> Result<RgbImage> {
        let file = oxideav_image::open(&self.ctx, path).map_err(|e| anyhow!("decode image {}: {e}", path.display()))?;
        let img = file.primary();
        let data = img.to_rgb8().map_err(|e| anyhow!("convert image to RGB: {e}"))?;
        Ok(RgbImage {
            width: img.width() as usize,
            height: img.height() as usize,
            data,
        })
    }

    /// [`Media::load_image`] for an encoded image in memory (PNG, JPEG, WebP, GIF, BMP).
    pub fn decode_image(&self, bytes: &[u8]) -> Result<RgbImage> {
        let file = oxideav_image::decode_bytes(&self.ctx, bytes).map_err(|e| anyhow!("decode image: {e}"))?;
        let img = file.primary();
        let data = img.to_rgb8().map_err(|e| anyhow!("convert image to RGB: {e}"))?;
        Ok(RgbImage {
            width: img.width() as usize,
            height: img.height() as usize,
            data,
        })
    }

    /// Decodes any supported audio container/codec and returns mono samples at `target_rate`.
    pub fn load_audio(&self, path: &Path, target_rate: usize) -> Result<Vec<f32>> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut input: Box<dyn oxideav_core::ReadSeek> = Box::new(file);
        let ext = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
        let format = self
            .ctx
            .containers
            .probe_input(&mut *input, ext.as_deref())
            .map_err(|e| anyhow!("detect audio format of {}: {e}", path.display()))?;
        let mut demuxer = self
            .ctx
            .containers
            .open_demuxer(&format, input, &self.ctx.codecs)
            .map_err(|e| anyhow!("open {format} demuxer: {e}"))?;
        let stream = demuxer
            .streams()
            .iter()
            .find(|s| s.params.media_type == MediaType::Audio)
            .cloned()
            .ok_or_else(|| anyhow!("{}: no audio stream", path.display()))?;
        let mut decoder = oxideav_pipeline::make_decoder(&self.ctx.codecs, &stream.params)
            .map_err(|e| anyhow!("no decoder for {:?}: {e}", stream.params.codec_id))?;

        let mut mono: Vec<f32> = Vec::new();
        let rate = stream.params.sample_rate.unwrap_or(0) as usize;
        let channels = stream.params.channels.unwrap_or(1).max(1) as usize;
        let declared = stream.params.sample_format;
        let take = |frame: &Frame, mono: &mut Vec<f32>| -> Result<()> {
            if let Frame::Audio(af) = frame {
                let fmt = declared
                    .or_else(|| infer_format(af, channels))
                    .ok_or_else(|| anyhow!("unknown sample format"))?;
                append_mono(af, fmt, channels, mono)?;
            }
            Ok(())
        };
        loop {
            let pkt = match demuxer.next_packet() {
                Ok(p) => p,
                Err(e) if e.is_eof() => break,
                Err(e) => bail!("demux error: {e}"),
            };
            if pkt.stream_index != stream.index {
                continue;
            }
            decoder.send_packet(&pkt).map_err(|e| anyhow!("decode: {e}"))?;
            loop {
                match decoder.receive_frame() {
                    Ok(f) => take(&f, &mut mono)?,
                    Err(e) if e.is_need_more() || e.is_eof() => break,
                    Err(e) => bail!("decode: {e}"),
                }
            }
        }
        let _ = decoder.flush();
        while let Ok(f) = decoder.receive_frame() {
            take(&f, &mut mono)?;
        }
        if rate == 0 {
            bail!("unknown sample rate");
        }
        Ok(if rate == target_rate {
            mono
        } else {
            resample(&mono, rate, target_rate)
        })
    }
}

/// Guesses the sample format from the frame's byte layout when the stream did not declare one.
fn infer_format(af: &oxideav_core::AudioFrame, channels: usize) -> Option<SampleFormat> {
    let n = af.samples as usize;
    if n == 0 || af.data.is_empty() {
        return None;
    }
    let planar = af.data.len() > 1;
    let per_sample = if planar {
        af.data[0].len() / n
    } else {
        af.data[0].len() / (n * channels)
    };
    Some(match (per_sample, planar) {
        (1, _) => SampleFormat::U8,
        (2, false) => SampleFormat::S16,
        (2, true) => SampleFormat::S16P,
        (3, _) => SampleFormat::S24,
        (4, false) => SampleFormat::F32,
        (4, true) => SampleFormat::F32P,
        (8, false) => SampleFormat::F64,
        (8, true) => SampleFormat::F64P,
        _ => return None,
    })
}

/// Appends the frame's samples, averaged across channels, as f32 in [-1, 1].
fn append_mono(af: &oxideav_core::AudioFrame, fmt: SampleFormat, channels: usize, out: &mut Vec<f32>) -> Result<()> {
    let n = af.samples as usize;
    let planar = af.data.len() > 1;
    let sample = |plane: &[u8], i: usize| -> f32 {
        match fmt {
            SampleFormat::U8 | SampleFormat::U8P => (plane[i] as f32 - 128.0) / 128.0,
            SampleFormat::S8 => plane[i] as i8 as f32 / 128.0,
            SampleFormat::S16 | SampleFormat::S16P => i16::from_le_bytes([plane[2 * i], plane[2 * i + 1]]) as f32 / 32768.0,
            SampleFormat::S24 => {
                let v = (plane[3 * i] as i32) | ((plane[3 * i + 1] as i32) << 8) | ((plane[3 * i + 2] as i8 as i32) << 16);
                v as f32 / 8_388_608.0
            }
            SampleFormat::S32 | SampleFormat::S32P => {
                i32::from_le_bytes(plane[4 * i..4 * i + 4].try_into().unwrap()) as f32 / 2_147_483_648.0
            }
            SampleFormat::F32 | SampleFormat::F32P => f32::from_le_bytes(plane[4 * i..4 * i + 4].try_into().unwrap()),
            SampleFormat::F64 | SampleFormat::F64P => f64::from_le_bytes(plane[8 * i..8 * i + 8].try_into().unwrap()) as f32,
            _ => f32::NAN,
        }
    };
    out.reserve(n);
    for s in 0..n {
        let mut acc = 0f32;
        for c in 0..channels {
            let v = if planar {
                sample(&af.data[c], s)
            } else {
                sample(&af.data[0], s * channels + c)
            };
            if v.is_nan() {
                bail!("unsupported sample format {fmt:?}");
            }
            acc += v;
        }
        out.push(acc / channels as f32);
    }
    Ok(())
}

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
