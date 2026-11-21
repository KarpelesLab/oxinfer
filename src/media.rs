//! Media decoding through oxideav (pure-Rust codecs): images to RGB8, audio to 16 kHz mono f32.
//!
//! Conversions of already-decoded frames (no file or codec code) are in [`crate::av`].

use crate::av;
use anyhow::{Context, Result, anyhow, bail};
use oxideav_core::{Frame, MediaType, RuntimeContext};
use std::path::Path;

pub use crate::audio::resample;
pub use crate::video::RgbImage;

pub struct Media {
    ctx: RuntimeContext,
}

impl Default for Media {
    fn default() -> Self {
        Self::new()
    }
}

/// Registers the codecs and containers `media` depends on, in the order `oxideav_meta::register_all`
/// calls them (alphabetical). oxinfer depends on these crates directly rather than on oxideav-meta,
/// so that crates registered by oxideav-meta can depend on oxinfer without a dependency cycle.
pub fn register(ctx: &mut RuntimeContext) {
    oxideav_aac::register(ctx);
    oxideav_av1::register(ctx);
    oxideav_avi::register(ctx);
    oxideav_basic::register(ctx);
    oxideav_bmp::register(ctx);
    oxideav_flac::register(ctx);
    oxideav_gif::register(ctx);
    oxideav_h264::register(ctx);
    oxideav_h265::register(ctx);
    oxideav_mjpeg::register(ctx);
    oxideav_mkv::register(ctx);
    oxideav_mov::registry::register(ctx);
    oxideav_mp3::register(ctx);
    oxideav_mp4::register(ctx);
    oxideav_mpeg4video::register(ctx);
    oxideav_ogg::register(ctx);
    oxideav_opus::register(ctx);
    oxideav_png::register(ctx);
    oxideav_vorbis::register(ctx);
    oxideav_vp8::register(ctx);
    oxideav_vp9::register(ctx);
    oxideav_webp::register(ctx);
}

impl Media {
    pub fn new() -> Self {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
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

    /// Decodes the first audio stream of a file: its decoded frames and stream parameters (see
    /// [`av::AudioClip`]).
    pub fn decode_audio(&self, path: &Path) -> Result<av::AudioClip> {
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
        let adts = AdtsHeader::for_stream(&stream.params);
        let mut frames = Vec::new();
        let take = |frame: Frame, frames: &mut Vec<oxideav_core::AudioFrame>| {
            if let Frame::Audio(af) = frame {
                frames.push(af);
            }
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
            let pkt = match adts.as_ref().and_then(|h| h.wrap(&pkt.data)) {
                Some(data) => oxideav_core::Packet { data, ..pkt },
                None => pkt,
            };
            decoder.send_packet(&pkt).map_err(|e| anyhow!("decode: {e}"))?;
            loop {
                match decoder.receive_frame() {
                    Ok(f) => take(f, &mut frames),
                    Err(e) if e.is_need_more() || e.is_eof() => break,
                    Err(e) => bail!("decode: {e}"),
                }
            }
        }
        let _ = decoder.flush();
        while let Ok(f) = decoder.receive_frame() {
            take(f, &mut frames);
        }
        Ok(av::AudioClip::new(frames, stream.params))
    }

    /// Decodes any supported audio container/codec and returns mono samples at `target_rate`.
    pub fn load_audio(&self, path: &Path, target_rate: usize) -> Result<Vec<f32>> {
        let clip = self.decode_audio(path)?;
        clip.to_mono(target_rate).with_context(|| path.display().to_string())
    }
}

/// The ADTS header of an AAC track whose packets are bare access units configured by an
/// `AudioSpecificConfig` (MP4's `esds`, Matroska's CodecPrivate): oxideav-aac 0.1.7, its latest
/// release, decodes only ADTS or LOAS ("packet has neither an ADTS nor a LOAS syncword"), so
/// those packets are given the ADTS header the configuration implies.
pub(crate) struct AdtsHeader {
    /// `profile_ObjectType` (`audioObjectType - 1`).
    profile: u8,
    /// `sampling_frequency_index`.
    rate_index: u8,
    /// `channel_configuration`.
    channels: u8,
}

impl AdtsHeader {
    /// For an `aac` stream with an AudioSpecificConfig ADTS can carry: AAC Main, LC, SSR or LTP
    /// (object types 1 to 4) at a tabulated rate, with a channel configuration of 1 to 7.
    pub(crate) fn for_stream(params: &oxideav_core::CodecParameters) -> Option<Self> {
        let asc = &params.extradata;
        if params.codec_id.as_str() != "aac" || asc.len() < 2 {
            return None;
        }
        let aot = asc[0] >> 3;
        let rate_index = ((asc[0] & 7) << 1) | (asc[1] >> 7);
        let channels = (asc[1] >> 3) & 15;
        ((1..=4).contains(&aot) && rate_index < 13 && (1..=7).contains(&channels)).then_some(Self {
            profile: aot - 1,
            rate_index,
            channels,
        })
    }

    /// `au` behind its 7-byte ADTS header; `None` when it already is an ADTS frame (a raw data
    /// block starts with a 3-bit element id below 7, never with the 12-bit syncword) or is too
    /// long for one.
    pub(crate) fn wrap(&self, au: &[u8]) -> Option<Vec<u8>> {
        let len = au.len() + 7;
        if (au.len() >= 2 && au[0] == 0xFF && au[1] & 0xF0 == 0xF0) || len >= 1 << 13 {
            return None;
        }
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(&[
            0xFF,
            // MPEG-4, layer 0, no CRC.
            0xF1,
            (self.profile << 6) | (self.rate_index << 2) | (self.channels >> 2),
            ((self.channels & 3) << 6) | (len >> 11) as u8,
            (len >> 3) as u8,
            ((len & 7) << 5) as u8 | 0x1F,
            // Buffer fullness 0x7FF (variable rate), one raw data block.
            0xFC,
        ]);
        out.extend_from_slice(au);
        Some(out)
    }
}
