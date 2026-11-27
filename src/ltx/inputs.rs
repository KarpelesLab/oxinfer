//! Media given to LTX-2.5's pipelines: a soundtrack to make the video for (audio to video,
//! `a2vid_two_stage`) and a video to regenerate a time window of (retake, `retake.py`), decoded
//! as the reference's `decode_audio_from_file` / `decode_video_from_file` do.

use anyhow::{Result, bail};
use std::path::Path;

/// A waveform: one vector per channel in `[-1, 1]` at `rate` Hz.
#[derive(Clone, Debug, Default)]
pub struct AudioInput {
    pub channels: Vec<Vec<f32>>,
    pub rate: u32,
}

impl AudioInput {
    /// Seconds of audio.
    pub fn duration(&self) -> f64 {
        self.channels.first().map_or(0.0, |c| c.len() as f64 / self.rate.max(1) as f64)
    }

    /// The samples from `start` seconds on, at most `max_duration` seconds of them (cut to
    /// `round(max_duration rate)` samples, as `decode_audio_from_file`).
    pub fn window(&self, start: f64, max_duration: Option<f64>) -> Self {
        let r = self.rate as f64;
        let skip = (start * r).round().max(0.0) as usize;
        let keep = max_duration.map(|d| (d * r).round().max(0.0) as usize);
        Self {
            channels: self
                .channels
                .iter()
                .map(|c| {
                    let s = &c[skip.min(c.len())..];
                    s[..keep.unwrap_or(s.len()).min(s.len())].to_vec()
                })
                .collect(),
            rate: self.rate,
        }
    }

    /// The first audio stream of a media file (any container and codec oxideav reads).
    pub fn load(path: &Path) -> Result<Self> {
        #[cfg(feature = "media")]
        {
            let clip = crate::media::Media::new().decode_audio(path)?;
            let (channels, rate) = clip.to_channels()?;
            if channels.is_empty() || channels[0].is_empty() {
                bail!("{}: the audio stream has no samples", path.display());
            }
            Ok(Self { channels, rate })
        }
        #[cfg(not(feature = "media"))]
        bail!("built without the `media` feature: {} cannot be decoded", path.display())
    }
}

/// A video's frames (RGB8, `width` x `height`) at `fps`, with its soundtrack when it has one.
#[derive(Clone, Debug, Default)]
pub struct VideoInput {
    pub frames: Vec<Vec<u8>>,
    pub width: usize,
    pub height: usize,
    pub fps: f32,
    pub audio: Option<AudioInput>,
}

impl VideoInput {
    /// Every frame of a video file (or a folder of frame images at 24 fps) and its first audio
    /// stream, if any.
    pub fn load(path: &Path) -> Result<Self> {
        #[cfg(feature = "media")]
        {
            let media = crate::media::Media::new();
            let all = |n: usize, _: f64| (0..n).collect::<Vec<_>>();
            let clip = crate::video::load_clip(&media, path, &all, None)?;
            let audio = if path.is_dir() {
                None
            } else {
                match media.decode_audio(path) {
                    Ok(c) => {
                        let (channels, rate) = c.to_channels()?;
                        (!channels.is_empty() && !channels[0].is_empty()).then_some(AudioInput { channels, rate })
                    }
                    Err(_) => None,
                }
            };
            Ok(Self {
                frames: clip.frames,
                width: clip.width,
                height: clip.height,
                fps: clip.fps as f32,
                audio,
            })
        }
        #[cfg(not(feature = "media"))]
        bail!("built without the `media` feature: {} cannot be decoded", path.display())
    }
}

/// A retake (`RetakePipeline`): the time window `[start, end)` (seconds) of [`VideoInput`] is
/// regenerated from the prompt, the rest kept; `video` / `audio`: whether each stream is
/// regenerated in the window (else kept whole).
#[derive(Clone, Debug)]
pub struct Retake {
    pub source: VideoInput,
    pub start: f32,
    pub end: f32,
    pub video: bool,
    pub audio: bool,
}

/// `TemporalRegionMask`: whether a span `[t0, t1)` (seconds) meets `[start, end)`.
pub fn in_region(t0: f32, t1: f32, start: f32, end: f32) -> bool {
    t1 > start && t0 < end
}
