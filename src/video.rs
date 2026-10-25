//! Video input: frame sampling as the Hugging Face video processors do it, and (with the
//! `media` feature) decoding through oxideav, from a video file or from a directory of frames.
//! Decoded frames in memory are [`crate::av::VideoClip`] (the `oxideav` feature).
//!
//! oxideav (pure Rust) demuxes MP4 / MOV / Matroska / WebM / AVI and decodes H.264, H.265, AV1,
//! VP8, VP9, MPEG-4 Part 2 and Motion JPEG; a directory of image files (sorted by name) is read
//! as an image sequence at `VideoOptions::frames_fps`.

/// Frame sampling. `None` fields take the model's processor defaults: Qwen samples `fps`
/// frames per second (2) between 4 and 768 frames; Gemma 4 takes 32 frames spread uniformly.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoOptions {
    /// Frames sampled per second of video.
    pub fps: Option<f64>,
    /// Most frames sampled (Gemma 4: the number of frames when `fps` is not set).
    pub max_frames: Option<usize>,
    /// Frame rate of a directory of frames (default 24, the rate the processors assume for a
    /// video without one).
    pub frames_fps: Option<f64>,
}

/// The rate assumed for videos that do not declare one.
pub const DEFAULT_FPS: f64 = 24.0;

/// `Qwen3VLVideoProcessor.sample_frames`: `fps` frames per second of the clip, clamped to
/// `[min_frames, max_frames]` and to the frame count, spread evenly from the first to the last
/// frame (numpy `linspace(...).round()`, half to even).
pub fn sample_qwen(total: usize, video_fps: f64, fps: f64, min_frames: usize, max_frames: usize) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    let want = (total as f64 / video_fps * fps) as usize;
    let n = want.max(min_frames).min(max_frames).min(total).max(1);
    linspace_round(total - 1, n)
}

/// numpy `linspace(0, last, n).round().astype(int)`.
fn linspace_round(last: usize, n: usize) -> Vec<usize> {
    if n == 1 {
        return vec![0];
    }
    let step = last as f64 / (n - 1) as f64;
    (0..n)
        .map(|i| {
            if i == n - 1 {
                last
            } else {
                (i as f64 * step).round_ties_even() as usize
            }
        })
        .collect()
}

/// `default_sample_indices_fn` (Gemma 4): `num_frames` frames at `i * total / num_frames`
/// (the frame count when `fps` is given: `total / video_fps * fps`), never more than `total`.
pub fn sample_uniform(total: usize, video_fps: f64, fps: Option<f64>, num_frames: usize) -> Vec<usize> {
    let n = match fps {
        Some(f) => ((total as f64 / video_fps * f) as usize).min(num_frames),
        None => num_frames,
    }
    .min(total)
    .max(1);
    if total == 0 {
        return Vec::new();
    }
    (0..n).map(|i| i * total / n).collect()
}

/// A decoded image: tightly packed RGB8.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    /// Interleaved RGB, row-major.
    pub data: Vec<u8>,
}

/// Sampled frames of a clip.
pub struct Clip {
    /// RGB8 frames, `width` x `height` each, in sampling order.
    pub frames: Vec<Vec<u8>>,
    pub width: usize,
    pub height: usize,
    /// Index of each sampled frame in the whole clip.
    pub indices: Vec<usize>,
    /// Frame rate of the clip.
    pub fps: f64,
    /// Frames in the clip.
    pub total: usize,
    /// Presentation time of each sampled frame in seconds from the first frame, when the frames
    /// carry timestamps (in-memory clips, [`crate::av::VideoClip`]); `None`: index / fps.
    pub pts: Option<Vec<f64>>,
}

impl Clip {
    /// Timestamp of each sampled frame in seconds.
    pub fn times(&self) -> Vec<f64> {
        match &self.pts {
            Some(t) => t.clone(),
            None => self.indices.iter().map(|&i| i as f64 / self.fps).collect(),
        }
    }
}

/// Frame selection: `(total frames, fps) -> indices`.
pub type Sampler<'s> = &'s dyn Fn(usize, f64) -> Vec<usize>;

#[cfg(feature = "media")]
pub use decode::load_clip;

#[cfg(feature = "media")]
mod decode {
    use super::*;
    use crate::media::Media;
    use anyhow::{Context, Result, anyhow, bail};
    use oxideav_core::{CodecParameters, Frame, MediaType, VideoFrame};
    use std::path::Path;

    /// Image files of a frame directory, sorted by name.
    fn frame_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let exts = ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tga", "tif", "tiff", "qoi"];
        let mut v: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| exts.contains(&e.to_ascii_lowercase().as_str()))
            })
            .collect();
        v.sort();
        if v.is_empty() {
            bail!("{}: no image files", dir.display());
        }
        Ok(v)
    }

    /// Decodes the frames `select` picks from a video file or a directory of frames
    /// (`frames_fps` frames per second, default `DEFAULT_FPS`).
    pub fn load_clip(media: &Media, path: &Path, select: Sampler, frames_fps: Option<f64>) -> Result<Clip> {
        if path.is_dir() {
            let files = frame_files(path)?;
            let fps = frames_fps.unwrap_or(DEFAULT_FPS);
            let indices = select(files.len(), fps);
            let mut frames = Vec::with_capacity(indices.len());
            let (mut w, mut h) = (0, 0);
            for &i in &indices {
                let img = media.load_image(&files[i])?;
                if frames.is_empty() {
                    (w, h) = (img.width, img.height);
                } else if (img.width, img.height) != (w, h) {
                    bail!(
                        "{}: frame {} is {}x{}, not {w}x{h}",
                        path.display(),
                        files[i].display(),
                        img.width,
                        img.height
                    );
                }
                frames.push(img.data);
            }
            return Ok(Clip {
                frames,
                width: w,
                height: h,
                indices,
                fps,
                total: files.len(),
                pts: None,
            });
        }
        let ctx = media.context();
        let open = || -> Result<Box<dyn oxideav_core::Demuxer>> {
            let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
            let mut input: Box<dyn oxideav_core::ReadSeek> = Box::new(file);
            let ext = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
            let format = ctx
                .containers
                .probe_input(&mut *input, ext.as_deref())
                .map_err(|e| anyhow!("detect the container of {}: {e}", path.display()))?;
            ctx.containers
                .open_demuxer(&format, input, &ctx.codecs)
                .map_err(|e| anyhow!("open {format} demuxer: {e}"))
        };
        // Pass 1: count the video packets (one per frame) for the sampling.
        let mut demux = open()?;
        let stream = demux
            .streams()
            .iter()
            .find(|s| s.params.media_type == MediaType::Video)
            .cloned()
            .ok_or_else(|| anyhow!("{}: no video stream", path.display()))?;
        let mut total = 0usize;
        let (mut first_pts, mut last_pts) = (None, None);
        loop {
            match demux.next_packet() {
                Ok(p) if p.stream_index == stream.index => {
                    total += 1;
                    if let Some(t) = p.pts {
                        first_pts = Some(first_pts.map_or(t, |f: i64| f.min(t)));
                        last_pts = Some(last_pts.map_or(t, |l: i64| l.max(t)));
                    }
                }
                Ok(_) => {}
                Err(e) if e.is_eof() => break,
                Err(e) => bail!("{}: demux error: {e}", path.display()),
            }
        }
        if total == 0 {
            bail!("{}: the video stream has no frames", path.display());
        }
        let tb = stream.time_base.0;
        let secs = |ticks: i64| ticks as f64 * tb.num as f64 / tb.den as f64;
        let fps = match (stream.params.frame_rate, stream.duration, first_pts, last_pts) {
            (Some(r), ..) if r.num > 0 && r.den > 0 => r.num as f64 / r.den as f64,
            (_, Some(d), ..) if d > 0 => total as f64 / secs(d),
            (_, _, Some(a), Some(b)) if b > a && total > 1 => (total - 1) as f64 / secs(b - a),
            _ => DEFAULT_FPS,
        };
        let indices = select(total, fps);
        let mut want = indices.clone();
        want.sort_unstable();
        want.dedup();

        // Pass 2: decode everything, keep the sampled frames.
        let mut params = stream.params.clone();
        // Some MOV files carry the codec configuration box with its header.
        if params.extradata.len() > 8 && matches!(&params.extradata[4..8], b"avcC" | b"hvcC" | b"av1C" | b"vpcC") {
            params.extradata.drain(..8);
        }
        let mut demux = open()?;
        let mut decoder = oxideav_pipeline::make_decoder(&ctx.codecs, &params)
            .map_err(|e| anyhow!("{}: no decoder for {:?}: {e}", path.display(), params.codec_id))?;
        let mut kept: Vec<(usize, Vec<u8>)> = Vec::with_capacity(want.len());
        let mut size = None;
        let mut n = 0usize;
        let mut take = |f: Frame, kept: &mut Vec<(usize, Vec<u8>)>| -> Result<()> {
            if let Frame::Video(v) = f {
                if want.binary_search(&n).is_ok() {
                    let (rgb, w, h) = to_rgb(v, &params)?;
                    match size {
                        None => size = Some((w, h)),
                        Some(s) if s != (w, h) => bail!("{}: the frame size changes mid-stream", path.display()),
                        _ => {}
                    }
                    kept.push((n, rgb));
                }
                n += 1;
            }
            Ok(())
        };
        let last = *want.last().unwrap();
        loop {
            let pkt = match demux.next_packet() {
                Ok(p) => p,
                Err(e) if e.is_eof() => break,
                Err(e) => bail!("{}: demux error: {e}", path.display()),
            };
            if pkt.stream_index != stream.index {
                continue;
            }
            decoder.send_packet(&pkt).map_err(|e| anyhow!("{}: decode: {e}", path.display()))?;
            loop {
                match decoder.receive_frame() {
                    Ok(f) => take(f, &mut kept)?,
                    Err(e) if e.is_need_more() || e.is_eof() => break,
                    Err(e) => bail!("{}: decode: {e}", path.display()),
                }
            }
            if kept.last().is_some_and(|k| k.0 == last) {
                break;
            }
        }
        if !kept.last().is_some_and(|k| k.0 == last) {
            let _ = decoder.flush();
            while let Ok(f) = decoder.receive_frame() {
                take(f, &mut kept)?;
            }
        }
        if kept.is_empty() {
            bail!("{}: no frame decoded", path.display());
        }
        let (w, h) = size.unwrap();
        // Frames past the decoder's output (fewer frames than packets) repeat the last one.
        let frames = indices
            .iter()
            .map(|&i| {
                let k = kept.partition_point(|k| k.0 <= i).max(1) - 1;
                kept[k].1.clone()
            })
            .collect();
        Ok(Clip {
            frames,
            width: w,
            height: h,
            indices,
            fps,
            total,
            pts: None,
        })
    }

    /// Converts a decoded frame to RGB8 through oxideav-image. Decoders do not always report
    /// their output pixel format in the stream parameters; it is then inferred from the plane
    /// layout ([`crate::av::complete_params`]).
    fn to_rgb(v: VideoFrame, params: &CodecParameters) -> Result<(Vec<u8>, usize, usize)> {
        let p = crate::av::complete_params(&v, params)?;
        let img = oxideav_image::Image::from_video_frame(v, &p).map_err(|e| anyhow!("decoded frame: {e}"))?;
        let rgb = img.to_rgb8().map_err(|e| anyhow!("convert frame to RGB: {e}"))?;
        Ok((rgb, img.width() as usize, img.height() as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_sampling_matches_reference() {
        // np.linspace(0, 119, 8).round() for a 4 s, 30 fps clip at 2 fps.
        assert_eq!(sample_qwen(120, 30.0, 2.0, 4, 768), vec![0, 17, 34, 51, 68, 85, 102, 119]);
        // Short clips still get min_frames (bounded by the frame count).
        assert_eq!(sample_qwen(10, 30.0, 2.0, 4, 768), vec![0, 3, 6, 9]);
        assert_eq!(sample_qwen(3, 30.0, 2.0, 4, 768), vec![0, 1, 2]);
        // linspace(0, 9, 4) = [0, 3, 6, 9]; linspace(0, 5, 3) = [0, 2.5, 5] -> round half even.
        assert_eq!(linspace_round(5, 3), vec![0, 2, 5]);
    }

    #[test]
    fn uniform_sampling_matches_reference() {
        assert_eq!(sample_uniform(120, 30.0, None, 4), vec![0, 30, 60, 90]);
        assert_eq!(sample_uniform(10, 30.0, None, 32).len(), 10);
        assert_eq!(sample_uniform(120, 30.0, Some(1.0), 32), vec![0, 30, 60, 90]);
    }
}
