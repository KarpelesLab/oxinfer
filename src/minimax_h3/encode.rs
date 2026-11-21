//! Generated videos as files, and references read from files (the `media` feature): Motion JPEG
//! with the soundtrack in AVI ([`avi_mjpeg`]), animated WebP ([`animated_webp`]), and
//! [`load_reference`] for ref2va (an image, an audio clip or a video with its soundtrack).
//!
//! No pure-Rust H.264, VP9 or AV1 encoder is available in the oxideav crates oxinfer uses, so
//! there is no `.mp4` / `.webm` output: AVI plays in ffmpeg-based players (`ffmpeg -i clip.avi
//! clip.mp4` converts it), animated WebP in browsers (without sound).

use super::ref2va::RefInput;
use super::video::{FPS, Soundtrack, Video};
use crate::sd::Image;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// An image file as RGB8.
pub fn load_image(path: &Path) -> Result<Image> {
    let im = crate::media::Media::new()
        .load_image(path)
        .with_context(|| format!("load {}", path.display()))?;
    Ok(Image {
        width: im.width,
        height: im.height,
        rgb: im.data,
    })
}

/// File extensions read as images and as audio clips by [`load_reference`].
pub const IMAGE_EXTENSIONS: [&str; 9] = ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tga", "tif", "tiff"];
pub const AUDIO_EXTENSIONS: [&str; 8] = ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", "f32"];

/// A reference file decoded at its own rates: an image or an audio clip (by extension), anything
/// else a video (or a folder of frames) with its soundtrack when it has one.
pub fn load_reference(path: &Path) -> Result<RefInput> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        return Ok(RefInput::Image(load_image(path)?));
    }
    let media = crate::media::Media::new();
    let sound =
        |p: &Path| -> Result<(Vec<Vec<f32>>, u32)> { media.decode_audio(p)?.to_channels().with_context(|| p.display().to_string()) };
    if AUDIO_EXTENSIONS.contains(&ext.as_str()) {
        let (c, rate) = sound(path)?;
        return Ok(RefInput::Audio(c, rate));
    }
    let all = |n: usize, _fps: f64| (0..n).collect::<Vec<_>>();
    let clip = crate::video::load_clip(&media, path, &all, None).with_context(|| format!("decode {}", path.display()))?;
    let audio = if path.is_dir() { None } else { sound(path).ok() };
    let (w, h) = (clip.width, clip.height);
    Ok(RefInput::Video {
        frames: clip.frames.into_iter().map(|rgb| Image { width: w, height: h, rgb }).collect(),
        fps: clip.fps,
        audio,
    })
}

/// Motion JPEG frames in an AVI (RIFF) container with an index; the soundtrack, when there is
/// one, as a second stream of 16-bit PCM, one chunk after each frame.
pub fn avi_mjpeg(v: &Video) -> Result<Vec<u8>> {
    if v.frames.is_empty() {
        bail!("no frames to write");
    }
    let opts = oxideav_mjpeg::EncodeOptions::default();
    let jpegs: Vec<Vec<u8>> = v
        .frames
        .iter()
        .map(|f| oxideav_mjpeg::encode_rgb8(v.width as u32, v.height as u32, f, &opts).map_err(|e| anyhow::anyhow!("JPEG encode: {e}")))
        .collect::<Result<_>>()?;
    let u = |x: usize| (x as u32).to_le_bytes();
    let chunk = |id: &[u8; 4], data: &[u8]| {
        let mut c = id.to_vec();
        c.extend_from_slice(&u(data.len()));
        c.extend_from_slice(data);
        if data.len() % 2 == 1 {
            c.push(0);
        }
        c
    };
    let list = |kind: &[u8; 4], body: &[u8]| {
        let mut c = b"LIST".to_vec();
        c.extend_from_slice(&u(body.len() + 4));
        c.extend_from_slice(kind);
        c.extend_from_slice(body);
        c
    };
    let (n, max) = (jpegs.len(), jpegs.iter().map(Vec::len).max().unwrap_or(0));
    // The soundtrack: its rate, bytes per sample of every channel, and the samples of each frame
    // (the last frame's chunk takes what is left).
    let audio = v.audio.as_ref().filter(|a| !a.is_empty() && n > 0);
    let (rate, block) = audio.map_or((0, 0), |a| (a.sample_rate as usize, 2 * a.channels.len()));
    let span = |a: &Soundtrack, i: usize| {
        let end = if i + 1 == n { a.len() } else { ((i + 1) * rate / FPS).min(a.len()) };
        (i * rate / FPS).min(end)..end
    };
    let max_audio = audio.map_or(0, |a| (0..n).map(|i| span(a, i).len() * block).max().unwrap_or(0));
    // avih: microseconds per frame, max bytes / s, padding, flags (has index), frames, initial
    // frames, streams, buffer size, width, height, reserved x4.
    let mut avih = Vec::new();
    let streams = 1 + audio.is_some() as usize;
    let rest = [0, 0x10, n, 0, streams, max.max(max_audio), v.width, v.height, 0, 0, 0, 0];
    for x in [1_000_000 / FPS, max * FPS + rate * block].into_iter().chain(rest) {
        avih.extend_from_slice(&u(x));
    }
    // strh: type, handler, flags, priority + language, initial frames, scale, rate, start, length,
    // buffer size, quality, sample size, frame rectangle.
    let mut strh = b"vidsMJPG".to_vec();
    for x in [0, 0, 0, 1, FPS, 0, n, max, 0xFFFF_FFFF, 0] {
        strh.extend_from_slice(&u(x));
    }
    for x in [0, 0, v.width, v.height] {
        strh.extend_from_slice(&(x as u16).to_le_bytes());
    }
    // strf: BITMAPINFOHEADER.
    let mut strf = Vec::new();
    strf.extend_from_slice(&u(40));
    strf.extend_from_slice(&u(v.width));
    strf.extend_from_slice(&u(v.height));
    strf.extend_from_slice(&1u16.to_le_bytes());
    strf.extend_from_slice(&24u16.to_le_bytes());
    strf.extend_from_slice(b"MJPG");
    for x in [v.width * v.height * 3, 0, 0, 0, 0] {
        strf.extend_from_slice(&u(x));
    }
    let mut strls = list(b"strl", &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat());
    if let Some(a) = audio {
        // strh as above: one unit of `scale / rate` seconds is a sample of every channel.
        let mut strh = b"auds\0\0\0\0".to_vec();
        for x in [0, 0, 0, block, rate * block, 0, a.len(), max_audio, 0xFFFF_FFFF, block] {
            strh.extend_from_slice(&u(x));
        }
        strh.extend_from_slice(&[0; 8]);
        // strf: WAVEFORMATEX, integer PCM.
        let mut strf = Vec::new();
        strf.extend_from_slice(&1u16.to_le_bytes());
        strf.extend_from_slice(&(a.channels.len() as u16).to_le_bytes());
        strf.extend_from_slice(&u(rate));
        strf.extend_from_slice(&u(rate * block));
        strf.extend_from_slice(&(block as u16).to_le_bytes());
        strf.extend_from_slice(&16u16.to_le_bytes());
        strf.extend_from_slice(&0u16.to_le_bytes());
        strls.extend(list(b"strl", &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat()));
    }
    let hdrl = list(b"hdrl", &[chunk(b"avih", &avih), strls].concat());
    let (mut movi, mut idx) = (Vec::new(), Vec::new());
    let mut add = |id: &[u8; 4], data: &[u8]| {
        idx.extend_from_slice(id);
        idx.extend_from_slice(&u(0x10));
        idx.extend_from_slice(&u(movi.len() + 4));
        idx.extend_from_slice(&u(data.len()));
        movi.extend(chunk(id, data));
    };
    for (i, j) in jpegs.iter().enumerate() {
        add(b"00dc", j);
        if let Some(a) = audio {
            add(b"01wb", &crate::tts::pcm16_bytes(&a.interleaved(span(a, i))));
        }
    }
    let body = [hdrl, list(b"movi", &movi), chunk(b"idx1", &idx)].concat();
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&u(body.len() + 4));
    out.extend_from_slice(b"AVI ");
    out.extend_from_slice(&body);
    Ok(out)
}

/// The frames as an animated WebP (lossless; no audio), 24 fps in whole milliseconds.
pub fn animated_webp(v: &Video) -> Result<Vec<u8>> {
    use oxideav_webp::anim_encode::AnimFrame;
    let frames: Vec<AnimFrame> = v
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let rgba: Vec<u8> = f.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
            // 41.67 ms on average.
            let d = ((i + 1) * 1000 / FPS - i * 1000 / FPS) as u32;
            AnimFrame::new(v.width as u32, v.height as u32, rgba, d)
        })
        .collect();
    oxideav_webp::encode_animation_frames(&frames, &oxideav_webp::Metadata::default(), &oxideav_webp::EncodeOptions::new())
        .map_err(|e| anyhow::anyhow!("WebP encode: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::MediaType;

    /// The AVI demuxed by oxideav: every frame and every audio sample comes back.
    #[test]
    fn avi_carries_the_soundtrack() {
        let (w, h, n) = (32usize, 16usize, 5usize);
        let frames: Vec<Vec<u8>> = (0..n).map(|f| (0..w * h * 3).map(|i| (i * 7 + f * 40) as u8).collect()).collect();
        // A little longer than the frames, as the model's soundtracks are.
        let len = n * 32000 / FPS + 123;
        let ramp = |k: usize| -> Vec<f32> { (0..len).map(|i| ((i * k) % 2000) as f32 / 1000.0 - 1.0).collect() };
        let audio = Soundtrack {
            sample_rate: 32000,
            channels: vec![ramp(3), ramp(7)],
        };
        let want = crate::tts::pcm16_bytes(&audio.interleaved(0..len));
        let v = Video {
            width: w,
            height: h,
            frames,
            audio: Some(audio),
        };
        let avi = avi_mjpeg(&v).unwrap();
        let media = crate::media::Media::new();
        let ctx = media.context();
        let mut input: Box<dyn oxideav_core::ReadSeek> = Box::new(std::io::Cursor::new(avi));
        let format = ctx.containers.probe_input(&mut *input, Some("avi")).unwrap();
        let mut demux = ctx.containers.open_demuxer(&format, input, &ctx.codecs).unwrap();
        let streams = demux.streams().to_vec();
        assert_eq!(streams.len(), 2);
        let a = streams.iter().find(|s| s.params.media_type == MediaType::Audio).unwrap();
        assert_eq!(a.params.sample_rate, Some(32000));
        assert_eq!(a.params.channels, Some(2));
        let (mut video_packets, mut pcm) = (0, Vec::new());
        while let Ok(p) = demux.next_packet() {
            if p.stream_index == a.index {
                pcm.extend_from_slice(&p.data);
            } else {
                video_packets += 1;
            }
        }
        assert_eq!(video_packets, n);
        assert_eq!(pcm, want);
        let webp = animated_webp(&v).unwrap();
        assert_eq!(&webp[..4], b"RIFF");
        assert_eq!(&webp[8..12], b"WEBP");
    }
}
