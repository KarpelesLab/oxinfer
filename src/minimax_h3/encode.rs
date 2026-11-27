//! Generated videos as files, and references read from files (the `media` feature): H.264 and
//! AAC in MP4 or VP9 and Opus in WebM ([`mp4_or_webm`], oxideav's encoders and muxers), Motion
//! JPEG with the soundtrack in AVI ([`avi_mjpeg`]), animated WebP ([`animated_webp`]), and
//! [`load_reference`] for ref2va (an image, an audio clip or a video with its soundtrack).

use super::ref2va::RefInput;
#[cfg(test)]
use super::video::FPS;
use super::video::{Soundtrack, Video};
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
        let end = if i + 1 == n {
            a.len()
        } else {
            ((i + 1) * rate / v.fps).min(a.len())
        };
        (i * rate / v.fps).min(end)..end
    };
    let max_audio = audio.map_or(0, |a| (0..n).map(|i| span(a, i).len() * block).max().unwrap_or(0));
    // avih: microseconds per frame, max bytes / s, padding, flags (has index), frames, initial
    // frames, streams, buffer size, width, height, reserved x4.
    let mut avih = Vec::new();
    let streams = 1 + audio.is_some() as usize;
    let rest = [0, 0x10, n, 0, streams, max.max(max_audio), v.width, v.height, 0, 0, 0, 0];
    for x in [1_000_000 / v.fps, max * v.fps + rate * block].into_iter().chain(rest) {
        avih.extend_from_slice(&u(x));
    }
    // strh: type, handler, flags, priority + language, initial frames, scale, rate, start, length,
    // buffer size, quality, sample size, frame rectangle.
    let mut strh = b"vidsMJPG".to_vec();
    for x in [0, 0, 0, 1, v.fps, 0, n, max, 0xFFFF_FFFF, 0] {
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

pub use super::video::VideoFormat;

/// `v` in `format` (see [`mp4_or_webm`] and [`avi_mjpeg`]).
pub fn encode_video(v: &Video, format: VideoFormat) -> Result<Vec<u8>> {
    match format {
        VideoFormat::Avi => avi_mjpeg(v),
        f => mp4_or_webm(v, f),
    }
}

/// An in-memory `Write + Seek` a muxer can own while the caller keeps the bytes.
#[derive(Clone, Default)]
struct SharedBuf(std::sync::Arc<std::sync::Mutex<std::io::Cursor<Vec<u8>>>>);

impl std::io::Write for SharedBuf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl std::io::Seek for SharedBuf {
    fn seek(&mut self, p: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).seek(p)
    }
}

/// RGB8 to planar YUV 4:2:0, BT.601 limited range (what players assume of an H.264 / VP9
/// stream without colour signalling at these sizes).
pub fn rgb_to_yuv420(rgb: &[u8], w: usize, h: usize) -> [Vec<u8>; 3] {
    let px = |x: usize, y: usize| {
        let i = (y * w + x) * 3;
        (rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32)
    };
    let mut yp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, y);
            yp[y * w + x] = (16.0 + 0.257 * r + 0.504 * g + 0.098 * b).round().clamp(0.0, 255.0) as u8;
        }
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (mut u, mut v) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut r, mut g, mut b, mut n) = (0.0, 0.0, 0.0, 0.0);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = (cx * 2 + dx, cy * 2 + dy);
                if x < w && y < h {
                    let p = px(x, y);
                    (r, g, b, n) = (r + p.0, g + p.1, b + p.2, n + 1.0);
                }
            }
            let (r, g, b) = (r / n, g / n, b / n);
            u[cy * cw + cx] = (128.0 - 0.148 * r - 0.291 * g + 0.439 * b).round().clamp(0.0, 255.0) as u8;
            v[cy * cw + cx] = (128.0 + 0.439 * r - 0.368 * g - 0.071 * b).round().clamp(0.0, 255.0) as u8;
        }
    }
    [yp, u, v]
}

/// Planar YUV 4:2:0 (BT.601 limited range) back to RGB8: for checking an encode.
pub fn yuv420_to_rgb(planes: [&[u8]; 3], strides: [usize; 3], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let yy = planes[0][y * strides[0] + x] as f32 - 16.0;
            let u = planes[1][(y / 2) * strides[1] + x / 2] as f32 - 128.0;
            let v = planes[2][(y / 2) * strides[2] + x / 2] as f32 - 128.0;
            let c = |f: f32| f.round().clamp(0.0, 255.0) as u8;
            let i = (y * w + x) * 3;
            out[i] = c(1.164 * yy + 1.596 * v);
            out[i + 1] = c(1.164 * yy - 0.392 * u - 0.813 * v);
            out[i + 2] = c(1.164 * yy + 2.017 * u);
        }
    }
    out
}

fn av(e: oxideav_core::Error) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// An RGB8 image (`w` x `h`, both even) through H.264 and back: one intra picture at constant
/// QP `qp` (4:2:0, BT.601 limited range, CABAC), decoded by oxideav's decoder. What a still
/// looks like after a video codec (LTX-2.5 conditions on images re-compressed so). oxideav's
/// encoder takes sides that are multiples of 16 only: the picture is padded by repeating its
/// last column and row, as x264 pads its macroblocks, and cropped back.
pub fn h264_roundtrip(rgb: &[u8], w: usize, h: usize, qp: u32) -> Result<Vec<u8>> {
    use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, Rational, VideoFrame, VideoPlane};
    if w == 0 || h == 0 || !w.is_multiple_of(2) || !h.is_multiple_of(2) || rgb.len() != w * h * 3 {
        bail!("h264_roundtrip: a {w}x{h} image (even sides) of {} bytes", rgb.len());
    }
    let (ow, oh) = (w, h);
    let (w, h) = (w.next_multiple_of(16), h.next_multiple_of(16));
    let padded: Vec<u8> = if (w, h) == (ow, oh) {
        rgb.to_vec()
    } else {
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            let row = &rgb[y.min(oh - 1) * ow * 3..][..ow * 3];
            v.extend_from_slice(row);
            for _ in ow..w {
                v.extend_from_slice(&row[(ow - 1) * 3..]);
            }
        }
        v
    };
    let rgb = &padded[..];
    let media = crate::media::Media::new();
    let ctx = media.context();
    let mut vp = CodecParameters::video(CodecId::new("h264"));
    vp.width = Some(w as u32);
    vp.height = Some(h as u32);
    vp.pixel_format = Some(PixelFormat::Yuv420P);
    vp.frame_rate = Some(Rational::new(1, 1));
    vp.options = oxideav_core::CodecOptions::new()
        .set("format", "avcc")
        .set("cabac", "true")
        .set("rc", "cqp")
        .set("qp", qp.min(51).to_string())
        .set("gop", "1");
    let mut enc = ctx.codecs.first_encoder(&vp).map_err(av).context("the h264 encoder")?;
    let [y, u, v] = rgb_to_yuv420(rgb, w, h);
    let cw = w / 2;
    enc.send_frame(&Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane { stride: cw, data: u },
            VideoPlane { stride: cw, data: v },
        ],
    }))
    .map_err(av)?;
    enc.flush().map_err(av)?;
    let mut packets = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        packets.push(p);
    }
    let mut dec = ctx
        .codecs
        .first_decoder(enc.output_params())
        .map_err(av)
        .context("the h264 decoder")?;
    let mut out = None;
    let mut take = |d: &mut Box<dyn oxideav_core::Decoder>| {
        while let Ok(Frame::Video(vf)) = d.receive_frame() {
            let p = vf.image_planes();
            out.get_or_insert_with(|| yuv420_to_rgb([&p[0].data, &p[1].data, &p[2].data], [p[0].stride, p[1].stride, p[2].stride], w, h));
        }
    };
    for p in &packets {
        dec.send_packet(p).map_err(av)?;
        take(&mut dec);
    }
    let _ = dec.flush();
    take(&mut dec);
    let out = out.context("the h264 decoder gave no picture")?;
    if (w, h) == (ow, oh) {
        return Ok(out);
    }
    Ok((0..oh).flat_map(|y| out[y * w * 3..][..ow * 3].to_vec()).collect())
}

/// Frames per independently encoded chunk of the video: H.264's IDR period and VP9's keyframe
/// interval (one second). Each chunk starts with an IDR picture / keyframe, so the chunks are
/// encoded in parallel, by one encoder each, and their packets concatenated. VP9's encoder codes
/// only its first frame as a keyframe (no interval option), so this is also what gives a WebM a
/// seek point every second (about 2 % more bytes than one keyframe in all).
#[cfg(test)]
const CHUNK: usize = FPS;

/// Frames per chunk of `v`: one second at its frame rate.
fn chunk_of(v: &Video) -> usize {
    v.fps.max(1)
}

/// The H.264 encoder's options: CABAC, so Main profile (every browser plays it; about 3x faster
/// than oxideav-h264's CAVLC path, whose P macroblocks are slower to decide), constant QP 19
/// (with CABAC, about the size of CAVLC at QP 20 for more quality), an IDR picture every chunk.
/// The packets are AVCC samples and the parameter sets the encoder's extradata (`avcC`).
fn h264_options(gop: usize) -> oxideav_core::CodecOptions {
    oxideav_core::CodecOptions::new()
        .set("format", "avcc")
        .set("cabac", "true")
        .set("rc", "cqp")
        .set("qp", "19")
        .set("gop", gop.to_string())
}

/// H.264 (Main profile, CABAC, constant QP 19, an IDR picture every second, 4:2:0) and AAC-LC
/// (64 kb/s per channel) in MP4, or VP9 (a keyframe every second) and Opus (48 kHz) in WebM,
/// through oxideav's encoders and muxers. The frame size must be a multiple of 16 (MiniMax-H3's
/// are multiples of 32).
pub fn mp4_or_webm(v: &Video, format: VideoFormat) -> Result<Vec<u8>> {
    use oxideav_core::{
        AudioFrame, CodecId, CodecParameters, Encoder, Frame, Packet, PixelFormat, Rational, SampleFormat, StreamInfo, TimeBase,
        VideoFrame, VideoPlane,
    };
    use std::sync::Mutex;
    if v.frames.is_empty() {
        bail!("no frames to write");
    }
    let (vcodec, acodec, container) = match format {
        VideoFormat::Mp4 => ("h264", "aac", "mp4"),
        VideoFormat::Webm => ("vp9", "opus", "webm"),
        VideoFormat::Avi => return avi_mjpeg(v),
    };
    let media = crate::media::Media::new();
    let ctx = media.context();
    let fps = v.fps as i64;
    let mut vp = CodecParameters::video(CodecId::new(vcodec));
    vp.width = Some(v.width as u32);
    vp.height = Some(v.height as u32);
    vp.pixel_format = Some(PixelFormat::Yuv420P);
    vp.frame_rate = Some(Rational::new(fps, 1));
    if format == VideoFormat::Mp4 {
        vp.options = h264_options(chunk_of(v));
    }
    let vtb = TimeBase::new(1, fps);
    // One encoder per chunk, each run on a thread of the pool.
    let starts: Vec<usize> = (0..v.frames.len()).step_by(chunk_of(v)).collect();
    let mut encoders = Vec::with_capacity(starts.len());
    for _ in &starts {
        let e = ctx
            .codecs
            .first_encoder(&vp)
            .map_err(av)
            .with_context(|| format!("the {vcodec} encoder"))?;
        encoders.push(Mutex::new(e));
    }
    let chunks: Vec<Mutex<Option<Result<Vec<Packet>>>>> = starts.iter().map(|_| Mutex::new(None)).collect();
    let encode_chunk = |enc: &mut Box<dyn Encoder>, start: usize| -> Result<Vec<Packet>> {
        let mut out = Vec::new();
        let cw = v.width.div_ceil(2);
        for (i, f) in v.frames.iter().enumerate().skip(start).take(chunk_of(v)) {
            let [y, u, w2] = rgb_to_yuv420(f, v.width, v.height);
            let frame = VideoFrame {
                pts: Some(i as i64),
                planes: vec![
                    VideoPlane { stride: v.width, data: y },
                    VideoPlane { stride: cw, data: u },
                    VideoPlane { stride: cw, data: w2 },
                ],
            };
            enc.send_frame(&Frame::Video(frame)).map_err(av)?;
            while let Ok(p) = enc.receive_packet() {
                out.push(p);
            }
        }
        enc.flush().map_err(av)?;
        while let Ok(p) = enc.receive_packet() {
            out.push(p);
        }
        Ok(out)
    };
    crate::pool::run(starts.len(), &|c| {
        let mut enc = encoders[c].lock().unwrap_or_else(|p| p.into_inner());
        let r = encode_chunk(&mut enc, starts[c]);
        *chunks[c].lock().unwrap_or_else(|p| p.into_inner()) = Some(r);
    });
    let mut vpk = Vec::new();
    for c in chunks {
        vpk.extend(
            c.into_inner()
                .unwrap_or_else(|p| p.into_inner())
                .context("a chunk was not encoded")??,
        );
    }
    let encoders: Vec<Box<dyn Encoder>> = encoders
        .into_iter()
        .map(|m| m.into_inner().unwrap_or_else(|p| p.into_inner()))
        .collect();
    let vparams = encoders[0].output_params().clone();
    // The chunks share one sample description (H.264: the avcC with the parameter sets).
    if encoders.iter().any(|e| e.output_params().extradata != vparams.extradata) {
        bail!("the {vcodec} encoders of the chunks disagree on their configuration");
    }
    // Opus takes 8 / 12 / 16 / 24 / 48 kHz (oxideav-opus 0.0.14: 48 only); a 32 kHz soundtrack is
    // resampled to 48 kHz, not down to 24 (which would cut its band at 12 kHz).
    let resampled = match (format, v.audio.as_ref().filter(|a| !a.is_empty())) {
        (VideoFormat::Webm, Some(a)) if a.sample_rate != 48_000 => Some(Soundtrack {
            sample_rate: 48_000,
            channels: a
                .channels
                .iter()
                .map(|c| crate::audio::resample(c, a.sample_rate as usize, 48_000))
                .collect(),
        }),
        _ => None,
    };
    let audio = resampled.as_ref().or(v.audio.as_ref()).filter(|a| !a.is_empty());
    // Samples per channel of each frame fed: AAC's frame, Opus' 20 ms.
    let chunk = if format == VideoFormat::Webm { 960 } else { 1024 };
    let mut aenc = match audio {
        Some(a) => {
            let mut ap = CodecParameters::audio(CodecId::new(acodec));
            ap.sample_rate = Some(a.sample_rate);
            ap.channels = Some(a.channels.len() as u16);
            ap.sample_format = Some(SampleFormat::S16);
            ap.bit_rate = Some(64_000 * a.channels.len() as u64);
            Some(
                ctx.codecs
                    .first_encoder(&ap)
                    .map_err(av)
                    .with_context(|| format!("the {acodec} encoder"))?,
            )
        }
        None => None,
    };
    let atb = TimeBase::new(1, audio.map_or(1, |a| a.sample_rate as i64));
    let mut apk = Vec::new();
    if let (Some(enc), Some(a)) = (aenc.as_mut(), audio) {
        // Interleaved 16-bit chunks (the last one padded with silence to a whole frame).
        let n = a.len();
        let mut at = 0;
        while at < n {
            let end = (at + chunk).min(n);
            let mut pcm = a.interleaved(at..end);
            pcm.resize(chunk * a.channels.len(), 0.0);
            let data = crate::tts::pcm16_bytes(&pcm);
            let end = at + chunk;
            let fr = AudioFrame {
                samples: (end - at) as u32,
                pts: Some(at as i64),
                data: vec![data],
            };
            enc.send_frame(&Frame::Audio(fr)).map_err(av)?;
            while let Ok(p) = enc.receive_packet() {
                apk.push(p);
            }
            at = end;
        }
        enc.flush().map_err(av)?;
        while let Ok(p) = enc.receive_packet() {
            apk.push(p);
        }
    }
    let mut streams = vec![StreamInfo {
        index: 0,
        time_base: vtb,
        duration: Some(v.frames.len() as i64),
        start_time: Some(0),
        params: vparams,
    }];
    if let (Some(enc), Some(a)) = (&aenc, audio) {
        let mut params = enc.output_params().clone();
        // oxideav-aac 0.1.7 leaves its AudioSpecificConfig out of the parameters (its packets are
        // ADTS frames, which oxideav-mp4 0.0.11 strips): the esds gets the AAC-LC one explicitly.
        if format == VideoFormat::Mp4 && params.extradata.is_empty() {
            params.extradata = oxideav_aac::asc_writer::aac_lc_asc(a.sample_rate, a.channels.len() as u8);
        }
        streams.push(StreamInfo {
            index: 1,
            time_base: atb,
            duration: Some(a.len() as i64),
            start_time: Some(0),
            params,
        });
    }
    let buf = SharedBuf::default();
    let mut mux = match format {
        // The concrete muxer, for the Segment's Duration: the registry's `webm` muxer writes
        // none, and browsers then report an infinite duration until the end is read.
        VideoFormat::Webm => {
            let mut m = oxideav_mkv::mux::MkvMuxer::new_webm(Box::new(buf.clone()), &streams).map_err(av)?;
            let secs = audio.map_or(0.0, |a| a.len() as f64 / a.sample_rate as f64);
            let secs = secs.max(v.frames.len() as f64 / v.fps as f64);
            m.set_duration(std::time::Duration::from_secs_f64(secs)).map_err(av)?;
            Box::new(m) as Box<dyn oxideav_core::Muxer>
        }
        _ => ctx.containers.open_muxer(container, Box::new(buf.clone()), &streams).map_err(av)?,
    };
    mux.write_header().map_err(av)?;
    // Interleaved by time.
    let key = |p: &Packet| {
        let t = p.dts.or(p.pts).unwrap_or(0) as f64;
        t * p.time_base.0.num as f64 / p.time_base.0.den as f64
    };
    for p in &mut vpk {
        p.stream_index = 0;
        // oxideav-vp9 labels its packets 1/1000 s although their pts count frames.
        p.time_base = vtb;
    }
    for p in &mut apk {
        p.stream_index = 1;
        // Every AAC / Opus packet is a random-access point. oxideav-opus leaves the flag unset,
        // and oxideav-mkv then writes SimpleBlocks without their keyframe bit, which Chrome
        // refuses to play ("demuxer seek failed").
        p.flags.keyframe = true;
    }
    let mut all: Vec<Packet> = vpk.into_iter().chain(apk).collect();
    all.sort_by(|a, b| key(a).total_cmp(&key(b)));
    for p in &all {
        mux.write_packet(p).map_err(av)?;
    }
    mux.write_trailer().map_err(av)?;
    drop(mux);
    let out = buf.0.lock().unwrap_or_else(|p| p.into_inner()).get_ref().clone();
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
            let d = ((i + 1) * 1000 / v.fps - i * 1000 / v.fps) as u32;
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

    /// A moving picture with detail (gradients and a moving square) and a stereo tone.
    pub(crate) fn test_video(w: usize, h: usize, n: usize) -> Video {
        let frames = (0..n)
            .map(|f| {
                let mut rgb = vec![0u8; w * h * 3];
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) * 3;
                        let inside = (x + 64 - (f * 3) % 64) % 64 < 16 && (y / 16) % 2 == 0;
                        rgb[i] = (x * 255 / w) as u8;
                        rgb[i + 1] = (y * 255 / h) as u8;
                        rgb[i + 2] = if inside { 230 } else { ((x + y + f * 4) % 256) as u8 / 2 };
                    }
                }
                rgb
            })
            .collect();
        let len = n * 32000 / FPS;
        let tone = |hz: f32| {
            (0..len)
                .map(|i| 0.3 * (i as f32 * hz * std::f32::consts::TAU / 32000.0).sin())
                .collect()
        };
        Video {
            width: w,
            height: h,
            frames,
            fps: FPS,
            audio: Some(Soundtrack {
                sample_rate: 32000,
                channels: vec![tone(440.0), tone(660.0)],
            }),
        }
    }

    /// What [`decode_back`] found.
    pub(crate) struct Decoded {
        pub frames: Vec<Vec<u8>>,
        /// The soundtrack as [`crate::media::Media::decode_audio`] decodes it, per channel.
        pub audio: Vec<Vec<f32>>,
        pub rate: u32,
        /// The container's duration (seconds), when it states one.
        pub duration: Option<f64>,
        /// Video packets flagged as keyframes, and audio packets not flagged.
        pub video_keyframes: usize,
        pub audio_not_keyframes: usize,
    }

    /// Demuxes `bytes` (container `fmt`) and decodes its video with oxideav, and its audio the
    /// way oxinfer reads a reference clip ([`crate::media::Media::decode_audio`]).
    pub(crate) fn decode_back(bytes: Vec<u8>, fmt: &str, w: usize, h: usize) -> Decoded {
        use oxideav_core::Frame;
        let media = crate::media::Media::new();
        let ctx = media.context();
        let file = std::env::temp_dir().join(format!("oxinfer-decode-back-{}-{w}x{h}.{fmt}", std::process::id()));
        std::fs::write(&file, &bytes).unwrap();
        let mut input: Box<dyn oxideav_core::ReadSeek> = Box::new(std::io::Cursor::new(bytes));
        let format = ctx.containers.probe_input(&mut *input, Some(fmt)).unwrap();
        let mut demux = ctx.containers.open_demuxer(&format, input, &ctx.codecs).unwrap();
        let duration = demux.duration_micros().map(|us| us as f64 / 1e6);
        let streams = demux.streams().to_vec();
        let vs = streams.iter().find(|s| s.params.media_type == MediaType::Video).unwrap().clone();
        let audio_index = streams.iter().find(|s| s.params.media_type == MediaType::Audio).unwrap().index;
        let mut vdec = ctx.codecs.first_decoder(&vs.params).unwrap();
        let (mut frames, mut video_keyframes, mut audio_not_keyframes) = (Vec::new(), 0, 0);
        let take = |d: &mut Box<dyn oxideav_core::Decoder>, frames: &mut Vec<Vec<u8>>| {
            while let Ok(Frame::Video(vf)) = d.receive_frame() {
                let p = vf.image_planes();
                frames.push(yuv420_to_rgb(
                    [&p[0].data, &p[1].data, &p[2].data],
                    [p[0].stride, p[1].stride, p[2].stride],
                    w,
                    h,
                ));
            }
        };
        while let Ok(p) = demux.next_packet() {
            if p.stream_index == vs.index {
                video_keyframes += p.flags.keyframe as usize;
                vdec.send_packet(&p).unwrap();
                take(&mut vdec, &mut frames);
            } else if p.stream_index == audio_index {
                audio_not_keyframes += !p.flags.keyframe as usize;
            }
        }
        let _ = vdec.flush();
        take(&mut vdec, &mut frames);
        let (audio, rate) = media.decode_audio(&file).unwrap().to_channels().unwrap();
        std::fs::remove_file(&file).ok();
        Decoded {
            frames,
            audio,
            rate,
            duration,
            video_keyframes,
            audio_not_keyframes,
        }
    }

    /// The signal-to-noise ratio (dB) of `got` against `want` over `want`'s middle, at the
    /// codec delay (up to 4096 samples) that fits best.
    pub(crate) fn snr(want: &[f32], got: &[f32]) -> f64 {
        let (lo, hi) = (want.len() / 8, want.len() * 7 / 8);
        (0..4096.min(got.len().saturating_sub(hi)))
            .map(|d| {
                let (mut s, mut e) = (0f64, 0f64);
                for i in lo..hi {
                    let (a, b) = (want[i] as f64, got[i + d] as f64);
                    (s, e) = (s + a * a, e + (a - b) * (a - b));
                }
                10.0 * (s / e.max(1e-12)).log10()
            })
            .fold(f64::MIN, f64::max)
    }

    pub(crate) fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len() as f64;
        10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
    }

    /// MP4 (H.264 + AAC) and WebM (VP9 + Opus) decode back with oxideav: every frame (two chunks,
    /// encoded apart, each starting with a keyframe) at a reasonable PSNR, and the soundtrack's
    /// two tones; the WebM states its duration and flags its audio packets as keyframes.
    #[test]
    fn mp4_and_webm_decode_back() {
        let (w, h, n) = (128usize, 64usize, 30usize);
        let v = test_video(w, h, n);
        // The YUV conversion alone.
        let [y, u, vv] = rgb_to_yuv420(&v.frames[0], w, h);
        let back = yuv420_to_rgb([&y, &u, &vv], [w, w / 2, w / 2], w, h);
        let conv = psnr(&v.frames[0], &back);
        let secs = v.audio.as_ref().unwrap().len() as f64 / 32000.0;
        for (f, fmt) in [(VideoFormat::Mp4, "mp4"), (VideoFormat::Webm, "webm")] {
            let bytes = encode_video(&v, f).unwrap_or_else(|e| panic!("{fmt}: {e:#}"));
            let size = bytes.len();
            if let Ok(dir) = std::env::var("OXINFER_ENCODE_OUT") {
                std::fs::write(format!("{dir}/test.{fmt}"), &bytes).unwrap();
                std::fs::write(format!("{dir}/test.rgb"), v.frames.concat()).unwrap();
            }
            let d = decode_back(bytes, fmt, w, h);
            assert_eq!(d.frames.len(), n, "{fmt}: frames");
            let p: Vec<f64> = d.frames.iter().zip(&v.frames).map(|(a, b)| psnr(a, b)).collect();
            let min = p.iter().cloned().fold(f64::MAX, f64::min);
            // The two tones at the decoded rate (48 kHz for Opus).
            let tone = |hz: f32, len: usize| -> Vec<f32> {
                (0..len)
                    .map(|i| 0.3 * (i as f32 * hz * std::f32::consts::TAU / d.rate as f32).sin())
                    .collect()
            };
            let want_len = (secs * d.rate as f64) as usize;
            let s: Vec<f64> = [440.0, 660.0]
                .iter()
                .zip(&d.audio)
                .map(|(&hz, got)| snr(&tone(hz, want_len), got))
                .collect();
            eprintln!(
                "[{fmt}: {size} bytes, PSNR min {min:.1} dB (the YUV conversion alone {conv:.1} dB), {} keyframes, \
                 audio {} Hz x{}, {} samples, SNR {s:.1?} dB, duration {:?}]",
                d.video_keyframes,
                d.rate,
                d.audio.len(),
                d.audio[0].len(),
                d.duration
            );
            assert!(min > 28.0, "{fmt}: PSNR {p:?}");
            assert_eq!(d.video_keyframes, n.div_ceil(CHUNK), "{fmt}: a keyframe per chunk");
            assert_eq!(d.audio.len(), 2, "{fmt}");
            assert!(
                d.audio[0].len() as f64 / d.rate as f64 + 0.05 >= secs,
                "{fmt}: {} samples",
                d.audio[0].len()
            );
            assert!(s.iter().all(|&x| x > 15.0), "{fmt}: SNR {s:?}");
            if f == VideoFormat::Webm {
                let dur = d.duration.expect("the WebM's Duration");
                assert!((dur - secs.max(n as f64 / FPS as f64)).abs() < 0.01, "{dur}");
                assert_eq!(d.audio_not_keyframes, 0);
            }
        }
    }

    /// A video of the tiny random MiniMax-H3 release, written as MP4 and decoded back by oxideav:
    /// every frame, at the PSNR such noise allows (the random weights make noise, the hardest
    /// content there is: about 19 dB at QP 20, as for libx264), and the soundtrack's AAC stream.
    #[test]
    fn tiny_minimax_h3_to_mp4() {
        let dir = std::env::temp_dir().join(format!("oxinfer-h3-mp4-{}", std::process::id()));
        crate::minimax_h3::tiny::write_tiny_video_release(&dir).unwrap();
        let pipe = crate::minimax_h3::video::VideoPipeline::load(&dir, &dir.join("text_encoder")).unwrap();
        let mut p = crate::minimax_h3::video::VideoParams::new("A red fox.");
        (p.size, p.frames, p.steps, p.seed) = (Some((64, 64)), 22, 3, 1);
        let v = pipe.generate(&p, None, &mut |_, _| {}).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let bytes = encode_video(&v, VideoFormat::Mp4).unwrap();
        let d = decode_back(bytes, "mp4", 64, 64);
        assert_eq!(d.frames.len(), v.frames.len());
        let p: Vec<f64> = d.frames.iter().zip(&v.frames).map(|(a, b)| psnr(a, b)).collect();
        let min = p.iter().cloned().fold(f64::MAX, f64::min);
        let a = v.audio.as_ref().unwrap();
        let s: Vec<f64> = a.channels.iter().zip(&d.audio).map(|(want, got)| snr(want, got)).collect();
        eprintln!(
            "[tiny MiniMax-H3 MP4: PSNR min {min:.1} dB, audio {} Hz x{}, SNR {s:.1?} dB]",
            d.rate,
            d.audio.len()
        );
        assert!(min > 15.0, "{p:?}");
        assert_eq!((d.rate, d.audio.len()), (32000, 2));
        assert!(d.audio[0].len() >= a.len());
    }

    /// How long a 124-frame 448x256 video (5.2 s) takes to write as MP4 and as WebM:
    /// `cargo test --release --lib encode_speed -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn encode_speed() {
        let v = test_video(448, 256, 124);
        for (f, fmt) in [(VideoFormat::Mp4, "mp4"), (VideoFormat::Webm, "webm")] {
            let t = std::time::Instant::now();
            let bytes = encode_video(&v, f).unwrap();
            let s = t.elapsed().as_secs_f64();
            eprintln!(
                "[{fmt}: {s:.2} s ({:.3} s per frame, {} threads), {} bytes]",
                s / v.frames.len() as f64,
                crate::pool::threads(),
                bytes.len()
            );
        }
    }

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
            fps: FPS,
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
