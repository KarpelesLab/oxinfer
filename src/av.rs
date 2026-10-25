//! oxideav-core types as oxinfer's in-memory media (the `oxideav` feature): decoded pictures,
//! audio and video as [`VideoFrame`] / [`AudioFrame`] plus the [`CodecParameters`] that describe
//! them, converted to what the towers take (RGB8, mono f32 at the model's rate), and RGB output
//! (Stable Diffusion) back to a frame.
//!
//! Only oxideav-core is used: no codec, container or file code, so these work in a build without
//! the `media` feature, e.g. inside an oxideav pipeline filter that already has decoded frames.
//!
//! oxideav frames carry no geometry or format of their own; [`CodecParameters`] gives `width`,
//! `height` and `pixel_format` (video), `sample_rate`, `channels` and `sample_format` (audio).
//! YCbCr is converted with the matrix and range of the colour signal (the frame's
//! [`VideoFrame::color_signal`] record, else [`CodecParameters::color_signal`]): unspecified
//! means BT.601, limited range (full for the `YuvJ*` formats). The arithmetic (Q15 fixed point,
//! nearest-neighbour chroma, deep samples narrowed to their top 8 bits) is the one of
//! oxideav-pixfmt, so every format both convert gives the same RGB bytes as
//! `oxideav_image::Image::to_rgb8`, which the path-based inputs use.

use crate::video::{Clip, DEFAULT_FPS, RgbImage, Sampler};
use anyhow::{Context, Result, anyhow, bail};
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, ColorRange, ColorSignal, MatrixCoefficients, PixelFormat, Rational, SampleFormat, TimeBase,
    VideoFrame, VideoPlane,
};

/// A decoded picture: one frame and the stream parameters describing it (`width`, `height`,
/// `pixel_format`, optionally the colour signal). An image attachment, or a decision's image.
#[derive(Clone, Debug)]
pub struct Picture {
    pub frame: VideoFrame,
    pub params: CodecParameters,
}

impl Picture {
    pub fn new(frame: VideoFrame, params: CodecParameters) -> Self {
        Self { frame, params }
    }

    /// A tightly packed `Rgb24` picture (`width * height * 3` bytes).
    pub fn from_rgb(width: usize, height: usize, rgb: Vec<u8>) -> Result<Self> {
        if rgb.len() != width * height * 3 {
            bail!("{width}x{height} RGB needs {} bytes, got {}", width * height * 3, rgb.len());
        }
        let (frame, params) = rgb_frame(width, height, rgb);
        Ok(Self { frame, params })
    }

    /// The picture as tightly packed RGB8 (see [`video_frame_to_rgb`]).
    pub fn to_rgb(&self) -> Result<RgbImage> {
        video_frame_to_rgb(&self.frame, &self.params)
    }
}

impl PartialEq for Picture {
    fn eq(&self, o: &Self) -> bool {
        video_params_eq(&self.params, &o.params) && frame_eq(&self.frame, &o.frame)
    }
}

/// Decoded audio: frames in stream order and the stream parameters describing them
/// (`sample_rate`, `channels`, `sample_format`). An audio attachment, or Whisper input.
#[derive(Clone, Debug)]
pub struct AudioClip {
    pub frames: Vec<AudioFrame>,
    pub params: CodecParameters,
}

impl AudioClip {
    pub fn new(frames: Vec<AudioFrame>, params: CodecParameters) -> Self {
        Self { frames, params }
    }

    /// One frame of mono `F32` samples at `sample_rate`.
    pub fn from_mono(samples: &[f32], sample_rate: u32) -> Self {
        let mut params = CodecParameters::audio(CodecId::new("pcm_f32le"));
        params.sample_rate = Some(sample_rate);
        params.channels = Some(1);
        params.sample_format = Some(SampleFormat::F32);
        let frame = AudioFrame {
            samples: samples.len() as u32,
            pts: Some(0),
            data: vec![samples.iter().flat_map(|s| s.to_le_bytes()).collect()],
        };
        Self {
            frames: vec![frame],
            params,
        }
    }

    /// The samples averaged over the channels, as f32 in [-1, 1], resampled to `rate` (see
    /// [`audio_frames_to_mono`]).
    pub fn to_mono(&self, rate: usize) -> Result<Vec<f32>> {
        audio_frames_to_mono(&self.frames, &self.params, rate)
    }
}

impl PartialEq for AudioClip {
    fn eq(&self, o: &Self) -> bool {
        let (a, b) = (&self.params, &o.params);
        (a.sample_rate, a.channels, a.sample_format) == (b.sample_rate, b.channels, b.sample_format)
            && self.frames.len() == o.frames.len()
            && self
                .frames
                .iter()
                .zip(&o.frames)
                .all(|(x, y)| (x.samples, x.pts, &x.data) == (y.samples, y.pts, &y.data))
    }
}

/// Decoded video: frames in presentation order, the stream parameters describing them and the
/// time base of their `pts`. Frames are sampled as the model's video processor does (see
/// [`crate::video::VideoOptions`]); the prompt timestamps of the sampled frames come from their
/// `pts` when every frame has one and `time_base` is set, otherwise from their index and the
/// frame rate. The frame rate is `params.frame_rate`, else the one the `pts` imply, else
/// [`DEFAULT_FPS`].
#[derive(Clone, Debug)]
pub struct VideoClip {
    pub frames: Vec<VideoFrame>,
    pub params: CodecParameters,
    pub time_base: Option<TimeBase>,
}

impl VideoClip {
    pub fn new(frames: Vec<VideoFrame>, params: CodecParameters, time_base: Option<TimeBase>) -> Self {
        Self { frames, params, time_base }
    }

    /// RGB frames at a constant `fps` (each `width * height * 3` bytes), timed by index.
    pub fn from_rgb(width: usize, height: usize, fps: f64, frames: Vec<Vec<u8>>) -> Result<Self> {
        let mut params = rgb_frame(width, height, Vec::new()).1;
        params.frame_rate = Some(rational(fps));
        let frames = frames
            .into_iter()
            .enumerate()
            .map(|(i, rgb)| {
                if rgb.len() != width * height * 3 {
                    bail!(
                        "frame {i}: {width}x{height} RGB needs {} bytes, got {}",
                        width * height * 3,
                        rgb.len()
                    );
                }
                let mut f = rgb_frame(width, height, rgb).0;
                f.pts = None;
                Ok(f)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            frames,
            params,
            time_base: None,
        })
    }

    /// Presentation time of each frame in seconds from the first one, when every frame has a
    /// `pts` and the time base is known.
    pub fn pts_seconds(&self) -> Option<Vec<f64>> {
        let tb = self.time_base?.0;
        if tb.num <= 0 || tb.den <= 0 {
            return None;
        }
        let pts: Vec<i64> = self.frames.iter().map(|f| f.pts).collect::<Option<_>>()?;
        let first = *pts.iter().min()?;
        Some(pts.iter().map(|&p| (p - first) as f64 * tb.num as f64 / tb.den as f64).collect())
    }

    /// The frame rate: `params.frame_rate`, else what the timestamps imply, else [`DEFAULT_FPS`].
    pub fn fps(&self) -> f64 {
        if let Some(r) = self.params.frame_rate
            && r.num > 0
            && r.den > 0
        {
            return r.num as f64 / r.den as f64;
        }
        if let Some(t) = self.pts_seconds() {
            let span = t.iter().cloned().fold(0.0, f64::max);
            if t.len() > 1 && span > 0.0 {
                return (t.len() - 1) as f64 / span;
            }
        }
        DEFAULT_FPS
    }

    /// The frames `select` picks (`(frame count, fps) -> indices`), converted to RGB8.
    pub fn sample(&self, select: Sampler) -> Result<Clip> {
        if self.frames.is_empty() {
            bail!("the video has no frames");
        }
        let fps = self.fps();
        let indices = select(self.frames.len(), fps);
        let mut done: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut frames = Vec::with_capacity(indices.len());
        let mut size = None;
        for &i in &indices {
            let f = self
                .frames
                .get(i)
                .with_context(|| format!("sampled frame {i} of {}", self.frames.len()))?;
            if let Some((_, rgb)) = done.iter().find(|d| d.0 == i) {
                frames.push(rgb.clone());
                continue;
            }
            let img = video_frame_to_rgb(f, &self.params).with_context(|| format!("video frame {i}"))?;
            match size {
                None => size = Some((img.width, img.height)),
                Some(s) if s != (img.width, img.height) => bail!("video frame {i} is {}x{}, not {}x{}", img.width, img.height, s.0, s.1),
                _ => {}
            }
            done.push((i, img.data.clone()));
            frames.push(img.data);
        }
        let (width, height) = size.context("no frame sampled")?;
        let pts = self.pts_seconds().map(|t| indices.iter().map(|&i| t[i]).collect());
        Ok(Clip {
            frames,
            width,
            height,
            indices,
            fps,
            total: self.frames.len(),
            pts,
        })
    }
}

impl PartialEq for VideoClip {
    fn eq(&self, o: &Self) -> bool {
        video_params_eq(&self.params, &o.params)
            && self.params.frame_rate == o.params.frame_rate
            && self.time_base == o.time_base
            && self.frames.len() == o.frames.len()
            && self.frames.iter().zip(&o.frames).all(|(a, b)| frame_eq(a, b))
    }
}

fn video_params_eq(a: &CodecParameters, b: &CodecParameters) -> bool {
    (a.width, a.height, a.pixel_format, a.color_signal) == (b.width, b.height, b.pixel_format, b.color_signal)
}

fn frame_eq(a: &VideoFrame, b: &VideoFrame) -> bool {
    a.pts == b.pts
        && a.planes.len() == b.planes.len()
        && a.planes
            .iter()
            .zip(&b.planes)
            .all(|(x, y)| x.stride == y.stride && x.data == y.data)
}

fn rational(fps: f64) -> Rational {
    // Exact for integer rates and the NTSC ones (30000/1001 ...), close otherwise.
    let den = if fps.fract() == 0.0 { 1 } else { 1001 };
    Rational::new((fps * den as f64).round() as i64, den)
}

/// A tightly packed `Rgb24` frame (`pts` 0) and its `rawvideo` parameters (`width`, `height`,
/// `pixel_format`), e.g. to hand a generated image to an oxideav encoder. The colour signal is
/// left unspecified, as oxideav-image does for raw RGB: an sRGB signal would carry the identity
/// matrix, which an RGB to YCbCr conversion downstream would then apply.
pub fn rgb_frame(width: usize, height: usize, rgb: Vec<u8>) -> (VideoFrame, CodecParameters) {
    let mut params = CodecParameters::video(CodecId::new("rawvideo"));
    params.width = Some(width as u32);
    params.height = Some(height as u32);
    params.pixel_format = Some(PixelFormat::Rgb24);
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: width * 3,
            data: rgb,
        }],
    };
    (frame, params)
}

// ---- video frames to RGB ----------------------------------------------------------------------

/// Converts a decoded frame to tightly packed RGB8. `params` gives `width`, `height` and
/// `pixel_format`; when one is missing it is inferred from the plane layout as for decoded files
/// ([`complete_params`]). Alpha is dropped (not composited), deeper samples are narrowed to their
/// top 8 bits, palettes (`Pal8`) are expanded. YCbCr uses the colour signal's matrix (BT.601 when
/// unspecified; BT.709, FCC, BT.470BG, SMPTE 170M / 240M, BT.2020 NCL and identity are
/// implemented) and range (limited unless signalled, or a `YuvJ*` format). Formats without an
/// RGB meaning here (CMYK, float, ...) are an error naming the format.
pub fn video_frame_to_rgb(frame: &VideoFrame, params: &CodecParameters) -> Result<RgbImage> {
    let completed;
    let params = if params.width.is_none() || params.height.is_none() || params.pixel_format.is_none() {
        completed = complete_params(frame, params)?;
        &completed
    } else {
        params
    };
    let (w, h) = (params.width.unwrap_or(0) as usize, params.height.unwrap_or(0) as usize);
    let fmt = params.pixel_format.context("video parameters have no pixel format")?;
    if w == 0 || h == 0 {
        bail!("frame geometry {w}x{h} has a zero side");
    }
    let signal = frame.color_signal().unwrap_or(params.color_signal);
    let data = to_rgb(frame.image_planes(), w, h, fmt, frame, signal)?;
    Ok(RgbImage { width: w, height: h, data })
}

/// The parameters a decoder left out (`width`, `height`, `pixel_format`), inferred from the
/// frame's plane layout: some decoders do not report their output format in the stream
/// parameters.
pub fn complete_params(frame: &VideoFrame, params: &CodecParameters) -> Result<CodecParameters> {
    let mut p = params.clone();
    let planes = frame.image_planes();
    if planes.is_empty() {
        bail!("decoded frame has no planes");
    }
    let rows = |i: usize| planes[i].data.len() / planes[i].stride.max(1);
    let w = p.width.map_or(planes[0].stride, |w| w as usize);
    let h = p.height.map_or(rows(0), |h| h as usize);
    p.width = Some(w as u32);
    p.height = Some(h as u32);
    if p.pixel_format.is_none() {
        let full = p.codec_id.as_str() == "mjpeg";
        let wide = planes[0].stride >= 2 * w && planes.len() == 3;
        p.pixel_format = Some(match planes.len() {
            3 => {
                let sub_v = rows(1) < h;
                let sub_h = planes[1].stride * if wide { 1 } else { 2 } <= planes[0].stride + 1;
                match (sub_v, sub_h, wide, full) {
                    (true, _, false, false) => PixelFormat::Yuv420P,
                    (true, _, false, true) => PixelFormat::YuvJ420P,
                    (true, _, true, _) => PixelFormat::Yuv420P10Le,
                    (false, true, false, false) => PixelFormat::Yuv422P,
                    (false, true, false, true) => PixelFormat::YuvJ422P,
                    (false, true, true, _) => PixelFormat::Yuv422P10Le,
                    (false, false, false, false) => PixelFormat::Yuv444P,
                    (false, false, false, true) => PixelFormat::YuvJ444P,
                    (false, false, true, _) => PixelFormat::Yuv444P10Le,
                }
            }
            2 => PixelFormat::Nv12,
            1 if planes[0].stride >= 4 * w => PixelFormat::Rgba,
            1 if planes[0].stride >= 3 * w => PixelFormat::Rgb24,
            _ => PixelFormat::Gray8,
        });
    }
    Ok(p)
}

/// Rows of one plane: `rows` rows of at least `row_bytes` bytes, `stride` apart.
struct Plane<'a> {
    data: &'a [u8],
    stride: usize,
}

impl<'a> Plane<'a> {
    fn row(&self, r: usize) -> &'a [u8] {
        &self.data[r * self.stride..]
    }
}

fn plane<'a>(planes: &'a [VideoPlane], i: usize, row_bytes: usize, rows: usize, fmt: PixelFormat) -> Result<Plane<'a>> {
    let p = planes
        .get(i)
        .with_context(|| format!("{fmt:?} frame has {} plane(s), needs plane {}", planes.len(), i + 1))?;
    let need = if rows == 0 { 0 } else { p.stride * (rows - 1) + row_bytes };
    if p.stride < row_bytes || p.data.len() < need {
        bail!(
            "{fmt:?} plane {i}: {} bytes with stride {} is too small for {rows} rows of {row_bytes} bytes",
            p.data.len(),
            p.stride
        );
    }
    Ok(Plane {
        data: &p.data,
        stride: p.stride,
    })
}

/// Sample depth of the deep (16-bit storage) formats: `(bits, little-endian word)`.
fn deep_bits(fmt: PixelFormat) -> Option<u32> {
    use PixelFormat as P;
    Some(match fmt {
        P::Yuv420P10Le | P::Yuv422P10Le | P::Yuv444P10Le | P::Yuv440P10Le | P::Yuva422P10Le | P::Yuva444P10Le | P::Yuva420P10Le => 10,
        P::Gbrp10Le | P::Gbrap10Le | P::Gray10Le => 10,
        P::Yuv420P12Le | P::Yuv422P12Le | P::Yuv444P12Le | P::Yuv440P12Le | P::Yuva422P12Le | P::Yuva444P12Le | P::Yuva420P12Le => 12,
        P::Gbrp12Le | P::Gbrap12Le | P::Gray12Le => 12,
        P::Gbrp14Le | P::Gbrap14Le => 14,
        P::Yuv420P16Le | P::Yuv422P16Le | P::Yuv444P16Le | P::Yuv440P16Le | P::Yuva422P16Le | P::Yuva444P16Le | P::Yuva420P16Le => 16,
        P::Gbrp16Le | P::Gbrap16Le | P::Gray16Le => 16,
        _ => return None,
    })
}

/// A 16-bit little-endian sample of `bits` significant bits narrowed to its top 8 bits (what
/// oxideav-pixfmt's depth-down conversion does).
#[inline]
fn narrow(lo: u8, hi: u8, bits: u32) -> u8 {
    let v = u16::from_le_bytes([lo, hi]) as u32 & ((1u32 << bits) - 1);
    (v >> (bits - 8)) as u8
}

/// Planar YCbCr / GBR layouts: `(horizontal, vertical chroma subsampling, GBR planes)`.
fn planar_layout(fmt: PixelFormat) -> Option<(usize, usize, bool)> {
    use PixelFormat as P;
    Some(match fmt {
        P::Yuv420P | P::YuvJ420P | P::Yuva420P | P::Yuv420P10Le | P::Yuv420P12Le | P::Yuv420P16Le => (2, 2, false),
        P::Yuva420P10Le | P::Yuva420P12Le | P::Yuva420P16Le => (2, 2, false),
        P::Yuv422P | P::YuvJ422P | P::Yuva422P | P::Yuv422P10Le | P::Yuv422P12Le | P::Yuv422P16Le => (2, 1, false),
        P::Yuva422P10Le | P::Yuva422P12Le | P::Yuva422P16Le => (2, 1, false),
        P::Yuv444P | P::YuvJ444P | P::Yuva444P | P::Yuv444P10Le | P::Yuv444P12Le | P::Yuv444P16Le => (1, 1, false),
        P::Yuva444P10Le | P::Yuva444P12Le | P::Yuva444P16Le => (1, 1, false),
        P::Yuv440P | P::Yuv440P10Le | P::Yuv440P12Le | P::Yuv440P16Le => (1, 2, false),
        P::Yuv411P => (4, 1, false),
        P::Gbrp8 | P::Gbrap8 | P::Gbrp10Le | P::Gbrap10Le | P::Gbrp12Le | P::Gbrap12Le => (1, 1, true),
        P::Gbrp14Le | P::Gbrap14Le | P::Gbrp16Le | P::Gbrap16Le => (1, 1, true),
        _ => return None,
    })
}

fn to_rgb(planes: &[VideoPlane], w: usize, h: usize, fmt: PixelFormat, frame: &VideoFrame, signal: ColorSignal) -> Result<Vec<u8>> {
    use PixelFormat as P;
    let mut out = vec![0u8; w * h * 3];
    // Packed RGB with 8-bit components: bytes per pixel and the R, G, B offsets.
    let packed8 = match fmt {
        P::Rgb24 => Some((3, [0, 1, 2])),
        P::Bgr24 => Some((3, [2, 1, 0])),
        P::Rgba | P::Rgbx => Some((4, [0, 1, 2])),
        P::Bgra | P::Bgrx => Some((4, [2, 1, 0])),
        P::Argb | P::Xrgb => Some((4, [1, 2, 3])),
        P::Abgr | P::Xbgr => Some((4, [3, 2, 1])),
        _ => None,
    };
    if let Some((bpp, [ro, go, bo])) = packed8 {
        let p = plane(planes, 0, w * bpp, h, fmt)?;
        for y in 0..h {
            let row = p.row(y);
            for (x, o) in out[y * w * 3..(y + 1) * w * 3].as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let s = &row[x * bpp..];
                o.copy_from_slice(&[s[ro], s[go], s[bo]]);
            }
        }
        return Ok(out);
    }
    // Everything else maps pixel by pixel through one of these readers.
    let mut each = |px: &mut dyn FnMut(usize, usize) -> [u8; 3]| {
        for y in 0..h {
            for (x, o) in out[y * w * 3..(y + 1) * w * 3].as_chunks_mut::<3>().0.iter_mut().enumerate() {
                o.copy_from_slice(&px(x, y));
            }
        }
    };
    match fmt {
        P::Rgb48Le | P::Rgba64Le => {
            let bpp = if fmt == P::Rgb48Le { 6 } else { 8 };
            let p = plane(planes, 0, w * bpp, h, fmt)?;
            each(&mut |x, y| {
                let s = &p.row(y)[x * bpp..];
                [s[1], s[3], s[5]]
            });
        }
        P::Gray8 | P::Ya8 | P::Gray16Le | P::Gray10Le | P::Gray12Le | P::Ya16Le => {
            let (bpp, bits) = match fmt {
                P::Gray8 => (1, 8),
                P::Ya8 => (2, 8),
                P::Ya16Le => (4, 16),
                _ => (2, deep_bits(fmt).unwrap()),
            };
            let p = plane(planes, 0, w * bpp, h, fmt)?;
            each(&mut |x, y| {
                let s = &p.row(y)[x * bpp..];
                let v = if bits == 8 { s[0] } else { narrow(s[0], s[1], bits) };
                [v, v, v]
            });
        }
        P::MonoBlack | P::MonoWhite => {
            let p = plane(planes, 0, w.div_ceil(8), h, fmt)?;
            let one = if fmt == P::MonoBlack { 255 } else { 0 };
            each(&mut |x, y| {
                let bit = (p.row(y)[x / 8] >> (7 - x % 8)) & 1;
                let v = if bit == 1 { one } else { 255 - one };
                [v, v, v]
            });
        }
        P::Pal8 => {
            let p = plane(planes, 0, w, h, fmt)?;
            if frame.palette().is_none() {
                bail!("Pal8 frame carries no palette");
            }
            each(&mut |x, y| frame.palette_rgb(p.row(y)[x]).unwrap_or([0, 0, 0]));
        }
        P::Rgb565Le | P::Rgb565Be | P::Bgr565Le | P::Bgr565Be | P::Rgb555Le | P::Rgb555Be | P::Bgr555Le | P::Bgr555Be => {
            let p = plane(planes, 0, w * 2, h, fmt)?;
            let be = matches!(fmt, P::Rgb565Be | P::Bgr565Be | P::Rgb555Be | P::Bgr555Be);
            let six = matches!(fmt, P::Rgb565Le | P::Rgb565Be | P::Bgr565Le | P::Bgr565Be);
            let bgr = matches!(fmt, P::Bgr565Le | P::Bgr565Be | P::Bgr555Le | P::Bgr555Be);
            // Bit replication widens 5 / 6-bit components to the full 8-bit range.
            let w5 = |v: u16| ((v << 3) | (v >> 2)) as u8;
            let w6 = |v: u16| ((v << 2) | (v >> 4)) as u8;
            each(&mut |x, y| {
                let s = &p.row(y)[x * 2..];
                let v = if be {
                    u16::from_be_bytes([s[0], s[1]])
                } else {
                    u16::from_le_bytes([s[0], s[1]])
                };
                let (hi, mid, lo) = if six {
                    (w5(v >> 11), w6((v >> 5) & 63), w5(v & 31))
                } else {
                    (w5((v >> 10) & 31), w5((v >> 5) & 31), w5(v & 31))
                };
                if bgr { [lo, mid, hi] } else { [hi, mid, lo] }
            });
        }
        P::Nv12 | P::Nv21 => {
            let yp = plane(planes, 0, w, h, fmt)?;
            let uv = plane(planes, 1, w.div_ceil(2) * 2, h.div_ceil(2), fmt)?;
            let m = YuvToRgb::new(signal, false)?;
            let (u_at, v_at) = if fmt == P::Nv12 { (0, 1) } else { (1, 0) };
            each(&mut |x, y| {
                let c = &uv.row(y / 2)[(x / 2) * 2..];
                m.px(yp.row(y)[x], c[u_at], c[v_at])
            });
        }
        P::Yuyv422 | P::Uyvy422 => {
            let p = plane(planes, 0, w.div_ceil(2) * 4, h, fmt)?;
            let m = YuvToRgb::new(signal, false)?;
            let (yo, uo, vo) = if fmt == P::Yuyv422 { ([0, 2], 1, 3) } else { ([1, 3], 0, 2) };
            each(&mut |x, y| {
                let s = &p.row(y)[(x / 2) * 4..];
                m.px(s[yo[x % 2]], s[uo], s[vo])
            });
        }
        _ => {
            let Some((ws, hs, gbr)) = planar_layout(fmt) else {
                bail!("unsupported pixel format {fmt:?} (no conversion to RGB)");
            };
            let (bytes, bits) = match deep_bits(fmt) {
                Some(b) => (2, b),
                None => (1, 8),
            };
            let (cw, ch) = (w.div_ceil(ws), h.div_ceil(hs));
            let p0 = plane(planes, 0, w * bytes, h, fmt)?;
            let p1 = plane(planes, 1, cw * bytes, ch, fmt)?;
            let p2 = plane(planes, 2, cw * bytes, ch, fmt)?;
            let at = |p: &Plane, x: usize, y: usize| -> u8 {
                if bytes == 1 {
                    p.row(y)[x]
                } else {
                    let s = &p.row(y)[2 * x..];
                    narrow(s[0], s[1], bits)
                }
            };
            if gbr {
                each(&mut |x, y| [at(&p2, x, y), at(&p0, x, y), at(&p1, x, y)]);
            } else {
                let full = matches!(fmt, P::YuvJ420P | P::YuvJ422P | P::YuvJ444P);
                let m = YuvToRgb::new(signal, full)?;
                each(&mut |x, y| m.px(at(&p0, x, y), at(&p1, x / ws, y / hs), at(&p2, x / ws, y / hs)));
            }
        }
    }
    Ok(out)
}

/// oxideav-pixfmt's Q15 fixed point.
const FP_SHIFT: i32 = 15;
const FP_ONE: i32 = 1 << FP_SHIFT;
const FP_HALF: i32 = 1 << (FP_SHIFT - 1);

fn q15(f: f32) -> i32 {
    let v = f * FP_ONE as f32;
    if v >= 0.0 { (v + 0.5) as i32 } else { -((-v + 0.5) as i32) }
}

/// YCbCr to RGB for one colour signal, in oxideav-pixfmt's arithmetic (same matrix constants,
/// computed in f32 and rounded to Q15).
struct YuvToRgb {
    /// Identity matrix (the planes carry G, B, R).
    identity: bool,
    y_scale: i32,
    y_off: i32,
    cr_r: i32,
    cb_b: i32,
    cg_cr: i32,
    cg_cb: i32,
}

impl YuvToRgb {
    /// `full_format`: the pixel format implies full range (`YuvJ*`).
    fn new(signal: ColorSignal, full_format: bool) -> Result<Self> {
        let limited = match signal.range {
            ColorRange::Limited => true,
            ColorRange::Full => false,
            _ => !full_format,
        };
        let (kr, kb): (f32, f32) = match signal.matrix.code_point() {
            0 => (0.299, 0.114), // identity: the weights are unused
            2 => (0.299, 0.114), // unspecified: BT.601
            1 => (0.2126, 0.0722),
            4 => (0.30, 0.11),
            5 | 6 => (0.299, 0.114),
            7 => (0.212, 0.087),
            9 => (0.2627, 0.0593),
            m => bail!(
                "unsupported YCbCr matrix coefficients {m} ({})",
                MatrixCoefficients::new(m).name().unwrap_or("reserved")
            ),
        };
        let kg = 1.0 - kr - kb;
        let identity = signal.matrix == MatrixCoefficients::IDENTITY;
        Ok(if limited {
            let s = 255.0f32 / 224.0;
            Self {
                identity,
                y_scale: q15(255.0 / 219.0),
                y_off: 16,
                cr_r: q15(2.0 * (1.0 - kr) * s),
                cb_b: q15(2.0 * (1.0 - kb) * s),
                cg_cr: q15((2.0 * kr * (1.0 - kr) / kg) * s),
                cg_cb: q15((2.0 * kb * (1.0 - kb) / kg) * s),
            }
        } else {
            Self {
                identity,
                y_scale: FP_ONE,
                y_off: 0,
                cr_r: q15(2.0 * (1.0 - kr)),
                cb_b: q15(2.0 * (1.0 - kb)),
                cg_cr: q15(2.0 * kr * (1.0 - kr) / kg),
                cg_cb: q15(2.0 * kb * (1.0 - kb) / kg),
            }
        })
    }

    #[inline]
    fn px(&self, y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let c = |v: i32| v.clamp(0, 255) as u8;
        if self.identity {
            // Y = G, Cb = B, Cr = R; limited range scales all three like luma.
            let s = |v: u8| c(((v as i32 - self.y_off) * self.y_scale + FP_HALF) >> FP_SHIFT);
            return [s(cr), s(y), s(cb)];
        }
        let yv = (y as i32 - self.y_off) * self.y_scale;
        let (cbv, crv) = (cb as i32 - 128, cr as i32 - 128);
        [
            c((yv + self.cr_r * crv + FP_HALF) >> FP_SHIFT),
            c((yv - self.cg_cr * crv - self.cg_cb * cbv + FP_HALF) >> FP_SHIFT),
            c((yv + self.cb_b * cbv + FP_HALF) >> FP_SHIFT),
        ]
    }
}

// ---- audio frames to mono f32 -----------------------------------------------------------------

/// Guesses the sample format from the frame's byte layout when the stream did not declare one.
fn infer_format(af: &AudioFrame, channels: usize) -> Option<SampleFormat> {
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
fn append_mono(af: &AudioFrame, fmt: SampleFormat, channels: usize, out: &mut Vec<f32>) -> Result<()> {
    let n = af.samples as usize;
    let bps = fmt.bytes_per_sample();
    // One plane per channel is read as planar even when the declared format is interleaved.
    let planar = channels > 1 && (fmt.is_planar() || af.data.len() > 1);
    let planes = if planar { channels } else { 1 };
    let need = if planar { n * bps } else { n * channels * bps };
    if af.data.len() < planes || af.data[..planes].iter().any(|p| p.len() < need) {
        bail!(
            "{fmt:?} audio frame of {n} samples x {channels} channels needs {planes} plane(s) of {need} bytes, has {:?}",
            af.data.iter().map(Vec::len).collect::<Vec<_>>()
        );
    }
    let sample = |plane: &[u8], i: usize| -> Option<f32> {
        let b = &plane[i * bps..i * bps + bps];
        Some(match fmt {
            SampleFormat::U8 | SampleFormat::U8P => (b[0] as f32 - 128.0) / 128.0,
            SampleFormat::S8 => b[0] as i8 as f32 / 128.0,
            SampleFormat::S16 | SampleFormat::S16P => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
            SampleFormat::S24 => ((b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i8 as i32) << 16)) as f32 / 8_388_608.0,
            SampleFormat::S32 | SampleFormat::S32P => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
            SampleFormat::F32 | SampleFormat::F32P => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            SampleFormat::F64 | SampleFormat::F64P => f64::from_le_bytes(b.try_into().unwrap()) as f32,
            _ => return None,
        })
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
            acc += v.ok_or_else(|| anyhow!("unsupported sample format {fmt:?}"))?;
        }
        out.push(acc / channels as f32);
    }
    Ok(())
}

/// Decoded audio frames to mono f32 in [-1, 1] (channels averaged) at `rate` Hz, resampled with
/// [`crate::audio::resample`] when the stream's rate differs. `params` gives `sample_rate`,
/// `channels` (default 1) and `sample_format` (inferred from the frame layout when missing).
/// Every [`SampleFormat`], interleaved or planar, is supported.
pub fn audio_frames_to_mono(frames: &[AudioFrame], params: &CodecParameters, rate: usize) -> Result<Vec<f32>> {
    let src = params.sample_rate.unwrap_or(0) as usize;
    if src == 0 {
        bail!("unknown sample rate");
    }
    let channels = params.channels.unwrap_or(1).max(1) as usize;
    let mut mono = Vec::new();
    for af in frames {
        let fmt = params
            .sample_format
            .or_else(|| infer_format(af, channels))
            .context("unknown sample format")?;
        append_mono(af, fmt, channels, &mut mono)?;
    }
    Ok(if src == rate {
        mono
    } else {
        crate::audio::resample(&mono, src, rate)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use PixelFormat as P;
    use oxideav_core::{ColorPrimaries, TransferCharacteristics};

    /// A deterministic byte stream.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
    }

    fn params(w: usize, h: usize, fmt: PixelFormat) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new("rawvideo"));
        p.width = Some(w as u32);
        p.height = Some(h as u32);
        p.pixel_format = Some(fmt);
        p
    }

    /// Image planes of `fmt` at `w` x `h`: `(row bytes, rows)` each.
    fn layout(fmt: PixelFormat, w: usize, h: usize) -> Vec<(usize, usize)> {
        let packed = |bpp: usize| vec![(w * bpp, h)];
        match fmt {
            P::Rgb24 | P::Bgr24 => packed(3),
            P::Rgba | P::Bgra | P::Argb | P::Abgr | P::Rgbx | P::Bgrx | P::Xrgb | P::Xbgr => packed(4),
            P::Rgb48Le => packed(6),
            P::Rgba64Le => packed(8),
            P::Gray8 | P::Pal8 => packed(1),
            P::Gray16Le | P::Gray10Le | P::Gray12Le | P::Ya8 => packed(2),
            P::Ya16Le => packed(4),
            P::Rgb565Le | P::Rgb565Be | P::Bgr565Le | P::Bgr565Be | P::Rgb555Le | P::Rgb555Be | P::Bgr555Le | P::Bgr555Be => packed(2),
            P::MonoBlack | P::MonoWhite => vec![(w.div_ceil(8), h)],
            P::Nv12 | P::Nv21 => vec![(w, h), (w.div_ceil(2) * 2, h.div_ceil(2))],
            P::Yuyv422 | P::Uyvy422 => vec![(w.div_ceil(2) * 4, h)],
            f => {
                let (ws, hs, _) = planar_layout(f).expect("planar");
                let b = if deep_bits(f).is_some() { 2 } else { 1 };
                let c = (w.div_ceil(ws) * b, h.div_ceil(hs));
                let mut v = vec![(w * b, h), c, c];
                if f.plane_count() == 4 {
                    v.push((w * b, h));
                }
                v
            }
        }
    }

    /// A frame of random samples (valid for the format's depth), rows `pad` bytes longer than
    /// needed.
    fn random_frame(fmt: PixelFormat, w: usize, h: usize, pad: usize, seed: u64) -> VideoFrame {
        let mut r = Lcg(seed);
        let bits = deep_bits(fmt);
        let planes = layout(fmt, w, h)
            .into_iter()
            .map(|(row, rows)| {
                let stride = row + pad;
                let mut data: Vec<u8> = (0..stride * rows).map(|_| r.next()).collect();
                if let Some(b) = bits {
                    for s in data.as_chunks_mut::<2>().0 {
                        let v = (u16::from_le_bytes([s[0], s[1]]) as u32 & ((1u32 << b) - 1)) as u16;
                        s.copy_from_slice(&v.to_le_bytes());
                    }
                }
                VideoPlane { stride, data }
            })
            .collect();
        let mut f = VideoFrame { pts: Some(0), planes };
        if fmt == P::Pal8 {
            f.set_palette((0..200 * 3).map(|_| r.next()).collect());
        }
        f
    }

    fn rgb(fmt: PixelFormat, w: usize, h: usize, planes: Vec<(usize, Vec<u8>)>) -> Vec<u8> {
        let frame = VideoFrame {
            pts: None,
            planes: planes.into_iter().map(|(stride, data)| VideoPlane { stride, data }).collect(),
        };
        video_frame_to_rgb(&frame, &params(w, h, fmt)).unwrap().data
    }

    #[test]
    fn packed_rgb_formats() {
        // 2x2 pixels: (10, 20, 30), (40, 50, 60) on both rows, one padding byte per row.
        let want = [10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60];
        let cases: Vec<(PixelFormat, Vec<u8>)> = vec![
            (P::Rgb24, vec![10, 20, 30, 40, 50, 60, 99]),
            (P::Bgr24, vec![30, 20, 10, 60, 50, 40, 99]),
            (P::Rgba, vec![10, 20, 30, 1, 40, 50, 60, 2, 99]),
            (P::Rgbx, vec![10, 20, 30, 1, 40, 50, 60, 2, 99]),
            (P::Bgra, vec![30, 20, 10, 1, 60, 50, 40, 2, 99]),
            (P::Bgrx, vec![30, 20, 10, 1, 60, 50, 40, 2, 99]),
            (P::Argb, vec![1, 10, 20, 30, 2, 40, 50, 60, 99]),
            (P::Xrgb, vec![1, 10, 20, 30, 2, 40, 50, 60, 99]),
            (P::Abgr, vec![1, 30, 20, 10, 2, 60, 50, 40, 99]),
            (P::Xbgr, vec![1, 30, 20, 10, 2, 60, 50, 40, 99]),
            // 16-bit: the high byte of each sample.
            (P::Rgb48Le, vec![7, 10, 7, 20, 7, 30, 7, 40, 7, 50, 7, 60, 99]),
            (P::Rgba64Le, vec![7, 10, 7, 20, 7, 30, 0, 255, 7, 40, 7, 50, 7, 60, 0, 255, 99]),
        ];
        for (fmt, row) in cases {
            let stride = row.len();
            let data = [row.clone(), row].concat();
            assert_eq!(rgb(fmt, 2, 2, vec![(stride, data)]), want, "{fmt:?}");
        }
    }

    #[test]
    fn gray_palette_mono_and_packed_16_bit() {
        assert_eq!(rgb(P::Gray8, 2, 1, vec![(2, vec![0, 200])]), [0, 0, 0, 200, 200, 200]);
        assert_eq!(rgb(P::Ya8, 2, 1, vec![(4, vec![9, 1, 77, 2])]), [9, 9, 9, 77, 77, 77]);
        assert_eq!(rgb(P::Gray16Le, 1, 1, vec![(2, vec![0xff, 0x80])]), [0x80; 3]);
        assert_eq!(rgb(P::Ya16Le, 1, 1, vec![(4, vec![0xff, 0x40, 0, 0])]), [0x40; 3]);
        // The top 8 bits: 10-bit 1023 -> 255 and 512 -> 128, 12-bit 4095 -> 255.
        let gray10 = [1023u16.to_le_bytes(), 512u16.to_le_bytes()].concat();
        assert_eq!(rgb(P::Gray10Le, 2, 1, vec![(4, gray10)]), [255, 255, 255, 128, 128, 128]);
        assert_eq!(rgb(P::Gray12Le, 1, 1, vec![(2, 4095u16.to_le_bytes().to_vec())]), [255; 3]);
        // 1 bit per pixel, MSB first: 0b1010_0000 over 3 pixels.
        assert_eq!(
            rgb(P::MonoBlack, 3, 1, vec![(1, vec![0b1010_0000])]),
            [255, 255, 255, 0, 0, 0, 255, 255, 255]
        );
        assert_eq!(
            rgb(P::MonoWhite, 3, 1, vec![(1, vec![0b1010_0000])]),
            [0, 0, 0, 255, 255, 255, 0, 0, 0]
        );
        // Pal8: entries 0 and 1; index 5 is past the table (black).
        let mut f = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![1, 0, 5],
            }],
        };
        f.set_palette(vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(
            video_frame_to_rgb(&f, &params(3, 1, P::Pal8)).unwrap().data,
            [4, 5, 6, 1, 2, 3, 0, 0, 0]
        );
        // 5-6-5: R 31 -> 255, G 32 of 63 -> 130, B 1 -> 8; 5-5-5: G 16 of 31 -> 132.
        let v565: u16 = (31 << 11) | (32 << 5) | 1;
        let v555: u16 = (31 << 10) | (16 << 5) | 1;
        let px = |fmt, v: u16, be: bool| {
            let b = if be { v.to_be_bytes() } else { v.to_le_bytes() };
            rgb(fmt, 1, 1, vec![(2, b.to_vec())])
        };
        assert_eq!(px(P::Rgb565Le, v565, false), [255, 130, 8]);
        assert_eq!(px(P::Rgb565Be, v565, true), [255, 130, 8]);
        assert_eq!(px(P::Bgr565Le, v565, false), [8, 130, 255]);
        assert_eq!(px(P::Bgr565Be, v565, true), [8, 130, 255]);
        assert_eq!(px(P::Rgb555Le, v555, false), [255, 132, 8]);
        assert_eq!(px(P::Rgb555Be, v555, true), [255, 132, 8]);
        assert_eq!(px(P::Bgr555Le, v555, false), [8, 132, 255]);
        assert_eq!(px(P::Bgr555Be, v555, true), [8, 132, 255]);
        // Planar GBR: planes G, B, R.
        assert_eq!(rgb(P::Gbrp8, 1, 1, vec![(1, vec![2]), (1, vec![3]), (1, vec![1])]), [1, 2, 3]);
        let w16 = |v: u16| (2, v.to_le_bytes().to_vec());
        assert_eq!(rgb(P::Gbrp16Le, 1, 1, vec![w16(0x0200), w16(0x0300), w16(0x0100)]), [1, 2, 3]);
        assert_eq!(rgb(P::Gbrp10Le, 1, 1, vec![w16(8), w16(12), w16(4)]), [1, 2, 3]);
    }

    /// YCbCr to RGB in f64 (the textbook equations), limited or full range.
    fn reference(y: u8, cb: u8, cr: u8, kr: f64, kb: f64, limited: bool) -> [f64; 3] {
        let (yf, cbf, crf) = if limited {
            let c = 255.0 / 224.0;
            ((y as f64 - 16.0) * 255.0 / 219.0, (cb as f64 - 128.0) * c, (cr as f64 - 128.0) * c)
        } else {
            (y as f64, cb as f64 - 128.0, cr as f64 - 128.0)
        };
        let r = yf + 2.0 * (1.0 - kr) * crf;
        let b = yf + 2.0 * (1.0 - kb) * cbf;
        let g = (yf - kr * r - kb * b) / (1.0 - kr - kb);
        [r, g, b].map(|v| v.clamp(0.0, 255.0))
    }

    /// The `(Y, Cb, Cr)` of pixel `(x, y)` of a frame produced by `random_frame`.
    fn yuv_at(fmt: PixelFormat, pl: &[VideoPlane], x: usize, y: usize) -> (u8, u8, u8) {
        let at = |i: usize, row: usize, col: usize| pl[i].data[row * pl[i].stride + col];
        match fmt {
            P::Nv12 => (at(0, y, x), at(1, y / 2, x / 2 * 2), at(1, y / 2, x / 2 * 2 + 1)),
            P::Nv21 => (at(0, y, x), at(1, y / 2, x / 2 * 2 + 1), at(1, y / 2, x / 2 * 2)),
            P::Yuyv422 => (at(0, y, x / 2 * 4 + (x % 2) * 2), at(0, y, x / 2 * 4 + 1), at(0, y, x / 2 * 4 + 3)),
            P::Uyvy422 => (at(0, y, x / 2 * 4 + 1 + (x % 2) * 2), at(0, y, x / 2 * 4), at(0, y, x / 2 * 4 + 2)),
            f => {
                let (ws, hs, _) = planar_layout(f).unwrap();
                (at(0, y, x), at(1, y / hs, x / ws), at(2, y / hs, x / ws))
            }
        }
    }

    #[test]
    fn yuv_hand_computed() {
        // Limited-range black, white and grey.
        for (y, v) in [(16u8, 0u8), (235, 255), (126, 128)] {
            assert_eq!(
                rgb(P::Yuv444P, 1, 1, vec![(1, vec![y]), (1, vec![128]), (1, vec![128])]),
                [v; 3],
                "Y {y}"
            );
        }
        // BT.601 red, limited (Y 81, Cb 90, Cr 240) and full range (76, 85, 255); the rounded
        // samples give R = 65 * 255/219 + 112 * 1.402 * 255/224 = 254.4 and 76 + 127 * 1.402 = 254.1.
        assert_eq!(
            rgb(P::Yuv444P, 1, 1, vec![(1, vec![81]), (1, vec![90]), (1, vec![240])]),
            [254, 0, 0]
        );
        assert_eq!(
            rgb(P::YuvJ444P, 1, 1, vec![(1, vec![76]), (1, vec![85]), (1, vec![255])]),
            [254, 0, 0]
        );
        // Every 8-bit sampling against the f64 equations (within 1), for two matrices and ranges.
        let mut r = Lcg(7);
        let bt601 = MatrixCoefficients::BT601_525;
        for (matrix, kr, kb) in [(bt601, 0.299, 0.114), (MatrixCoefficients::BT709, 0.2126, 0.0722)] {
            for range in [ColorRange::Limited, ColorRange::Full] {
                let formats = [
                    P::Yuv420P,
                    P::Yuv422P,
                    P::Yuv444P,
                    P::Yuv440P,
                    P::Yuv411P,
                    P::Nv12,
                    P::Nv21,
                    P::Yuyv422,
                    P::Uyvy422,
                ];
                for fmt in formats {
                    let (w, h) = (7, 5);
                    let frame = random_frame(fmt, w, h, 3, r.next() as u64);
                    let mut p = params(w, h, fmt);
                    p.color_signal = ColorSignal::new(range, ColorPrimaries::BT709, TransferCharacteristics::BT709, matrix);
                    let out = video_frame_to_rgb(&frame, &p).unwrap().data;
                    for y in 0..h {
                        for x in 0..w {
                            let (yy, cb, cr) = yuv_at(fmt, &frame.planes, x, y);
                            let want = reference(yy, cb, cr, kr, kb, range == ColorRange::Limited);
                            let got = &out[(y * w + x) * 3..(y * w + x) * 3 + 3];
                            for c in 0..3 {
                                let d = (got[c] as f64 - want[c]).abs();
                                assert!(d <= 1.0, "{fmt:?} {range:?} {matrix} ({x},{y}) {got:?} vs {want:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn colour_signal_precedence() {
        // Y 81, Cb 90, Cr 240: red in BT.601 limited, something else under BT.709 or full range.
        let frame = |sig: Option<ColorSignal>| {
            let planes = [81, 90, 240].map(|v| VideoPlane { stride: 1, data: vec![v] });
            let mut f = VideoFrame {
                pts: None,
                planes: planes.to_vec(),
            };
            if let Some(s) = sig {
                f.set_color_signal(s);
            }
            f
        };
        let mut p = params(1, 1, P::Yuv444P);
        let bt601 = video_frame_to_rgb(&frame(None), &p).unwrap().data;
        p.color_signal = ColorSignal::bt709_limited();
        let bt709 = video_frame_to_rgb(&frame(None), &p).unwrap().data;
        assert_ne!(bt601, bt709);
        // The frame's record wins over the stream's.
        let sig = ColorSignal::unspecified().with_matrix(MatrixCoefficients::BT470_SYSTEM_BG);
        assert_eq!(video_frame_to_rgb(&frame(Some(sig)), &p).unwrap().data, bt601);
        // YuvJ implies full range unless the signal says otherwise.
        let mut pj = params(1, 1, P::YuvJ444P);
        let full = video_frame_to_rgb(&frame(None), &pj).unwrap().data;
        pj.color_signal = ColorSignal::unspecified().with_range(ColorRange::Limited);
        assert_eq!(video_frame_to_rgb(&frame(None), &pj).unwrap().data, bt601);
        assert_ne!(full, bt601);
        // Identity: the planes are G, B, R (full range: as they are).
        let mut pi = params(1, 1, P::Yuv444P);
        pi.color_signal = ColorSignal::unspecified()
            .with_matrix(MatrixCoefficients::IDENTITY)
            .with_range(ColorRange::Full);
        assert_eq!(video_frame_to_rgb(&frame(None), &pi).unwrap().data, [240, 81, 90]);
        // Matrices without an implementation are named.
        pi.color_signal = ColorSignal::unspecified().with_matrix(MatrixCoefficients::YCGCO);
        let e = video_frame_to_rgb(&frame(None), &pi).unwrap_err().to_string();
        assert!(e.contains("matrix coefficients 8"), "{e}");
    }

    #[test]
    fn errors_name_the_problem() {
        let f = random_frame(P::Rgb24, 2, 2, 0, 1);
        let e = video_frame_to_rgb(&f, &params(2, 2, P::Cmyk)).unwrap_err().to_string();
        assert!(e.contains("Cmyk"), "{e}");
        let e = video_frame_to_rgb(&f, &params(2, 2, P::GrayF32Le)).unwrap_err().to_string();
        assert!(e.contains("GrayF32Le"), "{e}");
        // A missing pixel format is inferred from the layout (one plane of 3 bytes per pixel).
        let mut p = params(2, 2, P::Rgb24);
        let want = video_frame_to_rgb(&f, &p).unwrap();
        p.pixel_format = None;
        assert_eq!(video_frame_to_rgb(&f, &p).unwrap(), want);
        let e = video_frame_to_rgb(&f, &params(3, 2, P::Rgb24)).unwrap_err().to_string();
        assert!(e.contains("too small"), "{e}");
        let e = video_frame_to_rgb(&f, &params(2, 2, P::Yuv420P)).unwrap_err().to_string();
        assert!(e.contains("plane"), "{e}");
    }

    #[test]
    fn rgb_frame_round_trip() {
        let data: Vec<u8> = (0..4 * 3 * 3).map(|i| i as u8).collect();
        let (f, p) = rgb_frame(4, 3, data.clone());
        assert_eq!(p.pixel_format, Some(P::Rgb24));
        assert_eq!((p.width, p.height), (Some(4), Some(3)));
        let img = video_frame_to_rgb(&f, &p).unwrap();
        assert_eq!((img.width, img.height, &img.data), (4, 3, &data));
        assert_eq!(Picture::from_rgb(4, 3, data).unwrap(), Picture::new(f, p));
        assert!(Picture::from_rgb(4, 3, vec![0; 5]).is_err());
    }

    fn audio(fmt: SampleFormat, channels: u16, samples: u32, data: Vec<Vec<u8>>) -> Vec<f32> {
        let mut p = CodecParameters::audio(CodecId::new("pcm"));
        p.sample_rate = Some(16000);
        p.channels = Some(channels);
        p.sample_format = Some(fmt);
        audio_frames_to_mono(&[AudioFrame { samples, pts: None, data }], &p, 16000).unwrap()
    }

    #[test]
    fn every_sample_format() {
        // Two stereo samples: (L, R) = (0.5, -0.5) and (0, 0.25); mono is the channel mean.
        let s16 = |v: i16| v.to_le_bytes().to_vec();
        let s24 = |v: i32| v.to_le_bytes()[..3].to_vec();
        let s32 = |v: i32| v.to_le_bytes().to_vec();
        let f32b = |v: f32| v.to_le_bytes().to_vec();
        let f64b = |v: f64| v.to_le_bytes().to_vec();
        let il = |s: Vec<Vec<u8>>| vec![s.concat()];
        let pl = |l: Vec<Vec<u8>>, r: Vec<Vec<u8>>| vec![l.concat(), r.concat()];
        let cases: Vec<(SampleFormat, Vec<Vec<u8>>)> = vec![
            (SampleFormat::U8, il(vec![vec![192], vec![64], vec![128], vec![160]])),
            (SampleFormat::U8P, pl(vec![vec![192], vec![128]], vec![vec![64], vec![160]])),
            (SampleFormat::S8, il(vec![vec![64], vec![(-64i8) as u8], vec![0], vec![32]])),
            (SampleFormat::S16, il(vec![s16(16384), s16(-16384), s16(0), s16(8192)])),
            (SampleFormat::S16P, pl(vec![s16(16384), s16(0)], vec![s16(-16384), s16(8192)])),
            (SampleFormat::S24, il(vec![s24(1 << 22), s24(-(1 << 22)), s24(0), s24(1 << 21)])),
            (SampleFormat::S32, il(vec![s32(1 << 30), s32(-(1 << 30)), s32(0), s32(1 << 29)])),
            (
                SampleFormat::S32P,
                pl(vec![s32(1 << 30), s32(0)], vec![s32(-(1 << 30)), s32(1 << 29)]),
            ),
            (SampleFormat::F32, il(vec![f32b(0.5), f32b(-0.5), f32b(0.0), f32b(0.25)])),
            (SampleFormat::F32P, pl(vec![f32b(0.5), f32b(0.0)], vec![f32b(-0.5), f32b(0.25)])),
            (SampleFormat::F64, il(vec![f64b(0.5), f64b(-0.5), f64b(0.0), f64b(0.25)])),
            (SampleFormat::F64P, pl(vec![f64b(0.5), f64b(0.0)], vec![f64b(-0.5), f64b(0.25)])),
        ];
        assert_eq!(cases.len(), 12, "one case per SampleFormat variant");
        for (fmt, data) in cases {
            assert_eq!(audio(fmt, 2, 2, data), [0.0, 0.125], "{fmt:?}");
        }
        // Mono; full scale; S24 sign extension.
        assert_eq!(audio(SampleFormat::S24, 1, 1, vec![s24(-(1 << 23))]), [-1.0]);
        assert_eq!(
            audio(SampleFormat::S16, 1, 2, vec![[s16(-32768), s16(32767)].concat()]),
            [-1.0, 32767.0 / 32768.0]
        );
        assert_eq!(audio(SampleFormat::U8, 1, 2, vec![vec![0, 255]]), [-1.0, 127.0 / 128.0]);
    }

    #[test]
    fn audio_inference_resampling_and_errors() {
        // No declared format: inferred from the layout (2 bytes interleaved: S16).
        let mut p = CodecParameters::audio(CodecId::new("pcm"));
        p.sample_rate = Some(16000);
        p.channels = Some(1);
        let f = AudioFrame {
            samples: 2,
            pts: None,
            data: vec![[16384i16.to_le_bytes(), (-16384i16).to_le_bytes()].concat()],
        };
        assert_eq!(audio_frames_to_mono(std::slice::from_ref(&f), &p, 16000).unwrap(), [0.5, -0.5]);
        // 48 kHz to 16 kHz keeps a third of the samples and a constant level; frames concatenate.
        let clip = AudioClip::from_mono(&[0.25; 4800], 48000);
        let out = clip.to_mono(16000).unwrap();
        assert_eq!(out.len(), 1600);
        assert!(out[100..1500].iter().all(|v| (v - 0.25).abs() < 1e-3));
        let two = AudioClip::new(vec![clip.frames[0].clone(), clip.frames[0].clone()], clip.params.clone());
        assert_eq!(two.to_mono(48000).unwrap().len(), 9600);
        // Short planes and a missing rate are errors.
        p.sample_format = Some(SampleFormat::S32);
        let e = audio_frames_to_mono(std::slice::from_ref(&f), &p, 16000).unwrap_err().to_string();
        assert!(e.contains("needs"), "{e}");
        p.sample_rate = None;
        assert!(audio_frames_to_mono(&[f], &p, 16000).is_err());
    }

    #[test]
    fn video_clip_timing() {
        // 10 frames at 25 fps on a 1/90000 time base, starting at pts 3000.
        let frames: Vec<VideoFrame> = (0..10)
            .map(|i| {
                let mut f = random_frame(P::Gray8, 4, 2, 0, i);
                f.pts = Some(3000 + i as i64 * 3600);
                f
            })
            .collect();
        let mut clip = VideoClip::new(frames, params(4, 2, P::Gray8), Some(TimeBase::new(1, 90000)));
        assert!((clip.fps() - 25.0).abs() < 1e-9);
        let pick = |_: usize, _: f64| vec![0, 4, 9];
        let c = clip.sample(&pick).unwrap();
        assert_eq!(c.times(), [0.0, 0.16, 0.36]);
        assert_eq!((c.width, c.height, c.total, c.frames.len()), (4, 2, 10, 3));
        assert_eq!(c.frames[1], video_frame_to_rgb(&clip.frames[4], &clip.params).unwrap().data);
        // A declared frame rate wins for the sampling; the timestamps still give the times.
        clip.params.frame_rate = Some(Rational::new(30, 1));
        let select = |total: usize, fps: f64| {
            assert_eq!((total, fps), (10, 30.0));
            vec![9]
        };
        assert_eq!(clip.sample(&select).unwrap().times(), [0.36]);
        // Without timestamps: index / fps.
        clip.time_base = None;
        assert_eq!(clip.sample(&pick).unwrap().times(), [0.0, 4.0 / 30.0, 9.0 / 30.0]);
        clip.params.frame_rate = None;
        assert_eq!(clip.fps(), DEFAULT_FPS);
        let rgb = VideoClip::from_rgb(1, 1, 2.0, vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        assert_eq!(rgb.sample(&|_, _| vec![1]).unwrap().frames, [vec![4, 5, 6]]);
        assert_eq!(rgb.fps(), 2.0);
    }

    /// Every format oxideav-image (oxideav-pixfmt) converts too gives the same bytes, under several
    /// colour signals, at odd sizes and with padded rows. It does not convert the newer packed
    /// formats (`Rgbx`, 5-6-5, 5-5-5) and needs even / multiple-of-4 widths for packed 4:2:2 and
    /// 4:1:1; those are covered by the hand-computed tests above.
    #[cfg(feature = "media")]
    #[test]
    fn matches_oxideav_image() {
        let formats = [
            P::Rgb24,
            P::Bgr24,
            P::Rgba,
            P::Bgra,
            P::Argb,
            P::Abgr,
            P::Rgbx,
            P::Rgb48Le,
            P::Rgba64Le,
            P::Gray8,
            P::Gray16Le,
            P::Gray10Le,
            P::Gray12Le,
            P::Ya8,
            P::Ya16Le,
            P::Pal8,
            P::MonoBlack,
            P::MonoWhite,
            P::Rgb565Le,
            P::Bgr555Be,
            P::Yuv420P,
            P::Yuv422P,
            P::Yuv444P,
            P::Yuv440P,
            P::Yuv411P,
            P::YuvJ420P,
            P::YuvJ422P,
            P::YuvJ444P,
            P::Yuva420P,
            P::Yuva422P,
            P::Yuva444P,
            P::Nv12,
            P::Nv21,
            P::Yuyv422,
            P::Uyvy422,
            P::Gbrp8,
            P::Gbrap8,
            P::Yuv420P10Le,
            P::Yuv422P10Le,
            P::Yuv444P10Le,
            P::Yuv420P12Le,
            P::Yuv444P12Le,
            P::Yuv420P16Le,
            P::Yuv444P16Le,
            P::Yuva420P10Le,
            P::Gbrp10Le,
            P::Gbrp12Le,
            P::Gbrp16Le,
        ];
        let signals = [
            ColorSignal::unspecified(),
            ColorSignal::bt709_limited(),
            ColorSignal::unspecified().with_range(ColorRange::Full),
            ColorSignal::unspecified().with_matrix(MatrixCoefficients::BT2020_NCL),
            ColorSignal::unspecified().with_matrix(MatrixCoefficients::new(4)),
            ColorSignal::unspecified().with_matrix(MatrixCoefficients::new(7)),
        ];
        let (mut compared, mut skipped) = (Vec::new(), Vec::new());
        for (k, &fmt) in formats.iter().enumerate() {
            for (si, sig) in signals.iter().enumerate() {
                for (w, h, pad) in [(16, 8, 0), (7, 5, 3), (64, 33, 1)] {
                    let frame = random_frame(fmt, w, h, pad, (k * 31 + si) as u64);
                    let mut p = params(w, h, fmt);
                    p.color_signal = *sig;
                    let ours = video_frame_to_rgb(&frame, &p).unwrap().data;
                    // oxideav-pixfmt 0.1.11 has a debug assertion on the formats it lacks.
                    let convert = || oxideav_image::Image::from_video_frame(frame, &p).and_then(|i| i.to_rgb8());
                    let theirs = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(convert)) {
                        Ok(Ok(v)) => v,
                        Ok(Err(e)) => {
                            skipped.push(format!("{fmt:?} {w}x{h} ({e})"));
                            continue;
                        }
                        Err(_) => {
                            skipped.push(format!("{fmt:?} (panicked)"));
                            continue;
                        }
                    };
                    assert!(ours == theirs, "{fmt:?} {sig} {w}x{h}");
                    compared.push(fmt);
                }
            }
        }
        compared.dedup();
        skipped.dedup();
        eprintln!("same bytes: {compared:?}\nnot converted by oxideav-image: {skipped:?}");
        assert_eq!(compared.len(), formats.len() - 3, "all but Rgbx, 5-6-5 and 5-5-5");
    }
}
