//! Media in and out: image, audio and video references of requests (`data:` URLs, fetched URLs,
//! uploads), multipart bodies, and the image and audio endpoints.

use super::registry::{ImageModel, Kind, Loaded};
use super::{Api, ApiError, ApiRequest, ApiResponse, ApiResult, Output, Sink, now};
use crate::engine::Flow;
use crate::video::RgbImage;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ---- media references -------------------------------------------------------------------------

/// Files written for one request, removed when it is done.
#[derive(Default)]
pub(super) struct TempFiles(Vec<PathBuf>);

impl Drop for TempFiles {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

impl TempFiles {
    /// Writes `bytes` to a new temporary file with extension `ext`.
    pub fn write(&mut self, bytes: &[u8], ext: &str) -> ApiResult<PathBuf> {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join("oxinfer-server");
        std::fs::create_dir_all(&dir).map_err(|e| ApiError::new(500, format!("create {}: {e}", dir.display())))?;
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = dir.join(format!("{}-{n}.{ext}", std::process::id()));
        std::fs::write(&p, bytes).map_err(|e| ApiError::new(500, format!("write {}: {e}", p.display())))?;
        self.0.push(p.clone());
        Ok(p)
    }
}

/// The file extension for the content of `bytes` (magic numbers), else from a MIME type or a
/// file name.
pub(super) fn extension(bytes: &[u8], mime: Option<&str>, name: Option<&str>) -> String {
    let b = bytes;
    let sniffed = if b.starts_with(b"\x89PNG") {
        Some("png")
    } else if b.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("jpg")
    } else if b.starts_with(b"GIF8") {
        Some("gif")
    } else if b.starts_with(b"BM") {
        Some("bmp")
    } else if b.len() >= 12 && &b[..4] == b"RIFF" {
        match &b[8..12] {
            b"WAVE" => Some("wav"),
            b"WEBP" => Some("webp"),
            b"AVI " => Some("avi"),
            _ => None,
        }
    } else if b.starts_with(b"fLaC") {
        Some("flac")
    } else if b.starts_with(b"OggS") {
        Some("ogg")
    } else if b.starts_with(b"ID3") || (b.len() > 1 && b[0] == 0xff && b[1] & 0xe0 == 0xe0) {
        Some("mp3")
    } else if b.len() >= 8 && &b[4..8] == b"ftyp" {
        Some(if &b[8..b.len().min(12)] == b"M4A " { "m4a" } else { "mp4" })
    } else if b.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some("mkv")
    } else {
        None
    };
    if let Some(s) = sniffed {
        return s.into();
    }
    let from_mime = mime.and_then(|m| {
        Some(match m.split(';').next()?.trim() {
            "image/png" => "png",
            "image/jpeg" | "image/jpg" => "jpg",
            "image/webp" => "webp",
            "image/gif" => "gif",
            "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
            "audio/mpeg" | "audio/mp3" => "mp3",
            "audio/flac" => "flac",
            "audio/ogg" | "audio/opus" => "ogg",
            "audio/aac" => "aac",
            "audio/mp4" | "audio/m4a" | "audio/x-m4a" => "m4a",
            "video/mp4" => "mp4",
            "video/webm" => "webm",
            "video/quicktime" => "mov",
            "video/x-matroska" => "mkv",
            "video/x-msvideo" => "avi",
            _ => return None,
        })
    });
    if let Some(s) = from_mime {
        return s.into();
    }
    name.and_then(|n| Path::new(n.split(['?', '#']).next().unwrap_or(n)).extension())
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .filter(|e| e.len() <= 5 && e.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or_else(|| "bin".into())
}

/// The bytes of a media reference: a `data:` URL, an `http(s)://` URL (when
/// [`super::ServerOptions::remote_media`]) or a local path / `file://` URL (when
/// [`super::ServerOptions::local_media`]; returned as a path). Also the MIME type when known.
pub(super) enum Fetched {
    Bytes(Vec<u8>, Option<String>),
    Path(PathBuf),
}

pub(super) fn fetch(api: &Api, url: &str) -> ApiResult<Fetched> {
    if let Some(d) = url.strip_prefix("data:") {
        let (meta, b64) = d
            .split_once(',')
            .ok_or_else(|| ApiError::bad("malformed data: URL (expected data:MIME;base64,DATA)"))?;
        let Some(mime) = meta.strip_suffix(";base64") else {
            return Err(ApiError::bad("only base64 data: URLs are supported"));
        };
        let bytes = crate::b64::decode(b64).ok_or_else(|| ApiError::bad("bad base64 in a data: URL"))?;
        return Ok(Fetched::Bytes(bytes, (!mime.is_empty()).then(|| mime.to_string())));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        if !api.opts.remote_media {
            return Err(ApiError::bad(
                "this server does not fetch media URLs; send the media as a data: URL",
            ));
        }
        #[cfg(feature = "hf")]
        {
            let bytes = crate::hf::http_get(url).map_err(|e| ApiError::bad(format!("fetch {url}: {e:#}")))?;
            return Ok(Fetched::Bytes(bytes, None));
        }
        #[cfg(not(feature = "hf"))]
        return Err(ApiError::bad(
            "built without the `hf` feature: media URLs cannot be fetched; use a data: URL",
        ));
    }
    if api.opts.local_media {
        let p = PathBuf::from(url.strip_prefix("file://").unwrap_or(url));
        if p.is_file() {
            return Ok(Fetched::Path(p));
        }
        return Err(ApiError::bad(format!("no such file: {}", p.display())));
    }
    Err(ApiError::bad(format!(
        "unsupported media reference {:?}: expected a data: or http(s):// URL",
        url.chars().take(64).collect::<String>()
    )))
}

/// A media reference as a local file (written to `temp` unless it is one already).
pub(super) fn to_file(api: &Api, url: &str, temp: &mut TempFiles) -> ApiResult<PathBuf> {
    match fetch(api, url)? {
        Fetched::Path(p) => Ok(p),
        Fetched::Bytes(b, mime) => {
            let ext = extension(&b, mime.as_deref(), Some(url).filter(|u| !u.starts_with("data:")));
            temp.write(&b, &ext)
        }
    }
}

/// Decodes encoded images (PNG, JPEG, WebP, GIF, BMP; the `media` feature).
fn decode_images(encoded: Vec<Vec<u8>>) -> ApiResult<Vec<RgbImage>> {
    crate::mm::decode_images(&encoded).map_err(|e| ApiError::bad(format!("decode image: {e:#}")))
}

// ---- multipart/form-data ----------------------------------------------------------------------

/// One field of a `multipart/form-data` body.
#[derive(Debug, PartialEq)]
pub(super) struct Part {
    pub name: String,
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub data: Vec<u8>,
}

impl Part {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.data).trim().to_string()
    }
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|i| i + from)
}

/// Parses a `multipart/form-data` body.
pub(super) fn multipart(content_type: Option<&str>, body: &[u8]) -> ApiResult<Vec<Part>> {
    let ct = content_type.unwrap_or_default();
    if !ct.starts_with("multipart/form-data") {
        return Err(ApiError::new(415, "this endpoint takes a multipart/form-data body"));
    }
    let boundary = ct
        .split(';')
        .filter_map(|p| p.trim().strip_prefix("boundary="))
        .next()
        .map(|b| b.trim_matches('"'))
        .filter(|b| !b.is_empty())
        .ok_or_else(|| ApiError::bad("multipart body without a boundary"))?;
    let delim = format!("--{boundary}").into_bytes();
    let next_delim = [b"\r\n".as_slice(), &delim].concat();
    let bad = || ApiError::bad("malformed multipart body");
    let mut pos = find(body, &delim, 0).ok_or_else(bad)?;
    let mut parts = Vec::new();
    loop {
        let after = pos + delim.len();
        if body[after..].starts_with(b"--") {
            break;
        }
        let start = after + if body[after..].starts_with(b"\r\n") { 2 } else { 0 };
        let end = find(body, &next_delim, start).ok_or_else(bad)?;
        let part = &body[start..end];
        let h_end = find(part, b"\r\n\r\n", 0).ok_or_else(bad)?;
        let headers = String::from_utf8_lossy(&part[..h_end]);
        let (mut name, mut filename, mut content_type) = (None, None, None);
        for line in headers.split("\r\n") {
            let Some((k, v)) = line.split_once(':') else { continue };
            match k.trim().to_ascii_lowercase().as_str() {
                "content-disposition" => {
                    for p in v.split(';').map(str::trim) {
                        if let Some(n) = p.strip_prefix("name=") {
                            name = Some(n.trim_matches('"').to_string());
                        } else if let Some(f) = p.strip_prefix("filename=") {
                            filename = Some(f.trim_matches('"').to_string());
                        }
                    }
                }
                "content-type" => content_type = Some(v.trim().to_string()),
                _ => {}
            }
        }
        parts.push(Part {
            name: name.ok_or_else(bad)?,
            filename,
            content_type,
            data: part[h_end + 4..].to_vec(),
        });
        pos = end + 2;
    }
    Ok(parts)
}

/// The form fields (all parts without a file name, and every part as text) by name.
fn field<'a>(parts: &'a [Part], name: &str) -> Option<&'a Part> {
    parts.iter().find(|p| p.name == name)
}

// ---- images -----------------------------------------------------------------------------------

/// An image generation or edit request.
#[derive(Default)]
struct ImageJob {
    model: String,
    prompt: String,
    negative: Option<String>,
    n: usize,
    size: Option<(usize, usize)>,
    seed: Option<u64>,
    steps: Option<usize>,
    guidance: Option<f32>,
    strength: Option<f32>,
    as_url: bool,
    images: Vec<RgbImage>,
    mask: Option<RgbImage>,
}

impl ImageJob {
    /// From the fields of a JSON body or a form (`get` returns a field as JSON).
    fn parse(get: &dyn Fn(&str) -> Option<Value>) -> ApiResult<Self> {
        let s = |k: &str| get(k).and_then(|v| v.as_str().map(str::to_string));
        let num = |k: &'static str| -> ApiResult<Option<f64>> {
            match get(k) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Number(n)) => Ok(n.as_f64()),
                Some(Value::String(t)) => t
                    .trim()
                    .parse()
                    .map(Some)
                    .map_err(|_| ApiError::bad(format!("{k} must be a number")).param(k)),
                Some(_) => Err(ApiError::bad(format!("{k} must be a number")).param(k)),
            }
        };
        let model = s("model").ok_or_else(|| ApiError::bad("model is required").param("model"))?;
        let prompt = s("prompt").ok_or_else(|| ApiError::bad("prompt is required").param("prompt"))?;
        let n = num("n")?.unwrap_or(1.0) as usize;
        if !(1..=10).contains(&n) {
            return Err(ApiError::bad("n must be between 1 and 10").param("n"));
        }
        let size = match s("size").as_deref() {
            None | Some("auto") | Some("") => None,
            Some(sz) => {
                let (w, h) = sz
                    .split_once(['x', 'X'])
                    .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
                    .ok_or_else(|| ApiError::bad(format!("size {sz:?}: expected WIDTHxHEIGHT")).param("size"))?;
                if !(64..=4096).contains(&w) || !(64..=4096).contains(&h) {
                    return Err(ApiError::bad("size: each side must be between 64 and 4096").param("size"));
                }
                Some((w, h))
            }
        };
        let as_url = match s("response_format").as_deref() {
            None | Some("b64_json") => false,
            Some("url") => true,
            Some(f) => return Err(ApiError::bad(format!("response_format {f:?}: expected b64_json or url")).param("response_format")),
        };
        if let Some(f) = s("output_format")
            && f != "png"
        {
            return Err(ApiError::bad("output_format: only png is supported").param("output_format"));
        }
        Ok(Self {
            model,
            prompt,
            negative: s("negative_prompt").or_else(|| s("negative")),
            n,
            size,
            seed: num("seed")?.map(|v| v as u64),
            steps: num("steps")?.or(num("num_inference_steps")?).map(|v| v as usize),
            guidance: num("guidance")?.or(num("guidance_scale")?).map(|v| v as f32),
            strength: num("strength")?.map(|v| v as f32),
            as_url,
            ..Self::default()
        })
    }
}

/// `POST /v1/images/generations`.
pub(super) fn image_generations(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    let job = ImageJob::parse(&|k| body.get(k).cloned())?;
    run_images(api, job)
}

/// `POST /v1/images/edits` (multipart: `image` or `image[]` files, an optional `mask`).
pub(super) fn image_edits(api: &Arc<Api>, req: &ApiRequest) -> ApiResult<Output> {
    let parts = multipart(req.content_type.as_deref(), &req.body)?;
    let get = |k: &str| field(&parts, k).map(|p| Value::String(p.text()));
    let mut job = ImageJob::parse(&get)?;
    let images: Vec<Vec<u8>> = parts
        .iter()
        .filter(|p| p.name == "image" || p.name == "image[]" || p.name == "images[]")
        .map(|p| p.data.clone())
        .collect();
    if images.is_empty() {
        return Err(ApiError::bad("image is required").param("image"));
    }
    job.images = decode_images(images)?;
    if let Some(m) = field(&parts, "mask") {
        job.mask = decode_images(vec![m.data.clone()])?.pop();
    }
    run_images(api, job)
}

fn run_images(api: &Arc<Api>, job: ImageJob) -> ApiResult<Output> {
    api.models.check(&job.model, Kind::Image)?;
    let mut lease = api.models.acquire(&job.model, Kind::Image)?;
    let Loaded::Image(m) = lease.model() else { unreachable!() };
    let seed = job.seed.unwrap_or_else(random_seed);
    let imgs = generate_images(m, &job, seed).map_err(|e| ApiError::new(500, format!("{e:#}")))?;
    drop(lease);
    let data: Vec<Value> = imgs
        .iter()
        .map(|(w, h, rgb)| {
            let b64 = crate::b64::encode(&crate::png::encode_rgb(*w, *h, rgb));
            if job.as_url {
                json!({ "url": format!("data:image/png;base64,{b64}") })
            } else {
                json!({ "b64_json": b64 })
            }
        })
        .collect();
    Ok(Output::Full(ApiResponse::json(
        200,
        &json!({ "created": now(), "data": data, "output_format": "png" }),
    )))
}

pub(super) fn random_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    (t.as_nanos() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 16
}

/// Runs the pipeline `n` times (seeds `seed`, `seed + 1`, ...; Janus: one batch).
fn generate_images(m: &mut ImageModel, job: &ImageJob, seed: u64) -> anyhow::Result<Vec<(usize, usize, Vec<u8>)>> {
    use anyhow::bail;
    let (w, h) = (job.size.map(|s| s.0), job.size.map(|s| s.1));
    let first = job.images.first();
    if job.mask.is_some() && !matches!(m, ImageModel::Sd(_)) {
        bail!("masks are supported by Stable Diffusion models only");
    }
    let mut out = Vec::new();
    if let ImageModel::Janus(j) = m {
        if first.is_some() {
            bail!("Janus-Pro generates images from text only (no edits)");
        }
        let mut o = crate::janus::ImageOptions {
            seed,
            parallel_size: job.n,
            ..Default::default()
        };
        if let Some(g) = job.guidance {
            o.cfg_weight = g;
        }
        let (imgs, _) = j.generate_images(&job.prompt, &o, &mut |_| Flow::Continue)?;
        return Ok(imgs.into_iter().map(|i| (i.width, i.height, i.rgb)).collect());
    }
    for i in 0..job.n {
        let seed = seed.wrapping_add(i as u64);
        let img = match m {
            ImageModel::Sd(pipe) => {
                let edit = first.map(|im| {
                    let mut e = crate::sd::EditInput::new(im.clone());
                    if let Some(mk) = &job.mask {
                        e = e.with_mask(mk.clone());
                    }
                    e.strength = job.strength;
                    e
                });
                let kind = match &edit {
                    Some(e) => Some(pipe.edit_kind(e.mask.is_some())?),
                    None => None,
                };
                let strength = job.strength.unwrap_or(match kind {
                    Some(crate::sd::EditKind::Img2Img) => crate::sd::DEFAULT_STRENGTH,
                    _ => 1.0,
                });
                let side = pipe.default_size();
                let p = crate::sd::Params {
                    prompt: job.prompt.clone(),
                    negative: job.negative.clone().unwrap_or_default(),
                    steps: job.steps.unwrap_or(crate::sd::default_steps(pipe.turbo, kind, strength)),
                    guidance: job.guidance.unwrap_or(if pipe.turbo { 0.0 } else { 7.5 }),
                    seed,
                    // 0 x 0: an edit takes the image's size.
                    width: w.unwrap_or(if edit.is_some() { 0 } else { side }),
                    height: h.unwrap_or(if edit.is_some() { 0 } else { side }),
                    sampler: crate::sd::sched::Sampler::EulerAncestral,
                };
                let (img, _) = match &edit {
                    Some(e) => pipe.generate_from(&p, e, &mut |_, _| {})?,
                    None => pipe.generate(&p, &mut |_, _| {})?,
                };
                (img.width, img.height, img.rgb)
            }
            ImageModel::Flux(pipe) => {
                let o = crate::flux::Options {
                    steps: job.steps,
                    guidance: job.guidance,
                    width: w.or(first.map(|i| i.width)),
                    height: h.or(first.map(|i| i.height)),
                    seed,
                    negative_prompt: job.negative.clone(),
                    ..Default::default()
                };
                let (img, _) = match first {
                    Some(im) => pipe.edit(im, &job.prompt, &o, &mut |_, _| true)?,
                    None => pipe.generate(&job.prompt, &o, &mut |_, _| true)?,
                };
                (img.width, img.height, img.rgb)
            }
            ImageModel::QwenImage(pipe) => {
                let r = crate::qwen_image::Request {
                    prompt: job.prompt.clone(),
                    negative: job.negative.clone(),
                    steps: job.steps,
                    true_cfg: job.guidance,
                    seed: Some(seed),
                    width: w,
                    height: h,
                    ..Default::default()
                };
                let p = pipe.params(&r);
                let (img, _) = if job.images.is_empty() {
                    pipe.generate(&p, &mut |_, _| {})?
                } else {
                    pipe.edit(&job.images, &p, &mut |_, _| {})?
                };
                (img.width, img.height, img.rgb)
            }
            ImageModel::OmniGen2(pipe) => {
                let r = crate::omnigen2::Request {
                    prompt: job.prompt.clone(),
                    negative: job.negative.clone(),
                    steps: job.steps,
                    text_guidance: job.guidance,
                    seed: Some(seed),
                    width: w,
                    height: h,
                    ..Default::default()
                };
                let p = r.params();
                let (img, _) = if job.images.is_empty() {
                    pipe.generate(&p, &mut |_, _| {})?
                } else {
                    pipe.edit(&job.images, &p, &mut |_, _| {})?
                };
                (img.width, img.height, img.rgb)
            }
            ImageModel::Step1x(pipe) => {
                let r = crate::step1x::Request {
                    prompt: job.prompt.clone(),
                    negative: job.negative.clone(),
                    steps: job.steps,
                    true_cfg: job.guidance,
                    seed: Some(seed),
                    width: w,
                    height: h,
                    ..Default::default()
                };
                let p = r.params();
                let (img, _) = match first {
                    Some(im) => pipe.edit(im, &p, &mut |_, _| {})?,
                    None => pipe.generate(&p, &mut |_, _| {})?,
                };
                (img.width, img.height, img.rgb)
            }
            ImageModel::Bagel(b) => {
                let mut o = crate::bagel::ImageOptions {
                    width: w,
                    height: h,
                    seed,
                    cfg_text_scale: job.guidance,
                    ..Default::default()
                };
                if let Some(s) = job.steps {
                    o.steps = s;
                }
                let (img, _) = match first {
                    Some(im) => b.edit(im, &job.prompt, &o, &mut |_, _| true)?,
                    None => b.generate(&job.prompt, &o, &mut |_, _| true)?,
                };
                (img.width, img.height, img.rgb)
            }
            ImageModel::Janus(_) => unreachable!(),
        };
        out.push(img);
    }
    Ok(out)
}

// ---- speech to text ---------------------------------------------------------------------------

/// `POST /v1/audio/transcriptions` and `/v1/audio/translations` (multipart: `file`, `model`,
/// `language`, `prompt`, `response_format`, `temperature`, `timestamp_granularities[]`).
/// Beyond the OpenAI API: `beam_size`, `patience`, `length_penalty` and `best_of` (beam search
/// and best-of-n sampling, as openai-whisper), and several `file` parts, transcribed together
/// as a batch and answered with a JSON array of the per-file results, in order (`json` and
/// `verbose_json` only).
pub(super) fn transcriptions(api: &Arc<Api>, req: &ApiRequest, translate: bool) -> ApiResult<Output> {
    let parts = multipart(req.content_type.as_deref(), &req.body)?;
    let text = |k: &str| field(&parts, k).map(Part::text).filter(|s| !s.is_empty());
    let model = text("model").ok_or_else(|| ApiError::bad("model is required").param("model"))?;
    let files: Vec<&Part> = parts.iter().filter(|p| p.name == "file").collect();
    if files.is_empty() {
        return Err(ApiError::bad("file is required").param("file"));
    }
    let format = text("response_format").unwrap_or_else(|| "json".into());
    if !["json", "text", "verbose_json", "srt", "vtt"].contains(&format.as_str()) {
        return Err(
            ApiError::bad(format!("response_format {format:?}: expected json, text, verbose_json, srt or vtt")).param("response_format"),
        );
    }
    let words = parts
        .iter()
        .filter(|p| p.name.starts_with("timestamp_granularities"))
        .any(|p| p.text() == "word");
    if files.len() > 1 && !["json", "verbose_json"].contains(&format.as_str()) {
        return Err(
            ApiError::bad("several files are answered as a JSON array: response_format must be json or verbose_json")
                .param("response_format"),
        );
    }
    let number = |k: &'static str| -> ApiResult<Option<f32>> {
        text(k)
            .map(|v| {
                v.parse::<f32>()
                    .map_err(|_| ApiError::bad(format!("{k} must be a number")).param(k))
            })
            .transpose()
    };
    let count = |k: &'static str| -> ApiResult<Option<usize>> {
        text(k)
            .map(|v| {
                v.parse::<usize>()
                    .map_err(|_| ApiError::bad(format!("{k} must be a positive integer")).param(k))
            })
            .transpose()
    };
    let mut o = crate::whisper::TranscribeOptions {
        beam_size: count("beam_size")?,
        patience: number("patience")?,
        length_penalty: number("length_penalty")?,
        best_of: count("best_of")?,
        language: text("language"),
        initial_prompt: text("prompt"),
        task: translate.then_some(crate::whisper::Task::Translate),
        timestamps: format != "json" && format != "text",
        word_timestamps: words,
        ..Default::default()
    };
    if let Some(t) = text("temperature") {
        let t: f32 = t
            .parse()
            .map_err(|_| ApiError::bad("temperature must be a number").param("temperature"))?;
        if t > 0.0 {
            o.temperatures = vec![t];
        }
    }
    api.models.check(&model, Kind::Transcription)?;
    let pcms: Vec<Vec<f32>> = files.iter().map(|f| decode_audio(f)).collect::<ApiResult<_>>()?;
    let refs: Vec<&[f32]> = pcms.iter().map(|p| &p[..]).collect();
    let mut lease = api.models.acquire(&model, Kind::Transcription)?;
    let Loaded::Transcription(t) = lease.model() else { unreachable!() };
    // Bad search options (patience without beam_size, ...) are the request's fault.
    let mut trs = t.transcribe_batch(&refs, &o).map_err(|e| {
        let bad = ["beam_size", "patience", "best_of", "length_penalty"];
        let msg = format!("{e:#}");
        ApiError::new(if bad.iter().any(|k| msg.starts_with(k)) { 400 } else { 500 }, msg)
    })?;
    drop(lease);
    let text_body = |s: String, ct: &str| ApiResponse {
        status: 200,
        content_type: ct.into(),
        body: s.into_bytes(),
    };
    let object = |tr: &crate::whisper::Transcript| -> Value {
        if format != "verbose_json" {
            return json!({ "text": tr.text.trim(), "usage": { "type": "duration", "seconds": tr.stats.audio_secs.ceil() } });
        }
        let segments: Vec<Value> = tr
            .segments
            .iter()
            .map(|s| {
                json!({
                    "id": s.id, "seek": 0, "start": s.start, "end": s.end, "text": s.text, "tokens": s.tokens,
                    "temperature": s.temperature, "avg_logprob": s.avg_logprob,
                    "compression_ratio": s.compression_ratio, "no_speech_prob": s.no_speech_prob,
                })
            })
            .collect();
        let mut v = json!({
            "task": if translate { "translate" } else { "transcribe" },
            "language": tr.language,
            "duration": tr.stats.audio_secs,
            "text": tr.text.trim(),
            "segments": segments,
            "usage": { "type": "duration", "seconds": tr.stats.audio_secs.ceil() },
        });
        if words {
            let w: Vec<Value> = tr
                .segments
                .iter()
                .flat_map(|s| s.words.iter())
                .map(|w| json!({ "word": w.word.trim(), "start": w.start, "end": w.end }))
                .collect();
            v["words"] = Value::Array(w);
        }
        v
    };
    if trs.len() > 1 {
        return Ok(Output::Full(ApiResponse::json(
            200,
            &Value::Array(trs.iter().map(object).collect()),
        )));
    }
    let tr = trs.pop().unwrap();
    Ok(Output::Full(match format.as_str() {
        "text" => text_body(format!("{}\n", tr.text.trim()), "text/plain; charset=utf-8"),
        "srt" | "vtt" => text_body(subtitles(&tr.segments, format == "vtt"), "text/plain; charset=utf-8"),
        _ => ApiResponse::json(200, &object(&tr)),
    }))
}

/// An uploaded audio file as 16 kHz mono samples.
fn decode_audio(file: &Part) -> ApiResult<Vec<f32>> {
    let ext = extension(&file.data, file.content_type.as_deref(), file.filename.as_deref());
    #[cfg(feature = "media")]
    {
        let mut temp = TempFiles::default();
        let p = temp.write(&file.data, &ext)?;
        crate::whisper::Transcriber::load_audio(&p).map_err(|e| ApiError::bad(format!("decode audio: {e:#}")))
    }
    #[cfg(not(feature = "media"))]
    {
        if ext != "wav" {
            return Err(ApiError::bad("built without the `media` feature: only WAV audio can be decoded"));
        }
        let a = crate::tts::read_wav(&file.data).map_err(|e| ApiError::bad(format!("decode audio: {e:#}")))?;
        Ok(crate::audio::resample(
            &a.samples,
            a.sample_rate as usize,
            crate::whisper::mel::SAMPLE_RATE,
        ))
    }
}

fn subtitles(segments: &[crate::whisper::Segment], vtt: bool) -> String {
    let ts = |t: f64| {
        let ms = (t * 1000.0).round() as u64;
        let sep = if vtt { '.' } else { ',' };
        format!(
            "{:02}:{:02}:{:02}{sep}{:03}",
            ms / 3_600_000,
            ms / 60_000 % 60,
            ms / 1000 % 60,
            ms % 1000
        )
    };
    let mut s = String::new();
    if vtt {
        s.push_str("WEBVTT\n\n");
    }
    for (i, seg) in segments.iter().enumerate() {
        if !vtt {
            s.push_str(&format!("{}\n", i + 1));
        }
        s.push_str(&format!("{} --> {}\n{}\n\n", ts(seg.start), ts(seg.end), seg.text.trim()));
    }
    s
}

// ---- text to speech ---------------------------------------------------------------------------

/// OpenAI's voice names, which map to the model's default voice.
const OPENAI_VOICES: [&str; 13] = [
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse",
];

/// `POST /v1/audio/speech`: `wav` (default) or `pcm` (16-bit little-endian mono, streamed as it
/// is generated); `stream_format: "sse"` streams base64 PCM in `speech.audio.delta` events.
pub(super) fn speech(api: &Arc<Api>, body: &Value) -> ApiResult<Output> {
    let s = |k: &str| body.get(k).and_then(Value::as_str).map(str::to_string);
    let model = s("model").ok_or_else(|| ApiError::bad("model is required").param("model"))?;
    let input = s("input")
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| ApiError::bad("input is required").param("input"))?;
    let format = s("response_format").unwrap_or_else(|| "wav".into());
    if format != "wav" && format != "pcm" {
        return Err(ApiError::bad(format!("response_format {format:?}: this server writes wav or pcm")).param("response_format"));
    }
    if let Some(sp) = body.get("speed").and_then(Value::as_f64)
        && (sp - 1.0).abs() > 1e-6
    {
        return Err(ApiError::bad("speed: only 1.0 is supported").param("speed"));
    }
    let sse = match s("stream_format").as_deref() {
        None | Some("audio") => false,
        Some("sse") => true,
        Some(f) => return Err(ApiError::bad(format!("stream_format {f:?}: expected audio or sse")).param("stream_format")),
    };
    let mut o = crate::tts::SpeakOptions::default();
    let num = |k: &str| body.get(k).and_then(Value::as_f64);
    if let Some(v) = num("temperature") {
        o.temperature = v as f32;
    }
    if let Some(v) = num("top_p") {
        o.top_p = v as f32;
    }
    if let Some(v) = num("repetition_penalty") {
        o.repetition_penalty = v as f32;
    }
    if let Some(v) = num("max_tokens") {
        o.max_tokens = v as usize;
    }
    o.seed = num("seed").map(|v| v as u64).unwrap_or_else(random_seed);
    let voice = s("voice");
    api.models.check(&model, Kind::Speech)?;
    let api2 = api.clone();
    let mut run = move |sink: Option<Sink>| -> ApiResult<Option<crate::tts::Speech>> {
        let mut lease = api2.models.acquire(&model, Kind::Speech)?;
        let Loaded::Speech(sp) = lease.model() else { unreachable!() };
        let info = sp.info();
        o.voice = match voice.as_deref() {
            Some(v) if info.voices.iter().any(|x| x == v) => Some(v.to_string()),
            Some(v) if !OPENAI_VOICES.contains(&v) && !info.voices.is_empty() => {
                return Err(ApiError::bad(format!("voice {v:?}: this model's voices are {}", info.voices.join(", "))).param("voice"));
            }
            Some(v) if info.voices.is_empty() && !OPENAI_VOICES.contains(&v) => Some(v.to_string()),
            _ if info.voices.iter().any(|x| x == "tara") => Some("tara".into()),
            _ => info.voices.first().cloned(),
        };
        match sink {
            None => Ok(Some(
                sp.speak_detailed(&input, &o).map_err(|e| ApiError::new(500, format!("{e:#}")))?,
            )),
            Some(sink) => {
                let r = sp.speak_stream(&input, &o, &mut |c| {
                    let pcm: Vec<u8> = c
                        .samples
                        .iter()
                        .flat_map(|&x| ((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
                        .collect();
                    let ok = if sse {
                        let ev = json!({ "type": "speech.audio.delta", "audio": crate::b64::encode(&pcm) });
                        sink(format!("data: {ev}\n\n").as_bytes())
                    } else {
                        sink(&pcm)
                    };
                    if ok { Flow::Continue } else { Flow::Stop }
                });
                match r {
                    Ok(speech) if sse => {
                        let ev = json!({
                            "type": "speech.audio.done",
                            "usage": { "input_tokens": speech.stats.prompt_tokens, "output_tokens": speech.stats.tokens,
                                       "total_tokens": speech.stats.prompt_tokens + speech.stats.tokens },
                        });
                        sink(format!("data: {ev}\n\n").as_bytes());
                    }
                    Ok(_) => {}
                    Err(e) => return Err(ApiError::new(500, format!("{e:#}"))),
                }
                Ok(None)
            }
        }
    };
    if format == "pcm" || sse {
        let producer = move |sink: Sink| {
            if let Err(e) = run(Some(&mut *sink)) {
                if sse {
                    sink(format!("data: {}\n\n", e.to_json()).as_bytes());
                } else {
                    eprintln!("[speech: {}]", e.message);
                }
            }
        };
        return Ok(Output::Stream {
            content_type: if sse { "text/event-stream" } else { "audio/pcm" }.into(),
            producer: Box::new(producer),
        });
    }
    let speech = run(None)?.expect("a complete utterance");
    Ok(Output::Full(ApiResponse {
        status: 200,
        content_type: "audio/wav".into(),
        body: speech.audio.to_wav(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_fields_and_files() {
        let body = b"--XyZ\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper\r\n--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\x00\r\n\x01\r\n--XyZ--\r\n";
        let parts = multipart(Some("multipart/form-data; boundary=XyZ"), body).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].text(), "whisper");
        assert_eq!(parts[1].filename.as_deref(), Some("a.wav"));
        assert_eq!(parts[1].content_type.as_deref(), Some("audio/wav"));
        assert_eq!(parts[1].data, b"RIFF\x00\r\n\x01");
        assert!(multipart(Some("application/json"), body).is_err());
        assert!(multipart(Some("multipart/form-data; boundary=other"), body).is_err());
    }

    #[test]
    fn extensions() {
        assert_eq!(extension(b"\x89PNG\r\n", None, None), "png");
        assert_eq!(extension(b"RIFF\0\0\0\0WAVEfmt ", None, None), "wav");
        assert_eq!(extension(b"????", Some("audio/mpeg"), None), "mp3");
        assert_eq!(extension(b"????", None, Some("https://x/y/clip.webm?a=1")), "webm");
        assert_eq!(extension(b"????", None, None), "bin");
    }

    #[test]
    fn subtitle_formats() {
        let seg = crate::whisper::Segment {
            id: 0,
            start: 1.5,
            end: 62.25,
            text: " Hello.".into(),
            tokens: Vec::new(),
            temperature: 0.0,
            avg_logprob: 0.0,
            compression_ratio: 0.0,
            no_speech_prob: 0.0,
            words: Vec::new(),
        };
        assert_eq!(
            subtitles(std::slice::from_ref(&seg), false),
            "1\n00:00:01,500 --> 00:01:02,250\nHello.\n\n"
        );
        assert_eq!(subtitles(&[seg], true), "WEBVTT\n\n00:00:01.500 --> 00:01:02.250\nHello.\n\n");
    }

    #[test]
    fn image_job_fields() {
        let body = json!({"model": "sd", "prompt": "a cat", "size": "512x384", "n": 2, "seed": "7", "response_format": "url"});
        let j = ImageJob::parse(&|k| body.get(k).cloned()).unwrap();
        assert_eq!((j.size, j.n, j.seed, j.as_url), (Some((512, 384)), 2, Some(7), true));
        let bad = json!({"model": "sd", "prompt": "x", "size": "huge"});
        assert_eq!(ImageJob::parse(&|k| bad.get(k).cloned()).err().unwrap().param, Some("size"));
        let bad = json!({"model": "sd", "prompt": "x", "n": 0});
        assert!(ImageJob::parse(&|k| bad.get(k).cloned()).is_err());
    }
}
