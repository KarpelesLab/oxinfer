//! Endpoints of the non-generative models: voice isolation (`POST /v1/audio/isolations`), depth
//! estimation (`POST /v1/images/depth`, `POST /v1/videos/depth`) and OCR (`POST /v1/ocr`).
//!
//! Each takes `multipart/form-data` (the input as the `file` part, the other fields as form
//! fields) or a JSON object (the input base64-encoded in `file`, optionally as a `data:` URL;
//! the image endpoints also take `image`).

use super::media::{Part, TempFiles, extension, multipart};
use super::registry::{Kind, Loaded};
use super::{Api, ApiError, ApiRequest, ApiResponse, ApiResult, Output};
use serde_json::{Value, json};
use std::sync::Arc;

/// The request's fields as text and its input file, from either body format.
struct Fields {
    text: Vec<(String, String)>,
    file: Option<Part>,
}

impl Fields {
    fn parse(req: &ApiRequest, file_key: &str) -> ApiResult<Self> {
        if req.content_type.as_deref().is_some_and(|c| c.starts_with("multipart/")) {
            let parts = multipart(req.content_type.as_deref(), &req.body)?;
            let file = parts.iter().position(|p| p.name == file_key || p.name == "file");
            let mut text = Vec::new();
            for (i, p) in parts.iter().enumerate() {
                if Some(i) != file {
                    text.push((p.name.clone(), p.text()));
                }
            }
            let file = file.map(|i| Part {
                name: parts[i].name.clone(),
                filename: parts[i].filename.clone(),
                content_type: parts[i].content_type.clone(),
                data: parts[i].data.clone(),
            });
            return Ok(Self { text, file });
        }
        let v: Value = serde_json::from_slice(&req.body).map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
        let Value::Object(o) = v else {
            return Err(ApiError::bad("the body must be a JSON object"));
        };
        let mut text = Vec::new();
        let mut file = None;
        for (k, v) in o {
            if k == file_key || k == "file" {
                let s = v.as_str().ok_or_else(|| ApiError::bad(format!("{k}: expected a base64 string")))?;
                let (mime, data) = match s.strip_prefix("data:").and_then(|r| r.split_once(',')) {
                    Some((head, d)) => (Some(head.trim_end_matches(";base64").to_string()), d),
                    None => (None, s),
                };
                let data = crate::b64::decode(data.trim()).ok_or_else(|| ApiError::bad(format!("{k}: invalid base64")))?;
                file = Some(Part {
                    name: k,
                    filename: None,
                    content_type: mime,
                    data,
                });
            } else {
                let t = match v {
                    Value::String(s) => s,
                    Value::Null => continue,
                    other => other.to_string(),
                };
                text.push((k, t));
            }
        }
        Ok(Self { text, file })
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.text
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
            .filter(|v| !v.is_empty())
    }
}

/// `POST /v1/audio/isolations`: the voice of an audio file (any format the server decodes).
/// Fields: `model`, `file`, `response_format` (`wav`: float32 WAV, the default; `wav16`: 16-bit
/// PCM WAV; `pcm`: raw little-endian float32; `json`: base64 WAVs of the voice and the
/// background with the figures), `output` (`voice`, the default, or `background`: what the
/// voice was taken from), `output_sample_rate` (default 48000; 0 keeps the input's), and the
/// isolation options ([`crate::voice_isolator::IsolateOptions::KEYS`]).
pub(super) fn isolations(api: &Arc<Api>, req: &ApiRequest) -> ApiResult<Output> {
    let f = Fields::parse(req, "file")?;
    let model = f
        .get("model")
        .ok_or_else(|| ApiError::bad("model is required").param("model"))?
        .to_string();
    let file = f.file.as_ref().ok_or_else(|| ApiError::bad("file is required").param("file"))?;
    let format = f.get("response_format").unwrap_or("wav").to_string();
    if !["wav", "wav16", "pcm", "json"].contains(&format.as_str()) {
        return Err(ApiError::bad(format!("response_format {format:?}: expected wav, wav16, pcm or json")).param("response_format"));
    }
    let background = match f.get("output").unwrap_or("voice") {
        "voice" => false,
        "background" | "residual" | "noise" => true,
        o => return Err(ApiError::bad(format!("output {o:?}: expected voice or background")).param("output")),
    };
    let mut o = crate::voice_isolator::IsolateOptions::default();
    for k in crate::voice_isolator::IsolateOptions::KEYS {
        if let Some(v) = f.get(k) {
            o.set(k, v).map_err(|e| ApiError::bad(format!("{e:#}")))?;
        }
    }
    let out_rate: Option<u32> = match f.get("output_sample_rate") {
        None => Some(crate::voice_isolator::SAMPLE_RATE),
        Some(v) => {
            let r: u32 = v
                .parse()
                .map_err(|_| ApiError::bad("output_sample_rate must be a whole number").param("output_sample_rate"))?;
            (r != 0).then_some(r)
        }
    };
    api.models.check(&model, Kind::Isolation)?;
    let ext = extension(&file.data, file.content_type.as_deref(), file.filename.as_deref());
    let mut temp = TempFiles::default();
    let p = temp.write(&file.data, &ext)?;
    let (pcm, rate) = crate::voice_isolator::load_audio(&p).map_err(|e| ApiError::bad(format!("decode audio: {e:#}")))?;
    let t0 = std::time::Instant::now();
    let mut lease = api.models.acquire(&model, Kind::Isolation)?;
    let Loaded::Isolation(iso) = lease.model() else { unreachable!() };
    let sr = iso.sample_rate();
    let voice = iso.isolate(&pcm, rate, &o).map_err(|e| {
        let msg = format!("{e:#}");
        ApiError::new(
            if msg.contains("must") || msg.contains("applies") {
                400
            } else {
                500
            },
            msg,
        )
    })?;
    drop(lease);
    let secs = t0.elapsed().as_secs_f64();
    let (rate_out, voice, input) = match out_rate {
        Some(r) if r == sr => {
            let x = crate::audio::resample_kaiser_best(&pcm, rate as usize, sr as usize);
            (r, voice, x)
        }
        Some(r) => (
            r,
            crate::audio::resample_kaiser_best(&voice, sr as usize, r as usize),
            crate::audio::resample_kaiser_best(&pcm, rate as usize, r as usize),
        ),
        None => {
            let mut v = crate::audio::resample_kaiser_best(&voice, sr as usize, rate as usize);
            v.resize(pcm.len(), 0.0);
            (rate, v, pcm)
        }
    };
    let residual: Vec<f32> = input.iter().zip(&voice).map(|(a, v)| a - v).collect();
    let main = if background { &residual } else { &voice };
    let body = |ct: &str, b: Vec<u8>| {
        Output::Full(ApiResponse {
            status: 200,
            content_type: ct.into(),
            body: b,
        })
    };
    Ok(match format.as_str() {
        "wav" => body("audio/wav", crate::tts::wav_bytes_f32(main, rate_out)),
        "wav16" => {
            let peak = main.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s: Vec<f32> = if peak > 1.0 {
                main.iter().map(|v| v / peak).collect()
            } else {
                main.clone()
            };
            body("audio/wav", crate::tts::wav_bytes(&s, rate_out))
        }
        "pcm" => body("application/octet-stream", main.iter().flat_map(|v| v.to_le_bytes()).collect()),
        _ => body(
            "application/json",
            serde_json::to_vec(&json!({
                "object": "audio.isolation",
                "model": model,
                "sample_rate": rate_out,
                "duration": voice.len() as f64 / rate_out as f64,
                "seconds": secs,
                "voice": crate::b64::encode(&crate::tts::wav_bytes_f32(&voice, rate_out)),
                "background": crate::b64::encode(&crate::tts::wav_bytes_f32(&residual, rate_out)),
            }))
            .unwrap_or_default(),
        ),
    })
}

/// `POST /v1/images/depth`: the metric depth of an image (PNG, JPEG, WebP, GIF, BMP). Fields:
/// `model`, `file` (also `image`), `focal_px` (the focal length in pixels; default: the image's
/// EXIF 35 mm focal length, else the estimator's), `exif` (`false` ignores it),
/// `response_format` (`json`, the default: the figures with the base64 color preview `png`,
/// 16-bit `png16` and, with `include_depth`, the float32 depth `depth` (little-endian, meters);
/// `png`: the color preview alone; `png16`: the 16-bit PNG alone; `f32`: the raw depth alone).
pub(super) fn depth(api: &Arc<Api>, req: &ApiRequest) -> ApiResult<Output> {
    let f = Fields::parse(req, "image")?;
    let model = f
        .get("model")
        .ok_or_else(|| ApiError::bad("model is required").param("model"))?
        .to_string();
    let file = f
        .file
        .as_ref()
        .ok_or_else(|| ApiError::bad("file (or image) is required").param("file"))?;
    let format = f.get("response_format").unwrap_or("json").to_string();
    if !["json", "png", "png16", "f32"].contains(&format.as_str()) {
        return Err(ApiError::bad(format!("response_format {format:?}: expected json, png, png16 or f32")).param("response_format"));
    }
    let given: Option<f32> = match f.get("focal_px") {
        None => None,
        Some(v) => Some(
            v.parse::<f32>()
                .ok()
                .filter(|f| f.is_finite() && *f > 0.0)
                .ok_or_else(|| ApiError::bad("focal_px must be a positive number").param("focal_px"))?,
        ),
    };
    let flag = |k: &str| f.get(k).map(|v| matches!(v, "1" | "true" | "yes" | "on"));
    let use_exif = flag("exif").unwrap_or(true);
    api.models.check(&model, Kind::Depth)?;
    let img = crate::depth_pro::decode_image(&file.data).map_err(|e| ApiError::bad(format!("decode image: {e:#}")))?;
    let (f_px, source) = match (given, use_exif.then_some(img.focal_px).flatten()) {
        (Some(v), _) => (Some(v), "given"),
        (None, Some(v)) => (Some(v), "exif"),
        (None, None) => (None, "estimated"),
    };
    let t0 = std::time::Instant::now();
    let mut lease = api.models.acquire(&model, Kind::Depth)?;
    let Loaded::Depth(net) = lease.model() else { unreachable!() };
    let d = net
        .infer_rgb(&img.rgb, img.width, img.height, f_px)
        .map_err(|e| ApiError::new(500, format!("{e:#}")))?;
    drop(lease);
    let secs = t0.elapsed().as_secs_f64();
    let body = |ct: &str, b: Vec<u8>| {
        Output::Full(ApiResponse {
            status: 200,
            content_type: ct.into(),
            body: b,
        })
    };
    let png = || crate::png::encode_rgb(d.width, d.height, &d.colorize());
    let (g16, k) = d.gray16();
    let png16 = || crate::png::encode_gray16(d.width, d.height, &g16);
    let raw = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    Ok(match format.as_str() {
        "png" => body("image/png", png()),
        "png16" => body("image/png", png16()),
        "f32" => {
            let depth = d
                .depth()
                .ok_or_else(|| ApiError::bad("no focal length (give focal_px or load the estimator)"))?;
            body("application/octet-stream", raw(&depth))
        }
        _ => {
            let range = d.depth_range();
            let mut v = json!({
                "object": "image.depth",
                "model": model,
                "width": d.width,
                "height": d.height,
                "focal_px": d.focal_px,
                "focal_source": if d.focal_px.is_some() { Some(source) } else { None },
                "fov_deg": d.fov_deg,
                "depth_p05": range.map(|r| r.0).filter(|v| v.is_finite()),
                "depth_p95": range.map(|r| r.1).filter(|v| v.is_finite()),
                "far_fraction": range.map(|r| r.2),
                "png16_meters_numerator": d.focal_px.map(|_| k),
                "seconds": secs,
                "png": crate::b64::encode(&png()),
                "png16": crate::b64::encode(&png16()),
            });
            if flag("include_depth").unwrap_or(false)
                && let Some(depth) = d.depth()
            {
                v["depth"] = json!(crate::b64::encode(&raw(&depth)));
            }
            body("application/json", serde_json::to_vec(&v).unwrap_or_default())
        }
    })
}

/// `POST /v1/videos/depth`: the metric depth of every frame of a video (a file oxideav decodes:
/// MP4, MOV, Matroska / WebM, AVI; the `media` feature). Fields: `model`, `file` (also `video`),
/// `mode` (`buffered`, the default, or `unbuffered`), `window` (16), `overlap` (4), `focal_px`
/// (default: estimated over the first window), `response_format` (`json`, the default: the
/// figures of each frame, with `include_depth` the float32 depth of every frame, little-endian,
/// frame after frame, base64; `mp4`: the color preview; `f32`: the depth of every frame).
pub(super) fn depth_video(api: &Arc<Api>, req: &ApiRequest) -> ApiResult<Output> {
    let f = Fields::parse(req, "video")?;
    let model = f
        .get("model")
        .ok_or_else(|| ApiError::bad("model is required").param("model"))?
        .to_string();
    let file = f
        .file
        .as_ref()
        .ok_or_else(|| ApiError::bad("file (or video) is required").param("file"))?;
    let format = f.get("response_format").unwrap_or("json").to_string();
    if !["json", "mp4", "f32"].contains(&format.as_str()) {
        return Err(ApiError::bad(format!("response_format {format:?}: expected json, mp4 or f32")).param("response_format"));
    }
    let mut vo = crate::depth_pro::VideoOptions::default();
    for k in crate::depth_pro::VideoOptions::KEYS {
        if let Some(v) = f.get(k) {
            vo.set(k, v).map_err(|e| ApiError::bad(format!("{e:#}")).param(k))?;
        }
    }
    vo.check().map_err(|e| ApiError::bad(format!("{e:#}")))?;
    let flag = |k: &str| f.get(k).map(|v| matches!(v, "1" | "true" | "yes" | "on"));
    api.models.check(&model, Kind::Depth)?;
    #[cfg(feature = "media")]
    {
        let ext = file
            .filename
            .as_deref()
            .and_then(|n| std::path::Path::new(n).extension())
            .and_then(|e| e.to_str())
            .filter(|e| crate::depth_pro::VIDEO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or("mp4")
            .to_string();
        let tmp = std::env::temp_dir().join(format!(
            "oxinfer-depth-{}-{:x}.{ext}",
            std::process::id(),
            crate::clock::unix_time().as_nanos()
        ));
        std::fs::write(&tmp, &file.data).map_err(|e| ApiError::new(500, format!("temporary file: {e}")))?;
        let clip = crate::depth_pro::load_video(&tmp, None);
        let _ = std::fs::remove_file(&tmp);
        let clip = clip.map_err(|e| ApiError::bad(format!("decode video: {e:#}")))?;
        let t0 = std::time::Instant::now();
        let mut lease = api.models.acquire(&model, Kind::Depth)?;
        let Loaded::Depth(net) = lease.model() else { unreachable!() };
        let frames: Vec<&[u8]> = clip.frames.iter().map(|f| &f[..]).collect();
        let maps = net
            .infer_video(&frames, clip.width, clip.height, &vo)
            .map_err(|e| ApiError::new(500, format!("{e:#}")))?;
        drop(lease);
        let secs = t0.elapsed().as_secs_f64();
        let body = |ct: &str, b: Vec<u8>| {
            Output::Full(ApiResponse {
                status: 200,
                content_type: ct.into(),
                body: b,
            })
        };
        let all_depth = || -> ApiResult<Vec<u8>> {
            let mut out = Vec::with_capacity(maps.len() * clip.width * clip.height * 4);
            for m in &maps {
                let d = m
                    .depth()
                    .ok_or_else(|| ApiError::bad("no focal length (give focal_px or load the estimator)"))?;
                out.extend(d.iter().flat_map(|x| x.to_le_bytes()));
            }
            Ok(out)
        };
        Ok(match format.as_str() {
            "mp4" => body(
                "video/mp4",
                crate::depth_pro::preview_mp4(&maps, clip.fps).map_err(|e| ApiError::new(500, format!("{e:#}")))?,
            ),
            "f32" => body("application/octet-stream", all_depth()?),
            _ => {
                let first = maps.first().map(|m| m.scale());
                let mut v = json!({
                    "object": "video.depth",
                    "model": model,
                    "frames": maps.len(),
                    "width": clip.width,
                    "height": clip.height,
                    "fps": clip.fps,
                    "mode": vo.mode.as_str(),
                    "window": vo.window,
                    "overlap": vo.overlap,
                    "focal_px": maps.last().and_then(|m| m.focal_px),
                    "focal_source": maps.last().and_then(|m| m.focal_px).map(|_| if vo.focal_px.is_some() { "given" } else { "estimated" }),
                    "png16_meters_numerator": first.filter(|_| maps[0].focal_px.is_some()).map(|s| 65535.0 / s),
                    "seconds": secs,
                    "per_frame": maps.iter().enumerate().map(|(i, m)| {
                        let r = m.depth_range();
                        json!({
                            "index": i,
                            "depth_p05": r.map(|r| r.0).filter(|v| v.is_finite()),
                            "depth_p95": r.map(|r| r.1).filter(|v| v.is_finite()),
                            "far_fraction": r.map(|r| r.2),
                        })
                    }).collect::<Vec<_>>(),
                });
                if flag("include_depth").unwrap_or(false) {
                    v["depth"] = json!(crate::b64::encode(&all_depth()?));
                }
                body("application/json", serde_json::to_vec(&v).unwrap_or_default())
            }
        })
    }
    #[cfg(not(feature = "media"))]
    {
        let _ = (file, format, vo, flag);
        Err(ApiError::bad(
            "video input needs the `media` feature (this server was built without it)",
        ))
    }
}

/// `POST /v1/ocr`: the text of an image (PNG, JPEG, WebP, GIF, BMP) with OCRia-nano-1. Fields:
/// `model`, `file` (or `image`), `response_format` (`json`, the default: the page with its
/// lines, boxes, text and confidences; `text`: the text alone), and the page options
/// ([`crate::ocria::OcrRunOptions::KEYS`]: `plain`, `det_long_side`, `det_thresh`, the switches).
pub(super) fn ocr(api: &Arc<Api>, req: &ApiRequest) -> ApiResult<Output> {
    let f = Fields::parse(req, "image")?;
    let model = f
        .get("model")
        .ok_or_else(|| ApiError::bad("model is required").param("model"))?
        .to_string();
    let file = f
        .file
        .as_ref()
        .ok_or_else(|| ApiError::bad("file (or image) is required").param("file"))?;
    let format = f.get("response_format").unwrap_or("json").to_string();
    if !["json", "text"].contains(&format.as_str()) {
        return Err(ApiError::bad(format!("response_format {format:?}: expected json or text")).param("response_format"));
    }
    let mut o = crate::ocria::OcrRunOptions::default();
    // `plain` first: it resets the others.
    if let Some(v) = f.get("plain") {
        o.set("plain", v).map_err(|e| ApiError::bad(format!("{e:#}")).param("plain"))?;
    }
    for k in crate::ocria::OcrRunOptions::KEYS.iter().filter(|k| **k != "plain") {
        if let Some(v) = f.get(k) {
            o.set(k, v).map_err(|e| ApiError::bad(format!("{e:#}")).param(k))?;
        }
    }
    api.models.check(&model, Kind::Ocr)?;
    let img = crate::mm::decode_images(std::slice::from_ref(&file.data)).map_err(|e| ApiError::bad(format!("decode image: {e:#}")))?;
    let img = &img[0];
    let mut lease = api.models.acquire(&model, Kind::Ocr)?;
    let Loaded::Ocr(m) = lease.model() else { unreachable!() };
    let page = m
        .ocr_rgb(&img.data, img.width, img.height, &o)
        .map_err(|e| ApiError::new(500, format!("{e:#}")))?;
    drop(lease);
    let body = |ct: &str, b: Vec<u8>| {
        Output::Full(ApiResponse {
            status: 200,
            content_type: ct.into(),
            body: b,
        })
    };
    Ok(match format.as_str() {
        "text" => body("text/plain; charset=utf-8", page.text.into_bytes()),
        _ => {
            let mut v = serde_json::to_value(&page).unwrap_or_default();
            if let Value::Object(m) = &mut v {
                m.insert("object".into(), json!("ocr.page"));
                m.insert("model".into(), json!(model));
            }
            body("application/json", serde_json::to_vec(&v).unwrap_or_default())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerOptions;

    #[test]
    fn isolation_endpoint() {
        let path = crate::voice_isolator::tests::tiny("srv", false);
        let mut opts = ServerOptions::default();
        opts.set("model", &format!("vi={}", path.display())).unwrap();
        let api = Api::new(opts).unwrap();
        let x = crate::voice_isolator::tests::signal(1200);
        let wav = crate::tts::wav_bytes_f32(&x, 1600);
        let body = json!({ "model": "vi", "file": crate::b64::encode(&wav), "response_format": "json", "output_sample_rate": 0 });
        let req = ApiRequest::post_json("/v1/audio/isolations", &body.to_string());
        let Output::Full(r) = api.handle(&req) else {
            panic!("a full response")
        };
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["sample_rate"], 1600);
        let voice = crate::tts::read_wav(&crate::b64::decode(v["voice"].as_str().unwrap()).unwrap()).unwrap();
        let bg = crate::tts::read_wav(&crate::b64::decode(v["background"].as_str().unwrap()).unwrap()).unwrap();
        assert_eq!(voice.samples.len(), x.len());
        for ((a, b), c) in voice.samples.iter().zip(&bg.samples).zip(&x) {
            assert!((a + b - c).abs() < 1e-5);
        }
        // Raw WAV out, and a bad option.
        let body = json!({ "model": "vi", "file": crate::b64::encode(&wav) });
        let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/audio/isolations", &body.to_string())) else {
            panic!()
        };
        assert_eq!((r.status, r.content_type.as_str()), (200, "audio/wav"));
        let body = json!({ "model": "vi", "file": crate::b64::encode(&wav), "lookahead_frames": "x" });
        let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/audio/isolations", &body.to_string())) else {
            panic!()
        };
        assert_eq!(r.status, 400);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn depth_endpoint() {
        let (dir, fdir) = crate::depth_pro::tests::tiny("srv");
        let mut opts = ServerOptions::default();
        opts.set("model", &format!("d={},focal={}", dir.display(), fdir.display())).unwrap();
        let api = Api::new(opts).unwrap();
        let (w, h) = (40, 30);
        let png = crate::png::encode_rgb(w, h, &crate::depth_pro::tests::image(w, h));
        let body = json!({ "model": "d", "image": crate::b64::encode(&png), "include_depth": true });
        let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/images/depth", &body.to_string())) else {
            panic!()
        };
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(40), Some(30)));
        assert_eq!(v["focal_source"], "estimated");
        let depth = crate::b64::decode(v["depth"].as_str().unwrap()).unwrap();
        assert_eq!(depth.len(), w * h * 4);
        let body = json!({ "model": "d", "file": crate::b64::encode(&png), "focal_px": 50, "response_format": "png16" });
        let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/images/depth", &body.to_string())) else {
            panic!()
        };
        assert_eq!((r.status, r.content_type.as_str()), (200, "image/png"));
        // A video: its frames' depth (an MP4 made from depth previews of the tiny model).
        #[cfg(feature = "media")]
        {
            let (w, h) = (48, 32);
            let net = crate::depth_pro::DepthPro::load(
                &dir.to_string_lossy(),
                &crate::depth_pro::DepthOptions {
                    focal: None,
                    ..Default::default()
                },
            )
            .unwrap();
            let maps: Vec<_> = (0..3u8)
                .map(|k| {
                    let img: Vec<u8> = crate::depth_pro::tests::image(w, h)
                        .iter()
                        .map(|v| v.wrapping_add(40 * k))
                        .collect();
                    net.infer_rgb(&img, w, h, Some(40.0)).unwrap()
                })
                .collect();
            let mp4 = crate::depth_pro::preview_mp4(&maps, 10.0).unwrap();
            let body = json!({ "model": "d", "video": crate::b64::encode(&mp4), "mode": "unbuffered", "window": 2, "include_depth": true });
            let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/videos/depth", &body.to_string())) else {
                panic!()
            };
            assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
            let v: Value = serde_json::from_slice(&r.body).unwrap();
            assert_eq!(
                (v["object"].as_str(), v["frames"].as_u64(), v["mode"].as_str()),
                (Some("video.depth"), Some(3), Some("unbuffered"))
            );
            assert_eq!(v["per_frame"].as_array().unwrap().len(), 3);
            assert_eq!(crate::b64::decode(v["depth"].as_str().unwrap()).unwrap().len(), 3 * w * h * 4);
            let body = json!({ "model": "d", "video": crate::b64::encode(&mp4), "overlap": 16 });
            let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/videos/depth", &body.to_string())) else {
                panic!()
            };
            assert_eq!(r.status, 400, "{}", String::from_utf8_lossy(&r.body));
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ocr_endpoint() {
        let dir = crate::ocria::tests::tiny("srv");
        let mut opts = ServerOptions::default();
        opts.set("model", &format!("ocr={},scripts=latin", dir.display())).unwrap();
        let api = Api::new(opts).unwrap();
        let (w, h) = (320, 160);
        let png = crate::png::encode_rgb(
            w,
            h,
            &crate::ocria::tests::page_with(w, h, &[(40, 30, 260, 50), (40, 100, 280, 120)]),
        );
        let body = json!({ "model": "ocr", "image": crate::b64::encode(&png), "plain": true });
        let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/ocr", &body.to_string())) else {
            panic!("a full response")
        };
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["object"], "ocr.page");
        assert_eq!(v["text"], "x\nx");
        assert_eq!(v["lines"].as_array().unwrap().len(), 2);
        assert_eq!(v["lines"][0]["script"], "latin");
        // Text out, a bad option, a wrong kind of model.
        let body = json!({ "model": "ocr", "file": crate::b64::encode(&png), "response_format": "text" });
        let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/ocr", &body.to_string())) else {
            panic!()
        };
        assert_eq!(
            (r.status, r.content_type.as_str(), r.body.as_slice()),
            (200, "text/plain; charset=utf-8", &b"x\nx"[..])
        );
        let body = json!({ "model": "ocr", "file": crate::b64::encode(&png), "gutter": "maybe" });
        let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/ocr", &body.to_string())) else {
            panic!()
        };
        assert_eq!(r.status, 400);
        let Output::Full(r) = api.handle(&ApiRequest::post_json("/v1/ocr", &json!({ "model": "ocr" }).to_string())) else {
            panic!()
        };
        assert_eq!(r.status, 400);
        let _ = std::fs::remove_dir_all(dir);
    }
}
