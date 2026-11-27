//! Video generation (MiniMax-H3) as asynchronous jobs, in the shape of OpenAI's videos API
//! (`/v1/videos`, Sora's): a request creates a job and answers at once with its `video` object;
//! the job then waits for the model, loads it, encodes the prompt, denoises (`progress` 0 to
//! 100, `step` of `steps`) and decodes the frames and the soundtrack; its files are downloaded
//! once it is `completed`.
//!
//! | Endpoint | |
//! |---|---|
//! | `POST /v1/videos` (also `/v1/videos/generations`) | JSON or multipart: creates a job |
//! | `GET /v1/videos` | the jobs, newest first |
//! | `GET /v1/videos/{id}` | one job |
//! | `GET /v1/videos/{id}/events` | the job as server-sent events until it ends (an extension) |
//! | `GET /v1/videos/{id}/content?variant=video\|audio\|thumbnail` | the video (MP4: H.264 and AAC, by default), the WAV soundtrack, the first frame as PNG |
//! | `POST /v1/videos/{id}/cancel` | stops it between two steps (an extension) |
//! | `DELETE /v1/videos/{id}` | cancels it if it runs and forgets it |
//!
//! Request fields: `model`, `prompt` (required); `seconds` (OpenAI's: a number or a string; the
//! frame count at 24 fps, rounded up to `17 n + 5`) or `frames`; `size` (`WxH`, multiples of 32;
//! default 16:9 at a 768 short edge, or the first frame's aspect ratio); `steps` (points of the
//! sigma grid, default 50, or a turbo LoRA's); `seed`; `input_reference` or `first_frame`, and
//! `last_frame`: images (first / last frame to video); `references`: images, audio clips and
//! videos for ref2va (a model whose folder is the release's `Ref2VA/`); `ref_image_edge`;
//! `still_prompt`, `motion_scale`, `stg_scale`, `stg_layers`, `guidance_until`: the motion /
//! structure guidance of Sulphur-3's ComfyUI pack (`crate::minimax_h3::video::MotionGuidance`). In
//! JSON the media are `data:` or `http(s)://` URLs (as the chat endpoint takes them); in a
//! multipart body files (`references[]` repeated).

use super::media::{TempFiles, extension, multipart, to_file};
use super::registry::{Kind, Loaded, Stage};
use super::{Api, ApiError, ApiRequest, ApiResponse, ApiResult, Output, now};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Finished jobs are kept this long (their `expires_at`), and at most this many jobs.
const KEEP_SECS: u64 = 3600;
const KEEP_JOBS: usize = 32;

/// What a job makes: the files of a completed one.
#[derive(Default)]
struct Files {
    /// The video file in the request's format (the `media` feature).
    video: Option<(Vec<u8>, &'static str)>,
    wav: Option<Vec<u8>>,
    thumbnail: Option<Vec<u8>>,
}

struct State {
    /// `queued`, `in_progress`, `completed`, `failed`, `cancelled`.
    status: &'static str,
    /// What it is doing: `queued`, `waiting`, `downloading`, `loading`, `generating`,
    /// `decoding`, `encoding`, then the status.
    stage: &'static str,
    progress: f64,
    step: usize,
    steps: usize,
    download: Option<Value>,
    error: Option<String>,
    started: Option<Instant>,
    finished: Option<Instant>,
    completed_at: Option<u64>,
    /// The canvas, frame count and rate, once known.
    size: Option<(usize, usize)>,
    frames: Option<usize>,
    fps: Option<f64>,
    has_audio: bool,
    files: Files,
}

/// One video job.
pub(crate) struct Job {
    pub id: String,
    model: String,
    prompt: String,
    created_at: u64,
    seconds: f64,
    requested_size: Option<(usize, usize)>,
    cancel: Arc<AtomicBool>,
    st: Mutex<State>,
}

impl Job {
    fn running(&self) -> bool {
        matches!(lock(&self.st).status, "queued" | "in_progress")
    }

    pub fn to_json(&self) -> Value {
        let s = lock(&self.st);
        let size = s.size.or(self.requested_size).map(|(w, h)| format!("{w}x{h}"));
        let fps = s.fps.unwrap_or(crate::minimax_h3::video::FPS as f64);
        let seconds = s.frames.map_or(self.seconds, |f| f as f64 / fps);
        let elapsed = s
            .started
            .map(|t| s.finished.unwrap_or_else(Instant::now).duration_since(t).as_secs_f64());
        json!({
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": s.status,
            "progress": s.progress.floor() as u64,
            "created_at": self.created_at,
            "completed_at": s.completed_at,
            "expires_at": s.completed_at.map(|t| t + KEEP_SECS),
            "size": size,
            "seconds": format!("{:.2}", seconds).trim_end_matches('0').trim_end_matches('.').to_string(),
            "prompt": self.prompt,
            "error": s.error.as_ref().map(|m| json!({"code": if s.status == "cancelled" { "cancelled" } else { "generation_failed" }, "message": m})),
            "remixed_from_video_id": null,
            // Extensions.
            "stage": s.stage,
            "step": s.step,
            "steps": s.steps,
            "frames": s.frames,
            "fps": fps,
            "has_audio": s.has_audio,
            "download": s.download,
            "elapsed": elapsed.map(|e| (e * 10.0).round() / 10.0),
            "variants": {
                "video": s.files.video.as_ref().map(|b| b.0.len()),
                "audio": s.files.wav.as_ref().map(|b| b.len()),
                "thumbnail": s.files.thumbnail.as_ref().map(|b| b.len()),
            },
        })
    }

    fn set(&self, f: impl FnOnce(&mut State)) {
        f(&mut lock(&self.st))
    }
}

/// The jobs of an [`Api`].
#[derive(Default)]
pub(crate) struct Videos {
    jobs: Mutex<Vec<Arc<Job>>>,
}

impl Videos {
    fn find(&self, id: &str) -> ApiResult<Arc<Job>> {
        lock(&self.jobs)
            .iter()
            .find(|j| j.id == id)
            .cloned()
            .ok_or_else(|| ApiError::new(404, format!("no video {id:?}")).code("video_not_found"))
    }

    /// Cancels job `id` if it is running (as `POST /v1/videos/{id}/cancel`): false when there
    /// is no such running job.
    pub fn cancel(&self, id: &str) -> bool {
        let Ok(j) = self.find(id) else { return false };
        if !j.running() {
            return false;
        }
        j.cancel.store(true, Ordering::Relaxed);
        j.set(|s| {
            if s.status == "queued" || s.stage == "waiting" {
                s.stage = "cancelling";
            }
        });
        true
    }

    /// The running jobs (the status page).
    pub fn running(&self) -> Vec<Value> {
        lock(&self.jobs).iter().filter(|j| j.running()).map(|j| j.to_json()).collect()
    }

    /// Drops expired files and the oldest finished jobs beyond [`KEEP_JOBS`].
    fn prune(&self) {
        let mut jobs = lock(&self.jobs);
        let t = now();
        for j in jobs.iter() {
            let mut s = lock(&j.st);
            if s.completed_at.is_some_and(|c| c + KEEP_SECS < t) {
                s.files = Files::default();
            }
        }
        while jobs.len() > KEEP_JOBS {
            match jobs.iter().position(|j| !j.running()) {
                Some(i) => {
                    jobs.remove(i);
                }
                None => break,
            }
        }
    }
}

/// A video request's settings (from JSON or a form).
struct Request {
    model: String,
    prompt: String,
    frames: Option<usize>,
    seconds: Option<f64>,
    size: Option<(usize, usize)>,
    steps: Option<usize>,
    seed: Option<u64>,
    ref_image_edge: Option<usize>,
    /// Motion / structure guidance and its still caption
    /// ([`crate::minimax_h3::video::MotionGuidance`]).
    guidance: Option<crate::minimax_h3::video::MotionGuidance>,
    still_prompt: Option<String>,
    /// The video file's format: `mp4` (H.264 and AAC, the default), `webm` (VP9 and Opus), `avi`
    /// (Motion JPEG and PCM).
    format: crate::minimax_h3::video::VideoFormat,
    /// Local files of the first / last frame and of the references (`temp` owns them).
    first: Option<std::path::PathBuf>,
    last: Option<std::path::PathBuf>,
    refs: Vec<std::path::PathBuf>,
    /// Every field of the request (LTX-2.5 reads its own: `fps`, `one_stage`,
    /// `temporal_upscale`, `audio`, `eta`, `sigmas`, `stage2_sigmas`, `noise`, `strength`).
    fields: Value,
    temp: TempFiles,
}

fn parse_size(sz: &str) -> ApiResult<(usize, usize)> {
    let (w, h) = sz
        .split_once(['x', 'X'])
        .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
        .ok_or_else(|| ApiError::bad(format!("size {sz:?}: expected WIDTHxHEIGHT")).param("size"))?;
    if !(32..=4096).contains(&w) || !(32..=4096).contains(&h) || w % 32 != 0 || h % 32 != 0 {
        return Err(ApiError::bad("size: each side a multiple of 32, between 32 and 4096").param("size"));
    }
    Ok((w, h))
}

impl Request {
    fn parse(api: &Api, req: &ApiRequest) -> ApiResult<Self> {
        let mut temp = TempFiles::default();
        let multipart_body = req.content_type.as_deref().is_some_and(|c| c.starts_with("multipart/"));
        // Fields as JSON values, media as local files.
        let (fields, first, last, refs) = if multipart_body {
            let parts = multipart(req.content_type.as_deref(), &req.body)?;
            let mut fields = serde_json::Map::new();
            let (mut first, mut last, mut refs) = (None, None, Vec::new());
            for p in &parts {
                let ext = || extension(&p.data, p.content_type.as_deref(), p.filename.as_deref());
                match p.name.as_str() {
                    "input_reference" | "first_frame" | "image" => first = Some(media_part(api, p, &ext(), &mut temp)?),
                    "last_frame" => last = Some(media_part(api, p, &ext(), &mut temp)?),
                    "references" | "references[]" | "reference" | "reference[]" => refs.push(media_part(api, p, &ext(), &mut temp)?),
                    name => {
                        fields.insert(name.to_string(), Value::String(p.text()));
                    }
                }
            }
            (Value::Object(fields), first, last, refs)
        } else {
            let v = super::json_body(req)?;
            let url = |k: &'static str| -> ApiResult<Option<String>> {
                match v.get(k) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::String(s)) => Ok(Some(s.clone())),
                    // OpenAI's image references: {"image_url": "..."} or {"url": "..."}.
                    Some(Value::Object(o)) => Ok(o
                        .get("image_url")
                        .or_else(|| o.get("url"))
                        .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str)))
                        .map(str::to_string)),
                    Some(_) => Err(ApiError::bad(format!("{k}: a data: or http(s):// URL")).param(k)),
                }
            };
            let first = match url("input_reference")?.or(url("first_frame")?) {
                Some(u) => Some(to_file(api, &u, &mut temp)?),
                None => None,
            };
            let last = match url("last_frame")? {
                Some(u) => Some(to_file(api, &u, &mut temp)?),
                None => None,
            };
            let mut refs = Vec::new();
            match v.get("references") {
                None | Some(Value::Null) => {}
                Some(Value::Array(a)) => {
                    for r in a {
                        let u = r
                            .as_str()
                            .or_else(|| r.get("url").and_then(Value::as_str))
                            .ok_or_else(|| ApiError::bad("references: an array of data: or http(s):// URLs").param("references"))?;
                        refs.push(to_file(api, u, &mut temp)?);
                    }
                }
                Some(_) => return Err(ApiError::bad("references: an array of URLs").param("references")),
            }
            (v, first, last, refs)
        };
        let s = |k: &str| fields.get(k).and_then(|v| v.as_str().map(str::to_string));
        let num = |k: &'static str| -> ApiResult<Option<f64>> {
            match fields.get(k) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Number(n)) => Ok(n.as_f64()),
                Some(Value::String(t)) if t.trim().is_empty() => Ok(None),
                Some(Value::String(t)) => t
                    .trim()
                    .parse()
                    .map(Some)
                    .map_err(|_| ApiError::bad(format!("{k} must be a number")).param(k)),
                Some(_) => Err(ApiError::bad(format!("{k} must be a number")).param(k)),
            }
        };
        let model = s("model")
            .filter(|m| !m.is_empty())
            .ok_or_else(|| ApiError::bad("model is required").param("model"))?;
        let prompt = s("prompt")
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| ApiError::bad("prompt is required").param("prompt"))?;
        let seconds = num("seconds")?;
        if let Some(sec) = seconds
            && !(0.1..=60.0).contains(&sec)
        {
            return Err(ApiError::bad("seconds must be between 0.1 and 60").param("seconds"));
        }
        let frames = num("frames")?.map(|f| f as usize);
        if frames.is_some_and(|f| !(1..=1441).contains(&f)) {
            return Err(ApiError::bad("frames must be between 1 and 1441").param("frames"));
        }
        let steps = num("steps")?.or(num("num_inference_steps")?).map(|v| v as usize);
        if steps.is_some_and(|v| !(2..=200).contains(&v)) {
            return Err(ApiError::bad("steps must be between 2 and 200").param("steps"));
        }
        let size = match s("size").as_deref() {
            None | Some("") | Some("auto") => None,
            Some(sz) => Some(parse_size(sz)?),
        };
        let format = match s("response_format").or_else(|| s("format")).as_deref() {
            None | Some("") => crate::minimax_h3::video::VideoFormat::Mp4,
            Some(f) => crate::minimax_h3::video::VideoFormat::parse(f)
                .ok_or_else(|| ApiError::bad(format!("response_format {f:?}: mp4, webm or avi")).param("response_format"))?,
        };
        if !refs.is_empty() && (first.is_some() || last.is_some()) {
            return Err(ApiError::bad("references (ref2va) and first / last frames (fl2va) are different tasks").param("references"));
        }
        if refs.len() > crate::minimax_h3::ref2va::MAX_REFERENCES {
            return Err(ApiError::bad(format!("at most {} references", crate::minimax_h3::ref2va::MAX_REFERENCES)).param("references"));
        }
        let (guidance, still_prompt) = crate::minimax_h3::video::guidance_fields(&|k| crate::minimax_h3::video::json_field(fields.get(k)))
            .map_err(|e| ApiError::bad(format!("{e:#}")))?;
        Ok(Self {
            model,
            prompt,
            frames,
            seconds,
            size,
            steps,
            seed: num("seed")?.map(|v| v as u64),
            ref_image_edge: num("ref_image_edge")?.map(|v| v as usize),
            guidance,
            still_prompt,
            format,
            first,
            last,
            refs,
            fields,
            temp,
        })
    }
}

/// A media part of a form: an uploaded file, or a text field holding a URL.
fn media_part(api: &Api, p: &super::media::Part, ext: &str, temp: &mut TempFiles) -> ApiResult<std::path::PathBuf> {
    if p.filename.is_none() {
        let t = p.text();
        if !t.is_empty() && !t.contains('\n') && t.len() < 1 << 20 && (t.starts_with("data:") || t.contains("://")) {
            return to_file(api, &t, temp);
        }
    }
    if p.data.is_empty() {
        return Err(ApiError::bad(format!("{}: an empty file", p.name)));
    }
    temp.write(&p.data, ext)
}

/// `POST /v1/videos`: checks the request, starts the job and answers with it.
pub(super) fn create(api: &Arc<Api>, req: &ApiRequest) -> ApiResult<Output> {
    let r = Request::parse(api, req)?;
    api.models.check(&r.model, Kind::Video)?;
    let seconds = r
        .seconds
        .or(r.frames.map(|f| f as f64 / crate::minimax_h3::video::FPS as f64))
        .unwrap_or(124.0 / 24.0);
    let job = Arc::new(Job {
        id: api.new_id("video_"),
        model: r.model.clone(),
        prompt: r.prompt.clone(),
        created_at: now(),
        seconds,
        requested_size: r.size,
        cancel: Arc::new(AtomicBool::new(false)),
        st: Mutex::new(State {
            status: "queued",
            stage: "queued",
            progress: 0.0,
            step: 0,
            steps: 0,
            download: None,
            error: None,
            started: None,
            finished: None,
            completed_at: None,
            size: None,
            frames: None,
            fps: None,
            has_audio: false,
            files: Files::default(),
        }),
    });
    api.videos.prune();
    lock(&api.videos.jobs).push(job.clone());
    // As created: the job may have moved on by the time the thread below has started.
    let created = job.to_json();
    let (api2, j) = (api.clone(), job.clone());
    std::thread::Builder::new()
        .name("oxinfer-video".into())
        .spawn(move || {
            let log = api2.opts.log;
            if log {
                eprintln!("[video {}: {}]", j.id, j.model);
            }
            let r = run(&api2, &j, r);
            let cancelled = j.cancel.load(Ordering::Relaxed);
            j.set(|s| {
                s.finished = Some(Instant::now());
                match &r {
                    Ok(()) => {
                        s.status = "completed";
                        s.stage = "completed";
                        s.progress = 100.0;
                        s.completed_at = Some(now());
                    }
                    Err(_) if cancelled => {
                        s.status = "cancelled";
                        s.stage = "cancelled";
                        s.error = Some("cancelled".into());
                    }
                    Err(e) => {
                        s.status = "failed";
                        s.stage = "failed";
                        s.error = Some(e.message.clone());
                    }
                }
            });
            if log {
                match &r {
                    Ok(()) => eprintln!("[video {}: done]", j.id),
                    Err(e) => eprintln!("[video {}: {}]", j.id, e.message),
                }
            }
        })
        .map_err(|e| ApiError::internal(e.into()))?;
    Ok(Output::Full(ApiResponse::json(200, &created)))
}

/// The job's work, on its own thread.
fn run(api: &Api, j: &Job, r: Request) -> ApiResult<()> {
    let cancelled = || j.cancel.load(Ordering::Relaxed);
    let mut last = Instant::now() - Duration::from_secs(1);
    let mut on = |st: &Stage| {
        j.set(|s| match st {
            Stage::Waiting => s.stage = "waiting",
            Stage::Loading => {
                s.stage = "loading";
                s.status = "in_progress";
                s.download = None;
            }
            Stage::Downloading {
                file,
                index,
                count,
                done,
                total,
            } => {
                if last.elapsed() < Duration::from_millis(200) && total.is_none_or(|t| *done < t) {
                    return;
                }
                last = Instant::now();
                s.stage = "downloading";
                s.status = "in_progress";
                s.download = Some(json!({"file": file, "index": index, "count": count, "done": done, "total": total}));
            }
            Stage::Prompt { .. } => {}
        });
        !cancelled()
    };
    let mut lease = api.models.acquire_with(&r.model, Kind::Video, Some(&mut on))?;
    if cancelled() {
        return Err(ApiError::new(499, "cancelled"));
    }
    let Loaded::Video(m) = lease.model() else { unreachable!() };
    if let Some(l) = &m.ltx {
        let video = run_ltx(j, &r, l)?;
        drop(lease);
        drop(r.temp);
        return finish(j, video, r.format);
    }
    let Some(pipe) = &m.pipe else { unreachable!() };
    j.set(|s| {
        s.status = "in_progress";
        s.stage = "generating";
        s.started = Some(Instant::now());
        s.download = None;
        s.progress = 2.0;
    });
    let mut p = crate::minimax_h3::video::VideoParams::new(&r.prompt);
    p.size = r.size;
    if let Some(f) = r.frames {
        p.frames = f;
    } else if let Some(sec) = r.seconds {
        p.frames = (sec * crate::minimax_h3::video::FPS as f64).round().max(1.0) as usize;
    }
    if let Some(s) = r.steps.or(m.steps) {
        p.steps = s;
    }
    p.seed = r.seed.unwrap_or_else(super::media::random_seed);
    p.cancel = Some(j.cancel.clone());
    p.guidance = r.guidance.clone();
    p.still_prompt = r.still_prompt.clone();
    let bad = |e: anyhow::Error| ApiError::bad(format!("{e:#}"));
    #[cfg(feature = "media")]
    let load_image = |f: &std::path::Path| crate::minimax_h3::encode::load_image(f).map_err(bad);
    #[cfg(not(feature = "media"))]
    let load_image = |_: &std::path::Path| -> ApiResult<crate::sd::Image> {
        Err(ApiError::bad("built without the `media` feature: images cannot be decoded"))
    };
    if let Some(f) = &r.first {
        p.first = Some(load_image(f)?);
    }
    if let Some(f) = &r.last {
        p.last = Some(load_image(f)?);
    }
    let (h, w, frames, _) = pipe.geometry(&p).map_err(bad)?;
    j.set(|s| {
        s.size = Some((w, h));
        s.frames = Some(frames);
        s.steps = p.steps.saturating_sub(1);
        s.has_audio = pipe.audio_vae.is_some();
    });
    let mut progress = |i: usize, n: usize| {
        j.set(|s| {
            s.step = i;
            s.steps = n;
            s.progress = 2.0 + 88.0 * i as f64 / n.max(1) as f64;
            if i == n {
                s.stage = "decoding";
            }
        })
    };
    let fail = |e: anyhow::Error| {
        if cancelled() {
            ApiError::new(499, "cancelled")
        } else {
            ApiError::internal(e)
        }
    };
    let video = if r.refs.is_empty() {
        pipe.generate(&p, None, &mut progress).map_err(fail)?
    } else {
        #[cfg(feature = "media")]
        {
            use crate::minimax_h3::ref2va::{RefOptions, normalize};
            let inputs = r
                .refs
                .iter()
                .map(|f| crate::minimax_h3::encode::load_reference(f))
                .collect::<anyhow::Result<Vec<_>>>()
                .map_err(bad)?;
            let mut o = RefOptions {
                short_edge: p.short_edge,
                max_pixels: p.max_pixels,
                ..Default::default()
            };
            if let Some(e) = r.ref_image_edge {
                o.image_short_edge = e;
            }
            let rate = pipe.audio_vae.as_ref().map_or(32000, |a| a.cfg.sampling_rate);
            let refs = normalize(inputs, frames, rate, &o).map_err(bad)?;
            pipe.generate_ref(&p, &refs, None, &mut progress).map_err(fail)?
        }
        #[cfg(not(feature = "media"))]
        return Err(ApiError::bad("built without the `media` feature: references cannot be decoded"));
    };
    drop(lease);
    drop(r.temp);
    finish(j, video, r.format)
}

/// An LTX-2.5 job: the request's fields as [`crate::ltx::api::params_from_json`] reads them.
fn run_ltx(j: &Job, r: &Request, pipe: &crate::ltx::Pipeline) -> ApiResult<crate::minimax_h3::video::Video> {
    let cancelled = || j.cancel.load(Ordering::Relaxed);
    let bad = |e: anyhow::Error| ApiError::bad(format!("{e:#}"));
    let mut o = r.fields.clone();
    o["prompt"] = Value::String(r.prompt.clone());
    if let Some((w, h)) = r.size {
        o["size"] = Value::String(format!("{w}x{h}"));
    }
    let mut p = crate::ltx::api::params_from_json(&o).map_err(bad)?;
    if let Some(f) = r.frames {
        p.frames = Some(f);
    } else if let Some(sec) = r.seconds {
        p.frames = Some((sec * p.fps as f64).round().max(1.0) as usize);
    }
    p.seed = r.seed.unwrap_or_else(super::media::random_seed);
    p.cancel = Some(j.cancel.clone());
    let mut imgs = Vec::new();
    if let Some(f) = &r.first {
        imgs.push((f.clone(), Some(0), p.images.first().map_or(1.0, |i| i.strength)));
    }
    if let Some(f) = &r.last {
        imgs.push((f.clone(), None, 1.0));
    }
    #[cfg(feature = "media")]
    let load = |f: &std::path::Path| crate::minimax_h3::encode::load_image(f);
    #[cfg(not(feature = "media"))]
    let load = |_: &std::path::Path| -> anyhow::Result<crate::sd::Image> {
        anyhow::bail!("built without the `media` feature: images cannot be decoded")
    };
    crate::ltx::api::add_images(&mut p, imgs, &load).map_err(bad)?;
    j.set(|s| {
        s.status = "in_progress";
        s.stage = "generating";
        s.started = Some(Instant::now());
        s.download = None;
        s.progress = 2.0;
        s.size = Some((p.width, p.height));
        s.fps = Some(p.fps as f64);
    });
    // Progress: the text encoder, then the stages' steps (2 to 90 %), then the decoders.
    let mut progress = |stage: &str, i: usize, n: usize| {
        j.set(|s| {
            let (lo, hi) = match stage {
                "stage 1" => (2.0, 60.0),
                "stage 2" => (60.0, 85.0),
                "temporal" => (85.0, 90.0),
                _ => (90.0, 95.0),
            };
            s.step = i;
            s.steps = n;
            s.stage = if stage == "vae" { "decoding" } else { "generating" };
            s.progress = lo + (hi - lo) * i as f64 / n.max(1) as f64;
        })
    };
    let video = pipe.generate(&p, &mut progress).map_err(|e| {
        if cancelled() {
            ApiError::new(499, "cancelled")
        } else {
            ApiError::internal(e)
        }
    })?;
    j.set(|s| {
        s.size = Some((video.width, video.height));
        s.frames = Some(video.frames.len());
        s.fps = Some(video.fps as f64);
    });
    Ok(video)
}

/// The job's files from its video.
fn finish(j: &Job, video: crate::minimax_h3::video::Video, format: crate::minimax_h3::video::VideoFormat) -> ApiResult<()> {
    let _ = format;
    j.set(|s| {
        s.stage = "encoding";
        s.progress = 95.0;
        s.has_audio = video.audio.as_ref().is_some_and(|a| !a.is_empty());
    });
    let mut files = Files {
        wav: video.audio.as_ref().filter(|a| !a.is_empty()).map(|a| a.to_wav()),
        thumbnail: video.frames.first().map(|f| crate::png::encode_rgb(video.width, video.height, f)),
        video: None,
    };
    #[cfg(feature = "media")]
    {
        let f = format;
        let bytes = crate::minimax_h3::encode::encode_video(&video, f).map_err(ApiError::internal)?;
        files.video = Some((bytes, f.mime()));
    }
    j.set(|s| s.files = files);
    Ok(())
}

/// `GET /v1/videos`: newest first (`limit`, `after`, `order=asc|desc`).
pub(super) fn list(api: &Api, req: &ApiRequest) -> ApiResult<Output> {
    api.videos.prune();
    let mut jobs: Vec<Arc<Job>> = lock(&api.videos.jobs).clone();
    if req.query_param("order").as_deref() != Some("asc") {
        jobs.reverse();
    }
    if let Some(after) = req.query_param("after")
        && let Some(i) = jobs.iter().position(|j| j.id == after)
    {
        jobs.drain(..=i);
    }
    let limit = req
        .query_param("limit")
        .and_then(|l| l.parse().ok())
        .unwrap_or(100usize)
        .clamp(1, 100);
    let more = jobs.len() > limit;
    jobs.truncate(limit);
    let data: Vec<Value> = jobs.iter().map(|j| j.to_json()).collect();
    Ok(Output::Full(ApiResponse::json(
        200,
        &json!({
            "object": "list", "data": data, "has_more": more,
            "first_id": jobs.first().map(|j| j.id.clone()), "last_id": jobs.last().map(|j| j.id.clone()),
        }),
    )))
}

/// `/v1/videos/{id}...`.
pub(super) fn route(api: &Arc<Api>, req: &ApiRequest, rest: &str) -> ApiResult<Output> {
    let get = req.method.eq_ignore_ascii_case("GET") || req.method.eq_ignore_ascii_case("HEAD");
    let post = req.method.eq_ignore_ascii_case("POST");
    let delete = req.method.eq_ignore_ascii_case("DELETE");
    let segs: Vec<&str> = rest.split('/').collect();
    let wrong = || ApiError::new(405, format!("{} is not allowed on {}", req.method, req.path));
    match segs.as_slice() {
        [id] if get => Ok(Output::Full(ApiResponse::json(200, &api.videos.find(id)?.to_json()))),
        [id] if delete => {
            let j = api.videos.find(id)?;
            j.cancel.store(true, Ordering::Relaxed);
            lock(&api.videos.jobs).retain(|x| x.id != j.id);
            Ok(Output::Full(ApiResponse::json(
                200,
                &json!({"id": j.id, "object": "video.deleted", "deleted": true}),
            )))
        }
        [id, "cancel"] if post => {
            let j = api.videos.find(id)?;
            if j.running() {
                j.cancel.store(true, Ordering::Relaxed);
                // A job still waiting for the model stops at its next report; say so now.
                j.set(|s| {
                    if s.status == "queued" || s.stage == "waiting" {
                        s.stage = "cancelling";
                    }
                });
            }
            Ok(Output::Full(ApiResponse::json(200, &j.to_json())))
        }
        [id, "events"] if get => {
            let j = api.videos.find(id)?;
            Ok(Output::Stream {
                content_type: "text/event-stream".into(),
                producer: Box::new(move |sink| {
                    let mut last = String::new();
                    loop {
                        let running = j.running();
                        let v = j.to_json().to_string();
                        if v != last {
                            if !sink(format!("event: video\ndata: {v}\n\n").as_bytes()) {
                                return;
                            }
                            last = v;
                        } else if !sink(b": waiting\n\n") {
                            return;
                        }
                        if !running {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(500));
                    }
                }),
            })
        }
        [id, "content"] if get => {
            let j = api.videos.find(id)?;
            let variant = req.query_param("variant").unwrap_or_else(|| "video".into());
            let s = lock(&j.st);
            if s.status != "completed" {
                return Err(ApiError::new(
                    409,
                    format!("video {id} is {}: its content is ready once it is completed", s.status),
                )
                .code("video_not_ready"));
            }
            let (bytes, ct, what) = match variant.as_str() {
                "video" => (
                    s.files.video.as_ref().map(|v| &v.0),
                    s.files.video.as_ref().map_or("video/mp4", |v| v.1),
                    "a video file (built without the `media` feature, or expired)",
                ),
                "audio" | "wav" => (
                    s.files.wav.as_ref(),
                    "audio/wav",
                    "a soundtrack (the model has no audio VAE, or it expired)",
                ),
                "thumbnail" | "png" => (s.files.thumbnail.as_ref(), "image/png", "a thumbnail (expired)"),
                v => return Err(ApiError::bad(format!("variant {v:?}: video, audio or thumbnail")).param("variant")),
            };
            let b = bytes.ok_or_else(|| ApiError::new(404, format!("video {id} has no {what}")))?;
            Ok(Output::Full(ApiResponse {
                status: 200,
                content_type: ct.into(),
                body: b.clone(),
            }))
        }
        [_] | [_, "cancel" | "events" | "content"] => Err(wrong()),
        _ => Err(ApiError::new(404, format!("unknown endpoint {}", req.path))),
    }
}
