//! The web GUI's backend ([`super::ServerOptions::web`], `oxinfer web`): the Vue app of `web/`
//! (built into `src/server/webui/` and embedded in the binary) served at `/`, and its management
//! API under `/api/web/`, next to the unchanged OpenAI-compatible `/v1` API.
//!
//! | Endpoint | |
//! |---|---|
//! | `GET /api/web/info` | `{"oxinfer": true, "version", "auth_required", "admin", ...}`; needs no key |
//! | `GET /api/web/status` | the machine, the memory budget, the cache's disk, the models and downloads |
//! | `GET /api/web/catalog` | the tested models ([`crate::advisor`]) with what fits this machine and what is cached, and the presets |
//! | `GET /api/web/presets` | the catalog's presets with whether their base model is downloaded and whether one is listed |
//! | `GET /api/web/cache` | the cache folders ([`crate::model_cache`]) with size, last use, kind, in use |
//! | `POST /api/web/cache/delete` | `{"name"}`: removes a cache folder (refused while in use) |
//! | `GET /api/web/disk` | free and total bytes of the cache's file system |
//! | `POST /api/web/plan` | `{"model"}`: the files a download takes, their sizes, what is missing, whether it fits |
//! | `GET/POST /api/web/downloads` | the download jobs; `{"model"}` starts one (or returns the running one) |
//! | `GET /api/web/downloads/{id}` | one job; `/events` streams it as server-sent events until it ends |
//! | `POST /api/web/downloads/{id}/cancel` | cancels it (the partial file stays, a later download resumes it) |
//! | `GET /api/web/models` | the models clients can use (`/v1/models`), with their load state |
//! | `POST /api/web/models/load` | `{"model", "id", "options"}`: adds the model and loads it in the background |
//! | `POST /api/web/models/unload` | `{"id"}`: unloads it (it stays listed and loads again on use); with `"cancel": true` its requests are cancelled first |
//! | `GET /api/web/activity` | the status page: uptime, requests in flight and recent ones, models with memory and device, process, machine and GPU, prompt caches ([`Api::status`]) |
//! | `GET /api/web/requests` | the requests in flight (running and queued) and the running video jobs |
//! | `POST /api/web/requests/{id}/cancel` | cancels one (also `DELETE /api/web/requests/{id}`): it stops at its next token or step and frees its model |
//! | `GET /api/web/prompt-cache` | the prompt caches: per directory the snapshots by model |
//! | `POST /api/web/prompt-cache/clear` | `{"model"}`, `{"fingerprint"}` or `{}` (all), optionally `"dir"`: removes snapshots |
//! | `POST /api/web/caches/free` | `{"model"}` or `{}`: lets go of the streamed models' cached experts |
//! | `GET /api/web/update` | this version, the latest release, whether it is newer and can be installed from here |
//! | `POST /api/web/update` | installs the latest release over this executable and restarts into it ([`allow_self_update`]) |
//! | `GET /api/web/opencode` | opencode's configuration ([`crate::opencode`]): detected, its file, and per chat model whether it has an entry and whether that is up to date |
//! | `POST /api/web/opencode/add` | `{"model"}`: writes the model's entry (and the `oxinfer` provider) to opencode's configuration |
//! | `POST /api/web/opencode/remove` | `{"model"}`: removes its entry (and the provider when it has no model left) |
//!
//! Every endpoint but `info` needs the API key when one is set. Requests that change anything
//! (POST, DELETE) must be JSON and, when the browser sends an `Origin`, come from the page's own
//! origin; on a loopback listener the `Host` must be a loopback name (against DNS rebinding).
//! Bound to a non-loopback address without an API key, the management API is read-only:
//! downloads, deletions, loads, unloads, cancels, cache clearing and updates are refused with 403
//! (`oxinfer web` warns at start). The opencode endpoints also need the request to come from this
//! machine (a loopback listener or a loopback client): they write to the server user's home.

mod opencode;
mod update;

pub use update::{RESTART_ENV, Restart, allow_self_update};
use update::{Updates, start_update, update_status};

use super::{Api, ApiError, ApiRequest, ApiResponse, ApiResult, ModelSpec, Output};
use crate::server::registry::{Stage, detect_path};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

// `ASSETS`: (path, bytes) of every file of the built GUI (build.rs).
include!(concat!(env!("OUT_DIR"), "/webui_assets.rs"));

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A file of the embedded GUI: its bytes and content type.
pub fn asset(path: &str) -> Option<(&'static [u8], &'static str)> {
    let p = path.trim_start_matches('/');
    let p = if p.is_empty() { "index.html" } else { p };
    let (_, bytes) = ASSETS.iter().find(|(n, _)| *n == p)?;
    let ext = p.rsplit('.').next().unwrap_or_default();
    let ct = match ext {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "json" => "application/json",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    };
    Some((bytes, ct))
}

/// This machine's name (for the page title), without a `.local` / `.localdomain` suffix.
fn hostname() -> Option<String> {
    #[cfg(unix)]
    let name = {
        unsafe extern "C" {
            fn gethostname(name: *mut std::ffi::c_char, len: usize) -> std::ffi::c_int;
        }
        let mut buf = [0u8; 256];
        // SAFETY: the buffer is valid for its length; the name is NUL-terminated within it (or
        // truncated, which the search for the NUL below handles).
        let r = unsafe { gethostname(buf.as_mut_ptr().cast(), buf.len() - 1) };
        (r == 0).then(|| String::from_utf8_lossy(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(0)]).into_owned())
    };
    #[cfg(not(unix))]
    let name = std::env::var("COMPUTERNAME").ok();
    name.map(|n| n.trim_end_matches(".localdomain").trim_end_matches(".local").to_string())
        .filter(|n| !n.is_empty())
}

/// Whether the GUI is embedded in this build (it is unless `src/server/webui` was missing).
pub fn has_gui() -> bool {
    ASSETS.iter().any(|(n, _)| *n == "index.html")
}

/// One file of a download job.
#[derive(Clone, Debug)]
struct FileProg {
    name: String,
    done: u64,
    total: Option<u64>,
}

#[derive(Debug)]
struct JobState {
    /// `planning`, `downloading`, `done`, `error`, `cancelled`.
    status: &'static str,
    files: Vec<FileProg>,
    /// The file being fetched (1-based index of `count`).
    file: String,
    index: usize,
    count: usize,
    /// Bytes on disk when the job started (resumed files, files already whole).
    start_bytes: Option<u64>,
    /// (time, bytes done) over the last seconds, for the speed.
    samples: VecDeque<(Instant, u64)>,
    error: Option<String>,
    path: Option<String>,
    finished: Option<Instant>,
}

/// A download running in the background ([`WebState::start_download`]).
struct Job {
    id: String,
    spec: String,
    /// The cache folder it writes to.
    folder: Option<PathBuf>,
    cancel: AtomicBool,
    started: Instant,
    st: Mutex<JobState>,
}

impl Job {
    fn running(&self) -> bool {
        matches!(lock(&self.st).status, "planning" | "downloading")
    }

    fn to_json(&self) -> Value {
        let s = lock(&self.st);
        let done: u64 = s.files.iter().map(|f| f.done).sum();
        let known: u64 = s.files.iter().map(|f| f.total.unwrap_or(f.done)).sum();
        let total = (!s.files.is_empty() && s.files.iter().all(|f| f.total.is_some())).then_some(known);
        let speed = match (s.samples.front(), s.samples.back()) {
            (Some(a), Some(b)) if b.0 > a.0 && s.finished.is_none() => (b.1.saturating_sub(a.1)) as f64 / (b.0 - a.0).as_secs_f64(),
            _ => 0.0,
        };
        let eta = match total {
            Some(t) if speed > 1.0 && t > done => Some(((t - done) as f64 / speed).round()),
            _ => None,
        };
        let elapsed = s.finished.unwrap_or_else(Instant::now).duration_since(self.started).as_secs_f64();
        json!({
            "id": self.id,
            "model": self.spec,
            "status": s.status,
            "file": s.file,
            "index": s.index,
            "count": s.count,
            "done": done,
            "total": total,
            "known": known,
            "downloaded": done.saturating_sub(s.start_bytes.unwrap_or(done)),
            "speed": speed.round(),
            "eta": eta,
            "elapsed": (elapsed * 10.0).round() / 10.0,
            "error": s.error,
            "path": s.path,
            "files": s.files.iter().map(|f| json!({"name": f.name, "done": f.done, "total": f.total})).collect::<Vec<_>>(),
        })
    }

    /// A progress report of the fetch.
    #[cfg(feature = "hf")]
    fn progress(&self, p: &crate::hf::Progress) -> bool {
        let mut s = lock(&self.st);
        s.status = "downloading";
        if p.index > 0 {
            s.file = p.file.to_string();
            s.index = p.index;
            s.count = p.count;
        }
        match s.files.iter_mut().find(|f| f.name == p.file) {
            Some(f) => {
                f.done = p.done;
                if p.total.is_some() {
                    f.total = p.total;
                }
            }
            // A file the plan did not know (a component another loader fetches): counted from now.
            None if p.index > 0 => s.files.push(FileProg {
                name: p.file.to_string(),
                done: p.done,
                total: p.total,
            }),
            None => {}
        }
        let done: u64 = s.files.iter().map(|f| f.done).sum();
        if s.start_bytes.is_none() {
            s.start_bytes = Some(done);
        }
        let now = Instant::now();
        if s.samples
            .back()
            .is_none_or(|(t, _)| now.duration_since(*t) >= Duration::from_millis(200))
        {
            s.samples.push_back((now, done));
        }
        while s.samples.len() > 2
            && s.samples
                .front()
                .is_some_and(|(t, _)| now.duration_since(*t) > Duration::from_secs(6))
        {
            s.samples.pop_front();
        }
        !self.cancel.load(Ordering::Relaxed)
    }

    fn finish(&self, r: anyhow::Result<PathBuf>) {
        let mut s = lock(&self.st);
        s.finished = Some(Instant::now());
        match r {
            Ok(p) => {
                s.status = "done";
                s.path = Some(p.display().to_string());
                for f in &mut s.files {
                    if let Some(t) = f.total {
                        f.done = t;
                    }
                }
            }
            Err(_) if self.cancel.load(Ordering::Relaxed) => s.status = "cancelled",
            Err(e) => {
                s.status = "error";
                s.error = Some(format!("{e:#}"));
            }
        }
    }
}

/// A model load started from the GUI ([`WebState::start_load`]).
struct LoadJob {
    id: String,
    cancel: AtomicBool,
    started: Instant,
    /// `queued`, `waiting`, `downloading`, `loading`, `ready`, `error`, `cancelled`, and the
    /// stage's details.
    st: Mutex<Value>,
}

/// The web GUI's state in [`Api`].
pub(crate) struct WebState {
    jobs: Mutex<Vec<Arc<Job>>>,
    loads: Mutex<Vec<Arc<LoadJob>>>,
    next: AtomicU64,
    /// Destructive endpoints are allowed: a loopback listener, or an API key.
    admin: bool,
    loopback: bool,
    platform: OnceLock<crate::platform::Platform>,
    /// The CPU kernel rates of the catalog's speed estimates, once measured (in the background).
    rates: Mutex<Option<[f64; 7]>>,
    calibrating: AtomicBool,
    /// The release check of `GET /api/web/update` ([`super::ServerOptions::update_check`]).
    updates: Updates,
}

impl WebState {
    pub fn new(opts: &super::ServerOptions) -> Self {
        let loopback = std::net::ToSocketAddrs::to_socket_addrs(&opts.listen)
            .ok()
            .and_then(|mut a| a.next())
            .is_some_and(|a| a.ip().is_loopback());
        WebState {
            jobs: Mutex::new(Vec::new()),
            loads: Mutex::new(Vec::new()),
            next: AtomicU64::new(1),
            admin: loopback || opts.api_key.is_some(),
            loopback,
            platform: OnceLock::new(),
            rates: Mutex::new(None),
            calibrating: AtomicBool::new(false),
            updates: Updates::new(opts),
        }
    }

    /// Whether the management API is read-only (a non-loopback listener without an API key).
    pub fn read_only(&self) -> bool {
        !self.admin
    }

    fn platform(&self) -> &crate::platform::Platform {
        self.platform.get_or_init(crate::platform::detect)
    }

    fn new_id(&self, prefix: &str) -> String {
        format!("{prefix}{}", self.next.fetch_add(1, Ordering::Relaxed))
    }

    /// A download job is writing to `folder`.
    fn downloading(&self, folder: &Path) -> bool {
        let real = std::fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf());
        lock(&self.jobs).iter().any(|j| {
            j.running()
                && j.folder
                    .as_ref()
                    .is_some_and(|f| std::fs::canonicalize(f).unwrap_or_else(|_| f.clone()) == real)
        })
    }
}

fn json_ok(v: Value) -> ApiResult<Output> {
    Ok(Output::Full(ApiResponse::json(200, &v)))
}

fn body(req: &ApiRequest) -> ApiResult<Value> {
    if req.body.is_empty() {
        return Ok(json!({}));
    }
    super::json_body(req)
}

fn str_field<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|k| v.get(*k).and_then(Value::as_str))
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// The host part of a `Host` header or of an origin's authority (`[::1]:8080` gives `::1`).
pub(crate) fn host_name(authority: &str) -> &str {
    let a = authority.trim();
    if let Some(rest) = a.strip_prefix('[') {
        return rest.split(']').next().unwrap_or_default();
    }
    a.rsplit_once(':')
        .filter(|(_, p)| p.bytes().all(|b| b.is_ascii_digit()))
        .map_or(a, |(h, _)| h)
}

pub(crate) fn loopback_name(host: &str) -> bool {
    let h = host.to_ascii_lowercase();
    h == "localhost" || h.ends_with(".localhost") || h.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Serves the GUI's files and the management API; None for every other request (`/v1`).
pub(crate) fn route(api: &Arc<Api>, w: &WebState, req: &ApiRequest) -> Option<ApiResult<Output>> {
    let path = req.path.as_str();
    let get = req.method.eq_ignore_ascii_case("GET") || req.method.eq_ignore_ascii_case("HEAD");
    if let Some(rest) = path.strip_prefix("/api/web/").or(if path == "/api/web" { Some("") } else { None }) {
        return Some(management(api, w, req, rest.trim_end_matches('/')));
    }
    if get && !path.starts_with("/v1/") && path != "/health" {
        let p = if path == "/" || path.is_empty() { "/index.html" } else { path };
        if let Some((bytes, ct)) = asset(p) {
            return Some(Ok(Output::Full(ApiResponse {
                status: 200,
                content_type: ct.to_string(),
                body: bytes.to_vec(),
            })));
        }
        if p == "/index.html" {
            let msg = "this build has no web GUI (src/server/webui was empty when it was built; run `npm run build:native` in web/)";
            return Some(Err(ApiError::new(404, msg)));
        }
    }
    None
}

fn management(api: &Arc<Api>, w: &WebState, req: &ApiRequest, rest: &str) -> ApiResult<Output> {
    let get = req.method.eq_ignore_ascii_case("GET") || req.method.eq_ignore_ascii_case("HEAD");
    let post = req.method.eq_ignore_ascii_case("POST");
    let delete = req.method.eq_ignore_ascii_case("DELETE");
    // DNS rebinding: a page of another site whose name resolves to 127.0.0.1 sends its own Host.
    if w.loopback
        && let Some(h) = &req.host
        && !loopback_name(host_name(h))
    {
        return Err(ApiError::new(
            403,
            format!("Host {h:?} is not a loopback name; open the GUI at http://localhost or 127.0.0.1"),
        ));
    }
    if rest == "info" && get {
        return json_ok(json!({
            "oxinfer": true,
            "native": true,
            "version": env!("CARGO_PKG_VERSION"),
            "auth_required": api.opts.api_key.is_some(),
            "admin": w.admin,
            "api_base": "/v1",
            "hostname": hostname(),
            "features": {"hf": cfg!(feature = "hf"), "media": cfg!(feature = "media")},
        }));
    }
    api.check_auth(req)?;
    if post || delete {
        // Cross-site requests: a form or a text/plain fetch cannot set a JSON content type
        // without a CORS preflight, which is not answered; an Origin must be the page's own.
        let json_ct = req.content_type.as_deref().is_some_and(|c| c.starts_with("application/json"));
        if !json_ct && !req.body.is_empty() {
            return Err(ApiError::new(
                415,
                "the management API takes JSON bodies (Content-Type: application/json)",
            ));
        }
        if !json_ct && req.origin.is_some() {
            return Err(ApiError::new(415, "the management API needs Content-Type: application/json"));
        }
        if let Some(origin) = &req.origin {
            let authority = origin.split_once("://").map(|(_, a)| a).unwrap_or(origin.as_str());
            let same = req.host.as_deref().is_some_and(|h| h.eq_ignore_ascii_case(authority));
            if !same {
                return Err(ApiError::new(403, format!("cross-origin request from {origin} refused")));
            }
        }
        if !w.admin {
            return Err(ApiError::new(
                403,
                "the management API is read-only: the server listens on a non-loopback address without an API key (start it with --api-key KEY)",
            )
            .code("read_only"));
        }
    }
    let cache_dir = api.models.cache_dir();
    let segs: Vec<&str> = rest.split('/').collect();
    match (segs.as_slice(), get, post, delete) {
        (["status"], true, _, _) => json_ok(status(api, w)),
        (["catalog"], true, _, _) => json_ok(catalog(api, w)),
        (["presets"], true, _, _) => json_ok(json!({"data": presets(api)})),
        (["disk"], true, _, _) => {
            let d = crate::model_cache::disk_space(cache_dir);
            json_ok(
                json!({"path": crate::model_cache::root(cache_dir).display().to_string(), "free": d.map(|d| d.0), "total": d.map(|d| d.1)}),
            )
        }
        (["cache"], true, _, _) => json_ok(cache(api, w)?),
        (["cache", "delete"], _, true, _) => {
            let b = body(req)?;
            let name = str_field(&b, &["name", "model"]).ok_or_else(|| ApiError::bad("cache/delete: no \"name\"").param("name"))?;
            delete_entry(api, w, name)
        }
        (["cache", name], _, _, true) => delete_entry(api, w, name),
        (["plan"], _, true, _) => plan(api, &body(req)?),
        (["downloads"], true, _, _) => {
            let jobs: Vec<Value> = lock(&w.jobs).iter().map(|j| j.to_json()).collect();
            json_ok(json!({"data": jobs}))
        }
        (["downloads"], _, true, _) => {
            let b = body(req)?;
            let spec = str_field(&b, &["model", "spec"]).ok_or_else(|| ApiError::bad("downloads: no \"model\"").param("model"))?;
            // A preset (`kromoss-dmd2`) downloads its base model; its LoRAs come with the load.
            let j = start_download(api, w, base_of(spec))?;
            json_ok(j.to_json())
        }
        (["downloads", id], true, _, _) => json_ok(find_job(w, id)?.to_json()),
        (["downloads", id, "events"], true, _, _) => {
            let j = find_job(w, id)?;
            Ok(Output::Stream {
                content_type: "text/event-stream".into(),
                producer: Box::new(move |sink| {
                    loop {
                        let running = j.running();
                        let v = j.to_json();
                        if !sink(format!("data: {v}\n\n").as_bytes()) || !running {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(500));
                    }
                }),
            })
        }
        (["downloads", id, "cancel"], _, true, _) | (["downloads", id], _, _, true) => {
            let j = find_job(w, id)?;
            j.cancel.store(true, Ordering::Relaxed);
            json_ok(j.to_json())
        }
        (["update"], true, _, _) => json_ok(update_status(api, w)),
        (["update"], _, true, _) => start_update(api, w),
        (["opencode"], true, _, _) => json_ok(opencode::status(api, w, req)),
        (["opencode", "add"], _, true, _) => opencode::change(api, w, req, &body(req)?, true),
        (["opencode", "remove"], _, true, _) => opencode::change(api, w, req, &body(req)?, false),
        (["models"], true, _, _) => json_ok(models(api, w)),
        (["activity"], true, _, _) => json_ok(api.status()),
        (["requests"], true, _, _) => {
            let data: Vec<Value> = api.requests().iter().map(super::RequestInfo::to_json).collect();
            json_ok(json!({"data": data, "videos": api.videos.running()}))
        }
        (["requests", id, "cancel"], _, true, _) | (["requests", id], _, _, true) => {
            if !api.cancel(id) {
                return Err(ApiError::new(404, format!("no request {id:?} in flight")).code("request_not_found"));
            }
            if api.opts.log {
                eprintln!("[cancelled {id}]");
            }
            json_ok(json!({"id": id, "cancelled": true}))
        }
        (["prompt-cache"], true, _, _) => json_ok(super::status::prompt_cache_json(api, &api.models.status())),
        (["prompt-cache", "clear"], _, true, _) => {
            let b = body(req)?;
            let r = super::status::clear_prompt_cache(
                api,
                str_field(&b, &["model", "id"]),
                str_field(&b, &["fingerprint"]),
                str_field(&b, &["dir"]),
            )?;
            json_ok(json!({"removed": r.entries, "bytes": r.bytes}))
        }
        (["caches", "free"], _, true, _) => {
            let b = body(req)?;
            let freed = super::status::free_caches(api, str_field(&b, &["model", "id"]))?;
            json_ok(json!({"freed": freed}))
        }
        (["models", "load"], _, true, _) => load(api, w, &body(req)?),
        (["models", "unload"], _, true, _) => {
            let b = body(req)?;
            let id = str_field(&b, &["id", "model"]).ok_or_else(|| ApiError::bad("models/unload: no \"id\"").param("id"))?;
            // A load in progress is cancelled where it reports (downloads, waits); a model being
            // read into memory is unloaded once it is in.
            let loading = lock(&w.loads).iter().rev().find(|l| l.id == id).cloned().filter(|l| {
                matches!(
                    lock(&l.st)["state"].as_str(),
                    Some("queued" | "waiting" | "downloading" | "loading")
                )
            });
            if let Some(l) = loading {
                l.cancel.store(true, Ordering::Relaxed);
                return json_ok(json!({"id": id, "unloaded": false, "cancelling": true}));
            }
            // In use: its requests cancelled, unloaded once they have stopped.
            if b.get("cancel").and_then(Value::as_bool) == Some(true) {
                let n = api.activity.cancel_model(id);
                let was = api.models.unload_waiting(id, Duration::from_secs(10))?;
                return json_ok(json!({"id": id, "unloaded": was, "cancelled": n}));
            }
            let was = api.models.unload(id)?;
            json_ok(json!({"id": id, "unloaded": was}))
        }
        _ => Err(ApiError::new(404, format!("unknown endpoint {} {}", req.method, req.path))),
    }
}

fn find_job(w: &WebState, id: &str) -> ApiResult<Arc<Job>> {
    lock(&w.jobs)
        .iter()
        .find(|j| j.id == id)
        .cloned()
        .ok_or_else(|| ApiError::new(404, format!("no download {id:?}")))
}

/// The family of the model in `path` when it answers in speech (`lfm3_omni`, `haru`): the GUI's voice chat.
fn speaks(path: &Path) -> Option<&'static str> {
    if crate::lfm3_omni::is_omni(path) {
        Some("lfm3_omni")
    } else if crate::haru::is_haru(path) {
        Some("haru")
    } else {
        None
    }
}

/// The kind of what is in a cache folder, as the server would detect it.
fn cache_kind(e: &crate::model_cache::Entry) -> Option<&'static str> {
    if !e.usable {
        return None;
    }
    let path = match e.ggufs.first() {
        Some(g) if !e.path.join("config.json").is_file() => e.path.join(g),
        _ => e.path.clone(),
    };
    Some(detect_path(&path, e.repo.as_deref().unwrap_or(&e.name)).0.as_str())
}

fn cache(api: &Api, w: &WebState) -> ApiResult<Value> {
    let cache_dir = api.models.cache_dir();
    let mut v = crate::model_cache::to_json(cache_dir)?;
    let entries = crate::model_cache::list(cache_dir)?;
    let catalog: Vec<&str> = crate::advisor::CATALOG.iter().map(|m| m.repo).collect();
    if let Some(list) = v["entries"].as_array_mut() {
        for (j, e) in list.iter_mut().zip(&entries) {
            let in_catalog: Vec<&&str> = catalog
                .iter()
                .filter(|c| e.repo.as_deref().is_some_and(|r| c.split(':').next() == Some(r)))
                .collect();
            j["catalog"] = json!(in_catalog);
            j["kind"] = json!(cache_kind(e));
            j["audio_out"] = json!(speaks(&e.path).filter(|_| e.usable));
            j["loaded"] = json!(api.models.uses_folder(&e.path));
            j["downloading"] = json!(w.downloading(&e.path));
        }
    }
    Ok(v)
}

fn delete_entry(api: &Api, w: &WebState, name: &str) -> ApiResult<Output> {
    let cache_dir = api.models.cache_dir();
    let path = crate::model_cache::locate(name, cache_dir).map_err(|e| {
        let msg = format!("{e:#}");
        ApiError::new(if msg.contains("not in the cache") { 404 } else { 400 }, msg).param("name")
    })?;
    if api.models.uses_folder(&path) {
        return Err(ApiError::new(409, format!("{name} is loaded or in use: unload it first")).code("in_use"));
    }
    if w.downloading(&path) {
        return Err(ApiError::new(409, format!("{name} is being downloaded: cancel the download first")).code("in_use"));
    }
    let freed = crate::model_cache::remove(name, cache_dir).map_err(ApiError::internal)?;
    if api.opts.log {
        eprintln!("[removed {} from the cache ({:.2} GB)]", path.display(), freed as f64 / 1e9);
    }
    json_ok(json!({"name": name, "removed": true, "freed": freed}))
}

/// The model a spec downloads: a preset's base model, else the spec itself.
fn base_of(spec: &str) -> &str {
    match crate::advisor::preset(spec) {
        Some(p) if !Path::new(spec).exists() => p.base,
        _ => spec,
    }
}

/// The catalog's presets ([`crate::advisor::PRESETS`]) with whether their base model is
/// downloaded and whether one is listed under the preset's id. The ones the
/// server has no route for (LongCat-Video-Avatar's) are left out.
fn presets(api: &Api) -> Value {
    let entries = crate::model_cache::list(api.models.cache_dir()).unwrap_or_default();
    let listed: Vec<String> = api.models.status().into_iter().map(|s| s.spec.id).collect();
    let mut v = crate::advisor::presets_json();
    if let Some(list) = v.as_array_mut() {
        list.retain(|p| p["pipeline"] != crate::longcat_avatar::PIPELINE);
        for p in list {
            let base = p["base"].as_str().unwrap_or_default().to_string();
            p["base_cached"] = json!(entries.iter().any(|e| e.usable && e.repo.as_deref() == Some(base.as_str())));
            p["listed"] = json!(listed.iter().any(|id| p["id"] == id.as_str()));
        }
    }
    v
}

fn plan(api: &Api, b: &Value) -> ApiResult<Output> {
    let spec = str_field(b, &["model", "spec"]).ok_or_else(|| ApiError::bad("plan: no \"model\"").param("model"))?;
    let shown = spec;
    let spec = base_of(spec);
    let cache_dir = api.models.cache_dir();
    let disk = crate::model_cache::disk_space(cache_dir);
    if Path::new(spec).exists() {
        return json_ok(
            json!({"model": spec, "local": true, "total": 0, "missing": 0, "free": disk.map(|d| d.0), "fits": true, "files": []}),
        );
    }
    #[cfg(feature = "hf")]
    {
        let p = crate::hf::plan(spec, cache_dir).map_err(|e| ApiError::bad(format!("{e:#}")).param("model"))?;
        let mut v = p.to_json();
        v["model"] = json!(shown);
        v["base"] = json!(spec);
        v["local"] = json!(false);
        v["free"] = json!(disk.map(|d| d.0));
        v["fits"] = json!(disk.is_none_or(|d| d.0 >= p.missing()));
        json_ok(v)
    }
    #[cfg(not(feature = "hf"))]
    Err(ApiError::bad(format!(
        "{shown}: not a local path, and this build cannot fetch models (the `hf` feature)"
    )))
}

fn start_download(api: &Api, w: &WebState, spec: &str) -> ApiResult<Arc<Job>> {
    if let Some(j) = lock(&w.jobs).iter().find(|j| j.spec == spec && j.running()) {
        return Ok(j.clone());
    }
    if Path::new(spec).exists() {
        return Err(ApiError::bad(format!("{spec} is a local path: nothing to download")).param("model"));
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = (api, w);
        Err(ApiError::bad("this build cannot fetch models (the `hf` feature)"))
    }
    #[cfg(feature = "hf")]
    {
        let folder_name = crate::model_cache::folder_name(spec).map_err(|e| ApiError::bad(format!("{e:#}")).param("model"))?;
        let cache_dir = api.models.cache_dir().map(Path::to_path_buf);
        let job = Arc::new(Job {
            id: w.new_id("dl-"),
            spec: spec.to_string(),
            folder: Some(crate::model_cache::root(cache_dir.as_deref()).join(folder_name)),
            cancel: AtomicBool::new(false),
            started: Instant::now(),
            st: Mutex::new(JobState {
                status: "planning",
                files: Vec::new(),
                file: String::new(),
                index: 0,
                count: 0,
                start_bytes: None,
                samples: VecDeque::new(),
                error: None,
                path: None,
                finished: None,
            }),
        });
        {
            let mut jobs = lock(&w.jobs);
            // Finished jobs are kept for the GUI, the last 50.
            if jobs.len() >= 50
                && let Some(i) = jobs.iter().position(|j| !j.running())
            {
                jobs.remove(i);
            }
            jobs.push(job.clone());
        }
        let j = job.clone();
        let log = api.opts.log;
        std::thread::Builder::new()
            .name("oxinfer-download".into())
            .spawn(move || {
                if log {
                    eprintln!("[download {}: {}]", j.id, j.spec);
                }
                // The sizes up front (a failure only leaves the totals unknown).
                if let Ok(p) = crate::hf::plan(&j.spec, cache_dir.as_deref()) {
                    let mut s = lock(&j.st);
                    s.count = p.files.len();
                    s.files = p
                        .files
                        .iter()
                        .map(|f| FileProg {
                            name: f.name.clone(),
                            done: f.have,
                            total: f.size,
                        })
                        .collect();
                    s.start_bytes = Some(p.files.iter().map(|f| f.have).sum());
                }
                let r = if j.cancel.load(Ordering::Relaxed) {
                    Err(anyhow::anyhow!("download cancelled"))
                } else {
                    let mut hook = |p: &crate::hf::Progress| j.progress(p);
                    crate::hf::with_progress(&mut hook, || crate::engine::resolve_model(&j.spec, cache_dir.as_deref()))
                };
                if log {
                    match &r {
                        Ok(p) => eprintln!("[download {}: done, {}]", j.id, p.display()),
                        Err(e) => eprintln!("[download {}: {e:#}]", j.id),
                    }
                }
                j.finish(r);
            })
            .map_err(|e| ApiError::internal(e.into()))?;
        Ok(job)
    }
}

/// `options` of a load: an object of values, or `KEY=VALUE,...`.
fn load_options(v: Option<&Value>) -> ApiResult<Vec<(String, String)>> {
    let bad = |m: String| ApiError::bad(m).param("options");
    Ok(match v {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => s
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|kv| {
                kv.split_once('=')
                    .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                    .ok_or_else(|| bad(format!("option {kv:?} is not KEY=VALUE")))
            })
            .collect::<ApiResult<_>>()?,
        Some(Value::Object(o)) => o
            .iter()
            .filter(|(_, v)| !v.is_null() && v.as_str() != Some(""))
            .map(|(k, v)| {
                let v = match v {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => (if *b { "1" } else { "0" }).to_string(),
                    other => other.to_string(),
                };
                (k.clone(), v)
            })
            .collect(),
        Some(_) => return Err(bad("options: expected an object or KEY=VALUE,...".into())),
    })
}

fn load(api: &Arc<Api>, w: &WebState, b: &Value) -> ApiResult<Output> {
    let source = str_field(b, &["model", "spec", "source"]).ok_or_else(|| ApiError::bad("models/load: no \"model\"").param("model"))?;
    let mut spec = ModelSpec::new(source).map_err(|e| ApiError::bad(format!("{e:#}")))?;
    if let Some(id) = str_field(b, &["id"]) {
        spec.id = id.to_string();
    } else if Path::new(source).is_absolute() {
        // A path: the folder or file name is a friendlier id.
        let p = Path::new(source);
        let name = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| source.to_string());
        spec.id = name;
    }
    spec.options = load_options(b.get("options"))?;
    spec.check().map_err(|e| ApiError::bad(format!("{e:#}")).param("options"))?;
    let id = spec.id.clone();
    api.models.add(spec)?;
    let already = api.models.status().into_iter().any(|s| s.spec.id == id && s.loaded);
    let job = Arc::new(LoadJob {
        id: id.clone(),
        cancel: AtomicBool::new(false),
        started: Instant::now(),
        st: Mutex::new(json!({"state": if already { "ready" } else { "queued" }})),
    });
    {
        let mut loads = lock(&w.loads);
        loads.retain(|l| l.id != id);
        loads.push(job.clone());
    }
    if !already {
        let api2 = api.clone();
        let j = job.clone();
        std::thread::Builder::new()
            .name("oxinfer-load".into())
            .spawn(move || {
                let mut last = Instant::now() - Duration::from_secs(1);
                let mut on = |s: &Stage| {
                    let v = match s {
                        Stage::Waiting => json!({"state": "waiting"}),
                        Stage::Loading => json!({"state": "loading"}),
                        Stage::Downloading {
                            file,
                            index,
                            count,
                            done,
                            total,
                        } => {
                            // At most five updates a second.
                            if last.elapsed() < Duration::from_millis(200) && total.is_none_or(|t| *done < t) {
                                return !j.cancel.load(Ordering::Relaxed);
                            }
                            last = Instant::now();
                            json!({"state": "downloading", "file": file, "index": index, "count": count, "done": done, "total": total})
                        }
                        Stage::Prompt { .. } => return true,
                    };
                    *lock(&j.st) = v;
                    !j.cancel.load(Ordering::Relaxed)
                };
                let r = api2.models.load_now(&j.id, &mut on);
                let secs = (j.started.elapsed().as_secs_f64() * 10.0).round() / 10.0;
                *lock(&j.st) = match r {
                    // Cancelled while the weights were read in: unloaded again.
                    Ok(_) if j.cancel.load(Ordering::Relaxed) => {
                        let _ = api2.models.unload(&j.id);
                        json!({"state": "cancelled"})
                    }
                    Ok(kind) => json!({"state": "ready", "kind": kind.as_str(), "seconds": secs}),
                    Err(_) if j.cancel.load(Ordering::Relaxed) => json!({"state": "cancelled"}),
                    Err(e) => json!({"state": "error", "error": e.message}),
                };
            })
            .map_err(|e| ApiError::internal(e.into()))?;
    }
    json_ok(json!({"id": id, "load": *lock(&job.st)}))
}

fn models(api: &Api, w: &WebState) -> Value {
    let loads = lock(&w.loads);
    let data: Vec<Value> = api
        .models
        .status()
        .into_iter()
        .map(|s| {
            let load = loads.iter().rev().find(|l| l.id == s.spec.id).map(|l| lock(&l.st).clone());
            let state = if s.loaded {
                if s.busy { "busy" } else { "ready" }
            } else {
                match load.as_ref().and_then(|l| l["state"].as_str()) {
                    Some(st @ ("queued" | "waiting" | "downloading" | "loading" | "error")) => st,
                    _ if s.busy => "loading",
                    _ => "unloaded",
                }
            };
            json!({
                "id": s.spec.id,
                "source": s.spec.source,
                "options": s.spec.options.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
                "kind": s.kind.map(super::Kind::as_str),
                "loaded": s.loaded,
                "busy": s.busy,
                "memory": s.bytes,
                "audio_out": api.models.local_path(&s.spec.id).and_then(|p| speaks(&p)),
                "state": state,
                "load": load,
            })
        })
        .collect();
    json!({"data": data})
}

fn status(api: &Api, w: &WebState) -> Value {
    let plat = w.platform();
    let cache_dir = api.models.cache_dir();
    let disk = crate::model_cache::disk_space(cache_dir);
    let downloads: Vec<Value> = lock(&w.jobs).iter().map(|j| j.to_json()).collect();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "platform": plat.summary(),
        "cores": plat.cpu.cores,
        "memory": plat.cpu.memory_bytes,
        "memory_available": crate::platform::available_memory(),
        "memory_budget": api.models.budget(),
        "gpus": plat.gpus.iter().map(|g| json!({"backend": g.kind.as_str(), "name": g.name, "memory": g.memory_bytes, "unified": g.unified_memory})).collect::<Vec<_>>(),
        "accelerators": crate::advisor::accelerators(plat, crate::platform::DevicePref::Auto).iter().map(crate::advisor::Accelerator::to_json).collect::<Vec<_>>(),
        "cache": {"path": crate::model_cache::root(cache_dir).display().to_string(), "free": disk.map(|d| d.0), "total": disk.map(|d| d.1)},
        "admin": w.admin,
        "listen": api.opts.listen,
        "loopback": w.loopback,
        "auth_required": api.opts.api_key.is_some(),
        "models": models(api, w)["data"],
        "downloads": downloads,
    })
}

fn catalog(api: &Arc<Api>, w: &WebState) -> Value {
    let cache_dir = api.models.cache_dir();
    let mut hw = crate::advisor::Hardware::from_platform(w.platform(), crate::platform::DevicePref::Auto, cache_dir);
    let rates = *lock(&w.rates);
    match rates {
        Some(r) => hw.cpu_rates = r,
        // Timing the kernels takes 2 to 3 seconds: once, in the background.
        None => calibrate_later(api),
    }
    let rows = crate::advisor::survey(&hw);
    let mut v = crate::advisor::to_json(&hw, &rows);
    let entries = crate::model_cache::list(cache_dir).unwrap_or_default();
    if let Some(list) = v["models"].as_array_mut() {
        for (m, row) in list.iter_mut().zip(&rows) {
            let repo = row.model.repo;
            let (base, quant) = match repo.split_once(':') {
                Some((b, q)) => (b, Some(q)),
                None => (repo, None),
            };
            let e = entries.iter().find(|e| e.repo.as_deref() == Some(base));
            let cached = e.is_some_and(|e| match quant {
                Some(q) => e
                    .ggufs
                    .iter()
                    .any(|g| crate::model_cache::gguf_quant(g).is_some_and(|x| x.eq_ignore_ascii_case(q))),
                None => e.usable,
            });
            m["cached"] = json!(cached);
            m["cache_bytes"] = json!(e.map(|e| e.bytes));
            // (LongCat-Video-Avatar is in the catalog but has no route here.)
            let served = row.model.repo != crate::longcat_avatar::AVATAR_REPO;
            m["servable"] = json!(served);
            m["server_kind"] = json!(match row.model.kind {
                _ if !served => None,
                crate::advisor::Kind::Chat => Some("chat"),
                crate::advisor::Kind::Embedding => Some("embedding"),
                crate::advisor::Kind::Image => Some("image"),
                crate::advisor::Kind::Speech => Some("transcription"),
                crate::advisor::Kind::Tts => Some("speech"),
                crate::advisor::Kind::Video => Some("video"),
                crate::advisor::Kind::Audio => Some("isolation"),
                crate::advisor::Kind::Depth => Some("depth"),
            });
        }
    }
    v["calibrated"] = json!(rates.is_some());
    v["presets"] = presets(api);
    v
}

/// Starts timing the CPU kernels for the catalog's speed estimates, once (a background thread).
pub(crate) fn calibrate_later(api: &Arc<Api>) {
    let Some(w) = &api.web else { return };
    if w.calibrating.swap(true, Ordering::Relaxed) {
        return;
    }
    let api = api.clone();
    let _ = std::thread::Builder::new().name("oxinfer-calibrate".into()).spawn(move || {
        let r = crate::advisor::calibrate();
        if let Some(w) = &api.web {
            *lock(&w.rates) = Some(r);
        }
    });
}
