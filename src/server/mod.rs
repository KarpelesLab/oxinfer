//! An OpenAI-compatible API over the models oxinfer runs (feature `server`): chat completions
//! (text, images, video and audio input, tools, thinking, streaming), legacy completions,
//! embeddings, image generation and editing, speech to text (Whisper) and text to speech
//! (Orpheus).
//!
//! [`Api`] is the request handling without any transport: an HTTP-like request in
//! ([`ApiRequest`]: method, path, content type, body), a response out ([`Output`]: a complete body,
//! or a stream written through a callback, for server-sent events). [`http`] serves it over
//! HTTP(S) with [httpsd](https://crates.io/crates/httpsd); the C API (`oxi_server_*`) exposes both.
//!
//! Only the models of the whitelist ([`ServerOptions::models`], [`ModelSpec`]) can be requested,
//! by their id. Each is a local path, a repo id (`org/name`, `org/name:QUANT` for a GGUF repo;
//! fetched into the cache on first use, or at start with [`ServerOptions::download`]) with an
//! optional alias and per-model options (the [`crate::engine::EngineOptions`] keys for chat
//! models, the loader options of the other kinds, plus sampling and conversation defaults).
//! Models load on first use (or at start with [`ServerOptions::preload`]); each runs one request
//! at a time (others wait their turn), and when a model to load would not fit in
//! [`ServerOptions::memory_budget`] next to the loaded ones, the least recently used idle models
//! are unloaded first. Chat requests that extend the previous conversation of a model reuse its
//! cache ([`crate::engine::Engine::chat`]).
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use oxinfer::server::{Api, ApiRequest, Output, ServerOptions};
//! let mut opts = ServerOptions::default();
//! opts.set("model", "qwen=Qwen/Qwen3-0.6B,quant=q8")?;
//! let api = Api::new(opts)?;
//! let req = ApiRequest::post_json(
//!     "/v1/chat/completions",
//!     r#"{"model": "qwen", "messages": [{"role": "user", "content": "Hi!"}]}"#,
//! );
//! if let Output::Full(r) = api.handle(&req) {
//!     println!("{} {}", r.status, String::from_utf8_lossy(&r.body));
//! }
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod http;
mod media;
mod openai;
mod registry;

pub use registry::Kind;

/// One model of the whitelist: `[ID=]SOURCE[,KEY=VALUE...]`, e.g. `qwen=Qwen/Qwen3-0.6B,quant=q8`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ModelSpec {
    /// The name clients use (`model` in requests). Default: the source.
    pub id: String,
    /// A local directory or file, a repo id (`org/name`), or `org/name:QUANT` (GGUF).
    pub source: String,
    /// The model's options in order (see [`ModelSpec::KEYS`] and the loaders' own keys).
    pub options: Vec<(String, String)>,
}

impl ModelSpec {
    /// Server-side keys of a model, besides those of its loader (for chat models,
    /// [`crate::engine::EngineOptions::KEYS`]):
    /// - `kind`: `chat`, `embedding`, `image`, `transcription` or `speech` (default: detected);
    /// - `system`: the system prompt when a request has none (default none; `default` for the
    ///   CLI's prompt with the local date and time);
    /// - `think`: thinking by default (`1` / `0`; requests override it);
    /// - `max_tokens`: reply length when a request sets none (default 4096);
    /// - `temperature`, `top_p`, `top_k`, `repetition_penalty`: sampling defaults (OpenAI's:
    ///   temperature 1, top-p 1, no top-k);
    /// - `prompt_name`: embedding models, the named prompt used when a request names none;
    /// - `memory_gb`: the memory the model is counted for (default: estimated from its files).
    pub const KEYS: &'static [&'static str] = &[
        "kind",
        "system",
        "think",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "prompt_name",
        "memory_gb",
    ];

    /// Parses `[ID=]SOURCE[,KEY=VALUE...]`.
    pub fn parse(s: &str) -> Result<Self> {
        let mut items = s.split(',').map(str::trim);
        let head = items.next().unwrap_or_default();
        let (id, source) = match head.split_once('=') {
            Some((id, src)) => (id.trim().to_string(), src.trim().to_string()),
            None => (String::new(), head.to_string()),
        };
        if source.is_empty() {
            bail!("model {s:?}: no source (expected [ID=]SOURCE[,KEY=VALUE...])");
        }
        let mut options = Vec::new();
        for item in items.filter(|i| !i.is_empty()) {
            let (k, v) = item
                .split_once('=')
                .with_context(|| format!("model {s:?}: option {item:?} is not KEY=VALUE"))?;
            options.push((k.trim().to_string(), v.trim().to_string()));
        }
        let mut spec = Self::new(&source)?;
        if !id.is_empty() {
            spec.id = id;
        }
        spec.options = options;
        spec.check()?;
        Ok(spec)
    }

    /// A model with its source as id and no options.
    pub fn new(source: &str) -> Result<Self> {
        if source.is_empty() {
            bail!("empty model source");
        }
        Ok(Self {
            id: source.to_string(),
            source: source.to_string(),
            options: Vec::new(),
        })
    }

    /// From a configuration file entry: a spec string, or `{"id": ..., "model": ..., "options":
    /// {KEY: VALUE, ...}}` (values as strings, numbers or booleans).
    pub fn from_json(v: &Value) -> Result<Self> {
        if let Some(s) = v.as_str() {
            return Self::parse(s);
        }
        let o = v.as_object().context("model: expected a string or an object")?;
        let source = o
            .get("model")
            .or_else(|| o.get("source"))
            .and_then(Value::as_str)
            .context("model: no \"model\" (the source)")?;
        let mut spec = Self::new(source)?;
        if let Some(id) = o.get("id").and_then(Value::as_str) {
            spec.id = id.to_string();
        }
        if let Some(opts) = o.get("options") {
            for (k, v) in opts.as_object().context("model options: expected an object")? {
                let v = match v {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => (if *b { "1" } else { "0" }).to_string(),
                    other => other.to_string(),
                };
                spec.options.push((k.clone(), v));
            }
        }
        for k in o.keys() {
            if !["id", "model", "source", "options"].contains(&k.as_str()) {
                bail!("model: unknown key {k:?} (id, model, options)");
            }
        }
        spec.check()?;
        Ok(spec)
    }

    /// The last value of option `key`.
    pub fn option(&self, key: &str) -> Option<&str> {
        self.options.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    fn check(&self) -> Result<()> {
        if let Some(k) = self.option("kind") {
            Kind::parse(k).with_context(|| format!("{}: unknown kind {k:?} (chat, embedding, image, transcription, speech)", self.id))?;
        }
        for (k, _) in &self.options {
            if ["threads", "qos", "efficiency_threads"].contains(&k.as_str()) {
                bail!("{}: {k} is a server-wide setting, not a model option", self.id);
            }
        }
        Ok(())
    }
}

/// Server settings: the whitelist and how requests are handled ([`Api`]), and the HTTP listener
/// ([`http`]). Set fields, or [`ServerOptions::set`] key-value pairs (the CLI flags and the C API's
/// format).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ServerOptions {
    /// The models clients may use (key `model`, repeatable).
    pub models: Vec<ModelSpec>,
    /// Requests must carry `Authorization: Bearer KEY` (key `api_key`).
    pub api_key: Option<String>,
    /// `Access-Control-Allow-Origin` value (`*` or an origin) for browser clients; None: no CORS
    /// headers (key `cors`).
    pub cors: Option<String>,
    /// Largest request body in bytes (key `max_body_mb`, default 16).
    pub max_body: usize,
    /// Load every model at start instead of on first use (key `preload`).
    pub preload: bool,
    /// Fetch every model at start (key `download`); with neither, on first use.
    pub download: bool,
    /// Memory the loaded models may take together, in bytes (key `memory_gb`; default 80% of
    /// the physical memory, unlimited when unknown).
    pub memory_budget: Option<u64>,
    /// Where repo ids are fetched to (key `cache_dir`; default [`crate::hf::cache_root`]).
    pub cache_dir: Option<PathBuf>,
    /// CPU threads (process-wide; key `threads`, default: all cores).
    pub threads: Option<usize>,
    /// Image, audio and video URLs in requests may be `http(s)://` and are then fetched by the
    /// server (key `remote_media`, default on; `data:` URLs always work).
    pub remote_media: bool,
    /// Media references may also be local paths or `file://` URLs (key `local_media`, default
    /// off: anyone who can reach the server could read files with it).
    pub local_media: bool,
    /// HTTP listener address (key `listen`, default `127.0.0.1:8080`).
    pub listen: String,
    /// HTTPS: PEM certificate chain and key files (keys `tls_cert`, `tls_key`).
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    /// HTTPS with a generated self-signed certificate (key `self_signed`).
    pub self_signed: bool,
    /// HTTP worker threads, the most connections served at once (key `workers`, default 32).
    pub workers: usize,
    /// Log one line per request to stderr (key `log`, default on).
    pub log: bool,
    /// Session snapshots of the chat models kept in this directory across requests, many
    /// conversations and restarts ([`crate::engine::EngineOptions::prompt_cache`]): a request
    /// whose conversation starts with a stored one (or, for models without recurrent state, shares
    /// a prefix with it) restores it instead of recomputing it; conversations are stored after
    /// each reply (once the response is sent), and for recurrent models (Qwen3.5) also the prefix
    /// shared by all conversations with the same system prompt and tools. Key `prompt_cache` (a
    /// directory, created private; `auto` for `~/.cache/oxinfer/prompt-cache`; a model's own
    /// `prompt_cache` option overrides it, `none` turns it off for that model).
    pub prompt_cache: Option<PathBuf>,
    /// The prompt cache's size cap in GB (key `prompt_cache_gb`, default 20; see
    /// [`crate::engine::EngineOptions::prompt_cache_gb`]).
    pub prompt_cache_gb: Option<f64>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            api_key: None,
            cors: None,
            max_body: 16 << 20,
            preload: false,
            download: false,
            memory_budget: None,
            cache_dir: None,
            threads: None,
            remote_media: true,
            local_media: false,
            listen: "127.0.0.1:8080".into(),
            tls_cert: None,
            tls_key: None,
            self_signed: false,
            workers: 32,
            log: true,
            prompt_cache: None,
            prompt_cache_gb: None,
        }
    }
}

fn parse_bool(key: &str, v: &str) -> Result<bool> {
    match v.to_ascii_lowercase().as_str() {
        "" | "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{key}: expected a boolean (1/0, true/false), got {v:?}"),
    }
}

impl ServerOptions {
    /// The keys [`ServerOptions::set`] accepts.
    pub const KEYS: &'static [&'static str] = &[
        "model",
        "api_key",
        "cors",
        "max_body_mb",
        "preload",
        "download",
        "memory_gb",
        "cache_dir",
        "threads",
        "remote_media",
        "local_media",
        "listen",
        "tls_cert",
        "tls_key",
        "self_signed",
        "workers",
        "log",
        "prompt_cache",
        "prompt_cache_gb",
        "config",
    ];

    /// Sets an option by name; `model` adds a [`ModelSpec`], `config` reads a JSON file of
    /// options (`{"listen": "...", "models": [...], ...}`, see [`ServerOptions::from_json`]).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let opt_str = |v: &str| (!v.is_empty()).then(|| v.to_string());
        let num = |v: &str| -> Result<f64> { v.parse().with_context(|| format!("{key}: not a number: {v:?}")) };
        match key {
            "model" | "models" => self.models.push(ModelSpec::parse(value)?),
            "api_key" => self.api_key = opt_str(value),
            "cors" => self.cors = opt_str(value),
            "max_body_mb" => self.max_body = (num(value)? * (1u64 << 20) as f64) as usize,
            "preload" => self.preload = parse_bool(key, value)?,
            "download" => self.download = parse_bool(key, value)?,
            "memory_gb" => self.memory_budget = opt_str(value).map(|v| num(&v)).transpose()?.map(|g| (g * 1e9) as u64),
            "cache_dir" => self.cache_dir = opt_str(value).map(PathBuf::from),
            "threads" => self.threads = opt_str(value).map(|v| v.parse()).transpose().context("threads")?,
            "remote_media" => self.remote_media = parse_bool(key, value)?,
            "local_media" => self.local_media = parse_bool(key, value)?,
            "listen" => self.listen = value.to_string(),
            "tls_cert" => self.tls_cert = opt_str(value).map(PathBuf::from),
            "tls_key" => self.tls_key = opt_str(value).map(PathBuf::from),
            "self_signed" => self.self_signed = parse_bool(key, value)?,
            "workers" => self.workers = value.parse().context("workers")?,
            "log" => self.log = parse_bool(key, value)?,
            "prompt_cache" => {
                self.prompt_cache = match value {
                    "" | "none" | "off" | "0" => None,
                    "auto" | "on" | "1" => Some(crate::snapshot::default_dir()),
                    v => Some(PathBuf::from(v)),
                }
            }
            "prompt_cache_gb" => self.prompt_cache_gb = opt_str(value).map(|v| num(&v)).transpose()?,
            "config" => {
                let text = std::fs::read_to_string(value).with_context(|| format!("config: {value}"))?;
                let v: Value = serde_json::from_str(&text).with_context(|| format!("config: {value}"))?;
                self.apply_json(&v)?;
            }
            _ => bail!("unknown server option {key:?} (known: {})", Self::KEYS.join(", ")),
        }
        Ok(())
    }

    /// Options from a JSON object: the keys of [`ServerOptions::set`], with `models` an array
    /// of [`ModelSpec::from_json`] entries.
    pub fn from_json(v: &Value) -> Result<Self> {
        let mut o = Self::default();
        o.apply_json(v)?;
        Ok(o)
    }

    fn apply_json(&mut self, v: &Value) -> Result<()> {
        for (k, val) in v.as_object().context("server options: expected a JSON object")? {
            match (k.as_str(), val) {
                ("models" | "model", Value::Array(a)) => {
                    for m in a {
                        self.models.push(ModelSpec::from_json(m)?);
                    }
                }
                ("models" | "model", m @ Value::Object(_)) => self.models.push(ModelSpec::from_json(m)?),
                (_, Value::String(s)) => self.set(k, s)?,
                (_, Value::Bool(b)) => self.set(k, if *b { "1" } else { "0" })?,
                (_, Value::Null) => self.set(k, "")?,
                (_, other) => self.set(k, &other.to_string())?,
            }
        }
        Ok(())
    }

    /// Options from `key=value` lines (or `;`-separated pairs; `#` starts a comment line).
    pub fn from_pairs(text: &str) -> Result<Self> {
        let mut o = Self::default();
        for item in text.split(['\n', ';']) {
            let item = item.trim();
            if item.is_empty() || item.starts_with('#') {
                continue;
            }
            let (k, v) = item.split_once('=').with_context(|| format!("option {item:?} is not key=value"))?;
            o.set(k.trim(), v.trim())?;
        }
        Ok(o)
    }
}

/// A request to [`Api::handle`].
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ApiRequest {
    /// `GET`, `POST`, ...
    pub method: String,
    /// The path, without the query string (`/v1/chat/completions`).
    pub path: String,
    /// The `Content-Type` header (JSON or `multipart/form-data`).
    pub content_type: Option<String>,
    /// The `Authorization` header (`Bearer KEY`), checked when [`ServerOptions::api_key`] is set.
    pub authorization: Option<String>,
    pub body: Vec<u8>,
}

impl ApiRequest {
    pub fn new(method: &str, path: &str, content_type: Option<&str>, body: Vec<u8>) -> Self {
        Self {
            method: method.to_string(),
            path: path.split('?').next().unwrap_or_default().to_string(),
            content_type: content_type.map(str::to_string),
            authorization: None,
            body,
        }
    }

    /// A `POST` of a JSON body.
    pub fn post_json(path: &str, body: &str) -> Self {
        Self::new("POST", path, Some("application/json"), body.as_bytes().to_vec())
    }

    pub fn with_authorization(mut self, value: &str) -> Self {
        self.authorization = Some(value.to_string());
        self
    }
}

/// A complete response.
#[derive(Clone, Debug, PartialEq)]
pub struct ApiResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

impl ApiResponse {
    fn json(status: u16, v: &Value) -> Self {
        Self {
            status,
            content_type: "application/json".into(),
            body: v.to_string().into_bytes(),
        }
    }
}

/// Writes a streamed body: returns false once the client is gone (the producer then stops).
pub type Sink<'a> = &'a mut dyn FnMut(&[u8]) -> bool;

/// A streamed body's producer: runs the request, writing the body through the sink.
pub type Producer = Box<dyn FnOnce(Sink) + Send>;

/// What [`Api::handle`] answers.
pub enum Output {
    Full(ApiResponse),
    /// Status 200 with this content type (`text/event-stream` for server-sent events, raw audio
    /// for streamed speech); the producer writes the body (errors past this point are reported
    /// in the stream).
    Stream {
        content_type: String,
        producer: Producer,
    },
}

impl std::fmt::Debug for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Output::Full(r) => f.debug_tuple("Full").field(r).finish(),
            Output::Stream { content_type, .. } => f.debug_struct("Stream").field("content_type", content_type).finish_non_exhaustive(),
        }
    }
}

impl Output {
    /// Runs a streamed output into a buffer (status 200); a full one as it is.
    pub fn collect(self) -> ApiResponse {
        match self {
            Output::Full(r) => r,
            Output::Stream { content_type, producer } => {
                let mut body = Vec::new();
                producer(&mut |b: &[u8]| {
                    body.extend_from_slice(b);
                    true
                });
                ApiResponse {
                    status: 200,
                    content_type,
                    body,
                }
            }
        }
    }
}

/// An error in OpenAI's shape (`{"error": {"message", "type", "param", "code"}}`).
#[derive(Clone, Debug)]
pub(crate) struct ApiError {
    pub status: u16,
    pub message: String,
    pub param: Option<&'static str>,
    pub code: Option<&'static str>,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            param: None,
            code: None,
        }
    }
    pub fn bad(message: impl Into<String>) -> Self {
        Self::new(400, message)
    }
    pub fn param(mut self, p: &'static str) -> Self {
        self.param = Some(p);
        self
    }
    pub fn code(mut self, c: &'static str) -> Self {
        self.code = Some(c);
        self
    }
    /// A failure while running a request (the model's own error).
    pub fn internal(e: anyhow::Error) -> Self {
        Self::new(500, format!("{e:#}"))
    }
    fn kind(&self) -> &'static str {
        match self.status {
            401 => "authentication_error",
            404 => "not_found_error",
            413 | 400 | 405 | 415 | 422 => "invalid_request_error",
            503 => "service_unavailable",
            _ => "server_error",
        }
    }
    pub fn to_json(&self) -> Value {
        json!({ "error": { "message": self.message, "type": self.kind(), "param": self.param, "code": self.code } })
    }
    fn response(&self) -> ApiResponse {
        ApiResponse::json(self.status, &self.to_json())
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        Self::internal(e)
    }
}

pub(crate) type ApiResult<T> = std::result::Result<T, ApiError>;

/// The request handler: the whitelist, the loaded models and the endpoints.
pub struct Api {
    opts: ServerOptions,
    models: registry::Registry,
    ids: AtomicU64,
}

impl Api {
    /// Checks the options, then fetches or loads the models when asked to
    /// ([`ServerOptions::download`], [`ServerOptions::preload`]).
    pub fn new(opts: ServerOptions) -> Result<Arc<Api>> {
        let mut seen = std::collections::HashSet::new();
        for m in &opts.models {
            if !seen.insert(m.id.as_str()) {
                bail!("model id {:?} is listed twice (give one an alias: ID=SOURCE)", m.id);
            }
        }
        if let Some(t) = opts.threads {
            crate::pool::init_with(crate::pool::PoolOptions::threads(t));
        }
        // 80% of the memory, and never more than it can give now while keeping its reserve.
        let budget = opts
            .memory_budget
            .or_else(|| crate::platform::total_memory().map(|m| (m / 10 * 8).min(crate::platform::memory_headroom().unwrap_or(u64::MAX))));
        let mut models = registry::Registry::new(&opts.models, budget, opts.cache_dir.clone(), opts.log);
        models.prompt_cache = opts.prompt_cache.clone();
        models.prompt_cache_gb = opts.prompt_cache_gb;
        let api = Arc::new(Api {
            models,
            opts,
            ids: AtomicU64::new(0),
        });
        if api.opts.preload {
            api.models.preload()?;
        } else if api.opts.download {
            api.models.download()?;
        }
        Ok(api)
    }

    pub fn options(&self) -> &ServerOptions {
        &self.opts
    }

    /// The whitelist's ids with their kind (None until detected) and whether they are loaded.
    pub fn models(&self) -> Vec<(String, Option<Kind>, bool)> {
        self.models.list()
    }

    /// Writes a chat model's waiting prompt-cache snapshot on another thread, after the response
    /// (which this does not delay; the next request to the model waits for it).
    pub(crate) fn flush_prompt_cache_later(self: &Arc<Self>, id: &str) {
        let api = self.clone();
        let id = id.to_string();
        let _ = std::thread::Builder::new()
            .name("oxinfer-prompt-cache".into())
            .spawn(move || api.models.flush_prompt_cache(&id));
    }

    /// Unloads every idle model.
    pub fn unload_all(&self) {
        self.models.unload_idle(u64::MAX);
    }

    /// A fresh id after a prefix (`chatcmpl-` gives `chatcmpl-...`).
    pub(crate) fn new_id(&self, prefix: &str) -> String {
        let n = self.ids.fetch_add(1, Ordering::Relaxed);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        format!("{prefix}{:x}{:04x}", t.as_millis(), n & 0xffff)
    }

    /// Handles one request.
    pub fn handle(self: &Arc<Self>, req: &ApiRequest) -> Output {
        let t0 = std::time::Instant::now();
        let out = match self.route(req) {
            Ok(o) => o,
            Err(e) => Output::Full(e.response()),
        };
        if self.opts.log {
            let status = match &out {
                Output::Full(r) => r.status.to_string(),
                Output::Stream { .. } => "stream".into(),
            };
            eprintln!("[{} {} -> {status} in {:.2}s]", req.method, req.path, t0.elapsed().as_secs_f64());
        }
        out
    }

    fn route(self: &Arc<Self>, req: &ApiRequest) -> ApiResult<Output> {
        let path = req.path.trim_end_matches('/');
        let path = if path.starts_with("/v1/") || path == "/health" {
            path.to_string()
        } else {
            // Clients configured without the /v1 prefix.
            format!("/v1{path}")
        };
        if path == "/health" || path == "/v1/health" {
            return Ok(Output::Full(ApiResponse::json(200, &json!({ "status": "ok" }))));
        }
        self.check_auth(req)?;
        if req.body.len() > self.opts.max_body {
            return Err(ApiError::new(413, format!("request body over {} bytes", self.opts.max_body)));
        }
        let get = req.method.eq_ignore_ascii_case("GET");
        let post = req.method.eq_ignore_ascii_case("POST");
        let wrong_method = || ApiError::new(405, format!("{} is not allowed on {path}", req.method));
        match path.as_str() {
            "/v1/models" if get => Ok(Output::Full(ApiResponse::json(200, &self.models_json()))),
            p if p.starts_with("/v1/models/") && get => {
                let id = &p["/v1/models/".len()..];
                let m = self
                    .models_json()
                    .get("data")
                    .and_then(|d| d.as_array())
                    .and_then(|d| d.iter().find(|m| m["id"] == id).cloned())
                    .ok_or_else(|| ApiError::new(404, format!("the model {id:?} does not exist")).code("model_not_found"))?;
                Ok(Output::Full(ApiResponse::json(200, &m)))
            }
            "/v1/chat/completions" if post => openai::chat(self, &json_body(req)?),
            "/v1/completions" if post => openai::completions(self, &json_body(req)?),
            "/v1/embeddings" if post => openai::embeddings(self, &json_body(req)?),
            "/v1/images/generations" if post => media::image_generations(self, &json_body(req)?),
            "/v1/images/edits" if post => media::image_edits(self, req),
            "/v1/audio/transcriptions" if post => media::transcriptions(self, req, false),
            "/v1/audio/translations" if post => media::transcriptions(self, req, true),
            "/v1/audio/speech" if post => media::speech(self, &json_body(req)?),
            "/v1/models"
            | "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/embeddings"
            | "/v1/images/generations"
            | "/v1/images/edits"
            | "/v1/audio/transcriptions"
            | "/v1/audio/translations"
            | "/v1/audio/speech" => Err(wrong_method()),
            _ => Err(ApiError::new(404, format!("unknown endpoint {}", req.path))),
        }
    }

    fn check_auth(&self, req: &ApiRequest) -> ApiResult<()> {
        let Some(key) = &self.opts.api_key else { return Ok(()) };
        let given = req
            .authorization
            .as_deref()
            .and_then(|a| a.strip_prefix("Bearer ").or_else(|| a.strip_prefix("bearer ")))
            .map(str::trim)
            .unwrap_or_default();
        // Compared in constant time.
        let ok = given.len() == key.len() && given.bytes().zip(key.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0;
        if ok {
            Ok(())
        } else {
            Err(ApiError::new(401, "invalid or missing API key (Authorization: Bearer KEY)").code("invalid_api_key"))
        }
    }

    fn models_json(&self) -> Value {
        let data: Vec<Value> = self
            .models
            .list()
            .into_iter()
            .map(|(id, kind, loaded)| {
                json!({
                    "id": id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "oxinfer",
                    "kind": kind.map(Kind::as_str),
                    "loaded": loaded,
                })
            })
            .collect();
        json!({ "object": "list", "data": data })
    }
}

fn json_body(req: &ApiRequest) -> ApiResult<Value> {
    if let Some(ct) = &req.content_type
        && ct.starts_with("multipart/")
    {
        return Err(ApiError::new(415, "this endpoint takes a JSON body"));
    }
    let v: Value = serde_json::from_slice(&req.body).map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    if !v.is_object() {
        return Err(ApiError::bad("the body must be a JSON object"));
    }
    Ok(v)
}

/// Seconds since the epoch (the `created` fields).
pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_spec_syntax() {
        let m = ModelSpec::parse("qwen=Qwen/Qwen3-0.6B,quant=q8, think=1").unwrap();
        assert_eq!((m.id.as_str(), m.source.as_str()), ("qwen", "Qwen/Qwen3-0.6B"));
        assert_eq!(m.option("quant"), Some("q8"));
        assert_eq!(m.option("think"), Some("1"));
        let m = ModelSpec::parse("org/repo:Q4_K_M").unwrap();
        assert_eq!((m.id.as_str(), m.source.as_str()), ("org/repo:Q4_K_M", "org/repo:Q4_K_M"));
        assert!(ModelSpec::parse("x=y,novalue").is_err());
        assert!(ModelSpec::parse("x=y,kind=video").is_err());
        assert!(ModelSpec::parse("x=y,threads=4").is_err());
        let m =
            ModelSpec::from_json(&json!({"id": "e", "model": "BAAI/bge-small-en-v1.5", "options": {"kind": "embedding", "memory_gb": 1}}))
                .unwrap();
        assert_eq!(m.option("kind"), Some("embedding"));
        assert_eq!(m.option("memory_gb"), Some("1"));
    }

    #[test]
    fn server_options() {
        let o = ServerOptions::from_pairs("model=a=x/y\nmodel=x/z,quant=q4; api_key=k; cors=*; max_body_mb=2").unwrap();
        assert_eq!(o.models.len(), 2);
        assert_eq!(o.models[1].id, "x/z");
        assert_eq!(o.api_key.as_deref(), Some("k"));
        assert_eq!(o.max_body, 2 << 20);
        let o = ServerOptions::from_json(&json!({"listen": "0.0.0.0:1", "models": ["a=b/c"], "preload": true, "workers": 4})).unwrap();
        assert_eq!((o.listen.as_str(), o.preload, o.workers), ("0.0.0.0:1", true, 4));
        assert!(ServerOptions::from_pairs("nope=1").is_err());
    }

    fn api(opts: &str) -> Arc<Api> {
        Api::new(ServerOptions::from_pairs(opts).unwrap()).unwrap()
    }

    #[test]
    fn routing_and_errors() {
        let a = api("model=m=/nonexistent/model; api_key=secret; log=0");
        let r = a.handle(&ApiRequest::new("GET", "/v1/models", None, vec![])).collect();
        assert_eq!(r.status, 401);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["error"]["type"], "authentication_error");
        let auth = |r: ApiRequest| r.with_authorization("Bearer secret");
        let r = a.handle(&auth(ApiRequest::new("GET", "/v1/models", None, vec![]))).collect();
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["data"][0]["id"], "m");
        // Without the /v1 prefix too.
        let r = a.handle(&auth(ApiRequest::new("GET", "/models/m", None, vec![]))).collect();
        assert_eq!(r.status, 200);
        let r = a.handle(&auth(ApiRequest::new("GET", "/v1/models/x", None, vec![]))).collect();
        assert_eq!(r.status, 404);
        let r = a
            .handle(&auth(ApiRequest::new("GET", "/v1/chat/completions", None, vec![])))
            .collect();
        assert_eq!(r.status, 405);
        let r = a.handle(&auth(ApiRequest::post_json("/v1/chat/completions", "{"))).collect();
        assert_eq!(r.status, 400);
        let unknown = r#"{"model": "other", "messages": [{"role": "user", "content": "hi"}]}"#;
        let r = a.handle(&auth(ApiRequest::post_json("/v1/chat/completions", unknown))).collect();
        assert_eq!(r.status, 404);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["error"]["code"], "model_not_found");
        let r = a.handle(&ApiRequest::new("GET", "/health", None, vec![])).collect();
        assert_eq!(r.status, 200);
    }
}
