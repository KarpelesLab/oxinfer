//! What the server is doing: the model requests running or waiting for their model, with what
//! each has done so far, and their cancellation ([`Activity`]); the process's and the machine's
//! resources ([`process_json`], [`machine_json`]).
//!
//! A request is tracked from the moment it passes the API key check until its response is
//! written ([`Activity::start`], dropped with its [`Registration`]). The request's own thread
//! (the streamed body's producer thread, for streams) knows it as the current task
//! ([`enter`], [`current`]), so the code it runs reports its stage, tokens and steps, and checks
//! [`cancelled`] at its stopping points: a request waiting for its model gives up at once, a
//! generation stops at its next token (or denoising step), a streamed body at its next write.

use super::ApiError;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Finished requests kept for the status page.
const RECENT: usize = 20;

/// What a request is doing ([`Task::stage`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Parsing its body, fetching its media.
    Starting,
    /// Waiting for its model, which another request is using.
    Queued,
    /// Its model's files are being fetched.
    Downloading,
    /// Its model is being read into memory.
    Loading,
    /// The prompt is being read.
    Prompt,
    /// Generating (tokens, denoising steps, audio).
    Generating,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Starting => "starting",
            Stage::Queued => "queued",
            Stage::Downloading => "downloading",
            Stage::Loading => "loading",
            Stage::Prompt => "prompt",
            Stage::Generating => "generating",
        }
    }
}

#[derive(Debug)]
struct TaskState {
    model: Option<String>,
    stage: Stage,
    /// Prompt tokens read, of the prompt's.
    prompt: Option<(usize, usize)>,
    tokens: u64,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
    /// Denoising step done, of the steps.
    step: Option<(usize, usize)>,
    /// Bytes fetched of the file being downloaded.
    download: Option<(u64, Option<u64>)>,
}

/// A tracked request.
pub(crate) struct Task {
    pub id: String,
    /// The endpoint (`/v1/chat/completions`).
    pub endpoint: String,
    /// `chat`, `completion`, `embedding`, `image`, `transcription`, `speech`.
    pub kind: &'static str,
    /// The client's address, and its `User-Agent`.
    pub client: Option<String>,
    pub user_agent: Option<String>,
    started: Instant,
    started_unix: u64,
    stream: AtomicBool,
    cancel: AtomicBool,
    st: Mutex<TaskState>,
    /// Stop flags of what runs the request (the engine's cancel handle), set by a cancel.
    hooks: Mutex<Vec<Box<dyn Fn() + Send + Sync>>>,
}

impl Task {
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn state(&self) -> MutexGuard<'_, TaskState> {
        lock(&self.st)
    }

    /// Asks the request to stop (it does at its next stopping point).
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        for h in lock(&self.hooks).iter() {
            h();
        }
    }

    /// Runs `hook` when the request is cancelled (at once when it is already).
    pub fn on_cancel(&self, hook: Box<dyn Fn() + Send + Sync>) {
        if self.cancelled() {
            hook();
        }
        lock(&self.hooks).push(hook);
    }

    fn info(&self) -> RequestInfo {
        let s = self.state();
        let rate = match (s.first_token, s.last_token) {
            (Some(a), Some(b)) if s.tokens > 1 && b > a => Some((s.tokens - 1) as f64 / (b - a).as_secs_f64()),
            _ => None,
        };
        RequestInfo {
            id: self.id.clone(),
            endpoint: self.endpoint.clone(),
            kind: self.kind,
            model: s.model.clone(),
            stage: s.stage,
            stream: self.stream.load(Ordering::Relaxed),
            elapsed: self.started.elapsed().as_secs_f64(),
            started: self.started_unix,
            prompt: s.prompt,
            tokens: s.tokens,
            tokens_per_sec: rate,
            step: s.step,
            download: s.download,
            client: self.client.clone(),
            user_agent: self.user_agent.clone(),
            cancelling: self.cancelled(),
        }
    }
}

/// A request as [`super::Api::requests`] reports it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RequestInfo {
    /// `req-N`: what [`super::Api::cancel`] takes.
    pub id: String,
    pub endpoint: String,
    /// `chat`, `completion`, `embedding`, `image`, `transcription`, `speech`.
    pub kind: &'static str,
    /// The model it names (once known).
    pub model: Option<String>,
    pub stage: Stage,
    /// The response is streamed.
    pub stream: bool,
    /// Seconds since it arrived, and its arrival (Unix time).
    pub elapsed: f64,
    pub started: u64,
    /// Prompt tokens read of the prompt's, while reading it.
    pub prompt: Option<(usize, usize)>,
    /// Tokens generated so far, and the rate since the first one.
    pub tokens: u64,
    pub tokens_per_sec: Option<f64>,
    /// Denoising step done of the steps (image models).
    pub step: Option<(usize, usize)>,
    /// Bytes fetched of the model file being downloaded, of its size.
    pub download: Option<(u64, Option<u64>)>,
    /// The client's address and `User-Agent`, when known.
    pub client: Option<String>,
    pub user_agent: Option<String>,
    /// Cancelled, not stopped yet.
    pub cancelling: bool,
}

impl RequestInfo {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "endpoint": self.endpoint,
            "kind": self.kind,
            "model": self.model,
            "stage": self.stage.as_str(),
            "stream": self.stream,
            "elapsed": (self.elapsed * 10.0).round() / 10.0,
            "started": self.started,
            "prompt": self.prompt.map(|(d, t)| json!({"done": d, "total": t})),
            "tokens": self.tokens,
            "tokens_per_sec": self.tokens_per_sec.map(|r| (r * 10.0).round() / 10.0),
            "step": self.step.map(|(d, t)| json!({"done": d, "total": t})),
            "download": self.download.map(|(d, t)| json!({"done": d, "total": t})),
            "client": self.client,
            "user_agent": self.user_agent,
            "cancelling": self.cancelling,
        })
    }
}

/// The requests in flight, and the last ones finished.
pub(crate) struct Activity {
    tasks: Mutex<Vec<Arc<Task>>>,
    recent: Mutex<VecDeque<Value>>,
    next: AtomicU64,
    /// Requests served, and cancelled, since the start.
    served: AtomicU64,
    cancelled: AtomicU64,
    pub started: Instant,
    pub started_unix: u64,
    /// The last CPU time sample, for the CPU use between two status reads.
    cpu: Mutex<Option<(Instant, f64)>>,
    cpu_pct: Mutex<Option<f64>>,
}

/// Keeps a request listed until dropped (then it moves to the recent ones).
pub(crate) struct Registration {
    activity: Arc<Activity>,
    pub task: Arc<Task>,
    /// How it ended (`ok`, `error`, `cancelled`), when set.
    outcome: Option<(&'static str, Option<u16>)>,
}

impl Registration {
    pub fn finish(&mut self, outcome: &'static str, status: Option<u16>) {
        self.outcome = Some((outcome, status));
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let a = &self.activity;
        lock(&a.tasks).retain(|t| !Arc::ptr_eq(t, &self.task));
        let (outcome, status) = match self.outcome {
            _ if self.task.cancelled() => ("cancelled", Some(499)),
            Some(o) => o,
            None => ("ok", None),
        };
        a.served.fetch_add(1, Ordering::Relaxed);
        if outcome == "cancelled" {
            a.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        let mut v = self.task.info().to_json();
        v["outcome"] = json!(outcome);
        v["status"] = json!(status);
        let mut r = lock(&a.recent);
        r.push_front(v);
        r.truncate(RECENT);
    }
}

impl Activity {
    pub fn new() -> Self {
        Self {
            tasks: Mutex::new(Vec::new()),
            recent: Mutex::new(VecDeque::new()),
            next: AtomicU64::new(1),
            served: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            started: Instant::now(),
            started_unix: super::now(),
            cpu: Mutex::new(None),
            cpu_pct: Mutex::new(None),
        }
    }

    /// Lists a request until the returned registration is dropped.
    pub fn start(
        self: &Arc<Self>,
        kind: &'static str,
        endpoint: &str,
        model: Option<String>,
        client: Option<String>,
        user_agent: Option<String>,
    ) -> Registration {
        let task = Arc::new(Task {
            id: format!("req-{}", self.next.fetch_add(1, Ordering::Relaxed)),
            endpoint: endpoint.to_string(),
            kind,
            client,
            user_agent,
            started: Instant::now(),
            started_unix: super::now(),
            stream: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            st: Mutex::new(TaskState {
                model,
                stage: Stage::Starting,
                prompt: None,
                tokens: 0,
                first_token: None,
                last_token: None,
                step: None,
                download: None,
            }),
            hooks: Mutex::new(Vec::new()),
        });
        lock(&self.tasks).push(task.clone());
        Registration {
            activity: self.clone(),
            task,
            outcome: None,
        }
    }

    /// The requests in flight, oldest first.
    pub fn requests(&self) -> Vec<RequestInfo> {
        lock(&self.tasks).iter().map(|t| t.info()).collect()
    }

    /// Cancels request `id`: false when no such request is in flight.
    pub fn cancel(&self, id: &str) -> bool {
        let t = lock(&self.tasks).iter().find(|t| t.id == id).cloned();
        match t {
            Some(t) => {
                t.cancel();
                true
            }
            None => false,
        }
    }

    /// Cancels every request in flight for `model`; returns how many.
    pub fn cancel_model(&self, model: &str) -> usize {
        let ts: Vec<Arc<Task>> = lock(&self.tasks)
            .iter()
            .filter(|t| t.state().model.as_deref() == Some(model))
            .cloned()
            .collect();
        for t in &ts {
            t.cancel();
        }
        ts.len()
    }

    pub fn recent(&self) -> Vec<Value> {
        lock(&self.recent).iter().cloned().collect()
    }

    pub fn counters(&self) -> (u64, u64) {
        (self.served.load(Ordering::Relaxed), self.cancelled.load(Ordering::Relaxed))
    }

    /// The process's CPU use (100 = one core) since the previous call at least a second ago.
    pub fn cpu_percent(&self, cpu_secs: Option<f64>) -> Option<f64> {
        let now = Instant::now();
        let secs = cpu_secs?;
        let mut last = lock(&self.cpu);
        let mut pct = lock(&self.cpu_pct);
        match *last {
            Some((t, c)) if now.duration_since(t).as_secs_f64() >= 1.0 => {
                *pct = Some(((secs - c) / now.duration_since(t).as_secs_f64() * 100.0).max(0.0));
                *last = Some((now, secs));
            }
            Some(_) => {}
            None => {
                // The first sample: the average since the start.
                let up = self.started.elapsed().as_secs_f64();
                *pct = (up > 0.0).then(|| secs / up * 100.0);
                *last = Some((now, secs));
            }
        }
        pct.map(|p| (p * 10.0).round() / 10.0)
    }
}

/// The error of a cancelled request.
pub(crate) fn cancelled_error() -> ApiError {
    ApiError::new(499, "the request was cancelled (from the server's status page or API)").code("cancelled")
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Task>>> = const { RefCell::new(None) };
}

/// Makes `task` this thread's current request until the guard drops.
pub(crate) fn enter(task: &Arc<Task>) -> Scope {
    Scope(CURRENT.with(|c| c.replace(Some(task.clone()))))
}

pub(crate) struct Scope(Option<Arc<Task>>);

impl Drop for Scope {
    fn drop(&mut self) {
        let prev = self.0.take();
        CURRENT.with(|c| *c.borrow_mut() = prev);
    }
}

/// This thread's request, if it runs one.
pub(crate) fn current() -> Option<Arc<Task>> {
    CURRENT.with(|c| c.borrow().clone())
}

fn with_current(f: impl FnOnce(&Task)) {
    CURRENT.with(|c| {
        if let Some(t) = c.borrow().as_ref() {
            f(t)
        }
    })
}

/// Whether this thread's request was cancelled.
pub(crate) fn cancelled() -> bool {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(|t| t.cancelled()))
}

pub(crate) fn set_stage(stage: Stage) {
    with_current(|t| t.state().stage = stage);
}

pub(crate) fn set_model(model: &str) {
    with_current(|t| t.state().model = Some(model.to_string()));
}

pub(crate) fn set_stream(task: &Task) {
    task.stream.store(true, Ordering::Relaxed);
}

/// A download report of this thread's request's model.
#[cfg_attr(not(feature = "hf"), allow(dead_code))]
pub(crate) fn download(done: u64, total: Option<u64>) {
    with_current(|t| {
        let mut s = t.state();
        s.stage = Stage::Downloading;
        s.download = Some((done, total));
    });
}

/// Prompt progress of this thread's request.
pub(crate) fn prompt(done: usize, total: usize) {
    with_current(|t| {
        let mut s = t.state();
        s.stage = Stage::Prompt;
        s.prompt = Some((done, total));
    });
}

/// One token generated by this thread's request.
pub(crate) fn token() {
    with_current(|t| {
        let now = Instant::now();
        let mut s = t.state();
        s.stage = Stage::Generating;
        s.tokens += 1;
        s.first_token.get_or_insert(now);
        s.last_token = Some(now);
    });
}

/// A denoising step done by this thread's request (`total` 0 when unknown).
pub(crate) fn step(done: usize, total: usize) {
    with_current(|t| {
        let mut s = t.state();
        s.stage = Stage::Generating;
        s.step = Some((done, total));
    });
}

/// The kind of request a model endpoint serves (None: not tracked).
pub(crate) fn kind_of(path: &str) -> Option<&'static str> {
    Some(match path {
        "/v1/chat/completions" | "/v1/messages" => "chat",
        "/v1/completions" => "completion",
        "/v1/embeddings" => "embedding",
        "/v1/images/generations" | "/v1/images/edits" => "image",
        "/v1/audio/transcriptions" | "/v1/audio/translations" => "transcription",
        "/v1/audio/speech" => "speech",
        "/v1/audio/isolations" => "isolation",
        _ => return None,
    })
}

// ---- the process and the machine --------------------------------------------------------------

/// `(user + system CPU seconds, peak resident bytes)` of this process.
fn rusage() -> Option<(f64, u64)> {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios"))]
    {
        unsafe extern "C" {
            fn getrusage(who: std::ffi::c_int, usage: *mut i64) -> std::ffi::c_int;
        }
        // `struct rusage` is 144 bytes on these 64-bit targets: two `timeval`s, then 14 longs.
        let mut buf = [0i64; 32];
        // SAFETY: the buffer is larger than `struct rusage` and 8-byte aligned.
        if unsafe { getrusage(0, buf.as_mut_ptr()) } != 0 {
            return None;
        }
        // `tv_usec` is 32-bit on Apple systems (then 4 bytes of padding).
        let usec = |v: i64| {
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                v & 0xffff_ffff
            } else {
                v
            }
        };
        let secs = buf[0] as f64 + usec(buf[1]) as f64 * 1e-6 + buf[2] as f64 + usec(buf[3]) as f64 * 1e-6;
        // `ru_maxrss`: bytes on Apple systems, kilobytes on Linux.
        let peak = if cfg!(any(target_os = "macos", target_os = "ios")) {
            buf[4] as u64
        } else {
            buf[4] as u64 * 1024
        };
        Some((secs, peak))
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios")))]
    {
        None
    }
}

/// `(resident bytes, physical footprint)` of this process (the footprint on Apple systems only:
/// what Activity Monitor shows, mapped weights in the page cache excluded).
fn resident() -> (Option<u64>, Option<u64>) {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let kb = std::fs::read_to_string("/proc/self/status").ok().and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        });
        (kb.map(|k| k * 1024), None)
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        unsafe extern "C" {
            fn proc_pid_rusage(pid: std::ffi::c_int, flavor: std::ffi::c_int, buffer: *mut u64) -> std::ffi::c_int;
        }
        // `struct rusage_info_v0`: a 16-byte UUID, then 10 u64s (resident size 7th, physical
        // footprint 8th).
        let mut buf = [0u64; 32];
        // SAFETY: the buffer is larger than `rusage_info_v0` (96 bytes) and 8-byte aligned.
        if unsafe { proc_pid_rusage(std::process::id() as std::ffi::c_int, 0, buf.as_mut_ptr()) } != 0 {
            return (None, None);
        }
        (Some(buf[8]), Some(buf[9]))
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios")))]
    {
        (None, None)
    }
}

/// An integer `sysctl` by name (Apple systems).
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sysctl_int(name: &std::ffi::CStr) -> Option<i64> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut std::ffi::c_void,
            oldlen: *mut usize,
            new: *const std::ffi::c_void,
            newlen: usize,
        ) -> std::ffi::c_int;
    }
    let mut v = [0u8; 8];
    let mut len = v.len();
    // SAFETY: `v` holds `len` bytes; nothing is written to the kernel.
    let rc = unsafe { sysctlbyname(name.as_ptr(), v.as_mut_ptr().cast(), &mut len, std::ptr::null(), 0) };
    match (rc, len) {
        (0, 4) => Some(i32::from_ne_bytes(v[..4].try_into().ok()?) as i64),
        (0, 8) => Some(i64::from_ne_bytes(v)),
        _ => None,
    }
}

/// The process: pid, CPU time and use, memory.
pub(crate) fn process_json(a: &Activity) -> Value {
    let ru = rusage();
    let (rss, footprint) = resident();
    json!({
        "pid": std::process::id(),
        "cpu_seconds": ru.map(|r| (r.0 * 10.0).round() / 10.0),
        "cpu_percent": a.cpu_percent(ru.map(|r| r.0)),
        "resident": rss,
        "footprint": footprint,
        "peak_resident": ru.map(|r| r.1),
    })
}

/// How short of memory the system is, where cheap to tell (null elsewhere).
fn pressure_json() -> Value {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // 1 normal, 2 warning, 4 critical; and the percentage of memory free.
        let level = sysctl_int(c"kern.memorystatus_vm_pressure_level");
        let free = sysctl_int(c"kern.memorystatus_level");
        json!({
            "level": match level {
                Some(1) => Some("normal"),
                Some(2) => Some("warning"),
                Some(4) => Some("critical"),
                _ => None,
            },
            "free_percent": free,
        })
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        // Pressure stall information: the share of time some task waited for memory.
        match std::fs::read_to_string("/proc/pressure/memory") {
            Ok(s) => {
                let avg10 = s
                    .lines()
                    .find(|l| l.starts_with("some"))
                    .and_then(|l| l.split_whitespace().find_map(|f| f.strip_prefix("avg10=")))
                    .and_then(|v| v.parse::<f64>().ok());
                json!({ "some_avg10": avg10 })
            }
            Err(_) => Value::Null,
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android")))]
    {
        Value::Null
    }
}

/// The machine's memory, and how short of it the system is (where cheap to tell).
pub(crate) fn machine_json() -> Value {
    let pressure = pressure_json();
    let total = crate::platform::total_memory();
    // Elsewhere than Linux the platform reports no free memory (it answers the total).
    let available = crate::platform::available_memory().filter(|a| Some(*a) != total);
    json!({
        "memory": total,
        "memory_available": available,
        "memory_headroom": crate::platform::memory_headroom().filter(|_| available.is_some()),
        "pressure": pressure,
    })
}

/// The GPU backend in use, with the memory it reports.
pub(crate) fn gpu_json() -> Value {
    match crate::gpu::active() {
        None => Value::Null,
        Some(b) => {
            let dm = b.device_memory();
            json!({
                "backend": b.name(),
                "allocated": b.allocated_memory(),
                "working_set": b.working_set(),
                "device_free": dm.map(|d| d.0),
                "device_total": dm.map(|d| d.1),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_report_and_cancel() {
        let a = Arc::new(Activity::new());
        let mut r = a.start("chat", "/v1/chat/completions", Some("m".into()), Some("127.0.0.1:5".into()), None);
        {
            let _s = enter(&r.task);
            prompt(5, 10);
            assert_eq!(a.requests()[0].stage, Stage::Prompt);
            token();
            token();
            assert!(!cancelled());
            let hit = Arc::new(AtomicBool::new(false));
            let h = hit.clone();
            current().unwrap().on_cancel(Box::new(move || h.store(true, Ordering::Relaxed)));
            assert!(a.cancel(&r.task.id));
            assert!(cancelled() && hit.load(Ordering::Relaxed));
        }
        assert!(!cancelled(), "the scope restores the thread's request");
        let i = &a.requests()[0];
        assert_eq!(
            (i.tokens, i.stage, i.model.as_deref(), i.cancelling),
            (2, Stage::Generating, Some("m"), true)
        );
        assert!(!a.cancel("req-999"));
        r.finish("ok", Some(200));
        drop(r);
        assert!(a.requests().is_empty());
        assert_eq!(a.recent()[0]["outcome"], "cancelled");
        assert_eq!(a.counters(), (1, 1));
        let r = a.start("image", "/v1/images/generations", Some("sd".into()), None, None);
        assert_eq!(a.cancel_model("sd"), 1);
        assert!(r.task.cancelled());
        assert!(process_json(&a)["pid"].is_u64());
        assert!(machine_json().is_object());
    }
}
