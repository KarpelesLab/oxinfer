//! A small C API over [`crate::engine`] (feature `capi`), for Swift and other non-Rust hosts.
//! The header is `include/oxinfer.h`; build a static library with `cargo build-capi --target T`
//! (see `.cargo/config.toml`). Rust programs should use [`crate::engine`] directly.
//!
//! Conventions: strings are NUL-terminated UTF-8; functions returning `int` use `< 0` for errors
//! and pointer-returning ones NULL, with the message in `oxi_last_error()` (per thread). Strings
//! returned as `char *` are owned by the caller and freed with `oxi_string_free`. No Rust panic
//! crosses the boundary: a panic becomes an error return.
//!
//! Threads: an engine may be used from any thread, one call at a time, except `oxi_engine_cancel`,
//! which may be called from any thread at any time (it stops a running `oxi_engine_send` /
//! `oxi_engine_complete` after the current token).
#![allow(clippy::missing_safety_doc)] // the contract is documented once above and in the header

use crate::engine::{CancelHandle, Delta, Engine, EngineOptions, Flow, Reply, SamplingParams, StopReason, SystemPrompt};
use anyhow::{Context, Result, anyhow, bail};
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

/// Receives streamed text (`len` bytes, not NUL-terminated); return non-zero to stop.
pub type OxiTokenFn = Option<unsafe extern "C" fn(user: *mut c_void, text: *const c_char, len: usize, thinking: c_int) -> c_int>;
/// Receives download progress (`total` 0 when unknown); return non-zero to cancel.
pub type OxiProgressFn =
    Option<unsafe extern "C" fn(user: *mut c_void, file: *const c_char, done: u64, total: u64, index: c_int, count: c_int) -> c_int>;

/// An engine handle.
pub struct OxiEngine {
    cancel: CancelHandle,
    engine: Mutex<Engine>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: String) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|l| *l.borrow_mut() = c);
}

/// Runs `f`, turning errors and panics into `err` with the message stored for `oxi_last_error`.
fn guard<T>(err: T, f: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(format!("{e:#}"));
            err
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            set_error(format!("internal error (panic): {msg}"));
            err
        }
    }
}

unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> Result<&'a str> {
    if p.is_null() {
        bail!("{what} is NULL");
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .with_context(|| format!("{what} is not UTF-8"))
}

unsafe fn opt_str_arg<'a>(p: *const c_char, what: &str) -> Result<Option<&'a str>> {
    if p.is_null() {
        Ok(None)
    } else {
        unsafe { str_arg(p, what) }.map(Some)
    }
}

fn out_string(s: String) -> *mut c_char {
    CString::new(s.replace('\0', " ")).unwrap_or_default().into_raw()
}

unsafe fn engine<'a>(e: *const OxiEngine) -> Result<&'a OxiEngine> {
    unsafe { e.as_ref() }.context("engine is NULL")
}

/// The library version (static string).
#[unsafe(no_mangle)]
pub extern "C" fn oxi_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// The last error on this thread (valid until the next failing call on it; never NULL).
#[unsafe(no_mangle)]
pub extern "C" fn oxi_last_error() -> *const c_char {
    LAST_ERROR.with(|l| l.borrow().as_ptr())
}

/// Frees a string returned by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Parses `key=value` lines (or `;`-separated pairs) into engine options.
fn parse_options(text: Option<&str>) -> Result<EngineOptions> {
    let mut o = EngineOptions::default();
    for item in text.unwrap_or("").split(['\n', ';']) {
        let item = item.trim();
        if item.is_empty() || item.starts_with('#') {
            continue;
        }
        let (k, v) = item.split_once('=').with_context(|| format!("option {item:?} is not key=value"))?;
        o.set(k.trim(), v.trim())?;
    }
    Ok(o)
}

/// Runs `f` with download progress going to `progress`.
fn with_progress<T>(progress: OxiProgressFn, user: *mut c_void, f: impl FnOnce() -> Result<T>) -> Result<T> {
    #[cfg(feature = "hf")]
    if let Some(cb) = progress {
        let mut hook = |p: &crate::hf::Progress| -> bool {
            let file = CString::new(p.file).unwrap_or_default();
            let r = unsafe {
                cb(
                    user,
                    file.as_ptr(),
                    p.done,
                    p.total.unwrap_or(0),
                    p.index as c_int,
                    p.count as c_int,
                )
            };
            r == 0
        };
        return crate::hf::with_progress(&mut hook, f);
    }
    let _ = (progress, user);
    f()
}

/// Loads a model (directory, `.gguf` file or repo id; see `oxinfer.h` for the option keys).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_load(
    model: *const c_char,
    options: *const c_char,
    progress: OxiProgressFn,
    user: *mut c_void,
) -> *mut OxiEngine {
    guard(std::ptr::null_mut(), || {
        let model = unsafe { str_arg(model, "model") }?;
        let opts = parse_options(unsafe { opt_str_arg(options, "options") }?)?;
        let e = with_progress(progress, user, || Engine::load(model, &opts))?;
        Ok(Box::into_raw(Box::new(OxiEngine {
            cancel: e.cancel_handle(),
            engine: Mutex::new(e),
        })))
    })
}

/// Downloads (or finds in the cache) a model and returns its local path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_download(
    spec: *const c_char,
    cache_dir: *const c_char,
    progress: OxiProgressFn,
    user: *mut c_void,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let spec = unsafe { str_arg(spec, "spec") }?;
        let cache = unsafe { opt_str_arg(cache_dir, "cache_dir") }?.map(std::path::PathBuf::from);
        let p = with_progress(progress, user, || crate::engine::resolve_model(spec, cache.as_deref()))?;
        Ok(out_string(p.to_string_lossy().into_owned()))
    })
}

/// Frees an engine (NULL is ignored). No other call on it may be running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_free(e: *mut OxiEngine) {
    if !e.is_null() {
        guard((), || {
            drop(unsafe { Box::from_raw(e) });
            Ok(())
        })
    }
}

fn lock(e: &OxiEngine) -> std::sync::MutexGuard<'_, Engine> {
    e.engine.lock().unwrap_or_else(|p| p.into_inner())
}

/// Sets a conversation or sampling setting (see `oxinfer.h`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_set(e: *mut OxiEngine, key: *const c_char, value: *const c_char) -> c_int {
    guard(-1, || {
        let e = unsafe { engine(e) }?;
        let key = unsafe { str_arg(key, "key") }?;
        let value = unsafe { opt_str_arg(value, "value") }?;
        let mut g = lock(e);
        let v = value.unwrap_or("");
        let num = |what: &str| -> Result<f64> { v.parse().map_err(|_| anyhow!("{key}: expected {what}, got {v:?}")) };
        let mut s: SamplingParams = g.sampling();
        match key {
            "system" => {
                g.config_mut().system = match value {
                    None => SystemPrompt::Default,
                    Some("") => SystemPrompt::None,
                    Some(t) => SystemPrompt::Text(t.to_string()),
                }
            }
            "think" => g.config_mut().think = num("0 or 1")? != 0.0,
            "max_tokens" => g.config_mut().max_tokens = num("a count")? as usize,
            "temperature" => s.temperature = num("a number")? as f32,
            "top_k" => s.top_k = num("a count")? as usize,
            "top_p" => s.top_p = num("a number")? as f32,
            "seed" => s.seed = v.parse().map_err(|_| anyhow!("seed: expected an integer, got {v:?}"))?,
            _ => bail!("unknown setting {key:?} (system, think, max_tokens, temperature, top_k, top_p, seed)"),
        }
        if s != g.sampling() {
            g.set_sampling(s);
        }
        Ok(0)
    })
}

fn finish_code(r: &Reply) -> c_int {
    match r.finish {
        StopReason::Stop => 0,
        StopReason::Length => 1,
        StopReason::Cancelled => 2,
    }
}

fn run(
    e: *mut OxiEngine,
    text: *const c_char,
    cb: OxiTokenFn,
    user: *mut c_void,
    f: impl FnOnce(&mut Engine, &str, &mut dyn FnMut(&Delta) -> Flow) -> Result<Reply>,
) -> c_int {
    guard(-1, || {
        let e = unsafe { engine(e) }?;
        let text = unsafe { str_arg(text, "text") }?;
        let mut g = lock(e);
        let mut on_delta = |d: &Delta| -> Flow {
            match cb {
                Some(cb) if !d.text.is_empty() => {
                    let r = unsafe { cb(user, d.text.as_ptr() as *const c_char, d.text.len(), d.thinking as c_int) };
                    if r != 0 { Flow::Stop } else { Flow::Continue }
                }
                _ => Flow::Continue,
            }
        };
        let reply = f(&mut g, text, &mut on_delta)?;
        Ok(finish_code(&reply))
    })
}

/// Sends a user message and streams the reply. Returns 0 (stop token), 1 (length limit),
/// 2 (cancelled) or -1 (error).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_send(e: *mut OxiEngine, text: *const c_char, cb: OxiTokenFn, user: *mut c_void) -> c_int {
    run(e, text, cb, user, |g, t, d| g.send(t, d))
}

/// Feeds raw text (no chat template) and streams the continuation; returns as `oxi_engine_send`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_complete(e: *mut OxiEngine, text: *const c_char, cb: OxiTokenFn, user: *mut c_void) -> c_int {
    run(e, text, cb, user, |g, t, d| g.complete(t, d))
}

/// Starts a new conversation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_reset(e: *mut OxiEngine) -> c_int {
    guard(-1, || {
        lock(unsafe { engine(e) }?).reset();
        Ok(0)
    })
}

/// Stops the running generation (any thread).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_cancel(e: *const OxiEngine) {
    if let Some(e) = unsafe { e.as_ref() } {
        e.cancel.cancel();
    }
}

/// Model information as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_info_json(e: *mut OxiEngine) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let g = lock(unsafe { engine(e) }?);
        let i = g.info();
        let v = serde_json::json!({
            "path": i.path.to_string_lossy(),
            "model_type": i.model_type,
            "family": format!("{:?}", i.family),
            "layers": i.layers,
            "hidden_size": i.hidden_size,
            "vocab_size": i.vocab_size,
            "quant": i.quant.as_str(),
            "format": i.format,
            "checkpoint_bytes": i.checkpoint_bytes,
            "streamed_bytes": i.streamed_bytes,
            "device": i.device,
            "threads": i.threads,
            "mmap": i.mmap,
            "speculative": format!("{:?}", i.speculative),
            "load_secs": i.load_secs,
            "coreml": i.coreml.as_ref().map(|c| serde_json::json!({
                "units": c.units.as_str(),
                "prefill": c.prefill.as_str(),
                "architecture": c.architecture,
                "context_length": c.context_length,
                "batch_size": c.batch_size,
                "chunks": c.chunks,
                "recurrent": c.recurrent,
            })),
            "decision": g.decision_assets().map(|d| d.info_json()),
        });
        Ok(out_string(v.to_string()))
    })
}

/// Statistics of the last turn as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_stats_json(e: *mut OxiEngine) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let g = lock(unsafe { engine(e) }?);
        let s = g.stats();
        let v = serde_json::json!({
            "prompt_tokens": s.prompt_tokens,
            "prefill_secs": s.prefill_secs,
            "prefill_tok_s": s.prefill_tok_s(),
            "generated_tokens": s.generated_tokens,
            "decode_secs": s.decode_secs,
            "decode_tok_s": s.decode_tok_s(),
            "forwards": s.forwards,
            "drafted": s.drafted,
            "accepted": s.accepted,
            "acceptance": s.acceptance(),
            "context_tokens": s.context_tokens,
        });
        Ok(out_string(v.to_string()))
    })
}

/// Detected CPU features and GPUs (the CLI's `--devices` report).
#[unsafe(no_mangle)]
pub extern "C" fn oxi_devices() -> *mut c_char {
    guard(std::ptr::null_mut(), || Ok(out_string(crate::platform::detect().report())))
}

/// A typed decision (Jev models, see `oxinfer.h`): a `/v1/decide` JSON request in, the JSON
/// response out.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_decide(e: *mut OxiEngine, request_json: *const c_char) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let e = unsafe { engine(e) }?;
        let req = unsafe { str_arg(request_json, "request_json") }?;
        let req: serde_json::Value = serde_json::from_str(req).context("request_json is not JSON")?;
        let out = lock(e).decide_json(&req)?;
        Ok(out_string(out.to_string()))
    })
}

/// The decision assets as JSON (`/v1/decide/info`), or NULL with an error when none are loaded.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_decide_info_json(e: *mut OxiEngine) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let g = lock(unsafe { engine(e) }?);
        let d = g.decision_assets().context("no decision assets loaded")?;
        Ok(out_string(d.info_json().to_string()))
    })
}
