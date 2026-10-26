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
//! `oxi_engine_complete` after the current token). Whisper transcribers (`oxi_transcriber_*`,
//! over [`crate::whisper`]) and Orpheus speakers (`oxi_speaker_*`, `oxi_speak*`, over
//! [`crate::tts`]) follow the same rules, with `oxi_transcriber_cancel` / `oxi_speaker_cancel`.
#![allow(clippy::missing_safety_doc)] // the contract is documented once above and in the header

use crate::engine::{Attachments, CancelHandle, Delta, Engine, EngineOptions, Flow, Reply, SamplingParams, StopReason, SystemPrompt};
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
            "repetition_penalty" => s.repetition_penalty = num("a number")? as f32,
            _ => bail!("unknown setting {key:?} (system, think, max_tokens, temperature, top_k, top_p, seed, repetition_penalty)"),
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

/// Sends a user message with attachments (`media_json`, see `oxinfer.h`) and streams the reply;
/// returns as `oxi_engine_send`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_engine_send_media(
    e: *mut OxiEngine,
    text: *const c_char,
    media_json: *const c_char,
    cb: OxiTokenFn,
    user: *mut c_void,
) -> c_int {
    let media = match unsafe { opt_str_arg(media_json, "media_json") }.and_then(|j| match j {
        None => Ok(Attachments::default()),
        Some(j) => Attachments::from_json(&serde_json::from_str(j).context("media_json is not JSON")?),
    }) {
        Ok(m) => m,
        Err(err) => return guard(-1, || Err(err)),
    };
    run(e, text, cb, user, |g, t, d| g.send_with_media(t, &media, d))
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

// ---- speech to text (Whisper) ---------------------------------------------------------------

/// Receives one finished segment as JSON (`len` bytes, not NUL-terminated); return non-zero to
/// stop the transcription.
pub type OxiSegmentFn = Option<unsafe extern "C" fn(user: *mut c_void, segment_json: *const c_char, len: usize) -> c_int>;

/// A Whisper model handle.
pub struct OxiTranscriber {
    cancel: CancelHandle,
    asr: Mutex<crate::whisper::Transcriber>,
}

unsafe fn transcriber<'a>(t: *const OxiTranscriber) -> Result<&'a OxiTranscriber> {
    unsafe { t.as_ref() }.context("transcriber is NULL")
}

/// Loads a Whisper model (directory or repo id; `key=value` options as for engines).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_transcriber_load(
    model: *const c_char,
    options: *const c_char,
    progress: OxiProgressFn,
    user: *mut c_void,
) -> *mut OxiTranscriber {
    guard(std::ptr::null_mut(), || {
        let model = unsafe { str_arg(model, "model") }?;
        let mut o = crate::whisper::TranscriberOptions::default();
        for item in unsafe { opt_str_arg(options, "options") }?.unwrap_or("").split(['\n', ';']) {
            let item = item.trim();
            if item.is_empty() || item.starts_with('#') {
                continue;
            }
            let (k, v) = item.split_once('=').with_context(|| format!("option {item:?} is not key=value"))?;
            o.set(k.trim(), v.trim())?;
        }
        let t = with_progress(progress, user, || crate::whisper::Transcriber::load(model, &o))?;
        Ok(Box::into_raw(Box::new(OxiTranscriber {
            cancel: t.cancel_handle(),
            asr: Mutex::new(t),
        })))
    })
}

/// Frees a transcriber (NULL is ignored). No other call on it may be running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_transcriber_free(t: *mut OxiTranscriber) {
    if !t.is_null() {
        guard((), || {
            drop(unsafe { Box::from_raw(t) });
            Ok(())
        })
    }
}

/// Stops the running transcription (any thread); it returns the segments done so far.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_transcriber_cancel(t: *const OxiTranscriber) {
    if let Some(t) = unsafe { t.as_ref() } {
        t.cancel.cancel();
    }
}

/// Model information as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_transcriber_info_json(t: *mut OxiTranscriber) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let t = unsafe { transcriber(t) }?;
        let g = t.asr.lock().unwrap_or_else(|p| p.into_inner());
        Ok(out_string(serde_json::to_string(g.info())?))
    })
}

fn transcribe_common(
    t: *mut OxiTranscriber,
    options_json: *const c_char,
    cb: OxiSegmentFn,
    user: *mut c_void,
    pcm: impl FnOnce() -> Result<Vec<f32>>,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let t = unsafe { transcriber(t) }?;
        let opts: crate::whisper::TranscribeOptions = match unsafe { opt_str_arg(options_json, "options_json") }?.map(str::trim) {
            Some(s) if !s.is_empty() => serde_json::from_str(s).context("options_json")?,
            _ => Default::default(),
        };
        let samples = pcm()?;
        let g = t.asr.lock().unwrap_or_else(|p| p.into_inner());
        let mut on_segment = |s: &crate::whisper::Segment| -> Flow {
            match cb {
                Some(cb) => {
                    let j = serde_json::to_string(s).unwrap_or_default();
                    let r = unsafe { cb(user, j.as_ptr() as *const c_char, j.len()) };
                    if r != 0 { Flow::Stop } else { Flow::Continue }
                }
                None => Flow::Continue,
            }
        };
        let out = g.transcribe_with(&samples, &opts, &mut on_segment)?;
        Ok(out_string(serde_json::to_string(&out)?))
    })
}

/// Transcribes `n` mono samples in [-1, 1] at `sample_rate` Hz (resampled to 16 kHz). Returns
/// the transcript as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_transcribe_pcm(
    t: *mut OxiTranscriber,
    samples: *const f32,
    n: usize,
    sample_rate: c_int,
    options_json: *const c_char,
    cb: OxiSegmentFn,
    user: *mut c_void,
) -> *mut c_char {
    transcribe_common(t, options_json, cb, user, || {
        if samples.is_null() && n > 0 {
            bail!("samples is NULL");
        }
        let x = if n == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(samples, n) }
        };
        let rate = usize::try_from(sample_rate)
            .ok()
            .filter(|&r| r > 0)
            .context("sample_rate must be positive")?;
        Ok(crate::audio::resample(x, rate, crate::whisper::mel::SAMPLE_RATE))
    })
}

/// Transcribes an audio file (decoded with the `media` feature). Returns the transcript as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_transcribe_file(
    t: *mut OxiTranscriber,
    path: *const c_char,
    options_json: *const c_char,
    cb: OxiSegmentFn,
    user: *mut c_void,
) -> *mut c_char {
    transcribe_common(t, options_json, cb, user, || {
        let path = unsafe { str_arg(path, "path") }?;
        #[cfg(feature = "media")]
        {
            crate::whisper::Transcriber::load_audio(std::path::Path::new(path))
        }
        #[cfg(not(feature = "media"))]
        {
            let _ = path;
            bail!("decoding audio files needs the `media` feature; use oxi_transcribe_pcm")
        }
    })
}

// ---- Janus-Pro (text to image, image understanding) ----------------------------------------

/// Receives one generated image: `index` in the batch, tightly packed RGB8 (`len` =
/// `width * height * 3` bytes, valid during the call only).
pub type OxiImageFn =
    Option<unsafe extern "C" fn(user: *mut c_void, index: c_int, width: c_int, height: c_int, rgb: *const u8, len: usize)>;
/// Receives image-generation progress (`step` of `steps` image tokens); return non-zero to cancel.
pub type OxiStepFn = Option<unsafe extern "C" fn(user: *mut c_void, step: c_int, steps: c_int) -> c_int>;

/// A Janus-Pro model handle.
pub struct OxiJanus {
    cancel: crate::janus::CancelHandle,
    model: Mutex<crate::janus::Janus>,
}

unsafe fn janus<'a>(j: *const OxiJanus) -> Result<&'a OxiJanus> {
    unsafe { j.as_ref() }.context("janus is NULL")
}

fn lock_janus(j: &OxiJanus) -> std::sync::MutexGuard<'_, crate::janus::Janus> {
    j.model.lock().unwrap_or_else(|p| p.into_inner())
}

/// Loads a Janus-Pro model (directory or repo id; `key=value` options: quant, mmap, device,
/// cache_dir, generation, understanding).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_janus_load(
    model: *const c_char,
    options: *const c_char,
    progress: OxiProgressFn,
    user: *mut c_void,
) -> *mut OxiJanus {
    guard(std::ptr::null_mut(), || {
        let model = unsafe { str_arg(model, "model") }?;
        let mut o = crate::janus::JanusOptions::default();
        for item in unsafe { opt_str_arg(options, "options") }?.unwrap_or("").split(['\n', ';']) {
            let item = item.trim();
            if item.is_empty() || item.starts_with('#') {
                continue;
            }
            let (k, v) = item.split_once('=').with_context(|| format!("option {item:?} is not key=value"))?;
            o.set(k.trim(), v.trim())?;
        }
        let j = with_progress(progress, user, || crate::janus::Janus::load(model, &o))?;
        Ok(Box::into_raw(Box::new(OxiJanus {
            cancel: j.cancel_handle(),
            model: Mutex::new(j),
        })))
    })
}

/// Frees a Janus model (NULL is ignored). No other call on it may be running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_janus_free(j: *mut OxiJanus) {
    if !j.is_null() {
        guard((), || {
            drop(unsafe { Box::from_raw(j) });
            Ok(())
        })
    }
}

/// Stops the running generation or answer (any thread).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_janus_cancel(j: *const OxiJanus) {
    if let Some(j) = unsafe { j.as_ref() } {
        j.cancel.cancel();
    }
}

/// Text to image: generates `parallel_size` 384x384 images for `prompt` (`options_json`: see
/// `oxinfer.h`), each passed to `on_image`. Returns the number of images, or -1 (error or
/// cancelled).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_janus_generate(
    j: *mut OxiJanus,
    prompt: *const c_char,
    options_json: *const c_char,
    on_image: OxiImageFn,
    on_step: OxiStepFn,
    user: *mut c_void,
) -> c_int {
    guard(-1, || {
        let j = unsafe { janus(j) }?;
        let prompt = unsafe { str_arg(prompt, "prompt") }?;
        let opts: crate::janus::ImageOptions = match unsafe { opt_str_arg(options_json, "options_json") }?.map(str::trim) {
            Some(s) if !s.is_empty() => serde_json::from_str(s).context("options_json")?,
            _ => Default::default(),
        };
        let g = lock_janus(j);
        let mut progress = |p: crate::janus::Progress| -> Flow {
            match on_step {
                Some(cb) if unsafe { cb(user, p.step as c_int, p.steps as c_int) } != 0 => Flow::Stop,
                _ => Flow::Continue,
            }
        };
        let (imgs, _) = g.generate_images(prompt, &opts, &mut progress)?;
        if let Some(cb) = on_image {
            for (i, img) in imgs.iter().enumerate() {
                unsafe {
                    cb(
                        user,
                        i as c_int,
                        img.width as c_int,
                        img.height as c_int,
                        img.rgb.as_ptr(),
                        img.rgb.len(),
                    )
                };
            }
        }
        Ok(imgs.len() as c_int)
    })
}

fn janus_answer(
    j: *mut OxiJanus,
    question: *const c_char,
    options_json: *const c_char,
    cb: OxiTokenFn,
    user: *mut c_void,
    images: impl FnOnce() -> Result<Vec<crate::video::RgbImage>>,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let j = unsafe { janus(j) }?;
        let question = unsafe { str_arg(question, "question") }?;
        let opts: crate::janus::ChatOptions = match unsafe { opt_str_arg(options_json, "options_json") }?.map(str::trim) {
            Some(s) if !s.is_empty() => serde_json::from_str(s).context("options_json")?,
            _ => Default::default(),
        };
        let images = images()?;
        let g = lock_janus(j);
        let mut on_text = |t: &str| -> Flow {
            match cb {
                Some(cb) if !t.is_empty() && unsafe { cb(user, t.as_ptr() as *const c_char, t.len(), 0) } != 0 => Flow::Stop,
                _ => Flow::Continue,
            }
        };
        Ok(out_string(g.understand(question, &images, &opts, &mut on_text)?))
    })
}

/// Answers `question` about the images in `images_json` (a JSON array of file paths, decoded
/// with the `media` feature; NULL for none), streaming the text to `cb`. Returns the answer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_janus_understand(
    j: *mut OxiJanus,
    question: *const c_char,
    images_json: *const c_char,
    options_json: *const c_char,
    cb: OxiTokenFn,
    user: *mut c_void,
) -> *mut c_char {
    janus_answer(j, question, options_json, cb, user, || {
        let paths: Vec<String> = match unsafe { opt_str_arg(images_json, "images_json") }?.map(str::trim) {
            Some(s) if !s.is_empty() => serde_json::from_str(s).context("images_json: expected an array of paths")?,
            _ => Vec::new(),
        };
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "media")]
        {
            let m = crate::media::Media::new();
            paths.iter().map(|p| m.load_image(std::path::Path::new(p))).collect()
        }
        #[cfg(not(feature = "media"))]
        bail!("decoding image files needs the `media` feature; use oxi_janus_understand_rgb")
    })
}

/// [`oxi_janus_understand`] with one image given as tightly packed RGB8 pixels.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn oxi_janus_understand_rgb(
    j: *mut OxiJanus,
    question: *const c_char,
    rgb: *const u8,
    width: c_int,
    height: c_int,
    options_json: *const c_char,
    cb: OxiTokenFn,
    user: *mut c_void,
) -> *mut c_char {
    janus_answer(j, question, options_json, cb, user, || {
        let (w, h) = (usize::try_from(width)?, usize::try_from(height)?);
        if rgb.is_null() || w == 0 || h == 0 {
            bail!("rgb is NULL or the size is empty");
        }
        let data = unsafe { std::slice::from_raw_parts(rgb, w * h * 3) }.to_vec();
        Ok(vec![crate::video::RgbImage { width: w, height: h, data }])
    })
}

// ---- Text to speech (Orpheus) ---------------------------------------------------------------

/// Receives streamed audio: `n` mono samples in [-1, 1] at `sample_rate`, the first being sample
/// `start` of the utterance; return non-zero to stop.
pub type OxiAudioFn =
    Option<unsafe extern "C" fn(user: *mut c_void, samples: *const f32, n: usize, start: u64, sample_rate: c_int) -> c_int>;

/// An Orpheus speaker handle.
pub struct OxiSpeaker {
    cancel: CancelHandle,
    speaker: Mutex<crate::tts::Speaker>,
    stats: Mutex<Option<crate::tts::SpeakStats>>,
}

unsafe fn speaker<'a>(s: *const OxiSpeaker) -> Result<&'a OxiSpeaker> {
    unsafe { s.as_ref() }.context("speaker is NULL")
}

/// Loads an Orpheus model and the SNAC decoder (`key=value` options, see the header).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_speaker_load(
    model: *const c_char,
    options: *const c_char,
    progress: OxiProgressFn,
    user: *mut c_void,
) -> *mut OxiSpeaker {
    guard(std::ptr::null_mut(), || {
        let model = unsafe { str_arg(model, "model") }?;
        let mut o = crate::tts::SpeakerOptions::default();
        for item in unsafe { opt_str_arg(options, "options") }?.unwrap_or("").split(['\n', ';']) {
            let item = item.trim();
            if item.is_empty() || item.starts_with('#') {
                continue;
            }
            let (k, v) = item.split_once('=').with_context(|| format!("option {item:?} is not key=value"))?;
            o.set(k.trim(), v.trim())?;
        }
        let s = with_progress(progress, user, || crate::tts::Speaker::load(model, &o))?;
        Ok(Box::into_raw(Box::new(OxiSpeaker {
            cancel: s.cancel_handle(),
            speaker: Mutex::new(s),
            stats: Mutex::new(None),
        })))
    })
}

/// Frees a speaker (NULL is ignored). No other call on it may be running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_speaker_free(s: *mut OxiSpeaker) {
    if !s.is_null() {
        guard((), || {
            drop(unsafe { Box::from_raw(s) });
            Ok(())
        })
    }
}

/// Stops the running utterance (any thread); it returns the audio generated so far.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_speaker_cancel(s: *const OxiSpeaker) {
    if let Some(s) = unsafe { s.as_ref() } {
        s.cancel.cancel();
    }
}

/// Model information as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_speaker_info_json(s: *mut OxiSpeaker) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let s = unsafe { speaker(s) }?;
        let g = s.speaker.lock().unwrap_or_else(|p| p.into_inner());
        let i = g.info();
        let j = serde_json::json!({
            "path": i.path,
            "snac_path": i.snac_path,
            "layers": i.layers,
            "hidden_size": i.hidden_size,
            "quant": i.quant.as_str(),
            "device": i.device,
            "threads": i.threads,
            "sample_rate": i.sample_rate,
            "load_secs": i.load_secs,
            "voices": i.voices,
            "languages": i.languages,
            "pretrained": i.pretrained,
            "snac_encoder": i.snac_encoder,
        });
        Ok(out_string(j.to_string()))
    })
}

/// Encodes a voice-cloning reference: `n` mono samples in [-1, 1] at `sample_rate` saying
/// `text`. Returns the reference as JSON (`{"text", "frames"}`), to put in the speak options'
/// `"references"` array (several references: a steadier voice).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_speaker_encode_reference(
    s: *mut OxiSpeaker,
    samples: *const f32,
    n: usize,
    sample_rate: c_int,
    text: *const c_char,
) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let s = unsafe { speaker(s) }?;
        let text = unsafe { str_arg(text, "text") }?;
        if samples.is_null() && n > 0 {
            bail!("samples is NULL");
        }
        if sample_rate <= 0 {
            bail!("sample_rate must be positive");
        }
        let pcm: &[f32] = if n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(samples, n) }
        };
        let g = s.speaker.lock().unwrap_or_else(|p| p.into_inner());
        let r = g.encode_reference(pcm, sample_rate as u32, text)?;
        Ok(out_string(serde_json::to_string(&r)?))
    })
}

/// `oxi_speaker_encode_reference` of an audio file (WAV, or any format oxideav decodes when
/// built with `media`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_speaker_encode_reference_file(s: *mut OxiSpeaker, path: *const c_char, text: *const c_char) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let s = unsafe { speaker(s) }?;
        let path = unsafe { str_arg(path, "path") }?;
        let text = unsafe { str_arg(text, "text") }?;
        let g = s.speaker.lock().unwrap_or_else(|p| p.into_inner());
        let r = g.encode_reference_file(std::path::Path::new(path), text)?;
        Ok(out_string(serde_json::to_string(&r)?))
    })
}

/// Statistics of the last utterance as JSON (`null` before the first).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_speaker_stats_json(s: *mut OxiSpeaker) -> *mut c_char {
    guard(std::ptr::null_mut(), || {
        let s = unsafe { speaker(s) }?;
        let st = s.stats.lock().unwrap_or_else(|p| p.into_inner());
        let mut j = serde_json::to_value(&*st)?;
        if let (Some(st), Some(o)) = (&*st, j.as_object_mut()) {
            o.insert("tok_s".into(), st.tok_s().into());
            o.insert("rtf".into(), st.rtf().into());
        }
        Ok(out_string(j.to_string()))
    })
}

fn speak_common(
    s: *mut OxiSpeaker,
    text: *const c_char,
    options_json: *const c_char,
    cb: OxiAudioFn,
    user: *mut c_void,
) -> Result<crate::tts::Speech> {
    let s = unsafe { speaker(s) }?;
    let text = unsafe { str_arg(text, "text") }?;
    let opts = crate::tts::SpeakOptions::from_json(unsafe { opt_str_arg(options_json, "options_json") }?.unwrap_or(""))?;
    let g = s.speaker.lock().unwrap_or_else(|p| p.into_inner());
    let speech = match cb {
        Some(cb) => g.speak_stream(text, &opts, &mut |c| {
            let r = unsafe { cb(user, c.samples.as_ptr(), c.samples.len(), c.start as u64, c.sample_rate as c_int) };
            if r != 0 { Flow::Stop } else { Flow::Continue }
        })?,
        None => g.speak_detailed(text, &opts)?,
    };
    *s.stats.lock().unwrap_or_else(|p| p.into_inner()) = Some(speech.stats.clone());
    Ok(speech)
}

fn speech_code(f: StopReason) -> c_int {
    match f {
        StopReason::Stop => 0,
        StopReason::Length => 1,
        StopReason::Cancelled => 2,
    }
}

/// Speaks `text`; the samples go to `samples_out` (free with `oxi_audio_free`) when it is not
/// NULL. Returns 0 (end of speech), 1 (max_tokens), 2 (cancelled) or -1.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn oxi_speak(
    s: *mut OxiSpeaker,
    text: *const c_char,
    options_json: *const c_char,
    cb: OxiAudioFn,
    user: *mut c_void,
    samples_out: *mut *mut f32,
    n_out: *mut usize,
    sample_rate_out: *mut c_int,
) -> c_int {
    guard(-1, || {
        let speech = speak_common(s, text, options_json, cb, user)?;
        let a = speech.audio;
        unsafe {
            if let Some(p) = sample_rate_out.as_mut() {
                *p = a.sample_rate as c_int;
            }
            if let Some(p) = n_out.as_mut() {
                *p = a.samples.len();
            }
            if let Some(p) = samples_out.as_mut() {
                *p = Box::into_raw(a.samples.into_boxed_slice()) as *mut f32;
            }
        }
        Ok(speech_code(speech.finish))
    })
}

/// Frees samples returned by `oxi_speak` (`n` as returned; NULL is ignored).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_audio_free(samples: *mut f32, n: usize) {
    if !samples.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(samples, n)) });
    }
}

/// Speaks `text` into a 16-bit PCM WAV file. Returns as `oxi_speak`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxi_speak_wav(
    s: *mut OxiSpeaker,
    text: *const c_char,
    options_json: *const c_char,
    path: *const c_char,
    cb: OxiAudioFn,
    user: *mut c_void,
) -> c_int {
    guard(-1, || {
        let path = unsafe { str_arg(path, "path") }?;
        let speech = speak_common(s, text, options_json, cb, user)?;
        speech.audio.write_wav(path)?;
        Ok(speech_code(speech.finish))
    })
}
