//! oxinfer in the browser (and Deno), behind the `web` feature: the library built as a plain
//! C-ABI WebAssembly module (`cargo build-web`), no bindgen. The host
//! registers the model files in the module's virtual filesystem, then drives a chat session token
//! by token. GPU work goes the other way: the module imports `gpu_*` functions that the host
//! implements with WebGPU, and the host wraps them with JSPI (`WebAssembly.Suspending`) so the
//! synchronous `Backend::matmul` call can wait for the asynchronous buffer readback.
//!
//! # Safety
//! The exports taking pointers are `unsafe`: the host must pass pointers into this module's memory
//! that are valid for the lengths given (strings and buffers from `web_alloc`), and `*mut usize`
//! out-parameters must point at writable module memory. `web/src/lib/oxinfer.js` upholds this.
//!
//! ABI: strings are (ptr, len) UTF-8 in module memory; buffers for the host are allocated with
//! `web_alloc` and must be freed with `web_free`. Functions returning `i32` use `< 0` for errors
//! (the message is available through `web_last_error`).
#![allow(clippy::missing_safety_doc)] // the contract is documented once above

mod webgpu;

use crate::chat::{ChatFormat, GemmaChat, QwenChat};
use crate::model::{Model, Quant, Session};
use crate::qwen::{QwenModel, QwenSession};
use crate::sampler::Sampler;
use crate::{config, fsx, quant::PageVec, safetensors};
use anyhow::{Result, anyhow, bail};
use std::cell::RefCell;
use std::path::Path;
use tokenizers::Tokenizer;

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn host_log(ptr: *const u8, len: usize);
    fn host_random(ptr: *mut u8, len: usize);
}

/// `getrandom` custom backend (selected with `--cfg getrandom_backend="custom"` in
/// `.cargo/config.toml`): the tokenizer's hash maps seed from it.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(dest: *mut u8, len: usize) -> Result<(), getrandom::Error> {
    unsafe { host_random(dest, len) };
    Ok(())
}

pub(crate) fn log(s: &str) {
    unsafe { host_log(s.as_ptr(), s.len()) }
}

enum Lm {
    Gemma(Box<Model<'static>>, Session),
    Qwen(Box<QwenModel<'static>>, QwenSession, &'static QwenChat),
}

impl Lm {
    fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        match self {
            Lm::Gemma(m, s) => m.forward(s, tokens),
            Lm::Qwen(m, s, _) => m.forward(s, tokens),
        }
    }
    fn reset(&mut self) {
        match self {
            Lm::Gemma(m, s) => *s = m.new_session(),
            Lm::Qwen(m, s, _) => *s = m.new_session(),
        }
    }
    fn fmt(&self) -> &'static dyn ChatFormat {
        match self {
            Lm::Gemma(m, _) if m.cfg.gemma3 => &crate::chat::Gemma3Chat,
            Lm::Gemma(..) => &GemmaChat,
            Lm::Qwen(.., f) => *f,
        }
    }
}

struct Engine {
    lm: Lm,
    tok: Tokenizer,
    eos: Vec<u32>,
    sampler: Sampler,
    prefill_chunk: usize,
    /// Logits for the next token (after a prefill or a step), empty when the turn has ended.
    logits: Vec<f32>,
    /// Tokens generated in the current turn.
    out: Vec<u32>,
    turns: usize,
    max_tokens: usize,
    think: bool,
    /// Scratch for strings handed to the host (kept alive until the next call).
    text: String,
    /// Tools declared to the model at the start of each conversation (`web_set_tools`).
    tools: Vec<crate::tools::ToolSpec>,
    /// The token that opens a tool call; reply text is shown up to it.
    call_token: Option<u32>,
}

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(e: &anyhow::Error) -> i32 {
    let msg = format!("{e:#}");
    log(&format!("[error] {msg}"));
    LAST_ERROR.with(|l| *l.borrow_mut() = msg);
    -1
}

unsafe fn str_arg<'a>(ptr: *const u8, len: usize) -> &'a str {
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(ptr, len) }).unwrap_or("")
}

/// Panics are reported to the host console (the module itself aborts).
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| log(&format!("[panic] {info}"))));
}

// ---- memory -------------------------------------------------------------------------------------

/// Allocates `len` bytes (page-aligned, so the buffer can later be registered as a file).
#[unsafe(no_mangle)]
pub extern "C" fn web_alloc(len: usize) -> *mut u8 {
    let v = PageVec::<u64>::zeroed(len.div_ceil(8).max(1));
    let p = v.as_ptr() as *mut u8;
    std::mem::forget(v);
    p
}

/// Frees a buffer from `web_alloc` that was not handed to `web_put_file`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { PageVec::<u64>::from_raw(ptr as *mut u64, len.div_ceil(8).max(1)) });
    }
}

/// Registers the buffer (from `web_alloc`, `len` valid bytes) as file `path` in the virtual
/// filesystem; the module takes ownership.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_put_file(path: *const u8, path_len: usize, ptr: *mut u8, len: usize) {
    let path = unsafe { str_arg(path, path_len) }.to_string();
    let v = unsafe { PageVec::<u64>::from_raw(ptr as *mut u64, len.div_ceil(8).max(1)) };
    fsx::put(&path, v, len);
}

/// Registers file `path` as kept by the host (`len` bytes, read through `host_read(id, ...)`):
/// used for lookup-table shards (`*-tables.safetensors`) so they do not count against the 4 GiB
/// of module memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_put_host_file(path: *const u8, path_len: usize, id: u32, len: f64) -> i32 {
    let path = unsafe { str_arg(path, path_len) }.to_string();
    if len > usize::MAX as f64 {
        return set_error(&anyhow!("{path}: {len} bytes is too large for a host file in wasm32"));
    }
    fsx::put_host(&path, fsx::HostFile { id, len: len as usize });
    0
}

// ---- model --------------------------------------------------------------------------------------

/// Loads the model in directory `dir` of the virtual filesystem. `quant`: 0 bf16, 1 q8, 2 q4.
/// `gpu`: non-zero to use the host's WebGPU implementation for prefill matmuls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_load(dir: *const u8, dir_len: usize, quant: u32, gpu: u32, prefill_chunk: u32) -> i32 {
    install_panic_hook();
    let dir = unsafe { str_arg(dir, dir_len) }.to_string();
    match load(Path::new(&dir), quant, gpu != 0, prefill_chunk as usize) {
        Ok(e) => {
            ENGINE.with(|g| *g.borrow_mut() = Some(e));
            0
        }
        Err(e) => set_error(&e),
    }
}

fn load(dir: &Path, quant: u32, gpu: bool, prefill_chunk: usize) -> Result<Engine> {
    let quant = match quant {
        0 => Quant::Bf16,
        1 => Quant::Q8,
        2 => Quant::Q4,
        q => bail!("unknown quant {q}"),
    };
    if gpu {
        webgpu::install();
    }
    let t0 = crate::clock::Instant::now();
    let model_type = config::model_type(dir)?;
    let eos = config::load_eos_ids(dir);
    let tok = Tokenizer::from_bytes(fsx::read(&dir.join("tokenizer.json"))?).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let files: &'static safetensors::ModelFiles = Box::leak(Box::new(safetensors::ModelFiles::open(dir, true)?));
    let lm = if model_type.starts_with("qwen3") {
        let cfg = config::QwenConfig::load(dir)?;
        let m = QwenModel::load(cfg, files, quant)?;
        let s = m.new_session();
        Lm::Qwen(Box::new(m), s, QwenChat::for_model(dir))
    } else {
        let cfg = config::TextConfig::load(dir)?;
        let m = Model::load(cfg, files, quant)?;
        let s = m.new_session();
        Lm::Gemma(Box::new(m), s)
    };
    let (layers, hidden, streamed, stored) = match &lm {
        Lm::Gemma(m, _) => (m.cfg.num_hidden_layers, m.cfg.hidden_size, m.streamed_bytes(), m.quant),
        Lm::Qwen(m, ..) => (m.cfg.num_hidden_layers, m.cfg.hidden_size, m.streamed_bytes(), m.quant),
    };
    let _ = quant;
    files.release_unused_host_files();
    let eos = crate::chat::with_turn_end(lm.fmt(), &tok, eos);
    log(&format!(
        "[loaded {model_type}: {layers} layers, hidden {hidden}, {:.2} GB checkpoint, {stored:?} weights{} streaming {:.2} GB/token, device {}, {:.2}s]",
        files.total_bytes() as f64 / 1e9,
        if files.is_mlx() {
            " (MLX)"
        } else if Quant::stored(files).is_some() {
            " (pre-quantized)"
        } else {
            ""
        },
        streamed as f64 / 1e9,
        crate::gpu::active().map(|b| b.name().to_string()).unwrap_or_else(|| "cpu".into()),
        t0.elapsed().as_secs_f64()
    ));
    Ok(Engine {
        lm,
        tok,
        eos,
        sampler: Sampler::new(0.0, 0, 1.0, 0),
        // Each GPU batch costs a host round trip, so larger chunks amortize better there.
        prefill_chunk: if prefill_chunk != 0 {
            prefill_chunk
        } else if crate::gpu::active().is_some() {
            512
        } else {
            256
        },
        logits: Vec::new(),
        out: Vec::new(),
        turns: 0,
        max_tokens: 0,
        think: false,
        text: String::new(),
        tools: Vec::new(),
        call_token: None,
    })
    .map(|mut e| {
        e.call_token = e.tok.token_to_id(e.lm.fmt().tool_call_marker());
        e
    })
}

/// Sampling settings for the following turns.
#[unsafe(no_mangle)]
pub extern "C" fn web_set_sampling(temperature: f32, top_k: u32, top_p: f32, seed: u64) {
    ENGINE.with(|g| {
        if let Some(e) = g.borrow_mut().as_mut() {
            e.sampler = Sampler::new(temperature, top_k as usize, top_p, seed);
        }
    });
}

/// Starts a new conversation (drops the KV cache).
#[unsafe(no_mangle)]
pub extern "C" fn web_reset() {
    ENGINE.with(|g| {
        if let Some(e) = g.borrow_mut().as_mut() {
            e.lm.reset();
            e.turns = 0;
            e.logits.clear();
            e.out.clear();
        }
    });
}

/// Adds a user turn (and, on the first turn, the optional system prompt), runs the prefill and
/// returns the number of prompt tokens processed. Then call `web_next` until it returns -1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_chat_turn(
    user: *const u8,
    user_len: usize,
    system: *const u8,
    system_len: usize,
    think: u32,
    max_tokens: u32,
) -> i32 {
    let user = unsafe { str_arg(user, user_len) }.to_string();
    let system = if system.is_null() || system_len == 0 {
        None
    } else {
        Some(unsafe { str_arg(system, system_len) }.to_string())
    };
    ENGINE.with(|g| {
        let mut g = g.borrow_mut();
        let Some(e) = g.as_mut() else {
            return set_error(&anyhow!("no model loaded"));
        };
        match e.chat_turn(&user, system.as_deref(), think != 0, max_tokens as usize) {
            Ok(n) => n as i32,
            Err(err) => set_error(&err),
        }
    })
}

impl Engine {
    fn chat_turn(&mut self, user: &str, system: Option<&str>, think: bool, max_tokens: usize) -> Result<usize> {
        let fmt = self.lm.fmt();
        let text = if self.turns == 0 {
            fmt.first_turn(system, think, &self.tools, user)
        } else if !self.out.is_empty() {
            // The previous reply ended (with or without its stop token); close the model turn.
            fmt.end_of_model_turn() + &fmt.user_turn(user, think)
        } else {
            fmt.user_turn(user, think)
        };
        let ids = self
            .tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        self.out.clear();
        self.logits.clear();
        for c in ids.chunks(self.prefill_chunk) {
            self.logits = self.lm.forward(c);
        }
        self.turns += 1;
        self.max_tokens = max_tokens;
        self.think = think;
        Ok(ids.len())
    }

    /// Samples the next token and runs it through the model; None when the turn is over.
    fn next(&mut self) -> Option<u32> {
        if self.logits.is_empty() || self.out.len() >= self.max_tokens {
            self.logits.clear();
            return None;
        }
        let id = self.sampler.sample(&self.logits);
        if self.eos.contains(&id) {
            // Keep it in `out` so the next turn knows the model turn is closed, but do not emit it.
            self.out.push(id);
            self.logits.clear();
            return None;
        }
        self.out.push(id);
        self.logits = self.lm.forward(&[id]);
        Some(id)
    }

    /// The visible reply: up to the first tool call, without special tokens.
    fn decoded(&self) -> String {
        let end = self
            .call_token
            .and_then(|c| self.out.iter().position(|&t| t == c))
            .unwrap_or(self.out.len());
        let shown: Vec<u32> = self.out[..end].iter().copied().filter(|t| !self.eos.contains(t)).collect();
        self.tok.decode(&shown, true).unwrap_or_default()
    }

    /// Tool calls in the reply that just ended.
    fn tool_calls(&self) -> Vec<crate::tools::ToolCall> {
        let raw = self.tok.decode(&self.out, false).unwrap_or_default();
        self.lm.fmt().parse_tool_calls(&raw)
    }

    /// Feeds tool results after a reply that ended on its calls; generation then continues
    /// (`web_next`) with a new reply segment. Returns the number of tokens fed.
    fn tool_results(&mut self, results: &[crate::tools::ToolResult]) -> Result<usize> {
        let text = self.lm.fmt().tool_results(results, self.think);
        let ids = self
            .tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        self.out.clear();
        self.logits.clear();
        for c in ids.chunks(self.prefill_chunk) {
            self.logits = self.lm.forward(c);
        }
        Ok(ids.len())
    }
}

/// Declares tools for the next conversation: a JSON array of `{name, description, parameters}`
/// (parameters as a JSON schema object). An empty array removes them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_set_tools(json: *const u8, len: usize) -> i32 {
    let json = unsafe { str_arg(json, len) };
    let tools: Vec<crate::tools::ToolSpec> = match serde_json::from_str(json) {
        Ok(t) => t,
        Err(e) => return set_error(&anyhow!("tools: {e}")),
    };
    ENGINE.with(|g| match g.borrow_mut().as_mut() {
        Some(e) => {
            e.tools = tools;
            0
        }
        None => set_error(&anyhow!("no model loaded")),
    })
}

/// The tool calls of the reply that just ended, as a JSON array of `{name, arguments}` (pointer;
/// length to `*len`; `[]` when there are none).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_tool_calls(len: *mut usize) -> *const u8 {
    ENGINE.with(|g| {
        let mut g = g.borrow_mut();
        match g.as_mut() {
            Some(e) => {
                e.text = serde_json::to_string(&e.tool_calls()).unwrap_or_else(|_| "[]".into());
                unsafe { *len = e.text.len() };
                e.text.as_ptr()
            }
            None => {
                unsafe { *len = 0 };
                std::ptr::null()
            }
        }
    })
}

/// Feeds the results of those calls (a JSON array of `{name, content}`) and prepares the next
/// reply segment; then call `web_next` again. Returns the number of tokens fed, or < 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_tool_results(json: *const u8, len: usize) -> i32 {
    let json = unsafe { str_arg(json, len) };
    let results: Vec<crate::tools::ToolResult> = match serde_json::from_str(json) {
        Ok(r) => r,
        Err(e) => return set_error(&anyhow!("tool results: {e}")),
    };
    ENGINE.with(|g| match g.borrow_mut().as_mut() {
        Some(e) => match e.tool_results(&results) {
            Ok(n) => n as i32,
            Err(err) => set_error(&err),
        },
        None => set_error(&anyhow!("no model loaded")),
    })
}

/// Generates one token: returns its id, or -1 when the reply is complete (stop token or limit).
#[unsafe(no_mangle)]
pub extern "C" fn web_next() -> i32 {
    ENGINE.with(|g| g.borrow_mut().as_mut().and_then(|e| e.next()).map(|t| t as i32).unwrap_or(-1))
}

/// The reply so far, decoded (valid until the next call into the module); returns the pointer,
/// the length goes to `*len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_text(len: *mut usize) -> *const u8 {
    ENGINE.with(|g| {
        let mut g = g.borrow_mut();
        match g.as_mut() {
            Some(e) => {
                e.text = e.decoded();
                unsafe { *len = e.text.len() };
                e.text.as_ptr()
            }
            None => {
                unsafe { *len = 0 };
                std::ptr::null()
            }
        }
    })
}

/// Last error message (pointer; length to `*len`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_last_error(len: *mut usize) -> *const u8 {
    LAST_ERROR.with(|l| {
        let l = l.borrow();
        unsafe { *len = l.len() };
        l.as_ptr()
    })
}

/// Number of tokens generated in the current turn.
#[unsafe(no_mangle)]
pub extern "C" fn web_generated() -> u32 {
    ENGINE.with(|g| {
        g.borrow()
            .as_ref()
            .map(|e| e.out.iter().filter(|t| !e.eos.contains(t)).count() as u32)
            .unwrap_or(0)
    })
}

/// GPU statistics line (pointer; length to `*len`), empty without a GPU backend.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_gpu_stats(len: *mut usize) -> *const u8 {
    ENGINE.with(|g| {
        let mut g = g.borrow_mut();
        match g.as_mut() {
            Some(e) => {
                e.text = crate::gpu::active().and_then(|b| b.stats()).unwrap_or_default();
                unsafe { *len = e.text.len() };
                e.text.as_ptr()
            }
            None => {
                unsafe { *len = 0 };
                std::ptr::null()
            }
        }
    })
}

// ---- image generation (Stable Diffusion) -------------------------------------------------------

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Called after each denoising step.
    fn host_progress(step: u32, total: u32);
}

struct SdState {
    pipe: crate::sd::Pipeline,
    image: Vec<u8>,
    width: u32,
    height: u32,
    check: String,
}

thread_local! {
    static SD: RefCell<Option<SdState>> = const { RefCell::new(None) };
}

/// Loads the diffusers pipeline in directory `dir` of the virtual filesystem (its `.safetensors`
/// files are best registered with `web_put_host_file`: weights are converted on load, so the
/// files never need to be in module memory). `gpu` non-zero uses the host's WebGPU. `loras`:
/// LoRA files to merge, one `PATH[:SCALE]` per line (empty for none).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_sd_load(dir: *const u8, dir_len: usize, gpu: u32, loras: *const u8, loras_len: usize) -> i32 {
    install_panic_hook();
    let dir = unsafe { str_arg(dir, dir_len) }.to_string();
    let loras = if loras_len == 0 { "" } else { unsafe { str_arg(loras, loras_len) } };
    let loras: Result<Vec<_>> = loras
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| crate::sd::lora::LoraSpec::parse(l.trim()))
        .collect();
    let loras = match loras {
        Ok(l) => l,
        Err(e) => return set_error(&e),
    };
    if gpu != 0 {
        webgpu::install();
    }
    let t0 = crate::clock::Instant::now();
    match crate::sd::Pipeline::load_with_loras(Path::new(&dir), &loras, &mut |line| log(&format!("[lora: {line}]"))) {
        Ok(pipe) => {
            log(&format!(
                "[loaded diffusion pipeline in {:.1}s{}; device {}]",
                t0.elapsed().as_secs_f64(),
                if pipe.turbo { ", turbo" } else { "" },
                crate::gpu::active().map(|b| b.name().to_string()).unwrap_or_else(|| "cpu".into())
            ));
            SD.with(|s| {
                *s.borrow_mut() = Some(SdState {
                    pipe,
                    image: Vec::new(),
                    width: 0,
                    height: 0,
                    check: String::new(),
                })
            });
            0
        }
        Err(e) => set_error(&e),
    }
}

/// Defaults for the loaded pipeline into `out[0..4]`: image side, steps, guidance, turbo (0/1).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_sd_defaults(out: *mut f32) -> i32 {
    SD.with(|s| match s.borrow().as_ref() {
        Some(st) => {
            let v = [
                st.pipe.default_size() as f32,
                if st.pipe.turbo { 1.0 } else { 25.0 },
                if st.pipe.turbo { 0.0 } else { 7.5 },
                st.pipe.turbo as u8 as f32,
            ];
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out, 4) };
            0
        }
        None => set_error(&anyhow!("no image model loaded")),
    })
}

/// Generates an image; read it with `web_sd_image`. `sampler`: 0 Euler, 1 Euler ancestral.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn web_sd_generate(
    prompt: *const u8,
    prompt_len: usize,
    negative: *const u8,
    negative_len: usize,
    steps: u32,
    guidance: f32,
    seed: f64,
    width: u32,
    height: u32,
    sampler: u32,
) -> i32 {
    let prompt = unsafe { str_arg(prompt, prompt_len) }.to_string();
    let negative = if negative.is_null() {
        String::new()
    } else {
        unsafe { str_arg(negative, negative_len) }.to_string()
    };
    SD.with(|s| {
        let mut s = s.borrow_mut();
        let Some(st) = s.as_mut() else {
            return set_error(&anyhow!("no image model loaded"));
        };
        let params = crate::sd::Params {
            prompt,
            negative,
            steps: steps.max(1) as usize,
            guidance,
            seed: seed as u64,
            width: width as usize,
            height: height as usize,
            sampler: if sampler == 0 {
                crate::sd::sched::Sampler::Euler
            } else {
                crate::sd::sched::Sampler::EulerAncestral
            },
        };
        match st.pipe.generate(&params, &mut |i, n| unsafe { host_progress(i as u32, n as u32) }) {
            Ok((img, tm)) => {
                log(&format!(
                    "[text {:.2}s | denoise {:.2}s ({} UNet passes) | decode {:.2}s]",
                    tm.text, tm.denoise, tm.unet_calls, tm.decode
                ));
                st.width = img.width as u32;
                st.height = img.height as u32;
                st.image = img.rgb;
                0
            }
            Err(e) => set_error(&e),
        }
    })
}

/// Recompute every device op of the image pipeline on the CPU and keep the worst error per kind.
#[unsafe(no_mangle)]
pub extern "C" fn web_gpu_check_enable() {
    crate::sd::ops::gpu_check::enable();
}

/// The device-vs-CPU summary (pointer; length to `*len`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_gpu_check_summary(len: *mut usize) -> *const u8 {
    SD.with(|s| {
        let mut s = s.borrow_mut();
        match s.as_mut() {
            Some(st) => {
                st.check = crate::sd::ops::gpu_check::summary();
                unsafe { *len = st.check.len() };
                st.check.as_ptr()
            }
            None => {
                unsafe { *len = 0 };
                std::ptr::null()
            }
        }
    })
}

/// The last generated image as RGB bytes (`width * height * 3`); dimensions to `*w`, `*h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_sd_image(w: *mut u32, h: *mut u32) -> *const u8 {
    SD.with(|s| match s.borrow().as_ref() {
        Some(st) => {
            unsafe {
                *w = st.width;
                *h = st.height;
            }
            st.image.as_ptr()
        }
        None => std::ptr::null(),
    })
}
