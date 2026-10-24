//! High-level text generation: load a model by path or repo id, hold a conversation with it,
//! stream the reply token by token, cancel, and read statistics. This is what the CLI's text
//! modes do, as a library (the CLI is built from the same pieces: [`crate::generate`],
//! [`resolve_model`], [`load_tokenizer`], [`crate::mm`]).
//!
//! ```no_run
//! use oxinfer::engine::{Engine, EngineOptions, Flow};
//! # fn main() -> anyhow::Result<()> {
//! let mut opts = EngineOptions::default();
//! opts.quant = oxinfer::model::Quant::Q8;
//! opts.set("mtp", "1")?; // the same options as key-value pairs (for configuration files, FFI)
//! let mut engine = Engine::load("Qwen/Qwen3.5-0.8B", &opts)?;
//! let reply = engine.send("Why is the sky blue?", &mut |d| {
//!     print!("{}", d.text);
//!     Flow::Continue
//! })?;
//! eprintln!("{:.1} tok/s", reply.stats.decode_tok_s());
//! # Ok(())
//! # }
//! ```
//!
//! Supported checkpoints: Gemma 4 and Gemma 3 (text), Qwen3.5 / Qwen3.8 (dense and MoE), Qwen3,
//! as safetensors folders (bf16, oxinfer pre-quantized, MLX) or single GGUF files. Speculative
//! decoding: Qwen's MTP layer ([`EngineOptions::mtp`], optionally from a sidecar checkpoint) or
//! a Gemma 4 assistant model ([`EngineOptions::draft`], greedy). Images and audio for Gemma 4
//! with the `media` feature ([`Engine::send_with`]).
//!
//! Core ML models (on Apple Silicon macOS / iOS with the `gpu` feature): an ANEMLL-format
//! directory (`meta.yaml` + `.mlmodelc` parts, from ANEMLL or `tools/coreml_convert.py`: Qwen3,
//! Qwen3.5 / 3.6 / 3.8 dense, Gemma 4 E2B / E4B, Gemma 3, Llama 3) runs on the Neural Engine
//! through `crate::anemll`, behind the same API. [`EngineOptions::coreml_units`] and
//! [`EngineOptions::coreml_prefill`] choose where and how; the context length is fixed by the
//! conversion ([`CoreMlInfo::context_length`]); [`ModelInfo::coreml`] is set for these models.
//! Not available for them: attachments, speculative decoding, streaming, [`Engine::files`]
//! (empty).
//!
//! # Threads and concurrency
//! An [`Engine`] is `Send`: load it on one thread and use it from another. Its methods take
//! `&mut self`, so one engine runs one generation at a time; [`CancelHandle`] is the only part
//! meant for other threads (e.g. a UI's stop button). The CPU thread pool and the GPU backend
//! are process-wide: the thread count and device of the first engine (or of the latest
//! [`EngineOptions::threads`]) apply to all, and generations of different engines are
//! serialized by a global lock.
//!
//! Downloads: wrap [`Engine::load`] in [`crate::hf::with_progress`] to receive progress and to
//! cancel a download.

use crate::chat::{self, ChatFormat};
use crate::clock::Instant;
use crate::drafter::Drafter;
use crate::generate::{self, Decoded, Flow as GFlow, TextStream};
use crate::model::{Model, Quant, Session};
use crate::platform::DevicePref;
use crate::qwen::{QwenModel, QwenSession};
use crate::qwen_mtp::{MtpDecoder, MtpOptions, QwenMtp};
use crate::safetensors::ModelFiles;
use crate::sampler::Sampler;
use crate::tools::{ToolCall, ToolResult, ToolSpec};
use crate::{config, mm, pool};
use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokenizers::Tokenizer;

pub use crate::generate::{Flow, StopReason};

/// How to load and run a model. Construct with `EngineOptions::default()` and set fields (or
/// [`EngineOptions::set`] key-value pairs); new options are added as new fields and keys.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EngineOptions {
    /// CPU worker threads (process-wide; default: all cores). Key `threads`.
    pub threads: Option<usize>,
    /// Weight precision for checkpoints that are not already quantized (GGUF, pre-quantized and
    /// MLX checkpoints keep their own). Key `quant` (`bf16`, `q8`, `q4`). Default bf16.
    pub quant: Quant,
    /// Map the weights from disk instead of copying them into memory. Key `mmap`. Default on
    /// iOS (mapped file pages do not count as dirty memory there), off elsewhere.
    pub mmap: bool,
    /// With `mmap`, touch every weight page while loading. Key `preload`.
    pub preload: bool,
    /// Compute device (process-wide; set by the first engine). Key `device`
    /// (`auto`, `cpu`, `metal`, `cuda`, `vulkan`).
    pub device: DevicePref,
    /// Tokens per prefill chunk (default 256 on the CPU, 1024 with a GPU). Key `prefill_chunk`.
    pub prefill_chunk: Option<usize>,
    /// Maximum tokens in the conversation (prompt and replies); a turn that does not fit fails
    /// and replies stop at the limit. Key `max_context`. Default: unlimited.
    pub max_context: Option<usize>,
    /// Qwen3.5/3.8: speculative decoding with the checkpoint's multi-token prediction layer
    /// (greedy and sampled; same output distribution). Key `mtp`.
    pub mtp: bool,
    /// Implies `mtp`: take the MTP weights (and a draft head, if any) from this checkpoint
    /// (path or repo id), e.g. the original one for a converted copy. Key `mtp_from`.
    pub mtp_from: Option<String>,
    /// With `mtp`, always draft `draft_tokens` (default: adaptive between 1 and that).
    /// Key `mtp_fixed`.
    pub mtp_fixed: bool,
    /// Gemma 4: speculative decoding (greedy) with this assistant model (path or repo id).
    /// Key `draft`.
    pub draft: Option<String>,
    /// Maximum draft tokens per round (default 3 for MTP, 6 for a drafter). Key `draft_tokens`.
    pub draft_tokens: Option<usize>,
    /// Where repo ids are downloaded (default [`crate::hf::cache_root`]). Key `cache_dir`.
    pub cache_dir: Option<PathBuf>,
    /// Qwen3.5/3.8: stream the weights from the (mapped) file for models larger than memory,
    /// prefetching the next layers while one computes (implies `mmap`; see [`crate::stream`]).
    /// MTP then drafts up to 8 tokens with a cost-aware length. The routed experts of MoE models
    /// are read on demand, recently used ones kept in a part of the page-cache budget. Keys
    /// `stream` (on with the defaults), `cache_budget_gb`, `resident_mb`, `prefetch_mb`,
    /// `flash_mbps`, `expert_cache_mb`.
    pub stream: Option<crate::stream::StreamOptions>,
    /// Core ML models: where Core ML may run them. Default (None): the Neural Engine, or the
    /// CPU / GPU when `device` asks for it. Key `coreml_units` (`auto`, `ane`, `all`, `gpu`,
    /// `cpu`).
    pub coreml_units: Option<CoreMlUnits>,
    /// Core ML models: how prompts are fed. Key `coreml_prefill` (`auto`, `batch`, `pad`,
    /// `single`).
    pub coreml_prefill: CoreMlPrefill,
    /// GPU matmuls bit-identical to the CPU (no half-precision matrix units on M5/A19; slower
    /// prefill). Process-wide, applied on load; None leaves the current setting
    /// ([`crate::gpu::set_exact`]). Key `gpu_exact`.
    pub gpu_exact: Option<bool>,
    /// MoE experts on the GPU during prefill. Process-wide, applied on load; None leaves the
    /// current setting ([`crate::gpu::set_moe_offload`]). Key `gpu_moe`.
    pub gpu_moe: Option<bool>,
}

/// Where Core ML runs a model ([`EngineOptions::coreml_units`]; Core ML's `MLComputeUnits`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreMlUnits {
    /// CPU and Neural Engine (`ane`).
    Ane,
    /// Any device Core ML picks (`all`).
    All,
    /// CPU and GPU (`gpu`).
    Gpu,
    /// CPU only (`cpu`; Core ML returns NaN for some stateful fp16 graphs there).
    Cpu,
}

impl CoreMlUnits {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "ane" | "neural-engine" | "cpu-ane" => Self::Ane,
            "all" => Self::All,
            "gpu" | "cpu-gpu" => Self::Gpu,
            "cpu" => Self::Cpu,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ane => "ane",
            Self::All => "all",
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

/// How a Core ML model is fed prompts ([`EngineOptions::coreml_prefill`]; `PrefillMode` of
/// `crate::anemll` on Apple targets).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoreMlPrefill {
    /// What the model's `meta.yaml` supports (`auto`).
    #[default]
    Auto,
    /// Full batches through `prefill`, the rest token by token (`batch`).
    Batch,
    /// As `Batch` with the last partial batch padded; same as `Batch` for recurrent models
    /// (`pad`).
    Pad,
    /// Every token through `infer` (`single`).
    Single,
}

impl CoreMlPrefill {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "auto" => Self::Auto,
            "batch" | "batched" => Self::Batch,
            "pad" | "padded" => Self::Pad,
            "single" => Self::Single,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Batch => "batch",
            Self::Pad => "pad",
            Self::Single => "single",
        }
    }
}

/// A loaded Core ML model ([`ModelInfo::coreml`]).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CoreMlInfo {
    pub units: CoreMlUnits,
    /// The prefill mode in use (`auto` resolved).
    pub prefill: CoreMlPrefill,
    /// `model_info.architecture` of `meta.yaml` (`qwen3`, `qwen3_5`, `gemma4`, ...).
    pub architecture: String,
    /// Maximum tokens in the conversation (fixed by the conversion).
    pub context_length: usize,
    /// Tokens per `prefill` call.
    pub batch_size: usize,
    /// Decoder chunks (separate Core ML models).
    pub chunks: usize,
    /// Recurrent state (Qwen3.5 Gated DeltaNet): every token is fed exactly once and the
    /// conversation cannot be rewound (a reset starts a new state).
    pub recurrent: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            threads: None,
            quant: Quant::Bf16,
            mmap: cfg!(target_os = "ios"),
            preload: false,
            device: DevicePref::Auto,
            prefill_chunk: None,
            max_context: None,
            mtp: false,
            mtp_from: None,
            mtp_fixed: false,
            draft: None,
            draft_tokens: None,
            cache_dir: None,
            stream: None,
            coreml_units: None,
            coreml_prefill: CoreMlPrefill::Auto,
            gpu_exact: None,
            gpu_moe: None,
        }
    }
}

fn parse_bool(key: &str, v: &str) -> Result<bool> {
    match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{key}: expected a boolean (1/0, true/false), got {v:?}"),
    }
}

/// An empty value or 0 means "unset" for optional counts.
fn parse_opt_usize(key: &str, v: &str) -> Result<Option<usize>> {
    if v.is_empty() {
        return Ok(None);
    }
    let n: usize = v.parse().with_context(|| format!("{key}: expected a number, got {v:?}"))?;
    Ok((n > 0).then_some(n))
}

impl EngineOptions {
    /// The keys [`EngineOptions::set`] accepts.
    pub const KEYS: &'static [&'static str] = &[
        "threads",
        "quant",
        "mmap",
        "preload",
        "device",
        "prefill_chunk",
        "max_context",
        "mtp",
        "mtp_from",
        "mtp_fixed",
        "draft",
        "draft_tokens",
        "cache_dir",
        "stream",
        "cache_budget_gb",
        "resident_mb",
        "prefetch_mb",
        "flash_mbps",
        "expert_cache_mb",
        "coreml_units",
        "coreml_prefill",
        "gpu_exact",
        "gpu_moe",
    ];

    /// Sets an option by name (see the field docs for the keys). Unknown keys are errors.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let opt_str = |v: &str| (!v.is_empty()).then(|| v.to_string());
        match key {
            "threads" => self.threads = parse_opt_usize(key, value)?,
            "quant" => {
                self.quant = match value {
                    "bf16" => Quant::Bf16,
                    "q8" | "int8" => Quant::Q8,
                    "q4" | "q4_0" => Quant::Q4,
                    _ => bail!("quant: unknown quantization {value:?} (use bf16, q8 or q4)"),
                }
            }
            "mmap" => self.mmap = parse_bool(key, value)?,
            "preload" => self.preload = parse_bool(key, value)?,
            "device" => {
                self.device =
                    DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?} (auto|cpu|cuda|metal|vulkan)"))?
            }
            "prefill_chunk" => self.prefill_chunk = parse_opt_usize(key, value)?,
            "max_context" => self.max_context = parse_opt_usize(key, value)?,
            "mtp" => self.mtp = parse_bool(key, value)?,
            "mtp_from" => {
                self.mtp_from = opt_str(value);
                self.mtp |= self.mtp_from.is_some();
            }
            "mtp_fixed" => self.mtp_fixed = parse_bool(key, value)?,
            "draft" => self.draft = opt_str(value),
            "draft_tokens" => self.draft_tokens = parse_opt_usize(key, value)?,
            "cache_dir" => self.cache_dir = opt_str(value).map(PathBuf::from),
            "stream" => self.stream = parse_bool(key, value)?.then(Default::default),
            "cache_budget_gb" | "resident_mb" | "prefetch_mb" | "flash_mbps" | "expert_cache_mb" => {
                let st = self.stream.get_or_insert_with(Default::default);
                let v: f64 = value.parse().with_context(|| format!("{key}: not a number: {value:?}"))?;
                match key {
                    "cache_budget_gb" => st.cache_budget = Some((v * 1e9) as u64),
                    "resident_mb" => st.resident_bytes = Some((v * (1u64 << 20) as f64) as u64),
                    "prefetch_mb" => st.prefetch_bytes = (v * (1u64 << 20) as f64) as u64,
                    "expert_cache_mb" => st.part_cache_bytes = Some((v * (1u64 << 20) as f64) as u64),
                    _ => st.flash_bytes_per_sec = Some(v * 1e6),
                }
            }
            "coreml_units" => {
                self.coreml_units = match value {
                    "" | "auto" => None,
                    v => Some(CoreMlUnits::parse(v).with_context(|| format!("coreml_units: unknown value {v:?} (auto|ane|all|gpu|cpu)"))?),
                }
            }
            "coreml_prefill" => {
                self.coreml_prefill = CoreMlPrefill::parse(value)
                    .with_context(|| format!("coreml_prefill: unknown value {value:?} (auto|batch|pad|single)"))?
            }
            "gpu_exact" => self.gpu_exact = Some(parse_bool(key, value)?),
            "gpu_moe" => self.gpu_moe = Some(parse_bool(key, value)?),
            _ => bail!("unknown option {key:?} (known: {})", Self::KEYS.join(", ")),
        }
        Ok(())
    }
}

/// Sampling settings. Temperature 0 is greedy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    /// The CLI's defaults: greedy, top-k 64, top-p 0.95, seed 42.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 64,
            top_p: 0.95,
            seed: 42,
        }
    }
}

/// The system prompt of a conversation.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum SystemPrompt {
    /// A short prompt with the local date and time ([`crate::tools::default_system_prompt`]).
    #[default]
    Default,
    None,
    Text(String),
}

/// Conversation settings. `system`, `think` and `tools` shape the start of a conversation, so
/// changes to them apply after [`Engine::reset`]; `think` also applies to each later user turn,
/// and `max_tokens` to each reply.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ChatConfig {
    pub system: SystemPrompt,
    /// Thinking mode (models that have one).
    pub think: bool,
    /// Maximum tokens per reply (default 1024).
    pub max_tokens: usize,
    /// Tools declared to the model; calls come back in [`Reply::tool_calls`] and results go in
    /// with [`Engine::send_tool_results`]. [`crate::tools::builtin_specs`] and
    /// [`crate::tools::run_builtin`] are the CLI's built-in tools.
    pub tools: Vec<ToolSpec>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            system: SystemPrompt::Default,
            think: false,
            max_tokens: 1024,
            tools: Vec::new(),
        }
    }
}

/// A piece of the streamed reply.
#[derive(Clone, Copy, Debug)]
pub struct Delta<'a> {
    /// The token just generated (None for the final flush of held-back text).
    pub token: Option<u32>,
    /// New text (often empty: a token can end inside a multi-byte character, and thinking
    /// delimiters and tool calls are not shown).
    pub text: &'a str,
    /// Whether the text belongs to a thinking block.
    pub thinking: bool,
}

/// Timings and counters of the last turn.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Tokens fed for the turn (the formatted user message, tool results or raw prompt).
    pub prompt_tokens: usize,
    pub prefill_secs: f64,
    /// Tokens generated, without the stop token.
    pub generated_tokens: usize,
    pub decode_secs: f64,
    /// Main-model forward passes while decoding (fewer than tokens with speculative decoding).
    pub forwards: usize,
    /// Speculative decoding: draft tokens proposed and accepted.
    pub drafted: usize,
    pub accepted: usize,
    /// Tokens in the conversation after the turn.
    pub context_tokens: usize,
}

impl Stats {
    pub fn prefill_tok_s(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_secs.max(1e-9)
    }
    pub fn decode_tok_s(&self) -> f64 {
        self.generated_tokens as f64 / self.decode_secs.max(1e-9)
    }
    /// Fraction of draft tokens accepted (0 without speculative decoding).
    pub fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }
    pub fn tokens_per_forward(&self) -> f64 {
        self.generated_tokens as f64 / self.forwards.max(1) as f64
    }
}

/// A finished reply.
#[derive(Clone, Debug)]
pub struct Reply {
    /// The visible reply: without thinking, special tokens and tool calls.
    pub text: String,
    /// The thinking block's text, if any.
    pub thinking: String,
    /// Generated token ids, including the stop token if one ended the reply.
    pub tokens: Vec<u32>,
    pub finish: StopReason,
    /// Tool calls the reply ended on (only with [`ChatConfig::tools`]).
    pub tool_calls: Vec<ToolCall>,
    pub stats: Stats,
}

/// Model family of a loaded engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Gemma4,
    Gemma3,
    Qwen,
    /// Llama 3 (Core ML models only).
    Llama,
}

/// Speculative decoding in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speculative {
    None,
    /// Qwen multi-token prediction layer.
    Mtp,
    /// Gemma 4 assistant model.
    Drafter,
}

/// What was loaded.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ModelInfo {
    /// The local checkpoint (directory or GGUF file).
    pub path: PathBuf,
    pub model_type: String,
    pub family: Family,
    pub layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    /// Precision of the streamed weights (Core ML models: `Bf16`, meaningless there).
    pub quant: Quant,
    /// Checkpoint format: "gguf", "mlx", "oxinfer" (pre-quantized), "safetensors" or "coreml".
    pub format: &'static str,
    pub checkpoint_bytes: u64,
    /// Weight bytes read per decoded token.
    pub streamed_bytes: usize,
    /// "cpu" or the GPU backend's name (used for prefill matmuls); `coreml-<units>` for Core ML.
    pub device: String,
    /// CPU worker threads (0 for Core ML models, which do not use the pool).
    pub threads: usize,
    pub mmap: bool,
    pub speculative: Speculative,
    pub load_secs: f64,
    /// Set for Core ML models.
    pub coreml: Option<CoreMlInfo>,
}

/// Stops the running (or next) generation of an engine from any thread.
#[derive(Clone, Debug)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    /// Requests the current reply to stop after the token being generated (or, during the
    /// prefill, at its first token). Each turn clears the request when it starts, so one made
    /// while no turn runs has no effect.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Owns heap values that `'static` references handed out by [`Arena::keep`] point into, and
/// frees them in reverse order (models before the files they borrow).
struct Arena(Vec<(*mut (), Dropper)>);

/// Frees one value of the arena.
type Dropper = unsafe fn(*mut ());

// SAFETY: only `Send + Sync` values are stored, and the pointers are not used by the arena
// except to drop them.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    fn keep<T: Send + Sync + 'static>(&mut self, v: T) -> &'static T {
        unsafe fn drop_box<T>(p: *mut ()) {
            drop(unsafe { Box::from_raw(p as *mut T) });
        }
        let p = Box::into_raw(Box::new(v));
        self.0.push((p as *mut (), drop_box::<T>));
        // SAFETY: the box lives until the arena drops, after every user of the reference (the
        // engine declares the arena as its last field).
        unsafe { &*p }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        while let Some((p, d)) = self.0.pop() {
            unsafe { d(p) }
        }
    }
}

enum Lm {
    Gemma {
        model: &'static Model<'static>,
        sess: Session,
        drafter: Option<&'static Drafter<'static>>,
    },
    Qwen {
        model: &'static QwenModel<'static>,
        sess: QwenSession,
    },
    QwenMtp {
        dec: Box<MtpDecoder<'static>>,
    },
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
    CoreMl(Box<coreml_lm::CoreMlLm>),
}

#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
#[path = "engine_coreml.rs"]
mod coreml_lm;

impl Lm {
    fn context_len(&self) -> usize {
        match self {
            Lm::Gemma { sess, .. } => sess.pos,
            Lm::Qwen { sess, .. } => sess.pos,
            Lm::QwenMtp { dec } => dec.sess.pos,
            #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
            Lm::CoreMl(c) => c.context_len(),
        }
    }
    fn reset(&mut self) {
        match self {
            Lm::Gemma { model, sess, .. } => *sess = model.new_session(),
            Lm::Qwen { model, sess } => *sess = model.new_session(),
            Lm::QwenMtp { dec } => {
                let opts = dec.opts.clone();
                **dec = MtpDecoder::new(dec.model(), dec.mtp(), opts);
            }
            #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
            Lm::CoreMl(c) => c.reset(),
        }
    }
}

/// Serializes forward passes across engines (the thread pool runs one region at a time).
static RUN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A loaded model with one conversation.
pub struct Engine {
    lm: Lm,
    tok: Tokenizer,
    fmt: &'static (dyn ChatFormat + Sync),
    stop: Vec<u32>,
    sampler: Sampler,
    sampling: SamplingParams,
    config: ChatConfig,
    info: ModelInfo,
    files: &'static ModelFiles,
    prefill_chunk: usize,
    max_context: Option<usize>,
    draft_tokens: usize,
    /// Turns fed in the current conversation.
    turns: usize,
    /// The last reply's final token when the model has not run it (see [`Decoded::unfed`]).
    unfed: Option<u32>,
    cancel: Arc<AtomicBool>,
    stats: Stats,
    /// The weight streamer (`EngineOptions::stream`), stopped on drop before the files go.
    streamer: Option<Arc<crate::stream::Streamer>>,
    /// Must stay last: the model, sessions and references above point into it.
    _arena: Arena,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // The model holds handles too; stop the reader thread before the arena unmaps the files.
        if let Some(s) = &self.streamer {
            s.shutdown();
        }
    }
}

impl Engine {
    /// Loads a model: a local directory, a `.gguf` file, or (with the `hf` feature) a repo id
    /// (`org/name`, or `org/name:QUANT` for a GGUF repo) fetched into the cache.
    pub fn load(model: &str, opts: &EngineOptions) -> Result<Engine> {
        let t0 = Instant::now();
        let dir = resolve_model(model, opts.cache_dir.as_deref())?;
        let side_dir = opts
            .mtp_from
            .as_deref()
            .map(|s| resolve_model(s, opts.cache_dir.as_deref()))
            .transpose()?;
        let draft_dir = opts
            .draft
            .as_deref()
            .map(|s| resolve_model(s, opts.cache_dir.as_deref()))
            .transpose()?;
        if is_coreml_dir(&dir) {
            return Self::load_coreml(dir, opts, t0);
        }
        if let Some(t) = opts.threads {
            pool::configure(t);
        }
        if let Some(on) = opts.gpu_exact {
            crate::gpu::set_exact(on);
        }
        if let Some(on) = opts.gpu_moe {
            crate::gpu::set_moe_offload(on);
        }
        let gpu = if crate::gpu::initialized() {
            crate::gpu::active()
        } else {
            crate::gpu::init(&crate::platform::detect(), opts.device)
        };
        if crate::sd::is_diffusers(&dir) {
            bail!("{}: a diffusion pipeline; use oxinfer::sd::Pipeline", dir.display());
        }
        if crate::embed::is_embedding_model(&dir) {
            bail!("{}: an embedding model; use oxinfer::embed::Embedder", dir.display());
        }
        let model_type = config::model_type(&dir)?;
        let family = match model_type.as_str() {
            "gemma4" | "gemma4_text" => Family::Gemma4,
            "gemma3" | "gemma3_text" => Family::Gemma3,
            "qwen3" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_6" | "qwen3_8" => Family::Qwen,
            other => bail!("unsupported model_type {other}"),
        };
        let mtp = opts.mtp || side_dir.is_some();
        if mtp && family != Family::Qwen {
            bail!("MTP speculative decoding is for Qwen3.5/3.8 models");
        }
        if draft_dir.is_some() && family != Family::Gemma4 {
            bail!("a drafter model is for Gemma 4");
        }

        let mut arena = Arena(Vec::new());
        let mut streamer = None;
        // Streaming reads the weights in place: always mapped.
        let files = arena.keep(ModelFiles::open(&dir, !(opts.mmap || opts.stream.is_some()))?);
        let tok = load_tokenizer(&dir)?;
        let eos = config::load_eos_ids(&dir);
        let (lm, fmt, quant, layers, hidden, vocab, streamed): (Lm, &'static (dyn ChatFormat + Sync), _, _, _, _, _) = match family {
            Family::Gemma4 | Family::Gemma3 => {
                let model = arena.keep(Model::load(config::TextConfig::load(&dir)?, files, opts.quant)?);
                let drafter = match &draft_dir {
                    Some(d) => {
                        let df = arena.keep(ModelFiles::open(d, true)?);
                        let dr = Drafter::load(d, df)?;
                        if dr.backbone_hidden != model.cfg.hidden_size {
                            bail!(
                                "drafter backbone size {} does not match model hidden size {}",
                                dr.backbone_hidden,
                                model.cfg.hidden_size
                            );
                        }
                        Some(arena.keep(dr))
                    }
                    None => None,
                };
                let fmt: &'static (dyn ChatFormat + Sync) = if model.cfg.gemma3 { &chat::Gemma3Chat } else { &chat::GemmaChat };
                let c = &model.cfg;
                let info = (
                    model.quant,
                    c.num_hidden_layers,
                    c.hidden_size,
                    c.vocab_size,
                    model.streamed_bytes(),
                );
                let lm = Lm::Gemma {
                    model,
                    sess: model.new_session(),
                    drafter,
                };
                (lm, fmt, info.0, info.1, info.2, info.3, info.4)
            }
            Family::Llama => bail!("Llama checkpoints run as Core ML models only"),
            Family::Qwen => {
                let mut model = QwenModel::load(config::QwenConfig::load(&dir)?, files, opts.quant)?;
                let fmt: &'static (dyn ChatFormat + Sync) = chat::QwenChat::for_model(&dir);
                let side = match (&side_dir, mtp) {
                    (Some(d), true) => Some(arena.keep(ModelFiles::open(d, false)?)),
                    _ => None,
                };
                let mut m = if mtp {
                    let mut m = model.load_mtp(side.unwrap_or(files))?;
                    if let Some(side) = side {
                        m.attach_sidecar(&model.cfg, side, model.quant)?;
                    }
                    Some(m)
                } else {
                    None
                };
                // The streamer is attached before the model and MTP layer are frozen in the arena.
                let draft_tokens = opts.draft_tokens.unwrap_or(if opts.stream.is_some() { 8 } else { 3 });
                if let Some(so) = &opts.stream {
                    let mut fs = vec![files];
                    fs.extend(side);
                    streamer = Some(crate::stream::attach_qwen(&mut model, m.as_mut(), &fs, draft_tokens, so.clone())?);
                }
                let model = arena.keep(model);
                let c = &model.cfg;
                let info = (
                    model.quant,
                    c.num_hidden_layers,
                    c.hidden_size,
                    c.vocab_size,
                    model.streamed_bytes(),
                );
                let lm = match m {
                    Some(m) => {
                        let m: &'static QwenMtp<'static> = arena.keep(m);
                        let stop = chat::with_turn_end(fmt, &tok, eos.clone());
                        let mopts = MtpOptions {
                            draft_tokens,
                            adaptive: !opts.mtp_fixed,
                            stop,
                            // Streaming passes are IO-bound: long drafts pay off, so pick their
                            // length by cost and keep them exact.
                            cost_aware: !opts.mtp_fixed && opts.stream.is_some(),
                            exact: opts.stream.is_some(),
                        };
                        Lm::QwenMtp {
                            dec: Box::new(MtpDecoder::new(model, m, mopts)),
                        }
                    }
                    None => Lm::Qwen {
                        model,
                        sess: model.new_session(),
                    },
                };
                (lm, fmt, info.0, info.1, info.2, info.3, info.4)
            }
        };
        if opts.preload && opts.mmap {
            files.preload();
        }
        let stop = chat::with_turn_end(fmt, &tok, eos);
        let format = if files.metadata("format") == Some("gguf") {
            "gguf"
        } else if files.is_mlx() {
            "mlx"
        } else if Quant::stored(files).is_some() {
            "oxinfer"
        } else {
            "safetensors"
        };
        let speculative = match &lm {
            Lm::QwenMtp { .. } => Speculative::Mtp,
            Lm::Gemma { drafter: Some(_), .. } => Speculative::Drafter,
            _ => Speculative::None,
        };
        let info = ModelInfo {
            path: dir.clone(),
            model_type,
            family,
            layers,
            hidden_size: hidden,
            vocab_size: vocab,
            quant,
            format,
            checkpoint_bytes: files.total_bytes(),
            streamed_bytes: streamed,
            device: gpu.map(|b| b.name().to_string()).unwrap_or_else(|| "cpu".into()),
            threads: pool::threads(),
            mmap: opts.mmap,
            speculative,
            load_secs: t0.elapsed().as_secs_f64(),
            coreml: None,
        };
        let sampling = SamplingParams::default();
        Ok(Engine {
            lm,
            tok,
            fmt,
            stop,
            sampler: sampler_for(&sampling),
            sampling,
            config: ChatConfig::default(),
            info,
            files,
            // Bigger prefill chunks keep the GPU busier; on the CPU 256 keeps activations in cache.
            prefill_chunk: opts.prefill_chunk.unwrap_or(if gpu.is_some() { 1024 } else { 256 }),
            max_context: opts.max_context,
            draft_tokens: opts.draft_tokens.unwrap_or(6),
            turns: 0,
            unfed: None,
            cancel: Arc::new(AtomicBool::new(false)),
            stats: Stats::default(),
            streamer,
            _arena: arena,
        })
    }

    /// A Core ML model directory (see the module docs).
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
    fn load_coreml(dir: PathBuf, opts: &EngineOptions, t0: Instant) -> Result<Engine> {
        let l = coreml_lm::load(&dir, opts)?;
        let ctx = l.info.context_length;
        let info = ModelInfo {
            path: dir,
            model_type: l.meta.architecture.clone(),
            family: l.family,
            layers: l.layers,
            hidden_size: l.hidden_size,
            vocab_size: l.vocab_size,
            quant: Quant::Bf16,
            format: "coreml",
            checkpoint_bytes: l.bytes,
            streamed_bytes: 0,
            device: format!("coreml-{}", l.info.units.as_str()),
            threads: 0,
            mmap: false,
            speculative: Speculative::None,
            load_secs: t0.elapsed().as_secs_f64(),
            coreml: Some(l.info),
        };
        let mut arena = Arena(Vec::new());
        let files = arena.keep(ModelFiles::empty());
        let sampling = SamplingParams::default();
        Ok(Engine {
            lm: Lm::CoreMl(Box::new(l.lm)),
            tok: l.tok,
            fmt: l.fmt,
            stop: l.stop,
            sampler: sampler_for(&sampling),
            sampling,
            config: ChatConfig::default(),
            info,
            files,
            prefill_chunk: 0,
            // The conversion fixes the context.
            max_context: Some(opts.max_context.map_or(ctx, |m| m.min(ctx))),
            draft_tokens: 0,
            turns: 0,
            unfed: None,
            cancel: Arc::new(AtomicBool::new(false)),
            stats: Stats::default(),
            streamer: None,
            _arena: arena,
        })
    }

    #[cfg(not(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64")))]
    fn load_coreml(dir: PathBuf, _opts: &EngineOptions, _t0: Instant) -> Result<Engine> {
        bail!(
            "{}: a Core ML model; these run on Apple Silicon macOS / iOS (built with the `gpu` feature)",
            dir.display()
        )
    }

    /// The weight streamer when loaded with [`EngineOptions::stream`]: per-step IO statistics
    /// ([`crate::stream::Streamer::stats`], subtract two snapshots), the plan, the page-cache
    /// footprint, and the emulated flash bandwidth.
    pub fn streamer(&self) -> Option<&Arc<crate::stream::Streamer>> {
        self.streamer.as_ref()
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// The chat format (prompt rendering) of the model.
    pub fn chat_format(&self) -> &'static (dyn ChatFormat + Sync) {
        self.fmt
    }

    /// The checkpoint's tensors (e.g. for [`save_quantized`]).
    pub fn files(&self) -> &ModelFiles {
        self.files
    }

    /// Token ids that end a reply.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop
    }

    /// Replaces the sampler (and reseeds it).
    pub fn set_sampling(&mut self, s: SamplingParams) {
        self.sampling = s;
        self.sampler = sampler_for(&s);
    }

    pub fn sampling(&self) -> SamplingParams {
        self.sampling
    }

    pub fn config(&self) -> &ChatConfig {
        &self.config
    }

    /// Conversation settings (see [`ChatConfig`] for when changes apply).
    pub fn config_mut(&mut self) -> &mut ChatConfig {
        &mut self.config
    }

    /// A handle that stops generation from another thread.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancel.clone())
    }

    /// Statistics of the last turn.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Tokens in the conversation so far.
    pub fn context_len(&self) -> usize {
        self.lm.context_len()
    }

    /// Starts a new conversation (drops the cache).
    pub fn reset(&mut self) {
        self.lm.reset();
        self.turns = 0;
        self.unfed = None;
    }

    /// The text [`Engine::send`] would feed for `user` (the chat template applied).
    pub fn prompt_text(&self, user: &str) -> String {
        let (fmt, c) = (self.fmt, &self.config);
        if self.turns == 0 {
            let system = match &c.system {
                SystemPrompt::Default => Some(crate::tools::default_system_prompt(!c.tools.is_empty())),
                SystemPrompt::None => None,
                SystemPrompt::Text(t) => Some(t.clone()),
            };
            fmt.first_turn(system.as_deref(), c.think, &c.tools, user)
        } else {
            // The previous reply's stop token was sampled but never fed.
            fmt.end_of_model_turn() + &fmt.user_turn(user, c.think)
        }
    }

    /// Sends a user message and generates the reply, streaming it to `on_delta`.
    pub fn send(&mut self, user: &str, on_delta: &mut dyn FnMut(&Delta) -> Flow) -> Result<Reply> {
        self.send_with(user, &[], &[], on_delta)
    }

    /// [`Engine::send`] with image and audio attachments (Gemma 4 with the `media` feature),
    /// placed before the message text.
    pub fn send_with(
        &mut self,
        user: &str,
        images: &[PathBuf],
        audios: &[PathBuf],
        on_delta: &mut dyn FnMut(&Delta) -> Flow,
    ) -> Result<Reply> {
        let (mm_text, feats) = if images.is_empty() && audios.is_empty() {
            (String::new(), Vec::new())
        } else {
            if self.info.family != Family::Gemma4 || self.info.coreml.is_some() {
                bail!("image and audio inputs are supported for Gemma 4 only (not with Core ML)");
            }
            let req = mm::MmRequest {
                model: &self.info.path,
                images,
                audios,
                max_soft_tokens: 0,
                dump_mm: None,
            };
            mm::prepare(&req, self.files, self.info.hidden_size)?
        };
        let text = self.prompt_text(&format!("{mm_text}{user}"));
        let think = self.config.think;
        let reply = self.turn(&text, &feats, think)?.finish(self, on_delta)?;
        self.turns += 1;
        Ok(reply)
    }

    /// Feeds the results of the calls a reply ended on; the model continues its answer.
    pub fn send_tool_results(&mut self, results: &[ToolResult], on_delta: &mut dyn FnMut(&Delta) -> Flow) -> Result<Reply> {
        let text = self.fmt.tool_results(results, self.config.think);
        let think = self.config.think;
        self.turn(&text, &[], think)?.finish(self, on_delta)
    }

    /// Feeds `prompt` as is (no chat template; `<bos>` first for Gemma in an empty context) and
    /// generates a continuation. Call [`Engine::reset`] first for a fresh context.
    pub fn complete(&mut self, prompt: &str, on_delta: &mut dyn FnMut(&Delta) -> Flow) -> Result<Reply> {
        let gemma = matches!(self.info.family, Family::Gemma4 | Family::Gemma3);
        let text = if self.context_len() == 0 && self.unfed.is_none() && gemma {
            format!("<bos>{prompt}")
        } else {
            prompt.to_string()
        };
        self.turn(&text, &[], false)?.finish(self, on_delta)
    }

    /// Encodes `text` (after the unfed token of the last reply, if any) and checks the context
    /// limit; the actual work happens in [`PendingTurn::finish`].
    fn turn(&mut self, text: &str, feats: &[mm::MmFeature], think: bool) -> Result<PendingTurn> {
        let mut ids = self.tok.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec();
        if ids.is_empty() {
            bail!("empty prompt");
        }
        if let Some(t) = self.unfed {
            ids.insert(0, t);
        }
        let pos = self.context_len();
        if let Some(max) = self.max_context
            && pos + ids.len() > max
        {
            bail!(
                "context full: {pos} tokens + {} new exceed max_context {max}; reset the conversation",
                ids.len()
            );
        }
        let overrides = if feats.is_empty() {
            Vec::new()
        } else {
            mm::overrides(&ids, feats, self.info.hidden_size)?
        };
        if !overrides.is_empty() && matches!(self.lm, Lm::Qwen { .. } | Lm::QwenMtp { .. }) {
            bail!("attachments are not supported for this model");
        }
        self.unfed = None;
        Ok(PendingTurn { ids, overrides, think })
    }
}

/// A turn whose prompt is encoded, ready to run.
struct PendingTurn {
    ids: Vec<u32>,
    overrides: Vec<(usize, Vec<f32>)>,
    think: bool,
}

impl PendingTurn {
    fn finish(self, e: &mut Engine, on_delta: &mut dyn FnMut(&Delta) -> Flow) -> Result<Reply> {
        // Core ML models do not use the thread pool.
        let _run = e.info.coreml.is_none().then(|| RUN.lock().unwrap_or_else(|p| p.into_inner()));
        let PendingTurn { ids, overrides, think } = self;
        let Engine {
            lm,
            tok,
            fmt,
            stop,
            sampler,
            config,
            prefill_chunk,
            max_context,
            draft_tokens,
            cancel,
            ..
        } = e;
        let tok: &Tokenizer = tok;
        let chunk = *prefill_chunk;
        cancel.store(false, Ordering::SeqCst);

        let mut stream = TextStream::new(*fmt, think, tok);
        let (mut text, mut thinking) = (String::new(), String::new());
        let cancel_flag = cancel.clone();
        let mut cb = |out: &[u32]| -> GFlow {
            let id = *out.last().unwrap();
            stream.on_token(id);
            let d = stream.delta(tok, out, false);
            let th = stream.thinking();
            if th {
                thinking.push_str(&d);
            } else {
                text.push_str(&d);
            }
            let f = on_delta(&Delta {
                token: Some(id),
                text: &d,
                thinking: th,
            });
            if cancel_flag.load(Ordering::SeqCst) { GFlow::Stop } else { f }
        };

        let t0 = Instant::now();
        let mut st = Stats {
            prompt_tokens: ids.len(),
            ..Stats::default()
        };
        let (max_context, max_tokens) = (*max_context, config.max_tokens);
        let limit = |pos: usize| match max_context {
            Some(m) => max_tokens.min(m.saturating_sub(pos)),
            None => max_tokens,
        };
        let empty = |secs: f64| Decoded {
            tokens: Vec::new(),
            forwards: 0,
            secs,
            reason: StopReason::Length,
            unfed: None,
        };
        let d: Decoded = match lm {
            Lm::Gemma { model, sess, drafter } => {
                let hs = generate::prefill(*model, sess, &ids, chunk, &overrides);
                let h = model.cfg.hidden_size;
                let last = hs[hs.len() - h..].to_vec();
                st.prefill_secs = t0.elapsed().as_secs_f64();
                let max = limit(sess.pos);
                match *drafter {
                    _ if max == 0 => empty(0.0),
                    Some(dr) => {
                        let base = sess.pos;
                        let (d, ss) = generate::decode_speculative(model, dr, sess, last, *draft_tokens, stop, max, &mut cb);
                        // Keep the reply in the cache up to (not including) its last token.
                        sess.truncate(base + d.tokens.len() - 1);
                        st.drafted = ss.drafted;
                        st.accepted = ss.accepted;
                        d
                    }
                    None => {
                        let logits = model.logits(&last, 1);
                        generate::decode(*model, sess, logits, sampler, stop, max, &mut cb)
                    }
                }
            }
            Lm::Qwen { model, sess } => {
                let hs = generate::prefill(*model, sess, &ids, chunk, &[]);
                let logits = model.logits(&hs[hs.len() - model.cfg.hidden_size..], 1);
                st.prefill_secs = t0.elapsed().as_secs_f64();
                let max = limit(sess.pos);
                if max == 0 {
                    empty(0.0)
                } else {
                    generate::decode(*model, sess, logits, sampler, stop, max, &mut cb)
                }
            }
            Lm::QwenMtp { dec } => {
                let logits = dec.prefill(&ids, chunk);
                st.prefill_secs = t0.elapsed().as_secs_f64();
                let max = limit(dec.sess.pos);
                if max == 0 {
                    empty(0.0)
                } else {
                    let before = dec.stats.clone();
                    let d = generate::decode_mtp(dec, &logits, sampler, stop, max, &mut cb);
                    st.drafted = dec.stats.drafted - before.drafted;
                    st.accepted = dec.stats.accepted - before.accepted;
                    d
                }
            }
            #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
            Lm::CoreMl(c) => {
                let next = c.prefill(&ids)?;
                st.prefill_secs = t0.elapsed().as_secs_f64();
                let max = limit(c.context_len());
                if max == 0 {
                    empty(0.0)
                } else {
                    c.decode(next, sampler, stop, max, &mut cb)?
                }
            }
        };
        // Flush text held back for an incomplete character.
        let shown = match d.tokens.last() {
            Some(t) if stop.contains(t) => &d.tokens[..d.tokens.len() - 1],
            _ => &d.tokens[..],
        };
        let rest = stream.delta(tok, shown, true);
        if !rest.is_empty() {
            let th = stream.thinking();
            if th {
                thinking.push_str(&rest);
            } else {
                text.push_str(&rest);
            }
            on_delta(&Delta {
                token: None,
                text: &rest,
                thinking: th,
            });
        }
        st.generated_tokens = shown.len();
        st.decode_secs = d.secs;
        st.forwards = d.forwards;
        st.context_tokens = lm.context_len() + usize::from(d.unfed.is_some());
        let tool_calls = if config.tools.is_empty() {
            Vec::new()
        } else {
            let raw = tok.decode(&d.tokens, false).map_err(|e| anyhow!("decode: {e}"))?;
            fmt.parse_tool_calls(&raw)
        };
        e.unfed = d.unfed;
        e.stats = st.clone();
        Ok(Reply {
            text,
            thinking,
            tokens: d.tokens,
            finish: d.reason,
            tool_calls,
            stats: st,
        })
    }
}

fn sampler_for(s: &SamplingParams) -> Sampler {
    Sampler::new(s.temperature, s.top_k, s.top_p, s.seed)
}

/// Whether `dir` is a Core ML (ANEMLL-format) model directory: `meta.yaml` next to `.mlmodelc`
/// parts (checked on every platform, so other targets can say why it does not load).
fn is_coreml_dir(dir: &Path) -> bool {
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
    {
        coreml_lm::detect(dir)
    }
    #[cfg(not(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64")))]
    {
        dir.join("meta.yaml").is_file()
            && std::fs::read_dir(dir)
                .map(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".mlmodelc")))
                .unwrap_or(false)
    }
}

/// Resolves a model spec to a local checkpoint: a directory or GGUF file, or (with the `hf`
/// feature) a repo id (`org/name`, `org/name:QUANT`) fetched into `cache_dir` (default
/// [`crate::hf::cache_root`]).
pub fn resolve_model(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    #[cfg(feature = "hf")]
    {
        crate::hf::resolve(spec, cache_dir)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = cache_dir;
        let p = PathBuf::from(spec);
        if p.is_dir() || crate::gguf::is_gguf(&p) {
            Ok(p)
        } else {
            bail!("{spec}: not a directory or GGUF file (built without the `hf` feature, so repo ids cannot be fetched)")
        }
    }
}

/// Resolves a file spec: a local path, or (with the `hf` feature) `org/repo/path/in/repo`.
pub fn resolve_file(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    #[cfg(feature = "hf")]
    {
        crate::hf::resolve_file(spec, cache_dir)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = cache_dir;
        let p = PathBuf::from(spec);
        if p.is_file() {
            Ok(p)
        } else {
            bail!("{spec}: no such file (built without the `hf` feature, so repo files cannot be fetched)")
        }
    }
}

/// The model's tokenizer: `tokenizer.json`, or rebuilt from a GGUF file's vocabulary.
pub fn load_tokenizer(model: &Path) -> Result<Tokenizer> {
    if crate::gguf::is_gguf(model) {
        return crate::gguf::header(model)?.tokenizer();
    }
    Tokenizer::from_bytes(crate::fsx::read(&model.join("tokenizer.json"))?).map_err(|e| anyhow!("tokenizer: {e}"))
}

/// Writes the tensors the loader used, quantized as loaded, plus the config and tokenizer files:
/// a checkpoint that loads zero-copy with no quantization pass.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn save_quantized(src: &Path, out: &Path, files: &ModelFiles, quant: Quant) -> Result<()> {
    if out.exists() && std::fs::canonicalize(out)? == std::fs::canonicalize(src)? {
        bail!("refusing to write into the source directory {}", src.display());
    }
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let t0 = Instant::now();
    let table = crate::model::table_quant(quant);
    let source = src.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let written = files.write_quantized(
        out,
        &[
            ("format", "pt"),
            ("oxinfer_quant", quant.as_str()),
            ("oxinfer_table_quant", table.as_str()),
            ("oxinfer_source", &source),
        ],
    )?;
    let mut copied = Vec::new();
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        let keep = (name.ends_with(".json") && name != "model.safetensors.index.json") || name.ends_with(".jinja");
        if keep && e.file_type()?.is_file() {
            std::fs::copy(e.path(), out.join(&name))?;
            copied.push(name);
        }
    }
    copied.sort();
    for (p, size) in &written {
        eprintln!("[wrote {} ({:.2} GB)]", p.display(), *size as f64 / 1e9);
    }
    eprintln!(
        "[{} weights, tables {}; copied {}; {:.1}s]",
        quant.as_str(),
        table.as_str(),
        copied.join(", "),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn engine_is_send() {
        assert_send::<Engine>();
        assert_send::<CancelHandle>();
    }

    #[test]
    fn options_set() {
        let mut o = EngineOptions::default();
        o.set("quant", "q4").unwrap();
        o.set("threads", "4").unwrap();
        o.set("mtp_from", "org/repo").unwrap();
        o.set("max_context", "0").unwrap();
        assert_eq!(o.quant, Quant::Q4);
        assert_eq!(o.threads, Some(4));
        assert!(o.mtp);
        assert_eq!(o.max_context, None);
        assert!(o.set("nope", "1").is_err());
        assert!(o.set("mmap", "maybe").is_err());
        o.set("coreml_units", "ane").unwrap();
        o.set("coreml_prefill", "single").unwrap();
        assert_eq!((o.coreml_units, o.coreml_prefill), (Some(CoreMlUnits::Ane), CoreMlPrefill::Single));
        o.set("coreml_units", "auto").unwrap();
        assert_eq!(o.coreml_units, None);
        assert!(o.set("coreml_units", "npu").is_err());
        assert!(o.set("coreml_prefill", "fast").is_err());
        for u in [CoreMlUnits::Ane, CoreMlUnits::All, CoreMlUnits::Gpu, CoreMlUnits::Cpu] {
            assert_eq!(CoreMlUnits::parse(u.as_str()), Some(u));
        }
        for m in [CoreMlPrefill::Auto, CoreMlPrefill::Batch, CoreMlPrefill::Pad, CoreMlPrefill::Single] {
            assert_eq!(CoreMlPrefill::parse(m.as_str()), Some(m));
        }
        for k in EngineOptions::KEYS {
            // Every advertised key is accepted with some value.
            let v = match *k {
                "quant" => "q8",
                "device" => "cpu",
                "coreml_units" => "gpu",
                "coreml_prefill" => "pad",
                "mtp_from" | "draft" | "cache_dir" => "x",
                _ => "1",
            };
            o.set(k, v).unwrap();
        }
    }
}
