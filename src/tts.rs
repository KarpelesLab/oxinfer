//! Text to speech with Orpheus TTS (canopylabs): a Llama 3.2 3B fine-tune that writes SNAC
//! audio codes as extra vocabulary tokens, decoded to 24 kHz audio by the SNAC codec
//! ([`crate::snac`], `hubertsiuzdak/snac_24khz` by default).
//!
//! The prompt and token layout follow the official `orpheus_tts` package (engine_class.py,
//! decoder.py):
//! - prompt: start of human (128259), the tokenized `"{voice}: {text}"` with its
//!   `<|begin_of_text|>`, end of text (128009), end of human (128260), start of AI (128261),
//!   start of speech (128257);
//! - the reply: 7 tokens per frame (one code of codebook 0, two of codebook 1, four of codebook 2,
//!   in the order 0 1 2 2 1 2 2), token `128266 + position * 4096 + code`, then end of speech
//!   (128258);
//! - sampling: temperature 0.6, top-p 0.8, repetition penalty 1.1 (over the prompt and the
//!   reply, as transformers and vLLM apply it), at most 1200 tokens (about 14 s of audio).
//!
//! Streaming ([`Speaker::speak_stream`]) decodes overlapping windows of frames as tokens arrive,
//! as the official decoder does (it decodes the last 4 frames and keeps the second one: one frame
//! of left context and two of lookahead); here the first and last frames are emitted too, and
//! the context sizes are options ([`SpeakOptions::context_left`] / `context_right`). The
//! decoder's receptive field is under 3 frames each way: with 3 / 3 the stream is the full decode
//! (to 3e-7); the default 2 / 2 is within 1.7e-3 of it (59 dB SNR), the official 1 / 2 within
//! 2.4e-2 (27 dB).
//!
//! ```no_run
//! use oxinfer::tts::{Speaker, SpeakerOptions, SpeakOptions};
//! # fn main() -> anyhow::Result<()> {
//! let speaker = Speaker::load("unsloth/orpheus-3b-0.1-ft", &SpeakerOptions::default())?;
//! let mut opts = SpeakOptions::default();
//! opts.voice = Some("leah".into());
//! let audio = speaker.speak("Hello there, how are you today?", &opts)?;
//! audio.write_wav("hello.wav")?;
//! # Ok(())
//! # }
//! ```

use crate::clock::Instant;
use crate::engine::{CancelHandle, Flow, RUN, StopReason};
use crate::model::{Lin, Quant, linear};
use crate::platform::DevicePref;
use crate::pool;
use crate::qwen::{QwenModel, QwenSession};
use crate::safetensors::ModelFiles;
use crate::sampler::{Sampler, apply_repetition_penalty};
use crate::snac::{Noise, Snac, SnacOptions};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokenizers::Tokenizer;

/// `<|begin_of_text|>`.
pub const BOS: u32 = 128000;
/// `<|eot_id|>`, closing the text of the prompt.
pub const END_OF_TEXT: u32 = 128009;
pub const START_OF_SPEECH: u32 = 128257;
pub const END_OF_SPEECH: u32 = 128258;
pub const START_OF_HUMAN: u32 = 128259;
pub const END_OF_HUMAN: u32 = 128260;
pub const START_OF_AI: u32 = 128261;
pub const END_OF_AI: u32 = 128262;
/// Token of code 0 at frame position 0 (`<custom_token_10>`).
pub const AUDIO_BASE: u32 = 128266;
/// Codes per codebook.
pub const CODES: u32 = 4096;
/// Tokens per frame.
pub const FRAME_TOKENS: usize = 7;
/// Tokens that end the reply.
pub const STOP_TOKENS: [u32; 3] = [END_OF_SPEECH, END_OF_AI, END_OF_TEXT];

/// The voices of `canopylabs/orpheus-3b-0.1-ft` (English). Other checkpoints (the multilingual
/// research models) have their own; any name is passed through.
pub const VOICES: [&str; 8] = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"];

/// The default SNAC decoder.
pub const DEFAULT_SNAC: &str = "hubertsiuzdak/snac_24khz";

/// How to load a speaker. `SpeakerOptions::default()` then set fields, or
/// [`SpeakerOptions::set`] key-value pairs.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SpeakerOptions {
    /// CPU worker threads (process-wide; default: all cores). Key `threads`.
    pub threads: Option<usize>,
    /// Compute device for the prefill matmuls (process-wide; set by the first model loaded).
    /// Key `device`.
    pub device: DevicePref,
    /// Weight precision of the language model (bf16 checkpoints). Key `quant`. Default bf16.
    pub quant: Quant,
    /// Map the weights instead of reading them into memory. Key `mmap`.
    pub mmap: bool,
    /// The SNAC decoder: a directory or repo id (default [`DEFAULT_SNAC`]). Key `snac`.
    pub snac: Option<String>,
    /// SNAC matmul weights in f32 (exact, the default) or bf16. Key `snac_f32`.
    pub snac_f32: bool,
    /// Where repo ids are downloaded. Key `cache_dir`.
    pub cache_dir: Option<PathBuf>,
}

impl Default for SpeakerOptions {
    fn default() -> Self {
        Self {
            threads: None,
            device: DevicePref::Auto,
            quant: Quant::Bf16,
            mmap: cfg!(target_os = "ios"),
            snac: None,
            snac_f32: true,
            cache_dir: None,
        }
    }
}

fn parse_bool(key: &str, v: &str) -> Result<bool> {
    match v {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{key}: expected a boolean, got {v:?}"),
    }
}

impl SpeakerOptions {
    /// Sets an option by key (`threads`, `device`, `quant`, `mmap`, `snac`, `snac_f32`,
    /// `cache_dir`).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "threads" => self.threads = Some(value.parse().with_context(|| format!("threads: {value:?}"))?),
            "device" => self.device = DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?}"))?,
            "quant" => self.quant = Quant::parse(value).with_context(|| format!("quant: unknown quantization {value:?} (bf16, q8, q4)"))?,
            "mmap" => self.mmap = parse_bool(key, value)?,
            "snac" => self.snac = Some(value.to_string()),
            "snac_f32" => self.snac_f32 = parse_bool(key, value)?,
            "cache_dir" => self.cache_dir = Some(PathBuf::from(value)),
            _ => bail!("unknown option {key} (threads, device, quant, mmap, snac, snac_f32, cache_dir)"),
        }
        Ok(())
    }
}

/// Options of one utterance. Deserializes from JSON with every field optional (the C API's
/// option format), e.g. `{"voice": "leo", "temperature": 0.7, "seed": 3}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct SpeakOptions {
    /// Voice name put before the text (`"tara: ..."`); None or empty: no voice prefix.
    pub voice: Option<String>,
    /// 0 = greedy.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = no top-k cut (the official settings have none).
    pub top_k: usize,
    /// Over the prompt and the generated tokens (1 = off). Orpheus needs at least 1.1 for
    /// stable output; higher values speak faster.
    pub repetition_penalty: f32,
    /// Generated tokens at most (7 per frame of 2048 samples: 1200 is about 14 s at 24 kHz).
    pub max_tokens: usize,
    pub seed: u64,
    /// Only sample tokens the format allows at each position (the audio codes of the frame
    /// position, or a stop token), which also skips most of the output layer. Off: sample from
    /// the whole vocabulary as the reference does (tokens out of place are then dropped).
    pub constrained: bool,
    /// The SNAC noise blocks' noise (as the reference decoder adds it); off for a deterministic
    /// waveform independent of the seed.
    pub noise: bool,
    /// Streaming: frames of left context decoded with each window (default 2; the official
    /// decoder: 1; 3 makes the stream identical to a full decode).
    pub context_left: usize,
    /// Streaming: frames of lookahead before a frame is emitted (default and official decoder: 2;
    /// 3 for an exact stream).
    pub context_right: usize,
    /// Streaming: frames emitted per window (1 = lowest latency; more = fewer, larger windows).
    pub chunk_frames: usize,
}

impl Default for SpeakOptions {
    fn default() -> Self {
        Self {
            voice: Some("tara".into()),
            temperature: 0.6,
            top_p: 0.8,
            top_k: 0,
            repetition_penalty: 1.1,
            max_tokens: 1200,
            seed: 42,
            constrained: true,
            noise: true,
            context_left: 2,
            context_right: 2,
            chunk_frames: 1,
        }
    }
}

impl SpeakOptions {
    /// Options from a JSON object (unset fields keep their defaults).
    pub fn from_json(json: &str) -> Result<Self> {
        if json.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(json).context("speak options")
    }
}

/// Mono PCM audio.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Audio {
    /// Samples in [-1, 1].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Audio {
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate.max(1) as f64
    }

    /// The samples as 16-bit PCM (rounded, clipped), as the reference writes them.
    pub fn to_i16(&self) -> Vec<i16> {
        self.samples.iter().map(|&s| f32_to_i16(s)).collect()
    }

    /// A 16-bit PCM mono WAV file.
    pub fn to_wav(&self) -> Vec<u8> {
        wav_bytes(&self.samples, self.sample_rate)
    }

    /// Writes [`Audio::to_wav`] to `path`.
    pub fn write_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        let p = path.as_ref();
        std::fs::write(p, self.to_wav()).with_context(|| format!("write {}", p.display()))
    }

    /// One `F32` mono frame (pts 0, in samples) and the parameters describing it.
    #[cfg(feature = "oxideav")]
    pub fn to_frame(&self) -> (oxideav_core::AudioFrame, oxideav_core::CodecParameters) {
        crate::av::mono_audio_frame(&self.samples, self.sample_rate, 0)
    }

    /// The audio as an [`crate::av::AudioClip`] of one frame.
    #[cfg(feature = "oxideav")]
    pub fn to_clip(&self) -> crate::av::AudioClip {
        let (frame, params) = self.to_frame();
        crate::av::AudioClip::new(vec![frame], params)
    }
}

fn f32_to_i16(s: f32) -> i16 {
    (s * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

/// A 16-bit PCM mono WAV file of `samples` (in [-1, 1]).
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for &s in samples {
        b.extend_from_slice(&f32_to_i16(s).to_le_bytes());
    }
    b
}

/// A piece of streamed audio.
#[derive(Clone, Copy, Debug)]
pub struct AudioChunk<'a> {
    /// The new samples, following the previous chunk's.
    pub samples: &'a [f32],
    /// Index of the first sample in the utterance.
    pub start: usize,
    pub sample_rate: u32,
    /// Frames generated so far.
    pub frames: usize,
    /// Whether this is the last chunk.
    pub last: bool,
}

impl AudioChunk<'_> {
    /// The chunk as an `F32` mono frame (pts = `start`, in samples) and its parameters.
    #[cfg(feature = "oxideav")]
    pub fn to_frame(&self) -> (oxideav_core::AudioFrame, oxideav_core::CodecParameters) {
        crate::av::mono_audio_frame(self.samples, self.sample_rate, self.start as i64)
    }
}

/// Timings and counts of one utterance.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SpeakStats {
    pub prompt_tokens: usize,
    /// Generated tokens, the stop token included.
    pub tokens: usize,
    /// Frames of audio (7 tokens each).
    pub frames: usize,
    /// Generated tokens that were not audio codes in place (unconstrained sampling), dropped.
    pub dropped_tokens: usize,
    pub audio_secs: f64,
    pub prefill_secs: f64,
    /// Language-model time after the prefill.
    pub generate_secs: f64,
    /// SNAC decoding time.
    pub snac_secs: f64,
    pub total_secs: f64,
    /// Streaming: time from the call to the first audio chunk.
    pub first_chunk_secs: Option<f64>,
}

impl SpeakStats {
    /// Generated tokens per second (after the prefill).
    pub fn tok_s(&self) -> f64 {
        self.tokens as f64 / self.generate_secs.max(1e-9)
    }
    /// Real-time factor: total time over audio duration (below 1 = faster than real time).
    pub fn rtf(&self) -> f64 {
        self.total_secs / self.audio_secs.max(1e-9)
    }
}

/// The result of [`Speaker::speak_stream`].
#[derive(Clone, Debug)]
pub struct Speech {
    pub audio: Audio,
    /// The generated token ids (stop token included).
    pub tokens: Vec<u32>,
    /// The audio frames, as 7 codes each in token order (`0 1 2 2 1 2 2` codebooks).
    pub frames: Vec<[u32; 7]>,
    pub finish: StopReason,
    pub stats: SpeakStats,
}

/// What a loaded speaker is.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SpeakerInfo {
    pub path: PathBuf,
    pub snac_path: PathBuf,
    pub layers: usize,
    pub hidden_size: usize,
    pub quant: Quant,
    pub device: String,
    pub threads: usize,
    pub sample_rate: u32,
    pub load_secs: f64,
}

/// The output rows of the audio tokens of each frame position, and of the stop tokens: logits
/// for just these when sampling is constrained.
struct AudioHead {
    pos: Vec<Lin<'static>>,
    stop: Lin<'static>,
}

/// Orpheus: the language model, its tokenizer and the SNAC decoder.
pub struct Speaker {
    model: QwenModel<'static>,
    head: Option<AudioHead>,
    tok: Tokenizer,
    snac: Snac,
    info: SpeakerInfo,
    cancel: Arc<AtomicBool>,
    /// Must stay last: `model` and `head` borrow from it.
    _files: Box<ModelFiles>,
}

/// Rows `r0..r1` of a `[_, n_in]` bf16 matrix at the model's precision.
fn row_lin(w: &'static [u16], rows: &[std::ops::Range<usize>], n_in: usize, quant: Quant) -> Lin<'static> {
    let n: usize = rows.iter().map(|r| r.len()).sum();
    let contiguous = rows.len() == 1;
    let bf16 = || {
        let mut v = crate::quant::PageVec::<u16>::zeroed(n * n_in);
        let mut o = 0;
        for r in rows {
            let src = &w[r.start * n_in..r.end * n_in];
            v[o..o + src.len()].copy_from_slice(src);
            o += src.len();
        }
        v
    };
    match quant {
        Quant::Q8 => Lin::q8(crate::quant::quantize_bf16(&bf16(), n, n_in)),
        Quant::Q4 => Lin::q4(crate::quant::quantize_q4(&bf16(), n, n_in)),
        _ if contiguous => Lin::bf16(&w[rows[0].start * n_in..rows[0].end * n_in]),
        _ => Lin::bf16_owned(bf16()),
    }
}

/// The audio head of a GGUF file, in the output layer's own block format: whole rows, so each
/// frame position's codes are a borrowed view of the mapped file and the three stop rows a copy.
fn gguf_audio_head(files: &'static ModelFiles, name: &str, vocab: usize, n_in: usize) -> Result<Option<AudioHead>> {
    let Some(g) = files.ggml(name) else {
        return Ok(None);
    };
    let (kind, bytes) = g?;
    let rb = kind.row_bytes(n_in);
    if bytes.len() != vocab * rb {
        return Ok(None);
    }
    let rows = |r0: usize, n: usize| &bytes[r0 * rb..(r0 + n) * rb];
    let pos = (0..FRAME_TOKENS as u32)
        .map(|p| Lin::from_gguf(kind, rows((AUDIO_BASE + p * CODES) as usize, CODES as usize), CODES as usize, n_in))
        .collect::<Result<_>>()?;
    let stop: Vec<u8> = STOP_TOKENS.iter().flat_map(|&t| rows(t as usize, 1).iter().copied()).collect();
    let stop = Lin::gguf(crate::ggml::GgufMatrix::from_bytes(kind, &stop, STOP_TOKENS.len(), n_in)?);
    Ok(Some(AudioHead { pos, stop }))
}

impl Speaker {
    /// Loads an Orpheus checkpoint (directory or repo id, e.g. `unsloth/orpheus-3b-0.1-ft`, an
    /// ungated copy of `canopylabs/orpheus-3b-0.1-ft`) and the SNAC decoder.
    pub fn load(model: &str, opts: &SpeakerOptions) -> Result<Self> {
        let dir = crate::engine::resolve_model(model, opts.cache_dir.as_deref())?;
        let snac = opts.snac.as_deref().unwrap_or(DEFAULT_SNAC);
        let snac_dir = crate::engine::resolve_model(snac, opts.cache_dir.as_deref())?;
        Self::load_dirs(&dir, &snac_dir, opts)
    }

    /// Loads from local directories.
    pub fn load_dirs(dir: &Path, snac_dir: &Path, opts: &SpeakerOptions) -> Result<Self> {
        let t0 = Instant::now();
        if let Some(n) = opts.threads {
            let mut o = pool::options();
            o.threads = n;
            pool::configure_with(o);
        }
        let gpu = if crate::gpu::initialized() {
            crate::gpu::active()
        } else {
            crate::gpu::init(&crate::platform::detect(), opts.device)
        };
        let mt = crate::config::model_type(dir)?;
        if mt != "llama" {
            bail!("{}: model_type {mt} is not an Orpheus (Llama) checkpoint", dir.display());
        }
        let cfg = crate::config::QwenConfig::load(dir)?;
        if cfg.vocab_size < (AUDIO_BASE + FRAME_TOKENS as u32 * CODES) as usize {
            bail!("{}: vocabulary of {} has no Orpheus audio tokens", dir.display(), cfg.vocab_size);
        }
        let files = Box::new(ModelFiles::open(dir, !opts.mmap)?);
        // SAFETY: the files are boxed (their address does not move with the Speaker) and dropped
        // after the model and head that borrow them (`_files` is the last field).
        let fr: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let model = QwenModel::load(cfg, fr, opts.quant)?;
        let h = model.cfg.hidden_size;
        let head_name = if fr.has("lm_head.weight") && !model.cfg.tie_word_embeddings {
            "lm_head.weight".to_string()
        } else if fr.has("model.embed_tokens.weight") {
            "model.embed_tokens.weight".to_string()
        } else {
            String::new()
        };
        // The audio head needs the bf16 rows (not available in pre-quantized checkpoints, which
        // then use the full output layer).
        let head = match fr.bf16(&head_name) {
            Ok(w) if w.len() == model.cfg.vocab_size * h && matches!(model.quant, Quant::Bf16 | Quant::Q8 | Quant::Q4) => {
                let pos = (0..FRAME_TOKENS as u32)
                    .map(|p| {
                        let r0 = (AUDIO_BASE + p * CODES) as usize;
                        row_lin(w, std::slice::from_ref(&(r0..r0 + CODES as usize)), h, model.quant)
                    })
                    .collect();
                let stops: Vec<_> = STOP_TOKENS.iter().map(|&t| t as usize..t as usize + 1).collect();
                Some(AudioHead {
                    pos,
                    stop: row_lin(w, &stops, h, model.quant),
                })
            }
            _ => None,
        };
        let head = match head {
            Some(hd) => Some(hd),
            None => gguf_audio_head(fr, &head_name, model.cfg.vocab_size, h)?,
        };
        let tok = crate::engine::load_tokenizer(dir)?;
        let snac = Snac::load(
            snac_dir,
            &SnacOptions {
                f32_weights: opts.snac_f32,
            },
        )
        .with_context(|| format!("load SNAC from {}", snac_dir.display()))?;
        let info = SpeakerInfo {
            path: dir.to_path_buf(),
            snac_path: snac_dir.to_path_buf(),
            layers: model.cfg.num_hidden_layers,
            hidden_size: h,
            quant: model.quant,
            device: gpu.map_or("cpu".to_string(), |b| b.name().to_string()),
            threads: pool::threads(),
            sample_rate: snac.sample_rate(),
            load_secs: t0.elapsed().as_secs_f64(),
        };
        Ok(Self {
            model,
            head,
            tok,
            snac,
            info,
            cancel: Arc::new(AtomicBool::new(false)),
            _files: files,
        })
    }

    pub fn info(&self) -> &SpeakerInfo {
        &self.info
    }

    /// The SNAC decoder.
    pub fn snac(&self) -> &Snac {
        &self.snac
    }

    /// A handle that stops the running (or next) utterance from another thread; the audio
    /// generated so far is returned.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle::from_flag(self.cancel.clone())
    }

    /// The prompt token ids for `text` in `voice` (see the module docs).
    pub fn prompt_tokens(&self, text: &str, voice: Option<&str>) -> Result<Vec<u32>> {
        let t = match voice.map(str::trim).filter(|v| !v.is_empty()) {
            Some(v) => format!("{v}: {text}"),
            None => text.to_string(),
        };
        let enc = self.tok.encode(t.as_str(), false).map_err(|e| anyhow!("tokenize: {e}"))?;
        let mut ids = vec![START_OF_HUMAN, BOS];
        ids.extend_from_slice(enc.get_ids());
        ids.extend_from_slice(&[END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]);
        Ok(ids)
    }

    /// Speaks `text`: the whole utterance, decoded once at the end.
    pub fn speak(&self, text: &str, opts: &SpeakOptions) -> Result<Audio> {
        Ok(self.run(text, opts, None)?.audio)
    }

    /// Speaks `text`, handing audio to `on_chunk` as it is generated (return [`Flow::Stop`] to
    /// stop; the audio so far is returned).
    pub fn speak_stream(&self, text: &str, opts: &SpeakOptions, on_chunk: &mut dyn FnMut(&AudioChunk) -> Flow) -> Result<Speech> {
        self.run(text, opts, Some(on_chunk))
    }

    /// [`Speaker::speak`] with the tokens, frames and statistics.
    pub fn speak_detailed(&self, text: &str, opts: &SpeakOptions) -> Result<Speech> {
        self.run(text, opts, None)
    }

    /// Decodes Orpheus audio frames (7 codes each, see [`audio_frames`]) to audio.
    pub fn decode_frames(&self, frames: &[[u32; 7]], noise: Noise) -> Result<Audio> {
        let _run = RUN.lock().unwrap_or_else(|p| p.into_inner());
        decode_window(&self.snac, frames, 0, noise).map(|samples| Audio {
            samples,
            sample_rate: self.snac.sample_rate(),
        })
    }

    fn run(&self, text: &str, opts: &SpeakOptions, mut on_chunk: Option<&mut dyn FnMut(&AudioChunk) -> Flow>) -> Result<Speech> {
        let t0 = Instant::now();
        self.cancel.store(false, Ordering::SeqCst);
        let _run = RUN.lock().unwrap_or_else(|p| p.into_inner());
        let ids = self.prompt_tokens(text, opts.voice.as_deref())?;
        let mut st = SpeakStats {
            prompt_tokens: ids.len(),
            ..Default::default()
        };
        let noise = if opts.noise { Noise::Seeded(opts.seed) } else { Noise::Off };
        let rate = self.snac.sample_rate();
        let vocab = self.model.cfg.vocab_size;
        let h = self.model.cfg.hidden_size;
        let head = if opts.constrained { self.head.as_ref() } else { None };

        let mut sampler = Sampler::new(opts.temperature, opts.top_k, opts.top_p, opts.seed);
        let penalty = opts.repetition_penalty;
        let mut seen = vec![0u64; vocab.div_ceil(64)];
        let mark = |seen: &mut Vec<u64>, t: u32| {
            if let Some(w) = seen.get_mut(t as usize / 64) {
                *w |= 1 << (t % 64);
            }
        };
        let is_seen = |seen: &[u64], t: u32| seen.get(t as usize / 64).is_some_and(|w| (w >> (t % 64)) & 1 == 1);
        for &t in &ids {
            mark(&mut seen, t);
        }

        let mut sess: QwenSession = self.model.new_session();
        let hs = crate::generate::prefill(&self.model, &mut sess, &ids, 256, &[]);
        let mut hidden = hs[hs.len() - h..].to_vec();
        st.prefill_secs = t0.elapsed().as_secs_f64();

        let mut tokens = Vec::new();
        let mut frames: Vec<[u32; 7]> = Vec::new();
        let mut cur = [0u32; 7];
        let mut fill = 0usize;
        let mut finish = StopReason::Length;
        let mut out = Streamer {
            snac: &self.snac,
            noise,
            left: opts.context_left,
            right: opts.context_right,
            chunk: opts.chunk_frames.max(1),
            emitted: 0,
            audio: Vec::new(),
            snac_secs: 0.0,
            t0,
            first_chunk: None,
        };
        let tg = Instant::now();
        for _ in 0..opts.max_tokens {
            // Logits and the token ids they stand for.
            let (mut logits, ids_of): (Vec<f32>, Option<Vec<u32>>) = match head {
                Some(hd) => {
                    let mut l = vec![0f32; CODES as usize + STOP_TOKENS.len()];
                    let (a, b) = l.split_at_mut(CODES as usize);
                    linear(&hd.pos[fill], CODES as usize, h, &hidden, 1, a);
                    linear(&hd.stop, STOP_TOKENS.len(), h, &hidden, 1, b);
                    let base = AUDIO_BASE + fill as u32 * CODES;
                    let ids: Vec<u32> = (base..base + CODES).chain(STOP_TOKENS).collect();
                    if penalty != 1.0 {
                        for (v, &t) in l.iter_mut().zip(&ids) {
                            if is_seen(&seen, t) {
                                *v = if *v < 0.0 { *v * penalty } else { *v / penalty };
                            }
                        }
                    }
                    (l, Some(ids))
                }
                None => {
                    let mut l = self.model.logits(&hidden, 1);
                    if penalty != 1.0 {
                        apply_repetition_penalty(&mut l, &seen, penalty);
                    }
                    (l, None)
                }
            };
            if logits.iter().any(|v| v.is_nan()) {
                bail!("the model produced NaN logits");
            }
            let k = sampler.sample(&logits) as usize;
            let id = ids_of.as_ref().map_or(k as u32, |m| m[k]);
            logits.clear();
            tokens.push(id);
            mark(&mut seen, id);
            if STOP_TOKENS.contains(&id) {
                finish = StopReason::Stop;
                break;
            }
            let pos_base = AUDIO_BASE + fill as u32 * CODES;
            if (pos_base..pos_base + CODES).contains(&id) {
                cur[fill] = id - pos_base;
                fill += 1;
                if fill == FRAME_TOKENS {
                    frames.push(cur);
                    fill = 0;
                    let flow = match on_chunk.as_deref_mut() {
                        Some(cb) => {
                            let ts = Instant::now();
                            let f = out.pump(&frames, false, cb)?;
                            st.generate_secs -= ts.elapsed().as_secs_f64();
                            f
                        }
                        None => Flow::Continue,
                    };
                    if flow == Flow::Stop {
                        finish = StopReason::Cancelled;
                        break;
                    }
                }
            } else {
                st.dropped_tokens += 1;
            }
            if self.cancel.load(Ordering::SeqCst) {
                finish = StopReason::Cancelled;
                break;
            }
            hidden = self.model.forward_hidden_mm(&mut sess, &[id], &[]);
        }
        st.generate_secs += tg.elapsed().as_secs_f64();
        let samples = match on_chunk {
            Some(cb) => {
                if finish != StopReason::Cancelled {
                    out.pump(&frames, true, cb)?;
                }
                std::mem::take(&mut out.audio)
            }
            None => {
                let ts = Instant::now();
                let a = decode_window(&self.snac, &frames, 0, noise)?;
                out.snac_secs += ts.elapsed().as_secs_f64();
                a
            }
        };
        st.snac_secs = out.snac_secs;
        st.first_chunk_secs = out.first_chunk;
        st.tokens = tokens.len();
        st.frames = frames.len();
        st.audio_secs = samples.len() as f64 / rate as f64;
        st.total_secs = t0.elapsed().as_secs_f64();
        Ok(Speech {
            audio: Audio {
                samples,
                sample_rate: rate,
            },
            tokens,
            frames,
            finish,
            stats: st,
        })
    }
}

/// Streaming decode state: overlapping windows of frames, each emitting the frames that have
/// their lookahead.
struct Streamer<'s> {
    snac: &'s Snac,
    noise: Noise,
    left: usize,
    right: usize,
    chunk: usize,
    /// Frames emitted so far.
    emitted: usize,
    audio: Vec<f32>,
    snac_secs: f64,
    t0: Instant,
    first_chunk: Option<f64>,
}

impl Streamer<'_> {
    /// Emits every chunk of frames that has `right` frames of lookahead (all frames left when
    /// `last`).
    fn pump(&mut self, frames: &[[u32; 7]], last: bool, cb: &mut dyn FnMut(&AudioChunk) -> Flow) -> Result<Flow> {
        let fs = self.snac.samples_per_frame();
        loop {
            let avail = frames.len() - self.emitted;
            let n = if last { avail } else { self.chunk };
            if n == 0 || (!last && avail < self.chunk + self.right) {
                return Ok(Flow::Continue);
            }
            let w0 = self.emitted.saturating_sub(self.left);
            let w1 = (self.emitted + n + self.right).min(frames.len());
            let ts = Instant::now();
            let wav = decode_window(self.snac, &frames[w0..w1], w0, self.noise)?;
            self.snac_secs += ts.elapsed().as_secs_f64();
            let a = (self.emitted - w0) * fs;
            let start = self.audio.len();
            self.audio.extend_from_slice(&wav[a..a + n * fs]);
            self.emitted += n;
            self.first_chunk.get_or_insert_with(|| self.t0.elapsed().as_secs_f64());
            let chunk = AudioChunk {
                samples: &self.audio[start..],
                start,
                sample_rate: self.snac.sample_rate(),
                frames: frames.len(),
                last,
            };
            if cb(&chunk) == Flow::Stop {
                return Ok(Flow::Stop);
            }
            if last {
                return Ok(Flow::Continue);
            }
        }
    }
}

/// SNAC codes of Orpheus frames: codebook 0 takes position 0, codebook 1 positions 1 and 4,
/// codebook 2 positions 2, 3, 5 and 6.
pub fn frames_to_codes(frames: &[[u32; 7]]) -> [Vec<u32>; 3] {
    let mut c = [Vec::new(), Vec::new(), Vec::new()];
    for f in frames {
        c[0].push(f[0]);
        c[1].extend_from_slice(&[f[1], f[4]]);
        c[2].extend_from_slice(&[f[2], f[3], f[5], f[6]]);
    }
    c
}

/// Decodes `frames` whose first frame is frame `frame0` of the utterance.
fn decode_window(snac: &Snac, frames: &[[u32; 7]], frame0: usize, noise: Noise) -> Result<Vec<f32>> {
    let c = frames_to_codes(frames);
    snac.decode_at(&[&c[0], &c[1], &c[2]], frame0, noise)
}

/// The audio frames of a generated token sequence, as the reference decoder reads them: the
/// tokens after the last start of speech (if any), audio tokens taken in order with their codes
/// relative to the frame position they land on (tokens whose code is out of range for that
/// position are skipped), an incomplete last frame dropped.
pub fn audio_frames(tokens: &[u32]) -> Vec<[u32; 7]> {
    let from = tokens.iter().rposition(|&t| t == START_OF_SPEECH).map_or(0, |i| i + 1);
    let mut frames = Vec::new();
    let mut cur = [0u32; 7];
    let mut fill = 0;
    for &t in &tokens[from..] {
        let base = AUDIO_BASE + fill as u32 * CODES;
        if (base..base + CODES).contains(&t) {
            cur[fill] = t - base;
            fill += 1;
            if fill == FRAME_TOKENS {
                frames.push(cur);
                fill = 0;
            }
        }
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_and_codes() {
        let f = [1, 2, 3, 4, 5, 6, 7];
        let toks: Vec<u32> = std::iter::once(START_OF_SPEECH)
            .chain(f.iter().enumerate().map(|(p, &c)| AUDIO_BASE + p as u32 * CODES + c))
            .chain([AUDIO_BASE + 9, END_OF_SPEECH])
            .collect();
        let frames = audio_frames(&toks);
        assert_eq!(frames, vec![f]);
        let c = frames_to_codes(&frames);
        assert_eq!(c[0], vec![1]);
        assert_eq!(c[1], vec![2, 5]);
        assert_eq!(c[2], vec![3, 4, 6, 7]);
    }

    #[test]
    fn wav_header() {
        let w = wav_bytes(&[0.0, 1.0, -1.0], 24000);
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(w.len(), 44 + 6);
        assert_eq!(i16::from_le_bytes([w[46], w[47]]), 32767);
        assert_eq!(i16::from_le_bytes([w[48], w[49]]), -32767);
    }

    #[test]
    fn options_json() {
        let o = SpeakOptions::from_json(r#"{"voice": "leo", "temperature": 0.0, "seed": 5}"#).unwrap();
        assert_eq!(o.voice.as_deref(), Some("leo"));
        assert_eq!(o.temperature, 0.0);
        assert_eq!(o.seed, 5);
        assert_eq!(o.top_p, 0.8);
        assert!(SpeakOptions::from_json(r#"{"bogus": 1}"#).is_ok());
    }

    #[cfg(feature = "oxideav")]
    #[test]
    fn oxideav_frames() {
        let a = Audio {
            samples: vec![0.5, -0.25, 0.0],
            sample_rate: 24000,
        };
        let (f, p) = a.to_frame();
        assert_eq!((f.samples, f.pts, f.data[0].len()), (3, Some(0), 12));
        assert_eq!((p.sample_rate, p.channels), (Some(24000), Some(1)));
        assert_eq!(a.to_clip().to_mono(24000).unwrap(), a.samples);
        let c = AudioChunk {
            samples: &a.samples[1..],
            start: 1,
            sample_rate: 24000,
            frames: 1,
            last: true,
        };
        let (f, _) = c.to_frame();
        assert_eq!((f.samples, f.pts), (2, Some(1)));
    }
}
