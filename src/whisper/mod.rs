//! Speech to text with OpenAI Whisper checkpoints in the Hugging Face layout (`model_type:
//! whisper`): openai/whisper-{tiny,base,small,medium,large-v3,large-v3-turbo}, the `.en` models,
//! and distil-whisper (same architecture, fewer decoder layers). Sizes come from `config.json`
//! (80 or 128 mel bins), special tokens and decoding defaults from `generation_config.json`.
//!
//! Decoding follows transformers' `WhisperForConditionalGeneration.generate` (greedy by default,
//! so its transcripts can be compared token for token), and openai-whisper for beam search:
//! - language detection from the first window's first-token distribution over the language
//!   tokens when no language is given; transcribe or translate;
//! - `suppress_tokens` / `begin_suppress_tokens`; the timestamp rules of
//!   `WhisperTimeStampLogitsProcessor` with segment-level timestamps;
//! - long-form audio (over 30 s, timestamps forced on) in sequential 30 s windows, each moved to
//!   the last complete segment's end timestamp;
//! - optional temperature fallback (compression ratio, average log-probability and no-speech
//!   thresholds), conditioning on the previous window's text, an initial prompt and a prefix;
//! - optional word-level timestamps from the alignment heads' cross-attention (openai-whisper's
//!   DTW alignment);
//! - optional beam search ([`TranscribeOptions::beam_size`], `patience`, `length_penalty`) at
//!   temperature 0 and the best of several samples above it (`best_of`), as openai-whisper's
//!   `BeamSearchDecoder` and `MaximumLikelihoodRanker` do them;
//! - several inputs at once ([`Transcriber::transcribe_batch`]): their windows share the encoder
//!   and every decoder step, with the transcripts each input gets alone.
//!
//! ```no_run
//! use oxinfer::whisper::{Transcriber, TranscriberOptions, TranscribeOptions};
//! # fn main() -> anyhow::Result<()> {
//! let asr = Transcriber::load("openai/whisper-small", &TranscriberOptions::default())?;
//! let mut opts = TranscribeOptions::default();
//! opts.timestamps = true;
//! let pcm: Vec<f32> = vec![0.0; 16_000]; // 16 kHz mono in [-1, 1]
//! let t = asr.transcribe_with(&pcm, &opts, &mut |seg| {
//!     println!("[{:.2} -> {:.2}] {}", seg.start, seg.end, seg.text);
//!     oxinfer::engine::Flow::Continue
//! })?;
//! println!("{} ({})", t.text, t.language);
//! # Ok(())
//! # }
//! ```
//!
//! The encoder runs its matmuls and attention on the GPU backend when one is active (Metal,
//! CUDA, WebGPU); decoding is per token on the CPU, with q8 / q4 weights if asked
//! ([`TranscriberOptions::decoder`]).

pub mod align;
#[cfg(test)]
mod batch_tests;
mod compress;
mod decode;
pub mod mel;
pub mod model;
pub mod tokens;

use crate::clock::Instant;
// RUN serializes transcriptions with each other and with engine generations (the thread pool
// runs one region at a time).
use crate::engine::{CancelHandle, Flow, RUN};
use crate::platform::DevicePref;
use crate::pool;
use crate::safetensors::ModelFiles;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, bail};
pub use model::{Precision, WhisperConfig, WhisperModel};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
pub use tokens::{LANGUAGES, Special, language_code, language_name};

/// Seconds per mel frame and per timestamp token.
const FRAME_SECS: f64 = 0.01;
const TIMESTAMP_SECS: f64 = 0.02;

/// Whether `dir` holds a Whisper checkpoint.
pub fn is_whisper(dir: &Path) -> bool {
    crate::config::model_type(dir).is_ok_and(|t| t == "whisper")
}

/// How to load a Whisper model. `TranscriberOptions::default()` then set fields, or
/// [`TranscriberOptions::set`] key-value pairs.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TranscriberOptions {
    /// CPU worker threads (process-wide; default: all cores). Key `threads`.
    pub threads: Option<usize>,
    /// Compute device (process-wide; set by the first model loaded). Key `device`.
    pub device: DevicePref,
    /// Encoder matrices (and the cross-attention keys / values). Key `encoder`. Default bf16.
    pub encoder: Precision,
    /// Decoder matrices, including the tied embedding / output projection. Key `decoder`
    /// (`quant` sets it too, as for language models). Default bf16.
    pub decoder: Precision,
    /// Map the checkpoint instead of reading it into memory while converting. Key `mmap`.
    pub mmap: bool,
    /// Where repo ids are downloaded. Key `cache_dir`.
    pub cache_dir: Option<PathBuf>,
    /// GPU matmuls bit-identical to the CPU (no half-precision matrix units). Key `gpu_exact`.
    pub gpu_exact: Option<bool>,
}

impl Default for TranscriberOptions {
    fn default() -> Self {
        Self {
            threads: None,
            device: DevicePref::Auto,
            encoder: Precision::Bf16,
            decoder: Precision::Bf16,
            mmap: !cfg!(target_arch = "wasm32"),
            cache_dir: None,
            gpu_exact: None,
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

impl TranscriberOptions {
    /// Sets an option by key (`threads`, `device`, `encoder`, `decoder`, `quant`, `precision`
    /// for both, `mmap`, `cache_dir`, `gpu_exact`).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let prec = |v: &str| Precision::parse(v).with_context(|| format!("{key}: unknown precision {v:?} (f32, f16, bf16, q8, q4)"));
        match key {
            "threads" => self.threads = Some(value.parse().with_context(|| format!("threads: {value:?}"))?),
            "device" => self.device = DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?}"))?,
            "encoder" => self.encoder = prec(value)?,
            "decoder" | "quant" => self.decoder = prec(value)?,
            "precision" => {
                self.encoder = prec(value)?;
                self.decoder = self.encoder;
            }
            "mmap" => self.mmap = parse_bool(key, value)?,
            "cache_dir" => self.cache_dir = Some(PathBuf::from(value)),
            "gpu_exact" => self.gpu_exact = Some(parse_bool(key, value)?),
            _ => bail!("unknown option {key}"),
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    /// Speech to text in the spoken language.
    Transcribe,
    /// Speech to English text.
    Translate,
}

/// Options of one transcription. Deserializes from JSON with every field optional (the C API's
/// option format), e.g. `{"language": "fr", "timestamps": true}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct TranscribeOptions {
    /// Spoken language as a code (`fr`), an English name (`french`) or a token (`<|fr|>`);
    /// None detects it from the first 30 s (or uses the checkpoint's default language).
    pub language: Option<String>,
    /// None: the checkpoint's default task, else transcribe.
    pub task: Option<Task>,
    /// Segment-level timestamps (always on for audio over 30 s).
    pub timestamps: bool,
    /// Word-level timestamps (implies `timestamps`; needs `alignment_heads` in the
    /// generation config).
    pub word_timestamps: bool,
    /// Temperatures to try in order; the next one is used when a window fails the thresholds
    /// below. `[0.0]` (the default) is greedy decoding without fallback.
    pub temperatures: Vec<f32>,
    /// Retry when the window's tokens compress better than this (repetition loops), e.g. 2.4.
    pub compression_ratio_threshold: Option<f32>,
    /// Retry when the average token log-probability is below this, e.g. -1.0.
    pub logprob_threshold: Option<f32>,
    /// Skip a window as silence when its no-speech probability exceeds this and its average
    /// log-probability is below `logprob_threshold`, e.g. 0.6.
    pub no_speech_threshold: Option<f32>,
    /// Feed the previous windows' text (up to 223 tokens) as context to the next window.
    pub condition_on_prev_tokens: bool,
    /// Text given to the decoder as previous context (spelling, names, style).
    pub initial_prompt: Option<String>,
    /// Text the first window's transcript is forced to start with.
    pub prefix: Option<String>,
    /// Tokens never generated; None uses the generation config's `suppress_tokens`.
    pub suppress_tokens: Option<Vec<u32>>,
    /// Suppress a blank or end of text as the first token (`begin_suppress_tokens`).
    pub suppress_blank: bool,
    /// Generated tokens per window at most (the decoder holds 448 with the prompt).
    pub max_new_tokens: Option<usize>,
    /// Seed of the sampler (temperatures above 0).
    pub seed: u64,
    /// Beam search with this many beams at temperature 0 (openai-whisper's `beam_size`); None
    /// is greedy decoding. Temperatures above 0 (the fallback) sample instead.
    pub beam_size: Option<usize>,
    /// Beam search ends once `round(beam_size * patience)` hypotheses have ended (None: 1.0).
    pub patience: Option<f32>,
    /// Ranking of the ended hypotheses (beams, or the `best_of` samples): None divides the
    /// log-probability by the length, Some(a), 0 to 1, by `((5 + length) / 6)^a`.
    pub length_penalty: Option<f32>,
    /// Sequences sampled at each temperature above 0, of which the best ranked is kept (None: 1).
    pub best_of: Option<usize>,
    /// Inputs decoded together by [`Transcriber::transcribe_batch`] (the others wait for a free
    /// place). Each holds its features and one window's cross-attention keys and values.
    pub batch_size: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: None,
            timestamps: false,
            word_timestamps: false,
            temperatures: vec![0.0],
            compression_ratio_threshold: None,
            logprob_threshold: None,
            no_speech_threshold: None,
            condition_on_prev_tokens: false,
            initial_prompt: None,
            prefix: None,
            suppress_tokens: None,
            suppress_blank: true,
            max_new_tokens: None,
            seed: 0,
            beam_size: None,
            patience: None,
            length_penalty: None,
            best_of: None,
            batch_size: 8,
        }
    }
}

impl TranscribeOptions {
    /// openai-whisper's robustness defaults: temperatures 0, 0.2, ..., 1.0 with a compression
    /// ratio threshold of 2.4, log-probability -1 and no-speech 0.6.
    pub fn with_fallback(mut self) -> Self {
        self.temperatures = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        self.compression_ratio_threshold = Some(2.4);
        self.logprob_threshold = Some(-1.0);
        self.no_speech_threshold = Some(0.6);
        self
    }
}

/// One word with its time span (word timestamps).
#[derive(Clone, Debug, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    /// Mean probability of the word's tokens.
    pub probability: f32,
}

/// A transcribed span of audio.
#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub id: usize,
    /// Seconds from the start of the audio.
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// Generated token ids, timestamps included.
    pub tokens: Vec<u32>,
    /// Decoding temperature of the window (0 = greedy).
    pub temperature: f32,
    /// Mean log-probability of the window's tokens.
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    /// Probability of `<|nospeech|>` at the window's start.
    pub no_speech_prob: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
}

/// Timings of one transcription, in seconds. In a batch, `encode_secs` and `decode_secs` are the
/// time of the passes the input took part in (shared with the inputs decoded with it).
#[derive(Clone, Debug, Default, Serialize)]
pub struct TranscribeStats {
    pub audio_secs: f64,
    pub mel_secs: f64,
    /// Encoder and cross-attention keys / values, all windows.
    pub encode_secs: f64,
    pub decode_secs: f64,
    pub total_secs: f64,
    pub windows: usize,
    /// Decoder passes that produced a token, fallback retries and every beam or sample included.
    pub tokens: usize,
    /// Wall time over audio duration (lower is faster; 0.1 = ten times real time).
    pub rtf: f64,
}

/// The result of a transcription.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Transcript {
    pub text: String,
    /// Language code (`en`), given or detected (`en` for English-only models).
    pub language: String,
    /// The detected language's probability (None when the language was given).
    pub language_probability: Option<f32>,
    pub segments: Vec<Segment>,
    /// The transcription was stopped early (cancel, or the segment callback returned Stop).
    pub stopped: bool,
    pub stats: TranscribeStats,
}

/// What a loaded model is.
#[derive(Clone, Debug, Serialize)]
pub struct TranscriberInfo {
    pub path: PathBuf,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub n_mels: usize,
    pub vocab_size: usize,
    pub multilingual: bool,
    pub encoder: Precision,
    pub decoder: Precision,
    /// "cpu" or the GPU backend's name (used for the encoder).
    pub device: String,
    pub threads: usize,
    pub load_secs: f64,
}

/// A loaded Whisper model.
pub struct Transcriber {
    model: WhisperModel,
    tok: Tokenizer,
    sp: Special,
    info: TranscriberInfo,
    cancel: Arc<AtomicBool>,
}

/// The options of a transcription resolved against the checkpoint (shared by a batch).
struct Plan<'a> {
    o: &'a TranscribeOptions,
    task: Task,
    /// The language token when it is given (else detected per input).
    language: Option<u32>,
    prompt: Option<Vec<u32>>,
    prefix: Vec<u32>,
    suppress: Vec<bool>,
    /// Beam search: (beams, ended hypotheses that end the search).
    beam: Option<(usize, usize)>,
}

/// One input on its way through a transcription.
struct Stream {
    /// Position among the inputs.
    index: usize,
    feats: mel::Features,
    timestamps: bool,
    language: Option<u32>,
    /// Segment history for conditioning (raw tokens, the prompt first).
    history: Vec<Vec<u32>>,
    all_text: Vec<u32>,
    /// The next window's first frame.
    seek: usize,
    condition: bool,
    out: Transcript,
    stats: TranscribeStats,
    started: Instant,
}

/// The decoder prompt of a window.
struct Prompt {
    input: Vec<u32>,
    sot_index: usize,
    sot_len: usize,
    max_new: usize,
}

/// What the logits rules need.
struct Rules<'a> {
    suppress: &'a [bool],
    begin_suppress: &'a [u32],
    timestamps: bool,
}

impl Transcriber {
    /// Loads a checkpoint: a local directory or (with the `hf` feature) a repo id.
    pub fn load(model: &str, opts: &TranscriberOptions) -> Result<Self> {
        let dir = crate::engine::resolve_model(model, opts.cache_dir.as_deref())?;
        Self::load_dir(&dir, opts)
    }

    /// Loads the checkpoint in `dir`.
    pub fn load_dir(dir: &Path, opts: &TranscriberOptions) -> Result<Self> {
        let t0 = Instant::now();
        if let Some(n) = opts.threads {
            let mut o = pool::options();
            o.threads = n;
            pool::configure_with(o);
        }
        if let Some(on) = opts.gpu_exact {
            crate::gpu::set_exact(on);
        }
        let gpu = crate::gpu::ensure(opts.device)?;
        let model_type = crate::config::model_type(dir)?;
        if model_type != "whisper" {
            bail!("{}: model_type {model_type} is not a Whisper model", dir.display());
        }
        let cfg = WhisperConfig::load(dir)?;
        let files = ModelFiles::open(dir, !opts.mmap)?;
        let model = WhisperModel::load(cfg, &files, opts.encoder, opts.decoder)?;
        drop(files);
        let tok = tokens::load_tokenizer(dir)?;
        let sp = Special::load(dir, &tok, model.cfg.vocab_size, model.cfg.max_target_positions)?;
        let info = TranscriberInfo {
            path: dir.to_path_buf(),
            d_model: model.cfg.d_model,
            encoder_layers: model.cfg.encoder_layers,
            decoder_layers: model.cfg.decoder_layers,
            n_mels: model.cfg.num_mel_bins,
            vocab_size: model.cfg.vocab_size,
            multilingual: sp.multilingual,
            encoder: opts.encoder,
            decoder: opts.decoder,
            device: gpu.map_or("cpu".to_string(), |b| b.name().to_string()),
            threads: pool::threads(),
            load_secs: t0.elapsed().as_secs_f64(),
        };
        Ok(Self {
            model,
            tok,
            sp,
            info,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn info(&self) -> &TranscriberInfo {
        &self.info
    }
    pub fn model(&self) -> &WhisperModel {
        &self.model
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }
    pub fn special(&self) -> &Special {
        &self.sp
    }

    /// Stops the running (or next) transcription from any thread: it returns what is done,
    /// with `stopped` set. Each transcription clears the request when it starts.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle::from_flag(self.cancel.clone())
    }

    /// Transcribes 16 kHz mono samples in [-1, 1].
    pub fn transcribe(&self, pcm: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
        self.transcribe_with(pcm, opts, &mut |_| Flow::Continue)
    }

    /// Transcribes decoded audio frames (the `oxideav` feature): `params` gives `sample_rate`,
    /// `channels` and `sample_format`; the channels are averaged and the samples resampled to
    /// 16 kHz (see [`crate::av::audio_frames_to_mono`]).
    #[cfg(feature = "oxideav")]
    pub fn transcribe_frames(
        &self,
        frames: &[oxideav_core::AudioFrame],
        params: &oxideav_core::CodecParameters,
        opts: &TranscribeOptions,
    ) -> Result<Transcript> {
        self.transcribe_frames_with(frames, params, opts, &mut |_| Flow::Continue)
    }

    /// [`Transcriber::transcribe_frames`] with the segment callback of [`Transcriber::transcribe_with`].
    #[cfg(feature = "oxideav")]
    pub fn transcribe_frames_with(
        &self,
        frames: &[oxideav_core::AudioFrame],
        params: &oxideav_core::CodecParameters,
        opts: &TranscribeOptions,
        on_segment: &mut dyn FnMut(&Segment) -> Flow,
    ) -> Result<Transcript> {
        let pcm = crate::av::audio_frames_to_mono(frames, params, mel::SAMPLE_RATE)?;
        self.transcribe_with(&pcm, opts, on_segment)
    }

    /// Decodes an audio file (any container / codec oxideav reads), resampled to 16 kHz mono.
    #[cfg(feature = "media")]
    pub fn load_audio(path: &Path) -> Result<Vec<f32>> {
        crate::media::Media::new().load_audio(path, mel::SAMPLE_RATE)
    }

    /// Transcribes an audio file (see [`Transcriber::load_audio`]).
    #[cfg(feature = "media")]
    pub fn transcribe_file(&self, path: &Path, opts: &TranscribeOptions) -> Result<Transcript> {
        self.transcribe_file_with(path, opts, &mut |_| Flow::Continue)
    }

    /// [`Transcriber::transcribe_file`] with the segment callback of [`Transcriber::transcribe_with`].
    #[cfg(feature = "media")]
    pub fn transcribe_file_with(
        &self,
        path: &Path,
        opts: &TranscribeOptions,
        on_segment: &mut dyn FnMut(&Segment) -> Flow,
    ) -> Result<Transcript> {
        let pcm = Self::load_audio(path)?;
        self.transcribe_with(&pcm, opts, on_segment)
    }

    /// The language distribution of the first 30 s: (code, probability) by decreasing
    /// probability. Fails for English-only models.
    pub fn detect_language(&self, pcm: &[f32]) -> Result<Vec<(String, f32)>> {
        if !self.sp.multilingual {
            bail!("this model is English-only");
        }
        let _run = RUN.lock().unwrap_or_else(|p| p.into_inner());
        let feats = mel::log_mel(pcm, self.model.cfg.num_mel_bins);
        let cross = self.model.cross_kv(&self.model.encode(&self.window(&feats, 0)));
        Ok(self.language_probs(&cross))
    }

    fn window(&self, feats: &mel::Features, seek: usize) -> Vec<f32> {
        let mut w = feats.window(seek);
        w.resize(self.model.window_frames() * feats.n_mels, 0.0);
        w
    }

    fn language_probs(&self, cross: &model::CrossKv) -> Vec<(String, f32)> {
        let mut st = self.model.new_state();
        let h = self.model.decode(&[self.sp.sot], &mut st, cross);
        let logits = self.model.logits(&h);
        let langs = &self.sp.languages;
        let max = langs.iter().map(|l| logits[l.1 as usize]).fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<(String, f32)> = langs.iter().map(|l| (l.0.clone(), (logits[l.1 as usize] - max).exp())).collect();
        let sum: f32 = p.iter().map(|x| x.1).sum();
        for x in &mut p {
            x.1 /= sum;
        }
        // Stable sort: ties keep token order, so the first is argmax's choice.
        p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        p
    }

    /// Transcribes 16 kHz mono samples, calling `on_segment` as each segment is final (after
    /// its 30 s window is decoded); returning [`Flow::Stop`] ends the transcription there.
    pub fn transcribe_with(&self, pcm: &[f32], o: &TranscribeOptions, on_segment: &mut dyn FnMut(&Segment) -> Flow) -> Result<Transcript> {
        let mut out = self.transcribe_batch_with(&[pcm], o, &mut |_, s| on_segment(s))?;
        Ok(out.pop().unwrap())
    }

    /// Transcribes several inputs (16 kHz mono samples each) together: the current 30 s window
    /// of every input goes through the encoder and through each decoder step as one batch,
    /// [`TranscribeOptions::batch_size`] inputs at a time. Each transcript is the one
    /// [`Transcriber::transcribe`] gives for that input alone, token for token and bit for bit
    /// in its probabilities (a row of a batch does not depend on the others). The windows of one
    /// input stay sequential: each starts where the previous one's last segment ended.
    pub fn transcribe_batch(&self, pcms: &[&[f32]], opts: &TranscribeOptions) -> Result<Vec<Transcript>> {
        self.transcribe_batch_with(pcms, opts, &mut |_, _| Flow::Continue)
    }

    /// [`Transcriber::transcribe_batch`], calling `on_segment` with the input's index as each
    /// segment is final; returning [`Flow::Stop`] ends that input's transcription there (the
    /// others go on). Segments of different inputs arrive interleaved, in order for each input.
    pub fn transcribe_batch_with(
        &self,
        pcms: &[&[f32]],
        o: &TranscribeOptions,
        on_segment: &mut dyn FnMut(usize, &Segment) -> Flow,
    ) -> Result<Vec<Transcript>> {
        let _run = RUN.lock().unwrap_or_else(|p| p.into_inner());
        self.cancel.store(false, Ordering::SeqCst);
        let plan = self.plan(o)?;
        let batch = o.batch_size.max(1);
        let mut outs: Vec<Option<Transcript>> = pcms.iter().map(|_| None).collect();
        let mut active: Vec<Stream> = Vec::new();
        let mut next = 0usize;
        loop {
            let cancelled = self.cancel.load(Ordering::SeqCst);
            while !cancelled && active.len() < batch && next < pcms.len() {
                active.push(self.open(next, pcms[next], &plan));
                next += 1;
            }
            // Inputs that are done (or stopped) leave their place to the next ones.
            let mut i = 0;
            while i < active.len() {
                let s = &mut active[i];
                s.out.stopped |= cancelled;
                if s.out.stopped || s.seek >= s.feats.frames {
                    let s = active.remove(i);
                    let index = s.index;
                    outs[index] = Some(self.close(s));
                } else {
                    i += 1;
                }
            }
            if active.is_empty() {
                if cancelled || next >= pcms.len() {
                    break;
                }
                continue;
            }
            self.round(&plan, &mut active, on_segment)?;
        }
        // Cancelled before they started.
        Ok(outs
            .into_iter()
            .map(|t| {
                t.unwrap_or_else(|| Transcript {
                    language: if self.sp.multilingual { "" } else { "en" }.to_string(),
                    stopped: true,
                    ..Default::default()
                })
            })
            .collect())
    }

    /// Transcribes several audio files together (see [`Transcriber::transcribe_batch`]).
    #[cfg(feature = "media")]
    pub fn transcribe_files(&self, paths: &[&Path], opts: &TranscribeOptions) -> Result<Vec<Transcript>> {
        let pcms: Vec<Vec<f32>> = paths
            .iter()
            .map(|p| Self::load_audio(p).with_context(|| format!("{}", p.display())))
            .collect::<Result<_>>()?;
        let refs: Vec<&[f32]> = pcms.iter().map(|p| &p[..]).collect();
        self.transcribe_batch(&refs, opts)
    }

    /// Checks the options and resolves them against the checkpoint.
    fn plan<'a>(&self, o: &'a TranscribeOptions) -> Result<Plan<'a>> {
        let sp = &self.sp;
        let m = &self.model;
        if o.word_timestamps && sp.alignment_heads.is_empty() {
            bail!("word timestamps need alignment_heads in generation_config.json");
        }
        if o.temperatures.is_empty() {
            bail!("temperatures must not be empty");
        }
        if o.no_speech_threshold.is_some() && o.logprob_threshold.is_none() {
            bail!("no_speech_threshold needs logprob_threshold");
        }
        let beam = match o.beam_size {
            Some(0) => bail!("beam_size must be at least 1"),
            Some(size) => {
                // Python's round (half to even), as openai-whisper computes it.
                let max_candidates = (size as f64 * o.patience.unwrap_or(1.0) as f64).round_ties_even();
                if max_candidates.is_nan() || max_candidates < 1.0 {
                    bail!("patience {:?} leaves no hypothesis to wait for with {size} beams", o.patience);
                }
                Some((size, max_candidates as usize))
            }
            None if o.patience.is_some() => bail!("patience needs beam_size"),
            None => None,
        };
        if o.best_of == Some(0) {
            bail!("best_of must be at least 1");
        }
        if o.length_penalty.is_some_and(|a| !(0.0..=1.0).contains(&a)) {
            bail!("length_penalty must be between 0 and 1");
        }

        // Task and language tokens.
        let task = o.task.unwrap_or(match sp.default_task.as_deref() {
            Some("translate") => Task::Translate,
            _ => Task::Transcribe,
        });
        let mut language: Option<u32> = None;
        if sp.multilingual {
            if let Some(l) = o.language.as_deref().or(sp.default_language.as_deref()) {
                language = Some(sp.language_token(l)?);
            }
        } else if o.language.as_deref().is_some_and(|l| language_code(l) != Some("en")) || task == Task::Translate {
            bail!("this model is English-only (no language or translate task)");
        }
        let encode_text = |s: &str| -> Result<Vec<u32>> {
            let e = self.tok.encode(s, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
            Ok(e.get_ids().to_vec())
        };
        let prompt: Option<Vec<u32>> = match o.initial_prompt.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(p) => {
                let ids = encode_text(&format!(" {p}"))?;
                if ids.iter().any(|&t| t >= sp.eot) {
                    bail!("the initial prompt contains special tokens");
                }
                Some(ids)
            }
            None => None,
        };
        let prefix: Vec<u32> = match o.prefix.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(p) => encode_text(&format!(" {p}"))?,
            None => Vec::new(),
        };
        let vocab = m.cfg.vocab_size;
        let mut suppress = vec![false; vocab];
        for &t in o.suppress_tokens.as_deref().unwrap_or(&sp.suppress) {
            if (t as usize) < vocab {
                suppress[t as usize] = true;
            }
        }
        Ok(Plan {
            o,
            task,
            language,
            prompt,
            prefix,
            suppress,
            beam,
        })
    }

    /// Starts an input: its features and decoding state.
    fn open(&self, index: usize, pcm: &[f32], plan: &Plan) -> Stream {
        let o = plan.o;
        let started = Instant::now();
        let mut stats = TranscribeStats {
            audio_secs: pcm.len() as f64 / mel::SAMPLE_RATE as f64,
            ..Default::default()
        };
        let feats = mel::log_mel(pcm, self.model.cfg.num_mel_bins);
        stats.mel_secs = started.elapsed().as_secs_f64();
        let shortform = feats.frames <= self.model.window_frames();
        let mut history: Vec<Vec<u32>> = Vec::new();
        if o.condition_on_prev_tokens
            && let Some(p) = &plan.prompt
        {
            history.push(p.clone());
        }
        Stream {
            index,
            feats,
            timestamps: o.timestamps || o.word_timestamps || !shortform,
            language: plan.language,
            history,
            all_text: Vec::new(),
            seek: 0,
            condition: o.condition_on_prev_tokens,
            out: Transcript::default(),
            stats,
            started,
        }
    }

    /// Ends an input: its transcript.
    fn close(&self, s: Stream) -> Transcript {
        let sp = &self.sp;
        let mut out = s.out;
        out.text = tokens::decode_text(&self.tok, sp, &s.all_text).trim().to_string();
        out.language = match s.language {
            Some(l) => sp.language_of(l).unwrap_or("").to_string(),
            // English-only models (or stopped before detection).
            None => if sp.multilingual { "" } else { "en" }.to_string(),
        };
        let mut stats = s.stats;
        stats.total_secs = s.started.elapsed().as_secs_f64();
        stats.rtf = stats.total_secs / stats.audio_secs.max(1e-9);
        out.stats = stats;
        out
    }

    /// The decoder input of a stream's next window: context, then `<|startoftranscript|>
    /// [language task] [notimestamps]` and the prefix.
    fn prompt(&self, plan: &Plan, s: &Stream) -> Result<Prompt> {
        let sp = &self.sp;
        let o = plan.o;
        let cut_off = self.model.cfg.max_target_positions / 2 - 1;
        let mut input: Vec<u32> = Vec::new();
        if s.condition && !s.history.is_empty() {
            let flat: Vec<u32> = s
                .history
                .iter()
                .flat_map(|s| {
                    let skip = s.len() > 2 && s[s.len() - 2] >= sp.timestamp_begin;
                    s[..s.len() - skip as usize].iter().copied()
                })
                .collect();
            input.push(sp.sot_prev.context("no <|startofprev|> token")?);
            input.extend_from_slice(&flat[flat.len().saturating_sub(cut_off)..]);
        } else if let Some(p) = &plan.prompt {
            input.push(sp.sot_prev.context("no <|startofprev|> token")?);
            input.extend_from_slice(p);
        }
        let sot_index = input.len();
        let sot_len = if sp.multilingual { 3 } else { 1 };
        input.push(sp.sot);
        if sp.multilingual {
            input.push(s.language.unwrap());
            input.push(match plan.task {
                Task::Transcribe => sp.transcribe.context("no <|transcribe|> token")?,
                Task::Translate => sp.translate.context("no <|translate|> token")?,
            });
        }
        if !s.timestamps {
            input.push(sp.no_timestamps);
        }
        if s.seek == 0 {
            input.extend_from_slice(&plan.prefix);
        }
        if input.len() >= sp.max_length {
            bail!(
                "the prompt ({} tokens) leaves no room in the {}-token decoder",
                input.len(),
                sp.max_length
            );
        }
        let mut max_new = sp.max_length - input.len();
        if let Some(n) = o.max_new_tokens {
            max_new = max_new.min(n);
        }
        Ok(Prompt {
            input,
            sot_index,
            sot_len,
            max_new,
        })
    }

    /// Decodes the next window of every stream (none is done): one encoder pass, then the
    /// decoding attempts, the streams that need another temperature going on together.
    fn round(&self, plan: &Plan, streams: &mut [Stream], on_segment: &mut dyn FnMut(usize, &Segment) -> Flow) -> Result<()> {
        let m = &self.model;
        let sp = &self.sp;
        let o = plan.o;
        let win = m.window_frames();
        let n = streams.len();

        let te = Instant::now();
        let windows: Vec<Vec<f32>> = streams
            .iter()
            .map(|s| {
                let seek_num = (s.feats.frames - s.seek).min(win);
                let mut window = s.feats.window(s.seek);
                // Frames past this window's audio are zeros (the last window).
                window[seek_num * s.feats.n_mels..].fill(0.0);
                window.resize(win * s.feats.n_mels, 0.0);
                window
            })
            .collect();
        let refs: Vec<&[f32]> = windows.iter().map(|w| &w[..]).collect();
        let cross: Vec<model::CrossKv> = m.encode_batch(&refs).iter().map(|e| m.cross_kv(e)).collect();
        drop(windows);
        let encode_secs = te.elapsed().as_secs_f64();

        let td = Instant::now();
        let mut prompts: Vec<Prompt> = Vec::with_capacity(n);
        for (s, c) in streams.iter_mut().zip(&cross) {
            s.stats.encode_secs += encode_secs;
            s.stats.windows += 1;
            if sp.multilingual && s.language.is_none() {
                let probs = self.language_probs(c);
                let best = &probs[0];
                s.language = Some(sp.language_token(&best.0)?);
                s.out.language_probability = Some(best.1);
            }
            prompts.push(self.prompt(plan, s)?);
        }
        let begin_suppress: &[u32] = if o.suppress_blank { &sp.begin_suppress } else { &[] };
        let rules: Vec<Rules> = streams
            .iter()
            .map(|s| Rules {
                suppress: &plan.suppress,
                begin_suppress,
                timestamps: s.timestamps,
            })
            .collect();

        // Decode, retrying at higher temperatures while the thresholds say so.
        let mut chosen: Vec<Option<(decode::Attempt, f32)>> = (0..n).map(|_| None).collect();
        let mut skip = vec![false; n];
        let mut pending: Vec<usize> = (0..n).collect();
        for (fi, &temp) in o.temperatures.iter().enumerate() {
            // Beam search at temperature 0, sampling (the best of `best_of`) above it.
            let search = match plan.beam {
                Some((size, max_candidates)) if temp <= 0.0 => decode::Search::Beam { size, max_candidates },
                _ => decode::Search::Sample {
                    temperature: temp,
                    n: if temp > 0.0 { o.best_of.unwrap_or(1) } else { 1 },
                },
            };
            let attempts = {
                let jobs: Vec<decode::Job> = pending
                    .iter()
                    .map(|&i| decode::Job {
                        cross: &cross[i],
                        input: &prompts[i].input,
                        sot_index: prompts[i].sot_index,
                        rules: &rules[i],
                        search,
                        length_penalty: o.length_penalty,
                        // openai-whisper's beam search stops after half the context.
                        max_new: match search {
                            decode::Search::Beam { .. } => prompts[i].max_new.min(m.cfg.max_target_positions / 2),
                            decode::Search::Sample { .. } => prompts[i].max_new,
                        },
                        seed: o.seed.wrapping_add(streams[i].seek as u64 * 31 + fi as u64),
                    })
                    .collect();
                decode::run(m, sp, &jobs, &self.cancel)
            };
            let last = fi + 1 == o.temperatures.len();
            let mut again: Vec<usize> = Vec::new();
            for (&i, a) in pending.iter().zip(attempts) {
                let s = &mut streams[i];
                s.stats.tokens += a.steps;
                if a.cancelled {
                    s.out.stopped = true;
                    continue;
                }
                let cr = compress::compression_ratio(&a.tokens);
                let lp = a.logprobs.iter().sum::<f32>() / a.tokens.len().max(1) as f32;
                let mut fallback = o.compression_ratio_threshold.is_some_and(|t| cr > t) || o.logprob_threshold.is_some_and(|t| lp < t);
                if let (Some(nt), Some(lt)) = (o.no_speech_threshold, o.logprob_threshold)
                    && lp < lt
                    && a.no_speech_prob > nt
                {
                    fallback = false;
                    skip[i] = true;
                }
                chosen[i] = Some((a, temp));
                if fallback && !last {
                    again.push(i);
                }
            }
            pending = again;
            if pending.is_empty() {
                break;
            }
        }
        let decode_secs = td.elapsed().as_secs_f64();

        for (i, s) in streams.iter_mut().enumerate() {
            s.stats.decode_secs += decode_secs;
            if s.out.stopped {
                continue;
            }
            let (a, temp) = chosen[i].take().unwrap();
            self.finish_window(plan, s, &prompts[i], &cross[i], &a, temp, skip[i], on_segment);
        }
        Ok(())
    }

    /// Turns a window's chosen attempt into segments and moves the stream on.
    #[allow(clippy::too_many_arguments)]
    fn finish_window(
        &self,
        plan: &Plan,
        s: &mut Stream,
        prompt: &Prompt,
        cross: &model::CrossKv,
        a: &decode::Attempt,
        temp: f32,
        skip: bool,
        on_segment: &mut dyn FnMut(usize, &Segment) -> Flow,
    ) {
        let sp = &self.sp;
        let o = plan.o;
        let seek_num = (s.feats.frames - s.seek).min(self.model.window_frames());
        let time_offset = s.seek as f64 * FRAME_SECS;
        s.condition = o.condition_on_prev_tokens && temp < 0.5;
        if skip {
            s.seek += seek_num;
            return;
        }
        let cr = compress::compression_ratio(&a.tokens);
        let avg_lp = a.logprobs.iter().sum::<f32>() / a.tokens.len().max(1) as f32;
        let mut seq = a.tokens.clone();
        let mut probs: Vec<f32> = a.logprobs.iter().map(|v| v.exp()).collect();
        if seq.last() == Some(&sp.eot) {
            seq.pop();
            probs.pop();
        }
        let prefix = &plan.prefix;
        if s.seek == 0 && !prefix.is_empty() {
            seq.splice(0..0, prefix.iter().copied());
            probs.splice(0..0, prefix.iter().map(|_| 1.0));
        }
        let (raw, offset) = split_segments(sp.timestamp_begin, &seq, time_offset, seek_num);
        let mut segs: Vec<Segment> = Vec::new();
        let mut tok_at = 0usize;
        let mut seg_probs: Vec<Vec<f32>> = Vec::new();
        for (start, end, toks) in raw {
            // Locate the segment's tokens in `seq` for their probabilities.
            let n = toks.len();
            let pos = (tok_at..=seq.len().saturating_sub(n))
                .find(|&i| seq[i..i + n] == toks[..])
                .unwrap_or(tok_at);
            seg_probs.push(probs[pos..pos + n].to_vec());
            tok_at = pos + n;
            s.all_text.extend(toks.iter().copied().filter(|&t| t < sp.eot));
            s.history.push(toks.clone());
            let mut end = end;
            if !s.timestamps {
                end = end.min(s.stats.audio_secs);
            }
            segs.push(Segment {
                id: 0,
                start: round_ms(start),
                end: round_ms(end),
                text: tokens::decode_text(&self.tok, sp, &toks).trim().to_string(),
                tokens: toks,
                temperature: temp,
                avg_logprob: avg_lp,
                compression_ratio: cr,
                no_speech_prob: a.no_speech_prob,
                words: Vec::new(),
            });
        }
        if o.word_timestamps {
            self.add_words(
                &mut segs,
                &seg_probs,
                &prompt.input[prompt.sot_index..prompt.sot_index + prompt.sot_len],
                cross,
                seek_num,
                time_offset,
            );
        }
        s.seek += offset.max(1);
        for mut seg in segs {
            if !seg.tokens.iter().any(|&t| t < sp.eot) {
                continue;
            }
            seg.id = s.out.segments.len();
            if on_segment(s.index, &seg) == Flow::Stop {
                s.out.stopped = true;
            }
            s.out.segments.push(seg);
        }
    }

    /// Word timestamps for the segments of one window (openai-whisper's `add_word_timestamps`).
    fn add_words(
        &self,
        segs: &mut [Segment],
        probs: &[Vec<f32>],
        sot_seq: &[u32],
        cross: &model::CrossKv,
        seek_num: usize,
        time_offset: f64,
    ) {
        let sp = &self.sp;
        let mut text: Vec<u32> = Vec::new();
        let mut text_probs: Vec<f32> = Vec::new();
        for (s, p) in segs.iter().zip(probs) {
            for (&t, &pr) in s.tokens.iter().zip(p) {
                if t < sp.eot {
                    text.push(t);
                    text_probs.push(pr);
                }
            }
        }
        if text.is_empty() {
            return;
        }
        // [sot language task] <|notimestamps|> text <|endoftext|>
        let sot = sot_seq;
        let mut toks = sot.to_vec();
        toks.push(sp.no_timestamps);
        toks.extend_from_slice(&text);
        toks.push(sp.eot);
        if toks.len() > self.model.cfg.max_target_positions {
            return;
        }
        let scores = self.model.cross_attention_weights(&toks, cross, &sp.alignment_heads);
        let n_src = cross.layers[0].n;
        let n_frames = (seek_num / 2).clamp(1, n_src);
        let times = align::token_times(&scores, toks.len(), n_src, n_frames, sot.len());
        // Words over text + <|endoftext|>, grouped on Unicode boundaries first.
        let mut all = text.clone();
        all.push(sp.eot);
        let (mut texts, mut groups) = (Vec::new(), Vec::new());
        let mut cur: Vec<u32> = Vec::new();
        for &t in &all {
            cur.push(t);
            let s = self.tok.decode(&cur, false).unwrap_or_default();
            if !s.contains('\u{fffd}') || t >= sp.eot {
                texts.push(s);
                groups.push(std::mem::take(&mut cur));
            }
        }
        let no_spaces = matches!(
            self.lang_code_of(sot_seq).as_deref(),
            Some("zh" | "ja" | "th" | "lo" | "my" | "yue")
        );
        let (words, wtoks) = if no_spaces {
            (texts, groups)
        } else {
            align::split_on_spaces(texts, groups, sp.eot)
        };
        let mut raw = Vec::new();
        let mut k = 0usize;
        for (w, wt) in words.into_iter().zip(wtoks) {
            let start_i = k;
            k += wt.len();
            if wt[0] >= sp.eot {
                break;
            }
            let start = times.get(start_i).copied().unwrap_or(0.0) as f64 * TIMESTAMP_SECS + time_offset;
            let end = times.get(k).copied().unwrap_or(n_frames as f32) as f64 * TIMESTAMP_SECS + time_offset;
            let p = text_probs[start_i..k.min(text_probs.len())].iter().sum::<f32>() / wt.len() as f32;
            raw.push(align::RawWord {
                word: w,
                tokens: wt,
                start,
                end,
                probability: p,
            });
        }
        align::merge_punctuations(&mut raw);
        // Hand the words to the segments in token order.
        let mut wi = 0;
        for s in segs.iter_mut() {
            let mut need = s.tokens.iter().filter(|&&t| t < sp.eot).count();
            while need > 0 && wi < raw.len() {
                let w = &raw[wi];
                need = need.saturating_sub(w.tokens.len());
                let start = round_ms(w.start.max(s.start));
                s.words.push(Word {
                    word: w.word.clone(),
                    start,
                    end: round_ms(w.end.min(s.end)).max(start),
                    probability: w.probability,
                });
                wi += 1;
            }
        }
    }

    fn lang_code_of(&self, sot_seq: &[u32]) -> Option<String> {
        sot_seq.iter().find_map(|&t| self.sp.language_of(t)).map(str::to_string)
    }
}

/// Applies the logits rules (transformers' processor order: begin-suppress, suppress,
/// timestamps) for the tokens generated so far (`done`).
fn process(sp: &Special, logits: &mut [f32], done: &[u32], r: &Rules) {
    let ninf = f32::NEG_INFINITY;
    if done.is_empty() {
        for &t in r.begin_suppress {
            if let Some(v) = logits.get_mut(t as usize) {
                *v = ninf;
            }
        }
    }
    for (v, &s) in logits.iter_mut().zip(r.suppress) {
        if s {
            *v = ninf;
        }
    }
    if !r.timestamps {
        return;
    }
    let tb = sp.timestamp_begin as usize;
    let eot = sp.eot as usize;
    logits[sp.no_timestamps as usize] = ninf;
    let n = done.len();
    let last_ts = n >= 1 && done[n - 1] as usize >= tb;
    let penult_ts = n < 2 || done[n - 2] as usize >= tb;
    if last_ts {
        if penult_ts {
            logits[tb..].fill(ninf);
        } else {
            logits[..eot].fill(ninf);
        }
    }
    let mut ts_last = tb;
    if let Some(&t) = done.iter().rev().find(|&&t| t as usize >= tb) {
        ts_last = if last_ts && !penult_ts { t as usize } else { t as usize + 1 };
    }
    let hi = ts_last.min(logits.len());
    logits[tb..hi].fill(ninf);
    if done.is_empty() {
        logits[..tb].fill(ninf);
        if let Some(m) = sp.max_initial_timestamp_index {
            let last_allowed = tb + m as usize;
            if last_allowed + 1 < logits.len() {
                logits[last_allowed + 1..].fill(ninf);
            }
        }
    }
    // Timestamps when their total probability beats every single text token.
    let lp = log_softmax(logits);
    let mx = lp[tb..].iter().copied().fold(ninf, f32::max);
    let ts_lp = if mx == ninf {
        ninf
    } else {
        mx + lp[tb..].iter().map(|&v| (v - mx).exp()).sum::<f32>().ln()
    };
    let text_max = lp[..tb].iter().copied().fold(ninf, f32::max);
    if ts_lp > text_max {
        logits[..tb].fill(ninf);
    }
}

/// transformers' `_retrieve_segment`: cuts a window's tokens (no `<|endoftext|>`) at
/// timestamp pairs into (start, end, tokens), and returns how many frames to move on.
fn split_segments(tb: u32, seq: &[u32], time_offset: f64, seek_num: usize) -> (Vec<(f64, f64, Vec<u32>)>, usize) {
    let is_ts: Vec<bool> = seq.iter().map(|&t| t >= tb).collect();
    let n = seq.len();
    let single_ending = n >= 2 && !is_ts[n - 2] && is_ts[n - 1];
    let mut slices: Vec<usize> = (0..n.saturating_sub(1))
        .filter(|&i| is_ts[i] && is_ts[i + 1])
        .map(|i| i + 1)
        .collect();
    let pos = |t: u32| t.saturating_sub(tb) as f64;
    if !slices.is_empty() {
        if single_ending {
            slices.push(n);
        } else {
            *slices.last_mut().unwrap() += 1;
        }
        let mut segs = Vec::new();
        let mut last = 0;
        for (i, &cur) in slices.iter().enumerate() {
            let is_last = i + 1 == slices.len();
            let sl = &seq[last..cur];
            let end_idx = if !is_last || single_ending { sl.len() - 1 } else { sl.len() - 2 };
            segs.push((
                time_offset + pos(sl[0]) * TIMESTAMP_SECS,
                time_offset + pos(sl[end_idx]) * TIMESTAMP_SECS,
                sl.to_vec(),
            ));
            last = cur;
        }
        let offset = if single_ending {
            seek_num
        } else {
            // Seek to the last complete segment's end (2 mel frames per timestamp step).
            seq[last - 2].saturating_sub(tb) as usize * 2
        };
        (segs, offset)
    } else {
        let mut last_pos = (seek_num as f64 * FRAME_SECS / TIMESTAMP_SECS).floor();
        if let Some(&t) = seq.iter().rev().find(|&&t| t >= tb)
            && t != tb
        {
            last_pos = pos(t);
        }
        (vec![(time_offset, time_offset + last_pos * TIMESTAMP_SECS, seq.to_vec())], seek_num)
    }
}

/// Seconds rounded to the millisecond (timestamps are multiples of 20 ms).
fn round_ms(t: f64) -> f64 {
    (t * 1000.0).round() / 1000.0
}

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + x.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    x.iter().map(|&v| v - lse).collect()
}

fn log_softmax_at(x: &[f32], i: usize) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + x.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    x[i] - lse
}

fn softmax_prob(x: &[f32], i: usize) -> f32 {
    log_softmax_at(x, i).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy vocabulary: text 0..10, eot 10, sot 11, notimestamps 12, timestamps 13..20.
    fn toy() -> Special {
        Special {
            eot: 10,
            sot: 11,
            sot_prev: None,
            no_timestamps: 12,
            no_speech: 11,
            transcribe: None,
            translate: None,
            timestamp_begin: 13,
            languages: Vec::new(),
            multilingual: false,
            suppress: Vec::new(),
            begin_suppress: Vec::new(),
            max_initial_timestamp_index: Some(2),
            max_length: 448,
            alignment_heads: Vec::new(),
            default_language: None,
            default_task: None,
        }
    }

    fn allowed(sp: &Special, done: &[u32], logits: &[f32]) -> Vec<usize> {
        let r = Rules {
            suppress: &[false; 20],
            begin_suppress: &[0],
            timestamps: true,
        };
        let mut l = logits.to_vec();
        process(sp, &mut l, done, &r);
        (0..l.len()).filter(|&i| l[i] > f32::NEG_INFINITY).collect()
    }

    #[test]
    fn timestamp_rules() {
        let sp = toy();
        let flat = [0f32; 20];
        // First token: a timestamp of at most max_initial_timestamp_index (13..=15).
        assert_eq!(allowed(&sp, &[], &flat), vec![13, 14, 15]);
        // After text: text, end of text, or a timestamp after the opening one (increasing).
        let mut text = flat;
        text[3] = 5.0;
        assert_eq!(allowed(&sp, &[14, 3], &text), (0..12).chain(15..20).collect::<Vec<_>>());
        // After a single timestamp following text: a timestamp (closing pair) or end of text,
        // not text; and not before the last timestamp.
        let mut eot = flat;
        eot[10] = 5.0;
        assert_eq!(allowed(&sp, &[14, 3, 16], &eot), vec![10, 11, 16, 17, 18, 19]);
        // ... where all timestamps together outweigh the end of text when it is not likely.
        assert_eq!(allowed(&sp, &[14, 3, 16], &text), vec![16, 17, 18, 19]);
        // After a timestamp pair: text only (incl. eot), no timestamps.
        assert_eq!(allowed(&sp, &[14, 3, 16, 16], &text), (0..12).collect::<Vec<_>>());
        // Timestamps together more likely than any text token: timestamps only.
        let mut ts = flat;
        for v in &mut ts[13..] {
            *v = 1.0;
        }
        assert_eq!(allowed(&sp, &[14, 3], &ts), (15..20).collect::<Vec<_>>());
        // Begin-suppress applies to the first token only.
        let r = Rules {
            suppress: &[false; 20],
            begin_suppress: &[0],
            timestamps: false,
        };
        let mut l = flat.to_vec();
        process(&sp, &mut l, &[], &r);
        assert_eq!(l[0], f32::NEG_INFINITY);
        let mut l = flat.to_vec();
        process(&sp, &mut l, &[1], &r);
        assert_eq!(l[0], 0.0);
    }

    #[test]
    fn segments_like_transformers() {
        // <|0.00|> a b <|0.20|><|0.20|> c d <|0.40|>: a single timestamp ends it, all consumed.
        let (s, off) = split_segments(100, &[100, 5, 6, 110, 110, 7, 8, 120], 10.0, 3000);
        assert_eq!(s.len(), 2);
        assert!((s[0].0 - 10.0).abs() < 1e-9 && (s[0].1 - 10.2).abs() < 1e-9 && s[0].2 == [100, 5, 6, 110]);
        assert!((s[1].0 - 10.2).abs() < 1e-9 && (s[1].1 - 10.4).abs() < 1e-9 && s[1].2 == [110, 7, 8, 120]);
        assert_eq!(off, 3000);
        // An unfinished segment after the last pair is dropped and the window moves to the
        // pair (2 mel frames per timestamp step).
        let (s, off) = split_segments(100, &[100, 5, 110, 110, 6, 7], 0.0, 3000);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].2, [100, 5, 110, 110]);
        assert!((s[0].1 - 0.2).abs() < 1e-9);
        assert_eq!(off, 20);
        // No timestamps: one segment over the window.
        let (s, off) = split_segments(100, &[5, 6, 7], 30.0, 1001);
        assert_eq!((s[0].0, s[0].1, off), (30.0, 30.0 + 500.0 * 0.02, 1001));
    }

    #[test]
    fn options_parse() {
        let o: TranscribeOptions = serde_json::from_str(r#"{"language":"fr","task":"translate","timestamps":true}"#).unwrap();
        assert_eq!(o.language.as_deref(), Some("fr"));
        assert_eq!(o.task, Some(Task::Translate));
        assert!(o.timestamps && o.suppress_blank);
        assert_eq!(o.temperatures, vec![0.0]);
        assert_eq!(
            (o.beam_size, o.patience, o.length_penalty, o.best_of, o.batch_size),
            (None, None, None, None, 8)
        );
        let o: TranscribeOptions = serde_json::from_str(r#"{"beam_size":5,"patience":2.0,"length_penalty":0.6,"best_of":3}"#).unwrap();
        assert_eq!(
            (o.beam_size, o.patience, o.length_penalty, o.best_of),
            (Some(5), Some(2.0), Some(0.6), Some(3))
        );
        let mut t = TranscriberOptions::default();
        t.set("quant", "q8").unwrap();
        assert_eq!(t.decoder, Precision::Q8);
        assert!(t.set("bogus", "1").is_err());
        assert_eq!(language_code("French"), Some("fr"));
        assert_eq!(language_code("<|de|>"), Some("de"));
        assert_eq!(language_code("castilian"), Some("es"));
    }
}
