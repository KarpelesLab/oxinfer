//! Text to speech with Fish Audio Drama 3 (`fishaudio/drama-3`, `model_type: fish_omni`): a
//! 35B mixture-of-experts language model that writes one audio frame per step, a codebook tower
//! that completes the frame ([`tower`]) and a 44.1 kHz codec ([`codec`]). The prompt, the sampler,
//! the chunking of long texts and the decoding loop follow Fish Audio's reference
//! (`fish_speech/models/text2semantic/fish_omni/` of fishaudio/fish-speech: `generate.py`,
//! `tower.py`, `chunking.py`).
//!
//! The sequence is a table of rows `[token, code 0 .. code 9]`: text rows hold a token and zeros,
//! audio rows `[semantic_start + code 0, code 0 .. code 9]`.
//!
//! - **Prompt** ([`Drama::prompt`]): a system turn (`convert the provided text to speech`; with
//!   references `... reference to the following:` and per reference its transcript behind a
//!   speaker tag and its codes), earlier chunks as finished user / assistant turns, the text
//!   behind a speaker tag (`<|speaker:0|>` .. `<|speaker:15|>`), then `<|im_start|>assistant` and
//!   `<|voice|>`. Every literal is tokenized on its own.
//! - **Backbone**: the Qwen3.5 mixture-of-experts hybrid ([`crate::qwen`]: Gated DeltaNet and
//!   gated attention layers, 256 experts of which 8 run per token), with the experts in FP8 as
//!   the checkpoint stores them. On audio rows its input is `(embedding of the main token +
//!   embedding of code 1) / sqrt(2)`.
//! - **Frame**: the backbone's logits over the end token and the semantic band give the main
//!   token (code 0); the tower writes the codes 1 .. 9 from the backbone's hidden state.
//! - **Sampling**: temperature, top-k and top-p on the main token and on every code; a main token
//!   that repeats one of the last 16 rows' is drawn again with fixed settings (temperature 1.0,
//!   top-k 30, top-p 0.95). Temperature 0 is plain greedy decoding.
//! - **Long texts** are cut into chunks of about 300 bytes at sentence ends; each chunk is
//!   generated with the first chunk and the latest one as history.
//!
//! Voices: a reference clip with its transcript ([`Reference`]) is encoded by the codec and put
//! in the prompt; without one the model picks a voice (the seed decides which).
//!
//! ```no_run
//! use oxinfer::fish::{Drama, DramaOptions, DramaSpeak};
//! # fn main() -> anyhow::Result<()> {
//! let d = Drama::load("fishaudio/drama-3", &DramaOptions::default())?;
//! d.speak("Hello there, how are you today?", &DramaSpeak::default())?.audio.write_wav("hello.wav")?;
//! # Ok(())
//! # }
//! ```

pub mod codec;
pub mod tower;

use crate::clock::Instant;
use crate::config::QwenConfig;
use crate::engine::{CancelHandle, Flow, StopReason};
use crate::model::{Lin, Quant, linear};
use crate::qwen::{QwenModel, QwenSession};
use crate::safetensors::ModelFiles;
use crate::tokenizer::Tokenizer;
use crate::tts::{Audio, AudioChunk, SpeakStats, Speech};
use anyhow::{Context, Result, anyhow, bail};
use codec::Codec;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tower::{Tower, TowerConfig, TowerState};

/// The repository of the released model.
pub const REPO: &str = "fishaudio/drama-3";

const TTS_INSTRUCTION: &str = "convert the provided text to speech";
const REFERENCE_INSTRUCTION: &str = "convert the provided text to speech reference to the following:";
const MESSAGE_END: &str = "<|im_end|>\n";
const GENERATION_START: &str = "<|im_start|>assistant\n<|voice|>";
/// Only the prompt builder writes these.
const STRUCTURAL_TOKENS: [&str; 6] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
];
const MAX_SPEAKER_ID: u32 = 15;
/// Rows of a prompt at most, and frames generated after it when no limit is given.
pub const MAX_PROMPT_ROWS: usize = 8192;
const DEFAULT_MAX_NEW_TOKENS: usize = 2048;
const CHUNK_MAX_NEW_TOKENS: usize = 600;
const PREFILL_CHUNK: usize = 128;
/// A main token equal to one of this many last rows' is drawn again with the settings below.
const RAS_WINDOW: usize = 16;
const RAS_TEMPERATURE: f32 = 1.0;
const RAS_TOP_K: usize = 30;
const RAS_TOP_P: f32 = 0.95;
const MIN_TEMPERATURE: f32 = 1e-5;
/// A chunk of a long text: about this many UTF-8 bytes and at most this many speakers.
const CHUNK_BYTES: usize = 300;
const CHUNK_MAX_SPEAKERS: usize = 3;

/// The Unicode `White_Space` property (what the reference strips).
fn is_space(c: char) -> bool {
    matches!(c, '\u{9}'..='\u{d}' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// A speaker tag (`<|speaker:N|>`) at the start of `s`: its digits and its length in bytes.
fn speaker_tag(s: &str) -> Option<(&str, usize)> {
    let rest = s.strip_prefix("<|speaker:")?;
    let n = rest.bytes().take_while(u8::is_ascii_digit).count();
    if n == 0 || !rest[n..].starts_with("|>") {
        return None;
    }
    Some((&rest[..n], "<|speaker:".len() + n + 2))
}

/// The reference's `validate_drama3_text`: not empty, none of the tokens the prompt builder
/// writes, and only the speaker tags 0 .. 15 in their canonical spelling.
pub fn validate_text(text: &str, what: &str) -> Result<()> {
    if strip(text).is_empty() {
        bail!("{what}: the text is empty");
    }
    for t in STRUCTURAL_TOKENS {
        if text.contains(t) {
            bail!("{what}: {t} is written by the prompt builder");
        }
    }
    for (i, _) in text.match_indices("<|speaker") {
        let ok = speaker_tag(&text[i..])
            .is_some_and(|(d, _)| (d == "0" || !d.starts_with('0')) && d.parse::<u32>().is_ok_and(|n| n <= MAX_SPEAKER_ID));
        if !ok {
            bail!("{what}: speaker tags must be <|speaker:0|> to <|speaker:{MAX_SPEAKER_ID}|>");
        }
    }
    Ok(())
}

/// `text` behind `<|speaker:N|>` unless a tag already leads it.
fn with_speaker_tag(text: &str, speaker: u32) -> String {
    if speaker_tag(text.trim_start_matches(is_space)).is_some() {
        text.to_string()
    } else {
        format!("<|speaker:{speaker}|> {text}")
    }
}

/// The pieces of `text` that a chunk boundary must not cut (special tokens, paired tags such as
/// `<whisper>...</whisper>`, `[bracketed]` cues), flagged, between the free ones.
fn protected_pieces(text: &str) -> Vec<(&str, bool)> {
    // The end of a protected span starting at byte `i`, if one starts there.
    let span_end = |i: usize| -> Option<usize> {
        let s = &text[i..];
        if let Some(rest) = s.strip_prefix("<|") {
            // `<|...|>` without `|` or `>` inside.
            let n = rest.find(['|', '>'])?;
            if rest[n..].starts_with("|>") {
                return Some(i + 2 + n + 2);
            }
        }
        if let Some(tag) = s.strip_prefix('<') {
            // `<name ...>...</name>`.
            let name_len = tag
                .char_indices()
                .take_while(|&(j, c)| {
                    if j == 0 {
                        c.is_ascii_alphabetic()
                    } else {
                        c.is_alphanumeric() || c == '_' || c == '-'
                    }
                })
                .map(|(j, c)| j + c.len_utf8())
                .last()?;
            let (name, after) = tag.split_at(name_len);
            // `\b` after the name: the next character is not a word character.
            let open = after.find('>')?;
            if after.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_') {
                return None;
            }
            let body = &after[open + 1..];
            let close = format!("</{name}>");
            let n = body.find(&close)?;
            return Some(i + 1 + name_len + open + 1 + n + close.len());
        }
        if let Some(rest) = s.strip_prefix('[') {
            let n = rest.find(['[', ']'])?;
            if rest[n..].starts_with(']') {
                return Some(i + 1 + n + 1);
            }
        }
        None
    };
    let mut out = Vec::new();
    let (mut last, mut i) = (0, 0);
    while i < text.len() {
        let c = text[i..].chars().next().unwrap();
        if (c == '<' || c == '[')
            && let Some(end) = span_end(i)
        {
            if i > last {
                out.push((&text[last..i], false));
            }
            out.push((&text[i..end], true));
            last = end;
            i = end;
            continue;
        }
        i += c.len_utf8();
    }
    if last < text.len() {
        out.push((&text[last..], false));
    }
    out
}

/// Free pieces longer than `length` bytes cut after each of `marks` (a dot between two digits
/// stays).
fn break_at_marks<'t>(pieces: Vec<(&'t str, bool)>, length: usize, marks: &[char]) -> Vec<(&'t str, bool)> {
    let mut out = Vec::new();
    for (text, protected) in pieces {
        if protected || text.len() <= length {
            out.push((text, protected));
            continue;
        }
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut start = 0;
        for (n, &(i, c)) in chars.iter().enumerate() {
            if !marks.contains(&c) {
                continue;
            }
            let decimal = c == '.' && n > 0 && n + 1 < chars.len() && chars[n - 1].1.is_numeric() && chars[n + 1].1.is_numeric();
            if !decimal {
                let end = i + c.len_utf8();
                out.push((&text[start..end], false));
                start = end;
            }
        }
        if start < text.len() {
            out.push((&text[start..], false));
        }
    }
    out
}

/// Chunks of about `length` bytes, cut at sentence ends, then clause marks, then spaces, then
/// anywhere (`split_by_length` of the reference's `chunking.py`).
fn split_by_length(text: &str, length: usize) -> Vec<String> {
    if strip(text).is_empty() {
        return Vec::new();
    }
    if text.len() <= length {
        return vec![text.to_string()];
    }
    let mut pieces = break_at_marks(
        protected_pieces(text),
        0,
        &['.', '!', '?', ';', '\u{3002}', '\u{ff01}', '\u{ff1f}', '\u{ff1b}', '\u{2026}', '\n'],
    );
    pieces = break_at_marks(pieces, length, &[',', ':', '\u{ff0c}', '\u{ff1a}', '\u{3001}']);
    pieces = break_at_marks(pieces, length, &[' ']);
    let mut cut: Vec<&str> = Vec::new();
    for (text, protected) in pieces {
        if protected || text.len() <= length {
            cut.push(text);
            continue;
        }
        let mut start = 0;
        let mut size = 0;
        for (i, c) in text.char_indices() {
            if size > 0 && size + c.len_utf8() > length {
                cut.push(&text[start..i]);
                start = i;
                size = 0;
            }
            size += c.len_utf8();
        }
        cut.push(&text[start..]);
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for piece in cut {
        if !current.is_empty() && current.len() + piece.len() > length {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(piece);
    }
    chunks.push(current);
    chunks.iter().map(|c| strip(c).to_string()).filter(|c| !c.is_empty()).collect()
}

/// The chunks a long text is generated in (`split_drama3_text`): untagged text by length; tagged
/// text by speaker segment, regrouped so that a chunk holds at most three speakers and about 300
/// bytes.
pub fn split_text(text: &str) -> Vec<String> {
    if !text.match_indices("<|speaker:").any(|(i, _)| speaker_tag(&text[i..]).is_some()) {
        return split_by_length(text, CHUNK_BYTES);
    }
    // (speaker, text) per tagged segment; text before the first tag is speaker 0's.
    let mut segments: Vec<(String, String)> = Vec::new();
    let mut speaker = "0".to_string();
    let mut start = 0;
    let mut i = 0;
    let push = |speaker: &str, body: &str, segments: &mut Vec<(String, String)>| {
        for piece in split_by_length(body, CHUNK_BYTES) {
            segments.push((speaker.to_string(), format!("<|speaker:{speaker}|> {piece}")));
        }
    };
    while i < text.len() {
        if let Some((digits, len)) = speaker_tag(&text[i..]) {
            push(&speaker, &text[start..i], &mut segments);
            speaker = digits.to_string();
            i += len;
            start = i;
        } else {
            i += text[i..].chars().next().unwrap().len_utf8();
        }
    }
    push(&speaker, &text[start..], &mut segments);

    let mut chunks = Vec::new();
    let mut batch: Vec<String> = Vec::new();
    let mut speakers: Vec<String> = Vec::new();
    let mut size = 0;
    for (speaker, segment) in segments {
        let mut new_speaker = !speakers.contains(&speaker);
        let too_many = new_speaker && speakers.len() >= CHUNK_MAX_SPEAKERS;
        let too_long = !batch.is_empty() && size + segment.len() > CHUNK_BYTES;
        if too_many || too_long {
            chunks.push(batch.join("\n"));
            batch.clear();
            speakers.clear();
            size = 0;
            new_speaker = true;
        }
        size += segment.len();
        batch.push(segment);
        if new_speaker {
            speakers.push(speaker);
        }
    }
    if !batch.is_empty() {
        chunks.push(batch.join("\n"));
    }
    chunks
}

/// How to load the model.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DramaOptions {
    /// The precision of the matrices the checkpoint stores as bf16 (the backbone's mixers and
    /// shared experts, the tower): bf16 as stored (the default), or Q8 / Q4, quantized at load,
    /// which halve / quarter the memory read per frame. The FP8 experts stay as they are.
    pub quant: Quant,
    /// CPU worker threads (process-wide; default: all cores).
    pub threads: Option<usize>,
    /// Map the weights from disk instead of reading them into memory.
    pub mmap: bool,
    /// Where repo ids are downloaded.
    pub cache_dir: Option<PathBuf>,
}

impl Default for DramaOptions {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            threads: None,
            mmap: false,
            cache_dir: None,
        }
    }
}

/// A voice to clone: a clip and what is said in it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reference {
    /// An audio file, or the clip's codes as a `.npy` file (`[10, frames]`, as fish-speech writes
    /// them, or `[frames, 10]`).
    pub audio: PathBuf,
    /// The transcript of the clip (a leading `<|speaker:N|>` names the speaker it is; default 0).
    pub text: String,
}

/// Options of one utterance; the defaults are fish-speech's. Deserializes from JSON with every
/// field optional.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct DramaSpeak {
    /// 0 = greedy (and no redraw of repeated tokens).
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Frames generated at most per chunk of text, 21.5 per second of audio (0: 600 per chunk,
    /// 2048 with `single_chunk`).
    pub max_tokens: usize,
    pub seed: u64,
    /// The speaker tag (0 .. 15) put before text that does not start with one.
    pub speaker: u32,
    /// Voices to clone, in the order `[Reference 1]`, `[Reference 2]`, ...
    pub references: Vec<Reference>,
    /// Generate the whole text in one pass instead of chunk by chunk.
    pub single_chunk: bool,
    /// Draw a main token again when it repeats one of the last 16 rows' (the reference's
    /// repetition-aware sampling).
    pub redraw_repeats: bool,
}

impl Default for DramaSpeak {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 30,
            top_p: 0.7,
            max_tokens: 0,
            seed: 42,
            speaker: 0,
            references: Vec::new(),
            single_chunk: false,
            redraw_repeats: true,
        }
    }
}

impl DramaSpeak {
    /// Options from a JSON object (unset fields keep their defaults).
    pub fn from_json(json: &str) -> Result<Self> {
        if json.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(json).context("speak options")
    }
}

/// What a loaded model is.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DramaInfo {
    pub path: PathBuf,
    pub quant: Quant,
    pub layers: usize,
    pub hidden_size: usize,
    /// Bytes of the checkpoint and of the codec decoder.
    pub weight_bytes: u64,
    pub threads: usize,
    pub sample_rate: u32,
    /// Frames per second of audio.
    pub frame_rate: f64,
    pub n_codebooks: usize,
    pub load_secs: f64,
}

/// The frames the model wrote for one prompt.
#[derive(Clone, Debug)]
pub struct Generated {
    /// `[frames, 1 + n_codebooks]`: the main token and the codes of every frame.
    pub rows: Vec<u32>,
    pub finish: StopReason,
    pub prompt_rows: usize,
    pub prefill_secs: f64,
    pub generate_secs: f64,
    /// With `keep_logits`: per step, the logits over the main candidates (the end token, then
    /// the semantic band) followed by those of the nine codebook heads.
    pub logits: Vec<Vec<f32>>,
}

/// The ids named by `config.json` beside the two model configurations.
#[derive(Debug, Clone, Deserialize)]
struct TopConfig {
    semantic_start_token_id: u32,
    semantic_end_token_id: u32,
    eos_token_id: u32,
    audio_decoder_config: TowerConfig,
    #[serde(default)]
    speech_template_version: Option<String>,
}

pub struct Drama {
    model: QwenModel<'static>,
    tower: Tower<'static>,
    /// The output rows of the end token and of the semantic band, in that order.
    head: Lin<'static>,
    codec: Codec,
    tok: Tokenizer,
    semantic_start: u32,
    semantic_end: u32,
    eos: u32,
    info: DramaInfo,
    cancel: Arc<AtomicBool>,
    /// The last clip encoded, so that the chunks of a text in one cloned voice encode it once.
    last_clip: std::sync::Mutex<Option<(PathBuf, Vec<u32>)>>,
    /// Must stay last: `model`, `tower` and `head` borrow from it.
    _files: Box<ModelFiles>,
}

/// Whether `model` (a path or a repo id) is a Drama 3 model: a folder whose `config.json` says
/// `fish_omni`, or the repository's id.
pub fn is_drama(model: &Path) -> bool {
    if model.is_dir() {
        return crate::config::model_type(model).is_ok_and(|t| t == "fish_omni");
    }
    let s = model.to_string_lossy().to_ascii_lowercase();
    s.split('/').count() == 2 && s.starts_with("fishaudio/drama")
}

/// xoshiro256** seeded through splitmix64.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self([next(), next(), next(), next()])
    }
    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        (r >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn argmax(l: &[f32]) -> usize {
    l.iter().enumerate().fold(0, |b, (i, &v)| if v > l[b] { i } else { b })
}

/// One index of `logits` as the reference's `sample_top_k_top_p` draws it: over the logits sorted
/// by rank, the ranks from `top_k` on are dropped, and so is every rank whose cumulative
/// probability (its own included) exceeds `top_p`; the first rank always stays.
fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut Rng) -> usize {
    let t = temperature.max(MIN_TEMPERATURE);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let top = logits[order[0]] as f64 / t as f64;
    let p: Vec<f64> = order.iter().map(|&i| (logits[i] as f64 / t as f64 - top).exp()).collect();
    let total: f64 = p.iter().sum();
    let mut kept = Vec::with_capacity(top_k.max(1));
    let mut cum = 0.0;
    for (rank, &v) in p.iter().enumerate() {
        cum += v / total;
        if rank > 0 && (rank >= top_k || cum > top_p as f64) {
            break;
        }
        kept.push(v);
    }
    let mut u = rng.uniform() * kept.iter().sum::<f64>();
    for (rank, &v) in kept.iter().enumerate() {
        if u < v {
            return order[rank];
        }
        u -= v;
    }
    order[kept.len() - 1]
}

impl Drama {
    /// Loads the model: a repo id ([`REPO`]: the checkpoint and `codec.pth` are fetched, 41.5 GB)
    /// or a folder holding them.
    pub fn load(model: &str, opts: &DramaOptions) -> Result<Self> {
        let dir = crate::engine::resolve_model(model, opts.cache_dir.as_deref())?;
        #[cfg(feature = "hf")]
        if !dir.join("codec.pth").is_file() && !Path::new(model).is_dir() {
            crate::hf::resolve_files(model, &["codec.pth"], opts.cache_dir.as_deref())?;
        }
        Self::load_dir(&dir, opts)
    }

    /// Loads from a local folder (`config.json`, `tokenizer.json`, the safetensors shards and
    /// `codec.pth`).
    pub fn load_dir(dir: &Path, opts: &DramaOptions) -> Result<Self> {
        let t0 = Instant::now();
        if let Some(n) = opts.threads {
            let mut o = crate::pool::options();
            o.threads = n;
            crate::pool::configure_with(o);
        }
        let raw = crate::fsx::read(&dir.join("config.json")).with_context(|| format!("read {}/config.json", dir.display()))?;
        let top: TopConfig = serde_json::from_slice(&raw).context("parse the fish_omni config")?;
        if let Some(v) = top.speech_template_version.as_deref()
            && v != "reference-bracket-v1"
        {
            bail!("{}: speech_template_version {v} is not supported", dir.display());
        }
        let cfg = QwenConfig::load(dir)?;
        let tcfg = top.audio_decoder_config.clone();
        if tcfg.dim != cfg.hidden_size {
            bail!(
                "{}: the tower's width {} is not the backbone's {}",
                dir.display(),
                tcfg.dim,
                cfg.hidden_size
            );
        }
        let band = (top.semantic_end_token_id + 1).saturating_sub(top.semantic_start_token_id) as usize;
        if band != tcfg.vocab_size || top.semantic_end_token_id as usize >= cfg.vocab_size {
            bail!("{}: the semantic band does not match the tower's vocabulary", dir.display());
        }
        let codec_path = dir.join("codec.pth");
        if !codec_path.is_file() {
            bail!("{}: the codec is missing (codec.pth of {REPO})", codec_path.display());
        }
        let codec = Codec::load(&codec_path)?;
        if codec.n_codebooks() != tcfg.num_codebooks || codec.codebook_size(0) != tcfg.vocab_size {
            bail!("{}: not the codec of this model", codec_path.display());
        }
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{}/tokenizer.json: {e}", dir.display()))?;
        for (t, id) in [
            ("<|semantic:0|>".to_string(), top.semantic_start_token_id),
            (format!("<|semantic:{}|>", band - 1), top.semantic_end_token_id),
            ("<|im_end|>".to_string(), top.eos_token_id),
        ] {
            if tok.token_to_id(&t) != Some(id) {
                bail!("{}: the tokenizer and the config disagree on {t}", dir.display());
            }
        }
        let files = Box::new(ModelFiles::open(dir, !opts.mmap)?);
        // SAFETY: the files are boxed (their address does not move with the Drama) and dropped
        // after the model, the tower and the head that borrow them (`_files` is the last field).
        let fr: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let model = QwenModel::load(cfg, fr, opts.quant)?;
        let tower = Tower::load(tcfg, fr, model.quant)?;
        let h = model.cfg.hidden_size;
        // The rows of the main candidates: the end token, then the band.
        let w = fr.bf16("lm_head.weight")?;
        if w.len() != model.cfg.vocab_size * h {
            bail!("lm_head.weight: {} values, expected {}", w.len(), model.cfg.vocab_size * h);
        }
        let mut rows = crate::quant::PageVec::<u16>::zeroed((1 + band) * h);
        let (e, s) = (top.eos_token_id as usize, top.semantic_start_token_id as usize);
        rows[..h].copy_from_slice(&w[e * h..(e + 1) * h]);
        rows[h..].copy_from_slice(&w[s * h..(s + band) * h]);
        let info = DramaInfo {
            path: dir.to_path_buf(),
            quant: model.quant,
            layers: model.cfg.num_hidden_layers,
            hidden_size: h,
            weight_bytes: files.total_bytes() + codec.weight_bytes as u64,
            threads: crate::pool::threads(),
            sample_rate: codec::SAMPLE_RATE,
            frame_rate: codec::SAMPLE_RATE as f64 / codec::HOP as f64,
            n_codebooks: codec.n_codebooks(),
            load_secs: t0.elapsed().as_secs_f64(),
        };
        Ok(Self {
            model,
            tower,
            head: Lin::bf16_owned(rows),
            codec,
            tok,
            semantic_start: top.semantic_start_token_id,
            semantic_end: top.semantic_end_token_id,
            eos: top.eos_token_id,
            info,
            cancel: Arc::new(AtomicBool::new(false)),
            last_clip: std::sync::Mutex::new(None),
            _files: files,
        })
    }

    pub fn info(&self) -> &DramaInfo {
        &self.info
    }

    pub fn codec(&self) -> &Codec {
        &self.codec
    }

    /// The backbone (for checks: its full logits).
    pub fn backbone(&self) -> &QwenModel<'static> {
        &self.model
    }

    /// A handle that stops the running (or next) utterance from another thread; the audio
    /// generated so far is returned.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle::from_flag(self.cancel.clone())
    }

    /// The ids of one literal of the prompt.
    pub fn ids(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(text, true)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    fn check_codes(&self, codes: &[u32], what: &str) -> Result<()> {
        let n = self.info.n_codebooks;
        if codes.is_empty() || !codes.len().is_multiple_of(n) {
            bail!("{what}: {} values are not frames of {n} codes", codes.len());
        }
        if codes.iter().any(|&c| c as usize >= self.tower.cfg.vocab_size) {
            bail!("{what}: codes must be below {}", self.tower.cfg.vocab_size);
        }
        Ok(())
    }

    /// The rows (`[n, 1 + n_codebooks]`) of the prompt of `text`: `references` are `(transcript,
    /// codes)` of the voices to clone, `history` `(text, codes)` of earlier chunks, codes as
    /// `[frames, n_codebooks]`; `speaker` tags untagged text. Ends with the open assistant turn.
    pub fn prompt(&self, text: &str, speaker: u32, references: &[(String, Vec<u32>)], history: &[(String, Vec<u32>)]) -> Result<Vec<u32>> {
        if speaker > MAX_SPEAKER_ID {
            bail!("speaker {speaker}: the model has the speakers 0 to {MAX_SPEAKER_ID}");
        }
        validate_text(text, "text")?;
        let n_cb = self.info.n_codebooks;
        let mut rows: Vec<u32> = Vec::new();
        let lit = |rows: &mut Vec<u32>, s: &str| -> Result<()> {
            for id in self.ids(s)? {
                rows.push(id);
                rows.extend(std::iter::repeat_n(0, n_cb));
            }
            Ok(())
        };
        let audio = |rows: &mut Vec<u32>, codes: &[u32]| {
            for f in codes.chunks(n_cb) {
                rows.push(self.semantic_start + f[0]);
                rows.extend_from_slice(f);
            }
        };
        lit(&mut rows, "<|im_start|>system\n")?;
        lit(
            &mut rows,
            if references.is_empty() {
                TTS_INSTRUCTION
            } else {
                REFERENCE_INSTRUCTION
            },
        )?;
        for (i, (transcript, codes)) in references.iter().enumerate() {
            let what = format!("reference {}", i + 1);
            validate_text(transcript, &what)?;
            self.check_codes(codes, &what)?;
            // The tag follows "Text: " directly; leading white space moves behind it.
            let tagged = with_speaker_tag(transcript, 0);
            let body = tagged.trim_start_matches(is_space);
            let (_, tag_len) = speaker_tag(body).expect("tagged above");
            let leading = &tagged[..tagged.len() - body.len()];
            let transcript = format!("{}{leading}{}", &body[..tag_len], &body[tag_len..]);
            lit(&mut rows, &format!("\n\n[Reference {}]\nText: {transcript}\nSpeech: ", i + 1))?;
            audio(&mut rows, codes);
        }
        lit(&mut rows, MESSAGE_END)?;
        for (i, (turn, codes)) in history.iter().enumerate() {
            let what = format!("history {}", i + 1);
            validate_text(turn, &what)?;
            self.check_codes(codes, &what)?;
            lit(&mut rows, "<|im_start|>user\n")?;
            lit(&mut rows, &with_speaker_tag(turn, speaker))?;
            lit(&mut rows, MESSAGE_END)?;
            lit(&mut rows, GENERATION_START)?;
            audio(&mut rows, codes);
            lit(&mut rows, MESSAGE_END)?;
        }
        lit(&mut rows, "<|im_start|>user\n")?;
        lit(&mut rows, &with_speaker_tag(text, speaker))?;
        lit(&mut rows, MESSAGE_END)?;
        lit(&mut rows, GENERATION_START)?;
        if rows.len() / (1 + n_cb) > MAX_PROMPT_ROWS {
            bail!("the prompt has {} rows; at most {MAX_PROMPT_ROWS}", rows.len() / (1 + n_cb));
        }
        Ok(rows)
    }

    fn is_semantic(&self, token: u32) -> bool {
        (self.semantic_start..=self.semantic_end).contains(&token)
    }

    /// The backbone's states (`[t, hidden]`) of `rows` appended to the session.
    fn backbone_rows(&self, sess: &mut QwenSession, rows: &[u32]) -> Vec<f32> {
        let w = 1 + self.info.n_codebooks;
        let h = self.info.hidden_size;
        let tokens: Vec<u32> = rows.chunks(w).map(|r| r[0]).collect();
        let mut mixed: Vec<(usize, Vec<f32>)> = Vec::new();
        for (i, r) in rows.chunks(w).enumerate() {
            if self.is_semantic(r[0]) {
                let mut e = vec![0f32; h];
                self.model.embed.row(r[0] as usize, &mut e);
                self.tower.embed_backbone(&r[1..], &mut e);
                mixed.push((i, e));
            }
        }
        let mm: Vec<(usize, &[f32])> = mixed.iter().map(|(i, e)| (*i, &e[..])).collect();
        self.model.forward_hidden_mm(sess, &tokens, &mm)
    }

    /// The backbone's logits over the whole vocabulary for the last row of `prompt` (for checks
    /// against the reference).
    pub fn prompt_logits(&self, prompt: &[u32]) -> Vec<f32> {
        let h = self.info.hidden_size;
        let mut sess = self.model.new_session();
        let hs = self.backbone_rows(&mut sess, prompt);
        self.model.logits(&hs[hs.len() - h..], 1)
    }

    /// Generates at most `max_new` frames after `prompt` ([`Drama::prompt`]). `on_frame(rows so
    /// far)` is called after each frame; `keep_logits` records every step's logits.
    pub fn generate(
        &self,
        prompt: &[u32],
        o: &DramaSpeak,
        max_new: usize,
        keep_logits: bool,
        on_frame: &mut dyn FnMut(&[u32]) -> Flow,
    ) -> Result<Generated> {
        let w = 1 + self.info.n_codebooks;
        let h = self.info.hidden_size;
        let n = prompt.len() / w;
        if n == 0 || !prompt.len().is_multiple_of(w) {
            bail!("the prompt is not rows of {w} ids");
        }
        if o.top_k < 1 {
            bail!("top_k must be at least 1");
        }
        let t0 = Instant::now();
        let mut sess = self.model.new_session();
        let mut ts: TowerState = self.tower.new_state();
        let codes_of = |r: &[u32]| -> Option<Vec<u32>> { self.is_semantic(r[0]).then(|| r[1..].to_vec()) };
        // Prefill: the backbone over the prompt, and the tower over every row but the last, each
        // labelled with the row after it.
        let mut hidden = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + PREFILL_CHUNK).min(n);
            let hs = self.backbone_rows(&mut sess, &prompt[start * w..end * w]);
            let paired = end.min(n - 1);
            if paired > start {
                let own: Vec<Option<Vec<u32>>> = (start..paired).map(|i| codes_of(&prompt[i * w..(i + 1) * w])).collect();
                let next: Vec<Option<Vec<u32>>> = (start..paired).map(|i| codes_of(&prompt[(i + 1) * w..(i + 2) * w])).collect();
                let own: Vec<Option<&[u32]>> = own.iter().map(|c| c.as_deref()).collect();
                let next: Vec<Option<&[u32]>> = next.iter().map(|c| c.as_deref()).collect();
                self.tower.prefill(&mut ts, &hs[..(paired - start) * h], &own, &next);
            }
            hidden = hs[(end - start - 1) * h..].to_vec();
            start = end;
            if self.cancel.load(Ordering::Relaxed) {
                break;
            }
        }
        let prefill_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let greedy = o.temperature <= 0.0;
        let mut rng = Rng::new(o.seed);
        let band = self.tower.cfg.vocab_size;
        // The main tokens of the last rows (the prompt's included).
        let mut recent: Vec<u32> = prompt.chunks(w).map(|r| r[0]).collect();
        let mut last: Vec<u32> = prompt[(n - 1) * w..].to_vec();
        let mut rows: Vec<u32> = Vec::new();
        let mut logits_log = Vec::new();
        let mut finish = StopReason::Length;
        let mut main_logits = vec![0f32; 1 + band];
        for step in 0..max_new.max(1) {
            if self.cancel.swap(false, Ordering::Relaxed) {
                finish = StopReason::Cancelled;
                break;
            }
            linear(&self.head, 1 + band, h, &hidden, 1, &mut main_logits);
            let token_of = |i: usize| if i == 0 { self.eos } else { self.semantic_start + i as u32 - 1 };
            let main = if greedy {
                token_of(argmax(&main_logits))
            } else {
                let first = token_of(sample(&main_logits, o.temperature, o.top_k, o.top_p, &mut rng));
                // The second draw is taken whether it is used or not, as the reference does.
                let retry = token_of(sample(&main_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, &mut rng));
                let window = &recent[recent.len().saturating_sub(RAS_WINDOW)..];
                if o.redraw_repeats && self.is_semantic(first) && window.contains(&first) {
                    retry
                } else {
                    first
                }
            };
            let c0 = self.is_semantic(main).then(|| main - self.semantic_start);
            let mut step_logits = if keep_logits { main_logits.clone() } else { Vec::new() };
            let own = codes_of(&last);
            let codes = self.tower.step(&mut ts, &hidden, own.as_deref(), c0, &mut |_, l| {
                if keep_logits {
                    step_logits.extend_from_slice(l);
                }
                if greedy {
                    argmax(l) as u32
                } else {
                    sample(l, o.temperature, o.top_k, o.top_p, &mut rng) as u32
                }
            });
            if keep_logits {
                logits_log.push(step_logits);
            }
            if main == self.eos {
                finish = StopReason::Stop;
                break;
            }
            last = std::iter::once(main).chain(codes).collect();
            rows.extend_from_slice(&last);
            recent.push(main);
            if matches!(on_frame(&rows), Flow::Stop) {
                finish = StopReason::Cancelled;
                break;
            }
            if step + 1 < max_new {
                hidden = self.backbone_rows(&mut sess, &last);
            }
        }
        Ok(Generated {
            rows,
            finish,
            prompt_rows: n,
            prefill_secs,
            generate_secs: t1.elapsed().as_secs_f64(),
            logits: logits_log,
        })
    }

    /// The codes (`[frames, n_codebooks]`) of mono audio, resampled to the codec's 44.1 kHz.
    pub fn encode_audio(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<u32>> {
        if sample_rate == codec::SAMPLE_RATE {
            return self.codec.encode(pcm);
        }
        self.codec.encode(&crate::zonos::speaker::resample_sinc(
            pcm,
            sample_rate as usize,
            codec::SAMPLE_RATE as usize,
        ))
    }

    /// The codes of a reference: an audio file (WAV; any format oxideav decodes with the `media`
    /// feature), its channels averaged, or a `.npy` file of codes.
    pub fn reference_codes(&self, path: &Path) -> Result<Vec<u32>> {
        let n_cb = self.info.n_codebooks;
        if path.extension().is_some_and(|e| e == "npy") {
            let (shape, v) = crate::zonos::read_npy(path)?;
            let codes: Vec<u32> = match shape[..] {
                // fish-speech's layout, one row per codebook.
                [c, frames] if c == n_cb => (0..frames * n_cb).map(|i| v[(i % n_cb) * frames + i / n_cb] as u32).collect(),
                [_, c] if c == n_cb => v.iter().map(|&x| x as u32).collect(),
                _ => bail!("{}: {shape:?} is not codes [{n_cb}, frames]", path.display()),
            };
            self.check_codes(&codes, &path.display().to_string())?;
            return Ok(codes);
        }
        if let Some((p, c)) = &*self.last_clip.lock().unwrap()
            && p == path
        {
            return Ok(c.clone());
        }
        #[cfg(feature = "media")]
        let (pcm, rate) = {
            let clip = crate::media::Media::new().decode_audio(path)?;
            let rate = clip.params.sample_rate.unwrap_or(0) as usize;
            if rate == 0 {
                bail!("{}: unknown sample rate", path.display());
            }
            (clip.to_mono(rate).with_context(|| path.display().to_string())?, rate as u32)
        };
        #[cfg(not(feature = "media"))]
        let (pcm, rate) = {
            let a = crate::tts::read_wav(&crate::fsx::read(path)?)
                .with_context(|| format!("{} (built without `media`: WAV only)", path.display()))?;
            (a.samples, a.sample_rate)
        };
        let codes = self.encode_audio(&pcm, rate)?;
        *self.last_clip.lock().unwrap() = Some((path.to_path_buf(), codes.clone()));
        Ok(codes)
    }

    /// Frames (`[frames, n_codebooks]`) to audio.
    pub fn decode(&self, codes: &[u32]) -> Result<Audio> {
        Ok(Audio {
            samples: if codes.is_empty() { Vec::new() } else { self.codec.decode(codes)? },
            sample_rate: codec::SAMPLE_RATE,
        })
    }

    /// Speaks `text`.
    pub fn speak(&self, text: &str, o: &DramaSpeak) -> Result<Speech> {
        self.speak_stream(text, o, &mut |_| Flow::Continue)
    }

    /// Speaks `text`, calling `on_chunk` with the audio of each chunk of the text as soon as it
    /// is generated and decoded (a text of up to about 300 bytes is one chunk).
    pub fn speak_stream(&self, text: &str, o: &DramaSpeak, on_chunk: &mut dyn FnMut(&AudioChunk) -> Flow) -> Result<Speech> {
        let t0 = Instant::now();
        self.cancel.store(false, Ordering::Relaxed);
        let n_cb = self.info.n_codebooks;
        let w = 1 + n_cb;
        validate_text(text, "text")?;
        let references = o
            .references
            .iter()
            .map(|r| Ok((r.text.clone(), self.reference_codes(&r.audio)?)))
            .collect::<Result<Vec<_>>>()?;
        let mut chunks = if o.single_chunk { Vec::new() } else { split_text(text) };
        if chunks.is_empty() {
            chunks.push(text.to_string());
        }
        let requested = match o.max_tokens {
            0 if o.single_chunk => DEFAULT_MAX_NEW_TOKENS,
            0 => CHUNK_MAX_NEW_TOKENS,
            n => n,
        };
        let capacity = MAX_PROMPT_ROWS + DEFAULT_MAX_NEW_TOKENS;
        // The first chunk stays in the history, with the latest one.
        let mut pinned: Option<(String, Vec<u32>)> = None;
        let mut latest: Option<(String, Vec<u32>)> = None;
        let mut stats = SpeakStats::default();
        let mut samples: Vec<f32> = Vec::new();
        let mut all_codes: Vec<u32> = Vec::new();
        let mut finish = StopReason::Stop;
        for (ci, chunk) in chunks.iter().enumerate() {
            // As much history as fits: both turns, the pinned one, none.
            let candidates: [Vec<(String, Vec<u32>)>; 3] = [
                pinned.iter().chain(&latest).cloned().collect(),
                pinned.iter().cloned().collect(),
                Vec::new(),
            ];
            let mut prompt = Err(anyhow!("no prompt"));
            for h in &candidates {
                prompt = self.prompt(chunk, o.speaker, &references, h);
                if prompt.is_ok() {
                    break;
                }
            }
            let prompt = prompt?;
            let rows = prompt.len() / w;
            let budget = requested.min(capacity.saturating_sub(rows)).max(1);
            let mut seeded = o.clone();
            seeded.seed = o.seed.wrapping_add(ci as u64);
            let g = self.generate(&prompt, &seeded, budget, false, &mut |_| Flow::Continue)?;
            stats.prompt_tokens += g.prompt_rows;
            stats.prefill_secs += g.prefill_secs;
            stats.generate_secs += g.generate_secs;
            let frames = g.rows.len() / w;
            stats.tokens += frames;
            stats.frames += frames;
            finish = g.finish;
            let codes: Vec<u32> = g.rows.chunks(w).flat_map(|r| r[1..].iter().copied()).collect();
            let t1 = Instant::now();
            let audio = self.decode(&codes)?;
            stats.snac_secs += t1.elapsed().as_secs_f64();
            stats.first_chunk_secs.get_or_insert_with(|| t0.elapsed().as_secs_f64());
            let start = samples.len();
            samples.extend_from_slice(&audio.samples);
            all_codes.extend_from_slice(&codes);
            let last = ci + 1 == chunks.len() || finish == StopReason::Cancelled;
            let flow = on_chunk(&AudioChunk {
                samples: &samples[start..],
                start,
                sample_rate: codec::SAMPLE_RATE,
                frames: stats.frames,
                last,
            });
            if finish == StopReason::Cancelled || matches!(flow, Flow::Stop) {
                finish = StopReason::Cancelled;
                break;
            }
            if frames > 0 {
                if pinned.is_none() {
                    pinned = Some((chunk.clone(), codes));
                } else {
                    latest = Some((chunk.clone(), codes));
                }
            }
        }
        let audio = Audio {
            samples,
            sample_rate: codec::SAMPLE_RATE,
        };
        stats.audio_secs = audio.duration_secs();
        stats.total_secs = t0.elapsed().as_secs_f64();
        Ok(Speech {
            audio,
            tokens: all_codes,
            frames: Vec::new(),
            finish,
            stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_tags_and_validation() {
        assert_eq!(speaker_tag("<|speaker:12|> hi"), Some(("12", 14)));
        assert_eq!(speaker_tag("<|speaker:|>"), None);
        assert_eq!(with_speaker_tag("hi", 0), "<|speaker:0|> hi");
        assert_eq!(with_speaker_tag(" \u{3000}<|speaker:3|>hi", 0), " \u{3000}<|speaker:3|>hi");
        assert!(validate_text("hello <|speaker:15|> there", "t").is_ok());
        assert!(validate_text("hello <|speaker:16|>", "t").is_err());
        assert!(validate_text("hello <|speaker:01|>", "t").is_err());
        assert!(validate_text("a <|im_end|>", "t").is_err());
        assert!(validate_text(" \u{a0}\n", "t").is_err());
    }

    #[test]
    fn long_text_is_cut_at_sentences() {
        let s = "This is a sentence of forty bytes or so. ".repeat(12);
        let chunks = split_text(&s);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.len() <= CHUNK_BYTES && c.ends_with('.')));
        assert_eq!(chunks.join(" "), s.trim());
        // Protected spans stay whole, decimals too.
        let p = protected_pieces("a [laughs] b <whisper>x. y</whisper> <|speaker:1|> 3.14");
        let spans: Vec<&str> = p.iter().filter(|x| x.1).map(|x| x.0).collect();
        assert_eq!(spans, ["[laughs]", "<whisper>x. y</whisper>", "<|speaker:1|>"]);
        assert_eq!(break_at_marks(vec![("pi is 3.14. ok", false)], 0, &['.']).len(), 2);
    }

    #[test]
    fn tagged_text_groups_speakers() {
        let t = "<|speaker:0|> A.<|speaker:1|> B.<|speaker:2|> C.<|speaker:3|> D.<|speaker:0|> E.";
        let chunks = split_text(t);
        assert_eq!(
            chunks,
            // (The reference keeps the space after a tag and adds its own.)
            [
                "<|speaker:0|>  A.\n<|speaker:1|>  B.\n<|speaker:2|>  C.",
                "<|speaker:3|>  D.\n<|speaker:0|>  E."
            ]
        );
    }

    #[test]
    fn sampler_keeps_the_top_rank_and_honours_top_k_and_top_p() {
        let mut rng = Rng::new(7);
        let l = [0.0f32, 3.0, 1.0, 2.9];
        // top_p below the first probability: the first rank alone.
        for _ in 0..32 {
            assert_eq!(sample(&l, 1.0, 30, 0.1, &mut rng), 1);
        }
        // top_k 2: ranks 1 and 3 only, both drawn.
        let draws: Vec<usize> = (0..200).map(|_| sample(&l, 1.0, 2, 1.0, &mut rng)).collect();
        assert!(draws.iter().all(|&d| d == 1 || d == 3));
        assert!(draws.contains(&1) && draws.contains(&3));
        // A tiny temperature is greedy.
        assert_eq!(sample(&l, 0.0, 30, 1.0, &mut rng), 1);
    }
}
