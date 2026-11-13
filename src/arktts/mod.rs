//! Text to speech with Audio8 TTS (Edge0, `model_type: arktts`): a pair of transformers that
//! writes ten codes per frame ([`lm`]) and the codec that renders them at 44.1 kHz and reads
//! reference clips ([`codec`]), from a checkpoint directory (`config.json`, `tokenizer.json`,
//! `model.safetensors`, `codec.pth`) such as `Edge0/Audio8-TTS-Realistic-Air`. The prompt, the
//! sampler and the stopping rule follow the Python code the repository ships
//! (`processing_arktts.py`, `modeling_arktts.py`, `modeling_arktts_codec.py`).
//!
//! The prompt is a chat in rows of 11 ids (a token and the ten codes of a frame, zeros under
//! text): a system turn (`convert the provided text to speech`, or with a reference voice
//! `... reference to the following:`, the reference's transcript behind `<|speaker:0|>` and the
//! frames of its audio), the text as the user turn, and `<|voice|>` opening the assistant's.
//! Every literal piece is tokenized on its own. The model then writes one frame per step: the
//! slow transformer the semantic code (or the end token), the fast one the nine other codes.
//!
//! Without a reference the model picks a voice, which changes with the seed. A voice is a clip
//! and its exact transcript ([`ArkttsSpeak::voice`], [`ArkttsSpeak::voice_text`]); clips in a
//! `voices/` folder beside the weights (`NAME.wav` and `NAME.txt`) are voices by name.
//!
//! ```no_run
//! use oxinfer::arktts::{Arktts, ArkttsOptions, ArkttsSpeak};
//! # fn main() -> anyhow::Result<()> {
//! let m = Arktts::load("Edge0/Audio8-TTS-Realistic-Air", &ArkttsOptions::default())?;
//! m.speak("Hello there, how are you today?", &ArkttsSpeak::default())?.audio.write_wav("hello.wav")?;
//! # Ok(())
//! # }
//! ```

pub mod codec;
pub mod lm;

use crate::clock::Instant;
use crate::engine::{CancelHandle, Flow, StopReason};
use crate::tokenizer::Tokenizer;
use crate::tts::{Audio, AudioChunk, SpeakStats, Speech};
use anyhow::{Context, Result, anyhow, bail};
use codec::Codec;
use lm::{Lm, LmConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The repository of the released model.
pub const REPO: &str = "Edge0/Audio8-TTS-Realistic-Air";

/// Frames between two pieces of streamed audio (about 1.5 s).
const STREAM_FRAMES: usize = 32;

/// How to load the model.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ArkttsOptions {
    /// CPU worker threads (process-wide; default: all cores).
    pub threads: Option<usize>,
    /// The folder of named voices (default: `voices/` in the model's directory).
    pub voices_dir: Option<PathBuf>,
    /// Where repo ids are downloaded.
    pub cache_dir: Option<PathBuf>,
}

/// Options of one utterance. The sampling defaults are the checkpoint's
/// (`generation_config.json`: temperature 0.7, top-p 0.9, top-k 50 for the released model).
/// Deserializes from JSON with every field optional.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ArkttsSpeak {
    /// The reference voice: a name from the voices folder ([`Arktts::voices`]), an audio clip
    /// (WAV; any format oxideav decodes with the `media` feature), or its codes as a `.npy`
    /// array `[10, frames]` (what the reference's processor takes as `reference_codes`). None or
    /// `none`: no reference, the model picks a voice.
    pub voice: Option<String>,
    /// The exact transcript of the reference clip (default: the `.txt` file beside it).
    pub voice_text: Option<String>,
    /// 0 = greedy (both transformers take the most likely code). None: the checkpoint's.
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Frames at most (21.5 per second of audio; None: 1024, as the model card's example), within
    /// what the context of 2048 rows leaves after the prompt.
    pub max_tokens: Option<usize>,
    pub seed: u64,
}

impl ArkttsSpeak {
    /// Options from a JSON object (unset fields keep their defaults).
    pub fn from_json(json: &str) -> Result<Self> {
        if json.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(json).context("speak options")
    }
}

/// A voice of the voices folder.
#[derive(Clone, Debug)]
pub struct Voice {
    pub name: String,
    /// The clip, or its codes as `.npy`.
    pub clip: PathBuf,
    /// The transcript's file.
    pub text: PathBuf,
}

/// What a loaded model is.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ArkttsInfo {
    pub dir: PathBuf,
    pub layers: usize,
    pub fast_layers: usize,
    pub hidden_size: usize,
    /// Bytes of weights in memory (the language model and the codec's decoder).
    pub weight_bytes: usize,
    pub threads: usize,
    pub sample_rate: u32,
    /// Frames per second.
    pub frame_rate: f64,
    pub context: usize,
    pub load_secs: f64,
    pub voices: Vec<String>,
}

/// The codes of a reference clip (`[frames, 10]`) and its transcript.
#[derive(Clone, Debug)]
pub struct Reference {
    pub text: String,
    pub codes: Vec<u32>,
    pub frames: usize,
}

/// The frames the model wrote.
#[derive(Clone, Debug)]
pub struct Generated {
    /// `[frames, 10]`.
    pub codes: Vec<u32>,
    pub frames: usize,
    pub finish: StopReason,
    /// The logits of every step when asked for: the semantic codes' and the end token's
    /// (`codebook_size + 1`), then those of the nine other codebooks (`codebook_size` each); a
    /// step that ends the speech has only the first part.
    pub logits: Vec<Vec<f32>>,
}

/// The codes of a clip (`[frames, 10]`) and its frame count.
pub type Codes = Arc<(Vec<u32>, usize)>;

/// The sampling settings of one utterance.
#[derive(Clone, Copy, Debug)]
struct Sampling {
    temperature: f32,
    top_k: usize,
    top_p: f32,
}

pub struct Arktts {
    lm: Lm,
    codec: Codec,
    tok: Tokenizer,
    voices: Vec<Voice>,
    info: ArkttsInfo,
    defaults: Sampling,
    cancel: Arc<AtomicBool>,
    /// The clips already encoded, by path.
    encoded: Mutex<HashMap<PathBuf, Codes>>,
}

/// Whether `model` is the directory of an Audio8 TTS checkpoint (`model_type: arktts`).
pub fn is_arktts(model: &Path) -> bool {
    model.is_dir()
        && crate::fsx::read(&model.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|c| c["model_type"] == "arktts")
}

/// The model for the server, from a model entry's options (`threads`, `voices`: the folder of
/// named voices).
#[cfg(feature = "server")]
pub(crate) fn load_for_server(model: &str, options: &[(&str, &str)], cache_dir: Option<PathBuf>) -> Result<Arktts> {
    let mut o = ArkttsOptions {
        cache_dir,
        ..Default::default()
    };
    for (k, v) in options {
        match *k {
            "threads" => o.threads = Some(v.parse().with_context(|| format!("threads: {v:?}"))?),
            "voices" => o.voices_dir = Some(PathBuf::from(v)),
            _ => bail!("unknown Audio8 TTS option {k:?} (threads, voices)"),
        }
    }
    Arktts::load(model, &o)
}

/// The text as the processor cleans it: trimmed, every run of whitespace one space.
pub fn clean_text(text: &str) -> String {
    text.split(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether `text` holds a `<|speaker:N|>` tag.
fn has_speaker_tag(text: &str) -> bool {
    text.match_indices("<|speaker:").any(|(i, m)| {
        let rest = &text[i + m.len()..];
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        digits > 0 && rest[digits..].starts_with("|>")
    })
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

/// One draw from `logits` as the reference filters them (`ArkttsLegacyTopKTopPLogitsProcessor`,
/// then the temperature): the candidates are sorted, and one is dropped when the probabilities
/// (at temperature 1) up to and including it exceed `top_p` or when it is past the first `top_k`;
/// the most likely one always stays. The rest is sampled at `temperature`.
fn sample(logits: &[f32], s: Sampling, rng: &mut Rng) -> usize {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let max = logits[order[0]];
    let total: f32 = order.iter().map(|&i| (logits[i] - max).exp()).sum();
    let mut kept = Vec::with_capacity(s.top_k.min(order.len()));
    let mut cum = 0f32;
    for (rank, &i) in order.iter().enumerate() {
        cum += (logits[i] - max).exp() / total;
        if rank > 0 && (cum > s.top_p || rank >= s.top_k) {
            break;
        }
        kept.push(i);
    }
    let t = s.temperature.max(1e-5);
    let p: Vec<f64> = kept.iter().map(|&i| (((logits[i] - max) / t) as f64).exp()).collect();
    let mut u = rng.uniform() * p.iter().sum::<f64>();
    for (&i, &pv) in kept.iter().zip(&p) {
        if u < pv {
            return i;
        }
        u -= pv;
    }
    *kept.last().expect("the most likely candidate is kept")
}

impl Arktts {
    /// Loads the model: a checkpoint directory, or a repo id ([`REPO`]; `config.json`,
    /// `generation_config.json`, `tokenizer.json`, `model.safetensors` and `codec.pth` are
    /// fetched, 2.6 GB).
    pub fn load(model: &str, opts: &ArkttsOptions) -> Result<Self> {
        let t0 = Instant::now();
        if let Some(n) = opts.threads {
            let mut o = crate::pool::options();
            o.threads = n;
            crate::pool::configure_with(o);
        }
        let p = Path::new(model);
        let dir = if p.is_dir() {
            p.to_path_buf()
        } else {
            #[cfg(feature = "hf")]
            {
                let names = [
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "model.safetensors",
                    "codec.pth",
                ];
                crate::hf::resolve_files(model, &names, opts.cache_dir.as_deref())?
            }
            #[cfg(not(feature = "hf"))]
            bail!("{model}: not a folder (built without the `hf` feature, so repo ids cannot be fetched)");
        };
        let cfg: LmConfig =
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json")).with_context(|| format!("{}: config.json", dir.display()))?)
                .with_context(|| format!("{}: config.json of an arktts model", dir.display()))?;
        let codec_path = dir.join(&cfg.codec_filename);
        // A repository fetched as a plain checkpoint (`org--name` in the cache) has no codec yet.
        #[cfg(feature = "hf")]
        if !codec_path.is_file()
            && let (Some(name), Some(root)) = (dir.file_name().and_then(|n| n.to_str()), dir.parent())
            && let Some((org, repo)) = name.split_once("--")
            && !cfg.codec_filename.contains(['/', '\\'])
        {
            crate::hf::resolve_files(&format!("{org}/{repo}"), &[cfg.codec_filename.as_str()], Some(root))?;
        }
        if !codec_path.is_file() {
            bail!("{}: the codec is missing ({} of {REPO})", codec_path.display(), cfg.codec_filename);
        }
        let files = crate::safetensors::ModelFiles::open(&dir, false)?;
        let lm = Lm::load(&files, cfg)?;
        drop(files);
        let c = &lm.cfg;
        let codec = Codec::load(&codec_path, c.num_codebooks, c.codec_sample_rate, c.codec_frame_size)?;
        if codec.semantic_size != c.codebook_size {
            bail!("{}: not the codec of this model", codec_path.display());
        }
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{}: tokenizer.json: {e}", dir.display()))?;
        let mut defaults = Sampling {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        };
        if let Ok(b) = crate::fsx::read(&dir.join("generation_config.json"))
            && let Ok(g) = serde_json::from_slice::<serde_json::Value>(&b)
        {
            if let Some(v) = g["temperature"].as_f64() {
                defaults.temperature = v as f32;
            }
            if let Some(v) = g["top_k"].as_u64() {
                defaults.top_k = v as usize;
            }
            if let Some(v) = g["top_p"].as_f64() {
                defaults.top_p = v as f32;
            }
        }
        let voices = scan_voices(opts.voices_dir.as_deref().unwrap_or(&dir.join("voices")));
        let info = ArkttsInfo {
            layers: c.n_layer,
            fast_layers: c.n_fast_layer,
            hidden_size: c.dim,
            weight_bytes: lm.weight_bytes + codec.weight_bytes,
            threads: crate::pool::threads(),
            sample_rate: codec.sample_rate,
            frame_rate: codec.sample_rate as f64 / codec.frame_size as f64,
            context: c.max_seq_len,
            load_secs: t0.elapsed().as_secs_f64(),
            voices: voices.iter().map(|v| v.name.clone()).collect(),
            dir,
        };
        Ok(Self {
            lm,
            codec,
            tok,
            voices,
            info,
            defaults,
            cancel: Arc::new(AtomicBool::new(false)),
            encoded: Mutex::new(HashMap::new()),
        })
    }

    pub fn info(&self) -> &ArkttsInfo {
        &self.info
    }

    pub fn voices(&self) -> &[Voice] {
        &self.voices
    }

    pub fn lm(&self) -> &Lm {
        &self.lm
    }

    pub fn codec(&self) -> &Codec {
        &self.codec
    }

    /// A handle that stops the running (or next) utterance from another thread; the audio
    /// generated so far is returned.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle::from_flag(self.cancel.clone())
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// The codes of a reference clip: mono audio at `sample_rate`, resampled to the codec's rate.
    pub fn encode_reference(&self, pcm: &[f32], sample_rate: u32) -> Result<(Vec<u32>, usize)> {
        if sample_rate == self.codec.sample_rate {
            return self.codec.encode(pcm);
        }
        let x = crate::audio::resample(pcm, sample_rate as usize, self.codec.sample_rate as usize);
        self.codec.encode(&x)
    }

    /// The codes of a clip file, or of a `.npy` array of codes (`[10, frames]`); each file is
    /// encoded once.
    pub fn reference_codes(&self, path: &Path) -> Result<Codes> {
        if let Some(c) = self.encoded.lock().unwrap().get(path) {
            return Ok(c.clone());
        }
        let n = self.lm.cfg.num_codebooks;
        let codes = if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("npy")) {
            let (shape, v) = crate::zonos::read_npy(path)?;
            if shape.len() != 2 || shape[0] != n || shape[1] == 0 {
                bail!("{}: {shape:?} is not reference codes [{n}, frames]", path.display());
            }
            if v.iter().any(|&x| x < 0.0 || x >= self.lm.cfg.codebook_size as f64) {
                bail!("{}: values outside the codebook", path.display());
            }
            let frames = shape[1];
            let mut codes = vec![0u32; frames * n];
            for j in 0..n {
                for f in 0..frames {
                    codes[f * n + j] = v[j * frames + f] as u32;
                }
            }
            (codes, frames)
        } else {
            let pcm = crate::tts::load_audio_file(path, self.codec.sample_rate)?;
            self.codec.encode(&pcm).with_context(|| path.display().to_string())?
        };
        let codes = Arc::new(codes);
        self.encoded.lock().unwrap().insert(path.to_path_buf(), codes.clone());
        Ok(codes)
    }

    /// The reference `o` names, if any.
    pub fn reference(&self, o: &ArkttsSpeak) -> Result<Option<Reference>> {
        let name = match o.voice.as_deref() {
            None | Some("") | Some("none") => return Ok(None),
            Some(v) => v,
        };
        let (clip, text_file) = match self.voices.iter().find(|v| v.name.eq_ignore_ascii_case(name)) {
            Some(v) => (v.clip.clone(), v.text.clone()),
            None => {
                let p = PathBuf::from(name);
                if !p.is_file() {
                    let names: Vec<&str> = self.voices.iter().map(|v| v.name.as_str()).collect();
                    bail!(
                        "voice {name:?}: not a file and not one of the voices in the model's voices/ folder ({})",
                        if names.is_empty() {
                            "which is empty".to_string()
                        } else {
                            names.join(", ")
                        }
                    );
                }
                let t = p.with_extension("txt");
                (p, t)
            }
        };
        let text = match &o.voice_text {
            Some(t) => t.clone(),
            None => std::fs::read_to_string(&text_file).with_context(|| {
                format!(
                    "the reference clip {} needs its transcript (--ref-text, or {})",
                    clip.display(),
                    text_file.display()
                )
            })?,
        };
        if clean_text(&text).is_empty() {
            bail!("the transcript of the reference clip {} is empty", clip.display());
        }
        let c = self.reference_codes(&clip)?;
        Ok(Some(Reference {
            text,
            codes: c.0.clone(),
            frames: c.1,
        }))
    }

    /// The token ids of the prompt's two halves, the reference's frames going between them
    /// (`ArkttsProcessor._prompt_segments`).
    pub fn prompt_ids(&self, text: &str, reference_text: Option<&str>) -> Result<(Vec<u32>, Vec<u32>)> {
        let target = clean_text(text);
        if target.is_empty() {
            bail!("the text is empty");
        }
        let parts = |parts: &[&str]| -> Result<Vec<u32>> {
            let mut ids = Vec::new();
            for p in parts {
                ids.extend(self.encode_text(p)?);
            }
            Ok(ids)
        };
        let Some(rt) = reference_text else {
            let full = parts(&[
                "<|im_start|>system\n",
                "convert the provided text to speech",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                &target,
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ])?;
            return Ok((full, Vec::new()));
        };
        let mut rt = clean_text(rt);
        if !has_speaker_tag(&rt) {
            rt = format!("<|speaker:0|>{rt}");
        }
        let prefix = parts(&[
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            &rt,
            "\n\nSpeech:\n",
        ])?;
        let suffix = parts(&[
            "<|im_end|>\n",
            "<|im_start|>user\n",
            &target,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ])?;
        Ok((prefix, suffix))
    }

    /// The prompt as rows of `1 + 10` ids.
    pub fn prompt(&self, text: &str, reference: Option<&Reference>) -> Result<Vec<u32>> {
        let c = &self.lm.cfg;
        let (n, w) = (c.num_codebooks, c.num_codebooks + 1);
        let (prefix, suffix) = self.prompt_ids(text, reference.map(|r| r.text.as_str()))?;
        let mut rows = Vec::with_capacity((prefix.len() + suffix.len() + reference.map_or(0, |r| r.frames)) * w);
        let text_rows = |rows: &mut Vec<u32>, ids: &[u32]| {
            for &id in ids {
                rows.push(id);
                rows.extend(std::iter::repeat_n(0, n));
            }
        };
        text_rows(&mut rows, &prefix);
        if let Some(r) = reference {
            if r.codes.len() != r.frames * n || r.frames == 0 {
                bail!("the reference has {} codes for {} frames", r.codes.len(), r.frames);
            }
            for f in r.codes.chunks(n) {
                if f[0] as usize >= c.codebook_size {
                    bail!("reference code {} outside the codebook", f[0]);
                }
                rows.push(c.semantic_begin_id + f[0]);
                rows.extend_from_slice(f);
            }
        }
        text_rows(&mut rows, &suffix);
        Ok(rows)
    }

    fn sampling(&self, o: &ArkttsSpeak) -> Sampling {
        Sampling {
            temperature: o.temperature.unwrap_or(self.defaults.temperature),
            top_k: o.top_k.unwrap_or(self.defaults.top_k).max(1),
            top_p: o.top_p.filter(|p| *p > 0.0).unwrap_or(self.defaults.top_p).min(1.0),
        }
    }

    /// Generates the frames after `prompt` (rows of 11 ids). `on_frame` sees the codes so far
    /// (`[frames, 10]`).
    pub fn generate(
        &self,
        prompt: &[u32],
        o: &ArkttsSpeak,
        keep_logits: bool,
        on_frame: &mut dyn FnMut(&[u32]) -> Flow,
    ) -> Result<Generated> {
        let c = &self.lm.cfg;
        let (n, cs) = (c.num_codebooks, c.codebook_size);
        let n_prompt = prompt.len() / (n + 1);
        if n_prompt >= c.max_seq_len {
            bail!(
                "the prompt of {n_prompt} rows fills the model context of {} (the reference clip and the text share it)",
                c.max_seq_len
            );
        }
        let budget = o.max_tokens.unwrap_or(1024).max(1).min(c.max_seq_len - n_prompt);
        let s = self.sampling(o);
        let greedy = s.temperature <= 1e-6;
        // A semantic code seen in the last frames is drawn again with these settings.
        let again = Sampling {
            temperature: c.ras_temperature,
            top_k: s.top_k,
            top_p: c.ras_top_p,
        };
        let mut rng = Rng::new(o.seed);
        let mut cache = self.lm.new_cache();
        let mut fast = self.lm.new_fast_cache();
        let (mut logits, mut hidden) = self.lm.slow(&mut cache, prompt, n_prompt)?;
        let mut codes: Vec<u32> = Vec::new();
        let mut log = Vec::new();
        // The reference's window starts after the first frame, which never enters it.
        let mut recent: Option<std::collections::VecDeque<u32>> = None;
        let mut finish = StopReason::Length;
        for step in 0..budget {
            let pick = if greedy {
                argmax(&logits)
            } else {
                let first = sample(&logits, s, &mut rng);
                if first < cs && recent.as_ref().is_some_and(|r| r.contains(&(first as u32))) {
                    sample(&logits, again, &mut rng)
                } else {
                    first
                }
            };
            if pick == cs {
                if keep_logits {
                    log.push(std::mem::take(&mut logits));
                }
                finish = StopReason::Stop;
                break;
            }
            let mut row = vec![c.semantic_begin_id + pick as u32, pick as u32];
            let mut step_logits = if keep_logits { std::mem::take(&mut logits) } else { Vec::new() };
            self.lm.fast_begin(&mut fast, &hidden);
            for _ in 1..n {
                let l = self.lm.fast_step(&mut fast, *row.last().unwrap())?;
                let code = if greedy { argmax(&l) } else { sample(&l, s, &mut rng) };
                row.push(code as u32);
                if keep_logits {
                    step_logits.extend_from_slice(&l);
                }
            }
            if keep_logits {
                log.push(step_logits);
            }
            codes.extend_from_slice(&row[1..]);
            match &mut recent {
                None => recent = Some(std::collections::VecDeque::new()),
                Some(r) => {
                    r.push_back(pick as u32);
                    if r.len() > c.ras_window_size {
                        r.pop_front();
                    }
                }
            }
            if self.cancel.swap(false, Ordering::Relaxed) || matches!(on_frame(&codes), Flow::Stop) {
                finish = StopReason::Cancelled;
                break;
            }
            if step + 1 == budget {
                break;
            }
            (logits, hidden) = self.lm.slow(&mut cache, &row, 1)?;
        }
        Ok(Generated {
            frames: codes.len() / n,
            codes,
            finish,
            logits: log,
        })
    }

    /// Frames (`[frames, 10]`) to audio.
    pub fn decode(&self, codes: &[u32], frames: usize) -> Result<Audio> {
        Ok(Audio {
            samples: self.codec.decode(codes, frames)?,
            sample_rate: self.codec.sample_rate,
        })
    }

    /// Speaks `text`.
    pub fn speak(&self, text: &str, o: &ArkttsSpeak) -> Result<Speech> {
        self.speak_with(text, o, None)
    }

    /// Speaks `text`, calling `on_chunk` with the audio as it is generated, about 1.5 s at a time
    /// (the codec is causal: the pieces are the samples a decoding of the whole utterance gives).
    pub fn speak_stream(&self, text: &str, o: &ArkttsSpeak, on_chunk: &mut dyn FnMut(&AudioChunk) -> Flow) -> Result<Speech> {
        self.speak_with(text, o, Some(on_chunk))
    }

    fn speak_with(&self, text: &str, o: &ArkttsSpeak, mut on_chunk: Option<&mut dyn FnMut(&AudioChunk) -> Flow>) -> Result<Speech> {
        let t0 = Instant::now();
        self.cancel.store(false, Ordering::Relaxed);
        let reference = self.reference(o)?;
        let prompt = self.prompt(text, reference.as_ref())?;
        let n = self.lm.cfg.num_codebooks;
        let (sr, fs) = (self.codec.sample_rate, self.codec.frame_size);
        let mut samples: Vec<f32> = Vec::new();
        let mut first = None;
        let mut first_chunk = None;
        let mut codec_secs = 0f64;
        let mut err = None;
        let g = self.generate(&prompt, o, false, &mut |codes| {
            first.get_or_insert_with(|| t0.elapsed().as_secs_f64());
            let frames = codes.len() / n;
            let Some(on) = on_chunk.as_mut() else { return Flow::Continue };
            if !frames.is_multiple_of(STREAM_FRAMES) {
                return Flow::Continue;
            }
            let t1 = Instant::now();
            let start = samples.len();
            match self.codec.decode_tail(codes, frames, start / fs) {
                Ok(s) => samples.extend_from_slice(&s),
                Err(e) => {
                    err = Some(e);
                    return Flow::Stop;
                }
            }
            codec_secs += t1.elapsed().as_secs_f64();
            first_chunk.get_or_insert_with(|| t0.elapsed().as_secs_f64());
            on(&AudioChunk {
                samples: &samples[start..],
                start,
                sample_rate: sr,
                frames,
                last: false,
            })
        })?;
        if let Some(e) = err {
            return Err(e);
        }
        let t1 = Instant::now();
        let start = samples.len();
        if start < g.frames * fs {
            samples.extend_from_slice(&self.codec.decode_tail(&g.codes, g.frames, start / fs)?);
        }
        codec_secs += t1.elapsed().as_secs_f64();
        if let Some(on) = on_chunk.as_mut() {
            first_chunk.get_or_insert_with(|| t0.elapsed().as_secs_f64());
            on(&AudioChunk {
                samples: &samples[start..],
                start,
                sample_rate: sr,
                frames: g.frames,
                last: true,
            });
        }
        let audio = Audio { samples, sample_rate: sr };
        let prefill_secs = first.unwrap_or(0.0);
        let total = t0.elapsed().as_secs_f64();
        let stats = SpeakStats {
            prompt_tokens: prompt.len() / (n + 1),
            tokens: g.frames + usize::from(g.finish == StopReason::Stop),
            frames: g.frames,
            dropped_tokens: 0,
            loops: 0,
            audio_secs: audio.duration_secs(),
            prefill_secs,
            generate_secs: (total - prefill_secs - codec_secs).max(0.0),
            snac_secs: codec_secs,
            total_secs: total,
            first_chunk_secs: first_chunk,
        };
        Ok(Speech {
            audio,
            tokens: g.codes,
            frames: Vec::new(),
            finish: g.finish,
            stats,
        })
    }
}

/// The voices of a folder: every clip (or `.npy` codes) with a `.txt` transcript of the same
/// name beside it.
fn scan_voices(dir: &Path) -> Vec<Voice> {
    let mut out: Vec<Voice> = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_file() || p.extension().is_none_or(|x| x == "txt") {
            continue;
        }
        let text = p.with_extension("txt");
        let Some(name) = p.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if text.is_file() && !out.iter().any(|v| v.name == name) {
            out.push(Voice {
                name: name.to_string(),
                clip: p.clone(),
                text,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_cleaned_as_python_splits_it() {
        assert_eq!(clean_text("  Hello,\n\tworld \u{a0} again  "), "Hello, world again");
        assert_eq!(clean_text("\n"), "");
    }

    #[test]
    fn speaker_tags_are_recognized() {
        assert!(has_speaker_tag("<|speaker:0|>Hi"));
        assert!(has_speaker_tag("a <|speaker:12|> b"));
        assert!(!has_speaker_tag("<|speaker:|>Hi"));
        assert!(!has_speaker_tag("<|speaker:x|>Hi"));
    }

    #[test]
    fn sampling_keeps_the_candidates_the_reference_keeps() {
        // Probabilities 0.6, 0.3, 0.1: top_p 0.5 keeps only the first (it always stays), 0.95
        // the first two (the third brings the sum past it), and top_k 1 the first.
        let l = [0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()];
        let draws = |top_k, top_p| {
            let s = Sampling {
                temperature: 1.0,
                top_k,
                top_p,
            };
            let mut rng = Rng::new(7);
            let mut seen = [0usize; 3];
            for _ in 0..2000 {
                seen[sample(&l, s, &mut rng)] += 1;
            }
            seen
        };
        let a = draws(50, 0.5);
        assert_eq!((a[1], a[2]), (0, 0));
        let b = draws(50, 0.95);
        assert!(b[0] > 1200 && b[0] < 1460 && b[1] > 540 && b[2] == 0, "{b:?}");
        let c = draws(1, 1.0);
        assert_eq!(c[0], 2000);
        // Greedy at a tiny temperature.
        let s = Sampling {
            temperature: 1e-5,
            top_k: 50,
            top_p: 1.0,
        };
        assert_eq!(sample(&l, s, &mut Rng::new(1)), 0);
    }
}
