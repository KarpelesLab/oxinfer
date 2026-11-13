//! Text to speech with ZONOS3-Mini (Zyphra): a language model that writes ten codes per 15 Hz
//! frame ([`lm`]) and the ZAC 1 codec decoder that renders them at 44.1 kHz ([`codec`]), from
//! the GGUF files of `Zyphra/ZONOS3-Mini` (`gguf/backbone-ptq.gguf`, `gguf/zac-keep2.gguf` or
//! `gguf/zac.gguf`). The prompt, the sampler and the end-of-speech rule follow Zyphra's reference
//! (`model/src/zonos3/tts/{plan,prompt,sampler}.py`, `reference/synth.py` of Zyphra/ZONOS3-Mini).
//!
//! The prompt is a table of rows of 11 ids (ten codebooks and a text id), in this order:
//! - the speaker: either one row replaced by the projection of a 192-d speaker embedding, or the
//!   codes of a reference clip (the "reference prefix": codebook `j` delayed by `j` rows, the tail
//!   flushed with padding);
//! - three condition rows (loudness, bandwidth, speaking rate), each a bucket or "unknown";
//! - the text, one SuperBPE token per row between a start and an end id;
//! - three frames of the codec's silence, already in the delay pattern.
//!
//! The model then writes one row per step, codebook `j` of frame `f` in row `f + j`, until a
//! codebook holds the end token (EOA); ten more rows complete the last frames.
//!
//! Voices: the repository ships 27 default voices as a speaker embedding and reference codes
//! each ([`Voice`]); any `.npy` file of either kind works too, and so does an audio clip, whose
//! speaker embedding the speaker encoder computes ([`speaker`], `gguf/spk-encoder.gguf`). Turning
//! a clip into reference codes (the codec's encoder) is not implemented (see docs/zonos.md).
//!
//! ```no_run
//! use oxinfer::zonos::{Zonos, ZonosOptions, ZonosSpeak};
//! # fn main() -> anyhow::Result<()> {
//! let z = Zonos::load("Zyphra/ZONOS3-Mini", &ZonosOptions::default())?;
//! let mut o = ZonosSpeak::default();
//! o.voice = Some("Imogen".into());
//! z.speak("Hello there, how are you today?", &o)?.audio.write_wav("hello.wav")?;
//! # Ok(())
//! # }
//! ```

pub mod codec;
pub(crate) mod file;
pub mod lm;
pub mod speaker;

use crate::clock::Instant;
use crate::engine::{CancelHandle, Flow, StopReason};
use crate::tokenizer::Tokenizer;
use crate::tts::{Audio, AudioChunk, SpeakStats, Speech};
use anyhow::{Context, Result, anyhow, bail};
use codec::Decoder;
use file::Gg;
use lm::{Lm, LmState};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The repository of the released model.
pub const REPO: &str = "Zyphra/ZONOS3-Mini";

/// The default voices of the repository (`gguf/voices/<language>/<name>`).
#[cfg(feature = "hf")]
const VOICES: [&str; 27] = [
    "English/Cillian_en-Irish_M_1",
    "English/Imogen_en-British_F_1",
    "English/Jett_en-energetic_M_1",
    "English/Morgana_en-witch_F_1",
    "English/Rupert_en-British_M_1",
    "English/Sakura_en-anime_F_1",
    "English/Savannah_en-American_F_1",
    "English/Thaddeus_en-announcervoice_M_1",
    "English/Willow_en-asrm_F_1",
    "English/Wyatt_en-American_M_1",
    "French/Gaspard_fr_M_1",
    "German/Liesel_de_F_1",
    "German/Wolfgang_de_M_1",
    "Hindi/Ananya_hi_F_1",
    "Hindi/Arjun_hi_M_1",
    "Italian/Ginevra_it_F_1",
    "Italian/Lorenzo_it_M_1",
    "Korean/Minjun_ko_M_1",
    "Korean/Seoyeon_ko_F_1",
    "Mandarin Chinese/Haoran_zh-energetic_M_1",
    "Mandarin Chinese/Meilin_zh-energetic_F_1",
    "Portuguese/Leonor_pt_F_1",
    "Portuguese/Thiago_pt_M_1",
    "Spanish/Rodrigo_es_M_1",
    "Spanish/Ximena_es_F_1",
    "Telugu/Chaitanya_te_M_1",
    "Telugu/Sravya_te_F_1",
];

/// The voice used when none is named.
pub const DEFAULT_VOICE: &str = "Savannah";

/// How to load the model.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ZonosOptions {
    /// CPU worker threads (process-wide; default: all cores).
    pub threads: Option<usize>,
    /// The codec decoder: `keep2` (the default, `zac-keep2.gguf`: two transformer layers per
    /// stage), `full` (`zac.gguf`, about three times slower) or a GGUF file.
    pub decoder: Option<String>,
    /// Compute the quantized layers (Q8_0 / Q4_0) in f32 on the dequantized weights, as the
    /// PyTorch reference does on the same weights (the default: its greedy output is then the
    /// reference's row for row). Off: the int8 kernels, with the activations rounded to 8 bits
    /// per block of 32 as ggml does, which no longer follow the reference row for row.
    pub exact: bool,
    /// Where repo ids are downloaded.
    pub cache_dir: Option<PathBuf>,
}

impl Default for ZonosOptions {
    fn default() -> Self {
        Self {
            threads: None,
            decoder: None,
            exact: true,
            cache_dir: None,
        }
    }
}

/// Options of one utterance; the defaults are the reference's (`registries/generation.json`).
/// Deserializes from JSON with every field optional.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ZonosSpeak {
    /// A default voice by name (`Imogen`; see [`Zonos::voices`]), or a `.npy` file: a speaker
    /// embedding (float32 `[192]`) or reference codes (int32 `[frames, 10]`). None: the default
    /// voice ([`DEFAULT_VOICE`]) when the model has it; `none`: no speaker conditioning.
    pub voice: Option<String>,
    /// `embedding` (the default: one prompt row) or `reference` (the voice's codes before the
    /// text: closer to the clip, a longer prompt).
    pub speaker_mode: SpeakerMode,
    /// 0 = greedy.
    pub temperature: f32,
    /// 0 = the whole codebook.
    pub top_k: usize,
    /// 0 or 1 = off.
    pub top_p: f32,
    pub min_p: f32,
    /// Applied to the codes of the last `repetition_window` generated rows, in the first
    /// `repetition_codebooks` codebooks (1 = off).
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    /// Negative: every codebook.
    pub repetition_codebooks: i32,
    /// Generated rows at most (15 per second of audio).
    pub max_tokens: usize,
    pub seed: u64,
    /// Frames of codec silence opening the audio (0 to 30; they are part of the output).
    pub silence_prefill_frames: usize,
    /// Classifier-free guidance on the speaker (1 = off): a second pass without the speaker,
    /// `uncond + scale * (cond - uncond)`.
    pub speaker_cfg_scale: f32,
    /// The condition rows by raw value (`loudness` in LUFS, `fmax` in Hz, `speaking_rate` in
    /// UTF-8 bytes per second); unset rows are "unknown".
    pub loudness: Option<f64>,
    pub fmax: Option<f64>,
    pub speaking_rate: Option<f64>,
    /// The condition rows by bucket id (in the model's order), overriding the raw values.
    pub condition_buckets: Option<Vec<u32>>,
    /// Output gain in dB below the decoder's level (None: the decoder's own, 3 dB).
    pub headroom_db: Option<f32>,
    pub fade_out_ms: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeakerMode {
    #[default]
    Embedding,
    Reference,
}

impl Default for ZonosSpeak {
    fn default() -> Self {
        Self {
            voice: None,
            speaker_mode: SpeakerMode::Embedding,
            temperature: 0.75,
            top_k: 80,
            top_p: 0.69,
            min_p: 0.0,
            repetition_penalty: 2.4,
            repetition_window: 96,
            repetition_codebooks: 6,
            max_tokens: 2000,
            seed: 42,
            silence_prefill_frames: 3,
            speaker_cfg_scale: 1.0,
            loudness: None,
            fmax: None,
            speaking_rate: None,
            condition_buckets: None,
            headroom_db: None,
            fade_out_ms: 0.0,
        }
    }
}

impl ZonosSpeak {
    /// Options from a JSON object (unset fields keep their defaults).
    pub fn from_json(json: &str) -> Result<Self> {
        if json.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(json).context("speak options")
    }
}

/// A voice: a speaker embedding and / or the codec codes of its reference clip.
#[derive(Clone, Debug, Default)]
pub struct Voice {
    /// `Imogen`.
    pub name: String,
    /// `Imogen_en-British_F_1`.
    pub label: String,
    /// `English`.
    pub language: String,
    pub embedding: Option<PathBuf>,
    pub codes: Option<PathBuf>,
}

/// What a loaded model is.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ZonosInfo {
    pub backbone: PathBuf,
    pub decoder: PathBuf,
    /// `keep2_plain` or `full_mr`.
    pub decoder_variant: String,
    pub layers: usize,
    pub hidden_size: usize,
    pub weight_bytes: usize,
    pub threads: usize,
    pub sample_rate: u32,
    /// Frames (rows) per second.
    pub frame_rate: f64,
    pub load_secs: f64,
    pub voices: Vec<String>,
}

/// The prompt of one utterance.
#[derive(Clone, Debug)]
pub struct Prompt {
    /// `[rows, n_codebooks + 1]`.
    pub rows: Vec<u32>,
    /// The speaker row's embedding (row 0).
    pub speaker: Option<Vec<f32>>,
    /// Rows of the reference prefix.
    pub prefix_rows: usize,
    /// The condition rows (absolute row, bucket).
    pub cond: Vec<(usize, u32)>,
    /// The silence rows ending the prompt.
    pub prefill_rows: usize,
    /// The text tokens, start and end ids included.
    pub text_tokens: usize,
}

/// The rows the model wrote.
#[derive(Clone, Debug)]
pub struct Generated {
    /// `[rows, n_codebooks]`, in the delay pattern, the finishing rows included.
    pub rows: Vec<u32>,
    /// The frame (in generated rows) where the speech ends; None when the budget ran out.
    pub eos_frame: Option<usize>,
    pub finish: StopReason,
    /// The logits of every step (`[n_codebooks, audio_vocab]` each), when asked for.
    pub logits: Vec<Vec<f32>>,
}

pub struct Zonos {
    lm: Lm,
    decoder: Decoder,
    tok: Tokenizer,
    voices: Vec<Voice>,
    info: ZonosInfo,
    cancel: Arc<AtomicBool>,
    /// `spk-encoder.gguf` beside the backbone, loaded at its first use.
    speaker_path: Option<PathBuf>,
    speaker: std::sync::OnceLock<speaker::SpeakerEncoder>,
    /// The last clip embedded, so that a batch in one cloned voice encodes it once.
    last_clip: std::sync::Mutex<Option<(PathBuf, Vec<f32>)>>,
}

/// Whether `model` (a path or a repo id) is a ZONOS3 model: a GGUF backbone, a folder holding
/// one (`backbone-*.gguf`, in it or in its `gguf/`), or the repository's id.
pub fn is_zonos(model: &Path) -> bool {
    if crate::gguf::is_gguf(model) {
        return crate::gguf::header(model).is_ok_and(|h| h.arch() == "zonos3");
    }
    if model.is_dir() {
        return find_backbone(model).is_some();
    }
    let s = model.to_string_lossy();
    s.split('/').count() == 2 && s.to_ascii_lowercase().contains("zonos3")
}

fn find_backbone(dir: &Path) -> Option<PathBuf> {
    for d in [dir.to_path_buf(), dir.join("gguf")] {
        for name in ["backbone-ptq.gguf", "backbone-f16.gguf", "backbone-q8_0.gguf", "backbone-qat.gguf"] {
            let p = d.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// One plain space for every whitespace run that holds a line break, a tab or another non-space
/// whitespace; runs of plain spaces are kept (`zyphra_textnorm.collapse_line_breaks`).
pub fn collapse_line_breaks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = String::new();
    let flush = |run: &mut String, out: &mut String| {
        if run.chars().any(|c| c != ' ') {
            out.push(' ');
        } else {
            out.push_str(run);
        }
        run.clear();
    };
    for c in text.chars() {
        if c.is_whitespace() {
            run.push(c);
        } else {
            flush(&mut run, &mut out);
            out.push(c);
        }
    }
    flush(&mut run, &mut out);
    out
}

/// A `.npy` array (version 1 to 3, little endian, C order): its shape and values as f64.
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let bad = || anyhow!("{}: not a supported .npy file", path.display());
    if b.len() < 12 || &b[..6] != b"\x93NUMPY" {
        return Err(bad());
    }
    let (hlen, start) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        _ => return Err(bad()),
    };
    let header = std::str::from_utf8(b.get(start..start + hlen).ok_or_else(bad)?).map_err(|_| bad())?;
    let field = |k: &str| header.split_once(&format!("'{k}':")).map(|(_, r)| r.trim_start());
    let descr = field("descr")
        .and_then(|r| r.strip_prefix('\''))
        .and_then(|r| r.split('\'').next())
        .ok_or_else(bad)?;
    if field("fortran_order").is_none_or(|r| !r.starts_with("False")) {
        return Err(bad());
    }
    let shape: Vec<usize> = field("shape")
        .and_then(|r| r.strip_prefix('('))
        .and_then(|r| r.split(')').next())
        .ok_or_else(bad)?
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>().map_err(|_| bad()))
        .collect::<Result<_>>()?;
    let n: usize = shape.iter().product();
    let data = &b[start + hlen..];
    let (size, conv): (usize, fn(&[u8]) -> f64) = match descr {
        "<f4" => (4, |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        "<f8" => (8, |c| f64::from_le_bytes(c.try_into().unwrap())),
        "<i4" => (4, |c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        "<i8" => (8, |c| i64::from_le_bytes(c.try_into().unwrap()) as f64),
        "<u4" => (4, |c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        "<i2" => (2, |c| i16::from_le_bytes([c[0], c[1]]) as f64),
        _ => bail!("{}: .npy element type {descr} is not supported", path.display()),
    };
    if data.len() < n * size {
        return Err(bad());
    }
    Ok((shape, data[..n * size].chunks(size).map(conv).collect()))
}

/// A voice as the prompt takes it: a speaker embedding or reference codes (`[frames, n_codebooks]`).
type VoiceInputs = (Option<Vec<f32>>, Option<Vec<u32>>);

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

/// Samples one code of a codebook from its logits, as `zonos3.tts.sampler.sample_tts`: padding
/// and every special id but the end token are excluded; the repetition penalty divides positive
/// logits of the recent codes and multiplies negative ones; then temperature, top-k (ties with
/// the k-th value stay), top-p over the sorted probabilities and min-p.
fn sample_code(logits: &[f32], recent: &[u32], o: &ZonosSpeak, codebook_size: usize, eoa: u32, rng: &mut Rng) -> u32 {
    let mut l: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i >= codebook_size && i as u32 != eoa {
                f32::NEG_INFINITY
            } else {
                v
            }
        })
        .collect();
    let pen = o.repetition_penalty.max(1.0);
    if pen > 1.0 {
        for &r in recent {
            // Each id once, however often it was seen.
            if (r as usize) < codebook_size && l[r as usize] == logits[r as usize] {
                let v = &mut l[r as usize];
                *v = if *v > 0.0 { *v / pen } else { *v * pen };
            }
        }
    }
    let argmax = |l: &[f32]| l.iter().enumerate().fold(0, |b, (i, &v)| if v > l[b] { i } else { b }) as u32;
    if o.temperature <= 1e-6 {
        return argmax(&l);
    }
    l.iter_mut().for_each(|v| *v /= o.temperature);
    let k = if o.top_k >= 1 { o.top_k } else { codebook_size };
    if k < l.len() {
        let mut sorted = l.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let kth = sorted[k - 1];
        l.iter_mut().filter(|v| **v < kth).for_each(|v| *v = f32::NEG_INFINITY);
    }
    let greedy = argmax(&l);
    lm::softmax(&mut l);
    let mut p = l;
    let renorm = |p: &mut [f32]| {
        let s = p.iter().sum::<f32>().max(1e-8);
        p.iter_mut().for_each(|v| *v /= s);
    };
    let top_p = o.top_p.clamp(0.0, 1.0);
    if top_p > 0.0 && top_p < 1.0 {
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        let mut cum = 0f32;
        for &i in &order {
            // Dropped once the probabilities before it exceed top_p.
            if cum > top_p {
                p[i] = 0.0;
            }
            cum += p[i];
        }
        renorm(&mut p);
    }
    if o.min_p > 0.0 {
        let top = p.iter().cloned().fold(0f32, f32::max);
        p.iter_mut().filter(|v| **v < o.min_p * top).for_each(|v| *v = 0.0);
        renorm(&mut p);
    }
    let total: f64 = p.iter().map(|&v| v as f64).sum();
    if total <= 0.0 {
        return greedy;
    }
    let mut u = rng.uniform() * total;
    let mut last = greedy;
    for (i, &v) in p.iter().enumerate() {
        if v > 0.0 {
            last = i as u32;
            if u < v as f64 {
                return last;
            }
            u -= v as f64;
        }
    }
    last
}

impl Zonos {
    /// Loads the model: a repo id ([`REPO`]: the backbone, the decoder and the default voices are
    /// fetched, about 1.1 GB), a folder holding the `gguf/` files, or the backbone GGUF itself
    /// (the decoder and `voices/` beside it).
    pub fn load(model: &str, opts: &ZonosOptions) -> Result<Self> {
        let t0 = Instant::now();
        if let Some(n) = opts.threads {
            let mut o = crate::pool::options();
            o.threads = n;
            crate::pool::configure_with(o);
        }
        let decoder_file = match opts.decoder.as_deref() {
            None | Some("keep2") | Some("keep2_plain") => "zac-keep2.gguf".to_string(),
            Some("full") | Some("full_mr") => "zac.gguf".to_string(),
            Some(f) => f.to_string(),
        };
        let p = Path::new(model);
        let backbone = if crate::gguf::is_gguf(p) {
            p.to_path_buf()
        } else if p.is_dir() {
            find_backbone(p).with_context(|| format!("{model}: no backbone-ptq.gguf in the folder or its gguf/"))?
        } else {
            #[cfg(feature = "hf")]
            {
                let mut names = vec!["gguf/backbone-ptq.gguf".to_string(), "gguf/spk-encoder.gguf".to_string()];
                if !Path::new(&decoder_file).is_file() {
                    names.push(format!("gguf/{decoder_file}"));
                }
                for v in VOICES {
                    names.push(format!("gguf/voices/{v}.embedding.npy"));
                    names.push(format!("gguf/voices/{v}.codes.npy"));
                }
                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                let dir = crate::hf::resolve_files(model, &refs, opts.cache_dir.as_deref())?;
                dir.join("gguf/backbone-ptq.gguf")
            }
            #[cfg(not(feature = "hf"))]
            bail!("{model}: not a folder or a GGUF file (built without the `hf` feature, so repo ids cannot be fetched)");
        };
        let dir = backbone.parent().unwrap_or(Path::new(".")).to_path_buf();
        let decoder_path = if Path::new(&decoder_file).is_file() {
            PathBuf::from(&decoder_file)
        } else {
            dir.join(&decoder_file)
        };
        if !decoder_path.is_file() {
            bail!(
                "{}: the codec decoder is missing (gguf/{decoder_file} of {REPO})",
                decoder_path.display()
            );
        }
        let lm = Lm::load(&Gg::open(&backbone, "zonos3")?, opts.exact)?;
        let decoder = Decoder::load(&Gg::open(&decoder_path, "mossdual_istft")?)?;
        if decoder.n_codebooks != lm.cfg.n_codebooks || decoder.codebook_size != lm.cfg.codebook_size {
            bail!("{}: not the codec of {}", decoder_path.display(), backbone.display());
        }
        let tok = Tokenizer::from_bytes(lm.tokenizer_json.as_bytes()).context("the tokenizer of the backbone")?;
        let voices = scan_voices(&dir.join("voices"));
        let info = ZonosInfo {
            backbone,
            decoder: decoder_path,
            decoder_variant: decoder.variant.clone(),
            layers: lm.cfg.n_layers,
            hidden_size: lm.cfg.dim,
            weight_bytes: lm.weight_bytes + decoder.weight_bytes,
            threads: crate::pool::threads(),
            sample_rate: decoder.sample_rate,
            frame_rate: decoder.sample_rate as f64 / decoder.hop as f64,
            load_secs: t0.elapsed().as_secs_f64(),
            voices: voices.iter().map(|v| v.name.clone()).collect(),
        };
        Ok(Self {
            lm,
            decoder,
            tok,
            voices,
            info,
            cancel: Arc::new(AtomicBool::new(false)),
            speaker_path: Some(dir.join("spk-encoder.gguf")).filter(|p| p.is_file()),
            speaker: std::sync::OnceLock::new(),
            last_clip: std::sync::Mutex::new(None),
        })
    }

    /// The speaker embedding (192 values) of mono audio: at most its first 60 s are used. Needs
    /// `spk-encoder.gguf` beside the backbone.
    pub fn speaker_embedding(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let path = self
            .speaker_path
            .as_ref()
            .with_context(|| format!("cloning a voice from audio needs gguf/spk-encoder.gguf of {REPO} beside the backbone"))?;
        if self.speaker.get().is_none() {
            let enc = speaker::SpeakerEncoder::load(&Gg::open(path, "mect_spk")?)?;
            if enc.dim != self.lm.cfg.speaker_dim {
                bail!(
                    "{}: embeddings of {} values, the model takes {}",
                    path.display(),
                    enc.dim,
                    self.lm.cfg.speaker_dim
                );
            }
            let _ = self.speaker.set(enc);
        }
        self.speaker.get().expect("just set").embed(pcm, sample_rate)
    }

    /// [`Zonos::speaker_embedding`] of an audio file (WAV; any format oxideav decodes with the
    /// `media` feature), its channels averaged.
    pub fn speaker_embedding_file(&self, path: &Path) -> Result<Vec<f32>> {
        if let Some((p, e)) = &*self.last_clip.lock().unwrap()
            && p == path
        {
            return Ok(e.clone());
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
        let e = self.speaker_embedding(&pcm, rate)?;
        *self.last_clip.lock().unwrap() = Some((path.to_path_buf(), e.clone()));
        Ok(e)
    }

    pub fn info(&self) -> &ZonosInfo {
        &self.info
    }

    pub fn voices(&self) -> &[Voice] {
        &self.voices
    }

    pub fn lm(&self) -> &Lm {
        &self.lm
    }

    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// A handle that stops the running (or next) utterance from another thread; the audio
    /// generated so far is returned.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle::from_flag(self.cancel.clone())
    }

    /// The ids of `text` as the model reads them: start (2), the SuperBPE tokens shifted by 4,
    /// end (3). Line breaks become spaces; the text is otherwise taken as written.
    pub fn text_ids(&self, text: &str) -> Result<Vec<u32>> {
        let text = collapse_line_breaks(text);
        if let Some(c) = text.chars().find(|c| ('\u{e000}'..='\u{e00a}').contains(c)) {
            bail!(
                "the text holds the marker U+{:04X}: this checkpoint has no phoneme, start or pause tokens",
                c as u32
            );
        }
        let enc = self.tok.encode(text.as_str(), false).map_err(|e| anyhow!("tokenize: {e}"))?;
        let mut ids = vec![2u32];
        ids.extend(enc.get_ids().iter().map(|i| i + 4));
        ids.push(3);
        Ok(ids)
    }

    /// The speaker embedding and / or reference codes `o.voice` names.
    fn resolve_voice(&self, o: &ZonosSpeak) -> Result<VoiceInputs> {
        let c = &self.lm.cfg;
        let name = match o.voice.as_deref() {
            Some("") | Some("none") => return Ok((None, None)),
            Some(v) => v,
            None if self.voices.iter().any(|v| v.name == DEFAULT_VOICE) => DEFAULT_VOICE,
            None => match self.voices.first() {
                Some(v) => v.name.as_str(),
                None => return Ok((None, None)),
            },
        };
        let embedding = |p: &Path| -> Result<Vec<f32>> {
            let (shape, v) = read_npy(p)?;
            let n: usize = shape.iter().product();
            if n != c.speaker_dim {
                bail!("{}: {shape:?} is not a speaker embedding of {} values", p.display(), c.speaker_dim);
            }
            Ok(v.iter().map(|&x| x as f32).collect())
        };
        let codes = |p: &Path| -> Result<Vec<u32>> {
            let (shape, v) = read_npy(p)?;
            if shape.len() != 2 || shape[1] != c.n_codebooks || shape[0] == 0 {
                bail!("{}: {shape:?} is not reference codes [frames, {}]", p.display(), c.n_codebooks);
            }
            if v.iter().any(|&x| x < 0.0 || x >= c.codebook_size as f64) {
                bail!("{}: values outside the codebook", p.display());
            }
            Ok(v.iter().map(|&x| x as u32).collect())
        };
        let want = o.speaker_mode;
        if let Some(v) = self
            .voices
            .iter()
            .find(|v| v.name.eq_ignore_ascii_case(name) || v.label.eq_ignore_ascii_case(name))
        {
            return match (want, &v.embedding, &v.codes) {
                (SpeakerMode::Embedding, Some(e), _) => Ok((Some(embedding(e)?), None)),
                (SpeakerMode::Reference, _, Some(k)) => Ok((None, Some(codes(k)?))),
                (SpeakerMode::Embedding, None, _) => bail!("voice {name}: no speaker embedding (try the reference mode)"),
                (SpeakerMode::Reference, _, None) => bail!("voice {name}: no reference codes (try the embedding mode)"),
            };
        }
        let p = Path::new(name);
        if p.extension().is_some_and(|e| e == "npy") && p.is_file() {
            let (shape, _) = read_npy(p)?;
            return if shape.len() == 2 && shape[1] == c.n_codebooks {
                Ok((None, Some(codes(p)?)))
            } else {
                Ok((Some(embedding(p)?), None))
            };
        }
        if p.is_file() {
            if want == SpeakerMode::Reference {
                bail!("voice {name}: the reference mode takes codes (.npy); from a clip only the speaker embedding is computed");
            }
            return Ok((Some(self.speaker_embedding_file(p)?), None));
        }
        let known: Vec<&str> = self.voices.iter().map(|v| v.name.as_str()).collect();
        bail!(
            "voice {name:?}: not a .npy file and not one of the model's voices ({})",
            known.join(", ")
        )
    }

    /// The condition buckets of `o` in the model's order.
    fn condition_buckets(&self, o: &ZonosSpeak) -> Result<Vec<u32>> {
        let c = &self.lm.cfg;
        if let Some(b) = &o.condition_buckets {
            if b.len() != c.condition_buckets.len() || b.iter().zip(&c.condition_buckets).any(|(&b, &n)| b as usize > n) {
                bail!(
                    "condition_buckets: expected {} ids, each at most its bucket count {:?}",
                    c.condition_buckets.len(),
                    c.condition_buckets
                );
            }
            return Ok(b.clone());
        }
        Ok(c.condition_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let v = match name.as_str() {
                    "loudness" => o.loudness,
                    "fmax" => o.fmax,
                    "speaking_rate" => o.speaking_rate,
                    _ => None,
                };
                match v {
                    // bisect_right: bucket i holds [edge i-1, edge i).
                    Some(v) if !c.condition_edges[i].is_empty() => c.condition_edges[i].iter().filter(|&&e| e <= v).count() as u32,
                    _ => c.condition_buckets[i] as u32,
                }
            })
            .collect())
    }

    /// The prompt of `text` and, with speaker guidance, its twin without the speaker.
    pub fn prompt(&self, text: &str, o: &ZonosSpeak) -> Result<(Prompt, Option<Prompt>)> {
        let c = &self.lm.cfg;
        let n_cb = c.n_codebooks;
        let (pad, tpad) = (c.audio_pad_id, c.text_vocab as u32);
        let (embedding, codes) = self.resolve_voice(o)?;
        let text_ids = self.text_ids(text)?;
        let buckets = self.condition_buckets(o)?;
        if o.silence_prefill_frames > 30 {
            bail!("silence_prefill_frames must be 0..30");
        }
        let prefill = if c.silence_frame.len() == n_cb {
            o.silence_prefill_frames
        } else {
            0
        };
        let build = |speaker: bool, buckets: &[u32]| -> Prompt {
            let mut rows: Vec<u32> = Vec::new();
            let blank = |rows: &mut Vec<u32>| {
                rows.extend(std::iter::repeat_n(pad, n_cb));
                rows.push(tpad);
            };
            let mut prefix_rows = 0;
            let mut spk = None;
            if speaker {
                if let Some(e) = &embedding {
                    blank(&mut rows);
                    spk = Some(e.clone());
                } else if let Some(k) = &codes {
                    // 60 s at most; codebook j delayed by j rows, the tail flushed with padding.
                    let frames = (k.len() / n_cb).min(900);
                    prefix_rows = frames + n_cb - 1;
                    for r in 0..prefix_rows {
                        for j in 0..n_cb {
                            rows.push(if r >= j && r - j < frames { k[(r - j) * n_cb + j] } else { pad });
                        }
                        rows.push(tpad);
                    }
                }
            }
            let start = rows.len() / (n_cb + 1);
            let cond: Vec<(usize, u32)> = buckets.iter().enumerate().map(|(i, &b)| (start + i, b)).collect();
            for _ in 0..buckets.len() {
                blank(&mut rows);
            }
            for &t in &text_ids {
                rows.extend(std::iter::repeat_n(pad, n_cb));
                rows.push(t);
            }
            for r in 0..prefill {
                for j in 0..n_cb {
                    rows.push(if r >= j { c.silence_frame[j] } else { pad });
                }
                rows.push(tpad);
            }
            Prompt {
                rows,
                speaker: spk,
                prefix_rows,
                cond,
                prefill_rows: prefill,
                text_tokens: text_ids.len(),
            }
        };
        let cond = build(true, &buckets);
        let twin = if o.speaker_cfg_scale != 1.0 {
            if !o.speaker_cfg_scale.is_finite() || o.speaker_cfg_scale < 0.0 {
                bail!("speaker_cfg_scale must be finite and not negative");
            }
            if embedding.is_none() && codes.is_none() {
                bail!("speaker guidance needs a voice");
            }
            let unknown: Vec<u32> = c.condition_buckets.iter().map(|&n| n as u32).collect();
            Some(build(false, &unknown))
        } else {
            None
        };
        Ok((cond, twin))
    }

    /// Generates the rows of `prompt` (`twin`: its unconditional twin for speaker guidance).
    /// `on_row(rows so far, end frame)` is called after each row; `keep_logits` records every
    /// step's logits.
    pub fn generate(
        &self,
        prompt: &Prompt,
        twin: Option<&Prompt>,
        o: &ZonosSpeak,
        keep_logits: bool,
        on_row: &mut dyn FnMut(&[u32], Option<usize>) -> Flow,
    ) -> Result<Generated> {
        let c = &self.lm.cfg;
        let n_cb = c.n_codebooks;
        let w = n_cb + 1;
        let (pad, eoa, av) = (c.audio_pad_id, c.eoa_id, c.audio_vocab);
        let n_prompt = prompt.rows.len() / w;
        let longest = n_prompt.max(twin.map_or(0, |t| t.rows.len() / w));
        if longest >= c.max_seqlen {
            bail!("the prompt of {longest} rows fills the model context of {}", c.max_seqlen);
        }
        let budget = o.max_tokens.max(1).min(c.max_seqlen - longest);
        let has_audio_prefix = prompt.rows[(n_prompt - 1) * w..(n_prompt - 1) * w + n_cb].iter().any(|&v| v != pad);
        let spk = |p: &Prompt| p.speaker.clone().map(|e| (0usize, e));
        let mut branches: Vec<(&Prompt, LmState)> = vec![(prompt, self.lm.new_state())];
        if let Some(t) = twin {
            branches.push((t, self.lm.new_state()));
        }
        let mut outs: Vec<Vec<f32>> = Vec::new();
        for (p, st) in branches.iter_mut() {
            let s = spk(p);
            outs.push(
                self.lm
                    .forward(st, &p.rows, p.rows.len() / w, s.as_ref().map(|(r, e)| (*r, &e[..])), &p.cond)?,
            );
        }
        let reps = if o.repetition_codebooks < 0 {
            n_cb
        } else {
            (o.repetition_codebooks as usize).min(n_cb)
        };
        let mut rng = Rng::new(o.seed);
        let mut rows: Vec<u32> = Vec::new();
        let mut logits_log = Vec::new();
        let (mut eos_frame, mut countdown) = (None::<usize>, 0usize);
        let mut finish = StopReason::Length;
        loop {
            let n = rows.len() / n_cb;
            let logits: Vec<f32> = match outs.as_slice() {
                [cond, uncond] => {
                    let s = o.speaker_cfg_scale;
                    cond.iter().zip(uncond).map(|(c, u)| u + s * (c - u)).collect()
                }
                _ => std::mem::take(&mut outs[0]),
            };
            let window = if o.repetition_window > 0 && o.repetition_penalty > 1.0 {
                o.repetition_window.min(n)
            } else {
                0
            };
            let mut row = vec![0u32; n_cb];
            for (j, r) in row.iter_mut().enumerate() {
                let recent: Vec<u32> = if j < reps {
                    (n - window..n).map(|i| rows[i * n_cb + j]).collect()
                } else {
                    Vec::new()
                };
                *r = sample_code(&logits[j * av..(j + 1) * av], &recent, o, c.codebook_size, eoa, &mut rng);
            }
            // Without audio in the prompt, the codebooks whose delayed stream has not started
            // are padding (the model was never trained on those positions).
            if !has_audio_prefix && n < n_cb - 1 {
                row[n + 1..].iter_mut().for_each(|v| *v = pad);
            }
            if keep_logits {
                logits_log.push(logits);
            }
            rows.extend_from_slice(&row);
            let mut finished = n + 1 >= budget;
            if eos_frame.is_none()
                && let Some(col) = row.iter().rposition(|&v| v == eoa)
            {
                eos_frame = Some(n.saturating_sub(col));
                countdown = n_cb + 1;
            }
            if countdown > 0 {
                countdown -= 1;
                if countdown == 0 {
                    finished = true;
                    finish = StopReason::Stop;
                }
            }
            if self.cancel.swap(false, Ordering::Relaxed) || matches!(on_row(&rows, eos_frame), Flow::Stop) {
                finish = StopReason::Cancelled;
                finished = true;
            }
            if finished {
                return Ok(Generated {
                    rows,
                    eos_frame,
                    finish,
                    logits: logits_log,
                });
            }
            let mut ids = row;
            ids.push(c.text_vocab as u32);
            outs.clear();
            for (p, st) in branches.iter_mut() {
                let s = spk(p);
                outs.push(self.lm.forward(st, &ids, 1, s.as_ref().map(|(r, e)| (*r, &e[..])), &p.cond)?);
            }
        }
    }

    /// The complete frames of an utterance, the delay pattern undone: the silence rows of
    /// `prompt` followed by `rows` (`[n, n_codebooks]`), up to the end frame. `[frames,
    /// n_codebooks]` and the frame count.
    pub fn frames(&self, prompt: &Prompt, rows: &[u32], eos_frame: Option<usize>) -> (Vec<u32>, usize) {
        let c = &self.lm.cfg;
        let (n_cb, w) = (c.n_codebooks, c.n_codebooks + 1);
        let n_prompt = prompt.rows.len() / w;
        let mut buf: Vec<u32> = Vec::new();
        for r in n_prompt - prompt.prefill_rows..n_prompt {
            buf.extend_from_slice(&prompt.rows[r * w..r * w + n_cb]);
        }
        buf.extend_from_slice(rows);
        let n = buf.len() / n_cb;
        // The end: the aligned end frame, or the first frame an end token sits in.
        let mut end = eos_frame.map(|e| e + prompt.prefill_rows);
        for r in 0..n {
            if let Some(col) = buf[r * n_cb..(r + 1) * n_cb].iter().rposition(|&v| v == c.eoa_id) {
                let f = r.saturating_sub(col);
                end = Some(end.map_or(f, |e| e.min(f)));
            }
        }
        let mut target = n.saturating_sub(n_cb - 1);
        if let Some(e) = end {
            target = target.min(e);
        }
        let mut out = Vec::with_capacity(target * n_cb);
        for f in 0..target {
            for j in 0..n_cb {
                out.push(buf[(f + j) * n_cb + j]);
            }
        }
        // A frame with padding or an end token in it (a cancelled or cut utterance) ends the audio.
        let good = out
            .chunks(n_cb)
            .take_while(|f| f.iter().all(|&v| (v as usize) < c.codebook_size))
            .count();
        out.truncate(good * n_cb);
        (out, good)
    }

    /// Frames (`[frames, n_codebooks]`) to audio, with the output gain and the soft limiter of
    /// the reference servers (samples above -1 dBFS bend into full scale).
    pub fn decode(&self, frames: &[u32], n: usize, o: &ZonosSpeak) -> Result<Audio> {
        let mut samples = if n == 0 { Vec::new() } else { self.decoder.decode(frames, n)? };
        let gain = 10f32.powf(-o.headroom_db.unwrap_or(self.decoder.headroom_db) / 20.0);
        let knee = 10f32.powf(-1.0 / 20.0);
        let width = 1.0 - knee;
        for v in samples.iter_mut() {
            *v *= gain;
            if v.abs() > knee {
                *v = v.signum() * (knee + width * ((v.abs() - knee) / width).tanh());
            }
        }
        let fade = ((self.decoder.sample_rate as f32 * o.fade_out_ms.max(0.0) / 1000.0) as usize).min(samples.len());
        let ns = samples.len();
        for (i, v) in samples[ns - fade..].iter_mut().enumerate() {
            let x = if fade > 1 { i as f32 / (fade - 1) as f32 } else { 1.0 };
            *v *= 0.5 * (1.0 + (std::f32::consts::PI * x).cos());
        }
        Ok(Audio {
            samples,
            sample_rate: self.decoder.sample_rate,
        })
    }

    /// Speaks `text`.
    pub fn speak(&self, text: &str, o: &ZonosSpeak) -> Result<Speech> {
        self.speak_stream(text, o, &mut |_| Flow::Continue)
    }

    /// Speaks `text`, calling `on_chunk` with the audio. The utterance is decoded when all of it
    /// is generated (the decoder is not incremental here), so the chunks arrive together at the
    /// end, one second each.
    pub fn speak_stream(&self, text: &str, o: &ZonosSpeak, on_chunk: &mut dyn FnMut(&AudioChunk) -> Flow) -> Result<Speech> {
        let t0 = Instant::now();
        self.cancel.store(false, Ordering::Relaxed);
        let (prompt, twin) = self.prompt(text, o)?;
        let w = self.lm.cfg.n_codebooks + 1;
        let mut first = None;
        let g = self.generate(&prompt, twin.as_ref(), o, false, &mut |_, _| {
            first.get_or_insert_with(|| t0.elapsed().as_secs_f64());
            Flow::Continue
        })?;
        let t1 = Instant::now();
        let (frames, n) = self.frames(&prompt, &g.rows, g.eos_frame);
        let audio = self.decode(&frames, n, o)?;
        let codec_secs = t1.elapsed().as_secs_f64();
        let sr = audio.sample_rate as usize;
        let mut start = 0;
        let mut first_chunk = None;
        while start < audio.samples.len() {
            let end = (start + sr).min(audio.samples.len());
            first_chunk.get_or_insert_with(|| t0.elapsed().as_secs_f64());
            let chunk = AudioChunk {
                samples: &audio.samples[start..end],
                start,
                sample_rate: audio.sample_rate,
                frames: n,
                last: end == audio.samples.len(),
            };
            if matches!(on_chunk(&chunk), Flow::Stop) {
                break;
            }
            start = end;
        }
        let prefill_secs = first.unwrap_or(0.0);
        let total = t0.elapsed().as_secs_f64();
        let stats = SpeakStats {
            prompt_tokens: prompt.rows.len() / w,
            tokens: g.rows.len() / self.lm.cfg.n_codebooks,
            frames: n,
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
            tokens: frames,
            frames: Vec::new(),
            finish: g.finish,
            stats,
        })
    }
}

/// The voices of a `voices/` folder: `<language>/<Name>_<tags>.embedding.npy` and `.codes.npy`.
fn scan_voices(dir: &Path) -> Vec<Voice> {
    let mut out: Vec<Voice> = Vec::new();
    let mut dirs = vec![(dir.to_path_buf(), String::new())];
    while let Some((d, lang)) = dirs.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                dirs.push((p, name));
                continue;
            }
            let (label, emb) = match (name.strip_suffix(".embedding.npy"), name.strip_suffix(".codes.npy")) {
                (Some(l), _) => (l, true),
                (_, Some(l)) => (l, false),
                _ => continue,
            };
            let i = match out.iter().position(|v| v.label == label && v.language == lang) {
                Some(i) => i,
                None => {
                    out.push(Voice {
                        name: label.split('_').next().unwrap_or(label).to_string(),
                        label: label.to_string(),
                        language: lang.clone(),
                        ..Voice::default()
                    });
                    out.len() - 1
                }
            };
            if emb {
                out[i].embedding = Some(p);
            } else {
                out[i].codes = Some(p);
            }
        }
    }
    out.sort_by(|a, b| (&a.language, &a.label).cmp(&(&b.language, &b.label)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_breaks_collapse_to_one_space() {
        assert_eq!(collapse_line_breaks("a  b"), "a  b");
        assert_eq!(collapse_line_breaks("a \n b\tc\r\n"), "a b c ");
        assert_eq!(collapse_line_breaks("one\n\ntwo"), "one two");
    }

    #[test]
    fn npy_arrays_and_voice_folders() {
        let dir = std::env::temp_dir().join(format!("oxinfer-zonos-{}", std::process::id()));
        let lang = dir.join("English");
        std::fs::create_dir_all(&lang).unwrap();
        let npy = |descr: &str, shape: &str, data: &[u8]| {
            let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
            while (10 + h.len() + 1) % 64 != 0 {
                h.push(' ');
            }
            h.push('\n');
            let mut b = b"\x93NUMPY\x01\x00".to_vec();
            b.extend_from_slice(&(h.len() as u16).to_le_bytes());
            b.extend_from_slice(h.as_bytes());
            b.extend_from_slice(data);
            b
        };
        let codes: Vec<u8> = [7i32, 4095, 0, 12].iter().flat_map(|v| v.to_le_bytes()).collect();
        let emb: Vec<u8> = [0.5f32, -1.25].iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(lang.join("Ada_en_F_1.codes.npy"), npy("<i4", "(2, 2)", &codes)).unwrap();
        std::fs::write(lang.join("Ada_en_F_1.embedding.npy"), npy("<f4", "(2,)", &emb)).unwrap();
        let (shape, v) = read_npy(&lang.join("Ada_en_F_1.codes.npy")).unwrap();
        assert_eq!((shape, v), (vec![2, 2], vec![7.0, 4095.0, 0.0, 12.0]));
        let (shape, v) = read_npy(&lang.join("Ada_en_F_1.embedding.npy")).unwrap();
        assert_eq!((shape, v), (vec![2], vec![0.5, -1.25]));
        let voices = scan_voices(&dir);
        assert_eq!(voices.len(), 1);
        assert_eq!((voices[0].name.as_str(), voices[0].language.as_str()), ("Ada", "English"));
        assert!(voices[0].embedding.is_some() && voices[0].codes.is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sampler_masks_special_ids_and_penalizes_repeats() {
        let mut o = ZonosSpeak {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = Rng::new(1);
        // 4 codes, EOA (4), padding (5): padding wins on logits but is never sampled.
        let l = [1.0, 3.0, 2.0, -1.0, 0.5, 9.0];
        assert_eq!(sample_code(&l, &[], &o, 4, 4, &mut rng), 1);
        // The repeated code 1 is divided by 2.4 (1.25), below code 2.
        assert_eq!(sample_code(&l, &[1, 1], &o, 4, 4, &mut rng), 2);
        o.temperature = 1.0;
        o.top_k = 2;
        o.top_p = 0.0;
        for _ in 0..64 {
            assert!(matches!(sample_code(&l, &[], &o, 4, 4, &mut rng), 1 | 2));
        }
    }
}
