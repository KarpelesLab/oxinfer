//! Full-duplex speech to speech with Haru (higurashi-ai/Haru-4B, `model_type: haru`): the
//! model listens and speaks at the same time, one frame of 3200 samples at 24 kHz (133 ms) in
//! and one out. A port of the PyTorch reference of <https://github.com/higurashi-ai/Haru>
//! (`haru/session.py`, `prompt.py`, `sampling.py`, `audio.py`, `models/`).
//!
//! One position of the language model (the Qwen3.5-4B text model, [`crate::qwen`]) is one
//! frame. Its input is the sum of three streams: the embedding of the model's text token of that
//! frame, the model's own audio latent and the user's audio latent, each latent through its
//! projector ([`head::Projector`]). The hidden state at a frame gives the next frame: its text
//! token through the tied embedding matrix, and its audio latent through the MeanFlow head
//! ([`head::Head`]), which the codec ([`codec::Codec`]) decodes right away. Frames without text
//! hold a padding token; an end-of-padding token sits on the frame before the model speaks.
//!
//! The prompt is a character card as a ChatML system turn (text only), then the voice reference:
//! the latents of a clip on the model's audio stream, under padding tokens and one
//! end-of-padding token.
//!
//! ```no_run
//! use oxinfer::haru::{DuplexConfig, DuplexSession, Haru, HaruOptions};
//! # fn main() -> anyhow::Result<()> {
//! let model = Haru::load("higurashi-ai/Haru-4B", &HaruOptions::default())?;
//! let voice = model.load_reference("voice.wav".as_ref())?;
//! let mut session = DuplexSession::new(&model, "", Some(&voice), DuplexConfig::default());
//! let mut frame = session.start()?; // the model's first 133 ms: frame.audio, frame.text
//! # let user_audio = vec![0f32; 3200];
//! while !frame.done {
//!     frame = session.step(&user_audio)?; // 3200 samples in, 3200 out
//! }
//! # Ok(())
//! # }
//! ```

pub mod codec;
pub mod head;

use crate::config::QwenConfig;
use crate::model::Quant;
use crate::qwen::{QwenModel, QwenSession};
use crate::safetensors::ModelFiles;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow, bail};
use codec::{Codec, CodecConfig};
use head::{Head, HeadConfig, Projector};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The repository of the released model.
pub const REPO: &str = "higurashi-ai/Haru-4B";

/// How to load the model.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HaruOptions {
    /// Precision of the language model's matrices (bf16 as stored, or quantized at load).
    pub quant: Quant,
    /// Map the weights instead of reading them into memory.
    pub mmap: bool,
    /// CPU worker threads (process-wide; default: all cores).
    pub threads: Option<usize>,
    /// Where repo ids are downloaded.
    pub cache_dir: Option<PathBuf>,
}

impl Default for HaruOptions {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            mmap: false,
            threads: None,
            cache_dir: None,
        }
    }
}

/// The DRY penalty of the text stream: a token that would extend a sequence which already
/// occurred is penalized by `multiplier * base^(match length - allowed_length)`.
#[derive(Clone, Debug)]
pub struct DryConfig {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub lookback: usize,
    pub max_match_length: usize,
    pub max_penalty: f32,
}

impl Default for DryConfig {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 1,
            lookback: 1024,
            max_match_length: 32,
            max_penalty: 32.0,
        }
    }
}

/// `{token: penalty}` for the tokens that would extend an earlier repeat of the end of `history`.
pub fn dry_penalties(history: &[u32], c: &DryConfig) -> HashMap<u32, f32> {
    let n = history.len();
    let mut longest: HashMap<u32, usize> = HashMap::new();
    if n < c.allowed_length.max(1) {
        return HashMap::new();
    }
    let last = history[n - 1];
    let lo = (n - 1).saturating_sub(c.lookback);
    for j in (lo..n.saturating_sub(1)).rev() {
        if history[j] != last {
            continue;
        }
        let mut length = 1;
        while length < c.max_match_length && j >= length && history[j - length] == history[n - 1 - length] {
            length += 1;
        }
        if length >= c.allowed_length {
            let e = longest.entry(history[j + 1]).or_insert(0);
            *e = (*e).max(length);
        }
    }
    longest
        .into_iter()
        .map(|(t, l)| (t, c.max_penalty.min(c.multiplier * c.base.powi((l - c.allowed_length) as i32))))
        .collect()
}

/// Settings of one conversation; the defaults are the reference's.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DuplexConfig {
    /// Guidance of the audio head (1 to 2).
    pub guidance_scale: f32,
    /// Of the text stream; 0 is greedy.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub dry: Option<DryConfig>,
    /// Added to the end-of-padding logit: above 0 the model starts speaking sooner.
    pub eop_bias: f32,
    /// The model opens the conversation.
    pub speak_first: bool,
    pub max_frames: usize,
    /// At most this much of the end of the voice reference is used.
    pub reference_seconds: f32,
    pub seed: u64,
}

impl Default for DuplexConfig {
    fn default() -> Self {
        Self {
            guidance_scale: 1.5,
            temperature: 0.7,
            top_k: 25,
            top_p: 1.0,
            dry: Some(DryConfig::default()),
            eop_bias: 0.0,
            speak_first: false,
            max_frames: 27_000,
            reference_seconds: 180.0,
            seed: 0,
        }
    }
}

/// One frame of the model.
#[derive(Clone, Debug)]
pub struct Frame {
    /// `frame_samples` samples at 24 kHz, within -1 .. 1.
    pub audio: Vec<f32>,
    pub token: u32,
    /// The new text of this frame, empty if none.
    pub text: String,
    pub index: usize,
    /// The audio latent of the frame (as the codec takes it).
    pub latent: Vec<f32>,
    /// The language model's hidden state the frame came from.
    pub hidden: Vec<f32>,
    pub done: bool,
}

/// Where a session's standard normal values come from: its own generator, or values recorded
/// from another implementation (to compare with it draw for draw).
pub trait Noise {
    fn normal(&mut self, n: usize) -> Vec<f32>;
}

/// xoshiro256** seeded through splitmix64.
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Self {
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

impl Noise for Rng {
    fn normal(&mut self, n: usize) -> Vec<f32> {
        // Box-Muller.
        (0..n)
            .map(|_| {
                let (u, v) = (1.0 - self.uniform(), self.uniform());
                ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
            })
            .collect()
    }
}

/// Recorded values, handed out in order; zeros once they run out.
pub struct Replay {
    values: Vec<f32>,
    at: usize,
}

impl Replay {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values, at: 0 }
    }
    /// Values not handed out yet.
    pub fn left(&self) -> usize {
        self.values.len() - self.at
    }
}

impl Noise for Replay {
    fn normal(&mut self, n: usize) -> Vec<f32> {
        let end = (self.at + n).min(self.values.len());
        let mut v = self.values[self.at..end].to_vec();
        self.at = end;
        v.resize(n, 0.0);
        v
    }
}

#[derive(Deserialize)]
struct RawConfig {
    sample_rate: u32,
    frame_samples: usize,
    latent_dim: usize,
    pad_token_id: u32,
    eop_token_id: u32,
    lm: serde_json::Value,
    codec: CodecConfig,
    head: HeadConfig,
}

/// What a loaded model is.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HaruInfo {
    pub dir: PathBuf,
    pub layers: usize,
    pub hidden_size: usize,
    pub weight_bytes: u64,
    pub threads: usize,
    pub sample_rate: u32,
    pub frame_samples: usize,
    pub load_secs: f64,
}

pub struct Haru {
    // Field order: everything that borrows `_files` is dropped before it.
    lm: QwenModel<'static>,
    model_proj: Projector,
    user_proj: Projector,
    head: Head,
    codec: Codec,
    tok: Tokenizer,
    info: HaruInfo,
    pub pad_id: u32,
    pub eop_id: u32,
    pub eos_id: Option<u32>,
    latent_scale: f32,
    latent_bias: f32,
    _files: Box<ModelFiles>,
}

/// Whether `model` is the directory of a Haru checkpoint (`model_type: haru`).
pub fn is_haru(model: &Path) -> bool {
    model.is_dir()
        && crate::fsx::read(&model.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|c| c["model_type"] == "haru")
}

/// Leading and trailing audio quieter than -45 dBFS cut (RMS over windows of 20 ms, 0.2 s kept
/// around the speech), then at most the last `max_seconds`: the reference's `prepare_reference`.
pub fn prepare_reference(audio: &[f32], max_seconds: f32, sample_rate: u32) -> Vec<f32> {
    let window = (sample_rate / 50) as usize;
    let count = audio.len() / window.max(1);
    let mut trimmed = audio;
    if count > 0 {
        let voiced: Vec<usize> = (0..count)
            .filter(|&w| {
                let s = &audio[w * window..(w + 1) * window];
                let rms = (s.iter().map(|v| v * v).sum::<f32>() / window as f32 + 1e-12).sqrt();
                20.0 * rms.log10() > -45.0
            })
            .collect();
        trimmed = match (voiced.first(), voiced.last()) {
            (Some(&a), Some(&b)) => {
                let margin = (0.2 * sample_rate as f32) as usize;
                &audio[(a * window).saturating_sub(margin)..((b + 1) * window + margin).min(audio.len())]
            }
            _ => &audio[..0],
        };
    }
    let limit = (max_seconds * sample_rate as f32) as usize;
    trimmed[trimmed.len().saturating_sub(limit)..].to_vec()
}

impl Haru {
    /// Loads the model: a checkpoint directory (`config.json`, `model.safetensors`,
    /// `tokenizer.json`) or a repo id ([`REPO`], 10.5 GB).
    pub fn load(model: &str, opts: &HaruOptions) -> Result<Self> {
        let t0 = crate::clock::Instant::now();
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
                crate::hf::resolve_files(
                    model,
                    &["config.json", "tokenizer.json", "model.safetensors"],
                    opts.cache_dir.as_deref(),
                )?
            }
            #[cfg(not(feature = "hf"))]
            bail!("{model}: not a folder (built without the `hf` feature, so repo ids cannot be fetched)");
        };
        let raw: RawConfig =
            serde_json::from_slice(&crate::fsx::read(&dir.join("config.json")).with_context(|| format!("{}: config.json", dir.display()))?)
                .with_context(|| format!("{}: config.json of a haru model", dir.display()))?;
        let lm_cfg: QwenConfig = serde_json::from_value(raw.lm.clone()).context("haru: the lm section of config.json")?;
        if raw.lm["model_type"] != "qwen3_5_text" {
            bail!(
                "haru: a language model of type {} is not supported (qwen3_5_text)",
                raw.lm["model_type"]
            );
        }
        let mut files = Box::new(ModelFiles::open(&dir, !opts.mmap)?);
        // The language model under the names of a bare Qwen3.5 text model.
        files.rename(|n| n.strip_prefix("lm.").map(str::to_string));
        // SAFETY: the box's heap block does not move, and `Haru` drops everything that borrows
        // from it before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let eos_id = lm_cfg.eos_id();
        let lm = QwenModel::load(lm_cfg, fs, opts.quant)?;
        let h = lm.cfg.hidden_size;
        if raw.head.hidden_size != h || raw.head.latent_dim != raw.latent_dim || raw.codec.latent_dim != raw.latent_dim {
            bail!("haru: the head, the codec and the language model do not fit together");
        }
        let codec = Codec::load(fs, &raw.codec)?;
        if codec.frame_samples != raw.frame_samples {
            bail!("haru: the codec's ratios do not give frames of {} samples", raw.frame_samples);
        }
        let scalar = |name: &str, default: f32| -> Result<f32> {
            Ok(if fs.has(name) {
                fs.f32(name)?.first().copied().unwrap_or(default)
            } else {
                default
            })
        };
        let head = Head::load(fs, raw.head)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{}: tokenizer.json: {e}", dir.display()))?;
        let info = HaruInfo {
            layers: lm.cfg.num_hidden_layers,
            hidden_size: h,
            weight_bytes: fs.total_bytes(),
            threads: crate::pool::threads(),
            sample_rate: raw.sample_rate,
            frame_samples: raw.frame_samples,
            load_secs: t0.elapsed().as_secs_f64(),
            dir,
        };
        Ok(Self {
            model_proj: Projector::load(fs, "model_audio_proj", raw.latent_dim, h)?,
            user_proj: Projector::load(fs, "user_audio_proj", raw.latent_dim, h)?,
            latent_scale: scalar("latent_scale", 1.0)?,
            latent_bias: scalar("latent_bias", 0.0)?,
            lm,
            head,
            codec,
            tok,
            info,
            pad_id: raw.pad_token_id,
            eop_id: raw.eop_token_id,
            eos_id,
            _files: files,
        })
    }

    pub fn info(&self) -> &HaruInfo {
        &self.info
    }

    pub fn codec(&self) -> &Codec {
        &self.codec
    }

    pub fn head(&self) -> &Head {
        &self.head
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// A voice reference from an audio file: mono at 24 kHz, silence trimmed, at most its last
    /// 180 s.
    pub fn load_reference(&self, path: &Path) -> Result<Vec<f32>> {
        let pcm = crate::tts::load_audio_file(path, self.info.sample_rate)?;
        Ok(prepare_reference(&pcm, 180.0, self.info.sample_rate))
    }

    /// The token ids of the character card as a system turn (none for an empty card).
    pub fn system_ids(&self, system_prompt: &str) -> Result<Vec<u32>> {
        if system_prompt.is_empty() {
            return Ok(Vec::new());
        }
        let text = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
        Ok(self
            .tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Codec latents in the normalized space of the head and the projectors.
    fn scale(&self, raw: &[f32]) -> Vec<f32> {
        raw.iter().map(|v| (v + self.latent_bias) * self.latent_scale).collect()
    }

    fn unscale(&self, normalized: &[f32]) -> Vec<f32> {
        normalized.iter().map(|v| v / self.latent_scale - self.latent_bias).collect()
    }

    /// The input of one position: the token's embedding plus the projections of the latents
    /// that position carries.
    fn embed(&self, token: u32, model_latent: Option<&[f32]>, user_latent: Option<&[f32]>) -> Vec<f32> {
        let mut x = vec![0f32; self.lm.cfg.hidden_size];
        self.lm.embed.row(token as usize, &mut x);
        for (proj, latent) in [(&self.model_proj, model_latent), (&self.user_proj, user_latent)] {
            if let Some(l) = latent {
                let y = proj.forward(&self.scale(l));
                x.iter_mut().zip(&y).for_each(|(x, y)| *x += y);
            }
        }
        x
    }

    /// Latents around the codec's means (`[n, latent_dim]`), as the reference draws them: one
    /// scale for the whole call, then a value per element.
    fn sample_latents(&self, mean: &[f32], noise: &mut dyn Noise) -> Vec<f32> {
        let scale = noise.normal(1)[0] * (self.codec.latent_std / 0.8);
        let n = noise.normal(mean.len());
        mean.iter().zip(&n).map(|(m, n)| m + scale * n).collect()
    }

    /// The latent of digital silence (the last of 16 frames of zeros).
    fn silence_latent(&self) -> Vec<f32> {
        let d = self.codec.latent_dim;
        let m = self.codec.encode(&vec![0f32; 16 * self.codec.frame_samples]);
        m[m.len() - d..].to_vec()
    }
}

/// The text of the model's stream as it arrives: a run of padding is one space, and a character
/// of several tokens is held until all of them are there.
struct Detokenizer {
    pending: Vec<u32>,
    in_gap: bool,
}

/// A live conversation with [`Haru`]: [`DuplexSession::start`] gives the model's frame 0, then
/// each [`DuplexSession::step`] with the user's frame `k` gives the model's frame `k + 1`.
pub struct DuplexSession<'m> {
    model: &'m Haru,
    pub config: DuplexConfig,
    system_prompt: String,
    reference: Option<Vec<f32>>,
    noise: Box<dyn Noise + 'm>,
    rng: Rng,
    lm: QwenSession,
    enc: codec::State,
    dec: codec::State,
    token: u32,
    latent: Option<Vec<f32>>,
    force_text: usize,
    history: Vec<u32>,
    penalties: HashMap<u32, f32>,
    detok: Detokenizer,
    /// Positions of the prompt.
    pub prompt_length: usize,
    /// The prompt's text tokens.
    pub prompt_tokens: Vec<u32>,
    pub index: usize,
}

impl<'m> DuplexSession<'m> {
    /// `reference`: a mono clip at 24 kHz of the voice the model should speak with.
    pub fn new(model: &'m Haru, system_prompt: &str, reference: Option<&[f32]>, config: DuplexConfig) -> Self {
        let seed = config.seed;
        Self::with_noise(model, system_prompt, reference, config, Box::new(Rng::new(seed ^ 0x6861_7275)))
    }

    /// [`DuplexSession::new`] with the normal values of the latents taken from `noise`.
    pub fn with_noise(
        model: &'m Haru,
        system_prompt: &str,
        reference: Option<&[f32]>,
        config: DuplexConfig,
        noise: Box<dyn Noise + 'm>,
    ) -> Self {
        let limit = (config.reference_seconds * model.info.sample_rate as f32) as usize;
        let reference = reference
            .filter(|r| !r.is_empty())
            .map(|r| r[r.len().saturating_sub(limit)..].to_vec());
        Self {
            rng: Rng::new(config.seed),
            lm: model.lm.new_session(),
            enc: model.codec.encoder_state(),
            dec: model.codec.decoder_state(),
            token: model.pad_id,
            latent: None,
            force_text: 0,
            history: Vec::new(),
            penalties: HashMap::new(),
            detok: Detokenizer {
                pending: Vec::new(),
                in_gap: false,
            },
            prompt_length: 0,
            prompt_tokens: Vec::new(),
            index: 0,
            system_prompt: system_prompt.to_string(),
            reference,
            noise,
            model,
            config,
        }
    }

    fn silent(&self, token: u32) -> bool {
        token == self.model.pad_id || token == self.model.eop_id
    }

    fn banned(&self, token: u32) -> bool {
        self.silent(token) || Some(token) == self.model.eos_id
    }

    /// Runs the prompt (the card, then the voice reference) and gives frame 0.
    pub fn start(&mut self) -> Result<Frame> {
        let m = self.model;
        let (d, fs) = (m.codec.latent_dim, m.codec.frame_samples);
        let (pad, eop) = (m.pad_id, m.eop_id);
        self.lm = m.lm.new_session();
        self.enc = m.codec.encoder_state();
        self.dec = m.codec.decoder_state();
        let card = m.system_ids(&self.system_prompt)?;
        let ref_latents = match &self.reference {
            Some(r) => Some(m.sample_latents(&m.codec.encode(r), self.noise.as_mut())),
            None => None,
        };
        let n_ref = ref_latents.as_ref().map_or(0, |l| l.len() / d);
        // A reference already ends in the end-of-padding token, so an opening turn starts with
        // padding and that token.
        let mut grid: Vec<u32> = if self.config.speak_first { vec![pad, eop] } else { Vec::new() };
        if card.is_empty() && n_ref == 0 && grid.is_empty() {
            grid = vec![pad];
        }
        let mut tokens = card.clone();
        if n_ref > 0 {
            tokens.extend(std::iter::repeat_n(pad, n_ref - 1));
            tokens.push(eop);
        }
        tokens.extend(&grid);
        let (n, n_card, n_grid) = (tokens.len(), card.len(), grid.len());
        // Positions before any generated audio carry the speaker's room tone (the reference's
        // quietest frame) on the model's side and encoded silence on the user's.
        let (room, silence) = if n_grid > 0 {
            let silence = m.silence_latent();
            let whole = self.reference.as_ref().map_or(0, |r| r.len() / fs);
            let room = match (&self.reference, &ref_latents) {
                (Some(r), Some(l)) if whole > 0 => {
                    let energy = |f: usize| r[f * fs..(f + 1) * fs].iter().map(|v| v * v).sum::<f32>() / fs as f32;
                    let quiet = (0..whole).fold(0, |b, f| if energy(f) < energy(b) { f } else { b });
                    l[quiet * d..(quiet + 1) * d].to_vec()
                }
                _ => silence.clone(),
            };
            (room, silence)
        } else {
            (Vec::new(), Vec::new())
        };
        let h = m.lm.cfg.hidden_size;
        let mut x = Vec::with_capacity(n * h);
        for (i, &tok) in tokens.iter().enumerate() {
            let model_latent = if i >= n - n_grid {
                Some(&room[..])
            } else if i >= n_card {
                ref_latents.as_ref().map(|l| &l[(i - n_card) * d..(i - n_card + 1) * d])
            } else {
                None
            };
            let user_latent = (i >= n - n_grid).then_some(&silence[..]);
            x.extend(m.embed(tok, model_latent, user_latent));
        }
        let mm: Vec<(usize, &[f32])> = x.chunks(h).enumerate().collect();
        let hidden = m.lm.forward_hidden_mm(&mut self.lm, &tokens, &mm);
        self.force_text = usize::from(self.config.speak_first);
        self.prompt_length = n;
        self.prompt_tokens = tokens;
        self.index = 0;
        self.history.clear();
        self.penalties.clear();
        self.emit(&hidden[(n - 1) * h..])
    }

    /// Takes one frame of the user (`frame_samples` samples at 24 kHz) and gives the model's
    /// next frame.
    pub fn step(&mut self, user_audio: &[f32]) -> Result<Frame> {
        let m = self.model;
        let Some(latent) = self.latent.take() else {
            bail!("call start() first");
        };
        if user_audio.len() != m.codec.frame_samples {
            bail!("a user frame is {} samples, got {}", m.codec.frame_samples, user_audio.len());
        }
        let mean = m.codec.encode_stream(user_audio, &mut self.enc);
        let user_latent = m.sample_latents(&mean, self.noise.as_mut());
        let x = m.embed(self.token, Some(&latent), Some(&user_latent));
        let hidden = m.lm.forward_hidden_mm(&mut self.lm, &[self.token], &[(0, &x[..])]);
        self.emit(&hidden)
    }

    /// The next frame's text token and audio from a hidden state.
    fn emit(&mut self, hidden: &[f32]) -> Result<Frame> {
        let m = self.model;
        let c = self.config.clone();
        let mut logits = m.lm.logits(hidden, 1);
        for (&t, &p) in &self.penalties {
            logits[t as usize] -= p;
        }
        logits[m.eop_id as usize] += c.eop_bias;
        if self.force_text > 0 {
            for t in [Some(m.pad_id), Some(m.eop_id), m.eos_id].into_iter().flatten() {
                logits[t as usize] = f32::NEG_INFINITY;
            }
            self.force_text -= 1;
        }
        let token = sample_token(&logits, c.temperature, c.top_k, c.top_p, &mut self.rng);
        let noise = self.noise.normal(m.codec.latent_dim);
        let latent = m.unscale(&m.head.sample(hidden, &noise, c.guidance_scale));
        let mut audio = m.codec.decode_stream(&latent, &mut self.dec);
        audio.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
        if let Some(dry) = &c.dry
            && !self.banned(token)
        {
            self.history.push(token);
            let extra = self.history.len().saturating_sub(dry.lookback + dry.max_match_length + 1);
            self.history.drain(..extra);
            self.penalties = dry_penalties(&self.history, dry);
            let (pad, eop, eos) = (m.pad_id, m.eop_id, m.eos_id);
            self.penalties.retain(|&t, _| t != pad && t != eop && Some(t) != eos);
        }
        let text = self.push_text(token)?;
        self.token = token;
        self.latent = Some(latent.clone());
        let index = self.index;
        self.index += 1;
        Ok(Frame {
            audio,
            token,
            text,
            index,
            latent,
            hidden: hidden.to_vec(),
            done: self.index >= self.config.max_frames,
        })
    }

    fn push_text(&mut self, token: u32) -> Result<String> {
        if self.silent(token) {
            if self.detok.in_gap {
                return Ok(String::new());
            }
            self.detok.in_gap = true;
            return Ok(" ".to_string());
        }
        self.detok.in_gap = false;
        self.detok.pending.push(token);
        let text = self
            .model
            .tok
            .decode(&self.detok.pending, false)
            .map_err(|e| anyhow!("detokenize: {e}"))?;
        if text.ends_with('\u{fffd}') {
            return Ok(String::new());
        }
        self.detok.pending.clear();
        Ok(text)
    }
}

/// One token from `logits`: greedy at a temperature of 0, else among the `top_k` most likely
/// (all of them for 0) at `temperature`, those before which the probabilities reach `top_p`.
fn sample_token(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return logits.iter().enumerate().fold(0, |b, (i, &v)| if v > logits[b] { i } else { b }) as u32;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    let k = if top_k > 0 && top_k < logits.len() { top_k } else { logits.len() };
    let by = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b));
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, by);
        order.truncate(k);
    }
    order.sort_by(by);
    let max = logits[order[0]];
    let mut p: Vec<f64> = order.iter().map(|&i| (((logits[i] - max) / temperature) as f64).exp()).collect();
    let total: f64 = p.iter().sum();
    if top_p > 0.0 && top_p < 1.0 {
        let mut before = 0.0;
        for v in p.iter_mut() {
            let share = *v / total;
            if before >= top_p as f64 {
                *v = 0.0;
            }
            before += share;
        }
    }
    let mut u = rng.uniform() * p.iter().sum::<f64>();
    for (&i, &v) in order.iter().zip(&p) {
        if v > 0.0 {
            if u < v {
                return i as u32;
            }
            u -= v;
        }
    }
    order[0] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_penalizes_what_would_repeat() {
        let c = DryConfig::default();
        // "1 2 3 1 2": 3 would extend the repeat of "1 2" (length 2), and after the lone 2 too.
        let p = dry_penalties(&[1, 2, 3, 1, 2], &c);
        assert_eq!(p.len(), 1);
        assert!((p[&3] - 0.8 * 1.75).abs() < 1e-6);
        // A match of length 1 costs the multiplier; longer matches are capped.
        assert!((dry_penalties(&[7, 9, 7], &c)[&9] - 0.8).abs() < 1e-6);
        let long: Vec<u32> = (0..40).chain(0..40).collect();
        assert_eq!(dry_penalties(&long, &c)[&0], 32.0);
        assert!(dry_penalties(&[], &c).is_empty());
        assert!(dry_penalties(&[5], &c).is_empty());
    }

    #[test]
    fn sampling_is_greedy_at_zero_and_stays_in_the_top_k() {
        let logits = [0.1f32, 2.0, 1.9, -3.0, 1.8];
        let mut rng = Rng::new(1);
        assert_eq!(sample_token(&logits, 0.0, 25, 1.0, &mut rng), 1);
        let mut seen = [0usize; 5];
        for _ in 0..3000 {
            seen[sample_token(&logits, 0.7, 2, 1.0, &mut rng) as usize] += 1;
        }
        assert!(seen[1] > seen[2] && seen[2] > 900 && seen[0] + seen[3] + seen[4] == 0, "{seen:?}");
        // top_p 0.3: the first candidate alone already reaches it.
        for _ in 0..100 {
            assert_eq!(sample_token(&logits, 0.7, 0, 0.3, &mut rng), 1);
        }
    }

    #[test]
    fn references_are_trimmed_to_the_speech() {
        let sr = 24_000;
        let mut a = vec![0f32; sr as usize * 3];
        for (i, v) in a.iter_mut().enumerate().skip(sr as usize).take(sr as usize) {
            *v = 0.2 * (i as f32 * 0.05).sin();
        }
        let t = prepare_reference(&a, 180.0, sr);
        // One second of speech and 0.2 s on each side.
        assert_eq!(t.len(), (1.4 * sr as f32) as usize);
        assert_eq!(prepare_reference(&a, 0.5, sr).len(), sr as usize / 2);
        assert!(prepare_reference(&vec![0f32; 48_000], 180.0, sr).is_empty());
    }

    #[test]
    fn normal_values_have_the_moments_of_a_normal() {
        let v = Rng::new(3).normal(20_000);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05, "{mean} {var}");
        let mut r = Replay::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(r.normal(2), [1.0, 2.0]);
        assert_eq!(r.normal(2), [3.0, 0.0]);
    }
}
