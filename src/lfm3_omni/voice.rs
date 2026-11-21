//! Spoken replies of LFM3-Omni through the library: the speech head and the codec next to a
//! loaded language model ([`Voice`]), and a whole reply from a prompt ([`Voice::reply`]).
//! [`crate::engine::Engine::speak_omni`] runs it on a chat engine's own model (the server's
//! chat completions with `modalities: ["text", "audio"]` and `/v1/audio/speech`, the C API's
//! `oxi_engine_speak_omni`); `oxinfer --model LiquidAI/LFM3-Omni-2B --speak TEXT -o out.wav` and
//! `--audio question.wav -o reply.wav` are the command line's.

use super::codec::{CodecConfig, CodecDecoder};
use super::{Mode, OmniConfig, Speech, SpeechOptions};
use crate::chat::{ChatFormat, Lfm3Chat};
use crate::qwen::QwenModel;
use crate::safetensors::ModelFiles;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The voices of the text-to-speech system prompt (the model card's).
pub const VOICES: [&str; 2] = ["female", "male"];

/// The system prompt of a spoken reply without one of its own: text to speech ([`Mode::Audio`])
/// in American English with `voice` (`female`, the default, or `male`), or a voice chat
/// ([`Mode::Interleaved`]: text and speech interleaved).
pub fn default_system(mode: Mode, voice: Option<&str>) -> Result<String> {
    Ok(match mode {
        Mode::Audio => {
            let v = voice.unwrap_or("female");
            if !VOICES.contains(&v) {
                bail!("voice {v:?}: LFM3-Omni's voices are female and male");
            }
            format!("Perform TTS in American English using a {v} voice.")
        }
        Mode::Interleaved => "Respond with interleaved text and audio.".to_string(),
    })
}

/// How to make one spoken reply.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SpokenOptions {
    pub mode: Mode,
    /// The system prompt; None: [`default_system`] (an empty one: no system turn).
    pub system: Option<String>,
    /// `female` or `male` for [`Mode::Audio`] without a system prompt of its own.
    pub voice: Option<String>,
    /// The sampling of the speech codes and the reply's length; None: the checkpoint's
    /// (`generation_config.json`), seed 0, 1024 slots.
    pub speech: Option<SpeechOptions>,
}

impl SpokenOptions {
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            system: None,
            voice: None,
            speech: None,
        }
    }
}

/// A spoken reply.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct SpokenReply {
    /// The reply's text (empty for text to speech).
    pub text: String,
    /// Mono samples in `[-1, 1]` at `sample_rate` (24 kHz).
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Speech frames of 80 ms.
    pub frames: usize,
    pub prompt_tokens: usize,
    /// Generated slots: text tokens and speech frames.
    pub tokens: usize,
    /// The model ended the turn itself (not the slot budget).
    pub finished: bool,
}

impl SpokenReply {
    /// The audio as a 16-bit PCM WAV file.
    pub fn to_wav(&self) -> Vec<u8> {
        crate::tts::wav_bytes(&self.samples, self.sample_rate)
    }
}

/// LFM3-Omni's speech head (from the model's own files) and its codec (`audio_tokenizer/`).
pub struct Voice {
    // Declared before the codec's files it borrows.
    codec: CodecDecoder<'static>,
    pub speech: Speech<'static>,
    defaults: SpeechOptions,
    eos: Vec<u32>,
    _codec_files: Box<ModelFiles>,
}

impl Voice {
    /// Loads the speech head from `files` (the checkpoint's tensors, as the language model has
    /// them) and the codec from `dir/audio_tokenizer`.
    pub fn load(dir: &Path, files: &'static ModelFiles) -> Result<Self> {
        let speech = Speech::load(OmniConfig::load(dir)?, files)?;
        let codec_dir = dir.join("audio_tokenizer");
        let codec_files = Box::new(ModelFiles::open(&codec_dir, true).with_context(|| format!("the codec in {}", codec_dir.display()))?);
        // SAFETY: the box's heap block does not move, and the decoder borrowing it is declared
        // (and so dropped) before it.
        let cf: &'static ModelFiles = unsafe { &*(codec_files.as_ref() as *const ModelFiles) };
        let codec = CodecDecoder::load(CodecConfig::load(&codec_dir)?, cf)?;
        Ok(Self {
            codec,
            speech,
            defaults: SpeechOptions::load(dir),
            eos: crate::config::load_eos_ids(dir),
            _codec_files: codec_files,
        })
    }

    /// The codec's output rate.
    pub fn sample_rate(&self) -> u32 {
        self.codec.cfg.sample_rate as u32
    }

    /// The checkpoint's speech sampling settings.
    pub fn defaults(&self) -> &SpeechOptions {
        &self.defaults
    }

    /// One spoken reply to `user` (its attachments' placeholders `mm_text` before it, their
    /// features `feats`) from `model`, on a fresh session; `on_text` receives the reply's text as
    /// it grows.
    #[allow(clippy::too_many_arguments)]
    pub fn reply(
        &self,
        model: &QwenModel,
        tok: &Tokenizer,
        user: &str,
        mm_text: &str,
        feats: &[crate::mm::MmFeature],
        o: &SpokenOptions,
        prefill_chunk: usize,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<SpokenReply> {
        let system = match &o.system {
            Some(s) => s.clone(),
            None => default_system(o.mode, o.voice.as_deref())?,
        };
        let fmt = Lfm3Chat;
        let mut text = fmt.conversation_start(Some(system.as_str()).filter(|s| !s.is_empty()), false);
        text.push_str(&fmt.user_turn(&format!("{mm_text}{user}"), false));
        let mut ids = tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        ids.push(o.mode.marker(&self.speech.cfg));
        let opts = o.speech.clone().unwrap_or_else(|| self.defaults.clone());
        let mut sess = model.new_session();
        let overrides = crate::mm::bind(model, &mut sess, &ids, feats)?;
        let hs = crate::generate::prefill(model, &mut sess, &ids, prefill_chunk.max(1), &overrides);
        let (mut text_ids, mut shown, mut out) = (Vec::new(), 0usize, String::new());
        let reply = super::generate(
            model,
            &self.speech,
            &mut sess,
            &hs,
            o.mode,
            &opts,
            &self.eos,
            |t| {
                if self.eos.contains(&t) {
                    return;
                }
                text_ids.push(t);
                if let Ok(s) = tok.decode(&text_ids, true)
                    && !s.ends_with('\u{fffd}')
                    && s.len() > shown
                {
                    on_text(&s[shown..]);
                    out.push_str(&s[shown..]);
                    shown = s.len();
                }
            },
            |_| {},
        );
        let samples = if reply.frames == 0 {
            Vec::new()
        } else {
            self.codec.decode(&reply.codes, reply.frames)?
        };
        Ok(SpokenReply {
            text: out.trim().to_string(),
            samples,
            sample_rate: self.sample_rate(),
            frames: reply.frames,
            prompt_tokens: ids.len(),
            tokens: reply.text.len() + reply.frames,
            finished: reply.finished,
        })
    }
}
