//! Liquid LFM3-Omni (`model_type` "lfm3_omni", LiquidAI/LFM3-Omni-2B): an LFM3 language model
//! (`crate::lfm3`, on the Qwen code path) that reads audio and images as soft tokens and writes
//! text, speech, or both interleaved.
//!
//! ```text
//! audio (16 kHz) -> log-mel -> stride-8 downsampler -> causal LFM2 encoder -> projector
//!                   -> one <|audio|> soft token per 80 ms                         [audio]
//! image -> smart resize, 16x16 patches -> LFM3 vision tower -> 2x2 merge projector
//!                   -> one <|image|> soft token per 32x32 pixels                  [vision]
//! reply: text tokens from the LM head, and speech frames: after <|audio_start|> (speech) or
//!        inside <|mixed_start|> blocks (6 text tokens, then 12 frames, ...), each frame's 12
//!        codes come from the depth decoder, read off the language model's hidden state, and
//!        the frame goes back in as the sum of its codes' embeddings            [this file]
//! codes -> Liquid codec decoder -> 24 kHz audio                                   [codec]
//! ```
//!
//! The depth decoder is a bidirectional 3-layer LFM3 stack over 12 slots, one per codebook:
//! each slot is a slice of a projection of the hidden state plus either the embedding of its
//! code or a mask embedding. The codebooks are revealed over the passes of `maskgit_schedule`
//! (4, 4, 4) in a random order, each pass scoring the slots still masked. A frame in which any
//! code is `audio_eos_code` ends the speech.
//!
//! Not implemented: video input, the streaming encoder and live transcription, audio to codes
//! (the codec's encoder), speech frames in the prompt (earlier spoken turns), batches.

pub mod audio;
pub(crate) mod blocks;
pub mod codec;
pub mod vision;

use crate::kernels::*;
use crate::model::{Lin, Quant, linear, load_lin};
use crate::qwen::{QwenModel, QwenSession};
use crate::qwen_vision::VLin;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use blocks::{Span, Stack, StackConfig};
use std::path::Path;

pub fn is_model_type(m: &str) -> bool {
    m == "lfm3_omni"
}

/// Whether `dir` holds an LFM3-Omni checkpoint.
pub fn is_omni(dir: &Path) -> bool {
    crate::config::model_type(dir).is_ok_and(|m| is_model_type(&m))
}

/// The Omni keys of `config.json`.
#[derive(Clone, Debug)]
pub struct OmniConfig {
    pub audio_token_id: u32,
    pub audio_out_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub audio_start_token_id: u32,
    pub mixed_start_token_id: u32,
    pub text_end_token_id: u32,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub audio_eos_code: u32,
    /// Text tokens, then speech frames, of an interleaved block.
    pub interleaved_n_text: usize,
    pub interleaved_n_audio: usize,
    /// Codebooks revealed by each pass of the depth decoder.
    pub maskgit_schedule: Vec<usize>,
    pub text_hidden: usize,
    pub(crate) depth: StackConfig,
    pub raw: serde_json::Value,
}

impl OmniConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse config.json")?;
        Self::from_json(raw)
    }

    pub fn from_json(raw: serde_json::Value) -> Result<Self> {
        let int = |v: &serde_json::Value, k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("lfm3_omni config: missing {k}"))
        };
        let id = |k: &str| int(&raw, k).map(|v| v as u32);
        let d = &raw["depth_config"];
        let types: Vec<String> = serde_json::from_value(d["layer_types"].clone()).context("depth_config layer_types")?;
        if types.iter().any(|t| t != "full_attention") {
            bail!("LFM3-Omni depth decoder: only full_attention layers are supported");
        }
        let rope = &d["rope_parameters"]["full_attention"];
        if rope["partial_rotary_factor"].as_f64().is_some_and(|f| f != 1.0) || rope["rope_type"].as_str().is_some_and(|t| t != "default") {
            bail!("LFM3-Omni depth decoder: only the default full RoPE is supported");
        }
        let hidden = int(d, "hidden_size")?;
        let heads = int(d, "num_attention_heads")?;
        let num_codebooks = int(&raw, "num_codebooks")?;
        let schedule: Vec<usize> = serde_json::from_value(raw["maskgit_schedule"].clone()).context("maskgit_schedule")?;
        if schedule.iter().sum::<usize>() != num_codebooks || schedule.contains(&0) {
            bail!("lfm3_omni config: maskgit_schedule {schedule:?} does not cover {num_codebooks} codebooks");
        }
        Ok(Self {
            audio_token_id: id("audio_token_id")?,
            audio_out_token_id: id("audio_out_token_id")?,
            image_token_id: id("image_token_id")?,
            video_token_id: id("video_token_id")?,
            audio_start_token_id: id("audio_start_token_id")?,
            mixed_start_token_id: id("mixed_start_token_id")?,
            text_end_token_id: id("text_end_token_id")?,
            num_codebooks,
            codebook_size: int(&raw, "codebook_size")?,
            audio_eos_code: id("audio_eos_code")?,
            interleaved_n_text: int(&raw, "interleaved_n_text")?,
            interleaved_n_audio: int(&raw, "interleaved_n_audio")?,
            maskgit_schedule: schedule,
            text_hidden: int(&raw["text_config"], "hidden_size")?,
            depth: StackConfig {
                hidden,
                conv: vec![false; types.len()],
                heads,
                kv_heads: int(d, "num_key_value_heads")?,
                head_dim: d.get("head_dim").and_then(|x| x.as_u64()).map_or(hidden / heads, |x| x as usize),
                inter: int(d, "intermediate_size")?,
                kernel: 7,
                eps: d.get("norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
                sandwich: d.get("use_sandwich_norm").and_then(|x| x.as_bool()).unwrap_or(true),
                lfm3: true,
                rope_theta: rope["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            },
            raw,
        })
    }

    /// The tokens a text reply never holds (`generate(generation_mode="text")`): the four
    /// placeholders and the two markers that open speech.
    pub fn text_suppressed(&self) -> Vec<u32> {
        let mut v = self.placeholders().to_vec();
        v.extend([self.mixed_start_token_id, self.audio_start_token_id]);
        v
    }

    fn placeholders(&self) -> [u32; 4] {
        [
            self.audio_token_id,
            self.audio_out_token_id,
            self.image_token_id,
            self.video_token_id,
        ]
    }
}

/// [`OmniConfig::text_suppressed`] from the raw config, or nothing for another model.
pub(crate) fn text_suppressed(raw: &serde_json::Value) -> Vec<u32> {
    if raw.get("model_type").and_then(|v| v.as_str()).is_none_or(|m| !is_model_type(m)) {
        return Vec::new();
    }
    [
        "audio_token_id",
        "audio_out_token_id",
        "image_token_id",
        "video_token_id",
        "mixed_start_token_id",
        "audio_start_token_id",
    ]
    .iter()
    .filter_map(|k| raw.get(*k).and_then(|v| v.as_u64()).map(|v| v as u32))
    .collect()
}

/// The placeholder text of an image of `n` tokens / an audio clip of `n` tokens.
pub fn image_placeholder(n: usize) -> String {
    format!("<|vision_start|>{}<|vision_end|>", "<|image|>".repeat(n))
}
pub fn audio_placeholder(n: usize) -> String {
    format!("<|vision_start|>{}<|vision_end|>", "<|audio|>".repeat(n))
}

/// The speech head: the depth decoder and the embeddings of the speech frames.
pub struct Speech<'a> {
    pub cfg: OmniConfig,
    depth_linear: VLin<'a>,
    mask_embedding: Vec<f32>,
    stack: Stack<'a>,
    /// Per codebook: the code embeddings (`[codebook_size, depth hidden]`, also the output
    /// head) and the norm before the head.
    books: Vec<(&'a [u16], Lin<'a>, Vec<f32>)>,
    /// `[num_codebooks * codebook_size, text hidden]`.
    codebook_embedding: &'a [u16],
}

impl<'a> Speech<'a> {
    pub fn load(cfg: OmniConfig, files: &'a ModelFiles) -> Result<Self> {
        let p = "model.depth_decoder.";
        let (dh, th, nq, cs) = (cfg.depth.hidden, cfg.text_hidden, cfg.num_codebooks, cfg.codebook_size);
        let mask_embedding = files.f32(&format!("{p}mask_embedding"))?;
        if mask_embedding.len() != dh {
            bail!("{p}mask_embedding has {} values, expected {dh}", mask_embedding.len());
        }
        let mut books = Vec::with_capacity(nq);
        for i in 0..nq {
            let name = format!("{p}depth_embeddings.{i}.embedding.weight");
            let rows = files.bf16(&name)?;
            if rows.len() != cs * dh {
                bail!("{name}: expected [{cs}, {dh}]");
            }
            let norm = files.f32(&format!("{p}depth_embeddings.{i}.embedding_norm.weight"))?;
            if norm.len() != dh {
                bail!("{p}depth_embeddings.{i}.embedding_norm: expected {dh} values");
            }
            books.push((rows, load_lin(files, &name, cs, dh, Quant::Bf16)?, norm));
        }
        let codebook_embedding = files.bf16("model.codebook_embedding.weight")?;
        if codebook_embedding.len() != nq * cs * th {
            bail!("model.codebook_embedding.weight: expected [{}, {th}]", nq * cs);
        }
        Ok(Self {
            depth_linear: VLin::load(files, &format!("{p}depth_linear"), (nq * dh, th), (nq * dh, th))?,
            mask_embedding,
            stack: Stack::load(files, &format!("{p}model."), cfg.depth.clone())?,
            books,
            codebook_embedding,
            cfg,
        })
    }

    /// The logits `[num_codebooks, codebook_size]` of every codebook of a frame, given the
    /// language model's hidden state, the codes known so far and which slots are still masked.
    pub fn depth_logits(&self, hidden: &[f32], codes: &[u32], masked: &[bool]) -> Vec<f32> {
        let (dh, nq, cs) = (self.cfg.depth.hidden, self.cfg.num_codebooks, self.cfg.codebook_size);
        let mut x = vec![0f32; nq * dh];
        self.depth_linear.forward(hidden, 1, &mut x);
        let mut row = vec![0f32; dh];
        for i in 0..nq {
            let slot = &mut x[i * dh..(i + 1) * dh];
            if masked[i] {
                add_inplace(slot, &self.mask_embedding);
            } else {
                let c = codes[i] as usize;
                bf16_to_f32_slice(&self.books[i].0[c * dh..(c + 1) * dh], &mut row);
                add_inplace(slot, &row);
            }
        }
        self.stack.forward(&mut x, nq, Span::Bidirectional);
        let mut out = vec![0f32; nq * cs];
        for (i, (_, head, norm)) in self.books.iter().enumerate() {
            rmsnorm_f32w(&x[i * dh..(i + 1) * dh], norm, self.cfg.depth.eps, &mut row);
            linear(head, cs, dh, &row, 1, &mut out[i * cs..(i + 1) * cs]);
        }
        out
    }

    /// The codes of one frame from the hidden state before it. `history` holds the earlier
    /// frames the repetition penalty covers (`[frames, num_codebooks]`).
    pub fn sample_frame(&self, hidden: &[f32], history: &[Vec<u32>], opts: &SpeechOptions, rng: &mut Rng) -> Vec<u32> {
        let (nq, cs) = (self.cfg.num_codebooks, self.cfg.codebook_size);
        let mut order: Vec<usize> = (0..nq).collect();
        if !opts.fixed_order {
            for i in (1..nq).rev() {
                order.swap(i, rng.below(i + 1));
            }
        }
        let mut codes = vec![0u32; nq];
        let mut masked = vec![true; nq];
        let mut start = 0;
        for &n in &self.cfg.maskgit_schedule {
            let pick = &order[start..start + n];
            start += n;
            let logits = self.depth_logits(hidden, &codes, &masked);
            for &i in pick {
                let mut l = logits[i * cs..(i + 1) * cs].to_vec();
                if opts.repetition_penalty != 1.0 {
                    for f in history {
                        let v = &mut l[f[i] as usize];
                        // Applied once per distinct code, as transformers' processor does.
                        *v = logits[i * cs + f[i] as usize];
                        *v = if *v < 0.0 {
                            *v * opts.repetition_penalty
                        } else {
                            *v / opts.repetition_penalty
                        };
                    }
                }
                codes[i] = sample(&l, opts.temperature, opts.top_k, rng);
            }
            for &i in pick {
                masked[i] = false;
            }
        }
        codes
    }

    /// The embedding a frame enters the language model with: the sum of its codes' rows.
    pub fn frame_embedding(&self, codes: &[u32]) -> Vec<f32> {
        let (th, cs) = (self.cfg.text_hidden, self.cfg.codebook_size);
        let mut out = vec![0f32; th];
        let mut row = vec![0f32; th];
        for (i, &c) in codes.iter().enumerate() {
            let r = i * cs + c as usize;
            bf16_to_f32_slice(&self.codebook_embedding[r * th..(r + 1) * th], &mut row);
            add_inplace(&mut out, &row);
        }
        out
    }
}

/// A small generator for the speech sampling (xorshift64*).
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 11) as usize % n
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Argmax at temperature 0, else a draw from the softmax of `logits / temperature` over the
/// `top_k` largest (0: all).
fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return crate::sampler::argmax(logits);
    }
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    if top_k > 0 && top_k < idx.len() {
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        idx.truncate(top_k);
    }
    let max = idx.iter().map(|&i| logits[i as usize]).fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = idx.iter().map(|&i| ((logits[i as usize] - max) / temperature).exp()).collect();
    let mut r = rng.uniform() * p.iter().sum::<f32>();
    for (&i, &pi) in idx.iter().zip(&p) {
        r -= pi;
        if r <= 0.0 {
            return i;
        }
    }
    idx[idx.len() - 1]
}

/// What a reply is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Speech only: the reply opens with `<|audio_start|>`.
    Audio,
    /// Text and speech in alternating blocks: the reply opens with `<|mixed_start|>`.
    Interleaved,
}

impl Mode {
    /// The marker to append to the prompt.
    pub fn marker(self, cfg: &OmniConfig) -> u32 {
        match self {
            Mode::Audio => cfg.audio_start_token_id,
            Mode::Interleaved => cfg.mixed_start_token_id,
        }
    }
}

/// Sampling of the speech codes (the text tokens of a spoken reply are greedy).
#[derive(Clone, Debug)]
pub struct SpeechOptions {
    /// 0 picks the most likely code.
    pub temperature: f32,
    /// 0: no top-k.
    pub top_k: usize,
    /// On the codes each codebook emitted in the last `repetition_window` frames (1: off).
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    pub seed: u64,
    /// Generated slots at most (a text token or a speech frame each).
    pub max_new: usize,
    /// Reveal the codebooks in index order instead of a random one (comparisons with the
    /// reference, which draws its order from torch's generator).
    pub fixed_order: bool,
}

impl SpeechOptions {
    /// The checkpoint's `generation_config.json` settings (temperature 0.4, penalty 1.2 over
    /// 64 frames for LFM3-Omni-2B), or those values when the file has none.
    pub fn load(dir: &Path) -> Self {
        let v: serde_json::Value = crate::fsx::read(&dir.join("generation_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        Self {
            temperature: v["audio_temperature"].as_f64().unwrap_or(0.4) as f32,
            top_k: v["audio_top_k"].as_u64().unwrap_or(0) as usize,
            repetition_penalty: v["audio_repetition_penalty"].as_f64().unwrap_or(1.2) as f32,
            repetition_window: v["audio_repetition_window"].as_u64().unwrap_or(64) as usize,
            seed: 0,
            max_new: 1024,
            fixed_order: false,
        }
    }
}

/// A generated reply.
#[derive(Clone, Debug, Default)]
pub struct Reply {
    /// The text tokens, in order (markers and the end-of-turn token included).
    pub text: Vec<u32>,
    /// `[frames, num_codebooks]`, without the end-of-speech frame.
    pub codes: Vec<u32>,
    pub frames: usize,
    /// Whether the model ended the turn itself (not the slot budget).
    pub finished: bool,
}

/// Generates a reply of text and speech after a prompt that has been prefilled into `sess`
/// with the mode's marker as its last token; `hidden` is the final-normed hidden state of
/// that token. `eos` ends the turn in a text slot; `on_text` receives each text token and
/// `on_frame` each speech frame as it is made.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &QwenModel,
    speech: &Speech,
    sess: &mut QwenSession,
    hidden: &[f32],
    mode: Mode,
    opts: &SpeechOptions,
    eos: &[u32],
    mut on_text: impl FnMut(u32),
    mut on_frame: impl FnMut(&[u32]),
) -> Reply {
    #[derive(PartialEq, Clone, Copy)]
    enum M {
        Text,
        Audio,
        Interleaved,
    }
    let cfg = &speech.cfg;
    let (h, vocab) = (model.cfg.hidden_size, model.cfg.vocab_size);
    let mut mode = match mode {
        Mode::Audio => M::Audio,
        Mode::Interleaved => M::Interleaved,
    };
    let mut hidden = hidden[hidden.len() - h..].to_vec();
    let mut rng = Rng::new(opts.seed);
    let mut history: std::collections::VecDeque<Vec<u32>> = Default::default();
    let mut reply = Reply::default();
    let (mut slots_left, mut speaking, mut text_done) = (cfg.interleaved_n_text, false, false);
    for step in 0..opts.max_new {
        let interleaving = mode == M::Interleaved;
        let (token, emb) = if mode == M::Audio || (interleaving && speaking) {
            let hist: Vec<Vec<u32>> = history.iter().cloned().collect();
            let mut codes = speech.sample_frame(&hidden, &hist, opts, &mut rng);
            if codes.contains(&cfg.audio_eos_code) {
                // The end of the speech: the frame goes back in whole, and text follows.
                codes.fill(cfg.audio_eos_code);
                mode = M::Text;
            } else {
                on_frame(&codes);
                reply.codes.extend_from_slice(&codes);
                reply.frames += 1;
                history.push_back(codes.clone());
                if history.len() > opts.repetition_window {
                    history.pop_front();
                }
            }
            (cfg.audio_out_token_id, Some(speech.frame_embedding(&codes)))
        } else {
            let mut logits = vec![0f32; vocab];
            linear(&model.lm_head, vocab, h, &hidden, 1, &mut logits);
            for t in cfg.placeholders() {
                logits[t as usize] = f32::NEG_INFINITY;
            }
            let token = crate::sampler::argmax(&logits);
            reply.text.push(token);
            on_text(token);
            if token == cfg.text_end_token_id {
                if interleaving {
                    (text_done, speaking) = (true, true);
                } else {
                    reply.finished = true;
                }
            } else if eos.contains(&token) {
                reply.finished = true;
            } else if interleaving {
                // Inside a block the mode markers are ordinary text.
            } else if token == cfg.audio_start_token_id {
                mode = M::Audio;
            } else if token == cfg.mixed_start_token_id {
                mode = M::Interleaved;
                (slots_left, speaking, text_done) = (cfg.interleaved_n_text, false, false);
            }
            (token, None)
        };
        if interleaving && !text_done {
            slots_left -= 1;
            if slots_left == 0 {
                speaking = !speaking;
                slots_left = if speaking {
                    cfg.interleaved_n_audio
                } else {
                    cfg.interleaved_n_text
                };
            }
        }
        if reply.finished || step + 1 == opts.max_new {
            break;
        }
        hidden = match &emb {
            Some(e) => model.forward_hidden_mm(sess, &[token], &[(0, e.as_slice())]),
            None => model.forward_hidden_mm(sess, &[token], &[]),
        };
    }
    reply
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sample_is_argmax_and_top_k_bounds_the_draw() {
        let mut rng = Rng::new(3);
        let l = [0.1f32, 2.0, -1.0, 1.9];
        assert_eq!(sample(&l, 0.0, 0, &mut rng), 1);
        for _ in 0..200 {
            assert!(matches!(sample(&l, 1.0, 2, &mut rng), 1 | 3));
        }
    }

    #[test]
    fn placeholders_wrap_in_the_span_pair() {
        assert_eq!(audio_placeholder(2), "<|vision_start|><|audio|><|audio|><|vision_end|>");
        assert_eq!(image_placeholder(1), "<|vision_start|><|image|><|vision_end|>");
    }

    #[test]
    fn suppressed_tokens_only_for_omni() {
        let raw = serde_json::json!({"model_type": "lfm3_omni", "audio_token_id": 5, "image_token_id": 6,
            "video_token_id": 7, "audio_out_token_id": 8, "mixed_start_token_id": 9, "audio_start_token_id": 10});
        assert_eq!(text_suppressed(&raw), vec![5, 8, 6, 7, 9, 10]);
        assert!(text_suppressed(&serde_json::json!({"model_type": "lfm3"})).is_empty());
    }
}
