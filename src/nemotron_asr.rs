//! The audio side of NVIDIA Nemotron 3.5 Transcribe (`nemo_speechlm`,
//! nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B), as transformers'
//! `Nemotron3_5TranscribeForConditionalGeneration` runs NeMo's `AudioPerceptionModule`, in front
//! of the Nemotron-H language model of [`crate::nemotron`]:
//!
//! - Features (NeMo's `AudioToMelSpectrogramPreprocessor`): pre-emphasis 0.97, a centered STFT
//!   (512 points, a 25 ms Hann window every 10 ms), 128 Slaney mel bins of the power spectrum,
//!   `ln(x + 2^-24)`. The window and the filter bank are the checkpoint's own tensors (kept in
//!   bf16), or computed and rounded to bf16 when it has none. `samples / 160` frames are valid.
//! - The speech recognition encoder: the features normalized per mel bin over the recording,
//!   eight frames stacked per position (80 ms) and projected; 32 pre-norm transformer layers
//!   (LayerNorm on the queries and keys of each head, rotary positions on half of each head,
//!   GELU MLPs). A recording longer than 500 positions (40 s) is encoded in chunks of 500 that
//!   see 50 positions of context on each side.
//! - The speaker encoder (Nemotron 3 Diarization, a streaming Sortformer): the raw features
//!   through a smaller encoder of the same kind (rotary positions on the whole head), a
//!   projection, a sub-pixel convolution back to one frame per 10 ms and a two-layer head give
//!   the activity of up to eight speakers, in order of arrival. Chunks after the first also
//!   attend to the frames kept from the earlier ones: a FIFO of the latest, and the
//!   arrival-order speaker cache the FIFO overflows into, compressed to its best-scored frames
//!   per speaker.
//! - Each position's speaker activities (mean probability over its eight frames, thresholded)
//!   go through a LayerNorm and select fixed sinusoidal speaker embeddings, added to the
//!   LayerNorm'd encoder output; a linear projection gives one soft token per position, which
//!   takes the place of an `<|audio|>` token of the prompt.
//!
//! The prompt is NeMo's training prompt through the chat template: an empty system turn, the
//! instruction, a space and the audio tokens in the user turn, and a reply that opens with a
//! closed `<think></think>` block. Decoding is greedy (or sampled at a temperature) up to
//! `<|im_end|>`.
//!
//! Not here: the word timestamp head (`ctc_timestamp.*`, a CTC classifier over the encoder
//! states that NeMo force-aligns the transcript on), the multi-token prediction head, and
//! batching of several recordings in one pass.

use crate::bert::gelu_erf;
use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, attention_masked, linear, load_lin};
use crate::nemotron::{NemotronConfig, NemotronModel};
use crate::safetensors::ModelFiles;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

/// The instruction of a plain transcription (the processor's `default_transcription_prompt`).
pub const TRANSCRIBE_PROMPT: &str = "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.";

/// The instruction of a translation into `language` (the model card's).
pub fn translate_prompt(language: &str) -> String {
    format!("Please translate the given speech to {language}.")
}

/// Whether `dir` holds a NeMo speech LM checkpoint (Nemotron 3.5 Transcribe).
pub fn is_speech_lm(dir: &Path) -> bool {
    crate::config::model_type(dir).is_ok_and(|t| t == crate::nemotron::SPEECHLM)
}

const LN_EPS: f32 = 1e-5;
/// Added to the mel energies before the logarithm.
const LOG_GUARD: f32 = 1.0 / 16_777_216.0;

/// A NeMo `TransformerEncoder`.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// The share of each head that is rotated.
    pub rotary_fraction: f64,
    pub rope_theta: f64,
    /// LayerNorm on the queries and keys of each head.
    pub qk_norm: bool,
}

/// The settings of the audio side (`config.json` of a `nemo_speechlm` checkpoint).
#[derive(Clone, Debug)]
pub struct SpeechConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub preemphasis: f32,
    /// Spectrogram frames per encoder position.
    pub subsampling_factor: usize,
    pub asr: EncoderConfig,
    pub diar: EncoderConfig,
    /// Width of the diarization head.
    pub diar_head_size: usize,
    pub num_speakers: usize,
    /// Encoder positions per chunk, and the context a chunk sees on each side.
    pub chunk_length: usize,
    pub chunk_left_context: usize,
    pub chunk_right_context: usize,
    /// The diarization FIFO and what it moves to the speaker cache when it overflows.
    pub fifo_length: usize,
    pub speaker_cache_update_period: usize,
    pub speaker_cache_length: usize,
    pub silence_frames_per_speaker: usize,
    pub prediction_score_threshold: f32,
    pub latest_frames_score_boost: f32,
    pub strong_boost_rate: f64,
    pub weak_boost_rate: f64,
    pub min_positive_scores_rate: f64,
    pub speaker_activity_threshold: f32,
    pub audio_token: String,
    pub audio_token_id: u32,
}

impl SpeechConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        Self::from_json(&raw)
    }

    /// From transformers' keys (`audio_config`, written into the release's `config.json`), or
    /// from NeMo's (`pe_encoder_config`, with the defaults of its online inference).
    pub fn from_json(raw: &Value) -> Result<Self> {
        let num = |v: &Value, k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize);
        let float = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64);
        let pre = &raw["perception"]["preprocessor"];
        let sample_rate = num(pre, "sample_rate").unwrap_or(16_000);
        let secs = |k: &str, d: f64| (float(pre, k).unwrap_or(d) * sample_rate as f64).round() as usize;
        let audio = &raw["audio_config"];
        let nemo = &raw["pe_encoder_config"];
        let mut c = if audio.is_object() {
            let enc = |v: &Value, qk_norm: bool| -> Result<EncoderConfig> {
                let need = |k: &str| num(v, k).with_context(|| format!("audio config has no {k}"));
                let rope = &v["rope_parameters"];
                Ok(EncoderConfig {
                    hidden_size: need("hidden_size")?,
                    intermediate_size: need("intermediate_size")?,
                    num_hidden_layers: need("num_hidden_layers")?,
                    num_attention_heads: need("num_attention_heads")?,
                    rotary_fraction: float(rope, "partial_rotary_factor").unwrap_or(1.0),
                    rope_theta: float(rope, "rope_theta").unwrap_or(10000.0),
                    qk_norm,
                })
            };
            let d = &audio["diarization_config"];
            let (head, st) = (&d["head_config"], &d["streaming_config"]);
            let asr = &audio["asr_config"];
            Self {
                sample_rate,
                n_fft: num(pre, "n_fft").unwrap_or(512),
                hop_length: secs("window_stride", 0.01),
                win_length: secs("window_size", 0.025),
                n_mels: num(asr, "num_mel_bins").unwrap_or(128),
                preemphasis: 0.97,
                subsampling_factor: num(asr, "subsampling_factor").unwrap_or(8),
                asr: enc(asr, true)?,
                diar: enc(&d["audio_config"], false)?,
                diar_head_size: num(head, "hidden_size").context("diarization head_config has no hidden_size")?,
                num_speakers: num(head, "num_speakers").unwrap_or(8),
                chunk_length: num(d, "chunk_length").unwrap_or(500),
                chunk_left_context: num(d, "chunk_left_context").unwrap_or(50),
                chunk_right_context: num(d, "chunk_right_context").unwrap_or(50),
                fifo_length: num(d, "fifo_length").unwrap_or(40),
                speaker_cache_update_period: num(d, "speaker_cache_update_period").unwrap_or(300),
                speaker_cache_length: num(st, "speaker_cache_length").unwrap_or(188),
                silence_frames_per_speaker: num(st, "speaker_cache_silence_frames_per_speaker").unwrap_or(1),
                prediction_score_threshold: float(st, "prediction_score_threshold").unwrap_or(0.25) as f32,
                latest_frames_score_boost: float(st, "latest_frames_score_boost").unwrap_or(0.05) as f32,
                strong_boost_rate: float(st, "strong_boost_rate").unwrap_or(0.75),
                weak_boost_rate: float(st, "weak_boost_rate").unwrap_or(1.5),
                min_positive_scores_rate: float(st, "min_positive_scores_rate").unwrap_or(0.5),
                speaker_activity_threshold: float(audio, "speaker_activity_threshold").unwrap_or(0.5) as f32,
                audio_token: String::new(),
                audio_token_id: 0,
            }
        } else if nemo.is_object() {
            let enc = |v: &Value, fraction: f64| -> Result<EncoderConfig> {
                let d = num(v, "d_model").context("NeMo encoder config has no d_model")?;
                Ok(EncoderConfig {
                    hidden_size: d,
                    intermediate_size: (d as f64 * float(v, "ff_expansion").unwrap_or(4.0)) as usize,
                    num_hidden_layers: num(v, "n_layers").context("NeMo encoder config has no n_layers")?,
                    num_attention_heads: num(v, "n_heads").context("NeMo encoder config has no n_heads")?,
                    rotary_fraction: float(v, "rotary_fraction").unwrap_or(fraction),
                    rope_theta: float(v, "rope_base").unwrap_or(10000.0),
                    qk_norm: v.get("qk_norm").and_then(Value::as_bool).unwrap_or(false),
                })
            };
            let asr = &nemo["asr_encoder_cfg"];
            let dm = &nemo["diarization_model_cfg"];
            let sf = &dm["sortformer_modules"];
            Self {
                sample_rate,
                n_fft: num(pre, "n_fft").unwrap_or(512),
                hop_length: secs("window_stride", 0.01),
                win_length: secs("window_size", 0.025),
                n_mels: num(asr, "feat_in").unwrap_or(128),
                preemphasis: 0.97,
                subsampling_factor: num(asr, "subsampling_factor").unwrap_or(8),
                asr: enc(asr, 1.0)?,
                diar: enc(&dm["encoder"], 1.0)?,
                diar_head_size: num(sf, "tf_d_model").context("sortformer_modules has no tf_d_model")?,
                num_speakers: num(sf, "num_spks").unwrap_or(8),
                chunk_length: num(nemo, "online_inference_length").unwrap_or(500),
                chunk_left_context: num(nemo, "chunk_left_context").unwrap_or(50),
                chunk_right_context: num(nemo, "chunk_right_context").unwrap_or(50),
                fifo_length: num(nemo, "diar_fifo_len").unwrap_or(40),
                speaker_cache_update_period: num(nemo, "diar_spkcache_update_period").unwrap_or(300),
                speaker_cache_length: num(nemo, "diar_spkcache_len").unwrap_or(188),
                silence_frames_per_speaker: num(sf, "spkcache_sil_frames_per_spk").unwrap_or(1),
                prediction_score_threshold: float(sf, "pred_score_threshold").unwrap_or(0.25) as f32,
                latest_frames_score_boost: float(sf, "scores_boost_latest").unwrap_or(0.05) as f32,
                strong_boost_rate: float(sf, "strong_boost_rate").unwrap_or(0.75),
                weak_boost_rate: float(sf, "weak_boost_rate").unwrap_or(1.5),
                min_positive_scores_rate: float(sf, "min_pos_scores_rate").unwrap_or(0.5),
                speaker_activity_threshold: float(nemo, "speaker_activity_threshold").unwrap_or(0.5) as f32,
                audio_token: String::new(),
                audio_token_id: 0,
            }
        } else {
            bail!("config.json has neither audio_config nor pe_encoder_config: not a NeMo speech LM");
        };
        c.audio_token = raw
            .get("audio_locator_tag")
            .and_then(Value::as_str)
            .unwrap_or("<|audio|>")
            .to_string();
        c.audio_token_id = raw.get("audio_token_id").and_then(Value::as_u64).unwrap_or(0) as u32;
        for (what, e) in [("speech recognition", &c.asr), ("diarization", &c.diar)] {
            let hd = e.hidden_size / e.num_attention_heads.max(1);
            if e.num_attention_heads == 0 || hd * e.num_attention_heads != e.hidden_size || !rot_dims(e).is_multiple_of(2) {
                bail!("{what} encoder: {} heads over a width of {}", e.num_attention_heads, e.hidden_size);
            }
        }
        if c.chunk_length == 0 || c.subsampling_factor == 0 || c.num_speakers == 0 || c.speaker_cache_length < c.num_speakers {
            bail!("unusable audio chunking in config.json");
        }
        Ok(c)
    }
}

/// Rotated dimensions of each head.
fn rot_dims(e: &EncoderConfig) -> usize {
    ((e.hidden_size / e.num_attention_heads) as f64 * e.rotary_fraction) as usize
}

/// A matrix: float32 tensors exactly (a tiny reference checkpoint), others as bf16.
fn matrix<'a>(files: &'a ModelFiles, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
    if let Some(info) = files.info(full)
        && info.dtype == "F32"
        && files.ggml(full).is_none()
    {
        if info.shape.iter().product::<usize>() != n_out * n_in {
            bail!("tensor {full} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
        }
        files.note_use(full, true, false);
        return Lin::from_gguf(crate::ggml::GgmlType::F32, files.bytes(full)?, n_out, n_in);
    }
    load_lin(files, full, n_out, n_in, Quant::Bf16)
}

fn vector(files: &ModelFiles, full: &str, n: usize) -> Result<Vec<f32>> {
    let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
    if info.shape.iter().product::<usize>() != n {
        bail!("tensor {full} has shape {:?}, expected {n} values", info.shape);
    }
    files.f32(full)
}

/// LayerNorm weight and bias.
type Norm = (Vec<f32>, Vec<f32>);

fn norm(files: &ModelFiles, p: &str, n: usize) -> Result<Norm> {
    Ok((vector(files, &format!("{p}weight"), n)?, vector(files, &format!("{p}bias"), n)?))
}

/// LayerNorm over each row of `x` (`n` wide), in place.
fn layer_norm_rows(x: &mut [f32], n: usize, (w, b): &Norm) {
    for_chunks(x, n, x.len() > 64 * n, |_, r| {
        let mean = r.iter().sum::<f32>() / n as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for ((v, &wi), &bi) in r.iter_mut().zip(w).zip(b) {
            *v = (*v - mean) * inv * wi + bi;
        }
    });
}

/// A small float32 linear layer with a bias.
struct Dense {
    w: Vec<f32>,
    b: Vec<f32>,
    n_in: usize,
}

impl Dense {
    fn load(files: &ModelFiles, p: &str, n_out: usize, n_in: usize) -> Result<Self> {
        Ok(Self {
            w: vector(files, &format!("{p}weight"), n_out * n_in)?,
            b: vector(files, &format!("{p}bias"), n_out)?,
            n_in,
        })
    }
    fn n_out(&self) -> usize {
        self.b.len()
    }
    /// `y = W x + b` over the rows of `x`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (ni, no) = (self.n_in, self.n_out());
        let t = x.len() / ni;
        let mut y = vec![0f32; t * no];
        for_chunks(&mut y, no, t > 1, |ti, row| {
            let xr = &x[ti * ni..(ti + 1) * ni];
            for (o, v) in row.iter_mut().enumerate() {
                *v = self.b[o] + dot(&self.w[o * ni..(o + 1) * ni], xr);
            }
        });
        y
    }
}

struct EncLayer<'a> {
    ln1: Norm,
    /// `[3 d, d]`: query, key and value rows.
    qkv: Lin<'a>,
    qk_norm: Option<(Norm, Norm)>,
    out: Lin<'a>,
    out_b: Vec<f32>,
    ln2: Norm,
    fc1: Lin<'a>,
    fc1_b: Vec<f32>,
    fc2: Lin<'a>,
    fc2_b: Vec<f32>,
}

/// A NeMo `TransformerEncoder` (feature stacking, pre-norm layers, rotary self-attention).
struct Encoder<'a> {
    cfg: EncoderConfig,
    /// `[d, subsampling * mels]`, None for an encoder that is given positions.
    embed: Option<Lin<'a>>,
    in_norm: Norm,
    layers: Vec<EncLayer<'a>>,
    out_norm: Norm,
    inv_freq: Vec<f32>,
}

impl<'a> Encoder<'a> {
    /// The encoder under `p` (`perception.encoder.asr_encoder.`), in NeMo's names.
    fn load(files: &'a ModelFiles, p: &str, cfg: &EncoderConfig, stacked: Option<usize>) -> Result<Self> {
        let (d, inter, nh) = (cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads);
        let hd = d / nh;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{p}layers.{l}.");
            let qk_norm = if cfg.qk_norm {
                Some((
                    norm(files, &format!("{lp}attn.q_norm."), hd)?,
                    norm(files, &format!("{lp}attn.k_norm."), hd)?,
                ))
            } else {
                None
            };
            layers.push(EncLayer {
                ln1: norm(files, &format!("{lp}norm1."), d)?,
                qkv: matrix(files, &format!("{lp}attn.w_qkv.weight"), 3 * d, d)?,
                qk_norm,
                out: matrix(files, &format!("{lp}attn.out_proj.weight"), d, d)?,
                out_b: vector(files, &format!("{lp}attn.out_proj.bias"), d)?,
                ln2: norm(files, &format!("{lp}norm2."), d)?,
                fc1: matrix(files, &format!("{lp}ffn.net.0.weight"), inter, d)?,
                fc1_b: vector(files, &format!("{lp}ffn.net.0.bias"), inter)?,
                fc2: matrix(files, &format!("{lp}ffn.net.3.weight"), d, inter)?,
                fc2_b: vector(files, &format!("{lp}ffn.net.3.bias"), d)?,
            });
        }
        Ok(Self {
            embed: match stacked {
                Some(n) => Some(matrix(files, &format!("{p}pre_encode.proj.weight"), d, n)?),
                None => None,
            },
            in_norm: norm(files, &format!("{p}embed_norm."), d)?,
            layers,
            out_norm: norm(files, &format!("{p}final_norm."), d)?,
            inv_freq: rope_inv_freq(rot_dims(cfg), cfg.rope_theta, 1.0),
            cfg: cfg.clone(),
        })
    }

    /// The stacked features `[t, subsampling * mels]` projected to `[t, d]`.
    fn embed(&self, stacked: &[f32], n_in: usize) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        let t = stacked.len() / n_in;
        let mut y = vec![0f32; t * d];
        linear(
            self.embed.as_ref().expect("an encoder with a projection"),
            d,
            n_in,
            stacked,
            t,
            &mut y,
        );
        y
    }

    /// Runs the layers over `x` (`[t, d]`, positions 0..t), the first `valid` positions being
    /// the keys every position attends to (padding is on the right).
    fn forward(&self, mut x: Vec<f32>, valid: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (d, nh, inter) = (c.hidden_size, c.num_attention_heads, c.intermediate_size);
        let hd = d / nh;
        let t = x.len() / d;
        let rot = rot_dims(c);
        let positions: Vec<usize> = (0..t).collect();
        let (cos, sin) = rope_tables(&positions, &self.inv_freq);
        let half = rot / 2;
        let scale = (hd as f32).powf(-0.5);
        layer_norm_rows(&mut x, d, &self.in_norm);
        let mut hn = vec![0f32; t * d];
        let mut qkv = vec![0f32; t * 3 * d];
        let mut o = vec![0f32; t * d];
        let mut a = vec![0f32; t * inter];
        for layer in &self.layers {
            hn.copy_from_slice(&x);
            layer_norm_rows(&mut hn, d, &layer.ln1);
            linear(&layer.qkv, 3 * d, d, &hn, t, &mut qkv);
            let (mut q, mut k, mut v) = (vec![0f32; t * d], vec![0f32; valid * d], vec![0f32; valid * d]);
            for ti in 0..t {
                let row = &qkv[ti * 3 * d..(ti + 1) * 3 * d];
                q[ti * d..(ti + 1) * d].copy_from_slice(&row[..d]);
                if ti < valid {
                    k[ti * d..(ti + 1) * d].copy_from_slice(&row[d..2 * d]);
                    v[ti * d..(ti + 1) * d].copy_from_slice(&row[2 * d..]);
                }
            }
            if let Some((qn, kn)) = &layer.qk_norm {
                layer_norm_rows(&mut q, hd, qn);
                layer_norm_rows(&mut k, hd, kn);
            }
            if half > 0 {
                let rotate = |x: &mut [f32]| {
                    for_chunks(x, hd, true, |i, head| {
                        let ti = i / nh;
                        apply_rope(&mut head[..rot], &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                    });
                };
                rotate(&mut q);
                rotate(&mut k);
            }
            scale_inplace(&mut q, scale);
            let mut cache = LayerCache::new(nh, hd);
            cache.push(&k, &v, valid);
            let mut att = vec![0f32; t * d];
            attention_masked(&q, &cache, nh, t, 0, None, false, &mut att);
            linear(&layer.out, d, d, &att, t, &mut o);
            for_chunks(&mut x, d, t > 1, |ti, row| {
                for ((xv, &ov), &b) in row.iter_mut().zip(&o[ti * d..(ti + 1) * d]).zip(&layer.out_b) {
                    *xv += ov + b;
                }
            });
            hn.copy_from_slice(&x);
            layer_norm_rows(&mut hn, d, &layer.ln2);
            linear(&layer.fc1, inter, d, &hn, t, &mut a);
            for_chunks(&mut a, inter, t > 1, |_, row| {
                for (v, &b) in row.iter_mut().zip(&layer.fc1_b) {
                    *v = gelu_erf(*v + b);
                }
            });
            linear(&layer.fc2, d, inter, &a, t, &mut o);
            for_chunks(&mut x, d, t > 1, |ti, row| {
                for ((xv, &ov), &b) in row.iter_mut().zip(&o[ti * d..(ti + 1) * d]).zip(&layer.fc2_b) {
                    *xv += ov + b;
                }
            });
        }
        layer_norm_rows(&mut x, d, &self.out_norm);
        x
    }
}

/// The streaming state of the speaker encoder: the arrival-order speaker cache and the FIFO of
/// the most recent positions, which every chunk attends to.
struct SpeakerCache<'c> {
    cfg: &'c SpeechConfig,
    d: usize,
    embeds: Vec<f32>,
    probs: Vec<f32>,
    fifo: Vec<f32>,
    compressed: bool,
}

impl SpeakerCache<'_> {
    fn cache_frames(&self) -> usize {
        self.embeds.len() / self.d
    }
    fn fifo_frames(&self) -> usize {
        self.fifo.len() / self.d
    }

    /// Pushes a processed chunk to the FIFO, moving its oldest positions to the speaker cache
    /// when it overflows. `input`: the encoder input of the step (the cached positions, the
    /// look-back, the chunk, its look-ahead); `probs`: the speaker probabilities of each of
    /// those positions (zero on padding).
    fn update(&mut self, input: &[f32], probs: &[f32], silence: &[f32], chunk: usize, lookback: usize) {
        let (d, s) = (self.d, self.cfg.num_speakers);
        let (nc, nf) = (self.cache_frames(), self.fifo_frames());
        let start = nc + nf + lookback;
        let mut fifo = std::mem::take(&mut self.fifo);
        fifo.extend_from_slice(&input[start * d..(start + chunk) * d]);
        let len = fifo.len() / d;
        let popped = if len <= self.cfg.fifo_length {
            0
        } else {
            self.cfg.speaker_cache_update_period.max(len - self.cfg.fifo_length).min(len)
        };
        if popped > 0 {
            // The look-back positions sit between the FIFO's and the chunk's, and are skipped.
            let mut fifo_probs = probs[nc * s..(nc + nf) * s].to_vec();
            fifo_probs.extend_from_slice(&probs[start * s..(start + chunk) * s]);
            // An uncompressed cache still holds plain positions, re-estimated by this step; a
            // compressed one is out of order and keeps the probabilities stored with it.
            let mut cache_probs = if self.compressed {
                std::mem::take(&mut self.probs)
            } else {
                probs[..nc * s].to_vec()
            };
            cache_probs.extend_from_slice(&fifo_probs[..popped * s]);
            let mut cache_embeds = std::mem::take(&mut self.embeds);
            cache_embeds.extend_from_slice(&fifo[..popped * d]);
            fifo.drain(..popped * d);
            if cache_embeds.len() / d > self.cfg.speaker_cache_length {
                (cache_embeds, cache_probs) = self.compress(&cache_embeds, &cache_probs, silence);
                self.compressed = true;
            }
            self.embeds = cache_embeds;
            self.probs = cache_probs;
        }
        self.fifo = fifo;
    }

    /// Adds `boost` to the `k` best-scored positions of each speaker.
    fn boost(scores: &mut [f32], frames: usize, s: usize, k: usize, boost: f32) {
        for sp in 0..s {
            let mut idx: Vec<usize> = (0..frames).collect();
            idx.sort_by(|&a, &b| scores[b * s + sp].total_cmp(&scores[a * s + sp]).then(a.cmp(&b)));
            for &f in idx.iter().take(k) {
                scores[f * s + sp] += boost;
            }
        }
    }

    /// Keeps the `speaker_cache_length` most important positions, grouped by speaker and in
    /// their order within a speaker; the reserved slots of each speaker hold `silence`.
    fn compress(&self, embeds: &[f32], probs: &[f32], silence: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let c = self.cfg;
        let (d, s, keep, sil) = (self.d, c.num_speakers, c.speaker_cache_length, c.silence_frames_per_speaker);
        let frames = probs.len() / s;
        let budget = (keep / s).saturating_sub(sil) as f64;
        let min_positive = (budget * c.min_positive_scores_rate).floor() as usize;
        let strong = (budget * c.strong_boost_rate).floor() as usize;
        let weak = (budget * c.weak_boost_rate).floor() as usize;
        let thr = c.prediction_score_threshold;
        let ln_half = 0.5f32.ln();
        // Per position and speaker: the log-odds of the speaker, plus the log-probability that
        // the others are silent; only where the speaker talks, and above zero once enough are.
        let mut scores = vec![f32::NEG_INFINITY; frames * s];
        let mut positive = vec![0usize; s];
        for f in 0..frames {
            let p = &probs[f * s..(f + 1) * s];
            let silent: f32 = p.iter().map(|&x| (1.0 - x).max(thr).ln()).sum();
            for sp in 0..s {
                if p[sp] > 0.5 {
                    let v = p[sp].max(thr).ln() - (1.0 - p[sp]).max(thr).ln() + silent - ln_half;
                    scores[f * s + sp] = v;
                    positive[sp] += (v > 0.0) as usize;
                }
            }
        }
        for f in 0..frames {
            for sp in 0..s {
                let v = &mut scores[f * s + sp];
                if v.is_finite() && *v <= 0.0 && positive[sp] >= min_positive {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        // The positions just popped from the FIFO come after the cache's.
        for v in scores[keep.min(frames) * s..].iter_mut() {
            *v += c.latest_frames_score_boost;
        }
        Self::boost(&mut scores, frames, s, strong, -2.0 * ln_half);
        Self::boost(&mut scores, frames, s, weak, -ln_half);
        // Speaker by speaker, then the reserved silence slots (always kept); the best `keep`.
        let scored = frames + sil;
        let mut flat: Vec<(f32, usize)> = Vec::with_capacity(s * scored);
        for sp in 0..s {
            flat.extend((0..frames).map(|f| (scores[f * s + sp], sp * scored + f)));
            flat.extend((frames..scored).map(|f| (f32::INFINITY, sp * scored + f)));
        }
        flat.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let sentinel = scored * s;
        let mut chosen: Vec<usize> = flat[..keep.min(flat.len())]
            .iter()
            .map(|&(v, i)| if v == f32::NEG_INFINITY { sentinel } else { i })
            .collect();
        chosen.sort_unstable();
        let (mut e, mut p) = (Vec::with_capacity(keep * d), Vec::with_capacity(keep * s));
        for i in chosen {
            let f = if i == sentinel { frames } else { (i % scored).min(frames) };
            if f < frames {
                e.extend_from_slice(&embeds[f * d..(f + 1) * d]);
                p.extend_from_slice(&probs[f * s..(f + 1) * s]);
            } else {
                e.extend_from_slice(silence);
                p.extend(std::iter::repeat_n(0f32, s));
            }
        }
        (e, p)
    }
}

/// The speaker encoder (Nemotron 3 Diarization).
struct Diarizer<'a> {
    encoder: Encoder<'a>,
    proj: Dense,
    /// The sub-pixel convolution, `[factor * w][3 * w]` (tap by tap), and its bias.
    up_w: Vec<f32>,
    up_b: Vec<f32>,
    dense: Dense,
    out: Dense,
    silence: Vec<f32>,
}

impl<'a> Diarizer<'a> {
    fn load(files: &'a ModelFiles, cfg: &SpeechConfig) -> Result<Self> {
        let p = "perception.encoder.diarization_model.";
        let sp = format!("{p}sortformer_modules.");
        let (d, w, f) = (cfg.diar.hidden_size, cfg.diar_head_size, cfg.subsampling_factor);
        // Conv1d `[out, in, 3]` as `[out][tap * in]`.
        let raw = vector(files, &format!("{sp}subpixel_upsample.weight"), f * w * w * 3)?;
        let mut up_w = vec![0f32; f * w * 3 * w];
        for o in 0..f * w {
            for c in 0..w {
                for j in 0..3 {
                    up_w[(o * 3 + j) * w + c] = raw[(o * w + c) * 3 + j];
                }
            }
        }
        Ok(Self {
            encoder: Encoder::load(files, &format!("{p}encoder."), &cfg.diar, Some(f * cfg.n_mels))?,
            proj: Dense::load(files, &format!("{sp}encoder_proj."), w, d)?,
            up_w,
            up_b: vector(files, &format!("{sp}subpixel_upsample.bias"), f * w)?,
            dense: Dense::load(files, &format!("{sp}first_hidden_to_hidden."), w, w)?,
            out: Dense::load(files, &format!("{sp}single_hidden_to_spks."), cfg.num_speakers, w)?,
            silence: vector(files, &format!("{sp}learnable_sil_emb"), d)?,
        })
    }

    /// Speaker logits `[m * factor, speakers]` of one step over `m` encoder positions.
    fn step(&self, cfg: &SpeechConfig, input: Vec<f32>, valid: usize) -> Vec<f32> {
        let (w, f) = (cfg.diar_head_size, cfg.subsampling_factor);
        let h = self.proj.forward(&self.encoder.forward(input, valid));
        let m = h.len() / w;
        // The convolution over the positions (zero padding), each output split into `factor`
        // frames.
        let mut up = vec![0f32; m * f * w];
        for_chunks(&mut up, f * w, m > 1, |ti, row| {
            for (o, v) in row.iter_mut().enumerate() {
                let mut acc = self.up_b[o];
                for j in 0..3 {
                    if ti + j >= 1 && ti + j < m + 1 {
                        acc += dot(&self.up_w[(o * 3 + j) * w..(o * 3 + j + 1) * w], &h[(ti + j - 1) * w..(ti + j) * w]);
                    }
                }
                *v = acc;
            }
        });
        up.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut hid = self.dense.forward(&up);
        hid.iter_mut().for_each(|v| *v = v.max(0.0));
        self.out.forward(&hid)
    }

    /// Speaker logits `[n * factor, speakers]` of a whole recording: `embeds` (`[n, d]`) in
    /// chunks with context, each also attending to the speaker cache and the FIFO.
    fn forward(&self, cfg: &SpeechConfig, embeds: &[f32], valid: usize) -> Vec<f32> {
        let (d, s, f) = (cfg.diar.hidden_size, cfg.num_speakers, cfg.subsampling_factor);
        let n = embeds.len() / d;
        let mut cache = SpeakerCache {
            cfg,
            d,
            embeds: Vec::new(),
            probs: Vec::new(),
            fifo: Vec::new(),
            compressed: false,
        };
        let mut logits = Vec::with_capacity(n * f * s);
        let mut start = 0;
        while start < n {
            let end = (start + cfg.chunk_length).min(n);
            let ctx0 = start.saturating_sub(cfg.chunk_left_context);
            let ctx1 = (end + cfg.chunk_right_context).min(n);
            let lookback = start - ctx0;
            let cached = cache.cache_frames() + cache.fifo_frames();
            let mut input = Vec::with_capacity((cached + ctx1 - ctx0) * d);
            input.extend_from_slice(&cache.embeds);
            input.extend_from_slice(&cache.fifo);
            input.extend_from_slice(&embeds[ctx0 * d..ctx1 * d]);
            let m = cached + ctx1 - ctx0;
            // The cached positions are valid; the chunk's up to the end of the recording.
            let step_valid = cached + valid.clamp(ctx0, ctx1) - ctx0;
            let lg = self.step(cfg, input.clone(), step_valid);
            // The probabilities of each position: the mean over its frames, zero on padding.
            let mut probs = vec![0f32; m * s];
            for (i, row) in probs.chunks_mut(s).enumerate().take(step_valid) {
                for fr in 0..f {
                    for (p, &l) in row.iter_mut().zip(&lg[(i * f + fr) * s..(i * f + fr + 1) * s]) {
                        *p += 1.0 / (1.0 + (-l).exp());
                    }
                }
                row.iter_mut().for_each(|p| *p /= f as f32);
            }
            cache.update(&input, &probs, &self.silence, end - start, lookback);
            let at = (cached + lookback) * f * s;
            logits.extend_from_slice(&lg[at..at + (end - start) * f * s]);
            start = end;
        }
        logits
    }
}

/// Log-mel features of a recording, padded to whole encoder positions.
pub struct Features {
    /// `[frames, mels]`, zero past `valid`.
    pub data: Vec<f32>,
    /// Frames, a multiple of the subsampling factor.
    pub frames: usize,
    /// Frames of the recording (`samples / hop`).
    pub valid: usize,
}

/// What the audio side gives for a recording.
pub struct Encoded {
    /// The soft tokens, `[tokens, text hidden]`.
    pub embeds: Vec<f32>,
    pub tokens: usize,
    /// The speech recognition encoder's output `[frames / factor, d]`, before its LayerNorm
    /// and the speaker embeddings.
    pub asr_states: Vec<f32>,
    /// Speaker logits `[frames, speakers]`, one row per spectrogram frame.
    pub speaker_logits: Vec<f32>,
    /// Encoder chunks run.
    pub chunks: usize,
}

/// The audio side: features, both encoders and the projection.
pub struct Speech<'a> {
    pub cfg: SpeechConfig,
    /// `[mels, n_fft / 2 + 1]`.
    fb: Vec<f32>,
    window: Vec<f32>,
    asr: Encoder<'a>,
    asr_norm: Norm,
    diar: Diarizer<'a>,
    diar_norm: Norm,
    /// `[speakers, d]`.
    speaker_embedding: Vec<f32>,
    proj: Dense,
}

fn hz_to_mel(f: f64) -> f64 {
    if f < 1000.0 {
        3.0 * f / 200.0
    } else {
        15.0 + (f / 1000.0).ln() * 27.0 / 6.4f64.ln()
    }
}

fn mel_to_hz(m: f64) -> f64 {
    if m < 15.0 {
        200.0 * m / 3.0
    } else {
        1000.0 * ((m - 15.0) * 6.4f64.ln() / 27.0).exp()
    }
}

fn to_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// The Slaney mel filter bank `[mels, bins]` (area-normalized triangles), rounded to bf16 as
/// the checkpoint keeps it.
fn mel_filters(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let top = (sample_rate / 2) as f64;
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(top));
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut fb = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        let norm = 2.0 / (edges[m + 2] - edges[m]);
        for b in 0..bins {
            let f = top * b as f64 / (bins - 1) as f64;
            let down = (f - edges[m]) / (edges[m + 1] - edges[m]);
            let up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            fb[m * bins + b] = to_bf16((down.min(up).max(0.0) * norm) as f32);
        }
    }
    fb
}

impl<'a> Speech<'a> {
    pub fn load(cfg: SpeechConfig, files: &'a ModelFiles) -> Result<Self> {
        let p = "perception.";
        let ep = format!("{p}encoder.");
        if !files.has(&format!("{ep}asr_encoder.pre_encode.proj.weight")) {
            bail!("the checkpoint has no audio encoder ({ep}asr_encoder.*): not a NeMo speech LM");
        }
        let (d, mels, bins) = (cfg.asr.hidden_size, cfg.n_mels, cfg.n_fft / 2 + 1);
        let stacked = cfg.subsampling_factor * mels;
        let (fbn, wn) = (
            format!("{p}preprocessor.featurizer.fb"),
            format!("{p}preprocessor.featurizer.window"),
        );
        let fb = if files.has(&fbn) {
            vector(files, &fbn, mels * bins)?
        } else {
            mel_filters(mels, cfg.n_fft, cfg.sample_rate)
        };
        let window = if files.has(&wn) {
            vector(files, &wn, cfg.win_length)?
        } else {
            let n = cfg.win_length;
            (0..n)
                .map(|i| to_bf16((0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()) as f32))
                .collect()
        };
        // The fixed sinusoidal embedding of each speaker slot.
        let s = cfg.num_speakers;
        let mut speaker_embedding = vec![0f32; s * d];
        for sp in 0..s {
            for i in (0..d).step_by(2) {
                let w = (i as f32 * -(10000f32.ln() / d as f32)).exp();
                speaker_embedding[sp * d + i] = (sp as f32 * w).sin();
                if i + 1 < d {
                    speaker_embedding[sp * d + i + 1] = (sp as f32 * w).cos();
                }
            }
        }
        let proj_out = files
            .info(&format!("{p}proj.bias"))
            .with_context(|| format!("missing tensor {p}proj.bias"))?
            .shape
            .iter()
            .product();
        Ok(Self {
            fb,
            window,
            asr: Encoder::load(files, &format!("{ep}asr_encoder."), &cfg.asr, Some(stacked))?,
            asr_norm: norm(files, &format!("{ep}asr_norm."), d)?,
            diar: Diarizer::load(files, &cfg)?,
            diar_norm: norm(files, &format!("{ep}diar_norm."), s)?,
            speaker_embedding,
            proj: Dense::load(files, &format!("{p}proj."), proj_out, d)?,
            cfg,
        })
    }

    /// Width of the soft tokens (the language model's hidden size).
    pub fn out_dim(&self) -> usize {
        self.proj.n_out()
    }

    /// Soft tokens of a recording of `samples` samples.
    pub fn num_tokens(&self, samples: usize) -> usize {
        (samples / self.cfg.hop_length).div_ceil(self.cfg.subsampling_factor)
    }

    /// The log-mel features of 16 kHz mono samples.
    pub fn features(&self, pcm: &[f32]) -> Features {
        let c = &self.cfg;
        let (n_fft, hop, mels) = (c.n_fft, c.hop_length, c.n_mels);
        let bins = n_fft / 2 + 1;
        let valid = pcm.len() / hop;
        let frames = (valid + 1).next_multiple_of(c.subsampling_factor);
        // Pre-emphasis, then `n_fft / 2` zeros on each side (a centered STFT).
        let pad = n_fft / 2;
        let mut x = vec![0f32; pcm.len() + 2 * pad];
        for (i, &s) in pcm.iter().enumerate() {
            x[pad + i] = if i == 0 { s } else { s - c.preemphasis * pcm[i - 1] };
        }
        // The window centered in the transform.
        let off = (n_fft - c.win_length) / 2;
        let fft = crate::whisper::mel::Fft::new(n_fft);
        let mut data = vec![0f32; frames * mels];
        for_chunks(&mut data, mels, valid > 16, |t, row| {
            if t >= valid {
                return;
            }
            let mut buf = vec![(0f64, 0f64); n_fft];
            for (i, &w) in self.window.iter().enumerate() {
                buf[off + i].0 = (x[t * hop + off + i] * w) as f64;
            }
            let mut out = vec![(0f64, 0f64); n_fft];
            fft.run(&buf, &mut out);
            let power: Vec<f32> = out[..bins].iter().map(|&(re, im)| (re * re + im * im) as f32).collect();
            for (m, v) in row.iter_mut().enumerate() {
                *v = (dot(&self.fb[m * bins..(m + 1) * bins], &power) + LOG_GUARD).ln();
            }
        });
        Features { data, frames, valid }
    }

    /// Runs both encoders and the projection on the features of a recording.
    pub fn encode(&self, feats: &Features) -> Encoded {
        let c = &self.cfg;
        let (mels, f, d, s) = (c.n_mels, c.subsampling_factor, c.asr.hidden_size, c.num_speakers);
        let n = feats.frames / f;
        let tokens = feats.valid.div_ceil(f);
        // Encoder positions whose first frame is in the recording: the keys.
        let valid = tokens.min(n);
        // Per mel bin over the valid frames: zero mean, unit (unbiased) deviation.
        let mut normed = vec![0f32; feats.frames * mels];
        let nv = feats.valid.max(1) as f32;
        for m in 0..mels {
            let col = |t: usize| feats.data[t * mels + m];
            let mean = (0..feats.valid).map(col).sum::<f32>() / nv;
            let var = (0..feats.valid).map(|t| (col(t) - mean) * (col(t) - mean)).sum::<f32>() / (nv - 1.0).max(1.0);
            let std = var.sqrt() + 1e-5;
            for t in 0..feats.valid {
                normed[t * mels + m] = (col(t) - mean) / std;
            }
        }
        // The speech recognition encoder, in chunks with context.
        let mut asr_states = vec![0f32; n * d];
        let mut chunks = 0;
        let mut start = 0;
        while start < n {
            let end = (start + c.chunk_length).min(n);
            let ctx0 = start.saturating_sub(c.chunk_left_context);
            let ctx1 = (end + c.chunk_right_context).min(n);
            let x = self.asr.embed(&normed[ctx0 * f * mels..ctx1 * f * mels], f * mels);
            let y = self.asr.forward(x, valid.clamp(ctx0, ctx1) - ctx0);
            asr_states[start * d..end * d].copy_from_slice(&y[(start - ctx0) * d..(end - ctx0) * d]);
            chunks += 1;
            start = end;
        }
        // The speaker encoder on the raw features.
        let dx = self.diar.encoder.embed(&feats.data, f * mels);
        let speaker_logits = self.diar.forward(c, &dx, valid);
        // Thresholded mean activity per position, normed, selecting the speaker embeddings.
        let mut last = asr_states.clone();
        layer_norm_rows(&mut last, d, &self.asr_norm);
        for_chunks(&mut last, d, n > 1, |i, row| {
            let mut act = vec![0f32; s];
            for fr in 0..f {
                for (a, &l) in act.iter_mut().zip(&speaker_logits[(i * f + fr) * s..(i * f + fr + 1) * s]) {
                    *a += 1.0 / (1.0 + (-l).exp());
                }
            }
            for a in act.iter_mut() {
                *a = (*a / f as f32 > c.speaker_activity_threshold) as u8 as f32;
            }
            layer_norm_rows(&mut act, s, &self.diar_norm);
            for (sp, &a) in act.iter().enumerate() {
                axpy(row, a, &self.speaker_embedding[sp * d..(sp + 1) * d]);
            }
        });
        let mut embeds = self.proj.forward(&last);
        embeds.truncate(tokens * self.out_dim());
        Encoded {
            embeds,
            tokens,
            asr_states,
            speaker_logits,
            chunks,
        }
    }
}

/// Collapses the loops a generation can fall into (the model card's guardrail, applied to the
/// reported transcription results): a character repeated more than 20 times becomes one copy,
/// a block of up to 4096 characters repeated 20 times or more becomes one copy.
pub fn cleanup_generation_loops(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    // Runs of one character.
    let mut a: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut j = i;
        while j < chars.len() && chars[j] == chars[i] {
            j += 1;
        }
        let run = j - i;
        a.extend(std::iter::repeat_n(chars[i], if run >= 21 { 1 } else { run }));
        i = j;
    }
    // Repeated blocks: at each position the shortest block that repeats at least 20 times.
    let mut out: Vec<char> = Vec::with_capacity(a.len());
    let mut i = 0;
    while i < a.len() {
        let mut found = None;
        for len in 1..=((a.len() - i) / 20).min(4096) {
            if a[i] != a[i + len] {
                continue;
            }
            let mut reps = 1;
            while i + (reps + 1) * len <= a.len() && a[i..i + len] == a[i + reps * len..i + (reps + 1) * len] {
                reps += 1;
            }
            if reps >= 20 {
                found = Some((len, reps));
                break;
            }
        }
        match found {
            Some((len, reps)) => {
                out.extend_from_slice(&a[i..i + len]);
                i += len * reps;
            }
            None => {
                out.push(a[i]);
                i += 1;
            }
        }
    }
    out.into_iter().collect()
}

/// How to run one request.
#[derive(Clone, Debug)]
pub struct Request<'r> {
    /// The instruction in the user turn, before the audio.
    pub prompt: &'r str,
    pub system: Option<&'r str>,
    /// Lets the model reason before it answers (audio questions; off for transcription).
    pub think: bool,
    /// 0 is greedy decoding.
    pub temperature: f32,
    pub seed: u64,
    /// Default: twice the audio tokens plus 64.
    pub max_new_tokens: Option<usize>,
    /// Tokens of the prompt per forward pass.
    pub prefill_chunk: usize,
}

impl Default for Request<'_> {
    fn default() -> Self {
        Self {
            prompt: TRANSCRIBE_PROMPT,
            system: None,
            think: false,
            temperature: 0.0,
            seed: 0,
            max_new_tokens: None,
            prefill_chunk: 256,
        }
    }
}

/// The answer to a request.
#[derive(Clone, Debug, Default)]
pub struct Reply {
    /// The decoded text, special tokens other than the end of the turn kept (speaker tags).
    pub text: String,
    /// The generated tokens, without the token that ended the turn.
    pub tokens: Vec<u32>,
    /// Mean log-probability of the generated tokens.
    pub avg_logprob: f32,
    /// The prompt in tokens (the audio tokens among them).
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
    pub chunks: usize,
    /// Stopped by the callback or the token limit rather than by the end of the turn.
    pub stopped: bool,
    pub feature_secs: f64,
    pub encode_secs: f64,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

/// Nemotron 3.5 Transcribe: the audio side, the language model and the tokenizer of a
/// checkpoint folder.
pub struct SpeechLm {
    pub speech: Speech<'static>,
    pub lm: NemotronModel<'static>,
    pub tok: Tokenizer,
    stop: Vec<u32>,
    // Declared last: the models above borrow it.
    _files: Box<ModelFiles>,
}

impl SpeechLm {
    /// Loads the checkpoint in `dir`: the language model at precision `quant`, the encoders in
    /// bf16.
    pub fn load(dir: &Path, quant: Quant, mmap: bool) -> Result<Self> {
        let mt = crate::config::model_type(dir)?;
        if mt != crate::nemotron::SPEECHLM {
            bail!("{}: model_type {mt} is not a NeMo speech LM", dir.display());
        }
        let files = Box::new(ModelFiles::open(dir, !mmap)?);
        // SAFETY: the files are boxed (their address is stable) and dropped after the models,
        // which are declared first.
        let fr: &'static ModelFiles = unsafe { &*(&*files as *const ModelFiles) };
        let tok = crate::engine::load_tokenizer(dir)?;
        let mut cfg = SpeechConfig::load(dir)?;
        cfg.audio_token_id = tok
            .token_to_id(&cfg.audio_token)
            .with_context(|| format!("the tokenizer has no {} token", cfg.audio_token))?;
        let speech = Speech::load(cfg, fr)?;
        let lm = NemotronModel::load(NemotronConfig::load(dir)?, fr, quant)?;
        if speech.out_dim() != lm.cfg.hidden_size {
            bail!(
                "the audio projection gives {} values, the language model takes {}",
                speech.out_dim(),
                lm.cfg.hidden_size
            );
        }
        let fmt = crate::chat::NemotronChat;
        let stop = crate::chat::with_turn_end(&fmt, &tok, crate::config::load_eos_ids(dir));
        Ok(Self {
            speech,
            lm,
            tok,
            stop,
            _files: files,
        })
    }

    /// The prompt of a request over `n` audio tokens: the token ids (the audio positions hold
    /// the audio token) and the position of the first audio token.
    pub fn prompt_ids(&self, req: &Request, n: usize) -> Result<(Vec<u32>, usize)> {
        use crate::chat::ChatFormat;
        let fmt = crate::chat::NemotronChat;
        // The template's rendering of a text part and an audio part: "<text> <|audio|>".
        let text = fmt.first_turn(
            req.system,
            req.think,
            &[],
            &format!("{} {}", req.prompt, self.speech.cfg.audio_token),
        );
        let (head, tail) = text
            .rsplit_once(&self.speech.cfg.audio_token)
            .context("the prompt lost its audio token")?;
        let enc = |s: &str| -> Result<Vec<u32>> {
            Ok(self
                .tok
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec())
        };
        let mut ids = enc(head)?;
        let at = ids.len();
        ids.extend(std::iter::repeat_n(self.speech.cfg.audio_token_id, n));
        ids.extend(enc(tail)?);
        Ok((ids, at))
    }

    /// Answers `req` about 16 kHz mono samples. `on_token` sees the tokens generated so far
    /// after each one and can stop the decoding.
    pub fn run(&self, pcm: &[f32], req: &Request, on_token: &mut dyn FnMut(&[u32]) -> crate::generate::Flow) -> Result<Reply> {
        use crate::clock::Instant;
        let t0 = Instant::now();
        let feats = self.speech.features(pcm);
        if feats.valid == 0 {
            bail!("the recording is shorter than one frame ({} samples)", self.speech.cfg.hop_length);
        }
        let feature_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let enc = self.speech.encode(&feats);
        let encode_secs = t1.elapsed().as_secs_f64();
        let (ids, at) = self.prompt_ids(req, enc.tokens)?;
        let h = self.lm.cfg.hidden_size;
        let overrides: Vec<(usize, Vec<f32>)> = (0..enc.tokens).map(|i| (at + i, enc.embeds[i * h..(i + 1) * h].to_vec())).collect();
        let t2 = Instant::now();
        let mut sess = self.lm.new_session();
        let hs = crate::generate::prefill(&self.lm, &mut sess, &ids, req.prefill_chunk.max(1), &overrides);
        let logits = self.lm.logits(&hs[hs.len() - h..], 1);
        let prefill_secs = t2.elapsed().as_secs_f64();
        let max_new = req.max_new_tokens.unwrap_or(2 * enc.tokens + 64);
        let mut sampler = crate::sampler::Sampler::new(req.temperature, 0, 1.0, req.seed);
        // The log-probability of each generated token, from the logits it was drawn from.
        let (mut sum, mut count) = (0f64, 0usize);
        let lm = &self.lm;
        let mut logp = |lg: &[f32], id: u32| {
            let m = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = m + lg.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
            sum += lg[id as usize] as f64 - lse;
            count += 1;
        };
        let t3 = Instant::now();
        let mut out: Vec<u32> = Vec::new();
        let mut logits = logits;
        let mut stopped = true;
        for _ in 0..max_new {
            let id = sampler.sample(&logits);
            if self.stop.contains(&id) {
                stopped = false;
                break;
            }
            logp(&logits, id);
            out.push(id);
            if on_token(&out) == crate::generate::Flow::Stop {
                break;
            }
            logits = lm.forward(&mut sess, &[id]);
        }
        let decode_secs = t3.elapsed().as_secs_f64();
        let text = self.tok.decode(&out, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(Reply {
            text,
            avg_logprob: if count > 0 { (sum / count as f64) as f32 } else { 0.0 },
            tokens: out,
            prompt_tokens: ids.len(),
            audio_tokens: enc.tokens,
            chunks: enc.chunks,
            stopped,
            feature_secs,
            encode_secs,
            prefill_secs,
            decode_secs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loops_are_collapsed() {
        assert_eq!(cleanup_generation_loops("hello world"), "hello world");
        assert_eq!(cleanup_generation_loops(&format!("a{}b", "x".repeat(21))), "axb");
        // 20 copies are a block repeated 20 times; 19 are kept.
        assert_eq!(cleanup_generation_loops(&format!("a{}b", "x".repeat(20))), "axb");
        assert_eq!(
            cleanup_generation_loops(&format!("a{}b", "x".repeat(19))),
            format!("a{}b", "x".repeat(19))
        );
        assert_eq!(
            cleanup_generation_loops(&format!("so {}end", "and then ".repeat(20))),
            "so and then end"
        );
        let kept = format!("so {}end", "and then ".repeat(19));
        assert_eq!(cleanup_generation_loops(&kept), kept);
        assert_eq!(cleanup_generation_loops(&"日本".repeat(25)), "日本");
    }

    #[test]
    fn mel_bank_is_slaney() {
        let fb = mel_filters(128, 512, 16_000);
        let bins = 257;
        // Every filter is a triangle with a positive area, the first one starting at 0 Hz.
        assert_eq!(fb[0], 0.0);
        for m in 0..128 {
            let row = &fb[m * bins..(m + 1) * bins];
            assert!(row.iter().all(|x| *x >= 0.0) && row.iter().any(|x| *x > 0.0), "filter {m}");
        }
        // Slaney's normalization: 2 / (the triangle's width in Hz) at the peak, about 0.03 for
        // the 66.7 Hz spaced filters below 1 kHz.
        let peak = fb[bins..2 * bins].iter().copied().fold(0f32, f32::max);
        assert!((0.01..0.031).contains(&peak), "{peak}");
    }

    /// The speaker cache keeps a recording's first positions as they are, then compresses to
    /// its length with the silence slots of every speaker.
    #[test]
    fn speaker_cache_fills_and_compresses() {
        let raw = serde_json::json!({
            "audio_config": {
                "asr_config": {"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2},
                "diarization_config": {
                    "audio_config": {"hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2},
                    "head_config": {"hidden_size": 4, "num_speakers": 2},
                    "streaming_config": {"speaker_cache_length": 6},
                    "chunk_length": 4, "chunk_left_context": 1, "chunk_right_context": 1,
                    "fifo_length": 2, "speaker_cache_update_period": 3,
                },
            },
        });
        let cfg = SpeechConfig::from_json(&raw).unwrap();
        let d = 4;
        let mut c = SpeakerCache {
            cfg: &cfg,
            d,
            embeds: Vec::new(),
            probs: Vec::new(),
            fifo: Vec::new(),
            compressed: false,
        };
        let silence = vec![-1f32; d];
        // Chunk 1: positions 0..4 (and one of look-ahead); speaker 0 talks.
        let input: Vec<f32> = (0..5).flat_map(|i| vec![i as f32; d]).collect();
        let probs: Vec<f32> = (0..5).flat_map(|_| [0.9f32, 0.1]).collect();
        c.update(&input, &probs, &silence, 4, 0);
        // 4 positions overflow a FIFO of 2: 3 (the update period) move to the cache.
        assert_eq!((c.cache_frames(), c.fifo_frames(), c.compressed), (3, 1, false));
        assert_eq!(c.embeds[..d], [0.0; 4]);
        assert_eq!(c.fifo, vec![3.0; d]);
        // Chunk 2: the cache (3), the FIFO (1), one of look-back, positions 4..8; speaker 1.
        let mut input = c.embeds.clone();
        input.extend_from_slice(&c.fifo);
        input.extend((3..8).flat_map(|i| vec![i as f32; d]));
        let mut probs: Vec<f32> = (0..4).flat_map(|_| [0.9f32, 0.1]).collect();
        probs.extend((0..5).flat_map(|_| [0.2f32, 0.8]));
        c.update(&input, &probs, &silence, 4, 1);
        // FIFO 1 + 4 = 5 positions, 3 move: the cache would hold 6 of 6, not compressed.
        assert_eq!((c.cache_frames(), c.fifo_frames(), c.compressed), (6, 2, false));
        // Chunk 3 overflows the cache: compressed to 6, a silence slot for each speaker.
        let mut input = c.embeds.clone();
        input.extend_from_slice(&c.fifo);
        input.extend((7..12).flat_map(|i| vec![i as f32; d]));
        let mut probs: Vec<f32> = (0..3).flat_map(|_| [0.9f32, 0.1]).collect();
        probs.extend((0..5).flat_map(|_| [0.2f32, 0.8]));
        probs.extend((0..5).flat_map(|_| [0.9f32, 0.1]));
        c.update(&input, &probs, &silence, 4, 1);
        assert_eq!((c.cache_frames(), c.compressed), (6, true));
        let rows: Vec<f32> = c.embeds.chunks(d).map(|r| r[0]).collect();
        assert_eq!(rows.iter().filter(|x| **x == -1.0).count(), 2, "{rows:?}");
        // Speaker 0's positions come first, in order, then its silence slot, then speaker 1's.
        let first_silence = rows.iter().position(|x| *x == -1.0).unwrap();
        assert!(rows[..first_silence].windows(2).all(|w| w[0] < w[1]), "{rows:?}");
        assert_eq!(*rows.last().unwrap(), -1.0);
    }
}
