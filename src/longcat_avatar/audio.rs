//! The audio conditioning of LongCat-Video-Avatar 1.5 / 1.6: Whisper-large-v3 encoder hidden
//! states (`whisper-large-v3/` of the release), as `get_audio_embedding_whisper` of the
//! reference pipeline computes them.
//!
//! The speech (16 kHz mono) is cut into 30 s pieces, each featurized on its own (zero-padded to
//! 30 s, its own log-mel clamp) and encoded on its own; the 33 hidden states of every 20 ms
//! position (the embeddings, the stream after layers 1 to 31, the final layer norm's output) are
//! averaged in four groups of eight and the last kept alone, giving five vectors of 1280 per
//! position; the first `2 * frames` positions are then resampled linearly (`align_corners`) to
//! one row per video frame at 25 fps.

use crate::safetensors::ModelFiles;
use crate::whisper::mel::{CHUNK_SAMPLES, SAMPLE_RATE, log_mel};
use crate::whisper::model::{Precision, WhisperConfig, WhisperModel};
use anyhow::{Result, bail};
use std::path::Path;

/// Hidden-state groups per position.
pub const GROUPS: usize = 5;
/// The encoder's positions per second.
const ENC_FPS: usize = 50;

pub struct AudioEncoder {
    model: WhisperModel,
}

impl AudioEncoder {
    /// Loads the encoder of `dir` (`config.json`, `model.safetensors`); the decoder's layers are
    /// not read.
    pub fn load(dir: &Path) -> Result<Self> {
        let mut cfg = WhisperConfig::load(dir)?;
        if cfg.encoder_layers != 32 {
            bail!(
                "the audio features group 33 hidden states: a Whisper encoder of {} layers is not supported",
                cfg.encoder_layers
            );
        }
        cfg.decoder_layers = 0;
        let files = ModelFiles::open(dir, cfg!(target_arch = "wasm32"))?;
        // Half precision is exact for the release's float16 file; a float32 one keeps its values.
        let prec = match files.info("model.encoder.layers.0.fc1.weight").map(|t| t.dtype.as_str()) {
            Some("F16") => Precision::F16,
            _ => Precision::F32,
        };
        Ok(Self {
            model: WhisperModel::load(cfg, &files, prec, Precision::Bf16)?,
        })
    }

    /// Width of a feature vector (1280).
    pub fn dim(&self) -> usize {
        self.model.cfg.d_model
    }

    /// The features of `speech` (16 kHz mono) at `fps` rows per second: `[rows, 5, dim]` with
    /// `rows = floor(seconds * fps)`.
    pub fn embed(&self, speech: &[f32], fps: usize, progress: &mut dyn FnMut(usize, usize)) -> Result<(Vec<f32>, usize)> {
        let d = self.dim();
        let rows = speech.len() * fps / SAMPLE_RATE;
        if rows == 0 {
            bail!("the audio is shorter than one video frame");
        }
        // The reference keeps `2 * rows` encoder positions whatever the frame rate.
        let want = 2 * rows;
        let per_chunk = CHUNK_SAMPLES / SAMPLE_RATE * ENC_FPS;
        let chunks = speech.len().div_ceil(CHUNK_SAMPLES);
        let mut pos: Vec<f32> = Vec::with_capacity(want.min(chunks * per_chunk) * GROUPS * d);
        let mut have = 0;
        for (ci, piece) in speech.chunks(CHUNK_SAMPLES).enumerate() {
            if have >= want {
                break;
            }
            let mel = log_mel(piece, self.model.cfg.num_mel_bins);
            let states = self.model.encode_hidden_states(&mel.window(0));
            let n = per_chunk.min(want - have);
            let base = pos.len();
            pos.resize(base + n * GROUPS * d, 0.0);
            let out = &mut pos[base..];
            for (g, range) in [0..8, 8..16, 16..24, 24..32, 32..33].into_iter().enumerate() {
                let inv = 1.0 / range.len() as f32;
                for s in range {
                    let st = &states[s];
                    for p in 0..n {
                        let o = &mut out[(p * GROUPS + g) * d..(p * GROUPS + g + 1) * d];
                        for (a, &b) in o.iter_mut().zip(&st[p * d..(p + 1) * d]) {
                            *a += b * inv;
                        }
                    }
                }
            }
            have += n;
            progress(ci + 1, chunks);
        }
        // Linear resampling with the end points aligned.
        let mut out = vec![0f32; rows * GROUPS * d];
        let w = GROUPS * d;
        for (i, o) in out.chunks_mut(w).enumerate() {
            let x = if rows > 1 {
                i as f32 * (have - 1) as f32 / (rows - 1) as f32
            } else {
                0.0
            };
            let i0 = (x.floor() as usize).min(have - 1);
            let i1 = (i0 + 1).min(have - 1);
            let f = x - i0 as f32;
            let (a, b) = (&pos[i0 * w..(i0 + 1) * w], &pos[i1 * w..(i1 + 1) * w]);
            for ((v, &p), &q) in o.iter_mut().zip(a).zip(b) {
                *v = p + f * (q - p);
            }
        }
        Ok((out, rows))
    }
}

/// The audio rows of one clip: for each of `frames` video frames from `start` (every `stride`
/// rows), the rows at offsets -2 to 2, clamped to the ends: `[frames, 5, 5, dim]`.
pub fn clip_window(emb: &[f32], rows: usize, start: usize, frames: usize, stride: usize, window: usize) -> Vec<f32> {
    let w = emb.len() / rows;
    let half = (window / 2) as i64;
    let mut out = Vec::with_capacity(frames * window * w);
    for f in 0..frames {
        for o in 0..window as i64 {
            let i = ((start + f * stride) as i64 + o - half).clamp(0, rows as i64 - 1) as usize;
            out.extend_from_slice(&emb[i * w..(i + 1) * w]);
        }
    }
    out
}
