//! LTX-2.5's prompt encoding (`ltx_core.text_encoders.gemma`): the release's Gemma 4 12B text
//! encoder (`text_encoders/gemma4-12b-with-proj-ltx-2.5-*.safetensors`, transformers'
//! `gemma4_unified` text model with its config, tokenizer and the two projections in one file),
//! then the feature extractor:
//!
//! 1. the prompt (stripped) as `<bos>` and its tokens, at most 1024 (`LTXGemmaTokenizer`; the
//!    reference left-pads to 1024 with an attention mask, which leaves the valid rows as they are
//!    alone: RoPE is relative and the sliding window spans all of them);
//! 2. all 49 hidden states (the scaled embeddings, every layer's output, the last one after the
//!    final norm) of each token, each RMS-normed over its 3840 values (`PER_TOKEN_RMS`),
//!    interleaved as `[3840, 49]` (`FeatureExtractorV2`);
//! 3. `video_aggregate_embed` / `audio_aggregate_embed` of those rows times
//!    `sqrt(4096 / 3840)` / `sqrt(2048 / 3840)`.
//!
//! The transformer's connectors ([`super::dit::Connector`]) then make the 1024 context rows.

use crate::minimax_h3::dit::DLin;
use crate::model::{Model, Quant};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// The tokenizer's length (`TOKENIZER_MAX_LENGTH`): prompts are cut there and the connectors
/// fill the context up to it with registers.
pub const MAX_TOKENS: usize = 1024;

pub struct TextEncoder<'a> {
    pub model: Model<'a>,
    tok: crate::tokenizer::Tokenizer,
    video_proj: DLin<'a>,
    audio_proj: DLin<'a>,
    bos: u32,
}

/// The `[n, video_dim]` and `[n, audio_dim]` features of a prompt's tokens.
pub struct Features {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
    pub tokens: usize,
}

/// The Gemma config inside a text encoder file (`gemma_config` metadata).
pub fn gemma_config(files: &ModelFiles) -> Result<serde_json::Value> {
    let s = files
        .metadata("gemma_config")
        .context("LTX text encoder: no `gemma_config` metadata (not an LTX-2.5 text encoder file)")?;
    serde_json::from_str(s).context("LTX text encoder: `gemma_config` is not JSON")
}

impl<'a> TextEncoder<'a> {
    pub fn load(files: &'a ModelFiles) -> Result<Self> {
        let raw = gemma_config(files)?;
        let bos = raw
            .get("text_config")
            .and_then(|t| t.get("bos_token_id"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as u32;
        let cfg = crate::config::TextConfig::from_json(raw)?;
        let (h, nl) = (cfg.hidden_size, cfg.num_hidden_layers);
        let model = Model::load(cfg, files, Quant::Bf16)?;
        let tok = crate::tokenizer::Tokenizer::from_bytes(files.bytes("tokenizer_json").context("LTX text encoder: no tokenizer")?)
            .map_err(|e| anyhow::anyhow!("LTX text encoder tokenizer: {e}"))?;
        let flat = h * (nl + 1);
        let out = |name: &str| -> Result<usize> {
            let i = files
                .info(&format!("{name}.weight"))
                .with_context(|| format!("missing tensor {name}.weight"))?;
            if i.shape.len() != 2 || i.shape[1] != flat {
                bail!("{name}.weight: {:?}, expected [_, {flat}]", i.shape);
            }
            Ok(i.shape[0])
        };
        let (vn, an) = (
            "text_embedding_projection.video_aggregate_embed",
            "text_embedding_projection.audio_aggregate_embed",
        );
        let (vd, ad) = (out(vn)?, out(an)?);
        Ok(Self {
            video_proj: DLin::load(files, vn, vd, flat, true)?,
            audio_proj: DLin::load(files, an, ad, flat, true)?,
            model,
            tok,
            bos,
        })
    }

    /// The token ids the reference encodes: `<bos>` and the stripped prompt's tokens, at most
    /// [`MAX_TOKENS`].
    pub fn tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        let enc = self
            .tok
            .encode(prompt.trim(), false)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        if ids.first() != Some(&self.bos) {
            ids.insert(0, self.bos);
        }
        ids.truncate(MAX_TOKENS);
        Ok(ids)
    }

    /// Every hidden state of `ids` (`hidden_states[0..=layers]`), `[layers + 1][t * hidden]`.
    pub fn hidden_states(&self, ids: &[u32]) -> Vec<Vec<f32>> {
        let m = &self.model;
        let nl = m.cfg.num_hidden_layers;
        let mut states = Vec::with_capacity(nl + 1);
        states.push(ids.iter().flat_map(|&t| m.embed_scaled(t)).collect::<Vec<f32>>());
        let mut sess = m.new_session();
        let cap: Vec<usize> = (0..nl - 1).collect();
        // On the GPU prefill plan (activations on the device for the whole pass, every layer's
        // output copied out) when there is one; `OXINFER_LTX_TEXT_GPU=0`: the host path.
        if crate::gpu::active().is_some()
            && std::env::var("OXINFER_LTX_TEXT_GPU").map_or(true, |v| v != "0")
            && let Some((last, rows)) = m.forward_captures_gpu(&mut sess, ids, &cap)
        {
            let (t, h, nc) = (ids.len(), m.cfg.hidden_size, cap.len());
            for c in 0..nc {
                let mut s = vec![0f32; t * h];
                for (ti, r) in s.chunks_mut(h).enumerate() {
                    r.copy_from_slice(&rows[(ti * nc + c) * h..(ti * nc + c + 1) * h]);
                }
                states.push(s);
            }
            states.push(last);
            return states;
        }
        sess.capture = cap;
        let last = m.forward_hidden(&mut sess, ids);
        let mut caps = std::mem::take(&mut sess.captured);
        caps.sort_by_key(|c| c.0);
        states.extend(caps.into_iter().map(|c| c.1));
        states.push(last);
        states
    }

    /// The prompt's video and audio features ([`Features`]).
    pub fn encode(&self, prompt: &str) -> Result<Features> {
        let ids = self.tokens(prompt)?;
        let states = self.hidden_states(&ids);
        let h = self.model.cfg.hidden_size;
        Ok(self.features(&states, ids.len(), h))
    }

    /// [`Features`] from hidden states (`[l][t * h]`).
    pub fn features(&self, states: &[Vec<f32>], t: usize, h: usize) -> Features {
        let flat = features_flat(states, t, h);
        let width = flat.len() / t;
        let scaled = |dim: usize| -> Vec<f32> {
            let s = (dim as f32 / h as f32).sqrt();
            flat.iter().map(|&v| v * s).collect()
        };
        let _ = width;
        let video = self.video_proj.forward(&scaled(self.video_proj.n_out), t);
        let audio = self.audio_proj.forward(&scaled(self.audio_proj.n_out), t);
        Features { video, audio, tokens: t }
    }
}

/// `norm_and_concat_per_token_rms`: each state's rows RMS-normed (eps 1e-6), then per token
/// `[h, layers]` flattened (`d * layers + l`).
pub fn features_flat(states: &[Vec<f32>], t: usize, h: usize) -> Vec<f32> {
    let nl = states.len();
    let mut out = vec![0f32; t * h * nl];
    crate::kernels::for_chunks(&mut out, h * nl, t > 1, |r, row| {
        for (l, s) in states.iter().enumerate() {
            let x = &s[r * h..(r + 1) * h];
            let var = x.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for (d, &v) in x.iter().enumerate() {
                row[d * nl + l] = v * inv;
            }
        }
    });
    out
}
