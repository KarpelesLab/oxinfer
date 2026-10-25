//! Core ML (ANEMLL-format) models behind [`super::Engine`]: loading, prefill and the decoding
//! loop over [`crate::anemll::AnemllModel`].

use super::{CoreMlInfo, CoreMlPrefill, CoreMlUnits, EngineOptions, Family};
use crate::anemll::{AnemllModel, Meta, Next, Options, PrefillMode};
use crate::coreml::ComputeUnits;
use crate::generate::{Decoded, Flow, StopReason};
use crate::sampler::Sampler;
use anyhow::{Result, bail};
use std::path::Path;

/// A loaded Core ML model and its conversation state.
pub(super) struct CoreMlLm {
    pub model: AnemllModel,
    /// Logits past the tokenizer's vocabulary (padding of the LM head) are not sampled.
    vocab: usize,
    /// A reset whose new state could not be made; retried before the next turn.
    reset_failed: bool,
}

/// What `Engine::load` needs from a Core ML model directory.
pub(super) struct Loaded {
    pub lm: CoreMlLm,
    pub meta: Meta,
    pub tok: tokenizers::Tokenizer,
    pub fmt: &'static (dyn crate::chat::ChatFormat + Sync),
    pub stop: Vec<u32>,
    pub family: Family,
    pub info: CoreMlInfo,
    pub hidden_size: usize,
    pub layers: usize,
    pub vocab_size: usize,
    pub bytes: u64,
}

/// Whether `dir` holds a Core ML model this backend runs.
pub(super) fn detect(dir: &Path) -> bool {
    crate::anemll::is_anemll(dir)
}

fn units(opts: &EngineOptions) -> (ComputeUnits, CoreMlUnits) {
    use crate::platform::DevicePref;
    let u = opts.coreml_units.unwrap_or(match opts.device {
        DevicePref::Cpu => CoreMlUnits::Cpu,
        DevicePref::Gpu(_) => CoreMlUnits::Gpu,
        DevicePref::Auto => CoreMlUnits::Ane,
    });
    let cu = match u {
        CoreMlUnits::Ane => ComputeUnits::CpuAndNeuralEngine,
        CoreMlUnits::All => ComputeUnits::All,
        CoreMlUnits::Gpu => ComputeUnits::CpuAndGpu,
        CoreMlUnits::Cpu => ComputeUnits::CpuOnly,
    };
    (cu, u)
}

/// `eos_token_id` of `generation_config.json` / `config.json` (no defaults).
fn config_eos(dir: &Path) -> Vec<u32> {
    let ids = |v: &serde_json::Value| -> Option<Vec<u32>> {
        match v {
            serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as u32]),
            serde_json::Value::Array(a) => Some(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()),
            _ => None,
        }
    };
    let read = |name: &str| -> Option<serde_json::Value> { serde_json::from_slice(&std::fs::read(dir.join(name)).ok()?).ok() };
    if let Some(v) = read("generation_config.json").and_then(|v| ids(&v["eos_token_id"])) {
        return v;
    }
    read("config.json")
        .and_then(|v| ids(&v["eos_token_id"]).or_else(|| ids(&v["text_config"]["eos_token_id"])))
        .unwrap_or_default()
}

fn dir_bytes(p: &Path) -> u64 {
    let Ok(md) = std::fs::metadata(p) else { return 0 };
    if md.is_file() {
        return md.len();
    }
    std::fs::read_dir(p)
        .map(|rd| rd.flatten().map(|e| dir_bytes(&e.path())).sum())
        .unwrap_or(0)
}

pub(super) fn load(dir: &Path, opts: &EngineOptions) -> Result<Loaded> {
    if opts.mtp || opts.mtp_from.is_some() || opts.draft.is_some() || opts.stream.is_some() {
        bail!("speculative decoding and weight streaming are not available for Core ML models");
    }
    let meta = Meta::load(dir)?;
    let tok = meta.tokenizer()?;
    let fmt = meta.chat_format()?;
    let mut stop = meta.stop_tokens(&tok, fmt);
    for id in config_eos(dir) {
        if !stop.contains(&id) {
            stop.push(id);
        }
    }
    let arch = format!("{} {}", meta.architecture, meta.model_prefix).to_lowercase();
    let family = if arch.contains("gemma4") {
        Family::Gemma4
    } else if arch.contains("gemma") {
        Family::Gemma3
    } else if arch.contains("qwen") {
        Family::Qwen
    } else {
        Family::Llama
    };
    let (cu, u) = units(opts);
    let prefill = match opts.coreml_prefill {
        CoreMlPrefill::Auto => PrefillMode::Auto,
        CoreMlPrefill::Batch => PrefillMode::Batched,
        CoreMlPrefill::Pad => PrefillMode::Padded,
        CoreMlPrefill::Single => PrefillMode::Single,
    };
    let model = AnemllModel::load(dir, Options { units: cu, prefill })?;
    let resolved = match model.options.prefill {
        PrefillMode::Batched => CoreMlPrefill::Batch,
        PrefillMode::Padded if !(meta.no_rewind() || meta.prefill_full_batches_only) => CoreMlPrefill::Pad,
        PrefillMode::Padded => CoreMlPrefill::Batch,
        PrefillMode::Single => CoreMlPrefill::Single,
        PrefillMode::Auto => CoreMlPrefill::Auto,
    };
    let p = |k: &str| meta.params.get(k).and_then(crate::anemll::yaml::Value::as_usize);
    let hidden_size = p("hidden_size").unwrap_or(0);
    let layers = p("num_layers").unwrap_or(0);
    let vocab_size = meta
        .vocab_size
        .or_else(|| meta.lm_head_chunk_sizes.as_ref().map(|s| s.iter().sum()))
        .unwrap_or_else(|| tok.get_vocab_size(true));
    let parts = [&meta.embeddings, &meta.lm_head].into_iter().chain(&meta.ffn).chain(&meta.pf);
    let bytes = parts.map(|p| dir_bytes(p)).sum();
    let info = CoreMlInfo {
        units: u,
        prefill: resolved,
        architecture: meta.architecture.clone(),
        context_length: meta.context_length,
        batch_size: meta.batch_size,
        chunks: meta.ffn.len(),
        recurrent: meta.recurrent_state,
    };
    let vocab = tok.get_vocab_size(true).max(1);
    Ok(Loaded {
        lm: CoreMlLm {
            model,
            vocab,
            reset_failed: false,
        },
        meta,
        tok,
        fmt,
        stop,
        family,
        info,
        hidden_size,
        layers,
        vocab_size,
        bytes,
    })
}

impl CoreMlLm {
    pub fn context_len(&self) -> usize {
        self.model.pos()
    }

    /// A new state (the recurrent and KV states cannot be rewound).
    pub fn reset(&mut self) {
        self.reset_failed = self.model.reset().is_err();
    }

    /// Feeds the turn's tokens; the prediction after the last one.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<Next> {
        if self.reset_failed {
            self.model.reset()?;
            self.reset_failed = false;
        }
        self.model.prefill(ids)
    }

    /// Plain decoding as [`crate::generate::decode`]: every generated token is fed except a stop
    /// token or the token a cancellation stopped at ([`Decoded::unfed`]).
    pub fn decode(
        &mut self,
        mut next: Next,
        sampler: &mut Sampler,
        stop: &[u32],
        max_tokens: usize,
        on_token: &mut dyn FnMut(&[u32]) -> Flow,
    ) -> Result<Decoded> {
        let t0 = crate::clock::Instant::now();
        let mut out: Vec<u32> = Vec::new();
        let (mut forwards, mut reason, mut unfed) = (0usize, StopReason::Length, None);
        for _ in 0..max_tokens {
            let id = match &next {
                Next::Logits(l) => sampler.sample(&l[..l.len().min(self.vocab)]),
                Next::Argmax { token, .. } => *token,
            };
            out.push(id);
            if stop.contains(&id) {
                reason = StopReason::Stop;
                break;
            }
            if on_token(&out) == Flow::Stop {
                reason = StopReason::Cancelled;
                unfed = Some(id);
                break;
            }
            next = self.model.step(id)?;
            forwards += 1;
        }
        Ok(Decoded {
            tokens: out,
            forwards,
            secs: t0.elapsed().as_secs_f64(),
            reason,
            unfed,
        })
    }
}
