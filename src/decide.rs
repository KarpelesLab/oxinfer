//! Typed decisions ("System 1" of the Jev models, e.g. `autotrust/JEV-9B`, `autotrust/JEV-27B`):
//! a question about a piece of text answered with calibrated probabilities from one prefill pass,
//! with no generation.
//!
//! Kinds: `noul` (yes / no: probabilities for `["false", "true"]`), `score` (a 0-5 scale) and
//! `choice` (2-16 options labelled A-P, up to 256 with the "wide labels" Q-Z, AA, AB, ...).
//! The prompt (template `bare-v1`) is tokenized as one string without special tokens:
//!
//! ```text
//! [kind] choice
//! [state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
//! [question] Supplier response for this scenario.
//! [options]
//! A) issue_warning
//! B) renegotiate
//! [decision]:
//! ```
//!
//! (`noul` and `score` list their verbalizers, `false` / `true` and `0` ... `5`, as the option
//! lines). Two readouts of the last token's final-norm hidden state exist in the family:
//! - a slot head (`head.safetensors`: `proj.weight [slots, H]`, `proj.bias`), with the slot ranges
//!   of each kind in `judge_config.json`; the backbone runs with the unmerged LoRA in `adapter/`;
//! - the LM head (`adapter_vllm/decision_head.json`: verbalizer token ids and a bias): the logits
//!   of the verbalizer tokens plus the bias. With merged weights (JEV-35B) there is no adapter;
//!   otherwise `adapter_vllm/` holds the backbone LoRA and an `lm_head` LoRA.
//!
//! Either way the logits are divided by the per-kind temperature of `calibration.json` and soft-
//! maxed over the options. Choices beyond the trained slots read the LM-head logits of the wide
//! labels (bias 0), as the reference server does; with the slot head, the trained labels keep the
//! head's logits (the `lm_head` LoRA of `adapter_vllm/` re-expresses exactly those rows).
//!
//! The JSON wire format ([`DecisionRequest::from_json`], [`Decision::to_json`]) is the one of the
//! reference server's `POST /v1/decide`.

use crate::lora::LoraAdapter;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokenizers::Tokenizer;

/// Most options a `choice` takes (wide labels).
pub const MAX_OPTIONS: usize = 256;

/// The files of a decision bundle, fetched by `EngineOptions::decision_from` (no backbone).
pub const ASSET_FILES: &[&str] = &[
    "config.json",
    "judge_config.json",
    "calibration.json",
    "head.safetensors",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "adapter_vllm/decision_head.json",
];

/// The LM-head readout's adapter, fetched only when the bundle has no slot head.
pub const LM_ADAPTER_FILES: &[&str] = &["adapter_vllm/adapter_config.json", "adapter_vllm/adapter_model.safetensors"];

/// Whether `dir` holds decision assets.
pub fn has_assets(dir: &Path) -> bool {
    dir.is_dir() && (dir.join("judge_config.json").is_file() || dir.join("adapter_vllm/decision_head.json").is_file())
}

/// The question type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecisionKind {
    /// Yes / no: probabilities for `["false", "true"]`.
    Noul,
    /// A 0-5 scale: probabilities for `"0"` ... `"5"`.
    Score,
    /// One of the request's options.
    Choice,
}

impl DecisionKind {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "noul" => Self::Noul,
            "score" => Self::Score,
            "choice" => Self::Choice,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Score => "score",
            Self::Choice => "choice",
        }
    }
    fn index(self) -> usize {
        self as usize
    }
}

/// A typed question about `state`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DecisionRequest {
    pub kind: DecisionKind,
    /// A string is used as is; other JSON values are serialized as Python's `json.dumps` writes
    /// them (an array mixes text parts and JSON values, joined without separators).
    pub state: Value,
    pub question: String,
    /// `choice`: 2-256 options (one line each). `noul` / `score`: empty, or their verbalizers.
    pub options: Vec<String>,
}

impl DecisionRequest {
    pub fn new(kind: DecisionKind, state: impl Into<Value>, question: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            kind,
            state: state.into(),
            question: question.into(),
            options,
        }
    }
    pub fn noul(state: impl Into<Value>, question: impl Into<String>) -> Self {
        Self::new(DecisionKind::Noul, state, question, Vec::new())
    }
    pub fn score(state: impl Into<Value>, question: impl Into<String>) -> Self {
        Self::new(DecisionKind::Score, state, question, Vec::new())
    }
    pub fn choice<S: Into<String>>(state: impl Into<Value>, question: impl Into<String>, options: impl IntoIterator<Item = S>) -> Self {
        Self::new(DecisionKind::Choice, state, question, options.into_iter().map(Into::into).collect())
    }

    /// A `/v1/decide` request body: `{"kind", "state", "question", "options"}`.
    pub fn from_json(v: &Value) -> Result<Self> {
        let o = v.as_object().context("decision request: expected a JSON object")?;
        let kind = o
            .get("kind")
            .and_then(Value::as_str)
            .context("decision request: missing \"kind\"")?;
        let kind = DecisionKind::parse(kind).with_context(|| format!("decision request: unknown kind {kind:?} (noul, score, choice)"))?;
        let question = o
            .get("question")
            .and_then(Value::as_str)
            .context("decision request: missing \"question\"")?;
        let options = match o.get("options") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a
                .iter()
                .map(|x| x.as_str().map(str::to_string).context("decision request: options must be strings"))
                .collect::<Result<_>>()?,
            Some(_) => bail!("decision request: \"options\" must be a list of strings"),
        };
        Ok(Self::new(
            kind,
            o.get("state").cloned().unwrap_or(Value::String(String::new())),
            question,
            options,
        ))
    }

    /// The state as it appears in the prompt.
    pub fn state_text(&self) -> Result<String> {
        state_text(&self.state)
    }
}

fn state_text(v: &Value) -> Result<String> {
    use crate::tools::qwen::py_json;
    Ok(match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut s = String::new();
            for p in parts {
                match p {
                    Value::String(t) => s.push_str(t),
                    Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("text") => {
                        s.push_str(o.get("text").and_then(Value::as_str).unwrap_or_default())
                    }
                    Value::Object(o) if o.contains_key("image") || o.get("type").and_then(Value::as_str) == Some("image_url") => {
                        bail!("image parts in the state are not supported")
                    }
                    other => s.push_str(&py_json(other)),
                }
            }
            s
        }
        other => py_json(other),
    })
}

/// How the options of a decision were read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Adaptation {
    /// Within the trained labels (all `noul` / `score`, choices up to the trained slots).
    Native,
    /// A choice with more options than trained slots, read through the extra labels.
    WideLabels,
}

impl Adaptation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::WideLabels => "wide-labels",
        }
    }
}

/// Which readout a decision bundle uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readout {
    /// The fp32 slot head (`head.safetensors`).
    SlotHead,
    /// Verbalizer logits of the LM head plus a bias (`adapter_vllm/decision_head.json`).
    LmHead,
}

/// Timings of a decision.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct DecisionStats {
    pub prompt_tokens: usize,
    /// The forward pass (prefill and readout).
    pub prefill_secs: f64,
    /// Everything, tokenization included.
    pub total_secs: f64,
}

impl DecisionStats {
    pub fn prefill_tok_s(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_secs.max(1e-9)
    }
}

/// A decision: one probability per option.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Decision {
    pub kind: DecisionKind,
    /// The options the probabilities refer to (the verbalizers for `noul` / `score`).
    pub options: Vec<String>,
    pub probabilities: Vec<f64>,
    /// The most probable option.
    pub best_index: usize,
    pub best: String,
    pub adaptation: Adaptation,
    pub readout: Readout,
    pub stats: DecisionStats,
}

impl Decision {
    /// `score`: the expected level, `sum(i * p_i)`.
    pub fn expected_score(&self) -> Option<f64> {
        (self.kind == DecisionKind::Score).then(|| self.probabilities.iter().enumerate().map(|(i, p)| i as f64 * p).sum())
    }

    /// The `/v1/decide` response body (`model` names the decision model).
    pub fn to_json(&self, model: &str, protocol: &str) -> Value {
        let n = self.stats.prompt_tokens;
        json!({
            "kind": self.kind.as_str(),
            "effective_kind": self.kind.as_str(),
            "options": self.options,
            "probabilities": self.probabilities,
            "choice_index": self.best_index,
            "choice": self.best,
            "adaptation": self.adaptation.as_str(),
            "protocol": protocol,
            "model": model,
            "usage": {"prompt_tokens": n, "completion_tokens": 0, "total_tokens": n},
            "num_model_requests": 1,
            "elapsed_seconds": self.stats.total_secs,
        })
    }
}

/// Slot layout of a bundle: the verbalizers and their token ids, and each kind's slot range.
#[derive(Clone, Debug)]
struct Slots {
    verbalizers: Vec<String>,
    ids: Vec<u32>,
    /// `[lo, hi)` per kind (noul, score, choice).
    ranges: [(usize, usize); 3],
}

fn parse_slots(v: &Value, ids: &Value, what: &str) -> Result<Slots> {
    let s = v.get("slots").with_context(|| format!("{what}: missing \"slots\""))?;
    let verbalizers: Vec<String> = s
        .get("verbalizers")
        .and_then(Value::as_array)
        .with_context(|| format!("{what}: missing slots.verbalizers"))?
        .iter()
        .map(|x| {
            x.as_str()
                .map(str::to_string)
                .with_context(|| format!("{what}: verbalizers must be strings"))
        })
        .collect::<Result<_>>()?;
    let ids: Vec<u32> = ids
        .as_array()
        .with_context(|| format!("{what}: missing verbalizer_ids"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|v| v as u32)
                .with_context(|| format!("{what}: verbalizer ids must be integers"))
        })
        .collect::<Result<_>>()?;
    if ids.len() != verbalizers.len() {
        bail!("{what}: {} verbalizer ids for {} verbalizers", ids.len(), verbalizers.len());
    }
    let mut ranges = [(0, 0); 3];
    for k in [DecisionKind::Noul, DecisionKind::Score, DecisionKind::Choice] {
        let r = s
            .get("ranges")
            .and_then(|r| r.get(k.as_str()))
            .and_then(Value::as_array)
            .with_context(|| format!("{what}: missing slots.ranges.{}", k.as_str()))?;
        let (lo, hi) = (r.first().and_then(Value::as_u64), r.get(1).and_then(Value::as_u64));
        let (Some(lo), Some(hi)) = (lo, hi) else {
            bail!("{what}: bad slot range for {}", k.as_str());
        };
        let (lo, hi) = (lo as usize, hi as usize);
        if lo >= hi || hi > ids.len() {
            bail!("{what}: slot range {lo}..{hi} of {} outside {} slots", k.as_str(), ids.len());
        }
        ranges[k.index()] = (lo, hi);
    }
    let want = |k: DecisionKind| -> Vec<String> {
        match k {
            DecisionKind::Noul => vec!["false".into(), "true".into()],
            _ => (0..6).map(|i| i.to_string()).collect(),
        }
    };
    for k in [DecisionKind::Noul, DecisionKind::Score] {
        let (lo, hi) = ranges[k.index()];
        if verbalizers[lo..hi] != want(k)[..] {
            bail!("{what}: unexpected {} verbalizers {:?}", k.as_str(), &verbalizers[lo..hi]);
        }
    }
    Ok(Slots { verbalizers, ids, ranges })
}

/// The decision assets of a model: readout, temperatures, labels (see the module docs).
pub struct DecisionAssets {
    /// Where they were loaded from.
    pub source: PathBuf,
    /// The decision model's name (`judge_config.json` `model_name`, or the directory name).
    pub model_name: String,
    /// The template version (`bare-v1`).
    pub template: String,
    pub readout: Readout,
    slots: Slots,
    /// Slot head `[slots, hidden]` and bias.
    head: Option<(Vec<f32>, Vec<f32>)>,
    /// LM-head readout bias per slot.
    lm_bias: Vec<f32>,
    /// Temperature per kind.
    temps: [f64; 3],
    /// Text before `[kind]` (Gemma bundles: `<bos>`).
    prefix: String,
    /// Choice labels and their token ids, trained ones first (up to [`MAX_OPTIONS`]).
    labels: Vec<(String, u32)>,
    /// The adapter switch when the backbone has a runtime LoRA.
    pub(crate) switch: Option<Arc<AtomicBool>>,
    /// That adapter, once attached.
    pub adapter: Option<AdapterInfo>,
    hidden: usize,
}

/// The runtime LoRA attached for decisions.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AdapterInfo {
    pub dir: PathBuf,
    pub rank: usize,
    pub scale: f32,
    /// Linear layers patched.
    pub modules: usize,
    /// Parameter bytes (f32).
    pub bytes: usize,
}

/// Bundle facts to compare with the base model before attaching.
pub struct BaseCheck {
    pub hidden_size: usize,
    pub layers: usize,
    pub vocab_size: usize,
}

impl DecisionAssets {
    /// Loads the assets in `dir` for a base model of `base` dimensions using `tok`; the adapter
    /// (if any) is returned for the caller to attach (`QwenModel::attach_lora`).
    pub fn load(dir: &Path, tok: &Tokenizer, base: &BaseCheck, gemma: bool) -> Result<(Self, Option<LoraAdapter>)> {
        let read_json = |name: &str| -> Result<Option<Value>> {
            let p = dir.join(name);
            if !p.is_file() {
                return Ok(None);
            }
            Ok(Some(
                serde_json::from_slice(&crate::fsx::read(&p)?).with_context(|| format!("parse {}", p.display()))?,
            ))
        };
        let judge = read_json("judge_config.json")?;
        let dhead = read_json("adapter_vllm/decision_head.json")?;
        let head_path = dir.join("head.safetensors");
        let has_head = head_path.is_file() && judge.is_some();
        let (readout, slots) = if has_head {
            let j = judge.as_ref().unwrap();
            (
                Readout::SlotHead,
                parse_slots(j, j.get("verbalizer_ids").unwrap_or(&Value::Null), "judge_config.json")?,
            )
        } else if let Some(d) = &dhead {
            (
                Readout::LmHead,
                parse_slots(
                    d,
                    d.get("verbalizer_ids").unwrap_or(&Value::Null),
                    "adapter_vllm/decision_head.json",
                )?,
            )
        } else {
            bail!(
                "{}: no decision readout (head.safetensors with judge_config.json, or adapter_vllm/decision_head.json)",
                dir.display()
            );
        };
        let n = slots.ids.len();

        // The base must match: hidden size, layers, vocabulary, verbalizer tokens.
        let mut checks: Vec<(&str, Option<u64>, usize)> = Vec::new();
        if let Some(j) = &judge {
            checks.push(("hidden_size", j.get("hidden_size").and_then(Value::as_u64), base.hidden_size));
        }
        if let Some(c) = read_json("config.json")? {
            let c = c.get("text_config").cloned().unwrap_or(c);
            checks.push(("hidden_size", c.get("hidden_size").and_then(Value::as_u64), base.hidden_size));
            checks.push(("num_hidden_layers", c.get("num_hidden_layers").and_then(Value::as_u64), base.layers));
            checks.push(("vocab_size", c.get("vocab_size").and_then(Value::as_u64), base.vocab_size));
        }
        for (what, theirs, ours) in checks {
            if let Some(t) = theirs
                && t as usize != ours
            {
                bail!(
                    "decision assets in {} are for another base model: {what} {t}, the loaded model has {ours}",
                    dir.display()
                );
            }
        }
        for (v, &id) in slots.verbalizers.iter().zip(&slots.ids) {
            let enc = tok.encode(v.as_str(), false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            if enc.get_ids() != [id] {
                bail!(
                    "decision assets in {}: verbalizer {v:?} is token {id} in the bundle but {:?} for this tokenizer (another base model?)",
                    dir.display(),
                    enc.get_ids()
                );
            }
            if id as usize >= base.vocab_size {
                bail!("verbalizer token {id} is outside the vocabulary ({})", base.vocab_size);
            }
        }

        let head = if readout == Readout::SlotHead {
            let st = crate::safetensors::SafeTensors::open(&head_path, true)?;
            let get = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
                let info = st.info(name).with_context(|| format!("{}: missing {name}", head_path.display()))?;
                let b = st.bytes(name)?;
                let v: Vec<f32> = match info.dtype.as_str() {
                    "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
                    "BF16" => b
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
                        .collect(),
                    d => bail!("{}: {name} has dtype {d}", head_path.display()),
                };
                Ok((v, info.shape.clone()))
            };
            let (w, ws) = get("proj.weight")?;
            let (b, bs) = match get("proj.bias") {
                Ok(x) => x,
                Err(_) => (vec![0.0; n], vec![n]),
            };
            if ws != [n, base.hidden_size] || bs != [n] {
                bail!(
                    "{}: head is {ws:?} + {bs:?}, expected [{n}, {}] for {n} slots and the model's hidden size",
                    head_path.display(),
                    base.hidden_size
                );
            }
            Some((w, b))
        } else {
            None
        };
        let lm_bias: Vec<f32> = match &dhead {
            Some(d) => {
                let b: Vec<f32> = d
                    .get("bias")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                    .unwrap_or_default();
                if readout == Readout::LmHead && b.len() != n {
                    bail!("adapter_vllm/decision_head.json: {} biases for {n} slots", b.len());
                }
                b
            }
            None => Vec::new(),
        };

        let mut temps = [1.0f64; 3];
        if let Some(c) = read_json("calibration.json")? {
            for k in [DecisionKind::Noul, DecisionKind::Score, DecisionKind::Choice] {
                if let Some(t) = c.get("per_kind").and_then(|p| p.get(k.as_str())).and_then(Value::as_f64) {
                    if !(t > 0.0 && t.is_finite()) {
                        bail!("calibration.json: bad temperature {t} for {}", k.as_str());
                    }
                    temps[k.index()] = t;
                }
            }
        }

        // The backbone adapter: `adapter/` for the slot head, `adapter_vllm/` (backbone and
        // lm_head LoRA) for the LM-head readout unless the weights are merged.
        let merged = judge
            .as_ref()
            .and_then(|j| j.get("weights_mode"))
            .and_then(Value::as_str)
            .is_some_and(|m| m == "merged")
            || dhead.as_ref().and_then(|d| d.get("merged")).and_then(Value::as_bool) == Some(true);
        let adapter_dir = match readout {
            Readout::SlotHead => {
                let sub = judge
                    .as_ref()
                    .and_then(|j| j.get("adapter_subfolder"))
                    .and_then(Value::as_str)
                    .unwrap_or("adapter");
                dir.join(sub)
            }
            Readout::LmHead => dir.join("adapter_vllm"),
        };
        let adapter = if crate::lora::is_adapter_dir(&adapter_dir) {
            let ad = LoraAdapter::load(&adapter_dir)?;
            let l = ad.layer_count();
            if l != base.layers {
                bail!(
                    "LoRA adapter {} covers {l} layers, the loaded model has {}: the adapter is for another base model",
                    adapter_dir.display(),
                    base.layers
                );
            }
            Some(ad)
        } else if merged {
            None
        } else {
            bail!(
                "{}: the decision weights are unmerged but there is no adapter in {}",
                dir.display(),
                adapter_dir.display()
            );
        };

        let template = judge
            .as_ref()
            .or(dhead.as_ref())
            .and_then(|j| j.get("slots"))
            .and_then(|s| s.get("template_version"))
            .and_then(Value::as_str)
            .unwrap_or("bare-v1")
            .to_string();
        if !template.starts_with("bare-v1") {
            bail!("{}: template {template:?} is not supported (bare-v1)", dir.display());
        }
        let model_name = judge
            .as_ref()
            .and_then(|j| j.get("model_name"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| dir.file_name().map(|s| s.to_string_lossy().replace("--", "/")).unwrap_or_default());
        let prefix = match dhead.as_ref().and_then(|d| d.get("prefix_token")).and_then(Value::as_str) {
            Some(p) => p.to_string(),
            None if gemma => "<bos>".into(),
            None => String::new(),
        };
        let labels = choice_labels(tok, &slots)?;
        let switch = adapter.as_ref().map(LoraAdapter::switch);
        Ok((
            Self {
                source: dir.to_path_buf(),
                model_name,
                template,
                readout,
                slots,
                head,
                lm_bias,
                temps,
                prefix,
                labels,
                switch,
                adapter: None,
                hidden: base.hidden_size,
            },
            adapter,
        ))
    }

    /// The `protocol` field of responses.
    pub fn protocol(&self) -> String {
        if self.template == "bare-v1" {
            "jev27-bare-v1".into()
        } else {
            format!("jev-{}", self.template)
        }
    }

    /// Choice options the trained slots cover.
    pub fn native_choice_options(&self) -> usize {
        let (lo, hi) = self.slots.ranges[DecisionKind::Choice.index()];
        hi - lo
    }

    /// Most options a choice takes.
    pub fn max_choice_options(&self) -> usize {
        self.labels.len().max(self.native_choice_options())
    }

    /// Temperatures per kind (noul, score, choice).
    pub fn temperatures(&self) -> [f64; 3] {
        self.temps
    }

    /// Validates `req` and renders its prompt.
    pub fn plan(&self, req: &DecisionRequest) -> Result<Plan> {
        let k = req.kind;
        let (lo, hi) = self.slots.ranges[k.index()];
        let native = hi - lo;
        let (options, lines, adaptation) = match k {
            DecisionKind::Choice => {
                let n = req.options.len();
                let max = self.max_choice_options();
                if n < 2 || n > max {
                    bail!("choice needs 2-{max} options, got {n}");
                }
                if n > native && self.labels.len() < n {
                    bail!(
                        "choice with {n} options: this tokenizer has single-token labels for {}",
                        self.labels.len()
                    );
                }
                let lines: Vec<String> = req
                    .options
                    .iter()
                    .enumerate()
                    .map(|(i, o)| format!("{}) {o}", self.label(i)))
                    .collect();
                let a = if n > native { Adaptation::WideLabels } else { Adaptation::Native };
                (req.options.clone(), lines, a)
            }
            _ => {
                let v = self.slots.verbalizers[lo..hi].to_vec();
                if !req.options.is_empty() && req.options != v {
                    bail!("{} options must be {v:?} (or left out)", k.as_str());
                }
                (v.clone(), v, Adaptation::Native)
            }
        };
        let state = req.state_text()?;
        let prompt = format!(
            "{}[kind] {}\n[state] {state}\n[question] {}\n[options]\n{}\n[decision]:",
            self.prefix,
            k.as_str(),
            req.question,
            lines.join("\n")
        );
        Ok(Plan {
            kind: k,
            prompt,
            options,
            adaptation,
        })
    }

    /// The label of choice option `i`.
    fn label(&self, i: usize) -> &str {
        let (lo, hi) = self.slots.ranges[DecisionKind::Choice.index()];
        if lo + i < hi {
            &self.slots.verbalizers[lo + i]
        } else {
            &self.labels[i].0
        }
    }

    /// Whether the readout of `plan` needs the LM-head logits.
    pub fn needs_logits(&self, plan: &Plan) -> bool {
        self.readout == Readout::LmHead || plan.adaptation == Adaptation::WideLabels
    }

    /// The probabilities of `plan`'s options from the last token's final-norm hidden state
    /// (and the LM-head logits when [`DecisionAssets::needs_logits`]).
    pub fn read(&self, plan: &Plan, hidden: &[f32], logits: Option<&[f32]>) -> Result<Vec<f64>> {
        assert_eq!(hidden.len(), self.hidden);
        let k = plan.kind;
        let (lo, hi) = self.slots.ranges[k.index()];
        let n = plan.options.len();
        let logit = |id: u32| -> Result<f64> {
            let l = logits.context("readout needs the LM-head logits")?;
            Ok(l[id as usize] as f64)
        };
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            let slot = lo + i;
            let v = if slot < hi {
                match (&self.head, self.readout) {
                    (Some((w, b)), Readout::SlotHead) => {
                        let row = &w[slot * self.hidden..(slot + 1) * self.hidden];
                        row.iter().zip(hidden).map(|(&a, &x)| a as f64 * x as f64).sum::<f64>() + b[slot] as f64
                    }
                    _ => logit(self.slots.ids[slot])? + self.lm_bias[slot] as f64,
                }
            } else {
                // Wide labels: the LM-head logit, no bias.
                logit(self.labels[i].1)?
            };
            z.push(v / self.temps[k.index()]);
        }
        let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        Ok(e.into_iter().map(|v| v / s).collect())
    }

    /// Builds the decision from `plan` and its probabilities.
    pub fn decision(&self, plan: Plan, probabilities: Vec<f64>, stats: DecisionStats) -> Decision {
        let best_index = probabilities
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |b, (i, &p)| if p > b.1 { (i, p) } else { b })
            .0;
        Decision {
            kind: plan.kind,
            best: plan.options[best_index].clone(),
            options: plan.options,
            probabilities,
            best_index,
            adaptation: plan.adaptation,
            readout: self.readout,
            stats,
        }
    }

    /// The `GET /v1/decide/info` body.
    pub fn info_json(&self) -> Value {
        json!({
            "protocol": self.protocol(),
            "model": self.model_name,
            "readout": match self.readout { Readout::SlotHead => "slot-head", Readout::LmHead => "lm-head" },
            "adapter": self.adapter.as_ref().map(|a| json!({"dir": a.dir.display().to_string(), "rank": a.rank, "modules": a.modules})),
            "max_options": self.max_choice_options(),
            "native_choice_options": self.native_choice_options(),
            "temperatures": {"noul": self.temps[0], "score": self.temps[1], "choice": self.temps[2]},
            "source": self.source.display().to_string(),
        })
    }
}

/// A validated request: the prompt and the options it reads.
#[derive(Clone, Debug)]
pub struct Plan {
    pub kind: DecisionKind,
    pub prompt: String,
    pub options: Vec<String>,
    pub adaptation: Adaptation,
}

/// Choice labels beyond the trained slots, as the reference server picks them: A-Z then AA-ZZ,
/// keeping those that are one token on their own and inside an option line (`"x\n{L}) y"`). The
/// first ones must be the trained labels; otherwise only the trained slots are offered.
fn choice_labels(tok: &Tokenizer, slots: &Slots) -> Result<Vec<(String, u32)>> {
    let (lo, hi) = slots.ranges[DecisionKind::Choice.index()];
    let enc = |s: &str| -> Result<Vec<u32>> { Ok(tok.encode(s, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().to_vec()) };
    let letters: Vec<char> = ('A'..='Z').collect();
    let cands = letters
        .iter()
        .map(|c| c.to_string())
        .chain(letters.iter().flat_map(|a| letters.iter().map(move |b| format!("{a}{b}"))));
    let mut out: Vec<(String, u32)> = Vec::new();
    for lab in cands {
        if out.len() == MAX_OPTIONS {
            break;
        }
        let ids = enc(&lab)?;
        if ids.len() == 1 && enc(&format!("x\n{lab}) y"))?.contains(&ids[0]) {
            out.push((lab, ids[0]));
        }
    }
    let trained: Vec<(String, u32)> = slots.verbalizers[lo..hi]
        .iter()
        .cloned()
        .zip(slots.ids[lo..hi].iter().cloned())
        .collect();
    if out.len() < trained.len() || out[..trained.len()] != trained[..] {
        // The bundle's own labels (wide heads list all of theirs) and nothing beyond them.
        return Ok(trained);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_json() {
        let v: Value =
            serde_json::from_str(r#"{"kind":"choice","state":{"sku":"AX-330","stock":0.08},"question":"Q?","options":["a","b"]}"#).unwrap();
        let r = DecisionRequest::from_json(&v).unwrap();
        assert_eq!(r.kind, DecisionKind::Choice);
        assert_eq!(r.state_text().unwrap(), r#"{"sku": "AX-330", "stock": 0.08}"#);
        assert_eq!(r.options, ["a", "b"]);
        let v: Value = serde_json::from_str(r#"{"kind":"noul","state":"s","question":"q"}"#).unwrap();
        let r = DecisionRequest::from_json(&v).unwrap();
        assert!(r.options.is_empty() && r.state_text().unwrap() == "s");
        assert!(DecisionRequest::from_json(&serde_json::json!({"kind": "maybe", "question": "q"})).is_err());
    }

    #[test]
    fn slots_parse() {
        let j: Value = serde_json::from_str(
            r#"{"slots":{"ranges":{"noul":[0,2],"score":[2,8],"choice":[8,10]},
                "verbalizers":["false","true","0","1","2","3","4","5","A","B"]},
                "verbalizer_ids":[1,2,3,4,5,6,7,8,9,10]}"#,
        )
        .unwrap();
        let s = parse_slots(&j, &j["verbalizer_ids"], "t").unwrap();
        assert_eq!(s.ranges, [(0, 2), (2, 8), (8, 10)]);
    }
}
