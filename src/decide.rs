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
//! Images (Qwen3.5-family bases with their vision tower): the state is a list mixing text and
//! images (`["Photo: ", {"image": "data:image/png;base64,..."}, "\nTask: ..."]`), and each image
//! goes where it appears, as `<|vision_start|><|image_pad|>...<|vision_end|>` (one pad per merged
//! vision token, M-RoPE positions as in chat). There is no chat template around the prompt: the
//! reference server renders its parts with a raw template (text as is, an image as its
//! placeholder) and reads the same verbalizer logits. The readout, adapter and temperatures are
//! those of the text decisions; the heads never saw an image in training (zero-shot).
//!
//! The JSON wire format ([`DecisionRequest::from_json`], [`Decision::to_json`]) is the one of the
//! reference server's `POST /v1/decide`.

use crate::engine::Attachments;
use crate::lora::LoraAdapter;
use crate::video::RgbImage;
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

/// An image of a decision's state.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageSource {
    /// A local file.
    Path(PathBuf),
    /// An encoded image (PNG, JPEG, WebP, GIF, BMP).
    Bytes(Vec<u8>),
    /// An `http://` / `https://` URL, fetched when the decision runs (`hf` feature).
    Url(String),
    /// A decoded picture (the `oxideav` feature).
    #[cfg(feature = "oxideav")]
    Frame(Box<crate::av::Picture>),
}

impl ImageSource {
    /// An image reference as the reference server takes it: a `data:` URL (base64) or an
    /// `http(s)://` URL; also (oxinfer only) a `file://` URL or a local path.
    pub fn parse(s: &str) -> Result<Self> {
        if let Some(d) = s.strip_prefix("data:") {
            let (_, b64) = d
                .split_once(";base64,")
                .context("image data URL: only base64 data URLs are supported")?;
            return Ok(Self::Bytes(crate::b64::decode(b64).context("image data URL: bad base64")?));
        }
        if s.starts_with("http://") || s.starts_with("https://") {
            return Ok(Self::Url(s.to_string()));
        }
        if s.is_empty() {
            bail!("empty image reference");
        }
        Ok(Self::Path(PathBuf::from(s.strip_prefix("file://").unwrap_or(s))))
    }

    /// The encoded image (a decoded picture is encoded as PNG).
    pub fn bytes(&self) -> Result<Vec<u8>> {
        match self {
            Self::Path(p) => crate::fsx::read(p).with_context(|| format!("read image {}", p.display())),
            Self::Bytes(b) => Ok(b.clone()),
            #[cfg(feature = "hf")]
            Self::Url(u) => crate::hf::http_get(u),
            #[cfg(not(feature = "hf"))]
            Self::Url(u) => bail!("{u}: built without the `hf` feature, so image URLs cannot be fetched"),
            #[cfg(feature = "oxideav")]
            Self::Frame(f) => {
                let img = f.to_rgb()?;
                Ok(crate::png::encode_rgb(img.width, img.height, &img.data))
            }
        }
    }

    /// The images as RGB8, in order: decoded pictures converted, the others read and decoded
    /// (the `media` feature).
    pub fn to_rgb_all(images: &[ImageSource]) -> Result<Vec<RgbImage>> {
        let encoded = images
            .iter()
            .filter(|s| !s.is_frame())
            .map(|s| s.bytes())
            .collect::<Result<Vec<_>>>()?;
        let mut decoded = crate::mm::decode_images(&encoded)?.into_iter();
        images
            .iter()
            .map(|s| match s {
                #[cfg(feature = "oxideav")]
                Self::Frame(f) => f.to_rgb(),
                _ => Ok(decoded.next().expect("one decoded image per encoded one")),
            })
            .collect()
    }

    fn is_frame(&self) -> bool {
        #[cfg(feature = "oxideav")]
        if let Self::Frame(_) = self {
            return true;
        }
        false
    }

    /// The state part for this image (`{"image": ...}`; bytes as a base64 data URL, a decoded
    /// picture as a PNG data URL).
    pub fn to_part(&self) -> Value {
        let s = match self {
            Self::Path(p) => p.display().to_string(),
            Self::Bytes(b) => format!("data:application/octet-stream;base64,{}", crate::b64::encode(b)),
            Self::Url(u) => u.clone(),
            #[cfg(feature = "oxideav")]
            Self::Frame(_) => match self.bytes() {
                Ok(png) => format!("data:image/png;base64,{}", crate::b64::encode(&png)),
                Err(e) => format!("(unconvertible picture: {e})"),
            },
        };
        json!({ "image": s })
    }
}

/// A piece of a decision's state.
#[derive(Clone, Debug, PartialEq)]
pub enum StatePart {
    Text(String),
    Image(ImageSource),
}

/// A typed question about `state`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DecisionRequest {
    pub kind: DecisionKind,
    /// A string is used as is; other JSON values are serialized as Python's `json.dumps` writes
    /// them. An array mixes text parts, images and JSON values, joined without separators: a
    /// string or `{"type": "text", "text"}` is text, `{"image": REF}` or `{"type": "image_url",
    /// "image_url": {"url": REF}}` an image (see [`ImageSource::parse`]), anything else JSON.
    pub state: Value,
    pub question: String,
    /// `choice`: 2-256 options (one line each). `noul` / `score`: empty, or their verbalizers.
    pub options: Vec<String>,
    /// Image files (and decoded pictures, `image_frames`) placed at the start of the state,
    /// before its text, and the image token budget (`max_soft_tokens`, for the state's own images
    /// too; 0: the processor's default). Videos and audio are refused: the protocol takes images
    /// only.
    pub attachments: Attachments,
    /// How a choice with more options than the trained labels is read (JSON `"strategy"`).
    pub strategy: Strategy,
}

/// How a `choice` with more options than the trained labels (16) is read, as the reference
/// server's `strategy` field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Strategy {
    /// The bundle's default: `tournament` for Gemma bundles, `single` for Qwen ones (as the
    /// reference server's model profiles).
    #[default]
    Auto,
    /// One pass, the options past the trained labels read through the wide labels.
    Single,
    /// Groups of up to 16 options read natively, then a final of the group winners (topped up
    /// with the best others to 16), the group shares combined as the reference server does.
    Tournament,
}

impl Strategy {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "single" => Some(Self::Single),
            "tournament" => Some(Self::Tournament),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Single => "single",
            Self::Tournament => "tournament",
        }
    }
}

impl DecisionRequest {
    pub fn new(kind: DecisionKind, state: impl Into<Value>, question: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            kind,
            state: state.into(),
            question: question.into(),
            options,
            attachments: Attachments::default(),
            strategy: Strategy::Auto,
        }
    }
    /// With image files placed before the state text (see [`DecisionRequest::attachments`]).
    pub fn with_images<P: Into<PathBuf>>(mut self, images: impl IntoIterator<Item = P>) -> Self {
        self.attachments.images.extend(images.into_iter().map(Into::into));
        self
    }
    /// With decoded pictures placed before the state text, after the image files (the `oxideav`
    /// feature).
    #[cfg(feature = "oxideav")]
    pub fn with_image_frames(mut self, images: impl IntoIterator<Item = crate::av::Picture>) -> Self {
        self.attachments.image_frames.extend(images);
        self
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

    /// A `/v1/decide` request body: `{"kind", "state", "question", "options"}`, and (oxinfer
    /// only) `"attachments"` as [`Attachments::from_json`] takes them.
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
        let mut r = Self::new(
            kind,
            o.get("state").cloned().unwrap_or(Value::String(String::new())),
            question,
            options,
        );
        if let Some(a) = o.get("attachments").filter(|a| !a.is_null()) {
            r.attachments = Attachments::from_json(a).context("decision request")?;
        }
        if let Some(st) = o.get("strategy").filter(|v| !v.is_null()) {
            let st = st.as_str().context("decision request: \"strategy\" must be a string")?;
            r.strategy = Strategy::parse(st)
                .with_context(|| format!("decision request: strategy {st:?} is not supported (auto, single, tournament)"))?;
        }
        Ok(r)
    }

    /// The state as it appears in the prompt; an error when it has images (see
    /// [`DecisionRequest::state_parts`]).
    pub fn state_text(&self) -> Result<String> {
        let mut s = String::new();
        for p in self.state_parts()? {
            match p {
                StatePart::Text(t) => s.push_str(&t),
                StatePart::Image(_) => bail!("the state has images (see state_parts)"),
            }
        }
        Ok(s)
    }

    /// The state as text and images in prompt order: the attachments' images first (files, then
    /// decoded pictures), then the state (adjacent text merged).
    pub fn state_parts(&self) -> Result<Vec<StatePart>> {
        use crate::tools::qwen::py_json;
        let mut out: Vec<StatePart> = self
            .attachments
            .images
            .iter()
            .map(|p| StatePart::Image(ImageSource::Path(p.clone())))
            .collect();
        #[cfg(feature = "oxideav")]
        out.extend(
            self.attachments
                .image_frames
                .iter()
                .map(|f| StatePart::Image(ImageSource::Frame(Box::new(f.clone())))),
        );
        let text = |out: &mut Vec<StatePart>, t: &str| match out.last_mut() {
            Some(StatePart::Text(s)) => s.push_str(t),
            _ => out.push(StatePart::Text(t.to_string())),
        };
        let parts = match &self.state {
            Value::Array(parts) => parts,
            Value::String(s) => {
                text(&mut out, s);
                return Ok(out);
            }
            other => {
                text(&mut out, &py_json(other));
                return Ok(out);
            }
        };
        for p in parts {
            let Value::Object(o) = p else {
                match p {
                    Value::String(t) => text(&mut out, t),
                    other => text(&mut out, &py_json(other)),
                }
                continue;
            };
            let image = match (o.get("image"), o.get("type").and_then(Value::as_str)) {
                (Some(Value::String(s)), _) => Some(s.as_str()),
                (Some(_), _) => bail!("decision state: \"image\" must be a string (a data URL, URL or path)"),
                (None, Some("image_url")) => Some(
                    match o.get("image_url") {
                        Some(Value::String(s)) => Some(s.as_str()),
                        Some(u) => u.get("url").and_then(Value::as_str),
                        None => None,
                    }
                    .context("decision state: image_url part without a url")?,
                ),
                _ => None,
            };
            match image {
                Some(s) => out.push(StatePart::Image(ImageSource::parse(s)?)),
                None if o.get("type").and_then(Value::as_str) == Some("text") => {
                    text(&mut out, o.get("text").and_then(Value::as_str).unwrap_or_default())
                }
                None => text(&mut out, &py_json(p)),
            }
        }
        Ok(out)
    }
}

/// How the options of a decision were read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Adaptation {
    /// Within the trained labels (all `noul` / `score`, choices up to the trained slots).
    Native,
    /// A choice with more options than trained slots, read through the extra labels.
    WideLabels,
    /// A choice with more options than trained slots, read in groups ([`Strategy::Tournament`]).
    Tournament,
}

impl Adaptation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::WideLabels => "wide-labels",
            Self::Tournament => "tournament",
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
    /// Prompt tokens, image tokens included.
    pub prompt_tokens: usize,
    /// Images read, and their vision tokens.
    pub images: usize,
    pub image_tokens: usize,
    /// Fetching, decoding and encoding the images (vision tower).
    pub vision_secs: f64,
    /// The forward pass (prefill and readout).
    pub prefill_secs: f64,
    /// Everything, tokenization included.
    pub total_secs: f64,
    /// Forward passes (more than one for a tournament).
    pub passes: usize,
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
            "num_model_requests": self.stats.passes.max(1),
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
    /// Slot-head logits are soft-capped as `cap * tanh(z / cap)` (Gemma bundles: 30, like the
    /// model's own logits; `judge_config.json` `softcap`).
    softcap: Option<f64>,
    /// [`Strategy::Auto`] means a tournament (Gemma bundles, as the reference server's profile).
    tournament_default: bool,
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
            .unwrap_or_else(|| bundle_name(dir));
        let prefix = match dhead.as_ref().and_then(|d| d.get("prefix_token")).and_then(Value::as_str) {
            Some(p) => p.to_string(),
            None if gemma => "<bos>".into(),
            None => String::new(),
        };
        let labels = choice_labels(tok, &slots)?;
        let softcap = judge
            .as_ref()
            .and_then(|j| j.get("softcap"))
            .and_then(Value::as_f64)
            .filter(|c| *c > 0.0);
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
                softcap,
                tournament_default: gemma,
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

    /// Most options a choice takes (single passes need a label for each; tournaments read at most
    /// the trained labels per pass).
    pub fn max_choice_options(&self) -> usize {
        if self.tournament_default {
            return MAX_OPTIONS;
        }
        self.labels.len().max(self.native_choice_options())
    }

    /// The strategy `s` stands for with these assets ([`Strategy::Auto`]: the bundle's default).
    pub fn strategy(&self, s: Strategy) -> Strategy {
        match s {
            Strategy::Auto if self.tournament_default => Strategy::Tournament,
            Strategy::Auto => Strategy::Single,
            s => s,
        }
    }

    /// Whether `req` is read as a tournament (a choice with more options than trained labels).
    pub fn is_tournament(&self, req: &DecisionRequest) -> bool {
        req.kind == DecisionKind::Choice
            && req.options.len() > self.native_choice_options()
            && self.strategy(req.strategy) == Strategy::Tournament
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
        let a = &req.attachments;
        #[cfg(feature = "oxideav")]
        let frames = !a.video_frames.is_empty() || !a.audio_frames.is_empty();
        #[cfg(not(feature = "oxideav"))]
        let frames = false;
        if !a.videos.is_empty() || !a.audios.is_empty() || frames {
            bail!("decisions take images only (the /v1/decide protocol has no video or audio parts)");
        }
        // Text around the images: pieces[i] comes before image i, the last one after them all.
        let mut pieces = vec![format!("{}[kind] {}\n[state] ", self.prefix, k.as_str())];
        let mut images = Vec::new();
        for p in req.state_parts()? {
            match p {
                StatePart::Text(t) => pieces.last_mut().unwrap().push_str(&t),
                StatePart::Image(src) => {
                    images.push(src);
                    pieces.push(String::new());
                }
            }
        }
        pieces.last_mut().unwrap().push_str(&format!(
            "\n[question] {}\n[options]\n{}\n[decision]:",
            req.question,
            lines.join("\n")
        ));
        let prompt = pieces.join(IMAGE_PLACEHOLDER);
        Ok(Plan {
            kind: k,
            prompt,
            pieces,
            images,
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
                        let z = row.iter().zip(hidden).map(|(&a, &x)| a as f64 * x as f64).sum::<f64>() + b[slot] as f64;
                        self.softcap.map_or(z, |c| c * (z / c).tanh())
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

/// The tournament groups of `n` options: `ceil(n / k)` contiguous groups (`k` trained labels) of near-equal size.
pub fn tournament_groups(n: usize, k: usize) -> Vec<Vec<usize>> {
    let g = n.div_ceil(k);
    let (base, extra) = (n / g, n % g);
    let mut out = Vec::with_capacity(g);
    let mut i = 0;
    for j in 0..g {
        let size = base + usize::from(j < extra);
        out.push((i..i + size).collect());
        i += size;
    }
    out
}

/// The tournament's final options from the in-group probabilities (`p[o]` of option `o`):
/// each group's best, topped up with the best of the others to the trained label count, in
/// option order.
pub fn tournament_final(groups: &[Vec<usize>], p: &[f64], k: usize) -> Vec<usize> {
    let best = |g: &Vec<usize>| g.iter().copied().fold(g[0], |b, o| if p[o] > p[b] { o } else { b });
    let chosen: Vec<usize> = groups.iter().map(best).collect();
    let mut rest: Vec<usize> = groups.iter().flatten().copied().filter(|o| !chosen.contains(o)).collect();
    rest.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    let mut fin = chosen;
    fin.extend(rest.into_iter().take(k.saturating_sub(groups.len())));
    fin.sort_unstable();
    fin
}

/// The tournament's probabilities: final-round options get `final * among`, the others their
/// group's final share times their in-group probability (the reference server's formula).
pub fn tournament_combine(groups: &[Vec<usize>], in_group: &[f64], fin: &[usize], final_p: &[f64]) -> Vec<f64> {
    let n = in_group.len();
    let mut group_of = vec![0; n];
    for (gi, g) in groups.iter().enumerate() {
        for &o in g {
            group_of[o] = gi;
        }
    }
    let (mut share, mut cap) = (vec![0.0; groups.len()], vec![0.0; groups.len()]);
    for (&f, &q) in fin.iter().zip(final_p) {
        share[group_of[f]] += q;
        cap[group_of[f]] += in_group[f];
    }
    let among: f64 = share.iter().zip(&cap).map(|(a, b)| a * b).sum();
    let mut p: Vec<f64> = (0..n)
        .map(|o| match fin.iter().position(|&f| f == o) {
            Some(i) => final_p[i] * among,
            None => share[group_of[o]] * in_group[o],
        })
        .collect();
    let s: f64 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= s);
    p
}

/// An image in [`Plan::prompt`] before its expansion to one pad per vision token (the reference
/// server's raw template, Qwen3.5 family).
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// A validated request: the prompt and the options it reads.
#[derive(Clone, Debug)]
pub struct Plan {
    pub kind: DecisionKind,
    /// The prompt, with [`IMAGE_PLACEHOLDER`] for each image.
    pub prompt: String,
    /// The prompt's text around the images (`images.len() + 1` pieces).
    pub pieces: Vec<String>,
    pub images: Vec<ImageSource>,
    pub options: Vec<String>,
    pub adaptation: Adaptation,
}

/// A bundle's name from its directory: `org/name` for a cache directory (`org--name`), also for
/// a subfolder of one (`autotrust--JEV-9B/vl`: `autotrust/JEV-9B`).
fn bundle_name(dir: &Path) -> String {
    let name = |p: &Path| p.file_name().map(|s| s.to_string_lossy().to_string());
    let own = name(dir).unwrap_or_default();
    if !own.contains("--")
        && let Some(parent) = dir.parent().and_then(name)
        && parent.contains("--")
    {
        return parent.replace("--", "/");
    }
    own.replace("--", "/")
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
    fn tournament_rounds() {
        let g = tournament_groups(20, 16);
        assert_eq!(g, [(0..10).collect::<Vec<_>>(), (10..20).collect()]);
        assert_eq!(tournament_groups(33, 16).iter().map(Vec::len).collect::<Vec<_>>(), [11, 11, 11]);
        // In-group probabilities: option 3 leads group 0, option 12 group 1.
        let mut p = vec![0.01; 20];
        p[3] = 0.6;
        p[12] = 0.5;
        p[5] = 0.2;
        let fin = tournament_final(&g, &p, 16);
        assert_eq!(fin.len(), 16);
        assert!(fin.contains(&3) && fin.contains(&12) && fin.contains(&5));
        let final_p: Vec<f64> = fin.iter().map(|&o| if o == 12 { 0.7 } else { 0.3 / 15.0 }).collect();
        let out = tournament_combine(&g, &p, &fin, &final_p);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(
            out.iter()
                .enumerate()
                .fold((0, 0.0), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                .0,
            12
        );
        let v = json!({"kind": "choice", "question": "q", "options": ["a", "b"], "strategy": "tournament"});
        assert_eq!(DecisionRequest::from_json(&v).unwrap().strategy, Strategy::Tournament);
        assert!(DecisionRequest::from_json(&json!({"kind": "noul", "question": "q", "strategy": "permute"})).is_err());
    }

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
    fn state_images() {
        let v = json!({"kind": "noul", "question": "q", "state": [
            "Photo: ", {"image": "data:image/png;base64,Zm9vYmFy"}, "\nTask", {"a": 1},
            {"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}, {"type": "text", "text": "!"},
            {"image": "file:///tmp/x.png"}]});
        let r = DecisionRequest::from_json(&v).unwrap().with_images(["/tmp/first.png"]);
        let parts = r.state_parts().unwrap();
        assert_eq!(
            parts,
            [
                StatePart::Image(ImageSource::Path("/tmp/first.png".into())),
                StatePart::Text("Photo: ".into()),
                StatePart::Image(ImageSource::Bytes(b"foobar".to_vec())),
                StatePart::Text("\nTask{\"a\": 1}".into()),
                StatePart::Image(ImageSource::Url("https://example.com/x.jpg".into())),
                StatePart::Text("!".into()),
                StatePart::Image(ImageSource::Path("/tmp/x.png".into())),
            ]
        );
        assert!(r.state_text().is_err());
        assert_eq!(
            ImageSource::parse(ImageSource::Bytes(b"fo".to_vec()).to_part()["image"].as_str().unwrap()).unwrap(),
            ImageSource::Bytes(b"fo".to_vec())
        );
        assert!(ImageSource::parse("data:image/png,raw").is_err());
        let v = json!({"kind": "noul", "question": "q", "state": "s", "attachments": {"images": ["a.png"], "max_soft_tokens": 64}});
        let r = DecisionRequest::from_json(&v).unwrap();
        assert_eq!((r.attachments.images.len(), r.attachments.max_soft_tokens), (1, 64));
        assert_eq!(bundle_name(Path::new("/c/hub/autotrust--JEV-9B/vl")), "autotrust/JEV-9B");
        assert_eq!(bundle_name(Path::new("/c/hub/autotrust--JEV-27B-VL")), "autotrust/JEV-27B-VL");
    }

    #[cfg(feature = "oxideav")]
    #[test]
    fn state_image_frames() {
        // Decoded pictures follow the image files, before the state; they convert without a
        // decoder, and serialize as PNG data URLs.
        let pic = crate::av::Picture::from_rgb(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let r = DecisionRequest::noul("state", "q")
            .with_images(["/tmp/a.png"])
            .with_image_frames([pic.clone()]);
        let frame = ImageSource::Frame(Box::new(pic));
        assert_eq!(
            r.state_parts().unwrap(),
            [
                StatePart::Image(ImageSource::Path("/tmp/a.png".into())),
                StatePart::Image(frame.clone()),
                StatePart::Text("state".into()),
            ]
        );
        let rgb = ImageSource::to_rgb_all(std::slice::from_ref(&frame)).unwrap();
        assert_eq!((rgb[0].width, rgb[0].height, &rgb[0].data), (2, 1, &vec![1, 2, 3, 4, 5, 6]));
        let part = frame.to_part();
        assert!(part["image"].as_str().unwrap().starts_with("data:image/png;base64,"));
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
