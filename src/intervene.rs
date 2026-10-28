//! Runtime interventions on a running model (the weights are never changed): edits of chosen
//! MLP neurons, steering vectors added to the residual stream, directions projected out of it
//! (runtime "abliteration"), sparse-autoencoder feature steering and inspection, and capture of
//! residual activations to build such vectors (difference of means between two prompt sets).
//!
//! A user describes what to do with a [`Spec`] (serde / JSON, see the README), which is compiled
//! against a loaded model into a [`Compiled`] set of hooks; a session carries it in its
//! [`State`], so it can change between requests without reloading anything. The model families
//! call the hooks at fixed places of their forward pass:
//!
//! - residual points: point `2l` is the residual stream entering layer `l` (point 0: the
//!   embeddings), `2l + 1` the residual after layer `l`'s attention (or Gated DeltaNet) block
//!   was added, `2l + 2` the output of layer `l` (`hidden_states[l + 1]` in transformers);
//! - MLP neurons: the input of `down_proj` (`act(gate) * up`) of a dense MLP, of a MoE layer's
//!   shared expert, or of each routed expert.
//!
//! Everything runs on the CPU; the hooks that are cheap elementwise / reduction ops (steering,
//! ablation, neuron edits, JumpReLU / ReLU SAE features) also run inside the whole-pass GPU decode
//! graph ([`crate::gpu::decode`]), with their parameters in a buffer rewritten per step (so a
//! strength change does not rebuild anything). Capture and SAE inspection run on the CPU.

use crate::kernels::{dot, for_chunks};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub use crate::sae::Sae;

// ---- specification ------------------------------------------------------------------------

/// Which positions a hook applies to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Positions {
    /// Every position.
    #[default]
    All,
    /// Prompt tokens before the last one.
    Prompt,
    /// The last prompt token and every generated token: the positions whose outputs are sampled.
    Generated,
}

/// Where in a layer a residual hook sits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Point {
    /// The residual entering the layer (the previous layer's output; the embeddings for layer 0).
    Pre,
    /// After the attention (or Gated DeltaNet) block was added.
    Mid,
    /// The layer's output (`hidden_states[layer + 1]`).
    #[default]
    Post,
}

/// The residual point index of `point` of `layer` (see the module docs).
pub fn point_index(layer: usize, point: Point) -> usize {
    match point {
        Point::Pre => 2 * layer,
        Point::Mid => 2 * layer + 1,
        Point::Post => 2 * layer + 2,
    }
}

/// What a neuron edit does to the activation `a`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NeuronOp {
    /// `a * value`
    #[default]
    Scale,
    /// `0`
    Zero,
    /// `a + value` (a "dose")
    Add,
    /// `value`
    Set,
    /// `min(max(a, min), max)`
    Clamp,
}

/// The unit of a neuron edit's `value` / `dose` (add and set).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// The activation itself.
    #[default]
    Raw,
    /// `value * median_t |h_L,t| / |W_down[:, n]|`: the edit then moves the residual by `value`
    /// times the typical norm of layer L's output `h_L` (median over positions 1.. of the prompt,
    /// measured on the unedited model, or `resid_median` when given).
    ResidMedian,
}

/// Which MLP of a layer a neuron edit targets: a routed expert's id, or `"shared"` (a MoE
/// layer's shared expert). Omitted for dense MLPs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExpertRef {
    Id(usize),
    Name(String),
}

/// An edit of one MLP neuron (the input of `down_proj` at index `neuron`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeuronEdit {
    /// 0-based, as in the checkpoint's module names (`model.layers.{layer}.mlp`).
    pub layer: usize,
    pub neuron: usize,
    pub expert: Option<ExpertRef>,
    /// Default: `add` when `dose` is given, else `scale`.
    pub op: Option<NeuronOp>,
    pub value: Option<f32>,
    /// Same as `value` (with `op` defaulting to `add`).
    pub dose: Option<f32>,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub unit: Unit,
    /// With `unit: resid_median`: the median residual norm to use instead of measuring it.
    pub resid_median: Option<f32>,
    pub positions: Positions,
}

/// Adds `alpha * v` to the residual stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Steer {
    /// An inline vector (`hidden_size` values), applied at `layer`.
    pub vector: Option<Vec<f32>>,
    /// A vector file (see [`load_vectors`]): every layer it has, or only `layer` / `layers`.
    pub file: Option<String>,
    pub layer: Option<usize>,
    pub layers: Option<Vec<usize>>,
    pub point: Point,
    pub alpha: f32,
    /// Scale each vector to unit norm first (alpha is then the norm of what is added).
    pub normalize: bool,
    pub positions: Positions,
}

impl Default for Steer {
    fn default() -> Self {
        Steer {
            vector: None,
            file: None,
            layer: None,
            layers: None,
            point: Point::Post,
            alpha: 1.0,
            normalize: false,
            positions: Positions::All,
        }
    }
}

/// Projects a direction out of the residual stream (`x -= (x . r) r`, `r` unit) wherever a
/// block writes to it: the embeddings and after every attention and MLP block (Arditi et al.'s
/// directional ablation), in `layers` (default all).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Ablate {
    pub vector: Option<Vec<f32>>,
    pub file: Option<String>,
    /// Which of the file's vectors is the direction (required when it has several).
    pub direction_layer: Option<usize>,
    pub layers: Option<Vec<usize>>,
    pub positions: Positions,
}

/// What a feature edit does with SAE feature `f` (activation `a`, decoder row `d`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureOp {
    /// Set `a` to `value` (Neuronpedia's clamp): `x += (value - a) d`.
    #[default]
    Clamp,
    /// `x += (a * value - a) d`.
    Scale,
    /// `x += value * d`, whatever `a` is (Neuronpedia / SAELens steering).
    Add,
    /// Clamp to 0.
    Zero,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FeatureEdit {
    pub id: usize,
    pub op: FeatureOp,
    pub value: f32,
}

/// Feature steering with a sparse autoencoder. The rest of the residual (the SAE's error term
/// included) is kept: only `(a' - a) * W_dec[f]` is added.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SaeSteer {
    /// The SAE (see [`Sae::load`]): a directory, file, or `org/repo/path` on Hugging Face.
    pub sae: String,
    /// Override the SAE's hook layer (its output, `Point::Post`).
    pub layer: Option<usize>,
    pub features: Vec<FeatureEdit>,
    pub positions: Positions,
}

/// Records the `top_k` most active SAE features of every position ([`State::features`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SaeInspect {
    pub sae: String,
    pub layer: Option<usize>,
    pub top_k: usize,
}

/// Everything to apply while the model runs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Spec {
    pub neurons: Vec<NeuronEdit>,
    pub steer: Vec<Steer>,
    pub ablate: Vec<Ablate>,
    pub sae: Vec<SaeSteer>,
    pub sae_inspect: Vec<SaeInspect>,
}

impl Spec {
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty() && self.steer.is_empty() && self.ablate.is_empty() && self.sae.is_empty() && self.sae_inspect.is_empty()
    }

    /// Parses JSON (an object with the fields of [`Spec`]).
    pub fn from_json(text: &str) -> Result<Spec> {
        serde_json::from_str(text).context("intervention spec")
    }

    /// Whether some edit needs the median residual norm measured on a prompt
    /// ([`Unit::ResidMedian`] without `resid_median`).
    pub fn needs_calibration(&self) -> bool {
        self.neurons.iter().any(|n| n.unit == Unit::ResidMedian && n.resid_median.is_none())
    }

    /// Adds a neuron edit in the compact form `LAYER:NEURON[@EXPERT|@shared],OP[=VALUE][,key=value...]`,
    /// e.g. `14:8949,dose=1,unit=resid_median`, `3:120,zero`, `3:120,scale=2.5`,
    /// `3:120,clamp=-1:1`, `5:7@12,set=0` (keys: `scale`, `zero`, `add`, `dose`, `set`, `clamp`,
    /// `unit`, `resid_median`, `positions`).
    pub fn parse_neuron(s: &str) -> Result<NeuronEdit> {
        let mut parts = s.split(',');
        let head = parts.next().unwrap_or("").trim();
        let (ln, expert) = match head.split_once('@') {
            Some((a, e)) => (a, Some(e)),
            None => (head, None),
        };
        let (l, n) = ln
            .split_once(':')
            .with_context(|| format!("neuron edit {s:?}: expected LAYER:NEURON first"))?;
        let mut e = NeuronEdit {
            layer: l.trim().parse().with_context(|| format!("neuron edit {s:?}: layer"))?,
            neuron: n.trim().parse().with_context(|| format!("neuron edit {s:?}: neuron"))?,
            expert: expert.map(|x| match x.trim().parse() {
                Ok(i) => ExpertRef::Id(i),
                Err(_) => ExpertRef::Name(x.trim().to_string()),
            }),
            ..Default::default()
        };
        for kv in parts {
            let kv = kv.trim();
            if kv.is_empty() {
                continue;
            }
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            let num = |v: &str| -> Result<f32> { v.trim().parse().with_context(|| format!("neuron edit {s:?}: {k}={v:?}")) };
            match k.trim() {
                "scale" => (e.op, e.value) = (Some(NeuronOp::Scale), Some(num(v)?)),
                "zero" => e.op = Some(NeuronOp::Zero),
                "add" => (e.op, e.value) = (Some(NeuronOp::Add), Some(num(v)?)),
                "dose" => e.dose = Some(num(v)?),
                "set" => (e.op, e.value) = (Some(NeuronOp::Set), Some(num(v)?)),
                "clamp" => {
                    let (a, b) = v.split_once(':').with_context(|| format!("neuron edit {s:?}: clamp=MIN:MAX"))?;
                    e.op = Some(NeuronOp::Clamp);
                    e.min = (!a.trim().is_empty()).then(|| num(a)).transpose()?;
                    e.max = (!b.trim().is_empty()).then(|| num(b)).transpose()?;
                }
                "unit" => {
                    e.unit = serde_json::from_value(serde_json::Value::String(v.trim().into()))
                        .map_err(|_| anyhow!("neuron edit {s:?}: unit is raw or resid_median"))?
                }
                "resid_median" => e.resid_median = Some(num(v)?),
                "positions" => e.positions = parse_positions(v)?,
                other => bail!("neuron edit {s:?}: unknown key {other:?}"),
            }
        }
        Ok(e)
    }
}

/// `all`, `prompt` or `generated`.
pub fn parse_positions(v: &str) -> Result<Positions> {
    serde_json::from_value(serde_json::Value::String(v.trim().into()))
        .map_err(|_| anyhow!("positions: all, prompt or generated, got {v:?}"))
}

// ---- vector files ---------------------------------------------------------------------------

/// Per-layer vectors read from a file: `layer -> vector` where the vector belongs at that
/// layer's output ([`Point::Post`]).
#[derive(Clone, Debug, Default)]
pub struct Vectors {
    pub by_layer: BTreeMap<usize, Vec<f32>>,
    /// A single vector without a layer (a 1-D tensor alone in the file).
    pub single: Option<Vec<f32>>,
}

/// Reads vectors from a file:
/// - safetensors (oxinfer's format, as `--make-direction` writes it): 1-D tensors
///   `layers.{L}` (`hidden_size`, f32 / bf16 / f16) for the output of layer L (0-based), or one
///   2-D tensor `[n, hidden]` where `n = layers + 1` follows transformers' `hidden_states`
///   (row `i` is layer `i - 1`'s output; row 0, the embeddings, is dropped) and `n = layers` has
///   one row per layer output; a single 1-D tensor of another name is a layer-less direction;
/// - GGUF control vectors (llama.cpp `--control-vector`, repeng `export_gguf`): `direction.{N}`
///   is added to the output of layer N (0-based), as llama.cpp applies it;
/// - PyTorch `torch.save` state dicts with the same tensor names.
pub fn load_vectors(path: &Path, n_layers: usize, hidden: usize) -> Result<Vectors> {
    let bytes = std::fs::read(path).with_context(|| format!("{}", path.display()))?;
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let gguf = bytes.starts_with(b"GGUF");
    if gguf {
        let h = crate::gguf::parse_header(&bytes[..], bytes.len())?;
        for t in &h.tensors {
            let raw = bytes.get(t.start..t.end).context("gguf: tensor out of range")?;
            let v: Vec<f32> = match t.ggml_type {
                0 => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
                1 => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::f16::from_le_bytes(*c).to_f32())
                    .collect(),
                30 => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
                    .collect(),
                k => bail!("{}: tensor {} has ggml type {k} (expected F32/F16/BF16)", path.display(), t.name),
            };
            let shape: Vec<usize> = t.ne.iter().rev().copied().collect();
            tensors.push((t.name.clone(), shape, v));
        }
    } else if crate::torch::is_torch_zip(&bytes) {
        for (name, t) in crate::torch::read_tensors(&bytes)? {
            tensors.push((name, t.shape.clone(), decode_floats(t.dtype, &bytes[t.start..t.end])?));
        }
    } else {
        let st = crate::safetensors::SafeTensors::open(path, true)?;
        let names: Vec<String> = st.names().cloned().collect();
        for name in names {
            let info = st.info(&name).unwrap().clone();
            tensors.push((name.clone(), info.shape.clone(), decode_floats(&info.dtype, st.bytes(&name)?)?));
        }
    }
    let mut out = Vectors::default();
    let check = |v: &[f32], what: &str| -> Result<()> {
        if v.len() != hidden {
            bail!(
                "{}: {what} has {} values, the model's hidden size is {hidden}",
                path.display(),
                v.len()
            );
        }
        Ok(())
    };
    for (name, shape, v) in &tensors {
        let idx = name
            .strip_prefix("layers.")
            .or_else(|| name.strip_prefix("direction."))
            .and_then(|s| s.parse::<usize>().ok());
        if let Some(i) = idx {
            check(v, name)?;
            let layer = i;
            if layer >= n_layers {
                bail!("{}: {name} is past the model's {n_layers} layers", path.display());
            }
            out.by_layer.insert(layer, v.clone());
        } else if shape.len() == 2 && shape[1] == hidden && tensors.len() == 1 {
            let rows = shape[0];
            let skip = if rows == n_layers + 1 {
                1
            } else if rows == n_layers {
                0
            } else {
                bail!(
                    "{}: {name} has {rows} rows; expected {n_layers} or {}",
                    path.display(),
                    n_layers + 1
                );
            };
            for (l, r) in v.chunks(hidden).skip(skip).enumerate() {
                out.by_layer.insert(l, r.to_vec());
            }
        } else if (shape.len() == 1 || (shape.len() == 2 && shape[0] == 1)) && tensors.len() == 1 {
            check(v, name)?;
            out.single = Some(v.clone());
        }
    }
    if out.by_layer.is_empty() && out.single.is_none() {
        bail!(
            "{}: no vectors found (expected layers.N, direction.N, a [layers(+1), hidden] tensor or a single vector)",
            path.display()
        );
    }
    Ok(out)
}

fn decode_floats(dtype: &str, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "BF16" => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
        "F64" => b.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c) as f32).collect(),
        d => bail!("tensor dtype {d} is not a float type"),
    })
}

/// Writes per-layer vectors as safetensors (`layers.{L}`, f32) with string metadata.
pub fn save_vectors(path: &Path, vectors: &BTreeMap<usize, Vec<f32>>, metadata: &[(&str, String)]) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut meta = serde_json::Map::new();
    meta.insert("format".into(), "oxinfer-vectors".into());
    for (k, v) in metadata {
        meta.insert((*k).into(), v.clone().into());
    }
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));
    let mut off = 0usize;
    for (l, v) in vectors {
        let n = v.len() * 4;
        header.insert(
            format!("layers.{l}"),
            serde_json::json!({"dtype": "F32", "shape": [v.len()], "data_offsets": [off, off + n]}),
        );
        off += n;
    }
    let mut h = serde_json::to_string(&serde_json::Value::Object(header))?.into_bytes();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + off);
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    for v in vectors.values() {
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, out).with_context(|| format!("{}", path.display()))
}

// ---- compiled hooks -------------------------------------------------------------------------

/// Which MLP of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// The dense MLP.
    Mlp,
    /// A MoE layer's shared expert.
    Shared,
    /// A routed expert.
    Expert(usize),
}

/// The model facts compiling a [`Spec`] needs.
pub trait Host {
    fn n_layers(&self) -> usize;
    fn hidden(&self) -> usize;
    /// The intermediate size of `target` in `layer`, None when the layer has no such MLP.
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize>;
    /// Column `n` of that MLP's down projection (`[hidden]`).
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32>;
}

/// How to measure the residual for [`Unit::ResidMedian`]: per layer, the median over positions
/// 1.. of the norm of that layer's output, on the unedited model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Calibration {
    pub resid_median: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum NOp {
    Scale(f32),
    Add(f32),
    Set(f32),
    Clamp(f32, f32),
}

impl NOp {
    #[inline]
    fn apply(self, a: f32) -> f32 {
        match self {
            NOp::Scale(s) => a * s,
            NOp::Add(v) => a + v,
            NOp::Set(v) => v,
            NOp::Clamp(lo, hi) => a.max(lo).min(hi),
        }
    }
    /// (code, a, b) for the GPU kernel.
    fn code(self) -> (f32, f32, f32) {
        match self {
            NOp::Scale(s) => (0.0, s, 0.0),
            NOp::Add(v) => (1.0, v, 0.0),
            NOp::Set(v) => (2.0, v, 0.0),
            NOp::Clamp(lo, hi) => (3.0, lo, hi),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NeuronRule {
    pub target: Target,
    pub idx: usize,
    pub op: NOp,
    pub positions: Positions,
}

/// The nonlinearity giving a feature's activation from `enc . x + bias`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ItemAct {
    Linear,
    Relu,
    /// JumpReLU: `pre` where `pre > threshold`, else 0.
    JumpRelu(f32),
}

impl ItemAct {
    #[inline]
    fn apply(self, pre: f32) -> f32 {
        match self {
            ItemAct::Linear => pre,
            ItemAct::Relu => pre.max(0.0),
            ItemAct::JumpRelu(t) => {
                if pre > t {
                    pre
                } else {
                    0.0
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ItemOp {
    Set(f32),
    Scale(f32),
}

/// One directional edit of the residual: activation `a = act(enc . x + bias)`, new activation
/// `a' = op(a)`, `x += (a' - a) dec`. Directional ablation is `enc = dec = r`, linear, set 0.
#[derive(Clone, Debug)]
pub(crate) struct Item {
    pub enc: Vec<f32>,
    pub dec: Vec<f32>,
    pub bias: f32,
    pub act: ItemAct,
    pub op: ItemOp,
    pub positions: Positions,
}

impl Item {
    /// The coefficient of `dec` for row `x`.
    #[inline]
    fn coef(&self, x: &[f32]) -> f32 {
        let a = self.act.apply(dot(&self.enc, x) + self.bias);
        let a2 = match self.op {
            ItemOp::Set(v) => v,
            ItemOp::Scale(s) => a * s,
        };
        a2 - a
    }
}

/// Feature edits of an SAE whose activations need the whole encoder (TopK): CPU only.
#[derive(Clone)]
pub(crate) struct SaeOp {
    pub sae: Arc<Sae>,
    pub features: Vec<FeatureEdit>,
    pub positions: Positions,
}

#[derive(Clone, Default)]
pub(crate) struct PointOps {
    /// Vectors added (alpha folded in), one per position class.
    pub adds: Vec<(Positions, Vec<f32>)>,
    pub items: Vec<Item>,
    pub saes: Vec<SaeOp>,
}

#[derive(Clone)]
pub(crate) struct InspectOp {
    pub point: usize,
    pub sae: Arc<Sae>,
    pub top_k: usize,
}

/// The most active SAE features of one position ([`SaeInspect`]).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TokenFeatures {
    pub pos: usize,
    pub layer: usize,
    /// (feature, activation), most active first.
    pub top: Vec<(usize, f32)>,
}

static IDS: AtomicU64 = AtomicU64::new(1);

/// A [`Spec`] compiled against a model.
pub struct Compiled {
    pub(crate) n_layers: usize,
    pub(crate) hidden: usize,
    pub(crate) points: Vec<Option<PointOps>>,
    pub(crate) neurons: Vec<Vec<NeuronRule>>,
    pub(crate) inspect: Vec<InspectOp>,
    id: u64,
    gpu: OnceLock<Option<GpuData>>,
}

impl std::fmt::Debug for Compiled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiled")
            .field("points", &self.points.iter().filter(|p| p.is_some()).count())
            .field("neurons", &self.neurons.iter().map(|n| n.len()).sum::<usize>())
            .field("inspect", &self.inspect.len())
            .finish()
    }
}

/// Where compiled specs find their files.
pub struct Ctx<'a> {
    /// Resolves a file spec (path or `org/repo/file`) to a local path.
    pub resolve: &'a dyn Fn(&str) -> Result<PathBuf>,
    pub calibration: Option<&'a Calibration>,
    /// Loaded SAEs, by resolved path (reused across compiles).
    pub saes: Option<&'a Mutex<BTreeMap<PathBuf, Arc<Sae>>>>,
}

fn unit(v: &[f32]) -> Result<Vec<f32>> {
    let n = dot(v, v).sqrt();
    if !(n > 0.0 && n.is_finite()) {
        bail!("a direction has zero or non-finite norm");
    }
    Ok(v.iter().map(|x| x / n).collect())
}

impl Compiled {
    /// Compiles `spec` for the model described by `host`.
    pub fn new(spec: &Spec, host: &dyn Host, ctx: &Ctx) -> Result<Compiled> {
        let nl = host.n_layers();
        let h = host.hidden();
        let mut points: Vec<Option<PointOps>> = vec![None; 2 * nl + 1];
        let mut neurons: Vec<Vec<NeuronRule>> = vec![Vec::new(); nl];
        let check_layer = |l: usize| -> Result<()> {
            if l >= nl {
                bail!("layer {l} is out of range (the model has {nl} layers, 0..{})", nl - 1);
            }
            Ok(())
        };
        let mut vec_cache: BTreeMap<PathBuf, Vectors> = BTreeMap::new();
        let mut vectors = |f: &str| -> Result<Vectors> {
            let p = (ctx.resolve)(f)?;
            if !vec_cache.contains_key(&p) {
                let v = load_vectors(&p, nl, h)?;
                vec_cache.insert(p.clone(), v);
            }
            Ok(vec_cache[&p].clone())
        };
        let add_vec = |points: &mut Vec<Option<PointOps>>, p: usize, pos: Positions, v: &[f32], a: f32| {
            let ops = points[p].get_or_insert_with(PointOps::default);
            let slot = match ops.adds.iter().position(|(q, _)| *q == pos) {
                Some(i) => i,
                None => {
                    ops.adds.push((pos, vec![0f32; v.len()]));
                    ops.adds.len() - 1
                }
            };
            for (d, s) in ops.adds[slot].1.iter_mut().zip(v) {
                *d += a * s;
            }
        };

        // Neuron edits.
        for e in &spec.neurons {
            check_layer(e.layer)?;
            let target = match &e.expert {
                None => {
                    if host.mlp_inter(e.layer, Target::Mlp).is_some() {
                        Target::Mlp
                    } else {
                        bail!(
                            "layer {} is a MoE layer: give the neuron edit an expert (an id or \"shared\")",
                            e.layer
                        );
                    }
                }
                Some(ExpertRef::Name(n)) if n == "shared" => Target::Shared,
                Some(ExpertRef::Name(n)) => bail!("expert {n:?}: expected an expert id or \"shared\""),
                Some(ExpertRef::Id(i)) => Target::Expert(*i),
            };
            let inter = host
                .mlp_inter(e.layer, target)
                .with_context(|| format!("layer {} has no {target:?} MLP", e.layer))?;
            if e.neuron >= inter {
                bail!("neuron {} is out of range for layer {} ({inter} neurons)", e.neuron, e.layer);
            }
            let op = e.op.unwrap_or(if e.dose.is_some() { NeuronOp::Add } else { NeuronOp::Scale });
            let value = e.value.or(e.dose);
            let val = |what: &str| value.with_context(|| format!("neuron edit {}:{}: {what} needs a value", e.layer, e.neuron));
            let scale = match e.unit {
                Unit::Raw => 1.0,
                Unit::ResidMedian => {
                    if !matches!(op, NeuronOp::Add | NeuronOp::Set) {
                        bail!("unit resid_median goes with add / dose or set");
                    }
                    let med = match e.resid_median {
                        Some(m) => m,
                        None => *ctx
                            .calibration
                            .and_then(|c| c.resid_median.get(e.layer))
                            .context("unit resid_median needs a calibration pass (or an explicit resid_median)")?,
                    };
                    let col = host.down_column(e.layer, target, e.neuron);
                    let cn = dot(&col, &col).sqrt();
                    if cn.is_nan() || cn <= 0.0 {
                        bail!("neuron {}:{} has a zero down_proj column", e.layer, e.neuron);
                    }
                    med / cn
                }
            };
            let op = match op {
                NeuronOp::Scale => NOp::Scale(val("scale")?),
                NeuronOp::Zero => NOp::Scale(0.0),
                NeuronOp::Add => NOp::Add(val("add")? * scale),
                NeuronOp::Set => NOp::Set(val("set")? * scale),
                NeuronOp::Clamp => {
                    if e.min.is_none() && e.max.is_none() {
                        bail!("clamp needs min and/or max");
                    }
                    NOp::Clamp(e.min.unwrap_or(f32::NEG_INFINITY), e.max.unwrap_or(f32::INFINITY))
                }
            };
            neurons[e.layer].push(NeuronRule {
                target,
                idx: e.neuron,
                op,
                positions: e.positions,
            });
        }

        // Steering vectors.
        for s in &spec.steer {
            let mut list: Vec<(usize, Vec<f32>)> = Vec::new();
            match (&s.vector, &s.file) {
                (Some(v), None) => {
                    let l = s.layer.context("an inline steering vector needs a layer")?;
                    if v.len() != h {
                        bail!("steering vector has {} values, hidden size is {h}", v.len());
                    }
                    list.push((l, v.clone()));
                }
                (None, Some(f)) => {
                    let vs = vectors(f)?;
                    match (&vs.single, s.layer, &s.layers) {
                        (Some(v), Some(l), _) => list.push((l, v.clone())),
                        (Some(v), None, Some(ls)) => list.extend(ls.iter().map(|&l| (l, v.clone()))),
                        (Some(_), None, None) => bail!("{f}: a single vector needs a layer"),
                        (None, ..) => {
                            let want: Option<Vec<usize>> = s.layers.clone().or(s.layer.map(|l| vec![l]));
                            for (l, v) in vs.by_layer {
                                if want.as_ref().is_none_or(|w| w.contains(&l)) {
                                    list.push((l, v));
                                }
                            }
                            if list.is_empty() {
                                bail!("{f}: none of the requested layers is in the file");
                            }
                        }
                    }
                }
                _ => bail!("a steering entry needs exactly one of vector or file"),
            }
            for (l, v) in list {
                check_layer(l)?;
                let v = if s.normalize { unit(&v)? } else { v };
                add_vec(&mut points, point_index(l, s.point), s.positions, &v, s.alpha);
            }
        }

        // Directional ablation.
        for a in &spec.ablate {
            let r = match (&a.vector, &a.file) {
                (Some(v), None) => v.clone(),
                (None, Some(f)) => {
                    let vs = vectors(f)?;
                    match (vs.single, a.direction_layer) {
                        (Some(v), _) => v,
                        (None, Some(l)) => vs
                            .by_layer
                            .get(&l)
                            .cloned()
                            .with_context(|| format!("{f}: no vector for layer {l}"))?,
                        (None, None) if vs.by_layer.len() == 1 => vs.by_layer.into_values().next().unwrap(),
                        (None, None) => bail!("{f} has a vector per layer: choose one with direction_layer"),
                    }
                }
                _ => bail!("an ablation entry needs exactly one of vector or file"),
            };
            if r.len() != h {
                bail!("ablation direction has {} values, hidden size is {h}", r.len());
            }
            let r = unit(&r)?;
            let layers: Vec<usize> = a.layers.clone().unwrap_or_else(|| (0..nl).collect());
            let mut pts: Vec<usize> = Vec::new();
            for &l in &layers {
                check_layer(l)?;
                pts.push(point_index(l, Point::Pre));
                pts.push(point_index(l, Point::Mid));
                pts.push(point_index(l, Point::Post));
            }
            pts.sort_unstable();
            pts.dedup();
            for p in pts {
                let ops = points[p].get_or_insert_with(PointOps::default);
                // Several directions at one point: keep them orthonormal so that projecting them
                // out one at a time or all at once is the same.
                let mut d = r.clone();
                for it in ops
                    .items
                    .iter()
                    .filter(|i| i.act == ItemAct::Linear && i.op == ItemOp::Set(0.0) && i.enc == i.dec)
                {
                    let c = dot(&d, &it.dec);
                    for (x, y) in d.iter_mut().zip(&it.dec) {
                        *x -= c * y;
                    }
                }
                let Ok(d) = unit(&d) else { continue }; // already projected out
                ops.items.push(Item {
                    enc: d.clone(),
                    dec: d,
                    bias: 0.0,
                    act: ItemAct::Linear,
                    op: ItemOp::Set(0.0),
                    positions: a.positions,
                });
            }
        }

        // SAEs.
        let get_sae = |f: &str| -> Result<Arc<Sae>> {
            // A SAELens directory, or a file (local or in a repo).
            let p = if Path::new(f).is_dir() {
                PathBuf::from(f)
            } else {
                (ctx.resolve)(f)?
            };
            if let Some(c) = ctx.saes
                && let Some(s) = c.lock().unwrap().get(&p)
            {
                return Ok(s.clone());
            }
            // A weights file from a repo: fetch its config next to it (Qwen-Scope `config.json`,
            // SAELens `cfg.json`).
            let mut cfg = None;
            if !p.is_dir()
                && let Some((dir, _)) = f.rsplit_once('/')
            {
                for c in ["cfg.json", "config.json"] {
                    let local = p.parent().map(|d| d.join(c));
                    if local.as_ref().is_some_and(|l| l.is_file()) {
                        cfg = local;
                        break;
                    }
                    if let Ok(q) = (ctx.resolve)(&format!("{dir}/{c}")) {
                        cfg = Some(q);
                        break;
                    }
                }
            }
            let s = Arc::new(Sae::load_with(&p, cfg.as_deref())?);
            if s.d_in != h {
                bail!("{f}: the SAE reads {} values, the model's hidden size is {h}", s.d_in);
            }
            if let Some(c) = ctx.saes {
                c.lock().unwrap().insert(p, s.clone());
            }
            Ok(s)
        };
        for s in &spec.sae {
            let sae = get_sae(&s.sae)?;
            let l = s.layer.or(sae.layer).context("the SAE does not say its layer: give one")?;
            check_layer(l)?;
            let p = point_index(l, Point::Post);
            let mut cpu = Vec::new();
            for f in &s.features {
                if f.id >= sae.d_sae {
                    bail!("feature {} is out of range ({} features)", f.id, sae.d_sae);
                }
                let dec = sae.decoder_row(f.id);
                match f.op {
                    // Simultaneous with the SAE's other edits (one encode of the incoming x):
                    // an item whose activation is always 0 and becomes `value`.
                    FeatureOp::Add => points[p].get_or_insert_with(PointOps::default).items.push(Item {
                        enc: vec![0.0; h],
                        dec,
                        bias: 0.0,
                        act: ItemAct::Linear,
                        op: ItemOp::Set(f.value),
                        positions: s.positions,
                    }),
                    _ => {
                        let op = match f.op {
                            FeatureOp::Clamp => ItemOp::Set(f.value),
                            FeatureOp::Zero => ItemOp::Set(0.0),
                            _ => ItemOp::Scale(f.value),
                        };
                        match sae.item_act(f.id) {
                            Some(act) => {
                                let (enc, bias) = sae.encoder_column(f.id);
                                points[p].get_or_insert_with(PointOps::default).items.push(Item {
                                    enc,
                                    dec,
                                    bias,
                                    act,
                                    op,
                                    positions: s.positions,
                                });
                            }
                            None => cpu.push(f.clone()),
                        }
                    }
                }
            }
            if !cpu.is_empty() {
                points[p].get_or_insert_with(PointOps::default).saes.push(SaeOp {
                    sae: sae.clone(),
                    features: cpu,
                    positions: s.positions,
                });
            }
        }
        let mut inspect = Vec::new();
        for s in &spec.sae_inspect {
            let sae = get_sae(&s.sae)?;
            let l = s.layer.or(sae.layer).context("the SAE does not say its layer: give one")?;
            check_layer(l)?;
            inspect.push(InspectOp {
                point: point_index(l, Point::Post),
                sae,
                top_k: s.top_k.max(1),
            });
        }
        Ok(Compiled {
            n_layers: nl,
            hidden: h,
            points,
            neurons,
            inspect,
            id: IDS.fetch_add(1, Ordering::Relaxed),
            gpu: OnceLock::new(),
        })
    }

    /// Whether SAE inspection is on (the session then needs a feature sink).
    pub fn inspects(&self) -> bool {
        !self.inspect.is_empty()
    }

    /// Whether any neuron of `layer` / `target` is edited.
    pub fn edits_neurons(&self, layer: usize, target: Target) -> bool {
        self.neurons.get(layer).is_some_and(|v| v.iter().any(|r| r.target == target))
    }

    /// Whether any routed expert of `layer` has edits.
    pub fn edits_experts(&self, layer: usize) -> bool {
        self.neurons
            .get(layer)
            .is_some_and(|v| v.iter().any(|r| matches!(r.target, Target::Expert(_))))
    }
}

// ---- per-session state ----------------------------------------------------------------------

/// What a session applies and records (see the module docs). Cloning shares the compiled hooks
/// and the capture / inspection sinks.
#[derive(Clone)]
pub struct State {
    pub iv: Option<Arc<Compiled>>,
    /// Positions from here on are "generated" ([`Positions::Generated`]): the last prompt token
    /// and later ones. Set by [`crate::generate::prefill`] (`usize::MAX`: all prompt).
    pub gen_from: usize,
    pub capture: Option<Arc<Mutex<Capture>>>,
    /// SAE inspection results.
    pub features: Option<Arc<Mutex<Vec<TokenFeatures>>>>,
}

impl Default for State {
    fn default() -> Self {
        State {
            iv: None,
            gen_from: usize::MAX,
            capture: None,
            features: None,
        }
    }
}

impl State {
    /// Whether any hook is set (models skip all hook calls otherwise).
    #[inline]
    pub fn active(&self) -> bool {
        self.iv.is_some() || self.capture.is_some()
    }

    /// Whether a GPU decode step can apply everything: no capture or inspection, and the
    /// compiled hooks fit the device graph ([`Compiled::gpu_data`]).
    pub fn gpu_ok(&self) -> bool {
        self.capture.is_none() && self.iv.as_ref().is_none_or(|c| c.gpu_data().is_some())
    }

    #[inline]
    fn on(&self, p: Positions, pos: usize) -> bool {
        match p {
            Positions::All => true,
            Positions::Prompt => pos < self.gen_from,
            Positions::Generated => pos >= self.gen_from,
        }
    }

    /// Applies the hooks of residual point `p` to `x` (`t` rows of `h`, positions `start..`).
    pub fn point(&self, p: usize, x: &mut [f32], h: usize, start: usize) {
        let t = x.len() / h;
        if let Some(iv) = &self.iv
            && let Some(Some(ops)) = iv.points.get(p)
        {
            // Added vectors first, then the edits read the result: a direction ablated at the
            // same point also removes its component of what was added (as a hook on the next
            // layer's input would), and a clamped feature ends at its value.
            if !ops.adds.is_empty() {
                for_chunks(x, h, t > 1, |i, row| {
                    for (ps, v) in &ops.adds {
                        if self.on(*ps, start + i) {
                            for (r, a) in row.iter_mut().zip(v) {
                                *r += a;
                            }
                        }
                    }
                });
            }
            // Whole-encoder SAE edits (each encode runs on the thread pool, so rows go one by
            // one). All edits read the same x (simultaneous, as an SAE would).
            let mut sae_delta: Vec<Vec<(f32, &[f32])>> = vec![Vec::new(); if ops.saes.is_empty() { 0 } else { t }];
            for (i, d) in sae_delta.iter_mut().enumerate() {
                let row = &x[i * h..(i + 1) * h];
                for op in &ops.saes {
                    if self.on(op.positions, start + i) {
                        d.extend(op.sae.edit_coefs(row, &op.features));
                    }
                }
            }
            if !ops.items.is_empty() || !sae_delta.is_empty() {
                for_chunks(x, h, t > 1, |i, row| {
                    let pos = start + i;
                    let coefs: Vec<f32> = ops
                        .items
                        .iter()
                        .map(|it| if self.on(it.positions, pos) { it.coef(row) } else { 0.0 })
                        .collect();
                    for (it, c) in ops.items.iter().zip(coefs) {
                        if c != 0.0 {
                            for (v, d) in row.iter_mut().zip(&it.dec) {
                                *v += c * d;
                            }
                        }
                    }
                    for &(c, d) in sae_delta.get(i).map(|v| &v[..]).unwrap_or(&[]) {
                        for (v, d) in row.iter_mut().zip(d) {
                            *v += c * d;
                        }
                    }
                });
            }
        }
        if let Some(iv) = &self.iv
            && let Some(sink) = &self.features
        {
            for ins in iv.inspect.iter().filter(|i| i.point == p) {
                let mut out = sink.lock().unwrap();
                for (i, row) in x.chunks(h).enumerate() {
                    out.push(TokenFeatures {
                        pos: start + i,
                        layer: p.saturating_sub(2) / 2,
                        top: ins.sae.top_features(row, ins.top_k),
                    });
                }
            }
        }
        if let Some(c) = &self.capture {
            c.lock().unwrap().record(p, x, h, start);
        }
    }

    /// Applies the neuron edits of `layer` / `target` to `act` (`t` rows of `inter`, positions
    /// `start..`).
    pub fn neurons(&self, layer: usize, target: Target, act: &mut [f32], inter: usize, start: usize) {
        let Some(iv) = &self.iv else { return };
        let Some(rules) = iv.neurons.get(layer) else { return };
        for r in rules.iter().filter(|r| r.target == target) {
            for (i, row) in act.chunks_mut(inter).enumerate() {
                if self.on(r.positions, start + i) {
                    row[r.idx] = r.op.apply(row[r.idx]);
                }
            }
        }
    }

    /// Like [`State::neurons`] for rows at arbitrary positions (`pos[i]` for row `i`;
    /// `usize::MAX`: padding, left alone).
    pub fn neurons_at(&self, layer: usize, target: Target, act: &mut [f32], inter: usize, pos: &[usize]) {
        let Some(iv) = &self.iv else { return };
        let Some(rules) = iv.neurons.get(layer) else { return };
        for r in rules.iter().filter(|r| r.target == target) {
            for (row, &p) in act.chunks_mut(inter).zip(pos) {
                if p != usize::MAX && self.on(r.positions, p) {
                    row[r.idx] = r.op.apply(row[r.idx]);
                }
            }
        }
    }

    /// The neuron hook of `layer` for a pass starting at position `start`, when it has edits.
    pub fn hook(&self, layer: usize, start: usize) -> Option<NeuronHook<'_>> {
        let iv = self.iv.as_ref()?;
        (!iv.neurons.get(layer)?.is_empty()).then_some(NeuronHook { st: self, layer, start })
    }

    /// Whether `layer` has neuron edits of routed experts (the MoE then runs on the CPU paths).
    pub fn edits_experts(&self, layer: usize) -> bool {
        self.iv.as_ref().is_some_and(|c| c.edits_experts(layer))
    }

    /// Whether `layer` has neuron edits of `target`.
    pub fn edits(&self, layer: usize, target: Target) -> bool {
        self.iv.as_ref().is_some_and(|c| c.edits_neurons(layer, target))
    }
}

/// The neuron edits of one layer during one forward pass (rows are the pass's tokens).
#[derive(Clone, Copy)]
pub struct NeuronHook<'a> {
    st: &'a State,
    layer: usize,
    start: usize,
}

impl NeuronHook<'_> {
    /// `act`: one row of `inter` per token of the pass.
    pub fn apply(&self, target: Target, act: &mut [f32], inter: usize) {
        self.st.neurons(self.layer, target, act, inter, self.start);
    }
    /// `act`: row `i` belongs to token `rows[i]` of the pass (`usize::MAX`: padding).
    pub fn apply_rows(&self, target: Target, act: &mut [f32], inter: usize, rows: &[usize]) {
        let pos: Vec<usize> = rows.iter().map(|&r| if r == usize::MAX { r } else { self.start + r }).collect();
        self.st.neurons_at(self.layer, target, act, inter, &pos);
    }
    /// Whether routed experts are edited (they then stay on the CPU).
    pub fn experts(&self) -> bool {
        self.st.edits_experts(self.layer)
    }
    /// Whether `target` is edited.
    pub fn edits(&self, target: Target) -> bool {
        self.st.edits(self.layer, target)
    }
}

// ---- capture --------------------------------------------------------------------------------

/// How captured rows are reduced per forward sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pool {
    /// The last position fed.
    #[default]
    Last,
    /// The mean over positions 1.. (position 0, usually a massive-norm attention sink, is left
    /// out unless it is the only one).
    Mean,
    /// Every row.
    All,
    /// The L2 norm of every row.
    Norms,
}

/// Residual activations recorded while a session runs ([`State::capture`]).
#[derive(Clone, Debug, Default)]
pub struct Capture {
    pub pool: Pool,
    /// The residual points to record (empty: every layer output).
    pub points: Vec<usize>,
    pub hidden: usize,
    /// Per point: (sum or last row or all rows or norms, rows counted, position of the first).
    pub data: BTreeMap<usize, (Vec<f32>, usize)>,
    first0: BTreeMap<usize, Vec<f32>>,
}

impl Capture {
    pub fn new(pool: Pool, points: Vec<usize>) -> Self {
        Capture {
            pool,
            points,
            ..Default::default()
        }
    }

    fn record(&mut self, p: usize, x: &[f32], h: usize, start: usize) {
        if self.points.is_empty() {
            if p == 0 || !p.is_multiple_of(2) {
                return; // layer outputs only by default
            }
        } else if !self.points.contains(&p) {
            return;
        }
        self.hidden = h;
        let e = self.data.entry(p).or_insert_with(|| (Vec::new(), 0));
        match self.pool {
            Pool::Last => {
                e.0 = x[x.len() - h..].to_vec();
                e.1 = 1;
            }
            Pool::All => {
                e.0.extend_from_slice(x);
                e.1 += x.len() / h;
            }
            Pool::Norms => {
                e.0.extend(x.chunks(h).map(|r| dot(r, r).sqrt()));
                e.1 += x.len() / h;
            }
            Pool::Mean => {
                if e.0.is_empty() {
                    e.0 = vec![0.0; h];
                }
                for (i, r) in x.chunks(h).enumerate() {
                    if start + i == 0 {
                        self.first0.insert(p, r.to_vec());
                        continue;
                    }
                    for (s, v) in e.0.iter_mut().zip(r) {
                        *s += v;
                    }
                    e.1 += 1;
                }
            }
        }
    }

    /// The pooled vector of each recorded point (Last / Mean), keyed by point.
    pub fn pooled(&self) -> BTreeMap<usize, Vec<f32>> {
        self.data
            .iter()
            .map(|(&p, (v, n))| match self.pool {
                Pool::Mean if *n == 0 => (p, self.first0.get(&p).cloned().unwrap_or_else(|| v.clone())),
                Pool::Mean => (p, v.iter().map(|s| s / *n as f32).collect()),
                _ => (p, v.clone()),
            })
            .collect()
    }

    /// Forgets what was recorded (keeps the settings).
    pub fn clear(&mut self) {
        self.data.clear();
        self.first0.clear();
    }
}

/// The median of `v` (mean of the middle two for even lengths); 0 when empty.
pub fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

// ---- library helpers over any model ---------------------------------------------------------

use crate::generate::TextLm;

/// Runs `ids` through a fresh session of `model` (no interventions) and measures the median
/// norm of every layer's output over positions 1.. ([`Unit::ResidMedian`]).
pub fn calibrate<M: TextLm + ?Sized>(model: &M, ids: &[u32], chunk: usize) -> Result<Calibration> {
    if ids.len() < 2 {
        bail!("calibration needs a prompt of at least 2 tokens");
    }
    let mut sess = model.new_session();
    let cap = Arc::new(Mutex::new(Capture::new(Pool::Norms, Vec::new())));
    let st = model.iv_state(&mut sess).context("this model does not support interventions")?;
    st.capture = Some(cap.clone());
    crate::generate::prefill(model, &mut sess, ids, chunk, &[]);
    let c = cap.lock().unwrap();
    let nl = model.n_layers();
    let mut med = Vec::with_capacity(nl);
    for l in 0..nl {
        let (norms, _) = c.data.get(&point_index(l, Point::Post)).context("capture missed a layer")?;
        med.push(median(&norms[1..]));
    }
    Ok(Calibration { resid_median: med })
}

/// Runs each prompt (token ids) through a fresh session with no interventions and returns, per
/// prompt, the pooled output of every layer (`layer -> vector`).
pub fn capture_layers<M: TextLm + ?Sized>(
    model: &M,
    prompts: &[Vec<u32>],
    pool: Pool,
    chunk: usize,
) -> Result<Vec<BTreeMap<usize, Vec<f32>>>> {
    if !matches!(pool, Pool::Last | Pool::Mean) {
        bail!("capture_layers pools with last or mean");
    }
    let nl = model.n_layers();
    let mut out = Vec::with_capacity(prompts.len());
    for ids in prompts {
        if ids.is_empty() {
            bail!("empty prompt");
        }
        let mut sess = model.new_session();
        let cap = Arc::new(Mutex::new(Capture::new(pool, Vec::new())));
        model
            .iv_state(&mut sess)
            .context("this model does not support interventions")?
            .capture = Some(cap.clone());
        crate::generate::prefill(model, &mut sess, ids, chunk, &[]);
        let pooled = cap.lock().unwrap().pooled();
        out.push((0..nl).map(|l| (l, pooled[&point_index(l, Point::Post)].clone())).collect());
    }
    Ok(out)
}

/// The mean over prompts of per-layer vectors.
pub fn mean_layers(sets: &[BTreeMap<usize, Vec<f32>>]) -> BTreeMap<usize, Vec<f32>> {
    let mut acc: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for s in sets {
        for (&l, v) in s {
            let a = acc.entry(l).or_insert_with(|| vec![0.0; v.len()]);
            for (x, y) in a.iter_mut().zip(v) {
                *x += *y as f64;
            }
        }
    }
    let n = sets.len().max(1) as f64;
    acc.into_iter()
        .map(|(l, v)| (l, v.into_iter().map(|x| (x / n) as f32).collect()))
        .collect()
}

/// Difference of means per layer: `mean(positive) - mean(negative)` (not normalized: steering
/// with alpha 1 moves the negative mean onto the positive one; ablation normalizes it).
pub fn diff_of_means(pos: &[BTreeMap<usize, Vec<f32>>], neg: &[BTreeMap<usize, Vec<f32>>]) -> BTreeMap<usize, Vec<f32>> {
    let (a, b) = (mean_layers(pos), mean_layers(neg));
    a.into_iter()
        .filter_map(|(l, va)| b.get(&l).map(|vb| (l, va.iter().zip(vb).map(|(x, y)| x - y).collect())))
        .collect()
}

/// Reads a prompt list: a `.json` array of strings (or objects), `.jsonl` lines, `.csv` (first
/// column, header skipped), or plain text with one prompt per line. Objects give their
/// `prompt`, `instruction`, `goal` or `text` field.
pub fn read_prompts(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let from_obj = |v: &serde_json::Value| -> Option<String> {
        match v {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(o) => ["prompt", "instruction", "goal", "text"]
                .iter()
                .find_map(|k| o.get(*k).and_then(|x| x.as_str()).map(str::to_string)),
            _ => None,
        }
    };
    let v: Vec<String> = match ext.as_str() {
        "json" => {
            let j: serde_json::Value = serde_json::from_str(&text)?;
            j.as_array().context("expected a JSON array")?.iter().filter_map(from_obj).collect()
        }
        "jsonl" => text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok().as_ref().and_then(from_obj))
            .collect(),
        "csv" => text.lines().skip(1).filter_map(csv_first).filter(|s| !s.is_empty()).collect(),
        _ => text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect(),
    };
    if v.is_empty() {
        bail!("{}: no prompts", path.display());
    }
    Ok(v)
}

/// The first field of a CSV line (quotes handled).
fn csv_first(line: &str) -> Option<String> {
    let line = line.trim_end_matches('\r');
    if let Some(rest) = line.strip_prefix('"') {
        let mut out = String::new();
        let mut chars = rest.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    out.push('"');
                    chars.next();
                } else {
                    return Some(out);
                }
            } else {
                out.push(c);
            }
        }
        Some(out)
    } else {
        Some(line.split(',').next().unwrap_or("").to_string())
    }
}

// ---- GPU decode graph support ---------------------------------------------------------------

/// Floats per item / neuron entry in the per-step parameters.
pub const GPU_ENTRY: usize = 8;
/// Most items (directions, features) one residual point can have on the GPU.
pub const GPU_MAX_ITEMS: usize = 64;

/// Where one residual point's parameters sit: `hot` (rewritten every step: `n_add` add
/// coefficients padded to 4, then `GPU_ENTRY` floats per item: bias, threshold, act (0 linear,
/// 1 relu, 2 jumprelu), op (0 set, 1 scale), value, enabled, 0, 0) and `cold` (rewritten when the
/// compiled hooks change: the add vectors, then each item's encoder row, then its decoder row).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuPoint {
    pub hot: usize,
    pub cold: usize,
    pub n_add: usize,
    pub n_items: usize,
}

/// The neuron edits of one MLP: `n` entries of `GPU_ENTRY` floats at `hot` (expert id or -1,
/// neuron, op (0 scale, 1 add, 2 set, 3 clamp), a, b, enabled, 0, 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuNeurons {
    pub hot: usize,
    pub n: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuLayout {
    pub points: Vec<Option<GpuPoint>>,
    /// Per layer: (dense MLP or shared expert, routed experts).
    pub neurons: Vec<(Option<GpuNeurons>, Option<GpuNeurons>)>,
    pub hot_len: usize,
    pub cold_len: usize,
}

impl GpuLayout {
    /// A key for plans built from this layout (the structure only, not the values).
    pub fn signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut s = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut s);
        s.finish() | 1
    }
}

pub struct GpuData {
    pub layout: GpuLayout,
    pub cold: Vec<f32>,
}

impl Compiled {
    /// The device-graph form of the hooks, None when some cannot run there (TopK SAE features,
    /// inspection, too many items at a point).
    pub fn gpu_data(&self) -> Option<&GpuData> {
        self.gpu
            .get_or_init(|| {
                if !self.inspect.is_empty() {
                    return None;
                }
                let h = self.hidden;
                let mut lay = GpuLayout {
                    points: vec![None; self.points.len()],
                    neurons: vec![(None, None); self.n_layers],
                    hot_len: 0,
                    cold_len: 0,
                };
                let mut cold = Vec::new();
                for (p, ops) in self.points.iter().enumerate() {
                    let Some(ops) = ops else { continue };
                    if !ops.saes.is_empty() || ops.items.len() > GPU_MAX_ITEMS {
                        return None;
                    }
                    let gp = GpuPoint {
                        hot: lay.hot_len,
                        cold: cold.len(),
                        n_add: ops.adds.len(),
                        n_items: ops.items.len(),
                    };
                    lay.hot_len += ops.adds.len().next_multiple_of(4) + GPU_ENTRY * ops.items.len();
                    for (_, v) in &ops.adds {
                        cold.extend_from_slice(v);
                    }
                    for it in &ops.items {
                        cold.extend_from_slice(&it.enc);
                    }
                    for it in &ops.items {
                        cold.extend_from_slice(&it.dec);
                    }
                    debug_assert_eq!(cold.len(), gp.cold + (gp.n_add + 2 * gp.n_items) * h);
                    lay.points[p] = Some(gp);
                }
                for (l, rules) in self.neurons.iter().enumerate() {
                    // The same neuron edited twice would race on the device.
                    let mut seen = std::collections::HashSet::new();
                    if !rules.iter().all(|r| seen.insert((r.target, r.idx))) {
                        return None;
                    }
                    let dense = rules.iter().filter(|r| !matches!(r.target, Target::Expert(_))).count();
                    let ex = rules.len() - dense;
                    if rules.iter().any(|r| r.target == Target::Mlp) && rules.iter().any(|r| r.target == Target::Shared) {
                        return None; // one or the other per layer
                    }
                    if dense > 0 {
                        lay.neurons[l].0 = Some(GpuNeurons {
                            hot: lay.hot_len,
                            n: dense,
                        });
                        lay.hot_len += GPU_ENTRY * dense;
                    }
                    if ex > 0 {
                        lay.neurons[l].1 = Some(GpuNeurons { hot: lay.hot_len, n: ex });
                        lay.hot_len += GPU_ENTRY * ex;
                    }
                }
                lay.cold_len = cold.len();
                Some(GpuData { layout: lay, cold })
            })
            .as_ref()
    }

    /// The unique id of this compiled set (a device copy of [`GpuData::cold`] is refreshed when
    /// it changes).
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl State {
    /// The per-step parameters of a GPU decode step at position `pos` (see [`GpuPoint`]).
    pub fn gpu_hot(&self, pos: usize) -> Vec<f32> {
        let Some(iv) = &self.iv else { return Vec::new() };
        let Some(g) = iv.gpu_data() else { return Vec::new() };
        let mut hot = vec![0f32; g.layout.hot_len];
        for (p, gp) in g.layout.points.iter().enumerate() {
            let (Some(gp), Some(ops)) = (gp, &iv.points[p]) else { continue };
            for (i, (ps, _)) in ops.adds.iter().enumerate() {
                hot[gp.hot + i] = self.on(*ps, pos) as u8 as f32;
            }
            let base = gp.hot + gp.n_add.next_multiple_of(4);
            for (i, it) in ops.items.iter().enumerate() {
                let e = &mut hot[base + i * GPU_ENTRY..base + (i + 1) * GPU_ENTRY];
                let (act, thr) = match it.act {
                    ItemAct::Linear => (0.0, 0.0),
                    ItemAct::Relu => (1.0, 0.0),
                    ItemAct::JumpRelu(t) => (2.0, t),
                };
                let (op, v) = match it.op {
                    ItemOp::Set(v) => (0.0, v),
                    ItemOp::Scale(s) => (1.0, s),
                };
                e[..6].copy_from_slice(&[it.bias, thr, act, op, v, self.on(it.positions, pos) as u8 as f32]);
            }
        }
        for (l, (dn, en)) in g.layout.neurons.iter().enumerate() {
            let rules = &iv.neurons[l];
            for (g, experts) in [(dn, false), (en, true)] {
                let Some(g) = g else { continue };
                for (i, r) in rules
                    .iter()
                    .filter(|r| matches!(r.target, Target::Expert(_)) == experts)
                    .enumerate()
                {
                    let (code, a, b) = r.op.code();
                    let ex = match r.target {
                        Target::Expert(e) => e as f32,
                        _ => -1.0,
                    };
                    hot[g.hot + i * GPU_ENTRY..g.hot + i * GPU_ENTRY + 6].copy_from_slice(&[
                        ex,
                        r.idx as f32,
                        code,
                        a,
                        b,
                        self.on(r.positions, pos) as u8 as f32,
                    ]);
                }
            }
        }
        hot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ToyHost {
        down: Vec<Vec<f32>>, // [layers][h * inter] row-major [h, inter]
        h: usize,
        inter: usize,
    }
    impl Host for ToyHost {
        fn n_layers(&self) -> usize {
            self.down.len()
        }
        fn hidden(&self) -> usize {
            self.h
        }
        fn mlp_inter(&self, _l: usize, t: Target) -> Option<usize> {
            (t == Target::Mlp).then_some(self.inter)
        }
        fn down_column(&self, l: usize, _t: Target, n: usize) -> Vec<f32> {
            (0..self.h).map(|r| self.down[l][r * self.inter + n]).collect()
        }
    }

    fn ctx<'a>(cal: Option<&'a Calibration>) -> Ctx<'a> {
        static R: fn(&str) -> Result<PathBuf> = |s| Ok(PathBuf::from(s));
        Ctx {
            resolve: &R,
            calibration: cal,
            saes: None,
        }
    }

    fn toy() -> ToyHost {
        ToyHost {
            down: vec![vec![0.5; 4 * 6], (0..24).map(|i| i as f32 * 0.1).collect()],
            h: 4,
            inter: 6,
        }
    }

    #[test]
    fn neuron_ops_and_dose() {
        let host = toy();
        let mut spec = Spec::default();
        spec.neurons.push(Spec::parse_neuron("1:2,dose=2,unit=resid_median").unwrap());
        spec.neurons.push(Spec::parse_neuron("1:3,zero").unwrap());
        spec.neurons.push(Spec::parse_neuron("1:4,clamp=-1:0.5").unwrap());
        spec.neurons.push(Spec::parse_neuron("0:0,scale=3,positions=generated").unwrap());
        assert!(spec.needs_calibration());
        assert!(Compiled::new(&spec, &host, &ctx(None)).is_err());
        let cal = Calibration {
            resid_median: vec![1.0, 10.0],
        };
        let c = Compiled::new(&spec, &host, &ctx(Some(&cal))).unwrap();
        let st = State {
            iv: Some(Arc::new(c)),
            gen_from: 1,
            ..Default::default()
        };
        // Layer 1, neuron 2: add 2 * 10 / |col 2|.
        let col: Vec<f32> = (0..4).map(|r| (r * 6 + 2) as f32 * 0.1).collect();
        let want = 2.0 * 10.0 / dot(&col, &col).sqrt();
        let mut act = vec![1.0f32; 12];
        st.neurons(1, Target::Mlp, &mut act, 6, 0);
        assert!((act[2] - (1.0 + want)).abs() < 1e-5 && (act[8] - (1.0 + want)).abs() < 1e-5);
        assert_eq!(act[3], 0.0);
        assert_eq!(act[4], 0.5);
        // Positions: only the generated row (position >= 1) is scaled.
        let mut a0 = vec![1.0f32; 12];
        st.neurons(0, Target::Mlp, &mut a0, 6, 0);
        assert_eq!((a0[0], a0[6]), (1.0, 3.0));
    }

    #[test]
    fn steer_and_ablate() {
        let host = toy();
        let r = vec![1.0, 1.0, 0.0, 0.0];
        let spec = Spec {
            steer: vec![Steer {
                vector: Some(vec![0.0, 0.0, 1.0, 0.0]),
                layer: Some(0),
                alpha: 2.0,
                ..Default::default()
            }],
            ablate: vec![Ablate {
                vector: Some(r.clone()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let c = Compiled::new(&spec, &host, &ctx(None)).unwrap();
        let st = State {
            iv: Some(Arc::new(c)),
            ..Default::default()
        };
        let mut x = vec![3.0, 1.0, 5.0, 7.0];
        st.point(point_index(0, Point::Post), &mut x, 4, 0);
        // Projected out of (1,1)/sqrt2: (3,1) -> (1,-1); plus 2 on dim 2.
        let want = [1.0, -1.0, 7.0, 7.0];
        for (a, b) in x.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{x:?}");
        }
        // The GPU layout covers it.
        let g = st.iv.as_ref().unwrap().gpu_data().unwrap();
        let gp = g.layout.points[2].unwrap();
        assert_eq!((gp.n_add, gp.n_items), (1, 1));
        // 5 points (embeddings, mid and post of 2 layers) with an encoder and decoder row each,
        // plus the add vector.
        assert_eq!(g.cold.len(), 5 * 2 * 4 + 4);
    }

    #[test]
    fn vectors_round_trip() {
        let dir = std::env::temp_dir().join(format!("oxinfer-iv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("v.safetensors");
        let mut m = BTreeMap::new();
        m.insert(1usize, vec![1.0f32, 2.0, 3.0, 4.0]);
        m.insert(0usize, vec![0.5f32; 4]);
        save_vectors(&p, &m, &[("model", "toy".into())]).unwrap();
        let v = load_vectors(&p, 2, 4).unwrap();
        assert_eq!(v.by_layer, m);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn capture_pools() {
        let mut c = Capture::new(Pool::Mean, vec![2]);
        c.record(2, &[100.0, 100.0, 1.0, 2.0], 2, 0);
        c.record(2, &[3.0, 4.0], 2, 2);
        assert_eq!(c.pooled()[&2], vec![2.0, 3.0]);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        let diff = diff_of_means(&[BTreeMap::from([(0, vec![2.0, 2.0])])], &[BTreeMap::from([(0, vec![1.0, 0.0])])]);
        assert_eq!(diff[&0], vec![1.0, 2.0]);
    }

    #[test]
    fn parse_spec_json() {
        let s = Spec::from_json(
            r#"{"neurons":[{"layer":14,"neuron":8949,"dose":1.0,"unit":"resid_median"},{"layer":3,"neuron":1,"expert":"shared","op":"zero"}],
                "steer":[{"file":"v.safetensors","alpha":4,"positions":"generated"}]}"#,
        )
        .unwrap();
        assert_eq!(s.neurons[0].unit, Unit::ResidMedian);
        assert_eq!(s.neurons[1].expert, Some(ExpertRef::Name("shared".into())));
        assert_eq!(s.steer[0].positions, Positions::Generated);
        assert_eq!(s.steer[0].point, Point::Post);
        assert!(Spec::from_json(r#"{"bogus":1}"#).is_err());
    }
}
