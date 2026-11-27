//! Training a LoRA adapter from logged episodes: the offline step of learning from outcomes.
//!
//! The server logs each response as an episode (prompt and completion token ids) and is told
//! later how it went (a reward in `-1..1`); this module turns the (episode, outcome) pairs into a
//! PEFT adapter folder that `--lora` (and the server) loads on the same base model
//! ([`crate::lora`]), and decides from held-out episodes whether the adapter is worth keeping.
//!
//! What trains: the dense full-attention models of the Qwen path ([`crate::qwen::QwenModel`]):
//! Qwen3 and Llama 3 (see [`Trainer::new`] for the exact conditions). There is no autograd: the
//! forward keeps what the backward needs ([`Tape`]) and the backward is written by hand through
//! the LM head, RMSNorm, the SiLU MLP, causal attention with RoPE and query / key norms, down to
//! the lowest adapted layer. Only the adapter's `A` / `B` matrices get gradients (PEFT's
//! convention: `W x + (alpha / r) B A x`, `B` zero at the start).
//!
//! Precision: everything is f32 on the CPU. The frozen matmuls of the forward run through the
//! inference kernels on the weights as loaded (bf16, Q8 / Q4, GGUF blocks), so the loss is the
//! one the served model has. The backward needs `W^T g`, which no kernel computes, so each frozen
//! weight above the lowest adapted layer is transposed once at start ([`transpose`]): bf16
//! weights stay bf16 (the same values), every other format is dequantized to f32 (the values the
//! format stores). Quantized formats are then backpropagated as the linear map of their
//! dequantized weights, although their forward kernels also round the activations to int8: the
//! gradient is exact for bf16 checkpoints and a straight-through estimate for quantized ones.
//!
//! The objective ([`token_loss`]) is a reward-weighted token likelihood over the completion
//! tokens; see docs/learning.md.

use crate::ggml::{GgmlType, GgufMatrix};
use crate::kernels::{
    add_inplace, apply_rope, apply_rope_interleaved, axpy, dot, fast_exp, for_chunks, rmsnorm_f32w, rmsnorm_f32w_rows, rope_tables,
    silu_mul, sum_squares,
};
use crate::model::{LayerCache, Lin, Weight, attention, linear, linear_multi};
use crate::pool;
use crate::qwen::{Ffn, Mixer, QwenModel};
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The linear layers of a decoder layer an adapter can target, by their PEFT module name.
pub const TARGETS: [&str; 7] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
const Q: usize = 0;
const K: usize = 1;
const V: usize = 2;
const O: usize = 3;
const GATE: usize = 4;
const UP: usize = 5;
const DOWN: usize = 6;

/// The module path of target `slot` in layer `l`, from `layers.` on.
fn module_name(l: usize, slot: usize) -> String {
    let part = if slot <= O { "self_attn" } else { "mlp" };
    format!("layers.{l}.{part}.{}", TARGETS[slot])
}

/// What an adapter needs to know of the model class it is for: which linear layers of a decoder
/// layer it can target (at most seven, by index: a "slot"), their shapes and their PEFT names.
pub trait Family {
    /// The targets' names, by slot.
    fn target_names(&self) -> &'static [&'static str];
    fn n_layers(&self) -> usize;
    /// `(n_out, n_in)` of target `slot` in layer `l`.
    fn dims(&self, l: usize, slot: usize) -> (usize, usize);
    /// The module path of target `slot` in layer `l`, from `layers.` on.
    fn module_name(&self, l: usize, slot: usize) -> String;
    /// What an adapter file's keys put before the module path.
    fn key_prefix(&self) -> &'static str {
        "base_model.model.model."
    }
    /// The adapter row the model stores at each row of target `slot`'s weight, when the
    /// checkpoint reorders them.
    fn stored_rows(&self, _slot: usize) -> Option<Vec<usize>> {
        None
    }
}

impl Family for QwenModel<'_> {
    fn target_names(&self) -> &'static [&'static str] {
        &TARGETS
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn dims(&self, _l: usize, slot: usize) -> (usize, usize) {
        dims(self, slot)
    }
    fn module_name(&self, l: usize, slot: usize) -> String {
        module_name(l, slot)
    }
    fn stored_rows(&self, slot: usize) -> Option<Vec<usize>> {
        stored_rows(self, slot)
    }
}

/// One episode with its outcome.
#[derive(Clone, Debug)]
pub struct Episode {
    pub id: String,
    pub prompt: Vec<u32>,
    pub completion: Vec<u32>,
    /// The outcome's reward (clamped to `-1..1`) times its kind's weight.
    pub weight: f32,
}

/// What [`load_episodes`] read.
#[derive(Clone, Debug, Default)]
pub struct LoadStats {
    /// Lines of `episodes.jsonl`.
    pub episodes: usize,
    /// Episodes no outcome names.
    pub without_outcome: usize,
    /// Episodes whose weight is 0, or without completion tokens.
    pub empty: usize,
    /// Outcomes by kind (the last one of each episode).
    pub kinds: BTreeMap<String, usize>,
}

/// Parses `kind=weight,...` (`--kind-weight tests=1,user_praise=0.3`).
pub fn parse_kind_weights(s: &str) -> Result<Vec<(String, f32)>> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let (k, v) = p
                .split_once('=')
                .with_context(|| format!("kind weight {p:?}: expected KIND=WEIGHT"))?;
            let w: f32 = v.trim().parse().with_context(|| format!("kind weight {p:?}"))?;
            if !w.is_finite() {
                bail!("kind weight {p:?} is not a finite number");
            }
            Ok((k.trim().to_string(), w))
        })
        .collect()
}

/// Reads `episodes.jsonl` (`{"id", "prompt_tokens": [ids], "completion_tokens": [ids], ...}`)
/// and `outcomes.jsonl` (`{"id", "reward": -1..1, "kind", ...}`; the last line of an id with a
/// reward wins).
/// Episodes without an outcome are left out, as are those whose weight (the reward times the
/// weight of its kind in `kind_weights`, 1 for kinds not listed) is 0. The logged `logprobs` are
/// not read: an episode was sampled under whatever adapters were attached then, so they are not
/// the base model's.
pub fn load_episodes(episodes: &Path, outcomes: &Path, kind_weights: &[(String, f32)]) -> Result<(Vec<Episode>, LoadStats)> {
    let mut stats = LoadStats::default();
    let mut rewards: HashMap<String, (f32, String)> = HashMap::new();
    for (n, v) in json_lines(outcomes)? {
        let id = id_of(&v).with_context(|| format!("{}:{}: no id", outcomes.display(), n + 1))?;
        // A line without a reward (null: a note alone) says nothing to train on.
        let Some(r) = v.get("reward").and_then(|r| r.as_f64()).map(|r| r as f32).filter(|r| r.is_finite()) else {
            continue;
        };
        let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or_default().to_string();
        rewards.insert(id, (r.clamp(-1.0, 1.0), kind));
    }
    let mut out = Vec::new();
    let mut seen = HashMap::new();
    for (n, v) in json_lines(episodes)? {
        let at = || format!("{}:{}", episodes.display(), n + 1);
        let id = id_of(&v).with_context(|| format!("{}: no id", at()))?;
        stats.episodes += 1;
        let Some((reward, kind)) = rewards.get(&id) else {
            stats.without_outcome += 1;
            continue;
        };
        let toks = |k: &str| -> Result<Vec<u32>> {
            let a = v.get(k).and_then(|a| a.as_array()).with_context(|| format!("{}: no {k}", at()))?;
            a.iter()
                .map(|t| {
                    t.as_u64()
                        .map(|t| t as u32)
                        .with_context(|| format!("{}: {k} holds a non-integer", at()))
                })
                .collect()
        };
        let kw = kind_weights.iter().find(|(k, _)| k == kind).map_or(1.0, |(_, w)| *w);
        let ep = Episode {
            prompt: toks("prompt_tokens")?,
            completion: toks("completion_tokens")?,
            weight: reward * kw,
            id,
        };
        *stats.kinds.entry(kind.clone()).or_default() += 1;
        if ep.weight == 0.0 || ep.completion.is_empty() {
            stats.empty += 1;
            continue;
        }
        // An id logged twice: the last line wins, as for outcomes.
        match seen.insert(ep.id.clone(), out.len()) {
            Some(i) => out[i] = ep,
            None => out.push(ep),
        }
    }
    Ok((out, stats))
}

/// The JSON objects of a log, with their line index. A last line that does not parse is left
/// out (the server appends: a crash can cut the line it was writing); any other is an error.
fn json_lines(path: &Path) -> Result<Vec<(usize, serde_json::Value)>> {
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let lines: Vec<(usize, &str)> = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()).collect();
    let mut out = Vec::with_capacity(lines.len());
    for (i, (n, line)) in lines.iter().enumerate() {
        match serde_json::from_str(line) {
            Ok(v) => out.push((*n, v)),
            Err(_) if i + 1 == lines.len() => {}
            Err(e) => return Err(e).with_context(|| format!("{}:{}", path.display(), n + 1)),
        }
    }
    Ok(out)
}

/// The id of a log line: a string, or a number.
fn id_of(v: &serde_json::Value) -> Option<String> {
    match v.get("id")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// The trainable update of one linear layer: `y += scale * B (A x)`.
#[derive(Clone)]
pub(crate) struct Module {
    pub(crate) layer: usize,
    pub(crate) slot: usize,
    pub(crate) n_in: usize,
    pub(crate) n_out: usize,
    /// `[r, n_in]`
    pub(crate) a: Vec<f32>,
    /// `[r, n_out]`: B transposed, rows in the order the model stores the weight's rows.
    pub(crate) bt: Vec<f32>,
}

/// A LoRA adapter over the trainer's targets: the one being trained, or an existing one read
/// back as the starting point and as what the gate compares against.
#[derive(Clone)]
pub struct Adapter {
    pub rank: usize,
    pub alpha: f32,
    /// The targets, as indices into [`TARGETS`], sorted.
    slots: Vec<usize>,
    /// The adapted layers, sorted.
    layers: Vec<usize>,
    pub(crate) mods: Vec<Module>,
    /// The module of each (layer, slot).
    index: Vec<[Option<usize>; 7]>,
    /// The targets' names by slot ([`Family::target_names`]).
    names: &'static [&'static str],
}

/// Gradients of an adapter's modules, in their order: (`dA`, `dB^T`).
pub type Grads = Vec<(Vec<f32>, Vec<f32>)>;

/// xorshift64*: the trainer's random numbers (initialisation, shuffling).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A float tensor of an adapter file as f32.
fn tensor_f32(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let info = st.info(name).with_context(|| format!("missing tensor {name}"))?;
    let b = st.bytes(name)?;
    let v = match info.dtype.as_str() {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "BF16" => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
        d => bail!("LoRA tensor {name} has dtype {d}, expected a float type"),
    };
    Ok((v, info.shape.clone()))
}

impl Adapter {
    /// A new adapter of `rank` on `targets` (names of [`TARGETS`]) of `layers` (all when None):
    /// `A` uniform in `+-1 / sqrt(n_in)` (PEFT's default), `B` zero.
    pub fn new<F: Family + ?Sized>(
        model: &F,
        rank: usize,
        alpha: f32,
        targets: &[String],
        layers: Option<&[usize]>,
        seed: u64,
    ) -> Result<Self> {
        if rank == 0 {
            bail!("LoRA rank 0");
        }
        let nl = model.n_layers();
        let names = model.target_names();
        let mut slots = Vec::new();
        for t in targets {
            let s = names
                .iter()
                .position(|n| n == t)
                .with_context(|| format!("unknown LoRA target {t:?} (one of {})", names.join(", ")))?;
            if !slots.contains(&s) {
                slots.push(s);
            }
        }
        slots.sort_unstable();
        if slots.is_empty() {
            bail!("no LoRA targets");
        }
        let mut layers: Vec<usize> = layers.map_or_else(|| (0..nl).collect(), <[usize]>::to_vec);
        layers.sort_unstable();
        layers.dedup();
        if layers.is_empty() || layers.iter().any(|&l| l >= nl) {
            bail!("LoRA layers {layers:?}: the model has layers 0-{}", nl - 1);
        }
        let mut rng = Rng::new(seed);
        let mut ad = Self {
            rank,
            alpha,
            slots: slots.clone(),
            layers: layers.clone(),
            mods: Vec::new(),
            index: vec![[None; 7]; nl],
            names,
        };
        for &l in &layers {
            for &s in &slots {
                let (n_out, n_in) = model.dims(l, s);
                let bound = 1.0 / (n_in as f32).sqrt();
                let a = (0..rank * n_in).map(|_| (2.0 * rng.unit() - 1.0) * bound).collect();
                ad.push(Module {
                    layer: l,
                    slot: s,
                    n_in,
                    n_out,
                    a,
                    bt: vec![0f32; rank * n_out],
                });
            }
        }
        Ok(ad)
    }

    fn push(&mut self, m: Module) {
        self.index[m.layer][m.slot] = Some(self.mods.len());
        self.mods.push(m);
    }

    /// `alpha / r`.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// The target names.
    pub fn targets(&self) -> Vec<String> {
        self.slots.iter().map(|&s| self.names[s].to_string()).collect()
    }

    /// The lowest adapted layer.
    pub fn first_layer(&self) -> usize {
        self.layers[0]
    }

    /// Trainable parameters.
    pub fn params(&self) -> usize {
        self.mods.iter().map(|m| m.a.len() + m.bt.len()).sum()
    }

    /// Whether `o` has the same rank, scale, targets and layers (so that training can go on from
    /// its matrices).
    pub fn same_shape(&self, o: &Adapter) -> bool {
        self.rank == o.rank && self.alpha == o.alpha && self.slots == o.slots && self.layers == o.layers
    }

    /// The index (into the adapter's modules and into [`Grads`]) of the module on target `slot`
    /// of layer `l`.
    pub(crate) fn at(&self, l: usize, slot: usize) -> Option<usize> {
        self.index.get(l).and_then(|i| i[slot])
    }

    /// Each module's layer, target name, `A` (`[r, n_in]`) and `B^T` (`[r, n_out]`, rows in the
    /// order the model stores the weight's rows).
    pub fn modules(&self) -> impl Iterator<Item = (usize, &'static str, &[f32], &[f32])> {
        self.mods.iter().map(|m| (m.layer, self.names[m.slot], &m.a[..], &m.bt[..]))
    }

    /// Overwrites the matrices of the module on target `name` of layer `l` (`a`: `[r, n_in]`,
    /// `bt`: `[r, n_out]`).
    pub fn set_module(&mut self, l: usize, name: &str, a: &[f32], bt: &[f32]) -> Result<()> {
        let i = self
            .names
            .iter()
            .position(|n| *n == name)
            .and_then(|s| self.at(l, s))
            .with_context(|| format!("the adapter has no {name} in layer {l}"))?;
        let m = &mut self.mods[i];
        if a.len() != m.a.len() || bt.len() != m.bt.len() {
            bail!(
                "{name} of layer {l}: {} and {} values, expected {} and {}",
                a.len(),
                bt.len(),
                m.a.len(),
                m.bt.len()
            );
        }
        m.a.copy_from_slice(a);
        m.bt.copy_from_slice(bt);
        Ok(())
    }

    /// Zero gradients of this adapter's shapes.
    pub fn zero_grads(&self) -> Grads {
        self.mods.iter().map(|m| (vec![0f32; m.a.len()], vec![0f32; m.bt.len()])).collect()
    }

    /// Writes the PEFT folder `dir` (`adapter_config.json`, `adapter_model.safetensors` in f32)
    /// for the base model `base` (recorded as `base_model_name_or_path`).
    pub fn save<F: Family + ?Sized>(&self, model: &F, dir: &Path, base: &str) -> Result<()> {
        std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
        let cfg = serde_json::json!({
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
            "r": self.rank, "lora_alpha": self.alpha, "lora_dropout": 0.0, "bias": "none",
            "target_modules": self.targets(), "layers_to_transform": self.layers,
            "fan_in_fan_out": false, "use_rslora": false, "use_dora": false, "inference_mode": true,
            "init_lora_weights": true, "modules_to_save": null, "rank_pattern": {}, "alpha_pattern": {},
        });
        std::fs::write(dir.join("adapter_config.json"), serde_json::to_vec_pretty(&cfg)?)?;
        let r = self.rank;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let mut data: Vec<u8> = Vec::new();
        for m in &self.mods {
            let rows = model.stored_rows(m.slot);
            let mut b = vec![0f32; m.n_out * r];
            for p in 0..m.n_out {
                let o = rows.as_ref().map_or(p, |rs| rs[p]);
                for k in 0..r {
                    b[o * r + k] = m.bt[k * m.n_out + p];
                }
            }
            let key = format!("{}{}", model.key_prefix(), model.module_name(m.layer, m.slot));
            for (suffix, shape, vals) in [("lora_A", [r, m.n_in], &m.a), ("lora_B", [m.n_out, r], &b)] {
                let start = data.len();
                for x in vals {
                    data.extend_from_slice(&x.to_le_bytes());
                }
                header.insert(
                    format!("{key}.{suffix}.weight"),
                    serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
                );
            }
        }
        let mut h = serde_json::to_string(&serde_json::Value::Object(header))?.into_bytes();
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut out = Vec::with_capacity(8 + h.len() + data.len());
        out.extend_from_slice(&(h.len() as u64).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        let path = dir.join("adapter_model.safetensors");
        std::fs::write(&path, out).with_context(|| path.display().to_string())
    }

    /// Reads back a PEFT folder whose modules are all among [`TARGETS`] of `model`'s layers, with
    /// one rank (what [`Adapter::save`] writes; adapters with other modules, rsLoRA, DoRA or
    /// per-module ranks are an error: the trainer cannot run them).
    pub fn load<F: Family + ?Sized>(model: &F, dir: &Path) -> Result<Self> {
        let names = model.target_names();
        let cfg_path = dir.join("adapter_config.json");
        let cfg: serde_json::Value =
            serde_json::from_slice(&crate::fsx::read(&cfg_path)?).with_context(|| format!("parse {}", cfg_path.display()))?;
        for k in ["use_rslora", "use_dora", "fan_in_fan_out"] {
            if cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(false) {
                bail!("{}: {k} is not supported by the trainer", cfg_path.display());
            }
        }
        let rank = cfg.get("r").and_then(|v| v.as_u64()).context("adapter_config.json: missing r")? as usize;
        let alpha = cfg.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(rank as f64) as f32;
        let st = SafeTensors::open(&dir.join("adapter_model.safetensors"), true)?;
        let nl = model.n_layers();
        let mut found: BTreeMap<(usize, usize), (Option<String>, Option<String>)> = BTreeMap::new();
        for name in st.names() {
            let (module, is_a) = if let Some(m) = name.strip_suffix(".lora_A.weight") {
                (m, true)
            } else if let Some(m) = name.strip_suffix(".lora_B.weight") {
                (m, false)
            } else {
                bail!("{}: tensor {name} is not a lora_A / lora_B weight", dir.display());
            };
            let key = module
                .find("layers.")
                .map(|i| &module[i..])
                .with_context(|| format!("{}: module {module} is not in a decoder layer", dir.display()))?;
            let mut it = key.split('.').skip(1);
            let l: usize = it.next().and_then(|n| n.parse().ok()).with_context(|| format!("module {module}"))?;
            let rest: Vec<&str> = it.collect();
            let slot = names.iter().position(|t| rest.last() == Some(t));
            let slot = slot.filter(|&s| model.module_name(l, s) == key && l < nl);
            let slot = slot.with_context(|| format!("{}: the trainer has no module {module}", dir.display()))?;
            let e = found.entry((l, slot)).or_default();
            *(if is_a { &mut e.0 } else { &mut e.1 }) = Some(name.clone());
        }
        let mut ad = Self {
            rank,
            alpha,
            slots: Vec::new(),
            layers: Vec::new(),
            mods: Vec::new(),
            index: vec![[None; 7]; nl],
            names,
        };
        for ((l, slot), (an, bn)) in found {
            let (Some(an), Some(bn)) = (an, bn) else {
                bail!(
                    "{}: {} lacks its lora_A or lora_B weight",
                    dir.display(),
                    model.module_name(l, slot)
                );
            };
            let (n_out, n_in) = model.dims(l, slot);
            let (a, ash) = tensor_f32(&st, &an)?;
            let (b, bsh) = tensor_f32(&st, &bn)?;
            if ash != [rank, n_in] || bsh != [n_out, rank] {
                bail!(
                    "{}: {} is {bsh:?} x {ash:?}, expected [{n_out}, {rank}] x [{rank}, {n_in}]",
                    dir.display(),
                    model.module_name(l, slot)
                );
            }
            let rows = model.stored_rows(slot);
            let mut bt = vec![0f32; rank * n_out];
            for p in 0..n_out {
                let o = rows.as_ref().map_or(p, |rs| rs[p]);
                for k in 0..rank {
                    bt[k * n_out + p] = b[o * rank + k];
                }
            }
            if !ad.slots.contains(&slot) {
                ad.slots.push(slot);
            }
            if !ad.layers.contains(&l) {
                ad.layers.push(l);
            }
            ad.push(Module {
                layer: l,
                slot,
                n_in,
                n_out,
                a,
                bt,
            });
        }
        ad.slots.sort_unstable();
        ad.layers.sort_unstable();
        if ad.mods.is_empty() {
            bail!("{}: no LoRA modules", dir.display());
        }
        if ad.mods.len() != ad.slots.len() * ad.layers.len() {
            bail!("{}: not every adapted layer has every target", dir.display());
        }
        Ok(ad)
    }
}

/// `(n_out, n_in)` of target `slot`.
fn dims(model: &QwenModel, slot: usize) -> (usize, usize) {
    let c = &model.cfg;
    let (h, qw, kvw) = (
        c.hidden_size,
        c.num_attention_heads * c.head_dim,
        c.num_key_value_heads * c.head_dim,
    );
    match slot {
        Q => (qw, h),
        K | V => (kvw, h),
        O => (h, qw),
        GATE | UP => (c.intermediate_size, h),
        _ => (h, c.intermediate_size),
    }
}

/// The adapter row the model stores at each row of target `slot`'s weight, when the checkpoint
/// reorders them (Llama GGUF: query / key rows in llama.cpp's RoPE order).
fn stored_rows(model: &QwenModel, slot: usize) -> Option<Vec<usize>> {
    let c = &model.cfg;
    match slot {
        Q if c.rope_interleaved => Some(crate::lora::rope_pair_rows(c.num_attention_heads, c.head_dim)),
        K if c.rope_interleaved => Some(crate::lora::rope_pair_rows(c.num_key_value_heads, c.head_dim)),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// The transpose of the frozen weight `l` (`[n_out, n_in]`) as a weight `[n_in, n_out]` the
/// matmul kernels run, for `W^T g`: bf16 for a bf16 weight (the same values), f32 for every other
/// format (its dequantized values).
pub fn transpose(l: &Lin, n_out: usize, n_in: usize) -> Result<Lin<'static>> {
    if l.fold.is_some() {
        bail!("Hadamard-folded weights cannot be trained on (their rows are in another basis)");
    }
    const ROWS: usize = 64;
    let blocks = n_out.div_ceil(ROWS);
    if let Weight::Bf16(w, _) = &l.w {
        let mut out = crate::quant::PageVec::<u16>::zeroed(n_out * n_in);
        let op = SendPtr(out.as_mut_ptr());
        pool::run(blocks, &|b| {
            for r in b * ROWS..((b + 1) * ROWS).min(n_out) {
                for (i, &x) in w[r * n_in..(r + 1) * n_in].iter().enumerate() {
                    // SAFETY: column `r` of the output belongs to this block alone.
                    unsafe { *op.get().add(i * n_out + r) = x };
                }
            }
        });
        return Ok(Lin::bf16_owned(out));
    }
    let mut out = crate::quant::PageVec::<u8>::zeroed(n_out * n_in * 4);
    let op = SendPtr(out.as_mut_ptr());
    pool::run(blocks, &|b| {
        let mut row = vec![0f32; n_in];
        for r in b * ROWS..((b + 1) * ROWS).min(n_out) {
            l.rows(&[r as u32], n_in, &mut row);
            for (i, &x) in row.iter().enumerate() {
                // SAFETY: column `r` of the output belongs to this block alone.
                unsafe { (op.get().add((i * n_out + r) * 4) as *mut f32).write_unaligned(x) };
            }
        }
    });
    Ok(Lin::gguf(GgufMatrix::new(GgmlType::F32, out.into(), n_in, n_out)?))
}

/// How `model` stores its projections (the first layer's query weight): `bf16`, `q8`, `q4` or a
/// GGUF block type.
pub fn weight_format(model: &QwenModel) -> String {
    let Some(Mixer::Attn(at)) = model.layers.first().map(|l| &l.mixer) else {
        return "unknown".into();
    };
    match &at.q.w {
        Weight::Bf16(..) => "bf16".into(),
        Weight::Q8(_) => "q8".into(),
        Weight::Q4(_) => "q4".into(),
        Weight::Gguf(m) => format!("GGUF {}", m.kind),
    }
}

/// The transposed frozen weights of one layer.
struct LayerT {
    /// Query, key and value: None in the lowest adapted layer, below which nothing is needed.
    qkv: Option<[Lin<'static>; 3]>,
    o: Lin<'static>,
    gate: Lin<'static>,
    up: Lin<'static>,
    down: Lin<'static>,
}

/// What the backward needs of one layer's forward.
struct LayerRec {
    /// The residual entering the layer and after the attention.
    x_in: Vec<f32>,
    x_mid: Vec<f32>,
    /// Their norms (the inputs of the projections).
    hn: Vec<f32>,
    hn2: Vec<f32>,
    /// The query and key projections before their norms (models with them).
    qraw: Vec<f32>,
    kraw: Vec<f32>,
    /// The queries as attended with (normed, rotated, scaled).
    q: Vec<f32>,
    /// The keys (normed, rotated) and values.
    cache: LayerCache,
    att: Vec<f32>,
    /// The gate and up projections, and `silu(gate) * up`.
    g: Vec<f32>,
    u: Vec<f32>,
    act: Vec<f32>,
    /// `A x` of each adapted target.
    lora_u: [Option<Vec<f32>>; 7],
}

/// A forward pass recorded for the backward.
struct Tape {
    t: usize,
    /// Per layer, from the lowest adapted one.
    layers: Vec<LayerRec>,
    /// The residual before the final norm.
    x_last: Vec<f32>,
    cos: Vec<f32>,
    /// `-sin`: rotating with it undoes the rotation.
    neg_sin: Vec<f32>,
}

/// Loss options.
#[derive(Clone, Copy, Debug)]
pub struct Objective {
    /// The cap on the odds `p / (1 - p)` that scale the push-down of a negative episode's token.
    pub neg_clip: f32,
}

/// The loss of one completion token and its gradient: `row` holds the logits and is overwritten
/// by `d loss / d logits`. With `p` the model's probability of the token `target`:
/// - `positive`: `-ln p` (imitation);
/// - otherwise `-ln(1 - p)` (the token is pushed down until it is unlikely, where the loss and
///   its gradient vanish), continued past odds `p / (1 - p) = neg_clip` by its tangent in `ln p`,
///   so a token the model is nearly sure of is pushed with a bounded factor rather than
///   `1 / (1 - p)`.
pub fn token_loss(row: &mut [f32], target: usize, positive: bool, neg_clip: f32) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&z| ((z - max) as f64).exp()).sum();
    let lse = max as f64 + sum.ln();
    let logp = row[target] as f64 - lse;
    // d loss / d logit_j = coef * (p_j - [j == target])
    let (loss, coef) = if positive {
        (-logp, 1.0)
    } else {
        let p = logp.exp();
        let c = neg_clip as f64;
        if p < c * (1.0 - p) {
            (-(-p).ln_1p(), -p / (1.0 - p))
        } else {
            let p0 = c / (1.0 + c);
            (-(-p0).ln_1p() + c * (logp - p0.ln()), -c)
        }
    };
    for z in row.iter_mut() {
        *z = (coef * (*z as f64 - lse).exp()) as f32;
    }
    row[target] -= coef as f32;
    loss
}

/// `gx = d loss / d x` for `y = rmsnorm(x) * w` given `gy`.
fn rms_back(x: &[f32], w: &[f32], eps: f32, gy: &[f32], gx: &mut [f32]) {
    let n = x.len();
    let r = (sum_squares(x) / n as f32 + eps).powf(-0.5);
    let mut c = 0f32;
    for i in 0..n {
        c += gy[i] * w[i] * x[i];
    }
    let k = r * r * r * c / n as f32;
    for i in 0..n {
        gx[i] = r * w[i] * gy[i] - x[i] * k;
    }
}

/// [`rms_back`] over rows of `n`.
fn rms_back_rows(x: &[f32], n: usize, w: &[f32], eps: f32, gy: &[f32], gx: &mut [f32]) {
    for_chunks(gx, n, x.len() > 4 * n, |i, o| {
        rms_back(&x[i * n..(i + 1) * n], w, eps, &gy[i * n..(i + 1) * n], o)
    });
}

/// `y += scale * B (A x)` for `t` rows; returns `A x` (`[t, r]`).
fn lora_fwd(m: &Module, r: usize, scale: f32, x: &[f32], t: usize, y: &mut [f32]) -> Vec<f32> {
    let mut u = vec![0f32; t * r];
    for_chunks(&mut u, r, t > 1, |ti, ur| {
        let xr = &x[ti * m.n_in..(ti + 1) * m.n_in];
        for (k, uk) in ur.iter_mut().enumerate() {
            *uk = dot(&m.a[k * m.n_in..(k + 1) * m.n_in], xr);
        }
    });
    for_chunks(y, m.n_out, t > 1, |ti, yr| {
        for k in 0..r {
            axpy(yr, scale * u[ti * r + k], &m.bt[k * m.n_out..(k + 1) * m.n_out]);
        }
    });
    u
}

/// The backward of [`lora_fwd`]: adds the gradients of `A` and `B^T` to `g`, and, with `gx`, the
/// update's part of `d loss / d x`.
#[allow(clippy::too_many_arguments)]
fn lora_back(
    m: &Module,
    r: usize,
    scale: f32,
    x: &[f32],
    u: &[f32],
    gy: &[f32],
    t: usize,
    gx: Option<&mut [f32]>,
    g: &mut (Vec<f32>, Vec<f32>),
) {
    let (n_in, n_out) = (m.n_in, m.n_out);
    // z = B^T gy
    let mut z = vec![0f32; t * r];
    for_chunks(&mut z, r, t > 1, |ti, zr| {
        let gr = &gy[ti * n_out..(ti + 1) * n_out];
        for (k, zk) in zr.iter_mut().enumerate() {
            *zk = dot(&m.bt[k * n_out..(k + 1) * n_out], gr);
        }
    });
    for_chunks(&mut g.1, n_out, true, |k, row| {
        for ti in 0..t {
            axpy(row, scale * u[ti * r + k], &gy[ti * n_out..(ti + 1) * n_out]);
        }
    });
    for_chunks(&mut g.0, n_in, true, |k, row| {
        for ti in 0..t {
            axpy(row, scale * z[ti * r + k], &x[ti * n_in..(ti + 1) * n_in]);
        }
    });
    if let Some(gx) = gx {
        for_chunks(gx, n_in, t > 1, |ti, row| {
            for k in 0..r {
                axpy(row, scale * z[ti * r + k], &m.a[k * n_in..(k + 1) * n_in]);
            }
        });
    }
}

/// The forward and backward passes of one model class, as the training loop ([`run_with`]) drives
/// them.
pub trait Passes {
    /// The mean loss of `ep`'s completion tokens ([`token_loss`], not yet weighted) under adapter
    /// `ad` (None: the base model); with `grads`, also adds `factor` times its gradient to them.
    fn episode(&self, ad: Option<&Adapter>, ep: &Episode, obj: Objective, grads: Option<(&mut Grads, f32)>) -> Result<f64>;
    /// One line on what the passes hold, for the log.
    fn describe(&self) -> String;
    /// Bytes of transposed weights held.
    fn transposed_bytes(&self) -> usize {
        0
    }
}

impl Passes for Trainer<'_, '_> {
    fn episode(&self, ad: Option<&Adapter>, ep: &Episode, obj: Objective, grads: Option<(&mut Grads, f32)>) -> Result<f64> {
        Trainer::episode(self, ad, ep, obj, grads)
    }
    fn describe(&self) -> String {
        format!(
            "transposed weights from layer {}: {:.2} GB",
            self.first,
            self.transposed_bytes as f64 / 1e9
        )
    }
    fn transposed_bytes(&self) -> usize {
        self.transposed_bytes
    }
}

/// The forward and backward passes of a model for LoRA training.
pub struct Trainer<'m, 'a> {
    model: &'m QwenModel<'a>,
    /// The lowest layer the backward reaches.
    first: usize,
    /// Transposed weights of the layers from `first`.
    wt: Vec<LayerT>,
    head_t: Lin<'static>,
    /// Bytes held by the transposed weights.
    pub transposed_bytes: usize,
}

impl<'m, 'a> Trainer<'m, 'a> {
    /// A trainer for adapters whose lowest layer is `first` or above. Errors for models the
    /// backward does not cover: only layers of plain causal full attention (with or without
    /// per-head query / key norms, any RoPE layout, grouped key / value heads) and dense SiLU MLPs
    /// train, without attached adapters.
    pub fn new(model: &'m QwenModel<'a>, first: usize) -> Result<Self> {
        let c = &model.cfg;
        let unsupported = [
            (c.is_exp(), "Qwen4-Exp hyper-connections"),
            (c.lfm3.is_some(), "LFM3 layers"),
            (c.attn_output_gate, "gated attention (Qwen3.5 / 3.8)"),
            (c.attention_bias, "attention biases (Qwen2)"),
            (c.bidirectional, "bidirectional attention"),
            (c.granite_scales().is_some(), "Granite's multipliers"),
            (c.attention_multiplier.is_some(), "an attention multiplier"),
            (c.qk_norm_full(), "per-layer query / key norms"),
            (c.lightning, "lightning attention"),
            (c.num_experts.is_some(), "mixture-of-experts MLPs"),
            (!matches!(c.mlp_act(), crate::moe::Act::Silu), "an MLP activation other than SiLU"),
            (!c.suppress_tokens.is_empty(), "suppressed tokens"),
        ];
        if let Some((_, what)) = unsupported.iter().find(|(on, _)| *on) {
            bail!("LoRA training does not cover this model: {what}");
        }
        let (h, inter) = (c.hidden_size, c.intermediate_size);
        let (qw, kvw) = (c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim);
        if first >= model.layers.len() {
            bail!("layer {first}: the model has {} layers", model.layers.len());
        }
        for (l, layer) in model.layers.iter().enumerate() {
            let Mixer::Attn(at) = &layer.mixer else {
                bail!("LoRA training does not cover this model: layer {l} is not full attention (Gated DeltaNet, convolution)");
            };
            let Ffn::Dense(mlp) = &layer.ffn else {
                bail!("LoRA training does not cover this model: layer {l} has a mixture-of-experts MLP");
            };
            if at.indexer.is_some()
                || at.head_gate.is_some()
                || at.window.is_some()
                || layer.post.is_some()
                || c.postnorm_scales(l).is_some()
            {
                bail!("LoRA training does not cover this model: layer {l} has sparse, windowed or post-normed attention");
            }
            if at.lins().iter().chain(&[&mlp.gate, &mlp.up, &mlp.down]).any(|w| w.has_lora()) {
                bail!("the model has LoRA adapters attached: train on the base model");
            }
        }
        if model.lm_head.has_lora() {
            bail!("the model has LoRA adapters attached: train on the base model");
        }
        let mut wt = Vec::new();
        for (l, layer) in model.layers.iter().enumerate().skip(first) {
            let (Mixer::Attn(at), Ffn::Dense(mlp)) = (&layer.mixer, &layer.ffn) else {
                unreachable!()
            };
            wt.push(LayerT {
                qkv: if l > first {
                    Some([transpose(&at.q, qw, h)?, transpose(&at.k, kvw, h)?, transpose(&at.v, kvw, h)?])
                } else {
                    None
                },
                o: transpose(&at.o, h, qw)?,
                gate: transpose(&mlp.gate, inter, h)?,
                up: transpose(&mlp.up, inter, h)?,
                down: transpose(&mlp.down, h, inter)?,
            });
        }
        let head_t = transpose(&model.lm_head, c.vocab_size, h)?;
        let transposed_bytes = wt
            .iter()
            .map(|w| {
                w.qkv.iter().flatten().map(|l| l.bytes()).sum::<usize>() + w.o.bytes() + w.gate.bytes() + w.up.bytes() + w.down.bytes()
            })
            .sum::<usize>()
            + head_t.bytes();
        Ok(Self {
            model,
            first,
            wt,
            head_t,
            transposed_bytes,
        })
    }

    /// The final-normed hidden states `[t, hidden]` of `tokens` with adapter `ad` (None: the base
    /// model), and with `record` what the backward needs of the layers from `self.first`.
    fn forward(&self, ad: Option<&Adapter>, tokens: &[u32], record: bool) -> (Vec<f32>, Option<Tape>) {
        let model = self.model;
        let c = &model.cfg;
        let (t, h) = (tokens.len(), c.hidden_size);
        let (nh, nkv, hd, inter) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size);
        let (qw, kvw) = (nh * hd, nkv * hd);
        let eps = c.rms_norm_eps as f32;
        let par = t > 1;
        let half = model.inv_freq.len();
        let rope = if c.rope_interleaved { apply_rope_interleaved } else { apply_rope };
        let scale = c.attn_scale();
        let positions: Vec<usize> = (0..t).collect();
        let (cos, sin) = rope_tables(&positions, &model.inv_freq);
        let (r, s) = ad.map_or((0, 0.0), |a| (a.rank, a.scale()));
        let module = |l: usize, slot: usize| ad.and_then(|a| a.index[l][slot].map(|i| &a.mods[i]));

        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, par, |ti, row| model.embed.row(tokens[ti] as usize, row));
        let mut recs = Vec::new();
        for (l, layer) in model.layers.iter().enumerate() {
            let (Mixer::Attn(at), Ffn::Dense(mlp)) = (&layer.mixer, &layer.ffn) else {
                unreachable!("checked by Trainer::new")
            };
            let mut lora_u: [Option<Vec<f32>>; 7] = Default::default();
            let x_in = x.clone();
            let mut hn = vec![0f32; t * h];
            rmsnorm_f32w_rows(&x, h, &layer.ln_in, eps, &mut hn);
            let mut qraw = vec![0f32; t * qw];
            let mut kraw = vec![0f32; t * kvw];
            let mut v = vec![0f32; t * kvw];
            linear_multi(
                &mut [(&at.q, qw, &mut qraw[..]), (&at.k, kvw, &mut kraw[..]), (&at.v, kvw, &mut v[..])],
                h,
                &hn,
                t,
            );
            for (slot, y) in [(Q, &mut qraw), (K, &mut kraw), (V, &mut v)] {
                if let Some(m) = module(l, slot) {
                    lora_u[slot] = Some(lora_fwd(m, r, s, &hn, t, y));
                }
            }
            let mut q = vec![0f32; t * qw];
            for_chunks(&mut q, hd, par, |i, qo| {
                let ti = i / nh;
                let src = &qraw[i * hd..(i + 1) * hd];
                match &at.q_norm {
                    Some(w) => rmsnorm_f32w(src, w, eps, qo),
                    None => qo.copy_from_slice(src),
                }
                rope(
                    &mut qo[..2 * half],
                    &cos[ti * half..(ti + 1) * half],
                    &sin[ti * half..(ti + 1) * half],
                );
                for v in qo.iter_mut() {
                    *v *= scale;
                }
            });
            let mut kn = vec![0f32; t * kvw];
            for_chunks(&mut kn, hd, par, |i, ko| {
                let ti = i / nkv;
                let src = &kraw[i * hd..(i + 1) * hd];
                match &at.k_norm {
                    Some(w) => rmsnorm_f32w(src, w, eps, ko),
                    None => ko.copy_from_slice(src),
                }
                rope(
                    &mut ko[..2 * half],
                    &cos[ti * half..(ti + 1) * half],
                    &sin[ti * half..(ti + 1) * half],
                );
            });
            let mut cache = LayerCache::new(nkv, hd);
            cache.push(&kn, &v, t);
            let mut att = vec![0f32; t * qw];
            attention(&q, &cache, nh, t, 0, None, &mut att);
            let mut y = vec![0f32; t * h];
            linear(&at.o, h, qw, &att, t, &mut y);
            if let Some(m) = module(l, O) {
                lora_u[O] = Some(lora_fwd(m, r, s, &att, t, &mut y));
            }
            add_inplace(&mut x, &y);
            let x_mid = x.clone();

            let mut hn2 = vec![0f32; t * h];
            rmsnorm_f32w_rows(&x, h, &layer.ln_post, eps, &mut hn2);
            let mut g = vec![0f32; t * inter];
            let mut u = vec![0f32; t * inter];
            linear_multi(&mut [(&mlp.gate, inter, &mut g[..]), (&mlp.up, inter, &mut u[..])], h, &hn2, t);
            for (slot, y) in [(GATE, &mut g), (UP, &mut u)] {
                if let Some(m) = module(l, slot) {
                    lora_u[slot] = Some(lora_fwd(m, r, s, &hn2, t, y));
                }
            }
            let mut act = g.clone();
            silu_mul(&mut act, &u);
            linear(&mlp.down, h, inter, &act, t, &mut y);
            if let Some(m) = module(l, DOWN) {
                lora_u[DOWN] = Some(lora_fwd(m, r, s, &act, t, &mut y));
            }
            add_inplace(&mut x, &y);
            if record && l >= self.first {
                let normed = at.q_norm.is_some();
                recs.push(LayerRec {
                    x_in,
                    x_mid,
                    hn,
                    hn2,
                    qraw: if normed { qraw } else { Vec::new() },
                    kraw: if normed { kraw } else { Vec::new() },
                    q,
                    cache,
                    att,
                    g,
                    u,
                    act,
                    lora_u,
                });
            }
        }
        let mut xf = vec![0f32; t * h];
        rmsnorm_f32w_rows(&x, h, &model.final_norm, eps, &mut xf);
        let tape = record.then(|| Tape {
            t,
            layers: recs,
            x_last: x,
            cos,
            neg_sin: sin.iter().map(|v| -v).collect(),
        });
        (xf, tape)
    }

    /// The logits `[t, vocab]` of `tokens` with adapter `ad` (None: the base model).
    pub fn logits(&self, ad: Option<&Adapter>, tokens: &[u32]) -> Vec<f32> {
        let c = &self.model.cfg;
        let (xf, _) = self.forward(ad, tokens, false);
        let mut out = vec![0f32; tokens.len() * c.vocab_size];
        linear(&self.model.lm_head, c.vocab_size, c.hidden_size, &xf, tokens.len(), &mut out);
        out
    }

    /// The mean loss of `ep`'s completion tokens ([`token_loss`], not yet weighted) under adapter
    /// `ad`; with `grads`, also adds `factor` times its gradient to them.
    pub fn episode(&self, ad: Option<&Adapter>, ep: &Episode, obj: Objective, grads: Option<(&mut Grads, f32)>) -> Result<f64> {
        let c = &self.model.cfg;
        let (h, vocab) = (c.hidden_size, c.vocab_size);
        let mut tokens = ep.prompt.clone();
        tokens.extend_from_slice(&ep.completion);
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= vocab) {
            bail!("episode {}: token {bad} is outside the model's vocabulary of {vocab}", ep.id);
        }
        // Position `i` predicts token `i + 1`; a completion without a prompt has no prediction
        // for its first token.
        let first = ep.prompt.len().max(1) - 1;
        let t = tokens.len() - 1;
        if t <= first {
            bail!("episode {}: nothing to predict", ep.id);
        }
        let n = t - first;
        let (xf, tape) = self.forward(ad, &tokens[..t], grads.is_some());
        let positive = ep.weight > 0.0;
        // The head in chunks of rows: the logits of a long completion are large.
        const ROWS: usize = 64;
        let mut loss = 0f64;
        let mut g_xf = vec![0f32; if grads.is_some() { t * h } else { 0 }];
        let mut logits = vec![0f32; ROWS.min(n) * vocab];
        for r0 in (first..t).step_by(ROWS) {
            let rows = ROWS.min(t - r0);
            let lg = &mut logits[..rows * vocab];
            linear(&self.model.lm_head, vocab, h, &xf[r0 * h..(r0 + rows) * h], rows, lg);
            let losses: Vec<Mutex<f64>> = (0..rows).map(|_| Mutex::new(0.0)).collect();
            for_chunks(lg, vocab, true, |i, row| {
                *losses[i].lock().unwrap() = token_loss(row, tokens[r0 + i + 1] as usize, positive, obj.neg_clip);
            });
            loss += losses.iter().map(|l| *l.lock().unwrap()).sum::<f64>();
            if grads.is_some() {
                linear(&self.head_t, h, vocab, lg, rows, &mut g_xf[r0 * h..(r0 + rows) * h]);
            }
        }
        if let (Some((grads, factor)), Some(tape), Some(ad)) = (grads, tape, ad) {
            let k = factor / n as f32;
            g_xf.iter_mut().for_each(|v| *v *= k);
            self.backward(ad, &tape, &g_xf, grads);
        }
        Ok(loss / n as f64)
    }

    /// Adds to `grads` the gradients of `ad`'s matrices given `g_xf`, the gradient of the loss
    /// with respect to the final-normed hidden states of the pass recorded in `tape`.
    fn backward(&self, ad: &Adapter, tape: &Tape, g_xf: &[f32], grads: &mut Grads) {
        assert!(
            ad.first_layer() >= self.first,
            "the adapter reaches below the trainer's lowest layer"
        );
        let model = self.model;
        let c = &model.cfg;
        let (t, h) = (tape.t, c.hidden_size);
        let (nh, nkv, hd, inter) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size);
        let (qw, kvw) = (nh * hd, nkv * hd);
        let eps = c.rms_norm_eps as f32;
        let par = t > 1;
        let half = model.inv_freq.len();
        let rope = if c.rope_interleaved { apply_rope_interleaved } else { apply_rope };
        let scale = c.attn_scale();
        let (r, s) = (ad.rank, ad.scale());
        let rep = nh / nkv;

        let mut gx = vec![0f32; t * h];
        rms_back_rows(&tape.x_last, h, &model.final_norm, eps, g_xf, &mut gx);
        let mut tmp = vec![0f32; t * h];
        for l in (ad.first_layer()..model.layers.len()).rev() {
            let layer = &model.layers[l];
            let Mixer::Attn(at) = &layer.mixer else { unreachable!() };
            let rec = &tape.layers[l - self.first];
            let wt = &self.wt[l - self.first];
            let lowest = l == ad.first_layer();
            let mut back = |slot: usize, x: &[f32], gy: &[f32], gx: Option<&mut [f32]>| {
                if let (Some(i), Some(u)) = (ad.index[l][slot], &rec.lora_u[slot]) {
                    lora_back(&ad.mods[i], r, s, x, u, gy, t, gx, &mut grads[i]);
                }
            };

            // ---- MLP: x = x_mid + down(silu(gate(hn2)) * up(hn2)) ----
            let mut g_act = vec![0f32; t * inter];
            linear(&wt.down, inter, h, &gx, t, &mut g_act);
            back(DOWN, &rec.act, &gx, Some(&mut g_act));
            let mut g_g = vec![0f32; t * inter];
            let mut g_u = g_act;
            {
                let chunk = 4096;
                let up = SendPtr(g_u.as_mut_ptr());
                for_chunks(&mut g_g, chunk, true, |ci, gg| {
                    let o = ci * chunk;
                    // SAFETY: each task owns the same range of `g_u` as its chunk of `g_g`.
                    let gu = unsafe { std::slice::from_raw_parts_mut(up.get().add(o), gg.len()) };
                    for i in 0..gg.len() {
                        let (x, u, ga) = (rec.g[o + i], rec.u[o + i], gu[i]);
                        let sg = 1.0 / (1.0 + fast_exp(-x));
                        gg[i] = ga * u * sg * (1.0 + x * (1.0 - sg));
                        gu[i] = ga * x * sg;
                    }
                });
            }
            let mut g_hn = vec![0f32; t * h];
            linear(&wt.gate, h, inter, &g_g, t, &mut g_hn);
            linear(&wt.up, h, inter, &g_u, t, &mut tmp);
            add_inplace(&mut g_hn, &tmp);
            back(GATE, &rec.hn2, &g_g, Some(&mut g_hn));
            back(UP, &rec.hn2, &g_u, Some(&mut g_hn));
            rms_back_rows(&rec.x_mid, h, &layer.ln_post, eps, &g_hn, &mut tmp);
            add_inplace(&mut gx, &tmp);

            // ---- Attention: x_mid = x_in + o(att) ----
            let mut g_att = vec![0f32; t * qw];
            linear(&wt.o, qw, h, &gx, t, &mut g_att);
            back(O, &rec.att, &gx, Some(&mut g_att));
            let (mut g_q, mut g_k, g_v) = attention_back(&rec.q, &rec.cache, &g_att, nh, t);
            // Queries: scaled, rotated, normed.
            let mut g_qraw = vec![0f32; t * qw];
            {
                let qp = SendPtr(g_q.as_mut_ptr());
                for_chunks(&mut g_qraw, hd, par, |i, out| {
                    let ti = i / nh;
                    // SAFETY: each task owns head `i` of `g_q`.
                    let gq = unsafe { std::slice::from_raw_parts_mut(qp.get().add(i * hd), hd) };
                    gq.iter_mut().for_each(|v| *v *= scale);
                    rope(
                        &mut gq[..2 * half],
                        &tape.cos[ti * half..(ti + 1) * half],
                        &tape.neg_sin[ti * half..(ti + 1) * half],
                    );
                    match &at.q_norm {
                        Some(w) => rms_back(&rec.qraw[i * hd..(i + 1) * hd], w, eps, gq, out),
                        None => out.copy_from_slice(gq),
                    }
                });
            }
            let mut g_kraw = vec![0f32; t * kvw];
            {
                let kp = SendPtr(g_k.as_mut_ptr());
                for_chunks(&mut g_kraw, hd, par, |i, out| {
                    let ti = i / nkv;
                    // SAFETY: each task owns head `i` of `g_k`.
                    let gk = unsafe { std::slice::from_raw_parts_mut(kp.get().add(i * hd), hd) };
                    rope(
                        &mut gk[..2 * half],
                        &tape.cos[ti * half..(ti + 1) * half],
                        &tape.neg_sin[ti * half..(ti + 1) * half],
                    );
                    match &at.k_norm {
                        Some(w) => rms_back(&rec.kraw[i * hd..(i + 1) * hd], w, eps, gk, out),
                        None => out.copy_from_slice(gk),
                    }
                });
            }
            debug_assert_eq!(rep * nkv, nh);
            if lowest {
                // Nothing below is adapted: only the updates' own gradients.
                back(Q, &rec.hn, &g_qraw, None);
                back(K, &rec.hn, &g_kraw, None);
                back(V, &rec.hn, &g_v, None);
                break;
            }
            let [tq, tk, tv] = wt.qkv.as_ref().expect("transposes above the lowest layer");
            linear(tq, h, qw, &g_qraw, t, &mut g_hn);
            linear(tk, h, kvw, &g_kraw, t, &mut tmp);
            add_inplace(&mut g_hn, &tmp);
            linear(tv, h, kvw, &g_v, t, &mut tmp);
            add_inplace(&mut g_hn, &tmp);
            back(Q, &rec.hn, &g_qraw, Some(&mut g_hn));
            back(K, &rec.hn, &g_kraw, Some(&mut g_hn));
            back(V, &rec.hn, &g_v, Some(&mut g_hn));
            rms_back_rows(&rec.x_in, h, &layer.ln_in, eps, &g_hn, &mut tmp);
            add_inplace(&mut gx, &tmp);
        }
    }
}

/// The backward of causal attention over one sequence: given `g_att` (`[t, nh * hd]`), the
/// gradients of the queries (`[t, nh * hd]`), keys and values (`[t, n_kv * hd]`) as `cache`
/// holds them. The scores are computed again (as the forward's kernel does) rather than kept.
fn attention_back(q: &[f32], cache: &LayerCache, g_att: &[f32], nh: usize, t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hd, nkv) = (cache.hd, cache.n_kv);
    let rep = nh / nkv;
    let stride = nkv * hd;
    // Tasks of one key / value head each write that head's keys and values: each takes every
    // `split`-th query and sums into its own buffers.
    let split = (pool::threads() / nkv).clamp(1, t);
    let mut g_q = vec![0f32; t * nh * hd];
    let qp = SendPtr(g_q.as_mut_ptr());
    let parts: Vec<Mutex<(Vec<f32>, Vec<f32>)>> = (0..nkv * split).map(|_| Mutex::new((Vec::new(), Vec::new()))).collect();
    pool::run(nkv * split, &|task| {
        let (kvh, c) = (task / split, task % split);
        let mut gk = vec![0f32; t * hd];
        let mut gv = vec![0f32; t * hd];
        let mut p = vec![0f32; t];
        let mut gp = vec![0f32; t];
        for ti in (c..t).step_by(split) {
            let n = ti + 1;
            for hh in kvh * rep..(kvh + 1) * rep {
                let base = (ti * nh + hh) * hd;
                let (qv, ga) = (&q[base..base + hd], &g_att[base..base + hd]);
                crate::kernels::dots(qv, cache.k_from(0, kvh), stride, &mut p[..n]);
                crate::kernels::softmax_inplace(&mut p[..n]);
                crate::kernels::dots(ga, cache.v_from(0, kvh), stride, &mut gp[..n]);
                let mean: f32 = p[..n].iter().zip(&gp[..n]).map(|(a, b)| a * b).sum();
                // SAFETY: this task alone writes the queries of its heads.
                let gq = unsafe { std::slice::from_raw_parts_mut(qp.get().add(base), hd) };
                for j in 0..n {
                    let gs = p[j] * (gp[j] - mean);
                    axpy(gq, gs, cache.k_at(j, kvh));
                    axpy(&mut gk[j * hd..(j + 1) * hd], gs, qv);
                    axpy(&mut gv[j * hd..(j + 1) * hd], p[j], ga);
                }
            }
        }
        *parts[task].lock().unwrap() = (gk, gv);
    });
    let mut g_k = vec![0f32; t * stride];
    let mut g_v = vec![0f32; t * stride];
    for (task, part) in parts.into_iter().enumerate() {
        let kvh = task / split;
        let (gk, gv) = part.into_inner().unwrap();
        for j in 0..t {
            let o = j * stride + kvh * hd;
            add_inplace(&mut g_k[o..o + hd], &gk[j * hd..(j + 1) * hd]);
            add_inplace(&mut g_v[o..o + hd], &gv[j * hd..(j + 1) * hd]);
        }
    }
    (g_q, g_k, g_v)
}

/// AdamW over an adapter's matrices (PyTorch's update: decoupled weight decay, bias-corrected
/// moments).
pub struct AdamW {
    pub lr: f32,
    pub weight_decay: f32,
    step: u32,
    m: Grads,
    v: Grads,
}

impl AdamW {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    const EPS: f32 = 1e-8;

    pub fn new(ad: &Adapter, lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            step: 0,
            m: ad.zero_grads(),
            v: ad.zero_grads(),
        }
    }

    /// One update of `ad` with gradients `g`.
    pub fn step(&mut self, ad: &mut Adapter, g: &Grads) {
        self.step += 1;
        let c1 = 1.0 - Self::B1.powi(self.step as i32);
        let c2 = (1.0 - Self::B2.powi(self.step as i32)).sqrt();
        let (lr, wd) = (self.lr, self.weight_decay);
        let upd = |p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]| {
            for i in 0..p.len() {
                m[i] = Self::B1 * m[i] + (1.0 - Self::B1) * g[i];
                v[i] = Self::B2 * v[i] + (1.0 - Self::B2) * g[i] * g[i];
                p[i] -= lr * wd * p[i];
                p[i] -= lr / c1 * m[i] / (v[i].sqrt() / c2 + Self::EPS);
            }
        };
        for (i, md) in ad.mods.iter_mut().enumerate() {
            upd(&mut md.a, &g[i].0, &mut self.m[i].0, &mut self.v[i].0);
            upd(&mut md.bt, &g[i].1, &mut self.m[i].1, &mut self.v[i].1);
        }
    }
}

/// The norm of all gradients; scales them down to `max` when it is larger (`max` 0: no clipping).
pub fn clip_grads(g: &mut Grads, max: f32) -> f32 {
    let sq: f64 = g.iter().flat_map(|(a, b)| a.iter().chain(b)).map(|&v| v as f64 * v as f64).sum();
    let norm = sq.sqrt() as f32;
    if max > 0.0 && norm > max {
        let k = max / (norm + 1e-6);
        g.iter_mut().flat_map(|(a, b)| a.iter_mut().chain(b)).for_each(|v| *v *= k);
    }
    norm
}

/// Training options.
#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
    /// The adapted layers (None: all).
    pub layers: Option<Vec<usize>>,
    pub lr: f32,
    /// Optimizer steps (0: `epochs` passes over the training episodes).
    pub steps: usize,
    pub epochs: usize,
    /// Episodes per optimizer step.
    pub batch: usize,
    pub weight_decay: f32,
    /// Gradient-norm clipping (0: none).
    pub clip: f32,
    pub neg_clip: f32,
    /// The fraction of episodes held out for the gate.
    pub holdout: f32,
    /// Episodes of more tokens (prompt and completion) are left out.
    pub max_tokens: usize,
    pub seed: u64,
    /// Write the adapter whatever the gate says.
    pub force: bool,
    /// Start from a new adapter even when one exists.
    pub fresh: bool,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            targets: vec!["q_proj".into(), "v_proj".into()],
            layers: None,
            lr: 2e-4,
            steps: 0,
            epochs: 3,
            batch: 4,
            weight_decay: 0.0,
            clip: 1.0,
            neg_clip: 4.0,
            holdout: 0.2,
            max_tokens: 2048,
            seed: 0,
            force: false,
            fresh: false,
        }
    }
}

/// What a training run did.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Report {
    pub train_episodes: usize,
    pub holdout_episodes: usize,
    /// Episodes left out for their length.
    pub too_long: usize,
    pub steps: usize,
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
    pub trainable_params: usize,
    /// Whether training started from the existing adapter's matrices.
    pub resumed: bool,
    /// Why not, when there was an existing adapter.
    pub existing_note: Option<String>,
    /// The weighted loss of the training episodes before and after.
    pub train_loss_before: f64,
    pub train_loss_after: f64,
    /// The weighted loss of the held-out episodes: base model, existing adapter, new adapter.
    pub holdout_loss_base: Option<f64>,
    pub holdout_loss_existing: Option<f64>,
    pub holdout_loss_new: Option<f64>,
    /// What the gate compared the new adapter with: `base` or `existing`.
    pub gate_against: &'static str,
    /// The gate's verdict: the new adapter's held-out loss is lower.
    pub better: bool,
    pub written: bool,
    pub seconds_per_step: f64,
    pub transposed_bytes: usize,
}

/// FNV-1a of an episode id: which side of the split it is on, the same in every run.
fn id_hash(id: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in id.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    // FNV's high bits mix poorly for short keys.
    h ^= h >> 32;
    h.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Splits `eps` into training and held-out episodes: an episode is held out when the hash of its
/// id falls in the first `frac` of the range, so it stays held out in later runs over a longer
/// log (an adapter trained earlier never saw what a later gate measures it on); with episodes
/// but none held out that way, the one of lowest hash is.
pub fn split(eps: Vec<Episode>, frac: f32) -> (Vec<Episode>, Vec<Episode>) {
    if frac <= 0.0 || eps.len() < 2 {
        return (eps, Vec::new());
    }
    let limit = (frac.min(1.0) as f64 * u64::MAX as f64) as u64;
    let lowest = eps.iter().map(|e| id_hash(&e.id)).min().unwrap();
    let any = eps.iter().any(|e| id_hash(&e.id) < limit);
    let (hold, mut train): (Vec<_>, Vec<_>) = eps.into_iter().partition(|e| {
        let h = id_hash(&e.id);
        if any { h < limit } else { h == lowest }
    });
    let mut hold = hold;
    if train.is_empty() {
        // Everything fell on the held-out side: train on all but one.
        train = hold.split_off(1);
    }
    (train, hold)
}

/// The weighted loss of `eps` under `ad`: `sum |w| loss / sum |w|`.
pub fn eval(tr: &dyn Passes, ad: Option<&Adapter>, eps: &[Episode], obj: Objective) -> Result<f64> {
    let (mut num, mut den) = (0f64, 0f64);
    for e in eps {
        let w = e.weight.abs() as f64;
        num += w * tr.episode(ad, e, obj, None)?;
        den += w;
    }
    Ok(num / den)
}

/// Where a run reads its episodes and keeps its adapter.
#[derive(Clone, Debug)]
pub struct Paths {
    pub episodes: PathBuf,
    pub outcomes: PathBuf,
    /// The adapter folder, created or updated.
    pub adapter: PathBuf,
}

impl Paths {
    /// A profile directory as the server keeps it: `episodes.jsonl`, `outcomes.jsonl` and
    /// `adapter/`.
    pub fn profile(dir: &Path) -> Self {
        Self {
            episodes: dir.join("episodes.jsonl"),
            outcomes: dir.join("outcomes.jsonl"),
            adapter: dir.join("adapter"),
        }
    }

    /// `adapter` with `suffix` appended to its name (`adapter.prev`).
    fn sibling(&self, suffix: &str) -> PathBuf {
        let mut name = self.adapter.file_name().unwrap_or_default().to_os_string();
        name.push(suffix);
        self.adapter.with_file_name(name)
    }

    /// The previous adapter, kept when a new one replaces it.
    pub fn previous(&self) -> PathBuf {
        self.sibling(".prev")
    }
}

/// Trains an adapter for `model` (loaded from `base`) on the episodes of `paths` and, when the
/// gate passes (or `opts.force`), writes it to `paths.adapter`.
///
/// With an adapter already there, training starts from its matrices when it has the rank, alpha,
/// targets and layers asked for (else from a new one, said in the report), and the gate compares
/// the new adapter with it on the held-out episodes instead of with the base model; an existing
/// adapter the trainer cannot read is only replaced with `opts.force`. A replaced adapter is
/// kept as `adapter.prev`. The new folder is written beside the old one and renamed into place.
pub fn run(
    model: &QwenModel,
    base: &str,
    paths: &Paths,
    kind_weights: &[(String, f32)],
    opts: &TrainOptions,
    log: &mut dyn FnMut(&str),
) -> Result<Report> {
    run_with(
        model,
        &|first| Ok(Box::new(Trainer::new(model, first)?) as Box<dyn Passes + '_>),
        base,
        paths,
        kind_weights,
        opts,
        log,
    )
}

/// [`run`] for any model class: `model` says what an adapter of it looks like, `passes` makes its
/// forward and backward passes for adapters whose lowest layer is the one given.
pub fn run_with<'p, F: Family + ?Sized>(
    model: &F,
    passes: &dyn Fn(usize) -> Result<Box<dyn Passes + 'p>>,
    base: &str,
    paths: &Paths,
    kind_weights: &[(String, f32)],
    opts: &TrainOptions,
    log: &mut dyn FnMut(&str),
) -> Result<Report> {
    let (eps, stats) = load_episodes(&paths.episodes, &paths.outcomes, kind_weights)?;
    let kinds: Vec<String> = stats
        .kinds
        .iter()
        .map(|(k, n)| format!("{}: {n}", if k.is_empty() { "(none)" } else { k }))
        .collect();
    log(&format!(
        "{} episodes logged, {} without an outcome, {} with weight 0 or no completion; outcome kinds: {}",
        stats.episodes,
        stats.without_outcome,
        stats.empty,
        if kinds.is_empty() { "none".to_string() } else { kinds.join(", ") }
    ));
    let n_all = eps.len();
    let eps: Vec<Episode> = eps
        .into_iter()
        .filter(|e| e.prompt.len() + e.completion.len() <= opts.max_tokens)
        .collect();
    let too_long = n_all - eps.len();
    if too_long > 0 {
        log(&format!("{too_long} episodes of more than {} tokens left out", opts.max_tokens));
    }
    if eps.is_empty() {
        bail!("no episodes with an outcome to train on");
    }
    let (train, hold) = split(eps, opts.holdout);
    log(&format!("{} training episodes, {} held out", train.len(), hold.len()));

    let mut ad = Adapter::new(model, opts.rank, opts.alpha, &opts.targets, opts.layers.as_deref(), opts.seed)?;
    // The existing adapter: the starting point when it has the same shape, and what the gate
    // compares with.
    let mut existing_note = None;
    let mut unreadable = false;
    let existing = if crate::lora::is_adapter_dir(&paths.adapter) {
        match Adapter::load(model, &paths.adapter) {
            Ok(e) => Some(e),
            Err(e) => {
                unreadable = true;
                existing_note = Some(format!(
                    "the existing adapter cannot be read by the trainer ({e:#}): not compared with"
                ));
                None
            }
        }
    } else {
        None
    };
    let mut resumed = false;
    if let Some(e) = &existing {
        if opts.fresh {
            existing_note = Some("--fresh: a new adapter, compared with the existing one".into());
        } else if ad.same_shape(e) {
            ad = e.clone();
            resumed = true;
        } else {
            existing_note = Some(format!(
                "the existing adapter (rank {}, alpha {}, targets {}, {} layers) differs from the one asked for: starting from a new one",
                e.rank,
                e.alpha,
                e.targets().join(","),
                e.layers.len()
            ));
        }
    }
    if let Some(n) = &existing_note {
        log(n);
    }
    let first = ad.first_layer();
    let t0 = crate::clock::Instant::now();
    let tr = passes(first)?;
    let tr: &dyn Passes = &*tr;
    log(&format!(
        "{} in {:.1}s; {} trainable parameters (rank {}, alpha {}, {})",
        tr.describe(),
        t0.elapsed().as_secs_f64(),
        ad.params(),
        ad.rank,
        ad.alpha,
        ad.targets().join(",")
    ));
    let obj = Objective { neg_clip: opts.neg_clip };
    let holdout_loss_base = if hold.is_empty() { None } else { Some(eval(tr, None, &hold, obj)?) };
    let holdout_loss_existing = match &existing {
        Some(e) if !hold.is_empty() => Some(eval(tr, Some(e), &hold, obj)?),
        _ => None,
    };
    let train_loss_before = eval(tr, Some(&ad), &train, obj)?;
    log(&format!("training loss before: {train_loss_before:.5}"));

    let batch = opts.batch.max(1);
    let steps = if opts.steps > 0 {
        opts.steps
    } else {
        opts.epochs.max(1) * train.len().div_ceil(batch)
    };
    let mut opt = AdamW::new(&ad, opts.lr, opts.weight_decay);
    let mut rng = Rng::new(opts.seed ^ 0x5eed);
    let mut order: Vec<usize> = Vec::new();
    let t0 = crate::clock::Instant::now();
    for step in 0..steps {
        let mut picked = Vec::with_capacity(batch);
        while picked.len() < batch.min(train.len()) {
            if order.is_empty() {
                order = (0..train.len()).collect();
                for i in (1..order.len()).rev() {
                    order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
                }
            }
            picked.push(order.pop().unwrap());
        }
        let den: f32 = picked.iter().map(|&i| train[i].weight.abs()).sum();
        let mut grads = ad.zero_grads();
        let mut loss = 0f64;
        for &i in &picked {
            let w = train[i].weight.abs() / den;
            loss += w as f64 * tr.episode(Some(&ad), &train[i], obj, Some((&mut grads, w)))?;
        }
        let norm = clip_grads(&mut grads, opts.clip);
        opt.step(&mut ad, &grads);
        log(&format!(
            "step {}/{steps}: loss {loss:.5}, gradient norm {norm:.4}, {:.2}s",
            step + 1,
            t0.elapsed().as_secs_f64()
        ));
    }
    let seconds_per_step = t0.elapsed().as_secs_f64() / steps.max(1) as f64;
    let train_loss_after = eval(tr, Some(&ad), &train, obj)?;
    let holdout_loss_new = if hold.is_empty() {
        None
    } else {
        Some(eval(tr, Some(&ad), &hold, obj)?)
    };
    let (gate_against, reference) = match holdout_loss_existing {
        Some(l) => ("existing", Some(l)),
        None => ("base", holdout_loss_base),
    };
    let better = matches!((holdout_loss_new, reference), (Some(n), Some(r)) if n < r);
    log(&format!("training loss after: {train_loss_after:.5}"));
    match (holdout_loss_new, holdout_loss_base) {
        (Some(n), Some(b)) => log(&format!(
            "held-out loss: base model {b:.5}{}, new adapter {n:.5}: {} than the {gate_against}",
            holdout_loss_existing.map_or(String::new(), |e| format!(", existing adapter {e:.5}")),
            if better { "better" } else { "not better" },
        )),
        _ => log("no held-out episodes: the gate cannot tell whether the adapter helps"),
    }
    let write = opts.force || (better && !unreadable);
    if write {
        let tmp = paths.sibling(&format!(".tmp-{}", std::process::id()));
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp).with_context(|| tmp.display().to_string())?;
        }
        ad.save(model, &tmp, base)?;
        if paths.adapter.exists() {
            let prev = paths.previous();
            if prev.exists() {
                std::fs::remove_dir_all(&prev).with_context(|| prev.display().to_string())?;
            }
            std::fs::rename(&paths.adapter, &prev).with_context(|| format!("keep {} as {}", paths.adapter.display(), prev.display()))?;
        }
        std::fs::rename(&tmp, &paths.adapter).with_context(|| paths.adapter.display().to_string())?;
        log(&format!("adapter written to {}", paths.adapter.display()));
    } else if unreadable && better {
        log("adapter not written: it would replace an adapter it was not compared with (--force writes it)");
    } else {
        log("adapter not written (--force writes it anyway)");
    }
    Ok(Report {
        train_episodes: train.len(),
        holdout_episodes: hold.len(),
        too_long,
        steps,
        rank: ad.rank,
        alpha: ad.alpha,
        targets: ad.targets(),
        trainable_params: ad.params(),
        resumed,
        existing_note,
        train_loss_before,
        train_loss_after,
        holdout_loss_base,
        holdout_loss_existing,
        holdout_loss_new,
        gate_against,
        better,
        written: write,
        seconds_per_step,
        transposed_bytes: tr.transposed_bytes(),
    })
}

#[cfg(test)]
#[path = "lora_train_tests.rs"]
mod tests;
