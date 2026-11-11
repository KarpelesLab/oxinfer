//! LoRA files merged into the UNet and text-encoder weights at load. Every targeted weight becomes
//! `W + scale * alpha / rank * up @ down` in f32 before it is converted to bf16, so a merged model
//! runs exactly like the base one, on the CPU and on the GPU backends.
//!
//! Two key layouts are read:
//! - kohya / A1111: `lora_unet_<module path with _>.lora_down.weight`, `.lora_up.weight` and
//!   `.alpha`, and `lora_te_text_model_encoder_...` for the text encoder (`lora_te1_` and
//!   `lora_te2_` for SDXL's two). The underscore path is matched against the underscore form of
//!   every weight of the base model, and for the UNet also against its names in the original
//!   (SGM) layout, which kohya's SDXL files use (`lora_unet_input_blocks_4_1_...`).
//! - diffusers / PEFT: `unet.<module>.lora_A.weight` / `.lora_B.weight` (also `.lora.down` /
//!   `.lora.up`, and the older `processor.to_q_lora.down` attention processors), with
//!   `text_encoder.` (and SDXL's `text_encoder_2.`) for the text side.
//!
//! On a convolution (LoCon) the down projection is itself a convolution (`[rank, in, kh, kw]`)
//! and the up projection a 1x1 one (`[out, rank, 1, 1]`), so both flatten to the same product
//! `[out, rank] @ [rank, in * kh * kw]`.

use super::ops::Tensors;
use crate::kernels::for_chunks;
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// One LoRA file and the strength it is merged with.
#[derive(Clone, Debug)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub scale: f32,
}

impl LoraSpec {
    /// `FILE[:SCALE]`; the scale defaults to 1.
    pub fn parse(s: &str) -> Result<Self> {
        if let Some((p, v)) = s.rsplit_once(':')
            && !p.is_empty()
            && let Ok(scale) = v.parse::<f32>()
        {
            return Ok(Self {
                path: PathBuf::from(p),
                scale,
            });
        }
        if s.is_empty() {
            bail!("--lora expects FILE[:SCALE]");
        }
        Ok(Self {
            path: PathBuf::from(s),
            scale: 1.0,
        })
    }
}

/// The model a LoRA module patches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Part {
    Unet,
    Text,
    /// SDXL's second text encoder.
    Text2,
}

impl Part {
    /// The component's name in a diffusers file (key prefix, metadata section).
    fn section(self) -> &'static str {
        match self {
            Part::Unet => "unet",
            Part::Text => "text_encoder",
            Part::Text2 => "text_encoder_2",
        }
    }
}

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
}

/// Key suffixes, longest first where one ends another.
const SUFFIXES: [(&str, Role); 13] = [
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".lora_A.default.weight", Role::Down),
    (".lora_B.default.weight", Role::Up),
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".lora_linear_layer.down.weight", Role::Down),
    (".lora_linear_layer.up.weight", Role::Up),
    (".lora.down.weight", Role::Down),
    (".lora.up.weight", Role::Up),
    // Attention processors of older diffusers files: `attn1.processor.to_q_lora.down.weight`.
    ("_lora.down.weight", Role::Down),
    ("_lora.up.weight", Role::Up),
    (".alpha", Role::Alpha),
];

/// One low-rank pair of a LoRA file.
#[derive(Default)]
struct Module {
    file: usize,
    /// The original key prefix, for messages.
    key: String,
    down: Option<String>,
    up: Option<String>,
    alpha: Option<String>,
    /// The alpha from a PEFT `lora_adapter_metadata` header, for files without `.alpha` keys.
    meta_alpha: Option<f32>,
    /// 0 not looked up yet, 1 matches nothing, 2 tied to a base weight.
    state: AtomicU8,
    applied: AtomicBool,
}

struct File {
    t: Tensors,
    scale: f32,
    name: String,
}

/// The LoRA files of a pipeline, parsed into modules that are not yet tied to a weight.
pub struct Loras {
    files: Vec<File>,
    modules: Vec<Module>,
    /// (part, module path, kohya underscore form) per module.
    targets: Vec<(Part, String, bool)>,
    /// Keys that are not low-rank pairs (LoHa, LoKr, DoRA scales, full diffs, ...), per file.
    unsupported: Vec<Vec<String>>,
}

/// The base model's module path for a key prefix: the part it patches, the path, and whether it
/// is kohya's underscore form.
fn target(prefix: &str) -> (Part, String, bool) {
    for (p, part) in [
        ("lora_unet_", Part::Unet),
        ("lora_te_", Part::Text),
        ("lora_te1_", Part::Text),
        ("lora_te2_", Part::Text2),
    ] {
        if let Some(rest) = prefix.strip_prefix(p) {
            return (part, rest.to_string(), true);
        }
    }
    let (part, path) = if let Some(r) = prefix.strip_prefix("unet.") {
        (Part::Unet, r)
    } else if let Some(r) = prefix.strip_prefix("text_encoder.") {
        (Part::Text, r)
    } else if let Some(r) = prefix.strip_prefix("text_encoder_2.") {
        (Part::Text2, r)
    } else {
        (Part::Unet, prefix)
    };
    // `attn1.processor.to_out` -> `attn1.to_out.0`, `attn1.processor.to_q` -> `attn1.to_q`.
    let path = match path.split_once(".processor.") {
        Some((a, "to_out")) => format!("{a}.to_out.0"),
        Some((a, b)) => format!("{a}.{b}"),
        None => path.to_string(),
    };
    (part, path, false)
}

/// The alpha PEFT recorded for module `path` in a diffusers `lora_adapter_metadata` header: the
/// matching `alpha_pattern` entry, else `lora_alpha`.
fn peft_alpha(meta: &serde_json::Value, section: &str, path: &str) -> Option<f32> {
    if let Some(pats) = meta.get(format!("{section}.alpha_pattern")).and_then(|v| v.as_object()) {
        for (pat, v) in pats {
            // PEFT's rule: `(.*\.)?(pattern)$` matched from the start of the module name.
            if let Ok(re) = regex::Regex::new(&format!(r"^(.*\.)?({pat})$"))
                && re.is_match(path)
            {
                return v.as_f64().map(|a| a as f32);
            }
        }
    }
    meta.get(format!("{section}.lora_alpha")).and_then(|v| v.as_f64()).map(|a| a as f32)
}

impl Loras {
    pub fn open(specs: &[LoraSpec]) -> Result<Self> {
        let mut files = Vec::new();
        let mut modules: Vec<Module> = Vec::new();
        let mut targets = Vec::new();
        let mut unsupported = Vec::new();
        for (fi, s) in specs.iter().enumerate() {
            let t = Tensors::open(&s.path).with_context(|| format!("LoRA {}", s.path.display()))?;
            let meta: Option<serde_json::Value> = t.metadata("lora_adapter_metadata").and_then(|m| serde_json::from_str(m).ok());
            let mut by_prefix: BTreeMap<String, Module> = BTreeMap::new();
            let mut odd = Vec::new();
            for k in t.names() {
                let Some((prefix, role)) = SUFFIXES.iter().find_map(|(s, r)| k.strip_suffix(s).map(|p| (p, *r))) else {
                    odd.push(k.clone());
                    continue;
                };
                let m = by_prefix.entry(prefix.to_string()).or_insert_with(|| Module {
                    file: fi,
                    key: prefix.to_string(),
                    ..Default::default()
                });
                let slot = match role {
                    Role::Down => &mut m.down,
                    Role::Up => &mut m.up,
                    Role::Alpha => &mut m.alpha,
                };
                *slot = Some(k.clone());
            }
            for (prefix, mut m) in by_prefix {
                if m.down.is_none() || m.up.is_none() {
                    // An alpha without its pair belongs to an unsupported module kind.
                    odd.extend([m.down, m.up, m.alpha].into_iter().flatten());
                    continue;
                }
                let (part, path, kohya) = target(&prefix);
                if let Some(meta) = &meta
                    && m.alpha.is_none()
                {
                    m.meta_alpha = peft_alpha(meta, part.section(), &path);
                }
                targets.push((part, path, kohya));
                modules.push(m);
            }
            odd.sort();
            unsupported.push(odd);
            files.push(File {
                t,
                scale: s.scale,
                name: s.path.display().to_string(),
            });
        }
        Ok(Self {
            files,
            modules,
            targets,
            unsupported,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Ties the modules of `part` to the weights of `base`; returns the patches to install with
    /// `Tensors::set_lora` (None when nothing targets this part).
    pub fn resolve(self: &Arc<Self>, part: Part, base: &Tensors) -> Result<Option<Patches>> {
        let me = &**self;
        let kohya: HashMap<String, String> = if me.targets.iter().any(|(p, _, k)| *p == part && *k) {
            // The UNet also under its original (SGM) names: kohya's SDXL files use those.
            let sgm = if part == Part::Unet {
                super::xl::sgm_names(base)
            } else {
                Vec::new()
            };
            sgm.into_iter()
                .chain(
                    base.names()
                        .filter_map(|n| n.strip_suffix(".weight"))
                        .map(|m| (m.replace('.', "_"), format!("{m}.weight"))),
                )
                .collect()
        } else {
            HashMap::new()
        };
        let mut by_weight: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, (p, path, k)) in me.targets.iter().enumerate() {
            if *p != part {
                continue;
            }
            let m = &me.modules[i];
            let w = if *k {
                kohya.get(path).cloned()
            } else {
                // Text-encoder keys saved from newer transformers lack the `text_model.` level.
                [format!("{path}.weight"), format!("text_model.{path}.weight")]
                    .into_iter()
                    .find(|w| base.has(w) && (part != Part::Unet || !w.starts_with("text_model.")))
            };
            let Some(w) = w else {
                m.state.store(1, Ordering::Relaxed);
                continue;
            };
            // Shapes must agree with the base weight: a LoRA for another model fails here.
            let f = &me.files[m.file];
            let ws = base.shape(&w)?;
            let (ds, us) = (f.t.shape(m.down.as_ref().unwrap())?, f.t.shape(m.up.as_ref().unwrap())?);
            let (out, numel) = (ws[0], ws.iter().product::<usize>());
            let rank = ds.first().copied().unwrap_or(0);
            let ok = rank > 0
                && us.first() == Some(&out)
                && us.iter().product::<usize>() == out * rank
                && ds.iter().product::<usize>() * out == numel * rank;
            if !ok {
                bail!(
                    "LoRA {}: {} (down {ds:?}, up {us:?}) does not fit {w} {ws:?}; is it made for another base model?",
                    f.name,
                    m.key
                );
            }
            m.state.store(2, Ordering::Relaxed);
            by_weight.entry(w).or_default().push(i);
        }
        if by_weight.is_empty() {
            return Ok(None);
        }
        Ok(Some(Patches {
            loras: self.clone(),
            by_weight,
        }))
    }

    /// One line per file (`[lora FILE x1: 278 modules merged ...]`); errors when a file matched
    /// nothing at all.
    pub fn report(&self) -> Result<Vec<String>> {
        let mut lines = Vec::new();
        for (fi, f) in self.files.iter().enumerate() {
            let (matched, unmatched): (Vec<usize>, Vec<usize>) = (0..self.modules.len())
                .filter(|&i| self.modules[i].file == fi)
                .partition(|&i| self.modules[i].state.load(Ordering::Relaxed) == 2);
            // Matched but never read by the loaders (a weight the pipeline does not use) counts as
            // skipped.
            let (merged, unread): (Vec<usize>, Vec<usize>) =
                matched.into_iter().partition(|&i| self.modules[i].applied.load(Ordering::Relaxed));
            let count = |part| merged.iter().filter(|&&i| self.targets[i].0 == part).count();
            let text2 = match count(Part::Text2) {
                0 => String::new(),
                n => format!(", text encoder 2 {n}"),
            };
            let mut skipped: Vec<&str> = unmatched.iter().chain(&unread).map(|&i| self.modules[i].key.as_str()).collect();
            skipped.extend(self.unsupported[fi].iter().map(String::as_str));
            if merged.is_empty() {
                let mut names: Vec<&String> = f.t.names().collect();
                names.sort();
                bail!(
                    "LoRA {}: none of its {} keys match this model (e.g. {}); is it for another base model?",
                    f.name,
                    names.len(),
                    names.iter().take(3).map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                );
            }
            let mut s = format!(
                "[lora {} x{}: {} modules merged (unet {}, text encoder {}{text2})",
                f.name,
                f.scale,
                merged.len(),
                count(Part::Unet),
                count(Part::Text)
            );
            if !skipped.is_empty() {
                s.push_str(&format!(
                    "; {} modules or keys skipped (no such weight in this model, or an unsupported kind), e.g. {}",
                    skipped.len(),
                    skipped.iter().take(3).copied().collect::<Vec<_>>().join(", ")
                ));
            }
            s.push(']');
            lines.push(s);
        }
        Ok(lines)
    }
}

/// The LoRA updates of one component, by base weight name.
pub struct Patches {
    loras: Arc<Loras>,
    by_weight: HashMap<String, Vec<usize>>,
}

impl Patches {
    pub fn targets(&self, name: &str) -> bool {
        self.by_weight.contains_key(name)
    }

    /// Adds every update of `name` to `w` (PyTorch layout, `[out, ...]`).
    pub fn apply(&self, name: &str, out: usize, w: &mut [f32]) -> Result<()> {
        let Some(list) = self.by_weight.get(name) else {
            return Ok(());
        };
        let rest = w.len() / out;
        for &i in list {
            let m = &self.loras.modules[i];
            let f = &self.loras.files[m.file];
            let down = f.t.f32(m.down.as_ref().unwrap())?;
            let up = f.t.f32(m.up.as_ref().unwrap())?;
            let rank = down.len() / rest;
            let alpha = match &m.alpha {
                Some(a) => f.t.f32(a)?.first().copied().unwrap_or(rank as f32),
                None => m.meta_alpha.unwrap_or(rank as f32),
            };
            let s = f.scale * alpha / rank as f32;
            // Row o of the update is sum_r up[o][r] * down[r]; summed apart, then scaled and added.
            for_chunks(w, rest, true, |o, row| {
                let mut d = vec![0f32; rest];
                for (r, &u) in up[o * rank..(o + 1) * rank].iter().enumerate() {
                    for (dv, &x) in d.iter_mut().zip(&down[r * rest..(r + 1) * rest]) {
                        *dv += u * x;
                    }
                }
                for (wv, dv) in row.iter_mut().zip(d) {
                    *wv += s * dv;
                }
            });
            m.applied.store(true, Ordering::Relaxed);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_and_targets() {
        let s = LoraSpec::parse("a/b.safetensors:0.8").unwrap();
        assert_eq!((s.path, s.scale), (PathBuf::from("a/b.safetensors"), 0.8));
        assert_eq!(LoraSpec::parse("org/repo/x.safetensors").unwrap().scale, 1.0);
        assert_eq!(
            LoraSpec::parse("C:\\x.safetensors").unwrap().path,
            PathBuf::from("C:\\x.safetensors")
        );
        assert_eq!(
            target("lora_te_text_model_encoder_layers_0_mlp_fc1"),
            (Part::Text, "text_model_encoder_layers_0_mlp_fc1".into(), true)
        );
        assert_eq!(
            target("lora_te2_text_model_encoder_layers_31_self_attn_q_proj"),
            (Part::Text2, "text_model_encoder_layers_31_self_attn_q_proj".into(), true)
        );
        assert_eq!(
            target("text_encoder_2.text_model.encoder.layers.3.mlp.fc2"),
            (Part::Text2, "text_model.encoder.layers.3.mlp.fc2".into(), false)
        );
        assert_eq!(
            target("unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_out"),
            (
                Part::Unet,
                "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0".into(),
                false
            )
        );
        assert_eq!(
            target("text_encoder.text_model.encoder.layers.3.self_attn.q_proj"),
            (Part::Text, "text_model.encoder.layers.3.self_attn.q_proj".into(), false)
        );
    }
}
