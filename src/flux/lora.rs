//! LoRA files on the FLUX transformer (`--lora FILE[:SCALE]`), applied at load.
//!
//! Every targeted layer keeps its stored weights (bf16 borrowed from the mapped checkpoint, GGUF
//! blocks of any format, Q8 / Q4 made at load) and gains a low-rank side path
//! `y += scale * alpha / rank * up (down x)` in f32 ([`crate::lora::LoraDelta`], run after the
//! base matmul wherever that ran). That is the merged model `W + scale * alpha / rank * up @ down`
//! without rewriting `W`: a merge into a quantized matrix would have to dequantize and requantize
//! it (a 24 GB copy for the bf16 checkpoint, and the update of a few-step LoRA is smaller than
//! the rounding step of a Q4 block), so the update stays exact and the weights stay mapped.
//!
//! Key namings read:
//! - diffusers / PEFT: `transformer.<module>.lora_A.weight` / `.lora_B.weight` (also
//!   `.lora_B.bias`), alpha from the `lora_adapter_metadata` header or `.alpha` keys;
//! - kohya (sd-scripts, ai-toolkit): `lora_unet_<BFL module with _>.lora_down.weight` /
//!   `.lora_up.weight` / `.alpha`, and `lora_transformer_<diffusers module with _>`;
//! - Black Forest Labs / ComfyUI: `[diffusion_model.]double_blocks.0.img_attn.qkv.lora_A.weight`
//!   (BFL's own Canny / Depth control LoRAs, which also widen `img_in` to 128 channels and
//!   replace the QK norm scales);
//! - XLabs: `double_blocks.0.processor.qkv_lora1.down.weight` / `.up.weight` (`lora1` the image
//!   stream, `lora2` the text stream, `proj_lora*` the attention output).
//!
//! BFL modules are fused where diffusers' are not (`qkv`, the single blocks' `linear1`, the
//! final modulation with its halves swapped); a module on a fused matrix is split by rows over
//! the layers this crate keeps. A key that is not understood, or a module that matches no layer,
//! is an error; text-encoder modules (`lora_te*`, `text_encoder.*`) are not applied and counted
//! in the report line.

use super::transformer::{FluxTransformer, layer_limit};
use crate::lora::LoraDelta;
use crate::safetensors::SafeTensors;
pub use crate::sd::lora::LoraSpec;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;

/// A part of a module: the layer (this crate's name for it) and its rows of the module's
/// output (None: all of them).
type Part = (String, Option<(usize, usize)>);

/// The layers a LoRA module name stands for (`d` is the hidden size). Diffusers names of single
/// layers come back as they are; whether such a layer exists is the caller's check.
fn parts(m: &str, d: usize) -> Vec<Part> {
    let one = |s: String| vec![(s, None)];
    let rows = |names: &[String]| -> Vec<Part> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), Some((i * d, (i + 1) * d))))
            .collect()
    };
    let emb = |n: &str, l: &str| format!("time_text_embed.{n}.{}", if l == "in_layer" { "linear_1" } else { "linear_2" });
    match m {
        "img_in" => return one("x_embedder".into()),
        "txt_in" => return one("context_embedder".into()),
        "final_layer.linear" => return one("proj_out".into()),
        // BFL stores (shift, scale), diffusers (scale, shift).
        "final_layer.adaLN_modulation.1" => return rows(&["norm_out.linear.shift".into(), "norm_out.linear.scale".into()]),
        "norm_out.linear" => return rows(&["norm_out.linear.scale".into(), "norm_out.linear.shift".into()]),
        _ => {}
    }
    for (b, n) in [
        ("time_in", "timestep_embedder"),
        ("guidance_in", "guidance_embedder"),
        ("vector_in", "text_embedder"),
    ] {
        if let Some(l) = m.strip_prefix(b).and_then(|r| r.strip_prefix('.'))
            && matches!(l, "in_layer" | "out_layer")
        {
            return one(emb(n, l));
        }
    }
    if let Some((i, tail)) = m.strip_prefix("double_blocks.").and_then(|r| r.split_once('.')) {
        let p = |s: &str| format!("transformer_blocks.{i}.{s}");
        let single = match tail {
            "img_mod.lin" => "norm1.linear",
            "txt_mod.lin" => "norm1_context.linear",
            "img_attn.proj" => "attn.to_out.0",
            "txt_attn.proj" => "attn.to_add_out",
            "img_mlp.0" => "ff.net.0.proj",
            "img_mlp.2" => "ff.net.2",
            "txt_mlp.0" => "ff_context.net.0.proj",
            "txt_mlp.2" => "ff_context.net.2",
            "img_attn.qkv" => return rows(&[p("attn.to_q"), p("attn.to_k"), p("attn.to_v")]),
            "txt_attn.qkv" => return rows(&[p("attn.add_q_proj"), p("attn.add_k_proj"), p("attn.add_v_proj")]),
            _ => return one(m.to_string()),
        };
        return one(p(single));
    }
    if let Some((i, tail)) = m.strip_prefix("single_blocks.").and_then(|r| r.split_once('.')) {
        let p = |s: &str| format!("single_transformer_blocks.{i}.{s}");
        let qkv = [p("attn.to_q"), p("attn.to_k"), p("attn.to_v")];
        return match tail {
            "modulation.lin" => one(p("norm.linear")),
            "linear2" => one(p("proj_out")),
            // XLabs' `qkv_lora`: the attention rows of `linear1` only.
            "linear1.qkv" => rows(&qkv),
            "linear1" => {
                let mut v = rows(&qkv);
                v.push((p("proj_mlp"), Some((3 * d, usize::MAX))));
                v
            }
            _ => one(m.to_string()),
        };
    }
    one(m.to_string())
}

/// The diffusers name of a QK norm weight given by its BFL name (None: not a BFL norm).
fn bfl_norm(m: &str) -> Option<String> {
    if let Some((i, tail)) = m.strip_prefix("double_blocks.").and_then(|r| r.split_once('.')) {
        let n = match tail {
            "img_attn.norm.query_norm" => "attn.norm_q",
            "img_attn.norm.key_norm" => "attn.norm_k",
            "txt_attn.norm.query_norm" => "attn.norm_added_q",
            "txt_attn.norm.key_norm" => "attn.norm_added_k",
            _ => return None,
        };
        return Some(format!("transformer_blocks.{i}.{n}"));
    }
    let (i, tail) = m.strip_prefix("single_blocks.")?.split_once('.')?;
    let n = match tail {
        "norm.query_norm" => "attn.norm_q",
        "norm.key_norm" => "attn.norm_k",
        _ => return None,
    };
    Some(format!("single_transformer_blocks.{i}.{n}"))
}

/// XLabs' attention-processor modules under the BFL module they patch.
fn xlabs(m: &str) -> Option<String> {
    let (block, proc) = m.split_once(".processor.")?;
    let t = if block.starts_with("double_blocks.") {
        match proc {
            "qkv_lora1" => "img_attn.qkv",
            "qkv_lora2" => "txt_attn.qkv",
            "proj_lora1" => "img_attn.proj",
            "proj_lora2" => "txt_attn.proj",
            _ => return None,
        }
    } else {
        match proc {
            "qkv_lora" => "linear1.qkv",
            "proj_lora" => "linear2",
            _ => return None,
        }
    };
    Some(format!("{block}.{t}"))
}

/// Every module name a kohya underscore key can stand for: the BFL modules of a model with
/// these block counts, and the layers' own (diffusers) names.
fn module_names(layers: &[String], doubles: usize, singles: usize) -> Vec<String> {
    let mut v: Vec<String> = layers.to_vec();
    v.extend(
        [
            "img_in",
            "txt_in",
            "final_layer.linear",
            "final_layer.adaLN_modulation.1",
            "norm_out.linear",
        ]
        .map(String::from),
    );
    for b in ["time_in", "guidance_in", "vector_in"] {
        v.extend(["in_layer", "out_layer"].map(|l| format!("{b}.{l}")));
    }
    const D: [&str; 10] = [
        "img_mod.lin",
        "txt_mod.lin",
        "img_attn.qkv",
        "txt_attn.qkv",
        "img_attn.proj",
        "txt_attn.proj",
        "img_mlp.0",
        "img_mlp.2",
        "txt_mlp.0",
        "txt_mlp.2",
    ];
    for i in 0..doubles {
        v.extend(D.iter().map(|t| format!("double_blocks.{i}.{t}")));
    }
    for i in 0..singles {
        v.extend(
            ["modulation.lin", "linear1", "linear2"]
                .iter()
                .map(|t| format!("single_blocks.{i}.{t}")),
        );
    }
    v
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    Down,
    Up,
    UpBias,
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
    (".lora_B.bias", Role::UpBias),
    (".lora_up.bias", Role::UpBias),
    (".lora.down.weight", Role::Down),
    (".lora.up.weight", Role::Up),
    (".down.weight", Role::Down),
    (".up.weight", Role::Up),
    (".alpha", Role::Alpha),
];

/// Prefixes in front of the transformer's module path.
const PREFIXES: [&str; 5] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "base_model.model.",
    "transformer.",
    "unet.",
];

fn tensor_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let info = st.info(name).with_context(|| format!("missing tensor {name}"))?;
    let b = st.bytes_cow(name)?;
    Ok(match info.dtype.as_str() {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "F64" => b.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c) as f32).collect(),
        "BF16" => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
        d => bail!("LoRA tensor {name}: unsupported dtype {d}"),
    })
}

/// The alpha PEFT recorded for module `path` in a diffusers `lora_adapter_metadata` header (the
/// matching `alpha_pattern` entry, else `lora_alpha`), and whether the adapter is rank-stabilized.
fn peft_alpha(meta: &serde_json::Value, path: &str) -> (Option<f32>, bool) {
    let get = |k: &str| {
        meta.get(format!("transformer.{k}"))
            .or_else(|| meta.get("transformer").and_then(|t| t.get(k)))
    };
    let rs = get("use_rslora").and_then(|v| v.as_bool()).unwrap_or(false);
    if let Some(pats) = get("alpha_pattern").and_then(|v| v.as_object()) {
        for (pat, v) in pats {
            // PEFT's rule: the module name equals the key or ends with `.key`.
            if path == pat || path.ends_with(&format!(".{pat}")) {
                return (v.as_f64().map(|a| a as f32), rs);
            }
        }
    }
    (get("lora_alpha").and_then(|v| v.as_f64()).map(|a| a as f32), rs)
}

/// One module of a file.
#[derive(Default)]
struct Module {
    down: Option<String>,
    up: Option<String>,
    up_bias: Option<String>,
    alpha: Option<String>,
}

/// A low-rank pair on one layer: `down` `[r, n_in]`, `up` `[n_out, r]` (scale folded in).
struct Pair {
    down: Vec<f32>,
    up: Vec<f32>,
    r: usize,
    n_in: usize,
}

/// Applies the LoRA files to `tr` (see the module docs); returns one report line per file.
pub fn apply(tr: &mut FluxTransformer, specs: &[LoraSpec]) -> Result<Vec<String>> {
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    let d = tr.cfg.hidden();
    let (doubles, singles) = (tr.cfg.num_layers, tr.cfg.num_single_layers);
    let truncated = layer_limit().is_some();
    let shapes: HashMap<String, (usize, usize)> = tr.named_layers_mut().into_iter().map(|(n, l)| (n, (l.n_out, l.n_in))).collect();
    let norm_len: HashMap<String, usize> = tr.named_norms_mut().into_iter().map(|(n, v)| (n, v.len())).collect();
    let mut kohya: Option<HashMap<String, String>> = None;
    let mut pairs: HashMap<String, Vec<Pair>> = HashMap::new();
    let mut biases: HashMap<String, Vec<f32>> = HashMap::new();
    let mut norms: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lines = Vec::new();
    for spec in specs {
        let file = spec.path.display().to_string();
        let st = SafeTensors::open(&spec.path, false).with_context(|| format!("LoRA {file}"))?;
        let meta: Option<serde_json::Value> = st.metadata("lora_adapter_metadata").and_then(|m| serde_json::from_str(m).ok());
        let mut modules: std::collections::BTreeMap<String, Module> = Default::default();
        let mut norm_keys: Vec<(String, String)> = Vec::new();
        let mut odd: Vec<&String> = Vec::new();
        for k in st.names() {
            if let Some((p, role)) = SUFFIXES.iter().find_map(|(s, r)| k.strip_suffix(s).map(|p| (p, *r))) {
                let m = modules.entry(p.to_string()).or_default();
                let slot = match role {
                    Role::Down => &mut m.down,
                    Role::Up => &mut m.up,
                    Role::UpBias => &mut m.up_bias,
                    Role::Alpha => &mut m.alpha,
                };
                *slot = Some(k.clone());
                continue;
            }
            // Full replacements of the QK norm scales (BFL control LoRAs).
            let path = PREFIXES.iter().find_map(|p| k.strip_prefix(p)).unwrap_or(k);
            let norm = match (path.strip_suffix(".scale"), path.strip_suffix(".weight")) {
                (Some(m), _) => bfl_norm(m),
                (_, Some(m)) if norm_len.contains_key(m) => Some(m.to_string()),
                _ => None,
            };
            match norm {
                Some(n) => norm_keys.push((n, k.clone())),
                None => odd.push(k),
            }
        }
        if !odd.is_empty() {
            odd.sort();
            let kind = if odd.iter().any(|k| k.contains("dora_scale")) {
                " (DoRA)"
            } else if odd.iter().any(|k| k.contains("lokr_") || k.contains("hada_")) {
                " (LoKr / LoHa)"
            } else {
                ""
            };
            bail!(
                "LoRA {file}: {} keys are not low-rank pairs{kind}, e.g. {}; only LoRA (down / up) modules are supported",
                odd.len(),
                odd[0]
            );
        }
        let (mut applied, mut text, mut skipped) = (0usize, 0usize, 0usize);
        let (mut rmin, mut rmax) = (usize::MAX, 0usize);
        for (key, m) in &modules {
            let path = PREFIXES.iter().find_map(|p| key.strip_prefix(p)).unwrap_or(key);
            if path.starts_with("lora_te") || path.starts_with("text_encoder") || key.starts_with("text_encoder") {
                text += 1;
                continue;
            }
            let (Some(down), Some(up)) = (&m.down, &m.up) else {
                bail!("LoRA {file}: module {key} lacks its down or up matrix");
            };
            let name = if let Some(u) = path.strip_prefix("lora_unet_").or_else(|| path.strip_prefix("lora_transformer_")) {
                let map = kohya.get_or_insert_with(|| {
                    let layers: Vec<String> = shapes.keys().cloned().collect();
                    module_names(&layers, doubles, singles)
                        .into_iter()
                        .map(|n| (n.replace('.', "_"), n))
                        .collect()
                });
                map.get(u).cloned()
            } else {
                Some(xlabs(path).unwrap_or_else(|| path.to_string()))
            };
            let ps = name.as_deref().map(|n| parts(n, d)).unwrap_or_default();
            if ps.is_empty() || ps.iter().any(|(l, _)| !shapes.contains_key(l)) {
                if truncated {
                    skipped += 1;
                    continue;
                }
                bail!("LoRA {file}: module {key} matches no layer of this transformer; is it made for another model?");
            }
            let (ds, us) = (st.info(down).unwrap().shape.clone(), st.info(up).unwrap().shape.clone());
            let r = ds.first().copied().unwrap_or(0);
            let n_in = ds.iter().skip(1).product::<usize>();
            let rows_total = us.first().copied().unwrap_or(0);
            if r == 0 || us.iter().product::<usize>() != rows_total * r {
                bail!("LoRA {file}: module {key} has shapes {ds:?} / {us:?}");
            }
            let alpha = match &m.alpha {
                Some(a) => tensor_f32(&st, a)?.first().copied(),
                None => None,
            };
            let (meta_alpha, rs) = meta.as_ref().map_or((None, false), |mt| peft_alpha(mt, path));
            let alpha = alpha.or(meta_alpha).unwrap_or(r as f32);
            let scale = spec.scale * alpha / if rs { (r as f32).sqrt() } else { r as f32 };
            let (dw, uw) = (tensor_f32(&st, down)?, tensor_f32(&st, up)?);
            let ub = match &m.up_bias {
                Some(b) => Some(tensor_f32(&st, b)?),
                None => None,
            };
            let mut covered = 0;
            for (layer, range) in &ps {
                let (n_out, l_in) = shapes[layer];
                let (r0, r1) = match *range {
                    Some((a, b)) => (a, b.min(rows_total)),
                    None => (0, rows_total),
                };
                // Only `x_embedder` may be given more input channels (control LoRAs).
                let in_ok = n_in == l_in || (layer == "x_embedder" && n_in > l_in);
                if r1 < r0 || r1 - r0 != n_out || !in_ok {
                    bail!(
                        "LoRA {file}: module {key} (down {ds:?}, up {us:?}) does not fit {layer} [{n_out} x {l_in}]; is it made for another model?"
                    );
                }
                covered += n_out;
                pairs.entry(layer.clone()).or_default().push(Pair {
                    down: dw.clone(),
                    up: uw[r0 * r..r1 * r].iter().map(|v| v * scale).collect(),
                    r,
                    n_in,
                });
                if let Some(b) = &ub {
                    if b.len() != rows_total {
                        bail!("LoRA {file}: module {key} has a bias of {} values for {rows_total} rows", b.len());
                    }
                    let acc = biases.entry(layer.clone()).or_insert_with(|| vec![0.0; n_out]);
                    for (a, &v) in acc.iter_mut().zip(&b[r0..r1]) {
                        *a += scale * v;
                    }
                }
            }
            if covered != rows_total {
                bail!("LoRA {file}: module {key} has {rows_total} output rows, its layers {covered}");
            }
            applied += 1;
            (rmin, rmax) = (rmin.min(r), rmax.max(r));
        }
        for (n, k) in norm_keys {
            let v = tensor_f32(&st, &k)?;
            match norm_len.get(&n) {
                Some(&len) if len == v.len() => {
                    norms.insert(n, v);
                    applied += 1;
                }
                None if truncated => skipped += 1,
                _ => bail!("LoRA {file}: {k} does not fit this transformer"),
            }
        }
        if applied == 0 {
            bail!(
                "LoRA {file}: none of its {} modules target the FLUX transformer ({text} text-encoder modules)",
                modules.len()
            );
        }
        let mut s = format!("[lora {file} x{}: {applied} modules applied", spec.scale);
        if rmax > 0 {
            s.push_str(&if rmin == rmax {
                format!(", rank {rmin}")
            } else {
                format!(", ranks {rmin}-{rmax}")
            });
        }
        if text > 0 {
            s.push_str(&format!(
                "; {text} text-encoder modules not applied (only the transformer is patched)"
            ));
        }
        if skipped > 0 {
            s.push_str(&format!("; {skipped} modules of blocks cut by OXINFER_FLUX_LAYERS skipped"));
        }
        s.push(']');
        lines.push(s);
    }
    // Install: one stacked low-rank update per layer.
    let mut ext = None;
    for (name, layer) in tr.named_layers_mut() {
        if let Some(b) = biases.remove(&name) {
            match &mut layer.b {
                Some(lb) => lb.iter_mut().zip(&b).for_each(|(a, v)| *a += v),
                None => layer.b = Some(b),
            }
        }
        let Some(ps) = pairs.remove(&name) else { continue };
        let n_in = ps.iter().map(|p| p.n_in).max().unwrap();
        let r: usize = ps.iter().map(|p| p.r).sum();
        let mut a = vec![0f32; r * n_in];
        let mut b = vec![0f32; layer.n_out * r];
        let mut k0 = 0;
        for p in &ps {
            for k in 0..p.r {
                a[(k0 + k) * n_in..(k0 + k) * n_in + p.n_in].copy_from_slice(&p.down[k * p.n_in..(k + 1) * p.n_in]);
            }
            for o in 0..layer.n_out {
                b[o * r + k0..o * r + k0 + p.r].copy_from_slice(&p.up[o * p.r..(o + 1) * p.r]);
            }
            k0 += p.r;
        }
        let delta = Box::new(LoraDelta::fixed(a, &b, r, n_in, layer.n_out));
        if n_in > layer.n_in {
            ext = Some((n_in, delta));
        } else {
            layer.w.set_lora(Some(delta));
        }
    }
    if let Some((n_in, delta)) = ext {
        tr.set_input_extension(n_in, delta);
    }
    for (name, v) in tr.named_norms_mut() {
        if let Some(n) = norms.remove(&name) {
            *v = n;
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_parts() {
        let d = 8;
        assert_eq!(parts("img_in", d), vec![("x_embedder".to_string(), None)]);
        assert_eq!(parts("time_in.out_layer", d)[0].0, "time_text_embed.timestep_embedder.linear_2");
        let q = parts("double_blocks.3.txt_attn.qkv", d);
        assert_eq!(q[1], ("transformer_blocks.3.attn.add_k_proj".to_string(), Some((8, 16))));
        let l = parts("single_blocks.7.linear1", d);
        assert_eq!(l.len(), 4);
        assert_eq!(l[3], ("single_transformer_blocks.7.proj_mlp".to_string(), Some((24, usize::MAX))));
        assert_eq!(
            parts("final_layer.adaLN_modulation.1", d)[0],
            ("norm_out.linear.shift".to_string(), Some((0, 8)))
        );
        assert_eq!(parts("norm_out.linear", d)[0].0, "norm_out.linear.scale");
        assert_eq!(
            xlabs("double_blocks.2.processor.qkv_lora2").as_deref(),
            Some("double_blocks.2.txt_attn.qkv")
        );
        assert_eq!(
            xlabs("single_blocks.2.processor.qkv_lora").as_deref(),
            Some("single_blocks.2.linear1.qkv")
        );
        assert_eq!(parts("single_blocks.2.linear1.qkv", d).len(), 3);
        assert_eq!(
            bfl_norm("single_blocks.1.norm.key_norm").as_deref(),
            Some("single_transformer_blocks.1.attn.norm_k")
        );
        let names = module_names(&["transformer_blocks.0.attn.to_q".to_string()], 1, 1);
        let map: HashMap<String, String> = names.into_iter().map(|n| (n.replace('.', "_"), n)).collect();
        assert_eq!(map["double_blocks_0_img_attn_qkv"], "double_blocks.0.img_attn.qkv");
        assert_eq!(map["transformer_blocks_0_attn_to_q"], "transformer_blocks.0.attn.to_q");
        assert_eq!(map["final_layer_adaLN_modulation_1"], "final_layer.adaLN_modulation.1");
    }
}
