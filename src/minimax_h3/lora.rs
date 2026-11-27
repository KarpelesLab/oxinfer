//! LoRA files on the MiniMax-H3 transformer (`--lora FILE[:SCALE]`), applied at load: the 4- and
//! 8-step turbo LoRAs of `lightx2v/Minimax-h3-Turbo` (also in `Comfy-Org/MiniMax-H3`), and any
//! file with the same key naming.
//!
//! As for FLUX ([`crate::flux::lora`]), every targeted layer keeps its stored weights (mapped
//! bf16, FP8) and gains the low-rank side path `y += scale * alpha / rank * B (A x)` in float32
//! ([`crate::lora::LoraDelta`]): the merged model without rewriting 40 to 66 GB of weights.
//!
//! Key namings read (`.lora_A.weight` / `.lora_B.weight`, PEFT's `.lora_A.default.weight`, kohya's
//! `.lora_down.weight` / `.lora_up.weight`, under an optional `diffusion_model.` / `transformer.`
//! prefix):
//!
//! - ComfyUI's (the `*_comfyui_*` files; `comfy/lora.py` maps a MiniMax H3 module to its state
//!   dict name): `blocks.N.attn.qkv_proj` with `B` over `[q; k; v]` rows (the files' header says
//!   "block diagonal B; concat A; alpha multiplied by 3"), `.attn.out_proj`, `.mlp.fc1` over
//!   `[gate; value]` rows, `.mlp.fc2`, `.adaln_proj.linear`, the same under `token_refiner.`, and
//!   `final_layer.adaln_proj.linear`; an `.alpha` scalar per module;
//! - diffusers' (PEFT): `transformer_blocks.N.attn.to_q` / `to_k` / `to_v` / `to_out.0`,
//!   `.ff.net.0.proj` over `[value; gate]` rows, `.ff.net.2`, `token_refiner.refiner_blocks.N.*`,
//!   `norm_out.linear`; alpha from the header's `alpha` entry (lightx2v's files);
//! - kohya's flattened form of ComfyUI's (`lora_unet_blocks_N_attn_qkv_proj`, `..._mlp_fc1`,
//!   `lora_unet_token_refiner_blocks_N_*`, `lora_unet_final_layer_adaln_proj_linear`): the LoRAs
//!   of SulphurAI's Sulphur-3 (`s3_raw_lora`, `sulphur3_lora_r256`, `minimax_h3_max`, the
//!   `minimax_S3_tuned_*` and `s3rl*` files), extracted from or trained on ComfyUI's layout.
//!
//! Without an alpha the scale is 1 (alpha = rank). Either naming goes on either layout of the
//! base weights: a module on a matrix this checkpoint stores split, fused or with its rows in
//! another order is redistributed by rows ([`Dit::lora_parts`]). The release's own layout
//! (`qkv_proj` interleaved per head) has the names of ComfyUI's, so a `qkv_proj` module is always
//! read as ComfyUI's. A key that is not a low-rank pair or a module that matches no layer is an
//! error. Modules on the AdaLN projections (`minimax_h3_fl2v_turbo_8step_v1.2`) need the
//! unpruned transformer: a "pruned" checkpoint has no `[.., time_dim]` projection to update.

use super::dit::{Dit, LayerId};
use crate::flux::lora::tensor_f32;
use crate::lora::LoraDelta;
use crate::safetensors::SafeTensors;
pub use crate::sd::lora::LoraSpec;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
}

const SUFFIXES: [(&str, Role); 7] = [
    (".lora_A.default.weight", Role::Down),
    (".lora_B.default.weight", Role::Up),
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".alpha", Role::Alpha),
];

const PREFIXES: [&str; 4] = ["model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer."];

#[derive(Default)]
struct Module {
    down: Option<String>,
    up: Option<String>,
    alpha: Option<String>,
}

/// A low-rank pair on one stored layer: `down` `[r, n_in]`, `up` `[n_out, r]` (scale folded in).
struct Pair {
    down: Vec<f32>,
    up: Vec<f32>,
    r: usize,
}

/// The sampling a turbo LoRA was distilled for, from its file name (lightx2v's model table,
/// `github.com/ModelTC/Minimax-H3-Turbo`): the number of transformer evaluations in
/// `turbo_Nstep`, and video shift 6 for the `768p` files, 12 for the others (audio shift 3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turbo {
    /// Transformer evaluations: a sigma grid of `forwards + 1` points.
    pub forwards: usize,
    pub video_shift: f32,
    pub audio_shift: f32,
}

pub fn turbo_settings(path: &Path) -> Option<Turbo> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    let rest = &name[name.find("turbo_")? + "turbo_".len()..];
    let forwards: usize = rest.split_once("step")?.0.parse().ok()?;
    if forwards == 0 {
        return None;
    }
    Some(Turbo {
        forwards,
        video_shift: if name.contains("768p") { 6.0 } else { 12.0 },
        audio_shift: 3.0,
    })
}

/// The dotted module of a kohya-style flattened name (`lora_unet_blocks_12_attn_qkv_proj`, as
/// ComfyUI's LoRA tools and musubi-tuner write for MiniMax-H3: SulphurAI's Sulphur-3 LoRAs), or
/// None when `key` is not one.
fn kohya(key: &str) -> Option<String> {
    let rest = key.strip_prefix("lora_unet_")?;
    const TAILS: [(&str, &str); 5] = [
        ("attn_qkv_proj", "attn.qkv_proj"),
        ("attn_out_proj", "attn.out_proj"),
        ("mlp_fc1", "mlp.fc1"),
        ("mlp_fc2", "mlp.fc2"),
        ("adaln_proj_linear", "adaln_proj.linear"),
    ];
    if rest == "final_layer_adaln_proj_linear" {
        return Some("final_layer.adaln_proj.linear".into());
    }
    let (prefix, rest) = if let Some(r) = rest.strip_prefix("token_refiner_blocks_") {
        ("token_refiner.blocks.", r)
    } else {
        ("blocks.", rest.strip_prefix("blocks_")?)
    };
    let (idx, tail) = rest.split_once('_')?;
    idx.parse::<usize>().ok()?;
    let (_, t) = TAILS.iter().find(|(k, _)| *k == tail)?;
    Some(format!("{prefix}{idx}.{t}"))
}

/// ComfyUI's module for module path `path`, and for a pair that covers part of it or stores its
/// rows in another order, the module row of each row of the pair's `up`.
fn canonical(path: &str, inner: usize, ffn: usize) -> (String, Option<Vec<usize>>) {
    if let Some(m) = kohya(path) {
        return (m, None);
    }
    if path == "norm_out.linear" {
        return ("final_layer.adaln_proj.linear".into(), None);
    }
    let (prefix, rest) = if let Some(r) = path.strip_prefix("token_refiner.refiner_blocks.") {
        ("token_refiner.blocks.", r)
    } else if let Some(r) = path.strip_prefix("transformer_blocks.") {
        ("blocks.", r)
    } else {
        return (path.to_string(), None);
    };
    let Some((idx, tail)) = rest.split_once('.') else {
        return (path.to_string(), None);
    };
    let part = |j: usize| Some((j * inner..(j + 1) * inner).collect());
    let (tail, rows) = match tail {
        "attn.to_q" => ("attn.qkv_proj", part(0)),
        "attn.to_k" => ("attn.qkv_proj", part(1)),
        "attn.to_v" => ("attn.qkv_proj", part(2)),
        "attn.to_out.0" => ("attn.out_proj", None),
        // diffusers' [value; gate] against ComfyUI's [gate; value].
        "ff.net.0.proj" => ("mlp.fc1", Some((ffn..2 * ffn).chain(0..ffn).collect())),
        "ff.net.2" => ("mlp.fc2", None),
        t => (t, None),
    };
    (format!("{prefix}{idx}.{tail}"), rows)
}

/// Applies the LoRA files to `dit` (see the module docs); returns one report line per file.
pub fn apply(dit: &mut Dit, specs: &[LoraSpec]) -> Result<Vec<String>> {
    let (inner, ffn) = (dit.cfg.inner(), dit.cfg.ffn);
    let mut pairs: BTreeMap<LayerId, Vec<Pair>> = BTreeMap::new();
    let mut lines = Vec::new();
    for spec in specs {
        let file = spec.path.display().to_string();
        let st = SafeTensors::open(&spec.path, false).with_context(|| format!("LoRA {file}"))?;
        let meta_alpha: Option<f32> = st.metadata("alpha").and_then(|a| a.parse().ok());
        let mut modules: BTreeMap<String, Module> = BTreeMap::new();
        let mut odd: Vec<&String> = Vec::new();
        for k in st.names() {
            match SUFFIXES.iter().find_map(|(s, r)| k.strip_suffix(s).map(|p| (p, *r))) {
                Some((p, role)) => {
                    let m = modules.entry(p.to_string()).or_default();
                    *match role {
                        Role::Down => &mut m.down,
                        Role::Up => &mut m.up,
                        Role::Alpha => &mut m.alpha,
                    } = Some(k.clone());
                }
                None => odd.push(k),
            }
        }
        if !odd.is_empty() {
            odd.sort();
            bail!(
                "LoRA {file}: {} keys are not low-rank pairs, e.g. {}; only LoRA (down / up) modules are supported",
                odd.len(),
                odd[0]
            );
        }
        let (mut applied, mut rmin, mut rmax) = (0usize, usize::MAX, 0usize);
        let (mut smin, mut smax) = (f32::MAX, f32::MIN);
        for (key, m) in &modules {
            let (Some(down), Some(up)) = (&m.down, &m.up) else {
                bail!("LoRA {file}: module {key} lacks its down or up matrix");
            };
            let path = PREFIXES.iter().find_map(|p| key.strip_prefix(p)).unwrap_or(key);
            let (module, up_rows) = canonical(path, inner, ffn);
            let parts = dit.lora_parts(&module).with_context(|| {
                format!("LoRA {file}: module {key} matches no layer of this transformer; is it made for another model?")
            })?;
            let (ds, us) = (st.info(down).unwrap().shape.clone(), st.info(up).unwrap().shape.clone());
            let (&[r, n_in], &[rows_b, rb]) = (&ds[..], &us[..]) else {
                bail!("LoRA {file}: module {key} has shapes {ds:?} / {us:?}");
            };
            let total: usize = parts.iter().map(|(_, rows)| rows.len()).sum();
            // Where each row of the module is in the pair's `up`.
            let mut src: Vec<Option<usize>> = vec![None; total];
            match &up_rows {
                Some(map) if map.len() == rows_b && map.iter().all(|&m| m < total) => {
                    map.iter().enumerate().for_each(|(i, &m)| src[m] = Some(i));
                }
                None if rows_b == total => src.iter_mut().enumerate().for_each(|(i, s)| *s = Some(i)),
                _ => bail!("LoRA {file}: module {key} (up {us:?}) does not fit {module} of this transformer ({total} rows)"),
            }
            if r == 0 || rb != r {
                bail!("LoRA {file}: module {key} has shapes {ds:?} / {us:?}");
            }
            let alpha = match &m.alpha {
                Some(a) => tensor_f32(&st, a)?.first().copied(),
                None => meta_alpha,
            };
            let scale = spec.scale * alpha.unwrap_or(r as f32) / r as f32;
            let (dw, uw) = (tensor_f32(&st, down)?, tensor_f32(&st, up)?);
            for (id, rows) in parts {
                if rows.iter().all(|&m| src[m].is_none()) {
                    continue;
                }
                let l_in = dit.layer_mut(id).n_in;
                if n_in != l_in {
                    if module.ends_with("adaln_proj.linear") && dit.is_curve() {
                        bail!(
                            "LoRA {file}: module {key} updates an AdaLN projection, which this \"pruned\" transformer stores in curve form; use the unpruned transformer"
                        );
                    }
                    bail!("LoRA {file}: module {key} (down {ds:?}) does not fit its layer of {l_in} inputs; is it made for another model?");
                }
                // The rank components that reach this layer (a fused q / k / v pair is block
                // diagonal: a third of them per projection).
                let used: Vec<usize> = (0..r)
                    .filter(|&k| rows.iter().any(|&m| src[m].is_some_and(|i| uw[i * r + k] != 0.0)))
                    .collect();
                if used.is_empty() {
                    continue;
                }
                let ru = used.len();
                let mut a = vec![0f32; ru * n_in];
                for (j, &k) in used.iter().enumerate() {
                    a[j * n_in..(j + 1) * n_in].copy_from_slice(&dw[k * n_in..(k + 1) * n_in]);
                }
                let mut b = vec![0f32; rows.len() * ru];
                for (o, &m) in rows.iter().enumerate() {
                    if let Some(i) = src[m] {
                        for (j, &k) in used.iter().enumerate() {
                            b[o * ru + j] = scale * uw[i * r + k];
                        }
                    }
                }
                pairs.entry(id).or_default().push(Pair { down: a, up: b, r: ru });
            }
            applied += 1;
            (rmin, rmax) = (rmin.min(r), rmax.max(r));
            (smin, smax) = (smin.min(scale), smax.max(scale));
        }
        if applied == 0 {
            bail!("LoRA {file}: no low-rank modules");
        }
        let range = |a: String, b: String| if a == b { a } else { format!("{a}-{b}") };
        lines.push(format!(
            "[lora {file} x{}: {applied} modules applied, rank {}, alpha / rank x strength {}]",
            spec.scale,
            range(rmin.to_string(), rmax.to_string()),
            range(format!("{smin}"), format!("{smax}")),
        ));
    }
    // Install: one stacked low-rank update per layer.
    for (id, ps) in pairs {
        let lin = dit.layer_mut(id);
        let (n_in, n_out) = (lin.n_in, lin.n_out);
        let r: usize = ps.iter().map(|p| p.r).sum();
        let mut a = Vec::with_capacity(r * n_in);
        let mut b = vec![0f32; n_out * r];
        let mut k0 = 0;
        for p in &ps {
            a.extend_from_slice(&p.down);
            for o in 0..n_out {
                b[o * r + k0..o * r + k0 + p.r].copy_from_slice(&p.up[o * p.r..(o + 1) * p.r]);
            }
            k0 += p.r;
        }
        lin.w.set_lora(Some(Box::new(LoraDelta::fixed(a, &b, r, n_in, n_out))));
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_names() {
        let t = |n: &str| turbo_settings(Path::new(n));
        assert_eq!(
            t("loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
            Some(Turbo {
                forwards: 4,
                video_shift: 6.0,
                audio_shift: 3.0
            })
        );
        assert_eq!(
            t("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
            Some(Turbo {
                forwards: 8,
                video_shift: 12.0,
                audio_shift: 3.0
            })
        );
        assert_eq!(t("pika_h3_flex_fl2va_v1_bf16.safetensors"), None);
    }

    #[test]
    fn diffusers_modules() {
        let (m, rows) = canonical("transformer_blocks.3.attn.to_k", 4, 3);
        assert_eq!(m, "blocks.3.attn.qkv_proj");
        assert_eq!(rows, Some(vec![4, 5, 6, 7]));
        let (m, rows) = canonical("token_refiner.refiner_blocks.0.ff.net.0.proj", 4, 3);
        assert_eq!(m, "token_refiner.blocks.0.mlp.fc1");
        assert_eq!(rows, Some(vec![3, 4, 5, 0, 1, 2]));
        assert_eq!(canonical("blocks.1.mlp.fc2", 4, 3), ("blocks.1.mlp.fc2".to_string(), None));
    }

    #[test]
    fn kohya_modules() {
        let c = |p: &str| canonical(p, 4, 3).0;
        assert_eq!(c("lora_unet_blocks_12_attn_qkv_proj"), "blocks.12.attn.qkv_proj");
        assert_eq!(c("lora_unet_blocks_0_mlp_fc2"), "blocks.0.mlp.fc2");
        assert_eq!(c("lora_unet_blocks_3_adaln_proj_linear"), "blocks.3.adaln_proj.linear");
        assert_eq!(
            c("lora_unet_token_refiner_blocks_1_attn_out_proj"),
            "token_refiner.blocks.1.attn.out_proj"
        );
        assert_eq!(c("lora_unet_final_layer_adaln_proj_linear"), "final_layer.adaln_proj.linear");
        assert_eq!(kohya("lora_unet_blocks_x_mlp_fc1"), None);
        assert_eq!(kohya("lora_te_blocks_1_mlp_fc1"), None);
    }
}
