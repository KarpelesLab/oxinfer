//! LoRA files on LTX-2.5's transformer (`--lora FILE[:SCALE]`): the release's distilled LoRA
//! (`loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`, rank 450, for the dev
//! transformer) and LTX-2.x LoRAs with the same naming (`diffusion_model.<module>.lora_A.weight`
//! / `.lora_B.weight`, PEFT's `.lora_A.default.weight`, kohya's `.lora_down` / `.lora_up`, an
//! optional `.alpha`; as the release's loader, the file's `lora_alpha` metadata is not applied).
//!
//! Every targeted layer keeps its stored weights (mapped bf16, int8 ConvRot, NVFP4) and gains the
//! float32 side path `y += strength * B (A x)` (times `alpha / rank` with a module's `.alpha`;
//! [`crate::lora::LoraDelta`]).
//! A key that is not a low-rank pair, or a module no layer has, is an error.

use super::dit::Dit;
use crate::flux::lora::tensor_f32;
use crate::lora::LoraDelta;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32};

const PREFIXES: [&str; 4] = ["model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer."];
const DOWN: [&str; 3] = [".lora_A.default.weight", ".lora_A.weight", ".lora_down.weight"];
const UP: [&str; 3] = [".lora_B.default.weight", ".lora_B.weight", ".lora_up.weight"];

type Parts = (Option<String>, Option<String>, Option<String>);

/// Applies `(file, strength)` LoRAs to `dit`; returns the number of layers updated.
pub fn apply(dit: &mut Dit, loras: &[(PathBuf, f32)]) -> Result<usize> {
    apply_switched(dit, loras, None, None)
}

/// [`apply`] with the updates switched on and off by `switch` and scaled by `gain` (`f32` bits)
/// at run time (the distilled LoRA: on only in the second stage of the dev pipeline, at 0.25
/// then 0.5 in the HQ pipeline).
pub fn apply_switched(
    dit: &mut Dit,
    loras: &[(PathBuf, f32)],
    switch: Option<&Arc<AtomicBool>>,
    gain: Option<&Arc<AtomicU32>>,
) -> Result<usize> {
    let mut total = 0;
    for (path, strength) in loras {
        let st = SafeTensors::open(path, false).with_context(|| format!("open LoRA {}", path.display()))?;
        let mut pending: BTreeMap<String, Parts> = BTreeMap::new();
        for name in st.names() {
            let base = PREFIXES.iter().find_map(|p| name.strip_prefix(p)).unwrap_or(name);
            let (module, role) = if let Some(m) = DOWN.iter().find_map(|s| base.strip_suffix(s)) {
                (m, 0)
            } else if let Some(m) = UP.iter().find_map(|s| base.strip_suffix(s)) {
                (m, 1)
            } else if let Some(m) = base.strip_suffix(".alpha") {
                (m, 2)
            } else {
                bail!("{}: tensor {name} is not part of a low-rank pair", path.display());
            };
            let e = pending.entry(module.to_string()).or_default();
            match role {
                0 => e.0 = Some(name.clone()),
                1 => e.1 = Some(name.clone()),
                _ => e.2 = Some(name.clone()),
            }
        }
        let mut err: Option<anyhow::Error> = None;
        let mut applied = 0;
        dit.visit_layers(&mut |module, lin| {
            if err.is_some() {
                return;
            }
            let Some((d, u, a)) = pending.remove(module) else { return };
            let run = || -> Result<()> {
                let (d, u) = (d.context("no down matrix")?, u.context("no up matrix")?);
                let down = tensor_f32(&st, &d)?;
                let up = tensor_f32(&st, &u)?;
                let r = st.info(&d).map(|i| i.shape[0]).unwrap_or(0);
                if r == 0 || down.len() != r * lin.n_in || up.len() != lin.n_out * r {
                    bail!(
                        "{} / {} values for a [{}, {}] layer at rank {r}",
                        down.len(),
                        up.len(),
                        lin.n_out,
                        lin.n_in
                    );
                }
                // The release's loader fuses `strength B A` (no alpha): the file's `lora_alpha`
                // metadata is the rank of its largest updates (450), and the smaller ones (the
                // gates' rank 32, the patchify layers' 128) are not scaled by it. A module's own
                // `.alpha` (kohya's files) is applied, as ComfyUI does.
                let alpha = match &a {
                    Some(n) => tensor_f32(&st, n)?.first().copied().unwrap_or(r as f32),
                    None => r as f32,
                };
                let s = strength * alpha / r as f32;
                let down: Vec<f32> = down.iter().map(|v| v * s).collect();
                let mut d = LoraDelta::fixed(down, &up, r, lin.n_in, lin.n_out);
                if let Some(sw) = switch {
                    d = d.with_switch(sw.clone());
                }
                if let Some(g) = gain {
                    d = d.with_gain(g.clone());
                }
                lin.w.add_lora(Box::new(d));
                Ok(())
            };
            match run() {
                Ok(()) => applied += 1,
                Err(e) => err = Some(e.context(format!("{}: module {module}", path.display()))),
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        if let Some(m) = pending.keys().next() {
            bail!(
                "{}: module {m} matches no layer of the transformer ({} unmatched)",
                path.display(),
                pending.len()
            );
        }
        eprintln!("[LTX-2.5 LoRA {}: {applied} layers at strength {strength}]", path.display());
        total += applied;
    }
    Ok(total)
}
