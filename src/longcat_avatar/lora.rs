//! The release's LoRAs (`loraA/lora.safetensors`: DMD, `loraB/lora.safetensors`: OPSD), kept as
//! side paths as the reference does (`y += multiplier * alpha_scale * up(down(x))`), so that the
//! multipliers can change between denoising steps and the int8 base stays as stored.
//!
//! The files are Meituan's `LoRANetwork` state dicts: per target module
//! `lora___lorahyphen___blocks___lorahyphen___{i}___lorahyphen___attn___lorahyphen___qkv` the
//! tensors `.lora_down.weight` (`[sep * r, n_in]`), `.lora_up.weight` (`[n_out, r]`) or, for a
//! fused projection, `.lora_up.blocks.{j}.weight` (`[n_out / sep, r]` each: section `j` of the
//! outputs reads rows `j r .. (j + 1) r` of the down projection), and `.alpha_scale` (the
//! scalar `alpha / r`: the file's value is what the reference ends up using, since it loads the
//! state dict strictly).

use super::weights::Shards;
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::Path;

const PREFIX: &str = "lora___lorahyphen___";
const HYPHEN: &str = "___lorahyphen___";

/// The low-rank pair of one linear layer.
pub struct LoraMod {
    /// `[sep * r, n_in]`.
    pub down: Vec<f32>,
    /// `sep` matrices `[n_out / sep, r]`.
    pub ups: Vec<Vec<f32>>,
    pub r: usize,
    pub n_in: usize,
    pub n_out: usize,
    pub alpha_scale: f32,
}

/// One LoRA file.
pub struct LoraNet {
    pub mods: HashMap<String, LoraMod>,
    pub bytes: usize,
}

impl LoraNet {
    pub fn load(path: &Path) -> Result<Self> {
        let s = Shards::open_file(path)?;
        let mut mods = HashMap::new();
        let downs: Vec<String> = s.names().filter(|n| n.ends_with(".lora_down.weight")).cloned().collect();
        for dn in downs {
            let key = dn.strip_suffix(".lora_down.weight").unwrap();
            let Some(rest) = key.strip_prefix(PREFIX) else {
                bail!("{}: {key} is not a LongCat LoRA module name", path.display());
            };
            let module = rest.replace(HYPHEN, ".");
            let dshape = s.shape(&dn)?;
            let [rt, n_in] = dshape[..] else {
                bail!("{dn}: expected a matrix, got {dshape:?}");
            };
            let mut ups = Vec::new();
            let single = format!("{key}.lora_up.weight");
            if s.has(&single) {
                ups.push((s.shape(&single)?, s.f32(&single)?));
            } else {
                let mut j = 0;
                while s.has(&format!("{key}.lora_up.blocks.{j}.weight")) {
                    let n = format!("{key}.lora_up.blocks.{j}.weight");
                    ups.push((s.shape(&n)?, s.f32(&n)?));
                    j += 1;
                }
            }
            if ups.is_empty() || !rt.is_multiple_of(ups.len()) {
                bail!("{key}: {} up matrices for a down projection of {rt} rows", ups.len());
            }
            let r = rt / ups.len();
            let per = ups[0].0[0];
            for (shape, _) in &ups {
                if shape[..] != [per, r] {
                    bail!("{key}: an up matrix of shape {shape:?}, expected [{per}, {r}]");
                }
            }
            let an = format!("{key}.alpha_scale");
            let alpha_scale = if s.has(&an) { s.f32(&an)?[0] } else { 1.0 };
            let n_out = per * ups.len();
            mods.insert(
                module,
                LoraMod {
                    down: s.f32(&dn)?,
                    ups: ups.into_iter().map(|u| u.1).collect(),
                    r,
                    n_in,
                    n_out,
                    alpha_scale,
                },
            );
        }
        if mods.is_empty() {
            bail!("{}: no lora_down / lora_up pairs", path.display());
        }
        Ok(Self { mods, bytes: s.bytes() })
    }
}

/// The LoRAs of a forward and their multipliers (zero: not applied).
#[derive(Clone, Copy, Default)]
pub struct Active<'l> {
    pub nets: &'l [(&'l LoraNet, f32)],
}

impl<'l> Active<'l> {
    /// The pairs of `module` with their factors `multiplier * alpha_scale`.
    pub fn of(&self, module: &str) -> Vec<(&'l LoraMod, f32)> {
        self.nets
            .iter()
            .filter(|(_, m)| *m != 0.0)
            .filter_map(|(n, m)| n.mods.get(module).map(|l| (l, m * l.alpha_scale)))
            .collect()
    }
}

/// `y += f * up(down(x))` for each pair (`x`: `[rows, n_in]`, `y`: `[rows, n_out]`).
pub(crate) fn add_side(y: &mut [f32], x: &[f32], rows: usize, pairs: &[(&LoraMod, f32)]) {
    use crate::kernels::for_chunks;
    use crate::sd::ops::matmul_f32;
    for &(l, f) in pairs {
        let sep = l.ups.len();
        let (r, per) = (l.r, l.n_out / sep);
        debug_assert_eq!(y.len(), rows * l.n_out);
        let d = matmul_f32(&l.down, sep * r, l.n_in, x, rows);
        for (j, up) in l.ups.iter().enumerate() {
            let dj = if sep == 1 {
                std::borrow::Cow::Borrowed(&d[..])
            } else {
                let mut v = vec![0f32; rows * r];
                for_chunks(&mut v, r, rows * r >= 1 << 16, |i, o| {
                    o.copy_from_slice(&d[i * sep * r + j * r..i * sep * r + (j + 1) * r])
                });
                std::borrow::Cow::Owned(v)
            };
            let u = matmul_f32(up, per, r, &dj, rows);
            for_chunks(y, l.n_out, rows * per >= 1 << 16, |i, o| {
                for (v, &a) in o[j * per..(j + 1) * per].iter_mut().zip(&u[i * per..(i + 1) * per]) {
                    *v += f * a;
                }
            });
        }
    }
}
