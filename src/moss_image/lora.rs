//! LoRA files on the MOSS-Image transformer (`--lora FILE[:STRENGTH]`), applied at load.
//!
//! Low-rank modules (`lora_down` / `lora_up`, PEFT's `lora_A` / `lora_B`) keep the stored weights
//! and add `strength * alpha / rank * up (down x)` in f32 after the base matmul
//! (two float32 matmuls that run where the base one does): the release advises against merging its 8-step DMD LoRAs into the
//! bf16 weights, whose update is close to bf16's rounding step. With `merge` they are merged like
//! the full-rank kinds below (faster on a GPU, at that rounding).
//!
//! LyCORIS LoKr (`lokr_w1` / `lokr_w2`, or their `_a` / `_b` factors) and LoHa (`hada_w1_a` ...)
//! modules, and full `diff` tensors, are merged into the weights at load in f32, then stored at
//! the transformer's precision (bf16 as ComfyUI does, or quantized from the f32 sum): their update
//! is full rank (FameGrid's LoKr: the Kronecker product of a 4 x 4 and a 768-wide matrix).
//! Alpha follows ComfyUI: `alpha / dim` where a factor gives the dimension, else 1.
//!
//! Module names, after a `diffusion_model.` / `model.diffusion_model.` / `transformer.` /
//! `base_model.model.` prefix: the diffusers paths (`layers.3.proj.q`, `text_embedder.blocks.0.qkv`,
//! `shared_block_modulation`, ...), ComfyUI's fused `layers.3.proj.qkv` (split by rows over q / k /
//! v), and kohya's `lora_unet_layers_3_proj_q`. Text-encoder modules are counted, not applied. A
//! module that matches no layer is an error.

use crate::safetensors::SafeTensors;
pub use crate::sd::lora::LoraSpec;
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// A low-rank pair on one layer: `down` `[r, n_in]`, `up` `[n_out, r]` with the scale folded in.
struct Pair {
    down: Vec<f32>,
    up: Vec<f32>,
    r: usize,
    n_in: usize,
}

/// A full-rank update of rows `r0..r0 + n_out` of a module's matrix.
enum Full {
    /// `scale * kron(w1, w2)`; `w1` `[a1, b1]`, `w2` `[a2, b2]`.
    Kron {
        w1: Arc<Vec<f32>>,
        a1: usize,
        b1: usize,
        w2: Arc<Vec<f32>>,
        a2: usize,
        b2: usize,
        scale: f32,
        r0: usize,
    },
    /// `scale * w` for a dense `[rows, n_in]` update (LoHa's product, `diff`), rows `r0..`.
    Dense {
        w: Arc<Vec<f32>>,
        n_in: usize,
        scale: f32,
        r0: usize,
    },
}

impl Full {
    /// The same update for the layer whose first row is row `r0` of the module's matrix.
    fn at(&self, r0: usize) -> Full {
        match self {
            Full::Kron {
                w1,
                a1,
                b1,
                w2,
                a2,
                b2,
                scale,
                ..
            } => Full::Kron {
                w1: w1.clone(),
                a1: *a1,
                b1: *b1,
                w2: w2.clone(),
                a2: *a2,
                b2: *b2,
                scale: *scale,
                r0,
            },
            Full::Dense { w, n_in, scale, .. } => Full::Dense {
                w: w.clone(),
                n_in: *n_in,
                scale: *scale,
                r0,
            },
        }
    }

    fn cols(&self) -> usize {
        match self {
            Full::Kron { b1, b2, .. } => b1 * b2,
            Full::Dense { n_in, .. } => *n_in,
        }
    }

    fn rows_total(&self) -> usize {
        match self {
            Full::Kron { a1, a2, .. } => a1 * a2,
            Full::Dense { w, n_in, .. } => w.len() / n_in,
        }
    }

    /// Adds rows `0..n` (of this layer) of the update to `w` (`[n, n_in]`).
    fn add_to(&self, w: &mut [f32], n: usize, n_in: usize) {
        match self {
            Full::Kron {
                w1,
                b1,
                w2,
                a2,
                b2,
                scale,
                r0,
                ..
            } => {
                crate::kernels::for_chunks(w, n_in, n > 64, |i, row| {
                    let g = r0 + i;
                    let (p, r) = (g / a2, g % a2);
                    let w2r = &w2[r * b2..(r + 1) * b2];
                    for q in 0..*b1 {
                        let a = scale * w1[p * b1 + q];
                        for (o, &v) in row[q * b2..(q + 1) * b2].iter_mut().zip(w2r) {
                            *o += a * v;
                        }
                    }
                });
            }
            Full::Dense { w: d, scale, r0, .. } => {
                crate::kernels::for_chunks(w, n_in, n > 64, |i, row| {
                    let src = &d[(r0 + i) * n_in..(r0 + i + 1) * n_in];
                    for (o, &v) in row.iter_mut().zip(src) {
                        *o += scale * v;
                    }
                });
            }
        }
    }
}

/// The updates of a set of LoRA files, by layer (diffusers module path), handed to the
/// transformer's loader.
#[derive(Default)]
pub struct LoraSet {
    merge_all: bool,
    pairs: HashMap<String, Vec<Pair>>,
    full: HashMap<String, Vec<Full>>,
    /// Layers the loader asked for (to find modules that match none).
    used: Mutex<HashSet<String>>,
    reports: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Role {
    Down,
    Up,
    Alpha,
    LokrW1,
    LokrW2,
    LokrW1A,
    LokrW1B,
    LokrW2A,
    LokrW2B,
    LokrT2,
    HadaW1A,
    HadaW1B,
    HadaW2A,
    HadaW2B,
    Diff,
    DiffBias,
}

/// Key suffixes, longest first where one ends another.
const SUFFIXES: [(&str, Role); 20] = [
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".lora_A.default.weight", Role::Down),
    (".lora_B.default.weight", Role::Up),
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".lora.down.weight", Role::Down),
    (".lora.up.weight", Role::Up),
    (".alpha", Role::Alpha),
    (".lokr_w1_a", Role::LokrW1A),
    (".lokr_w1_b", Role::LokrW1B),
    (".lokr_w2_a", Role::LokrW2A),
    (".lokr_w2_b", Role::LokrW2B),
    (".lokr_w1", Role::LokrW1),
    (".lokr_w2", Role::LokrW2),
    (".lokr_t2", Role::LokrT2),
    (".hada_w1_a", Role::HadaW1A),
    (".hada_w1_b", Role::HadaW1B),
    (".hada_w2_a", Role::HadaW2A),
    (".hada_w2_b", Role::HadaW2B),
];

const PREFIXES: [&str; 5] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "base_model.model.",
    "transformer.",
    "unet.",
];

/// Every layer name of a transformer with `layers` blocks (diffusers module paths), and
/// ComfyUI's fused `qkv`.
pub fn module_names(layers: usize) -> Vec<String> {
    let mut v: Vec<String> = [
        "video_embedder",
        "text_embedder.proj",
        "time_embedding.0",
        "time_embedding.2",
        "shared_block_modulation",
        "video_head_modulation",
        "final_linear_video",
    ]
    .map(String::from)
    .to_vec();
    for j in 0..2 {
        v.extend(["qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"].map(|n| format!("text_embedder.blocks.{j}.{n}")));
    }
    for i in 0..layers {
        v.extend(["q", "k", "v", "qkv", "o", "ffn_up_gate", "ffn_down"].map(|n| format!("layers.{i}.proj.{n}")));
    }
    v
}

/// The layers a module stands for, with the rows of the module's output each takes: ComfyUI's
/// fused `layers.N.proj.qkv` is three layers of `dim` rows.
fn parts(m: &str, dim: usize) -> Vec<(String, usize, usize)> {
    if let Some(b) = m.strip_suffix(".proj.qkv")
        && b.starts_with("layers.")
    {
        return ["q", "k", "v"]
            .iter()
            .enumerate()
            .map(|(i, p)| (format!("{b}.proj.{p}"), i * dim, (i + 1) * dim))
            .collect();
    }
    vec![(m.to_string(), 0, usize::MAX)]
}

/// A LoKr factor: values, rows, columns, and the inner dimension when given as a product.
type Factor = (Vec<f32>, usize, usize, Option<usize>);

#[derive(Default)]
struct Module {
    keys: BTreeMap<u8, String>,
}

impl Module {
    fn get(&self, r: Role) -> Option<&String> {
        self.keys.get(&(r as u8))
    }
}

fn tensor(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let shape = st.info(name).with_context(|| format!("missing tensor {name}"))?.shape.clone();
    Ok((crate::flux::lora::tensor_f32(st, name)?, shape))
}

/// `[a, k] @ [k, b]`.
fn matmul(x: &[f32], y: &[f32], a: usize, k: usize, b: usize) -> Vec<f32> {
    let mut o = vec![0f32; a * b];
    crate::kernels::for_chunks(&mut o, b, a > 16, |i, row| {
        for kk in 0..k {
            let s = x[i * k + kk];
            for (r, &v) in row.iter_mut().zip(&y[kk * b..(kk + 1) * b]) {
                *r += s * v;
            }
        }
    });
    o
}

fn mat2(shape: &[usize], name: &str) -> Result<(usize, usize)> {
    match *shape {
        [a, b] => Ok((a, b)),
        [a, b, 1, 1] => Ok((a, b)),
        _ => bail!("LoRA tensor {name}: expected a matrix, got shape {shape:?}"),
    }
}

impl LoraSet {
    /// Reads `specs` for a transformer of width `dim` with `layers` blocks; `merge` merges the
    /// low-rank modules too.
    pub fn load(specs: &[LoraSpec], dim: usize, layers: usize, merge: bool) -> Result<Self> {
        let mut set = LoraSet {
            merge_all: merge,
            ..Default::default()
        };
        if specs.is_empty() {
            return Ok(set);
        }
        let names = module_names(layers);
        let known: HashSet<&str> = names.iter().map(String::as_str).collect();
        let kohya: HashMap<String, &str> = names.iter().map(|n| (n.replace('.', "_"), n.as_str())).collect();
        for spec in specs {
            let file = spec.path.display().to_string();
            let st = SafeTensors::open(&spec.path, false).with_context(|| format!("LoRA {file}"))?;
            let mut modules: BTreeMap<String, Module> = BTreeMap::new();
            let mut odd: Vec<&String> = Vec::new();
            for k in st.names() {
                let (base, role) = match SUFFIXES.iter().find_map(|(s, r)| k.strip_suffix(s).map(|p| (p, *r))) {
                    Some(x) => x,
                    None => match (k.strip_suffix(".diff"), k.strip_suffix(".diff_b")) {
                        (Some(p), _) => (p, Role::Diff),
                        (_, Some(p)) => (p, Role::DiffBias),
                        _ => {
                            odd.push(k);
                            continue;
                        }
                    },
                };
                modules.entry(base.to_string()).or_default().keys.insert(role as u8, k.clone());
            }
            if !odd.is_empty() {
                odd.sort();
                let kind = if odd.iter().any(|k| k.contains("dora_scale")) {
                    " (DoRA)"
                } else {
                    ""
                };
                bail!(
                    "LoRA {file}: {} keys are not LoRA / LoKr / LoHa modules{kind}, e.g. {}",
                    odd.len(),
                    odd[0]
                );
            }
            let (mut low, mut full, mut text, mut alpha_only) = (0usize, 0usize, 0usize, 0usize);
            let mut ranks: Vec<usize> = Vec::new();
            for (key, m) in &modules {
                let path = PREFIXES.iter().find_map(|p| key.strip_prefix(p)).unwrap_or(key);
                if path.starts_with("lora_te") || path.starts_with("text_encoder") || key.starts_with("text_encoder") {
                    text += 1;
                    continue;
                }
                if m.keys.len() == 1 && m.get(Role::Alpha).is_some() {
                    // An alpha without weights (FameGrid's per-q / k / v alphas beside its fused
                    // qkv LoKr): nothing to apply.
                    alpha_only += 1;
                    continue;
                }
                let name = match path.strip_prefix("lora_unet_").or_else(|| path.strip_prefix("lora_transformer_")) {
                    Some(u) => kohya.get(u).map(|s| s.to_string()),
                    None => known.contains(path).then(|| path.to_string()),
                };
                let Some(name) = name else {
                    bail!("LoRA {file}: module {key} matches no layer of the MOSS-Image transformer; is it made for another model?");
                };
                let alpha = match m.get(Role::Alpha) {
                    Some(a) => tensor(&st, a)?.0.first().copied(),
                    None => None,
                };
                if let (Some(d), Some(u)) = (m.get(Role::Down), m.get(Role::Up)) {
                    let (dw, ds) = tensor(&st, d)?;
                    let (uw, us) = tensor(&st, u)?;
                    let (r, n_in) = mat2(&ds, d)?;
                    let (rows, r2) = mat2(&us, u)?;
                    if r2 != r || r == 0 {
                        bail!("LoRA {file}: module {key} has shapes {ds:?} / {us:?}");
                    }
                    let scale = spec.scale * alpha.unwrap_or(r as f32) / r as f32;
                    for (layer, r0, r1) in parts(&name, dim) {
                        let r1 = r1.min(rows);
                        set.pairs.entry(layer).or_default().push(Pair {
                            down: dw.clone(),
                            up: uw[r0 * r..r1 * r].iter().map(|v| v * scale).collect(),
                            r,
                            n_in,
                        });
                    }
                    low += 1;
                    ranks.push(r);
                    continue;
                }
                let lokr = |w: Role, a: Role, b: Role| -> Result<Option<Factor>> {
                    if let Some(k) = m.get(w) {
                        let (v, s) = tensor(&st, k)?;
                        let (r, c) = mat2(&s, k)?;
                        return Ok(Some((v, r, c, None)));
                    }
                    match (m.get(a), m.get(b)) {
                        (Some(ka), Some(kb)) => {
                            let (va, sa) = tensor(&st, ka)?;
                            let (vb, sb) = tensor(&st, kb)?;
                            let ((ra, ca), (rb, cb)) = (mat2(&sa, ka)?, mat2(&sb, kb)?);
                            if ca != rb {
                                bail!("LoRA {file}: {ka} / {kb} have shapes {sa:?} / {sb:?}");
                            }
                            Ok(Some((matmul(&va, &vb, ra, ca, cb), ra, cb, Some(rb))))
                        }
                        _ => Ok(None),
                    }
                };
                if m.get(Role::LokrT2).is_some() {
                    bail!("LoRA {file}: module {key} is a LoKr with a Tucker core (lokr_t2), which is not supported");
                }
                let w1 = lokr(Role::LokrW1, Role::LokrW1A, Role::LokrW1B)?;
                let w2 = lokr(Role::LokrW2, Role::LokrW2A, Role::LokrW2B)?;
                let update = if let (Some((w1, a1, b1, d1)), Some((w2, a2, b2, d2))) = (w1, w2) {
                    // ComfyUI: alpha / dim when a factor gives the dimension (w2's wins).
                    let dim_f = d2.or(d1);
                    let s = match (alpha, dim_f) {
                        (Some(a), Some(dd)) => a / dd as f32,
                        _ => 1.0,
                    };
                    Some(Full::Kron {
                        w1: Arc::new(w1),
                        a1,
                        b1,
                        w2: Arc::new(w2),
                        a2,
                        b2,
                        scale: spec.scale * s,
                        r0: 0,
                    })
                } else if let (Some(k1a), Some(k1b), Some(k2a), Some(k2b)) = (
                    m.get(Role::HadaW1A),
                    m.get(Role::HadaW1B),
                    m.get(Role::HadaW2A),
                    m.get(Role::HadaW2B),
                ) {
                    let ((v1a, s1a), (v1b, s1b)) = (tensor(&st, k1a)?, tensor(&st, k1b)?);
                    let ((v2a, s2a), (v2b, s2b)) = (tensor(&st, k2a)?, tensor(&st, k2b)?);
                    let ((ra, r), (_, ca)) = (mat2(&s1a, k1a)?, mat2(&s1b, k1b)?);
                    let ((rb, r2), (_, cb)) = (mat2(&s2a, k2a)?, mat2(&s2b, k2b)?);
                    if ra != rb || ca != cb {
                        bail!("LoRA {file}: LoHa module {key} has mismatched factors");
                    }
                    let mut w = matmul(&v1a, &v1b, ra, r, ca);
                    for (a, b) in w.iter_mut().zip(matmul(&v2a, &v2b, rb, r2, cb)) {
                        *a *= b;
                    }
                    let s = alpha.map_or(1.0, |a| a / r as f32);
                    Some(Full::Dense {
                        w: Arc::new(w),
                        n_in: ca,
                        scale: spec.scale * s,
                        r0: 0,
                    })
                } else if let Some(k) = m.get(Role::Diff) {
                    let (w, s) = tensor(&st, k)?;
                    let (_, c) = mat2(&s, k)?;
                    Some(Full::Dense {
                        w: Arc::new(w),
                        n_in: c,
                        scale: spec.scale,
                        r0: 0,
                    })
                } else {
                    None
                };
                if m.get(Role::DiffBias).is_some() {
                    bail!("LoRA {file}: module {key} has a bias diff, which is not supported");
                }
                let Some(update) = update else {
                    bail!(
                        "LoRA {file}: module {key} lacks its weights ({:?})",
                        m.keys.values().collect::<Vec<_>>()
                    );
                };
                for (layer, r0, _) in parts(&name, dim) {
                    set.full.entry(layer).or_default().push(update.at(r0));
                }
                full += 1;
            }
            if low + full == 0 {
                bail!(
                    "LoRA {file}: none of its {} modules target the MOSS-Image transformer ({text} text-encoder modules)",
                    modules.len()
                );
            }
            let mut s = format!("[lora {file} x{}: ", spec.scale);
            if low > 0 {
                let (rmin, rmax) = (ranks.iter().min().unwrap(), ranks.iter().max().unwrap());
                s.push_str(&format!(
                    "{low} low-rank modules (rank {}){}",
                    if rmin == rmax { rmin.to_string() } else { format!("{rmin}-{rmax}") },
                    if merge { ", merged" } else { "" }
                ));
            }
            if full > 0 {
                s.push_str(&format!(
                    "{}{full} full-rank modules (LoKr / LoHa / diff), merged",
                    if low > 0 { ", " } else { "" }
                ));
            }
            if text > 0 {
                s.push_str(&format!("; {text} text-encoder modules not applied"));
            }
            let _ = alpha_only;
            s.push(']');
            set.reports.push(s);
        }
        Ok(set)
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty() && self.full.is_empty()
    }

    fn mark(&self, name: &str) {
        self.used.lock().unwrap_or_else(|p| p.into_inner()).insert(name.to_string());
    }

    /// Whether layer `name` gets its updates merged into the weights.
    pub fn merged(&self, name: &str) -> bool {
        self.full.contains_key(name) || (self.merge_all && self.pairs.contains_key(name))
    }

    /// Adds the merged updates of layer `name` to its f32 weights `w` (`[n, n_in]`).
    pub fn merge_into(&self, name: &str, w: &mut [f32], n: usize, n_in: usize) -> Result<()> {
        self.mark(name);
        for u in self.full.get(name).into_iter().flatten() {
            let rows_ok = match u {
                Full::Kron { r0, .. } | Full::Dense { r0, .. } => r0 + n <= u.rows_total(),
            };
            if u.cols() != n_in || !rows_ok {
                bail!(
                    "a LoRA update of {name} ({} x {}) does not fit the layer [{n} x {n_in}]; is it made for another model?",
                    u.rows_total(),
                    u.cols()
                );
            }
            u.add_to(w, n, n_in);
        }
        if self.merge_all {
            for p in self.pairs.get(name).into_iter().flatten() {
                if p.n_in != n_in || p.up.len() != n * p.r {
                    bail!("a LoRA module of {name} does not fit the layer [{n} x {n_in}]");
                }
                // w += up @ down
                crate::kernels::for_chunks(w, n_in, n > 64, |o, row| {
                    for k in 0..p.r {
                        let u = p.up[o * p.r + k];
                        for (x, &dv) in row.iter_mut().zip(&p.down[k * n_in..(k + 1) * n_in]) {
                            *x += u * dv;
                        }
                    }
                });
            }
        }
        Ok(())
    }

    /// The runtime low-rank update of layer `name` (`[n_out x n_in]`), its pairs stacked:
    /// `(down [r, n_in], up [n_out, r], r)` with the strengths in `up`.
    #[allow(clippy::type_complexity)]
    pub fn runtime(&self, name: &str, n_out: usize, n_in: usize) -> Result<Option<(Vec<f32>, Vec<f32>, usize)>> {
        if self.merge_all {
            return Ok(None);
        }
        let Some(ps) = self.pairs.get(name) else { return Ok(None) };
        self.mark(name);
        let r: usize = ps.iter().map(|p| p.r).sum();
        let mut a = vec![0f32; r * n_in];
        let mut b = vec![0f32; n_out * r];
        let mut k0 = 0;
        for p in ps {
            if p.n_in != n_in || p.up.len() != n_out * p.r {
                bail!(
                    "a LoRA module of {name} (rank {}, {} inputs) does not fit the layer [{n_out} x {n_in}]; is it made for another model?",
                    p.r,
                    p.n_in
                );
            }
            a[k0 * n_in..(k0 + p.r) * n_in].copy_from_slice(&p.down);
            for o in 0..n_out {
                b[o * r + k0..o * r + k0 + p.r].copy_from_slice(&p.up[o * p.r..(o + 1) * p.r]);
            }
            k0 += p.r;
        }
        Ok(Some((a, b, r)))
    }

    /// After the transformer is loaded: an error for updates no layer took (unless `truncated`,
    /// a run on fewer blocks); returns the report lines.
    pub fn finish(&self, truncated: bool) -> Result<Vec<String>> {
        let used = self.used.lock().unwrap_or_else(|p| p.into_inner());
        let mut missed: Vec<&String> = self.pairs.keys().chain(self.full.keys()).filter(|k| !used.contains(*k)).collect();
        missed.sort();
        missed.dedup();
        if !missed.is_empty() && !truncated {
            bail!("LoRA modules matched no loaded layer, e.g. {}", missed[0]);
        }
        Ok(self.reports.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_qkv_parts() {
        let p = parts("layers.3.proj.qkv", 8);
        assert_eq!(p.len(), 3);
        assert_eq!(p[1], ("layers.3.proj.k".to_string(), 8, 16));
        assert_eq!(parts("text_embedder.blocks.0.qkv", 8)[0].0, "text_embedder.blocks.0.qkv");
        let names = module_names(2);
        assert!(names.contains(&"layers.1.proj.ffn_up_gate".to_string()));
        assert!(names.contains(&"text_embedder.blocks.1.ffn_down".to_string()));
    }

    #[test]
    fn kron_rows() {
        // kron([[1, 2], [3, 4]], [[1, 0, 1]]) = [[1 0 1 2 0 2], [3 0 3 4 0 4]]; rows from 1.
        let k = Full::Kron {
            w1: Arc::new(vec![1.0, 2.0, 3.0, 4.0]),
            a1: 2,
            b1: 2,
            w2: Arc::new(vec![1.0, 0.0, 1.0]),
            a2: 1,
            b2: 3,
            scale: 0.5,
            r0: 1,
        };
        let mut w = vec![0f32; 6];
        k.add_to(&mut w, 1, 6);
        assert_eq!(w, vec![1.5, 0.0, 1.5, 2.0, 0.0, 2.0]);
    }
}
