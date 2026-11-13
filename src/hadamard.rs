//! Hadamard-folded weights (`prism.hadamard.*` GGUF metadata; PrismML's Ternary-Bonsai 2 models,
//! read by llama.cpp from b11556 on).
//!
//! A folded weight is `W_f = W D H`: the original `W` with a diagonal of `+1 / -1` signs `D` and
//! the normalized Sylvester Walsh-Hadamard matrix `H` (applied to consecutive blocks of
//! `block_size` inputs) multiplied in from the right, which spreads the weights out before they
//! are made ternary. `H H = I` and `D D = I`, so `W x = W_f (H (D x))`: the runtime flips the
//! signs of the matmul input and rotates it ([`Fold::apply`]). The token embedding is stored as
//! latent rows `z` and read back as `D (H z)` ([`Fold::unfold`]); with `tied_output` (version 2)
//! the same table is also the output head, a folded weight like the others.
//!
//! The Gated DeltaNet output projection (`ssm_out`) of a folded checkpoint keeps its value heads
//! in the checkpoint's grouped order while everything upstream of it uses llama.cpp's tiled
//! order (`prism.hadamard.gdn_v_grouped`), so its input is permuted first.
//!
//! Without the transform a folded file loads and produces noise, so anything this module does
//! not recognise (another transform, axis, architecture or tensor kind) is an error.

use crate::gguf::{Header, Value};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::sync::Arc;

/// The input transform of one folded weight (shared by all weights of the same width).
#[derive(Debug)]
pub struct Fold {
    /// Rotation block (a power of two dividing the width).
    pub block: usize,
    /// `+1 / -1` per input; empty for the identity sign mode.
    pub signs: Vec<f32>,
    /// `(head_dim, key_heads, repeat)`: the input holds `key_heads * repeat` value heads of
    /// `head_dim` in tiled order (head `k + key_heads * r`), read in grouped order (`r + repeat * k`).
    pub perm: Option<(usize, usize, usize)>,
}

/// The normalized Walsh-Hadamard transform of `x` (a power-of-two length), in Sylvester order:
/// `y[r] = sum_c (-1)^popcount(r & c) x[c] / sqrt(n)`.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for pair in x.chunks_exact_mut(2 * h) {
            let (a, b) = pair.split_at_mut(h);
            for (u, v) in a.iter_mut().zip(b) {
                let (s, d) = (*u + *v, *u - *v);
                *u = s;
                *v = d;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= scale);
}

impl Fold {
    /// `H (D x)` of every row of `x` (rows of `n_in`), into `out`.
    pub fn apply(&self, x: &[f32], n_in: usize, out: &mut [f32]) {
        debug_assert!(x.len() == out.len() && x.len().is_multiple_of(n_in) && n_in.is_multiple_of(self.block));
        for (xr, o) in x.chunks_exact(n_in).zip(out.chunks_exact_mut(n_in)) {
            match self.perm {
                Some((hd, nk, rep)) => {
                    for k in 0..nk {
                        for r in 0..rep {
                            let (src, dst) = (hd * (k + nk * r), hd * (r + rep * k));
                            o[dst..dst + hd].copy_from_slice(&xr[src..src + hd]);
                        }
                    }
                }
                None => o.copy_from_slice(xr),
            }
            for (v, &s) in o.iter_mut().zip(&self.signs) {
                *v *= s;
            }
            o.chunks_exact_mut(self.block).for_each(fwht);
        }
    }

    /// A latent table row back in the model's basis, in place: `D (H z)`.
    pub fn unfold(&self, row: &mut [f32]) {
        row.chunks_exact_mut(self.block).for_each(fwht);
        for (v, &s) in row.iter_mut().zip(&self.signs) {
            *v *= s;
        }
    }
}

/// The folded tensors of a GGUF file, by GGUF tensor name.
#[derive(Default)]
pub struct Spec {
    /// Matmul weights: their input is transformed ([`Fold::apply`]).
    pub weights: HashMap<String, Arc<Fold>>,
    /// Lookup tables: their rows are transformed back ([`Fold::unfold`]).
    pub tables: HashMap<String, Arc<Fold>>,
}

/// The tensor kinds whose matmul goes through the folded path (`blk.N.KIND.weight`).
const KINDS: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_gate",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ssm_out",
];

fn strings(h: &Header, key: &str) -> Result<Vec<String>> {
    h.get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("GGUF metadata {key} is missing"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .with_context(|| format!("GGUF metadata {key}: not a string"))
        })
        .collect()
}

fn ints(h: &Header, key: &str) -> Result<Vec<i64>> {
    h.get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("GGUF metadata {key} is missing"))?
        .iter()
        .map(|v| match v {
            Value::I(x) => Ok(*x),
            Value::U(x) => Ok(*x as i64),
            _ => bail!("GGUF metadata {key}: not an integer"),
        })
        .collect()
}

/// Reads the `prism.hadamard.*` metadata of `h` (None when the file has none), with the checks
/// of llama.cpp's `load_hparams_hadamard`.
pub fn parse(h: &Header) -> Result<Option<Spec>> {
    let Some(version) = h.get("prism.hadamard.version").and_then(Value::as_u64) else {
        return Ok(None);
    };
    let arch = h.arch();
    // The loaders that route every folded matmul through `model::linear_multi`.
    if arch != "qwen35" {
        bail!("Hadamard-folded weights (prism.hadamard) are not supported for the {arch} architecture");
    }
    let tied = matches!(h.get("prism.hadamard.tied_output"), Some(Value::Bool(true)));
    if !matches!(version, 1 | 2) || (version == 2) != tied {
        bail!("unsupported prism.hadamard version {version} (tied_output {tied})");
    }
    let has = |n: &str| h.tensors.iter().any(|t| t.name == n);
    if tied && has("output.weight") {
        bail!("prism.hadamard.tied_output with an output.weight tensor");
    }
    let s = |k: &str| h.get(k).and_then(Value::as_str).unwrap_or("");
    let block = h.get("prism.hadamard.block_size").and_then(Value::as_u64).unwrap_or(0) as usize;
    if block == 0 || !block.is_power_of_two() {
        bail!("invalid prism.hadamard.block_size {block}");
    }
    let (transform, axis, sign_mode) = (
        s("prism.hadamard.transform"),
        s("prism.hadamard.axis"),
        s("prism.hadamard.sign_mode"),
    );
    if transform != "normalized-sylvester-walsh-hadamard" {
        bail!("unsupported prism.hadamard.transform {transform:?}");
    }
    if axis != "input-last-dimension" {
        bail!("unsupported prism.hadamard.axis {axis:?}");
    }
    let mut signs: HashMap<usize, Vec<f32>> = HashMap::new();
    match sign_mode {
        "identity" => {}
        "explicit" => {
            let widths = ints(h, "prism.hadamard.sign_widths")?;
            let values = ints(h, "prism.hadamard.sign_values")?;
            if widths.is_empty() {
                bail!("prism.hadamard.sign_mode is explicit but sign_widths is empty");
            }
            let mut off = 0usize;
            for w in widths {
                let w = usize::try_from(w).ok().filter(|w| *w > 0 && w.is_multiple_of(block));
                let Some(w) = w.filter(|w| off + w <= values.len()) else {
                    bail!("invalid prism.hadamard sign width");
                };
                let v = &values[off..off + w];
                if v.iter().any(|x| !matches!(x, 1 | -1)) {
                    bail!("prism.hadamard sign values must be +1 or -1");
                }
                if signs.insert(w, v.iter().map(|&x| x as f32).collect()).is_some() {
                    bail!("duplicate prism.hadamard sign width {w}");
                }
                off += w;
            }
            if off != values.len() {
                bail!("prism.hadamard.sign_values length mismatch");
            }
        }
        m => bail!("unsupported prism.hadamard.sign_mode {m:?}"),
    }
    let grouped = matches!(h.get("prism.hadamard.gdn_v_grouped"), Some(Value::Bool(true)));
    let (n_v, n_k) = (
        h.get(&format!("{arch}.ssm.time_step_rank")).and_then(Value::as_u64).unwrap_or(0) as usize,
        h.get(&format!("{arch}.ssm.group_count")).and_then(Value::as_u64).unwrap_or(0) as usize,
    );

    // One Fold per (width, permutation), shared.
    let mut folds: HashMap<(usize, bool), Arc<Fold>> = HashMap::new();
    let mut fold = |name: &str, perm: bool| -> Result<Arc<Fold>> {
        let t = h
            .tensors
            .iter()
            .find(|t| t.name == name)
            .with_context(|| format!("prism.hadamard weight {name} is not in the file"))?;
        let width = *t.ne.first().context("prism.hadamard weight without dimensions")?;
        if !width.is_multiple_of(block) {
            bail!("prism.hadamard block size {block} does not divide the input width {width} of {name}");
        }
        if let Some(f) = folds.get(&(width, perm)) {
            return Ok(f.clone());
        }
        let sv = match signs.get(&width) {
            Some(v) => v.clone(),
            None if signs.is_empty() => Vec::new(),
            None => bail!("prism.hadamard has no sign vector for width {width} ({name})"),
        };
        let perm_dims = if perm {
            if n_k == 0 || n_v == 0 || !n_v.is_multiple_of(n_k) || !width.is_multiple_of(n_v) {
                bail!("prism.hadamard: bad Gated DeltaNet head geometry for {name}");
            }
            Some((width / n_v, n_k, n_v / n_k)).filter(|p| p.2 > 1)
        } else {
            None
        };
        let f = Arc::new(Fold {
            block,
            signs: sv,
            perm: perm_dims,
        });
        folds.insert((width, perm), f.clone());
        Ok(f)
    };

    let mut spec = Spec::default();
    for name in strings(h, "prism.hadamard.weight_names")? {
        let kind = name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .filter(|(l, _)| !l.is_empty() && l.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|(_, r)| r.strip_suffix(".weight"));
        if name != "output.weight" && !kind.is_some_and(|k| KINDS.contains(&k)) {
            bail!("prism.hadamard: weight {name} is not on a matmul path that applies the transform");
        }
        let f = fold(&name, grouped && kind == Some("ssm_out"))?;
        if spec.weights.insert(name.clone(), f).is_some() {
            bail!("duplicate prism.hadamard weight {name}");
        }
    }
    if spec.weights.is_empty() {
        bail!("prism.hadamard.weight_names is empty");
    }
    if h.get("prism.hadamard.inverse_weight_names").is_some() {
        for name in strings(h, "prism.hadamard.inverse_weight_names")? {
            if name != "token_embd.weight" || spec.weights.contains_key(&name) {
                bail!("prism.hadamard: {name} is not a table read through the inverse transform");
            }
            let f = fold(&name, false)?;
            spec.tables.insert(name, f);
        }
    }
    match (tied, spec.tables.get("token_embd.weight")) {
        (true, Some(f)) => {
            spec.weights.insert("token_embd.weight".into(), f.clone());
        }
        (true, None) => bail!("prism.hadamard.tied_output requires a latent token embedding"),
        (false, Some(_)) if !has("output.weight") => bail!("a tied Hadamard output requires version 2 and tied_output"),
        _ => {}
    }
    Ok(Some(spec))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// The butterfly gives llama.cpp's matrix: `(-1)^popcount(r & c) / sqrt(n)`.
    #[test]
    fn fwht_is_the_sylvester_matrix() {
        let n = 64usize;
        let mut seed = 7;
        let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        let mut y = x.clone();
        fwht(&mut y);
        for r in 0..n {
            let want: f32 = (0..n)
                .map(|c: usize| if (r & c).count_ones() & 1 == 1 { -x[c] } else { x[c] })
                .sum::<f32>()
                / (n as f32).sqrt();
            assert!((y[r] - want).abs() < 1e-5, "row {r}: {} vs {want}", y[r]);
        }
        // An involution.
        fwht(&mut y);
        assert!(y.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    /// `W x = W_f (H (D x))` for `W_f = W D H`, with and without the value-head permutation, and
    /// `unfold` inverts a latent row.
    #[test]
    fn fold_reproduces_the_unfolded_product() {
        let (hd, nk, rep) = (8, 4, 3);
        let (n_in, n_out, block) = (hd * nk * rep, 5, 32);
        let mut seed = 11;
        let signs: Vec<f32> = (0..n_in).map(|_| if lcg(&mut seed) > 0.0 { 1.0 } else { -1.0 }).collect();
        let w: Vec<f32> = (0..n_out * n_in).map(|_| lcg(&mut seed)).collect();
        // Fold each row: (W D) H, H symmetric so a row transforms like a vector.
        let mut wf = w.clone();
        for row in wf.chunks_exact_mut(n_in) {
            for (v, s) in row.iter_mut().zip(&signs) {
                *v *= s;
            }
            row.chunks_exact_mut(block).for_each(fwht);
        }
        let x: Vec<f32> = (0..2 * n_in).map(|_| lcg(&mut seed)).collect();
        let dot = |w: &[f32], x: &[f32]| -> Vec<f32> {
            x.chunks_exact(n_in)
                .flat_map(|xr| {
                    w.chunks_exact(n_in)
                        .map(move |wr| wr.iter().zip(xr).map(|(a, b)| a * b).sum::<f32>())
                })
                .collect()
        };
        for perm in [None, Some((hd, nk, rep))] {
            let f = Fold {
                block,
                signs: signs.clone(),
                perm,
            };
            let mut xf = vec![0f32; x.len()];
            f.apply(&x, n_in, &mut xf);
            // The reference input: grouped order when the fold permutes.
            let mut xg = x.clone();
            if perm.is_some() {
                for (xr, o) in x.chunks_exact(n_in).zip(xg.chunks_exact_mut(n_in)) {
                    for v in 0..nk * rep {
                        // Tiled head v = k + nk * r is grouped head r + rep * k.
                        let (k, r) = (v % nk, v / nk);
                        o[hd * (r + rep * k)..][..hd].copy_from_slice(&xr[hd * v..][..hd]);
                    }
                }
            }
            let (got, want) = (dot(&wf, &xf), dot(&w, &xg));
            assert!(
                got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4),
                "{perm:?}: {got:?} vs {want:?}"
            );
        }
        let f = Fold {
            block,
            signs: signs.clone(),
            perm: None,
        };
        // A latent row z = H (D e) reads back as e.
        let e: Vec<f32> = (0..n_in).map(|_| lcg(&mut seed)).collect();
        let mut z = vec![0f32; n_in];
        f.apply(&e, n_in, &mut z);
        f.unfold(&mut z);
        assert!(z.iter().zip(&e).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    fn header(meta: Vec<(&str, Value)>, tensors: &[(&str, &[usize])]) -> Header {
        Header {
            meta: meta.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            tensors: tensors
                .iter()
                .map(|(n, ne)| crate::gguf::Tensor {
                    name: n.to_string(),
                    ne: ne.to_vec(),
                    ggml_type: 42,
                    start: 0,
                    end: 0,
                })
                .collect(),
        }
    }

    fn base(version: u64, tied: bool) -> Vec<(&'static str, Value)> {
        let st = |s: &str| Value::Str(s.to_string());
        let names = |v: &[&str]| Value::Array(v.iter().map(|s| st(s)).collect());
        let signs: Vec<Value> = (0..64 + 128).map(|i| Value::I(if i % 3 == 0 { -1 } else { 1 })).collect();
        vec![
            ("general.architecture", st("qwen35")),
            ("qwen35.ssm.time_step_rank", Value::U(4)),
            ("qwen35.ssm.group_count", Value::U(2)),
            ("prism.hadamard.version", Value::U(version)),
            ("prism.hadamard.tied_output", Value::Bool(tied)),
            ("prism.hadamard.block_size", Value::U(32)),
            ("prism.hadamard.transform", st("normalized-sylvester-walsh-hadamard")),
            ("prism.hadamard.axis", st("input-last-dimension")),
            ("prism.hadamard.sign_mode", st("explicit")),
            ("prism.hadamard.sign_widths", Value::Array(vec![Value::I(64), Value::I(128)])),
            ("prism.hadamard.sign_values", Value::Array(signs)),
            ("prism.hadamard.gdn_v_grouped", Value::Bool(true)),
            (
                "prism.hadamard.weight_names",
                names(&[
                    "blk.0.ffn_down.weight",
                    "blk.0.ffn_up.weight",
                    "blk.0.ssm_out.weight",
                    "blk.1.attn_q.weight",
                ]),
            ),
            ("prism.hadamard.inverse_weight_names", names(&["token_embd.weight"])),
        ]
    }

    const TENSORS: &[(&str, &[usize])] = &[
        ("token_embd.weight", &[64, 100]),
        ("blk.0.ffn_down.weight", &[128, 64]),
        ("blk.0.ffn_up.weight", &[64, 128]),
        ("blk.0.ssm_out.weight", &[128, 64]),
        ("blk.1.attn_q.weight", &[64, 64]),
    ];

    #[test]
    fn metadata_is_parsed_and_checked() {
        // No metadata: no spec.
        assert!(
            parse(&header(vec![("general.architecture", Value::Str("qwen35".into()))], TENSORS))
                .unwrap()
                .is_none()
        );

        // Version 2: the embedding table is also the (folded) output head.
        let spec = parse(&header(base(2, true), TENSORS)).unwrap().unwrap();
        assert_eq!(spec.weights.len(), 5);
        assert!(Arc::ptr_eq(&spec.weights["token_embd.weight"], &spec.tables["token_embd.weight"]));
        let (down, up, out) = (
            &spec.weights["blk.0.ffn_down.weight"],
            &spec.weights["blk.0.ffn_up.weight"],
            &spec.weights["blk.0.ssm_out.weight"],
        );
        assert_eq!((down.signs.len(), up.signs.len(), down.block), (128, 64, 32));
        assert!(Arc::ptr_eq(up, &spec.weights["blk.1.attn_q.weight"]));
        // The second width's signs start after the first's 64 values.
        assert_eq!(down.signs[0], if 64 % 3 == 0 { -1.0 } else { 1.0 });
        assert_eq!(down.signs[2], -1.0);
        // ssm_out: 4 value heads of 32 over 2 key heads, permuted; same signs as ffn_down.
        assert_eq!(out.perm, Some((32, 2, 2)));
        assert!(down.perm.is_none() && out.signs == down.signs);

        // Version 1 needs its own output.weight.
        assert!(parse(&header(base(1, false), TENSORS)).is_err());
        let mut with_out = TENSORS.to_vec();
        with_out.push(("output.weight", &[64, 100]));
        let spec = parse(&header(base(1, false), &with_out)).unwrap().unwrap();
        assert!(!spec.weights.contains_key("token_embd.weight") && spec.tables.len() == 1);

        // Refused: version / tied mismatch, another transform, another architecture, a tensor
        // kind off the folded matmul path, a width without signs.
        assert!(parse(&header(base(2, false), TENSORS)).is_err());
        let edit = |k: &str, v: Value| {
            let mut m = base(2, true);
            m.iter_mut().find(|(n, _)| *n == k).unwrap().1 = v;
            parse(&header(m, TENSORS))
        };
        assert!(edit("prism.hadamard.transform", Value::Str("other".into())).is_err());
        assert!(edit("general.architecture", Value::Str("qwen35moe".into())).is_err());
        assert!(edit("prism.hadamard.block_size", Value::U(48)).is_err());
        assert!(
            edit(
                "prism.hadamard.weight_names",
                Value::Array(vec![Value::Str("blk.0.ssm_alpha.weight".into())])
            )
            .is_err()
        );
        assert!(edit("prism.hadamard.sign_widths", Value::Array(vec![Value::I(64)])).is_err());
    }
}
