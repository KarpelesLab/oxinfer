//! Sparse autoencoders over the residual stream, for feature inspection and steering
//! ([`crate::intervene`]).
//!
//! Formats:
//! - Qwen-Scope (`Qwen/SAE-Res-*`): `layer{L}.sae.pt` (a `torch.save` dict of `W_enc`
//!   `[d_sae, d_in]` (stored transposed), `W_dec` `[d_in, d_sae]`, `b_enc`, `b_dec`) with the
//!   repo's `config.json` beside it (`model_type: topk_sae`, `k`, `hook_point: resid_post`).
//!   Encode: `topk(relu(x W_enc^T + b_enc), k)`; no `b_dec` subtracted from the input.
//! - SAELens (`cfg.json` + `sae_weights.safetensors`): `W_enc` `[d_in, d_sae]`, `W_dec`
//!   `[d_sae, d_in]`, `b_enc`, `b_dec`, `threshold` (JumpReLU); `architecture` `standard`
//!   (ReLU), `jumprelu` or `topk`; `apply_b_dec_to_input`; `hook_name`
//!   `blocks.{L}.hook_resid_post`. Only `normalize_activations: none`.
//!
//! The hook point must be a layer's output (resid_post).

use crate::kernels::{dot, for_chunks};
use anyhow::{Context, Result, bail};
use std::path::Path;

/// How feature activations follow from the pre-activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// ReLU.
    Relu,
    /// JumpReLU with a per-feature threshold.
    JumpRelu,
    /// The `k` largest (after ReLU), others zero.
    TopK(usize),
}

/// A loaded SAE, weights in f32, feature-major.
pub struct Sae {
    pub d_in: usize,
    pub d_sae: usize,
    /// The layer whose output it reads, when the files say.
    pub layer: Option<usize>,
    pub arch: Arch,
    pub apply_b_dec_to_input: bool,
    w_enc: Vec<f32>, // [d_sae, d_in]
    b_enc: Vec<f32>,
    w_dec: Vec<f32>, // [d_sae, d_in]
    b_dec: Vec<f32>,
    threshold: Vec<f32>,
}

/// A 2-D float tensor read from a file: logical shape, strides, values in storage order.
struct T2 {
    shape: [usize; 2],
    stride: [usize; 2],
    data: Vec<f32>,
}

impl T2 {
    /// Row-major copy with the logical axes in order, or swapped with `t`.
    fn rows(&self, t: bool) -> Vec<f32> {
        let [r, c] = self.shape;
        let [sr, sc] = self.stride;
        let (nr, nc) = if t { (c, r) } else { (r, c) };
        let mut out = vec![0f32; nr * nc];
        let d = &self.data;
        for_chunks(&mut out, nc, true, |i, row| {
            for (j, o) in row.iter_mut().enumerate() {
                let (a, b) = if t { (j, i) } else { (i, j) };
                *o = d[a * sr + b * sc];
            }
        });
        out
    }
}

fn floats(dtype: &str, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        "BF16" => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::kernels::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect(),
        d => bail!("SAE tensor dtype {d} is not supported"),
    })
}

/// Reads every tensor of a torch or safetensors file as (name, shape, strides, values).
/// A tensor read from a file: name, shape, strides, values in storage order.
type Raw = (String, Vec<usize>, Vec<usize>, Vec<f32>);

fn read_all(path: &Path) -> Result<Vec<Raw>> {
    let bytes = std::fs::read(path).with_context(|| format!("{}", path.display()))?;
    let mut out = Vec::new();
    if crate::torch::is_torch_zip(&bytes) {
        for (name, t) in crate::torch::read_tensors_dense(&bytes)? {
            out.push((name, t.shape.clone(), t.stride.clone(), floats(t.dtype, &bytes[t.start..t.end])?));
        }
    } else {
        drop(bytes);
        let st = crate::safetensors::SafeTensors::open(path, true)?;
        let names: Vec<String> = st.names().cloned().collect();
        for n in names {
            let info = st.info(&n).unwrap().clone();
            let shape = info.shape.clone();
            let mut stride = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                stride[i] = stride[i + 1] * shape[i + 1];
            }
            out.push((n.clone(), shape, stride, floats(&info.dtype, st.bytes(&n)?)?));
        }
    }
    Ok(out)
}

/// The layer in `blocks.{L}.hook_resid_post` / `model.layers.{L}` / `layer{L}.sae.pt`.
fn layer_in(s: &str, pre: &str) -> Option<usize> {
    let i = s.find(pre)? + pre.len();
    let digits: String = s[i..].chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

impl Sae {
    /// Loads an SAE: a Qwen-Scope `layer{L}.sae.pt` (with its `config.json` in the same
    /// directory or given as `config`), a SAELens directory (`cfg.json` +
    /// `sae_weights.safetensors`), or such a weights file.
    pub fn load(path: &Path) -> Result<Sae> {
        Self::load_with(path, None)
    }

    /// [`Sae::load`] with the configuration file given explicitly.
    pub fn load_with(path: &Path, config: Option<&Path>) -> Result<Sae> {
        let (weights, cfg_path) = if path.is_dir() {
            let w = ["sae_weights.safetensors", "sae.safetensors"]
                .iter()
                .map(|f| path.join(f))
                .find(|p| p.is_file())
                .with_context(|| format!("{}: no sae_weights.safetensors", path.display()))?;
            (w, config.map(Path::to_path_buf).or_else(|| Some(path.join("cfg.json"))))
        } else {
            let dir = path.parent().unwrap_or(Path::new("."));
            let c = config
                .map(Path::to_path_buf)
                .or_else(|| ["cfg.json", "config.json"].iter().map(|f| dir.join(f)).find(|p| p.is_file()));
            (path.to_path_buf(), c)
        };
        let cfg: serde_json::Value = match &cfg_path {
            Some(p) if p.is_file() => serde_json::from_str(&std::fs::read_to_string(p)?).with_context(|| format!("{}", p.display()))?,
            _ => serde_json::Value::Null,
        };
        let s = |k: &str| cfg.get(k).and_then(|v| v.as_str()).unwrap_or("");
        let u = |k: &str| cfg.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        let norm = s("normalize_activations");
        if !norm.is_empty() && norm != "none" {
            bail!("SAE normalize_activations {norm:?} is not supported (only none)");
        }
        let k = u("k").or_else(|| {
            cfg.get("activation_fn_kwargs")
                .and_then(|a| a.get("k"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        });
        let arch_s = if s("model_type") == "topk_sae" { "topk" } else { s("architecture") };
        let hook = [s("hook_point"), s("hook_name")].concat();
        if hook.contains("resid_pre") || hook.contains("mlp") || hook.contains("attn") || hook.contains("resid_mid") {
            bail!("SAE hook {hook:?}: only residual-stream outputs (resid_post) are supported");
        }
        let fname = weights.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let layer = layer_in(s("hook_name"), "blocks.")
            .or_else(|| layer_in(s("hf_hook_name"), "layers."))
            .or_else(|| layer_in(&fname, "layer"));

        let mut ts = read_all(&weights)?;
        let mut take = |n: &str| -> Option<(Vec<usize>, Vec<usize>, Vec<f32>)> {
            let i = ts.iter().position(|t| t.0 == n)?;
            let (_, sh, st, d) = ts.swap_remove(i);
            Some((sh, st, d))
        };
        let (_, _, b_dec) = take("b_dec").context("SAE: no b_dec")?;
        let (_, _, b_enc) = take("b_enc").context("SAE: no b_enc")?;
        let d_in = b_dec.len();
        let d_sae = b_enc.len();
        let mat = |t: (Vec<usize>, Vec<usize>, Vec<f32>), what: &str| -> Result<Vec<f32>> {
            let (sh, st, d) = t;
            if sh.len() != 2 {
                bail!("SAE {what}: expected a matrix, shape {sh:?}");
            }
            let t2 = T2 {
                shape: [sh[0], sh[1]],
                stride: [st[0], st[1]],
                data: d,
            };
            // Feature-major [d_sae, d_in] either way round.
            if sh == [d_sae, d_in] {
                Ok(t2.rows(false))
            } else if sh == [d_in, d_sae] {
                Ok(t2.rows(true))
            } else {
                bail!("SAE {what}: shape {sh:?} does not match d_in {d_in} / d_sae {d_sae}")
            }
        };
        let w_enc = mat(take("W_enc").context("SAE: no W_enc")?, "W_enc")?;
        let w_dec = mat(take("W_dec").context("SAE: no W_dec")?, "W_dec")?;
        let threshold = take("threshold").map(|t| t.2).unwrap_or_default();
        let arch = match arch_s {
            "topk" => Arch::TopK(k.context("TopK SAE without k")?),
            "jumprelu" => Arch::JumpRelu,
            "standard" | "relu" | "" if threshold.is_empty() => Arch::Relu,
            "" => Arch::JumpRelu,
            a => bail!("SAE architecture {a:?} is not supported (standard, jumprelu, topk)"),
        };
        if arch == Arch::JumpRelu && threshold.len() != d_sae {
            bail!("JumpReLU SAE without a threshold per feature");
        }
        if let Some(d) = u("d_sae").or(u("d_in").map(|_| d_sae))
            && d != d_sae
        {
            bail!("SAE config d_sae {d} does not match the weights ({d_sae})");
        }
        Ok(Sae {
            d_in,
            d_sae,
            layer,
            arch,
            apply_b_dec_to_input: cfg.get("apply_b_dec_to_input").and_then(|v| v.as_bool()).unwrap_or(false),
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            threshold,
        })
    }

    /// Feature `f`'s decoder vector.
    pub fn decoder_row(&self, f: usize) -> Vec<f32> {
        self.dec(f).to_vec()
    }

    fn dec(&self, f: usize) -> &[f32] {
        &self.w_dec[f * self.d_in..(f + 1) * self.d_in]
    }

    /// Feature `f`'s encoder row and bias such that its pre-activation is `enc . x + bias`
    /// (the `b_dec` subtraction folded in).
    pub fn encoder_column(&self, f: usize) -> (Vec<f32>, f32) {
        let e = &self.w_enc[f * self.d_in..(f + 1) * self.d_in];
        let mut b = self.b_enc[f];
        if self.apply_b_dec_to_input {
            b -= dot(e, &self.b_dec);
        }
        (e.to_vec(), b)
    }

    /// The per-feature activation function when a feature's activation depends on that feature
    /// alone (ReLU, JumpReLU); None for TopK.
    pub(crate) fn item_act(&self, f: usize) -> Option<crate::intervene::ItemAct> {
        match self.arch {
            Arch::Relu => Some(crate::intervene::ItemAct::Relu),
            Arch::JumpRelu => Some(crate::intervene::ItemAct::JumpRelu(self.threshold[f])),
            Arch::TopK(_) => None,
        }
    }

    /// Pre-activations of all features for one residual row `x`.
    pub fn pre_acts(&self, x: &[f32]) -> Vec<f32> {
        let xin: Vec<f32> = if self.apply_b_dec_to_input {
            x.iter().zip(&self.b_dec).map(|(a, b)| a - b).collect()
        } else {
            x.to_vec()
        };
        let d = self.d_in;
        let mut pre = vec![0f32; self.d_sae];
        const C: usize = 512;
        for_chunks(&mut pre, C, true, |ci, ch| {
            for (j, p) in ch.iter_mut().enumerate() {
                let f = ci * C + j;
                *p = dot(&self.w_enc[f * d..(f + 1) * d], &xin) + self.b_enc[f];
            }
        });
        pre
    }

    /// Feature activations for one residual row (dense, `d_sae`).
    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        let mut a = self.pre_acts(x);
        match self.arch {
            Arch::Relu => a.iter_mut().for_each(|v| *v = v.max(0.0)),
            Arch::JumpRelu => {
                for (v, t) in a.iter_mut().zip(&self.threshold) {
                    if *v <= *t {
                        *v = 0.0;
                    }
                }
            }
            Arch::TopK(k) => {
                let mut idx: Vec<usize> = (0..a.len()).collect();
                let k = k.min(a.len());
                if k < a.len() {
                    idx.select_nth_unstable_by(k, |&i, &j| a[j].total_cmp(&a[i]));
                }
                let mut out = vec![0f32; a.len()];
                for &i in &idx[..k] {
                    out[i] = a[i].max(0.0);
                }
                a = out;
            }
        }
        a
    }

    /// Reconstruction `a W_dec + b_dec`.
    pub fn decode(&self, a: &[f32]) -> Vec<f32> {
        let mut x = self.b_dec.clone();
        for (f, &v) in a.iter().enumerate() {
            if v != 0.0 {
                for (o, d) in x.iter_mut().zip(self.dec(f)) {
                    *o += v * d;
                }
            }
        }
        x
    }

    /// The `k` most active features of `x` (nonzero ones only), most active first.
    pub fn top_features(&self, x: &[f32], k: usize) -> Vec<(usize, f32)> {
        let a = self.encode(x);
        let mut v: Vec<(usize, f32)> = a.into_iter().enumerate().filter(|p| p.1 > 0.0).collect();
        v.sort_by(|p, q| q.1.total_cmp(&p.1).then(p.0.cmp(&q.0)));
        v.truncate(k);
        v
    }

    /// For edits needing the whole encoder (TopK): `(a' - a, W_dec[f])` per edited feature.
    pub(crate) fn edit_coefs<'s>(&'s self, x: &[f32], feats: &[crate::intervene::FeatureEdit]) -> Vec<(f32, &'s [f32])> {
        use crate::intervene::FeatureOp;
        let a = self.encode(x);
        feats
            .iter()
            .map(|f| {
                let cur = a[f.id];
                let new = match f.op {
                    FeatureOp::Clamp => f.value,
                    FeatureOp::Zero => 0.0,
                    FeatureOp::Scale => cur * f.value,
                    FeatureOp::Add => cur + f.value, // not used: compiled into a plain vector
                };
                (new - cur, self.dec(f.id))
            })
            .collect()
    }

    /// A toy SAE for tests.
    #[cfg(test)]
    pub(crate) fn toy(arch: Arch, w_enc: Vec<f32>, b_enc: Vec<f32>, w_dec: Vec<f32>, b_dec: Vec<f32>, threshold: Vec<f32>) -> Sae {
        Sae {
            d_in: b_dec.len(),
            d_sae: b_enc.len(),
            layer: Some(0),
            arch,
            apply_b_dec_to_input: false,
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_archs() {
        // 3 features over 2 dims.
        let enc = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let dec = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let x = [2.0, -1.0];
        let s = Sae::toy(Arch::Relu, enc.clone(), vec![0.0, 0.0, -0.5], dec.clone(), vec![0.0, 0.0], vec![]);
        assert_eq!(s.encode(&x), vec![2.0, 0.0, 0.5]);
        let s = Sae::toy(
            Arch::TopK(1),
            enc.clone(),
            vec![0.0, 0.0, -0.5],
            dec.clone(),
            vec![0.0, 0.0],
            vec![],
        );
        assert_eq!(s.encode(&x), vec![2.0, 0.0, 0.0]);
        let s = Sae::toy(Arch::JumpRelu, enc, vec![0.0, 0.0, 0.0], dec, vec![0.1, 0.1], vec![2.5, 0.0, 0.5]);
        assert_eq!(s.encode(&x), vec![0.0, 0.0, 1.0]);
        let r = s.decode(&[0.0, 0.0, 1.0]);
        assert_eq!(r, vec![0.6, 0.6]);
        assert_eq!(s.top_features(&x, 3), vec![(2, 1.0)]);
    }

    #[test]
    fn transposed_rows() {
        // Logical [2, 3] stored transposed (stride [1, 2]).
        let t = T2 {
            shape: [2, 3],
            stride: [1, 2],
            data: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        };
        assert_eq!(t.rows(false), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.rows(true), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
