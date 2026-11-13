//! Liquid LFM3 (`model_type` "lfm3", e.g. LiquidAI/LFM3-1.7B): gated short-convolution blocks
//! interleaved with attention blocks, on the Qwen code path (`crate::qwen`).
//!
//! ```text
//! layer:  x += post_operator_norm(op(operator_norm(x)))        // op: conv or attention
//!         x += post_ffn_norm(w2(silu(w1(n)) * w3(n))),  n = ffn_norm(x)
//! conv:   [B | C | u] = in_proj(n);  y = out_proj(C * conv7(B * u))   // causal, depthwise
//! attn:   q, k RMS-normed per head, RoPE, softmax; each head's output times
//!         sigmoid(gate_proj(n)[head]); o_proj
//! ```
//!
//! `layer_types` names each layer `conv`, `full_attention` or `sliding_attention` (a causal
//! window of `sliding_window` keys, the query's own included). The two attention kinds have their
//! own RoPE parameters; a `partial_rotary_factor` keeps the frequency progression of the whole
//! head dimension and zeroes the tail, so the last pairs of each half are not rotated. The final
//! norm is `embedding_norm`, and the head is tied to the embeddings.
//!
//! The conv state (the last `conv_L_cache - 1` conv inputs of each layer) is recurrent state, like
//! Gated DeltaNet's: a session cannot be cut back to an arbitrary position, only inside the last
//! pass when it recorded its conv inputs (speculative verification, see `QwenSession::rewind`).

use crate::config::QwenConfig;
use crate::kernels::*;
use crate::model::{Lin, linear};
use crate::qwen::{Loader, Mlp};
use anyhow::{Result, bail};
use serde::Deserialize;

/// RoPE of one attention kind (`rope_parameters.full_attention` / `.sliding_attention`).
#[derive(Debug, Clone, Deserialize)]
pub struct Lfm3Rope {
    #[serde(default = "d_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

fn d_theta() -> f64 {
    1_000_000.0
}

impl Lfm3Rope {
    /// `head_dim / 2` inverse frequencies: the first `factor * head_dim / 2` of the progression
    /// over the whole head dimension, zeros after.
    pub fn inv_freq(&self, head_dim: usize) -> Vec<f32> {
        let angles = (self.partial_rotary_factor.unwrap_or(1.0) * head_dim as f64) as usize / 2;
        (0..head_dim / 2)
            .map(|i| {
                if i < angles {
                    (1.0 / self.rope_theta.powf((2 * i) as f64 / head_dim as f64)) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// The LFM3 keys of a config (`QwenConfig::lfm3`).
#[derive(Debug, Clone)]
pub struct Lfm3Config {
    /// Conv kernel width (`conv_L_cache`: 7).
    pub kernel: usize,
    pub conv_bias: bool,
    /// Keys a `sliding_attention` layer sees, the query's own included.
    pub window: usize,
    /// RMSNorms on the operator and MLP outputs (`use_sandwich_norm`).
    pub sandwich: bool,
    pub full: Lfm3Rope,
    pub sliding: Lfm3Rope,
}

impl Lfm3Config {
    /// From the raw `config.json` (the defaults are transformers' `Lfm3Config`'s).
    pub fn from_json(c: &serde_json::Value) -> Result<Self> {
        let rope = |kind: &str, theta: f64, factor: Option<f64>| -> Result<Lfm3Rope> {
            let r = match c.get("rope_parameters").and_then(|r| r.get(kind)) {
                Some(v) => serde_json::from_value(v.clone())?,
                None => Lfm3Rope {
                    rope_theta: theta,
                    rope_type: None,
                    partial_rotary_factor: factor,
                },
            };
            if r.rope_type.as_deref().is_some_and(|t| t != "default") {
                bail!("LFM3 {kind} RoPE type {:?} is not supported", r.rope_type);
            }
            Ok(r)
        };
        Ok(Lfm3Config {
            kernel: c.get("conv_L_cache").and_then(|v| v.as_u64()).unwrap_or(7) as usize,
            conv_bias: c.get("conv_bias").and_then(|v| v.as_bool()).unwrap_or(false),
            window: c.get("sliding_window").and_then(|v| v.as_u64()).unwrap_or(256) as usize,
            sandwich: c.get("use_sandwich_norm").and_then(|v| v.as_bool()).unwrap_or(true),
            full: rope("full_attention", 1_000_000.0, Some(0.5))?,
            sliding: rope("sliding_attention", 10_000.0, None)?,
        })
    }

    /// The default layer pattern: full attention on every fourth layer from the third, sliding
    /// attention between them, convs elsewhere.
    pub fn default_layer_types(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| match i % 4 {
                2 => "full_attention",
                0 if i > 0 => "sliding_attention",
                _ => "conv",
            })
            .map(str::to_string)
            .collect()
    }
}

/// The gated short convolution of a `conv` layer.
pub(crate) struct ShortConv<'a> {
    pub(crate) in_proj: Lin<'a>,  // [3 * H, H]: B, C, u
    pub(crate) out_proj: Lin<'a>, // [H, H]
    /// `[H, K]`: output `t` is `sum_k w[k] * input[t + k - (K - 1)]`.
    pub(crate) conv_w: Vec<f32>,
    pub(crate) kernel: usize,
    /// `in_proj`, conv and `out_proj` biases (`conv_bias`).
    pub(crate) bias: Option<[Vec<f32>; 3]>,
}

impl<'a> ShortConv<'a> {
    pub(crate) fn load(ld: &Loader<'a, '_>, p: &str, lf: &Lfm3Config) -> Result<Self> {
        let h = ld.cfg.hidden_size;
        let k = lf.kernel;
        Ok(ShortConv {
            in_proj: ld.lin(&format!("{p}in_proj.weight"), 3 * h, h)?,
            out_proj: ld.lin(&format!("{p}out_proj.weight"), h, h)?,
            conv_w: ld.getf(&format!("{p}conv.weight"), &[h, 1, k])?,
            kernel: k,
            bias: if lf.conv_bias {
                Some([
                    ld.getf(&format!("{p}in_proj.bias"), &[3 * h])?,
                    ld.getf(&format!("{p}conv.bias"), &[h])?,
                    ld.getf(&format!("{p}out_proj.bias"), &[h])?,
                ])
            } else {
                None
            },
        })
    }

    pub(crate) fn lins(&self) -> Vec<&Lin<'a>> {
        vec![&self.in_proj, &self.out_proj]
    }

    /// `t` tokens of normed input `hn` (`[t, H]`) to `y`; `state` (`[K - 1, H]`, oldest first)
    /// holds the conv inputs before them and is left holding the last `K - 1`. `inputs` receives
    /// the conv inputs of the `t` tokens (`[t, H]`: what rebuilds the state at any of them).
    pub(crate) fn forward(&self, state: &mut [f32], hn: &[f32], t: usize, h: usize, y: &mut [f32], inputs: Option<&mut Vec<f32>>) {
        let k = self.kernel;
        let keep = k - 1;
        debug_assert_eq!(state.len(), keep * h);
        let mut bcu = vec![0f32; t * 3 * h];
        linear(&self.in_proj, 3 * h, h, hn, t, &mut bcu);
        if let Some([b, _, _]) = &self.bias {
            for_chunks(&mut bcu, 3 * h, t > 1, |_, r| add_inplace(r, b));
        }
        // The conv inputs B * u, after the state's rows.
        let mut u = vec![0f32; (keep + t) * h];
        u[..keep * h].copy_from_slice(state);
        for_chunks(&mut u[keep * h..], h, t > 1, |ti, row| {
            let r = &bcu[ti * 3 * h..(ti + 1) * 3 * h];
            for (d, o) in row.iter_mut().enumerate() {
                *o = r[d] * r[2 * h + d];
            }
        });
        state.copy_from_slice(&u[t * h..]);
        if let Some(o) = inputs {
            o.clear();
            o.extend_from_slice(&u[keep * h..]);
        }
        let cb = self.bias.as_ref().map(|b| &b[1]);
        let mut g = vec![0f32; t * h];
        for_chunks(&mut g, h, t > 1, |ti, row| {
            let c = &bcu[ti * 3 * h + h..ti * 3 * h + 2 * h];
            for (d, o) in row.iter_mut().enumerate() {
                let w = &self.conv_w[d * k..(d + 1) * k];
                let mut acc = cb.map_or(0.0, |b| b[d]);
                for (j, &wj) in w.iter().enumerate() {
                    acc += wj * u[(ti + j) * h + d];
                }
                *o = c[d] * acc;
            }
        });
        linear(&self.out_proj, h, h, &g, t, y);
        if let Some([_, _, b]) = &self.bias {
            for_chunks(y, h, t > 1, |_, r| add_inplace(r, b));
        }
    }
}

/// The MLP of an LFM3 layer (`feed_forward.w1` / `w3` / `w2`: gate, up, down).
pub(crate) fn mlp<'a>(ld: &Loader<'a, '_>, p: &str) -> Result<Mlp<'a>> {
    let (h, inter) = (ld.cfg.hidden_size, ld.cfg.intermediate_size);
    Ok(Mlp {
        gate: ld.lin(&format!("{p}w1.weight"), inter, h)?,
        up: ld.lin(&format!("{p}w3.weight"), inter, h)?,
        down: ld.lin(&format!("{p}w2.weight"), h, inter)?,
        inter,
        act: crate::moe::Act::Silu,
    })
}

/// `cos` / `sin` rows (`[t, inv_freq.len()]`) of positions `start..start + t`.
pub(crate) fn rope_cs(inv_freq: &[f32], start: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0f32; t * half];
    let mut sin = vec![0f32; t * half];
    for ti in 0..t {
        let p = (start + ti) as f32;
        for (i, &f) in inv_freq.iter().enumerate() {
            let (s, c) = (p * f).sin_cos();
            cos[ti * half + i] = c;
            sin[ti * half + i] = s;
        }
    }
    (cos, sin)
}

/// Sets what the Qwen loader needs of an LFM3 config.
pub(crate) fn configure(cfg: &mut QwenConfig, raw: &serde_json::Value) -> Result<()> {
    let lf = Lfm3Config::from_json(raw)?;
    if lf.kernel < 2 {
        bail!("LFM3 conv_L_cache {} is not supported", lf.kernel);
    }
    if let Some(e) = raw.get("norm_eps").and_then(|v| v.as_f64()) {
        cfg.rms_norm_eps = e;
    }
    if raw.get("head_dim").is_none_or(|v| !v.is_u64()) {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads.max(1);
    }
    if cfg.layer_types.is_none() {
        cfg.layer_types = Some(Lfm3Config::default_layer_types(cfg.num_hidden_layers));
    }
    if let Some(t) = cfg
        .layer_types
        .as_ref()
        .unwrap()
        .iter()
        .find(|t| !matches!(t.as_str(), "conv" | "full_attention" | "sliding_attention"))
    {
        bail!("LFM3 layer type {t:?} is not supported (conv, full_attention, sliding_attention)");
    }
    if let Some(t) = raw.get("tie_embedding").and_then(|v| v.as_bool()) {
        cfg.tie_word_embeddings = t;
    } else if raw.get("tie_word_embeddings").is_none() {
        cfg.tie_word_embeddings = true;
    }
    cfg.attn_output_gate = false;
    cfg.zero_centered_norm = false;
    cfg.lfm3 = Some(lf);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_rope_zeroes_the_tail() {
        let r = Lfm3Rope {
            rope_theta: 1_000_000.0,
            rope_type: None,
            partial_rotary_factor: Some(0.5),
        };
        let f = r.inv_freq(128);
        assert_eq!(f.len(), 64);
        assert_eq!(f[0], 1.0);
        assert!((f[31] as f64 - 1.0 / 1_000_000f64.powf(62.0 / 128.0)).abs() < 1e-9);
        assert!(f[32..].iter().all(|&v| v == 0.0));
        let full = Lfm3Rope {
            partial_rotary_factor: None,
            ..r
        };
        assert!(full.inv_freq(128).iter().all(|&v| v > 0.0));
    }

    #[test]
    fn default_layer_pattern() {
        let t = Lfm3Config::default_layer_types(9);
        let want = [
            "conv",
            "conv",
            "full_attention",
            "conv",
            "sliding_attention",
            "conv",
            "full_attention",
            "conv",
            "sliding_attention",
        ];
        assert_eq!(t, want);
    }
}
