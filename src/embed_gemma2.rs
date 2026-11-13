//! EmbeddingGemma 2 (`model_type` embedding_gemma2, google/embeddinggemma-2): a bidirectional
//! Gemma-4-style encoder whose output is projected to the embedding size per token.
//!
//! ```text
//! x   = embed(ids) * sqrt(H)                      (or soft tokens of an image / audio clip)
//! ple = norm(per_layer_model_projection(x) / sqrt(H))    [layers, P] per token
//! layer i:
//!   x += post_attention_norm(attn(input_norm(x)))        q, k normed per head, v normed without
//!                                                        scale, RoPE, scores unscaled
//!   x += post_feedforward_norm(down(gelu(gate(n)) * up(n))),  n = pre_feedforward_norm(x)
//!   x += post_per_layer_input_norm(per_layer_projection(gelu(per_layer_input_gate(x)) * ple[i]))
//!   x *= layer_scalar
//! out = embedding_projection(norm(x))                    [t, embedding_dim]
//! ```
//!
//! Attention is bidirectional. `layer_types` names each layer `sliding_attention` (keys at most
//! `sliding_window` positions away, on either side) or `full_attention`; the two kinds have their
//! own RoPE base, and `per_layer_config` gives some layers another head size and number of
//! key / value heads (the full-attention ones: one KV head of 512). Unlike Gemma 4's per-layer
//! embeddings, the per-layer input is a projection of the input embeddings alone (no table).

use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, attention_masked, linear, linear_multi, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct EmbGemma2Config {
    /// Width of the encoder (512).
    pub width: usize,
    /// Width of the output (`embedding_dim`: 768); what the pooling sees.
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    /// Width of the per-layer input (`hidden_size_per_layer_input`).
    pub per_layer: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub pad_token_id: u32,
    /// Per layer: sliding attention, head size, key / value heads.
    pub layers: Vec<(bool, usize, usize)>,
    pub theta_full: f64,
    pub theta_sliding: f64,
    /// The placeholder tokens of soft tokens (`<|image|>`, `<|audio|>`).
    pub image_token_id: Option<u32>,
    pub audio_token_id: Option<u32>,
}

impl EmbGemma2Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let root: Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse EmbeddingGemma 2 config")?;
        Self::from_json(&root)
    }

    /// From a `config.json` (its `text_config`, and the placeholder token ids of the root).
    pub fn from_json(root: &Value) -> Result<Self> {
        let c = root.get("text_config").unwrap_or(root);
        let int = |k: &str| c[k].as_u64().map(|v| v as usize);
        let need = |k: &str| int(k).with_context(|| format!("EmbeddingGemma 2 config has no {k}"));
        let (width, n, nh) = (need("hidden_size")?, need("num_hidden_layers")?, need("num_attention_heads")?);
        let head_dim = int("head_dim").unwrap_or(width / nh.max(1));
        let nkv = int("num_key_value_heads").unwrap_or(nh);
        let types: Vec<String> = match c["layer_types"].as_array() {
            Some(a) => a.iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect(),
            None => vec!["sliding_attention".to_string(); n],
        };
        if types.len() != n {
            bail!("layer_types has {} entries for {n} layers", types.len());
        }
        let mut layers = Vec::with_capacity(n);
        for (l, ty) in types.iter().enumerate() {
            let sliding = match ty.as_str() {
                "sliding_attention" => true,
                "full_attention" => false,
                other => bail!("EmbeddingGemma 2 layer type {other:?} is not supported"),
            };
            // Overrides are keyed by the layer index (zero-padded) or by the layer type.
            let over = c["per_layer_config"].as_object().and_then(|o| {
                o.iter()
                    .find(|(k, _)| k.parse::<usize>().ok() == Some(l) || k.as_str() == ty)
                    .map(|(_, v)| v)
            });
            let get = |k: &str, d: usize| over.and_then(|o| o[k].as_u64()).map_or(d, |v| v as usize);
            let (hd, kv) = (get("head_dim", head_dim), get("num_key_value_heads", nkv));
            if kv == 0 || nh % kv != 0 || hd % 2 != 0 {
                bail!("layer {l}: {nh} heads of {hd} over {kv} key / value heads");
            }
            layers.push((sliding, hd, kv));
        }
        let theta = |kind: &str, d: f64| -> Result<f64> {
            let r = &c["rope_parameters"][kind];
            if r["rope_type"].as_str().is_some_and(|t| t != "default") {
                bail!("EmbeddingGemma 2 {kind} RoPE type {} is not supported", r["rope_type"]);
            }
            Ok(r["rope_theta"].as_f64().unwrap_or(d))
        };
        let tok = |k: &str| root[k].as_u64().map(|v| v as u32);
        Ok(EmbGemma2Config {
            width,
            hidden_size: int("embedding_dim").unwrap_or(width),
            num_hidden_layers: n,
            num_attention_heads: nh,
            intermediate_size: need("intermediate_size")?,
            per_layer: int("hidden_size_per_layer_input").unwrap_or(width),
            rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            vocab_size: need("vocab_size")?,
            sliding_window: int("sliding_window").unwrap_or(512),
            pad_token_id: int("pad_token_id").unwrap_or(0) as u32,
            layers,
            theta_full: theta("full_attention", 1_000_000.0)?,
            theta_sliding: theta("sliding_attention", 10_000.0)?,
            image_token_id: tok("image_token_id"),
            audio_token_id: tok("audio_token_id"),
        })
    }
}

struct Layer<'a> {
    q: Lin<'a>,
    k: Lin<'a>,
    v: Lin<'a>,
    o: Lin<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ln_in: Vec<f32>,
    ln_attn: Vec<f32>,
    ln_ff_in: Vec<f32>,
    ln_ff: Vec<f32>,
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
    ple_gate: Lin<'a>, // [P, H]
    ple_proj: Lin<'a>, // [H, P]
    ple_norm: Vec<f32>,
    scalar: f32,
}

pub struct EmbGemma2Model<'a> {
    pub cfg: EmbGemma2Config,
    embed: Table<'a>,
    layers: Vec<Layer<'a>>,
    norm: Vec<f32>,
    /// `[layers * P, H]`: every layer's per-layer input from the input embeddings.
    ple: Lin<'a>,
    ple_norm: Vec<f32>,
    /// `[embedding_dim, H]`.
    proj: Lin<'a>,
}

impl<'a> EmbGemma2Model<'a> {
    pub fn load(cfg: EmbGemma2Config, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let prefix = ["language_model.", "model.language_model.", ""]
            .into_iter()
            .find(|p| files.has(&format!("{p}embed_tokens.weight")))
            .context("cannot find language_model.embed_tokens in checkpoint")?;
        let (h, nh, inter, p) = (cfg.width, cfg.num_attention_heads, cfg.intermediate_size, cfg.per_layer);
        let vec = |name: String, n: usize| -> Result<Vec<f32>> {
            let v = files.f32(&name)?;
            if v.len() != n {
                bail!("tensor {name} has {} values, expected {n}", v.len());
            }
            Ok(v)
        };
        let lin = |name: String, n_out: usize, n_in: usize| load_lin(files, &name, n_out, n_in, quant);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (l, &(_, hd, kv)) in cfg.layers.iter().enumerate() {
            let lp = format!("{prefix}layers.{l}.");
            let n = |s: &str| format!("{lp}{s}");
            layers.push(Layer {
                q: lin(n("self_attn.q_proj.weight"), nh * hd, h)?,
                k: lin(n("self_attn.k_proj.weight"), kv * hd, h)?,
                v: lin(n("self_attn.v_proj.weight"), kv * hd, h)?,
                o: lin(n("self_attn.o_proj.weight"), h, nh * hd)?,
                q_norm: vec(n("self_attn.q_norm.weight"), hd)?,
                k_norm: vec(n("self_attn.k_norm.weight"), hd)?,
                ln_in: vec(n("input_layernorm.weight"), h)?,
                ln_attn: vec(n("post_attention_layernorm.weight"), h)?,
                ln_ff_in: vec(n("pre_feedforward_layernorm.weight"), h)?,
                ln_ff: vec(n("post_feedforward_layernorm.weight"), h)?,
                gate: lin(n("mlp.gate_proj.weight"), inter, h)?,
                up: lin(n("mlp.up_proj.weight"), inter, h)?,
                down: lin(n("mlp.down_proj.weight"), h, inter)?,
                ple_gate: lin(n("ple_block.per_layer_input_gate.weight"), p, h)?,
                ple_proj: lin(n("ple_block.per_layer_projection.weight"), h, p)?,
                ple_norm: vec(n("ple_block.post_per_layer_input_norm.weight"), h)?,
                // A buffer of ones in checkpoints that do not store it.
                scalar: files.f32(&n("layer_scalar")).ok().and_then(|v| v.first().copied()).unwrap_or(1.0),
            });
        }
        Ok(Self {
            embed: Table::load(files, &format!("{prefix}embed_tokens.weight"), cfg.vocab_size, h, Quant::Bf16)?,
            layers,
            norm: vec(format!("{prefix}norm.weight"), h)?,
            ple: lin(
                format!("{prefix}ple.per_layer_model_projection.weight"),
                cfg.num_hidden_layers * p,
                h,
            )?,
            ple_norm: vec(format!("{prefix}ple.per_layer_projection_norm.weight"), p)?,
            proj: lin(format!("{prefix}embedding_projection.weight"), cfg.hidden_size, h)?,
            cfg,
        })
    }

    /// The scaled input embeddings `[t, width]` of `tokens` (placeholders of soft tokens embed
    /// as the padding token; their rows are replaced by the caller).
    pub fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.width;
        let scale = (h as f32).sqrt();
        let placeholder = |t: u32| Some(t) == self.cfg.image_token_id || Some(t) == self.cfg.audio_token_id;
        let mut x = vec![0f32; tokens.len() * h];
        for_chunks(&mut x, h, tokens.len() > 1, |i, row| {
            let t = if placeholder(tokens[i]) { self.cfg.pad_token_id } else { tokens[i] };
            self.embed.row(t as usize, row);
            row.iter_mut().for_each(|v| *v *= scale);
        });
        x
    }

    /// Per-token outputs `[t, embedding_dim]` of `tokens`.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        self.forward_embeds(self.embed(tokens))
    }

    /// [`Self::forward`] from input embeddings `[t, width]` ([`Self::embed`], with soft tokens).
    pub fn forward_embeds(&self, mut x: Vec<f32>) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, inter, p, n) = (cfg.width, cfg.intermediate_size, cfg.per_layer, cfg.num_hidden_layers);
        let t = x.len() / h;
        let eps = cfg.rms_norm_eps as f32;
        let par = t > 1;

        // The per-layer inputs: [t, layers, P], each slice normed.
        let mut ple = vec![0f32; t * n * p];
        linear(&self.ple, n * p, h, &x, t, &mut ple);
        let s = (h as f32).powf(-0.5);
        for_chunks(&mut ple, p, par, |_, r| {
            r.iter_mut().for_each(|v| *v *= s);
            rmsnorm_f32w_inplace(r, &self.ple_norm, eps);
        });

        let mut hn = vec![0f32; t * h];
        let mut y = vec![0f32; t * h];
        let mut g = vec![0f32; t * inter];
        let mut u = vec![0f32; t * inter];
        let mut pg = vec![0f32; t * p];
        for (l, layer) in self.layers.iter().enumerate() {
            let (sliding, hd, kv) = cfg.layers[l];
            rmsnorm_f32w_rows(&x, h, &layer.ln_in, eps, &mut hn);
            self.attention(layer, sliding, hd, kv, &hn, t, &mut y);
            rmsnorm_f32w_rows_inplace(&mut y, h, &layer.ln_attn, eps);
            add_inplace(&mut x, &y);

            rmsnorm_f32w_rows(&x, h, &layer.ln_ff_in, eps, &mut hn);
            linear_multi(&mut [(&layer.gate, inter, &mut g[..]), (&layer.up, inter, &mut u[..])], h, &hn, t);
            gelu_mul(&mut g, &u);
            linear(&layer.down, h, inter, &g, t, &mut y);
            rmsnorm_f32w_rows_inplace(&mut y, h, &layer.ln_ff, eps);
            add_inplace(&mut x, &y);

            // The per-layer input, gated by the residual stream.
            linear(&layer.ple_gate, p, h, &x, t, &mut pg);
            for_chunks(&mut pg, p, par, |ti, r| {
                let e = &ple[(ti * n + l) * p..(ti * n + l + 1) * p];
                for (v, &e) in r.iter_mut().zip(e) {
                    *v = gelu_tanh(*v) * e;
                }
            });
            linear(&layer.ple_proj, h, p, &pg, t, &mut y);
            rmsnorm_f32w_rows_inplace(&mut y, h, &layer.ple_norm, eps);
            add_inplace(&mut x, &y);
            if layer.scalar != 1.0 {
                x.iter_mut().for_each(|v| *v *= layer.scalar);
            }
        }
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.norm, eps);
        let mut out = vec![0f32; t * cfg.hidden_size];
        linear(&self.proj, cfg.hidden_size, h, &x, t, &mut out);
        out
    }

    /// Bidirectional attention of one layer over the `t` normed rows `hn`, through `o_proj`.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, layer: &Layer, sliding: bool, hd: usize, kv: usize, hn: &[f32], t: usize, y: &mut [f32]) {
        let cfg = &self.cfg;
        let (h, nh) = (cfg.width, cfg.num_attention_heads);
        let eps = cfg.rms_norm_eps as f32;
        let par = t > 1;
        let mut q = vec![0f32; t * nh * hd];
        let mut k = vec![0f32; t * kv * hd];
        let mut v = vec![0f32; t * kv * hd];
        linear_multi(
            &mut [
                (&layer.q, nh * hd, &mut q[..]),
                (&layer.k, kv * hd, &mut k[..]),
                (&layer.v, kv * hd, &mut v[..]),
            ],
            h,
            hn,
            t,
        );
        // RoPE over the whole head, with the base of the layer's kind.
        let half = hd / 2;
        let theta = if sliding { cfg.theta_sliding } else { cfg.theta_full };
        let inv_freq: Vec<f32> = (0..half).map(|i| (1.0 / theta.powf((2 * i) as f64 / hd as f64)) as f32).collect();
        let mut cos = vec![0f32; t * half];
        let mut sin = vec![0f32; t * half];
        for ti in 0..t {
            for (i, &f) in inv_freq.iter().enumerate() {
                let (s, c) = (ti as f32 * f).sin_cos();
                cos[ti * half + i] = c;
                sin[ti * half + i] = s;
            }
        }
        for_chunks(&mut q, hd, par, |i, r| {
            let ti = i / nh;
            rmsnorm_f32w_inplace(r, &layer.q_norm, eps);
            apply_rope(r, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
        });
        for_chunks(&mut k, hd, par, |i, r| {
            let ti = i / kv;
            rmsnorm_f32w_inplace(r, &layer.k_norm, eps);
            apply_rope(r, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
        });
        let ones = vec![1f32; hd];
        for_chunks(&mut v, hd, par, |_, r| rmsnorm_f32w_inplace(r, &ones, eps));
        let mut cache = LayerCache::new(kv, hd);
        cache.push(&k, &v, t);
        // Keys at most `sliding_window` positions away (the kernel's window is exclusive).
        let window = sliding.then_some(cfg.sliding_window + 1);
        let mut att = vec![0f32; t * nh * hd];
        attention_masked(&q, &cache, nh, t, 0, window, false, &mut att);
        linear(&layer.o, h, nh * hd, &att, t, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_layer_shapes() {
        let root = serde_json::json!({
            "image_token_id": 258880, "audio_token_id": 258881,
            "text_config": {
                "hidden_size": 512, "embedding_dim": 768, "num_hidden_layers": 6, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 256, "intermediate_size": 2048, "vocab_size": 262144,
                "hidden_size_per_layer_input": 512, "sliding_window": 512,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "per_layer_config": {"05": {"head_dim": 512, "num_key_value_heads": 1}},
                "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}
            }
        });
        let c = EmbGemma2Config::from_json(&root).unwrap();
        assert_eq!((c.width, c.hidden_size, c.per_layer), (512, 768, 512));
        assert_eq!(c.layers[0], (true, 256, 2));
        assert_eq!(c.layers[5], (false, 512, 1));
        assert_eq!((c.theta_sliding, c.theta_full), (10_000.0, 1_000_000.0));
        assert_eq!((c.image_token_id, c.audio_token_id), (Some(258880), Some(258881)));
        let mut bad = root.clone();
        bad["text_config"]["layer_types"][0] = "conv".into();
        assert!(EmbGemma2Config::from_json(&bad).is_err());
    }
}
