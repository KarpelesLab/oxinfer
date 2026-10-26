//! NomicBERT encoders (`model_type` nomic_bert: nomic-embed-text v1 / v1.5, whose modeling code
//! lives in `nomic-ai/nomic-bert-2048`): a BERT variant with rotary positions on a fused Wqkv, a
//! gated MLP (`fc2(fc11(x) * act(fc12(x)))`), unbiased projections, LayerNorm after the embeddings
//! and post-norm (or pre-norm) blocks, bidirectional attention.
//!
//! Beyond `max_trained_positions` tokens, a `rotary_scaling_factor` turns on dynamic NTK scaling:
//! the RoPE base grows with the sequence length. The reference keeps the scaled frequencies and its
//! cos/sin cache across calls, so a short text encoded after a long one in the same process sees
//! the long text's tables; here every sequence gets the tables of its own length (what the
//! reference computes in a fresh process).

use crate::bert::{LayerNorm, gelu_erf};
use crate::kernels::*;
use crate::model::{LayerCache, Lin, Quant, Table, attention_masked, linear, linear_multi, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mlp {
    /// `fc2(fc11(x) * silu(fc12(x)))`
    SwiGlu,
    /// `fc2(fc11(x) * gelu(fc12(x)))`
    GeGlu,
    /// `fc2(fc11(x) * sigmoid(fc12(x)))`
    Glu,
    /// `fc2(gelu(fc1(x)))` (exact GELU whatever the variant's name: the reference passes `F.gelu`).
    Gelu,
}

/// The fields of `NomicBertConfig` (a `GPT2Config`) that matter for inference, with its defaults.
#[derive(Debug, Clone)]
pub struct NomicBertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub mlp: Mlp,
    pub layer_norm_eps: f64,
    pub prenorm: bool,
    pub qkv_bias: bool,
    pub fc1_bias: bool,
    pub fc2_bias: bool,
    /// Rotated dimensions per head (0: learned absolute positions instead).
    pub rotary_dim: usize,
    pub rotary_base: f64,
    pub rotary_interleaved: bool,
    pub rotary_scaling_factor: Option<f64>,
    pub max_trained_positions: usize,
}

impl NomicBertConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let c: Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("parse NomicBERT config")?;
        // GPT2Config names (n_embd, ...) first, then the BERT-style aliases.
        let int = |keys: &[&str]| keys.iter().find_map(|k| c[*k].as_u64()).map(|v| v as usize);
        let flag = |k: &str, d: bool| c[k].as_bool().unwrap_or(d);
        let num = |k: &str| c[k].as_f64();
        for k in ["parallel_block", "use_rms_norm", "rotary_head_dim", "norm_mlp"] {
            if flag(k, false) {
                bail!("NomicBERT with {k} is not supported");
            }
        }
        if int(&["moe_every_n_layers"]).unwrap_or(0) > 0 {
            bail!("NomicBERT mixture-of-experts layers are not supported");
        }
        if !c["rotary_emb_scale_base"].is_null() {
            bail!("NomicBERT rotary_emb_scale_base (xPos) is not supported");
        }
        let hidden_size = int(&["n_embd", "hidden_size"]).context("config: no n_embd")?;
        let num_attention_heads = int(&["n_head", "num_attention_heads"]).context("config: no n_head")?;
        let act = c["activation_function"].as_str().unwrap_or("gelu_new");
        let mlp = match act {
            "swiglu" => Mlp::SwiGlu,
            "geglu" => Mlp::GeGlu,
            "glu" => Mlp::Glu,
            "gelu" | "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" => Mlp::Gelu,
            a => bail!("NomicBERT activation_function {a} is not supported"),
        };
        let mut intermediate_size = int(&["n_inner", "intermediate_size"]).unwrap_or(4 * hidden_size);
        if mlp != Mlp::Gelu {
            intermediate_size = intermediate_size.div_ceil(256) * 256; // NomciBertGatedMLP's multiple_of
        }
        let hd = hidden_size / num_attention_heads;
        let rotary_dim = (hd as f64 * num("rotary_emb_fraction").unwrap_or(0.0)) as usize;
        Ok(Self {
            hidden_size,
            num_hidden_layers: int(&["n_layer", "num_hidden_layers"]).context("config: no n_layer")?,
            num_attention_heads,
            intermediate_size,
            mlp,
            layer_norm_eps: num("layer_norm_epsilon").unwrap_or(1e-5),
            prenorm: flag("prenorm", false),
            qkv_bias: flag("qkv_proj_bias", true),
            fc1_bias: flag("mlp_fc1_bias", true),
            fc2_bias: flag("mlp_fc2_bias", true),
            rotary_dim,
            rotary_base: num("rotary_emb_base").unwrap_or(10_000.0),
            rotary_interleaved: flag("rotary_emb_interleaved", false),
            // The reference tests the factor for truthiness: 0 and null both mean off.
            rotary_scaling_factor: num("rotary_scaling_factor").filter(|&f| f != 0.0),
            max_trained_positions: int(&["max_trained_positions"]).unwrap_or(2048),
        })
    }

    /// RoPE inverse frequencies for a sequence of `t` tokens (dynamic NTK past the trained length).
    fn inv_freq(&self, t: usize) -> Vec<f32> {
        let (dim, max) = (self.rotary_dim as f64, self.max_trained_positions);
        let base = match self.rotary_scaling_factor {
            Some(f) if t > max => self.rotary_base * (f * t as f64 / max as f64 - (f - 1.0)).powf(dim / (dim - 2.0)),
            _ => self.rotary_base,
        };
        rope_inv_freq(self.rotary_dim, base, 1.0)
    }
}

/// A linear layer, biased or not.
struct Proj<'a> {
    w: Lin<'a>,
    b: Option<Vec<f32>>,
    n_out: usize,
    n_in: usize,
}

impl<'a> Proj<'a> {
    fn load(files: &'a ModelFiles, p: &str, n_out: usize, n_in: usize, bias: bool, q: Quant) -> Result<Self> {
        Ok(Self {
            w: load_lin(files, &format!("{p}.weight"), n_out, n_in, q)?,
            b: if bias { Some(files.f32(&format!("{p}.bias"))?) } else { None },
            n_out,
            n_in,
        })
    }
    fn add_bias(&self, y: &mut [f32]) {
        if let Some(b) = &self.b {
            for_chunks(y, self.n_out, y.len() > 4 * self.n_out, |_, r| add_inplace(r, b));
        }
    }
    fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        linear(&self.w, self.n_out, self.n_in, x, t, y);
        self.add_bias(y);
    }
}

struct NomicLayer<'a> {
    wqkv: Proj<'a>,
    out: Proj<'a>,
    norm1: LayerNorm,
    /// fc11 (or fc1 for the plain GELU MLP).
    up: Proj<'a>,
    /// fc12, the gated half (gated MLPs only).
    gate: Option<Proj<'a>>,
    down: Proj<'a>,
    norm2: LayerNorm,
}

pub struct NomicBertModel<'a> {
    pub cfg: NomicBertConfig,
    pub quant: Quant,
    word: Table<'a>,
    /// Learned absolute positions, when the model has no rotary embeddings.
    pos: Option<(Table<'a>, usize)>,
    token_type: Vec<f32>,
    emb_ln: LayerNorm,
    layers: Vec<NomicLayer<'a>>,
}

impl<'a> NomicBertModel<'a> {
    pub fn load(cfg: NomicBertConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let prefix = ["", "bert.", "model."]
            .into_iter()
            .find(|p| files.has(&format!("{p}embeddings.word_embeddings.weight")))
            .context("cannot find embeddings.word_embeddings in checkpoint")?;
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let n = |s: &str| format!("{prefix}{s}");
        // Rows from the tensor: the config's vocab_size may be rounded up to pad_vocab_size_multiple.
        let rows = |name: &str| files.info(name).and_then(|i| i.shape.first().copied()).unwrap_or(0);
        let wname = n("embeddings.word_embeddings.weight");
        let word = Table::load(files, &wname, rows(&wname), h, Quant::Bf16)?;
        let pname = n("embeddings.position_embeddings.weight");
        let pos = if cfg.rotary_dim == 0 && files.has(&pname) {
            Some((Table::load(files, &pname, rows(&pname), h, Quant::Bf16)?, rows(&pname)))
        } else {
            None
        };
        let token_type = if files.has(&n("embeddings.token_type_embeddings.weight")) {
            files.f32(&n("embeddings.token_type_embeddings.weight"))?[..h].to_vec()
        } else {
            vec![0.0; h]
        };
        let emb_ln = LayerNorm::load(files, &n("emb_ln"), h)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = |s: &str| n(&format!("encoder.layers.{l}.{s}"));
            let d = |s: &str, o: usize, i: usize, bias: bool| Proj::load(files, &p(s), o, i, bias, quant);
            let gated = cfg.mlp != Mlp::Gelu;
            layers.push(NomicLayer {
                wqkv: d("attn.Wqkv", 3 * h, h, cfg.qkv_bias)?,
                out: d("attn.out_proj", h, h, cfg.qkv_bias)?,
                norm1: LayerNorm::load(files, &p("norm1"), h)?,
                up: d(if gated { "mlp.fc11" } else { "mlp.fc1" }, inter, h, cfg.fc1_bias)?,
                gate: if gated {
                    Some(d("mlp.fc12", inter, h, cfg.fc1_bias)?)
                } else {
                    None
                },
                down: d("mlp.fc2", h, inter, cfg.fc2_bias)?,
                norm2: LayerNorm::load(files, &p("norm2"), h)?,
            });
        }
        Ok(Self {
            cfg,
            quant,
            word,
            pos,
            token_type,
            emb_ln,
            layers,
        })
    }

    /// Last hidden states `[t, hidden]` for one sequence (special tokens included).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, nh, inter) = (cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size);
        let hd = h / nh;
        let t = self.pos.as_ref().map_or(tokens.len(), |(_, n)| tokens.len().min(*n));
        let tokens = &tokens[..t];
        let eps = cfg.layer_norm_eps as f32;
        let mut x = vec![0f32; t * h];
        let mut pe = vec![0f32; h];
        for (i, row) in x.chunks_mut(h).enumerate() {
            self.word.row(tokens[i] as usize, row);
            add_inplace(row, &self.token_type);
            if let Some((pos, _)) = &self.pos {
                pos.row(i, &mut pe);
                add_inplace(row, &pe);
            }
        }
        self.emb_ln.apply_rows(&mut x, eps);

        let rot = cfg.rotary_dim;
        let positions: Vec<usize> = (0..t).collect();
        let (cos, sin) = if rot > 0 {
            rope_tables(&positions, &cfg.inv_freq(t))
        } else {
            (Vec::new(), Vec::new())
        };
        let scale = (hd as f32).powf(-0.5);
        let mut qkv = vec![0f32; t * 3 * h];
        let (mut q, mut k, mut v) = (vec![0f32; t * h], vec![0f32; t * h], vec![0f32; t * h]);
        let mut att = vec![0f32; t * h];
        let mut o = vec![0f32; t * h];
        let (mut u, mut g) = (vec![0f32; t * inter], vec![0f32; t * inter]);
        // Pre-norm keeps the residual stream apart; the blocks then return the MLP output alone.
        let mut residual: Option<Vec<f32>> = None;
        for layer in &self.layers {
            if cfg.prenorm {
                let r = match residual.take() {
                    Some(mut r) => {
                        add_inplace(&mut r, &x);
                        r
                    }
                    None => x.clone(),
                };
                x.copy_from_slice(&r);
                layer.norm1.apply_rows(&mut x, eps);
                residual = Some(r);
            }

            // Attention: Wqkv rows are (three, head, dim).
            layer.wqkv.forward(&x, t, &mut qkv);
            for i in 0..t {
                let row = &qkv[i * 3 * h..(i + 1) * 3 * h];
                q[i * h..(i + 1) * h].copy_from_slice(&row[..h]);
                k[i * h..(i + 1) * h].copy_from_slice(&row[h..2 * h]);
                v[i * h..(i + 1) * h].copy_from_slice(&row[2 * h..]);
            }
            if rot > 0 {
                let half = rot / 2;
                for buf in [&mut q, &mut k] {
                    for_chunks(buf, h, t > 4, |i, r| {
                        let (c, s) = (&cos[i * half..(i + 1) * half], &sin[i * half..(i + 1) * half]);
                        for head in r.chunks_mut(hd) {
                            if cfg.rotary_interleaved {
                                apply_rope_interleaved(&mut head[..rot], c, s);
                            } else {
                                apply_rope(&mut head[..rot], c, s);
                            }
                        }
                    });
                }
            }
            scale_inplace(&mut q, scale);
            let mut cache = LayerCache::new(nh, hd);
            cache.push(&k, &v, t);
            attention_masked(&q, &cache, nh, t, 0, None, false, &mut att);
            layer.out.forward(&att, t, &mut o);

            if let Some(r) = residual.as_mut() {
                add_inplace(r, &o);
                x.copy_from_slice(r);
                layer.norm2.apply_rows(&mut x, eps);
            } else {
                add_inplace(&mut x, &o);
                layer.norm1.apply_rows(&mut x, eps);
            }

            // MLP.
            let par = g.len() > 8192;
            match &layer.gate {
                Some(gate) => {
                    linear_multi(&mut [(&layer.up.w, inter, &mut u[..]), (&gate.w, inter, &mut g[..])], h, &x, t);
                    layer.up.add_bias(&mut u);
                    gate.add_bias(&mut g);
                    match cfg.mlp {
                        Mlp::SwiGlu => silu_mul(&mut g, &u),
                        mode => {
                            for_chunks(&mut g, 4096, par, |ci, c| {
                                for (j, e) in c.iter_mut().enumerate() {
                                    let a = if mode == Mlp::GeGlu {
                                        gelu_erf(*e)
                                    } else {
                                        1.0 / (1.0 + (-*e).exp())
                                    };
                                    *e = a * u[ci * 4096 + j];
                                }
                            });
                        }
                    }
                }
                None => {
                    layer.up.forward(&x, t, &mut g);
                    for_chunks(&mut g, 4096, par, |_, c| {
                        for e in c.iter_mut() {
                            *e = gelu_erf(*e);
                        }
                    });
                }
            }
            layer.down.forward(&g, t, &mut o);

            if cfg.prenorm {
                x.copy_from_slice(&o);
            } else {
                add_inplace(&mut x, &o);
                layer.norm2.apply_rows(&mut x, eps);
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(factor: Option<f64>) -> NomicBertConfig {
        NomicBertConfig {
            hidden_size: 768,
            num_hidden_layers: 1,
            num_attention_heads: 12,
            intermediate_size: 3072,
            mlp: Mlp::SwiGlu,
            layer_norm_eps: 1e-12,
            prenorm: false,
            qkv_bias: false,
            fc1_bias: false,
            fc2_bias: false,
            rotary_dim: 64,
            rotary_base: 1000.0,
            rotary_interleaved: false,
            rotary_scaling_factor: factor,
            max_trained_positions: 2048,
        }
    }

    #[test]
    fn dynamic_ntk_scales_the_base_past_the_trained_length() {
        let plain = rope_inv_freq(64, 1000.0, 1.0);
        assert_eq!(cfg(Some(2.0)).inv_freq(2048), plain);
        assert_eq!(cfg(None).inv_freq(4096), plain);
        // 4096 tokens at factor 2: base * (2 * 4096 / 2048 - 1) ^ (64 / 62).
        let base = 1000.0 * 3f64.powf(64.0 / 62.0);
        assert_eq!(cfg(Some(2.0)).inv_freq(4096), rope_inv_freq(64, base, 1.0));
    }

    #[test]
    fn interleaved_rope_rotates_adjacent_pairs() {
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        let (c, s) = ([0.0f32, 1.0], [1.0f32, 0.0]);
        apply_rope_interleaved(&mut v, &c, &s);
        assert_eq!(v, [-2.0, 1.0, 3.0, 4.0]);
    }
}
