//! Ripple (`ricursive`: Ricursive Intelligence's Ripple-65B-Preview, 80 layers of 5120;
//! `RicursiveForCausalLM`), as Ricursive's `modeling_ricursive.py`: a dense transformer in the
//! shape of Gemma 3 with a few additions.
//!
//! - The embeddings are multiplied by `embed_scale` (when the config has one).
//! - Blocks with four norms: `x += norm(attn(norm(x)))`, `x += norm(mlp(norm(x)))` (plain RMS
//!   norms, `w x rsqrt(mean(x^2) + eps)`).
//! - Grouped-query attention whose head count is free of the width (80 heads of 128 on 5120, 8
//!   key / value heads): queries and keys RMS-normed per head (one weight for all heads), scores
//!   scaled by `head_dim^-0.5`.
//! - Two kinds of layers (`layer_types`): `sliding_attention` rotates queries and keys (RoPE over
//!   the whole head, absolute positions) and sees the last `sliding_window` positions, itself
//!   included; `full_attention` sees everything and has no positions at all.
//! - Gated attention: the heads' outputs are multiplied by `sigmoid(g_proj(x))` before `o_proj`.
//! - Value residuals (`use_valres`): the values are projected from
//!   `lerp(e, x, sigmoid(value_res_lambda))`, `e` being the token's scaled embedding and `x` the
//!   layer's normed input, in every layer.
//! - A SwiGLU MLP with clamped activations (`ffn_clamp_threshold`): the gate at most the
//!   threshold, the other branch within plus and minus the threshold.
//!
//! The attention over the cache, its window and the rotary tables are Gemma's
//! ([`crate::model::attention`]). Checkpoints: Ricursive's safetensors layout, in bf16 or with
//! the MLPs in NVFP4 (ModelOpt's layout, as Ripple-65B-Preview-NVFP4; its `input_scale` tensors,
//! for quantized activations, are not used).

use crate::intervene::Target;
use crate::kernels::*;
use crate::kimi::load_lin_exact;
use crate::model::{LayerCache, Lin, Quant, Table, attention, attention_prefill, linear, linear_multi, table_quant};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d_eps() -> f64 {
    1e-6
}
fn d_theta() -> f64 {
    10000.0
}
fn d_window() -> Option<usize> {
    Some(1024)
}
fn d_true() -> bool {
    true
}
fn d_silu() -> String {
    "silu".into()
}
fn d_head() -> usize {
    128
}
fn d_kv() -> usize {
    8
}

/// `config.json` of a Ripple checkpoint (`RicursiveConfig`).
#[derive(Debug, Clone, Deserialize)]
pub struct RippleConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "d_kv")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_head")]
    pub head_dim: usize,
    #[serde(default = "d_silu")]
    pub hidden_act: String,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "d_theta")]
    pub rope_theta: f64,
    #[serde(default = "d_window")]
    pub sliding_window: Option<usize>,
    /// `sliding_attention` or `full_attention` per layer; without it every fifth layer is full.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "d_true")]
    pub use_gated_attention: bool,
    #[serde(default)]
    pub use_valres: bool,
    #[serde(default)]
    pub ffn_clamp_threshold: Option<f64>,
    #[serde(default)]
    pub embed_scale: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl RippleConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::from_json(serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?)
    }

    pub fn from_json(raw: serde_json::Value) -> Result<Self> {
        let c: RippleConfig = serde_json::from_value(raw).context("parse the Ripple config")?;
        if let Some(lt) = &c.layer_types {
            if lt.len() != c.num_hidden_layers {
                bail!("ripple: layer_types has {} entries for {} layers", lt.len(), c.num_hidden_layers);
            }
            if let Some(bad) = lt.iter().find(|t| !matches!(t.as_str(), "sliding_attention" | "full_attention")) {
                bail!("ripple: layer type {bad:?} is not supported");
            }
        }
        // (The modeling code has SiLU only, whatever `hidden_act` says.)
        if c.hidden_act != "silu" {
            bail!("ripple: activation {:?} is not supported", c.hidden_act);
        }
        if c.head_dim == 0 || !c.head_dim.is_multiple_of(2) || c.num_key_value_heads == 0 {
            bail!("ripple: bad attention shapes");
        }
        if !c.num_attention_heads.is_multiple_of(c.num_key_value_heads) {
            bail!("ripple: the heads are not a multiple of the key / value heads");
        }
        if c.sliding_window == Some(0) {
            bail!("ripple: sliding_window 0");
        }
        Ok(c)
    }

    /// Whether layer `l` is a `sliding_attention` layer (rotary positions, windowed).
    pub fn is_sliding(&self, l: usize) -> bool {
        match &self.layer_types {
            Some(lt) => lt[l] == "sliding_attention",
            None => l % 5 != 4,
        }
    }
    /// The clamp of the MLP's activations (none for a threshold of zero, as the modeling code).
    fn clamp(&self) -> Option<f32> {
        self.ffn_clamp_threshold.filter(|&t| t != 0.0).map(|t| t as f32)
    }
}

/// Weight streaming for a checkpoint that does not fit in memory (as Mistral Large 4).
pub fn default_stream() -> crate::stream::StreamOptions {
    crate::mistral4::default_stream()
}

/// Whether a checkpoint of `bytes` is too big to hold in memory.
pub fn too_big(bytes: u64) -> bool {
    crate::mistral4::too_big(bytes)
}

struct Layer<'a> {
    sliding: bool,
    ln_in: Vec<f32>,
    ln_post_attn: Vec<f32>,
    ln_pre_ffn: Vec<f32>,
    ln_post_ffn: Vec<f32>,
    q: Lin<'a>, // [nh * hd, H]
    k: Lin<'a>, // [n_kv * hd, H]
    v: Lin<'a>,
    q_norm: Vec<f32>,      // [hd]
    k_norm: Vec<f32>,      // [hd]
    gate: Option<Lin<'a>>, // [nh * hd, H]
    o: Lin<'a>,            // [H, nh * hd]
    /// `sigmoid(value_res_lambda)`: the share of the layer's own input in what the values are
    /// projected from.
    blend: Option<f32>,
    mlp_gate: Lin<'a>, // [inter, H]
    mlp_up: Lin<'a>,
    mlp_down: Lin<'a>, // [H, inter]
}

/// Decoding state: every layer's keys and values (sliding layers keep theirs beyond the window
/// too, so a session can be cut back to any position).
#[derive(Clone)]
pub struct RippleSession {
    caches: Vec<LayerCache>,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
}

pub struct RippleModel<'a> {
    pub cfg: RippleConfig,
    pub quant: Quant,
    embed: Table<'a>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    inv_freq: Vec<f32>,
    layers: Vec<Layer<'a>>,
    /// Weight streaming: unit `l` is layer `l`, unit `num_hidden_layers` the LM head.
    stream: Option<crate::stream::StreamHook>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl<'a> RippleModel<'a> {
    pub fn load(cfg: RippleConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let (h, v, nh, nkv, hd) = (
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let getf = |full: &str, n: usize| -> Result<Vec<f32>> {
            let info = files.info(full).with_context(|| format!("missing tensor {full}"))?;
            if info.shape.iter().product::<usize>() != n {
                bail!("tensor {full} has shape {:?}, expected {n} values", info.shape);
            }
            files.f32(full)
        };
        let lin = |full: &str, n_out: usize, n_in: usize| load_lin_exact(files, full, n_out, n_in, quant);
        let embed_name = "model.embed_tokens.weight";
        let tied = cfg.tie_word_embeddings || !files.has("lm_head.weight");
        let embed = Table::load(files, embed_name, v, h, if tied { quant } else { table_quant(quant) })?;
        let lm_head = lin(if tied { embed_name } else { "lm_head.weight" }, v, h)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{l}.");
            let ap = format!("{lp}self_attn.");
            let inter = cfg.intermediate_size;
            layers.push(Layer {
                sliding: cfg.is_sliding(l),
                ln_in: getf(&format!("{lp}input_layernorm.weight"), h)?,
                ln_post_attn: getf(&format!("{lp}post_attention_layernorm.weight"), h)?,
                ln_pre_ffn: getf(&format!("{lp}pre_feedforward_layernorm.weight"), h)?,
                ln_post_ffn: getf(&format!("{lp}post_feedforward_layernorm.weight"), h)?,
                q: lin(&format!("{ap}q_proj.weight"), nh * hd, h)?,
                k: lin(&format!("{ap}k_proj.weight"), nkv * hd, h)?,
                v: lin(&format!("{ap}v_proj.weight"), nkv * hd, h)?,
                q_norm: getf(&format!("{ap}q_norm.weight"), hd)?,
                k_norm: getf(&format!("{ap}k_norm.weight"), hd)?,
                gate: if cfg.use_gated_attention {
                    Some(lin(&format!("{ap}g_proj.weight"), nh * hd, h)?)
                } else {
                    None
                },
                o: lin(&format!("{ap}o_proj.weight"), h, nh * hd)?,
                blend: if cfg.use_valres {
                    Some(sigmoid(getf(&format!("{ap}value_res_lambda"), 1)?[0]))
                } else {
                    None
                },
                mlp_gate: lin(&format!("{lp}mlp.gate_proj.weight"), inter, h)?,
                mlp_up: lin(&format!("{lp}mlp.up_proj.weight"), inter, h)?,
                mlp_down: lin(&format!("{lp}mlp.down_proj.weight"), h, inter)?,
            });
        }
        Ok(Self {
            final_norm: getf("model.norm.weight", h)?,
            inv_freq: rope_inv_freq(hd, cfg.rope_theta, 1.0),
            cfg,
            quant,
            embed,
            lm_head,
            layers,
            stream: None,
        })
    }

    pub fn new_session(&self) -> RippleSession {
        let c = &self.cfg;
        RippleSession {
            caches: (0..c.num_hidden_layers)
                .map(|_| LayerCache::new(c.num_key_value_heads, c.head_dim))
                .collect(),
            pos: 0,
            iv: Default::default(),
        }
    }

    fn lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let mut v = vec![&l.q, &l.k, &l.v, &l.o, &l.mlp_gate, &l.mlp_up, &l.mlp_down];
        v.extend(l.gate.as_ref());
        v
    }

    /// Weight bytes read per decoded token.
    pub fn streamed_bytes(&self) -> usize {
        self.lm_head.bytes()
            + self
                .layers
                .iter()
                .map(|l| self.lins(l).iter().map(|w| w.bytes()).sum::<usize>())
                .sum::<usize>()
    }

    /// The streaming units in execution order (see `crate::stream`): the layers, the LM head.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        let unit = |name: String, ranges| crate::stream::UnitSpec {
            name,
            ranges,
            uses: 1.0,
            parts: Vec::new(),
        };
        let mut v: Vec<_> = self
            .layers
            .iter()
            .enumerate()
            .map(|(l, layer)| unit(format!("layer.{l}"), self.lins(layer).iter().flat_map(|w| w.mem_ranges()).collect()))
            .collect();
        v.push(unit("lm_head".into(), self.lm_head.mem_ranges()));
        v
    }

    /// Marks the weights of the units with `streamed[i]` set as streamed (no accelerator state
    /// kept between calls).
    pub fn mark_streamed(&self, streamed: &[bool]) {
        let on = |i: usize| streamed.get(i).copied().unwrap_or(false);
        for (l, layer) in self.layers.iter().enumerate() {
            for w in self.lins(layer) {
                w.set_transient_device(on(l));
            }
        }
        self.lm_head.set_transient_device(on(self.layers.len()));
    }

    /// Brackets every layer and the LM head with `hook`'s units (None stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// LM-head logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * c.vocab_size];
        let _unit = crate::stream::enter(&self.stream, self.layers.len());
        linear(&self.lm_head, c.vocab_size, c.hidden_size, hidden, t, &mut logits);
        logits
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut RippleSession, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos` (positions in `mm` take the given embeddings, before the
    /// scale, instead of the token's), appending to the caches; returns the final-normed hidden
    /// states `[t, hidden]`.
    pub fn forward_hidden_mm(&self, sess: &mut RippleSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, &[], &mut Vec::new())
    }

    /// [`Self::forward_hidden_mm`] that also writes to `feats` (`[t, layers.len() * hidden]`) the
    /// residual stream entering each of `layers`, `num_hidden_layers` standing for the last
    /// layer's output.
    pub fn forward_capture(
        &self,
        sess: &mut RippleSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let c = &self.cfg;
        let h = c.hidden_size;
        let start = sess.pos;
        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, t > 1, |ti, row| self.embed.row(tokens[ti] as usize, row));
        for &(i, emb) in mm {
            x[i * h..(i + 1) * h].copy_from_slice(emb);
        }
        if let Some(s) = c.embed_scale {
            let s = s as f32;
            for_chunks(&mut x, h, t > 1, |_, row| scale_inplace(row, s));
        }
        // What the value residuals read: the scaled embeddings, before any intervention.
        let seed = c.use_valres.then(|| x.clone());
        let positions: Vec<usize> = (start..start + t).collect();
        let rope = rope_tables(&positions, &self.inv_freq);
        drop(timer);
        let iv = sess.iv.active().then(|| sess.iv.clone());
        if let Some(iv) = &iv {
            iv.point(0, &mut x, h, start);
        }
        let nf = layers.len();
        feats.clear();
        feats.resize(t * nf * h, 0.0);
        let mut capture = |l: usize, x: &[f32]| {
            for (k, _) in layers.iter().enumerate().filter(|e| *e.1 == l) {
                for ti in 0..t {
                    feats[(ti * nf + k) * h..(ti * nf + k + 1) * h].copy_from_slice(&x[ti * h..(ti + 1) * h]);
                }
            }
        };
        for (l, layer) in self.layers.iter().enumerate() {
            capture(l, &x);
            let _unit = crate::stream::enter(&self.stream, l);
            self.layer_forward(l, layer, &mut x, seed.as_deref(), t, start, &mut sess.caches[l], &rope, iv.as_ref());
        }
        capture(self.layers.len(), &x);
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.final_norm, c.rms_norm_eps as f32);
        sess.pos += t;
        x
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        l: usize,
        layer: &Layer,
        x: &mut [f32],
        seed: Option<&[f32]>,
        t: usize,
        start: usize,
        cache: &mut LayerCache,
        rope: &(Vec<f32>, Vec<f32>),
        iv: Option<&crate::intervene::State>,
    ) {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let eps = c.rms_norm_eps as f32;
        let half = hd / 2;
        let (cos, sin) = rope;
        let par = t > 1;

        // ---- Attention ----
        let mut hn = vec![0f32; t * h];
        rmsnorm_f32w_rows(x, h, &layer.ln_in, eps, &mut hn);
        let mut q = vec![0f32; t * nh * hd];
        let mut k = vec![0f32; t * nkv * hd];
        let mut v = vec![0f32; t * nkv * hd];
        let mut g = vec![0f32; if layer.gate.is_some() { t * nh * hd } else { 0 }];
        // The values come from a mix of the embeddings and the layer's input, the rest from the
        // input alone.
        let vin: Option<Vec<f32>> = match (layer.blend, seed) {
            (Some(b), Some(e)) => Some(e.iter().zip(&hn).map(|(&e, &x)| e + b * (x - e)).collect()),
            _ => None,
        };
        {
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&layer.q, nh * hd, &mut q[..]), (&layer.k, nkv * hd, &mut k[..])];
            if let Some(gp) = &layer.gate {
                jobs.push((gp, nh * hd, &mut g[..]));
            }
            if vin.is_none() {
                jobs.push((&layer.v, nkv * hd, &mut v[..]));
            }
            linear_multi(&mut jobs, h, &hn, t);
        }
        if let Some(vin) = &vin {
            linear(&layer.v, nkv * hd, h, vin, t, &mut v);
        }
        let scale = (hd as f32).powf(-0.5);
        for_chunks(&mut q, hd, par, |i, qh| {
            let ti = i / nh;
            rmsnorm_f32w_inplace(qh, &layer.q_norm, eps);
            if layer.sliding {
                apply_rope(qh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            }
            scale_inplace(qh, scale);
        });
        for_chunks(&mut k, hd, par, |i, kh| {
            let ti = i / nkv;
            rmsnorm_f32w_inplace(kh, &layer.k_norm, eps);
            if layer.sliding {
                apply_rope(kh, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            }
        });
        cache.push(&k, &v, t);
        let window = if layer.sliding { c.sliding_window } else { None };
        let mut att = vec![0f32; t * nh * hd];
        if t >= 32 {
            attention_prefill(&q, cache, nh, t, start, window, None, &mut att);
        } else {
            attention(&q, cache, nh, t, start, window, &mut att);
        }
        if layer.gate.is_some() {
            for (a, &g) in att.iter_mut().zip(&g) {
                *a *= sigmoid(g);
            }
        }
        let mut o = vec![0f32; t * h];
        linear(&layer.o, h, nh * hd, &att, t, &mut o);
        rmsnorm_f32w_rows_inplace(&mut o, h, &layer.ln_post_attn, eps);
        add_inplace(x, &o);
        if let Some(iv) = iv {
            iv.point(2 * l + 1, x, h, start);
        }

        // ---- MLP ----
        rmsnorm_f32w_rows(x, h, &layer.ln_pre_ffn, eps, &mut hn);
        let inter = c.intermediate_size;
        let mut a = vec![0f32; t * inter];
        let mut u = vec![0f32; t * inter];
        linear_multi(
            &mut [(&layer.mlp_gate, inter, &mut a[..]), (&layer.mlp_up, inter, &mut u[..])],
            h,
            &hn,
            t,
        );
        if let Some(th) = c.clamp() {
            a.iter_mut().for_each(|v| *v = v.min(th));
            u.iter_mut().for_each(|v| *v = v.clamp(-th, th));
        }
        silu_mul(&mut a, &u);
        if let Some(k) = iv.and_then(|s| s.hook(l, start)) {
            k.apply(Target::Mlp, &mut a, inter);
        }
        linear(&layer.mlp_down, h, inter, &a, t, &mut o);
        rmsnorm_f32w_rows_inplace(&mut o, h, &layer.ln_post_ffn, eps);
        add_inplace(x, &o);
        if let Some(iv) = iv {
            iv.point(2 * l + 2, x, h, start);
        }
    }

    /// Drops the positions at or beyond `len`.
    pub fn truncate(&self, sess: &mut RippleSession, len: usize) {
        for c in &mut sess.caches {
            c.truncate(len);
        }
        sess.pos = sess.pos.min(len);
    }
}

impl crate::generate::TextLm for RippleModel<'_> {
    type Sess = RippleSession;
    fn new_session(&self) -> RippleSession {
        RippleModel::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut RippleSession) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &RippleSession) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut RippleSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        RippleModel::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        RippleModel::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut RippleSession, tokens: &[u32]) -> Vec<f32> {
        RippleModel::forward(self, sess, tokens)
    }
}

/// The interface speculative decoding rounds use ([`crate::dspark::Target`]: prompt lookup; the
/// DFlash2 draft published for this model is not read).
impl crate::dspark::Target for RippleModel<'_> {
    fn forward_feats(
        &self,
        sess: &mut RippleSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, layers, feats)
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        if let Some(s) = &self.stream {
            s.need_ranges(&self.embed.row_ranges(token as usize, h));
        }
        self.embed.row(token as usize, out);
    }
    fn embed_scale(&self) -> f32 {
        self.cfg.embed_scale.unwrap_or(1.0) as f32
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        RippleModel::logits(self, hidden, t)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut RippleSession, len: usize) {
        RippleModel::truncate(self, sess, len);
    }
}

/// Runtime interventions (`crate::intervene`): the points around every sublayer on the residual
/// stream, and as neurons the input of the MLP's `down_proj` (after the clamp and the SiLU).
impl crate::intervene::Host for RippleModel<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        (target == Target::Mlp && layer < self.layers.len()).then_some(self.cfg.intermediate_size)
    }
    fn down_column(&self, layer: usize, _target: Target, n: usize) -> Vec<f32> {
        let (h, inter) = (self.cfg.hidden_size, self.cfg.intermediate_size);
        let mut one = vec![0f32; inter];
        one[n] = 1.0;
        let mut y = vec![0f32; h];
        linear(&self.layers[layer].mlp_down, h, inter, &one, 1, &mut y);
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_config() {
        let mut raw = serde_json::json!({
            "model_type": "ricursive", "embed_scale": 71.55417527999327, "ffn_clamp_threshold": 20.0,
            "gate_full_precision": true, "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120,
            "intermediate_size": 40960, "is_hybrid_swa": true, "max_position_embeddings": 2097152,
            "num_attention_heads": 80, "num_hidden_layers": 80, "num_key_value_heads": 8, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "sliding_window": 1024, "tie_word_embeddings": false, "use_gated_attention": true,
            "use_valres": true, "vocab_size": 100352,
        });
        // Without `layer_types` every fifth layer is full, which is what the release lists.
        let c = RippleConfig::from_json(raw.clone()).unwrap();
        let types: Vec<&str> = (0..80)
            .map(|l| if l % 5 == 4 { "full_attention" } else { "sliding_attention" })
            .collect();
        raw["layer_types"] = serde_json::json!(types);
        let listed = RippleConfig::from_json(raw.clone()).unwrap();
        assert!((0..80).all(|l| c.is_sliding(l) == listed.is_sliding(l) && c.is_sliding(l) == (l % 5 != 4)));
        assert_eq!((c.clamp(), c.sliding_window, c.use_valres), (Some(20.0), Some(1024), true));
        // What does not make a model is refused.
        let bad = |k: &str, v: serde_json::Value| {
            let mut r = raw.clone();
            r[k] = v;
            RippleConfig::from_json(r).is_err()
        };
        assert!(bad("layer_types", serde_json::json!(["full_attention"])));
        assert!(bad("num_key_value_heads", 7.into()));
        assert!(bad("hidden_act", "gelu".into()));
        assert!(bad("sliding_window", 0.into()));
        let mut odd = types.clone();
        odd[3] = "chunked_attention";
        assert!(bad("layer_types", serde_json::json!(odd)));
    }

    /// The config of the tiny model: every feature on, a window short enough to bind.
    fn tiny_config(over: serde_json::Value) -> RippleConfig {
        let mut c = serde_json::json!({
            "vocab_size": 48, "hidden_size": 32, "intermediate_size": 96, "num_hidden_layers": 5,
            "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "sliding_window": 6, "use_gated_attention": true, "use_valres": true,
            "ffn_clamp_threshold": 1.0, "embed_scale": 5.0, "tie_word_embeddings": false,
        });
        for (k, v) in over.as_object().unwrap() {
            c[k] = v.clone();
        }
        RippleConfig::from_json(c).unwrap()
    }

    /// Writes a tiny random checkpoint of [`tiny_config`] in Ricursive's naming (float32).
    fn tiny_checkpoint(dir: &std::path::Path) {
        let c = tiny_config(serde_json::json!({}));
        let (h, v, nh, nkv, hd, inter) = (
            c.hidden_size,
            c.vocab_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.intermediate_size,
        );
        let mut seed = 0x726970u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        // `scale` times uniform noise, plus `bias`.
        let mut add = |name: String, shape: &[usize], scale: f32, bias: f32| {
            let n: usize = shape.iter().product();
            let bytes = (0..n).flat_map(|_| (bias + scale * rng()).to_le_bytes()).collect();
            tensors.push((name, shape.to_vec(), bytes));
        };
        let mat = |fan_in: usize, gain: f32| gain * (3.0 / fan_in as f32).sqrt();
        add("model.embed_tokens.weight".into(), &[v, h], 0.35, 0.0);
        add("lm_head.weight".into(), &[v, h], mat(h, 3.0), 0.0);
        add("model.norm.weight".into(), &[h], 0.2, 1.0);
        for l in 0..c.num_hidden_layers {
            let p = format!("model.layers.{l}.");
            for n in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                add(format!("{p}{n}.weight"), &[h], 0.2, 1.0);
            }
            let ap = format!("{p}self_attn.");
            add(format!("{ap}q_proj.weight"), &[nh * hd, h], mat(h, 2.0), 0.0);
            add(format!("{ap}k_proj.weight"), &[nkv * hd, h], mat(h, 2.0), 0.0);
            add(format!("{ap}v_proj.weight"), &[nkv * hd, h], mat(h, 1.0), 0.0);
            add(format!("{ap}g_proj.weight"), &[nh * hd, h], mat(h, 1.5), 0.0);
            add(format!("{ap}o_proj.weight"), &[h, nh * hd], mat(nh * hd, 1.0), 0.0);
            add(format!("{ap}q_norm.weight"), &[hd], 0.2, 2.0);
            add(format!("{ap}k_norm.weight"), &[hd], 0.2, 2.0);
            add(format!("{ap}value_res_lambda"), &[1], 1.0, 0.0);
            add(format!("{p}mlp.gate_proj.weight"), &[inter, h], mat(h, 1.5), 0.0);
            add(format!("{p}mlp.up_proj.weight"), &[inter, h], mat(h, 1.5), 0.0);
            add(format!("{p}mlp.down_proj.weight"), &[h, inter], mat(inter, 1.0), 0.0);
        }
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, bytes) in tensors {
            let entry = serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]});
            head.insert(name, entry);
            data.extend(bytes);
        }
        let mut head = serde_json::Value::Object(head).to_string();
        while !head.len().is_multiple_of(8) {
            head.push(' ');
        }
        let mut file = (head.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(head.as_bytes());
        file.extend(data);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
    }

    /// A tiny random model, without a reference (tools/ref_ripple.py measures the agreement with
    /// Ricursive's code): the logits do not depend on how the tokens are split into passes (the
    /// long-pass attention kernel included), a session cut back leaves no trace, and each
    /// mechanism changes the logits.
    #[test]
    fn tiny_model() {
        let dir = std::env::temp_dir().join(format!("oxinfer-ripple-test-{}", std::process::id()));
        tiny_checkpoint(&dir);
        let files = ModelFiles::open(&dir, true).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let load = |over: serde_json::Value| RippleModel::load(tiny_config(over), &files, Quant::Bf16).unwrap();
        let m = load(serde_json::json!({}));
        let (h, v) = (m.cfg.hidden_size, m.cfg.vocab_size);
        let ids: Vec<u32> = (0..80).map(|i| (i * 7 + 3) % v as u32).collect();
        let more = [5u32, 17, 2, 9];
        // The logits after the prompt (in passes of `chunk`) and after each of `more`.
        let run = |m: &RippleModel, chunk: usize| -> Vec<f32> {
            let mut sess = m.new_session();
            let hs = crate::generate::prefill(m, &mut sess, &ids, chunk, &[]);
            let mut out = m.logits(&hs[hs.len() - h..], 1);
            for &t in &more {
                out.extend(m.forward(&mut sess, &[t]));
            }
            out
        };
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let base = run(&m, 80);
        assert!(base.iter().all(|x| x.is_finite()));
        let scale = base.iter().fold(0f32, |a, b| a.max(b.abs()));
        let tol = 2e-3 * scale;
        for chunk in [40, 17, 7, 1] {
            let d = diff(&run(&m, chunk), &base);
            assert!(d <= tol, "passes of {chunk} differ from one pass by {d} (logits up to {scale})");
        }
        // Cut back: the tokens after it run as if the dropped ones never had.
        let mut sess = m.new_session();
        m.forward(&mut sess, &ids);
        m.forward(&mut sess, &[30, 31, 29]);
        m.truncate(&mut sess, ids.len());
        assert_eq!(sess.pos, ids.len());
        let again: Vec<f32> = more.iter().flat_map(|&t| m.forward(&mut sess, &[t])).collect();
        let d = diff(&again, &base[v..]);
        assert!(d <= tol, "after a rollback the logits differ by {d}");
        // Each mechanism matters.
        let all_full = serde_json::json!(vec!["full_attention"; 5]);
        for (what, over) in [
            ("the window", serde_json::json!({"sliding_window": 64})),
            ("the layer kinds", serde_json::json!({"layer_types": all_full})),
            ("the gate", serde_json::json!({"use_gated_attention": false})),
            ("the value residuals", serde_json::json!({"use_valres": false})),
            ("the clamp", serde_json::json!({"ffn_clamp_threshold": null})),
            ("the embedding scale", serde_json::json!({"embed_scale": null})),
        ] {
            let d = diff(&run(&load(over), 80), &base);
            assert!(d > 20.0 * tol, "{what}: the logits moved by {d} only");
        }
    }

    /// The reasoning and the tool calls of a reply are marked in plain text: the stream reports
    /// the reasoning as thinking, shows neither the marks nor the line breaks that belong to
    /// them, and stops at the calls.
    #[test]
    fn reply_marks_are_filtered() {
        // One token per byte.
        let chars = crate::kimi::byte_chars();
        let normal: Vec<(String, u32)> = chars.iter().enumerate().map(|(b, c)| (c.to_string(), b as u32)).collect();
        let tok = crate::tokenizer::from_value(&crate::kimi::bpe_json(&normal, &[])).unwrap();
        // (thinking, text, flips, called) of a reply fed token by token.
        let read = |reply: &str| {
            let ids = tok.encode(reply, false).unwrap().get_ids().to_vec();
            let mut stream = crate::generate::TextStream::new(&crate::chat::RippleChat, true, &tok);
            assert!(!stream.thinking());
            let (mut thinking, mut text, mut flips) = (String::new(), String::new(), 0);
            for n in 1..=ids.len() {
                flips += usize::from(stream.on_token(ids[n - 1]));
                let d = stream.delta(&tok, &ids[..n], false);
                if stream.thinking() { &mut thinking } else { &mut text }.push_str(&d);
            }
            let d = stream.delta(&tok, &ids, true);
            if stream.thinking() { &mut thinking } else { &mut text }.push_str(&d);
            (thinking, text, flips, stream.called())
        };
        let own = |a: &str, b: &str, f: usize, c: bool| (a.to_string(), b.to_string(), f, c);
        assert_eq!(
            read("<think>\nLet me see: 1 < 2.\n\nYes.\n</think>\nThe answer is <b>2</b>.\n"),
            own("Let me see: 1 < 2.\n\nYes.", "The answer is <b>2</b>.\n", 2, false)
        );
        assert_eq!(
            read("<think>\nA tool.\n</think>\n\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"),
            own("A tool.", "", 2, true)
        );
        assert_eq!(
            read("Checking.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>\nmore"),
            own("", "Checking.", 0, true)
        );
        assert_eq!(read("No marks, a < b and </b>\n"), own("", "No marks, a < b and </b>\n", 0, false));
        // Cut inside the reasoning: what was written is reasoning, to its last byte.
        assert_eq!(read("<think>\nUnfinished <th"), own("Unfinished <th", "", 1, false));
    }
}
