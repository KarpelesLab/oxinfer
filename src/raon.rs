//! Raon-1 nano (`raon1_nano`: KRAFTON's Raon-1-nano, 24 layers of 1536; `Raon1NanoForCausalLM`),
//! as KRAFTON's `modeling_raon1_nano.py`:
//!
//! - No positions anywhere: order comes from the recurrent layers.
//! - A residual of `raon_mhc_streams` streams ("Raon_mHC"). Each sublayer (attention, MLP) has a
//!   selector reading the token's streams side by side, RMS-scaled: `h_pre = sigmoid(..)` weighs
//!   the streams into the sublayer's input (then RMS-normed), `h_post = post_factor sigmoid(..)`
//!   writes its output back to each stream, and the streams are mixed by
//!   `H = (1 - gamma) I + gamma Phi`, `gamma = sigmoid(eta)`, `Phi = B + E / (1 + rho)` with
//!   `B = (I + 11^T / n) / 2`, `E = Q Z Q^T` (`Q`: the Helmert basis, `Z` the selector's
//!   `(n - 1)^2` values) and `rho = max(0, max(-E / B))`, which keeps `Phi` non-negative with
//!   rows and columns summing to one. The model's output is the mean of the streams, normed.
//! - RDA layers (Raon Delta Attention, most layers): Kimi's KDA ([`crate::kimi::kda_forward`]:
//!   short causal convolutions, a gated delta rule per head with a per-channel decay bounded
//!   below, a gated RMS norm of the output) with a write gate: the value enters the state with
//!   `sigmoid(b + w)` where the state is erased with `sigmoid(b)`; queries and keys are normed
//!   as `x / max(|x|, 1e-6)`; the decay's low-rank projection has its own width
//!   (`rda_decay_rank`).
//! - MLA layers (`rda_config.mla_layers`): latent attention with a query LoRA, no rotary part and
//!   an output gate, as Kimi K3's ([`crate::kimi::mla_forward`]).
//! - A dense SwiGLU MLP in every layer.
//! - Per-layer embeddings (layout version 2): per token and layer
//!   `p = (RMSNorm(W_l e) + sqrt(d) T[token]_l) / sqrt(2)` (`e`: the token's embedding, `T` a
//!   `[vocab, layers * d]` table), and after each layer's MLP every stream receives
//!   `RMSNorm(P (gelu(G m) * p))`, `m` being the mean of the streams.
//! - The LM head is the embedding table.
//!
//! Checkpoints: KRAFTON's safetensors layout. The auxiliary multi-token prediction layer
//! (`mtp/model.safetensors`) is not read.

use crate::intervene::Target;
use crate::kernels::*;
use crate::kimi::{Kda, KdaDims, KdaRecord, KdaState, Mla, MlaCache, MlaDims, Mlp, kda_forward, kda_scan, load_lin_exact, mla_forward};
use crate::mistral4::load_kv_b;
use crate::model::{Lin, Quant, Table, linear, table_quant};
use crate::moe::Act;
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d_eps() -> f64 {
    1e-6
}
fn d_kernel() -> usize {
    4
}
fn d_lower() -> Option<f64> {
    Some(-5.0)
}
fn d_rank() -> usize {
    128
}
fn d_ple() -> usize {
    256
}
fn d_two() -> usize {
    2
}
fn d_true() -> bool {
    true
}

/// `rda_config`: which layers are RDA (1-based layer numbers; the others are MLA) and the RDA
/// shapes.
#[derive(Debug, Clone, Deserialize)]
pub struct RdaConfig {
    #[serde(default)]
    pub num_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "d_kernel")]
    pub conv_kernel_size: usize,
    #[serde(default = "d_lower")]
    pub gate_lower_bound: Option<f64>,
    pub rda_layers: Vec<usize>,
    #[serde(default)]
    pub mla_layers: Vec<usize>,
}

/// `config.json` of a Raon-1 nano checkpoint (`Raon1NanoConfig`).
#[derive(Debug, Clone, Deserialize)]
pub struct RaonConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "d_eps")]
    pub mla_latent_norm_eps: f64,
    #[serde(default)]
    pub raon_mhc_streams: Option<usize>,
    #[serde(default)]
    pub raon_mhc_post_factor: Option<f64>,
    #[serde(default)]
    pub raon_mhc_selector_eps: Option<f64>,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    #[serde(default = "d_true")]
    pub mla_use_output_gate: bool,
    pub rda_config: RdaConfig,
    #[serde(default = "d_rank")]
    pub rda_decay_rank: usize,
    #[serde(default = "d_ple")]
    pub per_layer_embedding_dim: usize,
    #[serde(default = "d_two")]
    pub per_layer_embedding_layout_version: usize,
}

impl RaonConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::from_json(serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?)
    }

    pub fn from_json(raw: serde_json::Value) -> Result<Self> {
        let c: RaonConfig = serde_json::from_value(raw).context("parse the Raon-1 config")?;
        let n = c.num_hidden_layers;
        let r = &c.rda_config;
        if r.rda_layers.iter().chain(&r.mla_layers).any(|&l| l == 0 || l > n) {
            bail!("raon: a layer of rda_config is out of range");
        }
        // (The modeling code reads `rda_layers` only: every other layer is MLA.)
        if !r.mla_layers.is_empty() && (0..n).any(|l| c.is_rda(l) == r.mla_layers.contains(&(l + 1))) {
            bail!("raon: rda_layers and mla_layers of rda_config do not split the layers");
        }
        if r.gate_lower_bound.is_none() {
            bail!("raon: RDA without gate_lower_bound (a softplus decay) is not supported");
        }
        if c.rda_heads() == 0 || c.rda_head_dim() == 0 || r.conv_kernel_size < 2 || c.rda_decay_rank == 0 {
            bail!("raon: bad RDA shapes");
        }
        if c.streams() < 2 {
            bail!("raon: fewer than two residual streams (raon_mhc_streams) is not supported");
        }
        if c.per_layer_embedding_dim != 0 && c.per_layer_embedding_layout_version != 2 {
            bail!("raon: only per-layer embedding layout version 2 is supported");
        }
        Ok(c)
    }

    /// Whether layer `l` (0-based) is an RDA layer.
    pub fn is_rda(&self, l: usize) -> bool {
        self.rda_config.rda_layers.contains(&(l + 1))
    }
    pub fn streams(&self) -> usize {
        self.raon_mhc_streams.unwrap_or(4)
    }
    fn rda_heads(&self) -> usize {
        self.rda_config.num_heads.unwrap_or(self.num_attention_heads)
    }
    fn rda_head_dim(&self) -> usize {
        self.rda_config
            .head_dim
            .or(self.head_dim)
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }
    fn rda_dims(&self) -> KdaDims {
        KdaDims {
            h: self.hidden_size,
            nh: self.rda_heads(),
            hd: self.rda_head_dim(),
            rank: self.rda_decay_rank,
            kc: self.rda_config.conv_kernel_size,
            lower: self.rda_config.gate_lower_bound.unwrap() as f32,
            eps: self.rms_norm_eps as f32,
            clamp_norm: true,
        }
    }
    fn mla_dims(&self) -> MlaDims {
        MlaDims {
            h: self.hidden_size,
            nh: self.num_attention_heads,
            lat: self.kv_lora_rank,
            vd: self.v_head_dim,
            dn: self.qk_nope_head_dim,
            dr: self.qk_rope_head_dim,
            ql: self.q_lora_rank,
            eps: self.mla_latent_norm_eps as f32,
        }
    }
}

/// One Raon_mHC site: the selector's rows (`[2n + (n - 1)^2, n * H]`: pre, post, then the mixing
/// chart's values), its three scales and its biases, and `sigmoid(eta)`.
struct Site {
    w: Vec<f32>,
    alpha: [f32; 3],
    bias: Vec<f32>,
    gamma: f32,
}

/// A layer's share of the per-layer embedding: `gate` `[d, H]`, `proj` `[H, d]` and the norm of
/// what the streams receive.
struct PleGate<'a> {
    gate: Lin<'a>,
    proj: Lin<'a>,
    norm: Vec<f32>,
}

/// The per-layer embedding inputs: the table `[vocab, layers * d]`, the projection of the token
/// embedding `[layers * d, H]` and its norm `[d]`.
struct Ple<'a> {
    table: Table<'a>,
    proj: Lin<'a>,
    norm: Vec<f32>,
}

#[allow(clippy::large_enum_variant)]
enum Attn<'a> {
    Rda(Kda<'a>),
    Mla(Mla<'a>),
}

struct Layer<'a> {
    ln_in: Vec<f32>,
    ln_post: Vec<f32>,
    hc_attn: Site,
    hc_mlp: Site,
    attn: Attn<'a>,
    mlp: Mlp<'a>,
    ple: Option<PleGate<'a>>,
}

#[derive(Clone)]
enum LayerState {
    Rda(KdaState),
    Mla(MlaCache),
}

/// The records of the last recording pass, which started at `pos` (see [`KdaRecord`]).
#[derive(Clone)]
struct Record {
    pos: usize,
    rda: Vec<(usize, KdaRecord)>,
}

/// Decoding state.
#[derive(Clone)]
pub struct RaonSession {
    layers: Vec<LayerState>,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
    rec: Option<Box<Record>>,
}

pub struct RaonModel<'a> {
    pub cfg: RaonConfig,
    pub quant: Quant,
    embed: Table<'a>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    ple: Option<Ple<'a>>,
    layers: Vec<Layer<'a>>,
    /// The Helmert basis `[n, n - 1]`.
    helmert: Vec<f32>,
    /// Weight streaming: unit 0 is the per-layer embedding projection, unit `l + 1` layer `l`,
    /// the last one the LM head.
    stream: Option<crate::stream::StreamHook>,
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The orthonormal `Q` (`[n, n - 1]`) with `Q^T Q = I` and `Q^T 1 = 0`.
fn helmert_basis(n: usize) -> Vec<f32> {
    let mut q = vec![0f32; n * (n - 1)];
    for k in 1..n {
        let s = ((k * (k + 1)) as f32).sqrt();
        for i in 0..k {
            q[i * (n - 1) + k - 1] = 1.0 / s;
        }
        q[k * (n - 1) + k - 1] = -(k as f32) / s;
    }
    q
}

/// Loads tensors by their names in KRAFTON's checkpoints, with shape checks.
struct Loader<'a, 'c> {
    files: &'a ModelFiles,
    cfg: &'c RaonConfig,
    quant: Quant,
}

impl<'a> Loader<'a, '_> {
    fn getf(&self, full: &str, n: usize) -> Result<Vec<f32>> {
        let info = self.files.info(full).with_context(|| format!("missing tensor {full}"))?;
        if info.shape.iter().product::<usize>() != n {
            bail!("tensor {full} has shape {:?}, expected {n} values", info.shape);
        }
        self.files.f32(full)
    }
    fn lin(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        load_lin_exact(self.files, full, n_out, n_in, self.quant)
    }

    fn site(&self, p: &str) -> Result<Site> {
        let (n, h) = (self.cfg.streams(), self.cfg.hidden_size);
        let m = (n - 1) * (n - 1);
        let one = |name: &str| Ok::<f32, anyhow::Error>(self.getf(&format!("{p}{name}"), 1)?[0]);
        let mut w = self.getf(&format!("{p}phi_pre"), n * n * h)?;
        w.extend(self.getf(&format!("{p}phi_post"), n * n * h)?);
        w.extend(self.getf(&format!("{p}res.phi_z"), m * n * h)?);
        let mut bias = self.getf(&format!("{p}b_pre"), n)?;
        bias.extend(self.getf(&format!("{p}b_post"), n)?);
        bias.extend(self.getf(&format!("{p}res.b_z"), m)?);
        Ok(Site {
            w,
            alpha: [one("alpha_pre")?, one("alpha_post")?, one("res.alpha_z")?],
            bias,
            gamma: sigmoid(one("res.eta")?),
        })
    }

    fn rda(&self, ap: &str) -> Result<Kda<'a>> {
        let dm = self.cfg.rda_dims();
        let (h, nh, d) = (dm.h, dm.nh, dm.nh * dm.hd);
        let conv = |n: &str| self.getf(&format!("{ap}{n}_conv1d.weight"), d * dm.kc);
        Ok(Kda {
            q: self.lin(&format!("{ap}q_proj.weight"), d, h)?,
            k: self.lin(&format!("{ap}k_proj.weight"), d, h)?,
            v: self.lin(&format!("{ap}v_proj.weight"), d, h)?,
            conv: [conv("q")?, conv("k")?, conv("v")?],
            f_a: self.lin(&format!("{ap}f_a_proj.weight"), dm.rank, h)?,
            f_b: self.lin(&format!("{ap}f_b_proj.weight"), d, dm.rank)?,
            exp_a: self.getf(&format!("{ap}A_log"), nh)?.iter().map(|v| v.exp()).collect(),
            dt_bias: self.getf(&format!("{ap}dt_bias"), d)?,
            b: self.lin(&format!("{ap}b_proj.weight"), nh, h)?,
            w: Some(self.lin(&format!("{ap}rda_w_proj.weight"), nh, h)?),
            g: self.lin(&format!("{ap}g_proj.weight"), d, h)?,
            o_norm: self.getf(&format!("{ap}o_norm.weight"), dm.hd)?,
            o: self.lin(&format!("{ap}o_proj.weight"), h, d)?,
        })
    }

    fn mla(&self, ap: &str) -> Result<Mla<'a>> {
        let dm = self.cfg.mla_dims();
        let (h, nh, lat, vd, qk) = (dm.h, dm.nh, dm.lat, dm.vd, dm.dn + dm.dr);
        let (k_b, v_b) = load_kv_b(self.files, ap, nh, lat, dm.dn, vd)?;
        Ok(Mla {
            q_a: self.lin(&format!("{ap}q_a_proj.weight"), dm.ql, h)?,
            q_norm: self.getf(&format!("{ap}q_a_layernorm.weight"), dm.ql)?,
            q_b: self.lin(&format!("{ap}q_b_proj.weight"), nh * qk, dm.ql)?,
            kv_a: self.lin(&format!("{ap}kv_a_proj_with_mqa.weight"), lat + dm.dr, h)?,
            kv_norm: self.getf(&format!("{ap}kv_a_layernorm.weight"), lat)?,
            k_b,
            v_b,
            gate: if self.cfg.mla_use_output_gate {
                Some(self.lin(&format!("{ap}g_proj.weight"), nh * vd, h)?)
            } else {
                None
            },
            o: self.lin(&format!("{ap}o_proj.weight"), h, nh * vd)?,
            indexer: None,
        })
    }
}

impl<'a> RaonModel<'a> {
    pub fn load(cfg: RaonConfig, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let ld = Loader { files, cfg: &cfg, quant };
        let (h, v, nl, d) = (cfg.hidden_size, cfg.vocab_size, cfg.num_hidden_layers, cfg.per_layer_embedding_dim);
        // The head is tied to the embedding (`lm_head.weight`, where a file repeats it, is not read).
        let embed_name = "model.embed_tokens.weight";
        let embed = Table::load(files, embed_name, v, h, quant)?;
        let lm_head = ld.lin(embed_name, v, h)?;
        let ple = if d > 0 {
            Some(Ple {
                table: Table::load(files, "model.per_layer_inputs.table", v, nl * d, table_quant(quant))?,
                proj: ld.lin("model.per_layer_inputs.embedding_projection", nl * d, h)?,
                norm: ld.getf("model.per_layer_inputs.projection_norm.weight", d)?,
            })
        } else {
            None
        };
        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let lp = format!("model.layers.{l}.");
            let ap = format!("{lp}self_attn.");
            let inter = cfg.intermediate_size;
            layers.push(Layer {
                ln_in: ld.getf(&format!("{lp}input_layernorm.weight"), h)?,
                ln_post: ld.getf(&format!("{lp}post_attention_layernorm.weight"), h)?,
                hc_attn: ld.site(&format!("{lp}self_attn_raon_mhc."))?,
                hc_mlp: ld.site(&format!("{lp}mlp_raon_mhc."))?,
                attn: if cfg.is_rda(l) {
                    Attn::Rda(ld.rda(&ap)?)
                } else {
                    Attn::Mla(ld.mla(&ap)?)
                },
                mlp: Mlp {
                    gate: ld.lin(&format!("{lp}mlp.gate_proj.weight"), inter, h)?,
                    up: ld.lin(&format!("{lp}mlp.up_proj.weight"), inter, h)?,
                    down: ld.lin(&format!("{lp}mlp.down_proj.weight"), h, inter)?,
                    inter,
                    act: Act::Silu,
                },
                ple: if d > 0 {
                    let pp = format!("{lp}per_layer_input_gate.");
                    Some(PleGate {
                        gate: ld.lin(&format!("{pp}gate"), d, h)?,
                        proj: ld.lin(&format!("{pp}projection"), h, d)?,
                        norm: ld.getf(&format!("{pp}output_norm.weight"), h)?,
                    })
                } else {
                    None
                },
            });
        }
        Ok(Self {
            final_norm: ld.getf("model.norm.weight", h)?,
            helmert: helmert_basis(cfg.streams()),
            cfg,
            quant,
            embed,
            lm_head,
            ple,
            layers,
            stream: None,
        })
    }

    pub fn new_session(&self) -> RaonSession {
        let dm = self.cfg.rda_dims();
        let d = dm.nh * dm.hd;
        RaonSession {
            layers: self
                .layers
                .iter()
                .map(|l| match &l.attn {
                    Attn::Rda(_) => LayerState::Rda(KdaState {
                        s: vec![0.0; d * dm.hd],
                        conv: vec![0.0; 3 * (dm.kc - 1) * d],
                    }),
                    Attn::Mla(_) => LayerState::Mla(MlaCache::default()),
                })
                .collect(),
            pos: 0,
            iv: Default::default(),
            rec: None,
        }
    }

    /// Weight bytes read per decoded token.
    pub fn streamed_bytes(&self) -> usize {
        self.lm_head.bytes()
            + self.ple.as_ref().map_or(0, |p| p.proj.bytes())
            + self
                .layers
                .iter()
                .map(|l| self.lins(l).iter().map(|w| w.bytes()).sum::<usize>())
                .sum::<usize>()
    }

    fn lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let mut v: Vec<&Lin> = match &l.attn {
            Attn::Rda(k) => {
                let mut v = vec![&k.q, &k.k, &k.v, &k.f_a, &k.f_b, &k.b, &k.g, &k.o];
                v.extend(k.w.as_ref());
                v
            }
            Attn::Mla(a) => {
                let mut v = vec![&a.q_a, &a.q_b, &a.kv_a, &a.o];
                v.extend(a.gate.as_ref());
                v.extend(a.k_b.iter().chain(&a.v_b));
                v
            }
        };
        v.extend([&l.mlp.gate, &l.mlp.up, &l.mlp.down]);
        if let Some(p) = &l.ple {
            v.extend([&p.gate, &p.proj]);
        }
        v
    }

    /// The streaming units in execution order (see `crate::stream`): the per-layer embedding
    /// projection, the layers, the LM head.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        let unit = |name: String, ranges| crate::stream::UnitSpec {
            name,
            ranges,
            uses: 1.0,
            parts: Vec::new(),
        };
        let mut v = vec![unit(
            "per_layer_inputs".into(),
            self.ple.as_ref().map_or(Vec::new(), |p| p.proj.mem_ranges()),
        )];
        for (l, layer) in self.layers.iter().enumerate() {
            v.push(unit(
                format!("layer.{l}"),
                self.lins(layer).iter().flat_map(|w| w.mem_ranges()).collect(),
            ));
        }
        v.push(unit("lm_head".into(), self.lm_head.mem_ranges()));
        v
    }

    /// Marks the weights of the units with `streamed[i]` set as streamed (no accelerator state
    /// kept between calls).
    pub fn mark_streamed(&self, streamed: &[bool]) {
        let on = |i: usize| streamed.get(i).copied().unwrap_or(false);
        if let Some(p) = &self.ple {
            p.proj.set_transient_device(on(0));
        }
        for (l, layer) in self.layers.iter().enumerate() {
            for w in self.lins(layer) {
                w.set_transient_device(on(l + 1));
            }
        }
        self.lm_head.set_transient_device(on(self.layers.len() + 1));
    }

    /// Brackets every unit with `hook`'s (None stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// LM-head logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * c.vocab_size];
        let _unit = crate::stream::enter(&self.stream, self.layers.len() + 1);
        linear(&self.lm_head, c.vocab_size, c.hidden_size, hidden, t, &mut logits);
        logits
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut RaonSession, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos` (positions in `mm` take the given embeddings instead of the
    /// token's), appending to the caches; returns the final-normed hidden states `[t, hidden]`.
    pub fn forward_hidden_mm(&self, sess: &mut RaonSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, &[], &mut Vec::new(), false)
    }

    /// One token's routing at a site: from its streams `x` (`[n * H]`), `pre` `[n]`, `post` `[n]`
    /// and the mixing matrix `res` (`[n, n]`, row = destination stream).
    fn route(&self, s: &Site, x: &[f32], pre: &mut [f32], post: &mut [f32], res: &mut [f32]) {
        let c = &self.cfg;
        let n = c.streams();
        let (nm, w) = (n - 1, x.len());
        let inv = (dot(x, x) / w as f32 + c.raon_mhc_selector_eps.unwrap_or(1e-5) as f32).powf(-0.5);
        let z = |i: usize| dot(&s.w[i * w..(i + 1) * w], x) * inv;
        let pf = c.raon_mhc_post_factor.unwrap_or(2.0) as f32;
        for i in 0..n {
            pre[i] = sigmoid(s.alpha[0] * z(i) + s.bias[i]);
            post[i] = pf * sigmoid(s.alpha[1] * z(n + i) + s.bias[n + i]);
        }
        let tz: Vec<f32> = (0..nm * nm).map(|i| s.alpha[2] * z(2 * n + i) + s.bias[2 * n + i]).collect();
        // E = Q Z Q^T; B = (I + 11^T / n) / 2.
        let q = &self.helmert;
        let b = |i: usize, j: usize| 0.5 * (f32::from(i == j) + 1.0 / n as f32);
        let mut rho = 0f32;
        for i in 0..n {
            for j in 0..n {
                let mut e = 0f32;
                for m in 0..nm {
                    for k in 0..nm {
                        e += q[i * nm + m] * tz[m * nm + k] * q[j * nm + k];
                    }
                }
                res[i * n + j] = e;
                rho = rho.max(-e / b(i, j));
            }
        }
        for i in 0..n {
            for j in 0..n {
                let phi = b(i, j) + res[i * n + j] / (1.0 + rho);
                res[i * n + j] = (1.0 - s.gamma) * f32::from(i == j) + s.gamma * phi;
            }
        }
    }

    /// Before a sublayer: per token the routing of `s`, and the weighted, `norm`-ed input
    /// (`[t, H]`); returns (input, post `[t, n]`, res `[t, n * n]`).
    fn hc_pre(&self, s: &Site, norm: &[f32], x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let (n, h) = (c.streams(), c.hidden_size);
        let mut y = vec![0f32; t * h];
        let mut post = vec![0f32; t * n];
        let mut res = vec![0f32; t * n * n];
        let (pp, rp) = (SendPtr(post.as_mut_ptr()), SendPtr(res.as_mut_ptr()));
        for_chunks(&mut y, h, t > 1, |ti, yr| {
            // SAFETY: each token writes its own rows.
            let (po, ro) = unsafe {
                (
                    std::slice::from_raw_parts_mut(pp.get().add(ti * n), n),
                    std::slice::from_raw_parts_mut(rp.get().add(ti * n * n), n * n),
                )
            };
            let xr = &x[ti * n * h..(ti + 1) * n * h];
            let mut pre = vec![0f32; n];
            self.route(s, xr, &mut pre, po, ro);
            yr.fill(0.0);
            for (i, &p) in pre.iter().enumerate() {
                axpy(yr, p, &xr[i * h..(i + 1) * h]);
            }
            rmsnorm_f32w_inplace(yr, norm, c.rms_norm_eps as f32);
        });
        (y, post, res)
    }

    /// After a sublayer: `x[i] = sum_j res[i][j] x[j] + post[i] f` per token (`f` `[t, H]`).
    fn hc_post(&self, x: &mut [f32], f: &[f32], post: &[f32], res: &[f32], t: usize) {
        let (n, h) = (self.cfg.streams(), self.cfg.hidden_size);
        for_chunks(x, n * h, t > 1, |ti, xr| {
            let old = xr.to_vec();
            let (po, ro) = (&post[ti * n..(ti + 1) * n], &res[ti * n * n..(ti + 1) * n * n]);
            let fr = &f[ti * h..(ti + 1) * h];
            for i in 0..n {
                let o = &mut xr[i * h..(i + 1) * h];
                for (v, &a) in o.iter_mut().zip(fr) {
                    *v = po[i] * a;
                }
                for j in 0..n {
                    axpy(o, ro[i * n + j], &old[j * h..(j + 1) * h]);
                }
            }
        });
    }

    /// The mean of each token's streams (`[t, H]`).
    fn mean(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (n, h) = (self.cfg.streams(), self.cfg.hidden_size);
        let mut m = vec![0f32; t * h];
        for_chunks(&mut m, h, t > 1, |ti, row| {
            for i in 0..n {
                axpy(row, 1.0 / n as f32, &x[(ti * n + i) * h..(ti * n + i + 1) * h]);
            }
        });
        m
    }

    /// The per-layer inputs of a pass (`[t, layers * d]`) from the tokens and their embeddings.
    fn ple_inputs(&self, p: &Ple, tokens: &[u32], e: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (h, d) = (c.hidden_size, c.per_layer_embedding_dim);
        let w = c.num_hidden_layers * d;
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| p.table.row_ranges(tk as usize, w)).collect();
            s.need_ranges(&rows);
        }
        let mut out = vec![0f32; t * w];
        {
            let _unit = crate::stream::enter(&self.stream, 0);
            linear(&p.proj, w, h, e, t, &mut out);
        }
        rmsnorm_f32w_rows_inplace(&mut out, d, &p.norm, c.rms_norm_eps as f32);
        let (sd, r2) = ((d as f32).sqrt(), std::f32::consts::FRAC_1_SQRT_2);
        for_chunks(&mut out, w, t > 1, |ti, o| {
            let mut row = vec![0f32; w];
            p.table.row(tokens[ti] as usize, &mut row);
            for (v, r) in o.iter_mut().zip(&row) {
                *v = (*v + sd * r) * r2;
            }
        });
        out
    }

    /// After layer `l`'s MLP: every stream receives the gated, projected, normed per-layer input.
    fn ple_add(&self, g: &PleGate, inputs: &[f32], l: usize, x: &mut [f32], t: usize) {
        let c = &self.cfg;
        let (n, h, d) = (c.streams(), c.hidden_size, c.per_layer_embedding_dim);
        let w = c.num_hidden_layers * d;
        let m = self.mean(x, t);
        let mut a = vec![0f32; t * d];
        linear(&g.gate, d, h, &m, t, &mut a);
        for (ti, row) in a.chunks_mut(d).enumerate() {
            for (v, p) in row.iter_mut().zip(&inputs[ti * w + l * d..ti * w + (l + 1) * d]) {
                // GELU, tanh approximation.
                let u = 0.797_884_6 * (*v + 0.044_715 * *v * *v * *v);
                *v = 0.5 * *v * (1.0 + u.tanh()) * p;
            }
        }
        let mut delta = vec![0f32; t * h];
        linear(&g.proj, h, d, &a, t, &mut delta);
        rmsnorm_f32w_rows_inplace(&mut delta, h, &g.norm, c.rms_norm_eps as f32);
        for_chunks(x, n * h, t > 1, |ti, xr| {
            for s in xr.chunks_mut(h) {
                add_inplace(s, &delta[ti * h..(ti + 1) * h]);
            }
        });
    }

    /// [`Self::forward_hidden_mm`] that also writes to `feats` (`[t, layers.len() * hidden]`) the
    /// mean of the streams entering each of `layers`, `num_hidden_layers` standing for the last
    /// layer's output. `record`: keep what a rollback inside this pass needs
    /// ([`RaonModel::truncate`]).
    pub fn forward_capture(
        &self,
        sess: &mut RaonSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
        record: bool,
    ) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let c = &self.cfg;
        let (h, n) = (c.hidden_size, c.streams());
        let start = sess.pos;
        // (New tokens end what the last recorded pass could roll back.)
        sess.rec = None;
        let mut rec = record.then(|| Record {
            pos: start,
            rda: Vec::new(),
        });
        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut e = vec![0f32; t * h];
        for_chunks(&mut e, h, t > 1, |ti, row| self.embed.row(tokens[ti] as usize, row));
        for &(i, emb) in mm {
            e[i * h..(i + 1) * h].copy_from_slice(emb);
        }
        let inputs = self.ple.as_ref().map(|p| self.ple_inputs(p, tokens, &e, t));
        let mut x = vec![0f32; t * n * h];
        for (xr, er) in x.chunks_mut(n * h).zip(e.chunks(h)) {
            xr.chunks_mut(h).for_each(|s| s.copy_from_slice(er));
        }
        drop(timer);
        // Interventions: residual points on the streams (as DeepSeek V4's copies).
        let iv = sess.iv.active().then(|| sess.iv.clone());
        let point = |p: usize, x: &mut [f32]| {
            if let Some(iv) = &iv {
                iv.point_copies(p, x, h, n, start);
            }
        };
        point(0, &mut x);
        let nf = layers.len();
        feats.clear();
        feats.resize(t * nf * h, 0.0);
        let mut capture = |l: usize, x: &[f32]| {
            if layers.contains(&l) {
                let m = self.mean(x, t);
                for (k, _) in layers.iter().enumerate().filter(|e| *e.1 == l) {
                    for ti in 0..t {
                        feats[(ti * nf + k) * h..(ti * nf + k + 1) * h].copy_from_slice(&m[ti * h..(ti + 1) * h]);
                    }
                }
            }
        };
        let (rda_dims, mla_dims) = (c.rda_dims(), c.mla_dims());
        let mut o = vec![0f32; t * h];
        for (l, layer) in self.layers.iter().enumerate() {
            capture(l, &x);
            let _unit = crate::stream::enter(&self.stream, l + 1);
            let (xa, post, res) = self.hc_pre(&layer.hc_attn, &layer.ln_in, &x, t);
            match (&layer.attn, &mut sess.layers[l]) {
                (Attn::Rda(k), LayerState::Rda(st)) => {
                    let mut kr = rec.as_ref().map(|_| KdaRecord::default());
                    kda_forward(&rda_dims, k, st, kr.as_mut(), &xa, t, &mut o);
                    if let (Some(r), Some(kr)) = (&mut rec, kr) {
                        r.rda.push((l, kr));
                    }
                }
                (Attn::Mla(a), LayerState::Mla(cache)) => mla_forward(&mla_dims, None, a, cache, &xa, t, start, &mut o),
                _ => unreachable!("layer state of another kind"),
            }
            self.hc_post(&mut x, &o, &post, &res, t);
            point(2 * l + 1, &mut x);
            let (xf, post, res) = self.hc_pre(&layer.hc_mlp, &layer.ln_post, &x, t);
            let hook = iv.as_ref().and_then(|s| s.hook(l, start));
            layer.mlp.forward(&xf, t, h, &mut o, hook.as_ref().map(|k| (k, Target::Mlp)));
            self.hc_post(&mut x, &o, &post, &res, t);
            if let (Some(g), Some(inputs)) = (&layer.ple, &inputs) {
                self.ple_add(g, inputs, l, &mut x, t);
            }
            point(2 * l + 2, &mut x);
        }
        capture(self.layers.len(), &x);
        let mut out = self.mean(&x, t);
        rmsnorm_f32w_rows_inplace(&mut out, h, &self.final_norm, c.rms_norm_eps as f32);
        sess.pos += t;
        sess.rec = rec.map(Box::new);
        out
    }

    /// Drops the positions at or beyond `len`: the MLA caches are cut, and the recurrent states
    /// go back to those before the last recorded pass, whose kept tokens are replayed (their
    /// convolutions and delta rule only). Panics when `len` is before that pass (see
    /// [`RaonSession::can_truncate`]).
    pub fn truncate(&self, sess: &mut RaonSession, len: usize) {
        if len >= sess.pos {
            return;
        }
        let mut rec = match sess.rec.take() {
            Some(r) if len >= r.pos => r,
            _ => panic!("Raon: the recurrent (RDA) state cannot go back before the last recorded pass"),
        };
        let c = &self.cfg;
        let dm = c.rda_dims();
        let (nh, d) = (dm.nh, dm.nh * dm.hd);
        let n = len - rec.pos;
        for (l, kr) in &mut rec.rda {
            let (Attn::Rda(kd), LayerState::Rda(st)) = (&self.layers[*l].attn, &mut sess.layers[*l]) else {
                unreachable!("an RDA record of another layer kind")
            };
            st.s.copy_from_slice(&kr.s0);
            st.conv.copy_from_slice(&kr.conv0);
            kr.raw.iter_mut().for_each(|r| r.truncate(n * d));
            kr.f.truncate(n * d);
            kr.beta.truncate(n * nh);
            kr.write.truncate(n * nh);
            if n > 0 {
                kda_scan(&dm, kd, st, &kr.raw, &kr.f, &kr.beta, &kr.write, n, None);
            }
        }
        for st in &mut sess.layers {
            if let LayerState::Mla(cache) = st {
                cache.c.truncate(len * c.kv_lora_rank);
                cache.r.truncate(len * c.qk_rope_head_dim);
                cache.len = cache.len.min(len);
            }
        }
        sess.pos = len;
        // (Still good for a rollback further into the pass.)
        sess.rec = Some(rec);
    }
}

impl RaonSession {
    /// Whether the session can go back to `len` positions: the current length, or a position
    /// inside the last recorded pass.
    pub fn can_truncate(&self, len: usize) -> bool {
        len >= self.pos || self.rec.as_ref().is_some_and(|r| len >= r.pos)
    }
}

impl crate::generate::TextLm for RaonModel<'_> {
    type Sess = RaonSession;
    fn new_session(&self) -> RaonSession {
        RaonModel::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut RaonSession) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &RaonSession) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut RaonSession, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        RaonModel::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        RaonModel::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut RaonSession, tokens: &[u32]) -> Vec<f32> {
        RaonModel::forward(self, sess, tokens)
    }
}

/// The interface speculative decoding rounds use ([`crate::dspark::Target`]: prompt lookup; no
/// draft exists for this model). Its passes are recorded, so they can be rolled back.
impl crate::dspark::Target for RaonModel<'_> {
    fn forward_feats(
        &self,
        sess: &mut RaonSession,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, layers, feats, true)
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        if let Some(s) = &self.stream {
            s.need_ranges(&self.embed.row_ranges(token as usize, h));
        }
        self.embed.row(token as usize, out);
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        RaonModel::logits(self, hidden, t)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut RaonSession, len: usize) {
        RaonModel::truncate(self, sess, len);
    }
}

/// Runtime interventions (`crate::intervene`): the points around every sublayer act on the
/// streams ([`crate::intervene::State::point_copies`]; captures read their mean, which is what
/// the modeling code gives as hidden states), and as neurons the input of the MLP's `down_proj`.
impl crate::intervene::Host for RaonModel<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        (target == Target::Mlp).then_some(self.layers.get(layer)?.mlp.inter)
    }
    fn down_column(&self, layer: usize, _target: Target, n: usize) -> Vec<f32> {
        let m = &self.layers[layer].mlp;
        let h = self.cfg.hidden_size;
        let mut one = vec![0f32; m.inter];
        one[n] = 1.0;
        let mut y = vec![0f32; h];
        linear(&m.down, h, m.inter, &one, 1, &mut y);
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_config() {
        let c = RaonConfig::from_json(serde_json::json!({
            "model_type": "raon1_nano", "hidden_size": 1536, "num_hidden_layers": 24, "num_attention_heads": 32,
            "head_dim": 128, "vocab_size": 259840, "tie_word_embeddings": true, "rms_norm_eps": 1e-6,
            "rope_parameters": null, "hidden_act": "silu", "intermediate_size": 6144, "q_lora_rank": 512,
            "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "mla_use_nope": true, "mla_use_output_gate": true, "raon_mhc_streams": 4,
            "rda_config": {
                "num_heads": 16, "head_dim": 128, "conv_kernel_size": 4, "full_rank_gate": true, "gate_lower_bound": -5.0,
                "mla_layers": [4, 8, 12, 16, 20, 24],
                "rda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23],
            },
            "per_layer_embedding_dim": 256, "per_layer_embedding_layout_version": 2,
        }))
        .unwrap();
        assert_eq!((0..24).filter(|&l| c.is_rda(l)).count(), 18);
        assert!(c.is_rda(0) && !c.is_rda(3) && !c.is_rda(23));
        let (r, m) = (c.rda_dims(), c.mla_dims());
        assert_eq!((r.nh, r.hd, r.rank, r.kc, r.lower), (16, 128, 128, 4, -5.0));
        assert_eq!((m.nh, m.lat, m.dn, m.dr, m.vd, m.ql), (32, 512, 128, 64, 128, 512));
        assert_eq!(c.streams(), 4);
        // What is not implemented, or does not make a model, is refused.
        let bad = |k: &str, v: serde_json::Value| {
            let mut raw = tiny_json();
            match k.strip_prefix("rda.") {
                Some(k) => raw["rda_config"][k] = v,
                None => raw[k] = v,
            }
            RaonConfig::from_json(raw).is_err()
        };
        assert!(bad("rda.gate_lower_bound", serde_json::Value::Null));
        assert!(bad("rda.rda_layers", serde_json::json!([1, 2, 9])));
        assert!(bad("rda.mla_layers", serde_json::json!([1, 4])));
        assert!(bad("raon_mhc_streams", 1.into()));
        assert!(bad("per_layer_embedding_layout_version", 1.into()));
    }

    #[test]
    fn helmert() {
        for n in 2..6 {
            let q = helmert_basis(n);
            let nm = n - 1;
            for a in 0..nm {
                // Columns are orthonormal and sum to zero.
                assert!((0..n).map(|i| q[i * nm + a]).sum::<f32>().abs() < 1e-6);
                for b in 0..nm {
                    let d: f32 = (0..n).map(|i| q[i * nm + a] * q[i * nm + b]).sum();
                    assert!((d - f32::from(a == b)).abs() < 1e-6);
                }
            }
        }
    }

    /// The config of the tiny model: RDA and MLA layers, four streams, per-layer embeddings.
    fn tiny_json() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 48, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 5,
            "num_attention_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6,
            "q_lora_rank": 12, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8,
            "mla_use_output_gate": true, "raon_mhc_streams": 4,
            "rda_config": {
                "num_heads": 3, "head_dim": 8, "conv_kernel_size": 4, "gate_lower_bound": -5.0,
                "mla_layers": [3, 5], "rda_layers": [1, 2, 4],
            },
            "rda_decay_rank": 6, "per_layer_embedding_dim": 4, "per_layer_embedding_layout_version": 2,
        })
    }

    fn tiny_config(over: serde_json::Value) -> RaonConfig {
        let mut c = tiny_json();
        for (k, v) in over.as_object().unwrap() {
            c[k] = v.clone();
        }
        RaonConfig::from_json(c).unwrap()
    }

    /// Writes a tiny random checkpoint of [`tiny_config`] in KRAFTON's naming (float32).
    fn tiny_checkpoint(dir: &std::path::Path) {
        let c = tiny_config(serde_json::json!({}));
        let (h, nh, v, n, nl) = (c.hidden_size, c.num_attention_heads, c.vocab_size, c.streams(), c.num_hidden_layers);
        let (r, m, pd) = (c.rda_dims(), c.mla_dims(), c.per_layer_embedding_dim);
        let d = r.nh * r.hd;
        let mut seed = 0x72616f6eu64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        // `scale` times uniform noise, plus `bias`.
        let mut add = |name: String, shape: &[usize], scale: f32, bias: f32| {
            let n: usize = shape.iter().product();
            let bytes = (0..n).flat_map(|_| (bias + scale * rng()).to_le_bytes()).collect();
            tensors.push((format!("model.{name}"), shape.to_vec(), bytes));
        };
        let mat = |fan_in: usize, gain: f32| gain * (3.0 / fan_in as f32).sqrt();
        add("embed_tokens.weight".into(), &[v, h], 1.7, 0.0);
        add("norm.weight".into(), &[h], 0.2, 1.0);
        add("per_layer_inputs.table".into(), &[v, nl * pd], 0.5, 0.0);
        add("per_layer_inputs.embedding_projection".into(), &[nl * pd, h], mat(h, 1.0), 0.0);
        add("per_layer_inputs.projection_norm.weight".into(), &[pd], 0.2, 1.0);
        for l in 0..nl {
            let p = format!("layers.{l}.");
            add(format!("{p}input_layernorm.weight"), &[h], 0.2, 1.0);
            add(format!("{p}post_attention_layernorm.weight"), &[h], 0.2, 1.0);
            for site in ["self_attn_raon_mhc", "mlp_raon_mhc"] {
                let sp = format!("{p}{site}.");
                add(format!("{sp}phi_pre"), &[n, n * h], mat(n * h, 1.0), 0.0);
                add(format!("{sp}phi_post"), &[n, n * h], mat(n * h, 1.0), 0.0);
                add(format!("{sp}res.phi_z"), &[(n - 1) * (n - 1), n * h], mat(n * h, 1.0), 0.0);
                for a in ["alpha_pre", "alpha_post", "res.alpha_z"] {
                    add(format!("{sp}{a}"), &[], 0.3, 1.0);
                }
                add(format!("{sp}b_pre"), &[n], 0.3, 0.0);
                add(format!("{sp}b_post"), &[n], 0.3, 0.0);
                add(format!("{sp}res.b_z"), &[(n - 1) * (n - 1)], 0.3, 0.0);
                add(format!("{sp}res.eta"), &[], 1.0, 0.0);
            }
            let ap = format!("{p}self_attn.");
            if c.is_rda(l) {
                for q in ["q", "k", "v"] {
                    add(format!("{ap}{q}_proj.weight"), &[d, h], mat(h, 1.5), 0.0);
                    add(format!("{ap}{q}_conv1d.weight"), &[d, r.kc], 0.7, 0.0);
                }
                add(format!("{ap}A_log"), &[r.nh], 0.6, 0.0);
                add(format!("{ap}dt_bias"), &[d], 0.5, 0.0);
                add(format!("{ap}f_a_proj.weight"), &[r.rank, h], mat(h, 1.0), 0.0);
                add(format!("{ap}f_b_proj.weight"), &[d, r.rank], mat(r.rank, 2.0), 0.0);
                add(format!("{ap}b_proj.weight"), &[r.nh, h], mat(h, 1.5), 0.0);
                add(format!("{ap}rda_w_proj.weight"), &[r.nh, h], mat(h, 3.0), 0.0);
                add(format!("{ap}g_proj.weight"), &[d, h], mat(h, 1.5), 0.0);
                add(format!("{ap}o_norm.weight"), &[r.hd], 0.2, 1.0);
                add(format!("{ap}o_proj.weight"), &[h, d], mat(d, 1.0), 0.0);
            } else {
                add(format!("{ap}q_a_proj.weight"), &[m.ql, h], mat(h, 1.0), 0.0);
                add(format!("{ap}q_a_layernorm.weight"), &[m.ql], 0.2, 1.0);
                add(format!("{ap}q_b_proj.weight"), &[nh * (m.dn + m.dr), m.ql], mat(m.ql, 2.0), 0.0);
                add(format!("{ap}kv_a_proj_with_mqa.weight"), &[m.lat + m.dr, h], mat(h, 1.0), 0.0);
                add(format!("{ap}kv_a_layernorm.weight"), &[m.lat], 0.2, 1.0);
                add(format!("{ap}kv_b_proj.weight"), &[nh * (m.dn + m.vd), m.lat], mat(m.lat, 2.0), 0.0);
                add(format!("{ap}g_proj.weight"), &[nh * m.vd, h], mat(h, 1.5), 0.0);
                add(format!("{ap}o_proj.weight"), &[h, nh * m.vd], mat(nh * m.vd, 1.0), 0.0);
            }
            let inter = c.intermediate_size;
            add(format!("{p}mlp.gate_proj.weight"), &[inter, h], mat(h, 1.0), 0.0);
            add(format!("{p}mlp.up_proj.weight"), &[inter, h], mat(h, 1.0), 0.0);
            add(format!("{p}mlp.down_proj.weight"), &[h, inter], mat(inter, 1.0), 0.0);
            add(format!("{p}per_layer_input_gate.gate"), &[pd, h], mat(h, 1.0), 0.0);
            add(format!("{p}per_layer_input_gate.projection"), &[h, pd], mat(pd, 1.0), 0.0);
            add(format!("{p}per_layer_input_gate.output_norm.weight"), &[h], 0.2, 1.0);
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

    /// A tiny random model, without a reference (tools/ref_raon.py measures the agreement with
    /// KRAFTON's code): the logits do not depend on how the tokens are split into passes, a
    /// rollback inside a recorded pass leaves no trace, the mixing matrices are doubly
    /// stochastic, and each mechanism changes the logits.
    #[test]
    fn tiny_model() {
        let dir = std::env::temp_dir().join(format!("oxinfer-raon-test-{}", std::process::id()));
        tiny_checkpoint(&dir);
        let files = ModelFiles::open(&dir, true).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let load = |over: serde_json::Value| RaonModel::load(tiny_config(over), &files, Quant::Bf16).unwrap();
        let m = load(serde_json::json!({}));
        let (h, v, n) = (m.cfg.hidden_size, m.cfg.vocab_size, m.cfg.streams());
        let ids: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % v as u32).collect();
        let more = [5u32, 17, 2, 9];
        // The logits after the prompt (in passes of `chunk`) and after each of `more`.
        let run = |m: &RaonModel, chunk: usize| -> Vec<f32> {
            let mut sess = m.new_session();
            let hs = crate::generate::prefill(m, &mut sess, &ids, chunk, &[]);
            let mut out = m.logits(&hs[hs.len() - h..], 1);
            for &t in &more {
                out.extend(m.forward(&mut sess, &[t]));
            }
            out
        };
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let base = run(&m, 40);
        assert!(base.iter().all(|x| x.is_finite()));
        let scale = base.iter().fold(0f32, |a, b| a.max(b.abs()));
        let tol = 2e-3 * scale;
        for chunk in [17, 7, 1] {
            let d = diff(&run(&m, chunk), &base);
            assert!(d <= tol, "passes of {chunk} differ from one pass by {d} (logits up to {scale})");
        }
        // A rollback inside a recorded pass: the tokens after it run as if the dropped ones never had.
        use crate::dspark::Target as _;
        let mut sess = m.new_session();
        m.forward(&mut sess, &ids[..30]);
        m.forward_feats(&mut sess, &ids[30..], &[], &[], &mut Vec::new());
        m.forward_feats(&mut sess, &[30, 31, 29], &[], &[], &mut Vec::new());
        assert!(sess.can_truncate(ids.len() + 1) && !sess.can_truncate(ids.len() - 1));
        m.truncate(&mut sess, ids.len() + 1);
        assert_eq!(sess.pos, ids.len() + 1);
        m.truncate(&mut sess, ids.len());
        let again: Vec<f32> = more.iter().flat_map(|&t| m.forward(&mut sess, &[t])).collect();
        let d = diff(&again, &base[v..]);
        assert!(d <= tol, "after a rollback the logits differ by {d}");
        // A site's mixing matrix has non-negative entries, rows and columns summing to one.
        let x: Vec<f32> = (0..n * h).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let (mut pre, mut post, mut res) = (vec![0f32; n], vec![0f32; n], vec![0f32; n * n]);
        for layer in &m.layers {
            m.route(&layer.hc_attn, &x, &mut pre, &mut post, &mut res);
            for i in 0..n {
                let row: f32 = res[i * n..(i + 1) * n].iter().sum();
                let col: f32 = (0..n).map(|j| res[j * n + i]).sum();
                assert!((row - 1.0).abs() < 1e-5 && (col - 1.0).abs() < 1e-5, "sums {row} {col}");
            }
            assert!(res.iter().all(|&e| e >= -1e-6));
            assert!(pre.iter().all(|&p| (0.0..=1.0).contains(&p)) && post.iter().all(|&p| (0.0..=2.0).contains(&p)));
        }
        // Each mechanism matters.
        for (what, over) in [
            ("the write-back factor", serde_json::json!({"raon_mhc_post_factor": 1.0})),
            ("the selector's epsilon", serde_json::json!({"raon_mhc_selector_eps": 0.5})),
            ("the MLA output gate", serde_json::json!({"mla_use_output_gate": false})),
            ("the per-layer embeddings", serde_json::json!({"per_layer_embedding_dim": 0})),
            ("the latent norm's epsilon", serde_json::json!({"mla_latent_norm_eps": 0.5})),
        ] {
            let d = diff(&run(&load(over), 40), &base);
            assert!(d > 20.0 * tol, "{what}: the logits moved by {d} only");
        }
    }
}
