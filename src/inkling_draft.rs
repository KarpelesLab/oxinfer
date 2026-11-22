//! Inkling-2-Mini's block drafter (`mtp.safetensors` of `thinkingmachines/Inkling-2-Mini`, and
//! its llama.cpp conversion `dspark-Inkling-2-Mini-*.gguf` of `unsloth/Inkling-2-Mini-GGUF`,
//! GGUF architecture `dflash`), run as a DSpark draft ([`crate::dspark`]). As llama.cpp's
//! `graph_inkling2` (ggml-org/llama.cpp PR 32218, "Inkling 2 DSpark support", `src/models/
//! dspark.cpp`; transformers does not have it):
//!
//! - **Context.** For every position the target ran, the residual streams entering four of its
//!   layer applications (`target_layer_ids` 5, 23, 41 of the 48 of a pass, and 48: the
//!   final-normed output) are mixed per draft layer with the softmax of a row of `fuse_weight`
//!   (`[layers + 1, 4]`), RMS-normed with the layer's `dflare_norm`, and projected by the
//!   layer's own context projections (`k_proj_ctx`, its key RMS-normed per head with
//!   `k_norm_ctx`, then RoPE; `v_proj_ctx`) into its key / value cache. The last row of
//!   `fuse_weight` mixes the streams for the head (normed with `markov_head.dflare_norm`).
//! - **Block.** The anchor (the committed token the target has not run, at position `n`; its
//!   embedding RMS-normed as the target's input) and 7 copies of `mask_embed` at `n + 1..n + 7`
//!   go through 3 pre-norm layers (sliding, full, sliding: the window of 1024 positions before
//!   the anchor), each with queries and keys RMS-normed per head, RoPE (rotate-half, theta 1e6,
//!   all 128 dims), non-causal attention over the cached context and the whole block, and a
//!   SiLU MLP; then `norm` and the target's LM head (`base`).
//! - **Head** (`markov_head`, "rnn_hidden"): slot `i` (1..7) updates a state of rank 256 from
//!   `[state, markov_w1[prev], hidden_i, context]` (`rnn_proj`, `[768, 5120]`; the context is
//!   the head's mix of the features of position `n - 1`): `gate = sigmoid(c0)`, `cand =
//!   tanh(c1)`, `state = cand + gate (state - cand)`; its logits are `30 tanh((base_i + head
//!   (hidden_out tanh(c2) + b)) / 30)`, and `prev` is the slot's token. The head's correction
//!   goes through the target's LM head, which is linear: it runs here as one product with
//!   `head * hidden_out` (`[vocab, 256]`, computed once from the target's LM head).

use crate::dspark::Target;
use crate::kernels::*;
use crate::model::{Lin, Quant, Table, linear_multi, linear_rows_device, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// The drafter's shapes.
#[derive(Clone, Debug)]
pub struct InkDraftConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub eps: f32,
    pub theta: f64,
    /// Per layer: context positions before the anchor its block sees (0: all).
    pub windows: Vec<usize>,
    pub block_size: usize,
    pub target_layers: Vec<usize>,
    pub softcap: f32,
}

impl InkDraftConfig {
    /// `mtp_config` of the release's `config.json`.
    pub fn from_hf(m: &serde_json::Value) -> Result<Self> {
        let int = |k: &str, d: Option<usize>| -> Result<usize> {
            match m.get(k).and_then(serde_json::Value::as_u64) {
                Some(v) => Ok(v as usize),
                None => d.with_context(|| format!("mtp_config.{k} is missing")),
            }
        };
        if m.get("markov_head_type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|t| t != "rnn_hidden")
        {
            bail!("Inkling-2 MTP: markov_head_type {:?} is not supported", m["markov_head_type"]);
        }
        let n_layers = int("num_hidden_layers", Some(3))?;
        let window = int("sliding_window_size", Some(1024))?;
        let local: Vec<usize> = m
            .get("local_layer_ids")
            .and_then(serde_json::Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect())
            .unwrap_or_default();
        let c = Self {
            hidden: int("hidden_size", None)?,
            n_layers,
            n_heads: int("num_attention_heads", None)?,
            n_kv: int("num_key_value_heads", None)?,
            head_dim: int("head_dim", Some(128))?,
            inter: int("intermediate_size", None)?,
            eps: m.get("rms_norm_eps").and_then(serde_json::Value::as_f64).unwrap_or(1e-6) as f32,
            theta: m.get("rope_theta").and_then(serde_json::Value::as_f64).unwrap_or(1e6),
            windows: (0..n_layers).map(|l| if local.contains(&l) { window } else { 0 }).collect(),
            block_size: int("block_size", Some(8))?,
            target_layers: m
                .get("target_layer_ids")
                .and_then(serde_json::Value::as_array)
                .context("mtp_config.target_layer_ids is missing")?
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect(),
            softcap: 30.0,
        };
        c.check()
    }

    /// A llama.cpp `dflash` GGUF of the drafter (its `dspark_fuse` tensor tells it apart).
    pub fn from_header(h: &crate::gguf::Header) -> Result<Self> {
        use crate::gguf::Value;
        let get = |k: &str| h.get(&format!("dflash.{k}"));
        let u = |k: &str| match get(k) {
            Some(Value::Array(a)) => a.first().and_then(Value::as_u64).map(|v| v as usize),
            v => v.and_then(Value::as_u64).map(|v| v as usize),
        };
        let need = |k: &str| u(k).with_context(|| format!("Inkling-2 draft GGUF has no dflash.{k}"));
        let n_layers = need("block_count")?;
        let window = u("attention.sliding_window").unwrap_or(0);
        let pattern: Vec<bool> = get("attention.sliding_window_pattern")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| matches!(v, Value::Bool(true))).collect())
            .unwrap_or_default();
        let c = Self {
            hidden: need("embedding_length")?,
            n_layers,
            n_heads: need("attention.head_count")?,
            n_kv: need("attention.head_count_kv")?,
            head_dim: need("attention.key_length")?,
            inter: need("feed_forward_length")?,
            eps: get("attention.layer_norm_rms_epsilon").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            theta: get("rope.freq_base").and_then(Value::as_f64).unwrap_or(1e6),
            windows: (0..n_layers)
                .map(|l| if pattern.get(l).copied().unwrap_or(false) { window } else { 0 })
                .collect(),
            block_size: need("block_size")?,
            target_layers: get("target_layers")
                .and_then(Value::as_array)
                .context("Inkling-2 draft GGUF has no dflash.target_layers")?
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect(),
            softcap: get("final_logit_softcapping").and_then(Value::as_f64).unwrap_or(30.0) as f32,
        };
        c.check()
    }

    fn check(self) -> Result<Self> {
        if self.block_size < 2
            || self.n_kv == 0
            || !self.n_heads.is_multiple_of(self.n_kv)
            || !self.head_dim.is_multiple_of(2)
            || self.target_layers.is_empty()
        {
            bail!("bad Inkling-2 draft config: {self:?}");
        }
        Ok(self)
    }

    /// As a DSpark config (the decoder's view: block, drafts, target layers).
    pub fn dspark(&self) -> crate::dspark::DsparkConfig {
        use crate::dspark::{Backbone, DsparkConfig, GqaConfig, MaskEmbed};
        DsparkConfig {
            hidden: self.hidden,
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            inter: self.inter,
            eps: self.eps,
            backbone: Backbone::Gqa(GqaConfig {
                n_kv: self.n_kv,
                head_dim: self.head_dim,
                k_eq_v: false,
                v_norm: false,
                sandwich: false,
                gelu: false,
                oai: false,
                attn_scale: (self.head_dim as f32).powf(-0.5),
                rope_dims: self.head_dim,
                rope_proportion: 1.0,
            }),
            windows: self.windows.clone(),
            causal: vec![false; self.n_layers],
            block_size: self.block_size,
            sample_from_anchor: false,
            target_layers: self.target_layers.clone(),
            softcap: Some(self.softcap),
            rope: crate::config::YarnParams {
                rope_theta: self.theta,
                rope_type: Some("default".into()),
                factor: None,
                original_max_position_embeddings: None,
                beta_fast: None,
                beta_slow: None,
                attention_factor: None,
                mscale: None,
                mscale_all_dim: None,
                truncate: None,
            },
            mask: MaskEmbed::Learned,
            embed_scale: Some(1.0),
            bridge: None,
        }
    }
}

/// Whether `files` hold the drafter (a GGUF or the release's `mtp.safetensors`).
pub fn is_ink_draft(files_has: impl Fn(&str) -> bool) -> bool {
    files_has("dspark_fuse.weight") || files_has("fuse_weight")
}

/// The release's `mtp.safetensors` beside a checkpoint folder, when its `config.json` describes
/// the drafter (`mtp_config`).
pub fn mtp_file(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let f = dir.join("mtp.safetensors");
    (f.is_file() && mtp_config(&f).is_some()).then_some(f)
}

/// The `mtp_config` of the checkpoint holding `mtp.safetensors` at `path`.
pub fn mtp_config(path: &std::path::Path) -> Option<serde_json::Value> {
    if path.file_name()? != "mtp.safetensors" {
        return None;
    }
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(path.parent()?.join("config.json")).ok()?).ok()?;
    let mt = raw.get("model_type").and_then(serde_json::Value::as_str).unwrap_or("");
    if !crate::inkling::is_model_type(mt) {
        return None;
    }
    raw.get("mtp_config").filter(|v| v.is_object()).cloned()
}

/// A layer's keys and values of some positions (`[t, n_kv * hd]` each).
pub type KeysValues = (Vec<f32>, Vec<f32>);

pub(crate) struct InkLayer<'a> {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) ctx_norm: Vec<f32>,
    pub(crate) q: Lin<'a>,
    pub(crate) k: Lin<'a>,
    pub(crate) v: Lin<'a>,
    pub(crate) o: Lin<'a>,
    pub(crate) k_ctx: Lin<'a>,
    pub(crate) v_ctx: Lin<'a>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) k_ctx_norm: Vec<f32>,
    pub(crate) gate: Lin<'a>,
    pub(crate) up: Lin<'a>,
    pub(crate) down: Lin<'a>,
}

/// The loaded drafter (weights borrowed from its files).
pub struct InkDraft<'a> {
    pub cfg: InkDraftConfig,
    pub(crate) layers: Vec<InkLayer<'a>>,
    /// Softmaxed rows of `fuse_weight`: per draft layer, then the head (`[layers + 1, n_targets]`).
    pub(crate) fuse: Vec<f32>,
    pub(crate) rnn_norm: Vec<f32>,
    pub(crate) out_norm: Vec<f32>,
    pub(crate) mask: Vec<f32>,
    pub(crate) w1: Table<'a>,
    pub(crate) joint: Lin<'a>,
    pub(crate) joint_b: Vec<f32>,
    /// `hidden_out` as f32 `[H, rank]`, and its bias.
    pub(crate) hid_out: Vec<f32>,
    pub(crate) hid_b: Vec<f32>,
    pub(crate) rank: usize,
    pub(crate) inv_freq: Vec<f32>,
    /// The target's LM head times `hidden_out` (`[vocab, rank]`, bf16) and times its bias, made on
    /// first use.
    pub(crate) proj: std::sync::OnceLock<(Lin<'static>, Vec<f32>)>,
    /// Run the matmuls of a few rows on the accelerator (see [`InkDraft::set_device`]).
    pub(crate) device: std::sync::atomic::AtomicBool,
    /// The drafter on the accelerator (see [`InkDraftDevice`]), made on the first draft pass
    /// that runs there.
    gpu: std::sync::OnceLock<Option<Box<dyn InkDraftDevice>>>,
    /// Rows of `markov_w1` (the padded vocabulary, the rows of the target's LM head).
    pub(crate) w1_rows: usize,
}

/// The drafter on an accelerator (`crate::metal::inkling_draft`): whole draft passes and context
/// injections in one submission each, over a device copy of a session's cache (`slot`).
pub(crate) trait InkDraftDevice: Send + Sync {
    /// Makes the device cache hold positions `from..to` of the host's (`rows`: per layer its
    /// keys and values of those positions).
    fn upload(&self, slot: &mut crate::gpu::decode::Slot, from: usize, to: usize, rows: &[(&[f32], &[f32])]) -> bool;
    /// [`InkDraft::context`] of `t` positions from `pos`, written to the device cache too.
    fn context(&self, slot: &mut crate::gpu::decode::Slot, feats: &[f32], t: usize, pos: usize) -> Option<(Vec<KeysValues>, Vec<f32>)>;
    /// A greedy draft pass: the block (`x0`: its input rows), the target's head over the first
    /// `k` draft rows and the head's chain, each slot's token the first maximum of its logits;
    /// returns the soft-capped logits `[k, vocab]` and the tokens.
    fn draft(
        &self,
        slot: &mut crate::gpu::decode::Slot,
        x0: &[f32],
        context: &[f32],
        pos: usize,
        anchor: u32,
        k: usize,
    ) -> Option<(Vec<f32>, Vec<u32>)>;
}

impl<'a> InkDraft<'a> {
    /// Loads the drafter from `files` (a `dflash` GGUF, `gguf`, or `mtp.safetensors`).
    pub fn load(files: &'a ModelFiles, cfg: InkDraftConfig, gguf: bool, quant: Quant) -> Result<Self> {
        let c = &cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.n_heads, c.n_kv, c.head_dim);
        let (qw, kw) = (nh * hd, nkv * hd);
        let shape = |n: &str, s: &[usize]| -> Result<()> {
            let info = files.info(n).with_context(|| format!("Inkling-2 draft: missing tensor {n}"))?;
            if info.shape != s {
                bail!("Inkling-2 draft: tensor {n} has shape {:?}, expected {s:?}", info.shape);
            }
            Ok(())
        };
        let lin = |n: &str, n_out: usize, n_in: usize| -> Result<Lin<'a>> {
            shape(n, &[n_out, n_in])?;
            load_lin(files, n, n_out, n_in, quant)
        };
        let vecf = |n: &str, len: usize| -> Result<Vec<f32>> {
            shape(n, &[len])?;
            files.f32(n)
        };
        let top = |g: &str, hf: &str| if gguf { g.to_string() } else { hf.to_string() };
        let ly = |l: usize, g: &str, hf: &str| if gguf { format!("blk.{l}.{g}") } else { format!("layers.{l}.{hf}") };
        let mut layers = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            let n = |g: &str, hf: &str| ly(l, g, hf);
            layers.push(InkLayer {
                attn_norm: vecf(&n("attn_norm.weight", "input_layernorm.weight"), h)?,
                ffn_norm: vecf(&n("ffn_norm.weight", "post_attention_layernorm.weight"), h)?,
                ctx_norm: vecf(&n("dflare_norm.weight", "dflare_norm.weight"), h)?,
                q: lin(&n("attn_q.weight", "self_attn.q_proj.weight"), qw, h)?,
                k: lin(&n("attn_k.weight", "self_attn.k_proj.weight"), kw, h)?,
                v: lin(&n("attn_v.weight", "self_attn.v_proj.weight"), kw, h)?,
                o: lin(&n("attn_output.weight", "self_attn.o_proj.weight"), h, qw)?,
                k_ctx: lin(&n("attn_k_ctx.weight", "self_attn.k_proj_ctx.weight"), kw, h)?,
                v_ctx: lin(&n("attn_v_ctx.weight", "self_attn.v_proj_ctx.weight"), kw, h)?,
                q_norm: vecf(&n("attn_q_norm.weight", "self_attn.q_norm.weight"), hd)?,
                k_norm: vecf(&n("attn_k_norm.weight", "self_attn.k_norm.weight"), hd)?,
                k_ctx_norm: vecf(&n("attn_k_ctx_norm.weight", "self_attn.k_norm_ctx.weight"), hd)?,
                gate: lin(&n("ffn_gate.weight", "mlp.gate_proj.weight"), c.inter, h)?,
                up: lin(&n("ffn_up.weight", "mlp.up_proj.weight"), c.inter, h)?,
                down: lin(&n("ffn_down.weight", "mlp.down_proj.weight"), h, c.inter)?,
            });
        }
        let nt = c.target_layers.len();
        let fname = top("dspark_fuse.weight", "fuse_weight");
        shape(&fname, &[c.n_layers + 1, nt])?;
        let mut fuse = files.f32(&fname)?;
        for row in fuse.chunks_mut(nt) {
            softmax_inplace(row);
        }
        let w1n = top("markov_w1.weight", "markov_head.markov_w1.weight");
        let w1i = files.info(&w1n).with_context(|| format!("Inkling-2 draft: missing tensor {w1n}"))?;
        let (vocab_rows, rank) = match w1i.shape[..] {
            [v, r] => (v, r),
            _ => bail!("Inkling-2 draft: {w1n} has shape {:?}", w1i.shape),
        };
        let hon = top("dspark_hidden_out.weight", "markov_head.hidden_out.weight");
        shape(&hon, &[h, rank])?;
        let hid_out = files.f32(&hon)?;
        Ok(Self {
            layers,
            fuse,
            rnn_norm: vecf(&top("dspark_norm.weight", "markov_head.dflare_norm.weight"), h)?,
            out_norm: vecf(&top("output_norm.weight", "norm.weight"), h)?,
            mask: vecf(&top("mask_embd.weight", "mask_embed"), h)?,
            w1: Table::load(files, &w1n, vocab_rows, rank, Quant::Bf16)?,
            joint: lin(&top("joint_proj.weight", "markov_head.rnn_proj.weight"), 3 * rank, 2 * rank + 2 * h)?,
            joint_b: vecf(&top("joint_proj.bias", "markov_head.rnn_proj.bias"), 3 * rank)?,
            hid_out,
            hid_b: vecf(&top("dspark_hidden_out.bias", "markov_head.hidden_out.bias"), h)?,
            rank,
            inv_freq: (0..hd / 2)
                .map(|i| 1.0 / (c.theta as f32).powf((2 * i) as f32 / hd as f32))
                .collect(),
            proj: std::sync::OnceLock::new(),
            device: std::sync::atomic::AtomicBool::new(false),
            gpu: std::sync::OnceLock::new(),
            w1_rows: vocab_rows,
            cfg,
        })
    }

    /// Runs the matmuls of up to 16 rows on the accelerator's matrix-vector kernels (4 rows per
    /// submission) when one is active, else on the CPU.
    pub fn set_device(&self, on: bool) {
        self.device.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// `linear_multi`, on the accelerator for a few rows when the draft runs there.
    fn lin(&self, jobs: &mut [(&Lin, usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
        if t <= 16 && self.device.load(std::sync::atomic::Ordering::Relaxed) {
            let mut ok = true;
            for r0 in (0..t).step_by(4) {
                let n = (t - r0).min(4);
                let mut outs: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0f32; n * j.1]).collect();
                let mut dj: Vec<(&Lin, Option<&Lin>, usize, &mut [f32])> = jobs
                    .iter()
                    .zip(outs.iter_mut())
                    .map(|(j, o)| (j.0, None, j.1, &mut o[..]))
                    .collect();
                if !linear_rows_device(&mut dj, n_in, &x[r0 * n_in..(r0 + n) * n_in], n) {
                    ok = false;
                    break;
                }
                for (j, o) in jobs.iter_mut().zip(&outs) {
                    j.2[r0 * j.1..(r0 + n) * j.1].copy_from_slice(o);
                }
            }
            if ok {
                return;
            }
        }
        linear_multi(jobs, n_in, x, t);
    }

    /// Weight bytes one draft pass reads (without the target's LM head and the head's product).
    pub fn pass_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down]
                    .iter()
                    .map(|w| w.bytes())
                    .sum::<usize>()
            })
            .sum::<usize>()
            + self.joint.bytes()
    }

    /// The streams of `feats` (`[t, n_targets * H]`) mixed by row `row` of the fuse weights,
    /// RMS-normed with `norm`: `[t, H]`.
    fn mix(&self, feats: &[f32], t: usize, row: usize, norm: &[f32]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let nt = self.cfg.target_layers.len();
        let w = &self.fuse[row * nt..(row + 1) * nt];
        let mut out = vec![0f32; t * h];
        for_chunks(&mut out, h, t > 1, |ti, o| {
            for (j, &wj) in w.iter().enumerate() {
                axpy(o, wj, &feats[(ti * nt + j) * h..(ti * nt + j + 1) * h]);
            }
            rmsnorm_f32w_inplace(o, norm, self.cfg.eps);
        });
        out
    }

    /// RMS norm of each head (`hd` values) with `w`, then RoPE at `pos0 + row` (heads of `heads`
    /// per row).
    fn norm_rope(&self, x: &mut [f32], heads: usize, w: &[f32], pos0: usize) {
        let hd = self.cfg.head_dim;
        let rows = x.len() / (heads * hd);
        let positions: Vec<usize> = (pos0..pos0 + rows).collect();
        let (cos, sin) = rope_tables(&positions, &self.inv_freq);
        let half = hd / 2;
        for_chunks(x, hd, true, |i, v| {
            let r = i / heads;
            rmsnorm_f32w_inplace(v, w, self.cfg.eps);
            apply_rope(v, &cos[r * half..(r + 1) * half], &sin[r * half..(r + 1) * half]);
        });
    }

    /// The context entries of `t` positions from `pos`: per layer its keys and values
    /// (`[t, n_kv * hd]` each), and the head's context rows (`[t, H]`).
    pub fn context(&self, feats: &[f32], t: usize, pos: usize) -> (Vec<KeysValues>, Vec<f32>) {
        let c = &self.cfg;
        let (h, kw) = (c.hidden, c.n_kv * c.head_dim);
        let mut kv = Vec::with_capacity(self.layers.len());
        for (li, l) in self.layers.iter().enumerate() {
            let ctx = self.mix(feats, t, li, &l.ctx_norm);
            let (mut k, mut v) = (vec![0f32; t * kw], vec![0f32; t * kw]);
            self.lin(&mut [(&l.k_ctx, kw, &mut k[..]), (&l.v_ctx, kw, &mut v[..])], h, &ctx, t);
            self.norm_rope(&mut k, c.n_kv, &l.k_ctx_norm, pos);
            kv.push((k, v));
        }
        (kv, self.mix(feats, t, self.layers.len(), &self.rnn_norm))
    }

    /// The block `[anchor, mask x (block - 1)]` at positions `pos..` through the layers, the
    /// context of layer `l` being `ctx(l, lo)` (its keys and values of positions `lo..pos`);
    /// returns the final-normed states `[block, H]`.
    pub fn block<'c, M: Target + ?Sized>(
        &self,
        target: &M,
        anchor: u32,
        pos: usize,
        ctx: &dyn Fn(usize, usize) -> (&'c [f32], &'c [f32]),
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.n_heads, c.n_kv, c.head_dim);
        let (qw, kw, nb) = (nh * hd, nkv * hd, c.block_size);
        let mut x = vec![0f32; nb * h];
        target.embed_row(anchor, &mut x[..h]);
        for i in 1..nb {
            x[i * h..(i + 1) * h].copy_from_slice(&self.mask);
        }
        let mut hn = vec![0f32; nb * h];
        let scale = (hd as f32).powf(-0.5);
        let group = nh / nkv;
        for (li, l) in self.layers.iter().enumerate() {
            rmsnorm_f32w_rows(&x, h, &l.attn_norm, c.eps, &mut hn);
            let (mut q, mut k, mut v) = (vec![0f32; nb * qw], vec![0f32; nb * kw], vec![0f32; nb * kw]);
            self.lin(
                &mut [(&l.q, qw, &mut q[..]), (&l.k, kw, &mut k[..]), (&l.v, kw, &mut v[..])],
                h,
                &hn,
                nb,
            );
            self.norm_rope(&mut q, nh, &l.q_norm, pos);
            self.norm_rope(&mut k, nkv, &l.k_norm, pos);
            let lo = match c.windows[li] {
                0 => 0,
                w => pos.saturating_sub(w),
            };
            let (ck, cv) = ctx(li, lo);
            let n_ctx = ck.len() / kw;
            let mut att = vec![0f32; nb * qw];
            let ap = SendPtr(att.as_mut_ptr());
            crate::pool::run(nb * nh, &|task| {
                let (i, hh) = (task / nh, task % nh);
                let g = hh / group;
                let qh = &q[(i * nh + hh) * hd..(i * nh + hh + 1) * hd];
                let row = |j: usize| -> (&[f32], &[f32]) {
                    if j < n_ctx {
                        let o = j * kw + g * hd;
                        (&ck[o..o + hd], &cv[o..o + hd])
                    } else {
                        let o = (j - n_ctx) * kw + g * hd;
                        (&k[o..o + hd], &v[o..o + hd])
                    }
                };
                let mut s: Vec<f32> = (0..n_ctx + nb).map(|j| dot(qh, row(j).0) * scale).collect();
                softmax_inplace(&mut s);
                // SAFETY: each (row, head) task writes its own slice.
                let out = unsafe { std::slice::from_raw_parts_mut(ap.get().add((i * nh + hh) * hd), hd) };
                for (j, &p) in s.iter().enumerate() {
                    axpy(out, p, row(j).1);
                }
            });
            let mut o = vec![0f32; nb * h];
            self.lin(&mut [(&l.o, h, &mut o[..])], qw, &att, nb);
            add_inplace(&mut x, &o);
            rmsnorm_f32w_rows(&x, h, &l.ffn_norm, c.eps, &mut hn);
            let (mut gt, mut up) = (vec![0f32; nb * c.inter], vec![0f32; nb * c.inter]);
            self.lin(&mut [(&l.gate, c.inter, &mut gt[..]), (&l.up, c.inter, &mut up[..])], h, &hn, nb);
            silu_mul(&mut gt, &up);
            self.lin(&mut [(&l.down, h, &mut o[..])], c.inter, &gt, nb);
            add_inplace(&mut x, &o);
        }
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.out_norm, c.eps);
        x
    }

    /// Makes what the first draft pass would otherwise make (the head's product with the
    /// target's LM head, and the accelerator's drafter): run at the end of a prefill, so that
    /// decoding starts at full speed.
    pub(crate) fn warm<M: Target + ?Sized>(&self, target: &M) {
        if self.device_for(target).is_none() {
            self.projection(target);
        }
    }

    /// The head's product with the target's LM head (see the module docs), made once.
    fn projection<M: Target + ?Sized>(&self, target: &M) -> &(Lin<'static>, Vec<f32>) {
        self.proj.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let (h, r) = (self.cfg.hidden, self.rank);
            // Rows of hidden_out^T (one per rank), then its bias: `[r + 1, H]` through the head.
            let mut rows = vec![0f32; (r + 1) * h];
            for i in 0..h {
                for j in 0..r {
                    rows[j * h + i] = self.hid_out[i * r + j];
                }
            }
            rows[r * h..].copy_from_slice(&self.hid_b);
            let y = target.head_raw(&rows, r + 1);
            let t1 = t0.elapsed().as_secs_f64() * 1e3;
            let v = y.len() / (r + 1);
            // `[r, v]` to bf16 `[v, r]`: tiles of 256 tokens in parallel, each reading its rows of y
            // contiguously (a plain loop over j then t wrote 51M scattered values: ~200 ms).
            let mut w = crate::quant::PageVec::<u16>::zeroed(v * r);
            const TILE: usize = 256;
            let wp = SendU16(w.as_mut_ptr());
            crate::pool::run(v.div_ceil(TILE), &|ti| {
                let (t0, t1) = (ti * TILE, (ti * TILE + TILE).min(v));
                // SAFETY: tile ti writes rows t0..t1 of w only.
                let out = unsafe { std::slice::from_raw_parts_mut(wp.get().add(t0 * r), (t1 - t0) * r) };
                for j in 0..r {
                    for (k, &yv) in y[j * v + t0..j * v + t1].iter().enumerate() {
                        out[k * r + j] = half::bf16::from_f32(yv).to_bits();
                    }
                }
            });
            if std::env::var_os("OXINFER_DEBUG").is_some() {
                eprintln!(
                    "[inkling draft: head product in {:.0} ms ({t1:.0} ms the head)]",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            (Lin::bf16_owned(w), y[r * v..].to_vec())
        })
    }

    /// The logits of draft slot `i` (1-based in the block, `hidden` its final-normed state and
    /// `base` its LM-head logits, raw) after `prev`, updating the head's `state`; `context` is
    /// the head's context row of the anchor's position minus one. Soft-capped, in `base`.
    #[allow(clippy::too_many_arguments)]
    pub fn slot<M: Target + ?Sized>(&self, target: &M, prev: u32, state: &mut [f32], hidden: &[f32], context: &[f32], base: &mut [f32]) {
        let (h, r) = (self.cfg.hidden, self.rank);
        let mut inp = Vec::with_capacity(2 * r + 2 * h);
        inp.extend_from_slice(state);
        let mut w1 = vec![0f32; r];
        self.w1.row(prev as usize, &mut w1);
        inp.extend_from_slice(&w1);
        inp.extend_from_slice(hidden);
        inp.extend_from_slice(context);
        let mut j = vec![0f32; 3 * r];
        // (7.9 MB: not worth a submission.)
        linear_multi(&mut [(&self.joint, 3 * r, &mut j[..])], 2 * r + 2 * h, &inp, 1);
        add_inplace(&mut j, &self.joint_b);
        for e in 0..r {
            let gate = 1.0 / (1.0 + (-j[e]).exp());
            let cand = j[r + e].tanh();
            state[e] = cand + gate * (state[e] - cand);
        }
        let latent: Vec<f32> = j[2 * r..].iter().map(|v| v.tanh()).collect();
        let (w, c) = self.projection(target);
        let v = c.len().min(base.len());
        let mut bias = vec![0f32; c.len()];
        self.lin(&mut [(w, c.len(), &mut bias[..])], r, &latent, 1);
        let cap = self.cfg.softcap;
        for_chunks(&mut base[..v], 8192, true, |ci, ch| {
            for (k, x) in ch.iter_mut().enumerate() {
                let t = ci * 8192 + k;
                *x = ((*x + bias[t] + c[t]) / cap).tanh() * cap;
            }
        });
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The block's input rows: the anchor's embedding (the target's input row) and the mask rows.
    pub fn block_input<M: Target + ?Sized>(&self, target: &M, anchor: u32) -> Vec<f32> {
        let (h, nb) = (self.cfg.hidden, self.cfg.block_size);
        let mut x = vec![0f32; nb * h];
        target.embed_row(anchor, &mut x[..h]);
        for i in 1..nb {
            x[i * h..(i + 1) * h].copy_from_slice(&self.mask);
        }
        x
    }

    /// The drafter on the accelerator, once a draft pass made it (injections then run there too).
    pub(crate) fn device_ready(&self) -> Option<&dyn InkDraftDevice> {
        if !self.device.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        self.gpu.get().and_then(|g| g.as_deref())
    }

    /// The drafter on the accelerator for target `target` (made on first use), when the draft
    /// runs there and the backend has the kernels (`OXINFER_INK_DRAFT_CPU=1` keeps it on the CPU).
    pub(crate) fn device_for<M: Target + ?Sized>(&self, target: &M) -> Option<&dyn InkDraftDevice> {
        if !self.device.load(std::sync::atomic::Ordering::Relaxed) || std::env::var_os("OXINFER_INK_DRAFT_CPU").is_some() {
            return None;
        }
        self.gpu
            .get_or_init(|| {
                #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64"))]
                {
                    let dbg = std::env::var_os("OXINFER_DEBUG").is_some();
                    let Some(head) = target.lm_head() else {
                        if dbg {
                            eprintln!("[inkling draft: the target has no LM head for the device]");
                        }
                        return None;
                    };
                    let (proj, cvec) = self.projection(target);
                    let t0 = std::time::Instant::now();
                    let d = crate::metal::inkling_draft::build(self, head, self.w1_rows, proj, cvec);
                    if dbg {
                        eprintln!(
                            "[inkling draft: on the device: {} ({:.0} ms)]",
                            d.is_some(),
                            t0.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    d
                }
                #[cfg(not(all(feature = "gpu", any(target_os = "macos", target_os = "ios"), target_arch = "aarch64")))]
                {
                    let _ = (target, self.w1_rows);
                    None
                }
            })
            .as_deref()
    }
}

#[derive(Clone, Copy)]
struct SendU16(*mut u16);
unsafe impl Send for SendU16 {}
unsafe impl Sync for SendU16 {}
impl SendU16 {
    #[inline(always)]
    fn get(self) -> *mut u16 {
        self.0
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The release's `mtp_config` reads as the drafter of llama.cpp's conversion: a block of 8,
    /// the target's final output last, sliding layers 0 and 2.
    #[test]
    fn release_mtp_config() {
        let m = serde_json::json!({
            "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24, "num_key_value_heads": 8,
            "head_dim": 128, "intermediate_size": 9216, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "local_layer_ids": [0, 2], "sliding_window_size": 1024, "block_size": 8, "markov_rank": 256,
            "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48]});
        let c = InkDraftConfig::from_hf(&m).unwrap();
        assert_eq!(c.windows, [1024, 0, 1024]);
        assert_eq!(c.target_layers, [5, 23, 41, 48]);
        let d = c.dspark();
        assert_eq!((d.max_draft(), d.softcap), (7, Some(30.0)));
        let mut bad = m.clone();
        bad["markov_head_type"] = "vanilla".into();
        assert!(InkDraftConfig::from_hf(&bad).is_err());
    }
}
