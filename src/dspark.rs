//! DSpark speculative decoding (llama.cpp `--spec-type draft-dspark`): a small draft model that
//! reads the target model's hidden states and proposes a whole block of tokens per forward pass.
//!
//! The draft (a llama.cpp `dflash` GGUF, e.g. unsloth's `dspark-Mistral-Large-4-1T-A52B-*.gguf`)
//! is a DFlash block-diffusion backbone plus a semi-autoregressive "Markov" head:
//!
//! - Context: for every position the target model has processed, the residual streams entering
//!   a few target layers (`dflash.target_layers`; the layer count stands for the last layer's
//!   output) are concatenated, projected by `fc`, RMS-normed (`enc.output_norm`), and each draft
//!   layer turns that into its own key / value cache entry (no attention norm on this path).
//! - Block: the last committed token (the "anchor", not yet run by the target) at position `n`,
//!   followed by learned mask embeddings at `n + 1..`, goes through the draft layers attending
//!   non-causally to the cached context of the last `sliding_window` positions and to the whole
//!   block. The final-normed states go through the target's LM head (soft-capped with the
//!   draft's own cap); with `sample_from_anchor` slot `i` predicts the token at `n + i + 1`.
//! - Markov head: each slot's logits get a low-rank bias keyed on the previous token
//!   (`markov_w2 * f(markov_w1[prev])`), chained across the block; the "rnn" variant carries a
//!   gated state through a joint projection of `[state, markov_w1[prev], hidden]`. The chain uses
//!   the drafted tokens, so the draft distribution of each slot is exactly what it was drawn from.
//!
//! The target verifies `[anchor, drafts..]` in one pass (standard speculative sampling, as
//! [`crate::qwen_mtp`]); the features of the committed positions then join the draft's cache.
//! Only the MLA backbone of the Mistral Large 4 drafts (dense MLA layers, latent K = V cache,
//! llama.cpp's `graph_mistral_large4`) is implemented; other backbones (Qwen3, Gemma 4, DeepSeek
//! V4) would add a `Layer` kind. The target side is the [`Target`] trait.

use crate::generate::TextLm;
use crate::gguf::{Header, Value};
use crate::kernels::*;
use crate::model::{Lin, Quant, Table, linear, linear_multi, linear_rows_device, load_lin};
use crate::qwen_mtp::{MtpStats, speculative_accept};
use crate::safetensors::ModelFiles;
use crate::sampler::Sampler;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// What a DSpark draft needs from its target model.
pub trait Target: TextLm {
    /// Runs `tokens` like [`TextLm::forward_hidden_mm`] (final-normed hidden states `[t,
    /// hidden]`) and writes to `feats` the residual stream entering each of `layers` (`[t,
    /// layers.len() * hidden]`; the layer count stands for the last layer's output before the
    /// final norm).
    fn forward_feats(
        &self,
        sess: &mut Self::Sess,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32>;
    /// The input embedding of `token` (`[hidden]`).
    fn embed_row(&self, token: u32, out: &mut [f32]);
    /// LM-head logits `[t, vocab]` without the target's own soft-capping.
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32>;
    /// [`Target::head_raw`] for a draft's few rows (1 to 4) on the accelerator's matrix-vector
    /// kernels ([`crate::gpu::Backend::matvec_rows`]); None when it cannot (the caller then uses
    /// `head_raw`).
    fn head_raw_device(&self, _hidden: &[f32], _t: usize) -> Option<Vec<f32>> {
        None
    }
    fn vocab_size(&self) -> usize;
    /// Drops cached positions at or beyond `len`.
    fn truncate(&self, sess: &mut Self::Sess, len: usize);
}

/// The draft's hyperparameters (`dflash.*` GGUF metadata).
#[derive(Clone, Debug)]
pub struct DsparkConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Latent (`kv_lora_rank`), non-rotary / rotary query-key and value head sizes.
    pub lat: usize,
    pub dn: usize,
    pub dr: usize,
    pub vd: usize,
    pub inter: usize,
    pub eps: f32,
    /// Context positions the block attends to (0: all).
    pub window: usize,
    /// Trained block: drafts per pass with `sample_from_anchor`, else one more slot than drafts.
    pub block_size: usize,
    pub sample_from_anchor: bool,
    /// Target layers whose input residual is read (the target layer count = its last output).
    pub target_layers: Vec<usize>,
    pub softcap: Option<f32>,
    pub rope: crate::config::YarnParams,
    /// The token the block's slots after the anchor are embedded as (DeepSeek V4; MLA drafts
    /// have learned mask embeddings instead).
    pub noise: Option<u32>,
    /// The stages' config of a DeepSeek V4 draft (None: an MLA draft).
    pub ds4: Option<Box<crate::config::DeepSeek4Config>>,
}

impl DsparkConfig {
    pub fn from_header(h: &Header) -> Result<Self> {
        if h.arch() != "dflash" {
            bail!("not a DFlash / DSpark draft (GGUF architecture {:?})", h.arch());
        }
        let get = |k: &str| h.get(&format!("dflash.{k}"));
        let u = |k: &str| get(k).and_then(Value::as_u64).map(|v| v as usize);
        let need = |k: &str| u(k).with_context(|| format!("DSpark GGUF has no dflash.{k}"));
        let f = |k: &str| get(k).and_then(Value::as_f64);
        let target_layers: Vec<usize> = get("target_layers")
            .and_then(Value::as_array)
            .context("DSpark GGUF has no dflash.target_layers")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as usize).context("bad dflash.target_layers"))
            .collect::<Result<_>>()?;
        // DeepSeek V4: full DeepSeek V4 stages (hyper-connections, MoE), `crate::deepseek4`.
        if get("hyper_connection.count").is_some() {
            let ds = crate::config::DeepSeek4Config::from_json(h.deepseek4_config()?)?;
            let c = Self {
                hidden: ds.hidden_size,
                n_layers: ds.num_hidden_layers,
                n_heads: ds.num_attention_heads,
                lat: 0,
                dn: 0,
                dr: ds.qk_rope_head_dim,
                vd: ds.head_dim,
                inter: ds.moe_intermediate_size,
                eps: ds.rms_norm_eps as f32,
                window: ds.sliding_window,
                block_size: need("block_size")?,
                sample_from_anchor: !matches!(get("sample_from_anchor"), Some(Value::Bool(false))),
                target_layers,
                softcap: None,
                rope: crate::config::YarnParams {
                    rope_theta: ds.rope_theta,
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
                noise: h.get("tokenizer.ggml.mask_token_id").and_then(Value::as_u64).map(|v| v as u32),
                ds4: Some(Box::new(ds)),
            };
            if c.block_size < 2 || c.target_layers.is_empty() || c.noise.is_none() {
                bail!("bad DeepSeek V4 DSpark metadata (block size, target layers, mask token)");
            }
            return Ok(c);
        }
        let Some(lat) = u("attention.kv_lora_rank") else {
            bail!("only the MLA (Mistral Large 4) and DeepSeek V4 DSpark backbones are supported");
        };
        let dr = need("rope.dimension_count")?;
        let qk = need("attention.key_length_mla")?;
        let theta = f("rope.freq_base").unwrap_or(10_000.0);
        let rope = match get("rope.scaling.type").and_then(Value::as_str) {
            Some("yarn") => {
                // llama.cpp's magnitude: mscale(factor, 1) / mscale(factor, yarn_log_multiplier),
                // or mscale(factor, 1) without the multiplier; queries keep 1/sqrt(head dim).
                let log_mul = f("rope.scaling.yarn_log_multiplier").filter(|&m| m != 0.0);
                crate::config::YarnParams {
                    rope_theta: theta,
                    rope_type: Some("yarn".into()),
                    factor: f("rope.scaling.factor"),
                    original_max_position_embeddings: u("rope.scaling.original_context_length"),
                    beta_fast: f("rope.scaling.yarn_beta_fast"),
                    beta_slow: f("rope.scaling.yarn_beta_slow"),
                    attention_factor: None,
                    mscale: log_mul.map(|_| 1.0),
                    mscale_all_dim: log_mul,
                    truncate: None,
                }
            }
            None | Some("none") => crate::config::YarnParams {
                rope_theta: theta,
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
            Some(t) => bail!("DSpark draft RoPE scaling {t:?} is not supported"),
        };
        let c = Self {
            hidden: need("embedding_length")?,
            n_layers: need("block_count")?,
            n_heads: need("attention.head_count")?,
            lat,
            dn: qk.checked_sub(dr).context("dflash.attention.key_length_mla < rope dims")?,
            dr,
            vd: need("attention.value_length_mla")?,
            inter: need("feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
            window: u("attention.sliding_window").unwrap_or(0),
            block_size: need("block_size")?,
            sample_from_anchor: !matches!(get("sample_from_anchor"), Some(Value::Bool(false))),
            target_layers,
            softcap: f("final_logit_softcapping").filter(|&c| c > 0.0).map(|c| c as f32),
            rope,
            noise: None,
            ds4: None,
        };
        if c.block_size < 2 || c.target_layers.is_empty() || !c.dr.is_multiple_of(2) {
            bail!("bad DSpark metadata: {c:?}");
        }
        Ok(c)
    }

    /// The most tokens one pass drafts.
    pub fn max_draft(&self) -> usize {
        if self.sample_from_anchor {
            self.block_size
        } else {
            self.block_size - 1
        }
    }
}

/// One dense MLA draft layer (llama.cpp `graph_mistral_large4`): absorbed latent attention, the
/// latent doubling as the value, then a SwiGLU MLP.
struct Layer<'a> {
    attn_norm: Vec<f32>,
    q: Lin<'a>,    // [nh * (dn + dr), H]
    kv_a: Lin<'a>, // [lat + dr, H]
    kv_norm: Vec<f32>,
    /// Per head `[lat, dn]` (query onto the latent) and `[vd, lat]` (attended latent onto value).
    k_b: Vec<Lin<'a>>,
    v_b: Vec<Lin<'a>>,
    o: Lin<'a>, // [H, nh * vd]
    ffn_norm: Vec<f32>,
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
}

/// The Markov head: `markov_w1` `[V, R]` embeds the previous token, `markov_w2` `[V, R]` maps the
/// (tanh'd) state onto a logit bias; the rnn variant's joint projection `[3R, 2R + H]`.
struct Markov<'a> {
    w1: Table<'a>,
    w2: Lin<'a>,
    joint: Option<(Lin<'a>, Vec<f32>)>,
    rank: usize,
}

/// A loaded DSpark draft (weights borrowed from its GGUF).
pub struct Dspark<'a> {
    pub cfg: DsparkConfig,
    /// The backbone: dense MLA layers (Mistral Large 4) or DeepSeek V4 stages.
    mla: Option<Mla<'a>>,
    ds4: Option<crate::deepseek4::Ds4Draft<'a>>,
    markov: Option<Markov<'a>>,
    /// The draft's own embeddings / LM head, when it ships them (else the target's are used).
    embed: Option<Table<'a>>,
    head: Option<Lin<'a>>,
    target_hidden: usize,
    /// Run the matmuls of the draft's few rows on the accelerator (see [`Dspark::set_device`]).
    device: std::sync::atomic::AtomicBool,
}

/// The MLA backbone (llama.cpp `graph_mistral_large4`).
struct Mla<'a> {
    fc: Lin<'a>, // [H, n_feat * target hidden]
    enc_norm: Vec<f32>,
    out_norm: Vec<f32>,
    mask: Table<'a>, // [block_size - 1, H]
    layers: Vec<Layer<'a>>,
    inv_freq: Vec<f32>,
    rope_mag: f32,
}

/// Positions the draft's context cache keeps beyond the window it attends to: a truncation by up
/// to this many positions leaves the cache exact (the dropped positions come back as the target
/// runs them again). The cache holds between `window + SLACK` and `window + 2 * SLACK`
/// positions once it is full; the oldest are dropped `SLACK` at a time.
pub const SLACK: usize = 256;

/// The draft's context cache: per layer and position the normed latent `[lat]` and the rotated
/// rotary key `[dr]`, for positions `start..len` of what the target has processed. The block
/// attends to the last `window` positions only, so with a window the older ones are dropped
/// (see [`SLACK`]); a draft without one keeps them all.
#[derive(Clone, Default)]
pub struct DsparkSession {
    c: Vec<Vec<f32>>,
    r: Vec<Vec<f32>>,
    lat: usize,
    dr: usize,
    /// Positions kept beyond `slack` (the draft's window; 0: all), and the slack.
    keep: usize,
    slack: usize,
    /// The first cached position.
    start: usize,
    pub len: usize,
}

impl DsparkSession {
    /// Drops cached positions at or beyond `len`. Going back more than [`SLACK`] positions may
    /// leave less than the window before `len` cached (positions dropped earlier do not come
    /// back): the block then attends to what is left until new positions fill the window, which
    /// changes the drafts but not the output (the target verifies them).
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            let n = len.saturating_sub(self.start);
            for c in &mut self.c {
                c.truncate(n * self.lat);
            }
            for r in &mut self.r {
                r.truncate(n * self.dr);
            }
            self.start = self.start.min(len);
            self.len = len;
        }
    }

    /// The first position still cached (positions before it were dropped).
    pub fn start(&self) -> usize {
        self.start
    }

    /// Bytes of the cached latents and keys.
    pub fn bytes(&self) -> usize {
        self.c.iter().chain(&self.r).map(|v| v.len() * 4).sum()
    }

    /// Appends one layer's entries for new positions (the caller then advances `len`).
    fn push(&mut self, layer: usize, c: &[f32], r: &[f32]) {
        self.c[layer].extend_from_slice(c);
        self.r[layer].extend_from_slice(r);
    }

    /// Drops the oldest positions once more than `keep + 2 * slack` are cached, down to
    /// `keep + slack` (moving the rest down once per `slack` positions).
    fn trim(&mut self) {
        if self.keep == 0 || self.len - self.start < self.keep + 2 * self.slack {
            return;
        }
        let drop = self.len - self.start - (self.keep + self.slack);
        for c in &mut self.c {
            c.drain(..drop * self.lat);
        }
        for r in &mut self.r {
            r.drain(..drop * self.dr);
        }
        self.start += drop;
    }

    /// Cached latents and rotary keys of `layer` for positions `lo..hi` (within `start..len`).
    fn range(&self, layer: usize, lo: usize, hi: usize) -> (&[f32], &[f32]) {
        let (a, b) = (lo - self.start, hi - self.start);
        (&self.c[layer][a * self.lat..b * self.lat], &self.r[layer][a * self.dr..b * self.dr])
    }
}

/// Whether a GGUF model uses the DSpark draft beside it unless told otherwise (CLI
/// `--no-draft`, engine `draft=none`): measured 6-10% faster on Mistral Large 4 streamed from
/// storage on a 128 GB M5 Max with [`STREAM_P_MIN`] (see the README).
pub const BY_DEFAULT: bool = true;

/// Whether the draft runs on the accelerator when one is active unless told otherwise
/// ([`DsparkOptions::device`]): measured faster on an M5 Max (see the README).
pub const DEVICE_BY_DEFAULT: bool = true;

/// The default `p_min` ([`DsparkOptions::p_min`]) when the target streams its weights. There a
/// verification pass reads the routed experts of every token it holds (experts shared by
/// consecutive tokens were page-cache hits anyway), so a rejected draft costs about one token's
/// reads: only drafts that are nearly always accepted pay. Measured on Mistral Large 4: drafts
/// the draft model gives at least 0.95 are accepted 98% of the time, 0.7-0.95 75%, 0.3-0.7 40%.
pub const STREAM_P_MIN: f32 = 0.85;

/// Default drafts per round with a DeepSeek V4 draft (its block holds 5): see the README.
pub const DS4_DRAFT_TOKENS: usize = 3;

/// The DSpark draft GGUF beside GGUF model `model` (a `dspark*.gguf` file in its folder or the
/// folder above, as unsloth lays them out), if there is exactly one.
pub fn draft_for(model: &Path) -> Option<PathBuf> {
    if !crate::gguf::is_gguf(model) {
        return None;
    }
    let dir = model.parent()?;
    for d in [Some(dir), dir.parent()].into_iter().flatten() {
        let Ok(rd) = std::fs::read_dir(d) else { continue };
        let mut v: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
                n.starts_with("dspark") && n.ends_with(".gguf")
            })
            .collect();
        v.sort();
        match v.len() {
            0 => continue,
            1 => return v.pop(),
            _ => return None,
        }
    }
    None
}

/// Whether `path` is a DFlash / DSpark draft GGUF.
pub fn is_draft(path: &Path) -> bool {
    crate::gguf::is_gguf(path) && crate::gguf::header(path).is_ok_and(|h| h.arch() == "dflash")
}

impl<'a> Dspark<'a> {
    /// Loads the draft in `files` (opened from `path`, a `dflash` GGUF) for a target whose
    /// hidden size is `target_hidden` and vocabulary `vocab`. Unquantized weights become `quant`.
    pub fn load(path: &Path, files: &'a ModelFiles, target_hidden: usize, vocab: usize, quant: Quant) -> Result<Self> {
        let hdr = crate::gguf::header(path)?;
        let cfg = DsparkConfig::from_header(&hdr)?;
        if hdr
            .tensors
            .iter()
            .any(|t| t.name == "d2t" || t.name.starts_with("selector_") || t.name.starts_with("lilicorr"))
        {
            bail!("this DFlash draft needs a reduced vocabulary, selector or LiLiCorr head (not supported)");
        }
        let c = &cfg;
        let (h, nh) = (c.hidden, c.n_heads);
        let shape = |n: &str, s: &[usize]| -> Result<()> {
            let info = files.info(n).with_context(|| format!("DSpark draft: missing tensor {n}"))?;
            if info.shape != s {
                bail!("DSpark draft: tensor {n} has shape {:?}, expected {s:?}", info.shape);
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
        // Per-head matrices `[nh, n_out, n_in]`, one Lin each (GGUF bytes split, or bf16 rows).
        let heads = |n: &str, n_out: usize, n_in: usize| -> Result<Vec<Lin<'a>>> {
            shape(n, &[nh, n_out, n_in])?;
            match files.ggml(n) {
                Some(g) => {
                    let (kind, bytes) = g?;
                    bytes
                        .chunks(bytes.len() / nh)
                        .map(|b| Lin::from_gguf(kind, b, n_out, n_in))
                        .collect()
                }
                None => Ok(files.bf16(n)?.chunks(n_out * n_in).map(Lin::bf16).collect()),
            }
        };
        let n_feat = c.target_layers.len() * target_hidden;
        let markov = if files.has("markov_w1.weight") {
            let rank = files.info("markov_w1.weight").unwrap().shape.last().copied().unwrap_or(0);
            shape("markov_w1.weight", &[vocab, rank])?;
            let joint = if files.has("joint_proj.weight") {
                Some((
                    lin("joint_proj.weight", 3 * rank, 2 * rank + h)?,
                    vecf("joint_proj.bias", 3 * rank)?,
                ))
            } else {
                None
            };
            Some(Markov {
                w1: Table::load(files, "markov_w1.weight", vocab, rank, quant)?,
                w2: lin("markov_w2.weight", vocab, rank)?,
                joint,
                rank,
            })
        } else {
            None
        };
        let embed = if files.has("token_embd.weight") {
            shape("token_embd.weight", &[vocab, h])?;
            Some(Table::load(files, "token_embd.weight", vocab, h, quant)?)
        } else {
            if h != target_hidden {
                bail!("the DSpark draft (hidden {h}) has no embeddings of its own for a target of hidden size {target_hidden}");
            }
            None
        };
        let head = if files.has("output.weight") {
            Some(lin("output.weight", vocab, h)?)
        } else {
            None
        };
        if let Some(ds) = &c.ds4 {
            let ds4 = crate::deepseek4::Ds4Draft::load(files, (**ds).clone(), quant, n_feat)?;
            return Ok(Self {
                mla: None,
                ds4: Some(ds4),
                markov,
                embed,
                head,
                target_hidden,
                device: std::sync::atomic::AtomicBool::new(false),
                cfg,
            });
        }
        let mut layers = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            let p = format!("blk.{l}.");
            layers.push(Layer {
                attn_norm: vecf(&format!("{p}attn_norm.weight"), h)?,
                q: lin(&format!("{p}attn_q.weight"), nh * (c.dn + c.dr), h)?,
                kv_a: lin(&format!("{p}attn_kv_a_mqa.weight"), c.lat + c.dr, h)?,
                kv_norm: vecf(&format!("{p}attn_kv_a_norm.weight"), c.lat)?,
                k_b: heads(&format!("{p}attn_k_b.weight"), c.lat, c.dn)?,
                v_b: heads(&format!("{p}attn_v_b.weight"), c.vd, c.lat)?,
                o: lin(&format!("{p}attn_output.weight"), h, nh * c.vd)?,
                ffn_norm: vecf(&format!("{p}ffn_norm.weight"), h)?,
                gate: lin(&format!("{p}ffn_gate.weight"), c.inter, h)?,
                up: lin(&format!("{p}ffn_up.weight"), c.inter, h)?,
                down: lin(&format!("{p}ffn_down.weight"), h, c.inter)?,
            });
        }
        shape("mask_embd.weight", &[c.block_size - 1, h])?;
        let (inv_freq, rope_mag) = cfg.rope.inv_freq(cfg.dr);
        let mla = Mla {
            fc: lin("fc.weight", h, n_feat)?,
            enc_norm: vecf("enc.output_norm.weight", h)?,
            out_norm: vecf("output_norm.weight", h)?,
            mask: Table::load(files, "mask_embd.weight", c.block_size - 1, h, quant)?,
            layers,
            inv_freq,
            rope_mag,
        };
        Ok(Self {
            mla: Some(mla),
            ds4: None,
            markov,
            embed,
            head,
            target_hidden,
            device: std::sync::atomic::AtomicBool::new(false),
            cfg,
        })
    }

    /// Runs the matmuls of a draft pass and of the injection of a round's committed tokens (the
    /// weights, the target's LM head) on the active accelerator's matrix-vector kernels
    /// (`Backend::matvec_rows`) instead of the CPU's; the attention and the small per-head
    /// matrices stay on the CPU. Without an accelerator that takes them, the CPU runs them.
    /// The drafts may differ in the last bits (the target's verification keeps the output
    /// exact). Off by default; [`DsparkDecoder::new`] sets it from [`DsparkOptions::device`].
    pub fn set_device(&self, on: bool) {
        self.device.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the draft runs its matmuls on the accelerator (see [`Dspark::set_device`]).
    pub fn device(&self) -> bool {
        self.device.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `linear_multi` of the draft: for a few rows on the accelerator's matrix-vector kernels
    /// when the draft runs there, else (or when they decline) as usual.
    fn lin(&self, jobs: &mut [(&Lin, usize, &mut [f32])], n_in: usize, x: &[f32], t: usize) {
        if t <= 4 && self.device() {
            let mut dj: Vec<(&Lin, Option<&Lin>, usize, &mut [f32])> = jobs.iter_mut().map(|(l, n, y)| (*l, None, *n, &mut **y)).collect();
            if linear_rows_device(&mut dj, n_in, x, t) {
                return;
            }
        }
        linear_multi(jobs, n_in, x, t);
    }

    /// The MLP's `silu(gate x) * up x` (`[t, inter]`), as [`Dspark::lin`].
    fn glu(&self, l: &Layer, x: &[f32], t: usize) -> Vec<f32> {
        let (h, inter) = (self.cfg.hidden, self.cfg.inter);
        let mut g = vec![0f32; t * inter];
        if t <= 4 && self.device() && linear_rows_device(&mut [(&l.gate, Some(&l.up), inter, &mut g[..])], h, x, t) {
            return g;
        }
        let mut u = vec![0f32; t * inter];
        linear_multi(&mut [(&l.gate, inter, &mut g[..]), (&l.up, inter, &mut u[..])], h, x, t);
        silu_mul(&mut g, &u);
        g
    }

    fn mla(&self) -> &Mla<'a> {
        self.mla.as_ref().expect("MLA draft")
    }

    pub fn new_session(&self) -> DsparkSession {
        // DeepSeek V4: per stage its window keys, `head_dim` per position.
        let (n, lat, dr) = match &self.ds4 {
            Some(d) => (d.n_stages(), d.cfg.head_dim, 0),
            None => (self.mla().layers.len(), self.cfg.lat, self.cfg.dr),
        };
        DsparkSession {
            c: vec![Vec::new(); n],
            r: vec![Vec::new(); n],
            lat,
            dr,
            // The DeepSeek V4 stages index their window keys from position 0: no trimming.
            keep: if self.ds4.is_some() { 0 } else { self.cfg.window },
            slack: SLACK,
            start: 0,
            len: 0,
        }
    }

    /// Weight bytes one draft pass reads (without the target's LM head).
    pub fn pass_bytes(&self) -> usize {
        if let Some(d) = &self.ds4 {
            return d.pass_bytes(self.cfg.block_size)
                + self
                    .markov
                    .as_ref()
                    .map_or(0, |m| m.w2.bytes() + m.joint.as_ref().map_or(0, |j| j.0.bytes()));
        }
        let mut n = self.mla().fc.bytes();
        for l in &self.mla().layers {
            n += [&l.q, &l.kv_a, &l.o, &l.gate, &l.up, &l.down]
                .iter()
                .map(|w| w.bytes())
                .sum::<usize>();
            n += l.k_b.iter().chain(&l.v_b).map(|w| w.bytes()).sum::<usize>();
        }
        if let Some(m) = &self.markov {
            n += m.w2.bytes() + m.joint.as_ref().map_or(0, |j| j.0.bytes());
        }
        n
    }

    /// Cos / sin tables (scaled by the YaRN magnitude) for `positions`.
    fn rope(&self, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let m = self.mla();
        let (mut cos, mut sin) = rope_tables(positions, &m.inv_freq);
        if m.rope_mag != 1.0 {
            cos.iter_mut().chain(sin.iter_mut()).for_each(|v| *v *= m.rope_mag);
        }
        (cos, sin)
    }

    /// A layer's latent and rotated rotary key for `t` rows of its `kv_a` projection (`kv`,
    /// `[t, lat + dr]`) at positions `start..`: `([t, lat], [t, dr])`.
    fn latent(&self, l: &Layer, kv: &[f32], t: usize, start: usize) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let (lat, dr) = (c.lat, c.dr);
        let positions: Vec<usize> = (start..start + t).collect();
        let (cos, sin) = self.rope(&positions);
        let half = dr / 2;
        let (mut cl, mut kr) = (vec![0f32; t * lat], vec![0f32; t * dr]);
        for ti in 0..t {
            let row = &kv[ti * (lat + dr)..(ti + 1) * (lat + dr)];
            rmsnorm_f32w(&row[..lat], &l.kv_norm, c.eps, &mut cl[ti * lat..(ti + 1) * lat]);
            let k = &mut kr[ti * dr..(ti + 1) * dr];
            k.copy_from_slice(&row[lat..]);
            apply_rope_interleaved(k, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
        }
        (cl, kr)
    }

    /// Adds the target features of the next `t` positions (`feats` `[t, n_feat]`, as
    /// [`Target::forward_feats`] returns them) to the draft's context cache.
    pub fn inject(&self, sess: &mut DsparkSession, feats: &[f32], t: usize) {
        if t == 0 {
            return;
        }
        let _timer = crate::prof::start(crate::prof::Cat::Other);
        let h = self.cfg.hidden;
        let n_feat = self.cfg.target_layers.len() * self.target_hidden;
        assert_eq!(feats.len(), t * n_feat);
        if let Some(d) = &self.ds4 {
            for (li, kv) in d.context(feats, t, sess.len).into_iter().enumerate() {
                sess.c[li].extend_from_slice(&kv);
            }
            sess.len += t;
            return;
        }
        let m = self.mla();
        let mut x = vec![0f32; t * h];
        self.lin(&mut [(&m.fc, h, &mut x[..])], n_feat, feats, t);
        rmsnorm_f32w_rows_inplace(&mut x, h, &m.enc_norm, self.cfg.eps);
        // Every layer's latent projection of the same rows, in one batch.
        let kd = self.cfg.lat + self.cfg.dr;
        let mut kvs = vec![vec![0f32; t * kd]; m.layers.len()];
        {
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = m
                .layers
                .iter()
                .zip(kvs.iter_mut())
                .map(|(l, kv)| (&l.kv_a, kd, &mut kv[..]))
                .collect();
            self.lin(&mut jobs, h, &x, t);
        }
        for (li, (l, kv)) in m.layers.iter().zip(&kvs).enumerate() {
            let (cl, kr) = self.latent(l, kv, t, sess.len);
            sess.push(li, &cl, &kr);
        }
        sess.len += t;
        sess.trim();
    }

    /// Runs the block `[anchor, mask_1, ..]` of `nb` rows at positions `sess.len..` through the
    /// draft layers; returns the final-normed states `[nb, H]`.
    fn block<M: Target + ?Sized>(&self, target: &M, sess: &DsparkSession, anchor: u32, nb: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (h, nh, lat, dn, dr, vd) = (c.hidden, c.n_heads, c.lat, c.dn, c.dr, c.vd);
        let qk = dn + dr;
        let pos = sess.len;
        let mut x = vec![0f32; nb * h];
        let embed = |t: u32, out: &mut [f32]| match &self.embed {
            Some(e) => e.row(t as usize, out),
            None => target.embed_row(t, out),
        };
        embed(anchor, &mut x[..h]);
        if let Some(d) = &self.ds4 {
            // The noise token after the anchor; the window of the last `window` positions.
            for i in 1..nb {
                embed(c.noise.unwrap_or(0), &mut x[i * h..(i + 1) * h]);
            }
            let hd = d.cfg.head_dim;
            let lo = pos.saturating_sub(c.window);
            let ctx: Vec<&[f32]> = sess.c.iter().map(|kv| &kv[lo * hd..pos * hd]).collect();
            return d.block(&x, nb, pos, &ctx);
        }
        let m = self.mla();
        for i in 1..nb {
            m.mask.row(i - 1, &mut x[i * h..(i + 1) * h]);
        }
        let positions: Vec<usize> = (pos..pos + nb).collect();
        let (cos, sin) = self.rope(&positions);
        let half = dr / 2;
        // The context the block sees: the last `window` positions, whatever the slot (fewer
        // after a deep truncation, see `DsparkSession::truncate`).
        let lo = if c.window > 0 { pos.saturating_sub(c.window) } else { 0 };
        let lo = lo.max(sess.start);
        let n_ctx = pos - lo;
        let scale = (qk as f32).powf(-0.5);
        let mut hn = vec![0f32; nb * h];
        for (li, l) in m.layers.iter().enumerate() {
            rmsnorm_f32w_rows(&x, h, &l.attn_norm, c.eps, &mut hn);
            let mut q = vec![0f32; nb * nh * qk];
            let mut kv = vec![0f32; nb * (lat + dr)];
            self.lin(&mut [(&l.q, nh * qk, &mut q[..]), (&l.kv_a, lat + dr, &mut kv[..])], h, &hn, nb);
            let (bc, br) = self.latent(l, &kv, nb, pos);
            // Non-rotary queries per head [nh][nb][dn] onto the latent; rotary parts rotated.
            let mut qn = vec![0f32; nh * nb * dn];
            for i in 0..nb {
                for hh in 0..nh {
                    let qh = &mut q[(i * nh + hh) * qk..(i * nh + hh + 1) * qk];
                    qn[(hh * nb + i) * dn..(hh * nb + i + 1) * dn].copy_from_slice(&qh[..dn]);
                    apply_rope_interleaved(&mut qh[dn..], &cos[i * half..(i + 1) * half], &sin[i * half..(i + 1) * half]);
                }
            }
            let ql = crate::mistral4::per_head(&l.k_b, &qn, nb, dn, lat);
            let (cc, cr) = sess.range(li, lo, pos);
            let mut ol = vec![0f32; nh * nb * lat];
            let op = SendPtr(ol.as_mut_ptr());
            crate::pool::run(nb * nh, &|task| {
                let (i, hh) = (task / nh, task % nh);
                let qlh = &ql[(hh * nb + i) * lat..(hh * nb + i + 1) * lat];
                let qr = &q[(i * nh + hh) * qk + dn..(i * nh + hh + 1) * qk];
                let key = |j: usize| -> (&[f32], &[f32]) {
                    if j < n_ctx {
                        (&cc[j * lat..(j + 1) * lat], &cr[j * dr..(j + 1) * dr])
                    } else {
                        let b = j - n_ctx;
                        (&bc[b * lat..(b + 1) * lat], &br[b * dr..(b + 1) * dr])
                    }
                };
                let mut s: Vec<f32> = (0..n_ctx + nb)
                    .map(|j| {
                        let (kc, kr) = key(j);
                        (dot(qlh, kc) + dot(qr, kr)) * scale
                    })
                    .collect();
                softmax_inplace(&mut s);
                // SAFETY: each (row, head) task writes its own latent row.
                let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((hh * nb + i) * lat), lat) };
                for (j, &p) in s.iter().enumerate() {
                    axpy(o, p, key(j).0);
                }
            });
            let v = crate::mistral4::per_head(&l.v_b, &ol, nb, lat, vd);
            let mut att = vec![0f32; nb * nh * vd];
            for i in 0..nb {
                for hh in 0..nh {
                    att[(i * nh + hh) * vd..(i * nh + hh + 1) * vd].copy_from_slice(&v[(hh * nb + i) * vd..(hh * nb + i + 1) * vd]);
                }
            }
            let mut o = vec![0f32; nb * h];
            self.lin(&mut [(&l.o, h, &mut o[..])], nh * vd, &att, nb);
            add_inplace(&mut x, &o);
            rmsnorm_f32w_rows(&x, h, &l.ffn_norm, c.eps, &mut hn);
            let g = self.glu(l, &hn, nb);
            self.lin(&mut [(&l.down, h, &mut o[..])], c.inter, &g, nb);
            add_inplace(&mut x, &o);
        }
        rmsnorm_f32w_rows_inplace(&mut x, h, &m.out_norm, c.eps);
        x
    }

    /// Drafts up to `k` tokens (1 to [`DsparkConfig::max_draft`]) after `anchor`, the committed
    /// token at position `sess.len` the target has not run yet. The block is cut before the
    /// first draft whose probability under the draft's own softmax is below `p_min` (0: never),
    /// which leaves the output distribution unchanged.
    pub fn propose<M: Target + ?Sized>(
        &self,
        target: &M,
        sess: &DsparkSession,
        anchor: u32,
        k: usize,
        p_min: f32,
        sampler: &mut Sampler,
    ) -> Proposal {
        let mut p = Proposal::default();
        p.tokens = self.slots(target, sess, anchor, k, &mut |_, logits| {
            let q = sampler.distribution(logits);
            let d = if sampler.greedy() { q[0].0 } else { sampler.sample_from(&q) };
            let pd = softmax_at(logits, d);
            if pd < p_min {
                return None;
            }
            p.dists.push(q);
            p.probs.push(pd);
            Some(d)
        });
        p
    }

    /// The final logits of each of `k` slots (`[k, vocab]`) with the chain following the
    /// argmax, and those greedy drafts (for checks against a reference).
    pub fn draft_logits<M: Target + ?Sized>(&self, target: &M, sess: &DsparkSession, anchor: u32, k: usize) -> (Vec<f32>, Vec<u32>) {
        let mut out = Vec::new();
        let drafts = self.slots(target, sess, anchor, k, &mut |_, logits| {
            out.extend_from_slice(logits);
            Some(crate::sampler::argmax(logits))
        });
        (out, drafts)
    }

    /// One draft pass of `k` slots: the block, the (soft-capped) logits of the draft slots, and
    /// the Markov chain, `choose(slot, logits)` picking each slot's token (the next slot's
    /// "previous token"; None ends the block). Returns the chosen tokens.
    fn slots<M: Target + ?Sized>(
        &self,
        target: &M,
        sess: &DsparkSession,
        anchor: u32,
        k: usize,
        choose: &mut dyn FnMut(usize, &[f32]) -> Option<u32>,
    ) -> Vec<u32> {
        let _timer = crate::prof::start(crate::prof::Cat::Other);
        let c = &self.cfg;
        let k = k.clamp(1, c.max_draft());
        let first = usize::from(!c.sample_from_anchor);
        // DeepSeek V4's block attends without a mask: always the trained block, whatever `k`.
        let nb = if self.ds4.is_some() { c.block_size } else { k + first };
        let hid = self.block(target, sess, anchor, nb);
        let h = c.hidden;
        let rows = &hid[first * h..(first + k) * h];
        let mut logits = match &self.head {
            Some(hd) => {
                let v = target.vocab_size();
                let mut y = vec![0f32; k * v];
                self.lin(&mut [(hd, v, &mut y[..])], h, rows, k);
                y
            }
            None => match self.device().then(|| target.head_raw_device(rows, k)).flatten() {
                Some(y) => y,
                None => target.head_raw(rows, k),
            },
        };
        // The cap applies to the backbone's logits, not to the Markov bias.
        if let Some(cap) = c.softcap {
            for_chunks(&mut logits, 8192, true, |_, ch| {
                for v in ch.iter_mut() {
                    *v = (*v / cap).tanh() * cap;
                }
            });
        }
        let v = logits.len() / k;
        let mut out = Vec::with_capacity(k);
        let mut prev = anchor;
        let mut state = self.markov.as_ref().map(|m| vec![0f32; m.rank]);
        for i in 0..k {
            let li = &mut logits[i * v..(i + 1) * v];
            if let Some(m) = &self.markov {
                let r = m.rank;
                let mut w1 = vec![0f32; r];
                m.w1.row(prev as usize, &mut w1);
                // Vanilla head: the previous token's embedding; rnn head: gated state update,
                // `state = gate * state + (1 - gate) * cand`, output tanh(third chunk).
                let feat = match (&m.joint, state.as_mut()) {
                    (Some((jp, b)), Some(st)) => {
                        let mut inp = Vec::with_capacity(2 * r + h);
                        inp.extend_from_slice(st);
                        inp.extend_from_slice(&w1);
                        inp.extend_from_slice(&rows[i * h..(i + 1) * h]);
                        let mut j = vec![0f32; 3 * r];
                        // Small (11 MB on Mistral Large 4's draft): not worth a submission.
                        linear(jp, 3 * r, 2 * r + h, &inp, 1, &mut j);
                        add_inplace(&mut j, b);
                        for e in 0..r {
                            let gate = 1.0 / (1.0 + (-j[e]).exp());
                            let cand = j[r + e].tanh();
                            st[e] = cand + gate * (st[e] - cand);
                        }
                        j[2 * r..].iter().map(|v| v.tanh()).collect()
                    }
                    _ => w1,
                };
                let mut bias = vec![0f32; v];
                self.lin(&mut [(&m.w2, v, &mut bias[..])], r, &feat, 1);
                add_inplace(li, &bias);
            }
            let Some(t) = choose(i, li) else { break };
            prev = t;
            out.push(t);
        }
        out
    }
}

/// The drafts of one pass.
#[derive(Clone, Debug, Default)]
pub struct Proposal {
    pub tokens: Vec<u32>,
    /// The distribution each draft was drawn from (the sampler's temperature / top-k / top-p
    /// over the draft logits; one token with probability 1 when greedy).
    pub dists: Vec<Vec<(u32, f32)>>,
    /// Each draft's probability under the draft's own softmax (temperature 1).
    pub probs: Vec<f32>,
}

/// DSpark decoding settings.
#[derive(Clone, Debug)]
pub struct DsparkOptions {
    /// Draft tokens per round (at most the draft's block, [`DsparkConfig::max_draft`]).
    pub draft_tokens: usize,
    /// Tokens that end generation: nothing after one is committed within a round.
    pub stop: Vec<u32>,
    /// Choose each round's draft length (1 to `draft_tokens`) for the most expected tokens per
    /// second from the measured round times and acceptance per position (as
    /// `qwen_mtp::MtpOptions::cost_aware`). A draft pass costs about the same for any length,
    /// but the verification reads the union of the tokens' routed experts.
    pub cost_aware: bool,
    /// Cut each block before the first draft the draft model itself gives less than this
    /// probability (llama.cpp's `--spec-draft-p-min`; 0 = off): an unlikely draft costs a
    /// verification row, and its routed experts when they stream from storage.
    pub p_min: f32,
    /// Where the draft runs its matmuls ([`Dspark::set_device`]): the accelerator
    /// (`Some(true)`), the CPU (`Some(false)`), or by default [`DEVICE_BY_DEFAULT`] when an
    /// accelerator is active (None).
    pub device: Option<bool>,
}

impl Default for DsparkOptions {
    fn default() -> Self {
        Self {
            draft_tokens: 3,
            stop: Vec::new(),
            cost_aware: false,
            p_min: 0.0,
            device: None,
        }
    }
}

/// Speculative generation with a DSpark draft: the plain decoding distribution exactly (greedy
/// gives the plain greedy tokens: the verification computes the same logits as one-token
/// forwards; sampling uses standard speculative sampling, see
/// [`crate::qwen_mtp::speculative_accept`]).
///
/// Usage: `prefill` the prompt, choose the first token from the returned logits and `push` it,
/// then call `step` until a stop token or the length limit; `prefill` again for a later turn.
pub struct DsparkDecoder<'m, M: Target> {
    model: &'m M,
    draft: &'m Dspark<'m>,
    pub sess: M::Sess,
    dsess: DsparkSession,
    pub opts: DsparkOptions,
    /// The committed token the target has not processed yet.
    next: Option<u32>,
    k_cur: usize,
    pub stats: MtpStats,
}

impl<'m, M: Target> DsparkDecoder<'m, M> {
    pub fn new(model: &'m M, draft: &'m Dspark<'m>, opts: DsparkOptions) -> Self {
        let opts = DsparkOptions {
            draft_tokens: opts.draft_tokens.min(draft.cfg.max_draft()),
            ..opts
        };
        // The verification of up to `draft + 1` tokens must stay on the decode kernels.
        if opts.draft_tokens + 1 > crate::kernels::decode_max_tokens() {
            crate::kernels::set_decode_max_tokens(opts.draft_tokens + 1);
        }
        draft.set_device(opts.device.unwrap_or(DEVICE_BY_DEFAULT && crate::gpu::active().is_some()));
        Self {
            model,
            draft,
            sess: model.new_session(),
            dsess: draft.new_session(),
            k_cur: opts.draft_tokens,
            opts,
            next: None,
            stats: MtpStats::default(),
        }
    }

    pub fn model(&self) -> &'m M {
        self.model
    }

    pub fn draft(&self) -> &'m Dspark<'m> {
        self.draft
    }

    /// Current draft length.
    pub fn draft_len(&self) -> usize {
        self.k_cur
    }

    /// The draft's context cache (for checks).
    pub fn draft_session(&self) -> &DsparkSession {
        &self.dsess
    }

    /// Runs `tokens` through the target and adds their features to the draft's cache; returns
    /// the target's final-normed hidden states.
    fn run(&mut self, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        let mut feats = Vec::new();
        let hs = self
            .model
            .forward_feats(&mut self.sess, tokens, mm, &self.draft.cfg.target_layers, &mut feats);
        self.draft.inject(&mut self.dsess, &feats, tokens.len());
        hs
    }

    /// Appends `tokens` (a prompt, or the next turn of a conversation) in chunks of `chunk` and
    /// returns the logits after the last one. A committed token that was never processed (the
    /// stop token ending the previous reply; a reply cut short has its last token put back in
    /// front of `tokens` by the caller) is dropped.
    pub fn prefill(&mut self, tokens: &[u32], chunk: usize) -> Vec<f32> {
        self.prefill_mm(tokens, chunk, &[])
    }

    /// `prefill` with multimodal soft tokens replacing the embeddings at some positions of
    /// `tokens` (`(index in tokens, embedding)`).
    pub fn prefill_mm(&mut self, tokens: &[u32], chunk: usize, overrides: &[(usize, Vec<f32>)]) -> Vec<f32> {
        assert!(!tokens.is_empty());
        self.next = None;
        let h = self.model.hidden_size();
        let mut last = Vec::new();
        let mut start = 0;
        for c in tokens.chunks(chunk.max(1)) {
            let end = start + c.len();
            let local: Vec<(usize, &[f32])> = overrides
                .iter()
                .filter(|(p, _)| *p >= start && *p < end)
                .map(|(p, e)| (p - start, e.as_slice()))
                .collect();
            start = end;
            let hs = self.run(c, &local);
            last = hs[hs.len() - h..].to_vec();
        }
        self.model.logits(&last, 1)
    }

    /// Commits the token chosen from `prefill`'s logits (the first token of a reply).
    pub fn push(&mut self, token: u32) {
        assert!(self.next.is_none(), "push follows prefill; step commits its own tokens");
        self.next = Some(token);
    }

    /// One speculative round: drafts up to the current draft length (and at most
    /// `max_new - 1`), verifies them with one target forward and returns the committed tokens
    /// (1 to `draft + 1` of them, ending at the first stop token if any).
    pub fn step(&mut self, sampler: &mut Sampler, max_new: usize) -> Vec<u32>
    where
        M::Sess: Clone,
    {
        let next = self.next.expect("push a first token before step");
        let (model, draft) = (self.model, self.draft);
        let v = model.vocab_size();
        let n = model.session_len(&self.sess); // position of `next`
        debug_assert_eq!(self.dsess.len, n);
        let k = self.k_cur.min(max_new.max(1) - 1);

        let t_round = crate::clock::Instant::now();
        let prop = if k > 0 {
            draft.propose(model, &self.dsess, next, k, self.opts.p_min, sampler)
        } else {
            Proposal::default()
        };
        let (drafts, qs) = (&prop.tokens, &prop.dists);
        let draft_secs = t_round.elapsed().as_secs_f64();
        // `p_min` may have cut the block.
        let kd = drafts.len();

        let mut seq = Vec::with_capacity(kd + 1);
        seq.push(next);
        seq.extend_from_slice(drafts);
        let check = std::env::var_os("OXINFER_CHECK_SPEC").is_some().then(|| self.sess.clone());
        let mut feats = Vec::new();
        let hs = model.forward_feats(&mut self.sess, &seq, &[], &draft.cfg.target_layers, &mut feats);
        let logits = model.logits(&hs, seq.len());
        let verify_secs = t_round.elapsed().as_secs_f64() - draft_secs;
        if let Some(mut s1) = check {
            // Debug: the batched verification must match one-token forwards exactly.
            for (i, &tk) in seq.iter().enumerate() {
                let l1 = model.forward(&mut s1, &[tk]);
                let lk = &logits[i * v..(i + 1) * v];
                let diff = l1.iter().zip(lk).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                if diff > 0.0 {
                    eprintln!("[check: pos {} of {}: max logit diff {diff:e}]", n + i, seq.len());
                }
            }
        }

        let mut out = Vec::with_capacity(kd + 1);
        let mut accepted = 0;
        for (j, (&d, q)) in drafts.iter().zip(qs).enumerate() {
            let p = sampler.distribution(&logits[j * v..(j + 1) * v]);
            match speculative_accept(sampler, &p, q, d) {
                None => {
                    out.push(d);
                    accepted += 1;
                    if self.opts.stop.contains(&d) {
                        break;
                    }
                }
                Some(t) => {
                    out.push(t);
                    break;
                }
            }
        }
        if accepted == drafts.len() && !out.last().is_some_and(|t| self.opts.stop.contains(t)) {
            out.push(sampler.sample(&logits[kd * v..(kd + 1) * v]));
        }
        if let Some(i) = out.iter().position(|t| self.opts.stop.contains(t)) {
            out.truncate(i + 1);
        }

        // Commit: the target keeps [next, out[..c-1]] (positions n..n+c), whose features join
        // the draft's context; out[c-1] is the new `next`.
        if std::env::var_os("OXINFER_DEBUG_DRAFT").is_some() {
            eprintln!("[draft: p {:?} accepted {accepted}]", prop.probs);
        }
        let c = out.len();
        model.truncate(&mut self.sess, n + c);
        let nf = feats.len() / seq.len();
        draft.inject(&mut self.dsess, &feats[..c * nf], c);
        self.next = out.last().copied();

        let st = &mut self.stats;
        st.rounds += 1;
        st.drafted += kd;
        st.accepted += accepted;
        st.emitted += c;
        if st.per_round.len() <= accepted {
            st.per_round.resize(accepted + 1, 0);
        }
        st.per_round[accepted] += 1;
        st.draft_secs += draft_secs;
        st.verify_secs += verify_secs;
        if k == self.k_cur && kd > 0 {
            // One pass drafts the whole block: no per-step cost between lengths.
            st.costs.record(kd, draft_secs + verify_secs, accepted, Some(0.0));
        }
        if self.opts.cost_aware
            && let Some(b) = st.costs.best(self.opts.draft_tokens.max(1))
        {
            self.k_cur = b;
        }
        out
    }
}

impl<M: Target> crate::generate::RoundDecoder for DsparkDecoder<'_, M>
where
    M::Sess: Clone,
{
    fn push(&mut self, token: u32) {
        DsparkDecoder::push(self, token)
    }
    fn step(&mut self, sampler: &mut Sampler, max_new: usize) -> Vec<u32> {
        DsparkDecoder::step(self, sampler, max_new)
    }
    fn rounds(&self) -> usize {
        self.stats.rounds
    }
}

/// `softmax(logits)[t]`.
fn softmax_at(logits: &[f32], t: u32) -> f32 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f32 = logits.iter().map(|&v| (v - m).exp()).sum();
    (logits[t as usize] - m).exp() / s
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
    use crate::generate::Flow;

    const V: usize = 64;
    const H: usize = 32;

    /// A deterministic toy target: the hidden state carries the token and its position, the next
    /// token is `token + 1` except after every fourth position (`token + 2`), with some spread
    /// for sampling.
    struct Toy;

    #[derive(Clone, Default)]
    struct ToySess(Vec<u32>);

    impl TextLm for Toy {
        type Sess = ToySess;
        fn new_session(&self) -> ToySess {
            ToySess::default()
        }
        fn hidden_size(&self) -> usize {
            H
        }
        fn n_layers(&self) -> usize {
            2
        }
        fn session_len(&self, s: &ToySess) -> usize {
            s.0.len()
        }
        fn forward_hidden_mm(&self, s: &mut ToySess, tokens: &[u32], _mm: &[(usize, &[f32])]) -> Vec<f32> {
            let mut out = vec![0f32; tokens.len() * H];
            for (i, &t) in tokens.iter().enumerate() {
                out[i * H] = t as f32;
                out[i * H + 1] = s.0.len() as f32;
                s.0.push(t);
            }
            out
        }
        fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
            let mut l = vec![0f32; t * V];
            for i in 0..t {
                let (tok, pos) = (hidden[i * H] as usize, hidden[i * H + 1] as usize);
                let next = (tok + if pos % 4 == 3 { 2 } else { 1 }) % V;
                for (j, x) in l[i * V..(i + 1) * V].iter_mut().enumerate() {
                    *x = ((j * 7 + pos * 3) % 11) as f32 * 0.1;
                }
                l[i * V + next] += 3.0;
            }
            l
        }
    }

    impl Target for Toy {
        fn forward_feats(
            &self,
            s: &mut ToySess,
            tokens: &[u32],
            mm: &[(usize, &[f32])],
            layers: &[usize],
            feats: &mut Vec<f32>,
        ) -> Vec<f32> {
            let hs = self.forward_hidden_mm(s, tokens, mm);
            feats.clear();
            for row in hs.chunks(H) {
                for _ in layers {
                    feats.extend(row.iter().map(|v| v * 0.01));
                }
            }
            hs
        }
        fn embed_row(&self, token: u32, out: &mut [f32]) {
            out.fill(0.0);
            out[token as usize % H] = 1.0;
        }
        fn head_raw(&self, _hidden: &[f32], t: usize) -> Vec<f32> {
            vec![0.0; t * V]
        }
        fn vocab_size(&self) -> usize {
            V
        }
        fn truncate(&self, s: &mut ToySess, len: usize) {
            s.0.truncate(len);
        }
    }

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }

    /// A one-layer MLA draft with zero backbone weights and a vanilla Markov head that maps the
    /// previous token `p` onto `p + 1` (bias 10): it always drafts `p + 1, p + 2, ..`. With
    /// `seed`, the backbone weights are random instead and the draft has its own LM head, so its
    /// logits depend on the attended context.
    fn toy_draft(path: &Path, seed: Option<u64>) {
        let mut st = seed.unwrap_or(0);
        let mut fill = |len: usize| -> Vec<f32> {
            match seed {
                None => vec![0.0; len],
                Some(_) => (0..len)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
                    })
                    .collect(),
            }
        };
        let (lat, dn, dr, vd, inter) = (32usize, 32usize, 32usize, 32usize, 32usize);
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        let mut kv: Vec<u8> = Vec::new();
        let mut n_kv = 0u64;
        let mut u32kv = |kv: &mut Vec<u8>, k: &str, v: u32| {
            put_str(kv, &format!("dflash.{k}"));
            kv.extend(4u32.to_le_bytes());
            kv.extend(v.to_le_bytes());
            n_kv += 1;
        };
        for (k, v) in [
            ("block_count", 1),
            ("embedding_length", H as u32),
            ("feed_forward_length", inter as u32),
            ("attention.head_count", 1),
            ("attention.kv_lora_rank", lat as u32),
            ("attention.key_length_mla", (dn + dr) as u32),
            ("attention.value_length_mla", vd as u32),
            ("rope.dimension_count", dr as u32),
            ("attention.sliding_window", 4),
            ("block_size", 3),
        ] {
            u32kv(&mut kv, k, v);
        }
        put_str(&mut kv, "dflash.target_layers");
        kv.extend(9u32.to_le_bytes());
        kv.extend(4u32.to_le_bytes());
        kv.extend(2u64.to_le_bytes());
        kv.extend(1u32.to_le_bytes());
        kv.extend(2u32.to_le_bytes());
        put_str(&mut kv, "general.architecture");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "dflash");
        n_kv += 2;
        // Tensors: (name, ggml ne, values).
        let ones = |n: usize| vec![1f32; n];
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            ("fc.weight".into(), vec![2 * H, H], fill(2 * H * H)),
            ("enc.output_norm.weight".into(), vec![H], ones(H)),
            ("output_norm.weight".into(), vec![H], ones(H)),
            ("mask_embd.weight".into(), vec![H, 2], fill(2 * H)),
        ];
        if seed.is_some() {
            tensors.push(("output.weight".into(), vec![H, V], fill(H * V)));
        }
        let mut w1 = vec![0f32; V * V];
        let mut w2 = vec![0f32; V * V];
        for p in 0..V {
            w1[p * V + p] = 1.0;
            w2[((p + 1) % V) * V + p] = 10.0;
        }
        tensors.push(("markov_w1.weight".into(), vec![V, V], w1));
        tensors.push(("markov_w2.weight".into(), vec![V, V], w2));
        for (n, ne) in [
            ("attn_norm.weight", vec![H]),
            ("attn_q.weight", vec![H, dn + dr]),
            ("attn_kv_a_mqa.weight", vec![H, lat + dr]),
            ("attn_kv_a_norm.weight", vec![lat]),
            ("attn_k_b.weight", vec![dn, lat, 1]),
            ("attn_v_b.weight", vec![lat, vd, 1]),
            ("attn_output.weight", vec![vd, H]),
            ("ffn_norm.weight", vec![H]),
            ("ffn_gate.weight", vec![H, inter]),
            ("ffn_up.weight", vec![H, inter]),
            ("ffn_down.weight", vec![inter, H]),
        ] {
            let len: usize = ne.iter().product();
            let v = if n.contains("norm") { ones(len) } else { fill(len) };
            tensors.push((format!("blk.0.{n}"), ne, v));
        }
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend(n_kv.to_le_bytes());
        b.extend(kv);
        let mut off = 0u64;
        for (name, ne, v) in &tensors {
            put_str(&mut b, name);
            b.extend((ne.len() as u32).to_le_bytes());
            for &d in ne {
                b.extend((d as u64).to_le_bytes());
            }
            b.extend(0u32.to_le_bytes());
            b.extend(off.to_le_bytes());
            off += (v.len() as u64 * 4).next_multiple_of(32);
        }
        b.resize(b.len().next_multiple_of(32), 0);
        for (_, _, v) in &tensors {
            let start = b.len();
            b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            b.resize(start + (v.len() * 4).next_multiple_of(32), 0);
        }
        std::fs::write(path, b).unwrap();
    }

    /// Greedy DSpark decoding commits exactly the plain greedy tokens, through rounds that accept
    /// every draft, some or none, and keeps the draft's context in step with the target.
    #[test]
    fn greedy_dspark_matches_plain_decoding() {
        let path = std::env::temp_dir().join(format!("oxinfer-dspark-test-{}.gguf", std::process::id()));
        toy_draft(&path, None);
        assert!(is_draft(&path));
        let files = ModelFiles::open(&path, true).unwrap();
        let d = Dspark::load(&path, &files, H, V, Quant::Bf16).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(d.cfg.max_draft(), 3);
        let toy = Toy;
        let prompt = [5u32, 9, 13];
        let mut plain_s = toy.new_session();
        let hs = crate::generate::prefill(&toy, &mut plain_s, &prompt, 2, &[]);
        let lg = toy.logits(&hs[hs.len() - H..], 1);
        let mut s = Sampler::new(0.0, 0, 1.0, 1);
        let plain = crate::generate::decode(&toy, &mut plain_s, lg, &mut s, &[], 40, &mut |_| Flow::Continue);
        for k in 1..=3 {
            let mut dec = DsparkDecoder::new(
                &toy,
                &d,
                DsparkOptions {
                    draft_tokens: k,
                    ..Default::default()
                },
            );
            let lg = dec.prefill(&prompt, 2);
            let spec = crate::generate::decode_rounds(&mut dec, &lg, &mut s, &[], 40, &mut |_| Flow::Continue);
            assert_eq!(spec.tokens, plain.tokens, "k = {k}");
            let st = &dec.stats;
            assert!(st.accepted > 0 && st.accepted < st.drafted, "k = {k}: {st:?}");
            assert!(st.per_round[k] > 0, "k = {k}: no fully accepted round: {:?}", st.per_round);
            assert_eq!(dec.draft_session().len, dec.sess.0.len());
        }
    }

    /// The bounded context cache (window 4, slack 2) gives bit for bit the drafts of a cache that
    /// keeps every position, through injections of any size and truncations within the slack,
    /// and stays bounded.
    #[test]
    fn bounded_context_cache_matches_full_cache() {
        let path = std::env::temp_dir().join(format!("oxinfer-dspark-window-{}.gguf", std::process::id()));
        toy_draft(&path, Some(7));
        let files = ModelFiles::open(&path, true).unwrap();
        let d = Dspark::load(&path, &files, H, V, Quant::Bf16).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(d.cfg.window, 4);
        let toy = Toy;
        let mut bounded = d.new_session();
        bounded.slack = 2;
        let mut full = d.new_session();
        full.keep = 0;
        let n_feat = 2 * H;
        let mut rng = 1u64;
        let mut next = |m: u64| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) % m
        };
        let mut most = 0;
        for step in 0..60 {
            let t = 1 + next(5) as usize;
            let feats: Vec<f32> = (0..t * n_feat).map(|_| next(1000) as f32 / 500.0 - 1.0).collect();
            d.inject(&mut bounded, &feats, t);
            d.inject(&mut full, &feats, t);
            if step % 3 == 2 {
                // A rejected tail: back by up to the slack.
                let back = (next(3) as usize).min(bounded.len - 1);
                bounded.truncate(bounded.len - back);
                full.truncate(full.len - back);
            }
            assert_eq!(bounded.len, full.len);
            // The window before `len` is always cached.
            assert!(bounded.len - bounded.start() >= bounded.len.min(4));
            most = most.max(bounded.len - bounded.start());
            let anchor = next(V as u64) as u32;
            let (lb, db) = d.draft_logits(&toy, &bounded, anchor, 3);
            let (lf, df) = d.draft_logits(&toy, &full, anchor, 3);
            assert_eq!(db, df, "step {step}");
            assert!(lb.iter().zip(&lf).all(|(a, b)| a.to_bits() == b.to_bits()), "step {step}");
        }
        assert!(bounded.start() > 0 && most < 4 + 2 * 2, "start {} most {most}", bounded.start());
        assert!(bounded.bytes() < full.bytes() / 4);
        // A deep truncation keeps what is left; the block then sees fewer positions.
        let cut = bounded.start() - 1;
        bounded.truncate(cut);
        assert_eq!((bounded.start(), bounded.len), (cut, cut));
        d.inject(&mut bounded, &vec![0.5; n_feat], 1);
        assert_eq!(bounded.len - bounded.start(), 1);
        let (l, _) = d.draft_logits(&toy, &bounded, 3, 3);
        assert!(l.iter().all(|v| v.is_finite()));
    }
}
