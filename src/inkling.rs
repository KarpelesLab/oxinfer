//! Thinking Machines' Inkling-2 text decoder, as transformers' `Inkling2TextModel` (`inkling2`:
//! thinkingmachines/Inkling-2-Small and Inkling-2-Small-MXFP8). The vision tower is in
//! [`crate::inkling_vision`].
//!
//! - The token embedding goes through an RMS norm. There are no position embeddings and no
//!   rotary positions: attention gets its positions from a relative bias and from short causal
//!   convolutions.
//! - Attention (grouped-query, softmax): queries scaled per channel then clipped to an L2 norm
//!   (`qk_l2_clip_threshold`), keys clipped, values RMS-normed without a gain. A fourth
//!   projection gives each query head a `d_rel` vector (scaled, clipped) that mixes a learned
//!   bank of bias-versus-distance profiles (`rel_logits_proj`, `[d_rel, extent]`) into one bias
//!   per backward distance below the extent. Every query also has a sink: one more logit (a
//!   projection of the input) with a zero value, which only absorbs probability. The attended
//!   values are gated per channel (`sigmoid(gate_proj x)`) before the output projection.
//! - Keys and values pass a short causal depthwise convolution (kernel 4) with a skip, as do the
//!   output of the attention and of the MLP: `(conv(x) + x) / sqrt(kernel + 1)`.
//! - "Sliding" layers (5 of 6) see the last `sliding_window_size` tokens and may have their own
//!   head counts (`swa_*`). The others see everything: there the relative extent is `rel_extent`,
//!   queries and relative vectors grow with `1 + alpha ln(max(1, (pos + 1) / n_floor))`, and the
//!   key / value convolutions add per-token taps (a projection of the input, shared by the
//!   channels of a key/value head) to the learned ones, over `sqrt(2 kernel + 1)`.
//! - Attention residual: there is no plain residual stream across blocks of
//!   `n_layers_per_attn_res_block` layers. The input of the first layer of each block is kept as
//!   a "block state"; before its attention, before its MLP and before the final norm, a layer
//!   reads a softmax mix (a learned query against the RMS-normed states, scaled by
//!   `hidden^-0.5`) of the block states and of the stream inside the current block. A full layer
//!   reads three mixes (for its queries, keys and values), each with its own norm. Inside a
//!   block the stream is the sum of the layers' outputs; the first layer of a block starts it
//!   again from its attention output.
//! - MoE without a shared expert: sigmoid scores, the top k of score plus a selection bias,
//!   renormalized and scaled by `route_scale`; experts are SwiGLU with soft caps on the linear
//!   gate factor and on `up` (`cap tanh(x / cap)`; [`crate::moe::Act::Situ`]).
//! - The logits are soft-capped (`final_logit_softcapping`) and cut to `unpadded_vocab_size`
//!   (the unembedding is padded).
//!
//! Not here: the multi-token prediction head (`mtp.safetensors`, which transformers does not
//! run either).
//!
//! Checkpoints: the release's tensor names (`model.llm.*`, which `save_pretrained` writes too).
//! The routed experts (`w13_weight`, gate and up rows interleaved, and `w2_weight`) are read as
//! stored: MXFP8 (E4M3 values with an E8M0 exponent per 32) in place, NVFP4 (E2M1 pairs with an
//! E4M3 scale per 16 and `amax / outer_scale_denominator` per expert) repacked at load into
//! ggml's NVFP4 blocks, or plain floats.
//!
//! The convolutions keep only their last inputs, so a session cannot go back to an earlier
//! position (no prompt lookup, no draft model).

use crate::intervene::{NeuronHook, Target};
use crate::kernels::*;
use crate::model::{Lin, Quant, Table, linear, linear_multi, load_lin, load_lin_experts, table_quant};
use crate::moe::{Act, Experts, NeedExperts, route_sigmoid};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

/// The epsilon of the norm of the attention residual's states (the module's default, not
/// `rms_norm_eps`).
const MIX_EPS: f32 = 1e-6;
/// Added to a norm before an L2 clip divides by it.
const CLIP_EPS: f32 = 1e-6;
/// The prefix of the text decoder's tensors.
const P: &str = "model.llm.";

/// Whether `model_type` is one of this family's.
pub fn is_model_type(mt: &str) -> bool {
    matches!(mt, "inkling2" | "inkling2_text")
}

/// The head layout of one kind of layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heads {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// The settings of an Inkling-2 text decoder (`text_config`, in the release's spelling or in
/// transformers').
#[derive(Debug, Clone)]
pub struct Inkling2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Rows of the embedding and of the unembedding.
    pub vocab_size: usize,
    /// Rows of the unembedding that are tokens (the logits).
    pub unpadded_vocab_size: usize,
    pub full: Heads,
    pub swa: Heads,
    pub sliding_window_size: usize,
    /// Per layer: sliding-window attention (otherwise full).
    pub sliding: Vec<bool>,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: f64,
    pub log_scaling_alpha: f64,
    pub conv_kernel_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub route_scale: f64,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,
    pub n_layers_per_attn_res_block: usize,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    pub final_logit_softcapping: f64,
    pub rms_norm_eps: f64,
    /// The placeholder token of an image feature.
    pub image_token_id: u32,
}

impl Inkling2Config {
    /// The config of a checkpoint folder (`config.json`).
    pub fn load(dir: &Path) -> Result<Self> {
        let raw: Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &Value) -> Result<Self> {
        let mt = raw.get("model_type").and_then(Value::as_str).unwrap_or("");
        if !is_model_type(mt) {
            bail!("model_type {mt:?} is not an Inkling-2 model");
        }
        let t = raw.get("text_config").filter(|v| v.is_object()).unwrap_or(raw);
        let get = |keys: &[&str]| keys.iter().find_map(|k| t.get(*k).filter(|v| !v.is_null()));
        let int = |keys: &[&str], default: Option<usize>| -> Result<usize> {
            match get(keys) {
                Some(v) => v
                    .as_u64()
                    .map(|v| v as usize)
                    .with_context(|| format!("config: {} is not an integer", keys[0])),
                None => default.with_context(|| format!("config: {} is missing", keys[0])),
            }
        };
        let num = |keys: &[&str], default: f64| -> Result<f64> {
            match get(keys) {
                Some(v) => v.as_f64().with_context(|| format!("config: {} is not a number", keys[0])),
                None => Ok(default),
            }
        };
        // Variants of the architecture the release does not use and transformers does not have.
        for (key, want) in [
            ("q_bias", false),
            ("o_bias", false),
            ("pre_norm_pre_affine", false),
            ("final_norm_pre_affine", false),
            ("use_value_norm_no_gamma", true),
            ("attn_res_normalize_values", true),
            ("use_gate_bias", true),
        ] {
            if let Some(v) = t.get(key).and_then(Value::as_bool)
                && v != want
            {
                bail!("Inkling-2: {key} = {v} is not supported");
            }
        }
        if int(&["n_shared_experts"], Some(0))? != 0 {
            bail!("Inkling-2: shared experts are not supported");
        }
        if let Some(f) = get(&["scoring_func"]).and_then(Value::as_str)
            && f != "sigmoid"
        {
            bail!("Inkling-2: scoring_func {f:?} is not supported");
        }
        let n = int(&["num_hidden_layers"], Some(48))?;
        let block = int(&["n_layers_per_attn_res_block"], Some(6))?;
        let sliding: Vec<bool> = if let Some(lt) = get(&["layer_types"]).and_then(Value::as_array) {
            lt.iter()
                .map(|v| match v.as_str() {
                    Some("hybrid_sliding") => Ok(true),
                    Some("hybrid") => Ok(false),
                    other => bail!("Inkling-2: layer type {other:?}"),
                })
                .collect::<Result<_>>()?
        } else if let Some(ids) = get(&["local_layer_ids"]).and_then(Value::as_array) {
            let ids: Vec<u64> = ids.iter().filter_map(Value::as_u64).collect();
            (0..n as u64).map(|i| ids.contains(&i)).collect()
        } else {
            (0..n).map(|i| (i + 1) % 6 != 0).collect()
        };
        if sliding.len() != n {
            bail!("Inkling-2: {} layer types for {n} layers", sliding.len());
        }
        let full = Heads {
            heads: int(&["num_attention_heads"], Some(48))?,
            kv_heads: int(&["num_key_value_heads"], Some(8))?,
            head_dim: int(&["head_dim"], Some(128))?,
        };
        let swa = Heads {
            heads: int(&["swa_num_attention_heads"], Some(full.heads))?,
            kv_heads: int(&["swa_num_key_value_heads"], Some(full.kv_heads))?,
            head_dim: int(&["swa_head_dim"], Some(full.head_dim))?,
        };
        for hd in [full, swa] {
            if hd.kv_heads == 0 || !hd.heads.is_multiple_of(hd.kv_heads) {
                bail!("Inkling-2: {} heads over {} key/value heads", hd.heads, hd.kv_heads);
            }
        }
        let vocab_size = int(&["vocab_size"], Some(201024))?;
        let c = Self {
            hidden_size: int(&["hidden_size"], None)?,
            num_hidden_layers: n,
            vocab_size,
            unpadded_vocab_size: int(&["unpadded_vocab_size"], Some(vocab_size))?.min(vocab_size),
            full,
            swa,
            sliding_window_size: int(&["sliding_window_size", "sliding_window"], Some(1024))?,
            sliding,
            d_rel: int(&["d_rel"], Some(16))?,
            rel_extent: int(&["rel_extent"], Some(1024))?,
            log_scaling_n_floor: num(&["log_scaling_n_floor"], 128000.0)?,
            log_scaling_alpha: num(&["log_scaling_alpha"], 0.1)?,
            conv_kernel_size: int(&["conv_kernel_size", "sconv_kernel_size"], Some(4))?,
            moe_intermediate_size: int(&["moe_intermediate_size", "intermediate_size"], None)?,
            n_routed_experts: int(&["n_routed_experts", "num_local_experts"], None)?,
            num_experts_per_tok: int(&["num_experts_per_tok"], None)?,
            route_scale: num(&["route_scale", "routed_scaling_factor"], 8.0)?,
            qk_l2_clip_threshold: num(&["qk_l2_clip_threshold"], 22.627416997969522)?,
            rel_l2_clip_threshold: num(&["rel_l2_clip_threshold"], 8.0)?,
            n_layers_per_attn_res_block: block,
            mlp_gate_softcap: num(&["mlp_gate_softcap"], 4.0)?,
            mlp_up_softcap: num(&["mlp_up_softcap"], 25.0)?,
            final_logit_softcapping: num(&["final_logit_softcapping"], 30.0)?,
            rms_norm_eps: num(&["rms_norm_eps"], 1e-6)?,
            image_token_id: raw.get("image_token_id").and_then(Value::as_u64).unwrap_or(200054) as u32,
        };
        if c.conv_kernel_size == 0 || c.n_layers_per_attn_res_block == 0 || c.sliding_window_size == 0 {
            bail!("Inkling-2: a kernel, a window or a block of 0");
        }
        if c.num_experts_per_tok == 0 || c.num_experts_per_tok > c.n_routed_experts {
            bail!("Inkling-2: {} experts per token of {}", c.num_experts_per_tok, c.n_routed_experts);
        }
        Ok(c)
    }

    /// The heads of layer `l`.
    pub fn heads(&self, l: usize) -> Heads {
        if self.sliding[l] { self.swa } else { self.full }
    }
}

/// Weight streaming for a checkpoint that does not fit in memory (as Mistral Large 4).
pub fn default_stream() -> crate::stream::StreamOptions {
    crate::mistral4::default_stream()
}

/// Whether a checkpoint of `bytes` is too big to hold in memory (over 60% of RAM).
pub fn too_big(bytes: u64) -> bool {
    crate::mistral4::too_big(bytes)
}

/// The convolution of keys or values.
// Two per layer, built once.
#[allow(clippy::large_enum_variant)]
enum KvConv<'a> {
    /// Learned taps, `[lag][channels]` (lag 0 is the current token).
    Static(Vec<f32>),
    /// Learned taps plus per-token ones: `proj` (`[kv_heads * kernel, H]`) gives one tap per
    /// key/value head and lag.
    Dynamic { taps: Vec<f32>, proj: Lin<'a> },
}

struct Attn<'a> {
    hd: Heads,
    /// Sliding-window attention over this many tokens (the current one included).
    window: Option<usize>,
    /// Backward distances that have a relative bias.
    extent: usize,
    q: Lin<'a>,        // [heads * d, H]
    k: Lin<'a>,        // [kv * d, H]
    v: Lin<'a>,        // [kv * d, H]
    r: Lin<'a>,        // [heads * d_rel, H]
    gate: Lin<'a>,     // [heads * d, H]
    sink: Lin<'a>,     // [heads, H]
    o: Lin<'a>,        // [H, heads * d]
    q_scale: Vec<f32>, // [heads * d]
    r_scale: Vec<f32>, // [heads * d_rel]
    rel: Vec<f32>,     // [d_rel, extent]
    k_conv: KvConv<'a>,
    v_conv: KvConv<'a>,
}

/// The norms before the attention: one, or one each for queries, keys and values.
enum InNorm {
    One(Vec<f32>),
    Qkv(Vec<f32>, Vec<f32>, Vec<f32>),
}

struct Layer<'a> {
    /// The first layer of a block: its input becomes a block state.
    boundary: bool,
    /// The attention residual's queries before the attention (none on layer 0): `[H]`, or
    /// `[3, H]` on a full layer.
    attn_query: Option<Vec<f32>>,
    mlp_query: Vec<f32>,
    ln_in: InNorm,
    ln_post: Vec<f32>,
    attn: Attn<'a>,
    /// The convolutions after the attention and after the MLP, `[lag][H]`.
    attn_conv: Vec<f32>,
    mlp_conv: Vec<f32>,
    router: Vec<f32>, // [n_exp, H]
    bias: Vec<f32>,   // [n_exp]
    experts: Experts<'a>,
}

/// The decoding state of one layer.
#[derive(Clone, Default)]
struct LayerState {
    /// Clipped keys and normed values of positions `first..`, `[kv * d]` each.
    k: Vec<f32>,
    v: Vec<f32>,
    first: usize,
    /// The last `kernel - 1` inputs of the key, value, attention and MLP convolutions.
    hist: [Vec<f32>; 4],
}

#[derive(Clone)]
pub struct Inkling2Session {
    layers: Vec<LayerState>,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
}

pub struct Inkling2Model<'a> {
    pub cfg: Inkling2Config,
    pub quant: Quant,
    embed: Table<'a>,
    embed_norm: Vec<f32>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    final_query: Vec<f32>,
    layers: Vec<Layer<'a>>,
    /// Weight streaming: unit `l` is layer `l`, unit `num_hidden_layers` the LM head.
    stream: Option<crate::stream::StreamHook>,
}

struct Loader<'a, 'c> {
    files: &'a ModelFiles,
    cfg: &'c Inkling2Config,
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
    /// A matrix: float32 tensors exactly, others through `load_lin`.
    fn lin(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        if let Some(info) = self.files.info(full)
            && info.dtype == "F32"
        {
            if info.shape != [n_out, n_in] {
                bail!("tensor {full} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
            }
            self.files.note_use(full, true, false);
            return Lin::from_gguf(crate::ggml::GgmlType::F32, self.files.bytes(full)?, n_out, n_in);
        }
        load_lin(self.files, full, n_out, n_in, self.quant)
    }
    /// A small projection kept in float precision (never quantized at load).
    fn lin_fp(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        Loader {
            quant: Quant::Bf16,
            ..*self
        }
        .lin(full, n_out, n_in)
    }

    /// A depthwise convolution `[channels, 1, kernel]` as `[lag][channels]`: the last tap of the
    /// kernel is the current token's.
    fn conv(&self, full: &str, ch: usize) -> Result<Vec<f32>> {
        let k = self.cfg.conv_kernel_size;
        let w = self.getf(full, ch * k)?;
        let mut taps = vec![0f32; k * ch];
        for c in 0..ch {
            for j in 0..k {
                taps[(k - 1 - j) * ch + c] = w[c * k + j];
            }
        }
        Ok(taps)
    }

    fn kv_conv(&self, p: &str, ch: usize, kv: usize) -> Result<KvConv<'a>> {
        let k = self.cfg.conv_kernel_size;
        if self.files.has(&format!("{p}static_w")) {
            Ok(KvConv::Dynamic {
                // `[kernel, channels]`, tap index = lag.
                taps: self.getf(&format!("{p}static_w"), k * ch)?,
                proj: self.lin_fp(&format!("{p}weight_proj"), kv * k, self.cfg.hidden_size)?,
            })
        } else {
            Ok(KvConv::Static(self.conv(&format!("{p}weight"), ch)?))
        }
    }

    /// The experts of `[n_exp, n_out, n_in]` tensor `full`, in the format it is stored in.
    fn experts(&self, full: &str, n_out: usize, n_in: usize) -> Result<Vec<Lin<'a>>> {
        use crate::ggml::{GgmlType, GgufMatrix};
        use crate::quant::{PageVec, QBuf};
        let f = self.files;
        let ne = self.cfg.n_routed_experts;
        let info = f.info(full).with_context(|| format!("missing tensor {full}"))?;
        let sn = format!("{full}_scale");
        let scale_shape = |per: usize| -> Result<()> {
            let si = f.info(&sn).with_context(|| format!("missing tensor {sn}"))?;
            if !n_in.is_multiple_of(per) || si.shape != [ne, n_out, n_in / per] {
                bail!("{sn}: shape {:?} for [{ne}, {n_out}, {n_in}] in blocks of {per}", si.shape);
            }
            Ok(())
        };
        match info.dtype.as_str() {
            // NVFP4: E2M1 pairs, an E4M3 scale per 16, and per expert `amax / denominator`.
            "U8" => {
                if info.shape != [ne, n_out, n_in / 2] || !n_in.is_multiple_of(16) {
                    bail!("{full}: NVFP4 shape {:?} for [{ne}, {n_out}, {n_in}]", info.shape);
                }
                scale_shape(16)?;
                if f.info(&sn).is_some_and(|i| i.dtype != "F8_E4M3") {
                    bail!("{sn}: NVFP4 scales must be E4M3");
                }
                let amax = self.getf(&format!("{full}_amax"), ne)?;
                let den = self.getf(&format!("{full}_outer_scale_denominator"), 1)?[0];
                if den == 0.0 {
                    bail!("{full}_outer_scale_denominator is 0");
                }
                let (data, sc) = (f.bytes(full)?, f.bytes(&sn)?);
                for n in [full, &sn] {
                    f.note_use(n, true, false);
                }
                let (per, sper) = (n_out * n_in / 2, n_out * n_in / 16);
                (0..ne)
                    .map(|e| {
                        let (d, s) = (&data[e * per..(e + 1) * per], &sc[e * sper..(e + 1) * sper]);
                        let factor = amax[e] / den;
                        if n_in.is_multiple_of(64) {
                            return Ok(Lin::gguf(crate::model::nvfp4_blocks(d, s, factor, n_out, n_in)?));
                        }
                        // Rows that are not whole ggml blocks (tiny test checkpoints): as float32.
                        const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                        let mut out = PageVec::<u8>::zeroed(n_out * n_in * 4);
                        for (i, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                            let code = (d[i / 2] >> (4 * (i % 2))) & 0xF;
                            let v = E2M1[(code & 7) as usize] * if code & 8 != 0 { -1.0 } else { 1.0 };
                            *o = (v * crate::ggml::e4m3_to_f32(s[i / 16]) * factor).to_le_bytes();
                        }
                        Ok(Lin::gguf(GgufMatrix::new(GgmlType::F32, out.into(), n_out, n_in)?))
                    })
                    .collect()
            }
            // MXFP8: E4M3 values, an E8M0 exponent per 32 along each row, used in place.
            "F8_E4M3" => {
                if info.shape != [ne, n_out, n_in] {
                    bail!("{full}: MXFP8 shape {:?} for [{ne}, {n_out}, {n_in}]", info.shape);
                }
                scale_shape(32)?;
                if f.info(&sn).is_some_and(|i| i.dtype != "U8") {
                    bail!("{sn}: MXFP8 scales must be E8M0 bytes");
                }
                let (data, sc) = (f.bytes(full)?, f.bytes(&sn)?);
                for n in [full, &sn] {
                    f.note_use(n, true, false);
                }
                let (per, sper) = (n_out * n_in, n_out * n_in / 32);
                (0..ne)
                    .map(|e| {
                        let mut s = PageVec::<f32>::zeroed(sper);
                        for (d, &b) in s.iter_mut().zip(&sc[e * sper..(e + 1) * sper]) {
                            *d = if b == 255 { f32::NAN } else { (2f32).powi(b as i32 - 127) };
                        }
                        // SAFETY: the weight borrows `files`, which outlives every model holding it.
                        let d = unsafe { QBuf::borrowed(&data[e * per..(e + 1) * per]) };
                        Ok(Lin::gguf(GgufMatrix::fp8(d, s.into(), n_out, n_in, (1, 32))?))
                    })
                    .collect()
            }
            _ => load_lin_experts(f, full, ne, n_out, n_in, self.quant),
        }
    }

    fn attn(&self, l: usize) -> Result<Attn<'a>> {
        let c = self.cfg;
        let (h, hd, dr) = (c.hidden_size, c.heads(l), c.d_rel);
        let (w, kw) = (hd.heads * hd.head_dim, hd.kv_heads * hd.head_dim);
        let ap = format!("{P}layers.{l}.attn.");
        let extent = if c.sliding[l] { c.sliding_window_size } else { c.rel_extent };
        Ok(Attn {
            hd,
            window: c.sliding[l].then_some(c.sliding_window_size),
            extent,
            q: self.lin(&format!("{ap}wq_du.weight"), w, h)?,
            k: self.lin(&format!("{ap}wk_dv.weight"), kw, h)?,
            v: self.lin(&format!("{ap}wv_dv.weight"), kw, h)?,
            r: self.lin(&format!("{ap}wr_du.weight"), hd.heads * dr, h)?,
            gate: self.lin(&format!("{ap}gate_attn.weight"), w, h)?,
            sink: self.lin_fp(&format!("{ap}attn_sink_proj.weight"), hd.heads, h)?,
            o: self.lin(&format!("{ap}wo_ud.weight"), h, w)?,
            q_scale: self.getf(&format!("{ap}q_l2_clip.scale"), w)?,
            r_scale: self.getf(&format!("{ap}rel_l2_clip.scale"), hd.heads * dr)?,
            rel: self.getf(&format!("{ap}rel_logits_proj.proj"), dr * extent)?,
            k_conv: self.kv_conv(&format!("{ap}k_sconv."), kw, hd.kv_heads)?,
            v_conv: self.kv_conv(&format!("{ap}v_sconv."), kw, hd.kv_heads)?,
        })
    }

    fn layer(&self, l: usize) -> Result<Layer<'a>> {
        let c = self.cfg;
        let h = c.hidden_size;
        let lp = format!("{P}layers.{l}.");
        let attn = self.attn(l)?;
        if matches!(attn.k_conv, KvConv::Dynamic { .. }) == c.sliding[l] || matches!(attn.v_conv, KvConv::Dynamic { .. }) == c.sliding[l] {
            bail!(
                "the key / value convolutions are not those of a {} layer",
                if c.sliding[l] { "sliding" } else { "full" }
            );
        }
        let ln_in = if c.sliding[l] {
            InNorm::One(self.getf(&format!("{lp}attn_norm.weight"), h)?)
        } else {
            InNorm::Qkv(
                self.getf(&format!("{lp}q_input_norm.weight"), h)?,
                self.getf(&format!("{lp}k_input_norm.weight"), h)?,
                self.getf(&format!("{lp}v_input_norm.weight"), h)?,
            )
        };
        // `attn_queries.{i}` belongs to layer `i + 1` (layer 0 reads its input as it is).
        let attn_query = match l {
            0 => None,
            _ => Some(self.getf(&format!("{P}attn_res.attn_queries.{}", l - 1), if c.sliding[l] { h } else { 3 * h })?),
        };
        let (ne, k, mi) = (c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size);
        let act = Act::Situ {
            beta: c.mlp_gate_softcap as f32,
            linear_beta: c.mlp_up_softcap as f32,
        };
        let ep = format!("{lp}mlp.experts.");
        let experts = Experts::interleaved(
            self.experts(&format!("{ep}w13_weight"), 2 * mi, h)?,
            self.experts(&format!("{ep}w2_weight"), h, mi)?,
            k,
            mi,
            act,
        );
        Ok(Layer {
            boundary: l.is_multiple_of(c.n_layers_per_attn_res_block),
            attn_query,
            mlp_query: self.getf(&format!("{P}attn_res.mlp_queries.{l}"), h)?,
            ln_in,
            ln_post: self.getf(&format!("{lp}mlp_norm.weight"), h)?,
            attn,
            attn_conv: self.conv(&format!("{lp}attn_sconv.weight"), h)?,
            mlp_conv: self.conv(&format!("{lp}mlp_sconv.weight"), h)?,
            router: self.getf(&format!("{lp}mlp.gate.weight"), ne * h)?,
            bias: self.getf(&format!("{lp}mlp.gate.bias"), ne)?,
            experts,
        })
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scales `x` down to L2 norm `threshold` when it is longer.
#[inline]
fn l2_clip(x: &mut [f32], threshold: f32) {
    let f = threshold / (dot(x, x).sqrt() + CLIP_EPS);
    if f < 1.0 {
        scale_inplace(x, f);
    }
}

/// `x * rsqrt(mean(x^2) + eps)`.
#[inline]
fn rms_unit(x: &mut [f32], eps: f32) {
    let r = 1.0 / (dot(x, x) / x.len() as f32 + eps).sqrt();
    scale_inplace(x, r);
}

/// A causal depthwise convolution with a skip over `x` (`[t, ch]`), in place:
/// `(sum_lag taps[lag] * x[t - lag] + x[t]) * norm`. `hist` holds the `kernel - 1` inputs before
/// `x` and is left holding the last ones. With `dynamic` (`[t, groups * kernel]`), channel `c`
/// of token `t` adds `dynamic[t][c / (ch / groups) * kernel + lag]` to its taps.
fn short_conv(taps: &[f32], dynamic: Option<(&[f32], usize)>, hist: &mut Vec<f32>, x: &mut [f32], ch: usize, norm: f32) {
    let k = taps.len() / ch;
    let t = x.len() / ch;
    let back = (k - 1) * ch;
    debug_assert_eq!(hist.len(), back);
    let mut seq = std::mem::take(hist);
    seq.extend_from_slice(x);
    for_chunks(x, ch, t > 1, |ti, row| {
        // Row `ti` of `x` is row `ti + k - 1` of `seq`.
        for (c, o) in row.iter_mut().enumerate() {
            *o = seq[back + ti * ch + c];
        }
        for lag in 0..k {
            let src = &seq[(ti + k - 1 - lag) * ch..(ti + k - lag) * ch];
            let w = &taps[lag * ch..(lag + 1) * ch];
            match dynamic {
                None => {
                    for ((o, &s), &wv) in row.iter_mut().zip(src).zip(w) {
                        *o += wv * s;
                    }
                }
                Some((d, groups)) => {
                    let per = ch / groups;
                    for (g, ((oc, sc), wc)) in row.chunks_mut(per).zip(src.chunks(per)).zip(w.chunks(per)).enumerate() {
                        let dw = d[(ti * groups + g) * k + lag];
                        for ((o, &s), &wv) in oc.iter_mut().zip(sc).zip(wc) {
                            *o += (wv + dw) * s;
                        }
                    }
                }
            }
        }
        scale_inplace(row, norm);
    });
    *hist = seq[t * ch..].to_vec();
}

/// The attention residual: for each of the `nq` queries (`[nq, h]`) and each token, the softmax
/// mix (`query . state * h^-0.5`) of the block states (`[t, h]` each, already RMS-normed) and of
/// the RMS-normed `x` (`[t, h]`). One `[t, h]` result per query.
fn mix(states: &[Vec<f32>], x: &[f32], queries: &[f32], h: usize) -> Vec<Vec<f32>> {
    let _timer = prof::start(Cat::Norm);
    let nq = queries.len() / h;
    let t = x.len() / h;
    let scale = (h as f32).powf(-0.5);
    let mut out = vec![0f32; t * nq * h];
    for_chunks(&mut out, nq * h, t > 4, |ti, o| {
        let mut xn = x[ti * h..(ti + 1) * h].to_vec();
        rms_unit(&mut xn, MIX_EPS);
        let mut s = vec![0f32; states.len() + 1];
        for (q, oq) in queries.chunks(h).zip(o.chunks_mut(h)) {
            for (v, st) in s.iter_mut().zip(states) {
                *v = dot(q, &st[ti * h..(ti + 1) * h]) * scale;
            }
            *s.last_mut().unwrap() = dot(q, &xn) * scale;
            softmax_inplace(&mut s);
            for (&w, st) in s.iter().zip(states) {
                axpy(oq, w, &st[ti * h..(ti + 1) * h]);
            }
            axpy(oq, *s.last().unwrap(), &xn);
        }
    });
    if nq == 1 {
        return vec![out];
    }
    (0..nq)
        .map(|q| {
            (0..t)
                .flat_map(|ti| out[(ti * nq + q) * h..(ti * nq + q + 1) * h].iter().copied())
                .collect()
        })
        .collect()
}

impl<'a> Inkling2Model<'a> {
    pub fn load(cfg: Inkling2Config, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let en = format!("{P}embed.weight");
        if !files.has(&en) {
            bail!("cannot find the token embedding ({en}) in the checkpoint");
        }
        let ld = Loader { files, cfg: &cfg, quant };
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let embed = Table::load(files, &en, v, h, table_quant(quant))?;
        let lm_head = ld.lin(&format!("{P}unembed.weight"), v, h)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            layers.push(ld.layer(l).with_context(|| format!("layer {l}"))?);
        }
        let m = Self {
            embed,
            embed_norm: ld.getf(&format!("{P}embed_norm.weight"), h)?,
            lm_head,
            final_norm: ld.getf(&format!("{P}norm.weight"), h)?,
            final_query: ld.getf(&format!("{P}attn_res.output_query"), h)?,
            layers,
            cfg,
            quant,
            stream: None,
        };
        // Routed experts that cannot all stay in memory are read through the page cache as they
        // are routed: no accelerator mapping may pin them.
        if too_big(files.total_bytes()) {
            for l in &m.layers {
                for e in l.experts.lins().into_iter().flatten() {
                    e.set_transient_device(true);
                }
            }
        }
        Ok(m)
    }

    pub fn new_session(&self) -> Inkling2Session {
        let c = &self.cfg;
        let back = c.conv_kernel_size - 1;
        Inkling2Session {
            layers: (0..c.num_hidden_layers)
                .map(|l| {
                    let hd = c.heads(l);
                    let kw = hd.kv_heads * hd.head_dim;
                    LayerState {
                        hist: [
                            vec![0f32; back * kw],
                            vec![0f32; back * kw],
                            vec![0f32; back * c.hidden_size],
                            vec![0f32; back * c.hidden_size],
                        ],
                        ..Default::default()
                    }
                })
                .collect(),
            pos: 0,
            iv: Default::default(),
        }
    }

    fn dense_lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let a = &l.attn;
        let mut v = vec![&a.q, &a.k, &a.v, &a.r, &a.gate, &a.sink, &a.o];
        for c in [&a.k_conv, &a.v_conv] {
            if let KvConv::Dynamic { proj, .. } = c {
                v.push(proj);
            }
        }
        v
    }

    /// Weight bytes read per decoded token (the routed experts at top-k).
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes();
        for l in &self.layers {
            n += self.dense_lins(l).iter().map(|w| w.bytes()).sum::<usize>();
            n += l.experts.bytes_per_token();
        }
        n
    }

    /// The streaming units in execution order (see `crate::stream`): the layers (routed experts
    /// read on demand), then the LM head.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        let mut v: Vec<crate::stream::UnitSpec> = self
            .layers
            .iter()
            .enumerate()
            .map(|(l, layer)| crate::stream::UnitSpec {
                name: format!("layer.{l}"),
                ranges: self.dense_lins(layer).iter().flat_map(|w| w.mem_ranges()).collect(),
                uses: 1.0,
                parts: layer.experts.ranges(),
            })
            .collect();
        v.push(crate::stream::UnitSpec {
            name: "lm_head".into(),
            ranges: self.lm_head.mem_ranges(),
            uses: 1.0,
            parts: Vec::new(),
        });
        v
    }

    /// Marks the weights of the units with `streamed[i]` set as streamed (no accelerator state
    /// kept between calls), and every routed expert.
    pub fn mark_streamed(&self, streamed: &[bool]) {
        for (l, layer) in self.layers.iter().enumerate() {
            let s = streamed.get(l).copied().unwrap_or(false);
            for w in self.dense_lins(layer) {
                w.set_transient_device(s);
            }
            for e in layer.experts.lins().into_iter().flatten() {
                e.set_transient_device(true);
            }
        }
        self.lm_head
            .set_transient_device(streamed.get(self.layers.len()).copied().unwrap_or(false));
    }

    /// Brackets every layer and the LM head with `hook`'s units (None stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// The unembedding over `hidden` (`[t, hidden]`), cut to the tokens: `[t, unpadded_vocab]`.
    fn head(&self, hidden: &[f32], t: usize, cap: bool) -> Vec<f32> {
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let (v, u) = (c.vocab_size, c.unpadded_vocab_size);
        let mut logits = vec![0f32; t * v];
        {
            let _unit = crate::stream::enter(&self.stream, self.layers.len());
            linear(&self.lm_head, v, c.hidden_size, hidden, t, &mut logits);
        }
        if u < v {
            for ti in 1..t {
                logits.copy_within(ti * v..ti * v + u, ti * u);
            }
            logits.truncate(t * u);
        }
        if cap {
            let s = c.final_logit_softcapping as f32;
            let chunk = u.min(4096);
            for_chunks(&mut logits, chunk, t * u > 16 * chunk, |_, r| {
                for x in r.iter_mut() {
                    *x = (*x / s).tanh() * s;
                }
            });
        }
        logits
    }

    /// Soft-capped logits `[t, unpadded_vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.head(hidden, t, true)
    }

    /// The input embedding of `token` (`[hidden]`, before its norm).
    pub fn embed_row(&self, token: u32, out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        if let Some(s) = &self.stream {
            s.need_ranges(&self.embed.row_ranges(token as usize, h));
        }
        self.embed.row(token as usize, out);
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut Inkling2Session, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden(sess, tokens);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos`, appending to the caches; returns the final-normed hidden
    /// states `[t, hidden]`.
    pub fn forward_hidden(&self, sess: &mut Inkling2Session, tokens: &[u32]) -> Vec<f32> {
        self.forward_capture(sess, tokens, &[], &[], &mut Vec::new())
    }

    /// [`Self::forward_hidden`] where the positions listed in `mm` (relative to this call) take
    /// the given embedding (an image feature, which skips the embedding norm), and that also
    /// writes to `feats` (`[t, layers.len() * hidden]`) the stream entering each of `layers`,
    /// `num_hidden_layers` standing for the last layer's output.
    pub fn forward_capture(
        &self,
        sess: &mut Inkling2Session,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        let t = tokens.len();
        assert!(t > 0);
        let c = &self.cfg;
        let h = c.hidden_size;
        let eps = c.rms_norm_eps as f32;
        let start = sess.pos;
        let timer = prof::start(Cat::Embed);
        if let Some(s) = &self.stream {
            let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| self.embed.row_ranges(tk as usize, h)).collect();
            s.need_ranges(&rows);
        }
        let mut x = vec![0f32; t * h];
        for_chunks(&mut x, h, t > 1, |ti, row| {
            self.embed.row(tokens[ti] as usize, row);
            rmsnorm_f32w_inplace(row, &self.embed_norm, eps);
        });
        for &(i, e) in mm {
            x[i * h..(i + 1) * h].copy_from_slice(e);
        }
        drop(timer);
        let iv = sess.iv.active().then(|| sess.iv.clone());
        let point = |p: usize, x: &mut [f32]| {
            if let Some(iv) = &iv {
                iv.point(p, x, h, start);
            }
        };
        point(0, &mut x);
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
        // The inputs of the first layer of each block so far, RMS-normed (all that reads them).
        let mut states: Vec<Vec<f32>> = Vec::new();
        let conv_norm = 1.0 / ((c.conv_kernel_size + 1) as f32).sqrt();
        for (l, layer) in self.layers.iter().enumerate() {
            capture(l, &x);
            let _unit = crate::stream::enter(&self.stream, l);
            let st = &mut sess.layers[l];
            // What the attention reads: the layer's input, or mixes of it with the block states.
            let mixed = layer.attn_query.as_ref().map(|q| mix(&states, &x, q, h));
            let input = |i: usize| -> &[f32] {
                match &mixed {
                    Some(m) => &m[i.min(m.len() - 1)],
                    None => &x,
                }
            };
            let normed = |w: &[f32], src: &[f32]| {
                let mut y = vec![0f32; t * h];
                rmsnorm_f32w_rows(src, h, w, eps, &mut y);
                y
            };
            let mut a = vec![0f32; t * h];
            match &layer.ln_in {
                InNorm::One(w) => {
                    let xn = normed(w, input(0));
                    self.attention(&layer.attn, [&xn, &xn, &xn], t, start, st, &mut a);
                }
                InNorm::Qkv(wq, wk, wv) => {
                    let (xq, xk, xv) = (normed(wq, input(0)), normed(wk, input(1)), normed(wv, input(2)));
                    self.attention(&layer.attn, [&xq, &xk, &xv], t, start, st, &mut a);
                }
            }
            drop(mixed);
            short_conv(&layer.attn_conv, None, &mut st.hist[2], &mut a, h, conv_norm);
            // The stream inside the block: a block's first layer starts it from its attention.
            if layer.boundary {
                for_chunks(&mut x, h, t > 4, |_, r| rms_unit(r, MIX_EPS));
                states.push(std::mem::replace(&mut x, a));
            } else {
                add_inplace(&mut x, &a);
            }
            point(2 * l + 1, &mut x);
            let mut hn = mix(&states, &x, &layer.mlp_query, h).pop().unwrap();
            rmsnorm_f32w_rows_inplace(&mut hn, h, &layer.ln_post, eps);
            let mut o = vec![0f32; t * h];
            let hook = iv.as_ref().and_then(|s| s.hook(l, start));
            {
                let need = |ids: &[usize]| {
                    if let Some(s) = &self.stream {
                        s.need_parts(l, ids)
                    }
                };
                let need: NeedExperts = self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize]));
                self.moe(layer, &hn, t, &mut o, need, hook.as_ref());
            }
            short_conv(&layer.mlp_conv, None, &mut st.hist[3], &mut o, h, conv_norm);
            add_inplace(&mut x, &o);
            point(2 * l + 2, &mut x);
        }
        capture(self.layers.len(), &x);
        let mut out = mix(&states, &x, &self.final_query, h).pop().unwrap();
        rmsnorm_f32w_rows_inplace(&mut out, h, &self.final_norm, eps);
        sess.pos += t;
        out
    }

    /// Routes each token (sigmoid scores, the top k of score plus bias, renormalized and scaled)
    /// and runs its experts.
    fn moe(&self, layer: &Layer, x: &[f32], t: usize, y: &mut [f32], need: NeedExperts, hook: Option<&NeuronHook>) {
        let c = &self.cfg;
        let h = c.hidden_size;
        let ne = c.n_routed_experts;
        let mut logits = vec![0f32; t * ne];
        for_chunks(&mut logits, ne, t > 1, |ti, lg| {
            for (v, w) in lg.iter_mut().zip(layer.router.chunks(h)) {
                *v = dot(w, &x[ti * h..(ti + 1) * h]);
            }
        });
        let routes: Vec<_> = logits
            .chunks(ne)
            .map(|lg| route_sigmoid(lg, &layer.bias, c.num_experts_per_tok, c.route_scale as f32))
            .collect();
        layer.experts.forward(x, t, h, &routes, y, need, hook, None);
    }

    /// Attention over the normed inputs `xs` (`[t, H]` each: what the queries, the keys and the
    /// values are projected from) into `out`, appending to the layer's cache.
    fn attention(&self, a: &Attn, xs: [&[f32]; 3], t: usize, start: usize, st: &mut LayerState, out: &mut [f32]) {
        let c = &self.cfg;
        let (h, dr, kn) = (c.hidden_size, c.d_rel, c.conv_kernel_size);
        let Heads {
            heads: nh,
            kv_heads: nkv,
            head_dim: d,
        } = a.hd;
        let (w, kw) = (nh * d, nkv * d);
        let [xq, xk, xv] = xs;
        let mut q = vec![0f32; t * w];
        let mut r = vec![0f32; t * nh * dr];
        let mut gate = vec![0f32; t * w];
        let mut sink = vec![0f32; t * nh];
        let mut k = vec![0f32; t * kw];
        let mut v = vec![0f32; t * kw];
        let dynamic = matches!(a.k_conv, KvConv::Dynamic { .. });
        let mut kd = vec![0f32; if dynamic { t * nkv * kn } else { 0 }];
        let mut vd = vec![0f32; if dynamic { t * nkv * kn } else { 0 }];
        {
            let mut jq: Vec<(&Lin, usize, &mut [f32])> = vec![
                (&a.q, w, &mut q[..]),
                (&a.r, nh * dr, &mut r[..]),
                (&a.gate, w, &mut gate[..]),
                (&a.sink, nh, &mut sink[..]),
            ];
            let mut jk: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.k, kw, &mut k[..])];
            let mut jv: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.v, kw, &mut v[..])];
            if let KvConv::Dynamic { proj, .. } = &a.k_conv {
                jk.push((proj, nkv * kn, &mut kd[..]));
            }
            if let KvConv::Dynamic { proj, .. } = &a.v_conv {
                jv.push((proj, nkv * kn, &mut vd[..]));
            }
            if std::ptr::eq(xq, xk) && std::ptr::eq(xq, xv) {
                // One input: all the projections in one parallel region.
                jq.append(&mut jk);
                jq.append(&mut jv);
                linear_multi(&mut jq, h, xq, t);
            } else {
                linear_multi(&mut jq, h, xq, t);
                linear_multi(&mut jk, h, xk, t);
                linear_multi(&mut jv, h, xv, t);
            }
        }
        let _timer = prof::start(Cat::Attention);
        for (conv, x, dy, hist) in [(&a.k_conv, &mut k, &kd, 0), (&a.v_conv, &mut v, &vd, 1)] {
            match conv {
                KvConv::Static(taps) => short_conv(taps, None, &mut st.hist[hist], x, kw, 1.0 / ((kn + 1) as f32).sqrt()),
                KvConv::Dynamic { taps, .. } => {
                    short_conv(taps, Some((dy, nkv)), &mut st.hist[hist], x, kw, 1.0 / ((2 * kn + 1) as f32).sqrt())
                }
            }
        }
        // The full layers scale queries and relative vectors up with the position.
        let tau = |ti: usize| -> f32 {
            if a.window.is_some() {
                return 1.0;
            }
            let n = ((start + ti + 1) as f64 / c.log_scaling_n_floor).max(1.0);
            (1.0 + c.log_scaling_alpha * n.ln()) as f32
        };
        let (qk_clip, rel_clip) = (c.qk_l2_clip_threshold as f32, c.rel_l2_clip_threshold as f32);
        for_chunks(&mut q, d, t > 1, |i, qh| {
            let hh = i % nh;
            for (x, &s) in qh.iter_mut().zip(&a.q_scale[hh * d..(hh + 1) * d]) {
                *x *= s;
            }
            l2_clip(qh, qk_clip);
            let f = tau(i / nh);
            if f != 1.0 {
                scale_inplace(qh, f);
            }
        });
        for_chunks(&mut r, dr, t > 1, |i, rh| {
            let hh = i % nh;
            for (x, &s) in rh.iter_mut().zip(&a.r_scale[hh * dr..(hh + 1) * dr]) {
                *x *= s;
            }
            l2_clip(rh, rel_clip);
            let f = tau(i / nh);
            if f != 1.0 {
                scale_inplace(rh, f);
            }
        });
        for_chunks(&mut k, d, t > 1, |_, kh| l2_clip(kh, qk_clip));
        let eps = c.rms_norm_eps as f32;
        for_chunks(&mut v, d, t > 1, |_, vh| rms_unit(vh, eps));
        st.k.extend_from_slice(&k);
        st.v.extend_from_slice(&v);

        let groups = nh / nkv;
        let scaling = (d as f32).powf(-0.5);
        let (first, extent) = (st.first, a.extent);
        let (ck, cv) = (&st.k, &st.v);
        let mut att = vec![0f32; t * w];
        for_chunks(&mut att, d, true, |task, o| {
            let (i, hh) = (task / nh, task % nh);
            let kvh = hh / groups;
            let p = start + i;
            let lo = match a.window {
                Some(win) => (p + 1).saturating_sub(win).max(first),
                None => first,
            };
            let n = p + 1 - lo;
            let qh = &q[task * d..(task + 1) * d];
            // The relative bias by backward distance: this head's mix of the profiles.
            let nb = n.min(extent);
            let mut bias = vec![0f32; nb];
            for (&rv, prof) in r[task * dr..(task + 1) * dr].iter().zip(a.rel.chunks(extent)) {
                axpy(&mut bias, rv, &prof[..nb]);
            }
            // Scores of positions `lo..=p`, then the sink.
            let mut s = vec![0f32; n + 1];
            for (j, sv) in s[..n].iter_mut().enumerate() {
                let at = (lo + j - first) * kw + kvh * d;
                let dist = p - (lo + j);
                *sv = dot(qh, &ck[at..at + d]) * scaling + if dist < nb { bias[dist] } else { 0.0 };
            }
            s[n] = sink[task];
            softmax_inplace(&mut s);
            for (j, &wj) in s[..n].iter().enumerate() {
                let at = (lo + j - first) * kw + kvh * d;
                axpy(o, wj, &cv[at..at + d]);
            }
        });
        // A sliding layer never reads further back than its window: what is older goes, a window's
        // worth at a time.
        if let Some(win) = a.window {
            let keep = (start + t + 1).saturating_sub(win);
            if keep >= st.first + win {
                let cut = (keep - st.first) * kw;
                st.k.drain(..cut);
                st.v.drain(..cut);
                st.first = keep;
            }
        }
        for (x, &g) in att.iter_mut().zip(&gate) {
            *x *= sigmoid(g);
        }
        drop(_timer);
        linear(&a.o, h, w, &att, t, out);
    }
}

impl crate::generate::TextLm for Inkling2Model<'_> {
    type Sess = Inkling2Session;
    fn new_session(&self) -> Inkling2Session {
        Inkling2Model::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut Inkling2Session) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &Inkling2Session) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut Inkling2Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, &[], &mut Vec::new())
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        Inkling2Model::logits(self, hidden, t)
    }
}

/// Runtime interventions (`crate::intervene`): the stream after the embedding, after each
/// layer's attention and after each layer (inside a block: the sum of the layers' outputs so
/// far), and as neurons the input of `down_proj` of the routed experts (`Target::Expert`, before
/// the routing weight).
impl crate::intervene::Host for Inkling2Model<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        match target {
            Target::Expert(e) if e < self.layers.get(layer)?.experts.n_experts() => Some(self.cfg.moe_intermediate_size),
            _ => None,
        }
    }
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32> {
        let Target::Expert(e) = target else {
            unreachable!("Inkling-2 has routed experts only");
        };
        let (h, mi) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        let mut one = vec![0f32; mi];
        one[n] = 1.0;
        let mut y = vec![0f32; h];
        // A streamed checkpoint reads the layer (and the routed expert) as a pass would.
        let _unit = crate::stream::enter(&self.stream, layer);
        if let Some(s) = &self.stream {
            s.need_parts(layer, &[e]);
        }
        linear(&self.layers[layer].experts.down[e], h, mi, &one, 1, &mut y);
        y
    }
}

/// What a speculative decoder needs of its target. The convolutions cannot go back to an earlier
/// position, so no draft is offered for this model (`truncate` then panics).
impl crate::dspark::Target for Inkling2Model<'_> {
    fn forward_feats(
        &self,
        sess: &mut Inkling2Session,
        tokens: &[u32],
        mm: &[(usize, &[f32])],
        layers: &[usize],
        feats: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, layers, feats)
    }
    fn embed_row(&self, token: u32, out: &mut [f32]) {
        Inkling2Model::embed_row(self, token, out)
    }
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        self.head(hidden, t, false)
    }
    // The logits cover the tokens only (the unembedding is padded).
    #[allow(clippy::misnamed_getters)]
    fn vocab_size(&self) -> usize {
        self.cfg.unpadded_vocab_size
    }
    fn truncate(&self, sess: &mut Inkling2Session, len: usize) {
        assert!(len >= sess.pos, "Inkling-2: the convolutions cannot go back to an earlier position");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The release's config spelling and transformers' give the same settings.
    #[test]
    fn config_spellings() {
        let hub = json!({"model_type": "inkling2", "text_config": {
            "hidden_size": 4608, "num_hidden_layers": 12, "vocab_size": 201024, "unpadded_vocab_size": 200058,
            "num_attention_heads": 48, "num_key_value_heads": 8, "head_dim": 128,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], "sconv_kernel_size": 4,
            "n_routed_experts": 384, "num_experts_per_tok": 8, "n_shared_experts": 0, "intermediate_size": 2304,
            "route_scale": 8.0, "use_gate_bias": true, "use_value_norm_no_gamma": true}});
        let c = Inkling2Config::from_json(&hub).unwrap();
        assert_eq!(
            (c.moe_intermediate_size, c.conv_kernel_size, c.unpadded_vocab_size),
            (2304, 4, 200058)
        );
        assert_eq!(c.swa, c.full);
        assert_eq!(c.sliding.iter().filter(|s| !**s).count(), 2);
        assert!(!c.sliding[5] && !c.sliding[11]);
        let tf = json!({"model_type": "inkling2", "text_config": {
            "hidden_size": 64, "num_hidden_layers": 6, "conv_kernel_size": 4, "moe_intermediate_size": 32,
            "n_routed_experts": 16, "num_experts_per_tok": 4, "swa_num_attention_heads": 4, "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "layer_types": ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid"]}});
        let c = Inkling2Config::from_json(&tf).unwrap();
        assert_eq!(c.sliding, [true, false, true, true, true, false]);
        assert_eq!((c.heads(0).heads, c.heads(1).heads), (4, 8));
        // A variant of the architecture that is not implemented is refused.
        let bad = json!({"model_type": "inkling2", "text_config": {"hidden_size": 64, "q_bias": true}});
        assert!(Inkling2Config::from_json(&bad).is_err());
    }

    /// The convolution against its definition, fed whole or token by token.
    #[test]
    fn short_conv_steps() {
        let (ch, k, t) = (4usize, 3usize, 5usize);
        let taps: Vec<f32> = (0..k * ch).map(|i| 0.1 * i as f32 - 0.3).collect();
        let x: Vec<f32> = (0..t * ch).map(|i| ((i * 7 % 11) as f32) * 0.25 - 1.0).collect();
        let dy: Vec<f32> = (0..t * 2 * k).map(|i| ((i * 5 % 7) as f32) * 0.1).collect();
        for dynamic in [false, true] {
            let want: Vec<f32> = (0..t * ch)
                .map(|i| {
                    let (ti, c) = (i / ch, i % ch);
                    let mut s = x[i];
                    for lag in 0..=ti.min(k - 1) {
                        let d = if dynamic { dy[(ti * 2 + c / 2) * k + lag] } else { 0.0 };
                        s += (taps[lag * ch + c] + d) * x[(ti - lag) * ch + c];
                    }
                    s * 0.5
                })
                .collect();
            let d = |a: usize, b: usize| dynamic.then(|| (&dy[a * 2 * k..b * 2 * k], 2));
            let mut whole = x.clone();
            short_conv(&taps, d(0, t), &mut vec![0f32; (k - 1) * ch], &mut whole, ch, 0.5);
            let mut hist = vec![0f32; (k - 1) * ch];
            let mut steps = x.clone();
            for ti in 0..t {
                short_conv(&taps, d(ti, ti + 1), &mut hist, &mut steps[ti * ch..(ti + 1) * ch], ch, 0.5);
            }
            for ((a, b), w) in whole.iter().zip(&steps).zip(&want) {
                assert!((a - w).abs() < 1e-5 && (b - w).abs() < 1e-5, "{a} {b} {w}");
            }
        }
    }

    /// Without block states the mix is the RMS-normed input; a zero query averages.
    #[test]
    fn mix_is_a_softmax_over_normed_states() {
        let h = 4;
        let x = [3.0f32, 0.0, 0.0, 0.0];
        let alone = mix(&[], &x, &[1.0, 2.0, 3.0, 4.0], h);
        assert!((alone[0][0] - 2.0).abs() < 1e-4 && alone[0][1] == 0.0);
        let state = vec![0.0f32, 2.0, 0.0, 0.0];
        let both = mix(&[state], &x, &[0.0; 8], h);
        assert_eq!(both.len(), 2);
        assert!((both[1][0] - 1.0).abs() < 1e-4 && (both[1][1] - 1.0).abs() < 1e-4);
    }
}
