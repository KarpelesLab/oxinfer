//! Mistral Large 4 (`mistral_large4`, Mistral-Large-4-1T-A52B) text model, as transformers'
//! `MistralLarge4TextModel`:
//!
//! - MLA attention without a query LoRA: `q_proj` gives each head a non-rotary and a rotary part;
//!   `kv_a_proj_with_mqa` compresses the token into a latent (RMS-normed, `kv_lora_rank` wide) and
//!   one shared rotary key; `kv_b_proj` expands the latent into each head's non-rotary key and its
//!   value. Only the latent and the rotary key are cached. Decoding absorbs `kv_b_proj` into the
//!   query and the output (scores against the latent directly, as llama.cpp does); prefill
//!   expands the keys and values and runs ordinary attention as GEMMs (on the GPU when it can).
//! - NoPE layers (`no_rope_layers[i] == 0`) have no rotary part: the whole query / key head is
//!   non-rotary and the latent comes alone.
//! - Per-position query scaling ("SSMax"): each head's query times `s_h * ln(pos + 1) + b_h`.
//! - RoPE on interleaved pairs with YaRN frequencies times learned factors (`exp(alpha)`,
//!   "LeRoPE"). Queries and keys are rotated in place (the reference concatenates the rotated
//!   even and odd lanes instead; the dot products are the same).
//! - KV sharing: a layer whose `kv_source_layers` entry is another layer has no `kv_a`
//!   projection and attends to that layer's latent cache with its own `kv_b`.
//! - DSA (DeepSeek sparse attention): an indexer (32 heads of 128 with its own keys, LayerNorm'd,
//!   rotated on the first rotary dims) scores every earlier token, `sum_h w_h relu(q_h . k)`, and
//!   each query attends only to its `index_topk` best keys; "shared" layers reuse the previous
//!   full layer's choice. Queries with at most `index_topk` keys see them all, so short contexts
//!   are dense attention.
//! - DeepSeek-style MoE after a dense first layer: softmax router over the routed experts (top
//!   k, renormalized, optional groups), plus a shared expert; final logits soft-capped.
//!
//! Checkpoints: transformers-named safetensors, Mistral's own `consolidated*.safetensors` layout
//! (renamed on open, `native_name`), FP8 block-quantized weights, and llama.cpp GGUF files
//! (`mistral-large4`, which split `kv_b` per head into `attn_k_b` / `attn_v_b`).

use crate::config::Mistral4Config;
use crate::intervene::{NeuronHook, Target};
use crate::kernels::*;
use crate::model::{Lin, Quant, Table, Weight, linear, linear_grouped, linear_multi, load_lin, table_quant};
use crate::moe::{Act, Experts, NeedExperts, route_top_k};
use crate::prof::{self, Cat};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// Chunks of at least this many tokens expand the latent cache into keys and values (prefill);
/// shorter ones score against the latent (decode).
const EXPAND_MIN: usize = 32;

/// The transformers name of a tensor in Mistral's own checkpoint layout (`consolidated*`,
/// `params.json` naming), as transformers' conversion mapping renames it; None for names that
/// stay (or that no loader reads). FP8 block scales (`qscale_weight`) become `weight_scale_inv`.
pub(crate) fn native_name(n: &str) -> Option<String> {
    let n = n.replace(".qscale_weight", ".weight_scale_inv");
    let top = match n.as_str() {
        "tok_embeddings.weight" => Some("model.language_model.embed_tokens.weight"),
        "norm.weight" => Some("model.language_model.norm.weight"),
        "output.weight" => Some("lm_head.weight"),
        "lerope.alpha" => Some("model.language_model.rotary_emb.alpha"),
        "pre_mm_projector_norm.weight" => Some("model.multi_modal_projector.norm.weight"),
        "patch_merger.merging_layer.weight" => Some("model.multi_modal_projector.patch_merger.merging_layer.weight"),
        "vision_language_adapter.w_in.weight" => Some("model.multi_modal_projector.linear_1.weight"),
        "vision_language_adapter.w_out.weight" => Some("model.multi_modal_projector.linear_2.weight"),
        _ => None,
    };
    if let Some(t) = top {
        return Some(t.to_string());
    }
    if let Some(r) = n.strip_prefix("vision_encoder.") {
        // Pixtral: q / k keep Mistral's interleaved RoPE row order (the vision tower rotates
        // adjacent pairs, see `crate::pixtral`).
        let r = r
            .replace(".feed_forward.w1.", ".feed_forward.gate_proj.")
            .replace(".feed_forward.w2.", ".feed_forward.down_proj.")
            .replace(".feed_forward.w3.", ".feed_forward.up_proj.")
            .replace(".attention.wq.", ".attention.q_proj.")
            .replace(".attention.wk.", ".attention.k_proj.")
            .replace(".attention.wv.", ".attention.v_proj.")
            .replace(".attention.wo.", ".attention.o_proj.");
        return Some(format!("model.vision_tower.{r}"));
    }
    let rest = n.strip_prefix("layers.")?;
    let (l, rest) = rest.split_once('.')?;
    let l: usize = l.parse().ok()?;
    let sub = if let Some(e) = rest.strip_prefix("experts.") {
        let (e, w) = e.split_once('.')?;
        let w = w
            .replace("w1.", "gate_proj.")
            .replace("w2.", "down_proj.")
            .replace("w3.", "up_proj.");
        format!("mlp.experts.{e}.{w}")
    } else if let Some(w) = rest.strip_prefix("shared_experts.") {
        let w = w
            .replace("w1.", "gate_proj.")
            .replace("w2.", "down_proj.")
            .replace("w3.", "up_proj.");
        format!("mlp.shared_experts.{w}")
    } else if let Some(w) = rest.strip_prefix("feed_forward.") {
        let w = w
            .replace("w1.", "gate_proj.")
            .replace("w2.", "down_proj.")
            .replace("w3.", "up_proj.");
        format!("mlp.{w}")
    } else if let Some(w) = rest.strip_prefix("attention.indexer.") {
        format!(
            "self_attn.indexer.{}",
            w.replace("ssmax_scaling.s", "softmax_scaling.scale")
                .replace("ssmax_scaling.b", "softmax_scaling.bias")
        )
    } else if let Some(w) = rest.strip_prefix("attention.") {
        let w = match w.split_once('.') {
            Some(("wq", s)) => format!("q_proj.{s}"),
            Some(("wkv_a_with_mqa", s)) => format!("kv_a_proj_with_mqa.{s}"),
            Some(("kv_a_norm", s)) => format!("kv_a_layernorm.{s}"),
            Some(("wkv_b", s)) => format!("kv_b_proj.{s}"),
            Some(("wo", s)) => format!("o_proj.{s}"),
            Some(("ssmax_scaling", "s")) => "softmax_scaling.scale".into(),
            Some(("ssmax_scaling", "b")) => "softmax_scaling.bias".into(),
            _ => return None,
        };
        format!("self_attn.{w}")
    } else {
        match rest {
            "attention_norm.weight" => "input_layernorm.weight".into(),
            "ffn_norm.weight" => "post_attention_layernorm.weight".into(),
            "gate.weight" => "mlp.gate.weight".into(),
            _ => return None,
        }
    };
    Some(format!("model.language_model.layers.{l}.{sub}"))
}

/// Weight streaming for a checkpoint that does not fit in memory: a page-cache budget of 70% of
/// RAM (the dense weights stay resident, the rest caches recently routed experts).
pub fn default_stream() -> crate::stream::StreamOptions {
    let mut o = crate::stream::StreamOptions::default().with_env();
    if o.cache_budget.is_none() {
        o.cache_budget = crate::platform::total_memory().map(|m| (m as f64 * 0.7) as u64);
    }
    o
}

/// Whether a checkpoint of `bytes` is too big to hold in memory (over 60% of RAM).
pub fn too_big(bytes: u64) -> bool {
    crate::platform::total_memory().is_some_and(|m| bytes as f64 > 0.6 * m as f64)
}

/// Whether `files` is a checkpoint in Mistral's own naming (see [`native_name`]).
pub(crate) fn is_native(files: &ModelFiles) -> bool {
    files.has("tok_embeddings.weight") && files.has("layers.0.attention_norm.weight")
}

struct Mlp<'a> {
    gate: Lin<'a>,
    up: Lin<'a>,
    down: Lin<'a>,
    inter: usize,
}

impl Mlp<'_> {
    fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
    /// `y = down(silu(gate(x)) * up(x))` for `t` rows (`hook`: neuron edits of `target`).
    fn forward(&self, x: &[f32], t: usize, h: usize, y: &mut [f32], hook: Option<(&NeuronHook, Target)>) {
        let mut g = vec![0f32; t * self.inter];
        let mut u = vec![0f32; t * self.inter];
        linear_multi(
            &mut [(&self.gate, self.inter, &mut g[..]), (&self.up, self.inter, &mut u[..])],
            h,
            x,
            t,
        );
        silu_mul(&mut g, &u);
        if let Some((hk, tg)) = hook {
            hk.apply(tg, &mut g, self.inter);
        }
        linear(&self.down, h, self.inter, &g, t, y);
    }
}

/// Routed experts with a softmax router (DeepSeek's group-limited choice when `n_group` > 1)
/// and the shared expert beside them.
struct Moe<'a> {
    router: Vec<f32>, // [n_exp, H]
    experts: Experts<'a>,
    shared: Mlp<'a>,
}

// One per layer, built once.
#[allow(clippy::large_enum_variant)]
enum Ffn<'a> {
    Dense(Mlp<'a>),
    Moe(Moe<'a>),
}

impl Ffn<'_> {
    /// The intermediate size of `target` (the neurons a [`crate::intervene`] edit indexes: the
    /// input of its `down_proj`), None when this layer has no such MLP.
    fn inter(&self, target: Target) -> Option<usize> {
        match (self, target) {
            (Ffn::Dense(m), Target::Mlp) => Some(m.inter),
            (Ffn::Moe(m), Target::Shared) => Some(m.shared.inter),
            (Ffn::Moe(m), Target::Expert(e)) if e < m.experts.n_experts() => Some(m.experts.inter),
            _ => None,
        }
    }

    /// Column `n` of `target`'s down projection (`[h]`).
    fn down_column(&self, target: Target, n: usize, h: usize) -> Vec<f32> {
        let (down, inter) = match (self, target) {
            (Ffn::Dense(m), _) => (&m.down, m.inter),
            (Ffn::Moe(m), Target::Shared | Target::Mlp) => (&m.shared.down, m.shared.inter),
            (Ffn::Moe(m), Target::Expert(e)) => (&m.experts.down[e], m.experts.inter),
        };
        let mut one = vec![0f32; inter];
        one[n] = 1.0;
        let mut y = vec![0f32; h];
        linear(down, h, inter, &one, 1, &mut y);
        y
    }
}

/// The DSA indexer of a "full" layer.
struct Indexer<'a> {
    q_norm: (Vec<f32>, Vec<f32>), // LayerNorm [H]
    wq: Lin<'a>,                  // [n_heads * D, H]
    wk: Lin<'a>,                  // [D, H]
    k_norm: (Vec<f32>, Vec<f32>), // LayerNorm [D]
    proj: Lin<'a>,                // [n_heads, H]
    ssmax: (Vec<f32>, Vec<f32>),  // per-head query scale and bias
}

struct Attn<'a> {
    q: Lin<'a>, // [nh * (dn + dr), H]
    /// The latent + rotary key projection (`[lat + dr, H]`) and the latent's RMSNorm, on layers
    /// that own their cache.
    kv_a: Option<(Lin<'a>, Vec<f32>)>,
    /// Per head: `k_b[h]` `[lat, dn]` maps a non-rotary query onto the latent (the transpose of
    /// `kv_b_proj`'s key rows), `v_b[h]` `[v, lat]` the attended latent onto the value.
    k_b: Vec<Lin<'a>>,
    v_b: Vec<Lin<'a>>,
    o: Lin<'a>,                  // [H, nh * v]
    ssmax: (Vec<f32>, Vec<f32>), // per-head query scale and bias
    indexer: Option<Indexer<'a>>,
    /// Non-rotary and rotary query / key sizes (dr = 0 on NoPE layers).
    dn: usize,
    dr: usize,
    /// The layer whose latent cache this one reads (itself when it owns one).
    src: usize,
}

struct Layer<'a> {
    ln_in: Vec<f32>,
    ln_post: Vec<f32>,
    attn: Attn<'a>,
    ffn: Ffn<'a>,
}

/// The latent cache of one layer: per position the normed latent `[lat]` and the rotated
/// rotary key `[dr]`.
#[derive(Clone, Default)]
pub struct LatentCache {
    c: Vec<f32>,
    r: Vec<f32>,
    lat: usize,
    dr: usize,
    pub len: usize,
}

impl LatentCache {
    fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.c.truncate(len * self.lat);
            self.r.truncate(len * self.dr);
            self.len = len;
        }
    }
}

/// Decoding state: latent caches of the layers that own one, indexer key caches of the full
/// indexer layers, and the position.
#[derive(Clone)]
pub struct Mistral4Session {
    kv: Vec<Option<LatentCache>>,
    /// `[len, idx_dim]` keys per full indexer layer.
    idx: Vec<Option<Vec<f32>>>,
    idx_dim: usize,
    pub pos: usize,
    /// Runtime interventions ([`crate::intervene`]).
    pub iv: crate::intervene::State,
}

impl Mistral4Session {
    /// Drops cached positions at or beyond `pos`.
    pub fn truncate(&mut self, pos: usize) {
        for c in self.kv.iter_mut().flatten() {
            c.truncate(pos);
        }
        for k in self.idx.iter_mut().flatten() {
            k.truncate(pos * self.idx_dim);
        }
        self.pos = self.pos.min(pos);
    }
}

/// The keys each query of a pass attends to: None = every earlier key (causal), or the sorted
/// positions the indexer chose.
type Selection = Vec<Option<Vec<u32>>>;

pub struct Mistral4Model<'a> {
    pub cfg: Mistral4Config,
    pub quant: Quant,
    embed: Table<'a>,
    lm_head: Lin<'a>,
    final_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
    /// Rotary inverse frequencies (YaRN times the learned LeRoPE factors) and the cos / sin
    /// magnitude.
    inv_freq: Vec<f32>,
    rope_mag: f32,
    /// Weight streaming: unit `l` is layer `l`, unit `num_hidden_layers` the LM head.
    stream: Option<crate::stream::StreamHook>,
}

/// LayerNorm with weight and bias over each row of `x` (`n` wide), in place.
fn layer_norm_rows(x: &mut [f32], n: usize, w: &[f32], b: &[f32], eps: f32) {
    for_chunks(x, n, x.len() > 64 * n, |_, r| {
        let mean = r.iter().sum::<f32>() / n as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ((v, &wi), &bi) in r.iter_mut().zip(w).zip(b) {
            *v = (*v - mean) * inv * wi + bi;
        }
    });
}

/// A linear weight as f32 `[n_out, n_in]` (small per-head matrices, expanded per prefill pass).
fn lin_f32(l: &Lin, n_out: usize, n_in: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_out * n_in];
    match &l.w {
        Weight::Bf16(w, _) => bf16_to_f32_slice(&w[..n_out * n_in], &mut out),
        Weight::Q8(m) => crate::quant::dequantize_rows(m, 0, n_out, &mut out),
        Weight::Q4(m) => crate::quant::dequantize_rows_q4(m, 0, n_out, &mut out),
        Weight::Gguf(m) => m.dequantize_rows(0, n_out, &mut out),
    }
    out
}

/// Per-head matmuls in one grouped call: `x` holds `nh` blocks of `t` rows of `n_in`, block `h`
/// goes through `ls[h]` (`[n_out, n_in]`); returns `nh` blocks of `t` rows of `n_out`.
pub(crate) fn per_head(ls: &[Lin], x: &[f32], t: usize, n_in: usize, n_out: usize) -> Vec<f32> {
    let nh = ls.len();
    let tp = t.next_multiple_of(TN);
    let mut xp = vec![0f32; nh * tp * n_in];
    for h in 0..nh {
        xp[h * tp * n_in..(h * tp + t) * n_in].copy_from_slice(&x[h * t * n_in..(h + 1) * t * n_in]);
    }
    let segs: Vec<(usize, usize)> = (0..nh).map(|h| (h * tp, tp)).collect();
    let refs: Vec<&Lin> = ls.iter().collect();
    let mut yp = vec![0f32; nh * tp * n_out];
    linear_grouped(&refs, &segs, n_out, n_in, &xp, &mut yp);
    let mut y = vec![0f32; nh * t * n_out];
    for h in 0..nh {
        y[h * t * n_out..(h + 1) * t * n_out].copy_from_slice(&yp[h * tp * n_out..(h * tp + t) * n_out]);
    }
    y
}

/// The `k` largest of `scores` (ties to the lower position), sorted by position.
fn top_positions(scores: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
    if k < idx.len() {
        idx.select_nth_unstable_by(k, |&a, &b| scores[b as usize].total_cmp(&scores[a as usize]).then(a.cmp(&b)));
        idx.truncate(k);
    }
    idx.sort_unstable();
    idx
}

/// Softmax-routes one token (DeepSeek-V3 / Mistral 4 `TopkRouter`): softmax over all experts;
/// with groups, only the `topk_group` groups with the best top-2 sums compete; the top `k`
/// weights renormalized, times `scale`.
fn route(logits: &[f32], k: usize, n_group: usize, topk_group: usize, scale: f32) -> Vec<(usize, f32)> {
    let mut r = if n_group <= 1 {
        route_top_k(logits, k)
    } else {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        let p: Vec<f32> = e.iter().map(|v| v / s).collect();
        let per = p.len() / n_group;
        let mut gs: Vec<(usize, f32)> = p
            .chunks(per)
            .enumerate()
            .map(|(g, c)| {
                let mut v = c.to_vec();
                v.sort_by(|a, b| b.total_cmp(a));
                (g, v.iter().take(2).sum())
            })
            .collect();
        gs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let keep: Vec<usize> = gs.iter().take(topk_group).map(|g| g.0).collect();
        let mut idx: Vec<usize> = (0..p.len()).filter(|i| keep.contains(&(i / per))).collect();
        idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        idx.truncate(k);
        let s: f32 = idx.iter().map(|&i| p[i]).sum::<f32>() + 1e-20;
        idx.into_iter().map(|i| (i, p[i] / s)).collect()
    };
    if scale != 1.0 {
        r.iter_mut().for_each(|w| w.1 *= scale);
    }
    r
}

/// Loads the model's tensors: shape checks, both name prefixes, GGUF's split `kv_b`.
struct Loader<'a, 'c> {
    files: &'a ModelFiles,
    cfg: &'c Mistral4Config,
    quant: Quant,
    prefix: &'static str,
}

impl<'a> Loader<'a, '_> {
    fn has(&self, n: &str) -> bool {
        self.files.has(n) || self.files.has(&format!("{n}.q8")) || self.files.has(&format!("{n}.q4"))
    }
    fn getf(&self, full: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.files.info(full).with_context(|| format!("missing tensor {full}"))?;
        if info.shape != shape {
            bail!("tensor {full} has shape {:?}, expected {:?}", info.shape, shape);
        }
        self.files.f32(full)
    }
    fn lin(&self, full: &str, n_out: usize, n_in: usize) -> Result<Lin<'a>> {
        load_lin(self.files, full, n_out, n_in, self.quant)
    }
    fn mlp(&self, p: &str, inter: usize) -> Result<Mlp<'a>> {
        let h = self.cfg.hidden_size;
        Ok(Mlp {
            gate: self.lin(&format!("{p}gate_proj.weight"), inter, h)?,
            up: self.lin(&format!("{p}up_proj.weight"), inter, h)?,
            down: self.lin(&format!("{p}down_proj.weight"), h, inter)?,
            inter,
        })
    }

    /// `kv_b` as per-head `k_b` (`[lat, dn]`) and `v_b` (`[v, lat]`): GGUF stores them so; a
    /// `kv_b_proj` (`[nh * (dn + v), lat]`) has its key rows transposed into a bf16 copy and its
    /// value rows borrowed.
    fn kv_b(&self, ap: &str, dn: usize) -> Result<(Vec<Lin<'a>>, Vec<Lin<'a>>)> {
        let c = self.cfg;
        let (nh, lat, v) = (c.num_attention_heads, c.kv_lora_rank, c.v_head_dim);
        let (kn, vn) = (format!("{ap}k_b"), format!("{ap}v_b"));
        if self.files.has(&kn) {
            let mut out = (Vec::with_capacity(nh), Vec::with_capacity(nh));
            for (name, n_out, n_in, dst) in [(&kn, lat, dn, &mut out.0), (&vn, v, lat, &mut out.1)] {
                let info = self.files.info(name).with_context(|| format!("missing tensor {name}"))?;
                if info.shape != [nh, n_out, n_in] {
                    bail!("tensor {name} has shape {:?}, expected [{nh}, {n_out}, {n_in}]", info.shape);
                }
                self.files.note_use(name, true, false);
                match self.files.ggml(name) {
                    Some(g) => {
                        let (kind, bytes) = g?;
                        for b in bytes.chunks(bytes.len() / nh) {
                            dst.push(Lin::from_gguf(kind, b, n_out, n_in)?);
                        }
                    }
                    None => {
                        let w = self.files.bf16(name)?;
                        dst.extend(w.chunks(n_out * n_in).map(Lin::bf16));
                    }
                }
            }
            return Ok(out);
        }
        let name = format!("{ap}kv_b_proj.weight");
        let info = self.files.info(&name).with_context(|| format!("missing tensor {name}"))?;
        if info.shape != [nh * (dn + v), lat] {
            bail!("tensor {name} has shape {:?}, expected [{}, {lat}]", info.shape, nh * (dn + v));
        }
        let w: &'a [u16] = if info.dtype == "F8_E4M3" {
            bail!("{name}: an FP8 kv_b_proj is not supported (the official checkpoint keeps it in bf16)");
        } else {
            self.files.bf16(&name)?
        };
        let (mut kb, mut vb) = (Vec::with_capacity(nh), Vec::with_capacity(nh));
        for h in 0..nh {
            let rows = &w[h * (dn + v) * lat..(h + 1) * (dn + v) * lat];
            let mut t = crate::quant::PageVec::<u16>::zeroed(lat * dn);
            for c in 0..dn {
                for r in 0..lat {
                    t[r * dn + c] = rows[c * lat + r];
                }
            }
            kb.push(Lin::bf16_owned(t));
            vb.push(Lin::bf16(&rows[dn * lat..]));
        }
        Ok((kb, vb))
    }

    fn attn(&self, l: usize) -> Result<Attn<'a>> {
        let c = self.cfg;
        let (h, nh, lat) = (c.hidden_size, c.num_attention_heads, c.kv_lora_rank);
        let dr = if c.has_rope(l) { c.qk_rope_head_dim } else { 0 };
        let dn = c.qk_head_dim() - dr;
        let ap = format!("{}layers.{l}.self_attn.", self.prefix);
        let src = c.kv_source(l);
        let kv_a = if src == l {
            Some((
                self.lin(&format!("{ap}kv_a_proj_with_mqa.weight"), lat + dr, h)?,
                self.getf(&format!("{ap}kv_a_layernorm.weight"), &[lat])?,
            ))
        } else {
            None
        };
        let (k_b, v_b) = self.kv_b(&ap, dn)?;
        let indexer = if c.full_indexer(l) {
            let ip = format!("{ap}indexer.");
            let (ni, d) = (c.index_n_heads, c.index_head_dim);
            Some(Indexer {
                q_norm: (
                    self.getf(&format!("{ip}q_input_norm.weight"), &[h])?,
                    self.getf(&format!("{ip}q_input_norm.bias"), &[h])?,
                ),
                wq: self.lin(&format!("{ip}wq_b.weight"), ni * d, h)?,
                wk: self.lin(&format!("{ip}wk.weight"), d, h)?,
                k_norm: (
                    self.getf(&format!("{ip}k_norm.weight"), &[d])?,
                    self.getf(&format!("{ip}k_norm.bias"), &[d])?,
                ),
                // Kept in float precision by transformers (`_keep_in_fp32_modules`): never quantized.
                proj: load_lin(self.files, &format!("{ip}weights_proj.weight"), ni, h, Quant::Bf16)?,
                ssmax: (
                    self.getf(&format!("{ip}softmax_scaling.scale"), &[ni])?,
                    self.getf(&format!("{ip}softmax_scaling.bias"), &[ni])?,
                ),
            })
        } else {
            None
        };
        Ok(Attn {
            q: self.lin(&format!("{ap}q_proj.weight"), nh * (dn + dr), h)?,
            kv_a,
            k_b,
            v_b,
            o: self.lin(&format!("{ap}o_proj.weight"), h, nh * c.v_head_dim)?,
            ssmax: (
                self.getf(&format!("{ap}softmax_scaling.scale"), &[nh])?,
                self.getf(&format!("{ap}softmax_scaling.bias"), &[nh])?,
            ),
            indexer,
            dn,
            dr,
            src,
        })
    }

    fn ffn(&self, l: usize) -> Result<Ffn<'a>> {
        let c = self.cfg;
        let h = c.hidden_size;
        let mp = format!("{}layers.{l}.mlp.", self.prefix);
        if !c.is_moe(l) {
            return Ok(Ffn::Dense(self.mlp(&mp, c.intermediate_size)?));
        }
        let (ne, mi) = (c.n_routed_experts, c.moe_intermediate_size);
        Ok(Ffn::Moe(Moe {
            router: self.getf(&format!("{mp}gate.weight"), &[ne, h])?,
            experts: Experts::load(
                self.files,
                &format!("{mp}experts."),
                ne,
                c.num_experts_per_tok,
                mi,
                h,
                self.quant,
                Act::Silu,
            )?,
            shared: self.mlp(&format!("{mp}shared_experts."), mi * c.n_shared_experts)?,
        }))
    }
}

impl<'a> Mistral4Model<'a> {
    pub fn load(cfg: Mistral4Config, files: &'a ModelFiles, quant: Quant) -> Result<Self> {
        let quant = Quant::stored(files).unwrap_or(quant);
        let prefix = if files.has("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else if files.has("model.embed_tokens.weight") {
            "model."
        } else if is_native(files) {
            bail!("a Mistral-native checkpoint must be opened with its names mapped (ModelFiles::open does it)");
        } else {
            bail!("cannot find embed_tokens in checkpoint");
        };
        let ld = Loader {
            files,
            cfg: &cfg,
            quant,
            prefix,
        };
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let tied = cfg.tie_word_embeddings || !ld.has("lm_head.weight");
        let embed_name = format!("{prefix}embed_tokens.weight");
        let embed = Table::load(files, &embed_name, v, h, if tied { quant } else { table_quant(quant) })?;
        let lm_head = if tied {
            ld.lin(&embed_name, v, h)?
        } else {
            ld.lin("lm_head.weight", v, h)?
        };
        let final_norm = ld.getf(&format!("{prefix}norm.weight"), &[h])?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}layers.{l}.");
            layers.push(Layer {
                ln_in: ld.getf(&format!("{lp}input_layernorm.weight"), &[h])?,
                ln_post: ld.getf(&format!("{lp}post_attention_layernorm.weight"), &[h])?,
                attn: ld.attn(l)?,
                ffn: ld.ffn(l)?,
            });
        }
        let (mut inv_freq, rope_mag) = cfg.rope();
        let half = inv_freq.len();
        // LeRoPE: transformers scales the frequencies by exp(alpha); llama.cpp's converter stores
        // exp(-alpha) as ggml's frequency divisors.
        let alpha = format!("{prefix}rotary_emb.alpha");
        if files.has(&alpha) {
            for (f, a) in inv_freq.iter_mut().zip(ld.getf(&alpha, &[half])?) {
                *f *= a.exp();
            }
        } else if files.has("rope_freqs") {
            for (f, d) in inv_freq.iter_mut().zip(ld.getf("rope_freqs", &[half])?) {
                *f /= d;
            }
        }
        let m = Self {
            cfg,
            quant,
            embed,
            lm_head,
            final_norm,
            layers,
            inv_freq,
            rope_mag,
            stream: None,
        };
        // Routed experts that cannot all stay in memory are read through the page cache as they
        // are routed: no accelerator mapping may pin them (the GPU MoE path wraps every expert).
        let ram = crate::platform::total_memory().unwrap_or(u64::MAX);
        if files.total_bytes() as f64 > 0.6 * ram as f64 {
            for l in &m.layers {
                if let Ffn::Moe(moe) = &l.ffn {
                    for e in moe.experts.lins().into_iter().flatten() {
                        e.set_transient_device(true);
                    }
                }
            }
        }
        Ok(m)
    }

    pub fn new_session(&self) -> Mistral4Session {
        let c = &self.cfg;
        Mistral4Session {
            kv: self
                .layers
                .iter()
                .enumerate()
                .map(|(l, layer)| {
                    (layer.attn.src == l).then(|| LatentCache {
                        lat: c.kv_lora_rank,
                        dr: layer.attn.dr,
                        ..Default::default()
                    })
                })
                .collect(),
            idx: self.layers.iter().map(|l| l.attn.indexer.as_ref().map(|_| Vec::new())).collect(),
            idx_dim: c.index_head_dim,
            pos: 0,
            iv: Default::default(),
        }
    }

    /// Weight bytes read per decoded token (the routed experts at top-k).
    pub fn streamed_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes();
        for l in &self.layers {
            n += self.attn_lins(&l.attn).iter().map(|w| w.bytes()).sum::<usize>();
            n += match &l.ffn {
                Ffn::Dense(m) => m.bytes(),
                Ffn::Moe(m) => m.shared.bytes() + m.experts.bytes_per_token(),
            };
        }
        n
    }

    fn attn_lins<'s>(&'s self, a: &'s Attn<'a>) -> Vec<&'s Lin<'a>> {
        let mut v = vec![&a.q, &a.o];
        v.extend(a.kv_a.iter().map(|k| &k.0));
        v.extend(a.k_b.iter().chain(&a.v_b));
        if let Some(ix) = &a.indexer {
            v.extend([&ix.wq, &ix.wk, &ix.proj]);
        }
        v
    }

    fn dense_lins<'s>(&'s self, l: &'s Layer<'a>) -> Vec<&'s Lin<'a>> {
        let mut v = self.attn_lins(&l.attn);
        match &l.ffn {
            Ffn::Dense(m) => v.extend([&m.gate, &m.up, &m.down]),
            Ffn::Moe(m) => v.extend([&m.shared.gate, &m.shared.up, &m.shared.down]),
        }
        v
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
                parts: match &layer.ffn {
                    Ffn::Moe(m) => m.experts.ranges(),
                    Ffn::Dense(_) => Vec::new(),
                },
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
            if let Ffn::Moe(m) = &layer.ffn {
                for e in m.experts.lins().into_iter().flatten() {
                    e.set_transient_device(true);
                }
            }
        }
        self.lm_head
            .set_transient_device(streamed.get(self.layers.len()).copied().unwrap_or(false));
    }

    /// Brackets every layer and the LM head with `hook`'s units (None stops streaming).
    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// LM-head logits `[t, vocab]` (not soft-capped) for final-normed hidden states `[t, hidden]`.
    pub fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * c.vocab_size];
        let _unit = crate::stream::enter(&self.stream, self.layers.len());
        linear(&self.lm_head, c.vocab_size, c.hidden_size, hidden, t, &mut logits);
        logits
    }

    /// Soft-capped logits `[t, vocab]` for final-normed hidden states `[t, hidden]`.
    pub fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let mut logits = self.head_raw(hidden, t);
        let _timer = prof::start(Cat::Head);
        if let Some(cap) = c.final_logit_softcapping.filter(|&c| c > 0.0) {
            let cap = cap as f32;
            for_chunks(&mut logits, 8192, true, |_, ch| {
                for v in ch.iter_mut() {
                    *v = (*v / cap).tanh() * cap;
                }
            });
        }
        logits
    }

    /// Runs `tokens` and returns the logits after the last one.
    pub fn forward(&self, sess: &mut Mistral4Session, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let hs = self.forward_hidden_mm(sess, tokens, &[]);
        self.logits(&hs[hs.len() - h..], 1)
    }

    /// Runs `tokens` from `sess.pos` (positions in `mm` take the given embeddings instead of the
    /// token's), appending to the caches; returns the final-normed hidden states `[t, hidden]`.
    pub fn forward_hidden_mm(&self, sess: &mut Mistral4Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        self.forward_capture(sess, tokens, mm, &[], &mut Vec::new())
    }

    /// [`Self::forward_hidden_mm`] that also writes to `feats` (`[t, layers.len() * hidden]`) the
    /// residual stream entering each of `layers`, `num_hidden_layers` standing for the last
    /// layer's output before the final norm (the target features of a DSpark draft).
    pub fn forward_capture(
        &self,
        sess: &mut Mistral4Session,
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
        let eps = c.rms_norm_eps as f32;
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
        let positions: Vec<usize> = (start..start + t).collect();
        let (mut cos, mut sin) = rope_tables(&positions, &self.inv_freq);
        if self.rope_mag != 1.0 {
            cos.iter_mut().chain(sin.iter_mut()).for_each(|v| *v *= self.rope_mag);
        }
        // SSMax: the log-position basis of every query's scale.
        let lp: Vec<f32> = positions.iter().map(|&p| ((p + 1) as f32).ln()).collect();
        drop(timer);
        let iv = sess.iv.active().then(|| sess.iv.clone());
        if let Some(iv) = &iv {
            iv.point(0, &mut x, h, start);
        }
        let mut sel: Selection = Vec::new();
        let mut hn = vec![0f32; t * h];
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
            rmsnorm_f32w_rows(&x, h, &layer.ln_in, eps, &mut hn);
            let mut o = vec![0f32; t * h];
            self.attn_forward(l, &layer.attn, &hn, t, start, sess, &mut sel, (&cos, &sin), &lp, &mut o);
            add_inplace(&mut x, &o);
            if let Some(iv) = &iv {
                iv.point(2 * l + 1, &mut x, h, start);
            }
            rmsnorm_f32w_rows(&x, h, &layer.ln_post, eps, &mut hn);
            o.fill(0.0);
            let need = |ids: &[usize]| {
                if let Some(s) = &self.stream {
                    s.need_parts(l, ids)
                }
            };
            let need: NeedExperts = self.stream.as_ref().map(|_| &need as &dyn Fn(&[usize]));
            let hook = iv.as_ref().and_then(|s| s.hook(l, start));
            match &layer.ffn {
                Ffn::Dense(m) => m.forward(&hn, t, h, &mut o, hook.as_ref().map(|k| (k, Target::Mlp))),
                Ffn::Moe(m) => self.moe_forward(m, &hn, t, &mut o, need, hook.as_ref()),
            }
            add_inplace(&mut x, &o);
            if let Some(iv) = &iv {
                iv.point(2 * l + 2, &mut x, h, start);
            }
            // Streaming: guess the next layer's experts from its router on this residual, so
            // they are read while its attention runs (decode-sized passes; prefill needs most).
            if let (Some(s), Some(next)) = (&self.stream, self.layers.get(l + 1))
                && let Ffn::Moe(m) = &next.ffn
                && t <= 4
                && s.options().predict_experts
            {
                let ne = m.experts.n_experts();
                let mut guess = Vec::new();
                for ti in 0..t {
                    let mut xn = vec![0f32; h];
                    rmsnorm_f32w(&x[ti * h..(ti + 1) * h], &next.ln_post, eps, &mut xn);
                    let lg: Vec<f32> = m.router.chunks(h).take(ne).map(|w| dot(w, &xn)).collect();
                    guess.extend(route(&lg, m.experts.top_k, c.n_group, c.topk_group, 1.0).into_iter().map(|r| r.0));
                }
                guess.sort_unstable();
                guess.dedup();
                s.prefetch_parts(l + 1, &guess);
            }
        }
        capture(self.layers.len(), &x);
        rmsnorm_f32w_rows_inplace(&mut x, h, &self.final_norm, eps);
        sess.pos += t;
        x
    }

    fn moe_forward(&self, m: &Moe, x: &[f32], t: usize, y: &mut [f32], need: NeedExperts, hook: Option<&NeuronHook>) {
        let c = &self.cfg;
        let h = c.hidden_size;
        let shared = crate::moe::Shared {
            gate: &m.shared.gate,
            up: &m.shared.up,
            down: &m.shared.down,
            inter: m.shared.inter,
            act: Act::Silu,
            hook,
            scale: None,
        };
        let ne = m.experts.n_experts();
        let mut logits = vec![0f32; t * ne];
        for_chunks(&mut logits, ne, t > 1, |ti, lg| {
            for (v, w) in lg.iter_mut().zip(m.router.chunks(h)) {
                *v = dot(w, &x[ti * h..(ti + 1) * h]);
            }
        });
        let routes: Vec<_> = logits
            .chunks(ne)
            .map(|lg| route(lg, m.experts.top_k, c.n_group, c.topk_group, c.routed_scaling_factor as f32))
            .collect();
        m.experts.forward(x, t, h, &routes, y, need, hook, Some(&shared));
    }

    /// One layer's attention over `hn` (`[t, H]`, the normed residual) into `out` (`[t, H]`).
    /// `sel` holds the previous full indexer layer's choice (replaced on full layers).
    #[allow(clippy::too_many_arguments)]
    fn attn_forward(
        &self,
        l: usize,
        a: &Attn,
        hn: &[f32],
        t: usize,
        start: usize,
        sess: &mut Mistral4Session,
        sel: &mut Selection,
        rope: (&[f32], &[f32]),
        lp: &[f32],
        out: &mut [f32],
    ) {
        let c = &self.cfg;
        let (h, nh, lat, vd) = (c.hidden_size, c.num_attention_heads, c.kv_lora_rank, c.v_head_dim);
        let (dn, dr) = (a.dn, a.dr);
        let qk = dn + dr;
        let half = cos_half(rope.0, t);
        let (cos, sin) = rope;
        let par = t > 1;

        // Projections of the normed residual: queries, the latent (own-cache layers), and the
        // indexer's key and head weights (full indexer layers).
        let mut q = vec![0f32; t * nh * qk];
        let mut kv = vec![0f32; if a.kv_a.is_some() { t * (lat + dr) } else { 0 }];
        let (ni, d) = (c.index_n_heads, c.index_head_dim);
        let mut ik = vec![0f32; if a.indexer.is_some() { t * d } else { 0 }];
        let mut iw = vec![0f32; if a.indexer.is_some() { t * ni } else { 0 }];
        {
            let mut jobs: Vec<(&Lin, usize, &mut [f32])> = vec![(&a.q, nh * qk, &mut q[..])];
            if let Some((ka, _)) = &a.kv_a {
                jobs.push((ka, lat + dr, &mut kv[..]));
            }
            if let Some(ix) = &a.indexer {
                jobs.push((&ix.wk, d, &mut ik[..]));
                jobs.push((&ix.proj, ni, &mut iw[..]));
            }
            linear_multi(&mut jobs, h, hn, t);
        }

        // The indexer: its keys join its cache, and each query's best keys are chosen.
        if let Some(ix) = &a.indexer {
            let eps = c.indexer_norm_eps as f32;
            let mut qn = hn.to_vec();
            layer_norm_rows(&mut qn, h, &ix.q_norm.0, &ix.q_norm.1, eps);
            let mut iq = vec![0f32; t * ni * d];
            linear(&ix.wq, ni * d, h, &qn, t, &mut iq);
            let rot = if dr > 0 { c.qk_rope_head_dim } else { 0 };
            for_chunks(&mut iq, d, par, |i, qh| {
                let (ti, hh) = (i / ni, i % ni);
                let f = ix.ssmax.0[hh] * lp[ti] + ix.ssmax.1[hh];
                scale_inplace(qh, f);
                if rot > 0 {
                    apply_rope_interleaved(&mut qh[..rot], &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                }
            });
            layer_norm_rows(&mut ik, d, &ix.k_norm.0, &ix.k_norm.1, eps);
            if rot > 0 {
                for_chunks(&mut ik, d, par, |ti, kh| {
                    apply_rope_interleaved(&mut kh[..rot], &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                });
            }
            let wscale = (ni as f32).powf(-0.5);
            iw.iter_mut().for_each(|w| *w *= wscale);
            let keys = sess.idx[l].as_mut().expect("indexer cache");
            keys.extend_from_slice(&ik);
            *sel = self.select(&iq, &iw, keys, t, start);
        }

        // The latent cache: normed latent and rotated rotary key per token.
        if let Some((_, norm)) = &a.kv_a {
            let eps = c.rms_norm_eps as f32;
            let cache = sess.kv[l].as_mut().expect("latent cache");
            for ti in 0..t {
                let row = &mut kv[ti * (lat + dr)..(ti + 1) * (lat + dr)];
                let (cl, kr) = row.split_at_mut(lat);
                rmsnorm_f32w_inplace(cl, norm, eps);
                if dr > 0 {
                    apply_rope_interleaved(kr, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                }
                cache.c.extend_from_slice(cl);
                cache.r.extend_from_slice(kr);
            }
            cache.len += t;
        }

        // Queries: rotary part rotated, every head scaled by its SSMax factor.
        for_chunks(&mut q, qk, par, |i, qh| {
            let (ti, hh) = (i / nh, i % nh);
            if dr > 0 {
                apply_rope_interleaved(&mut qh[dn..], &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
            }
            scale_inplace(qh, a.ssmax.0[hh] * lp[ti] + a.ssmax.1[hh]);
        });

        let cache = sess.kv[a.src].as_ref().expect("kv source cache");
        debug_assert_eq!(cache.len, start + t);
        let mut att = vec![0f32; t * nh * vd];
        if t >= EXPAND_MIN {
            self.attend_expanded(a, &q, cache, t, start, sel, &mut att);
        } else {
            self.attend_latent(a, &q, cache, t, start, sel, &mut att);
        }
        linear(&a.o, h, nh * vd, &att, t, out);
    }

    /// Per query of the pass, the keys its indexer scores rank in the top `index_topk` (None
    /// when it has no more keys than that): `iq` `[t, n_heads, D]` (scaled, rotated), `iw` `[t,
    /// n_heads]` (head weights, scaled), `keys` `[start + t, D]`.
    fn select(&self, iq: &[f32], iw: &[f32], keys: &[f32], t: usize, start: usize) -> Selection {
        let c = &self.cfg;
        let (ni, d, topk) = (c.index_n_heads, c.index_head_dim, c.index_topk);
        let mut sel: Selection = vec![None; t];
        let rows: Vec<usize> = (0..t).filter(|&i| start + i + 1 > topk).collect();
        if rows.is_empty() {
            return sel;
        }
        let _timer = prof::start(Cat::Attention);
        let n = start + t;
        let sm = (d as f32).powf(-0.5);
        // Rows per GEMM keep the score block to ~32 MB.
        let rb = ((1usize << 23) / (ni * n)).clamp(1, 64);
        for blk in rows.chunks(rb) {
            let mut qb = vec![0f32; blk.len() * ni * d];
            for (r, &i) in blk.iter().enumerate() {
                qb[r * ni * d..(r + 1) * ni * d].copy_from_slice(&iq[i * ni * d..(i + 1) * ni * d]);
            }
            let sc = crate::sd::ops::matmul_f32(&keys[..n * d], n, d, &qb, blk.len() * ni);
            let picks: Vec<Vec<u32>> = {
                let mut out = vec![Vec::new(); blk.len()];
                let op = SendPtrV(out.as_mut_ptr());
                crate::pool::run(blk.len(), &|r| {
                    let i = blk[r];
                    let p = start + i;
                    let mut s = vec![0f32; p + 1];
                    for hh in 0..ni {
                        let w = iw[i * ni + hh];
                        let row = &sc[(r * ni + hh) * n..(r * ni + hh) * n + p + 1];
                        for (o, &v) in s.iter_mut().zip(row) {
                            *o += w * (v * sm).max(0.0);
                        }
                    }
                    // SAFETY: each task writes its own slot.
                    unsafe { *op.get().add(r) = top_positions(&s, topk) };
                });
                out
            };
            for (&i, pk) in blk.iter().zip(picks) {
                sel[i] = Some(pk);
            }
        }
        sel
    }

    /// Attention scored against the latent cache (`kv_b` absorbed): each head's non-rotary
    /// query is mapped onto the latent (`k_b`), scored with the rotary part against every chosen
    /// position, the attended latent mapped onto the value (`v_b`). `out` `[t, nh * v]`.
    #[allow(clippy::too_many_arguments)]
    fn attend_latent(&self, a: &Attn, q: &[f32], cache: &LatentCache, t: usize, start: usize, sel: &Selection, out: &mut [f32]) {
        let c = &self.cfg;
        let (nh, lat, vd) = (c.num_attention_heads, c.kv_lora_rank, c.v_head_dim);
        let (dn, dr) = (a.dn, a.dr);
        let qk = dn + dr;
        // Non-rotary queries per head: [nh][t][dn] -> latent queries [nh][t][lat].
        let mut qn = vec![0f32; nh * t * dn];
        for ti in 0..t {
            for hh in 0..nh {
                qn[(hh * t + ti) * dn..(hh * t + ti + 1) * dn].copy_from_slice(&q[(ti * nh + hh) * qk..(ti * nh + hh) * qk + dn]);
            }
        }
        let ql = per_head(&a.k_b, &qn, t, dn, lat);
        let _timer = prof::start(Cat::Attention);
        let scale = c.attn_scale();
        // Tasks of a few heads of one token, so each cached latent is read once per group.
        let g = if t * nh >= 4 * crate::pool::threads() {
            nh.min(16)
        } else {
            4.min(nh)
        };
        let groups = nh.div_ceil(g);
        let mut ol = vec![0f32; nh * t * lat];
        let op = SendPtr(ol.as_mut_ptr());
        crate::pool::run(t * groups, &|task| {
            let (ti, h0) = (task / groups, (task % groups) * g);
            let h1 = (h0 + g).min(nh);
            let p = start + ti;
            let keys: Vec<u32> = match sel.get(ti).and_then(|s| s.as_ref()) {
                Some(s) => s.clone(),
                None => (0..=p as u32).collect(),
            };
            let n = keys.len();
            let mut sc = vec![0f32; (h1 - h0) * n];
            for (jj, &j) in keys.iter().enumerate() {
                let j = j as usize;
                let cj = &cache.c[j * lat..(j + 1) * lat];
                let rj = &cache.r[j * dr..(j + 1) * dr];
                for hh in h0..h1 {
                    let qlh = &ql[(hh * t + ti) * lat..(hh * t + ti + 1) * lat];
                    let mut s = dot(qlh, cj);
                    if dr > 0 {
                        s += dot(&q[(ti * nh + hh) * qk + dn..(ti * nh + hh + 1) * qk], rj);
                    }
                    sc[(hh - h0) * n + jj] = s * scale;
                }
            }
            for hh in h0..h1 {
                softmax_inplace(&mut sc[(hh - h0) * n..(hh - h0 + 1) * n]);
            }
            for hh in h0..h1 {
                // SAFETY: each (token, head) writes its own latent row.
                let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((hh * t + ti) * lat), lat) };
                for (jj, &j) in keys.iter().enumerate() {
                    axpy(o, sc[(hh - h0) * n + jj], &cache.c[j as usize * lat..(j as usize + 1) * lat]);
                }
            }
        });
        drop(_timer);
        let v = per_head(&a.v_b, &ol, t, lat, vd);
        for ti in 0..t {
            for hh in 0..nh {
                out[(ti * nh + hh) * vd..(ti * nh + hh + 1) * vd].copy_from_slice(&v[(hh * t + ti) * vd..(hh * t + ti + 1) * vd]);
            }
        }
    }

    /// Attention with the keys and values expanded from the latent cache (prefill): `kv_b` over
    /// every cached latent, then per head scores, the indexer's (or the causal) mask, softmax and
    /// values as GEMMs; a chunk that starts the cache with dense attention runs on the
    /// accelerator when there is one. `out` `[t, nh * v]`.
    #[allow(clippy::too_many_arguments)]
    fn attend_expanded(&self, a: &Attn, q: &[f32], cache: &LatentCache, t: usize, start: usize, sel: &Selection, out: &mut [f32]) {
        let c = &self.cfg;
        let (nh, lat, vd) = (c.num_attention_heads, c.kv_lora_rank, c.v_head_dim);
        let (dn, dr) = (a.dn, a.dr);
        let qk = dn + dr;
        let n = start + t;
        // kv_b as one f32 matrix: per head the key rows (transposed k_b) then the value rows.
        let rows = dn + vd;
        let mut w = vec![0f32; nh * rows * lat];
        for_chunks(&mut w, rows * lat, true, |hh, wh| {
            let kb = lin_f32(&a.k_b[hh], lat, dn);
            for r in 0..lat {
                for cc in 0..dn {
                    wh[cc * lat + r] = kb[r * dn + cc];
                }
            }
            wh[dn * lat..].copy_from_slice(&lin_f32(&a.v_b[hh], vd, lat));
        });
        let kvx = crate::sd::ops::matmul_f32(&w, nh * rows, lat, &cache.c[..n * lat], n);
        drop(w);
        let _timer = prof::start(Cat::Attention);
        let mut k = vec![0f32; n * nh * qk];
        let mut v = vec![0f32; n * nh * vd];
        for_chunks(&mut k, nh * qk, true, |j, kr| {
            for hh in 0..nh {
                let src = &kvx[(j * nh + hh) * rows..(j * nh + hh) * rows + dn];
                kr[hh * qk..hh * qk + dn].copy_from_slice(src);
                kr[hh * qk + dn..(hh + 1) * qk].copy_from_slice(&cache.r[j * dr..(j + 1) * dr]);
            }
        });
        for_chunks(&mut v, nh * vd, true, |j, vr| {
            for hh in 0..nh {
                vr[hh * vd..(hh + 1) * vd].copy_from_slice(&kvx[(j * nh + hh) * rows + dn..(j * nh + hh + 1) * rows]);
            }
        });
        drop(kvx);
        let scale = c.attn_scale();
        let dense = sel.iter().all(|s| s.is_none());
        if dense
            && start == 0
            && qk == vd
            && let Some(be) = crate::gpu::active()
            && 4.0 * (t * n * nh * qk) as f64 >= 1e8
        {
            // The device scales by 1/sqrt(head dim).
            let s = scale * (qk as f32).sqrt();
            let qs: Vec<f32> = if s == 1.0 { q.to_vec() } else { q.iter().map(|x| x * s).collect() };
            if be.attention(&qs, &k, &v, t, n, nh, qk, true, out) {
                return;
            }
        }
        let dp = qk.next_multiple_of(32);
        let np = n.next_multiple_of(32);
        let tb = ((1usize << 23) / n.max(1)).clamp(8, 256);
        for hh in 0..nh {
            let mut kh = vec![0f32; n * dp];
            for_chunks(&mut kh, dp, n > 64, |j, row| {
                row[..qk].copy_from_slice(&k[(j * nh + hh) * qk..(j * nh + hh + 1) * qk])
            });
            let mut vt = vec![0f32; vd * np];
            for_chunks(&mut vt, np, true, |cc, row| {
                for (j, x) in row[..n].iter_mut().enumerate() {
                    *x = v[(j * nh + hh) * vd + cc];
                }
            });
            let mut b0 = 0;
            while b0 < t {
                let b1 = (b0 + tb).min(t);
                let r = b1 - b0;
                let mut qh = vec![0f32; r * dp];
                for_chunks(&mut qh, dp, r > 64, |i, row| {
                    row[..qk].copy_from_slice(&q[((b0 + i) * nh + hh) * qk..((b0 + i) * nh + hh + 1) * qk]);
                });
                let kn = start + b1;
                let sc = crate::sd::ops::matmul_f32(&kh[..kn * dp], kn, dp, &qh, r);
                let mut p = vec![0f32; r * np];
                for_chunks(&mut p, np, true, |i, row| {
                    let ti = b0 + i;
                    let s = &sc[i * kn..(i + 1) * kn];
                    match &sel[ti] {
                        None => {
                            let hi = start + ti + 1;
                            for (o, &x) in row[..hi].iter_mut().zip(s) {
                                *o = x * scale;
                            }
                            softmax_inplace(&mut row[..hi]);
                        }
                        Some(ks) => {
                            let mut tmp: Vec<f32> = ks.iter().map(|&j| s[j as usize] * scale).collect();
                            softmax_inplace(&mut tmp);
                            for (&j, &pv) in ks.iter().zip(&tmp) {
                                row[j as usize] = pv;
                            }
                        }
                    }
                });
                let o = crate::sd::ops::matmul_f32(&vt, vd, np, &p, r);
                for i in 0..r {
                    out[((b0 + i) * nh + hh) * vd..((b0 + i) * nh + hh + 1) * vd].copy_from_slice(&o[i * vd..(i + 1) * vd]);
                }
                b0 = b1;
            }
        }
    }
}

/// The rotary table width (`dim / 2` entries per token).
fn cos_half(cos: &[f32], t: usize) -> usize {
    cos.len() / t.max(1)
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

#[derive(Clone, Copy)]
struct SendPtrV(*mut Vec<u32>);
unsafe impl Send for SendPtrV {}
unsafe impl Sync for SendPtrV {}
impl SendPtrV {
    #[inline(always)]
    fn get(self) -> *mut Vec<u32> {
        self.0
    }
}

impl crate::generate::TextLm for Mistral4Model<'_> {
    type Sess = Mistral4Session;
    fn new_session(&self) -> Mistral4Session {
        Mistral4Model::new_session(self)
    }
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn iv_state<'s>(&self, sess: &'s mut Mistral4Session) -> Option<&'s mut crate::intervene::State> {
        Some(&mut sess.iv)
    }
    fn session_len(&self, sess: &Mistral4Session) -> usize {
        sess.pos
    }
    fn forward_hidden_mm(&self, sess: &mut Mistral4Session, tokens: &[u32], mm: &[(usize, &[f32])]) -> Vec<f32> {
        Mistral4Model::forward_hidden_mm(self, sess, tokens, mm)
    }
    fn logits(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        Mistral4Model::logits(self, hidden, t)
    }
    fn forward(&self, sess: &mut Mistral4Session, tokens: &[u32]) -> Vec<f32> {
        Mistral4Model::forward(self, sess, tokens)
    }
}

/// The target of a DSpark draft ([`crate::dspark`]).
impl crate::dspark::Target for Mistral4Model<'_> {
    fn forward_feats(
        &self,
        sess: &mut Mistral4Session,
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
    fn head_raw(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        Mistral4Model::head_raw(self, hidden, t)
    }
    fn head_raw_device(&self, hidden: &[f32], t: usize) -> Option<Vec<f32>> {
        let c = &self.cfg;
        let _timer = prof::start(Cat::Head);
        let mut logits = vec![0f32; t * c.vocab_size];
        let _unit = crate::stream::enter(&self.stream, self.layers.len());
        let jobs = &mut [(&self.lm_head, None, c.vocab_size, &mut logits[..])];
        crate::model::linear_rows_device(jobs, c.hidden_size, hidden, t).then_some(logits)
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn truncate(&self, sess: &mut Mistral4Session, len: usize) {
        sess.truncate(len);
    }
}

/// Runtime interventions (`crate::intervene`): the residual points around every layer, and as
/// neurons the input of `down_proj` of the dense MLP (layer 0: `Target::Mlp`), of a MoE layer's
/// shared expert (`Target::Shared`) and of each routed expert (`Target::Expert`, before the
/// routing weight scales its output), as for Qwen MoE and Gemma 4 26B-A4B.
impl crate::intervene::Host for Mistral4Model<'_> {
    fn n_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }
    fn mlp_inter(&self, layer: usize, target: Target) -> Option<usize> {
        self.layers.get(layer)?.ffn.inter(target)
    }
    fn down_column(&self, layer: usize, target: Target, n: usize) -> Vec<f32> {
        self.layers[layer].ffn.down_column(target, n, self.cfg.hidden_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_names_map_to_transformers() {
        for (a, b) in [
            ("tok_embeddings.weight", "model.language_model.embed_tokens.weight"),
            (
                "layers.3.attention.wq.qscale_weight",
                "model.language_model.layers.3.self_attn.q_proj.weight_scale_inv",
            ),
            (
                "layers.3.attention.wkv_a_with_mqa.weight",
                "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
            ),
            (
                "layers.3.attention.ssmax_scaling.s",
                "model.language_model.layers.3.self_attn.softmax_scaling.scale",
            ),
            (
                "layers.3.attention.indexer.ssmax_scaling.b",
                "model.language_model.layers.3.self_attn.indexer.softmax_scaling.bias",
            ),
            (
                "layers.3.attention.indexer.wq_b.weight",
                "model.language_model.layers.3.self_attn.indexer.wq_b.weight",
            ),
            (
                "layers.5.experts.17.w3.weight",
                "model.language_model.layers.5.mlp.experts.17.up_proj.weight",
            ),
            (
                "layers.5.shared_experts.w2.weight",
                "model.language_model.layers.5.mlp.shared_experts.down_proj.weight",
            ),
            (
                "layers.0.feed_forward.w1.weight",
                "model.language_model.layers.0.mlp.gate_proj.weight",
            ),
            ("layers.5.gate.weight", "model.language_model.layers.5.mlp.gate.weight"),
            (
                "layers.5.ffn_norm.weight",
                "model.language_model.layers.5.post_attention_layernorm.weight",
            ),
            ("lerope.alpha", "model.language_model.rotary_emb.alpha"),
            ("output.weight", "lm_head.weight"),
        ] {
            assert_eq!(native_name(a).as_deref(), Some(b), "{a}");
        }
        assert_eq!(native_name("mtp_dspark.backbone.final_norm.weight"), None);
    }

    #[test]
    fn grouped_routing_matches_plain_with_one_group() {
        let lg = [0.3f32, -1.0, 2.0, 0.5, 1.5, -0.2, 0.0, 0.9];
        let a = route(&lg, 3, 1, 1, 1.0);
        let b = route(&lg, 3, 2, 2, 1.0);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.0, y.0);
            assert!((x.1 - y.1).abs() < 1e-6);
        }
        // One group of four kept: the best expert of the other group is out.
        let c = route(&lg, 2, 2, 1, 1.0);
        assert!(c.iter().all(|r| r.0 >= 4) || c.iter().all(|r| r.0 < 4));
    }
}
