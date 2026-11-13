//! ERNIE 5.5 (`ernie5_5_moe`: baidu/ERNIE-5.5-Flash-Base, 280B with 14B active, 43 layers of 4096)
//! on the DeepSeek V4 model of the parent module, as Baidu's vLLM plugin (github.com/baidu/vLLM-ERNIE,
//! `src/vllm_ernie/models/ernie5_5/`, which subclasses vLLM's DeepSeek V4 attention). What changes
//! from V4:
//!
//! - Two kinds of attention layer. A layer with a positive compression rate (128 in the release,
//!   37 of the 43 layers) is a V4 "HCA" layer: one shared key / value head, a sliding window of
//!   raw keys plus one pooled key per `rate` tokens, YaRN at `compress_rope_theta`. A layer whose
//!   `compress_ratios` entry is 0 or less (one in seven) is a multi-head latent attention layer
//!   (MLA, as DeepSeek V3: `q_a` / `q_b`, a key / value latent of `kv_lora_rank` plus a rotated
//!   part, `kv_b` giving each head its keys and values, plain RoPE at `rope_theta`) over the whole
//!   context, sparse past `index_topk + sliding_window` tokens: each query attends to its window
//!   and to the `index_topk` older tokens a lightning indexer scores highest (`index_n_heads`
//!   queries from the query latent against one key per token, `LayerNorm(wk(x))`, both rotated on
//!   their first dimensions in half-split pairs; `sum_h w_h relu(q_h . k)`). Both kinds have a
//!   learned sink logit per head.
//! - On every attention output: the heads mixed by `I + V U^T` (VHA, rank 16), and a sigmoid gate
//!   computed from the normed query latent (`gate_proj`) on what the last projection reads (the
//!   grouped low-rank output of an HCA layer, the concatenated heads of an MLA layer).
//! - Hyper-connections as V4's with the RMS factor `1 / (sqrt(mean) + eps)` before a block (the
//!   final collapse keeps `rsqrt(mean + eps)`).
//! - Feed-forward: the first `first_k_dense_replace` layers a dense MLP, the others a latent MoE:
//!   two routers scored `sigmoid(l0) + sigmoid(l1)`, the top-k of score plus a selection bias,
//!   weights renormalized and multiplied by a learned scale per expert; the routed experts run on
//!   `fc1(x)` (`moe_latent_size` wide), their weighted sum goes through an RMS norm and `fc2`; one
//!   shared expert on `x`. The activation is SiTU: `beta tanh(g / beta) sigmoid(g) * linear_beta
//!   tanh(u / linear_beta)`.
//! - MultiMax on the logits: `l + t0 relu(r0 - l) + t1 relu(l - r1) + t2 relu(r2 - l)^2 +
//!   t3 relu(l - r3)^2`.
//! - An MTP layer after the model's (ECHO): for position `i` it takes the residual copies before
//!   the final collapse and the embedding of token `i + 1` (`h_proj(hnorm(h)) + e_proj(enorm(e))`
//!   on every copy), runs one MLA layer with its own cache, its own collapse and norm, and the
//!   model's head: the prediction of token `i + 2`. Further drafts feed it its own residual
//!   ([`ErnieMtpDecoder`]).
//!
//! As for V4, the reference's quantization of cached keys and of the indexer is left out.
//!
//! Checkpoints: Baidu's naming (`model.layers.N.self_attn...`, bf16 or the FP8 release with its
//! `weight_scale_inv` blocks), read under V4-style names ([`native_name`]).

use super::*;

/// Whether `files` is a checkpoint in Baidu's naming.
pub(crate) fn is_native(files: &ModelFiles) -> bool {
    files.has("model.mhc_contract.hc_head_fn") && files.has("model.embed_tokens.weight")
}

/// The V4-style name the loader reads a tensor of Baidu's naming under (as the plugin's
/// `ernie5_5_mapper`); None for names kept.
pub(crate) fn native_name(n: &str) -> Option<String> {
    let top = match n {
        "model.embed_tokens.weight" => "embed.weight",
        "lm_head.weight" => "head.weight",
        "model.norm.weight" => "norm.weight",
        "model.lm_head.multimax_ranges" => "multimax_ranges",
        "model.lm_head.multimax_ts" => "multimax_ts",
        _ => "",
    };
    if !top.is_empty() {
        return Some(top.to_string());
    }
    if let Some(r) = n.strip_prefix("model.mhc_contract.") {
        return Some(r.to_string());
    }
    let rest = n.strip_prefix("model.layers.")?;
    let (l, rest) = rest.split_once('.')?;
    // (The MTP layer nests its hyper-connections.)
    let rest = rest.strip_prefix("transformer_layer.").unwrap_or(rest);
    // First match wins: longer prefixes before the ones they extend.
    const MAP: &[(&str, &str)] = &[
        ("self_attn.core_attention.softmax_offset", "attn.attn_sink"),
        ("self_attn.core_attention.compressor.linear_wkv.", "attn.compressor.wkv."),
        ("self_attn.core_attention.compressor.linear_wgate.", "attn.compressor.wgate."),
        ("self_attn.core_attention.", "attn."),
        ("self_attn.gate_proj.", "attn.o_gate."),
        ("self_attn.vha_postmix_", "attn.vha_"),
        ("self_attn.q_layernorm.", "attn.q_norm."),
        ("self_attn.q_a_layernorm.", "attn.q_norm."),
        ("self_attn.kv_layernorm.", "attn.kv_norm."),
        ("self_attn.kv_a_layernorm.", "attn.kv_norm."),
        ("self_attn.linear_q_down_proj.", "attn.wq_a."),
        ("self_attn.q_a_proj.", "attn.wq_a."),
        ("self_attn.linear_q_up_proj.", "attn.wq_b."),
        ("self_attn.q_b_proj.", "attn.wq_b."),
        ("self_attn.linear_kv_proj.", "attn.wkv."),
        ("self_attn.kv_a_proj_with_mqa.", "attn.wkv."),
        ("self_attn.kv_b_proj.", "attn.wkv_b."),
        ("self_attn.linear_o_group_proj_scale_inv", "attn.wo_a.weight_scale_inv"),
        ("self_attn.linear_o_group_proj", "attn.wo_a.weight"),
        ("self_attn.o_proj.", "attn.wo_b."),
        ("input_layernorm.", "attn_norm."),
        ("post_attention_layernorm.", "ffn_norm."),
        ("self_attention_hyper_connection.mapping_proj.weight", "hc_attn_fn"),
        ("self_attention_hyper_connection.bias", "hc_attn_base"),
        ("self_attention_hyper_connection.alpha_", "hc_attn_alpha_"),
        ("mlp_hyper_connection.mapping_proj.weight", "hc_ffn_fn"),
        ("mlp_hyper_connection.bias", "hc_ffn_base"),
        ("mlp_hyper_connection.alpha_", "hc_ffn_alpha_"),
        ("mlp.", "ffn.mlp."),
        ("block_sparse_moe.e_score_correction_bias", "ffn.gate.bias"),
        ("block_sparse_moe.gate.routed_scaling_factor_param", "ffn.gate.scale"),
        ("block_sparse_moe.fc1_latent_proj.", "ffn.fc1."),
        ("block_sparse_moe.fc2_latent_proj.", "ffn.fc2."),
        ("block_sparse_moe.", "ffn."),
    ];
    Some(match MAP.iter().find(|(from, _)| rest.starts_with(from)) {
        Some((from, to)) => format!("layers.{l}.{to}{}", &rest[from.len()..]),
        // (The MTP layer's own tensors: `e_proj`, `hnorm`, `hc_head_fn`, ...)
        None => format!("layers.{l}.{rest}"),
    })
}

/// What an MLA layer adds: `kv_b` per head (keys absorbed into the queries, values applied to
/// the attended latent) and the indexer.
pub(super) struct Mla<'a> {
    /// Per head `[kv_lora, nope]`: the transposed key rows of `kv_b`.
    kbt: Vec<f32>,
    /// Per head `[v, kv_lora]`: the value rows of `kv_b`.
    vb: Vec<f32>,
    iwq: Lin<'a>, // [index heads * dim, q_lora]
    iwk: Lin<'a>, // [index dim, H]
    ik_w: Vec<f32>,
    ik_b: Vec<f32>,
    iproj: Lin<'a>, // [index heads, H]
}

/// What a latent MoE layer adds.
pub(super) struct Latent<'a> {
    fc1: Lin<'a>, // [latent, H]
    fc2: Lin<'a>, // [H, latent]
    norm: Vec<f32>,
    /// The learned scale of each expert's routing weight.
    scale: Vec<f32>,
}

/// What ERNIE 5.5 adds to a layer of the parent module (whose `Attn` holds, for an MLA layer,
/// `q_a` / `q_b` / `kv_a` / the output projection under `wq_a` / `wq_b` / `wkv` / `wo_b`, and
/// whose shared expert is, in a dense layer, the MLP).
pub(super) struct LayerE<'a> {
    gate: Lin<'a>,
    gate_out: usize,
    q_lora: usize,
    /// `I + V U^T`, `[heads, heads]`.
    vha: Vec<f32>,
    heads: usize,
    mla: Option<Mla<'a>>,
    latent: Option<Latent<'a>>,
}

pub(super) struct Mtp<'a> {
    layer: Layer<'a>,
    lx: LayerE<'a>,
    e_proj: Lin<'a>,
    h_proj: Lin<'a>,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    norm: Vec<f32>,
    hc_head: Hc,
}

pub(super) struct Ernie<'a> {
    layers: Vec<LayerE<'a>>,
    mtp: Option<Mtp<'a>>,
    /// MultiMax ranges then temperatures.
    mm: [f32; 8],
}

/// A hyper-connection mixer in either layout: the mixer `[rows, hc * H]` or as Baidu stores it
/// (`[hc * H, rows]`), the three scales as one tensor or as `alpha_pre` / `alpha_post` /
/// `alpha_res`.
pub(super) fn hc(ld: &Loader, p: &str, rows: usize) -> Result<Hc> {
    let c = ld.cfg;
    let w = c.hc_mult * c.hidden_size;
    let fname = format!("{p}_fn");
    let mut f = ld.getf(&fname, &[rows, w])?;
    if ld.files.info(&fname).is_some_and(|i| i.shape.first() == Some(&w)) {
        let src = std::mem::replace(&mut f, vec![0f32; rows * w]);
        for (i, col) in src.chunks(rows).enumerate() {
            for (r, &v) in col.iter().enumerate() {
                f[r * w + i] = v;
            }
        }
    }
    let sname = format!("{p}_scale");
    let scale = if rows == c.hc_mult {
        ld.getf(&sname, &[1])?
    } else if ld.files.has(&sname) {
        ld.getf(&sname, &[3])?
    } else {
        let mut s = Vec::with_capacity(3);
        for a in ["pre", "post", "res"] {
            s.extend(ld.getf(&format!("{p}_alpha_{a}"), &[1])?);
        }
        s
    };
    Ok(Hc {
        f,
        base: ld.getf(&format!("{p}_base"), &[rows])?,
        scale,
    })
}

/// Matrix `name` (`[n_out, n_in]`) as f32, whatever it is stored as.
fn dense(ld: &Loader, name: &str, n_out: usize, n_in: usize) -> Result<Vec<f32>> {
    let info = ld.files.info(name).with_context(|| format!("missing tensor {name}"))?;
    if info.shape.iter().product::<usize>() != n_out * n_in {
        bail!("tensor {name} has shape {:?}, expected [{n_out}, {n_in}]", info.shape);
    }
    if info.dtype == "F8_E4M3" && ld.files.ggml(name).is_none() {
        let m = crate::model::fp8_matrix(ld.files, name, n_out, n_in)?;
        let mut o = vec![0f32; n_out * n_in];
        m.dequantize_rows(0, n_out, &mut o);
        return Ok(o);
    }
    ld.files.f32(name)
}

/// The grouped output projection `name` (`[groups * rank, n_in]`) as one matrix per group; an FP8
/// one is split along its scale blocks.
fn wo_a<'a>(ld: &Loader<'a, '_>, name: &str, groups: usize, rank: usize, n_in: usize) -> Result<Vec<Lin<'a>>> {
    let info = ld.files.info(name).with_context(|| format!("missing tensor {name}"))?;
    if info.dtype != "F8_E4M3" || ld.files.ggml(name).is_some() {
        return ld.row_parts(name, groups, rank, n_in);
    }
    let whole = crate::model::fp8_matrix(ld.files, name, groups * rank, n_in)?;
    let bs = whole.block_scales.as_ref().context("FP8 matrix without block scales")?;
    let (br, bc) = (bs.rows, bs.cols);
    let ncb = n_in.div_ceil(bc);
    if !rank.is_multiple_of(br) && groups * rank > br {
        bail!("tensor {name}: groups of {rank} rows do not fall on its {br}-row scale blocks");
    }
    let data = ld.files.bytes(name)?;
    ld.files.note_use(name, true, false);
    (0..groups)
        .map(|g| {
            let rows = rank.div_ceil(br);
            let first = g * rank / br;
            let mut s = crate::quant::PageVec::<f32>::zeroed(rows * ncb);
            s.copy_from_slice(&bs.s[first * ncb..(first + rows) * ncb]);
            let d = &data[g * rank * n_in..(g + 1) * rank * n_in];
            // SAFETY: the weight borrows `files`, which outlives every model holding it.
            let m = crate::ggml::GgufMatrix::fp8(unsafe { crate::quant::QBuf::borrowed(d) }, s.into(), rank, n_in, (br, bc))?;
            Ok(Lin::gguf(m))
        })
        .collect()
}

/// Layer `l` (the model's or, past them, the MTP layer).
fn load_layer<'a>(ld: &Loader<'a, '_>, l: usize) -> Result<(Layer<'a>, LayerE<'a>)> {
    let c = ld.cfg;
    let e = c.ernie.as_ref().unwrap();
    let h = c.hidden_size;
    let p = format!("layers.{l}.");
    let ap = format!("{p}attn.");
    let mix = (2 + c.hc_mult) * c.hc_mult;
    let is_mla = e.mla_layers[l];
    let (attn, gate, gate_out, q_lora, heads, mla) = if is_mla {
        let (mh, ql, r) = (e.mla_heads, e.mla_q_lora_rank, e.mla_kv_lora_rank);
        let (nope, rd, vd) = (e.mla_qk_nope_head_dim, e.mla_qk_rope_head_dim, e.mla_v_head_dim);
        let (ni, di) = (c.index_n_heads, c.index_head_dim);
        let kvb = dense(ld, &format!("{ap}wkv_b.weight"), mh * (nope + vd), r)?;
        let mut kbt = vec![0f32; mh * r * nope];
        let mut vb = vec![0f32; mh * vd * r];
        for (hh, rows) in kvb.chunks((nope + vd) * r).enumerate() {
            for n in 0..nope {
                for j in 0..r {
                    kbt[(hh * r + j) * nope + n] = rows[n * r + j];
                }
            }
            vb[hh * vd * r..(hh + 1) * vd * r].copy_from_slice(&rows[nope * r..]);
        }
        let attn = Attn {
            wq_a: ld.lin(&format!("{ap}wq_a.weight"), ql, h)?,
            q_norm: ld.getf(&format!("{ap}q_norm.weight"), &[ql])?,
            wq_b: ld.lin(&format!("{ap}wq_b.weight"), mh * (nope + rd), ql)?,
            wkv: ld.lin(&format!("{ap}wkv.weight"), r + rd, h)?,
            kv_norm: ld.getf(&format!("{ap}kv_norm.weight"), &[r])?,
            wo_a: Vec::new(),
            wo_b: ld.lin(&format!("{ap}wo_b.weight"), h, mh * vd)?,
            sink: ld.getf(&format!("{ap}attn_sink"), &[mh])?,
            comp: None,
            indexer: None,
        };
        let ip = format!("{ap}indexer.");
        let x = Mla {
            kbt,
            vb,
            iwq: ld.lin(&format!("{ip}wq_b.weight"), ni * di, ql)?,
            iwk: ld.lin(&format!("{ip}wk.weight"), di, h)?,
            ik_w: ld.getf(&format!("{ip}k_norm.weight"), &[di])?,
            ik_b: ld.getf(&format!("{ip}k_norm.bias"), &[di])?,
            iproj: ld.lin(&format!("{ip}weights_proj.weight"), ni, h)?,
        };
        (attn, ld.lin(&format!("{ap}o_gate.weight"), mh * vd, ql)?, mh * vd, ql, mh, Some(x))
    } else {
        let (nh, hd, ql) = (c.num_attention_heads, c.head_dim, c.q_lora_rank);
        let (g, r) = (c.o_groups, c.o_lora_rank);
        let attn = Attn {
            wq_a: ld.lin(&format!("{ap}wq_a.weight"), ql, h)?,
            q_norm: ld.getf(&format!("{ap}q_norm.weight"), &[ql])?,
            wq_b: ld.lin(&format!("{ap}wq_b.weight"), nh * hd, ql)?,
            wkv: ld.lin(&format!("{ap}wkv.weight"), hd, h)?,
            kv_norm: ld.getf(&format!("{ap}kv_norm.weight"), &[hd])?,
            wo_a: wo_a(ld, &format!("{ap}wo_a.weight"), g, r, nh * hd / g)?,
            wo_b: ld.lin(&format!("{ap}wo_b.weight"), h, g * r)?,
            sink: ld.getf(&format!("{ap}attn_sink"), &[nh])?,
            comp: Some(ld.compressor(&format!("{ap}compressor."), c.ratio(l), hd)?),
            indexer: None,
        };
        (attn, ld.lin(&format!("{ap}o_gate.weight"), g * r, ql)?, g * r, ql, nh, None)
    };
    // The head mix, folded: `I + V U^T`.
    let rank = ld
        .files
        .info(&format!("{ap}vha_U"))
        .map_or(0, |i| i.shape.iter().product::<usize>() / heads);
    let (u, v) = (
        ld.getf(&format!("{ap}vha_U"), &[heads, rank])?,
        ld.getf(&format!("{ap}vha_V"), &[heads, rank])?,
    );
    let mut vha = vec![0f32; heads * heads];
    for hh in 0..heads {
        for g in 0..heads {
            vha[hh * heads + g] = dot(&v[hh * rank..(hh + 1) * rank], &u[g * rank..(g + 1) * rank]) + f32::from(hh == g);
        }
    }
    let fp = format!("{p}ffn.");
    let act = act(c);
    let (moe, latent) = if l < e.first_k_dense_replace {
        let moe = Moe {
            router: Vec::new(),
            pick: Pick::Score(None),
            bias_vl: None,
            experts: Experts::load(ld.files, &format!("{fp}experts."), 0, 0, 0, h, ld.quant, act)?,
            shared: ld.mlp(&format!("{fp}mlp."), e.intermediate_size)?,
        };
        (moe, None)
    } else {
        let (ne, k, mi, ls) = (
            c.n_routed_experts,
            c.num_experts_per_tok,
            c.moe_intermediate_size,
            e.moe_latent_size,
        );
        let mut router = ld.getf(&format!("{fp}gate.weight"), &[ne, h])?;
        router.extend(ld.getf(&format!("{fp}gate.weight_1"), &[ne, h])?);
        let moe = Moe {
            router,
            pick: Pick::Score(Some(ld.getf(&format!("{fp}gate.bias"), &[ne])?)),
            bias_vl: None,
            experts: Experts::load(ld.files, &format!("{fp}experts."), ne, k, mi, ls, ld.quant, act)?,
            shared: ld.mlp(&format!("{fp}shared_experts."), mi * c.n_shared_experts)?,
        };
        let latent = Latent {
            fc1: ld.lin(&format!("{fp}fc1.weight"), ls, h)?,
            fc2: ld.lin(&format!("{fp}fc2.weight"), h, ls)?,
            norm: ld.getf(&format!("{fp}latent_norm.weight"), &[ls])?,
            scale: ld.getf(&format!("{fp}gate.scale"), &[ne])?,
        };
        (moe, Some(latent))
    };
    let layer = Layer {
        attn_norm: ld.getf(&format!("{p}attn_norm.weight"), &[h])?,
        ffn_norm: ld.getf(&format!("{p}ffn_norm.weight"), &[h])?,
        hc_attn: hc(ld, &format!("{p}hc_attn"), mix)?,
        hc_ffn: hc(ld, &format!("{p}hc_ffn"), mix)?,
        attn,
        moe,
        ratio: c.ratio(l),
    };
    let lx = LayerE {
        gate,
        gate_out,
        q_lora,
        vha,
        heads,
        mla,
        latent,
    };
    Ok((layer, lx))
}

fn act(c: &DeepSeek4Config) -> Act {
    let e = c.ernie.as_ref().unwrap();
    Act::Situ {
        beta: e.situ_beta as f32,
        linear_beta: e.situ_linear_beta as f32,
    }
}

/// Loads the layers of the model into `layers`, and what ERNIE 5.5 adds to them.
pub(super) fn load<'a>(ld: &Loader<'a, '_>, layers: &mut Vec<Layer<'a>>) -> Result<Ernie<'a>> {
    let c = ld.cfg;
    let e = c.ernie.as_ref().unwrap();
    let (n, h) = (c.num_hidden_layers, c.hidden_size);
    let mut xs = Vec::with_capacity(n);
    for l in 0..n {
        let (layer, lx) = load_layer(ld, l).with_context(|| format!("layer {l}"))?;
        layers.push(layer);
        xs.push(lx);
    }
    // The MTP layer (one: with more the reference cycles through them, which is not done here).
    let p = format!("layers.{n}.");
    let mtp = if e.mtp_layers == 1 && ld.files.has(&format!("{p}e_proj.weight")) {
        let (layer, lx) = load_layer(ld, n).context("MTP layer")?;
        Some(Mtp {
            layer,
            lx,
            e_proj: ld.lin(&format!("{p}e_proj.weight"), h, h)?,
            h_proj: ld.lin(&format!("{p}h_proj.weight"), h, h)?,
            enorm: ld.getf(&format!("{p}enorm.weight"), &[h])?,
            hnorm: ld.getf(&format!("{p}hnorm.weight"), &[h])?,
            norm: ld.getf(&format!("{p}shared_head.norm.weight"), &[h])?,
            hc_head: hc(ld, &format!("{p}hc_head"), c.hc_mult)?,
        })
    } else {
        None
    };
    let mm: Vec<f32> = match &e.multimax {
        Some(v) => v.clone(),
        None => {
            let mut v = ld.getf("multimax_ranges", &[4])?;
            v.extend(ld.getf("multimax_ts", &[4])?);
            v
        }
    };
    let mm: [f32; 8] = mm
        .try_into()
        .map_err(|_| anyhow::anyhow!("ernie5_5_moe: MultiMax takes 4 ranges and 4 temperatures"))?;
    Ok(Ernie { layers: xs, mtp, mm })
}

/// The cache of an MLA layer: per position the attention row (normed latent, rotated part) in
/// `comp` and the indexer key in `idx`.
pub(super) fn mla_state(c: &DeepSeek4Config) -> LayerState {
    let e = c.ernie.as_ref().unwrap();
    let cs = |d: usize| CompState {
        d,
        ratio: 1,
        ..Default::default()
    };
    LayerState {
        comp: Some(cs(e.mla_kv_lora_rank + e.mla_qk_rope_head_dim)),
        idx: Some(cs(c.index_head_dim)),
        ..Default::default()
    }
}

/// `x[h] = sum_g m[h, g] x[g]` over the heads of each token (`x` `[t, heads, d]`).
fn head_mix(m: &[f32], heads: usize, d: usize, x: &mut [f32], t: usize) {
    for_chunks(x, heads * d, t > 8, |_, row| {
        let src = row.to_vec();
        row.fill(0.0);
        for hh in 0..heads {
            let o = &mut row[hh * d..(hh + 1) * d];
            for g in 0..heads {
                axpy(o, m[hh * heads + g], &src[g * d..(g + 1) * d]);
            }
        }
    });
}

/// Rotates `v` in half-split pairs (`v[i]` with `v[i + half]`).
#[inline]
fn rope_neox(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for i in 0..half {
        let (x1, x2) = (v[i], v[i + half]);
        v[i] = x1 * cos[i] - x2 * sin[i];
        v[i + half] = x2 * cos[i] + x1 * sin[i];
    }
}

fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let r = 1.0 / (var + eps).sqrt();
    for ((v, &w), &b) in x.iter_mut().zip(w).zip(b) {
        *v = (*v - mean) * r * w + b;
    }
}

impl<'a> Ernie<'a> {
    pub(super) fn is_mla(&self, l: usize) -> bool {
        self.layers.get(l).is_some_and(|x| x.mla.is_some())
    }

    /// The matrices layer `l` reads beside those of its `Layer`.
    pub(super) fn lins(&self, l: usize) -> Vec<&Lin<'a>> {
        let Some(x) = self.layers.get(l) else {
            return Vec::new();
        };
        let mut v = vec![&x.gate];
        if let Some(m) = &x.mla {
            v.extend([&m.iwq, &m.iwk, &m.iproj]);
        }
        if let Some(lt) = &x.latent {
            v.extend([&lt.fc1, &lt.fc2]);
        }
        v
    }

    /// For an HCA layer: mixes the heads of its attention output `att` (`[t, heads, hd]`) and
    /// returns the gate logits (`[t, groups * o_lora]`) of the normed query latent `qa`.
    pub(super) fn mix_and_gate(&self, l: usize, qa: &[f32], att: &mut [f32], t: usize) -> Vec<f32> {
        let x = &self.layers[l];
        head_mix(&x.vha, x.heads, att.len() / (t * x.heads), att, t);
        let mut gate = vec![0f32; t * x.gate_out];
        linear(&x.gate, x.gate_out, x.q_lora, qa, t, &mut gate);
        gate
    }

    /// The MultiMax transform of logits.
    pub(super) fn multimax(&self, logits: &mut [f32]) {
        let (r, ts) = self.mm.split_at(4);
        for_chunks(logits, 8192, logits.len() > 65536, |_, row| {
            for v in row {
                let l = *v;
                let (a, b) = ((r[2] - l).max(0.0), (l - r[3]).max(0.0));
                *v = l + ts[0] * (r[0] - l).max(0.0) + ts[1] * (l - r[1]).max(0.0) + ts[2] * a * a + ts[3] * b * b;
            }
        });
    }
}

/// An MLA layer's attention over `xa` (`[t, H]`, the normed block input) into `out`.
#[allow(clippy::too_many_arguments)]
fn mla_attn(
    m: &DeepSeek4Model,
    layer: &Layer,
    lx: &LayerE,
    xa: &[f32],
    t: usize,
    start: usize,
    st: &mut LayerState,
    rope: (&[f32], &[f32]),
    out: &mut [f32],
) {
    let c = &m.cfg;
    let e = c.ernie.as_ref().unwrap();
    let (a, x) = (&layer.attn, lx.mla.as_ref().unwrap());
    let (h, mh, ql, r) = (c.hidden_size, e.mla_heads, e.mla_q_lora_rank, e.mla_kv_lora_rank);
    let (nope, rd, vd) = (e.mla_qk_nope_head_dim, e.mla_qk_rope_head_dim, e.mla_v_head_dim);
    let (ni, di, topk, win) = (c.index_n_heads, c.index_head_dim, c.index_topk, c.sliding_window);
    let eps = c.rms_norm_eps as f32;
    let (kw, qd) = (r + rd, nope + rd);
    let (cos, sin) = rope;
    let half = cos.len() / t;
    let cs_of = |ti: usize| (&cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
    let mut qa = vec![0f32; t * ql];
    let mut kv = vec![0f32; t * kw];
    let mut ik = vec![0f32; t * di];
    let mut iw = vec![0f32; t * ni];
    linear_multi(
        &mut [
            (&a.wq_a, ql, &mut qa[..]),
            (&a.wkv, kw, &mut kv[..]),
            (&x.iwk, di, &mut ik[..]),
            (&x.iproj, ni, &mut iw[..]),
        ],
        h,
        xa,
        t,
    );
    rmsnorm_f32w_rows_inplace(&mut qa, ql, &a.q_norm, eps);
    let mut gate = vec![0f32; t * mh * vd];
    linear(&lx.gate, mh * vd, ql, &qa, t, &mut gate);
    let mut q = vec![0f32; t * mh * qd];
    linear(&a.wq_b, mh * qd, ql, &qa, t, &mut q);
    // The cached row of each token: the normed latent, then the rotated part.
    for_chunks(&mut kv, kw, t > 64, |ti, row| {
        rmsnorm_f32w_inplace(&mut row[..r], &a.kv_norm, eps);
        let (cs, sn) = cs_of(ti);
        rope_tail(row, cs, sn, false);
    });
    for_chunks(&mut ik, di, t > 64, |ti, row| {
        layer_norm(row, &x.ik_w, &x.ik_b, 1e-6);
        let (cs, sn) = cs_of(ti);
        rope_neox(&mut row[..rd], cs, sn);
    });
    {
        let (cs, is) = (st.comp.as_mut().unwrap(), st.idx.as_mut().unwrap());
        debug_assert_eq!((cs.n, is.n), (start, start));
        cs.keys.extend_from_slice(&kv);
        cs.n += t;
        is.keys.extend_from_slice(&ik);
        is.n += t;
    }
    // Queries against the cached rows: each head's key projection absorbed (`K_h^T q`), then its
    // rotated part; scaled so that the row width's `kw^-0.5` of the shared attention is the
    // layer's `(nope + rope)^-0.5`.
    let fix = (kw as f32 / qd as f32).sqrt();
    let mut qx = vec![0f32; t * mh * kw];
    for_chunks(&mut qx, kw, t * mh > 16, |i, o| {
        let (ti, hh) = (i / mh, i % mh);
        let qh = &q[i * qd..(i + 1) * qd];
        let kb = &x.kbt[hh * r * nope..(hh + 1) * r * nope];
        for (v, row) in o[..r].iter_mut().zip(kb.chunks(nope)) {
            *v = dot(row, &qh[..nope]) * fix;
        }
        o[r..].copy_from_slice(&qh[nope..]);
        let (cs, sn) = cs_of(ti);
        rope_tail(&mut o[r..], cs, sn, false);
        scale_inplace(&mut o[r..], fix);
    });
    // Each query's tokens older than its window: all of them, or the indexer's choice once there
    // are more than `index_topk`.
    let older = |i: usize| (start + i + 1).saturating_sub(win);
    let mut sel: Vec<Option<Vec<u32>>> = vec![None; t];
    if older(t - 1) > topk {
        let mut iq = vec![0f32; t * ni * di];
        linear(&x.iwq, ni * di, ql, &qa, t, &mut iq);
        for_chunks(&mut iq, di, t * ni > 64, |i, qh| {
            let (cs, sn) = cs_of(i / ni);
            rope_neox(&mut qh[..rd], cs, sn);
        });
        let ws = (di as f32).powf(-0.5) * (ni as f32).powf(-0.5);
        iw.iter_mut().for_each(|w| *w *= ws);
        sel = select_lag(c, &iq, &iw, st.idx.as_ref().unwrap(), t, start, win, &|_, s| top_positions(s, topk));
    }
    for (i, s) in sel.iter_mut().enumerate() {
        if s.is_none() {
            *s = Some((0..older(i) as u32).collect());
        }
    }
    let cs = st.comp.as_ref().unwrap();
    let wlo = (start + 1).saturating_sub(win);
    let window = LayerState {
        raw: cs.keys[wlo * kw..(start + t) * kw].to_vec(),
        raw_start: wlo,
        ..Default::default()
    };
    let mut att = vec![0f32; t * mh * kw];
    attend_dims((mh, kw, win), &a.sink, &qx, &window, Some(cs), &sel, 1, t, start, &mut att);
    // Each head's values of the attended latent.
    let mut o = vec![0f32; t * mh * vd];
    for_chunks(&mut o, vd, t * mh > 16, |i, oh| {
        let lat = &att[i * kw..i * kw + r];
        let vb = &x.vb[(i % mh) * vd * r..(i % mh + 1) * vd * r];
        for (v, row) in oh.iter_mut().zip(vb.chunks(r)) {
            *v = dot(row, lat);
        }
    });
    head_mix(&lx.vha, mh, vd, &mut o, t);
    o.iter_mut().zip(&gate).for_each(|(v, &g)| *v *= sigmoid(g));
    linear(&a.wo_b, h, mh * vd, &o, t, out);
}

/// An MLP with the SiTU activation (the dense layers, the shared expert) over `x` into `y`.
fn mlp(s: &Mlp, act: Act, x: &[f32], t: usize, h: usize, y: &mut [f32]) {
    let mi = s.inter;
    let mut g = vec![0f32; t * mi];
    let mut u = vec![0f32; t * mi];
    linear_multi(&mut [(&s.gate, mi, &mut g[..]), (&s.up, mi, &mut u[..])], h, x, t);
    act.apply(&mut g, &mut u);
    linear(&s.down, h, mi, &g, t, y);
}

/// The feed-forward block over `x` (`[t, H]`) into `y`.
#[allow(clippy::too_many_arguments)]
fn ffn(m: &DeepSeek4Model, layer: &Layer, lx: &LayerE, x: &[f32], t: usize, y: &mut [f32], need: NeedExperts, hook: Option<&NeuronHook>) {
    let c = &m.cfg;
    let e = c.ernie.as_ref().unwrap();
    let act = act(c);
    let h = c.hidden_size;
    let mo = &layer.moe;
    let Some(lt) = &lx.latent else {
        return mlp(&mo.shared, act, x, t, h, y);
    };
    let (ne, k, ls) = (mo.experts.n_experts(), c.num_experts_per_tok, e.moe_latent_size);
    let bias = match &mo.pick {
        Pick::Score(Some(b)) => &b[..],
        _ => &[],
    };
    // Scores: the two routers' sigmoids summed.
    let mut scores = vec![0f32; t * ne];
    for_chunks(&mut scores, ne, t > 1, |ti, sc| {
        let xr = &x[ti * h..(ti + 1) * h];
        let (r0, r1) = mo.router.split_at(ne * h);
        for (i, v) in sc.iter_mut().enumerate() {
            *v = sigmoid(dot(&r0[i * h..(i + 1) * h], xr)) + sigmoid(dot(&r1[i * h..(i + 1) * h], xr));
        }
    });
    let routes: Vec<Vec<(usize, f32)>> = scores
        .chunks(ne)
        .map(|sc| {
            let pick = |i: usize| sc[i] + bias.get(i).copied().unwrap_or(0.0);
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| pick(b).total_cmp(&pick(a)).then(a.cmp(&b)));
            idx.truncate(k);
            let sum = if c.norm_topk_prob {
                idx.iter().map(|&i| sc[i]).sum::<f32>().max(1e-20)
            } else {
                1.0
            };
            idx.into_iter().map(|i| (i, sc[i] / sum * lt.scale[i])).collect()
        })
        .collect();
    let mut xl = vec![0f32; t * ls];
    linear(&lt.fc1, ls, h, x, t, &mut xl);
    let mut yl = vec![0f32; t * ls];
    mo.experts.forward(&xl, t, ls, &routes, &mut yl, need, hook, None);
    rmsnorm_f32w_rows_inplace(&mut yl, ls, &lt.norm, c.rms_norm_eps as f32);
    linear(&lt.fc2, h, ls, &yl, t, y);
    let mut s = vec![0f32; t * h];
    mlp(&mo.shared, act, x, t, h, &mut s);
    y.iter_mut().zip(&s).for_each(|(v, &s)| *v += s);
}

/// One layer over the residual copies `x` (`[t, hc, H]`): attention then feed-forward, each
/// between a collapse and a spread of the copies. `unit`: the model's layer (its streaming unit,
/// residual points and neuron hooks), None for the MTP layer.
#[allow(clippy::too_many_arguments)]
fn block(
    m: &DeepSeek4Model,
    layer: &Layer,
    lx: &LayerE,
    unit: Option<usize>,
    x: &mut [f32],
    t: usize,
    start: usize,
    st: &mut LayerState,
    ropes: &[(Vec<f32>, Vec<f32>); 2],
    iv: Option<&crate::intervene::State>,
) {
    let c = &m.cfg;
    let (h, n) = (c.hidden_size, c.hc_mult);
    let (xa, post, comb) = hc_pre(c, &layer.hc_attn, &layer.attn_norm, x, t);
    let mut o = vec![0f32; t * h];
    if lx.mla.is_some() {
        mla_attn(m, layer, lx, &xa, t, start, st, (&ropes[0].0, &ropes[0].1), &mut o);
    } else {
        let l = unit.expect("an HCA layer of the model");
        m.attn_forward(l, layer, &xa, t, start, st, (&ropes[1].0, &ropes[1].1), &mut o);
    }
    hc_post(c, x, &o, &post, &comb, t);
    if let (Some(l), Some(iv)) = (unit, iv) {
        iv.point_copies(2 * l + 1, x, h, n, start);
    }
    let (xf, post, comb) = hc_pre(c, &layer.hc_ffn, &layer.ffn_norm, x, t);
    o.fill(0.0);
    let need = |ids: &[usize]| {
        if let (Some(s), Some(l)) = (&m.stream, unit) {
            s.need_parts(l, ids)
        }
    };
    let need: NeedExperts = m.stream.as_ref().and(unit).map(|_| &need as &dyn Fn(&[usize]));
    let hook = unit.and_then(|l| iv.and_then(|s| s.hook(l, start)));
    ffn(m, layer, lx, &xf, t, &mut o, need, hook.as_ref());
    hc_post(c, x, &o, &post, &comb, t);
    if let (Some(l), Some(iv)) = (unit, iv) {
        iv.point_copies(2 * l + 2, x, h, n, start);
    }
}

/// The parent's `forward_capture` for ERNIE 5.5; `resid` also receives the residual copies
/// before the final collapse (`[t, hc * H]`, what the MTP layer reads).
pub(super) fn forward(
    m: &DeepSeek4Model,
    sess: &mut DeepSeek4Session,
    tokens: &[u32],
    layers: &[usize],
    feats: &mut Vec<f32>,
    resid: Option<&mut Vec<f32>>,
) -> Vec<f32> {
    let t = tokens.len();
    assert!(t > 0);
    let c = &m.cfg;
    let e = m.ern.as_ref().unwrap();
    let (h, n) = (c.hidden_size, c.hc_mult);
    let start = sess.pos;
    let timer = prof::start(Cat::Embed);
    if let Some(s) = &m.stream {
        let rows: Vec<(usize, usize)> = tokens.iter().flat_map(|&tk| m.embed.row_ranges(tk as usize, h)).collect();
        s.need_ranges(&rows);
    }
    let mut x = vec![0f32; t * n * h];
    for_chunks(&mut x, n * h, t > 1, |ti, row| {
        m.embed.row(tokens[ti] as usize, &mut row[..h]);
        for k in 1..n {
            row.copy_within(..h, k * h);
        }
    });
    let positions: Vec<usize> = (start..start + t).collect();
    let ropes = [rope_tables(&positions, &m.inv_main), rope_tables(&positions, &m.inv_comp)];
    drop(timer);
    let nf = layers.len();
    feats.clear();
    feats.resize(t * nf * h, 0.0);
    let iv = sess.iv.active().then(|| sess.iv.clone());
    if let Some(iv) = &iv {
        iv.point_copies(0, &mut x, h, n, start);
    }
    let mut capture = |l: usize, x: &[f32]| {
        for (k, _) in layers.iter().enumerate().filter(|e| *e.1 == l) {
            for ti in 0..t {
                let o = &mut feats[(ti * nf + k) * h..(ti * nf + k + 1) * h];
                hc_collapse(&vec![1.0 / n as f32; n], &x[ti * n * h..(ti + 1) * n * h], o);
            }
        }
    };
    for (l, layer) in m.layers.iter().enumerate() {
        capture(l, &x);
        let _unit = crate::stream::enter(&m.stream, l);
        block(
            m,
            layer,
            &e.layers[l],
            Some(l),
            &mut x,
            t,
            start,
            &mut sess.layers[l],
            &ropes,
            iv.as_ref(),
        );
    }
    capture(m.layers.len(), &x);
    let mut out = hc_head(c, &m.hc_head, &x, t);
    rmsnorm_f32w_rows_inplace(&mut out, h, &m.norm, c.rms_norm_eps as f32);
    if let Some(r) = resid {
        *r = x;
    }
    sess.pos += t;
    sess.trim();
    out
}

/// The cache of the MTP layer: one entry per position it has run.
#[derive(Clone)]
pub struct ErnieMtpSession {
    st: LayerState,
    pub pos: usize,
}

impl ErnieMtpSession {
    /// Drops the entries at or beyond `pos`.
    pub fn truncate(&mut self, pos: usize) {
        if pos < self.pos {
            for cs in [&mut self.st.comp, &mut self.st.idx].into_iter().flatten() {
                cs.truncate(pos);
            }
            self.pos = pos;
        }
    }
}

impl<'a> DeepSeek4Model<'a> {
    /// Whether the checkpoint has ERNIE 5.5's MTP layer.
    pub fn has_mtp(&self) -> bool {
        self.ern.as_ref().is_some_and(|e| e.mtp.is_some())
    }

    /// An empty cache for the MTP layer (None without one).
    pub fn mtp_session(&self) -> Option<ErnieMtpSession> {
        self.has_mtp().then(|| ErnieMtpSession {
            st: mla_state(&self.cfg),
            pos: 0,
        })
    }

    /// ERNIE 5.5: [`Self::forward_hidden`] that also gives the residual copies before the final
    /// collapse (`[t, hc_mult * hidden]`), the input of the MTP layer.
    pub fn forward_residual(&self, sess: &mut DeepSeek4Session, tokens: &[u32], resid: &mut Vec<f32>) -> Vec<f32> {
        assert!(self.ern.is_some(), "the residual before the head is read for ERNIE 5.5 only");
        forward(self, sess, tokens, &[], &mut Vec::new(), Some(resid))
    }

    /// ERNIE 5.5's MTP layer over the positions `ms.pos..`: `resid` (`[t, hc_mult * hidden]`) the
    /// model's residual copies at those positions (or, for a draft after a draft, the layer's own
    /// `out_resid`), `tokens` the token after each. Returns the final-normed hidden states
    /// (`[t, hidden]`, for [`Self::logits`]: the prediction of the token two after each position).
    pub fn mtp_forward(&self, ms: &mut ErnieMtpSession, resid: &[f32], tokens: &[u32], out_resid: &mut Vec<f32>) -> Vec<f32> {
        let c = &self.cfg;
        let mp = self.ern.as_ref().and_then(|e| e.mtp.as_ref()).expect("no MTP layer");
        let (h, n, t) = (c.hidden_size, c.hc_mult, tokens.len());
        assert_eq!(resid.len(), t * n * h);
        let eps = c.rms_norm_eps as f32;
        let mut emb = vec![0f32; t * h];
        for_chunks(&mut emb, h, t > 1, |ti, row| {
            self.embed.row(tokens[ti] as usize, row);
            rmsnorm_f32w_inplace(row, &mp.enorm, eps);
        });
        let mut ep = vec![0f32; t * h];
        linear(&mp.e_proj, h, h, &emb, t, &mut ep);
        let mut hn = resid.to_vec();
        rmsnorm_f32w_rows_inplace(&mut hn, h, &mp.hnorm, eps);
        let mut x = vec![0f32; t * n * h];
        linear(&mp.h_proj, h, h, &hn, t * n, &mut x);
        for_chunks(&mut x, n * h, t > 8, |ti, row| {
            for copy in row.chunks_mut(h) {
                axpy(copy, 1.0, &ep[ti * h..(ti + 1) * h]);
            }
        });
        let positions: Vec<usize> = (ms.pos..ms.pos + t).collect();
        let ropes = [rope_tables(&positions, &self.inv_main), (Vec::new(), Vec::new())];
        block(self, &mp.layer, &mp.lx, None, &mut x, t, ms.pos, &mut ms.st, &ropes, None);
        ms.pos += t;
        let mut out = hc_head(c, &mp.hc_head, &x, t);
        rmsnorm_f32w_rows_inplace(&mut out, h, &mp.norm, eps);
        *out_resid = x;
        out
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |x| x.0 as u32)
}

/// Greedy decoding of ERNIE 5.5 with drafts from its MTP layer: each round the layer proposes
/// `drafts` tokens after the next one, the model runs them in one pass and keeps those it would
/// have chosen itself, so the output is the model's own greedy output.
pub struct ErnieMtpDecoder<'m, 'a> {
    m: &'m DeepSeek4Model<'a>,
    pub sess: DeepSeek4Session,
    ms: ErnieMtpSession,
    /// The model's residual copies at the positions the MTP layer has not run yet
    /// (`ms.pos..sess.pos`), and the tokens at `ms.pos + 1..sess.pos`.
    pend: Vec<f32>,
    pend_tokens: Vec<u32>,
    /// The token after the ones the model has run (chosen, not fed yet).
    next: u32,
    pub drafts: usize,
    /// Rounds, tokens drafted, drafts kept.
    pub stats: (usize, usize, usize),
}

impl<'m, 'a> ErnieMtpDecoder<'m, 'a> {
    pub fn new(m: &'m DeepSeek4Model<'a>, drafts: usize) -> Result<Self> {
        let ms = m.mtp_session().context("this checkpoint has no MTP layer")?;
        Ok(Self {
            m,
            sess: m.new_session(),
            ms,
            pend: Vec::new(),
            pend_tokens: Vec::new(),
            next: 0,
            drafts: drafts.max(1),
            stats: (0, 0, 0),
        })
    }

    /// Runs the prompt; returns the first generated token (which [`Self::step`] then feeds).
    pub fn prefill(&mut self, prompt: &[u32]) -> u32 {
        let h = self.m.cfg.hidden_size;
        let mut resid = Vec::new();
        let hs = self.m.forward_residual(&mut self.sess, prompt, &mut resid);
        self.next = argmax(&self.m.logits(&hs[hs.len() - h..], 1));
        self.pend.extend(resid);
        self.pend_tokens.extend(&prompt[1..]);
        self.next
    }

    /// One round: returns the tokens it settles after the last one returned (one more than the
    /// drafts kept); the last of them is fed by the next round.
    pub fn step(&mut self) -> Vec<u32> {
        let m = self.m;
        let (h, w) = (m.cfg.hidden_size, m.cfg.hc_mult * m.cfg.hidden_size);
        let p = self.sess.pos;
        // The MTP layer over the positions the model has settled since its last run; its last row
        // is the first draft.
        let mut toks = std::mem::take(&mut self.pend_tokens);
        toks.push(self.next);
        let mut r = Vec::new();
        let hs = m.mtp_forward(&mut self.ms, &self.pend, &toks, &mut r);
        let mut seq = vec![self.next, argmax(&m.logits(&hs[hs.len() - h..], 1))];
        let mut last = r[r.len() - w..].to_vec();
        for _ in 1..self.drafts {
            let hs = m.mtp_forward(&mut self.ms, &last, &seq[seq.len() - 1..], &mut r);
            seq.push(argmax(&m.logits(&hs, 1)));
            last = std::mem::take(&mut r);
        }
        self.ms.truncate(p);
        // The model over the next token and the drafts: its own choice after each.
        let mut resid = Vec::new();
        let hs = m.forward_residual(&mut self.sess, &seq, &mut resid);
        let lg = m.logits(&hs, seq.len());
        let v = m.cfg.vocab_size;
        let own: Vec<u32> = lg.chunks(v).map(argmax).collect();
        let kept = (1..seq.len()).take_while(|&j| seq[j] == own[j - 1]).count();
        self.stats.0 += 1;
        self.stats.1 += seq.len() - 1;
        self.stats.2 += kept;
        self.sess.truncate(p + 1 + kept);
        self.pend = resid[..(kept + 1) * w].to_vec();
        self.pend_tokens = seq[1..=kept].to_vec();
        self.next = own[kept];
        let mut settled = seq[1..=kept].to_vec();
        settled.push(self.next);
        settled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let n = |s: &str| native_name(s).unwrap();
        assert_eq!(
            n("model.layers.3.self_attn.linear_q_down_proj.weight_scale_inv"),
            "layers.3.attn.wq_a.weight_scale_inv"
        );
        assert_eq!(
            n("model.layers.3.self_attn.linear_o_group_proj_scale_inv"),
            "layers.3.attn.wo_a.weight_scale_inv"
        );
        assert_eq!(
            n("model.layers.7.self_attn.core_attention.softmax_offset"),
            "layers.7.attn.attn_sink"
        );
        assert_eq!(
            n("model.layers.7.self_attn.core_attention.indexer.k_norm.bias"),
            "layers.7.attn.indexer.k_norm.bias"
        );
        assert_eq!(
            n("model.layers.0.self_attn.core_attention.compressor.linear_wgate.weight"),
            "layers.0.attn.compressor.wgate.weight"
        );
        assert_eq!(
            n("model.layers.43.transformer_layer.mlp_hyper_connection.alpha_res"),
            "layers.43.hc_ffn_alpha_res"
        );
        assert_eq!(n("model.layers.0.mlp.up_proj.weight"), "layers.0.ffn.mlp.up_proj.weight");
        assert_eq!(
            n("model.layers.1.block_sparse_moe.experts.511.w3.weight"),
            "layers.1.ffn.experts.511.w3.weight"
        );
        assert_eq!(n("model.layers.1.block_sparse_moe.gate.weight_1"), "layers.1.ffn.gate.weight_1");
        assert_eq!(n("model.layers.43.shared_head.norm.weight"), "layers.43.shared_head.norm.weight");
        assert_eq!(n("model.mhc_contract.hc_head_fn"), "hc_head_fn");
        assert_eq!(n("model.lm_head.multimax_ts"), "multimax_ts");
        assert!(native_name("embed.weight").is_none());
    }

    /// FP8 matrices smaller than, or not a multiple of, their scale blocks (a tiny checkpoint's)
    /// multiply as their dequantized rows.
    #[test]
    fn fp8_small_blocks() {
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for (n_out, n_in) in [(48usize, 256usize), (64, 96), (384, 256), (160, 48), (256, 256)] {
            let mut d = crate::quant::PageVec::<u8>::zeroed(n_out * n_in);
            for b in d.iter_mut() {
                *b = (rnd() % 127) as u8 | (((rnd() & 1) as u8) << 7);
            }
            let ns = n_out.div_ceil(128) * n_in.div_ceil(128);
            let mut s = crate::quant::PageVec::<f32>::zeroed(ns);
            for v in s.iter_mut() {
                *v = 0.001 + (rnd() % 100) as f32 * 1e-4;
            }
            let m = crate::ggml::GgufMatrix::fp8(d.into(), s.into(), n_out, n_in, (128, 128)).unwrap();
            let mut w = vec![0f32; n_out * n_in];
            m.dequantize_rows(0, n_out, &mut w);
            let lin = Lin::gguf(m);
            for t in [1usize, 5, 40] {
                let x: Vec<f32> = (0..t * n_in).map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0).collect();
                let mut y = vec![0f32; t * n_out];
                linear(&lin, n_out, n_in, &x, t, &mut y);
                for ti in 0..t {
                    for o in 0..n_out {
                        let want = dot(&w[o * n_in..(o + 1) * n_in], &x[ti * n_in..(ti + 1) * n_in]);
                        let got = y[ti * n_out + o];
                        assert!(
                            (want - got).abs() <= 1e-3 * (1.0 + want.abs()),
                            "[{n_out}, {n_in}] t {t} row {o}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }
}
