//! BAGEL's language model: Qwen2.5-7B as a Mixture-of-Transformers. Every decoder layer holds two
//! experts with the same shape, the understanding expert (the original Qwen2 weights: text and
//! SigLIP tokens) and the generation expert (`*_moe_gen`: VAE latent tokens). Each token goes
//! through its own expert's norms, projections and MLP; the experts meet in self-attention, which
//! runs over the joint sequence. Both experts add per-head RMSNorm on queries and keys
//! (`qk_norm`) to Qwen2's attention (q/k/v bias, GQA, rotate-half RoPE).
//!
//! The KV cache is one flat sequence per layer (the reference's `NaiveCache`, batch 1). A block
//! of new tokens attends to a prefix of a cache plus itself: text blocks causally (positions
//! increase), image blocks fully (every token of an image shares one RoPE position). Several
//! guidance branches read prefixes of the same cache without copying it.

use crate::kernels::{add_inplace, apply_rope, axpy, dot, for_chunks, rmsnorm_rows, rope_inv_freq, rope_tables, silu_mul};
use crate::model::{Lin, Quant, Table, linear_multi, load_lin};
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};

/// `llm_config.json` (a transformers `Qwen2Config`).
#[derive(Clone, Debug)]
pub struct LlmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
}

impl LlmConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let get = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as usize);
        let hidden_size = get("hidden_size", 3584);
        let num_attention_heads = get("num_attention_heads", 28);
        let cfg = Self {
            hidden_size,
            intermediate_size: get("intermediate_size", 18944),
            num_hidden_layers: get("num_hidden_layers", 28),
            num_attention_heads,
            num_key_value_heads: get("num_key_value_heads", 4),
            head_dim: hidden_size / num_attention_heads.max(1),
            vocab_size: get("vocab_size", 152064),
            rms_norm_eps: v.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6) as f32,
            rope_theta: v.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(1e6),
        };
        if v.get("rope_scaling").is_some_and(|s| !s.is_null()) {
            bail!("llm_config: rope_scaling is not supported");
        }
        if cfg.num_key_value_heads == 0 || !cfg.num_attention_heads.is_multiple_of(cfg.num_key_value_heads) {
            bail!(
                "llm_config: {} heads for {} KV heads",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            );
        }
        Ok(cfg)
    }
}

/// A linear layer with an optional f32 bias.
pub(crate) struct Proj<'a> {
    pub w: Lin<'a>,
    pub b: Option<Vec<f32>>,
    pub n_out: usize,
    pub n_in: usize,
}

impl<'a> Proj<'a> {
    pub fn load(files: &'a ModelFiles, p: &str, n_out: usize, n_in: usize, quant: Quant, bias: bool) -> Result<Self> {
        let b = if bias {
            let b = files.f32(&format!("{p}.bias"))?;
            if b.len() != n_out {
                bail!("{p}.bias: {} values, expected {n_out}", b.len());
            }
            Some(b)
        } else {
            None
        };
        Ok(Self {
            // Quantized blocks span 32 inputs (SigLIP's fc2 has 4304): other widths stay bf16.
            w: load_lin(
                files,
                &format!("{p}.weight"),
                n_out,
                n_in,
                if n_in.is_multiple_of(crate::quant::QBLOCK) {
                    quant
                } else {
                    Quant::Bf16
                },
            )?,
            b,
            n_out,
            n_in,
        })
    }

    fn add_bias(&self, y: &mut [f32]) {
        if let Some(b) = &self.b {
            for_chunks(y, self.n_out, y.len() > 4 * self.n_out, |_, r| {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            });
        }
    }
}

/// Several projections of one input (one fused matmul region / GPU submission).
pub(crate) fn project(ps: &[&Proj], n_in: usize, x: &[f32], rows: usize) -> Vec<Vec<f32>> {
    let mut ys: Vec<Vec<f32>> = ps.iter().map(|p| vec![0f32; rows * p.n_out]).collect();
    if rows == 0 {
        return ys;
    }
    {
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = ps.iter().zip(ys.iter_mut()).map(|(p, y)| (&p.w, p.n_out, &mut y[..])).collect();
        linear_multi(&mut jobs, n_in, x, rows);
    }
    for (p, y) in ps.iter().zip(ys.iter_mut()) {
        p.add_bias(y);
    }
    ys
}

/// One expert of one layer.
struct Expert<'a> {
    ln_in: &'a [u16],
    ln_post: &'a [u16],
    q: Proj<'a>,
    k: Proj<'a>,
    v: Proj<'a>,
    o: Proj<'a>,
    q_norm: &'a [u16],
    k_norm: &'a [u16],
    gate: Proj<'a>,
    up: Proj<'a>,
    down: Proj<'a>,
}

struct Layer<'a> {
    und: Expert<'a>,
    gen_: Option<Expert<'a>>,
}

pub struct Llm<'a> {
    pub cfg: LlmConfig,
    embed: Table<'a>,
    layers: Vec<Layer<'a>>,
    norm: &'a [u16],
    norm_gen: Option<&'a [u16]>,
    lm_head: Option<Proj<'a>>,
    inv_freq: Vec<f32>,
    scratch: std::sync::Mutex<Scratch>,
}

/// A KV cache: per layer and KV head, `[len, head_dim]` keys and values.
pub struct Kv {
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
    pub len: usize,
}

impl Kv {
    pub fn new(layers: usize, kv_heads: usize) -> Self {
        Self {
            k: (0..layers).map(|_| vec![Vec::new(); kv_heads]).collect(),
            v: (0..layers).map(|_| vec![Vec::new(); kv_heads]).collect(),
            len: 0,
        }
    }

    /// Drops everything after the first `len` positions.
    pub fn truncate(&mut self, len: usize, hd: usize) {
        for l in self.k.iter_mut().chain(self.v.iter_mut()) {
            for h in l.iter_mut() {
                h.truncate(len * hd);
            }
        }
        self.len = self.len.min(len);
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.k.iter().chain(&self.v).flatten().map(|h| h.len() * 4).sum()
    }
}

/// What a block of new tokens attends to besides itself.
pub enum Ctx<'c> {
    /// The whole cache, which the block's keys and values are appended to.
    Append(&'c mut Kv),
    /// The first `n` positions of a cache, left unchanged.
    Prefix(&'c Kv, usize),
}

/// A block of new tokens: `und` rows (text, SigLIP and image-marker tokens: the understanding
/// expert) and `gen` rows (VAE latent tokens: the generation expert), both `[rows, hidden]`.
/// Text blocks (`text`, no `gen` rows) are causal with positions `pos..`; image blocks are
/// bidirectional with every token at position `pos`.
pub struct Block {
    pub und: Vec<f32>,
    pub gen_: Vec<f32>,
    pub pos: usize,
    pub text: bool,
}

/// Optional `OXINFER_BAGEL_LAYERS=N`: run only the first N decoder layers (reference checks
/// against a truncated model).
fn layer_limit() -> Option<usize> {
    std::env::var("OXINFER_BAGEL_LAYERS").ok()?.trim().parse().ok()
}

impl<'a> Llm<'a> {
    /// Loads `language_model.*`; the generation expert only with `gen_`, the LM head only with
    /// `head` (text output: understanding, thinking).
    pub fn load(cfg: LlmConfig, files: &'a ModelFiles, quant: Quant, gen_: bool, head: bool) -> Result<Self> {
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let vec = |n: &str, len: usize| -> Result<&'a [u16]> {
            let w = files.bf16(n)?;
            if w.len() != len {
                bail!("{n}: {} values, expected {len}", w.len());
            }
            Ok(w)
        };
        let expert = |i: usize, s: &str| -> Result<Expert<'a>> {
            let p = format!("language_model.model.layers.{i}");
            let a = format!("{p}.self_attn");
            let m = format!("{p}.mlp{s}");
            Ok(Expert {
                ln_in: vec(&format!("{p}.input_layernorm{s}.weight"), h)?,
                ln_post: vec(&format!("{p}.post_attention_layernorm{s}.weight"), h)?,
                q: Proj::load(files, &format!("{a}.q_proj{s}"), nh * hd, h, quant, true)?,
                k: Proj::load(files, &format!("{a}.k_proj{s}"), nkv * hd, h, quant, true)?,
                v: Proj::load(files, &format!("{a}.v_proj{s}"), nkv * hd, h, quant, true)?,
                o: Proj::load(files, &format!("{a}.o_proj{s}"), h, nh * hd, quant, false)?,
                q_norm: vec(&format!("{a}.q_norm{s}.weight"), hd)?,
                k_norm: vec(&format!("{a}.k_norm{s}.weight"), hd)?,
                gate: Proj::load(files, &format!("{m}.gate_proj"), inter, h, quant, false)?,
                up: Proj::load(files, &format!("{m}.up_proj"), inter, h, quant, false)?,
                down: Proj::load(files, &format!("{m}.down_proj"), h, inter, quant, false)?,
            })
        };
        let n_layers = layer_limit().unwrap_or(cfg.num_hidden_layers).min(cfg.num_hidden_layers);
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(Layer {
                und: expert(i, "").with_context(|| format!("layer {i}"))?,
                gen_: if gen_ {
                    Some(expert(i, "_moe_gen").with_context(|| format!("layer {i} (generation expert)"))?)
                } else {
                    None
                },
            });
        }
        let embed =
            Table::load(files, "language_model.model.embed_tokens.weight", cfg.vocab_size, h, Quant::Bf16).context("token embeddings")?;
        let lm_head = if head {
            Some(Proj::load(files, "language_model.lm_head", cfg.vocab_size, h, quant, false)?)
        } else {
            None
        };
        Ok(Self {
            inv_freq: rope_inv_freq(hd, cfg.rope_theta, 1.0),
            norm: vec("language_model.model.norm.weight", h)?,
            norm_gen: if gen_ {
                Some(vec("language_model.model.norm_moe_gen.weight", h)?)
            } else {
                None
            },
            cfg,
            embed,
            layers,
            lm_head,
            scratch: Default::default(),
        })
    }

    pub fn new_kv(&self) -> Kv {
        Kv::new(self.layers.len(), self.cfg.num_key_value_heads)
    }

    pub fn has_head(&self) -> bool {
        self.lm_head.is_some()
    }

    /// Token embeddings `[ids.len(), hidden]`.
    pub fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let mut x = vec![0f32; ids.len() * h];
        for (r, &id) in x.chunks_mut(h).zip(ids) {
            self.embed.row(id as usize, r);
        }
        x
    }

    /// Text logits `[rows, vocab]` of final-normed hidden states.
    pub fn logits(&self, hidden: &[f32], rows: usize) -> Result<Vec<f32>> {
        let head = self.lm_head.as_ref().context("the LM head was not loaded")?;
        Ok(project(&[head], self.cfg.hidden_size, hidden, rows).swap_remove(0))
    }

    /// Runs `b` through the decoder; returns the final-normed hidden states of its `und` and
    /// `gen` rows.
    pub fn forward(&self, ctx: Ctx, b: Block) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok(self.forward_multi(vec![ctx], vec![b])?.swap_remove(0))
    }

    /// Runs several blocks of the same shape (one per context, e.g. the guidance branches of a
    /// denoising step) as one batch: every matmul takes all their rows, attention stays per
    /// block. Returns each block's final-normed `und` and `gen` hidden states.
    pub fn forward_multi(&self, ctxs: Vec<Ctx>, blocks: Vec<Block>) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let c = &self.cfg;
        let h = c.hidden_size;
        let nb = blocks.len();
        let (xu, xg, nu, ng) = self.forward_layers(ctxs, blocks)?;
        let mut hu = vec![0f32; xu.len()];
        let mut hg = vec![0f32; xg.len()];
        rmsnorm_rows(&xu, h, Some(self.norm), c.rms_norm_eps, &mut hu);
        if ng > 0 {
            rmsnorm_rows(&xg, h, self.norm_gen, c.rms_norm_eps, &mut hg);
        }
        Ok((0..nb)
            .map(|bi| {
                (
                    hu[bi * nu * h..(bi + 1) * nu * h].to_vec(),
                    hg[bi * ng * h..(bi + 1) * ng * h].to_vec(),
                )
            })
            .collect())
    }

    /// [`Llm::forward_multi`] for the generation expert's rows, before the final norm: the last
    /// layer's hidden states of each block's latent tokens (`[n_gen, hidden]` per block), the
    /// feature a [`crate::step_cache::TaylorSeer`] keeps. [`Llm::final_norm_gen`] completes them.
    pub fn forward_multi_gen_raw(&self, ctxs: Vec<Ctx>, blocks: Vec<Block>) -> Result<Vec<Vec<f32>>> {
        let h = self.cfg.hidden_size;
        let nb = blocks.len();
        let (_, xg, _, ng) = self.forward_layers(ctxs, blocks)?;
        Ok((0..nb).map(|bi| xg[bi * ng * h..(bi + 1) * ng * h].to_vec()).collect())
    }

    /// The generation expert's final norm over `[rows, hidden]` last-layer states.
    pub fn final_norm_gen(&self, xg: &[f32]) -> Vec<f32> {
        let mut hg = vec![0f32; xg.len()];
        rmsnorm_rows(xg, self.cfg.hidden_size, self.norm_gen, self.cfg.rms_norm_eps, &mut hg);
        hg
    }

    /// The layers of [`Llm::forward_multi`]: every block's understanding rows and generation
    /// rows after the last layer (before the final norms), and the rows per block of each.
    fn forward_layers(&self, mut ctxs: Vec<Ctx>, blocks: Vec<Block>) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
        let c = &self.cfg;
        let (h, inter) = (c.hidden_size, c.intermediate_size);
        let (nh, nkv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let nb = blocks.len();
        if nb == 0 || ctxs.len() != nb {
            bail!("forward: {} contexts for {nb} blocks", ctxs.len());
        }
        let (nu, ng, text) = (blocks[0].und.len() / h, blocks[0].gen_.len() / h, blocks[0].text);
        for b in &blocks {
            if b.und.len() != nu * h || b.gen_.len() != ng * h || b.text != text {
                bail!("forward: blocks of different shapes");
            }
        }
        if text && ng > 0 {
            bail!("forward: a text block has no latent tokens");
        }
        if ng > 0 && self.layers.first().is_some_and(|l| l.gen_.is_none()) {
            bail!("the generation expert was not loaded");
        }
        if nb > 1 && ctxs.iter().any(|c| matches!(c, Ctx::Append(_))) {
            bail!("forward: only a single block can extend its cache");
        }
        let t = nu + ng;
        let mut n_ctx = Vec::with_capacity(nb);
        for ctx in &ctxs {
            n_ctx.push(match ctx {
                Ctx::Append(kv) => kv.len,
                Ctx::Prefix(kv, n) => {
                    if *n > kv.len {
                        bail!("forward: prefix {n} of a {}-position cache", kv.len);
                    }
                    *n
                }
            });
        }
        let half = self.inv_freq.len();
        // RoPE tables per block: one position per image block, increasing positions for text.
        let tables: Vec<(Vec<f32>, Vec<f32>)> = blocks
            .iter()
            .map(|b| {
                let positions: Vec<usize> = if text { (b.pos..b.pos + t).collect() } else { vec![b.pos] };
                rope_tables(&positions, &self.inv_freq)
            })
            .collect();
        let (ru, rg) = (nb * nu, nb * ng);
        let mut xu = Vec::with_capacity(ru * h);
        let mut xg = Vec::with_capacity(rg * h);
        for b in blocks {
            xu.extend_from_slice(&b.und);
            xg.extend_from_slice(&b.gen_);
        }
        let mut guard = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let sc = &mut *guard;
        let (qd, kd) = (nh * hd, nkv * hd);
        let rows = nb * t;
        let mut hu = vec![0f32; ru * h];
        let mut hg = vec![0f32; rg * h];
        for (li, l) in self.layers.iter().enumerate() {
            // Attention inputs, block by block, und rows first (the order within a
            // bidirectional block is free).
            grow(&mut sc.q, rows * qd);
            grow(&mut sc.k, rows * kd);
            grow(&mut sc.v, rows * kd);
            let qkv = |sc: &mut Scratch, e: &Expert, x: &[f32], hbuf: &mut [f32], n: usize, off: usize| {
                if n == 0 {
                    return;
                }
                let r = nb * n;
                rmsnorm_rows(x, h, Some(e.ln_in), c.rms_norm_eps, hbuf);
                grow(&mut sc.qq, r * qd);
                grow(&mut sc.kk, r * kd);
                grow(&mut sc.vv, r * kd);
                let (qq, kk, vv) = (&mut sc.qq[..r * qd], &mut sc.kk[..r * kd], &mut sc.vv[..r * kd]);
                project_into(&[&e.q, &e.k, &e.v], h, hbuf, r, &mut [&mut qq[..], &mut kk[..], &mut vv[..]]);
                head_norm(qq, e.q_norm, hd, c.rms_norm_eps);
                head_norm(kk, e.k_norm, hd, c.rms_norm_eps);
                for bi in 0..nb {
                    let (src, dst) = (bi * n, bi * t + off);
                    sc.q[dst * qd..(dst + n) * qd].copy_from_slice(&qq[src * qd..(src + n) * qd]);
                    sc.k[dst * kd..(dst + n) * kd].copy_from_slice(&kk[src * kd..(src + n) * kd]);
                    sc.v[dst * kd..(dst + n) * kd].copy_from_slice(&vv[src * kd..(src + n) * kd]);
                }
            };
            qkv(sc, &l.und, &xu, &mut hu, nu, 0);
            if ng > 0 {
                qkv(sc, l.gen_.as_ref().unwrap(), &xg, &mut hg, ng, nu);
            }
            let rope = |m: &mut [f32], heads: usize| {
                for_chunks(m, hd, rows * heads > 64, |i, r| {
                    let (bi, ti) = (i / (t * heads), (i / heads) % t);
                    let ti = if text { ti } else { 0 };
                    let (cos, sin) = &tables[bi];
                    apply_rope(r, &cos[ti * half..(ti + 1) * half], &sin[ti * half..(ti + 1) * half]);
                });
            };
            rope(&mut sc.q[..rows * qd], nh);
            rope(&mut sc.k[..rows * kd], nkv);
            grow(&mut sc.a, rows * qd);
            for (bi, ctx) in ctxs.iter().enumerate() {
                let (kc, vc): (&[Vec<f32>], &[Vec<f32>]) = match ctx {
                    Ctx::Append(kv) => (&kv.k[li], &kv.v[li]),
                    Ctx::Prefix(kv, _) => (&kv.k[li], &kv.v[li]),
                };
                let (r0, r1) = (bi * t, (bi + 1) * t);
                let o = attend(
                    &sc.q[r0 * qd..r1 * qd],
                    kc,
                    vc,
                    n_ctx[bi],
                    &sc.k[r0 * kd..r1 * kd],
                    &sc.v[r0 * kd..r1 * kd],
                    t,
                    nh,
                    nkv,
                    hd,
                    text,
                );
                sc.a[r0 * qd..r1 * qd].copy_from_slice(&o);
            }
            if let Some(Ctx::Append(kv)) = ctxs.first_mut() {
                for g in 0..nkv {
                    let (kh, vh) = (&mut kv.k[li][g], &mut kv.v[li][g]);
                    for i in 0..t {
                        kh.extend_from_slice(&sc.k[(i * nkv + g) * hd..(i * nkv + g + 1) * hd]);
                        vh.extend_from_slice(&sc.v[(i * nkv + g) * hd..(i * nkv + g + 1) * hd]);
                    }
                }
            }
            // Output projection, residual, MLP: per expert over every block's rows.
            let step = |sc: &mut Scratch, e: &Expert, x: &mut [f32], hbuf: &mut [f32], n: usize, off: usize| {
                if n == 0 {
                    return;
                }
                let r = nb * n;
                grow(&mut sc.qq, r * qd);
                for bi in 0..nb {
                    let src = bi * t + off;
                    sc.qq[bi * n * qd..(bi + 1) * n * qd].copy_from_slice(&sc.a[src * qd..(src + n) * qd]);
                }
                grow(&mut sc.o, r * h);
                project_into(&[&e.o], qd, &sc.qq[..r * qd], r, &mut [&mut sc.o[..r * h]]);
                add_inplace(x, &sc.o[..r * h]);
                rmsnorm_rows(x, h, Some(e.ln_post), c.rms_norm_eps, hbuf);
                // The MLP in row chunks bounds the intermediate activations.
                const CH: usize = 4096;
                for c0 in (0..r).step_by(CH) {
                    let m = CH.min(r - c0);
                    let xs = &hbuf[c0 * h..(c0 + m) * h];
                    grow(&mut sc.g, m * inter);
                    grow(&mut sc.u, m * inter);
                    let (g, u) = (&mut sc.g[..m * inter], &mut sc.u[..m * inter]);
                    project_into(&[&e.gate, &e.up], h, xs, m, &mut [&mut g[..], &mut u[..]]);
                    silu_mul(g, u);
                    let d = &mut sc.o[..m * h];
                    project_into(&[&e.down], inter, g, m, &mut [&mut d[..]]);
                    add_inplace(&mut x[c0 * h..(c0 + m) * h], d);
                }
            };
            step(sc, &l.und, &mut xu, &mut hu, nu, 0);
            if ng > 0 {
                step(sc, l.gen_.as_ref().unwrap(), &mut xg, &mut hg, ng, nu);
            }
        }
        drop(guard);
        if let Some(Ctx::Append(kv)) = ctxs.first_mut() {
            kv.len += t;
        }
        Ok((xu, xg, nu, ng))
    }
}

/// Reused activation buffers (a denoising run would otherwise map and fault in gigabytes of
/// fresh memory per pass).
#[derive(Default)]
pub(crate) struct Scratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    qq: Vec<f32>,
    kk: Vec<f32>,
    vv: Vec<f32>,
    a: Vec<f32>,
    o: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
}

/// Grows `v` to at least `n` values (never shrinks; contents unspecified).
fn grow(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// [`project`] into caller-provided outputs.
pub(crate) fn project_into(ps: &[&Proj], n_in: usize, x: &[f32], rows: usize, ys: &mut [&mut [f32]]) {
    if rows == 0 {
        return;
    }
    {
        let mut jobs: Vec<(&Lin, usize, &mut [f32])> = ps.iter().zip(ys.iter_mut()).map(|(p, y)| (&p.w, p.n_out, &mut y[..])).collect();
        linear_multi(&mut jobs, n_in, x, rows);
    }
    for (p, y) in ps.iter().zip(ys.iter_mut()) {
        p.add_bias(y);
    }
}

/// RMSNorm over each `hd`-wide head of `x`.
fn head_norm(x: &mut [f32], w: &[u16], hd: usize, eps: f32) {
    crate::kernels::rmsnorm_rows_inplace(x, hd, Some(w), eps);
}

/// Attention of `t` new tokens (`q` `[t, nh*hd]`, rotated; new keys / values `[t, nkv*hd]`)
/// over the first `n_ctx` cached positions (`kc[g]`, `vc[g]`: `[len, hd]` per KV head) and the
/// new tokens. `causal`: new token i sees the context and new tokens 0..=i.
#[allow(clippy::too_many_arguments)]
fn attend(
    q: &[f32],
    kc: &[Vec<f32>],
    vc: &[Vec<f32>],
    n_ctx: usize,
    kn: &[f32],
    vn: &[f32],
    t: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    causal: bool,
) -> Vec<f32> {
    let _timer = crate::prof::start(crate::prof::Cat::Attention);
    let rep = nh / nkv;
    let tk = n_ctx + t;
    let mut out = vec![0f32; t * nh * hd];
    // Each KV head's keys and values as one [tk, hd] matrix: the context prefix, then the block.
    let gather = |src: &[Vec<f32>], new: &[f32], g: usize| -> Vec<f32> {
        let mut m = Vec::with_capacity(tk * hd);
        m.extend_from_slice(&src[g][..n_ctx * hd]);
        for i in 0..t {
            m.extend_from_slice(&new[(i * nkv + g) * hd..(i * nkv + g + 1) * hd]);
        }
        m
    };
    let scale = 1.0 / (hd as f32).sqrt();
    if causal || t * rep < 64 {
        // Few queries (text, decoding): one task per (token, head).
        let ks: Vec<Vec<f32>> = (0..nkv).map(|g| gather(kc, kn, g)).collect();
        let vs: Vec<Vec<f32>> = (0..nkv).map(|g| gather(vc, vn, g)).collect();
        let op = SendPtr(out.as_mut_ptr());
        let run = |task: usize| {
            let (i, hh) = (task / nh, task % nh);
            let g = hh / rep;
            let n = if causal { n_ctx + i + 1 } else { tk };
            let qv = &q[task * hd..(task + 1) * hd];
            let kg = &ks[g];
            let mut s: Vec<f32> = (0..n).map(|j| dot(qv, &kg[j * hd..(j + 1) * hd]) * scale).collect();
            crate::kernels::softmax_inplace(&mut s);
            // SAFETY: each task writes its own [hd] slice of `out`.
            let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(task * hd), hd) };
            let vg = &vs[g];
            for (j, &p) in s.iter().enumerate() {
                axpy(o, p, &vg[j * hd..(j + 1) * hd]);
            }
        };
        let ntasks = t * nh;
        if ntasks > 1 && tk * ntasks > 4096 {
            crate::pool::run(ntasks, &run);
        } else {
            (0..ntasks).for_each(run);
        }
        return out;
    }
    // Bidirectional blocks (images): per KV head, its `rep` query heads of every token are rows of
    // one single-head attention (row order is free without a mask), in query chunks that bound
    // the CPU path's [rows, keys] score matrix (GPU flash attention needs no such bound, and
    // every chunk would upload the keys and values again).
    let rows = t * rep;
    let chunk = if crate::gpu::active().is_some() {
        rows.min(1 << 16)
    } else {
        ((1usize << 24) / tk).clamp(64, 16384).min(rows)
    };
    for g in 0..nkv {
        let kg = gather(kc, kn, g);
        let vg = gather(vc, vn, g);
        // Rows (token i, head g*rep + r) are contiguous in `q` for a fixed i.
        let mut qg = vec![0f32; rows * hd];
        for_chunks(&mut qg, rep * hd, t > 64, |i, r| {
            r.copy_from_slice(&q[(i * nh + g * rep) * hd..(i * nh + (g + 1) * rep) * hd])
        });
        let mut og = vec![0f32; rows * hd];
        for r0 in (0..rows).step_by(chunk) {
            let n = chunk.min(rows - r0);
            let o = crate::sd::ops::attention(&qg[r0 * hd..(r0 + n) * hd], &kg, &vg, n, tk, 1, hd, false);
            og[r0 * hd..(r0 + n) * hd].copy_from_slice(&o);
        }
        let op = SendPtr(out.as_mut_ptr());
        let og = &og;
        let copy = |i: usize| {
            // SAFETY: token i's heads of group g are a disjoint slice of `out`.
            let dst = unsafe { std::slice::from_raw_parts_mut(op.get().add((i * nh + g * rep) * hd), rep * hd) };
            dst.copy_from_slice(&og[i * rep * hd..(i + 1) * rep * hd]);
        };
        if t > 256 {
            crate::pool::run(t, &copy);
        } else {
            (0..t).for_each(copy);
        }
    }
    out
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
