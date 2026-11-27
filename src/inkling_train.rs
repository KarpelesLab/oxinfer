//! LoRA training for Inkling-2 (`inkling2`, `inkling2_loop`): the text forward with low-rank
//! updates on the attention projections, and its backward pass, giving the gradient of a
//! token-level loss with respect to every `A` and `B` ([`Inkling2Model::train_grads`]).
//!
//! - **What is trained.** PEFT's LoRA, `W x + (alpha / r) B A x`, on any of the seven attention
//!   projections of a layer ([`Proj`]; `wo_ud`, the output projection, by default). A looped
//!   layer's update is one pair used in every pass: its gradient is the sum over the passes.
//!   Everything else is frozen.
//! - **Precision.** float32 activations and gradients on the CPU. The frozen weights are used as
//!   loaded, at their stored values: the forward products through the f32 GEMM (which decodes
//!   bf16, NVFP4, MXFP8 and GGUF blocks to f32), the transposed products of the backward pass on
//!   rows dequantized to f32 a block at a time ([`lin_t`]); no weight is dequantized whole, and
//!   only the experts a sequence routes to are read.
//! - **Routing.** The top-k choice of experts is taken from the forward pass and held fixed
//!   (it has no gradient); the routing weights are the renormalized sigmoid scores of the chosen
//!   experts, and gradients flow through them into the router's input, as in PyTorch autograd
//!   on transformers' model.
//! - **Memory.** The forward pass keeps the stream entering each layer application (and the
//!   routing): `(steps + 1) * hidden` floats per token. The backward pass runs each application
//!   again from its saved input to rebuild what its gradient reads, one application at a time.
//!   [`TrainOptions::grad_window`] truncates the gradient to the positions from that many
//!   tokens before the first target on (the prompt before still runs forward in full).
//!
//! The whole sequence is one pass from position 0 (no session, no cache). Images are not
//! handled.

use super::*;
use crate::moe::GateUp;
use std::path::Path;

/// An attention projection that can take a low-rank update.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Proj {
    Q,
    K,
    V,
    R,
    Gate,
    Sink,
    O,
}

impl Proj {
    pub const ALL: [Proj; 7] = [Proj::Q, Proj::K, Proj::V, Proj::R, Proj::Gate, Proj::Sink, Proj::O];

    /// The module's name in the release (what `--lora` matches after `layers.N.attn.`).
    pub fn name(self) -> &'static str {
        match self {
            Proj::Q => "wq_du",
            Proj::K => "wk_dv",
            Proj::V => "wv_dv",
            Proj::R => "wr_du",
            Proj::Gate => "gate_attn",
            Proj::Sink => "attn_sink_proj",
            Proj::O => "wo_ud",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }

    /// `(n_out, n_in)` of this projection in layer `l`.
    pub fn shape(self, c: &Inkling2Config, l: usize) -> (usize, usize) {
        let hd = c.heads(l);
        let (w, kw, h) = (hd.heads * hd.head_dim, hd.kv_heads * hd.head_dim, c.hidden_size);
        match self {
            Proj::Q | Proj::Gate => (w, h),
            Proj::K | Proj::V => (kw, h),
            Proj::R => (hd.heads * c.d_rel, h),
            Proj::Sink => (hd.heads, h),
            Proj::O => (h, w),
        }
    }
}

/// One low-rank pair: `a` is `[rank, n_in]`, `b` `[n_out, rank]` (PEFT's `lora_A` / `lora_B`).
#[derive(Clone, Debug)]
pub struct Site {
    pub layer: usize,
    pub proj: Proj,
    pub n_out: usize,
    pub n_in: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

/// A LoRA adapter being trained: the update of a site is `(alpha / rank) B A x`.
#[derive(Clone, Debug)]
pub struct Adapter {
    pub rank: usize,
    pub alpha: f32,
    pub sites: Vec<Site>,
}

/// Gradients of the loss with respect to each site's `a` and `b` (same order and shapes).
#[derive(Clone, Debug)]
pub struct Grads {
    pub a: Vec<Vec<f32>>,
    pub b: Vec<Vec<f32>>,
}

impl Adapter {
    /// A fresh adapter on projections `projs` of `layers` (all layers when empty): `A` uniform
    /// in `±1 / sqrt(n_in)` (PEFT's default, Kaiming with `a = sqrt(5)`), `B` zero.
    pub fn new(cfg: &Inkling2Config, rank: usize, alpha: f32, projs: &[Proj], layers: &[usize], seed: u64) -> Self {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        let mut sites = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            if !layers.is_empty() && !layers.contains(&l) {
                continue;
            }
            for &p in projs {
                let (n_out, n_in) = p.shape(cfg, l);
                let bound = 1.0 / (n_in as f32).sqrt();
                sites.push(Site {
                    layer: l,
                    proj: p,
                    n_out,
                    n_in,
                    a: (0..rank * n_in).map(|_| next() * bound).collect(),
                    b: vec![0f32; n_out * rank],
                });
            }
        }
        Self { rank, alpha, sites }
    }

    /// The factor on `B A x`.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    pub fn n_params(&self) -> usize {
        self.sites.iter().map(|s| s.a.len() + s.b.len()).sum()
    }

    /// Zero gradients of this adapter's shapes.
    pub fn zeros(&self) -> Grads {
        Grads {
            a: self.sites.iter().map(|s| vec![0f32; s.a.len()]).collect(),
            b: self.sites.iter().map(|s| vec![0f32; s.b.len()]).collect(),
        }
    }

    fn site(&self, layer: usize, proj: Proj) -> Option<usize> {
        self.sites.iter().position(|s| s.layer == layer && s.proj == proj)
    }

    /// Writes the adapter as a PEFT folder (`adapter_config.json`, `adapter_model.safetensors`
    /// with `base_model.model.model.llm.layers.N.attn.NAME.lora_A.weight` / `lora_B.weight`
    /// in f32), which `--lora` reads. `base` is recorded as `base_model_name_or_path`.
    pub fn save(&self, dir: &Path, base: &str) -> Result<()> {
        std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
        let mut targets: Vec<&str> = self.sites.iter().map(|s| s.proj.name()).collect();
        targets.sort_unstable();
        targets.dedup();
        let mut layers: Vec<usize> = self.sites.iter().map(|s| s.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        let cfg = serde_json::json!({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": base,
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "fan_in_fan_out": false,
            "use_rslora": false,
            "use_dora": false,
            "inference_mode": true,
            "target_modules": targets,
            "layers_to_transform": layers,
        });
        std::fs::write(dir.join("adapter_config.json"), serde_json::to_vec_pretty(&cfg)?)?;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let mut off = 0usize;
        let mut parts: Vec<&[f32]> = Vec::new();
        for s in &self.sites {
            let p = format!("base_model.model.model.llm.layers.{}.attn.{}", s.layer, s.proj.name());
            for (leaf, shape, v) in [("lora_A", [self.rank, s.n_in], &s.a), ("lora_B", [s.n_out, self.rank], &s.b)] {
                let n = v.len() * 4;
                header.insert(
                    format!("{p}.{leaf}.weight"),
                    serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + n]}),
                );
                off += n;
                parts.push(v);
            }
        }
        let mut h = serde_json::to_string(&Value::Object(header))?.into_bytes();
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut out = Vec::with_capacity(8 + h.len() + off);
        out.extend_from_slice(&(h.len() as u64).to_le_bytes());
        out.extend_from_slice(&h);
        // (The header lists the tensors in key order; the offsets say where each one is.)
        for v in parts {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        let path = dir.join("adapter_model.safetensors");
        std::fs::write(&path, out).with_context(|| path.display().to_string())
    }

    /// Reads back an adapter written by [`Adapter::save`] (or any PEFT LoRA folder whose modules
    /// are attention projections of this model under the release's names), to go on training it.
    pub fn load(cfg: &Inkling2Config, dir: &Path) -> Result<Self> {
        let raw: Value = serde_json::from_slice(&crate::fsx::read(&dir.join("adapter_config.json"))?)?;
        let rank = raw.get("r").and_then(Value::as_u64).context("adapter_config.json: missing r")? as usize;
        let alpha = raw.get("lora_alpha").and_then(Value::as_f64).unwrap_or(rank as f64) as f32;
        let st = crate::safetensors::SafeTensors::open(&dir.join("adapter_model.safetensors"), true)?;
        let f32s = |name: &str, n: usize| -> Result<Vec<f32>> {
            let info = st.info(name).with_context(|| format!("missing tensor {name}"))?;
            if info.dtype != "F32" || info.shape.iter().product::<usize>() != n {
                bail!("{name}: {} {:?}, expected {n} float32 values", info.dtype, info.shape);
            }
            Ok(st.bytes(name)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
        };
        let mut names: Vec<String> = st.names().filter(|n| n.ends_with(".lora_A.weight")).cloned().collect();
        names.sort();
        let mut sites = Vec::new();
        for an in names {
            let module = an.strip_suffix(".lora_A.weight").unwrap();
            let rest = module.rsplit_once("layers.").map(|x| x.1).unwrap_or("");
            let mut it = rest.splitn(3, '.');
            let (l, mid, leaf) = (it.next(), it.next(), it.next());
            let (Some(Ok(layer)), Some("attn"), Some(proj)) = (l.map(str::parse::<usize>), mid, leaf.and_then(Proj::parse)) else {
                bail!("{an}: not an attention projection this trainer knows");
            };
            if layer >= cfg.num_hidden_layers {
                bail!("{an}: the model has {} layers", cfg.num_hidden_layers);
            }
            let (n_out, n_in) = proj.shape(cfg, layer);
            sites.push(Site {
                layer,
                proj,
                n_out,
                n_in,
                a: f32s(&an, rank * n_in)?,
                b: f32s(&format!("{module}.lora_B.weight"), n_out * rank)?,
            });
        }
        sites.sort_by_key(|s| (s.layer, Proj::ALL.iter().position(|p| *p == s.proj)));
        Ok(Self { rank, alpha, sites })
    }
}

/// What [`Inkling2Model::train_grads`] may leave out.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrainOptions {
    /// Positions before the first target that still get gradient (None: all of them). Earlier
    /// positions run forward as they are and are constants of the backward pass.
    pub grad_window: Option<usize>,
}

/// The result of one sequence's forward and backward pass.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The sum of the targets' loss terms.
    pub loss: f64,
    /// The log-probability of each target token, in the order given.
    pub logp: Vec<f32>,
    pub grads: Grads,
    /// The first position that got gradient.
    pub grad_from: usize,
    /// Bytes the forward pass kept for the backward pass.
    pub tape_bytes: usize,
    pub forward_secs: f64,
    pub backward_secs: f64,
}

/// The default loss term of a target of weight `w` whose token has log-probability `logp`, and
/// its derivative with respect to `logp`: `-w logp` for `w >= 0` (the token is imitated), and
/// for `w < 0` the bounded `-|w| ln(1 - p)` (the token is pushed down; `1 - p` floored at 1e-4).
pub fn token_loss(w: f32, logp: f32) -> (f64, f64) {
    let (w, lp) = (w as f64, logp as f64);
    if w >= 0.0 {
        return (-w * lp, -w);
    }
    let p = lp.exp();
    let q = (1.0 - p).max(1e-4);
    (w * q.ln(), -w * p / q)
}

// ---- helpers ----------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct P(*mut f32);
unsafe impl Send for P {}
unsafe impl Sync for P {}
impl P {
    /// # Safety
    /// The range is in bounds of the buffer `P` points into, and no other live reference
    /// covers it (the tasks of a region take disjoint ranges).
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    unsafe fn s<'x>(self, at: usize, n: usize) -> &'x mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(at), n) }
    }
}

/// `f(i)` for `i` in `lo..hi`, over the pool.
fn par(lo: usize, hi: usize, f: impl Fn(usize) + Sync) {
    let n = hi.saturating_sub(lo);
    let tasks = n.min(crate::pool::threads() * 4);
    if tasks <= 1 {
        (lo..hi).for_each(f);
        return;
    }
    crate::pool::run(tasks, &|k| {
        for i in lo + n * k / tasks..lo + n * (k + 1) / tasks {
            f(i);
        }
    });
}

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

fn acc(v: &mut Option<Vec<f32>>, n: usize) -> &mut Vec<f32> {
    v.get_or_insert_with(|| vec![0f32; n])
}

/// Whether the weight is stored as floats (its matmuls are f32 at any number of rows).
fn float_stored(l: &Lin) -> bool {
    match &l.w {
        crate::model::Weight::Bf16(..) => true,
        crate::model::Weight::Gguf(m) => m.kind.is_float() && m.block_scales.is_none(),
        _ => false,
    }
}

/// Rows `r0..r1` of `l` as f32 (`[r1 - r0, n_in]`).
fn lin_rows(l: &Lin, r0: usize, r1: usize, n_in: usize, out: &mut [f32]) {
    use crate::model::Weight;
    match &l.w {
        Weight::Bf16(w, _) => bf16_to_f32_slice(&w[r0 * n_in..r1 * n_in], out),
        Weight::Q8(m) => crate::quant::dequantize_rows(m, r0, r1, out),
        Weight::Q4(m) => crate::quant::dequantize_rows_q4(m, r0, r1, out),
        Weight::Gguf(m) => m.dequantize_rows(r0, r1, out),
    }
}

/// Rows of a weight dequantized at a time by the transposed products (a multiple of 4: GGUF
/// matrices stored in groups of four rows decode whole groups).
const RB: usize = 16;

/// `y = W x` for `t` rows (`[t, n_in]` to `[t, n_out]`), frozen weight only, in f32: the f32 GEMM
/// for a pass long enough to take it (or float storage), else row by row on dequantized rows
/// (the decode kernels of a short pass round the activations to int8).
fn lin_fwd(l: &Lin, n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
    let mut y = vec![0f32; t * n_out];
    if t > decode_max_tokens() || float_stored(l) {
        linear(l, n_out, n_in, x, t, &mut y);
        return y;
    }
    let yp = P(y.as_mut_ptr());
    let blocks = n_out.div_ceil(RB);
    crate::pool::run(blocks, &|b| {
        let (r0, r1) = (b * RB, ((b + 1) * RB).min(n_out));
        let mut w = vec![0f32; (r1 - r0) * n_in];
        lin_rows(l, r0, r1, n_in, &mut w);
        for (r, wr) in (r0..r1).zip(w.chunks(n_in)) {
            for ti in 0..t {
                // SAFETY: column `r` of `y` belongs to this task.
                unsafe { yp.s(ti * n_out + r, 1)[0] = dot(wr, &x[ti * n_in..(ti + 1) * n_in]) };
            }
        }
    });
    y
}

/// `out[i] += sum_r g[i * gs + r] * W[r]` for rows `r0..r1` of `l` and `n` tokens, on the calling
/// thread (`g` is `[n, gs]`, `out` `[n, n_in]`).
#[allow(clippy::too_many_arguments)]
fn axpy_rows(l: &Lin, r0: usize, r1: usize, n_in: usize, g: &[f32], gs: usize, n: usize, out: &mut [f32], wb: &mut Vec<f32>) {
    wb.resize(RB * n_in, 0.0);
    let mut b = r0;
    while b < r1 {
        let nb = RB.min(r1 - b);
        lin_rows(l, b, b + nb, n_in, &mut wb[..nb * n_in]);
        for i in 0..n {
            let c = &g[i * gs + b..i * gs + b + nb];
            let o = &mut out[i * n_in..(i + 1) * n_in];
            let mut r = 0;
            while r + 4 <= nb {
                let (c0, c1, c2, c3) = (c[r], c[r + 1], c[r + 2], c[r + 3]);
                let w = &wb[r * n_in..(r + 4) * n_in];
                let (w0, rest) = w.split_at(n_in);
                let (w1, rest) = rest.split_at(n_in);
                let (w2, w3) = rest.split_at(n_in);
                for ((((o, &a0), &a1), &a2), &a3) in o.iter_mut().zip(w0).zip(w1).zip(w2).zip(w3) {
                    *o = fma(c3, a3, fma(c2, a2, fma(c1, a1, fma(c0, a0, *o))));
                }
                r += 4;
            }
            while r < nb {
                let cr = c[r];
                for (o, &a) in o.iter_mut().zip(&wb[r * n_in..(r + 1) * n_in]) {
                    *o = fma(cr, a, *o);
                }
                r += 1;
            }
        }
        b += nb;
    }
}

/// The transposed product of a frozen weight: `out += g W` for `g` `[n, rows]` (the gradient of
/// the first `rows` outputs) and `out` `[n, n_in]`, on rows of `W` dequantized to f32 a block at
/// a time. Many tokens: each task takes a range of tokens; few: a range of rows into an
/// accumulator of its own, added in task order.
fn lin_t(l: &Lin, rows: usize, n_in: usize, g: &[f32], n: usize, out: &mut [f32]) {
    if n == 0 || rows == 0 {
        return;
    }
    debug_assert!(g.len() == n * rows && out.len() == n * n_in);
    let th = crate::pool::threads();
    if n >= 8 * th || rows <= RB {
        let tasks = (n / 8).clamp(1, th * 2);
        let op = P(out.as_mut_ptr());
        crate::pool::run(tasks, &|k| {
            let (i0, i1) = (n * k / tasks, n * (k + 1) / tasks);
            let mut wb = Vec::new();
            // SAFETY: the tokens `i0..i1` belong to this task.
            let o = unsafe { op.s(i0 * n_in, (i1 - i0) * n_in) };
            axpy_rows(l, 0, rows, n_in, &g[i0 * rows..i1 * rows], rows, i1 - i0, o, &mut wb);
        });
        return;
    }
    let tasks = rows.div_ceil(RB).min(th);
    let per = rows.div_ceil(tasks).next_multiple_of(RB);
    let mut local = vec![0f32; tasks * n * n_in];
    let lp = P(local.as_mut_ptr());
    crate::pool::run(tasks, &|k| {
        let (r0, r1) = ((k * per).min(rows), ((k + 1) * per).min(rows));
        let mut wb = Vec::new();
        // SAFETY: accumulator `k` belongs to this task.
        let o = unsafe { lp.s(k * n * n_in, n * n_in) };
        axpy_rows(l, r0, r1, n_in, g, rows, n, o, &mut wb);
    });
    for part in local.chunks(n * n_in) {
        add_inplace(out, part);
    }
}

/// `rms_unit` of each row.
fn rms_rows(x: &[f32], h: usize, eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    for_chunks(&mut y, h, x.len() > 4 * h, |_, r| rms_unit(r, eps));
    y
}

/// The gradient of an RMS norm (`y = x rsqrt(mean(x^2) + eps) * w`): `gx += J^T gy` for the rows
/// from `lo` on.
fn norm_bwd(x: &[f32], w: Option<&[f32]>, h: usize, eps: f32, gy: &[f32], lo: usize, gx: &mut [f32]) {
    let t = x.len() / h;
    let gp = P(gx.as_mut_ptr());
    par(lo, t, |i| {
        let xr = &x[i * h..(i + 1) * h];
        let g = &gy[i * h..(i + 1) * h];
        // SAFETY: row `i` belongs to this task.
        let o = unsafe { gp.s(i * h, h) };
        let r = 1.0 / (dot(xr, xr) / h as f32 + eps).sqrt();
        let mut ug = 0f64;
        for j in 0..h {
            ug += (xr[j] * g[j] * w.map_or(1.0, |w| w[j])) as f64;
        }
        let c = (ug * r as f64 / h as f64) as f32;
        for j in 0..h {
            o[j] += r * (g[j] * w.map_or(1.0, |w| w[j]) - xr[j] * r * c);
        }
    });
}

/// The attention residual's mix ([`mix`]) over `items` (`[t, h]` each, RMS-normed): one output
/// per query.
fn mix_fwd(items: &[&[f32]], queries: &[f32], h: usize) -> Vec<Vec<f32>> {
    let nq = queries.len() / h;
    let t = items[0].len() / h;
    let scale = (h as f32).powf(-0.5);
    let mut outs: Vec<Vec<f32>> = (0..nq).map(|_| vec![0f32; t * h]).collect();
    let ps: Vec<P> = outs.iter_mut().map(|o| P(o.as_mut_ptr())).collect();
    par(0, t, |i| {
        let mut s = vec![0f32; items.len()];
        for (qi, q) in queries.chunks(h).enumerate() {
            for (v, it) in s.iter_mut().zip(items) {
                *v = dot(q, &it[i * h..(i + 1) * h]) * scale;
            }
            softmax_inplace(&mut s);
            // SAFETY: row `i` of each output belongs to this task.
            let o = unsafe { ps[qi].s(i * h, h) };
            for (&w, it) in s.iter().zip(items) {
                axpy(o, w, &it[i * h..(i + 1) * h]);
            }
        }
    });
    outs
}

/// The gradient of one query's mix: `gitems[j] += d out / d items[j]^T gout`.
fn mix_bwd(items: &[&[f32]], q: &[f32], h: usize, gout: &[f32], lo: usize, gitems: &[P]) {
    let t = items[0].len() / h;
    let scale = (h as f32).powf(-0.5);
    par(lo, t, |i| {
        let g = &gout[i * h..(i + 1) * h];
        let mut s: Vec<f32> = items.iter().map(|it| dot(q, &it[i * h..(i + 1) * h]) * scale).collect();
        softmax_inplace(&mut s);
        let c: Vec<f32> = items.iter().map(|it| dot(&it[i * h..(i + 1) * h], g)).collect();
        let m: f32 = s.iter().zip(&c).map(|(p, c)| p * c).sum();
        for (j, gp) in gitems.iter().enumerate() {
            // SAFETY: row `i` of each accumulator belongs to this task.
            let o = unsafe { gp.s(i * h, h) };
            axpy(o, s[j], g);
            axpy(o, s[j] * (c[j] - m) * scale, q);
        }
    });
}

/// A causal depthwise convolution with a skip over the whole sequence ([`short_conv`] from
/// position 0): `y[t] = (x[t] + sum_lag (taps[lag] + dyn[t][group][lag]) x[t - lag]) * norm`.
fn conv_fwd(taps: &[f32], dynamic: Option<(&[f32], usize)>, x: &[f32], ch: usize, norm: f32) -> Vec<f32> {
    let k = taps.len() / ch;
    let mut y = vec![0f32; x.len()];
    for_chunks(&mut y, ch, true, |ti, row| {
        row.copy_from_slice(&x[ti * ch..(ti + 1) * ch]);
        for lag in 0..k.min(ti + 1) {
            let src = &x[(ti - lag) * ch..(ti - lag + 1) * ch];
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
    y
}

/// The gradient of [`conv_fwd`]: `gx += J^T gy`, and with dynamic taps `gdyn += d y / d dyn^T gy`,
/// for the positions from `lo` on.
#[allow(clippy::too_many_arguments)]
fn conv_bwd(
    taps: &[f32],
    dynamic: Option<(&[f32], usize)>,
    x: &[f32],
    ch: usize,
    norm: f32,
    gy: &[f32],
    lo: usize,
    gx: &mut [f32],
    gdyn: Option<&mut [f32]>,
) {
    let k = taps.len() / ch;
    let t = x.len() / ch;
    let gp = P(gx.as_mut_ptr());
    par(lo, t, |s| {
        // SAFETY: row `s` belongs to this task.
        let o = unsafe { gp.s(s * ch, ch) };
        axpy(o, norm, &gy[s * ch..(s + 1) * ch]);
        for lag in 0..k.min(t - s) {
            let ti = s + lag;
            let g = &gy[ti * ch..(ti + 1) * ch];
            let w = &taps[lag * ch..(lag + 1) * ch];
            match dynamic {
                None => {
                    for ((o, &g), &wv) in o.iter_mut().zip(g).zip(w) {
                        *o += norm * g * wv;
                    }
                }
                Some((d, groups)) => {
                    let per = ch / groups;
                    for (gi, ((oc, gc), wc)) in o.chunks_mut(per).zip(g.chunks(per)).zip(w.chunks(per)).enumerate() {
                        let dw = d[(ti * groups + gi) * k + lag];
                        for ((o, &g), &wv) in oc.iter_mut().zip(gc).zip(wc) {
                            *o += norm * g * (wv + dw);
                        }
                    }
                }
            }
        }
    });
    if let (Some(gd), Some((_, groups))) = (gdyn, dynamic) {
        let per = ch / groups;
        let dp = P(gd.as_mut_ptr());
        par(lo, t, |ti| {
            // SAFETY: row `ti` belongs to this task.
            let o = unsafe { dp.s(ti * groups * k, groups * k) };
            for lag in 0..k.min(ti + 1) {
                let src = &x[(ti - lag) * ch..(ti - lag + 1) * ch];
                let g = &gy[ti * ch..(ti + 1) * ch];
                for gi in 0..groups {
                    o[gi * k + lag] += norm * dot(&g[gi * per..(gi + 1) * per], &src[gi * per..(gi + 1) * per]);
                }
            }
        });
    }
}

/// The gradient of [`l2_clip`] at `x` (before the clip): `J^T g` in place of `g`.
fn clip_bwd(x: &[f32], threshold: f32, g: &mut [f32]) {
    let n = dot(x, x).sqrt();
    let f = threshold / (n + CLIP_EPS);
    if f >= 1.0 || n == 0.0 {
        return;
    }
    let c = threshold * dot(x, g) / (n * (n + CLIP_EPS) * (n + CLIP_EPS));
    for (g, &x) in g.iter_mut().zip(x) {
        *g = f * *g - c * x;
    }
}

/// Derivatives of the experts' activation `act(g, u)` with respect to `g` and `u`.
#[inline]
fn act_grad(act: Act, g: f32, u: f32) -> (f32, f32) {
    match act {
        Act::Situ { beta, linear_beta } => {
            let (uc, du) = if linear_beta > 0.0 {
                let th = (u / linear_beta).tanh();
                (linear_beta * th, 1.0 - th * th)
            } else {
                (u, 1.0)
            };
            let th = (g / beta).tanh();
            let sg = 1.0 / (1.0 + (-g).exp());
            let f = beta * th * sg;
            (((1.0 - th * th) * sg + beta * th * sg * (1.0 - sg)) * uc, f * du)
        }
        _ => unreachable!("Inkling-2's experts are SiTU"),
    }
}

// ---- LoRA sites -------------------------------------------------------------------------------

/// `y += s B (A x)`; returns `A x` (`[t, rank]`).
fn lora_fwd(site: &Site, rank: usize, s: f32, x: &[f32], t: usize, y: &mut [f32]) -> Vec<f32> {
    let (n_in, n_out) = (site.n_in, site.n_out);
    let mut bt = vec![0f32; rank * n_out];
    for o in 0..n_out {
        for k in 0..rank {
            bt[k * n_out + o] = site.b[o * rank + k];
        }
    }
    let mut u = vec![0f32; t * rank];
    let (up, yp) = (P(u.as_mut_ptr()), P(y.as_mut_ptr()));
    par(0, t, |i| {
        // SAFETY: row `i` of `u` and of `y` belong to this task.
        let (ur, yr) = unsafe { (up.s(i * rank, rank), yp.s(i * n_out, n_out)) };
        let xr = &x[i * n_in..(i + 1) * n_in];
        for k in 0..rank {
            ur[k] = dot(&site.a[k * n_in..(k + 1) * n_in], xr);
            axpy(yr, s * ur[k], &bt[k * n_out..(k + 1) * n_out]);
        }
    });
    u
}

/// The gradient of a site's update: adds to `ga` / `gb` and `gx += s A^T B^T gy`, for the rows
/// from `lo` on.
#[allow(clippy::too_many_arguments)]
fn lora_bwd(
    site: &Site,
    rank: usize,
    s: f32,
    x: &[f32],
    u: &[f32],
    gy: &[f32],
    lo: usize,
    t: usize,
    gx: &mut [f32],
    ga: &mut [f32],
    gb: &mut [f32],
) {
    let (n_in, n_out) = (site.n_in, site.n_out);
    let mut bt = vec![0f32; rank * n_out];
    for o in 0..n_out {
        for k in 0..rank {
            bt[k * n_out + o] = site.b[o * rank + k];
        }
    }
    // gu = s B^T gy
    let mut gu = vec![0f32; t * rank];
    let (gup, gxp) = (P(gu.as_mut_ptr()), P(gx.as_mut_ptr()));
    par(lo, t, |i| {
        // SAFETY: row `i` of `gu` and of `gx` belong to this task.
        let (gr, xr) = unsafe { (gup.s(i * rank, rank), gxp.s(i * n_in, n_in)) };
        let g = &gy[i * n_out..(i + 1) * n_out];
        for k in 0..rank {
            gr[k] = s * dot(g, &bt[k * n_out..(k + 1) * n_out]);
            axpy(xr, gr[k], &site.a[k * n_in..(k + 1) * n_in]);
        }
    });
    let (gap, gbp) = (P(ga.as_mut_ptr()), P(gb.as_mut_ptr()));
    crate::pool::run(rank, &|k| {
        // SAFETY: row `k` of `ga` and column `k` of `gb` belong to this task.
        let ar = unsafe { gap.s(k * n_in, n_in) };
        let mut col = vec![0f32; n_out];
        for i in lo..t {
            axpy(ar, gu[i * rank + k], &x[i * n_in..(i + 1) * n_in]);
            axpy(&mut col, s * u[i * rank + k], &gy[i * n_out..(i + 1) * n_out]);
        }
        for (o, c) in col.into_iter().enumerate() {
            unsafe { gbp.s(o * rank + k, 1)[0] += c };
        }
    });
}

// ---- the unrolled pass ------------------------------------------------------------------------

/// One layer application of the unrolled pass, with what it reads and writes: streams (`[t, H]`
/// each, numbered) and block states (the RMS-normed stream entering a block's first layer).
struct PStep {
    step: Step,
    /// The stream entering the step, and the one leaving it.
    x_in: usize,
    x_out: usize,
    /// The block states the attention's mixes read.
    before: Vec<usize>,
    /// The state this step makes of its input (a block's first layer).
    push: Option<usize>,
    /// In the passes after the first: the previous pass's block states after its first, and its
    /// output stream.
    bank: Option<(Vec<usize>, usize)>,
}

struct Plan {
    steps: Vec<PStep>,
    n_streams: usize,
    /// The stream each state is the norm of.
    state_src: Vec<usize>,
    final_states: Vec<usize>,
    final_stream: usize,
}

impl Plan {
    fn new(c: &Inkling2Config, boundary: impl Fn(usize) -> bool) -> Self {
        let mut steps = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        let mut state_src = Vec::new();
        let (mut cur, mut n_streams) = (0usize, 1usize);
        let mut pre = (0usize, 0usize);
        let mut bank: Option<(Vec<usize>, usize)> = None;
        for step in c.steps() {
            let l = step.layer;
            if let Some(lc) = c.loop_cfg
                && l == lc.start
            {
                if step.pass == 0 {
                    pre = (cur, stack.len());
                } else {
                    let prev = stack.split_off(pre.1 + 1);
                    stack.truncate(pre.1);
                    bank = Some((prev, cur));
                    cur = pre.0;
                }
            }
            let before = stack.clone();
            let push = boundary(l).then(|| {
                state_src.push(cur);
                stack.push(state_src.len() - 1);
                state_src.len() - 1
            });
            let x_out = n_streams;
            n_streams += 1;
            steps.push(PStep {
                step,
                x_in: cur,
                x_out,
                before,
                push,
                bank: if step.pass > 0 && c.looped(l) { bank.clone() } else { None },
            });
            cur = x_out;
        }
        Self {
            steps,
            n_streams,
            state_src,
            final_states: stack,
            final_stream: cur,
        }
    }
}

/// What the forward pass keeps: every stream, and the experts each step routed to.
struct Tape {
    streams: Vec<Vec<f32>>,
    routes: Vec<Vec<u32>>,
}

struct GateTape {
    cat: Vec<f32>,
    tf: Vec<f32>,
    tg: Vec<f32>,
}

#[derive(Default)]
struct AttnTape {
    q0: Vec<f32>,
    r0: Vec<f32>,
    gate: Vec<f32>,
    sink: Vec<f32>,
    k0: Vec<f32>,
    v0: Vec<f32>,
    kd: Vec<f32>,
    vd: Vec<f32>,
    k1: Vec<f32>,
    v1: Vec<f32>,
    q: Vec<f32>,
    r: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    lse: Vec<f32>,
    y: Vec<f32>,
    /// `A x` of each adapted projection ([`Proj::ALL`]'s order).
    u: [Option<Vec<f32>>; 7],
}

struct MoeTape {
    logits: Vec<f32>,
    /// `[t, k]`: the chosen experts and their rows in the gathered layout.
    routes: Vec<u32>,
    row: Vec<u32>,
    segs: Vec<(usize, usize)>,
    count: Vec<usize>,
    /// Per gathered row: its token (`u32::MAX`: padding) and routing weight.
    src: Vec<u32>,
    roww: Vec<f32>,
    gu: Vec<f32>,
    o: Vec<f32>,
}

struct StepTape {
    xn_in: Option<Vec<f32>>,
    bank_xn: Option<Vec<f32>>,
    msgs: Option<Vec<Vec<f32>>>,
    mixed: Option<Vec<Vec<f32>>>,
    /// The gated attention inputs (with messages), before their norms.
    gated: Vec<Vec<f32>>,
    gates: Vec<Option<GateTape>>,
    xn: Vec<Vec<f32>>,
    at: AttnTape,
    a_raw: Vec<f32>,
    x_mid: Vec<f32>,
    xn_mid: Vec<f32>,
    hn0: Vec<f32>,
    hn1: Option<Vec<f32>>,
    moe: MoeTape,
    o_raw: Vec<f32>,
}

impl<'a> Inkling2Model<'a> {
    fn check_trainable(&self) -> Result<()> {
        let k = prefill_kernel();
        if !matches!(k, PrefillGemm::F32 | PrefillGemm::Sme) {
            bail!("training needs the exact prefill GEMM (OXINFER_PREFILL_GEMM=f32), not {k:?}");
        }
        if crate::gpu::active().is_some() {
            bail!("Inkling-2 training runs on the CPU: start without an accelerator backend");
        }
        for l in &self.layers {
            let a = &l.attn;
            let mut lins = vec![&a.q, &a.k, &a.v, &a.r, &a.gate, &a.sink, &a.o];
            lins.extend(l.experts.lins().into_iter().flatten());
            if lins.iter().any(|l| l.fold.is_some() || l.has_lora()) {
                bail!("training needs a model without Hadamard-folded weights and without attached adapters");
            }
            if !matches!(l.experts.act, Act::Situ { .. }) {
                bail!("training needs SiTU experts");
            }
        }
        Ok(())
    }

    fn tau(&self, a: &Attn, i: usize) -> f32 {
        if a.window.is_some() {
            return 1.0;
        }
        let n = ((i + 1) as f64 / self.cfg.log_scaling_n_floor).max(1.0);
        (1.0 + self.cfg.log_scaling_alpha * n.ln()) as f32
    }

    /// A frozen projection plus its low-rank update, if the adapter has one there.
    #[allow(clippy::too_many_arguments)]
    fn proj_fwd(&self, ad: Option<&Adapter>, l: usize, p: Proj, lin: &Lin, x: &[f32], t: usize, u: &mut [Option<Vec<f32>>; 7]) -> Vec<f32> {
        let (n_out, n_in) = p.shape(&self.cfg, l);
        let mut y = lin_fwd(lin, n_out, n_in, x, t);
        if let Some(ad) = ad
            && let Some(si) = ad.site(l, p)
        {
            u[p as usize] = Some(lora_fwd(&ad.sites[si], ad.rank, ad.scale(), x, t, &mut y));
        }
        y
    }

    /// `gx += W^T gy` (rows from `lo` on) and the gradients of the site's pair.
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(
        &self,
        ad: &Adapter,
        l: usize,
        p: Proj,
        lin: &Lin,
        x: &[f32],
        u: &[Option<Vec<f32>>; 7],
        gy: &[f32],
        lo: usize,
        t: usize,
        gx: &mut [f32],
        grads: &mut Grads,
    ) {
        let (n_out, n_in) = p.shape(&self.cfg, l);
        lin_t(lin, n_out, n_in, &gy[lo * n_out..], t - lo, &mut gx[lo * n_in..]);
        if let (Some(si), Some(u)) = (ad.site(l, p), &u[p as usize]) {
            lora_bwd(
                &ad.sites[si],
                ad.rank,
                ad.scale(),
                x,
                u,
                gy,
                lo,
                t,
                gx,
                &mut grads.a[si],
                &mut grads.b[si],
            );
        }
    }

    fn gate_fwd(&self, g: &Gate, x: &[f32], m: &[f32], t: usize) -> (Vec<f32>, GateTape) {
        let h = self.cfg.hidden_size;
        let r = self.cfg.loop_cfg.map_or(0, |c| c.gate_rank);
        let eps = self.cfg.rms_norm_eps as f32;
        let mut cat = vec![0f32; t * 2 * h];
        for_chunks(&mut cat, 2 * h, t > 4, |ti, row| {
            row[..h].copy_from_slice(&x[ti * h..(ti + 1) * h]);
            row[h..].copy_from_slice(&m[ti * h..(ti + 1) * h]);
            rms_unit(&mut row[..h], eps);
            rms_unit(&mut row[h..], eps);
        });
        let d = lin_fwd(&g.down, r, 2 * h, &cat, t);
        let mut tf = lin_fwd(&g.forget, h, r, &d, t);
        let mut tg = lin_fwd(&g.gain, h, r, &d, t);
        let mut out = vec![0f32; t * h];
        let (fp, gp) = (P(tf.as_mut_ptr()), P(tg.as_mut_ptr()));
        for_chunks(&mut out, h, t > 4, |ti, o| {
            // SAFETY: row `ti` of each belongs to this task.
            let (f, gn) = unsafe { (fp.s(ti * h, h), gp.s(ti * h, h)) };
            for j in 0..h {
                f[j] = f[j].tanh();
                gn[j] = gn[j].tanh();
                o[j] = (1.0 - f[j]) * x[ti * h + j] + gn[j] * m[ti * h + j];
            }
        });
        (out, GateTape { cat, tf, tg })
    }

    #[allow(clippy::too_many_arguments)]
    fn gate_bwd(&self, g: &Gate, x: &[f32], m: &[f32], tp: &GateTape, gout: &[f32], lo: usize, t: usize, gx: &mut [f32], gm: &mut [f32]) {
        let h = self.cfg.hidden_size;
        let r = self.cfg.loop_cfg.map_or(0, |c| c.gate_rank);
        let eps = self.cfg.rms_norm_eps as f32;
        let mut gf = vec![0f32; t * h];
        let mut gg = vec![0f32; t * h];
        let (gfp, ggp, gxp, gmp) = (P(gf.as_mut_ptr()), P(gg.as_mut_ptr()), P(gx.as_mut_ptr()), P(gm.as_mut_ptr()));
        par(lo, t, |i| {
            // SAFETY: row `i` of each belongs to this task.
            let (f, gn, xo, mo) = unsafe { (gfp.s(i * h, h), ggp.s(i * h, h), gxp.s(i * h, h), gmp.s(i * h, h)) };
            for j in 0..h {
                let at = i * h + j;
                let (tf, tg, go) = (tp.tf[at], tp.tg[at], gout[at]);
                f[j] = -(1.0 - tf * tf) * x[at] * go;
                gn[j] = (1.0 - tg * tg) * m[at] * go;
                xo[j] += (1.0 - tf) * go;
                mo[j] += tg * go;
            }
        });
        let n = t - lo;
        let mut gd = vec![0f32; n * r];
        lin_t(&g.forget, h, r, &gf[lo * h..], n, &mut gd);
        lin_t(&g.gain, h, r, &gg[lo * h..], n, &mut gd);
        let mut gcat = vec![0f32; n * 2 * h];
        lin_t(&g.down, r, 2 * h, &gd, n, &mut gcat);
        // The two halves of `cat` are the RMS-normed `x` and `m`.
        for (src, gp, off) in [(x, gxp, 0), (m, gmp, h)] {
            par(lo, t, |i| {
                let xr = &src[i * h..(i + 1) * h];
                let gc = &gcat[(i - lo) * 2 * h + off..(i - lo) * 2 * h + off + h];
                let u = &tp.cat[i * 2 * h + off..i * 2 * h + off + h];
                // SAFETY: row `i` belongs to this task.
                let o = unsafe { gp.s(i * h, h) };
                let rr = 1.0 / (dot(xr, xr) / h as f32 + eps).sqrt();
                let c = dot(u, gc) / h as f32;
                for j in 0..h {
                    o[j] += rr * (gc[j] - u[j] * c);
                }
            });
        }
    }

    /// Attention over the whole sequence ([`Inkling2Model::attention`] from position 0, with the
    /// adapter's updates), keeping what its gradient reads.
    fn attn_fwd(&self, ad: Option<&Adapter>, l: usize, a: &Attn, xs: [&[f32]; 3], t: usize) -> (Vec<f32>, AttnTape) {
        let c = &self.cfg;
        let (h, dr, kn) = (c.hidden_size, c.d_rel, c.conv_kernel_size);
        let Heads {
            heads: nh,
            kv_heads: nkv,
            head_dim: d,
        } = a.hd;
        let (w, kw) = (nh * d, nkv * d);
        let [xq, xk, xv] = xs;
        let mut tp = AttnTape::default();
        tp.q0 = self.proj_fwd(ad, l, Proj::Q, &a.q, xq, t, &mut tp.u);
        tp.r0 = self.proj_fwd(ad, l, Proj::R, &a.r, xq, t, &mut tp.u);
        tp.gate = self.proj_fwd(ad, l, Proj::Gate, &a.gate, xq, t, &mut tp.u);
        tp.sink = self.proj_fwd(ad, l, Proj::Sink, &a.sink, xq, t, &mut tp.u);
        tp.k0 = self.proj_fwd(ad, l, Proj::K, &a.k, xk, t, &mut tp.u);
        tp.v0 = self.proj_fwd(ad, l, Proj::V, &a.v, xv, t, &mut tp.u);
        for (conv, x, src, dy, out) in [
            (&a.k_conv, xk, &tp.k0, &mut tp.kd, &mut tp.k1),
            (&a.v_conv, xv, &tp.v0, &mut tp.vd, &mut tp.v1),
        ] {
            match conv {
                KvConv::Static(taps) => *out = conv_fwd(taps, None, src, kw, 1.0 / ((kn + 1) as f32).sqrt()),
                KvConv::Dynamic { taps, proj } => {
                    *dy = lin_fwd(proj, nkv * kn, h, x, t);
                    *out = conv_fwd(taps, Some((dy, nkv)), src, kw, 1.0 / ((2 * kn + 1) as f32).sqrt());
                }
            }
        }
        let (qk_clip, rel_clip) = (c.qk_l2_clip_threshold as f32, c.rel_l2_clip_threshold as f32);
        tp.q = tp.q0.clone();
        for_chunks(&mut tp.q, d, t > 1, |i, qh| {
            let hh = i % nh;
            for (x, &s) in qh.iter_mut().zip(&a.q_scale[hh * d..(hh + 1) * d]) {
                *x *= s;
            }
            l2_clip(qh, qk_clip);
            let f = self.tau(a, i / nh);
            if f != 1.0 {
                scale_inplace(qh, f);
            }
        });
        tp.r = tp.r0.clone();
        for_chunks(&mut tp.r, dr, t > 1, |i, rh| {
            let hh = i % nh;
            for (x, &s) in rh.iter_mut().zip(&a.r_scale[hh * dr..(hh + 1) * dr]) {
                *x *= s;
            }
            l2_clip(rh, rel_clip);
            let f = self.tau(a, i / nh);
            if f != 1.0 {
                scale_inplace(rh, f);
            }
        });
        tp.k = tp.k1.clone();
        for_chunks(&mut tp.k, d, t > 1, |_, kh| l2_clip(kh, qk_clip));
        let eps = c.rms_norm_eps as f32;
        tp.v = tp.v1.clone();
        for_chunks(&mut tp.v, d, t > 1, |_, vh| rms_unit(vh, eps));

        // One softmax per (query, head) over its keys and the sink.
        let groups = nh / nkv;
        let scaling = (d as f32).powf(-0.5);
        let extent = a.extent;
        tp.att = vec![0f32; t * w];
        tp.lse = vec![0f32; t * nh];
        let (ap, lp) = (P(tp.att.as_mut_ptr()), P(tp.lse.as_mut_ptr()));
        let (q, r, k, v, sink) = (&tp.q, &tp.r, &tp.k, &tp.v, &tp.sink);
        par(0, t, |i| {
            let lo = a.window.map_or(0, |win| (i + 1).saturating_sub(win));
            let n = i + 1 - lo;
            let nb = extent.min(n);
            let mut s = vec![0f32; n];
            let mut b = vec![0f32; nb];
            for hh in 0..nh {
                let g = hh / groups;
                dots(&q[i * w + hh * d..i * w + (hh + 1) * d], &k[lo * kw + g * d..], kw, &mut s);
                b.fill(0.0);
                for (&rv, prof) in r[(i * nh + hh) * dr..(i * nh + hh + 1) * dr].iter().zip(a.rel.chunks(extent)) {
                    axpy(&mut b, rv, &prof[..nb]);
                }
                let sk = sink[i * nh + hh];
                let mut m = sk;
                for (jj, x) in s.iter_mut().enumerate() {
                    let dist = n - 1 - jj;
                    *x = *x * scaling + if dist < nb { b[dist] } else { 0.0 };
                    m = m.max(*x);
                }
                let mut z = (sk - m).exp();
                for x in s.iter_mut() {
                    *x = (*x - m).exp();
                    z += *x;
                }
                scale_inplace(&mut s, 1.0 / z);
                // SAFETY: this query's head slice and log-sum belong to this task.
                let (o, ls) = unsafe { (ap.s(i * w + hh * d, d), lp.s(i * nh + hh, 1)) };
                weighted_values(&s, &v[lo * kw + g * d..], kw, d, o);
                ls[0] = m + z.ln();
            }
        });
        tp.y = tp.att.clone();
        for (x, &g) in tp.y.iter_mut().zip(tp.gate.iter()) {
            *x *= sigmoid(g);
        }
        let out = self.proj_fwd(ad, l, Proj::O, &a.o, &tp.y, t, &mut tp.u);
        (out, tp)
    }

    /// The gradient of [`Self::attn_fwd`]: the gradients of its three inputs (rows from `lo`
    /// on), and of the adapter's pairs into `grads`.
    #[allow(clippy::too_many_arguments)]
    fn attn_bwd(
        &self,
        ad: &Adapter,
        l: usize,
        a: &Attn,
        xs: [&[f32]; 3],
        tp: &AttnTape,
        gout: &[f32],
        lo: usize,
        t: usize,
        grads: &mut Grads,
    ) -> [Vec<f32>; 3] {
        let c = &self.cfg;
        let (h, dr, kn) = (c.hidden_size, c.d_rel, c.conv_kernel_size);
        let Heads {
            heads: nh,
            kv_heads: nkv,
            head_dim: d,
        } = a.hd;
        let (w, kw) = (nh * d, nkv * d);
        let [xq, xk, xv] = xs;
        let n = t - lo;
        let mut gy = vec![0f32; t * w];
        self.proj_bwd(ad, l, Proj::O, &a.o, &tp.y, &tp.u, gout, lo, t, &mut gy, grads);
        // y = att * sigmoid(gate)
        let mut g_att = vec![0f32; t * w];
        let mut g_gate = vec![0f32; t * w];
        {
            let (ap, gp) = (P(g_att.as_mut_ptr()), P(g_gate.as_mut_ptr()));
            par(lo, t, |i| {
                // SAFETY: row `i` of each belongs to this task.
                let (ga, gg) = unsafe { (ap.s(i * w, w), gp.s(i * w, w)) };
                for j in 0..w {
                    let at = i * w + j;
                    let sg = sigmoid(tp.gate[at]);
                    ga[j] = gy[at] * sg;
                    gg[j] = gy[at] * tp.att[at] * sg * (1.0 - sg);
                }
            });
        }
        drop(gy);
        let groups = nh / nkv;
        let scaling = (d as f32).powf(-0.5);
        let extent = a.extent;
        let (q, r, k, v) = (&tp.q, &tp.r, &tp.k, &tp.v);
        // The queries' side: gradients of q, r and the sink; `mean` is `g_att . att`.
        let mut gq = vec![0f32; t * w];
        let mut gr = vec![0f32; t * nh * dr];
        let mut g_sink = vec![0f32; t * nh];
        let mut mean = vec![0f32; t * nh];
        {
            let (qp, rp, sp, mp) = (P(gq.as_mut_ptr()), P(gr.as_mut_ptr()), P(g_sink.as_mut_ptr()), P(mean.as_mut_ptr()));
            let g_att = &g_att;
            par(lo, t, |i| {
                let klo = a.window.map_or(0, |win| (i + 1).saturating_sub(win));
                let nk = i + 1 - klo;
                let nb = extent.min(nk);
                let mut s = vec![0f32; nk];
                let mut cv = vec![0f32; nk];
                let mut b = vec![0f32; nb];
                let mut rev = vec![0f32; nb];
                for hh in 0..nh {
                    let g = hh / groups;
                    let ga = &g_att[i * w + hh * d..i * w + (hh + 1) * d];
                    let ls = tp.lse[i * nh + hh];
                    dots(&q[i * w + hh * d..i * w + (hh + 1) * d], &k[klo * kw + g * d..], kw, &mut s);
                    b.fill(0.0);
                    for (&rv, prof) in r[(i * nh + hh) * dr..(i * nh + hh + 1) * dr].iter().zip(a.rel.chunks(extent)) {
                        axpy(&mut b, rv, &prof[..nb]);
                    }
                    dots(ga, &v[klo * kw + g * d..], kw, &mut cv);
                    let m = dot(ga, &tp.att[i * w + hh * d..i * w + (hh + 1) * d]);
                    for (jj, x) in s.iter_mut().enumerate() {
                        let dist = nk - 1 - jj;
                        let p = (*x * scaling + if dist < nb { b[dist] } else { 0.0 } - ls).exp();
                        *x = p * (cv[jj] - m);
                        if dist < nb {
                            rev[dist] = *x;
                        }
                    }
                    // SAFETY: this (query, head)'s slices belong to this task.
                    let (oq, or, os, om) = unsafe {
                        (
                            qp.s(i * w + hh * d, d),
                            rp.s((i * nh + hh) * dr, dr),
                            sp.s(i * nh + hh, 1),
                            mp.s(i * nh + hh, 1),
                        )
                    };
                    weighted_values(&s, &k[klo * kw + g * d..], kw, d, oq);
                    scale_inplace(oq, scaling);
                    for (o, prof) in or.iter_mut().zip(a.rel.chunks(extent)) {
                        *o = dot(&rev, &prof[..nb]);
                    }
                    os[0] = -(tp.sink[i * nh + hh] - ls).exp() * m;
                    om[0] = m;
                }
            });
        }
        // The keys' side: each key and value collects from the queries that see it.
        let mut gk = vec![0f32; t * kw];
        let mut gv = vec![0f32; t * kw];
        {
            let mut rel_t = vec![0f32; extent * dr];
            for dd in 0..dr {
                for e in 0..extent {
                    rel_t[e * dr + dd] = a.rel[dd * extent + e];
                }
            }
            let (kp, vp) = (P(gk.as_mut_ptr()), P(gv.as_mut_ptr()));
            let (g_att, mean, rel_t) = (&g_att, &mean, &rel_t);
            par(lo, t, |j| {
                let i1 = a.window.map_or(t, |win| t.min(j + win));
                let ni = i1 - j;
                let mut s = vec![0f32; ni];
                let mut cv = vec![0f32; ni];
                let mut tmp = vec![0f32; d];
                for g in 0..nkv {
                    let kj = &k[j * kw + g * d..j * kw + (g + 1) * d];
                    let vj = &v[j * kw + g * d..j * kw + (g + 1) * d];
                    // SAFETY: this key's head slices belong to this task.
                    let (ok, ov) = unsafe { (kp.s(j * kw + g * d, d), vp.s(j * kw + g * d, d)) };
                    for hh in g * groups..(g + 1) * groups {
                        dots(kj, &q[j * w + hh * d..], w, &mut s);
                        dots(vj, &g_att[j * w + hh * d..], w, &mut cv);
                        for (ii, x) in s.iter_mut().enumerate() {
                            let i = j + ii;
                            let bias = if ii < extent {
                                dot(&r[(i * nh + hh) * dr..(i * nh + hh + 1) * dr], &rel_t[ii * dr..(ii + 1) * dr])
                            } else {
                                0.0
                            };
                            let p = (*x * scaling + bias - tp.lse[i * nh + hh]).exp();
                            *x = p;
                            cv[ii] = p * (cv[ii] - mean[i * nh + hh]);
                        }
                        weighted_values(&cv, &q[j * w + hh * d..], w, d, &mut tmp);
                        axpy(ok, scaling, &tmp);
                        weighted_values(&s, &g_att[j * w + hh * d..], w, d, &mut tmp);
                        add_inplace(ov, &tmp);
                    }
                }
            });
        }
        drop((g_att, mean));
        // Back through the clips, the scales and the value norm.
        let (qk_clip, rel_clip) = (c.qk_l2_clip_threshold as f32, c.rel_l2_clip_threshold as f32);
        for (g, x0, scale, clip, dim) in [
            (&mut gq, &tp.q0, &a.q_scale, qk_clip, d),
            (&mut gr, &tp.r0, &a.r_scale, rel_clip, dr),
        ] {
            let gp = P(g.as_mut_ptr());
            par(lo * nh, t * nh, |i| {
                let hh = i % nh;
                // SAFETY: this (query, head)'s slice belongs to this task.
                let gh = unsafe { gp.s(i * dim, dim) };
                let sc = &scale[hh * dim..(hh + 1) * dim];
                let pre: Vec<f32> = x0[i * dim..(i + 1) * dim].iter().zip(sc).map(|(x, s)| x * s).collect();
                scale_inplace(gh, self.tau(a, i / nh));
                clip_bwd(&pre, clip, gh);
                for (g, &s) in gh.iter_mut().zip(sc) {
                    *g *= s;
                }
            });
        }
        {
            let (kp, vp) = (P(gk.as_mut_ptr()), P(gv.as_mut_ptr()));
            let eps = c.rms_norm_eps as f32;
            par(lo * nkv, t * nkv, |i| {
                // SAFETY: this (key, head)'s slices belong to this task.
                let (gkh, gvh) = unsafe { (kp.s(i * d, d), vp.s(i * d, d)) };
                clip_bwd(&tp.k1[i * d..(i + 1) * d], qk_clip, gkh);
                let x = &tp.v1[i * d..(i + 1) * d];
                let u = &tp.v[i * d..(i + 1) * d];
                let rr = 1.0 / (dot(x, x) / d as f32 + eps).sqrt();
                let cc = dot(u, gvh) / d as f32;
                for (g, &u) in gvh.iter_mut().zip(u) {
                    *g = rr * (*g - u * cc);
                }
            });
        }
        let mut gxq = vec![0f32; t * h];
        let mut gxk = vec![0f32; t * h];
        let mut gxv = vec![0f32; t * h];
        for (conv, x, src, dy, g1, gx, p, lin) in [
            (&a.k_conv, xk, &tp.k0, &tp.kd, &gk, &mut gxk, Proj::K, &a.k),
            (&a.v_conv, xv, &tp.v0, &tp.vd, &gv, &mut gxv, Proj::V, &a.v),
        ] {
            let mut g0 = vec![0f32; t * kw];
            match conv {
                KvConv::Static(taps) => conv_bwd(taps, None, src, kw, 1.0 / ((kn + 1) as f32).sqrt(), g1, lo, &mut g0, None),
                KvConv::Dynamic { taps, proj } => {
                    let mut gd = vec![0f32; t * nkv * kn];
                    conv_bwd(
                        taps,
                        Some((dy, nkv)),
                        src,
                        kw,
                        1.0 / ((2 * kn + 1) as f32).sqrt(),
                        g1,
                        lo,
                        &mut g0,
                        Some(&mut gd),
                    );
                    lin_t(proj, nkv * kn, h, &gd[lo * nkv * kn..], n, &mut gx[lo * h..]);
                }
            }
            self.proj_bwd(ad, l, p, lin, x, &tp.u, &g0, lo, t, gx, grads);
        }
        self.proj_bwd(ad, l, Proj::Q, &a.q, xq, &tp.u, &gq, lo, t, &mut gxq, grads);
        self.proj_bwd(ad, l, Proj::R, &a.r, xq, &tp.u, &gr, lo, t, &mut gxq, grads);
        self.proj_bwd(ad, l, Proj::Gate, &a.gate, xq, &tp.u, &g_gate, lo, t, &mut gxq, grads);
        self.proj_bwd(ad, l, Proj::Sink, &a.sink, xq, &tp.u, &g_sink, lo, t, &mut gxq, grads);
        [gxq, gxk, gxv]
    }

    /// The routed experts over the whole sequence ([`Inkling2Model::moe`]'s prefill path),
    /// keeping the routing, the experts' pre-activations and their outputs. `fixed`: the experts
    /// of each token (`[t, k]`), instead of choosing them.
    fn moe_fwd(&self, layer: &Layer, x: &[f32], t: usize, fixed: Option<&[u32]>) -> (Vec<f32>, MoeTape) {
        let c = &self.cfg;
        let (h, ne, k) = (c.hidden_size, c.n_routed_experts, c.num_experts_per_tok);
        let ex = &layer.experts;
        let mi = ex.inter;
        let mut logits = vec![0f32; t * ne];
        crate::moe::router_logits(&layer.router, x, h, t, &mut logits);
        let mut routes = vec![0u32; t * k];
        let mut w = vec![0f32; t * k];
        for ti in 0..t {
            let lg = &logits[ti * ne..(ti + 1) * ne];
            let chosen: Vec<usize> = match fixed {
                Some(f) => f[ti * k..(ti + 1) * k].iter().map(|&e| e as usize).collect(),
                None => route_sigmoid(lg, &layer.bias, k, c.route_scale as f32)
                    .into_iter()
                    .map(|r| r.0)
                    .collect(),
            };
            let s: Vec<f32> = chosen.iter().map(|&e| 1.0 / (1.0 + (-lg[e]).exp())).collect();
            let sum: f32 = s.iter().sum();
            for (j, (&e, &sv)) in chosen.iter().zip(&s).enumerate() {
                routes[ti * k + j] = e as u32;
                w[ti * k + j] = sv / sum * c.route_scale as f32;
            }
        }
        let mut count = vec![0usize; ne];
        for &e in &routes {
            count[e as usize] += 1;
        }
        let mut segs = Vec::with_capacity(ne);
        let mut rows = 0;
        for &n in &count {
            let len = n.next_multiple_of(TN);
            segs.push((rows, len));
            rows += len;
        }
        let mut fill: Vec<usize> = segs.iter().map(|s| s.0).collect();
        let mut src = vec![u32::MAX; rows];
        let mut roww = vec![0f32; rows];
        let mut row = vec![0u32; t * k];
        for i in 0..t * k {
            let e = routes[i] as usize;
            row[i] = fill[e] as u32;
            src[fill[e]] = (i / k) as u32;
            roww[fill[e]] = w[i];
            fill[e] += 1;
        }
        let mut xs = vec![0f32; rows * h];
        for_chunks(&mut xs, h, true, |i, r| {
            if src[i] != u32::MAX {
                r.copy_from_slice(&x[src[i] as usize * h..(src[i] as usize + 1) * h]);
            }
        });
        let mut gu = vec![0f32; rows * 2 * mi];
        let mut a = vec![0f32; rows * mi];
        let act = ex.act;
        match &ex.gate_up[0] {
            GateUp::Fused(_) => {
                let ls: Vec<&Lin> = ex
                    .gate_up
                    .iter()
                    .map(|g| if let GateUp::Fused(l) = g { l } else { unreachable!() })
                    .collect();
                crate::model::linear_grouped(&ls, &segs, 2 * mi, h, &xs, &mut gu);
            }
            GateUp::Split(..) => {
                // (Kept as gate rows, then up rows.)
                let (gs, us): (Vec<&Lin>, Vec<&Lin>) = ex
                    .gate_up
                    .iter()
                    .map(|g| if let GateUp::Split(g, u) = g { (g, u) } else { unreachable!() })
                    .unzip();
                let mut g = vec![0f32; rows * mi];
                let mut u = vec![0f32; rows * mi];
                crate::model::linear_grouped(&gs, &segs, mi, h, &xs, &mut g);
                crate::model::linear_grouped(&us, &segs, mi, h, &xs, &mut u);
                for_chunks(&mut gu, 2 * mi, true, |i, r| {
                    r[..mi].copy_from_slice(&g[i * mi..(i + 1) * mi]);
                    r[mi..].copy_from_slice(&u[i * mi..(i + 1) * mi]);
                });
            }
        }
        let inter = ex.interleaved;
        for_chunks(&mut a, mi, true, |i, ar| {
            let r = &gu[i * 2 * mi..(i + 1) * 2 * mi];
            for (j, o) in ar.iter_mut().enumerate() {
                let (g, u) = if inter { (r[2 * j], r[2 * j + 1]) } else { (r[j], r[mi + j]) };
                *o = act_value(act, g, u);
            }
        });
        let ds: Vec<&Lin> = ex.down.iter().collect();
        let mut o = vec![0f32; rows * h];
        crate::model::linear_grouped(&ds, &segs, h, mi, &a, &mut o);
        let mut y = vec![0f32; t * h];
        for_chunks(&mut y, h, true, |ti, yr| {
            for j in 0..k {
                let i = row[ti * k + j] as usize;
                axpy(yr, w[ti * k + j], &o[i * h..(i + 1) * h]);
            }
        });
        (
            y,
            MoeTape {
                logits,
                routes,
                row,
                segs,
                count,
                src,
                roww,
                gu,
                o,
            },
        )
    }

    /// The gradient of [`Self::moe_fwd`] with respect to its input (rows from `lo` on).
    fn moe_bwd(&self, layer: &Layer, tp: &MoeTape, gy: &[f32], lo: usize, t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (h, ne, k) = (c.hidden_size, c.n_routed_experts, c.num_experts_per_tok);
        let ex = &layer.experts;
        let mi = ex.inter;
        let scale = c.route_scale as f32;
        let mut gx = vec![0f32; t * h];
        // Through the routing weights `scale * s_c / sum(s)` into the router's input.
        {
            let gp = P(gx.as_mut_ptr());
            par(lo, t, |ti| {
                // SAFETY: row `ti` belongs to this task.
                let o = unsafe { gp.s(ti * h, h) };
                let g = &gy[ti * h..(ti + 1) * h];
                let lg = &tp.logits[ti * ne..(ti + 1) * ne];
                let es = &tp.routes[ti * k..(ti + 1) * k];
                let s: Vec<f32> = es.iter().map(|&e| 1.0 / (1.0 + (-lg[e as usize]).exp())).collect();
                let sum: f32 = s.iter().sum();
                let gw: Vec<f32> = (0..k)
                    .map(|j| {
                        let i = tp.row[ti * k + j] as usize;
                        dot(g, &tp.o[i * h..(i + 1) * h])
                    })
                    .collect();
                let m: f32 = gw.iter().zip(&s).map(|(g, s)| g * s).sum::<f32>() / sum;
                for j in 0..k {
                    let gs = scale / sum * (gw[j] - m);
                    let e = es[j] as usize;
                    axpy(o, gs * s[j] * (1.0 - s[j]), &layer.router[e * h..(e + 1) * h]);
                }
            });
        }
        // Through each expert, over the rows routed to it.
        let rows = tp.src.len();
        let mut gxs = vec![0f32; rows * h];
        let xp = P(gxs.as_mut_ptr());
        let (act, inter) = (ex.act, ex.interleaved);
        crate::pool::run(ne, &|e| {
            let (s0, n) = (tp.segs[e].0, tp.count[e]);
            // (A segment's rows are in token order.)
            let a0 = tp.src[s0..s0 + n].partition_point(|&ti| (ti as usize) < lo);
            let na = n - a0;
            if na == 0 {
                return;
            }
            let r0 = s0 + a0;
            let mut go = vec![0f32; na * h];
            for (i, g) in go.chunks_mut(h).enumerate() {
                let ti = tp.src[r0 + i] as usize;
                axpy(g, tp.roww[r0 + i], &gy[ti * h..(ti + 1) * h]);
            }
            let mut wb = Vec::new();
            let mut ga = vec![0f32; na * mi];
            axpy_rows(&ex.down[e], 0, h, mi, &go, h, na, &mut ga, &mut wb);
            drop(go);
            let mut ggu = vec![0f32; na * 2 * mi];
            for i in 0..na {
                let pre = &tp.gu[(r0 + i) * 2 * mi..(r0 + i + 1) * 2 * mi];
                let o = &mut ggu[i * 2 * mi..(i + 1) * 2 * mi];
                for j in 0..mi {
                    let (gi, ui) = if inter { (2 * j, 2 * j + 1) } else { (j, mi + j) };
                    let (dg, du) = act_grad(act, pre[gi], pre[ui]);
                    o[gi] = ga[i * mi + j] * dg;
                    o[ui] = ga[i * mi + j] * du;
                }
            }
            // SAFETY: the rows of expert `e` belong to this task.
            let out = unsafe { xp.s(r0 * h, na * h) };
            match &ex.gate_up[e] {
                GateUp::Fused(l) => axpy_rows(l, 0, 2 * mi, h, &ggu, 2 * mi, na, out, &mut wb),
                GateUp::Split(g, u) => {
                    axpy_rows(g, 0, mi, h, &ggu, 2 * mi, na, out, &mut wb);
                    axpy_rows(u, 0, mi, h, &ggu[mi..], 2 * mi, na, out, &mut wb);
                }
            }
        });
        let gp = P(gx.as_mut_ptr());
        par(lo, t, |ti| {
            // SAFETY: row `ti` belongs to this task.
            let o = unsafe { gp.s(ti * h, h) };
            for j in 0..k {
                let i = tp.row[ti * k + j] as usize;
                add_inplace(o, &gxs[i * h..(i + 1) * h]);
            }
        });
        gx
    }

    /// One layer application over the whole sequence: the stream leaving it, and what its
    /// gradient reads. `states`: the block states by id, `bank_x` the previous pass's output.
    #[allow(clippy::too_many_arguments)]
    fn step_fwd(
        &self,
        ad: Option<&Adapter>,
        ps: &PStep,
        x: &[f32],
        states: &[Option<Vec<f32>>],
        bank_x: Option<&[f32]>,
        t: usize,
        fixed: Option<&[u32]>,
    ) -> (Vec<f32>, StepTape) {
        let c = &self.cfg;
        let h = c.hidden_size;
        let eps = c.rms_norm_eps as f32;
        let l = ps.step.layer;
        let layer = &self.layers[l];
        let conv_norm = 1.0 / ((c.conv_kernel_size + 1) as f32).sqrt();
        let st = |id: &usize| -> &[f32] { states[*id].as_deref().expect("block state") };
        // The messages of this layer, in the passes after the first.
        let (bank_xn, msgs) = match (&ps.bank, &layer.looped) {
            (Some((ids, _)), Some(li)) => {
                let bxn = rms_rows(bank_x.expect("bank stream"), h, MIX_EPS);
                let mut items: Vec<&[f32]> = ids.iter().map(st).collect();
                items.push(&bxn);
                let m = mix_fwd(&items, &li.query, h);
                (Some(bxn), Some(m))
            }
            _ => (None, None),
        };
        let xn_in = (layer.attn_query.is_some() || layer.boundary).then(|| rms_rows(x, h, MIX_EPS));
        let mixed = layer.attn_query.as_ref().map(|q| {
            let mut items: Vec<&[f32]> = ps.before.iter().map(st).collect();
            items.push(xn_in.as_deref().unwrap());
            mix_fwd(&items, q, h)
        });
        let n_in = match (&layer.ln_in, msgs.is_some()) {
            (InNorm::One(_), false) => 1,
            _ => 3,
        };
        let input = |i: usize| -> &[f32] {
            match &mixed {
                Some(m) => &m[i.min(m.len() - 1)],
                None => x,
            }
        };
        let mut gates: Vec<Option<GateTape>> = (0..4).map(|_| None).collect();
        let mut gated: Vec<Vec<f32>> = Vec::new();
        if let (Some(m), Some(li)) = (&msgs, &layer.looped) {
            for i in 0..3 {
                let (o, tp) = self.gate_fwd(&li.gates[i], input(i), &m[i], t);
                gated.push(o);
                gates[i] = Some(tp);
            }
        }
        let xn: Vec<Vec<f32>> = (0..n_in)
            .map(|i| {
                let w = match &layer.ln_in {
                    InNorm::One(w) => w,
                    InNorm::Qkv(wq, wk, wv) => [wq, wk, wv][i],
                };
                let src = if gated.is_empty() { input(i) } else { &gated[i] };
                let mut y = vec![0f32; t * h];
                rmsnorm_f32w_rows(src, h, w, eps, &mut y);
                y
            })
            .collect();
        let pick = |i: usize| -> &[f32] { &xn[i.min(n_in - 1)] };
        let (a_raw, at) = self.attn_fwd(ad, l, &layer.attn, [pick(0), pick(1), pick(2)], t);
        let a = conv_fwd(&layer.attn_conv, None, &a_raw, h, conv_norm);
        let x_mid = if layer.boundary {
            a
        } else {
            let mut v = x.to_vec();
            add_inplace(&mut v, &a);
            v
        };
        let xn_mid = rms_rows(&x_mid, h, MIX_EPS);
        let hn0 = {
            let mut items: Vec<&[f32]> = ps.before.iter().map(st).collect();
            if layer.boundary {
                items.push(xn_in.as_deref().unwrap());
            }
            items.push(&xn_mid);
            mix_fwd(&items, &layer.mlp_query, h).pop().unwrap()
        };
        let hn1 = match (&msgs, &layer.looped) {
            (Some(m), Some(li)) => {
                let (o, tp) = self.gate_fwd(&li.gates[3], &hn0, &m[3], t);
                gates[3] = Some(tp);
                Some(o)
            }
            _ => None,
        };
        let mut hn = vec![0f32; t * h];
        rmsnorm_f32w_rows(hn1.as_deref().unwrap_or(&hn0), h, &layer.ln_post, eps, &mut hn);
        let (o_raw, moe) = self.moe_fwd(layer, &hn, t, fixed);
        let mut out = conv_fwd(&layer.mlp_conv, None, &o_raw, h, conv_norm);
        add_inplace(&mut out, &x_mid);
        (
            out,
            StepTape {
                xn_in,
                bank_xn,
                msgs,
                mixed,
                gated,
                gates,
                xn,
                at,
                a_raw,
                x_mid,
                xn_mid,
                hn0,
                hn1,
                moe,
                o_raw,
            },
        )
    }

    /// The gradient of [`Self::step_fwd`]: from `g_out` (the gradient of the stream leaving the
    /// step) into the accumulators of what the step read (`gstream`, `gstate`), and the adapter's
    /// gradients.
    #[allow(clippy::too_many_arguments)]
    fn step_bwd(
        &self,
        ad: &Adapter,
        ps: &PStep,
        x: &[f32],
        states: &[Option<Vec<f32>>],
        bank_x: Option<&[f32]>,
        tp: &StepTape,
        mut g: Vec<f32>,
        lo: usize,
        t: usize,
        gstream: &mut [Option<Vec<f32>>],
        gstate: &mut [Option<Vec<f32>>],
        grads: &mut Grads,
    ) {
        let c = &self.cfg;
        let h = c.hidden_size;
        let eps = c.rms_norm_eps as f32;
        let l = ps.step.layer;
        let layer = &self.layers[l];
        let conv_norm = 1.0 / ((c.conv_kernel_size + 1) as f32).sqrt();
        let st = |id: &usize| -> &[f32] { states[*id].as_deref().expect("block state") };
        let n = t * h;
        for id in ps.before.iter().chain(&ps.push).chain(ps.bank.iter().flat_map(|b| &b.0)) {
            acc(&mut gstate[*id], n);
        }
        let state_p = |gstate: &mut [Option<Vec<f32>>], id: usize| P(gstate[id].as_mut().unwrap().as_mut_ptr());
        let mut g_msg: Vec<Vec<f32>> = if tp.msgs.is_some() {
            (0..4).map(|_| vec![0f32; n]).collect()
        } else {
            Vec::new()
        };
        // out = x_mid + conv(o_raw): `g` goes on as the gradient of x_mid.
        let mut g_oraw = vec![0f32; n];
        conv_bwd(&layer.mlp_conv, None, &tp.o_raw, h, conv_norm, &g, lo, &mut g_oraw, None);
        let g_hn = self.moe_bwd(layer, &tp.moe, &g_oraw, lo, t);
        drop(g_oraw);
        let mut g_hn1 = vec![0f32; n];
        norm_bwd(
            tp.hn1.as_deref().unwrap_or(&tp.hn0),
            Some(&layer.ln_post),
            h,
            eps,
            &g_hn,
            lo,
            &mut g_hn1,
        );
        drop(g_hn);
        let g_hn0 = match (&tp.msgs, &layer.looped, &tp.gates[3]) {
            (Some(m), Some(li), Some(gt)) => {
                let mut gx = vec![0f32; n];
                self.gate_bwd(&li.gates[3], &tp.hn0, &m[3], gt, &g_hn1, lo, t, &mut gx, &mut g_msg[3]);
                gx
            }
            _ => g_hn1,
        };
        // The MLP's mix: the block states (the one this step pushed included) and the stream.
        let mut g_xn_mid = vec![0f32; n];
        {
            let mut items: Vec<&[f32]> = ps.before.iter().map(st).collect();
            let mut gp: Vec<P> = ps.before.iter().map(|&id| state_p(gstate, id)).collect();
            if let Some(id) = ps.push {
                items.push(tp.xn_in.as_deref().unwrap());
                gp.push(state_p(gstate, id));
            }
            items.push(&tp.xn_mid);
            gp.push(P(g_xn_mid.as_mut_ptr()));
            mix_bwd(&items, &layer.mlp_query, h, &g_hn0, lo, &gp);
        }
        drop(g_hn0);
        norm_bwd(&tp.x_mid, None, h, MIX_EPS, &g_xn_mid, lo, &mut g);
        drop(g_xn_mid);
        // x_mid = a (a block's first layer) or x + a.
        let mut g_araw = vec![0f32; n];
        conv_bwd(&layer.attn_conv, None, &tp.a_raw, h, conv_norm, &g, lo, &mut g_araw, None);
        let mut g_x = if layer.boundary { vec![0f32; n] } else { g };
        let n_in = tp.xn.len();
        let pick = |i: usize| -> &[f32] { &tp.xn[i.min(n_in - 1)] };
        let g_xn = self.attn_bwd(ad, l, &layer.attn, [pick(0), pick(1), pick(2)], &tp.at, &g_araw, lo, t, grads);
        drop(g_araw);
        let input = |i: usize| -> &[f32] {
            match &tp.mixed {
                Some(m) => &m[i.min(m.len() - 1)],
                None => x,
            }
        };
        // The gradients of the mixes (or of the stream, on layer 0).
        let nq = tp.mixed.as_ref().map_or(0, Vec::len);
        let mut g_mixed: Vec<Vec<f32>> = (0..nq).map(|_| vec![0f32; n]).collect();
        for i in 0..n_in {
            // (One input for the three projections: their gradients add.)
            let mut gi = vec![0f32; n];
            if n_in == 1 {
                for g in &g_xn {
                    add_inplace(&mut gi[lo * h..], &g[lo * h..]);
                }
            } else {
                gi[lo * h..].copy_from_slice(&g_xn[i][lo * h..]);
            }
            let w = match &layer.ln_in {
                InNorm::One(w) => w,
                InNorm::Qkv(wq, wk, wv) => [wq, wk, wv][i],
            };
            let src = if tp.gated.is_empty() { input(i) } else { &tp.gated[i] };
            let mut g_pre = vec![0f32; n];
            norm_bwd(src, Some(w), h, eps, &gi, lo, &mut g_pre);
            let dst: &mut Vec<f32> = if nq > 0 { &mut g_mixed[i.min(nq - 1)] } else { &mut g_x };
            match (&tp.msgs, &layer.looped, &tp.gates[i]) {
                (Some(m), Some(li), Some(gt)) => self.gate_bwd(&li.gates[i], input(i), &m[i], gt, &g_pre, lo, t, dst, &mut g_msg[i]),
                _ => add_inplace(&mut dst[lo * h..], &g_pre[lo * h..]),
            }
        }
        drop(g_xn);
        let mut g_xn_in = vec![0f32; if tp.xn_in.is_some() { n } else { 0 }];
        if let (Some(q), Some(xn_in)) = (&layer.attn_query, &tp.xn_in) {
            let mut items: Vec<&[f32]> = ps.before.iter().map(st).collect();
            items.push(xn_in);
            let mut gp: Vec<P> = ps.before.iter().map(|&id| state_p(gstate, id)).collect();
            gp.push(P(g_xn_in.as_mut_ptr()));
            for (qi, gm) in g_mixed.iter().enumerate() {
                mix_bwd(&items, &q[qi * h..(qi + 1) * h], h, gm, lo, &gp);
            }
        }
        // The state this step pushed is the norm of its input, like the mixes' last item.
        if let Some(id) = ps.push {
            add_inplace(&mut g_xn_in, gstate[id].as_ref().unwrap());
            gstate[id] = None;
        }
        if tp.xn_in.is_some() {
            norm_bwd(x, None, h, MIX_EPS, &g_xn_in, lo, &mut g_x);
        }
        add_inplace(acc(&mut gstream[ps.x_in], n), &g_x);
        // The messages: mixes of the previous pass's states and output.
        if let (Some((ids, bx)), Some(li), Some(bxn)) = (&ps.bank, &layer.looped, &tp.bank_xn) {
            let mut items: Vec<&[f32]> = ids.iter().map(st).collect();
            items.push(bxn);
            let mut g_bxn = vec![0f32; n];
            let mut gp: Vec<P> = ids.iter().map(|&id| state_p(gstate, id)).collect();
            gp.push(P(g_bxn.as_mut_ptr()));
            for (qi, gm) in g_msg.iter().enumerate() {
                mix_bwd(&items, &li.query[qi * h..(qi + 1) * h], h, gm, lo, &gp);
            }
            norm_bwd(
                bank_x.expect("bank stream"),
                None,
                h,
                MIX_EPS,
                &g_bxn,
                lo,
                acc(&mut gstream[*bx], n),
            );
        }
    }

    /// The whole pass over `x0` (the normed embeddings, `[t, H]`): every stream, and the experts
    /// each step routed to (`fixed`: those of an earlier pass, used instead of choosing).
    fn tape(&self, ad: Option<&Adapter>, plan: &Plan, x0: Vec<f32>, t: usize, fixed: Option<&[Vec<u32>]>) -> Tape {
        let h = self.cfg.hidden_size;
        let mut streams: Vec<Vec<f32>> = Vec::with_capacity(plan.n_streams);
        streams.push(x0);
        let mut states: Vec<Option<Vec<f32>>> = (0..plan.state_src.len()).map(|_| None).collect();
        let mut routes = Vec::with_capacity(plan.steps.len());
        for (si, ps) in plan.steps.iter().enumerate() {
            let bank_x = ps.bank.as_ref().map(|b| &streams[b.1][..]);
            let (out, tp) = self.step_fwd(ad, ps, &streams[ps.x_in], &states, bank_x, t, fixed.map(|f| &f[si][..]));
            if let Some(id) = ps.push {
                states[id] = tp.xn_in;
            }
            routes.push(tp.moe.routes);
            debug_assert_eq!((streams.len(), out.len()), (ps.x_out, t * h));
            streams.push(out);
        }
        Tape { streams, routes }
    }

    /// The final-normed hidden states of a pass (`[t, H]`), and the mix they are the norm of.
    fn final_fwd(&self, plan: &Plan, tape: &Tape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.cfg.hidden_size;
        let x = &tape.streams[plan.final_stream];
        let t = x.len() / h;
        let sts: Vec<Vec<f32>> = plan
            .final_states
            .iter()
            .map(|&id| rms_rows(&tape.streams[plan.state_src[id]], h, MIX_EPS))
            .collect();
        let xn = rms_rows(x, h, MIX_EPS);
        let mut items: Vec<&[f32]> = sts.iter().map(|s| &s[..]).collect();
        items.push(&xn);
        let fm = mix_fwd(&items, &self.final_query, h).pop().unwrap();
        let mut out = vec![0f32; t * h];
        rmsnorm_f32w_rows(&fm, h, &self.final_norm, self.cfg.rms_norm_eps as f32, &mut out);
        (out, fm, xn)
    }

    /// The raw (uncapped) logits of `n` hidden rows, cut to the tokens: `[n, unpadded vocab]`.
    fn head_raw_rows(&self, hidden: &[f32], n: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (v, u) = (c.vocab_size, c.unpadded_vocab_size);
        let mut logits = lin_fwd(&self.lm_head, v, c.hidden_size, hidden, n);
        if u < v {
            for ti in 1..n {
                logits.copy_within(ti * v..ti * v + u, ti * u);
            }
            logits.truncate(n * u);
        }
        logits
    }

    fn plan(&self) -> Plan {
        Plan::new(&self.cfg, |l| self.layers[l].boundary)
    }

    /// The soft-capped logits (`[positions.len(), unpadded vocab]`) of `tokens` run as one pass
    /// from position 0 with the adapter's updates (none: the base model): the forward pass
    /// training differentiates.
    pub fn train_logits(&self, ad: Option<&Adapter>, tokens: &[u32], positions: &[usize]) -> Result<Vec<f32>> {
        self.check_trainable()?;
        let (h, t) = (self.cfg.hidden_size, tokens.len());
        if t == 0 || positions.iter().any(|&p| p >= t) {
            bail!("train_logits: positions out of the {t} tokens");
        }
        let plan = self.plan();
        let tape = self.tape(ad, &plan, self.embed_rows(tokens, &[]), t, None);
        let (out, _, _) = self.final_fwd(&plan, &tape);
        let mut rows = vec![0f32; positions.len() * h];
        for (r, &p) in rows.chunks_mut(h).zip(positions) {
            r.copy_from_slice(&out[p * h..(p + 1) * h]);
        }
        let mut logits = self.head_raw_rows(&rows, positions.len());
        let s = self.cfg.final_logit_softcapping as f32;
        for x in logits.iter_mut() {
            *x = (*x / s).tanh() * s;
        }
        Ok(logits)
    }

    /// The loss of `targets` (`(index of the token in `tokens`, weight)`: the token at that
    /// index, at least 1, is predicted from the positions before it; each contributes
    /// [`token_loss`]) and the log-probability of each target, without gradients.
    pub fn train_loss(&self, ad: Option<&Adapter>, tokens: &[u32], targets: &[(usize, f32)]) -> Result<(f64, Vec<f32>)> {
        self.train_loss_with(ad, tokens, targets, &token_loss, None)
    }

    fn train_loss_with(
        &self,
        ad: Option<&Adapter>,
        tokens: &[u32],
        targets: &[(usize, f32)],
        term: &(dyn Fn(f32, f32) -> (f64, f64) + Sync),
        fixed: Option<&[Vec<u32>]>,
    ) -> Result<(f64, Vec<f32>)> {
        self.check_trainable()?;
        check_targets(tokens, targets, self.cfg.unpadded_vocab_size)?;
        let plan = self.plan();
        let t = tokens.len();
        let tape = self.tape(ad, &plan, self.embed_rows(tokens, &[]), t, fixed);
        let (out, _, _) = self.final_fwd(&plan, &tape);
        let mut loss = 0f64;
        let mut logp = Vec::with_capacity(targets.len());
        self.head_loss(&out, tokens, targets, term, &mut loss, &mut logp, None);
        Ok((loss, logp))
    }

    /// The loss terms of `targets` from the final hidden states `out`, in chunks of rows; with
    /// `g_out` (`[t, H]`), adds the gradient of the loss with respect to `out`.
    #[allow(clippy::too_many_arguments)]
    fn head_loss(
        &self,
        out: &[f32],
        tokens: &[u32],
        targets: &[(usize, f32)],
        term: &(dyn Fn(f32, f32) -> (f64, f64) + Sync),
        loss: &mut f64,
        logp: &mut Vec<f32>,
        mut g_out: Option<&mut [f32]>,
    ) {
        let c = &self.cfg;
        let (h, u) = (c.hidden_size, c.unpadded_vocab_size);
        let cap = c.final_logit_softcapping as f32;
        for chunk in targets.chunks(64) {
            let n = chunk.len();
            let mut rows = vec![0f32; n * h];
            for (r, &(j, _)) in rows.chunks_mut(h).zip(chunk) {
                r.copy_from_slice(&out[(j - 1) * h..j * h]);
            }
            let mut z = self.head_raw_rows(&rows, n);
            // Per row: the capped logits' log-softmax, the loss term, and in place of the raw
            // logits the gradient with respect to them.
            let res: Vec<std::sync::Mutex<(f64, f32)>> = (0..n).map(|_| std::sync::Mutex::new((0.0, 0.0))).collect();
            let want_grad = g_out.is_some();
            for_chunks(&mut z, u, n > 1, |i, row| {
                let (j, w) = chunk[i];
                let y = tokens[j] as usize;
                let capped: Vec<f32> = row.iter().map(|&x| (x / cap).tanh() * cap).collect();
                let m = capped.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = m + capped.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
                let lp = (capped[y] as f64 - lse) as f32;
                let (ls, dl) = term(w, lp);
                *res[i].lock().unwrap() = (ls, lp);
                if want_grad {
                    for (v, (o, &cz)) in row.iter_mut().zip(&capped).enumerate() {
                        let p = (cz as f64 - lse).exp();
                        let gz = dl * (if v == y { 1.0 } else { 0.0 } - p);
                        let th = cz / cap;
                        *o = (gz * (1.0 - (th * th) as f64)) as f32;
                    }
                }
            });
            for r in res {
                let (ls, lp) = r.into_inner().unwrap();
                *loss += ls;
                logp.push(lp);
            }
            if let Some(g) = g_out.as_deref_mut() {
                let mut gh = vec![0f32; n * h];
                lin_t(&self.lm_head, u, h, &z, n, &mut gh);
                for (r, &(j, _)) in gh.chunks(h).zip(chunk) {
                    add_inplace(&mut g[(j - 1) * h..j * h], r);
                }
            }
        }
    }

    /// The loss of `targets` over `tokens` (see [`Self::train_loss`]) and its gradient with
    /// respect to every `A` and `B` of the adapter.
    pub fn train_grads(&self, ad: &Adapter, tokens: &[u32], targets: &[(usize, f32)], opts: TrainOptions) -> Result<Outcome> {
        self.train_grads_with(ad, tokens, targets, opts, &token_loss)
    }

    /// [`Self::train_grads`] with another loss term: `term(weight, logp)` returns a target's loss
    /// and its derivative with respect to the target token's log-probability.
    pub fn train_grads_with(
        &self,
        ad: &Adapter,
        tokens: &[u32],
        targets: &[(usize, f32)],
        opts: TrainOptions,
        term: &(dyn Fn(f32, f32) -> (f64, f64) + Sync),
    ) -> Result<Outcome> {
        self.check_trainable()?;
        check_targets(tokens, targets, self.cfg.unpadded_vocab_size)?;
        for s in &ad.sites {
            if s.layer >= self.layers.len() || (s.n_out, s.n_in) != s.proj.shape(&self.cfg, s.layer) {
                bail!("the adapter's {} of layer {} is not this model's", s.proj.name(), s.layer);
            }
        }
        let h = self.cfg.hidden_size;
        let t = tokens.len();
        let n = t * h;
        let t0 = std::time::Instant::now();
        let plan = self.plan();
        let tape = self.tape(Some(ad), &plan, self.embed_rows(tokens, &[]), t, None);
        let (out, fm, xn) = self.final_fwd(&plan, &tape);
        let forward_secs = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let first = targets.iter().map(|x| x.0 - 1).min().unwrap_or(0);
        let lo = opts.grad_window.map_or(0, |w| first.saturating_sub(w));
        let mut loss = 0f64;
        let mut logp = Vec::with_capacity(targets.len());
        let mut g_out = vec![0f32; n];
        self.head_loss(&out, tokens, targets, term, &mut loss, &mut logp, Some(&mut g_out));
        let mut grads = ad.zeros();
        let mut gstream: Vec<Option<Vec<f32>>> = (0..plan.n_streams).map(|_| None).collect();
        let mut gstate: Vec<Option<Vec<f32>>> = (0..plan.state_src.len()).map(|_| None).collect();
        let mut states: Vec<Option<Vec<f32>>> = (0..plan.state_src.len()).map(|_| None).collect();
        let ensure = |states: &mut [Option<Vec<f32>>], id: usize| {
            if states[id].is_none() {
                states[id] = Some(rms_rows(&tape.streams[plan.state_src[id]], h, MIX_EPS));
            }
        };
        // The final norm and mix.
        {
            let mut g_fm = vec![0f32; n];
            norm_bwd(&fm, Some(&self.final_norm), h, self.cfg.rms_norm_eps as f32, &g_out, lo, &mut g_fm);
            drop((g_out, out));
            for &id in &plan.final_states {
                ensure(&mut states, id);
                acc(&mut gstate[id], n);
            }
            let mut items: Vec<&[f32]> = plan.final_states.iter().map(|&id| states[id].as_deref().unwrap()).collect();
            items.push(&xn);
            let mut g_xn = vec![0f32; n];
            let mut gp: Vec<P> = plan
                .final_states
                .iter()
                .map(|&id| P(gstate[id].as_mut().unwrap().as_mut_ptr()))
                .collect();
            gp.push(P(g_xn.as_mut_ptr()));
            mix_bwd(&items, &self.final_query, h, &g_fm, lo, &gp);
            norm_bwd(
                &tape.streams[plan.final_stream],
                None,
                h,
                MIX_EPS,
                &g_xn,
                lo,
                acc(&mut gstream[plan.final_stream], n),
            );
        }
        drop((fm, xn));
        for (si, ps) in plan.steps.iter().enumerate().rev() {
            for &id in ps.before.iter().chain(ps.bank.iter().flat_map(|b| &b.0)) {
                ensure(&mut states, id);
            }
            let x = &tape.streams[ps.x_in];
            let bank_x = ps.bank.as_ref().map(|b| &tape.streams[b.1][..]);
            // The step again from its saved input, for what its gradient reads.
            let (_, tp) = self.step_fwd(Some(ad), ps, x, &states, bank_x, t, Some(&tape.routes[si]));
            let g = gstream[ps.x_out].take().unwrap_or_else(|| vec![0f32; n]);
            self.step_bwd(ad, ps, x, &states, bank_x, &tp, g, lo, t, &mut gstream, &mut gstate, &mut grads);
            if let Some(id) = ps.push {
                states[id] = None;
            }
        }
        let tape_bytes = tape.streams.iter().map(|s| s.len() * 4).sum::<usize>() + tape.routes.iter().map(|r| r.len() * 4).sum::<usize>();
        Ok(Outcome {
            loss,
            logp,
            grads,
            grad_from: lo,
            tape_bytes,
            forward_secs,
            backward_secs: t1.elapsed().as_secs_f64(),
        })
    }

    /// Bytes [`Self::train_grads`] keeps per token of a sequence between its forward and its
    /// backward pass (the streams and the routing).
    pub fn train_tape_bytes_per_token(&self) -> usize {
        let steps = self.cfg.steps().len();
        (steps + 1) * self.cfg.hidden_size * 4 + steps * self.cfg.num_experts_per_tok * 4
    }
}

/// The value of the experts' activation (as [`Act`] computes it).
#[inline]
fn act_value(act: Act, g: f32, u: f32) -> f32 {
    match act {
        Act::Situ { beta, linear_beta } => {
            let u = if linear_beta > 0.0 {
                linear_beta * (u / linear_beta).tanh()
            } else {
                u
            };
            beta * (g / beta).tanh() / (1.0 + (-g).exp()) * u
        }
        _ => unreachable!("Inkling-2's experts are SiTU"),
    }
}

fn check_targets(tokens: &[u32], targets: &[(usize, f32)], vocab: usize) -> Result<()> {
    if tokens.is_empty() {
        bail!("no tokens");
    }
    if let Some(&tk) = tokens.iter().find(|&&tk| tk as usize >= vocab) {
        bail!("token {tk} is outside the vocabulary ({vocab})");
    }
    if let Some(&(j, _)) = targets.iter().find(|x| x.0 == 0 || x.0 >= tokens.len()) {
        bail!("target {j} is not one of the tokens 1..{}", tokens.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::LoraSpec;
    use std::path::PathBuf;

    fn tiny(tag: &str) -> (PathBuf, Inkling2Model<'static>) {
        let dir = std::env::temp_dir().join(format!("oxinfer-ink-train-{tag}-{}", std::process::id()));
        tiny::write_tiny(&dir, 11).unwrap();
        let cfg = Inkling2Config::load(&dir).unwrap();
        let files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(&dir, true).unwrap()));
        let m = Inkling2Model::load(cfg, files, Quant::Bf16).unwrap();
        let _ = m.device.set(None);
        (dir, m)
    }

    fn tokens(n: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 37 + 5 * (i / 7) + 11) % 250) as u32).collect()
    }

    fn noise(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
            })
            .collect()
    }

    /// An adapter on projections `projs` of every layer, with a nonzero `B`.
    fn adapter(m: &Inkling2Model, projs: &[Proj]) -> Adapter {
        let mut ad = Adapter::new(&m.cfg, 2, 4.0, projs, &[], 3);
        for (i, s) in ad.sites.iter_mut().enumerate() {
            s.b = noise(100 + i as u64, s.b.len(), 0.05);
        }
        ad
    }

    fn targets(t: usize) -> Vec<(usize, f32)> {
        (t - 14..t)
            .map(|j| (j, if j % 4 == 1 { -0.7 } else { 1.0 + 0.1 * (j % 3) as f32 }))
            .collect()
    }

    /// The gradients against central differences of the loss along random directions, one
    /// projection kind and one factor at a time, with the routing of the base point held fixed
    /// (as the gradient holds it).
    #[test]
    fn gradients_match_finite_differences() {
        let (_dir, m) = tiny("fd");
        let t = 44;
        let toks = tokens(t);
        let tg = targets(t);
        let ad = adapter(&m, &Proj::ALL);
        let out = m.train_grads(&ad, &toks, &tg, TrainOptions::default()).unwrap();
        let plan = m.plan();
        let routes = m.tape(Some(&ad), &plan, m.embed_rows(&toks, &[]), t, None).routes;
        let loss = |ad: &Adapter| m.train_loss_with(Some(ad), &toks, &tg, &token_loss, Some(&routes)).unwrap().0;
        assert!(
            (loss(&ad) - out.loss).abs() < 1e-5 * out.loss.abs().max(1.0),
            "{} {}",
            loss(&ad),
            out.loss
        );
        let mut worst = 0f64;
        for &p in &Proj::ALL {
            for which in 0..2 {
                let dirs: Vec<Vec<f32>> = ad
                    .sites
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        let n = if which == 0 { s.a.len() } else { s.b.len() };
                        if s.proj == p {
                            noise(7 + (i * 2 + which) as u64, n, 1.0)
                        } else {
                            vec![0f32; n]
                        }
                    })
                    .collect();
                let g = if which == 0 { &out.grads.a } else { &out.grads.b };
                let analytic: f64 = g
                    .iter()
                    .zip(&dirs)
                    .flat_map(|(g, d)| g.iter().zip(d))
                    .map(|(&g, &d)| g as f64 * d as f64)
                    .sum();
                let eps = 1e-3f32;
                let moved = |sign: f32| {
                    let mut a2 = ad.clone();
                    for (s, d) in a2.sites.iter_mut().zip(&dirs) {
                        let v = if which == 0 { &mut s.a } else { &mut s.b };
                        for (x, &d) in v.iter_mut().zip(d) {
                            *x += sign * eps * d;
                        }
                    }
                    loss(&a2)
                };
                let fd = (moved(1.0) - moved(-1.0)) / (2.0 * eps as f64);
                // (float32 losses: the difference itself is only good to about 1e-2.)
                let err = ((fd - analytic).abs() - 0.02).max(0.0) / fd.abs().max(analytic.abs());
                eprintln!(
                    "{} {}: analytic {analytic:+.5e} fd {fd:+.5e} rel {err:.2e}",
                    p.name(),
                    ["A", "B"][which]
                );
                worst = worst.max(err);
            }
        }
        assert!(worst < 2e-2, "worst relative error {worst}");
    }

    /// A gradient window that covers the whole sequence changes nothing; a shorter one gives
    /// other (truncated) gradients of the same loss.
    #[test]
    fn gradient_window() {
        let (_dir, m) = tiny("win");
        let t = 40;
        let toks = tokens(t);
        let tg = targets(t);
        let ad = adapter(&m, &[Proj::O, Proj::Q]);
        let full = m.train_grads(&ad, &toks, &tg, TrainOptions::default()).unwrap();
        let wide = m.train_grads(&ad, &toks, &tg, TrainOptions { grad_window: Some(t) }).unwrap();
        assert_eq!(wide.grad_from, 0);
        assert_eq!(full.grads.b, wide.grads.b);
        let cut = m.train_grads(&ad, &toks, &tg, TrainOptions { grad_window: Some(4) }).unwrap();
        assert_eq!(cut.grad_from, t - 14 - 1 - 4);
        assert_eq!(cut.loss, full.loss);
        assert!(cut.grads.b.iter().flatten().all(|x| x.is_finite()));
        assert!(cut.grads.b != full.grads.b);
    }

    /// The adapter saved as a PEFT folder and attached to the model as `--lora` does gives the
    /// logits of the training forward; so does a round trip through [`Adapter::load`].
    #[test]
    fn saved_adapter_runs_as_the_training_forward() {
        let (dir, m) = tiny("save");
        let t = 40;
        let toks = tokens(t);
        let ad = adapter(&m, &Proj::ALL);
        let pos: Vec<usize> = (0..t).collect();
        let train = m.train_logits(Some(&ad), &toks, &pos).unwrap();
        let base = m.train_logits(None, &toks, &pos).unwrap();
        let out = dir.join("adapter");
        ad.save(&out, "tiny").unwrap();
        let back = Adapter::load(&m.cfg, &out).unwrap();
        assert_eq!(back.sites.len(), ad.sites.len());
        for (a, b) in ad.sites.iter().zip(&back.sites) {
            assert_eq!((a.layer, a.proj, &a.a, &a.b), (b.layer, b.proj, &b.a, &b.b));
        }
        let run = |m: &Inkling2Model| {
            let mut s = m.new_session();
            let hs = m.forward_hidden(&mut s, &toks);
            m.logits(&hs, t)
        };
        let plain = run(&m);
        let (_d2, mut m2) = tiny("save");
        let at = crate::lora::attach_specs(&mut m2, &[LoraSpec { path: out, scale: 1.0 }], &dir, false).unwrap();
        assert_eq!(at[0].modules, ad.sites.len());
        let _ = m2.device.set(None);
        let adapted = run(&m2);
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        eprintln!(
            "train vs --lora {:.2e}, base train vs inference {:.2e}, adapter moves {:.2e}",
            diff(&train, &adapted),
            diff(&base, &plain),
            diff(&train, &base)
        );
        assert!(diff(&train, &base) > 1e-2);
        assert!(diff(&base, &plain) < 2e-4);
        assert!(diff(&train, &adapted) < 2e-4);
    }
}
