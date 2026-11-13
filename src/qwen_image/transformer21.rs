//! `QwenImage21Transformer2DModel`: the 32-block single-stream transformer of Qwen-Image-2.1
//! (about 7B parameters, no biases anywhere).
//!
//! Text and image tokens share one sequence: the Qwen3-VL hidden states (`txt_in`: a zero-centred
//! RMSNorm, two linear layers around a tanh GELU), with the latent tokens of each condition image
//! standing where the text encoder had that image's placeholders (four latent tokens per
//! placeholder; the hidden states at those positions are dropped), then the target image's
//! latent tokens (`img_in`; 64 channels per token, no patching). Attention is block-causal: a
//! token sees everything before it, and every image is bidirectional inside. One shared
//! `modulation` layer turns the timestep embedding into the scales and gates of all blocks
//! (`x += tanh(gate) * f(layer_norm(x) * (1 + scale))`), and with `causal_condition` only the
//! target's tokens take the sampled timestep: text and condition images are modulated as at
//! timestep 0. Rotary positions have three axes: text tokens advance all three together; an
//! image holds the first at the position the text before it reached (which then advances by the
//! image's longer side) and lays rows and columns out centred on zero.
//!
//! Nothing before the target depends on the target or on the timestep, so the prefix runs once
//! ([`Transformer21::prefill`], keeping every block's keys and values) and each denoising step
//! computes the target's tokens alone against them ([`Transformer21::step`]): diffusers'
//! `use_kv_cache`, here without the first full pass.

use super::transformer::{CosSin, ImageShape, gated_add, gelu_tanh_exact, norm_modulate, temb};
use super::weights::{Dense, Loader};
use crate::kernels::*;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d1() -> usize {
    1
}
fn d3() -> usize {
    3
}
fn d32() -> usize {
    32
}
fn d64() -> usize {
    64
}
fn d128() -> usize {
    128
}
fn d4096() -> usize {
    4096
}
fn daxes() -> [usize; 3] {
    [16, 56, 56]
}
fn deps() -> f64 {
    1e-6
}
fn dtrue() -> bool {
    true
}

/// `transformer/config.json` of a `QwenImage21Transformer2DModel`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config21 {
    #[serde(default = "d1")]
    pub patch_size: usize,
    #[serde(default = "d64")]
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: Option<usize>,
    #[serde(default = "d32")]
    pub num_layers: usize,
    #[serde(default = "d128")]
    pub attention_head_dim: usize,
    #[serde(default = "d32")]
    pub num_attention_heads: usize,
    #[serde(default = "d4096")]
    pub context_in_dim: usize,
    #[serde(default = "d3")]
    pub mlp_ratio: usize,
    #[serde(default = "daxes")]
    pub axes_dims_rope: [usize; 3],
    #[serde(default = "deps")]
    pub eps: f64,
    #[serde(default = "dtrue")]
    pub causal_condition: bool,
}

/// Whether `dir` holds the config of a Qwen-Image-2.1 transformer.
pub fn is_transformer21(dir: &Path) -> bool {
    crate::fsx::read(&dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["_class_name"].as_str() == Some("QwenImage21Transformer2DModel"))
}

impl Config21 {
    pub fn load(dir: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?).context("transformer/config.json")?;
        if c.patch_size != 1 {
            bail!("Qwen-Image-2.1 transformer: patch_size {} is not supported (1)", c.patch_size);
        }
        if (c.eps - 1e-6).abs() > 1e-12 {
            bail!("Qwen-Image-2.1 transformer: eps {} is not supported (1e-6)", c.eps);
        }
        if !c.causal_condition {
            bail!("Qwen-Image-2.1 transformer: causal_condition = false is not supported");
        }
        if c.attention_head_dim != 2 * (c.axes_dims_rope.iter().sum::<usize>() / 2) {
            bail!(
                "axes_dims_rope {:?} does not cover the head size {}",
                c.axes_dims_rope,
                c.attention_head_dim
            );
        }
        Ok(c)
    }
    pub fn inner(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
    pub fn out(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }
}

struct Block<'a> {
    to_q: Dense<'a>,
    to_k: Dense<'a>,
    to_v: Dense<'a>,
    to_out: Dense<'a>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    proj: Dense<'a>,
    gate: Dense<'a>,
    out: Dense<'a>,
}

impl<'a> Block<'a> {
    fn lins(&self) -> [&Dense<'a>; 7] {
        [&self.to_q, &self.to_k, &self.to_v, &self.to_out, &self.proj, &self.gate, &self.out]
    }
}

/// The module paths of a block's linear layers (LoRA targets).
pub const BLOCK_LINEARS: [&str; 7] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.proj",
    "img_mlp.gate_layer",
    "img_mlp.out",
];
/// The module paths of the linear layers outside the blocks.
pub const TOP_LINEARS: [&str; 8] = [
    "img_in",
    "txt_in.in_layer",
    "txt_in.out_layer",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "modulation.1",
    "norm_out.linear",
    "proj_out",
];

/// A run of tokens before the target: text, or one condition image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seg {
    Text(usize),
    Image(ImageShape),
}

impl Seg {
    fn tokens(&self) -> usize {
        match self {
            Seg::Text(n) => *n,
            Seg::Image(s) => s.tokens(),
        }
    }
}

/// What a prompt (and its condition images) leaves for the denoising steps: every block's keys
/// and values of the tokens before the target (after the head norm and RoPE).
pub struct Prefix {
    /// Tokens before the target.
    pub n: usize,
    pub segs: Vec<Seg>,
    pub target: ImageShape,
    /// Per block: keys and values, `[n, inner]` (with room for the target's behind them).
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// RoPE cosines and sines of the target's tokens.
    rope: CosSin,
}

pub struct Transformer21<'a> {
    pub cfg: Config21,
    img_in: Dense<'a>,
    /// `txt_in.text_norm.weight + 1` (the checkpoint stores the scale zero-centred).
    txt_norm: Vec<f32>,
    txt_in: Dense<'a>,
    txt_out: Dense<'a>,
    t_lin1: Dense<'a>,
    t_lin2: Dense<'a>,
    modulation: Dense<'a>,
    blocks: Vec<Block<'a>>,
    norm_out: Dense<'a>,
    proj_out: Dense<'a>,
    inv_freq: [Vec<f32>; 3],
}

/// The token runs before the target for `slots` (per text-encoder token: whether it is an image
/// placeholder, each standing for four latent tokens) and the condition images' latent grids,
/// in order.
pub fn segments(slots: &[bool], cond: &[ImageShape]) -> Result<Vec<Seg>> {
    let n_slots = slots.iter().filter(|&&s| s).count();
    let n_cond: usize = cond.iter().map(|s| s.tokens()).sum();
    if n_slots * 4 != n_cond {
        bail!("the prompt holds {n_slots} image placeholders (4 latent tokens each) for {n_cond} condition-image tokens");
    }
    let mut segs = Vec::new();
    let (mut text, mut img, mut left) = (0usize, 0usize, 0usize);
    for &s in slots {
        if !s {
            text += 1;
            continue;
        }
        if left == 0 {
            // A new image starts here.
            if text > 0 {
                segs.push(Seg::Text(std::mem::take(&mut text)));
            }
            let shape = cond[img];
            img += 1;
            left = shape.tokens();
            if left == 0 || !left.is_multiple_of(4) {
                bail!("condition image {img}: {left} latent tokens (expected a multiple of 4)");
            }
            segs.push(Seg::Image(shape));
        } else if text > 0 {
            bail!("condition image {img}: its placeholders are interrupted by text");
        }
        left -= 4;
    }
    if left != 0 {
        bail!("condition image {img}: the prompt ends inside its placeholders");
    }
    if text > 0 {
        segs.push(Seg::Text(text));
    }
    Ok(segs)
}

/// Rotary positions `[frame, row, column]` of the tokens of `segs`, then of `target`.
pub fn positions(segs: &[Seg], target: ImageShape) -> Vec<[i64; 3]> {
    let mut pos = Vec::new();
    let mut p = 0i64;
    for s in segs.iter().chain(std::iter::once(&Seg::Image(target))) {
        match *s {
            Seg::Text(n) => {
                pos.extend((0..n as i64).map(|i| [p + i; 3]));
                p += n as i64;
            }
            Seg::Image(sh) => {
                let (h, w) = (sh.rows as i64, sh.cols as i64);
                for _ in 0..sh.frames {
                    for r in 0..h {
                        for c in 0..w {
                            pos.push([p, r - (h - h / 2), c - (w - w / 2)]);
                        }
                    }
                }
                p += h.max(w);
            }
        }
    }
    pos
}

/// Multi-head attention of `tq` queries over `tk` keys (no mask), in query chunks that keep the
/// score matrix under ~256 MB (GPU when one is active).
fn attend(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, nh: usize, hd: usize) -> Vec<f32> {
    let d = nh * hd;
    let chunk = ((1usize << 26) / tk.max(1)).clamp(64, tq.max(64));
    if chunk >= tq {
        return crate::sd::ops::attention(q, k, v, tq, tk, nh, hd, false);
    }
    let mut out = vec![0f32; tq * d];
    let mut s = 0;
    while s < tq {
        let e = (s + chunk).min(tq);
        let o = crate::sd::ops::attention(&q[s * d..e * d], k, v, e - s, tk, nh, hd, false);
        out[s * d..e * d].copy_from_slice(&o);
        s = e;
    }
    out
}

/// Causal attention of a text run: query `i` (of `tq`, the tokens `off..off + tq` of the
/// sequence) sees the keys `0..=off + i` (`k`, `v`: the first `off + tq` tokens).
fn attend_causal(q: &[f32], k: &[f32], v: &[f32], tq: usize, off: usize, nh: usize, hd: usize) -> Vec<f32> {
    let _t = crate::sd::ops::timing::start(crate::sd::ops::timing::ATTN);
    let d = nh * hd;
    let tk = off + tq;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; tq * d];
    let mut qh = vec![0f32; tq * hd];
    let mut kh = vec![0f32; tk * hd];
    for h in 0..nh {
        for_chunks(&mut qh, hd, tq > 64, |i, r| {
            r.copy_from_slice(&q[i * d + h * hd..i * d + (h + 1) * hd])
        });
        for_chunks(&mut kh, hd, tk > 64, |j, r| {
            r.copy_from_slice(&k[j * d + h * hd..j * d + (h + 1) * hd])
        });
        let s = crate::sd::ops::matmul_f32(&kh, tk, hd, &qh, tq);
        let mut o = vec![0f32; tq * hd];
        for_chunks(&mut o, hd, tq > 1, |i, r| {
            let row = &s[i * tk..i * tk + off + i + 1];
            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x * scale));
            let mut acc = vec![0f64; hd];
            let mut sum = 0f64;
            for (j, &x) in row.iter().enumerate() {
                let e = ((x * scale - mx) as f64).exp();
                sum += e;
                for (a, &vv) in acc.iter_mut().zip(&v[j * d + h * hd..j * d + (h + 1) * hd]) {
                    *a += e * vv as f64;
                }
            }
            for (o, a) in r.iter_mut().zip(&acc) {
                *o = (a / sum) as f32;
            }
        });
        for_chunks(&mut out, d, tq > 64, |i, r| {
            r[h * hd..(h + 1) * hd].copy_from_slice(&o[i * hd..(i + 1) * hd])
        });
    }
    out
}

impl<'a> Transformer21<'a> {
    pub fn load(cfg: Config21, ld: &Loader<'a, '_>) -> Result<Self> {
        let d = cfg.inner();
        let (hd, ctx) = (cfg.attention_head_dim, cfg.context_in_dim);
        let hid = d * cfg.mlp_ratio;
        // The layers outside the blocks stay in their checkpoint precision (GGUF files keep
        // their own types).
        let top = Loader {
            files: ld.files,
            quant: crate::model::Quant::Bf16,
            lora: ld.lora,
            prefix: ld.prefix.clone(),
        };
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let b = format!("transformer_blocks.{l}.");
            let dn = |n: &str, o: usize, i: usize| ld.dense(&format!("{b}{n}"), o, i);
            blocks.push(Block {
                to_q: dn("attn.to_q", d, d)?,
                to_k: dn("attn.to_k", d, d)?,
                to_v: dn("attn.to_v", d, d)?,
                to_out: dn("attn.to_out.0", d, d)?,
                norm_q: ld.f32(&format!("{b}attn.norm_q.weight"), hd)?,
                norm_k: ld.f32(&format!("{b}attn.norm_k.weight"), hd)?,
                proj: dn("img_mlp.proj", hid, d)?,
                gate: dn("img_mlp.gate_layer", hid, d)?,
                out: dn("img_mlp.out", d, hid)?,
            });
        }
        let inv = |dim: usize| -> Vec<f32> { (0..dim / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32)).collect() };
        let ax = cfg.axes_dims_rope;
        Ok(Self {
            img_in: top.dense("img_in", d, cfg.in_channels)?,
            txt_norm: ld.f32("txt_in.text_norm.weight", ctx)?.into_iter().map(|w| w + 1.0).collect(),
            txt_in: top.dense("txt_in.in_layer", d, ctx)?,
            txt_out: top.dense("txt_in.out_layer", d, d)?,
            t_lin1: top.dense("time_text_embed.timestep_embedder.linear_1", d, 256)?,
            t_lin2: top.dense("time_text_embed.timestep_embedder.linear_2", d, d)?,
            modulation: top.dense("modulation.1", 4 * d, d)?,
            blocks,
            norm_out: top.dense("norm_out.linear", d, d)?,
            proj_out: top.dense("proj_out", cfg.out(), d)?,
            inv_freq: [inv(ax[0]), inv(ax[1]), inv(ax[2])],
            cfg,
        })
    }

    /// Weight bytes of the blocks.
    pub fn bytes(&self) -> usize {
        self.blocks.iter().flat_map(|b| b.lins()).map(|d| d.w.bytes()).sum()
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// RoPE cosines and sines (`head_dim / 2` per token) of `pos`.
    fn rope(&self, pos: &[[i64; 3]]) -> CosSin {
        let half = self.cfg.attention_head_dim / 2;
        let (mut c, mut s) = (vec![0f32; pos.len() * half], vec![0f32; pos.len() * half]);
        for (t, p) in pos.iter().enumerate() {
            let mut j = t * half;
            for (ax, fr) in self.inv_freq.iter().enumerate() {
                for &f in fr {
                    let a = p[ax] as f32 * f;
                    c[j] = a.cos();
                    s[j] = a.sin();
                    j += 1;
                }
            }
        }
        (c, s)
    }

    /// The timestep embedding of `sigma` and the modulation it gives every block:
    /// `[scale 1, gate 1, scale 2, gate 2]` (`inner` each), the gates already through `tanh`.
    fn modulate(&self, sigma: f32) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.inner();
        let e = temb(&self.t_lin1, &self.t_lin2, sigma);
        let se: Vec<f32> = e.iter().map(|&v| crate::sd::ops::silu(v)).collect();
        let mut m = self.modulation.forward_small(&se, 1);
        for g in [1, 3] {
            m[g * d..(g + 1) * d].iter_mut().for_each(|v| *v = v.tanh());
        }
        (e, m)
    }

    /// Queries, keys and values of `x` (`[n, inner]`, modulated), with the head norm and RoPE.
    fn qkv(&self, b: &Block, x: &[f32], n: usize, rope: &CosSin) -> [Vec<f32>; 3] {
        let d = self.cfg.inner();
        let (nh, hd) = (self.cfg.num_attention_heads, self.cfg.attention_head_dim);
        let half = hd / 2;
        let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
        Dense::forward_multi(&mut [(&b.to_q, &mut q), (&b.to_k, &mut k), (&b.to_v, &mut v)], x, n);
        for (buf, w) in [(&mut q, &b.norm_q), (&mut k, &b.norm_k)] {
            for_chunks(buf, hd, true, |i, head| {
                let row = i / nh;
                rmsnorm_f32w_inplace(head, w, 1e-6);
                apply_rope_interleaved(head, &rope.0[row * half..(row + 1) * half], &rope.1[row * half..(row + 1) * half]);
            });
        }
        [q, k, v]
    }

    /// `out(silu(gate_layer(x)) * proj(x))` in row chunks that keep the hidden activations bounded.
    fn mlp(&self, b: &Block, x: &[f32], rows: usize) -> Vec<f32> {
        let (d, h) = (b.proj.n_in, b.proj.n_out);
        let mut y = vec![0f32; rows * d];
        let chunk = ((128usize << 20) / (h * 4)).max(64);
        let mut r0 = 0;
        while r0 < rows {
            let n = chunk.min(rows - r0);
            let (mut g, mut p) = (vec![0f32; n * h], vec![0f32; n * h]);
            Dense::forward_multi(&mut [(&b.gate, &mut g), (&b.proj, &mut p)], &x[r0 * d..(r0 + n) * d], n);
            for_chunks(&mut g, 4096, true, |i, r| {
                for (a, &pv) in r.iter_mut().zip(&p[i * 4096..]) {
                    *a = crate::sd::ops::silu(*a) * pv;
                }
            });
            drop(p);
            let o = b.out.forward(&g, n);
            y[r0 * d..(r0 + n) * d].copy_from_slice(&o);
            r0 += n;
        }
        y
    }

    /// The second half of a block on `x`: the gated attention output, then the gated MLP.
    fn finish(&self, b: &Block, x: &mut [f32], att: &[f32], n: usize, m: &[f32]) {
        let d = self.cfg.inner();
        let zero = vec![0f32; d];
        let o = b.to_out.forward(att, n);
        gated_add(x, &o, d, &[&m[d..2 * d]], |_| 0);
        let xm = norm_modulate(x, d, &[(&zero, &m[2 * d..3 * d])], |_| 0);
        let y = self.mlp(b, &xm, n);
        gated_add(x, &y, d, &[&m[3 * d..4 * d]], |_| 0);
    }

    /// Runs the tokens before the target (modulated as at timestep 0): `hidden` holds the text
    /// encoder's hidden states (`[slots.len(), context_in_dim]`, the last layer's output before
    /// its final norm), `slots` marks the image placeholders among them, `cond` the latent
    /// tokens of the condition images (`[tokens, in_channels]`, in order) with grids
    /// `cond_shapes`; `target` is the grid of the image to denoise.
    pub fn prefill(&self, hidden: &[f32], slots: &[bool], cond: &[f32], cond_shapes: &[ImageShape], target: ImageShape) -> Result<Prefix> {
        let cfg = &self.cfg;
        let d = cfg.inner();
        let (nh, hd, ctx) = (cfg.num_attention_heads, cfg.attention_head_dim, cfg.context_in_dim);
        let nt = slots.len();
        if hidden.len() != nt * ctx {
            bail!("text hidden states: {} values for {nt} tokens of width {ctx}", hidden.len());
        }
        let segs = segments(slots, cond_shapes)?;
        let n: usize = segs.iter().map(|s| s.tokens()).sum();
        if cond.len() != cond_shapes.iter().map(|s| s.tokens()).sum::<usize>() * cfg.in_channels {
            bail!("condition latents do not match their grids");
        }
        // Text rows, with the condition images' latent tokens at their placeholders.
        let mut hn = vec![0f32; nt * ctx];
        rmsnorm_f32w_rows(hidden, ctx, &self.txt_norm, 1e-6, &mut hn);
        let mut mid = self.txt_in.forward(&hn, nt);
        drop(hn);
        for_chunks(&mut mid, 4096, true, |_, r| r.iter_mut().for_each(|v| *v = gelu_tanh_exact(*v)));
        let te = self.txt_out.forward(&mid, nt);
        drop(mid);
        let ce = self.img_in.forward(cond, cond.len() / cfg.in_channels);
        let mut x = Vec::with_capacity(n * d);
        let (mut ti, mut ci) = (0usize, 0usize);
        for s in &segs {
            match s {
                Seg::Text(k) => {
                    let mut left = *k;
                    while left > 0 {
                        if !slots[ti] {
                            x.extend_from_slice(&te[ti * d..(ti + 1) * d]);
                            left -= 1;
                        }
                        ti += 1;
                    }
                }
                Seg::Image(sh) => {
                    x.extend_from_slice(&ce[ci * d..(ci + sh.tokens()) * d]);
                    ci += sh.tokens();
                }
            }
        }
        debug_assert_eq!(x.len(), n * d);
        let pos = positions(&segs, target);
        let rope = self.rope(&pos[..n]);
        let (_, m) = self.modulate(0.0);
        let zero = vec![0f32; d];
        let mut kv = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let xm = norm_modulate(&x, d, &[(&zero, &m[..d])], |_| 0);
            let [q, k, v] = self.qkv(b, &xm, n, &rope);
            drop(xm);
            let mut att = Vec::with_capacity(n * d);
            let mut s0 = 0;
            for s in &segs {
                let e = s0 + s.tokens();
                let qs = &q[s0 * d..e * d];
                att.extend(match s {
                    Seg::Text(_) => attend_causal(qs, &k[..e * d], &v[..e * d], e - s0, s0, nh, hd),
                    Seg::Image(_) => attend(qs, &k[..e * d], &v[..e * d], e - s0, e, nh, hd),
                });
                s0 = e;
            }
            self.finish(b, &mut x, &att, n, &m);
            kv.push((k, v));
        }
        Ok(Prefix {
            n,
            segs,
            target,
            kv,
            rope: self.rope(&pos[n..]),
        })
    }

    /// One denoising pass: the velocity of the target's latent tokens `x` (`[tokens,
    /// in_channels]`) at noise level `sigma`, against the prefix `pre` (whose buffers it uses as
    /// scratch space).
    pub fn step(&self, pre: &mut Prefix, x: &[f32], sigma: f32) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.inner();
        let (nh, hd) = (cfg.num_attention_heads, cfg.attention_head_dim);
        let n = pre.target.tokens();
        assert_eq!(x.len(), n * cfg.in_channels, "target tokens do not match the grid");
        let mut x = self.img_in.forward(x, n);
        let (e, m) = self.modulate(sigma);
        let zero = vec![0f32; d];
        let p = pre.n;
        for (b, (pk, pv)) in self.blocks.iter().zip(pre.kv.iter_mut()) {
            let xm = norm_modulate(&x, d, &[(&zero, &m[..d])], |_| 0);
            let [q, k, v] = self.qkv(b, &xm, n, &pre.rope);
            drop(xm);
            pk.truncate(p * d);
            pk.extend_from_slice(&k);
            pv.truncate(p * d);
            pv.extend_from_slice(&v);
            drop((k, v));
            let att = attend(&q, pk, pv, n, p + n, nh, hd);
            drop(q);
            self.finish(b, &mut x, &att, n, &m);
        }
        // Output: layer norm scaled by the timestep embedding (no shift), then the projection.
        let se: Vec<f32> = e.iter().map(|&v| crate::sd::ops::silu(v)).collect();
        let scale = self.norm_out.forward_small(&se, 1);
        let xo = norm_modulate(&x, d, &[(&zero, &scale)], |_| 0);
        self.proj_out.forward(&xo, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(rows: usize, cols: usize) -> ImageShape {
        ImageShape { frames: 1, rows, cols }
    }

    /// The layout of `QwenImage21Rope` / `build_token_metadata` (diffusers 0.44.0.dev0).
    #[test]
    fn layout_matches_diffusers() {
        // Text to image: 3 text tokens, then a 2x4 target.
        let segs = segments(&[false; 3], &[]).unwrap();
        assert_eq!(segs, [Seg::Text(3)]);
        let p = positions(&segs, img(2, 4));
        assert_eq!(p[..3], [[0; 3], [1; 3], [2; 3]]);
        assert_eq!(p[3], [3, -1, -2]);
        assert_eq!(p[10], [3, 0, 1]);
        // Two condition images back to back (2 + 1 placeholders), text around them.
        let slots = [false, true, true, true, false, false];
        let segs = segments(&slots, &[img(2, 4), img(2, 2)]).unwrap();
        assert_eq!(segs, [Seg::Text(1), Seg::Image(img(2, 4)), Seg::Image(img(2, 2)), Seg::Text(2)]);
        let p = positions(&segs, img(4, 2));
        assert_eq!(p.len(), 1 + 8 + 4 + 2 + 8);
        assert_eq!(p[1], [1, -1, -2]);
        assert_eq!(p[9], [5, -1, -1]);
        assert_eq!(p[13], [7; 3]);
        assert_eq!(p[15], [9, -2, -1]);
        assert!(segments(&[true], &[img(2, 4)]).is_err());
    }

    /// The causal attention of a text run equals full attention over the keys it may see.
    #[test]
    fn causal_run() {
        let (nh, hd, off, tq) = (2usize, 4usize, 3usize, 4usize);
        let d = nh * hd;
        let f = |n: usize, s: usize| -> Vec<f32> { (0..n * d).map(|i| (((i * 31 + s * 17) % 23) as f32 - 11.0) / 9.0).collect() };
        let (k, v) = (f(off + tq, 1), f(off + tq, 2));
        let q = f(tq, 3);
        let got = attend_causal(&q, &k, &v, tq, off, nh, hd);
        for i in 0..tq {
            let lim = off + i + 1;
            let want = crate::sd::ops::attention(&q[i * d..(i + 1) * d], &k[..lim * d], &v[..lim * d], 1, lim, nh, hd, false);
            for (a, b) in got[i * d..(i + 1) * d].iter().zip(&want) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
        }
    }
}
