//! `QwenImageTransformer2DModel`: the 60-block MMDiT of Qwen-Image (about 20B parameters).
//!
//! Each block keeps two streams, image tokens (2x2 patches of the 16-channel latents, the noisy
//! target first, then the reference images of an edit) and text tokens (the Qwen2.5-VL hidden
//! states), with their own AdaLN modulation (shift / scale / gate from the timestep embedding),
//! projections and GELU MLPs, and one joint attention over `[text, image]` with per-head RMSNorm
//! on queries and keys and the 3-axis MSRoPE (frame, row, column; rows and columns centred on the
//! image, text positions on the diagonal after the largest image). With `zero_cond_t` (Edit-2511)
//! the reference-image tokens are modulated as at timestep 0.

use super::weights::{Dense, Loader};
use crate::kernels::*;
use crate::pool;
use anyhow::{Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d2() -> usize {
    2
}
fn d64() -> usize {
    64
}
fn d60() -> usize {
    60
}
fn d128() -> usize {
    128
}
fn d24() -> usize {
    24
}
fn d3584() -> usize {
    3584
}
fn daxes() -> [usize; 3] {
    [16, 56, 56]
}

/// `transformer/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformerConfig {
    #[serde(default = "d2")]
    pub patch_size: usize,
    #[serde(default = "d64")]
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: Option<usize>,
    #[serde(default = "d60")]
    pub num_layers: usize,
    #[serde(default = "d128")]
    pub attention_head_dim: usize,
    #[serde(default = "d24")]
    pub num_attention_heads: usize,
    #[serde(default = "d3584")]
    pub joint_attention_dim: usize,
    #[serde(default)]
    pub guidance_embeds: bool,
    #[serde(default = "daxes")]
    pub axes_dims_rope: [usize; 3],
    #[serde(default)]
    pub zero_cond_t: bool,
    #[serde(default)]
    pub use_additional_t_cond: bool,
    #[serde(default)]
    pub use_layer3d_rope: bool,
}

impl TransformerConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        if c.guidance_embeds || c.use_additional_t_cond || c.use_layer3d_rope {
            bail!("Qwen-Image transformer variants with guidance embeddings, additional t conditions or layered RoPE are not supported");
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
        self.out_channels.unwrap_or(self.in_channels / 4)
    }
}

struct Block<'a> {
    img_mod: Dense<'a>,
    txt_mod: Dense<'a>,
    to_q: Dense<'a>,
    to_k: Dense<'a>,
    to_v: Dense<'a>,
    add_q: Dense<'a>,
    add_k: Dense<'a>,
    add_v: Dense<'a>,
    to_out: Dense<'a>,
    to_add_out: Dense<'a>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    norm_added_q: Vec<f32>,
    norm_added_k: Vec<f32>,
    img_fc1: Dense<'a>,
    img_fc2: Dense<'a>,
    txt_fc1: Dense<'a>,
    txt_fc2: Dense<'a>,
}

impl<'a> Block<'a> {
    fn lins(&self) -> [&Dense<'a>; 14] {
        [
            &self.img_mod,
            &self.txt_mod,
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.to_out,
            &self.to_add_out,
            &self.img_fc1,
            &self.img_fc2,
            &self.txt_fc1,
            &self.txt_fc2,
        ]
    }
}

/// RoPE cosines and sines, `head_dim / 2` per token.
pub type CosSin = (Vec<f32>, Vec<f32>);

/// One image of the token sequence: `frames x rows x cols` patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageShape {
    pub frames: usize,
    pub rows: usize,
    pub cols: usize,
}

impl ImageShape {
    pub fn tokens(&self) -> usize {
        self.frames * self.rows * self.cols
    }
}

pub struct Transformer<'a> {
    pub cfg: TransformerConfig,
    img_in: Dense<'a>,
    txt_in: Dense<'a>,
    txt_norm: Vec<f32>,
    t_lin1: Dense<'a>,
    t_lin2: Dense<'a>,
    blocks: Vec<Block<'a>>,
    norm_out: Dense<'a>,
    proj_out: Dense<'a>,
    /// Inverse frequencies per RoPE axis.
    inv_freq: [Vec<f32>; 3],
    /// Weight streaming: unit `i` is block `i`.
    pub(crate) stream: Option<crate::stream::StreamHook>,
}

/// LayerNorm without affine parameters (eps 1e-6) of each `n`-row, then `x * (1 + scale) +
/// shift` with the modulation of each row's group.
fn norm_modulate(x: &[f32], n: usize, mods: &[(&[f32], &[f32])], group: impl Fn(usize) -> usize + Sync) -> Vec<f32> {
    let _t = crate::sd::ops::timing::start(crate::sd::ops::timing::LN);
    let mut y = vec![0f32; x.len()];
    for_chunks(&mut y, n, x.len() > n, |i, r| {
        let src = &x[i * n..(i + 1) * n];
        let mean = src.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        let (shift, scale) = mods[group(i)];
        for j in 0..n {
            r[j] = ((src[j] as f64 - mean) * inv) as f32 * (1.0 + scale[j]) + shift[j];
        }
    });
    y
}

/// `x[i] += gate * y[i]` per row, with the gate of each row's group.
fn gated_add(x: &mut [f32], y: &[f32], n: usize, gates: &[&[f32]], group: impl Fn(usize) -> usize + Sync) {
    let _t = crate::sd::ops::timing::start(6);
    for_chunks(x, n, x.len() > n, |i, r| {
        let g = gates[group(i)];
        for ((a, &b), &gg) in r.iter_mut().zip(&y[i * n..(i + 1) * n]).zip(g) {
            *a += gg * b;
        }
    });
}

impl<'a> Transformer<'a> {
    pub fn load(cfg: TransformerConfig, ld: &Loader<'a, '_>) -> Result<Self> {
        let d = cfg.inner();
        let (hd, ja) = (cfg.attention_head_dim, cfg.joint_attention_dim);
        let p2 = cfg.patch_size * cfg.patch_size;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let b = format!("transformer_blocks.{l}.");
            let dn = |n: &str, o: usize, i: usize| ld.dense(&format!("{b}{n}"), o, i);
            blocks.push(Block {
                img_mod: dn("img_mod.1", 6 * d, d)?,
                txt_mod: dn("txt_mod.1", 6 * d, d)?,
                to_q: dn("attn.to_q", d, d)?,
                to_k: dn("attn.to_k", d, d)?,
                to_v: dn("attn.to_v", d, d)?,
                add_q: dn("attn.add_q_proj", d, d)?,
                add_k: dn("attn.add_k_proj", d, d)?,
                add_v: dn("attn.add_v_proj", d, d)?,
                to_out: dn("attn.to_out.0", d, d)?,
                to_add_out: dn("attn.to_add_out", d, d)?,
                norm_q: ld.f32(&format!("{b}attn.norm_q.weight"), hd)?,
                norm_k: ld.f32(&format!("{b}attn.norm_k.weight"), hd)?,
                norm_added_q: ld.f32(&format!("{b}attn.norm_added_q.weight"), hd)?,
                norm_added_k: ld.f32(&format!("{b}attn.norm_added_k.weight"), hd)?,
                img_fc1: dn("img_mlp.net.0.proj", 4 * d, d)?,
                img_fc2: dn("img_mlp.net.2", d, 4 * d)?,
                txt_fc1: dn("txt_mlp.net.0.proj", 4 * d, d)?,
                txt_fc2: dn("txt_mlp.net.2", d, 4 * d)?,
            });
        }
        let inv = |dim: usize| -> Vec<f32> { (0..dim / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32)).collect() };
        let ax = cfg.axes_dims_rope;
        Ok(Self {
            img_in: ld.dense("img_in", d, cfg.in_channels)?,
            txt_in: ld.dense("txt_in", d, ja)?,
            txt_norm: ld.f32("txt_norm.weight", ja)?,
            t_lin1: ld.dense("time_text_embed.timestep_embedder.linear_1", d, 256)?,
            t_lin2: ld.dense("time_text_embed.timestep_embedder.linear_2", d, d)?,
            blocks,
            norm_out: ld.dense("norm_out.linear", 2 * d, d)?,
            proj_out: ld.dense("proj_out", p2 * cfg.out(), d)?,
            inv_freq: [inv(ax[0]), inv(ax[1]), inv(ax[2])],
            stream: None,
            cfg,
        })
    }

    /// Streaming units (one per block, in execution order) and their weight memory.
    pub fn stream_units(&self) -> Vec<crate::stream::UnitSpec> {
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, b)| crate::stream::UnitSpec {
                name: format!("block.{i}"),
                ranges: b.lins().iter().flat_map(|d| d.w.mem_ranges()).collect(),
                uses: 1.0,
                parts: Vec::new(),
            })
            .collect()
    }

    /// Marks the weights of the units with `streamed[i]` as streamed (no device state between calls).
    pub fn mark_streamed(&self, streamed: &[bool]) {
        for (b, &s) in self.blocks.iter().zip(streamed) {
            for d in b.lins() {
                d.w.set_transient_device(s);
            }
        }
    }

    pub fn set_stream(&mut self, hook: Option<crate::stream::StreamHook>) {
        self.stream = hook;
    }

    /// Weight bytes of the blocks.
    pub fn bytes(&self) -> usize {
        self.blocks.iter().flat_map(|b| b.lins()).map(|d| d.w.bytes()).sum()
    }

    /// The timestep embedding of `sigma` (the pipeline passes `t / 1000`): `[inner]`.
    fn temb(&self, sigma: f32) -> Vec<f32> {
        // Timesteps(256, flip_sin_to_cos, shift 0, scale 1000) on sigma, in f32 as diffusers.
        let half = 128;
        let mut e = vec![0f32; 256];
        for i in 0..half {
            let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            let a = sigma * f * 1000.0;
            e[i] = a.cos();
            e[half + i] = a.sin();
        }
        let mut h = self.t_lin1.forward_small(&e, 1);
        h.iter_mut().for_each(|v| *v = crate::sd::ops::silu(*v));
        self.t_lin2.forward_small(&h, 1)
    }

    /// RoPE angles (cos, sin; `head_dim / 2` per token) of the image tokens of `shapes` (image
    /// `k` is frame `k`) and of `txt` text tokens.
    pub fn rope(&self, shapes: &[ImageShape], txt: usize) -> (CosSin, CosSin) {
        let half = self.cfg.attention_head_dim / 2;
        let [f0, f1, f2] = &self.inv_freq;
        let angles = |pos: [i64; 3], out_c: &mut [f32], out_s: &mut [f32]| {
            let mut j = 0;
            for (ax, fr) in [f0, f1, f2].iter().enumerate() {
                for &f in fr.iter() {
                    let a = pos[ax] as f32 * f;
                    out_c[j] = a.cos();
                    out_s[j] = a.sin();
                    j += 1;
                }
            }
        };
        let n: usize = shapes.iter().map(|s| s.tokens()).sum();
        let (mut ic, mut is) = (vec![0f32; n * half], vec![0f32; n * half]);
        let mut t = 0;
        let mut max_vid = 0i64;
        for (k, s) in shapes.iter().enumerate() {
            let (h, w) = (s.rows as i64, s.cols as i64);
            for f in 0..s.frames {
                for r in 0..h {
                    for c in 0..w {
                        // scale_rope: positions centred, -(h - h/2) .. h/2 - 1.
                        let pos = [(k + f) as i64, r - (h - h / 2), c - (w - w / 2)];
                        angles(pos, &mut ic[t * half..(t + 1) * half], &mut is[t * half..(t + 1) * half]);
                        t += 1;
                    }
                }
            }
            max_vid = max_vid.max(h / 2).max(w / 2);
        }
        let (mut tc, mut ts) = (vec![0f32; txt * half], vec![0f32; txt * half]);
        for i in 0..txt {
            let p = max_vid + i as i64;
            angles([p, p, p], &mut tc[i * half..(i + 1) * half], &mut ts[i * half..(i + 1) * half]);
        }
        ((ic, is), (tc, ts))
    }

    /// The text stream input: `txt_in(rmsnorm(hidden))` for the `[t, joint_attention_dim]`
    /// Qwen2.5-VL hidden states.
    pub fn embed_text(&self, hidden: &[f32], t: usize) -> Vec<f32> {
        let ja = self.cfg.joint_attention_dim;
        let mut hn = vec![0f32; t * ja];
        rmsnorm_f32w_rows(hidden, ja, &self.txt_norm, 1e-6, &mut hn);
        self.txt_in.forward(&hn, t)
    }

    /// One forward pass: `img` holds the packed latent patches of `shapes` (`[tokens,
    /// in_channels]`; `shapes[0]` the noisy target, the rest reference images), `txt` the text
    /// stream (`embed_text`, `[t_txt, inner]`), `sigma` the noise level. Returns the velocity
    /// prediction for the target's tokens (`[shapes[0].tokens(), patch^2 * out_channels]`).
    pub fn forward(&self, img: &[f32], shapes: &[ImageShape], txt: &[f32], sigma: f32) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.inner();
        let (nh, hd) = (cfg.num_attention_heads, cfg.attention_head_dim);
        let half = hd / 2;
        let ni: usize = shapes.iter().map(|s| s.tokens()).sum();
        let n0 = shapes[0].tokens();
        let nt = txt.len() / d;
        assert_eq!(img.len(), ni * cfg.in_channels, "image tokens do not match the shapes");
        let mut x = self.img_in.forward(img, ni);
        let mut tx = txt.to_vec();
        let temb = self.temb(sigma);
        let zero = cfg.zero_cond_t && ni > n0;
        // Modulation rows: the target's timestep, and 0 for the reference images (zero_cond_t).
        let mut temb_rows = temb.clone();
        if zero {
            temb_rows.extend(self.temb(0.0));
        }
        let nrows = temb_rows.len() / d;
        let silu_rows: Vec<f32> = temb_rows.iter().map(|&v| crate::sd::ops::silu(v)).collect();
        let img_group = |i: usize| (zero && i >= n0) as usize;
        let ((ic, is), (tc, ts)) = self.rope(shapes, nt);
        let nall = nt + ni;
        let mut q = vec![0f32; nall * d];
        let mut k = vec![0f32; nall * d];
        let mut v = vec![0f32; nall * d];
        for (li, b) in self.blocks.iter().enumerate() {
            let _unit = crate::stream::enter(&self.stream, li);
            let im = b.img_mod.forward_small(&silu_rows, nrows); // [nrows, 6d]
            let tm = b.txt_mod.forward_small(&silu_rows[..d], 1); // [6d]
            let ch = |row: usize, c: usize| row * 6 * d + c * d..row * 6 * d + (c + 1) * d;
            // chunk order per half: shift, scale, gate.
            let imods: Vec<(&[f32], &[f32])> = (0..nrows).map(|r| (&im[ch(r, 0)], &im[ch(r, 1)])).collect();
            let igates: Vec<&[f32]> = (0..nrows).map(|r| &im[ch(r, 2)]).collect();
            let tmods = [(&tm[0..d], &tm[d..2 * d])];
            let xm = norm_modulate(&x, d, &imods, img_group);
            let tmm = norm_modulate(&tx, d, &tmods, |_| 0);

            // Projections into the joint [text, image] rows.
            {
                let (qt, qi) = q.split_at_mut(nt * d);
                let (kt, ki) = k.split_at_mut(nt * d);
                let (vt, vi) = v.split_at_mut(nt * d);
                Dense::forward_multi(&mut [(&b.to_q, qi), (&b.to_k, ki), (&b.to_v, vi)], &xm, ni);
                Dense::forward_multi(&mut [(&b.add_q, qt), (&b.add_k, kt), (&b.add_v, vt)], &tmm, nt);
            }
            drop(xm);
            drop(tmm);
            // Per-head RMSNorm and RoPE (adjacent pairs).
            let tq = crate::sd::ops::timing::start(6);
            for (buf, wt, wi) in [(&mut q, &b.norm_added_q, &b.norm_q), (&mut k, &b.norm_added_k, &b.norm_k)] {
                for_chunks(buf, hd, true, |i, head| {
                    let row = i / nh;
                    let (w, c, s) = if row < nt {
                        (wt, &tc[row * half..(row + 1) * half], &ts[row * half..(row + 1) * half])
                    } else {
                        let r = row - nt;
                        (wi, &ic[r * half..(r + 1) * half], &is[r * half..(r + 1) * half])
                    };
                    rmsnorm_f32w_inplace(head, w, 1e-6);
                    apply_rope_interleaved(head, c, s);
                });
            }
            drop(tq);
            let att = crate::qwen_vision::attention_chunked(&q, &k, &v, nall, nh, hd);
            let ia = b.to_out.forward(&att[nt * d..], ni);
            let ta = b.to_add_out.forward(&att[..nt * d], nt);
            drop(att);
            gated_add(&mut x, &ia, d, &igates, img_group);
            gated_add(&mut tx, &ta, d, &[&tm[2 * d..3 * d]], |_| 0);

            // MLPs.
            let imods2: Vec<(&[f32], &[f32])> = (0..nrows).map(|r| (&im[ch(r, 3)], &im[ch(r, 4)])).collect();
            let igates2: Vec<&[f32]> = (0..nrows).map(|r| &im[ch(r, 5)]).collect();
            let xm = norm_modulate(&x, d, &imods2, img_group);
            let y = mlp(&b.img_fc1, &b.img_fc2, &xm, ni);
            gated_add(&mut x, &y, d, &igates2, img_group);
            let tmods2 = [(&tm[3 * d..4 * d], &tm[4 * d..5 * d])];
            let tmm = norm_modulate(&tx, d, &tmods2, |_| 0);
            let y = mlp(&b.txt_fc1, &b.txt_fc2, &tmm, nt);
            gated_add(&mut tx, &y, d, &[&tm[5 * d..6 * d]], |_| 0);
        }
        // Output: AdaLayerNormContinuous (scale first, then shift) on the target's tokens.
        let silu_t: Vec<f32> = temb.iter().map(|&v| crate::sd::ops::silu(v)).collect();
        let e = self.norm_out.forward_small(&silu_t, 1);
        let xo = norm_modulate(&x[..n0 * d], d, &[(&e[d..2 * d], &e[..d])], |_| 0);
        self.proj_out.forward(&xo, n0)
    }
}

/// `fc2(gelu_tanh(fc1(x)))` in row chunks that keep the hidden activations bounded.
fn mlp(fc1: &Dense, fc2: &Dense, x: &[f32], rows: usize) -> Vec<f32> {
    let (d, h) = (fc1.n_in, fc1.n_out);
    let mut y = vec![0f32; rows * d];
    let chunk = ((256usize << 20) / (h * 4)).max(64);
    let mut r0 = 0;
    while r0 < rows {
        let n = chunk.min(rows - r0);
        let mut mid = fc1.forward(&x[r0 * d..(r0 + n) * d], n);
        let c = (mid.len() / pool::threads().max(1)).max(4096);
        for_chunks(&mut mid, c, true, |_, r| r.iter_mut().for_each(|v| *v = gelu_tanh_exact(*v)));
        let o = fc2.forward(&mid, n);
        y[r0 * d..(r0 + n) * d].copy_from_slice(&o);
        r0 += n;
    }
    y
}

/// GELU with the tanh approximation, with a real tanh (PyTorch's `gelu(approximate="tanh")`).
#[inline]
fn gelu_tanh_exact(x: f32) -> f32 {
    const K: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (K * (x + 0.044_715 * x * x * x)).tanh())
}
