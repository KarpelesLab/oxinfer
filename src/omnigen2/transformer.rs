//! `OmniGen2Transformer2DModel`: a Lumina-style single-stream diffusion transformer (32 blocks,
//! 2520 wide, 21 query / 7 key-value heads of 120) with refiner blocks per input stream.
//!
//! - The text (Qwen2.5-VL hidden states) goes through an RMSNorm + linear caption embedder and
//!   two unmodulated `context_refiner` blocks, on its own.
//! - The noisy target's 2x2 latent patches go through `x_embedder` and two `noise_refiner` blocks;
//!   each reference image's through `ref_image_patch_embedder` plus a learned per-image index
//!   embedding and two `ref_image_refiner` blocks, each image on its own.
//! - The 32 joint blocks run over `[text, reference images..., target]`.
//!
//! Blocks: sandwich RMSNorms around attention and the SwiGLU MLP, tanh gates and `1 + scale`
//! modulation from the timestep embedding (`LuminaRMSNormZero`), per-head RMSNorm on queries and
//! keys, and the 3-axis "Omni RoPE" (complex pairs, 40 dimensions per axis): text token `i` sits at
//! `(i, i, i)`; each image at `(s, row, col)` with `s` the text length plus the larger side of each
//! image before it, so every image has its own grid starting at (0, 0).

use super::weights::{Linear, Loader};
use crate::kernels::*;
use anyhow::{Result, bail};
use serde::Deserialize;
use std::path::Path;

fn d2() -> usize {
    2
}
fn d16() -> usize {
    16
}
fn d256() -> usize {
    256
}
fn deps() -> f32 {
    1e-5
}
fn d1000() -> f32 {
    1000.0
}

/// `transformer/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformerConfig {
    #[serde(default = "d2")]
    pub patch_size: usize,
    #[serde(default = "d16")]
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: Option<usize>,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_refiner_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    #[serde(default = "d256")]
    pub multiple_of: usize,
    #[serde(default)]
    pub ffn_dim_multiplier: Option<f32>,
    #[serde(default = "deps")]
    pub norm_eps: f32,
    pub axes_dim_rope: [usize; 3],
    pub axes_lens: [usize; 3],
    pub text_feat_dim: usize,
    #[serde(default = "d1000")]
    pub timestep_scale: f32,
}

impl TransformerConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        if !c.hidden_size.is_multiple_of(c.num_attention_heads) || !c.num_attention_heads.is_multiple_of(c.num_kv_heads) {
            bail!(
                "transformer: {} heads / {} kv heads do not divide {}",
                c.num_attention_heads,
                c.num_kv_heads,
                c.hidden_size
            );
        }
        if c.head_dim() != c.axes_dim_rope.iter().sum::<usize>() || c.axes_dim_rope.iter().any(|a| a % 2 != 0) {
            bail!("axes_dim_rope {:?} does not cover the head size {}", c.axes_dim_rope, c.head_dim());
        }
        Ok(c)
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn out(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }
    /// Features per patch token (`patch^2 * in_channels`).
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.in_channels
    }
    /// `LuminaFeedForward`'s inner width: `4 * hidden`, times the multiplier, rounded up to
    /// `multiple_of`.
    pub fn ffn_dim(&self) -> usize {
        let mut inner = 4 * self.hidden_size;
        if let Some(m) = self.ffn_dim_multiplier {
            inner = (m * inner as f32) as usize;
        }
        self.multiple_of * inner.div_ceil(self.multiple_of)
    }
    /// Width of the timestep embedding (`min(hidden, 1024)`).
    pub fn temb_dim(&self) -> usize {
        self.hidden_size.min(1024)
    }
}

struct Block<'a> {
    /// `norm1.linear` (modulated blocks): `[4 * hidden, temb]`.
    modulation: Option<Linear<'a>>,
    norm1: Vec<f32>,
    to_q: Linear<'a>,
    to_k: Linear<'a>,
    to_v: Linear<'a>,
    to_out: Linear<'a>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    norm2: Vec<f32>,
    ffn_norm1: Vec<f32>,
    ffn_norm2: Vec<f32>,
    w1: Linear<'a>,
    w2: Linear<'a>,
    w3: Linear<'a>,
}

impl<'a> Block<'a> {
    fn load(ld: &Loader<'a>, p: &str, cfg: &TransformerConfig, modulated: bool) -> Result<Self> {
        let d = cfg.hidden_size;
        let hd = cfg.head_dim();
        let kv = cfg.num_kv_heads * hd;
        let f = cfg.ffn_dim();
        let l = |n: &str, o: usize, i: usize| ld.linear(&format!("{p}.{n}"), o, i);
        let v = |n: &str, len: usize| ld.f32(&format!("{p}.{n}.weight"), len);
        Ok(Self {
            modulation: if modulated {
                Some(l("norm1.linear", 4 * d, cfg.temb_dim())?)
            } else {
                None
            },
            norm1: if modulated { v("norm1.norm", d)? } else { v("norm1", d)? },
            to_q: l("attn.to_q", d, d)?,
            to_k: l("attn.to_k", kv, d)?,
            to_v: l("attn.to_v", kv, d)?,
            to_out: l("attn.to_out.0", d, d)?,
            norm_q: v("attn.norm_q", hd)?,
            norm_k: v("attn.norm_k", hd)?,
            norm2: v("norm2", d)?,
            ffn_norm1: v("ffn_norm1", d)?,
            ffn_norm2: v("ffn_norm2", d)?,
            w1: l("feed_forward.linear_1", f, d)?,
            w2: l("feed_forward.linear_2", d, f)?,
            w3: l("feed_forward.linear_3", f, d)?,
        })
    }

    fn lins(&self) -> Vec<&Linear<'a>> {
        let mut v = vec![&self.to_q, &self.to_k, &self.to_v, &self.to_out, &self.w1, &self.w2, &self.w3];
        v.extend(self.modulation.as_ref());
        v
    }
}

/// RoPE cosines and sines, `head_dim / 2` per token.
pub struct CosSin {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

/// One image of the token sequence: `rows x cols` patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub rows: usize,
    pub cols: usize,
}

impl Grid {
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }
}

/// The per-step conditioning shared by every pass of a step: the timestep embedding and each
/// modulated block's `(scale_msa, gate_msa, scale_mlp, gate_mlp)` (`[4 * hidden]`).
pub struct StepCond {
    noise: Vec<Vec<f32>>,
    refs: Vec<Vec<f32>>,
    layers: Vec<Vec<f32>>,
    /// `norm_out`'s scale.
    out_scale: Vec<f32>,
}

pub struct Transformer<'a> {
    pub cfg: TransformerConfig,
    x_embedder: Linear<'a>,
    ref_embedder: Linear<'a>,
    t_lin1: Linear<'a>,
    t_lin2: Linear<'a>,
    cap_norm: Vec<f32>,
    cap_lin: Linear<'a>,
    noise_refiner: Vec<Block<'a>>,
    ref_refiner: Vec<Block<'a>>,
    context_refiner: Vec<Block<'a>>,
    layers: Vec<Block<'a>>,
    norm_out_lin1: Linear<'a>,
    norm_out_lin2: Linear<'a>,
    /// `[n, hidden]` learned embeddings added to the tokens of reference image `j`.
    image_index_embedding: Vec<f32>,
    n_index: usize,
    /// Inverse frequencies per RoPE axis.
    inv_freq: [Vec<f64>; 3],
}

impl<'a> Transformer<'a> {
    pub fn load(cfg: TransformerConfig, ld: &Loader<'a>) -> Result<Self> {
        let d = cfg.hidden_size;
        let te = cfg.temb_dim();
        let blocks = |p: &str, n: usize, m: bool| -> Result<Vec<Block<'a>>> {
            (0..n).map(|i| Block::load(ld, &format!("{p}.{i}"), &cfg, m)).collect()
        };
        let idx_name = "image_index_embedding";
        let info_len = {
            let full = format!("{}{idx_name}", ld.prefix);
            ld.files.info(&full).map(|i| i.shape.iter().product::<usize>()).unwrap_or(0)
        };
        if info_len == 0 || info_len % d != 0 {
            bail!("transformer: missing or malformed {idx_name}");
        }
        let inv = |dim: usize| -> Vec<f64> { (0..dim / 2).map(|i| 1.0 / 10000f64.powf((2 * i) as f64 / dim as f64)).collect() };
        let ax = cfg.axes_dim_rope;
        Ok(Self {
            x_embedder: ld.linear("x_embedder", d, cfg.patch_dim())?,
            ref_embedder: ld.linear("ref_image_patch_embedder", d, cfg.patch_dim())?,
            t_lin1: ld.linear("time_caption_embed.timestep_embedder.linear_1", te, 256)?,
            t_lin2: ld.linear("time_caption_embed.timestep_embedder.linear_2", te, te)?,
            cap_norm: ld.f32("time_caption_embed.caption_embedder.0.weight", cfg.text_feat_dim)?,
            cap_lin: ld.linear("time_caption_embed.caption_embedder.1", d, cfg.text_feat_dim)?,
            noise_refiner: blocks("noise_refiner", cfg.num_refiner_layers, true)?,
            ref_refiner: blocks("ref_image_refiner", cfg.num_refiner_layers, true)?,
            context_refiner: blocks("context_refiner", cfg.num_refiner_layers, false)?,
            layers: blocks("layers", cfg.num_layers, true)?,
            norm_out_lin1: ld.linear("norm_out.linear_1", d, te)?,
            norm_out_lin2: ld.linear("norm_out.linear_2", cfg.patch_size * cfg.patch_size * cfg.out(), d)?,
            image_index_embedding: ld.f32(idx_name, info_len)?,
            n_index: info_len / d,
            inv_freq: [inv(ax[0]), inv(ax[1]), inv(ax[2])],
            cfg,
        })
    }

    /// The most reference images the model has index embeddings for (5).
    pub fn max_refs(&self) -> usize {
        self.n_index
    }

    /// Weight bytes.
    pub fn bytes(&self) -> usize {
        let blocks = self
            .noise_refiner
            .iter()
            .chain(&self.ref_refiner)
            .chain(&self.context_refiner)
            .chain(&self.layers);
        blocks.flat_map(|b| b.lins()).map(|l| l.bytes()).sum::<usize>()
            + [
                &self.x_embedder,
                &self.ref_embedder,
                &self.t_lin1,
                &self.t_lin2,
                &self.cap_lin,
                &self.norm_out_lin1,
                &self.norm_out_lin2,
            ]
            .iter()
            .map(|l| l.bytes())
            .sum::<usize>()
    }

    /// `Timesteps(256, flip_sin_to_cos, shift 0, scale)` then the 2-layer MLP: `[temb_dim]`.
    pub fn temb(&self, t: f32) -> Vec<f32> {
        let half = 128;
        let mut e = vec![0f32; 256];
        for i in 0..half {
            let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            let a = self.cfg.timestep_scale * (t * f);
            e[i] = a.cos();
            e[half + i] = a.sin();
        }
        let mut h = self.t_lin1.forward_small(&e, 1);
        h.iter_mut().for_each(|v| *v = crate::sd::ops::silu(*v));
        self.t_lin2.forward_small(&h, 1)
    }

    /// The modulation of every block for timestep `t`.
    pub fn step_cond(&self, t: f32) -> StepCond {
        let temb = self.temb(t);
        let s: Vec<f32> = temb.iter().map(|&v| crate::sd::ops::silu(v)).collect();
        let m = |bs: &[Block]| bs.iter().map(|b| b.modulation.as_ref().unwrap().forward_small(&s, 1)).collect();
        StepCond {
            noise: m(&self.noise_refiner),
            refs: m(&self.ref_refiner),
            layers: m(&self.layers),
            out_scale: self.norm_out_lin1.forward_small(&s, 1),
        }
    }

    /// RoPE angles for 3-axis positions.
    pub fn rope(&self, pos: &[[usize; 3]]) -> Result<CosSin> {
        let half = self.cfg.head_dim() / 2;
        for p in pos {
            for a in 0..3 {
                if p[a] >= self.cfg.axes_lens[a] {
                    bail!(
                        "position {} on RoPE axis {a} exceeds the model's {} (prompt or images too large)",
                        p[a],
                        self.cfg.axes_lens[a]
                    );
                }
            }
        }
        let mut cos = vec![0f32; pos.len() * half];
        let mut sin = vec![0f32; pos.len() * half];
        for (t, p) in pos.iter().enumerate() {
            let mut j = t * half;
            for (ax, fr) in self.inv_freq.iter().enumerate() {
                for &f in fr {
                    let a = p[ax] as f64 * f;
                    cos[j] = a.cos() as f32;
                    sin[j] = a.sin() as f32;
                    j += 1;
                }
            }
        }
        Ok(CosSin { cos, sin })
    }

    /// Positions of an image grid at axis-0 position `s`.
    fn grid_pos(g: Grid, s: usize, out: &mut Vec<[usize; 3]>) {
        for r in 0..g.rows {
            for c in 0..g.cols {
                out.push([s, r, c]);
            }
        }
    }

    /// The text stream: caption embedder and context refiner over the `[l, text_feat_dim]` hidden
    /// states. Returns `[l, hidden]`.
    pub fn embed_text(&self, hidden: &[f32], l: usize) -> Result<Vec<f32>> {
        let tf = self.cfg.text_feat_dim;
        let mut hn = vec![0f32; l * tf];
        rmsnorm_f32w_rows(hidden, tf, &self.cap_norm, self.cfg.norm_eps, &mut hn);
        let mut x = self.cap_lin.forward(&hn, l);
        let pos: Vec<[usize; 3]> = (0..l).map(|i| [i, i, i]).collect();
        let cs = self.rope(&pos)?;
        for b in &self.context_refiner {
            self.block(b, &mut x, l, &cs, None);
        }
        Ok(x)
    }

    /// The noisy target's patch tokens (`[g.tokens(), patch_dim]`) through `x_embedder` and the
    /// noise refiner.
    pub fn embed_noise(&self, tokens: &[f32], g: Grid, sc: &StepCond) -> Result<Vec<f32>> {
        let n = g.tokens();
        let mut x = self.x_embedder.forward(tokens, n);
        let mut pos = Vec::with_capacity(n);
        Self::grid_pos(g, 0, &mut pos);
        let cs = self.rope(&pos)?;
        for (b, m) in self.noise_refiner.iter().zip(&sc.noise) {
            self.block(b, &mut x, n, &cs, Some(m));
        }
        Ok(x)
    }

    /// Reference image `j`'s patch tokens through the patch embedder, its index embedding and the
    /// reference refiner.
    pub fn embed_ref(&self, tokens: &[f32], g: Grid, j: usize, sc: &StepCond) -> Result<Vec<f32>> {
        if j >= self.n_index {
            bail!("OmniGen2 takes at most {} input images", self.n_index);
        }
        let d = self.cfg.hidden_size;
        let n = g.tokens();
        let mut x = self.ref_embedder.forward(tokens, n);
        let e = &self.image_index_embedding[j * d..(j + 1) * d];
        for_chunks(&mut x, d, n > 64, |_, r| r.iter_mut().zip(e).for_each(|(a, &b)| *a += b));
        let mut pos = Vec::with_capacity(n);
        Self::grid_pos(g, 0, &mut pos);
        let cs = self.rope(&pos)?;
        for (b, m) in self.ref_refiner.iter().zip(&sc.refs) {
            self.block(b, &mut x, n, &cs, Some(m));
        }
        Ok(x)
    }

    /// The joint sequence `[text, refs..., target]` (`[n, hidden]`) and its RoPE angles.
    fn joint_input(&self, text: &[f32], refs: &[(&[f32], Grid)], target: &[f32], tg: Grid) -> Result<(Vec<f32>, CosSin, usize)> {
        let d = self.cfg.hidden_size;
        let l = text.len() / d;
        let nr: usize = refs.iter().map(|r| r.1.tokens()).sum();
        let n0 = tg.tokens();
        let n = l + nr + n0;
        let mut pos: Vec<[usize; 3]> = (0..l).map(|i| [i, i, i]).collect();
        let mut x = Vec::with_capacity(n * d);
        x.extend_from_slice(text);
        let mut s = l;
        for (t, g) in refs {
            Self::grid_pos(*g, s, &mut pos);
            s += g.rows.max(g.cols);
            x.extend_from_slice(t);
        }
        Self::grid_pos(tg, s, &mut pos);
        x.extend_from_slice(target);
        let cs = self.rope(&pos)?;
        Ok((x, cs, n))
    }

    /// The joint blocks over `[text, refs..., target]` and the output layer: the velocity of the
    /// target's patches, `[target.tokens(), patch^2 * out_channels]`.
    pub fn joint(&self, text: &[f32], refs: &[(&[f32], Grid)], target: &[f32], tg: Grid, sc: &StepCond) -> Result<Vec<f32>> {
        let h = self.joint_hidden(text, refs, target, tg, sc)?;
        Ok(self.output(&h, tg.tokens(), sc))
    }

    /// The joint blocks alone: the target rows of the last block's hidden states,
    /// `[target.tokens(), hidden]` (the feature a [`crate::step_cache::TaylorSeer`] keeps);
    /// [`Transformer::output`] turns them into the velocity.
    pub fn joint_hidden(&self, text: &[f32], refs: &[(&[f32], Grid)], target: &[f32], tg: Grid, sc: &StepCond) -> Result<Vec<f32>> {
        let d = self.cfg.hidden_size;
        let n0 = tg.tokens();
        let (mut x, cs, n) = self.joint_input(text, refs, target, tg)?;
        for (b, m) in self.layers.iter().zip(&sc.layers) {
            self.block(b, &mut x, n, &cs, Some(m));
        }
        x.drain(..(n - n0) * d);
        Ok(x)
    }

    /// [`Transformer::joint`] under a [`TeaCache`](crate::step_cache::TeaCache) (the official
    /// transformer's `enable_teacache`): the first block's modulated input over the whole
    /// sequence decides whether the blocks run; when they do not, the residual of the last pass
    /// that ran them is added to the input instead. Returns the velocity and whether the blocks
    /// ran.
    #[allow(clippy::too_many_arguments)]
    pub fn joint_teacache(
        &self,
        text: &[f32],
        refs: &[(&[f32], Grid)],
        target: &[f32],
        tg: Grid,
        sc: &StepCond,
        cache: &mut crate::step_cache::TeaCache,
        first_or_last: bool,
    ) -> Result<(Vec<f32>, bool)> {
        let d = self.cfg.hidden_size;
        let n0 = tg.tokens();
        let (mut x, cs, n) = self.joint_input(text, refs, target, tg)?;
        // `layers[0].norm1(hidden_states, temb)`: RMSNorm times (1 + scale_msa).
        let mut modulated = vec![0f32; n * d];
        if let (Some(b), Some(m)) = (self.layers.first(), sc.layers.first()) {
            rmsnorm_f32w_rows(&x, d, &b.norm1, self.cfg.norm_eps, &mut modulated);
            scale_rows(&mut modulated, d, &m[..d]);
        }
        let ran = cache.should_calc(modulated, first_or_last);
        let h = match (ran, cache.residual()) {
            (false, Some(res)) if res.len() == n0 * d => target.iter().zip(res).map(|(&a, &b)| a + b).collect(),
            _ => {
                for (b, m) in self.layers.iter().zip(&sc.layers) {
                    self.block(b, &mut x, n, &cs, Some(m));
                }
                x.drain(..(n - n0) * d);
                cache.set_residual(x.iter().zip(target).map(|(&a, &b)| a - b).collect());
                x
            }
        };
        Ok((self.output(&h, n0, sc), ran))
    }

    /// `norm_out` over the target's final hidden states (`[n0, hidden]`): LayerNorm (no affine,
    /// eps 1e-6) * (1 + scale), then the projection to `[n0, patch^2 * out_channels]`.
    pub fn output(&self, xo: &[f32], n0: usize, sc: &StepCond) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        let mut y = vec![0f32; n0 * d];
        let sc_out = &sc.out_scale;
        for_chunks(&mut y, d, n0 > 16, |i, r| {
            let src = &xo[i * d..(i + 1) * d];
            let mean = src.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for j in 0..d {
                r[j] = ((src[j] as f64 - mean) * inv) as f32 * (1.0 + sc_out[j]);
            }
        });
        self.norm_out_lin2.forward(&y, n0)
    }

    /// One block over `x` (`[n, hidden]`), in place; `m` is its modulation when it has one.
    fn block(&self, b: &Block, x: &mut [f32], n: usize, cs: &CosSin, m: Option<&Vec<f32>>) {
        let cfg = &self.cfg;
        let d = cfg.hidden_size;
        let eps = cfg.norm_eps;
        let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim());
        let kvd = nkv * hd;
        let half = hd / 2;
        let chunk = |k: usize| m.map(|m| &m[k * d..(k + 1) * d]);
        // Attention input: RMSNorm, times (1 + scale_msa) when modulated.
        let mut xn = vec![0f32; n * d];
        rmsnorm_f32w_rows(x, d, &b.norm1, eps, &mut xn);
        if let Some(s) = chunk(0) {
            scale_rows(&mut xn, d, s);
        }
        let mut q = vec![0f32; n * d];
        let mut k = vec![0f32; n * kvd];
        let mut v = vec![0f32; n * kvd];
        Linear::forward_multi(&mut [(&b.to_q, &mut q[..]), (&b.to_k, &mut k[..]), (&b.to_v, &mut v[..])], &xn, n);
        drop(xn);
        {
            let _t = crate::sd::ops::timing::start(6);
            for (buf, w, heads) in [(&mut q, &b.norm_q, nh), (&mut k, &b.norm_k, nkv)] {
                for_chunks(buf, hd, true, |i, head| {
                    let row = i / heads;
                    rmsnorm_f32w_inplace(head, w, eps);
                    apply_rope_interleaved(head, &cs.cos[row * half..(row + 1) * half], &cs.sin[row * half..(row + 1) * half]);
                });
            }
        }
        let att = gqa_attention(&q, &k, &v, n, nh, nkv, hd);
        drop((q, k, v));
        let mut o = b.to_out.forward(&att, n);
        drop(att);
        rmsnorm_f32w_rows_inplace(&mut o, d, &b.norm2, eps);
        gated_add(x, &o, d, chunk(1));
        drop(o);
        // MLP, in row chunks that bound the hidden activations.
        let f = b.w1.n_out;
        let rows = ((256usize << 20) / (f * 4 * 2)).max(64);
        let mut r0 = 0;
        while r0 < n {
            let r = rows.min(n - r0);
            let xs = &mut x[r0 * d..(r0 + r) * d];
            let mut h = vec![0f32; r * d];
            rmsnorm_f32w_rows(xs, d, &b.ffn_norm1, eps, &mut h);
            if let Some(s) = chunk(2) {
                scale_rows(&mut h, d, s);
            }
            let mut a = vec![0f32; r * f];
            let mut g = vec![0f32; r * f];
            Linear::forward_multi(&mut [(&b.w1, &mut a[..]), (&b.w3, &mut g[..])], &h, r);
            silu_mul(&mut a, &g);
            drop(g);
            let mut y = b.w2.forward(&a, r);
            rmsnorm_f32w_rows_inplace(&mut y, d, &b.ffn_norm2, eps);
            gated_add(xs, &y, d, chunk(3));
            r0 += r;
        }
    }
}

/// `x[i] *= 1 + s` per row.
fn scale_rows(x: &mut [f32], d: usize, s: &[f32]) {
    for_chunks(x, d, x.len() > 64 * d, |_, r| r.iter_mut().zip(s).for_each(|(a, &b)| *a *= 1.0 + b));
}

/// `x += tanh(gate) * y` per row (`x += y` without a gate).
fn gated_add(x: &mut [f32], y: &[f32], d: usize, gate: Option<&[f32]>) {
    match gate {
        None => add_inplace(x, y),
        Some(g) => {
            let t: Vec<f32> = g.iter().map(|v| v.tanh()).collect();
            for_chunks(x, d, x.len() > 64 * d, |i, r| {
                for ((a, &b), &gg) in r.iter_mut().zip(&y[i * d..(i + 1) * d]).zip(&t) {
                    *a += gg * b;
                }
            });
        }
    }
}

/// Bidirectional attention with grouped key-value heads (query head `h` reads kv head
/// `h / (nh / nkv)`), scale `hd^-0.5`. Heads are zero-padded to a multiple of 64 (the queries
/// pre-scaled so the padded softmax scale gives the original one) for the GPU's flash kernels.
pub(crate) fn gqa_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, nkv: usize, hd: usize) -> Vec<f32> {
    let hp = hd.next_multiple_of(64);
    let g = nh / nkv;
    let qs = ((hp as f64).sqrt() / (hd as f64).sqrt()) as f32;
    let dp = nh * hp;
    let mut qp = vec![0f32; n * dp];
    let mut kp = vec![0f32; n * dp];
    let mut vp = vec![0f32; n * dp];
    for_chunks(&mut qp, dp, n > 16, |t, r| {
        for h in 0..nh {
            let src = &q[t * nh * hd + h * hd..][..hd];
            for (o, &s) in r[h * hp..h * hp + hd].iter_mut().zip(src) {
                *o = s * qs;
            }
        }
    });
    for (dst, src) in [(&mut kp, k), (&mut vp, v)] {
        for_chunks(dst, dp, n > 16, |t, r| {
            for h in 0..nh {
                r[h * hp..h * hp + hd].copy_from_slice(&src[t * nkv * hd + (h / g) * hd..][..hd]);
            }
        });
    }
    let o = crate::qwen_vision::attention_chunked(&qp, &kp, &vp, n, nh, hp);
    drop((qp, kp, vp));
    let mut out = vec![0f32; n * nh * hd];
    for_chunks(&mut out, nh * hd, n > 16, |t, r| {
        for h in 0..nh {
            r[h * hd..(h + 1) * hd].copy_from_slice(&o[t * dp + h * hp..][..hd]);
        }
    });
    out
}

/// OmniGen2's patchify (`c (h p1) (w p2) -> (h w) (p1 p2 c)`): channels-last latents
/// `[lh * lw, c]` into tokens `[(lh/2) * (lw/2), 4c]`, feature `(dy * 2 + dx) * c + ch`.
pub fn patchify(lat: &[f32], lh: usize, lw: usize, c: usize) -> Vec<f32> {
    let (ph, pw) = (lh / 2, lw / 2);
    let mut out = vec![0f32; ph * pw * 4 * c];
    for_chunks(&mut out, 4 * c, ph * pw > 64, |i, t| {
        let (y, x) = (i / pw, i % pw);
        for dy in 0..2 {
            for dx in 0..2 {
                let src = &lat[((2 * y + dy) * lw + 2 * x + dx) * c..][..c];
                t[(dy * 2 + dx) * c..(dy * 2 + dx + 1) * c].copy_from_slice(src);
            }
        }
    });
    out
}

/// The inverse of [`patchify`].
pub fn unpatchify(tok: &[f32], lh: usize, lw: usize, c: usize) -> Vec<f32> {
    let pw = lw / 2;
    let mut lat = vec![0f32; lh * lw * c];
    for_chunks(&mut lat, c, lh * lw > 64, |p, r| {
        let (y, x) = (p / lw, p % lw);
        let (dy, dx) = (y % 2, x % 2);
        r.copy_from_slice(&tok[((y / 2) * pw + x / 2) * 4 * c + (dy * 2 + dx) * c..][..c]);
    });
    lat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_roundtrip() {
        let (h, w, c) = (4, 6, 3);
        let z: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        let t = patchify(&z, h, w, c);
        // Token 0: pixel (0,0) channels, then (0,1), (1,0), (1,1).
        assert_eq!(&t[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&t[6..9], &[18.0, 19.0, 20.0]);
        assert_eq!(unpatchify(&t, h, w, c), z);
    }

    #[test]
    fn gqa_matches_naive() {
        let (n, nh, nkv, hd) = (5, 6, 2, 12);
        let mut rng = crate::sd::ops::Rng::new(1);
        let q = rng.normals(n * nh * hd);
        let k = rng.normals(n * nkv * hd);
        let v = rng.normals(n * nkv * hd);
        let o = gqa_attention(&q, &k, &v, n, nh, nkv, hd);
        for h in 0..nh {
            let kh = h / (nh / nkv);
            for i in 0..n {
                let s: Vec<f32> = (0..n)
                    .map(|j| {
                        (0..hd)
                            .map(|c| q[i * nh * hd + h * hd + c] * k[j * nkv * hd + kh * hd + c])
                            .sum::<f32>()
                            / (hd as f32).sqrt()
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = s.iter().map(|x| (x - mx).exp()).collect();
                let z: f32 = e.iter().sum();
                for c in 0..hd {
                    let want: f32 = (0..n).map(|j| e[j] / z * v[j * nkv * hd + kh * hd + c]).sum();
                    assert!((o[i * nh * hd + h * hd + c] - want).abs() < 1e-4);
                }
            }
        }
    }
}
