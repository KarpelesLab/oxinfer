//! The Stable Diffusion 1.x / 2.x denoiser (diffusers `UNet2DConditionModel`): ResNet blocks with
//! timestep conditioning, spatial transformers with self- and cross-attention to the text
//! embedding, and skip connections from the down path to the up path. Channel counts are taken
//! from the weight shapes, so the same code loads SD 1.5, SD 2.1 and SD-Turbo.

use super::ops::{
    Conv, GroupNorm, LayerNorm, Linear, Pre, Tensors, attention, concat_channels, gelu_erf, silu_inplace, timestep_embedding, upsample2x,
};
use crate::kernels::{add_inplace, for_chunks};
use anyhow::{Context, Result, bail};
use std::path::Path;

pub struct Resnet {
    n1: GroupNorm,
    c1: Conv,
    temb: Option<Linear>,
    n2: GroupNorm,
    c2: Conv,
    short: Option<Linear>,
    pub cout: usize,
}

impl Resnet {
    pub fn load(t: &Tensors, p: &str, eps: f32) -> Result<Self> {
        let c1 = Conv::load(t, &format!("{p}.conv1"))?;
        let cout = c1.cout;
        Ok(Self {
            n1: GroupNorm::load(t, &format!("{p}.norm1"), 32, eps)?,
            c1,
            temb: if t.has(&format!("{p}.time_emb_proj.weight")) {
                Some(Linear::load(t, &format!("{p}.time_emb_proj"))?)
            } else {
                None
            },
            n2: GroupNorm::load(t, &format!("{p}.norm2"), 32, eps)?,
            c2: Conv::load(t, &format!("{p}.conv2"))?,
            short: if t.has(&format!("{p}.conv_shortcut.weight")) {
                Some(Linear::load(t, &format!("{p}.conv_shortcut"))?)
            } else if t.has(&format!("{p}.nin_shortcut.weight")) {
                // Taming-transformers naming (the Janus / LlamaGen VQ decoder).
                Some(Linear::load(t, &format!("{p}.nin_shortcut"))?)
            } else {
                None
            },
            cout,
        })
    }

    /// `x` is `[h*w, cin]`; `temb` is the (already SiLU-activated) time embedding.
    pub fn forward(&self, x: &[f32], h: usize, w: usize, temb: Option<&[f32]>) -> Vec<f32> {
        let n = h * w;
        // Each norm + SiLU is fused into the convolution that follows it (on the device).
        let (mut y, _, _) = self.c1.forward_pre(x, h, w, 1, Pre::Group(&self.n1, true));
        if let (Some(l), Some(te)) = (&self.temb, temb) {
            let add = l.forward(te, 1);
            for_chunks(&mut y, self.cout, true, |_, r| add_inplace(r, &add));
        }
        let (mut y, _, _) = self.c2.forward_pre(&y, h, w, 1, Pre::Group(&self.n2, true));
        match &self.short {
            Some(s) => add_inplace(&mut y, &s.forward(x, n)),
            None => add_inplace(&mut y, x),
        }
        y
    }
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
}

impl Attn {
    fn load(t: &Tensors, p: &str, heads: usize) -> Result<Self> {
        Ok(Self {
            q: Linear::load(t, &format!("{p}.to_q"))?,
            k: Linear::load(t, &format!("{p}.to_k"))?,
            v: Linear::load(t, &format!("{p}.to_v"))?,
            o: Linear::load(t, &format!("{p}.to_out.0"))?,
            heads,
        })
    }

    /// Queries from `pre(x)` (`[n, dim]`), keys and values from `ctx` (`[m, ctx_dim]`; `pre(x)`
    /// itself for self-attention). On the device the norm is fused into each projection;
    /// otherwise `x` is normalized once.
    fn forward(&self, x: &[f32], n: usize, pre: Pre, ctx: Option<(&[f32], usize)>) -> Vec<f32> {
        let (q, k, v);
        if self.q.on_device(n) {
            q = self.q.forward_pre(x, n, pre);
            (k, v) = match ctx {
                Some((c, m)) => (self.k.forward(c, m), self.v.forward(c, m)),
                None => (self.k.forward_pre(x, n, pre), self.v.forward_pre(x, n, pre)),
            };
        } else {
            let xn = pre.apply(x, n);
            q = self.q.forward(&xn, n);
            (k, v) = match ctx {
                Some((c, m)) => (self.k.forward(c, m), self.v.forward(c, m)),
                None => (self.k.forward(&xn, n), self.v.forward(&xn, n)),
            };
        }
        let m = ctx.map_or(n, |(_, m)| m);
        let d = self.q.n_out;
        let a = attention(&q, &k, &v, n, m, self.heads, d / self.heads, false);
        self.o.forward(&a, n)
    }
}

struct Block {
    n1: LayerNorm,
    a1: Attn,
    n2: LayerNorm,
    a2: Attn,
    n3: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

/// diffusers `Transformer2DModel` with one or more `BasicTransformerBlock`s.
pub struct SpatialTransformer {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<Block>,
    proj_out: Linear,
}

impl SpatialTransformer {
    fn load(t: &Tensors, p: &str, heads: usize) -> Result<Self> {
        let mut blocks = Vec::new();
        while t.has(&format!("{p}.transformer_blocks.{}.attn1.to_q.weight", blocks.len())) {
            let b = format!("{p}.transformer_blocks.{}", blocks.len());
            blocks.push(Block {
                n1: LayerNorm::load(t, &format!("{b}.norm1"), 1e-5)?,
                a1: Attn::load(t, &format!("{b}.attn1"), heads)?,
                n2: LayerNorm::load(t, &format!("{b}.norm2"), 1e-5)?,
                a2: Attn::load(t, &format!("{b}.attn2"), heads)?,
                n3: LayerNorm::load(t, &format!("{b}.norm3"), 1e-5)?,
                ff1: Linear::load(t, &format!("{b}.ff.net.0.proj"))?,
                ff2: Linear::load(t, &format!("{b}.ff.net.2"))?,
            });
        }
        if blocks.is_empty() {
            bail!("{p}: no transformer blocks");
        }
        Ok(Self {
            norm: GroupNorm::load(t, &format!("{p}.norm"), 32, 1e-6)?,
            // A 1x1 conv (SD 1.x) and a linear layer (SD 2.x) are the same thing channels-last.
            proj_in: Linear::load(t, &format!("{p}.proj_in"))?,
            blocks,
            proj_out: Linear::load(t, &format!("{p}.proj_out"))?,
        })
    }

    fn forward(&self, x: &[f32], n: usize, ctx: &[f32], m: usize) -> Vec<f32> {
        let mut h = self.proj_in.forward_pre(x, n, Pre::Group(&self.norm, false));
        for b in &self.blocks {
            let a = b.a1.forward(&h, n, Pre::Layer(&b.n1), None);
            add_inplace(&mut h, &a);
            let a = b.a2.forward(&h, n, Pre::Layer(&b.n2), Some((ctx, m)));
            add_inplace(&mut h, &a);
            // GEGLU: the projection's first half is the value, the second half its gate.
            let g = b.ff1.forward_pre(&h, n, Pre::Layer(&b.n3));
            let inner = b.ff1.n_out / 2;
            let mut gated = vec![0f32; n * inner];
            for_chunks(&mut gated, inner, true, |i, r| {
                let row = &g[i * 2 * inner..(i + 1) * 2 * inner];
                for j in 0..inner {
                    r[j] = row[j] * gelu_erf(row[inner + j]);
                }
            });
            add_inplace(&mut h, &b.ff2.forward(&gated, n));
        }
        let mut out = self.proj_out.forward(&h, n);
        add_inplace(&mut out, x);
        out
    }
}

struct DownBlock {
    resnets: Vec<Resnet>,
    attns: Vec<SpatialTransformer>,
    down: Option<Conv>,
}

struct UpBlock {
    resnets: Vec<Resnet>,
    attns: Vec<SpatialTransformer>,
    up: Option<Conv>,
}

pub struct Unet {
    conv_in: Conv,
    t1: Linear,
    t2: Linear,
    downs: Vec<DownBlock>,
    mid1: Resnet,
    mid_attn: SpatialTransformer,
    mid2: Resnet,
    ups: Vec<UpBlock>,
    norm_out: GroupNorm,
    conv_out: Conv,
    time_dim: usize,
    flip_sin_to_cos: bool,
    freq_shift: f32,
    pub in_channels: usize,
    pub sample_size: usize,
    pub cross_dim: usize,
}

fn count(t: &Tensors, f: impl Fn(usize) -> String) -> usize {
    let mut n = 0;
    while t.has(&f(n)) {
        n += 1;
    }
    n
}

impl Unet {
    /// `dir` holds `config.json`; `t` is the opened weight file (with any LoRA updates attached).
    pub fn load(dir: &Path, t: Tensors) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        let n_blocks = cfg["block_out_channels"].as_array().context("block_out_channels")?.len();
        // diffusers names the head count `attention_head_dim` for these models (int or per block).
        let heads_cfg = if cfg["num_attention_heads"].is_null() {
            &cfg["attention_head_dim"]
        } else {
            &cfg["num_attention_heads"]
        };
        let heads: Vec<usize> = match heads_cfg {
            serde_json::Value::Number(n) => vec![n.as_u64().unwrap() as usize; n_blocks],
            serde_json::Value::Array(a) => a.iter().map(|v| v.as_u64().unwrap_or(8) as usize).collect(),
            _ => vec![8; n_blocks],
        };
        let eps = cfg["norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        for (k, want) in [("class_embed_type", None), ("addition_embed_type", None)] {
            if cfg[k].as_str() != want {
                bail!("UNet {k} = {} is not supported (SDXL-style conditioning)", cfg[k]);
            }
        }
        let mut downs = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let p = format!("down_blocks.{i}");
            let nr = count(&t, |j| format!("{p}.resnets.{j}.conv1.weight"));
            let na = count(&t, |j| format!("{p}.attentions.{j}.proj_in.weight"));
            downs.push(DownBlock {
                resnets: (0..nr)
                    .map(|j| Resnet::load(&t, &format!("{p}.resnets.{j}"), eps))
                    .collect::<Result<_>>()?,
                attns: (0..na)
                    .map(|j| SpatialTransformer::load(&t, &format!("{p}.attentions.{j}"), heads[i]))
                    .collect::<Result<_>>()?,
                down: if t.has(&format!("{p}.downsamplers.0.conv.weight")) {
                    Some(Conv::load(&t, &format!("{p}.downsamplers.0.conv"))?)
                } else {
                    None
                },
            });
        }
        let mut ups = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let p = format!("up_blocks.{i}");
            let nr = count(&t, |j| format!("{p}.resnets.{j}.conv1.weight"));
            let na = count(&t, |j| format!("{p}.attentions.{j}.proj_in.weight"));
            let h = heads[n_blocks - 1 - i];
            ups.push(UpBlock {
                resnets: (0..nr)
                    .map(|j| Resnet::load(&t, &format!("{p}.resnets.{j}"), eps))
                    .collect::<Result<_>>()?,
                attns: (0..na)
                    .map(|j| SpatialTransformer::load(&t, &format!("{p}.attentions.{j}"), h))
                    .collect::<Result<_>>()?,
                up: if t.has(&format!("{p}.upsamplers.0.conv.weight")) {
                    Some(Conv::load(&t, &format!("{p}.upsamplers.0.conv"))?)
                } else {
                    None
                },
            });
        }
        let conv_in = Conv::load(&t, "conv_in")?;
        let t1 = Linear::load(&t, "time_embedding.linear_1")?;
        Ok(Self {
            in_channels: conv_in.cin,
            time_dim: t1.n_in,
            conv_in,
            t1,
            t2: Linear::load(&t, "time_embedding.linear_2")?,
            downs,
            mid1: Resnet::load(&t, "mid_block.resnets.0", eps)?,
            mid_attn: SpatialTransformer::load(&t, "mid_block.attentions.0", heads[n_blocks - 1])?,
            mid2: Resnet::load(&t, "mid_block.resnets.1", eps)?,
            ups,
            norm_out: GroupNorm::load(&t, "conv_norm_out", 32, eps)?,
            conv_out: Conv::load(&t, "conv_out")?,
            flip_sin_to_cos: cfg["flip_sin_to_cos"].as_bool().unwrap_or(true),
            freq_shift: cfg["freq_shift"].as_f64().unwrap_or(0.0) as f32,
            sample_size: cfg["sample_size"].as_u64().unwrap_or(64) as usize,
            cross_dim: cfg["cross_attention_dim"].as_u64().unwrap_or(768) as usize,
        })
    }

    /// Predicted noise (or v) for `latent` (`[h*w, in_channels]`) at timestep `timestep`, given the
    /// text embedding `ctx` (`[m, cross_dim]`).
    pub fn forward(&self, latent: &[f32], h: usize, w: usize, timestep: f32, ctx: &[f32], m: usize) -> Vec<f32> {
        let te = timestep_embedding(timestep, self.time_dim, self.flip_sin_to_cos, self.freq_shift);
        let mut temb = self.t1.forward(&te, 1);
        silu_inplace(&mut temb);
        let mut temb = self.t2.forward(&temb, 1);
        // Every ResNet applies SiLU to the embedding before its projection.
        silu_inplace(&mut temb);
        let temb = Some(&temb[..]);

        let (mut x, mut h, mut w) = self.conv_in.forward(latent, h, w, 1);
        let mut c = self.conv_in.cout;
        let mut skips: Vec<(Vec<f32>, usize)> = vec![(x.clone(), c)];
        for b in &self.downs {
            for (j, r) in b.resnets.iter().enumerate() {
                x = r.forward(&x, h, w, temb);
                c = r.cout;
                if let Some(a) = b.attns.get(j) {
                    x = a.forward(&x, h * w, ctx, m);
                }
                skips.push((x.clone(), c));
            }
            if let Some(d) = &b.down {
                let (y, oh, ow) = d.forward(&x, h, w, 2);
                (x, h, w) = (y, oh, ow);
                skips.push((x.clone(), c));
            }
        }
        x = self.mid1.forward(&x, h, w, temb);
        x = self.mid_attn.forward(&x, h * w, ctx, m);
        x = self.mid2.forward(&x, h, w, temb);
        for b in &self.ups {
            for (j, r) in b.resnets.iter().enumerate() {
                let (s, cs) = skips.pop().expect("skip connections exhausted");
                let cat = concat_channels(&x, c, &s, cs, h * w);
                x = r.forward(&cat, h, w, temb);
                c = r.cout;
                if let Some(a) = b.attns.get(j) {
                    x = a.forward(&x, h * w, ctx, m);
                }
            }
            if let Some(u) = &b.up {
                let up = upsample2x(&x, h, w, c);
                (h, w) = (h * 2, w * 2);
                x = u.forward(&up, h, w, 1).0;
            }
        }
        self.conv_out.forward_pre(&x, h, w, 1, Pre::Group(&self.norm_out, true)).0
    }
}
