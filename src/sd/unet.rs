//! The Stable Diffusion 1.x / 2.x denoiser (diffusers `UNet2DConditionModel`): ResNet blocks with
//! timestep conditioning, spatial transformers with self- and cross-attention to the text
//! embedding, and skip connections from the down path to the up path. Channel counts are taken
//! from the weight shapes, so the same code loads SD 1.5, SD 2.1, SD-Turbo and SDXL (several
//! transformer layers per block, none at the top level, and the added size / pooled-text
//! conditioning of `xl.rs`).

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
    /// SDXL's added conditioning (`addition_embed_type: "text_time"`).
    pub add: Option<super::xl::AddEmbed>,
    /// Whole passes on the accelerator ([`crate::gpu::unet`]): the device session, once made.
    gpu: std::sync::Mutex<Option<Box<dyn crate::gpu::unet::Session>>>,
    /// Whether [`Unet::forward_batch`] tries the whole-pass device plan (default on, or
    /// `OXINFER_SD_GPU_PASS=0` for off).
    gpu_pass: std::sync::atomic::AtomicBool,
}

/// One image of a [`Unet::forward_batch`]: latents, text context and SDXL's added embedding.
pub struct BatchItem<'a> {
    pub latent: &'a [f32],
    pub ctx: &'a [f32],
    pub added: Option<&'a [f32]>,
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
        if !cfg["class_embed_type"].is_null() {
            bail!("UNet class_embed_type = {} is not supported", cfg["class_embed_type"]);
        }
        let add = match cfg["addition_embed_type"].as_str() {
            None => None,
            Some("text_time") => Some(super::xl::AddEmbed::load(&t, &cfg)?),
            Some(other) => bail!("UNet addition_embed_type = {other} is not supported"),
        };
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
            add,
            gpu: std::sync::Mutex::new(None),
            gpu_pass: std::sync::atomic::AtomicBool::new(std::env::var("OXINFER_SD_GPU_PASS").map_or(true, |v| v != "0")),
        })
    }

    /// How much the down path shrinks each side (2 per downsampler): the latent's sides must
    /// be multiples of it.
    pub fn down_factor(&self) -> usize {
        1 << self.downs.iter().filter(|b| b.down.is_some()).count()
    }

    /// Predicted noise (or v) for `latent` (`[h*w, in_channels]`) at timestep `timestep`, given the
    /// text embedding `ctx` (`[m, cross_dim]`).
    pub fn forward(&self, latent: &[f32], h: usize, w: usize, timestep: f32, ctx: &[f32], m: usize) -> Vec<f32> {
        self.forward_added(latent, h, w, timestep, ctx, m, None)
    }

    /// [`Unet::forward`] with SDXL's added embedding ([`super::xl::AddEmbed::forward`]: pooled
    /// text and size conditioning), which joins the time embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_added(
        &self,
        latent: &[f32],
        h: usize,
        w: usize,
        timestep: f32,
        ctx: &[f32],
        m: usize,
        added: Option<&[f32]>,
    ) -> Vec<f32> {
        let te = timestep_embedding(timestep, self.time_dim, self.flip_sin_to_cos, self.freq_shift);
        let mut temb = self.t1.forward(&te, 1);
        silu_inplace(&mut temb);
        let mut temb = self.t2.forward(&temb, 1);
        if let Some(a) = added {
            add_inplace(&mut temb, a);
        }
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

    /// Whether whole passes run on the accelerator when it can ([`Unet::forward_batch`]).
    pub fn set_gpu_pass(&self, on: bool) {
        self.gpu_pass.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// See [`Unet::set_gpu_pass`].
    pub fn gpu_pass(&self) -> bool {
        self.gpu_pass.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// [`Unet::forward_added`] for several images at the same timestep (the guidance branches of a
    /// step), all `[h*w, in_channels]` latents with `[m, cross_dim]` contexts: in one batch on the
    /// accelerator when it runs whole passes, else one after the other.
    pub fn forward_batch(&self, items: &[BatchItem], h: usize, w: usize, timestep: f32, m: usize) -> Vec<Vec<f32>> {
        if self.gpu_pass()
            && let Some(v) = self.forward_device(items, h, w, timestep, m)
        {
            return v;
        }
        items
            .iter()
            .map(|it| self.forward_added(it.latent, h, w, timestep, it.ctx, m, it.added))
            .collect()
    }

    /// The time embedding after its SiLU (what every ResNet projects), per image.
    fn time_embeddings(&self, items: &[BatchItem], timestep: f32) -> Vec<f32> {
        let te = timestep_embedding(timestep, self.time_dim, self.flip_sin_to_cos, self.freq_shift);
        let mut temb = self.t1.forward(&te, 1);
        silu_inplace(&mut temb);
        let temb = self.t2.forward(&temb, 1);
        let mut all = Vec::with_capacity(items.len() * temb.len());
        for it in items {
            let mut t = temb.clone();
            if let Some(a) = it.added {
                add_inplace(&mut t, a);
            }
            silu_inplace(&mut t);
            all.extend_from_slice(&t);
        }
        all
    }

    fn forward_device(&self, items: &[BatchItem], h: usize, w: usize, timestep: f32, m: usize) -> Option<Vec<Vec<f32>>> {
        let be = crate::gpu::active()?;
        // Checking device ops against the CPU (OXINFER_GPU_CHECK) and exact mode keep the per-op
        // path.
        if super::ops::gpu_check::on() || crate::gpu::exact() {
            return None;
        }
        if items.is_empty() || items.iter().any(|it| it.ctx.len() != m * self.cross_dim) {
            return None;
        }
        let _t = super::ops::timing::start(super::ops::timing::PASS);
        let temb = self.time_embeddings(items, timestep);
        let g = self.plan_graph(h, w, &temb, items.len())?;
        let mut sess = self.gpu.lock().unwrap_or_else(|e| e.into_inner());
        if sess.is_none() {
            *sess = Some(be.unet_session()?);
        }
        let s = sess.as_mut()?;
        let gi: Vec<crate::gpu::unet::Item> = items
            .iter()
            .map(|it| crate::gpu::unet::Item {
                latent: it.latent,
                ctx: it.ctx,
                ctx_len: m,
            })
            .collect();
        let mut out = vec![Vec::new(); items.len()];
        s.forward(&g, &gi, &mut out).then_some(out)
    }

    /// One pass as a device graph (`temb`: the activated time embedding of each of `nb` images).
    fn plan_graph(&self, h: usize, w: usize, temb: &[f32], nb: usize) -> Option<crate::gpu::unet::Graph<'_>> {
        use crate::gpu::unet::{Graph, Op};
        let mut g = Graph {
            tensors: vec![[h, w, self.in_channels]],
            ops: Vec::new(),
            input: 0,
            output: 0,
            ctx_dim: self.cross_dim,
        };
        let (mut h, mut w) = (h, w);
        let mut c = self.conv_in.cout;
        let mut x = new_tensor(&mut g, [h, w, c]);
        g.ops.push(conv_op(&self.conv_in, 0, x, None, None, false)?);
        let mut skips = vec![x];
        for b in &self.downs {
            for (j, r) in b.resnets.iter().enumerate() {
                x = r.plan(&mut g, x, temb, nb)?;
                c = r.cout;
                if let Some(a) = b.attns.get(j) {
                    x = a.plan(&mut g, x)?;
                }
                skips.push(x);
            }
            if let Some(d) = &b.down {
                (h, w) = (h / 2, w / 2);
                let y = new_tensor(&mut g, [h, w, c]);
                g.ops.push(Op::Conv {
                    x,
                    out: y,
                    w: d.plan_lin()?,
                    k: d.k,
                    stride: 2,
                    pad: d.pad_before(),
                    up: false,
                    pre: None,
                    bias: None,
                    bias_per_image: false,
                    add: None,
                });
                x = y;
                skips.push(x);
            }
        }
        x = self.mid1.plan(&mut g, x, temb, nb)?;
        x = self.mid_attn.plan(&mut g, x)?;
        x = self.mid2.plan(&mut g, x, temb, nb)?;
        for b in &self.ups {
            for (j, r) in b.resnets.iter().enumerate() {
                let s = skips.pop()?;
                let cs = g.tensors[s][2];
                let cat = new_tensor(&mut g, [h, w, c + cs]);
                g.ops.push(Op::Concat { a: x, b: s, out: cat });
                x = r.plan(&mut g, cat, temb, nb)?;
                c = r.cout;
                if let Some(a) = b.attns.get(j) {
                    x = a.plan(&mut g, x)?;
                }
            }
            if let Some(u) = &b.up {
                (h, w) = (h * 2, w * 2);
                let y = new_tensor(&mut g, [h, w, c]);
                g.ops.push(conv_op(u, x, y, None, None, true)?);
                x = y;
            }
        }
        let y = new_tensor(&mut g, [h, w, self.conv_out.cout]);
        g.ops
            .push(conv_op(&self.conv_out, x, y, Some((self.norm_out.plan_norm(), true)), None, false)?);
        g.output = y;
        Some(g)
    }
}

pub(super) fn new_tensor(g: &mut crate::gpu::unet::Graph, s: [usize; 3]) -> usize {
    g.tensors.push(s);
    g.tensors.len() - 1
}

/// A same-size convolution op (`up`: of the 2x upsampled input).
pub(super) fn conv_op<'a>(
    c: &'a Conv,
    x: usize,
    out: usize,
    pre: Option<(crate::gpu::unet::Norm<'a>, bool)>,
    add: Option<usize>,
    up: bool,
) -> Option<crate::gpu::unet::Op<'a>> {
    Some(crate::gpu::unet::Op::Conv {
        x,
        out,
        w: c.plan_lin()?,
        k: c.k,
        stride: 1,
        pad: c.pad_before(),
        up,
        pre,
        bias: None,
        bias_per_image: false,
        add,
    })
}

impl Resnet {
    /// This block's ops on tensor `x` of `g` (`temb`: the activated time embedding of each of
    /// `nb` images); returns the output tensor.
    pub(super) fn plan<'a>(&'a self, g: &mut crate::gpu::unet::Graph<'a>, x: usize, temb: &[f32], nb: usize) -> Option<usize> {
        use crate::gpu::unet::Op;
        let [h, w, _] = g.tensors[x];
        let t1 = new_tensor(g, [h, w, self.cout]);
        let out = new_tensor(g, [h, w, self.cout]);
        let short = self.short.as_ref().map(|_| new_tensor(g, [h, w, self.cout]));
        if let (Some(s), Some(ts)) = (&self.short, short) {
            g.ops.push(Op::Conv {
                x,
                out: ts,
                w: s.plan_lin()?,
                k: 1,
                stride: 1,
                pad: 0,
                up: false,
                pre: None,
                bias: None,
                bias_per_image: false,
                add: None,
            });
        }
        let c1 = self.c1.plan_lin()?;
        // The first convolution's bias plus the time-embedding projection, per image.
        let bias = match &self.temb {
            Some(l) => {
                let mut b = l.forward(temb, nb);
                if let Some(cb) = c1.bias {
                    for r in b.chunks_mut(self.cout) {
                        add_inplace(r, &cb[..self.cout]);
                    }
                }
                Some(std::borrow::Cow::Owned(b))
            }
            None => None,
        };
        g.ops.push(Op::Conv {
            x,
            out: t1,
            w: c1,
            k: self.c1.k,
            stride: 1,
            pad: self.c1.pad_before(),
            up: false,
            pre: Some((self.n1.plan_norm(), true)),
            bias_per_image: bias.is_some(),
            bias,
            add: None,
        });
        g.ops.push(Op::Conv {
            x: t1,
            out,
            w: self.c2.plan_lin()?,
            k: self.c2.k,
            stride: 1,
            pad: self.c2.pad_before(),
            up: false,
            pre: Some((self.n2.plan_norm(), true)),
            bias: None,
            bias_per_image: false,
            add: Some(short.unwrap_or(x)),
        });
        Some(out)
    }
}

impl Attn {
    fn plan(&self) -> Option<crate::gpu::unet::Attn<'_>> {
        Some(crate::gpu::unet::Attn {
            q: self.q.plan_lin()?,
            k: self.k.plan_lin()?,
            v: self.v.plan_lin()?,
            o: self.o.plan_lin()?,
            heads: self.heads,
        })
    }
}

impl SpatialTransformer {
    fn plan<'a>(&'a self, g: &mut crate::gpu::unet::Graph<'a>, x: usize) -> Option<usize> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            blocks.push(crate::gpu::unet::Block {
                n1: b.n1.plan_norm(),
                a1: b.a1.plan()?,
                n2: b.n2.plan_norm(),
                a2: b.a2.plan()?,
                n3: b.n3.plan_norm(),
                ff1: b.ff1.plan_lin()?,
                ff2: b.ff2.plan_lin()?,
            });
        }
        let t = crate::gpu::unet::Transformer {
            norm: self.norm.plan_norm(),
            proj_in: self.proj_in.plan_lin()?,
            blocks,
            proj_out: self.proj_out.plan_lin()?,
        };
        let s = g.tensors[x];
        let out = new_tensor(g, s);
        g.ops.push(crate::gpu::unet::Op::Transformer { x, out, t });
        Some(out)
    }
}
