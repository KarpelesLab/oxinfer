//! The FLUX.1 rectified-flow transformer (diffusers `FluxTransformer2DModel`, BFL `Flux`):
//! 19 double-stream blocks (separate weights for text and image tokens, joint attention) and 38
//! single-stream blocks (one fused attention + MLP over the joint sequence), adaptive layer-norm
//! modulation from the timestep / guidance / pooled-text embedding, QK RMS norms and 3-axis RoPE.
//!
//! Weights load from a diffusers folder (`transformer/`, split `to_q`/`to_k`/`to_v`) or from the
//! original BFL layout (fused `qkv` / `linear1`, e.g. GGUF files), split into the same layers.

use super::rope::Rope;
use super::weights::{Dense, Loader, forward_many};
use crate::kernels::{for_chunks, gelu_tanh};
use crate::model::Quant;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// Architecture hyper-parameters (`transformer/config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub joint_dim: usize,
    pub pooled_dim: usize,
    pub guidance_embeds: bool,
    pub axes: Vec<usize>,
    pub theta: f64,
}

impl Config {
    pub fn hidden(&self) -> usize {
        self.heads * self.head_dim
    }

    fn from_json(c: &serde_json::Value) -> Result<Self> {
        let u = |k: &str, d: usize| c[k].as_u64().map(|v| v as usize).unwrap_or(d);
        if u("patch_size", 1) != 1 {
            bail!("patch_size {} is not supported", u("patch_size", 1));
        }
        let in_channels = u("in_channels", 64);
        Ok(Self {
            in_channels,
            out_channels: c["out_channels"].as_u64().map(|v| v as usize).unwrap_or(in_channels),
            num_layers: u("num_layers", 19),
            num_single_layers: u("num_single_layers", 38),
            heads: u("num_attention_heads", 24),
            head_dim: u("attention_head_dim", 128),
            joint_dim: u("joint_attention_dim", 4096),
            pooled_dim: u("pooled_projection_dim", 768),
            guidance_embeds: c["guidance_embeds"].as_bool().unwrap_or(false),
            axes: c["axes_dims_rope"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect())
                .unwrap_or_else(|| vec![16, 56, 56]),
            theta: 10000.0,
        })
    }
}

/// Tensor naming of the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    Diffusers,
    /// The original black-forest-labs names (`double_blocks.N.img_attn.qkv`, ...).
    Bfl,
}

/// A double-stream block: separate modulation, projections and MLPs for the image and text
/// tokens, one joint attention over `[text; image]`.
pub struct DoubleBlock<'a> {
    img_mod: Dense<'a>,
    txt_mod: Dense<'a>,
    img_qkv: Vec<Dense<'a>>,
    txt_qkv: Vec<Dense<'a>>,
    norms: [Vec<f32>; 4], // img q, img k, txt q, txt k
    img_out: Dense<'a>,
    txt_out: Dense<'a>,
    img_mlp: [Dense<'a>; 2],
    txt_mlp: [Dense<'a>; 2],
    pub heads: usize,
    pub head_dim: usize,
}

/// A single-stream block over the joint sequence: one modulation, q / k / v and the MLP input
/// from the same normalized activations, attention and MLP outputs projected together.
pub struct SingleBlock<'a> {
    modl: Dense<'a>,
    /// q, k, v, mlp-in: all reading the same normalized input.
    proj: Vec<Dense<'a>>,
    norms: [Vec<f32>; 2],
    out: Dense<'a>,
    pub heads: usize,
    pub head_dim: usize,
}

impl<'a> DoubleBlock<'a> {
    /// Block `i` (`transformer_blocks.{i}` / `double_blocks.{i}`) of a `heads` x `head_dim`
    /// model; the matrices are quantized as the loader says.
    pub fn load(ld: &Loader<'a>, layout: Layout, i: usize, heads: usize, head_dim: usize) -> Result<Self> {
        let d = heads * head_dim;
        let bfl = layout == Layout::Bfl;
        let (dp, bp) = (format!("transformer_blocks.{i}"), format!("double_blocks.{i}"));
        let p = |dn: &str, bn: &str| if bfl { format!("{bp}.{bn}") } else { format!("{dp}.{dn}") };
        let qkv = |dn: [&str; 3], bn: &str| -> Result<Vec<Dense<'a>>> {
            if bfl {
                ld.split(&p("", bn), &[d, d, d], true)
            } else {
                dn.iter().map(|x| ld.dense(&p(x, ""), true)).collect()
            }
        };
        let norm = |dn: &str, bn: &str| {
            ld.f32(&if bfl {
                format!("{bp}.{bn}.scale")
            } else {
                format!("{dp}.{dn}.weight")
            })
        };
        Ok(Self {
            img_mod: ld.dense(&p("norm1.linear", "img_mod.lin"), true)?,
            txt_mod: ld.dense(&p("norm1_context.linear", "txt_mod.lin"), true)?,
            img_qkv: qkv(["attn.to_q", "attn.to_k", "attn.to_v"], "img_attn.qkv")?,
            txt_qkv: qkv(["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"], "txt_attn.qkv")?,
            norms: [
                norm("attn.norm_q", "img_attn.norm.query_norm")?,
                norm("attn.norm_k", "img_attn.norm.key_norm")?,
                norm("attn.norm_added_q", "txt_attn.norm.query_norm")?,
                norm("attn.norm_added_k", "txt_attn.norm.key_norm")?,
            ],
            img_out: ld.dense(&p("attn.to_out.0", "img_attn.proj"), true)?,
            txt_out: ld.dense(&p("attn.to_add_out", "txt_attn.proj"), true)?,
            img_mlp: [
                ld.dense(&p("ff.net.0.proj", "img_mlp.0"), true)?,
                ld.dense(&p("ff.net.2", "img_mlp.2"), true)?,
            ],
            txt_mlp: [
                ld.dense(&p("ff_context.net.0.proj", "txt_mlp.0"), true)?,
                ld.dense(&p("ff_context.net.2", "txt_mlp.2"), true)?,
            ],
            heads,
            head_dim,
        })
    }

    /// Bytes of weights held.
    pub fn bytes(&self) -> usize {
        self.lins().iter().map(|l| l.bytes()).sum()
    }

    /// The block's linear layers (weight streaming: their memory ranges).
    pub fn lins(&self) -> Vec<&Dense<'a>> {
        [&self.img_mod, &self.txt_mod, &self.img_out, &self.txt_out]
            .into_iter()
            .chain(&self.img_qkv)
            .chain(&self.txt_qkv)
            .chain(&self.img_mlp)
            .chain(&self.txt_mlp)
            .collect()
    }

    /// Updates the image (`hi`, `[n_img, d]`) and text (`ht`, `[n_txt, d]`) streams. `st` is
    /// SiLU of the conditioning vector; `rope` covers `n_txt + n_img` tokens, text first.
    pub fn forward(&self, hi: &mut [f32], n_img: usize, ht: &mut [f32], n_txt: usize, st: &[f32], rope: &Rope) {
        let d = self.heads * self.head_dim;
        let nt = n_txt + n_img;
        let mi = self.img_mod.forward_vec(st);
        let mt = self.txt_mod.forward_vec(st);
        let (mi, mt) = (chunks(&mi, d), chunks(&mt, d));
        // [0] shift, [1] scale, [2] gate (attention), [3..6] the same for the MLP.
        let xi = ln_mod(hi, n_img, mi[1], mi[0]);
        let xt = ln_mod(ht, n_txt, mt[1], mt[0]);
        let (mut q, mut k, mut v) = (vec![0f32; nt * d], vec![0f32; nt * d], vec![0f32; nt * d]);
        {
            let (qt, qi) = q.split_at_mut(n_txt * d);
            let (kt, ki) = k.split_at_mut(n_txt * d);
            let (vt, vi) = v.split_at_mut(n_txt * d);
            forward_many_into(&self.txt_qkv, &xt, n_txt, &mut [qt, kt, vt]);
            forward_many_into(&self.img_qkv, &xi, n_img, &mut [qi, ki, vi]);
            rms_heads(qt, &self.norms[2], self.head_dim);
            rms_heads(kt, &self.norms[3], self.head_dim);
            rms_heads(qi, &self.norms[0], self.head_dim);
            rms_heads(ki, &self.norms[1], self.head_dim);
        }
        drop((xi, xt));
        rope.apply(&mut q, self.heads);
        rope.apply(&mut k, self.heads);
        let a = crate::sd::ops::attention(&q, &k, &v, nt, nt, self.heads, self.head_dim, false);
        drop((q, k, v));
        let (at, ai) = a.split_at(n_txt * d);
        gated_add(hi, &self.img_out.forward(ai, n_img), mi[2]);
        gated_add(ht, &self.txt_out.forward(at, n_txt), mt[2]);
        drop(a);
        let xi = ln_mod(hi, n_img, mi[4], mi[3]);
        gated_add(hi, &mlp(&self.img_mlp, &xi, n_img), mi[5]);
        drop(xi);
        let xt = ln_mod(ht, n_txt, mt[4], mt[3]);
        gated_add(ht, &mlp(&self.txt_mlp, &xt, n_txt), mt[5]);
    }
}

impl<'a> SingleBlock<'a> {
    /// Block `i` (`single_transformer_blocks.{i}` / `single_blocks.{i}`).
    pub fn load(ld: &Loader<'a>, layout: Layout, i: usize, heads: usize, head_dim: usize) -> Result<Self> {
        let d = heads * head_dim;
        let bfl = layout == Layout::Bfl;
        let (dp, bp) = (format!("single_transformer_blocks.{i}"), format!("single_blocks.{i}"));
        let p = |dn: &str, bn: &str| if bfl { format!("{bp}.{bn}") } else { format!("{dp}.{dn}") };
        let proj = if bfl {
            let (r, _) = ld.shape2(&format!("{bp}.linear1.weight"))?;
            if r <= 3 * d {
                bail!("{bp}.linear1 has {r} rows, expected more than {}", 3 * d);
            }
            ld.split(&format!("{bp}.linear1"), &[d, d, d, r - 3 * d], true)?
        } else {
            ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"]
                .iter()
                .map(|x| ld.dense(&format!("{dp}.{x}"), true))
                .collect::<Result<_>>()?
        };
        let norm = |dn: &str, bn: &str| {
            ld.f32(&if bfl {
                format!("{bp}.{bn}.scale")
            } else {
                format!("{dp}.{dn}.weight")
            })
        };
        Ok(Self {
            modl: ld.dense(&p("norm.linear", "modulation.lin"), true)?,
            proj,
            norms: [norm("attn.norm_q", "norm.query_norm")?, norm("attn.norm_k", "norm.key_norm")?],
            out: ld.dense(&p("proj_out", "linear2"), true)?,
            heads,
            head_dim,
        })
    }

    pub fn bytes(&self) -> usize {
        self.lins().iter().map(|l| l.bytes()).sum()
    }

    /// The block's linear layers.
    pub fn lins(&self) -> Vec<&Dense<'a>> {
        [&self.modl, &self.out].into_iter().chain(&self.proj).collect()
    }

    /// Updates the joint sequence `h` (`[nt, d]`, text first).
    pub fn forward(&self, h: &mut [f32], nt: usize, st: &[f32], rope: &Rope) {
        let d = self.heads * self.head_dim;
        let m = self.modl.forward_vec(st);
        let m = chunks(&m, d);
        let x = ln_mod(h, nt, m[1], m[0]);
        let mut ys = forward_many(&self.proj.iter().collect::<Vec<_>>(), &x, nt);
        drop(x);
        let mut mlp_in = ys.pop().unwrap();
        let v = ys.pop().unwrap();
        let mut k = ys.pop().unwrap();
        let mut q = ys.pop().unwrap();
        rms_heads(&mut q, &self.norms[0], self.head_dim);
        rms_heads(&mut k, &self.norms[1], self.head_dim);
        rope.apply(&mut q, self.heads);
        rope.apply(&mut k, self.heads);
        let a = crate::sd::ops::attention(&q, &k, &v, nt, nt, self.heads, self.head_dim, false);
        drop((q, k, v));
        let dm = mlp_in.len() / nt;
        let w = d + dm;
        let mut cat = vec![0f32; nt * w];
        gelu_inplace(&mut mlp_in);
        for_chunks(&mut cat, w, true, |r, row| {
            row[..d].copy_from_slice(&a[r * d..(r + 1) * d]);
            row[d..].copy_from_slice(&mlp_in[r * dm..(r + 1) * dm]);
        });
        drop((a, mlp_in));
        gated_add(h, &self.out.forward(&cat, nt), m[2]);
    }
}

pub struct FluxTransformer {
    pub cfg: Config,
    pub layout: Layout,
    x_emb: Dense<'static>,
    ctx_emb: Dense<'static>,
    time: [Dense<'static>; 2],
    guide: Option<[Dense<'static>; 2]>,
    text: [Dense<'static>; 2],
    /// Final modulation: (scale, shift) halves.
    norm_out: [Dense<'static>; 2],
    proj_out: Dense<'static>,
    pub doubles: Vec<DoubleBlock<'static>>,
    pub singles: Vec<SingleBlock<'static>>,
    /// Bytes of weights held (borrowed from the mapped checkpoint or quantized copies).
    pub weight_bytes: usize,
    // Declared last: dropped after every layer borrowing it.
    _files: Box<ModelFiles>,
}

/// Optional `OXINFER_FLUX_LAYERS=D,S`: run only the first D double and S single blocks (for
/// reference checks against a truncated diffusers model).
fn layer_limit() -> Option<(usize, usize)> {
    let v = std::env::var("OXINFER_FLUX_LAYERS").ok()?;
    let (a, b) = v.split_once(',')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

impl FluxTransformer {
    /// Loads `dir` (a diffusers `transformer/` folder) or a single GGUF file in BFL layout.
    /// Float weights are quantized to `quant` (bf16 keeps them as stored, borrowed from the
    /// mapping); GGUF weights run in their own format whatever `quant` says.
    pub fn load(path: &Path, quant: Quant) -> Result<Self> {
        let gguf = crate::gguf::is_gguf(path);
        // Mapped (zero-copy where the stored format runs as is; the page cache streams it).
        let files = Box::new(ModelFiles::open(path, cfg!(target_arch = "wasm32"))?);
        // SAFETY: the box's heap block does not move, and every layer borrowing it is dropped
        // before `_files` (field order).
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let layout = if fs.has("img_in.weight") { Layout::Bfl } else { Layout::Diffusers };
        let ld = Loader { files: fs, quant };
        let mut cfg = if !gguf && crate::fsx::exists(&path.join("config.json")) {
            Config::from_json(&serde_json::from_slice(&crate::fsx::read(&path.join("config.json"))?)?)?
        } else {
            Self::infer_config(&ld, layout)?
        };
        if let Some((d, s)) = layer_limit() {
            cfg.num_layers = cfg.num_layers.min(d);
            cfg.num_single_layers = cfg.num_single_layers.min(s);
        }
        let d = cfg.hidden();
        let bfl = layout == Layout::Bfl;
        let n = |dn: &str, bn: &str| if bfl { bn.to_string() } else { dn.to_string() };
        let top = |dn: &str, bn: &str| ld.dense(&n(dn, bn), false);
        let x_emb = top("x_embedder", "img_in")?;
        let ctx_emb = top("context_embedder", "txt_in")?;
        let time = [
            top("time_text_embed.timestep_embedder.linear_1", "time_in.in_layer")?,
            top("time_text_embed.timestep_embedder.linear_2", "time_in.out_layer")?,
        ];
        let guide = if cfg.guidance_embeds {
            Some([
                top("time_text_embed.guidance_embedder.linear_1", "guidance_in.in_layer")?,
                top("time_text_embed.guidance_embedder.linear_2", "guidance_in.out_layer")?,
            ])
        } else {
            None
        };
        let text = [
            top("time_text_embed.text_embedder.linear_1", "vector_in.in_layer")?,
            top("time_text_embed.text_embedder.linear_2", "vector_in.out_layer")?,
        ];
        let norm_out = {
            let mut v = ld.split(&n("norm_out.linear", "final_layer.adaLN_modulation.1"), &[d, d], false)?;
            let (b, a) = (v.pop().unwrap(), v.pop().unwrap());
            // diffusers stores (scale, shift); BFL (shift, scale).
            if bfl { [b, a] } else { [a, b] }
        };
        let proj_out = top("proj_out", "final_layer.linear")?;
        let (heads, hd) = (cfg.heads, cfg.head_dim);
        let doubles = (0..cfg.num_layers)
            .map(|i| DoubleBlock::load(&ld, layout, i, heads, hd))
            .collect::<Result<Vec<_>>>()?;
        let singles = (0..cfg.num_single_layers)
            .map(|i| SingleBlock::load(&ld, layout, i, heads, hd))
            .collect::<Result<Vec<_>>>()?;
        let weight_bytes = [&x_emb, &ctx_emb, &proj_out]
            .into_iter()
            .chain(time.iter())
            .chain(text.iter())
            .chain(norm_out.iter())
            .chain(guide.iter().flatten())
            .map(|l| l.bytes())
            .sum::<usize>()
            + doubles.iter().map(|b| b.bytes()).sum::<usize>()
            + singles.iter().map(|b| b.bytes()).sum::<usize>();
        Ok(Self {
            cfg,
            layout,
            x_emb,
            ctx_emb,
            time,
            guide,
            text,
            norm_out,
            proj_out,
            doubles,
            singles,
            weight_bytes,
            _files: files,
        })
    }

    /// The configuration of a checkpoint without `config.json` (GGUF / BFL single files).
    fn infer_config(ld: &Loader, layout: Layout) -> Result<Config> {
        let (pre_d, pre_s, xe, ce, ve) = match layout {
            Layout::Bfl => ("double_blocks", "single_blocks", "img_in", "txt_in", "vector_in.in_layer"),
            Layout::Diffusers => (
                "transformer_blocks",
                "single_transformer_blocks",
                "x_embedder",
                "context_embedder",
                "time_text_embed.text_embedder.linear_1",
            ),
        };
        let count = |p: &str, probe: &str| (0..).take_while(|i| ld.has(&format!("{p}.{i}.{probe}"))).count();
        let (num_layers, num_single_layers) = match layout {
            Layout::Bfl => (count(pre_d, "img_mod.lin.weight"), count(pre_s, "modulation.lin.weight")),
            Layout::Diffusers => (count(pre_d, "norm1.linear.weight"), count(pre_s, "norm.linear.weight")),
        };
        let (d, in_channels) = ld.shape2(&format!("{xe}.weight")).context("not a FLUX transformer")?;
        let (_, joint_dim) = ld.shape2(&format!("{ce}.weight"))?;
        let (_, pooled_dim) = ld.shape2(&format!("{ve}.weight"))?;
        let head_dim = 128;
        if !d.is_multiple_of(head_dim) || num_layers + num_single_layers == 0 {
            bail!("unexpected FLUX transformer shape (hidden {d}, {num_layers} + {num_single_layers} blocks)");
        }
        let guidance_embeds = match layout {
            Layout::Bfl => ld.has("guidance_in.in_layer.weight"),
            Layout::Diffusers => ld.has("time_text_embed.guidance_embedder.linear_1.weight"),
        };
        Ok(Config {
            in_channels,
            out_channels: in_channels,
            num_layers,
            num_single_layers,
            heads: d / head_dim,
            head_dim,
            joint_dim,
            pooled_dim,
            guidance_embeds,
            axes: vec![16, 56, 56],
            theta: 10000.0,
        })
    }

    /// The conditioning vector: timestep (+ guidance) embeddings plus the pooled-text projection.
    /// `t` is the timestep in [0, 1000] (sigma * 1000), `guidance` the distilled guidance scale.
    pub fn temb(&self, t: f32, guidance: Option<f32>, pooled: &[f32]) -> Vec<f32> {
        let mlp = |l: &[Dense; 2], x: &[f32]| {
            let mut h = l[0].forward_vec(x);
            h.iter_mut().for_each(|v| *v = silu(*v));
            l[1].forward_vec(&h)
        };
        let mut e = mlp(&self.time, &timestep_proj(t, 256));
        if let (Some(g), Some(gs)) = (&self.guide, guidance) {
            add(&mut e, &mlp(g, &timestep_proj(gs * 1000.0, 256)));
        }
        add(&mut e, &mlp(&self.text, pooled));
        e
    }

    /// One denoising call. `img` is `[n_img, in_channels]` (packed latent tokens: the noisy
    /// latent first, then any reference tokens), `txt` `[n_txt, joint_dim]` (T5 states),
    /// `rope` covers the `n_txt + n_img` tokens in that order. Returns `[n_img, out_channels]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &[f32],
        n_img: usize,
        txt: &[f32],
        n_txt: usize,
        pooled: &[f32],
        t: f32,
        guidance: Option<f32>,
        rope: &Rope,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.hidden();
        let nt = n_txt + n_img;
        assert_eq!(rope.rows, nt, "rope table rows");
        let temb = self.temb(t, guidance, pooled);
        let st: Vec<f32> = temb.iter().map(|&v| silu(v)).collect();
        let mut hi = self.x_emb.forward(img, n_img);
        let mut ht = self.ctx_emb.forward(txt, n_txt);
        for b in &self.doubles {
            b.forward(&mut hi, n_img, &mut ht, n_txt, &st, rope);
        }
        // Single-stream blocks over the joint sequence [text; image].
        let mut h = ht;
        h.extend_from_slice(&hi);
        drop(hi);
        for b in &self.singles {
            b.forward(&mut h, nt, &st, rope);
        }
        let hi = &h[n_txt * d..];
        let m = [self.norm_out[0].forward_vec(&st), self.norm_out[1].forward_vec(&st)];
        let x = ln_mod(hi, n_img, &m[0], &m[1]);
        self.proj_out.forward(&x, n_img)
    }
}

fn forward_many_into(ls: &[Dense], x: &[f32], rows: usize, outs: &mut [&mut [f32]]) {
    let n_in = ls[0].n_in;
    {
        let mut jobs: Vec<(&crate::model::Lin, usize, &mut [f32])> =
            ls.iter().zip(outs.iter_mut()).map(|(l, y)| (&l.w, l.n_out, &mut **y)).collect();
        crate::model::linear_multi(&mut jobs, n_in, x, rows);
    }
    for (l, y) in ls.iter().zip(outs.iter_mut()) {
        if let Some(b) = &l.b {
            for_chunks(y, l.n_out, rows > 64, |_, r| {
                for (v, &bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            });
        }
    }
}

fn mlp(l: &[Dense; 2], x: &[f32], rows: usize) -> Vec<f32> {
    let mut h = l[0].forward(x, rows);
    gelu_inplace(&mut h);
    l[1].forward(&h, rows)
}

fn gelu_inplace(x: &mut [f32]) {
    for_chunks(x, 1 << 16, x.len() > 1 << 16, |_, c| c.iter_mut().for_each(|v| *v = gelu_tanh(*v)));
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn add(a: &mut [f32], b: &[f32]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

fn chunks(v: &[f32], d: usize) -> Vec<&[f32]> {
    v.chunks(d).collect()
}

/// `x += gate * y` per row (`gate` has one value per channel).
fn gated_add(x: &mut [f32], y: &[f32], gate: &[f32]) {
    let d = gate.len();
    for_chunks(x, d, x.len() > d * 16, |r, row| {
        let yr = &y[r * d..(r + 1) * d];
        for ((a, &b), &g) in row.iter_mut().zip(yr).zip(gate) {
            *a += g * b;
        }
    });
}

/// Layer norm without affine parameters (eps 1e-6), then `* (1 + scale) + shift` per channel.
pub fn ln_mod(x: &[f32], rows: usize, scale: &[f32], shift: &[f32]) -> Vec<f32> {
    let d = scale.len();
    debug_assert_eq!(x.len(), rows * d);
    let mut y = vec![0f32; rows * d];
    for_chunks(&mut y, d, rows > 16, |r, out| {
        let row = &x[r * d..(r + 1) * d];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = (1.0 / (var + 1e-6).sqrt()) as f32;
        let m = mean as f32;
        for i in 0..d {
            out[i] = (row[i] - m) * inv * (1.0 + scale[i]) + shift[i];
        }
    });
    y
}

/// RMS norm (eps 1e-6, weight `w`) of every `dh`-wide head of `x`.
fn rms_heads(x: &mut [f32], w: &[f32], dh: usize) {
    for_chunks(x, dh * 32, x.len() > dh * 64, |_, c| {
        for h in c.chunks_mut(dh) {
            let ms = h.iter().map(|&v| v * v).sum::<f32>() / dh as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for (v, &g) in h.iter_mut().zip(w) {
                *v = *v * inv * g;
            }
        }
    });
}

/// Sinusoidal timestep features in f32 as diffusers' `Timesteps(dim, flip_sin_to_cos=True,
/// downscale_freq_shift=0)`: `[cos(t * f_i), sin(t * f_i)]`, `f_i = 10000^(-i / (dim / 2))`.
pub fn timestep_proj(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut e = vec![0f32; dim];
    let l = -(10000f32.ln());
    for i in 0..half {
        let f = (l * i as f32 / half as f32).exp();
        let a = (t * f) as f64;
        e[i] = a.cos() as f32;
        e[half + i] = a.sin() as f32;
    }
    e
}
