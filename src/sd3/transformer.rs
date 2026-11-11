//! The Stable Diffusion 3 / 3.5 transformer (diffusers `SD3Transformer2DModel`, Stability AI's
//! `MMDiT` / `MMDiT-X`): latents cut into 2x2 patches with a learned-size sinusoidal position
//! table cropped around its centre, text tokens projected to the same width, then joint blocks
//! with separate weights for the two streams and one attention over `[text; image]`. Every norm
//! is modulated by the conditioning vector (timestep + pooled text). SD 3.5 adds RMS norms on
//! the queries and keys, and SD 3.5 medium (MMDiT-X) a second, image-only attention in its first
//! blocks. The last block only reads the text stream (`context_pre_only`).
//!
//! Weights load from a diffusers `transformer/` folder or from a GGUF file in Stability AI's
//! original layout (`joint_blocks.N.x_block.attn.qkv`, as in city96's SD 3.5 quants): fused
//! `qkv` rows are split into separate layers without copying.

use crate::flux::transformer::{forward_many_into, gated_add, ln_mod, mlp, rms_heads, silu, timestep_proj};
use crate::flux::weights::{Dense, Loader, forward_many};
use crate::kernels::for_chunks;
use crate::model::Quant;
use crate::safetensors::ModelFiles;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// Architecture hyper-parameters (`transformer/config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub num_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Width of the text states given to the model (T5's 4096).
    pub joint_dim: usize,
    /// Width of the pooled text embedding (CLIP-L + CLIP-G: 2048).
    pub pooled_dim: usize,
    /// Side of the stored position table, in patches.
    pub pos_embed_max_size: usize,
    /// Blocks with a second, image-only attention (SD 3.5 medium: the first 13).
    pub dual_attention_layers: Vec<usize>,
    /// RMS norms on queries and keys (SD 3.5).
    pub qk_norm: bool,
}

impl Config {
    pub fn hidden(&self) -> usize {
        self.heads * self.head_dim
    }

    fn from_json(c: &serde_json::Value) -> Result<Self> {
        let u = |k: &str, d: usize| c[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let in_channels = u("in_channels", 16);
        let cfg = Self {
            in_channels,
            out_channels: u("out_channels", in_channels),
            patch_size: u("patch_size", 2),
            num_layers: u("num_layers", 18),
            heads: u("num_attention_heads", 18),
            head_dim: u("attention_head_dim", 64),
            joint_dim: u("joint_attention_dim", 4096),
            pooled_dim: u("pooled_projection_dim", 2048),
            pos_embed_max_size: u("pos_embed_max_size", 96),
            dual_attention_layers: c["dual_attention_layers"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect())
                .unwrap_or_default(),
            qk_norm: match c["qk_norm"].as_str() {
                None => false,
                Some("rms_norm") => true,
                Some(o) => bail!("qk_norm {o} is not supported"),
            },
        };
        if u("caption_projection_dim", cfg.hidden()) != cfg.hidden() {
            bail!(
                "caption_projection_dim {} differs from the hidden size {}",
                u("caption_projection_dim", 0),
                cfg.hidden()
            );
        }
        Ok(cfg)
    }
}

/// Tensor naming of the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    Diffusers,
    /// Stability AI's original names (`joint_blocks.N.x_block.attn.qkv`, ...).
    Sai,
}

/// One stream's attention projections: q, k, v, the optional RMS norm weights of q and k.
struct Qkv<'a> {
    qkv: Vec<Dense<'a>>,
    norms: Option<[Vec<f32>; 2]>,
}

impl<'a> Qkv<'a> {
    /// `dn`: the diffusers names of q / k / v and of the two norms; `sai`: the prefix holding
    /// `qkv`, `ln_q`, `ln_k` in the original layout.
    fn load(ld: &Loader<'a>, d: usize, qk_norm: bool, diffusers: Option<[String; 5]>, sai: &str) -> Result<Self> {
        let (qkv, names) = match diffusers {
            Some([q, k, v, nq, nk]) => (
                [q, k, v].iter().map(|n| ld.dense(n, true)).collect::<Result<Vec<_>>>()?,
                [format!("{nq}.weight"), format!("{nk}.weight")],
            ),
            None => (
                ld.split(&format!("{sai}.qkv"), &[d, d, d], true)?,
                [format!("{sai}.ln_q.weight"), format!("{sai}.ln_k.weight")],
            ),
        };
        let norms = if qk_norm {
            Some([ld.f32(&names[0])?, ld.f32(&names[1])?])
        } else {
            None
        };
        Ok(Self { qkv, norms })
    }

    fn norm(&self, q: &mut [f32], k: &mut [f32], head_dim: usize) {
        if let Some([nq, nk]) = &self.norms {
            rms_heads(q, nq, head_dim);
            rms_heads(k, nk, head_dim);
        }
    }
}

/// A joint block: modulated norms, projections and MLPs per stream, one attention over
/// `[text; image]`, optionally a second attention over the image tokens alone.
pub struct JointBlock<'a> {
    /// 6 (or 9 with the second attention) modulation vectors of the image stream.
    x_mod: Dense<'a>,
    /// 6 modulation vectors of the text stream; 2 (scale, shift) in a `context_pre_only` block.
    c_mod: Dense<'a>,
    /// The pre-only modulation is stored (shift, scale) in the original layout.
    c_mod_shift_first: bool,
    x_attn: Qkv<'a>,
    c_attn: Qkv<'a>,
    x_out: Dense<'a>,
    /// The text stream's output projection and MLP (None: `context_pre_only`).
    c_tail: Option<(Dense<'a>, [Dense<'a>; 2])>,
    /// The image-only attention and its output projection.
    attn2: Option<(Qkv<'a>, Dense<'a>)>,
    x_mlp: [Dense<'a>; 2],
    pub heads: usize,
    pub head_dim: usize,
}

impl<'a> JointBlock<'a> {
    /// Block `i` (`transformer_blocks.{i}` / `joint_blocks.{i}`).
    pub fn load(ld: &Loader<'a>, layout: Layout, i: usize, cfg: &Config) -> Result<Self> {
        let d = cfg.hidden();
        let sai = layout == Layout::Sai;
        let (dp, sx, sc) = (
            format!("transformer_blocks.{i}"),
            format!("joint_blocks.{i}.x_block"),
            format!("joint_blocks.{i}.context_block"),
        );
        let pre_only = i + 1 == cfg.num_layers;
        let dual = cfg.dual_attention_layers.contains(&i);
        let x = |dn: &str, sn: &str| if sai { format!("{sx}.{sn}") } else { format!("{dp}.{dn}") };
        let c = |dn: &str, sn: &str| if sai { format!("{sc}.{sn}") } else { format!("{dp}.{dn}") };
        let dnames = |n: [&str; 5]| (!sai).then(|| n.map(|s| format!("{dp}.{s}")));
        let x_mod = ld.dense(&x("norm1.linear", "adaLN_modulation.1"), true)?;
        let c_mod = ld.dense(&c("norm1_context.linear", "adaLN_modulation.1"), true)?;
        let want = (if dual { 9 } else { 6 } * d, if pre_only { 2 } else { 6 } * d);
        if (x_mod.n_out, c_mod.n_out) != want {
            bail!(
                "block {i}: modulation layers have {} / {} outputs, expected {} / {}",
                x_mod.n_out,
                c_mod.n_out,
                want.0,
                want.1
            );
        }
        let c_tail = if pre_only {
            None
        } else {
            Some((
                ld.dense(&c("attn.to_add_out", "attn.proj"), true)?,
                [
                    ld.dense(&c("ff_context.net.0.proj", "mlp.fc1"), true)?,
                    ld.dense(&c("ff_context.net.2", "mlp.fc2"), true)?,
                ],
            ))
        };
        let attn2 = if dual {
            Some((
                Qkv::load(
                    ld,
                    d,
                    cfg.qk_norm,
                    dnames(["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.norm_q", "attn2.norm_k"]),
                    &format!("{sx}.attn2"),
                )?,
                ld.dense(&x("attn2.to_out.0", "attn2.proj"), true)?,
            ))
        } else {
            None
        };
        Ok(Self {
            x_mod,
            c_mod,
            c_mod_shift_first: sai,
            x_attn: Qkv::load(
                ld,
                d,
                cfg.qk_norm,
                dnames(["attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k"]),
                &format!("{sx}.attn"),
            )?,
            c_attn: Qkv::load(
                ld,
                d,
                cfg.qk_norm,
                dnames([
                    "attn.add_q_proj",
                    "attn.add_k_proj",
                    "attn.add_v_proj",
                    "attn.norm_added_q",
                    "attn.norm_added_k",
                ]),
                &format!("{sc}.attn"),
            )?,
            x_out: ld.dense(&x("attn.to_out.0", "attn.proj"), true)?,
            c_tail,
            attn2,
            x_mlp: [
                ld.dense(&x("ff.net.0.proj", "mlp.fc1"), true)?,
                ld.dense(&x("ff.net.2", "mlp.fc2"), true)?,
            ],
            heads: cfg.heads,
            head_dim: cfg.head_dim,
        })
    }

    /// The block's linear layers.
    pub fn lins(&self) -> Vec<&Dense<'a>> {
        let mut v: Vec<&Dense<'a>> = vec![&self.x_mod, &self.c_mod, &self.x_out];
        v.extend(&self.x_attn.qkv);
        v.extend(&self.c_attn.qkv);
        v.extend(&self.x_mlp);
        if let Some((o, m)) = &self.c_tail {
            v.push(o);
            v.extend(m);
        }
        if let Some((a, o)) = &self.attn2 {
            v.extend(&a.qkv);
            v.push(o);
        }
        v
    }

    /// Bytes of weights held.
    pub fn bytes(&self) -> usize {
        self.lins().iter().map(|l| l.bytes()).sum()
    }

    /// Updates the image (`hi`, `[n_img, d]`) and text (`ht`, `[n_txt, d]`) streams. `st` is
    /// SiLU of the conditioning vector. A `context_pre_only` block leaves `ht` as it was.
    pub fn forward(&self, hi: &mut [f32], n_img: usize, ht: &mut [f32], n_txt: usize, st: &[f32]) {
        let d = self.heads * self.head_dim;
        let nt = n_txt + n_img;
        let mi = self.x_mod.forward_vec(st);
        let mt = self.c_mod.forward_vec(st);
        let (mi, mt): (Vec<&[f32]>, Vec<&[f32]>) = (mi.chunks(d).collect(), mt.chunks(d).collect());
        // Image: [0] shift, [1] scale, [2] gate (attention), [3..6] the same for the MLP,
        // [6..9] for the second attention.
        let xi = ln_mod(hi, n_img, mi[1], mi[0]);
        let xi2 = self.attn2.as_ref().map(|_| ln_mod(hi, n_img, mi[7], mi[6]));
        let xt = match (&self.c_tail, self.c_mod_shift_first) {
            (Some(_), _) | (None, true) => ln_mod(ht, n_txt, mt[1], mt[0]),
            (None, false) => ln_mod(ht, n_txt, mt[0], mt[1]),
        };
        let (mut q, mut k, mut v) = (vec![0f32; nt * d], vec![0f32; nt * d], vec![0f32; nt * d]);
        {
            let (qt, qi) = q.split_at_mut(n_txt * d);
            let (kt, ki) = k.split_at_mut(n_txt * d);
            let (vt, vi) = v.split_at_mut(n_txt * d);
            forward_many_into(&self.c_attn.qkv, &xt, n_txt, &mut [&mut *qt, &mut *kt, vt]);
            forward_many_into(&self.x_attn.qkv, &xi, n_img, &mut [&mut *qi, &mut *ki, vi]);
            self.c_attn.norm(qt, kt, self.head_dim);
            self.x_attn.norm(qi, ki, self.head_dim);
        }
        drop((xi, xt));
        let a = crate::sd::ops::attention(&q, &k, &v, nt, nt, self.heads, self.head_dim, false);
        drop((q, k, v));
        let (at, ai) = a.split_at(n_txt * d);
        gated_add(hi, &self.x_out.forward(ai, n_img), mi[2]);
        if let (Some((attn, out)), Some(x2)) = (&self.attn2, &xi2) {
            let mut ys = forward_many(&attn.qkv.iter().collect::<Vec<_>>(), x2, n_img);
            let v = ys.pop().unwrap();
            let mut k = ys.pop().unwrap();
            let mut q = ys.pop().unwrap();
            attn.norm(&mut q, &mut k, self.head_dim);
            let a2 = crate::sd::ops::attention(&q, &k, &v, n_img, n_img, self.heads, self.head_dim, false);
            gated_add(hi, &out.forward(&a2, n_img), mi[8]);
        }
        drop(xi2);
        let xi = ln_mod(hi, n_img, mi[4], mi[3]);
        gated_add(hi, &mlp(&self.x_mlp, &xi, n_img), mi[5]);
        drop(xi);
        if let Some((out, ff)) = &self.c_tail {
            gated_add(ht, &out.forward(at, n_txt), mt[2]);
            let xt = ln_mod(ht, n_txt, mt[4], mt[3]);
            gated_add(ht, &mlp(ff, &xt, n_txt), mt[5]);
        }
    }
}

pub struct Sd3Transformer {
    pub cfg: Config,
    pub layout: Layout,
    /// The patch convolution as a `[d, c * p * p]` matrix (features ordered channel, row,
    /// column) and its bias.
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    /// Name of the position table `[1, max * max, d]`.
    pos: &'static str,
    ctx_emb: Dense<'static>,
    time: [Dense<'static>; 2],
    text: [Dense<'static>; 2],
    /// Final modulation: (scale, shift) halves.
    norm_out: [Dense<'static>; 2],
    proj_out: Dense<'static>,
    pub blocks: Vec<JointBlock<'static>>,
    /// Bytes of weights held (borrowed from the mapped checkpoint or quantized copies).
    pub weight_bytes: usize,
    // Declared last: dropped after every layer borrowing it.
    files: Box<ModelFiles>,
}

impl Sd3Transformer {
    /// Loads `path`: a diffusers `transformer/` folder or a single GGUF file in the original
    /// layout. Float weights are quantized to `quant` (bf16 keeps them as stored, borrowed from
    /// the mapping); GGUF weights run in their own format whatever `quant` says.
    pub fn load(path: &Path, quant: Quant) -> Result<Self> {
        let gguf = crate::gguf::is_gguf(path);
        let mut files = Box::new(ModelFiles::open(path, cfg!(target_arch = "wasm32"))?);
        // Single-file checkpoints keep the transformer under `model.diffusion_model.`.
        files.rename(|n| n.strip_prefix("model.diffusion_model.").map(str::to_string));
        // SAFETY: the box's heap block does not move, and every layer borrowing it is dropped
        // before `files` (field order).
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let layout = if fs.has("x_embedder.proj.weight") {
            Layout::Sai
        } else {
            Layout::Diffusers
        };
        let sai = layout == Layout::Sai;
        let ld = Loader { files: fs, quant };
        let cfg = if !gguf && crate::fsx::exists(&path.join("config.json")) {
            Config::from_json(&serde_json::from_slice(&crate::fsx::read(&path.join("config.json"))?)?)?
        } else {
            Self::infer_config(&ld, layout)?
        };
        let d = cfg.hidden();
        let n = |dn: &'static str, sn: &'static str| if sai { sn } else { dn };
        let top = |dn: &'static str, sn: &'static str| ld.dense(n(dn, sn), false);
        let pk = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let pn = n("pos_embed.proj", "x_embedder.proj");
        let patch_w = ld.f32(&format!("{pn}.weight"))?;
        let patch_b = ld.f32(&format!("{pn}.bias"))?;
        if patch_w.len() != d * pk || patch_b.len() != d {
            bail!("patch embedding has {} weights, expected {d} x {pk}", patch_w.len());
        }
        let pos = n("pos_embed.pos_embed", "pos_embed");
        let rows = fs
            .info(pos)
            .with_context(|| format!("missing tensor {pos}"))?
            .shape
            .iter()
            .product::<usize>()
            / d;
        if rows != cfg.pos_embed_max_size * cfg.pos_embed_max_size {
            bail!("position table has {rows} rows, expected {0} x {0}", cfg.pos_embed_max_size);
        }
        let ctx_emb = top("context_embedder", "context_embedder")?;
        let time = [
            top("time_text_embed.timestep_embedder.linear_1", "t_embedder.mlp.0")?,
            top("time_text_embed.timestep_embedder.linear_2", "t_embedder.mlp.2")?,
        ];
        let text = [
            top("time_text_embed.text_embedder.linear_1", "y_embedder.mlp.0")?,
            top("time_text_embed.text_embedder.linear_2", "y_embedder.mlp.2")?,
        ];
        let norm_out = {
            let mut v = ld.split(n("norm_out.linear", "final_layer.adaLN_modulation.1"), &[d, d], false)?;
            let (b, a) = (v.pop().unwrap(), v.pop().unwrap());
            // diffusers stores (scale, shift); the original layout (shift, scale).
            if sai { [b, a] } else { [a, b] }
        };
        let proj_out = top("proj_out", "final_layer.linear")?;
        if proj_out.n_out != cfg.patch_size * cfg.patch_size * cfg.out_channels || ctx_emb.n_in != cfg.joint_dim {
            bail!("unexpected output / text projection shapes");
        }
        let blocks = (0..cfg.num_layers)
            .map(|i| JointBlock::load(&ld, layout, i, &cfg).with_context(|| format!("block {i}")))
            .collect::<Result<Vec<_>>>()?;
        let weight_bytes = [&ctx_emb, &proj_out]
            .into_iter()
            .chain(time.iter())
            .chain(text.iter())
            .chain(norm_out.iter())
            .map(|l| l.bytes())
            .sum::<usize>()
            + blocks.iter().map(|b| b.bytes()).sum::<usize>();
        Ok(Self {
            cfg,
            layout,
            patch_w,
            patch_b,
            pos,
            ctx_emb,
            time,
            text,
            norm_out,
            proj_out,
            blocks,
            weight_bytes,
            files,
        })
    }

    /// The configuration of a checkpoint without `config.json` (GGUF / single files).
    fn infer_config(ld: &Loader, layout: Layout) -> Result<Config> {
        if layout != Layout::Sai {
            bail!("not an SD 3 transformer (no x_embedder.proj.weight and no config.json)");
        }
        let shape = |n: &str| {
            ld.files
                .info(n)
                .map(|i| i.shape.clone())
                .with_context(|| format!("missing tensor {n}"))
        };
        let (d, in_channels, patch_size) = match shape("x_embedder.proj.weight")?[..] {
            [d, c, p, q] if p == q => (d, c, p),
            ref s => bail!("x_embedder.proj.weight: unexpected shape {s:?}"),
        };
        let num_layers = (0..)
            .take_while(|i| ld.has(&format!("joint_blocks.{i}.x_block.attn.qkv.weight")))
            .count();
        let head_dim = 64;
        if !d.is_multiple_of(head_dim) || num_layers == 0 {
            bail!("unexpected SD 3 transformer shape (hidden {d}, {num_layers} blocks)");
        }
        let pos_rows = shape("pos_embed")?.iter().product::<usize>() / d;
        let pos_embed_max_size = pos_rows.isqrt();
        if pos_embed_max_size * pos_embed_max_size != pos_rows {
            bail!("pos_embed has {pos_rows} rows, not a square");
        }
        Ok(Config {
            in_channels,
            out_channels: ld.shape2("final_layer.linear.weight")?.0 / (patch_size * patch_size),
            patch_size,
            num_layers,
            heads: d / head_dim,
            head_dim,
            joint_dim: ld.shape2("context_embedder.weight")?.1,
            pooled_dim: ld.shape2("y_embedder.mlp.0.weight")?.1,
            pos_embed_max_size,
            dual_attention_layers: (0..num_layers)
                .filter(|i| ld.has(&format!("joint_blocks.{i}.x_block.attn2.qkv.weight")))
                .collect(),
            qk_norm: ld.has("joint_blocks.0.x_block.attn.ln_q.weight"),
        })
    }

    /// Row `r` of the position table as f32.
    fn pos_row(&self, r: usize, out: &mut [f32]) -> Result<()> {
        let d = out.len();
        if let Some(g) = self.files.ggml(self.pos) {
            let (kind, b) = g?;
            let rb = kind.row_bytes(d);
            crate::ggml::dequantize_row(kind, &b[r * rb..(r + 1) * rb], out);
            return Ok(());
        }
        let b = self.files.bytes(self.pos)?;
        match self.files.info(self.pos).map(|i| i.dtype.as_str()) {
            Some("F16") => {
                for (o, c) in out.iter_mut().zip(b[r * d * 2..(r + 1) * d * 2].as_chunks::<2>().0) {
                    *o = half::f16::from_le_bytes(*c).to_f32();
                }
            }
            Some("BF16") => {
                for (o, c) in out.iter_mut().zip(b[r * d * 2..(r + 1) * d * 2].as_chunks::<2>().0) {
                    *o = f32::from_bits((u16::from_le_bytes(*c) as u32) << 16);
                }
            }
            Some("F32") => {
                for (o, c) in out.iter_mut().zip(b[r * d * 4..(r + 1) * d * 4].as_chunks::<4>().0) {
                    *o = f32::from_le_bytes(*c);
                }
            }
            t => bail!("position table dtype {t:?} is not supported"),
        }
        Ok(())
    }

    /// The conditioning vector: timestep embedding plus the pooled-text projection. `t` is the
    /// timestep in [0, 1000] (sigma * 1000).
    pub fn temb(&self, t: f32, pooled: &[f32]) -> Vec<f32> {
        let mlp = |l: &[Dense; 2], x: &[f32]| {
            let mut h = l[0].forward_vec(x);
            h.iter_mut().for_each(|v| *v = silu(*v));
            l[1].forward_vec(&h)
        };
        let mut e = mlp(&self.time, &timestep_proj(t, 256));
        crate::flux::transformer::add(&mut e, &mlp(&self.text, pooled));
        e
    }

    /// Patch tokens with their positions: `latent` is `[h * w, in_channels]` (channels last);
    /// returns `[(h / p) * (w / p), d]`.
    fn embed(&self, latent: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (p, ch, d, max) = (c.patch_size, c.in_channels, c.hidden(), c.pos_embed_max_size);
        let (gh, gw) = (h / p, w / p);
        if gh > max || gw > max {
            bail!(
                "a {gw}x{gh} patch grid exceeds the model's {max}x{max} position table ({} pixels a side)",
                max * p * 8
            );
        }
        let (top, left) = ((max - gh) / 2, (max - gw) / 2);
        let pk = ch * p * p;
        let mut x = vec![0f32; gh * gw * d];
        let err = std::sync::Mutex::new(None);
        for_chunks(&mut x, d, true, |i, out| {
            let (gy, gx) = (i / gw, i % gw);
            let mut patch = vec![0f32; pk];
            for py in 0..p {
                for px in 0..p {
                    let src = &latent[((gy * p + py) * w + gx * p + px) * ch..][..ch];
                    for (cc, &v) in src.iter().enumerate() {
                        patch[(cc * p + py) * p + px] = v;
                    }
                }
            }
            if let Err(e) = self.pos_row((top + gy) * max + left + gx, out) {
                *err.lock().unwrap() = Some(e);
                return;
            }
            for (o, (wr, &b)) in out.iter_mut().zip(self.patch_w.chunks(pk).zip(&self.patch_b)) {
                *o += b + wr.iter().zip(&patch).map(|(&a, &b)| a * b).sum::<f32>();
            }
        });
        match err.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(x),
        }
    }

    /// One denoising call. `latent` is `[h * w, in_channels]` (channels last, `h` and `w`
    /// multiples of the patch size), `ctx` `[n_ctx, joint_dim]` (CLIP + T5 states), `pooled`
    /// `[pooled_dim]`, `t` the timestep (sigma * 1000). Returns `[h * w, out_channels]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, latent: &[f32], h: usize, w: usize, ctx: &[f32], n_ctx: usize, pooled: &[f32], t: f32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (p, co) = (c.patch_size, c.out_channels);
        if !h.is_multiple_of(p) || !w.is_multiple_of(p) || latent.len() != h * w * c.in_channels {
            bail!("latent {w}x{h} does not split into {p}x{p} patches");
        }
        if ctx.len() != n_ctx * c.joint_dim || pooled.len() != c.pooled_dim {
            bail!("text conditioning has the wrong shape");
        }
        let (gh, gw) = (h / p, w / p);
        let n_img = gh * gw;
        let temb = self.temb(t, pooled);
        let st: Vec<f32> = temb.iter().map(|&v| silu(v)).collect();
        let mut hi = self.embed(latent, h, w)?;
        let mut ht = self.ctx_emb.forward(ctx, n_ctx);
        for b in &self.blocks {
            b.forward(&mut hi, n_img, &mut ht, n_ctx, &st);
        }
        drop(ht);
        let m = [self.norm_out[0].forward_vec(&st), self.norm_out[1].forward_vec(&st)];
        let x = ln_mod(&hi, n_img, &m[0], &m[1]);
        let y = self.proj_out.forward(&x, n_img);
        // Unpatchify: token features are ordered (row, column, channel).
        let mut out = vec![0f32; h * w * co];
        for_chunks(&mut out, w * co, true, |yy, row| {
            let (gy, py) = (yy / p, yy % p);
            for xx in 0..w {
                let (gx, px) = (xx / p, xx % p);
                let src = &y[(gy * gw + gx) * p * p * co + (py * p + px) * co..][..co];
                row[xx * co..(xx + 1) * co].copy_from_slice(src);
            }
        });
        Ok(out)
    }
}
