//! FLUX.1 Redux image prompts (diffusers `FluxPriorReduxPipeline`): an image goes through the
//! SigLIP vision tower (`SiglipVisionModel`, so400m/14 at 384x384: 729 patch tokens of width
//! 1152, the states after `post_layernorm`) and the Redux projector (`redux_down(silu(redux_up(x)))`,
//! 1152 -> 12288 -> 4096) into tokens of the T5 width, which are appended to the prompt's T5
//! states. The FLUX transformer then reads them as more text tokens (position ids 0, 0, 0).
//!
//! The folder is the layout of `black-forest-labs/FLUX.1-Redux-dev`: `image_encoder/` (config +
//! weights) and `image_embedder/`.

use super::weights::{Dense, Loader};
use crate::kernels::{dot, for_chunks, gelu_tanh};
use crate::model::Quant;
use crate::safetensors::ModelFiles;
use crate::video::RgbImage;
use anyhow::{Context, Result, bail};
use std::path::Path;

struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    fn load(ld: &Loader, p: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            w: ld.f32(&format!("{p}.weight"))?,
            b: ld.f32(&format!("{p}.bias"))?,
            eps,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = self.w.len();
        let mut y = vec![0f32; x.len()];
        for_chunks(&mut y, n, x.len() > n * 16, |r, out| {
            let row = &x[r * n..(r + 1) * n];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
            let inv = (1.0 / (var + self.eps as f64).sqrt()) as f32;
            for i in 0..n {
                out[i] = (row[i] - mean as f32) * inv * self.w[i] + self.b[i];
            }
        });
        y
    }
}

struct Block {
    ln1: LayerNorm,
    q: Dense<'static>,
    k: Dense<'static>,
    v: Dense<'static>,
    o: Dense<'static>,
    ln2: LayerNorm,
    fc1: Dense<'static>,
    fc2: Dense<'static>,
}

/// The SigLIP tower and the Redux projector.
pub struct Redux {
    /// Side of the square the image is resized to (384).
    pub image_size: usize,
    pub patch: usize,
    hidden: usize,
    heads: usize,
    /// `[hidden, 3 * patch * patch]` (channel, row, column), and its bias.
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    /// `[tokens, hidden]`.
    pos: Vec<f32>,
    blocks: Vec<Block>,
    post_ln: LayerNorm,
    up: Dense<'static>,
    down: Dense<'static>,
    /// Width of the tokens produced (the transformer's `joint_attention_dim`).
    pub out_dim: usize,
    // Declared last: dropped after every layer borrowing them.
    _files: [Box<ModelFiles>; 2],
}

impl Redux {
    /// Loads `dir/image_encoder` and `dir/image_embedder`.
    pub fn load(dir: &Path) -> Result<Self> {
        let enc_dir = dir.join("image_encoder");
        let cfg: serde_json::Value =
            serde_json::from_slice(&crate::fsx::read(&enc_dir.join("config.json")).context("image_encoder/config.json")?)?;
        // A whole SiglipModel's config nests the tower's.
        let cfg = if cfg["vision_config"].is_object() {
            &cfg["vision_config"]
        } else {
            &cfg
        };
        let u = |k: &str, d: usize| cfg[k].as_u64().map_or(d, |v| v as usize);
        let (hidden, layers, heads) = (u("hidden_size", 1152), u("num_hidden_layers", 27), u("num_attention_heads", 16));
        let (patch, image_size) = (u("patch_size", 14), u("image_size", 384));
        let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let act = cfg["hidden_act"].as_str().unwrap_or("gelu_pytorch_tanh");
        if act != "gelu_pytorch_tanh" {
            bail!("SigLIP activation {act} is not supported");
        }
        let in_memory = cfg!(target_arch = "wasm32");
        let enc = Box::new(ModelFiles::open(&enc_dir, in_memory)?);
        let emb = Box::new(ModelFiles::open(&dir.join("image_embedder"), in_memory)?);
        // SAFETY: the boxes' heap blocks do not move, and every layer borrowing them is dropped
        // before `_files` (field order).
        let (ef, mf): (&'static ModelFiles, &'static ModelFiles) =
            unsafe { (&*(enc.as_ref() as *const ModelFiles), &*(emb.as_ref() as *const ModelFiles)) };
        let ld = Loader {
            files: ef,
            quant: Quant::Bf16,
        };
        // Checkpoints saved by newer transformers drop the `vision_model.` level.
        let p = if ef.has("vision_model.embeddings.patch_embedding.weight") {
            "vision_model."
        } else {
            ""
        };
        let pn = format!("{p}embeddings.patch_embedding");
        let patch_w = ef.f32(&format!("{pn}.weight"))?;
        if patch_w.len() != hidden * 3 * patch * patch {
            bail!("{pn}.weight: {} values for {hidden} x 3 x {patch} x {patch}", patch_w.len());
        }
        let pos = ef.f32(&format!("{p}embeddings.position_embedding.weight"))?;
        let grid = image_size / patch;
        if pos.len() != grid * grid * hidden {
            bail!(
                "SigLIP position embedding: {} values for a {grid}x{grid} grid of width {hidden}",
                pos.len()
            );
        }
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let b = format!("{p}encoder.layers.{i}");
            let d = |n: &str| ld.dense(&format!("{b}.{n}"), false);
            blocks.push(Block {
                ln1: LayerNorm::load(&ld, &format!("{b}.layer_norm1"), eps)?,
                q: d("self_attn.q_proj")?,
                k: d("self_attn.k_proj")?,
                v: d("self_attn.v_proj")?,
                o: d("self_attn.out_proj")?,
                ln2: LayerNorm::load(&ld, &format!("{b}.layer_norm2"), eps)?,
                fc1: d("mlp.fc1")?,
                fc2: d("mlp.fc2")?,
            });
        }
        let ml = Loader {
            files: mf,
            quant: Quant::Bf16,
        };
        let (up, down) = (ml.dense("redux_up", false)?, ml.dense("redux_down", false)?);
        if up.n_in != hidden {
            bail!("the Redux projector reads {} channels, the image encoder gives {hidden}", up.n_in);
        }
        Ok(Self {
            image_size,
            patch,
            hidden,
            heads,
            patch_w,
            patch_b: ef.f32(&format!("{pn}.bias"))?,
            pos,
            post_ln: LayerNorm::load(&ld, &format!("{p}post_layernorm"), eps)?,
            blocks,
            out_dim: down.n_out,
            up,
            down,
            _files: [enc, emb],
        })
    }

    /// Tokens per image (729).
    pub fn tokens(&self) -> usize {
        (self.image_size / self.patch).pow(2)
    }

    /// `SiglipImageProcessor`: resized to the square (Pillow's bicubic, as transformers'
    /// PIL-backed processor), `x / 255` normalized with mean and std 0.5; `[3, s, s]`.
    pub fn preprocess(&self, img: &RgbImage) -> Vec<f32> {
        let s = self.image_size;
        let px = if (img.width, img.height) == (s, s) {
            img.data.clone()
        } else {
            crate::resample::resize_bicubic_rgb8(&img.data, img.width, img.height, s, s)
        };
        let mut out = vec![0f32; 3 * s * s];
        for (i, p) in px.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                out[c * s * s + i] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        out
    }

    /// The SigLIP states of preprocessed pixels (`[3, s, s]`): `[tokens, hidden]`.
    pub fn features(&self, px: &[f32]) -> Vec<f32> {
        let (s, p, d) = (self.image_size, self.patch, self.hidden);
        let g = s / p;
        let n = g * g;
        let k = 3 * p * p;
        let mut patches = vec![0f32; n * k];
        for_chunks(&mut patches, k, true, |t, row| {
            let (ty, tx) = (t / g, t % g);
            for c in 0..3 {
                for y in 0..p {
                    let src = c * s * s + (ty * p + y) * s + tx * p;
                    row[(c * p + y) * p..(c * p + y + 1) * p].copy_from_slice(&px[src..src + p]);
                }
            }
        });
        let mut x = vec![0f32; n * d];
        for_chunks(&mut x, d, true, |t, row| {
            let pr = &patches[t * k..(t + 1) * k];
            for (o, v) in row.iter_mut().enumerate() {
                *v = dot(&self.patch_w[o * k..(o + 1) * k], pr) + self.patch_b[o] + self.pos[t * d + o];
            }
        });
        let dh = d / self.heads;
        for b in &self.blocks {
            let h = b.ln1.forward(&x);
            let (q, kk, v) = (b.q.forward(&h, n), b.k.forward(&h, n), b.v.forward(&h, n));
            let a = crate::sd::ops::attention(&q, &kk, &v, n, n, self.heads, dh, false);
            crate::kernels::add_inplace(&mut x, &b.o.forward(&a, n));
            let h = b.ln2.forward(&x);
            let mut m = b.fc1.forward(&h, n);
            m.iter_mut().for_each(|v| *v = gelu_tanh(*v));
            crate::kernels::add_inplace(&mut x, &b.fc2.forward(&m, n));
        }
        self.post_ln.forward(&x)
    }

    /// The Redux tokens of SigLIP states: `[tokens, out_dim]`.
    pub fn project(&self, feats: &[f32]) -> Vec<f32> {
        let n = feats.len() / self.hidden;
        let mut h = self.up.forward(feats, n);
        h.iter_mut().for_each(|v| *v /= 1.0 + (-*v).exp());
        self.down.forward(&h, n)
    }

    /// The tokens an image prompt adds to the text stream: `[tokens(), out_dim]`.
    pub fn encode(&self, img: &RgbImage) -> Vec<f32> {
        self.project(&self.features(&self.preprocess(img)))
    }
}
