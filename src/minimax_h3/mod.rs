//! MiniMax-H3-Image: text to image with the MiniMax-H3 DiT distilled to six sampling steps without
//! classifier-free guidance, following diffusers' modular `minimax_h3_image` pipeline:
//!
//! - the prompt verbatim (no chat template, no special tokens) through MiniMax-H3's Qwen3-VL text
//!   encoder, conditioning on `hidden_states[50]` (the residual stream after its 50th layer; the
//!   text encoder lives in the MiniMax-H3 repository, `text_encoder/`);
//! - a canvas of at least `min_area` pixels (2048 x 2048 by default; a smaller request is generated
//!   larger at the same aspect ratio and resized back with Lanczos), sides multiples of 32;
//! - one packed sequence `[text | image]`: text rows at rotary positions `(i, 0, 0)`, the image's
//!   2x2 latent patches at `(n_text, y, x)` with `y` / `x` aspect-normalized
//!   (`linspace(left, left + side / sqrt(area), n, endpoint = False) * 32`);
//! - the six steps of `pdd_config.json` (their sigmas rounded through float32 `1 - sigma`), each
//!   with its own output head, Euler along the predicted velocity: `x += (sigma - sigma_next) v`;
//! - the VAE (`vae`), ImageNet de-normalization, 8-bit rounding (truncation, as `to(uint8)`).

pub mod dit;
pub mod vae;

use crate::clock::Instant;
use crate::model::Quant;
use crate::safetensors::ModelFiles;
pub use crate::sd::Image;
use anyhow::{Context, Result, bail};
use dit::{Dit, DitConfig, Packed, TAG_IMAGE, TAG_TEXT};
use std::path::{Path, PathBuf};
use vae::{VaeConfig, VaeDecoder};

/// Whether `dir` is a MiniMax-H3-Image release (`model_index.json`'s `MiniMaxH3ImagePipeline`).
pub fn is_h3_image(dir: &Path) -> bool {
    let Ok(b) = crate::fsx::read(&dir.join("model_index.json")) else {
        return false;
    };
    serde_json::from_slice::<serde_json::Value>(&b)
        .ok()
        .and_then(|v| v.get("_class_name").and_then(|c| c.as_str()).map(|c| c == "MiniMaxH3ImagePipeline"))
        .unwrap_or(false)
}

/// What [`Pipeline::generate`] needs.
#[derive(Clone, Debug)]
pub struct Params {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    /// The smallest area generated at (diffusers' `min_generation_area`).
    pub min_area: usize,
}

impl Params {
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            width: 1024,
            height: 1024,
            seed: 42,
            min_area: 2048 * 2048,
        }
    }
}

pub struct Pipeline {
    pub dir: PathBuf,
    pub text_encoder_dir: PathBuf,
    pub dit: Dit<'static>,
    pub vae: VaeDecoder<'static>,
    /// The sampling grid (`pdd_config.json`), ending at 0.
    pub sigmas: Vec<f32>,
    tok: crate::tokenizer::Tokenizer,
}

/// Rows of patches `[C, pt, ph, pw]` from latents `[C, h, w]` (one frame), row-major.
pub fn patchify(lat: &[f32], c: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (gh, gw) = (h / p, w / p);
    let mut out = vec![0f32; gh * gw * c * p * p];
    for gy in 0..gh {
        for gx in 0..gw {
            let row = &mut out[(gy * gw + gx) * c * p * p..(gy * gw + gx + 1) * c * p * p];
            for ch in 0..c {
                for y in 0..p {
                    for x in 0..p {
                        row[(ch * p + y) * p + x] = lat[(ch * h + gy * p + y) * w + gx * p + x];
                    }
                }
            }
        }
    }
    out
}

/// The inverse of [`patchify`].
pub fn unpatchify(rows: &[f32], c: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (gh, gw) = (h / p, w / p);
    let mut lat = vec![0f32; c * h * w];
    for gy in 0..gh {
        for gx in 0..gw {
            let row = &rows[(gy * gw + gx) * c * p * p..(gy * gw + gx + 1) * c * p * p];
            for ch in 0..c {
                for y in 0..p {
                    for x in 0..p {
                        lat[(ch * h + gy * p + y) * w + gx * p + x] = row[(ch * p + y) * p + x];
                    }
                }
            }
        }
    }
    lat
}

/// One aspect-normalized rotary axis (`_spatial_position_grid`): `side / patch` coordinates.
fn axis_grid(side: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = side as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let n = side / patch;
    (0..n).map(|i| (left + i as f64 * ratio / n as f64) * 32.0).collect()
}

/// The rotary positions of `[text | image]` (float64 like the reference, then float32).
pub fn positions(n_text: usize, lh: usize, lw: usize, patch: usize) -> Vec<[f32; 3]> {
    let sa = ((lh * lw) as f64).sqrt();
    let (ys, xs) = (axis_grid(lh, patch, sa), axis_grid(lw, patch, sa));
    let mut p: Vec<[f32; 3]> = (0..n_text).map(|i| [i as f32, 0.0, 0.0]).collect();
    for &y in &ys {
        for &x in &xs {
            p.push([n_text as f32, y as f32, x as f32]);
        }
    }
    p
}

/// `get_image_canvas_size`: `(height, width)` scaled up to `min_area` when smaller, rounded to
/// multiples of `multiple`.
pub fn canvas(height: usize, width: usize, min_area: usize, multiple: usize) -> (usize, usize) {
    let scale = (min_area as f64 / (width * height) as f64).sqrt().max(1.0);
    let r = |v: usize| ((v as f64 * scale / multiple as f64).round() as usize * multiple).max(multiple);
    (r(height), r(width))
}

impl Pipeline {
    /// Loads the DiT and the VAE of the release in `dir` (bf16, mapped); the text encoder from
    /// `text_encoder` (MiniMax-H3's `text_encoder/` folder) when it is used.
    pub fn load(dir: &Path, text_encoder: &Path) -> Result<Self> {
        if !is_h3_image(dir) {
            bail!("{}: not a MiniMax-H3-Image release", dir.display());
        }
        let tdir = dir.join("transformer");
        let dit_files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(&tdir, false)?));
        let index: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("model_index.json"))?)?;
        let heads = index
            .get("_minimax_h3")
            .and_then(|m| m.get("pdd_fused_heads"))
            .and_then(|v| v.as_str())
            .map(|f| ModelFiles::open(&dir.join(f), false))
            .transpose()?;
        let dit = Dit::load(DitConfig::load(&tdir)?, dit_files, heads.as_ref())?;
        let vdir = dir.join("vae");
        let vae_files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open(&vdir, false)?));
        let vae = VaeDecoder::load(VaeConfig::load(&vdir)?, vae_files)?;
        let pdd: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("pdd_config.json"))?)?;
        let sigmas: Vec<f32> = pdd
            .get("sigmas")
            .and_then(|s| s.as_array())
            .context("pdd_config.json has no sigmas")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        if sigmas.len() != dit.n_heads() + 1 && dit.n_heads() != 1 {
            bail!("{} sigmas for {} output heads", sigmas.len(), dit.n_heads());
        }
        let tok = crate::tokenizer::Tokenizer::from_bytes(crate::fsx::read(&dir.join("tokenizer").join("tokenizer.json"))?)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            text_encoder_dir: text_encoder.to_path_buf(),
            dit,
            vae,
            sigmas,
            tok,
        })
    }

    /// The conditioning of `prompt`: `[n, 5120]` after the text encoder's 50th layer.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        let ids = self
            .tok
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
            .ids
            .clone();
        if ids.is_empty() {
            bail!("empty prompt");
        }
        let files = ModelFiles::open(&self.text_encoder_dir, false)?;
        let cfg = crate::config::QwenConfig::load(&self.text_encoder_dir)?;
        if cfg.num_hidden_layers <= 50 {
            bail!(
                "the text encoder has {} layers; MiniMax-H3 conditions on its 50th",
                cfg.num_hidden_layers
            );
        }
        let model = crate::qwen::QwenModel::load(cfg, &files, Quant::Bf16)?;
        let mut sess = model.new_session();
        sess.capture = vec![49];
        model.forward_hidden_mm(&mut sess, &ids, &[]);
        let (_, x) = sess.captured.pop().context("layer 50 not captured")?;
        Ok(x)
    }

    /// Denoises `noise` (latent patch rows) under `text` (`[n, text_dim]`) on an `lh` x `lw` latent
    /// canvas; returns the final rows.
    pub fn denoise(&self, text: &[f32], noise: &[f32], lh: usize, lw: usize, progress: &mut dyn FnMut(usize, usize)) -> Vec<f32> {
        let cfg = &self.dit.cfg;
        let n_text = text.len() / cfg.text_dim;
        let p = cfg.patch[1];
        let pos = positions(n_text, lh, lw, p);
        let n = pos.len();
        let tags: Vec<u8> = (0..n).map(|i| if i < n_text { TAG_TEXT } else { TAG_IMAGE }).collect();
        let tidx = vec![0usize; n];
        let emb = self.dit.embed_text(text);
        // The released sampler rounds each sigma through its float32 timestep.
        let steps = self.sigmas.len() - 1;
        let ts: Vec<f32> = self.sigmas[..steps].iter().map(|&s| 1.0 - s).collect();
        let mut sig: Vec<f32> = ts.iter().map(|&t| 1.0 - t).collect();
        sig.push(0.0);
        let mut x = noise.to_vec();
        for (i, &t) in ts.iter().enumerate() {
            let v = self.dit.forward(
                &Packed {
                    text,
                    image: &x,
                    positions: &pos,
                    timesteps: &[t],
                    timestep_index: &tidx,
                    tags: &tags,
                },
                &emb,
                i,
            );
            let d = sig[i] - sig[i + 1];
            for (xv, vv) in x.iter_mut().zip(&v) {
                *xv += d * vv;
            }
            progress(i + 1, steps);
        }
        x
    }

    /// Decodes latent rows into an image of the canvas size (`[3, H, W]` in `[0, 1]`).
    pub fn decode(&self, rows: &[f32], lh: usize, lw: usize) -> Vec<f32> {
        let c = self.vae.cfg.latent_channels;
        let lat = unpatchify(rows, c, lh, lw, self.dit.cfg.patch[1]);
        let img = self.vae.decode(&lat, lh, lw);
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let plane = img.len() / 3;
        img.iter()
            .enumerate()
            .map(|(i, &v)| (v * std[i / plane] + mean[i / plane]).clamp(0.0, 1.0))
            .collect()
    }

    /// Text to image.
    pub fn generate(&self, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<Image> {
        let t0 = Instant::now();
        let text = self.encode_prompt(&p.prompt)?;
        let t_enc = t0.elapsed().as_secs_f64();
        let (ch, cw) = canvas(p.height, p.width, p.min_area, 16 * self.dit.cfg.patch[1]);
        let (lh, lw) = (ch / 16, cw / 16);
        let c = self.vae.cfg.latent_channels;
        let noise = crate::sd::ops::Rng::new(p.seed).normals(c * lh * lw);
        let rows = patchify(&noise, c, lh, lw, self.dit.cfg.patch[1]);
        let t1 = Instant::now();
        let rows = self.denoise(&text, &rows, lh, lw, progress);
        let t_dit = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let img = self.decode(&rows, lh, lw);
        let plane = ch * cw;
        let mut rgb = vec![0u8; plane * 3];
        for i in 0..plane {
            for k in 0..3 {
                rgb[i * 3 + k] = (img[k * plane + i] * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
        if (ch, cw) != (p.height, p.width) {
            rgb = crate::resample::resize_lanczos_rgb8(&rgb, cw, ch, p.width, p.height);
        }
        eprintln!(
            "[H3-Image: {cw}x{ch} canvas, text {:.1}s, {} steps {:.1}s, VAE {:.1}s]",
            t_enc,
            self.sigmas.len() - 1,
            t_dit,
            t2.elapsed().as_secs_f64()
        );
        Ok(Image {
            width: p.width,
            height: p.height,
            rgb,
        })
    }
}
