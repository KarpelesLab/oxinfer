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

pub mod audio_vae;
pub mod corrector;
#[cfg(all(feature = "gpu", not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))))]
mod cuda;
pub mod dit;
#[cfg(feature = "media")]
pub mod encode;
pub mod lora;
pub mod ref2va;
mod stack_dev;
#[doc(hidden)]
pub mod tiny;
pub mod vae;
pub mod video;
pub mod video_vae;

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

/// Whether `path` is a single-file MiniMax-H3-Image transformer in ComfyUI's layout
/// (`Comfy-Org/MiniMax-H3-Image`'s `diffusion_models/minimax_h3_image_*.safetensors`): the H3
/// DiT's tensor names without the audio stream.
pub fn is_h3_image_file(path: &Path) -> bool {
    if !path.is_file() || path.extension().is_none_or(|e| e != "safetensors") {
        return false;
    }
    crate::safetensors::SafeTensors::open(path, false).is_ok_and(|st| {
        [
            "video_patch_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
            "final_layer.video_out.weight",
        ]
        .iter()
        .all(|n| st.info(n).is_some())
            && st.info("audio_patch_proj.weight").is_none()
    })
}

/// MiniMax-H3-Image's sampling grid (`pdd_config.json`'s `sigmas`), for a single file whose
/// header does not state it.
pub const PDD_SIGMAS: [f32; 7] = [1.0, 0.967_523_7, 0.887_548_5, 0.75, 0.532_773_9, 0.232_008_87, 0.0];

/// The release's `vae/config.json` (`MiniMaxH3ImageVAE`), for ComfyUI's single-file image VAE
/// (`vae/minimax_h3_image_vae_fp16.safetensors`: the release's tensors in float16, no
/// configuration in its header).
const IMAGE_VAE_JSON: &str = r#"{"_class_name": "MiniMaxH3ImageVAE", "latent_channels": 24,
"latents_mean": [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264],
"latents_std": [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523],
"vae_tile_overlap_min": 64, "vae_tile_size": 256,
"vit_decoder_kwargs": {"dim_head": 64, "heads": 16, "num_layers": 24, "rope_dim_ratio": 0.5, "rope_theta": 100.0}}"#;

/// Whether a checkpoint folder holds its weights (not only its configuration).
fn has_weights(dir: &Path) -> bool {
    if dir.join("model.safetensors").exists() {
        return true;
    }
    let Ok(b) = crate::fsx::read(&dir.join("model.safetensors.index.json")) else {
        return false;
    };
    serde_json::from_slice::<serde_json::Value>(&b).is_ok_and(|v| {
        v["weight_map"]
            .as_object()
            .is_some_and(|m| m.values().filter_map(|f| f.as_str()).all(|f| dir.join(f).exists()))
    })
}

/// MiniMax-H3's text encoder (Qwen3-VL-32B, `text_encoder/` of `MiniMaxAI/MiniMax-H3`), which
/// MiniMax-H3-Image and the video pipeline both read: `OXINFER_H3_TEXT_ENCODER`, else
/// `text_encoder/` (or `FL2VA/text_encoder/`) of `model` (or of the release above a `FL2VA/` /
/// `Ref2VA/` folder), else of `MiniMaxAI/MiniMax-H3` in the oxinfer download cache (`cache_dir`,
/// or the default one) or in the Hugging Face cache.
pub fn find_text_encoder(model: &Path, cache_dir: Option<&Path>) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("OXINFER_H3_TEXT_ENCODER") {
        return Ok(PathBuf::from(p));
    }
    let mut roots = vec![model.to_path_buf()];
    if let Some(parent) = model.parent().filter(|_| model.ends_with("Ref2VA") || model.ends_with("FL2VA")) {
        roots.push(parent.to_path_buf());
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let mut caches: Vec<PathBuf> = cache_dir.map(Path::to_path_buf).into_iter().collect();
    caches.push(PathBuf::from(format!("{home}/.cache/oxinfer/hub")));
    caches.push(PathBuf::from(format!("{home}/.cache/huggingface/hub")));
    for cache in caches {
        roots.push(cache.join("MiniMaxAI--MiniMax-H3"));
        if let Ok(rd) = std::fs::read_dir(cache.join("models--MiniMaxAI--MiniMax-H3/snapshots")) {
            roots.extend(rd.flatten().map(|e| e.path()));
        }
    }
    for r in &roots {
        for sub in ["text_encoder", "FL2VA/text_encoder"] {
            let p = r.join(sub);
            if p.join("config.json").exists() && has_weights(&p) {
                return Ok(p);
            }
        }
    }
    bail!(
        "MiniMax-H3's text encoder was not found: download it (`hf download MiniMaxAI/MiniMax-H3 --include 'text_encoder/*'`), set OXINFER_H3_TEXT_ENCODER, or name it (the text_encoder option)"
    )
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
    /// The text encoder: the release's `text_encoder/` folder, or a file ([`Pipeline::load_files`]).
    pub text_encoder_dir: PathBuf,
    pub dit: Dit<'static>,
    pub vae: VaeDecoder<'static>,
    /// The sampling grid (`pdd_config.json`), ending at 0.
    pub sigmas: Vec<f32>,
    /// The release's tokenizer (None: the text encoder file's, see [`Pipeline::load_files`]).
    tok: Option<crate::tokenizer::Tokenizer>,
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
            tok: Some(tok),
        })
    }

    /// Loads ComfyUI's single files (`Comfy-Org/MiniMax-H3-Image`): the transformer
    /// ([`is_h3_image_file`]: the PDD heads stacked in `final_layer.video_out`, the grid in its
    /// header or [`PDD_SIGMAS`]; any weight format the video transformer takes), the image VAE (its
    /// single file, or a folder with `config.json`) and the text encoder: a folder in the
    /// release's layout (with `tokenizer/` beside the transformer), a GGUF of
    /// Qwen3-VL-32B-Instruct, or ComfyUI's `qwen3vl_32b_minimax_h3_*.safetensors` (with the
    /// release's `tokenizer.json` where [`video::load_text_encoder`] looks for it).
    pub fn load_files(transformer: &Path, vae: &Path, text_encoder: &Path) -> Result<Self> {
        if !is_h3_image_file(transformer) {
            bail!("{}: not a single-file MiniMax-H3-Image transformer", transformer.display());
        }
        let dit_files: &'static ModelFiles = Box::leak(Box::new(ModelFiles::open_file(transformer, false)?));
        let cfg = DitConfig::from_files(dit_files)?;
        let dit = Dit::load(cfg, dit_files, None)?;
        let sigmas = dit::comfy_pdd_sigmas(dit_files).unwrap_or_else(|| PDD_SIGMAS.to_vec());
        if sigmas.len() != dit.n_heads() + 1 {
            bail!("{} sigmas for {} output heads", sigmas.len(), dit.n_heads());
        }
        let (vae_files, vcfg) = if vae.is_file() {
            let f = ModelFiles::open_file(vae, false)?;
            (f, VaeConfig::from_json(&serde_json::from_str(IMAGE_VAE_JSON)?)?)
        } else {
            (ModelFiles::open(vae, false)?, VaeConfig::load(vae)?)
        };
        let vae_files: &'static ModelFiles = Box::leak(Box::new(vae_files));
        let vae_dec = VaeDecoder::load(vcfg, vae_files).with_context(|| format!("image VAE {}", vae.display()))?;
        if vae_dec.cfg.axes != 2 || vae_dec.cfg.latent_channels != dit.cfg.in_channels {
            bail!("{}: not MiniMax-H3-Image's VAE", vae.display());
        }
        let dir = transformer.parent().map(Path::to_path_buf).unwrap_or_default();
        let tok = if text_encoder.is_dir() {
            let t = crate::tokenizer::Tokenizer::from_bytes(crate::fsx::read(&dir.join("tokenizer").join("tokenizer.json"))?)
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
            Some(t)
        } else {
            None
        };
        Ok(Self {
            dir,
            text_encoder_dir: text_encoder.to_path_buf(),
            dit,
            vae: vae_dec,
            sigmas,
            tok,
        })
    }

    /// The conditioning of `prompt`: `[n, 5120]` after the text encoder's 50th layer.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        let Some(tok) = &self.tok else {
            // A text encoder file (GGUF, or ComfyUI's single file) through Qwen3-VL's loader.
            let vl = video::load_text_encoder(&self.dir, &self.text_encoder_dir, Quant::Bf16, false)?;
            let ids = vl.tokenize(prompt)?;
            if ids.is_empty() {
                bail!("empty prompt");
            }
            let mut hs = vl.hidden_states_tokens(ids, &[], None, &[video::TEXT_LAYER - 1])?;
            return Ok(hs.layers.pop().context("layer 50 not captured")?.1);
        };
        let ids = tok
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
