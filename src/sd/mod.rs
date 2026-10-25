//! Text-to-image with Stable Diffusion 1.x / 2.x and SD-Turbo from a diffusers folder
//! (`model_index.json`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`, `scheduler/`).
//!
//! The UNet's convolutions run through the same matmul as the language models, so they use the
//! SIMD kernels on every CPU and the GPU backends (Metal, CUDA, WebGPU) when one is active.

pub mod clip;
pub mod lora;
pub mod ops;
pub mod sched;
pub mod unet;
pub mod vae;

use crate::clock::Instant;
use anyhow::{Context, Result, bail};
use ops::Rng;
use sched::{Sampler, Schedule};
use std::path::{Path, PathBuf};

pub struct Pipeline {
    pub tokenizer: clip::ClipTokenizer,
    pub text: clip::ClipText,
    pub unet: unet::Unet,
    pub vae: vae::VaeDecoder,
    pub schedule: Schedule,
    /// Distilled single-step models (SD-Turbo) default to 1 step and no guidance.
    pub turbo: bool,
}

pub struct Params {
    pub prompt: String,
    pub negative: String,
    pub steps: usize,
    /// Classifier-free guidance scale; at or below 1 the unconditional pass is skipped.
    pub guidance: f32,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    pub sampler: Sampler,
}

/// A generated image: tightly packed RGB8.
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

#[cfg(feature = "oxideav")]
impl Image {
    /// The image as an oxideav `Rgb24` frame (`pts` 0) and its `rawvideo` parameters (`width`,
    /// `height`, `pixel_format`; see [`crate::av::rgb_frame`]), e.g. for an oxideav encoder or filter graph.
    pub fn to_video_frame(&self) -> (oxideav_core::VideoFrame, oxideav_core::CodecParameters) {
        crate::av::rgb_frame(self.width, self.height, self.rgb.clone())
    }

    /// [`Image::to_video_frame`] without copying the pixels.
    pub fn into_video_frame(self) -> (oxideav_core::VideoFrame, oxideav_core::CodecParameters) {
        crate::av::rgb_frame(self.width, self.height, self.rgb)
    }
}

/// Timings of one generation, in seconds.
#[derive(Default, Debug, Clone)]
pub struct Timings {
    pub text: f64,
    pub denoise: f64,
    pub decode: f64,
    pub unet_calls: usize,
}

/// True for a diffusers pipeline folder.
pub fn is_diffusers(dir: &Path) -> bool {
    crate::fsx::exists(&dir.join("model_index.json"))
}

/// The weight file of a component: prefers full precision, then the fp16 variant.
fn weights(dir: &Path, base: &str) -> Result<PathBuf> {
    for name in [format!("{base}.safetensors"), format!("{base}.fp16.safetensors")] {
        let p = dir.join(&name);
        if crate::fsx::exists(&p) {
            return Ok(p);
        }
    }
    bail!("no {base}.safetensors or {base}.fp16.safetensors in {}", dir.display())
}

fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("OXINFER_SD_DUMP").map(PathBuf::from)
}

/// Writes `[h*w, c]` channels-last data as `[c, h, w]` f32 (PyTorch's layout) for comparisons.
fn dump_chw(name: &str, x: &[f32], h: usize, w: usize, c: usize) {
    if let Some(d) = dump_dir() {
        let mut out = Vec::with_capacity(x.len() * 4);
        for ch in 0..c {
            for p in 0..h * w {
                out.extend_from_slice(&x[p * c + ch].to_le_bytes());
            }
        }
        let _ = std::fs::write(d.join(name), out);
    }
}

fn dump_raw(name: &str, x: &[f32]) {
    if let Some(d) = dump_dir() {
        let _ = std::fs::write(d.join(name), x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    }
}

impl Pipeline {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with_loras(dir, &[], &mut |_| {})
    }

    /// Loads the pipeline with LoRA files merged into the UNet and text-encoder weights;
    /// `report` receives one summary line per LoRA file.
    pub fn load_with_loras(dir: &Path, loras: &[lora::LoraSpec], report: &mut dyn FnMut(&str)) -> Result<Self> {
        if !is_diffusers(dir) {
            bail!("{} is not a diffusers pipeline (no model_index.json)", dir.display());
        }
        let index: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("model_index.json"))?)?;
        let class = index["_class_name"].as_str().unwrap_or("");
        if class.contains("XL") || class.contains("StableDiffusion3") || class.contains("Flux") {
            bail!("{class} is not supported yet (Stable Diffusion 1.x / 2.x and SD-Turbo are)");
        }
        let te = dir.join("text_encoder");
        let un = dir.join("unet");
        let va = dir.join("vae");
        let mut te_t = ops::Tensors::open(&weights(&te, "model")?)?;
        let mut un_t = ops::Tensors::open(&weights(&un, "diffusion_pytorch_model")?)?;
        let set = std::sync::Arc::new(lora::Loras::open(loras)?);
        if !set.is_empty() {
            te_t.set_lora(set.resolve(lora::Part::Text, &te_t)?);
            un_t.set_lora(set.resolve(lora::Part::Unet, &un_t)?);
        }
        let text = clip::ClipText::load(&te, te_t)?;
        let unet = unet::Unet::load(&un, un_t)?;
        if !set.is_empty() {
            for line in set.report()? {
                report(&line);
            }
        }
        if unet.cross_dim != text.hidden {
            bail!(
                "text encoder width {} does not match the UNet's cross-attention width {}",
                text.hidden,
                unet.cross_dim
            );
        }
        let vae = vae::VaeDecoder::load(&va, &weights(&va, "diffusion_pytorch_model")?)?;
        // Distilled single-step models are named as such (folder, or the name recorded at export).
        let name = format!("{} {}", dir.to_string_lossy(), index["_name_or_path"].as_str().unwrap_or("")).to_lowercase();
        Ok(Self {
            tokenizer: clip::ClipTokenizer::load(&dir.join("tokenizer"))?,
            text,
            unet,
            vae,
            schedule: Schedule::load(&dir.join("scheduler").join("scheduler_config.json"))?,
            turbo: name.contains("turbo"),
        })
    }

    /// Default image side: the UNet's training sample size times the VAE factor of 8.
    pub fn default_size(&self) -> usize {
        self.unet.sample_size * 8
    }

    pub fn encode_prompt(&self, text: &str) -> Vec<f32> {
        let ids = self.tokenizer.encode(text);
        if dump_dir().is_some() && !text.is_empty() {
            eprintln!("[ids {:?}]", &ids[..24.min(ids.len())]);
        }
        self.text.forward(&ids)
    }

    /// Generates one image. `progress(step, steps)` is called after each denoising step.
    pub fn generate(&self, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        if !p.width.is_multiple_of(8) || !p.height.is_multiple_of(8) || p.width < 64 || p.height < 64 {
            bail!("image size must be a multiple of 8 and at least 64 (got {}x{})", p.width, p.height);
        }
        // Three 2x downsamplings in the UNet: the latent side must divide by 8.
        let (lh, lw) = (p.height / 8, p.width / 8);
        if !lh.is_multiple_of(8) || !lw.is_multiple_of(8) {
            bail!("image size must be a multiple of 64 for this UNet (got {}x{})", p.width, p.height);
        }
        let mut tm = Timings::default();
        let t0 = Instant::now();
        let m = self.tokenizer.max_len;
        let cond = self.encode_prompt(&p.prompt);
        let guided = p.guidance > 1.0;
        let uncond = if guided { Some(self.encode_prompt(&p.negative)) } else { None };
        dump_raw("text_cond.bin", &cond);
        tm.text = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let c = self.unet.in_channels;
        let (timesteps, sigmas, init) = self.schedule.plan(p.steps.max(1));
        let mut rng = Rng::new(p.seed);
        let mut x: Vec<f32> = match dump_dir().map(|d| d.join("latents_in.bin")).filter(|f| f.exists()) {
            // Validation: start from the reference's noise ([c, h, w] f32).
            Some(f) => {
                let b = std::fs::read(&f).with_context(|| format!("read {}", f.display()))?;
                let v: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                if v.len() != c * lh * lw {
                    bail!("{} has {} values, expected {}", f.display(), v.len(), c * lh * lw);
                }
                let mut hwc = vec![0f32; v.len()];
                for ch in 0..c {
                    for px in 0..lh * lw {
                        hwc[px * c + ch] = v[ch * lh * lw + px];
                    }
                }
                hwc.iter().map(|&v| v * init as f32).collect()
            }
            None => rng.normals(lh * lw * c).iter().map(|&v| v * init as f32).collect(),
        };
        for (i, &t) in timesteps.iter().enumerate() {
            let (sigma, next) = (sigmas[i], sigmas[i + 1]);
            let s = Schedule::scale_input(sigma);
            let xin: Vec<f32> = x.iter().map(|&v| v * s).collect();
            let mut out = self.unet.forward(&xin, lh, lw, t as f32, &cond, m);
            tm.unet_calls += 1;
            if let Some(u) = &uncond {
                let un = self.unet.forward(&xin, lh, lw, t as f32, u, m);
                tm.unet_calls += 1;
                for (o, &n) in out.iter_mut().zip(&un) {
                    *o = n + p.guidance * (*o - n);
                }
            }
            if i == 0 {
                dump_chw("unet_out0.bin", &out, lh, lw, c);
            }
            self.schedule.step(&mut x, &out, sigma, next, p.sampler, &mut rng);
            progress(i + 1, timesteps.len());
        }
        dump_chw("latents_out.bin", &x, lh, lw, c);
        tm.denoise = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let (px, h, w) = self.vae.decode(&x, lh, lw);
        dump_chw("image.bin", &px, h, w, 3);
        let rgb = px
            .iter()
            .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        tm.decode = t2.elapsed().as_secs_f64();
        Ok((Image { width: w, height: h, rgb }, tm))
    }
}
