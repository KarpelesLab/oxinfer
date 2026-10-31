//! Text-to-image with Stable Diffusion 1.x / 2.x and SD-Turbo from a diffusers folder
//! (`model_index.json`, `text_encoder/`, `tokenizer/`, `unet/`, `vae/`, `scheduler/`), and image
//! editing: image-to-image, inpainting (dedicated 9-channel inpainting UNets, or mask blending
//! with a text-to-image one) and InstructPix2Pix instruction edits ([`Pipeline::generate_from`]).
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
use crate::video::RgbImage;
use anyhow::{Context, Result, bail};
use ops::{Rng, concat_channels};
use sched::{Sampler, Schedule};
use std::path::{Path, PathBuf};

pub struct Pipeline {
    pub tokenizer: clip::ClipTokenizer,
    pub text: clip::ClipText,
    pub unet: unet::Unet,
    pub vae: vae::VaeDecoder,
    /// The VAE encoder, loaded on the first edit.
    encoder: std::sync::OnceLock<vae::VaeEncoder>,
    vae_weights: PathBuf,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
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
    /// Encoding the input image (and masked image) to latents; 0 for text to image.
    pub encode: f64,
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
        let vae_weights = weights(&va, "diffusion_pytorch_model")?;
        let vae = vae::VaeDecoder::load(&va, &vae_weights)?;
        // Distilled single-step models are named as such (folder, or the name recorded at export).
        let name = format!("{} {}", dir.to_string_lossy(), index["_name_or_path"].as_str().unwrap_or("")).to_lowercase();
        Ok(Self {
            tokenizer: clip::ClipTokenizer::load(&dir.join("tokenizer"))?,
            text,
            unet,
            vae,
            encoder: std::sync::OnceLock::new(),
            vae_weights,
            cancel: Default::default(),
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

    /// What an edit with this pipeline does, from the UNet's input channels: 4 (a text-to-image
    /// UNet) gives image-to-image, or inpainting by mask blending when there is a mask; 9 (an
    /// inpainting UNet: latents + mask + masked-image latents) needs a mask; 8 (InstructPix2Pix:
    /// latents + image latents) follows an instruction.
    pub fn edit_kind(&self, with_mask: bool) -> Result<EditKind> {
        let lc = self.vae.latent_channels;
        let c = self.unet.in_channels;
        Ok(match (c, with_mask) {
            _ if c == lc => {
                if with_mask {
                    EditKind::InpaintBlend
                } else {
                    EditKind::Img2Img
                }
            }
            _ if c == 2 * lc + 1 => {
                if !with_mask {
                    bail!("this is an inpainting model ({c}-channel UNet): it needs a mask (white = repaint)");
                }
                EditKind::Inpaint
            }
            _ if c == 2 * lc => {
                if with_mask {
                    bail!("this is an InstructPix2Pix model: it takes an image and an instruction, not a mask");
                }
                EditKind::Instruct
            }
            _ => bail!("a UNet with {c} input channels is not supported (VAE latents have {lc})"),
        })
    }

    /// Whether this pipeline only edits images (inpainting and InstructPix2Pix UNets).
    pub fn needs_image(&self) -> bool {
        self.unet.in_channels != self.vae.latent_channels
    }

    /// The size an edit of a `w` x `h` image runs at when the parameters give none: the model's
    /// default area with the image's aspect ratio, each side a multiple of 64.
    pub fn edit_size(&self, w: usize, h: usize) -> (usize, usize) {
        let side = self.default_size() as f64;
        let aspect = w.max(1) as f64 / h.max(1) as f64;
        let tw = side * aspect.sqrt();
        let th = side / aspect.sqrt();
        let r = |v: f64| ((v / 64.0).round() as usize).max(1) * 64;
        (r(tw), r(th))
    }

    /// Generates one image from text. `progress(step, steps)` is called after each denoising
    /// step. Inpainting and InstructPix2Pix models need an image: see [`Pipeline::generate_from`].
    pub fn generate(&self, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        if self.needs_image() {
            bail!(
                "this {}-channel UNet edits images: give it an image (and a mask for inpainting)",
                self.unet.in_channels
            );
        }
        self.run(p, None, progress)
    }

    /// Image-to-image: `image` noised to `strength` (0..=1: the fraction of the `p.steps`
    /// schedule that is run) and denoised towards the prompt.
    pub fn img2img(&self, p: &Params, image: &RgbImage, strength: f32, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.generate_from(p, &EditInput::new(image.clone()).with_strength(strength), progress)
    }

    /// Inpainting: repaints where `mask` is white (luma >= 128), keeps the rest. With an
    /// inpainting UNet the model sees the mask and the masked image; with a text-to-image UNet
    /// the kept area is blended back in after each step. `strength` 1 starts from pure noise.
    pub fn inpaint(
        &self,
        p: &Params,
        image: &RgbImage,
        mask: &RgbImage,
        strength: f32,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<(Image, Timings)> {
        self.generate_from(
            p,
            &EditInput::new(image.clone()).with_mask(mask.clone()).with_strength(strength),
            progress,
        )
    }

    /// An instruction edit with InstructPix2Pix: `p.prompt` says what to change ("make it
    /// snowy"); `image_guidance` (default 1.5) is how closely the result follows the input
    /// image, `p.guidance` how strongly it follows the instruction.
    pub fn edit(
        &self,
        p: &Params,
        image: &RgbImage,
        image_guidance: f32,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<(Image, Timings)> {
        if self.edit_kind(false)? != EditKind::Instruct {
            bail!(
                "instruction edits need an InstructPix2Pix model (e.g. timbrooks/instruct-pix2pix); use img2img or inpaint with this one"
            );
        }
        self.generate_from(p, &EditInput::new(image.clone()).with_image_guidance(image_guidance), progress)
    }

    /// Any edit: what it does follows from the model and the input ([`Pipeline::edit_kind`]).
    /// `p.width` / `p.height` 0 pick [`Pipeline::edit_size`]; the image (and mask) are scaled to
    /// cover that size and centre-cropped.
    pub fn generate_from(&self, p: &Params, e: &EditInput, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.run(p, Some(e), progress)
    }

    /// A flag that stops the running generation (from any thread) when set: it then fails with
    /// "cancelled" after the current denoising step. Each generation clears it when it starts.
    pub fn cancel_flag(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.cancel.clone()
    }

    fn encoder(&self) -> Result<&vae::VaeEncoder> {
        if let Some(e) = self.encoder.get() {
            return Ok(e);
        }
        let e = vae::VaeEncoder::load(&self.vae_weights)?.context("this VAE has no encoder weights (decoder-only export)")?;
        Ok(self.encoder.get_or_init(|| e))
    }

    fn run(&self, p: &Params, e: Option<&EditInput>, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        let kind = match e {
            Some(e) => Some(self.edit_kind(e.mask.is_some())?),
            None => None,
        };
        let (width, height) = match e {
            Some(e) if p.width == 0 || p.height == 0 => self.edit_size(e.image.width, e.image.height),
            _ => (p.width, p.height),
        };
        if !width.is_multiple_of(8) || !height.is_multiple_of(8) || width < 64 || height < 64 {
            bail!("image size must be a multiple of 8 and at least 64 (got {width}x{height})");
        }
        // Three 2x downsamplings in the UNet: the latent side must divide by 8.
        let (lh, lw) = (height / 8, width / 8);
        if !lh.is_multiple_of(8) || !lw.is_multiple_of(8) {
            bail!("image size must be a multiple of 64 for this UNet (got {width}x{height})");
        }
        let n = lh * lw;
        let lc = self.vae.latent_channels;
        let sf = self.vae.scaling_factor;
        let steps = p.steps.max(1);
        let strength = match kind {
            Some(EditKind::Img2Img) => e.and_then(|e| e.strength).unwrap_or(DEFAULT_STRENGTH),
            Some(EditKind::Inpaint | EditKind::InpaintBlend) => e.and_then(|e| e.strength).unwrap_or(1.0),
            _ => 1.0,
        };
        if !(strength > 0.0 && strength <= 1.0) {
            bail!("strength must be in (0, 1] (got {strength})");
        }
        let image_guidance = e.and_then(|e| e.image_guidance).unwrap_or(DEFAULT_IMAGE_GUIDANCE);
        // diffusers' get_timesteps: the last int(steps * strength) steps of the schedule run.
        let run_steps = ((steps as f64 * strength as f64 + 1e-4).floor() as usize).min(steps);
        if run_steps == 0 {
            bail!(
                "strength {strength} of {steps} steps leaves no step to run: use at least {} steps",
                (1.0 / strength as f64).ceil() as usize
            );
        }
        let start = steps - run_steps;

        use std::sync::atomic::Ordering::Relaxed;
        self.cancel.store(false, Relaxed);
        let mut tm = Timings::default();
        let t0 = Instant::now();
        let m = self.tokenizer.max_len;
        let cond = self.encode_prompt(&p.prompt);
        let guided = match kind {
            Some(EditKind::Instruct) => p.guidance > 1.0 && image_guidance >= 1.0,
            _ => p.guidance > 1.0,
        };
        let uncond = if guided { Some(self.encode_prompt(&p.negative)) } else { None };
        dump_raw("text_cond.bin", &cond);
        tm.text = t0.elapsed().as_secs_f64();

        let (timesteps, sigmas, init) = self.schedule.plan(steps);
        let mut rng = Rng::new(p.seed);

        // The image and mask at the working size, and their latents.
        let te = Instant::now();
        let pixels = e.map(|e| to_unit_range(&fit(&e.image, width, height)));
        let mask_px: Option<Vec<f32>> = e.and_then(|e| e.mask.as_ref()).map(|mk| binary_mask(&fit(mk, width, height)));
        let mut enc_dumped = false;
        let mut encode = |px: &[f32], role: &str, rng: &mut Rng, mode: bool| -> Result<Vec<f32>> {
            let d = self.encoder()?.encode(px, height, width);
            if !enc_dumped {
                dump_chw("enc_mean.bin", &d.mean, lh, lw, lc);
                enc_dumped = true;
            }
            Ok(if mode {
                d.mean
            } else {
                let noise = draw_noise(role, n, lc, lh, lw, rng)?;
                d.sample_with(&noise).iter().map(|v| v * sf).collect()
            })
        };
        // diffusers' order of random draws: image latents, the initial noise, masked-image latents.
        let image_latents = match kind {
            Some(EditKind::Img2Img | EditKind::InpaintBlend) => Some(encode(pixels.as_deref().unwrap(), "enc_image", &mut rng, false)?),
            Some(EditKind::Inpaint) if strength < 1.0 => Some(encode(pixels.as_deref().unwrap(), "enc_image", &mut rng, false)?),
            // InstructPix2Pix conditions on the unscaled mode of the latent distribution.
            Some(EditKind::Instruct) => Some(encode(pixels.as_deref().unwrap(), "enc_image", &mut rng, true)?),
            _ => None,
        };
        let noise = draw_noise("latents_in", n, lc, lh, lw, &mut rng)?;
        // The 9-channel UNet's extra input: the latent-resolution mask and the masked image's latents.
        let mut extra: Option<(Vec<f32>, usize)> = None;
        let mut blend_mask: Option<Vec<f32>> = None;
        if let (Some(mpx), Some(px)) = (&mask_px, &pixels) {
            // Nearest-neighbour downsampling (torch's interpolate default): pixel (8y, 8x).
            let ml: Vec<f32> = (0..n).map(|i| mpx[(i / lw) * 8 * width + (i % lw) * 8]).collect();
            if kind == Some(EditKind::Inpaint) {
                let masked: Vec<f32> = px
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .zip(mpx)
                    .flat_map(|(c, &mv)| c.iter().map(move |&v| if mv < 0.5 { v } else { 0.0 }))
                    .collect();
                let ml_lat = encode(&masked, "enc_masked", &mut rng, false)?;
                let mut x = Vec::with_capacity(n * (lc + 1));
                for i in 0..n {
                    x.push(ml[i]);
                    x.extend_from_slice(&ml_lat[i * lc..(i + 1) * lc]);
                }
                extra = Some((x, lc + 1));
            } else {
                blend_mask = Some(ml);
            }
        }
        if let Some(il) = &image_latents
            && kind == Some(EditKind::Instruct)
        {
            extra = Some((il.clone(), lc));
        }
        tm.encode = te.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut x: Vec<f32> = match (&image_latents, kind) {
            (Some(z), Some(EditKind::Img2Img)) => z.iter().zip(&noise).map(|(&z, &e)| z + e * sigmas[start] as f32).collect(),
            (Some(z), Some(EditKind::Inpaint | EditKind::InpaintBlend)) if strength < 1.0 => {
                z.iter().zip(&noise).map(|(&z, &e)| z + e * sigmas[start] as f32).collect()
            }
            _ => noise.iter().map(|&v| v * init as f32).collect(),
        };
        let zeros = vec![0f32; n * lc];
        for i in start..steps {
            let (t, sigma, next) = (timesteps[i], sigmas[i], sigmas[i + 1]);
            let s = Schedule::scale_input(sigma);
            let xs: Vec<f32> = x.iter().map(|&v| v * s).collect();
            let xin = match &extra {
                Some((ex, ce)) => concat_channels(&xs, lc, ex, *ce, n),
                None => xs.clone(),
            };
            let mut out = self.unet.forward(&xin, lh, lw, t as f32, &cond, m);
            tm.unet_calls += 1;
            if let Some(u) = &uncond {
                if kind == Some(EditKind::Instruct) {
                    // Three passes: text + image, image only, neither (image latents zeroed).
                    let img = self.unet.forward(&xin, lh, lw, t as f32, u, m);
                    let xin0 = concat_channels(&xs, lc, &zeros, lc, n);
                    let un = self.unet.forward(&xin0, lh, lw, t as f32, u, m);
                    tm.unet_calls += 2;
                    for ((o, &im), &un) in out.iter_mut().zip(&img).zip(&un) {
                        *o = un + p.guidance * (*o - im) + image_guidance * (im - un);
                    }
                } else {
                    let un = self.unet.forward(&xin, lh, lw, t as f32, u, m);
                    tm.unet_calls += 1;
                    for (o, &n) in out.iter_mut().zip(&un) {
                        *o = n + p.guidance * (*o - n);
                    }
                }
            }
            if i == start {
                dump_chw("unet_out0.bin", &out, lh, lw, lc);
            }
            self.schedule.step(&mut x, &out, sigma, next, p.sampler, &mut rng);
            if let (Some(bm), Some(z)) = (&blend_mask, &image_latents) {
                // Keep the unmasked area: the input's latents noised to the next level.
                let nsig = if i + 1 < steps { sigmas[i + 1] as f32 } else { 0.0 };
                for (j, xv) in x.iter_mut().enumerate() {
                    let mk = bm[j / lc];
                    let keep = z[j] + noise[j] * nsig;
                    *xv = (1.0 - mk) * keep + mk * *xv;
                }
            }
            progress(i + 1 - start, steps - start);
            if self.cancel.load(Relaxed) {
                bail!("cancelled");
            }
        }
        dump_chw("latents_out.bin", &x, lh, lw, lc);
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

/// img2img's default strength (diffusers' `StableDiffusionImg2ImgPipeline`).
pub const DEFAULT_STRENGTH: f32 = 0.8;
/// InstructPix2Pix's default image guidance (diffusers').
pub const DEFAULT_IMAGE_GUIDANCE: f32 = 1.5;

/// Default step count: 25 (20 for instruction edits); distilled (turbo) models 1 for text to
/// image, and for edits enough that `steps * strength` reaches one step (at least 2).
pub fn default_steps(turbo: bool, kind: Option<EditKind>, strength: f32) -> usize {
    match kind {
        _ if turbo => ((1.0 / strength.max(0.01) as f64 - 1e-4).ceil() as usize).max(if kind.is_some() { 2 } else { 1 }),
        Some(EditKind::Instruct) => 20,
        _ => 25,
    }
}

/// What [`Pipeline::generate_from`] does with an input image (see [`Pipeline::edit_kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditKind {
    /// Noise the image's latents to `strength`, then denoise towards the prompt.
    Img2Img,
    /// A 9-channel inpainting UNet: the mask and masked image are part of its input.
    Inpaint,
    /// A text-to-image UNet with a mask: the unmasked area is reset after every step.
    InpaintBlend,
    /// InstructPix2Pix: the image latents are part of the UNet input; the prompt is an instruction.
    Instruct,
}

/// The image (and mask) an edit starts from.
#[derive(Clone, Debug)]
pub struct EditInput {
    pub image: RgbImage,
    /// White (luma >= 128) = repaint, black = keep; any size (scaled like the image).
    pub mask: Option<RgbImage>,
    /// img2img and inpainting: how much of the schedule is re-run, in (0, 1]; default
    /// [`DEFAULT_STRENGTH`] for img2img, 1 (start from pure noise) for inpainting.
    pub strength: Option<f32>,
    /// InstructPix2Pix: how closely to follow the input image (default [`DEFAULT_IMAGE_GUIDANCE`];
    /// below 1 turns all guidance off).
    pub image_guidance: Option<f32>,
}

impl EditInput {
    pub fn new(image: RgbImage) -> Self {
        Self {
            image,
            mask: None,
            strength: None,
            image_guidance: None,
        }
    }

    pub fn with_mask(mut self, mask: RgbImage) -> Self {
        self.mask = Some(mask);
        self
    }

    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = Some(strength);
        self
    }

    pub fn with_image_guidance(mut self, g: f32) -> Self {
        self.image_guidance = Some(g);
        self
    }
}

#[cfg(feature = "oxideav")]
impl EditInput {
    /// An edit of an oxideav picture (any pixel format [`crate::av::Picture::to_rgb`] converts).
    pub fn from_picture(image: &crate::av::Picture) -> Result<Self> {
        Ok(Self::new(image.to_rgb()?))
    }

    /// Adds a mask given as an oxideav picture (white = repaint).
    pub fn with_mask_picture(self, mask: &crate::av::Picture) -> Result<Self> {
        Ok(self.with_mask(mask.to_rgb()?))
    }
}

/// `img` scaled to cover `w` x `h` (Pillow-style bicubic, keeping its aspect ratio) and
/// centre-cropped; unchanged when it already has that size.
pub fn fit(img: &RgbImage, w: usize, h: usize) -> RgbImage {
    if img.width == w && img.height == h {
        return img.clone();
    }
    let s = (w as f64 / img.width as f64).max(h as f64 / img.height as f64);
    let sw = ((img.width as f64 * s).round() as usize).max(w);
    let sh = ((img.height as f64 * s).round() as usize).max(h);
    let big = crate::resample::resize_bicubic_rgb8(&img.data, img.width, img.height, sw, sh);
    let (x0, y0) = ((sw - w) / 2, (sh - h) / 2);
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let r = ((y0 + y) * sw + x0) * 3;
        data.extend_from_slice(&big[r..r + w * 3]);
    }
    RgbImage { width: w, height: h, data }
}

/// RGB8 to `[h*w, 3]` in [-1, 1] (diffusers' `VaeImageProcessor`: x / 255 * 2 - 1).
fn to_unit_range(img: &RgbImage) -> Vec<f32> {
    img.data.iter().map(|&v| 2.0 * (v as f32 / 255.0) - 1.0).collect()
}

/// Per pixel 1 (repaint) or 0 (keep): Pillow's luma (`convert("L")`) binarized at 0.5.
fn binary_mask(img: &RgbImage) -> Vec<f32> {
    img.data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            let l = (c[0] as u32 * 19595 + c[1] as u32 * 38470 + c[2] as u32 * 7471 + 0x8000) >> 16;
            if l as f32 / 255.0 < 0.5 { 0.0 } else { 1.0 }
        })
        .collect()
}

/// `n * c` standard normals, channels-last; for validation, `OXINFER_SD_DUMP/<role>.bin`
/// (`[c, h, w]` f32, as tools/ref_sd.py writes the reference's draws) when it exists.
fn draw_noise(role: &str, n: usize, c: usize, h: usize, w: usize, rng: &mut Rng) -> Result<Vec<f32>> {
    if let Some(f) = dump_dir().map(|d| d.join(format!("{role}.bin"))).filter(|f| f.exists()) {
        let b = std::fs::read(&f).with_context(|| format!("read {}", f.display()))?;
        let v: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        if v.len() != c * h * w {
            bail!("{} has {} values, expected {}", f.display(), v.len(), c * h * w);
        }
        let mut hwc = vec![0f32; v.len()];
        for ch in 0..c {
            for px in 0..h * w {
                hwc[px * c + ch] = v[ch * h * w + px];
            }
        }
        return Ok(hwc);
    }
    Ok(rng.normals(n * c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_is_binarized_luma() {
        let img = RgbImage {
            width: 4,
            height: 1,
            data: vec![0, 0, 0, 255, 255, 255, 128, 128, 128, 127, 127, 127],
        };
        assert_eq!(binary_mask(&img), vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn fit_keeps_same_size_and_crops_to_cover() {
        let data: Vec<u8> = (0..8 * 4 * 3).map(|i| i as u8).collect();
        let img = RgbImage { width: 8, height: 4, data };
        assert_eq!(fit(&img, 8, 4).data, img.data);
        let sq = fit(&img, 4, 4);
        assert_eq!((sq.width, sq.height, sq.data.len()), (4, 4, 48));
        // A 2:1 image into a square: the centre half of each row (no scaling needed).
        assert_eq!(&sq.data[..12], &img.data[6..18]);
    }

    #[test]
    fn unit_range() {
        let img = RgbImage {
            width: 1,
            height: 1,
            data: vec![0, 255, 51],
        };
        let v = to_unit_range(&img);
        assert_eq!(v[0], -1.0);
        assert_eq!(v[1], 1.0);
        assert!((v[2] - (-0.6)).abs() < 1e-6);
    }
}
