//! Stable Diffusion 3 / 3.5: text to image and image to image, following diffusers'
//! `StableDiffusion3Pipeline` / `StableDiffusion3Img2ImgPipeline` (SD 3 medium, SD 3.5 medium,
//! large and large-turbo).
//!
//! New here: [`transformer`] (the MMDiT / MMDiT-X joint-attention transformer) and [`clip`]
//! (CLIP-L and OpenCLIP-bigG with their projection heads and penultimate states). The rest is
//! shared with FLUX: [`crate::flux::t5`] (T5-XXL, optional at run time), [`crate::flux::vae`]
//! (the 16-channel `AutoencoderKL`, here with SD 3's scale 1.5305 / shift 0.0609 from its
//! config), [`crate::flux::sched`] (flow-matching Euler, constant or resolution-dependent
//! shift) and [`crate::flux::weights`] (bf16 / q8 / q4 / GGUF linear layers).
//!
//! A pipeline loads from a diffusers folder (`model_index.json` with a `StableDiffusion3*`
//! class), optionally with the transformer replaced by a GGUF file in Stability AI's original
//! layout (e.g. `city96/stable-diffusion-3.5-medium-gguf`); the text encoders, VAE, tokenizers
//! and scheduler then come from the base folder.
//!
//! Conditioning: the penultimate states of both CLIP encoders, concatenated to 2048 channels
//! and zero-padded to T5's 4096, are followed by the T5 states (zeros without T5); the two
//! projected pooled CLIP embeddings (768 + 1280) go into the timestep conditioning. Guidance is
//! classifier-free (two passes per step when the scale is above 1).

pub mod clip;
pub mod transformer;

use crate::clock::Instant;
use crate::flux::sched::{FlowMatchEuler, Plan};
use crate::flux::t5::T5Encoder;
use crate::flux::vae::{self, Vae};
use crate::model::Quant;
use crate::sd::clip::ClipTokenizer;
use crate::video::RgbImage;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use crate::flux::{Image, Progress};

/// Stops a running generation from another thread.
#[derive(Clone, Debug)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// How to load a pipeline.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Precision of the transformer's float weights (GGUF weights keep their own format).
    pub quant: Quant,
    /// Precision of the T5 encoder's weights (default: `quant`).
    pub text_quant: Option<Quant>,
    /// A transformer to use instead of the base folder's `transformer/`: a GGUF file (original
    /// layout), a diffusers `transformer` folder, or (with the `hf` feature) `org/name:QUANT`.
    pub transformer: Option<String>,
    /// A T5 encoder to use instead of the base folder's `text_encoder_3/`: a llama.cpp
    /// `t5encoder` GGUF file (e.g. `city96/t5-v1_1-xxl-encoder-gguf:Q8_0`) or a folder.
    pub text_encoder: Option<String>,
    /// Where the text encoders, VAE, tokenizers and scheduler come from when the model given is
    /// a GGUF transformer (required then: the Stability AI repositories are gated).
    pub base: Option<String>,
    /// Load T5 (default true). Without it (or when the folder has none) its tokens are zeros,
    /// as in diffusers with `text_encoder_3=None`.
    pub t5: bool,
    pub device: crate::platform::DevicePref,
    pub cache_dir: Option<PathBuf>,
    /// Load the VAE encoder (needed for image to image; text to image only decodes).
    pub vae_encoder: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            text_quant: None,
            transformer: None,
            text_encoder: None,
            base: None,
            t5: true,
            device: crate::platform::DevicePref::Auto,
            cache_dir: None,
            vae_encoder: true,
        }
    }
}

impl LoadOptions {
    /// Sets one `key=value` option: quant, text_quant, transformer, text_encoder, base, t5,
    /// device, cache_dir, vae_encoder.
    pub fn set(&mut self, k: &str, v: &str) -> Result<()> {
        let q = |v: &str| Quant::parse(v).with_context(|| format!("{k}={v}: expected bf16, q8 or q4"));
        let flag = |v: &str| !matches!(v, "0" | "false" | "no");
        match k {
            "quant" => self.quant = q(v)?,
            "text_quant" => self.text_quant = Some(q(v)?),
            "transformer" => self.transformer = Some(v.to_string()),
            "text_encoder" => self.text_encoder = Some(v.to_string()),
            "base" => self.base = Some(v.to_string()),
            "t5" => self.t5 = flag(v),
            "device" => self.device = crate::platform::DevicePref::parse(v).with_context(|| format!("device={v}"))?,
            "cache_dir" => self.cache_dir = Some(PathBuf::from(v)),
            "vae_encoder" => self.vae_encoder = flag(v),
            _ => bail!("unknown SD 3 option {k}"),
        }
        Ok(())
    }
}

/// Generation options (diffusers' `__call__` arguments; `None` = the pipeline default).
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Options {
    /// Denoising steps (default 28; 50 for image to image; 4 for turbo models).
    pub steps: Option<usize>,
    /// Classifier-free guidance scale (default 7; 0 for turbo models). At 1 or below only the
    /// conditional pass runs.
    pub guidance: Option<f32>,
    /// The negative prompt of guided runs (default: empty).
    pub negative_prompt: Option<String>,
    /// Output size, multiples of 16 (default 1024 x 1024; image to image: the input's size).
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// T5 tokens (default 256, at most 512).
    pub max_sequence_length: usize,
    /// Image to image: how much of the schedule is run, 0..1 (default 0.6).
    pub strength: f32,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            steps: None,
            guidance: None,
            negative_prompt: None,
            width: None,
            height: None,
            max_sequence_length: 256,
            strength: 0.6,
            seed: 42,
        }
    }
}

/// Timings of one run, in seconds.
#[derive(Default, Debug, Clone)]
pub struct Timings {
    pub text: f64,
    pub encode: f64,
    pub denoise: f64,
    pub decode: f64,
    pub transformer_calls: usize,
    /// Tokens per transformer call (text + image).
    pub tokens: usize,
}

/// The text conditioning of one prompt.
#[derive(Clone, Debug)]
pub struct Conditioning {
    /// `[tokens, joint_dim]`: 77 CLIP tokens, then the T5 tokens.
    pub states: Vec<f32>,
    pub tokens: usize,
    /// `[pooled_dim]`: CLIP-L then CLIP-G projected pooled embeddings.
    pub pooled: Vec<f32>,
}

pub struct Pipeline {
    pub transformer: transformer::Sd3Transformer,
    pub clip_l: clip::ClipTextProj,
    pub clip_g: clip::ClipTextProj,
    pub tokenizer_l: ClipTokenizer,
    pub tokenizer_g: ClipTokenizer,
    /// None: run without T5 (zeros in its place).
    pub t5: Option<T5Encoder>,
    pub vae: Vae,
    pub scheduler: FlowMatchEuler,
    /// The base folder's pipeline class (`StableDiffusion3Pipeline`).
    pub class: String,
    /// A few-step distilled model (recognised by "turbo" in its name): 4 steps, no guidance.
    pub turbo: bool,
    pub load_secs: f64,
    cancel: Arc<AtomicBool>,
}

/// True for a diffusers Stable Diffusion 3 pipeline folder.
pub fn is_sd3_dir(dir: &Path) -> bool {
    crate::fsx::read(&dir.join("model_index.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["_class_name"].as_str().is_some_and(|c| c.starts_with("StableDiffusion3")))
}

/// True for a GGUF file holding an SD 3 transformer.
pub fn is_sd3_gguf(path: &Path) -> bool {
    crate::gguf::is_gguf(path) && crate::gguf::header(path).is_ok_and(|h| h.arch() == "sd3")
}

fn resolve_dir(spec: &str, skip: &[&str], cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    #[cfg(feature = "hf")]
    {
        crate::hf::resolve_diffusers(spec, skip, cache_dir)
    }
    #[cfg(not(feature = "hf"))]
    {
        let _ = (skip, cache_dir);
        bail!("{spec}: not a local folder (built without the `hf` feature)")
    }
}

fn resolve_file(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    crate::engine::resolve_model(spec, cache_dir)
}

/// Dumps for reference checks: `OXINFER_SD3_DUMP=DIR` writes intermediate tensors (f32 LE) and
/// reads `latents_in.bin` (the initial noise, `[16, h, w]`) and `posterior_noise_in.bin` (image
/// to image: the noise of the VAE posterior sample) from it.
fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("OXINFER_SD3_DUMP").map(PathBuf::from)
}

fn dump(name: &str, x: &[f32]) {
    if let Some(d) = dump_dir() {
        let _ = std::fs::write(d.join(name), x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    }
}

fn read_override(name: &str, len: usize) -> Result<Option<Vec<f32>>> {
    let Some(f) = dump_dir().map(|d| d.join(name)).filter(|f| f.exists()) else {
        return Ok(None);
    };
    let b = std::fs::read(&f).with_context(|| format!("read {}", f.display()))?;
    let v: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    if v.len() != len {
        bail!("{} has {} values, expected {len}", f.display(), v.len());
    }
    eprintln!("[using {}]", f.display());
    Ok(Some(v))
}

/// `np.linspace(start, stop, n)` in float32, as numpy computes it.
fn linspace_f32(start: f32, stop: f32, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f32;
    (0..n).map(|i| if i == n - 1 { stop } else { i as f32 * step + start }).collect()
}

/// The denoising plan diffusers' SD 3 pipelines get from `set_timesteps(steps, mu=...)`:
/// timesteps evenly spaced from `sigma_max` to `sigma_min` of the training schedule (which the
/// constructor already shifted by the constant `shift`), then shifted, by `shift` again or, with
/// dynamic shifting, by `mu` computed from the number of image tokens.
pub fn sd3_plan(s: &FlowMatchEuler, steps: usize, image_seq_len: usize) -> Result<Plan> {
    let n = s.num_train_timesteps as f32;
    let k = s.shift as f32;
    // The constructor's sigmas: linspace(1, N, N) / N, shifted unless the shift is dynamic.
    let shifted = |v: f32| if s.use_dynamic_shifting { v } else { k * v / (1.0 + (k - 1.0) * v) };
    let (smax, smin) = (shifted(1.0), shifted(1.0 / n));
    let base: Vec<f32> = linspace_f32(smax * n, smin * n, steps).iter().map(|&t| t / n).collect();
    s.plan(&base, s.use_dynamic_shifting.then(|| s.mu(image_seq_len)))
}

impl Pipeline {
    /// Loads a pipeline: `model` is a diffusers SD 3 folder or repo id, or an SD 3 GGUF
    /// transformer (file, or `org/name:QUANT`) whose other components come from `opts.base`.
    pub fn load(model: &str, opts: &LoadOptions) -> Result<Self> {
        let t0 = Instant::now();
        let cache = opts.cache_dir.as_deref();
        let model_path = Path::new(model);
        let gguf_model = model.contains(':') && !model_path.exists() || is_sd3_gguf(model_path);
        let (base, tr_spec) = if gguf_model {
            let base = opts.base.clone().context(
                "an SD 3 GGUF transformer needs `base`: a diffusers SD 3 folder or repo with the text encoders, VAE and tokenizers",
            )?;
            (base, Some(model.to_string()))
        } else {
            (model.to_string(), opts.transformer.clone())
        };
        let mut skip: Vec<&str> = Vec::new();
        if tr_spec.is_some() {
            skip.push("transformer");
        }
        if opts.text_encoder.is_some() || !opts.t5 {
            skip.push("text_encoder_3");
        }
        let dir = resolve_dir(&base, &skip, cache)?;
        let index: serde_json::Value = serde_json::from_slice(
            &crate::fsx::read(&dir.join("model_index.json"))
                .with_context(|| format!("{}: not a diffusers folder (no model_index.json)", dir.display()))?,
        )?;
        let class = index["_class_name"].as_str().unwrap_or("").to_string();
        if !class.starts_with("StableDiffusion3") {
            bail!("{}: {class} is not a Stable Diffusion 3 pipeline", dir.display());
        }
        crate::gpu::ensure(opts.device)?;
        let tr_path = match &tr_spec {
            Some(s) => resolve_file(s, cache)?,
            None => dir.join("transformer"),
        };
        let transformer =
            transformer::Sd3Transformer::load(&tr_path, opts.quant).with_context(|| format!("transformer {}", tr_path.display()))?;
        let t5_path = match &opts.text_encoder {
            Some(s) => Some(resolve_file(s, cache)?),
            None => {
                let d = dir.join("text_encoder_3");
                let has = ["model.safetensors", "model.safetensors.index.json"]
                    .iter()
                    .any(|n| crate::fsx::exists(&d.join(n)));
                has.then_some(d)
            }
        };
        let t5 = match t5_path {
            Some(p) if opts.t5 => Some(
                T5Encoder::load(&p, &dir.join("tokenizer_3"), opts.text_quant.unwrap_or(opts.quant))
                    .with_context(|| format!("T5 encoder {}", p.display()))?,
            ),
            _ => None,
        };
        let clip_l = clip::ClipTextProj::load(&dir.join("text_encoder")).context("text_encoder (CLIP-L)")?;
        let clip_g = clip::ClipTextProj::load(&dir.join("text_encoder_2")).context("text_encoder_2 (CLIP-G)")?;
        let cfg = &transformer.cfg;
        if clip_l.projection_dim + clip_g.projection_dim != cfg.pooled_dim || clip_l.hidden + clip_g.hidden > cfg.joint_dim {
            bail!(
                "the CLIP encoders ({} + {} wide, pooled {} + {}) do not fit the transformer (text {}, pooled {})",
                clip_l.hidden,
                clip_g.hidden,
                clip_l.projection_dim,
                clip_g.projection_dim,
                cfg.joint_dim,
                cfg.pooled_dim
            );
        }
        if let Some(t) = &t5
            && t.d_model != cfg.joint_dim
        {
            bail!("the T5 encoder is {} wide, the transformer takes {}", t.d_model, cfg.joint_dim);
        }
        let vae = Vae::load(&dir.join("vae"), opts.vae_encoder)?;
        if vae.latent_channels != cfg.in_channels {
            bail!(
                "transformer takes {} channels, the VAE has {}",
                cfg.in_channels,
                vae.latent_channels
            );
        }
        let scheduler = FlowMatchEuler::load(&dir.join("scheduler").join("scheduler_config.json"))?;
        let turbo = [model, base.as_str()]
            .iter()
            .chain(tr_spec.iter().map(|s| s.as_str()).collect::<Vec<_>>().iter())
            .any(|s| s.to_lowercase().contains("turbo"));
        Ok(Self {
            transformer,
            clip_l,
            clip_g,
            tokenizer_l: ClipTokenizer::load(&dir.join("tokenizer"))?,
            tokenizer_g: ClipTokenizer::load(&dir.join("tokenizer_2"))?,
            t5,
            vae,
            scheduler,
            class,
            turbo,
            load_secs: t0.elapsed().as_secs_f64(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancel.clone())
    }

    /// The defaults of `o` filled in: (steps, guidance scale).
    pub fn defaults(&self, o: &Options, img2img: bool) -> (usize, f32) {
        (
            o.steps
                .unwrap_or(if self.turbo {
                    4
                } else if img2img {
                    50
                } else {
                    28
                })
                .max(1),
            o.guidance.unwrap_or(if self.turbo { 0.0 } else { 7.0 }),
        )
    }

    /// The conditioning of `prompt` with `t5_len` T5 tokens (diffusers `encode_prompt`).
    pub fn encode_prompt(&self, prompt: &str, t5_len: usize) -> Result<Conditioning> {
        let jd = self.transformer.cfg.joint_dim;
        let run = |tok: &ClipTokenizer, enc: &clip::ClipTextProj| {
            let ids = tok.encode(prompt);
            let eos = ids.iter().position(|&i| i == tok.eos).unwrap_or(ids.len() - 1);
            (enc.forward(&ids, eos), ids.len())
        };
        let (l, nl) = run(&self.tokenizer_l, &self.clip_l);
        let (g, ng) = run(&self.tokenizer_g, &self.clip_g);
        if nl != ng {
            bail!("the two CLIP tokenizers pad to different lengths ({nl}, {ng})");
        }
        let (hl, hg) = (self.clip_l.hidden, self.clip_g.hidden);
        let tokens = nl + t5_len;
        let mut states = vec![0f32; tokens * jd];
        for i in 0..nl {
            let row = &mut states[i * jd..(i + 1) * jd];
            row[..hl].copy_from_slice(&l.penultimate[i * hl..(i + 1) * hl]);
            row[hl..hl + hg].copy_from_slice(&g.penultimate[i * hg..(i + 1) * hg]);
        }
        if let Some(t5) = &self.t5 {
            states[nl * jd..].copy_from_slice(&t5.encode(prompt, t5_len)?);
        }
        let mut pooled = l.pooled;
        pooled.extend_from_slice(&g.pooled);
        Ok(Conditioning { states, tokens, pooled })
    }

    /// Text to image (diffusers `StableDiffusion3Pipeline`).
    pub fn generate(&self, prompt: &str, o: &Options, progress: Progress) -> Result<(Image, Timings)> {
        let (w, h) = (o.width.unwrap_or(1024), o.height.unwrap_or(1024));
        self.run(prompt, None, w, h, o, progress)
    }

    /// Image to image (diffusers `StableDiffusion3Img2ImgPipeline`): `image`, resized to the
    /// output size, is noised to `o.strength` of the schedule and denoised following `prompt`.
    pub fn img2img(&self, image: &RgbImage, prompt: &str, o: &Options, progress: Progress) -> Result<(Image, Timings)> {
        if image.width < 16 || image.height < 16 || image.data.len() != image.width * image.height * 3 {
            bail!("the input image must be RGB8 and at least 16x16");
        }
        if !(0.0..=1.0).contains(&o.strength) {
            bail!("strength must be between 0 and 1 (got {})", o.strength);
        }
        let (w, h) = (o.width.unwrap_or(image.width), o.height.unwrap_or(image.height));
        self.run(prompt, Some(image), w, h, o, progress)
    }

    fn run(
        &self,
        prompt: &str,
        init: Option<&RgbImage>,
        width: usize,
        height: usize,
        o: &Options,
        progress: Progress,
    ) -> Result<(Image, Timings)> {
        self.cancel.store(false, Ordering::Relaxed);
        let mut tm = Timings::default();
        let cfg = &self.transformer.cfg;
        let c = cfg.in_channels;
        // Multiples of the VAE factor times the patch size.
        let unit = self.vae.factor * cfg.patch_size;
        let (width, height) = (width / unit * unit, height / unit * unit);
        if width == 0 || height == 0 {
            bail!("image size must be at least {unit}x{unit}");
        }
        let (steps, guidance) = self.defaults(o, init.is_some());
        let seq = o.max_sequence_length.clamp(1, 512);

        let t0 = Instant::now();
        let cond = self.encode_prompt(prompt, seq)?;
        dump("prompt_embeds.bin", &cond.states);
        dump("pooled.bin", &cond.pooled);
        let uncond = if guidance > 1.0 {
            let u = self.encode_prompt(o.negative_prompt.as_deref().unwrap_or(""), seq)?;
            dump("neg_prompt_embeds.bin", &u.states);
            dump("neg_pooled.bin", &u.pooled);
            Some(u)
        } else {
            None
        };
        tm.text = t0.elapsed().as_secs_f64();

        let (lh, lw) = (height / self.vae.factor, width / self.vae.factor);
        let n_img = (lh / cfg.patch_size) * (lw / cfg.patch_size);
        tm.tokens = cond.tokens + n_img;
        let plan = sd3_plan(&self.scheduler, steps, n_img)?;
        dump("sigmas.bin", &plan.sigmas);
        let mut rng = crate::sd::ops::Rng::new(o.seed);

        // Image to image: the VAE posterior sample of the input, then noise at the start sigma.
        let t1 = Instant::now();
        let (init_latents, first) = match init {
            None => (None, 0),
            Some(img) => {
                let px = if (width, height) == (img.width, img.height) {
                    img.data.clone()
                } else {
                    crate::resample::resize_lanczos_rgb8(&img.data, img.width, img.height, width, height)
                };
                let enc = self.vae.encoder.as_ref().context("the VAE was loaded without its encoder")?;
                let (m, mh, mw) = enc.moments(&vae::rgb8_to_unit(&px), height, width);
                if (mh, mw) != (lh, lw) {
                    bail!("the VAE gave a {mw}x{mh} latent for {width}x{height}");
                }
                let eps = match read_override("posterior_noise_in.bin", c * lh * lw)? {
                    Some(v) => v,
                    None => rng.normals(c * lh * lw),
                };
                let eps = vae::chw_to_hwc(&eps, c, lh, lw);
                let (sc, sh) = (self.vae.scaling_factor, self.vae.shift_factor);
                let z: Vec<f32> = (0..lh * lw * c)
                    .map(|i| {
                        let (p, ch) = (i / c, i % c);
                        let (mean, logvar) = (m[p * 2 * c + ch], m[p * 2 * c + c + ch].clamp(-30.0, 20.0));
                        (mean + (0.5 * logvar).exp() * eps[i] - sh) * sc
                    })
                    .collect();
                dump("init_latents.bin", &vae::hwc_to_chw(&z, c, lh, lw));
                // get_timesteps: the last `steps * strength` steps.
                let run = ((steps as f32 * o.strength) as usize).min(steps);
                if run == 0 {
                    bail!("strength {} with {steps} steps leaves no denoising step", o.strength);
                }
                (Some(z), steps - run)
            }
        };
        tm.encode = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let noise = match read_override("latents_in.bin", c * lh * lw)? {
            Some(v) => v,
            None => rng.normals(c * lh * lw),
        };
        let noise = vae::chw_to_hwc(&noise, c, lh, lw);
        let mut x = match &init_latents {
            Some(z) => FlowMatchEuler::scale_noise(z, &noise, plan.sigmas[first]),
            None => noise,
        };
        let total = steps - first;
        for i in first..steps {
            if self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            let t = plan.timesteps[i];
            let mut v = self.transformer.forward(&x, lh, lw, &cond.states, cond.tokens, &cond.pooled, t)?;
            tm.transformer_calls += 1;
            if i == first {
                dump("noise_pred0.bin", &vae::hwc_to_chw(&v, c, lh, lw));
            }
            if let Some(u) = &uncond {
                let nu = self.transformer.forward(&x, lh, lw, &u.states, u.tokens, &u.pooled, t)?;
                tm.transformer_calls += 1;
                if i == first {
                    dump("noise_pred0_uncond.bin", &vae::hwc_to_chw(&nu, c, lh, lw));
                }
                for (a, &b) in v.iter_mut().zip(&nu) {
                    *a = b + guidance * (*a - b);
                }
            }
            FlowMatchEuler::step(&plan, i, &mut x, &v);
            if !progress(i + 1 - first, total) {
                bail!("cancelled");
            }
        }
        tm.denoise = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        dump("latents_out.bin", &vae::hwc_to_chw(&x, c, lh, lw));
        let (px, h, w) = self.vae.decode(&x, lh, lw);
        dump("image.bin", &vae::hwc_to_chw(&px, 3, h, w));
        tm.decode = t3.elapsed().as_secs_f64();
        Ok((
            Image {
                width: w,
                height: h,
                rgb: vae::unit_to_rgb8(&px),
            },
            tm,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd3_schedule() {
        // diffusers 0.42 FlowMatchEulerDiscreteScheduler(shift=3.0).set_timesteps(4): the base
        // timesteps run down to the training schedule's already shifted sigma_min, and are
        // shifted again.
        let s = FlowMatchEuler {
            shift: 3.0,
            ..Default::default()
        };
        let p = sd3_plan(&s, 4, 4096).unwrap();
        let want = [1.0f32, 0.857_692_3, 0.602_150_56, 0.008_928_572, 0.0];
        for (a, b) in p.sigmas.iter().zip(want) {
            assert!((a - b).abs() < 2e-6, "{:?}", p.sigmas);
        }
        assert!((p.timesteps[1] - 857.692_3).abs() < 2e-3);
        let p = sd3_plan(&s, 28, 4096).unwrap();
        assert!((p.sigmas[1] - 0.987_380_6).abs() < 2e-6 && (p.sigmas[27] - 0.008_928_572).abs() < 2e-6);
        // use_dynamic_shifting=True, mu = calculate_shift(1024, 256, 4096, 0.5, 1.15) = 0.63.
        let s = FlowMatchEuler {
            shift: 3.0,
            use_dynamic_shifting: true,
            ..Default::default()
        };
        let p = sd3_plan(&s, 4, 1024).unwrap();
        let want = [1.0f32, 0.789_953_9, 0.484_967_35, 0.001_875_964, 0.0];
        for (a, b) in p.sigmas.iter().zip(want) {
            assert!((a - b).abs() < 2e-6, "{:?}", p.sigmas);
        }
        assert_eq!(linspace_f32(1.0, 0.25, 4), vec![1.0, 0.75, 0.5, 0.25]);
    }
}
