//! FLUX.1: text to image ([schnell] / [dev]) and instruction-based image editing (FLUX.1
//! Kontext [dev]), following diffusers' `FluxPipeline` / `FluxKontextPipeline`.
//!
//! The pieces are reusable on their own: [`vae`] (the 16-channel FLUX autoencoder, encode and
//! decode, 2x2 latent packing), [`sched`] (flow-matching Euler with resolution-dependent shift),
//! [`rope`] (multi-axis rotary embeddings over token ids), [`transformer`] (the double / single
//! stream blocks), [`t5`] (the T5 v1.1 encoder) and [`weights`] (linear layers from safetensors
//! or GGUF, bf16 / q8 / q4 / any GGUF block format).
//!
//! A pipeline loads from a diffusers folder (`model_index.json`; e.g. `unsloth/FLUX.1-Kontext-dev`,
//! an ungated copy of `black-forest-labs/FLUX.1-Kontext-dev`), optionally with the transformer
//! replaced by a GGUF file in the original BFL layout (e.g. `QuantStack/FLUX.1-Kontext-dev-GGUF`,
//! `city96/FLUX.1-schnell-gguf`); the text encoders, VAE and tokenizers then come from the base
//! folder. Every FLUX.1 variant shares those.
//!
//! Kontext editing: the reference image is resized to the preferred resolution closest to its
//! aspect ratio, encoded by the VAE, packed and appended to the image tokens with RoPE index 1;
//! the transformer's output for those tokens is dropped. Guidance is the distilled guidance
//! embedding (no classifier-free guidance unless `true_cfg_scale` > 1 with a negative prompt).

pub mod rope;
pub mod sched;
pub mod t5;
pub mod transformer;
pub mod vae;
pub mod weights;

use crate::clock::Instant;
use crate::model::Quant;
use crate::video::RgbImage;
use anyhow::{Context, Result, bail};
use rope::Rope;
use sched::FlowMatchEuler;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// FLUX.1 Kontext's preferred resolutions (width, height), about one megapixel each.
pub const PREFERRED_KONTEXT_RESOLUTIONS: [(usize, usize); 17] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
];

/// The default diffusers base for GGUF transformers: an ungated copy of FLUX.1 Kontext [dev]
/// (its text encoders, VAE, tokenizers and scheduler are those of every FLUX.1 model).
pub const DEFAULT_BASE: &str = "unsloth/FLUX.1-Kontext-dev";

/// How to load a pipeline.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Precision of the transformer's float weights (GGUF weights keep their own format).
    pub quant: Quant,
    /// Precision of the T5 encoder's weights (default: `quant`).
    pub text_quant: Option<Quant>,
    /// A transformer to use instead of the base folder's `transformer/`: a GGUF file (BFL layout),
    /// a diffusers `transformer` folder, or (with the `hf` feature) `org/name:QUANT` / `org/name`.
    pub transformer: Option<String>,
    /// A T5 encoder to use instead of the base folder's `text_encoder_2/`: a llama.cpp
    /// `t5encoder` GGUF file (e.g. `city96/t5-v1_1-xxl-encoder-gguf:Q8_0`) or a folder.
    pub text_encoder: Option<String>,
    /// Where the text encoders, VAE, tokenizers and scheduler come from when the model given is a
    /// GGUF transformer (default [`DEFAULT_BASE`]).
    pub base: Option<String>,
    pub device: crate::platform::DevicePref,
    pub cache_dir: Option<PathBuf>,
    /// Load the VAE encoder (needed for editing; text-to-image only decodes).
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
            device: crate::platform::DevicePref::Auto,
            cache_dir: None,
            vae_encoder: true,
        }
    }
}

impl LoadOptions {
    /// Sets one `key=value` option (C API): quant, text_quant, transformer, text_encoder, base,
    /// device, cache_dir, vae_encoder.
    pub fn set(&mut self, k: &str, v: &str) -> Result<()> {
        let q = |v: &str| Quant::parse(v).with_context(|| format!("{k}={v}: expected bf16, q8 or q4"));
        match k {
            "quant" => self.quant = q(v)?,
            "text_quant" => self.text_quant = Some(q(v)?),
            "transformer" => self.transformer = Some(v.to_string()),
            "text_encoder" => self.text_encoder = Some(v.to_string()),
            "base" => self.base = Some(v.to_string()),
            "device" => self.device = crate::platform::DevicePref::parse(v).with_context(|| format!("device={v}"))?,
            "cache_dir" => self.cache_dir = Some(PathBuf::from(v)),
            "vae_encoder" => self.vae_encoder = !matches!(v, "0" | "false" | "no"),
            _ => bail!("unknown FLUX option {k}"),
        }
        Ok(())
    }
}

/// Generation / editing options (diffusers' `__call__` arguments; `None` = the pipeline default).
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Options {
    /// Denoising steps (default 28; 4 for models without guidance embedding, i.e. schnell).
    pub steps: Option<usize>,
    /// Distilled guidance (default 3.5 as diffusers; BFL suggests 2.5 for Kontext edits).
    pub guidance: Option<f32>,
    /// Output size before `max_area` scaling (default 1024 x 1024, as diffusers; the CLI passes
    /// the reference image's size).
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Kontext: the output is rescaled to this many pixels at the requested aspect ratio
    /// (default 1024^2).
    pub max_area: usize,
    /// Kontext: resize the reference image to the closest preferred resolution (default true).
    pub auto_resize: bool,
    /// T5 tokens (default 512; schnell is trained with 256).
    pub max_sequence_length: Option<usize>,
    pub seed: u64,
    /// True classifier-free guidance (an extra pass per step) when > 1 with a negative prompt.
    pub true_cfg_scale: f32,
    pub negative_prompt: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            steps: None,
            guidance: None,
            width: None,
            height: None,
            max_area: 1024 * 1024,
            auto_resize: true,
            max_sequence_length: None,
            seed: 42,
            true_cfg_scale: 1.0,
            negative_prompt: None,
        }
    }
}

/// A generated image: tightly packed RGB8.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl Image {
    pub fn into_rgb(self) -> RgbImage {
        RgbImage {
            width: self.width,
            height: self.height,
            data: self.rgb,
        }
    }
}

#[cfg(feature = "oxideav")]
impl Image {
    /// The image as an oxideav picture.
    pub fn to_picture(&self) -> Result<crate::av::Picture> {
        crate::av::Picture::from_rgb(self.width, self.height, self.rgb.clone())
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
    /// Tokens per transformer call (text + image + reference).
    pub tokens: usize,
}

/// Progress callback: `(step, steps)`; return false to cancel.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

/// Stops a running generation from another thread.
#[derive(Clone, Debug)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

pub struct Pipeline {
    pub transformer: transformer::FluxTransformer,
    pub t5: t5::T5Encoder,
    pub clip_tokenizer: crate::sd::clip::ClipTokenizer,
    pub clip: crate::sd::clip::ClipText,
    pub vae: vae::Vae,
    pub scheduler: FlowMatchEuler,
    /// The base folder's pipeline class (`FluxKontextPipeline`, `FluxPipeline`, ...).
    pub class: String,
    pub load_secs: f64,
    cancel: Arc<AtomicBool>,
}

/// True for a diffusers FLUX pipeline folder.
pub fn is_flux_dir(dir: &Path) -> bool {
    let p = dir.join("model_index.json");
    crate::fsx::read(&p)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["_class_name"].as_str().is_some_and(|c| c.starts_with("Flux")))
}

/// True for a GGUF file holding a FLUX transformer.
pub fn is_flux_gguf(path: &Path) -> bool {
    crate::gguf::is_gguf(path) && crate::gguf::header(path).is_ok_and(|h| h.arch() == "flux")
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

fn resolve_transformer(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    crate::engine::resolve_model(spec, cache_dir)
}

/// Dumps for reference checks: `OXINFER_FLUX_DUMP=DIR` writes intermediate tensors (f32 LE) and
/// reads `latents_in.bin` (the initial noise, `[16, h, w]`) and `*_in.bin` overrides from it.
fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("OXINFER_FLUX_DUMP").map(PathBuf::from)
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

/// Python's `round()` (half to even).
fn py_round(x: f64) -> usize {
    x.round_ties_even().max(0.0) as usize
}

/// The output size diffusers' Kontext pipeline uses for a requested `width` x `height`: rescaled
/// to `max_area` pixels at that aspect ratio, rounded down to multiples of 16.
pub fn kontext_size(width: usize, height: usize, max_area: usize) -> (usize, usize) {
    let ar = width as f64 / height as f64;
    let w = py_round((max_area as f64 * ar).sqrt());
    let h = py_round((max_area as f64 / ar).sqrt());
    (w / 16 * 16, h / 16 * 16)
}

/// The size diffusers' Kontext pipeline resizes a `width` x `height` reference image to.
pub fn kontext_reference_size(width: usize, height: usize, auto_resize: bool) -> (usize, usize) {
    // get_default_height_width: down to multiples of the processor's factor (16).
    let (w, h) = (width - width % 16, height - height % 16);
    let (mut w, mut h) = (w, h);
    if auto_resize {
        let ar = w as f64 / h as f64;
        let mut best = (f64::INFINITY, 0, 0);
        for &(pw, ph) in &PREFERRED_KONTEXT_RESOLUTIONS {
            let c = ((ar - pw as f64 / ph as f64).abs(), pw, ph);
            if c.0 < best.0 || (c.0 == best.0 && (c.1, c.2) < (best.1, best.2)) {
                best = c;
            }
        }
        (w, h) = (best.1, best.2);
    }
    (w / 16 * 16, h / 16 * 16)
}

impl Pipeline {
    /// Loads a pipeline: `model` is a diffusers FLUX folder or repo id, or a FLUX GGUF
    /// transformer (file, or `org/name:QUANT`) whose other components come from `opts.base`.
    pub fn load(model: &str, opts: &LoadOptions) -> Result<Self> {
        let t0 = Instant::now();
        let cache = opts.cache_dir.as_deref();
        let model_path = Path::new(model);
        let gguf_model = model.contains(':') && !model_path.exists() || is_flux_gguf(model_path);
        let (base, tr_spec) = if gguf_model {
            (
                opts.base.clone().unwrap_or_else(|| DEFAULT_BASE.to_string()),
                Some(model.to_string()),
            )
        } else {
            (model.to_string(), opts.transformer.clone())
        };
        let mut skip: Vec<&str> = Vec::new();
        if tr_spec.is_some() {
            skip.push("transformer");
        }
        if opts.text_encoder.is_some() {
            skip.push("text_encoder_2");
        }
        let dir = resolve_dir(&base, &skip, cache)?;
        let index: serde_json::Value = serde_json::from_slice(
            &crate::fsx::read(&dir.join("model_index.json"))
                .with_context(|| format!("{}: not a diffusers folder (no model_index.json)", dir.display()))?,
        )?;
        let class = index["_class_name"].as_str().unwrap_or("").to_string();
        if !class.starts_with("Flux") {
            bail!("{}: {class} is not a FLUX pipeline", dir.display());
        }
        crate::gpu::ensure(opts.device)?;
        let tr_path = match &tr_spec {
            Some(s) => resolve_transformer(s, cache)?,
            None => dir.join("transformer"),
        };
        let transformer =
            transformer::FluxTransformer::load(&tr_path, opts.quant).with_context(|| format!("transformer {}", tr_path.display()))?;
        let t5_path = match &opts.text_encoder {
            Some(s) => resolve_transformer(s, cache)?,
            None => dir.join("text_encoder_2"),
        };
        let t5 = t5::T5Encoder::load(&t5_path, &dir.join("tokenizer_2"), opts.text_quant.unwrap_or(opts.quant))
            .with_context(|| format!("T5 encoder {}", t5_path.display()))?;
        let te = dir.join("text_encoder");
        let te_w = ["model.safetensors", "model.fp16.safetensors"]
            .iter()
            .map(|n| te.join(n))
            .find(|p| crate::fsx::exists(p))
            .context("no text_encoder/model.safetensors")?;
        let clip = crate::sd::clip::ClipText::load(&te, crate::sd::ops::Tensors::open(&te_w)?)?;
        let clip_tokenizer = crate::sd::clip::ClipTokenizer::load(&dir.join("tokenizer"))?;
        let vae = vae::Vae::load(&dir.join("vae"), opts.vae_encoder)?;
        let mut scheduler = FlowMatchEuler::load(&dir.join("scheduler").join("scheduler_config.json"))?;
        if tr_spec.is_some() && !transformer.cfg.guidance_embeds && scheduler.use_dynamic_shifting {
            // A schnell transformer over a dev / Kontext base: schnell's scheduler (no shift).
            scheduler = FlowMatchEuler::default();
        }
        Ok(Self {
            transformer,
            t5,
            clip_tokenizer,
            clip,
            vae,
            scheduler,
            class,
            load_secs: t0.elapsed().as_secs_f64(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancel.clone())
    }

    /// True when the transformer has the distilled guidance embedding (dev, Kontext).
    pub fn has_guidance(&self) -> bool {
        self.transformer.cfg.guidance_embeds
    }

    fn defaults(&self, o: &Options) -> (usize, Option<f32>, usize) {
        let g = self.has_guidance();
        (
            o.steps.unwrap_or(if g { 28 } else { 4 }).max(1),
            g.then(|| o.guidance.unwrap_or(3.5)),
            o.max_sequence_length.unwrap_or(if g { 512 } else { 256 }).clamp(1, 512),
        )
    }

    /// The prompt's T5 states (`[len, 4096]`) and CLIP pooled embedding (`[768]`).
    pub fn encode_prompt(&self, prompt: &str, len: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let ids = self.clip_tokenizer.encode(prompt);
        let hs = self.clip.forward(&ids);
        // CLIPTextModel's pooler output: the final state at the first end-of-text token.
        let eos = ids.iter().position(|&i| i == self.clip_tokenizer.eos).unwrap_or(ids.len() - 1);
        let h = self.clip.hidden;
        let pooled = hs[eos * h..(eos + 1) * h].to_vec();
        let t5 = self.t5.encode(prompt, len)?;
        Ok((t5, pooled))
    }

    /// Text to image (diffusers `FluxPipeline`).
    pub fn generate(&self, prompt: &str, o: &Options, progress: Progress) -> Result<(Image, Timings)> {
        let (w, h) = (o.width.unwrap_or(1024), o.height.unwrap_or(1024));
        if w < 16 || h < 16 {
            bail!("image size must be at least 16x16 (got {w}x{h})");
        }
        self.run(prompt, None, w / 16 * 16, h / 16 * 16, o, progress)
    }

    /// Instruction-based editing (diffusers `FluxKontextPipeline`): `image` is the reference,
    /// `prompt` the instruction.
    pub fn edit(&self, image: &RgbImage, prompt: &str, o: &Options, progress: Progress) -> Result<(Image, Timings)> {
        let (w, h) = kontext_size(o.width.unwrap_or(1024), o.height.unwrap_or(1024), o.max_area);
        if w < 16 || h < 16 {
            bail!("output size {w}x{h} is too small");
        }
        if image.width < 16 || image.height < 16 || image.data.len() != image.width * image.height * 3 {
            bail!("the reference image must be RGB8 and at least 16x16");
        }
        self.run(prompt, Some(image), w, h, o, progress)
    }

    #[cfg(feature = "oxideav")]
    /// [`Pipeline::edit`] with an oxideav picture.
    pub fn edit_picture(&self, image: &crate::av::Picture, prompt: &str, o: &Options, progress: Progress) -> Result<(Image, Timings)> {
        self.edit(&image.to_rgb()?, prompt, o, progress)
    }

    fn run(
        &self,
        prompt: &str,
        reference: Option<&RgbImage>,
        width: usize,
        height: usize,
        o: &Options,
        progress: Progress,
    ) -> Result<(Image, Timings)> {
        self.cancel.store(false, Ordering::Relaxed);
        let mut tm = Timings::default();
        let (steps, guidance, seq) = self.defaults(o);
        let c = self.vae.latent_channels;
        let cfg = &self.transformer.cfg;
        if cfg.in_channels != 4 * c {
            bail!("transformer takes {} channels, the VAE has {c} (x4 packed)", cfg.in_channels);
        }
        let t0 = Instant::now();
        let (txt, pooled) = match (
            read_override("t5_in.bin", seq * cfg.joint_dim)?,
            read_override("pooled_in.bin", cfg.pooled_dim)?,
        ) {
            (Some(t), Some(p)) => (t, p),
            _ => self.encode_prompt(prompt, seq)?,
        };
        dump("t5.bin", &txt);
        dump("pooled.bin", &pooled);
        let neg = match (&o.negative_prompt, o.true_cfg_scale > 1.0) {
            (Some(n), true) => Some(self.encode_prompt(n, seq)?),
            _ => None,
        };
        tm.text = t0.elapsed().as_secs_f64();

        // Reference tokens (Kontext).
        let t1 = Instant::now();
        let (ref_tok, ref_grid) = match reference {
            None => (Vec::new(), None),
            Some(img) => {
                let (rw, rh) = kontext_reference_size(img.width, img.height, o.auto_resize);
                let px = if (rw, rh) == (img.width, img.height) {
                    img.data.clone()
                } else {
                    crate::resample::resize_lanczos_rgb8(&img.data, img.width, img.height, rw, rh)
                };
                let x = vae::rgb8_to_unit(&px);
                let (lh, lw) = (rh / 8, rw / 8);
                let z = match read_override("ref_latents_in.bin", c * lh * lw)? {
                    Some(v) => vae::chw_to_hwc(&v, c, lh, lw),
                    None => self.vae.encode(&x, rh, rw)?.0,
                };
                dump("ref_latents.bin", &vae::hwc_to_chw(&z, c, lh, lw));
                (vae::pack(&z, lh, lw, c), Some((lh / 2, lw / 2)))
            }
        };
        tm.encode = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let (lh, lw) = (2 * (height / 16), 2 * (width / 16));
        let n_noise = (lh / 2) * (lw / 2);
        let noise = match read_override("latents_in.bin", c * lh * lw)? {
            Some(v) => v,
            None => crate::sd::ops::Rng::new(o.seed).normals(c * lh * lw),
        };
        let mut x = vae::pack(&vae::chw_to_hwc(&noise, c, lh, lw), lh, lw, c);
        let extra: Vec<(f32, usize, usize)> = ref_grid.map(|(gh, gw)| (1.0, gh, gw)).into_iter().collect();
        let ids = Rope::flux_ids(seq, lh / 2, lw / 2, &extra);
        let rope = Rope::new(&ids, &cfg.axes, cfg.theta);
        let n_img = n_noise + ref_tok.len() / cfg.in_channels;
        tm.tokens = seq + n_img;
        let plan = self.scheduler.flux_plan(steps, n_noise)?;
        let mut input = vec![0f32; n_img * cfg.in_channels];
        input[n_noise * cfg.in_channels..].copy_from_slice(&ref_tok);
        for i in 0..steps {
            if self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            input[..n_noise * cfg.in_channels].copy_from_slice(&x);
            let t = plan.timesteps[i];
            let mut v = self.transformer.forward(&input, n_img, &txt, seq, &pooled, t, guidance, &rope);
            tm.transformer_calls += 1;
            v.truncate(n_noise * cfg.out_channels);
            if let Some((nt, np)) = &neg {
                let mut u = self.transformer.forward(&input, n_img, nt, seq, np, t, guidance, &rope);
                tm.transformer_calls += 1;
                u.truncate(n_noise * cfg.out_channels);
                for (a, &b) in v.iter_mut().zip(&u) {
                    *a = b + o.true_cfg_scale * (*a - b);
                }
            }
            if i == 0 {
                dump("noise_pred0.bin", &v);
            }
            FlowMatchEuler::step(&plan, i, &mut x, &v);
            if !progress(i + 1, steps) {
                bail!("cancelled");
            }
        }
        tm.denoise = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        let z = vae::unpack(&x, lh, lw, c);
        dump("latents_out.bin", &vae::hwc_to_chw(&z, c, lh, lw));
        let (px, h, w) = self.vae.decode(&z, lh, lw);
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
    fn kontext_sizes() {
        // diffusers: 1024x1024 request -> 1024x1024; 800x600 -> sqrt scaling then //16.
        assert_eq!(kontext_size(1024, 1024, 1024 * 1024), (1024, 1024));
        assert_eq!(kontext_size(800, 600, 1024 * 1024), (1168, 880));
        assert_eq!(kontext_size(512, 512, 512 * 512), (512, 512));
        // A 4:3 photo goes to 1184x880, a portrait 3:4 to 880x1184, square to 1024x1024.
        assert_eq!(kontext_reference_size(1600, 1200, true), (1184, 880));
        assert_eq!(kontext_reference_size(600, 800, true), (880, 1184));
        assert_eq!(kontext_reference_size(500, 500, true), (1024, 1024));
        assert_eq!(kontext_reference_size(500, 500, false), (496, 496));
    }
}
