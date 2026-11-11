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
//! Several references are appended one after the other, each at its own position
//! ([`RefLayout`]).
//!
//! [`Pipeline::render`] runs every mode from one set of [`Inputs`]: image to image and inpainting
//! (diffusers' `FluxImg2ImgPipeline` / `FluxInpaintPipeline`: `strength`, mask blending in latent
//! space), FLUX.1 Fill (384 input channels: noise, masked-image latents and the mask), FLUX.1
//! Canny / Depth (128: noise and the control image's latents; also as BFL's control LoRAs on
//! FLUX.1 [dev]) and FLUX.1 Redux image prompts ([`redux`]). LoRA files are applied at load
//! ([`lora`]); the transformer also loads from BFL's single-file safetensors and the VAE from
//! `ae.safetensors`.

pub mod lora;
pub mod redux;
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
    /// LoRA files applied to the transformer (see [`lora`]).
    pub loras: Vec<lora::LoraSpec>,
    /// An autoencoder to use instead of the base folder's `vae/`: BFL's `ae.safetensors` or a
    /// diffusers `vae` folder.
    pub vae: Option<String>,
    /// A FLUX.1 Redux folder or repo (`image_encoder/` SigLIP tower + `image_embedder/`), for
    /// image prompts.
    pub redux: Option<String>,
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
            loras: Vec::new(),
            vae: None,
            redux: None,
        }
    }
}

impl LoadOptions {
    /// Sets one `key=value` option (C API): quant, text_quant, transformer, text_encoder, base,
    /// device, cache_dir, vae_encoder, lora (`FILE[:SCALE]`, repeatable), vae, redux.
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
            "lora" => self.loras.push(lora::LoraSpec::parse(v)?),
            "vae" => self.vae = Some(v.to_string()),
            "redux" => self.redux = Some(v.to_string()),
            _ => bail!("unknown FLUX option {k}"),
        }
        Ok(())
    }
}

/// Where the tokens of Kontext reference images sit (their RoPE ids `[index, row, column]`; the
/// output's are `[0, y, x]`). One reference is at `[1, y, x]` in every layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RefLayout {
    /// ComfyUI's default (`ref_latents_method = "offset"`): index 1 for all, each reference
    /// placed beside or below the ones before it, whichever keeps the canvas more square.
    #[default]
    Offset,
    /// ComfyUI's `index` method: reference `i` at index `i + 1`, all at the origin.
    Index,
    /// diffusers' community `FluxKontextPipeline` for multiple images: every reference resized
    /// to the first one's size, index 1, reference `i` shifted along the columns by `i` times
    /// the grid height.
    Diffusers,
}

impl RefLayout {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "offset" => Self::Offset,
            "index" => Self::Index,
            "diffusers" => Self::Diffusers,
            _ => return None,
        })
    }
}

/// The position of each reference grid (`(rows, columns)` in tokens): `(index, row0, column0)`.
pub fn reference_positions(layout: RefLayout, grids: &[(usize, usize)]) -> Vec<(f32, usize, usize)> {
    let (mut h, mut w) = (0, 0);
    grids
        .iter()
        .enumerate()
        .map(|(i, &(gh, gw))| match layout {
            RefLayout::Index => ((i + 1) as f32, 0, 0),
            RefLayout::Diffusers => (1.0, 0, i * gh),
            RefLayout::Offset => {
                let (mut y0, mut x0) = (0, 0);
                if gh + h > gw + w {
                    x0 = w;
                } else {
                    y0 = h;
                }
                h = h.max(gh + y0);
                w = w.max(gw + x0);
                (1.0, y0, x0)
            }
        })
        .collect()
}

/// The images a run reads besides the prompt. With none it is text to image.
#[derive(Clone, Copy, Default)]
pub struct Inputs<'a> {
    /// FLUX.1 Kontext: the reference images the instruction edits or combines.
    pub references: &'a [RgbImage],
    /// Image to image, inpainting, FLUX.1 Fill: the starting image.
    pub init: Option<&'a RgbImage>,
    /// Inpainting / Fill: white (>= 50% luma) is repainted, black kept; any size (resized to the
    /// output like the image).
    pub mask: Option<&'a RgbImage>,
    /// FLUX.1 Canny / Depth (or [dev] with a control LoRA): the edge or depth map.
    pub control: Option<&'a RgbImage>,
    /// FLUX.1 Redux: image prompts (the pipeline must be loaded with `LoadOptions::redux`).
    pub image_prompts: &'a [RgbImage],
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
    /// Image to image / inpainting: how much of the schedule is run from the noised input
    /// (0..1; default 0.6 as diffusers, 1 for FLUX.1 Fill). Given to [`Pipeline::edit`], it makes
    /// the edit an image-to-image run instead of a Kontext one.
    pub strength: Option<f32>,
    /// Kontext with several references: where their tokens sit.
    pub ref_layout: RefLayout,
    /// Redux: the weight of each image prompt's tokens (default 1).
    pub image_prompt_scale: f32,
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
            strength: None,
            ref_layout: RefLayout::default(),
            image_prompt_scale: 1.0,
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
    /// One line per LoRA file applied.
    pub lora_report: Vec<String>,
    /// The Redux image-prompt encoder, when loaded.
    pub redux: Option<redux::Redux>,
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

/// True for a single safetensors file holding a FLUX transformer (BFL's `flux1-dev.safetensors`
/// layout, alone or inside a ComfyUI checkpoint, or diffusers names).
pub fn is_flux_safetensors(path: &Path) -> bool {
    if !path.is_file() || path.extension().is_none_or(|e| e != "safetensors") {
        return false;
    }
    let Ok(st) = crate::safetensors::SafeTensors::open(path, false) else {
        return false;
    };
    let has = |n: &str| st.info(n).is_some();
    has("double_blocks.0.img_attn.qkv.weight")
        || has("model.diffusion_model.double_blocks.0.img_attn.qkv.weight")
        || (has("x_embedder.weight") && has("single_transformer_blocks.0.norm.linear.weight"))
}

fn resolve_transformer(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    if spec.ends_with(".safetensors") {
        // `org/name/path/in/repo.safetensors`.
        return crate::engine::resolve_file(spec, cache_dir);
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
        let gguf_model = (model.contains(':') || model.ends_with(".safetensors")) && !model_path.exists()
            || is_flux_gguf(model_path)
            || is_flux_safetensors(model_path);
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
        let mut transformer =
            transformer::FluxTransformer::load(&tr_path, opts.quant).with_context(|| format!("transformer {}", tr_path.display()))?;
        // LoRA files: local paths, or `org/name/file.safetensors` (with the `hf` feature).
        let loras = opts
            .loras
            .iter()
            .map(|l| {
                Ok(lora::LoraSpec {
                    path: if l.path.is_file() {
                        l.path.clone()
                    } else {
                        crate::engine::resolve_file(&l.path.to_string_lossy(), cache)?
                    },
                    scale: l.scale,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let lora_report = lora::apply(&mut transformer, &loras)?;
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
        let vae = match &opts.vae {
            None => vae::Vae::load(&dir.join("vae"), opts.vae_encoder)?,
            Some(s) => {
                let p = resolve_transformer(s, cache)?;
                if p.is_file() {
                    vae::Vae::load_bfl(&p, opts.vae_encoder)
                } else {
                    vae::Vae::load(&p, opts.vae_encoder)
                }
                .with_context(|| format!("VAE {}", p.display()))?
            }
        };
        let redux = match &opts.redux {
            None => None,
            Some(s) => {
                let p = resolve_dir(s, &[], cache)?;
                Some(redux::Redux::load(&p).with_context(|| format!("Redux {}", p.display()))?)
            }
        };
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
            lora_report,
            redux,
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
        self.render(prompt, &Inputs::default(), o, progress)
    }

    /// Instruction-based editing (diffusers `FluxKontextPipeline`): `image` is the reference,
    /// `prompt` the instruction. With `o.strength` set it is an image-to-image run instead
    /// (`FluxImg2ImgPipeline`), for the models that are not Kontext.
    pub fn edit(&self, image: &RgbImage, prompt: &str, o: &Options, progress: Progress) -> Result<(Image, Timings)> {
        let inputs = if o.strength.is_some() {
            Inputs {
                init: Some(image),
                ..Default::default()
            }
        } else {
            Inputs {
                references: std::slice::from_ref(image),
                ..Default::default()
            }
        };
        self.render(prompt, &inputs, o, progress)
    }

    #[cfg(feature = "oxideav")]
    /// [`Pipeline::edit`] with an oxideav picture.
    pub fn edit_picture(&self, image: &crate::av::Picture, prompt: &str, o: &Options, progress: Progress) -> Result<(Image, Timings)> {
        self.edit(&image.to_rgb()?, prompt, o, progress)
    }

    /// `img` resized to `w` x `h` (Pillow's Lanczos, as diffusers' `VaeImageProcessor`).
    fn resized(img: &RgbImage, w: usize, h: usize) -> Vec<u8> {
        if (w, h) == (img.width, img.height) {
            img.data.clone()
        } else {
            crate::resample::resize_lanczos_rgb8(&img.data, img.width, img.height, w, h)
        }
    }

    /// Runs the pipeline on `inputs` (see [`Inputs`]): what they hold and the transformer's
    /// input width choose the mode.
    ///
    /// - nothing: text to image;
    /// - `references`: Kontext editing (the output is `width` x `height` rescaled to `max_area`);
    /// - `init`: image to image from the image noised to `strength`; with `mask`, inpainting
    ///   (after every step the kept region is reset to the image at that step's noise level), or
    ///   FLUX.1 Fill when the transformer takes 384 channels (the masked image's latents and the
    ///   mask ride along every token; `strength` defaults to 1);
    /// - `control`: FLUX.1 Canny / Depth (the transformer takes 128 channels: the control
    ///   image's latents ride along);
    /// - `image_prompts`: Redux tokens appended to the text tokens.
    ///
    /// The output size defaults to the init / control image's (multiples of 16), else 1024^2.
    pub fn render(&self, prompt: &str, inp: &Inputs, o: &Options, progress: Progress) -> Result<(Image, Timings)> {
        self.cancel.store(false, Ordering::Relaxed);
        let mut tm = Timings::default();
        let (steps, guidance, seq) = self.defaults(o);
        let c = self.vae.latent_channels;
        let cfg = &self.transformer.cfg;
        let tok = 4 * c;
        let f = self.vae.factor;
        // What rides along the noise tokens: nothing, a control latent, or Fill's masked latent + mask.
        let fill_w = tok + f * f * 4;
        #[derive(PartialEq)]
        enum Extra {
            None,
            Control,
            Fill,
        }
        let extra = match cfg.in_channels {
            n if n == tok => Extra::None,
            n if n == 2 * tok => Extra::Control,
            n if n == tok + fill_w => Extra::Fill,
            n => bail!("transformer takes {n} channels, the VAE has {c} (x4 packed)"),
        };
        if cfg.out_channels != tok {
            bail!("transformer predicts {} channels, the VAE has {c} (x4 packed)", cfg.out_channels);
        }
        for img in inp
            .references
            .iter()
            .chain(inp.init)
            .chain(inp.mask)
            .chain(inp.control)
            .chain(inp.image_prompts)
        {
            if img.width < 16 || img.height < 16 || img.data.len() != img.width * img.height * 3 {
                bail!("input images must be RGB8 and at least 16x16");
            }
        }
        match extra {
            Extra::Control if inp.control.is_none() => {
                bail!("this transformer takes a control image (FLUX.1 Canny / Depth, or a control LoRA): pass one")
            }
            Extra::Fill if inp.init.is_none() || inp.mask.is_none() => {
                bail!("this transformer is FLUX.1 Fill: pass the image and a mask")
            }
            Extra::None | Extra::Fill if inp.control.is_some() => {
                bail!("a control image needs FLUX.1 Canny / Depth (or FLUX.1 [dev] with a control LoRA)")
            }
            _ => {}
        }
        if inp.mask.is_some() && inp.init.is_none() {
            bail!("a mask needs the image to inpaint");
        }
        if !inp.references.is_empty() && (extra != Extra::None || inp.init.is_some()) {
            bail!("Kontext reference images cannot be combined with image-to-image, inpainting or control inputs");
        }
        if !inp.image_prompts.is_empty() && self.redux.is_none() {
            bail!("image prompts need the Redux encoder (LoadOptions::redux / --redux)");
        }
        let (width, height) = if !inp.references.is_empty() {
            kontext_size(o.width.unwrap_or(1024), o.height.unwrap_or(1024), o.max_area)
        } else {
            let (dw, dh) = inp.init.or(inp.control).map_or((1024, 1024), |i| (i.width, i.height));
            (o.width.unwrap_or(dw) / 16 * 16, o.height.unwrap_or(dh) / 16 * 16)
        };
        if width < 16 || height < 16 {
            bail!("image size must be at least 16x16 (got {width}x{height})");
        }

        let t0 = Instant::now();
        let (mut txt, pooled) = match (
            read_override("t5_in.bin", seq * cfg.joint_dim)?,
            read_override("pooled_in.bin", cfg.pooled_dim)?,
        ) {
            (Some(t), Some(p)) => (t, p),
            _ => self.encode_prompt(prompt, seq)?,
        };
        dump("t5.bin", &txt);
        dump("pooled.bin", &pooled);
        let mut neg = match (&o.negative_prompt, o.true_cfg_scale > 1.0) {
            (Some(n), true) => Some(self.encode_prompt(n, seq)?),
            _ => None,
        };
        // Redux: every image prompt's tokens after the text tokens.
        let mut n_txt = seq;
        if let Some(rx) = self.redux.as_ref().filter(|_| !inp.image_prompts.is_empty()) {
            if rx.out_dim != cfg.joint_dim {
                bail!("Redux tokens are {} wide, the transformer reads {}", rx.out_dim, cfg.joint_dim);
            }
            for img in inp.image_prompts {
                let mut e = rx.encode(img);
                dump("redux.bin", &e);
                e.iter_mut().for_each(|v| *v *= o.image_prompt_scale);
                n_txt += e.len() / cfg.joint_dim;
                if let Some((nt, _)) = &mut neg {
                    nt.extend_from_slice(&e);
                }
                txt.extend_from_slice(&e);
            }
        }
        tm.text = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let (lh, lw) = (2 * (height / 16), 2 * (width / 16));
        let n_noise = (lh / 2) * (lw / 2);
        // Kontext reference tokens, one grid per image.
        let mut ref_tok: Vec<f32> = Vec::new();
        let mut grids: Vec<(usize, usize)> = Vec::new();
        for (i, img) in inp.references.iter().enumerate() {
            // diffusers' multi-image pipeline gives every reference the first one's size.
            let sized = if o.ref_layout == RefLayout::Diffusers {
                &inp.references[0]
            } else {
                img
            };
            let (rw, rh) = kontext_reference_size(sized.width, sized.height, o.auto_resize);
            let x = vae::rgb8_to_unit(&Self::resized(img, rw, rh));
            let (rlh, rlw) = (rh / 8, rw / 8);
            let over = if inp.references.len() == 1 {
                read_override("ref_latents_in.bin", c * rlh * rlw)?
            } else {
                None
            };
            let z = match over {
                Some(v) => vae::chw_to_hwc(&v, c, rlh, rlw),
                None => self.vae.encode(&x, rh, rw)?.0,
            };
            if i == 0 {
                dump("ref_latents.bin", &vae::hwc_to_chw(&z, c, rlh, rlw));
            }
            ref_tok.extend(vae::pack(&z, rlh, rlw, c));
            grids.push((rlh / 2, rlw / 2));
        }
        // The starting image, the mask and what rides along every noise token.
        let init_tok = match inp.init {
            None => None,
            Some(img) => {
                let x = vae::rgb8_to_unit(&Self::resized(img, width, height));
                let z = self.vae.encode(&x, height, width)?.0;
                dump("init_latents.bin", &vae::hwc_to_chw(&z, c, lh, lw));
                Some((vae::pack(&z, lh, lw, c), x))
            }
        };
        let mask = inp.mask.map(|m| binary_mask(m, width, height));
        // Inpainting without a Fill transformer: the mask at latent resolution (nearest pixel,
        // as `F.interpolate`), one value per latent pixel and channel.
        let mut blend_mask: Option<Vec<f32>> = None;
        let mut cond: Vec<f32> = Vec::new();
        match (&extra, &mask) {
            (Extra::Fill, Some(m)) => {
                let (_, x) = init_tok.as_ref().unwrap();
                // The image with the repainted pixels at mid grey (`image * (1 - mask)`).
                let masked: Vec<f32> = x.iter().enumerate().map(|(i, &v)| v * (1.0 - m[i / 3])).collect();
                let z = self.vae.encode(&masked, height, width)?.0;
                dump("masked_latents.bin", &vae::hwc_to_chw(&z, c, lh, lw));
                // The mask at full resolution: the f x f pixels of each latent pixel as channels.
                let mut mc = vec![0f32; lh * lw * f * f];
                for_chunks_mask(&mut mc, f * f, |p, row| {
                    let (y, x) = (p / lw, p % lw);
                    for py in 0..f {
                        for px in 0..f {
                            row[py * f + px] = m[(y * f + py) * width + x * f + px];
                        }
                    }
                });
                let (zt, mt) = (vae::pack(&z, lh, lw, c), vae::pack(&mc, lh, lw, f * f));
                cond = vec![0f32; n_noise * fill_w];
                for t in 0..n_noise {
                    cond[t * fill_w..t * fill_w + tok].copy_from_slice(&zt[t * tok..(t + 1) * tok]);
                    cond[t * fill_w + tok..(t + 1) * fill_w].copy_from_slice(&mt[t * f * f * 4..(t + 1) * f * f * 4]);
                }
            }
            (_, Some(m)) => {
                let mut ml = vec![0f32; lh * lw * c];
                for_chunks_mask(&mut ml, c, |p, row| row.fill(m[(p / lw) * f * width + (p % lw) * f]));
                blend_mask = Some(vae::pack(&ml, lh, lw, c));
            }
            _ => {}
        }
        if let (Extra::Control, Some(img)) = (&extra, inp.control) {
            let x = vae::rgb8_to_unit(&Self::resized(img, width, height));
            let z = self.vae.encode(&x, height, width)?.0;
            dump("control_latents.bin", &vae::hwc_to_chw(&z, c, lh, lw));
            cond = vae::pack(&z, lh, lw, c);
        }
        tm.encode = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let noise = match read_override("latents_in.bin", c * lh * lw)? {
            Some(v) => v,
            None => crate::sd::ops::Rng::new(o.seed).normals(c * lh * lw),
        };
        let noise = vae::pack(&vae::chw_to_hwc(&noise, c, lh, lw), lh, lw, c);
        let pos = reference_positions(o.ref_layout, &grids);
        let extra_grids: Vec<rope::Grid> = grids
            .iter()
            .zip(&pos)
            .map(|(&(h, w), &(index, y0, x0))| rope::Grid { index, h, w, y0, x0 })
            .collect();
        let ids = Rope::flux_ids_at(n_txt, lh / 2, lw / 2, &extra_grids);
        let rope = Rope::new(&ids, &cfg.axes, cfg.theta);
        let ci = cfg.in_channels;
        let n_img = n_noise + ref_tok.len() / tok;
        tm.tokens = n_txt + n_img;
        let plan = self.scheduler.flux_plan(steps, n_noise)?;
        // Image to image: skip the first part of the schedule and start from the noised image.
        let first = match &init_tok {
            None => 0,
            Some(_) => start_step(steps, o.strength.unwrap_or(if extra == Extra::Fill { 1.0 } else { 0.6 })),
        };
        if first >= steps {
            bail!("strength {} leaves no denoising step out of {steps}", o.strength.unwrap_or(0.6));
        }
        let mut x = match &init_tok {
            Some((z, _)) => FlowMatchEuler::scale_noise(z, &noise, plan.sigmas[first]),
            None => noise.clone(),
        };
        let mut input = vec![0f32; n_img * ci];
        for t in 0..n_noise {
            input[t * ci + tok..(t + 1) * ci].copy_from_slice(&cond[t * (ci - tok)..(t + 1) * (ci - tok)]);
        }
        input[n_noise * ci..].copy_from_slice(&ref_tok);
        let run = steps - first;
        for i in first..steps {
            if self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            for t in 0..n_noise {
                input[t * ci..t * ci + tok].copy_from_slice(&x[t * tok..(t + 1) * tok]);
            }
            let t = plan.timesteps[i];
            let mut v = self.transformer.forward(&input, n_img, &txt, n_txt, &pooled, t, guidance, &rope);
            tm.transformer_calls += 1;
            v.truncate(n_noise * tok);
            if let Some((nt, np)) = &neg {
                let mut u = self.transformer.forward(&input, n_img, nt, n_txt, np, t, guidance, &rope);
                tm.transformer_calls += 1;
                u.truncate(n_noise * tok);
                for (a, &b) in v.iter_mut().zip(&u) {
                    *a = b + o.true_cfg_scale * (*a - b);
                }
            }
            if i == first {
                dump("noise_pred0.bin", &v);
            }
            FlowMatchEuler::step(&plan, i, &mut x, &v);
            if let (Some(m), Some((z, _))) = (&blend_mask, &init_tok) {
                // The kept region follows the image itself, at the noise level just reached.
                let s = if i + 1 < steps { plan.sigmas[i + 1] } else { 0.0 };
                for (j, a) in x.iter_mut().enumerate() {
                    let keep = s * noise[j] + (1.0 - s) * z[j];
                    *a = (1.0 - m[j]) * keep + m[j] * *a;
                }
            }
            if !progress(i + 1 - first, run) {
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

/// `f(i, chunk)` over the `size`-wide chunks of `x`.
fn for_chunks_mask(x: &mut [f32], size: usize, f: impl Fn(usize, &mut [f32])) {
    for (i, r) in x.chunks_exact_mut(size).enumerate() {
        f(i, r);
    }
}

/// The step an image-to-image run of `steps` steps starts at for `strength` (diffusers'
/// `get_timesteps`: `int(max(steps - min(steps * strength, steps), 0))`).
pub fn start_step(steps: usize, strength: f32) -> usize {
    // The decimal the user wrote, as Python's float would hold it.
    let s: f64 = format!("{strength}").parse().unwrap_or(strength as f64);
    let init = (steps as f64 * s.clamp(0.0, 1.0)).min(steps as f64);
    (steps as f64 - init).max(0.0) as usize
}

/// A mask image as diffusers' mask processor reads it: resized to `w` x `h` (Lanczos), luma
/// (Pillow's `L`), 1 where it is at least half white, else 0. `[h * w]`.
pub fn binary_mask(mask: &RgbImage, w: usize, h: usize) -> Vec<f32> {
    let px = Pipeline::resized(mask, w, h);
    px.as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            let l = (p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 0x8000) >> 16;
            if l as f32 / 255.0 >= 0.5 { 1.0 } else { 0.0 }
        })
        .collect()
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

    #[test]
    fn reference_layouts() {
        // (rows, columns) in tokens: a landscape 44x74, a square 64x64, a portrait 74x44.
        let g = [(44, 74), (64, 64), (74, 44)];
        // ComfyUI `offset`: the first at the origin; the second below it (64 + 44 <= 64 + 74);
        // the third to the right (74 + 108 > 44 + 74).
        assert_eq!(
            reference_positions(RefLayout::Offset, &g),
            vec![(1.0, 0, 0), (1.0, 44, 0), (1.0, 0, 74)]
        );
        assert_eq!(
            reference_positions(RefLayout::Index, &g),
            vec![(1.0, 0, 0), (2.0, 0, 0), (3.0, 0, 0)]
        );
        // diffusers' community pipeline: column offset i * rows.
        assert_eq!(
            reference_positions(RefLayout::Diffusers, &[(64, 64); 3]),
            vec![(1.0, 0, 0), (1.0, 0, 64), (1.0, 0, 128)]
        );
        assert_eq!(reference_positions(RefLayout::Offset, &[(64, 64)]), vec![(1.0, 0, 0)]);
    }

    #[test]
    fn strength_steps() {
        // diffusers' get_timesteps: int(max(steps - min(steps * strength, steps), 0)).
        assert_eq!(start_step(28, 0.6), 11);
        assert_eq!(start_step(10, 0.3), 7);
        assert_eq!(start_step(4, 0.5), 2);
        assert_eq!(start_step(50, 1.0), 0);
        assert_eq!(start_step(8, 0.0), 8);
    }

    #[test]
    fn mask_luma() {
        let m = RgbImage {
            width: 2,
            height: 1,
            data: vec![127, 127, 127, 128, 128, 128],
        };
        assert_eq!(binary_mask(&m, 2, 1), vec![0.0, 1.0]);
    }
}
