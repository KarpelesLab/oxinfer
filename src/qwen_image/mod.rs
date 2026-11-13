//! Qwen-Image: instruction-based image editing (Qwen-Image-Edit, -Edit-2509, -Edit-2511) and
//! text to image (Qwen-Image, -2512) from a diffusers folder, following diffusers'
//! `QwenImageEditPlusPipeline` / `QwenImageEditPipeline` / `QwenImagePipeline`:
//!
//! - prompt encoding with Qwen2.5-VL-7B (`crate::qwen_vl`): the pipeline's chat template (a
//!   system turn, then for an edit `Picture N: <|vision_start|><image>...<|vision_end|>` per
//!   input image, then the instruction), the input images resized to ~384x384 for the vision
//!   tower (Edit-2509 / 2511; the original Edit uses the full image), the last hidden states
//!   minus the template's first 64 (34 for text to image) tokens;
//! - the 60-block MMDiT (`transformer`), the target's latent patches followed by the reference
//!   images' (VAE-encoded at ~1024x1024) as extra tokens;
//! - flow matching with Euler steps and the resolution-dependent shift (`sched`), true
//!   classifier-free guidance with a negative prompt and the norm-preserving rescale of the
//!   combined prediction;
//! - the Wan-style VAE (`vae`), latents normalized by the per-channel mean / std.
//!
//! Beyond those three pipelines:
//!
//! - image to image and inpainting (`QwenImageImg2ImgPipeline`, `QwenImageInpaintPipeline`,
//!   `QwenImageEditInpaintPipeline`): denoising starts from the input image noised to the level
//!   `strength` selects, and with a mask the kept region returns to the image (noised to the
//!   step's level) after every step;
//! - ControlNet (`QwenImageControlNetPipeline`, `QwenImageControlNetInpaintPipeline` with
//!   InstantX's Union and Inpainting models, `controlnet`): residuals added to the transformer's
//!   image stream, with a conditioning scale and a start / end window;
//! - Qwen-Image-Layered (`QwenImageLayeredPipeline`): an image decomposed into RGBA layers
//!   ([`Pipeline::decompose`]);
//! - FP8 "scaled" single-file transformers (`weights`).
//!
//! LoRA files (Lightning step distillation for 4 / 8 steps, style LoRAs) are merged into the
//! transformer at load; the transformer can come from a GGUF file (e.g. unsloth's
//! `Qwen-Image-Edit-2511-GGUF`) and stream from storage. Components load on first use, and the
//! 16 GB text encoder is dropped after encoding unless asked to stay.
//!
//! Qwen-Image-2.1 and -2.1-Turbo (`QwenImage21Pipeline`) are a different model behind the same
//! [`Pipeline`]: a Qwen3-VL-8B text encoder, a 7B single-stream transformer (`transformer21`)
//! and a 16x RGBA VAE; see `v21`.

pub mod caption;
pub mod controlnet;
pub mod sched;
pub mod transformer;
pub mod transformer21;
mod v21;
pub mod vae;
pub mod weights;

use crate::clock::Instant;
use crate::model::Quant;
use crate::qwen_vl::{QwenVl, QwenVlOptions, VlImage};
use crate::safetensors::ModelFiles;
pub use crate::sd::Image;
use crate::video::RgbImage;
use anyhow::{Context, Result, bail};
use sched::FlowMatch;
use std::path::{Path, PathBuf};
use transformer::{Extra, ImageShape, Transformer, TransformerConfig};
pub use weights::LoraSpec;

/// Which pipeline a folder holds (`model_index.json`'s `_class_name`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `QwenImagePipeline`: text to image.
    TextToImage,
    /// `QwenImageEditPipeline` (Qwen-Image-Edit): one input image.
    Edit,
    /// `QwenImageEditPlusPipeline` (Edit-2509 / 2511): one or more input images.
    EditPlus,
    /// `QwenImageLayeredPipeline` (Qwen-Image-Layered): an image into RGBA layers.
    Layered,
    /// `QwenImage21Pipeline` (Qwen-Image-2.1, -2.1-Turbo): text to image, and editing with one
    /// or more input images; RGBA.
    V21,
}

/// The default `strength` of image to image and inpainting with a text-to-image model
/// (diffusers' default).
pub const DEFAULT_STRENGTH: f32 = 0.6;

/// Whether `dir` is a Qwen-Image diffusers folder.
pub fn is_qwen_image(dir: &Path) -> bool {
    kind_of(dir).is_some()
}

fn kind_of(dir: &Path) -> Option<Kind> {
    let v: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("model_index.json")).ok()?).ok()?;
    match v["_class_name"].as_str()? {
        "QwenImagePipeline"
        | "QwenImageImg2ImgPipeline"
        | "QwenImageInpaintPipeline"
        | "QwenImageControlNetPipeline"
        | "QwenImageControlNetInpaintPipeline" => Some(Kind::TextToImage),
        "QwenImageEditPipeline" | "QwenImageEditInpaintPipeline" => Some(Kind::Edit),
        "QwenImageEditPlusPipeline" => Some(Kind::EditPlus),
        "QwenImageLayeredPipeline" => Some(Kind::Layered),
        "QwenImage21Pipeline" => Some(Kind::V21),
        _ => None,
    }
}

/// How to load a [`Pipeline`].
#[derive(Clone, Debug)]
pub struct Options {
    /// Precision of the transformer's matmul weights (a GGUF transformer keeps its own).
    pub quant: Quant,
    /// Precision of the Qwen2.5-VL text encoder's language model.
    pub text_quant: Quant,
    /// Map weights from disk instead of reading them into memory.
    pub mmap: bool,
    /// LoRA files merged into the transformer (e.g. a Lightning LoRA).
    pub loras: Vec<LoraSpec>,
    /// The transformer from elsewhere: a GGUF file, a single safetensors file (bf16, or an FP8
    /// "scaled" checkpoint) or a diffusers `transformer/` folder.
    pub transformer: Option<PathBuf>,
    /// A ControlNet (InstantX's Qwen-Image-ControlNet-Union or -Inpainting): a folder with
    /// `config.json` and `diffusion_pytorch_model.safetensors`, or a repo id.
    pub controlnet: Option<PathBuf>,
    /// Stream the transformer's blocks from storage (weights mapped, read ahead per block).
    pub stream: Option<crate::stream::StreamOptions>,
    /// Keep the text encoder loaded between calls (otherwise dropped after each encoding).
    pub keep_text_encoder: bool,
    /// Accelerator choice, when the process has not initialized one yet.
    pub device: crate::platform::DevicePref,
}

impl Options {
    /// Sets an option by name (the C API's `key=value` options): `quant` and `text_quant` (bf16,
    /// q8, q4), `mmap`, `transformer` (a GGUF or safetensors file, or a folder), `controlnet`
    /// (a folder or repo id), `lora` (`FILE[:SCALE]`, repeatable), `keep_text_encoder`, `stream`.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let flag = || match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(anyhow::anyhow!("{key}: expected a boolean (1/0, true/false), got {value:?}")),
        };
        let quant = || Quant::parse(value).with_context(|| format!("{key}: unknown quantization {value:?} (bf16, q8, q4)"));
        match key {
            "quant" => self.quant = quant()?,
            "text_quant" => self.text_quant = quant()?,
            "mmap" => self.mmap = flag()?,
            "transformer" => self.transformer = (!value.is_empty()).then(|| PathBuf::from(value)),
            "controlnet" => self.controlnet = (!value.is_empty()).then(|| PathBuf::from(value)),
            "lora" => self.loras.push(LoraSpec::parse(value)?),
            "keep_text_encoder" => self.keep_text_encoder = flag()?,
            "stream" => self.stream = flag()?.then(crate::stream::StreamOptions::default),
            "device" => {
                self.device = crate::platform::DevicePref::parse(value)
                    .with_context(|| format!("device: unknown device {value:?} (auto|cpu|cuda|metal|vulkan)"))?
            }
            _ => {
                bail!("unknown option {key:?} (quant, text_quant, mmap, transformer, controlnet, lora, keep_text_encoder, stream, device)")
            }
        }
        Ok(())
    }

    /// The steps of a Lightning LoRA among `loras`, from its file name (`...-4steps-...`), or of
    /// a single-file transformer with the LoRA merged (`..._fp8_e4m3fn_scaled_4steps_...`).
    pub fn lightning_steps(&self) -> Option<usize> {
        let steps = |p: &Path, lora: bool| {
            let n = p.file_name()?.to_string_lossy().to_lowercase();
            if lora && !n.contains("lightning") {
                return None;
            }
            let i = n.find("steps")?;
            let digits: String = n[..i].chars().rev().take_while(|c| c.is_ascii_digit()).collect();
            digits.chars().rev().collect::<String>().parse().ok()
        };
        self.loras
            .iter()
            .find_map(|l| steps(&l.path, true))
            .or_else(|| self.transformer.as_ref().filter(|t| t.is_file()).and_then(|t| steps(t, false)))
    }
}

/// A generation request as JSON (the C API's `request_json`; every key but `prompt` optional).
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub prompt: String,
    pub negative: Option<String>,
    pub steps: Option<usize>,
    pub true_cfg: Option<f32>,
    pub seed: Option<u64>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Lightning settings (fixed shift 3, no guidance); default: when a Lightning LoRA is loaded.
    pub lightning: Option<bool>,
    /// Input image files (edit; decoded with the `media` feature).
    pub images: Vec<String>,
    /// Image to image / inpainting: how much of the schedule is run from the input image, 0..1.
    pub strength: Option<f32>,
    /// Inpainting mask file (white = repaint; decoded with the `media` feature).
    pub mask: Option<String>,
    /// ControlNet condition image files (decoded with the `media` feature) and their scales
    /// (one for all, or one per image), and the part of the schedule the ControlNet acts on.
    pub control_images: Vec<String>,
    pub control_scale: Vec<f32>,
    pub control_start: Option<f32>,
    pub control_end: Option<f32>,
    /// Rescale the guided prediction to the conditional one's norm (default true).
    pub cfg_normalize: Option<bool>,
}

/// A ControlNet condition: the control image (canny edges, soft edges, a depth map, a pose
/// drawing; the image to inpaint for the inpainting ControlNet) and its conditioning scale.
#[derive(Clone, Debug)]
pub struct Control {
    pub image: RgbImage,
    pub scale: f32,
}

/// An RGBA image (a layer of Qwen-Image-Layered).
#[derive(Clone, Debug)]
pub struct RgbaImage {
    pub width: usize,
    pub height: usize,
    /// Interleaved RGBA, row-major.
    pub rgba: Vec<u8>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            text_quant: Quant::Bf16,
            mmap: false,
            loras: Vec::new(),
            transformer: None,
            controlnet: None,
            stream: None,
            keep_text_encoder: false,
            device: crate::platform::DevicePref::Auto,
        }
    }
}

/// One generation's settings.
#[derive(Clone, Debug)]
pub struct Params {
    pub prompt: String,
    /// Negative prompt; guidance runs when it is set and `true_cfg > 1` (the pipelines'
    /// convention: Qwen recommends `" "` with 4.0).
    pub negative: Option<String>,
    pub steps: usize,
    pub true_cfg: f32,
    pub seed: u64,
    /// Output size; None: from the (last) input image's aspect ratio at ~1024x1024 (edit), or
    /// 1024x1024 (text to image). Rounded down to multiples of 16.
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Scheduler override (e.g. [`FlowMatch::lightning`]); None: the folder's.
    pub scheduler: Option<FlowMatch>,
    /// Image to image / inpainting ([`Pipeline::edit`]): the fraction of the schedule run from
    /// the input image (1 starts from pure noise). None: [`DEFAULT_STRENGTH`] with a
    /// text-to-image model, a plain edit with an edit model (1 once a mask is given).
    pub strength: Option<f32>,
    /// Inpainting mask (any size; white = repaint, black = keep), or the mask of the inpainting
    /// ControlNet.
    pub mask: Option<RgbImage>,
    /// ControlNet conditions (several images of one Union ControlNet add up).
    pub control: Vec<Control>,
    /// The ControlNet acts from this fraction of the steps to that one (0 and 1: all of them).
    pub control_start: f32,
    pub control_end: f32,
    /// Rescale the guided prediction to the norm of the conditional one (the pipelines always
    /// do; Qwen-Image-Layered makes it a choice).
    pub cfg_normalize: bool,
    /// Qwen-Image-Layered: the number of layers, and the working resolution (640 or 1024).
    pub layers: usize,
    pub resolution: usize,
    /// Qwen-Image-Layered with an empty prompt: caption the image in English (the model card's
    /// `use_en_prompt=True`) or in Chinese.
    pub caption_en: bool,
}

impl Params {
    /// Qwen's recommended settings: 40 steps, true CFG 4 with a `" "` negative prompt.
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            negative: Some(" ".into()),
            steps: 40,
            true_cfg: 4.0,
            seed: 42,
            width: None,
            height: None,
            scheduler: None,
            strength: None,
            mask: None,
            control: Vec::new(),
            control_start: 0.0,
            control_end: 1.0,
            cfg_normalize: true,
            layers: 4,
            resolution: 640,
            caption_en: true,
        }
    }

    /// Qwen-Image-Layered's settings (the pipeline's defaults): 50 steps, true CFG 4 with a
    /// `" "` negative prompt and no norm rescale, `layers` layers at resolution 640; `caption`
    /// describes the whole image (empty: the text encoder writes it).
    pub fn layered(caption: &str, layers: usize) -> Self {
        Self {
            steps: 50,
            layers,
            cfg_normalize: false,
            ..Self::new(caption)
        }
    }

    /// Settings for a Lightning (step-distilled) LoRA: `steps` steps (4 or 8), no guidance,
    /// the distillation's fixed shift of 3.
    pub fn lightning(prompt: &str, steps: usize) -> Self {
        Self {
            negative: None,
            steps,
            true_cfg: 1.0,
            scheduler: Some(FlowMatch::lightning()),
            ..Self::new(prompt)
        }
    }
}

/// Timings of one call, in seconds.
#[derive(Default, Debug, Clone)]
pub struct Timings {
    pub load_text: f64,
    pub text: f64,
    pub load_transformer: f64,
    pub vae_encode: f64,
    pub denoise: f64,
    pub decode: f64,
    pub transformer_calls: usize,
}

struct LoadedControl {
    // Drop order: the ControlNet (borrowing `_files`), then the files.
    c: controlnet::ControlNet<'static>,
    _files: Box<ModelFiles>,
}

struct LoadedTransformer {
    // Drop order: the transformer (borrowing `_files`, holding a stream hook), the streamer,
    // then the files.
    t: Transformer<'static>,
    _stream: Option<std::sync::Arc<crate::stream::Streamer>>,
    _files: Box<ModelFiles>,
}

/// A Qwen-Image pipeline (see the module docs).
pub struct Pipeline {
    pub dir: PathBuf,
    pub kind: Kind,
    pub opts: Options,
    pub sched: FlowMatch,
    pub tcfg: TransformerConfig,
    text: Option<QwenVl>,
    tr: Option<LoadedTransformer>,
    /// Qwen-Image-2.1: its transformer's config, the transformer once loaded, and the sampling
    /// sigmas a distilled checkpoint saves (`model_index.json`'s `sample_sigmas`).
    pub cfg21: Option<transformer21::Config21>,
    tr21: Option<v21::LoadedTransformer21>,
    pub sample_sigmas: Option<Vec<f32>>,
    /// The alpha channel of the last image of an RGBA model (Qwen-Image-2.1), row-major.
    pub alpha: Option<Vec<u8>>,
    control: Option<LoadedControl>,
    /// The caption the text encoder wrote for the last [`Pipeline::decompose`] without a prompt.
    pub caption: Option<String>,
    vae: vae::Vae,
    /// LoRA load summaries.
    pub lora_report: Vec<String>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// `calculate_dimensions`: a `width x height` of about `area` pixels with aspect `ratio`
/// (width / height), both multiples of 32 (Python's half-to-even rounding).
pub fn calculate_dimensions(area: f64, ratio: f64) -> (usize, usize) {
    let w = (area * ratio).sqrt();
    let h = w / ratio;
    (
        ((w / 32.0).round_ties_even() * 32.0) as usize,
        ((h / 32.0).round_ties_even() * 32.0) as usize,
    )
}

fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("OXINFER_QI_DUMP").map(PathBuf::from)
}

fn dump(name: &str, v: &[f32]) {
    if let Some(d) = dump_dir() {
        let _ = std::fs::write(d.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    }
}

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// RGB8 to `[h*w, 3]` in [-1, 1].
fn to_unit(rgb: &[u8]) -> Vec<f32> {
    rgb.iter().map(|&v| v as f32 / 255.0 * 2.0 - 1.0).collect()
}

/// Decoded pixels in [-1, 1] to bytes.
fn to_u8(px: &[f32]) -> Vec<u8> {
    px.iter()
        .map(|&v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// RGB8 pixels with their width and height.
type Rgb = (Vec<u8>, usize, usize);

/// `img` at `w x h` (Pillow's Lanczos, as the pipelines' image processor resizes).
fn resized(img: &RgbImage, (w, h): (usize, usize)) -> Rgb {
    if (img.width, img.height) == (w, h) {
        return (img.data.clone(), w, h);
    }
    (crate::resample::resize_lanczos_rgb8(&img.data, img.width, img.height, w, h), w, h)
}

/// `[c, n]` planes to `[n, c]` channels-last.
fn chw_to_hwc(chw: &[f32], c: usize) -> Vec<f32> {
    let n = chw.len() / c;
    let mut out = vec![0f32; chw.len()];
    for ch in 0..c {
        for px in 0..n {
            out[px * c + ch] = chw[ch * n + px];
        }
    }
    out
}

/// `n` standard normals from `rng`, or the contents of `name` in the dump folder (validation
/// against the reference with its noise).
fn normals_or_dump(name: &str, rng: &mut crate::sd::ops::Rng, n: usize) -> Result<Vec<f32>> {
    match dump_dir().map(|d| d.join(name)).filter(|f| f.exists()) {
        Some(f) => {
            let v = read_f32(&f)?;
            if v.len() != n {
                bail!("{} has {} values, expected {n}", f.display(), v.len());
            }
            Ok(v)
        }
        None => Ok(rng.normals(n)),
    }
}

/// A mask image at `w x h` as 0 / 1 per pixel (1 = repaint), the way diffusers' mask processor
/// reads it: resized (Lanczos), converted to grey (Pillow's ITU-R 601 weights), 1 from 0.5 up.
fn mask_pixels(mask: &RgbImage, w: usize, h: usize) -> Vec<f32> {
    let (rgb, _, _) = resized(mask, (w, h));
    rgb.as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            let l = (p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 0x8000) >> 16;
            if l >= 128 { 1.0 } else { 0.0 }
        })
        .collect()
}

/// A `w x h` pixel mask at latent resolution `lw x lh` (`F.interpolate`'s nearest neighbour).
fn latent_mask(m: &[f32], w: usize, h: usize, lw: usize, lh: usize) -> Vec<f32> {
    let (sy, sx) = (h as f32 / lh as f32, w as f32 / lw as f32);
    let mut out = Vec::with_capacity(lh * lw);
    for y in 0..lh {
        let yy = ((y as f32 * sy) as usize).min(h - 1);
        for x in 0..lw {
            let xx = ((x as f32 * sx) as usize).min(w - 1);
            out.push(m[yy * w + xx]);
        }
    }
    out
}

/// True classifier-free guidance on `[tokens, c]` predictions: `v = vn + g * (v - vn)`, each
/// token rescaled to the norm of its conditional prediction when `normalize`.
fn guide(v: &mut [f32], vn: &[f32], c: usize, g: f32, normalize: bool) {
    crate::kernels::for_chunks(v, c, true, |t, row| {
        let un = &vn[t * c..(t + 1) * c];
        let cn = row.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
        for (a, &b) in row.iter_mut().zip(un) {
            *a = b + g * (*a - b);
        }
        if normalize {
            let nn = row.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
            let r = (cn / nn) as f32;
            row.iter_mut().for_each(|a| *a *= r);
        }
    });
}

/// Restores the GPU attention precision when a run ends.
struct AttnGuard(bool);

impl Drop for AttnGuard {
    fn drop(&mut self) {
        crate::gpu::set_exact_attention(self.0);
    }
}

/// `_pack_latents`: channels-last latents `[lh*lw, c]` into `[(lh/2)*(lw/2), c*4]` 2x2 patches
/// (channel `c*4 + dy*2 + dx`).
pub fn pack(lat: &[f32], lh: usize, lw: usize, c: usize) -> Vec<f32> {
    let (ph, pw) = (lh / 2, lw / 2);
    let mut out = vec![0f32; ph * pw * c * 4];
    for y in 0..ph {
        for x in 0..pw {
            let t = &mut out[(y * pw + x) * c * 4..(y * pw + x + 1) * c * 4];
            for dy in 0..2 {
                for dx in 0..2 {
                    let src = &lat[((2 * y + dy) * lw + 2 * x + dx) * c..((2 * y + dy) * lw + 2 * x + dx + 1) * c];
                    for ch in 0..c {
                        t[ch * 4 + dy * 2 + dx] = src[ch];
                    }
                }
            }
        }
    }
    out
}

/// The inverse of [`pack`].
pub fn unpack(tok: &[f32], lh: usize, lw: usize, c: usize) -> Vec<f32> {
    let (ph, pw) = (lh / 2, lw / 2);
    let mut lat = vec![0f32; lh * lw * c];
    for y in 0..ph {
        for x in 0..pw {
            let t = &tok[(y * pw + x) * c * 4..(y * pw + x + 1) * c * 4];
            for dy in 0..2 {
                for dx in 0..2 {
                    let dst = &mut lat[((2 * y + dy) * lw + 2 * x + dx) * c..((2 * y + dy) * lw + 2 * x + dx + 1) * c];
                    for ch in 0..c {
                        dst[ch] = t[ch * 4 + dy * 2 + dx];
                    }
                }
            }
        }
    }
    lat
}

/// A reference image ready for the transformer: its packed latents and patch grid.
struct RefImage {
    tokens: Vec<f32>,
    shape: ImageShape,
}

impl Pipeline {
    /// Opens a Qwen-Image diffusers folder (or a Hugging Face repo id, fetched into the cache):
    /// configs and the VAE now, the text encoder and transformer on first use.
    pub fn load(path_or_repo: &str, opts: Options) -> Result<Self> {
        let dir = crate::engine::resolve_model(path_or_repo, None)?;
        crate::gpu::ensure(opts.device)?;
        let kind = kind_of(&dir).with_context(|| format!("{}: not a Qwen-Image diffusers pipeline", dir.display()))?;
        let tdir = dir.join("transformer");
        let cfg21 = match kind {
            Kind::V21 => Some(transformer21::Config21::load(&tdir)?),
            _ => None,
        };
        let tcfg = if let Some(c) = &cfg21 {
            v21::shared_config(c)
        } else if crate::fsx::exists(&tdir.join("config.json")) {
            TransformerConfig::load(&tdir)?
        } else {
            bail!("{}: no transformer/config.json", dir.display());
        };
        let sched = FlowMatch::load(&dir.join("scheduler").join("scheduler_config.json"))?;
        let vae = vae::Vae::load(&dir.join("vae"), true)?;
        if matches!(kind, Kind::Layered | Kind::V21) != (vae.image_channels() == 4) {
            bail!(
                "{}: a {}-channel VAE does not fit this pipeline (Qwen-Image-Layered's and Qwen-Image-2.1's are RGBA, the others' RGB)",
                dir.display(),
                vae.image_channels()
            );
        }
        if (kind == Kind::Layered) != (tcfg.use_layer3d_rope && tcfg.use_additional_t_cond) {
            bail!("{}: the transformer's layered options do not fit the pipeline class", dir.display());
        }
        if vae.cfg.z_dim * tcfg.patch_size * tcfg.patch_size != tcfg.in_channels {
            bail!(
                "VAE latent channels {} do not match the transformer input {}",
                vae.cfg.z_dim,
                tcfg.in_channels
            );
        }
        let sample_sigmas = if kind == Kind::V21 { v21::sample_sigmas(&dir) } else { None };
        Ok(Self {
            dir,
            kind,
            opts,
            sched,
            tcfg,
            text: None,
            tr: None,
            cfg21,
            tr21: None,
            sample_sigmas,
            alpha: None,
            control: None,
            caption: None,
            vae,
            lora_report: Vec::new(),
            cancel: Default::default(),
        })
    }

    /// The settings for `prompt` with this pipeline: Qwen's recommended ones ([`Params::new`]), or
    /// the Lightning ones when a Lightning LoRA is loaded (steps from its name).
    pub fn default_params(&self, prompt: &str) -> Params {
        if self.kind == Kind::V21 {
            return self.params21(prompt);
        }
        match self.opts.lightning_steps() {
            Some(n) => Params::lightning(prompt, n),
            None => Params::new(prompt),
        }
    }

    /// [`Params`] from a [`Request`].
    pub fn params(&self, r: &Request) -> Params {
        let mut p = match (r.lightning, self.opts.lightning_steps()) {
            _ if self.kind == Kind::V21 => self.default_params(&r.prompt),
            (Some(true), n) => Params::lightning(&r.prompt, n.unwrap_or(4)),
            (Some(false), _) => Params::new(&r.prompt),
            (None, _) => self.default_params(&r.prompt),
        };
        if let Some(s) = r.steps {
            p.steps = s;
        }
        if let Some(g) = r.true_cfg {
            p.true_cfg = g;
            if g > 1.0 && p.negative.is_none() {
                p.negative = Some(" ".into());
            }
        }
        if r.negative.is_some() {
            p.negative = r.negative.clone();
        }
        if let Some(s) = r.seed {
            p.seed = s;
        }
        p.width = r.width;
        p.height = r.height;
        p.strength = r.strength;
        if let Some(v) = r.control_start {
            p.control_start = v;
        }
        if let Some(v) = r.control_end {
            p.control_end = v;
        }
        if let Some(v) = r.cfg_normalize {
            p.cfg_normalize = v;
        }
        p
    }

    /// The ControlNet conditions of a request for its decoded control images: the request's
    /// scales, one for all or one each (default 1).
    pub fn controls(r: &Request, images: Vec<RgbImage>) -> Result<Vec<Control>> {
        if !matches!(r.control_scale.len(), 0 | 1) && r.control_scale.len() != images.len() {
            bail!(
                "control_scale: {} values for {} control images",
                r.control_scale.len(),
                images.len()
            );
        }
        Ok(images
            .into_iter()
            .enumerate()
            .map(|(i, image)| Control {
                image,
                scale: *r.control_scale.get(i).or(r.control_scale.first()).unwrap_or(&1.0),
            })
            .collect())
    }

    /// A flag that stops the running (or next) call between denoising steps (any thread).
    pub fn cancel_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.cancel.clone()
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.swap(false, std::sync::atomic::Ordering::Relaxed) {
            bail!("cancelled");
        }
        Ok(())
    }

    /// Loads the text encoder (if not loaded).
    pub fn load_text_encoder(&mut self) -> Result<f64> {
        if self.text.is_some() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let pdir = self.dir.join("processor");
        let has_proc = crate::fsx::exists(&pdir.join("tokenizer.json"));
        let vo = QwenVlOptions {
            quant: self.opts.text_quant,
            mmap: self.opts.mmap,
            tokenizer_dir: has_proc.then(|| pdir.clone()),
            processor_dir: crate::fsx::exists(&pdir.join("preprocessor_config.json")).then(|| pdir.clone()),
            // (Qwen-Image-Layered captions the image when it gets no prompt.)
            vision: self.kind != Kind::TextToImage,
        };
        let te = self.dir.join("text_encoder");
        let mut vl = QwenVl::load(&te, &vo)?;
        if !has_proc {
            vl.tokenizer = match crate::fsx::exists(&self.dir.join("tokenizer").join("tokenizer.json")) {
                true => crate::engine::load_tokenizer(&self.dir.join("tokenizer"))?,
                false => crate::qwen_vl::qwen2_tokenizer_from_vocab(&self.dir.join("tokenizer"))?,
            };
        }
        if vl.hidden_size() != self.tcfg.joint_attention_dim {
            bail!(
                "text encoder width {} does not match the transformer's {}",
                vl.hidden_size(),
                self.tcfg.joint_attention_dim
            );
        }
        self.text = Some(vl);
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Drops the text encoder.
    pub fn unload_text_encoder(&mut self) {
        self.text = None;
    }

    /// Loads the transformer (if not loaded), merging the LoRA files.
    pub fn load_transformer(&mut self) -> Result<f64> {
        if self.tr.is_some() || self.tr21.is_some() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let src = self.opts.transformer.clone().unwrap_or_else(|| self.dir.join("transformer"));
        let in_memory = !(self.opts.mmap || self.opts.stream.is_some());
        let files = Box::new(ModelFiles::open(&src, in_memory).with_context(|| format!("transformer weights {}", src.display()))?);
        // SAFETY: the box's heap block does not move; `LoadedTransformer` drops the transformer
        // (every borrow of it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        if fs.names().any(|n| n.ends_with(".comfy_quant")) {
            bail!(
                "{}: a ComfyUI mixed-precision checkpoint (`comfy_quant` layers), which is not supported; use the FP8 scaled, GGUF or bf16 weights",
                src.display()
            );
        }
        let prefix = ["", "model.diffusion_model.", "diffusion_model.", "transformer."]
            .into_iter()
            .find(|p| fs.has(&format!("{p}img_in.weight")))
            .with_context(|| format!("{}: no img_in.weight (not a Qwen-Image transformer)", src.display()))?
            .to_string();
        let is21 = fs.has(&format!("{prefix}modulation.1.weight"));
        if is21 != (self.kind == Kind::V21) {
            bail!(
                "{}: {} transformer, which does not fit this pipeline",
                src.display(),
                if is21 { "a Qwen-Image-2.1" } else { "not a Qwen-Image-2.1" }
            );
        }
        if is21 {
            self.load_transformer21(files, prefix)?;
            return Ok(t0.elapsed().as_secs_f64());
        }
        let lora = if self.opts.loras.is_empty() {
            None
        } else {
            let names: Vec<String> = (0..self.tcfg.num_layers)
                .flat_map(|l| {
                    [
                        "img_mod.1",
                        "txt_mod.1",
                        "attn.to_q",
                        "attn.to_k",
                        "attn.to_v",
                        "attn.add_q_proj",
                        "attn.add_k_proj",
                        "attn.add_v_proj",
                        "attn.to_out.0",
                        "attn.to_add_out",
                        "img_mlp.net.0.proj",
                        "img_mlp.net.2",
                        "txt_mlp.net.0.proj",
                        "txt_mlp.net.2",
                    ]
                    .map(|m| format!("{prefix}transformer_blocks.{l}.{m}.weight"))
                })
                .chain(["img_in", "txt_in", "proj_out", "norm_out.linear"].map(|m| format!("{prefix}{m}.weight")))
                .collect();
            Some(weights::LoraSet::open(&self.opts.loras, fs, &names, &prefix)?)
        };
        let ld = weights::Loader {
            files: fs,
            quant: Quant::stored(fs).unwrap_or(self.opts.quant),
            lora: lora.as_ref(),
            prefix,
        };
        let mut t = Transformer::load(self.tcfg.clone(), &ld)?;
        self.lora_report.clear();
        if let Some(l) = &lora {
            for (f, n) in &l.files {
                self.lora_report.push(format!("[lora {f}: {n} modules]"));
            }
            if l.applied() != l.modules() {
                bail!(
                    "LoRA: {} of {} modules matched no loaded weight",
                    l.modules() - l.applied(),
                    l.modules()
                );
            }
        }
        let stream = match &self.opts.stream {
            Some(so) => {
                let s = crate::stream::Streamer::new(&[fs], t.stream_units(), so.clone())?;
                let streamed: Vec<bool> = s.plan().units.iter().map(|u| !u.2).collect();
                t.mark_streamed(&streamed);
                t.set_stream(Some(crate::stream::StreamHook::new(s.clone(), 0)));
                Some(s)
            }
            None => None,
        };
        self.tr = Some(LoadedTransformer {
            t,
            _stream: stream,
            _files: files,
        });
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Drops the transformer.
    pub fn unload_transformer(&mut self) {
        self.tr = None;
        self.tr21 = None;
    }

    /// Loads the ControlNet of `Options::controlnet` (if not loaded), in its checkpoint precision.
    pub fn load_controlnet(&mut self) -> Result<f64> {
        if self.control.is_some() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let spec = self
            .opts
            .controlnet
            .clone()
            .context("no ControlNet was given (Options::controlnet)")?;
        let dir = crate::engine::resolve_model(&spec.to_string_lossy(), None)?;
        if !controlnet::is_controlnet(&dir) {
            bail!(
                "{}: not a Qwen-Image ControlNet (config.json with QwenImageControlNetModel)",
                dir.display()
            );
        }
        let cfg = TransformerConfig::load(&dir)?;
        if cfg.inner() != self.tcfg.inner()
            || cfg.in_channels != self.tcfg.in_channels
            || cfg.joint_attention_dim != self.tcfg.joint_attention_dim
        {
            bail!("{}: the ControlNet's widths do not match the transformer's", dir.display());
        }
        let files = Box::new(ModelFiles::open(&dir, !self.opts.mmap).with_context(|| format!("ControlNet weights {}", dir.display()))?);
        // SAFETY: the box's heap block does not move; `LoadedControl` drops the ControlNet (every
        // borrow of it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let ld = weights::Loader {
            files: fs,
            quant: Quant::Bf16,
            lora: None,
            prefix: String::new(),
        };
        let c = controlnet::ControlNet::load(cfg, &ld)?;
        self.control = Some(LoadedControl { c, _files: files });
        Ok(t0.elapsed().as_secs_f64())
    }

    /// The ControlNet, once loaded.
    pub fn controlnet(&self) -> Option<&controlnet::ControlNet<'static>> {
        self.control.as_ref().map(|c| &c.c)
    }

    /// The prompt template and the number of leading hidden states dropped.
    fn template(&self) -> (&'static str, usize) {
        match self.kind {
            Kind::TextToImage | Kind::Layered => (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                34,
            ),
            Kind::Edit => (
                "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n",
                64,
            ),
            // (Qwen-Image-2.1 builds its prompt in `v21`.)
            Kind::EditPlus | Kind::V21 => (
                "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                64,
            ),
        }
    }

    /// The conditioning hidden states of `prompt` with the condition images: `[t, 3584]`.
    fn encode_prompt(&self, prompt: &str, cond: &[VlImage]) -> Result<Vec<f32>> {
        let vl = self.text.as_ref().context("text encoder not loaded")?;
        let (tpl, drop) = self.template();
        let user = match self.kind {
            Kind::EditPlus => {
                let mut s = String::new();
                for (i, c) in cond.iter().enumerate() {
                    s.push_str(&format!("Picture {}: {}", i + 1, c.placeholder()));
                }
                s + prompt
            }
            _ => prompt.to_string(),
        };
        let mut text = tpl.replacen("{}", &user, 1);
        if self.kind == Kind::Edit {
            let c = cond.first().context("an edit needs an image")?;
            text = text.replacen("<|vision_start|><|image_pad|><|vision_end|>", &c.placeholder(), 1);
        }
        let h = vl.hidden_states(&text, cond, &[])?;
        let d = h.hidden;
        let n = h.ids.len();
        if n <= drop {
            bail!("the prompt template is shorter than the {drop} tokens it drops");
        }
        let mut keep = n - drop;
        if matches!(self.kind, Kind::TextToImage | Kind::Layered) {
            keep = keep.min(512);
        }
        Ok(h.last[drop * d..(drop + keep) * d].to_vec())
    }

    /// Text to image (any Qwen-Image folder; an edit model then generates without a reference),
    /// with the ControlNet images of `p.control` when a ControlNet is loaded.
    pub fn generate(&mut self, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        if self.kind == Kind::Layered {
            bail!("{}: a layer-decomposition model; use decompose", self.dir.display());
        }
        if self.kind == Kind::V21 {
            return self.run21(&[], p, progress);
        }
        if p.mask.is_some() && p.control.is_empty() {
            bail!("a mask needs the image to inpaint");
        }
        self.run(&[], p, progress)
    }

    /// Edits `images` following the instruction in `p.prompt` (one image for Qwen-Image-Edit; one
    /// or more for Edit-2509 / 2511). With `p.strength` or `p.mask` the first image (the last one
    /// for Edit-2509 / 2511) is also where denoising starts from, and what the kept region of an
    /// inpainting returns to. A text-to-image model (Qwen-Image, -2512) takes one image:
    /// image to image (`p.strength`, default 0.6) or inpainting (`p.mask`).
    pub fn edit(&mut self, images: &[RgbImage], p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        if images.is_empty() {
            bail!("an edit needs at least one input image");
        }
        match self.kind {
            Kind::V21 => return self.run21(images, p, progress),
            Kind::Layered => bail!("{}: a layer-decomposition model; use decompose", self.dir.display()),
            Kind::TextToImage if images.len() > 1 => bail!("image to image takes one image"),
            Kind::Edit if images.len() > 1 => bail!("Qwen-Image-Edit takes one image (Edit-2509 / 2511 take several)"),
            _ => {}
        }
        self.run(images, p, progress)
    }

    /// [`Pipeline::edit`] on oxideav pictures.
    #[cfg(feature = "oxideav")]
    pub fn edit_pictures(
        &mut self,
        pictures: &[crate::av::Picture],
        p: &Params,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<(Image, Timings)> {
        let imgs = pictures.iter().map(|pc| pc.to_rgb()).collect::<Result<Vec<_>>>()?;
        self.edit(&imgs, p, progress)
    }

    /// The prompt (and negative prompt) conditioning with the condition images, with the text
    /// encoder loaded for the call (and dropped after it unless it is kept).
    fn conditioning(&mut self, cond_rgb: &[Rgb], p: &Params, tm: &mut Timings) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        // The text encoder's hidden states carry outlier channels (|x| ~ 200) that the GPU's
        // half-precision matrix units round visibly (1e-2 relative), so the encoding runs with
        // exact GPU matmuls; it is a small part of the work.
        tm.load_text = self.load_text_encoder()?;
        let t0 = Instant::now();
        let was_exact = crate::gpu::exact();
        crate::gpu::set_exact(true);
        let enc = self.encode_all(cond_rgb, p);
        crate::gpu::set_exact(was_exact);
        let (mut pos, mut neg) = enc?;
        // Validation: the reference's conditioning instead of ours (`prompt_embeds_in.bin`,
        // `negative_embeds_in.bin` in the dump folder), to check the transformer alone.
        if let Some(d) = dump_dir() {
            for (name, dst) in [("prompt_embeds_in.bin", Some(&mut pos)), ("negative_embeds_in.bin", neg.as_mut())] {
                if let (Some(dst), true) = (dst, d.join(name).exists()) {
                    *dst = read_f32(&d.join(name))?;
                }
            }
        }
        dump("prompt_embeds.bin", &pos);
        if let Some(n) = &neg {
            dump("negative_embeds.bin", n);
        }
        tm.text = t0.elapsed().as_secs_f64();
        if !self.opts.keep_text_encoder {
            self.unload_text_encoder();
        }
        self.check_cancel()?;
        Ok((pos, neg))
    }

    fn run(&mut self, images: &[RgbImage], p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut tm = Timings::default();
        let sf = self.vae.scale();
        let mult = sf * 2;
        for img in images.iter().chain(p.control.iter().map(|c| &c.image)).chain(&p.mask) {
            if img.data.len() != img.width * img.height * 3 || img.data.is_empty() {
                bail!("input image: {} bytes for {}x{} RGB", img.data.len(), img.width, img.height);
            }
        }
        if !p.control.is_empty() {
            if self.opts.controlnet.is_none() {
                bail!("control images need a ControlNet (Options::controlnet, --controlnet)");
            }
            if self.kind != Kind::TextToImage || !images.is_empty() {
                bail!("the ControlNet runs with text to image only (Qwen-Image, Qwen-Image-2512, no input image)");
            }
        }
        // Output size.
        let (cw, ch) = match (self.kind, images.last(), images.first()) {
            (Kind::EditPlus, Some(last), _) => calculate_dimensions(1024.0 * 1024.0, last.width as f64 / last.height as f64),
            (Kind::Edit, _, Some(first)) => calculate_dimensions(1024.0 * 1024.0, first.width as f64 / first.height as f64),
            // Image to image and ControlNet: the input's size.
            (Kind::TextToImage, _, Some(i)) => (i.width, i.height),
            _ => match p.control.first() {
                Some(c) => (c.image.width, c.image.height),
                None => (1024, 1024),
            },
        };
        let width = p.width.unwrap_or(cw) / mult * mult;
        let height = p.height.unwrap_or(ch) / mult * mult;
        if width < mult || height < mult {
            bail!("output size {width}x{height} is too small");
        }
        // Condition images (vision tower) and VAE images.
        let mut cond_rgb: Vec<Rgb> = Vec::new();
        let mut vae_rgb: Vec<Rgb> = Vec::new();
        if self.kind != Kind::TextToImage {
            for img in images {
                let ratio = img.width as f64 / img.height as f64;
                match self.kind {
                    Kind::EditPlus => {
                        cond_rgb.push(resized(img, calculate_dimensions(384.0 * 384.0, ratio)));
                        vae_rgb.push(resized(img, calculate_dimensions(1024.0 * 1024.0, ratio)));
                    }
                    _ => {
                        let r = resized(img, (cw, ch));
                        cond_rgb.push(r.clone());
                        vae_rgb.push(r);
                    }
                }
            }
        }
        // Image to image / inpainting: the image denoising starts from, at the output size.
        let inpaint_cn = p.mask.is_some() && !p.control.is_empty();
        let from_image = !images.is_empty() && (self.kind == Kind::TextToImage || p.strength.is_some() || p.mask.is_some());
        let init = match (from_image, self.kind) {
            (false, _) => None,
            (true, Kind::EditPlus) => images.last().map(|i| resized(i, (width, height))),
            (true, _) => images.first().map(|i| resized(i, (width, height))),
        };
        let mask = p.mask.as_ref().map(|m| mask_pixels(m, width, height));
        let (pos, neg) = self.conditioning(&cond_rgb, p, &mut tm)?;

        // Reference latents (the posterior's mode, as the edit pipelines take it).
        let t1 = Instant::now();
        let z = self.vae.cfg.z_dim;
        let (lh, lw) = (2 * (height / mult), 2 * (width / mult));
        let mut rng = crate::sd::ops::Rng::new(p.seed);
        // Initial noise, generated as [c, h, w] (or read from the dump folder), then packed.
        let noise = pack(&chw_to_hwc(&normals_or_dump("noise.bin", &mut rng, z * lh * lw)?, z), lh, lw, z);
        // The image to start from: a sample of its posterior, as diffusers' image-to-image and
        // inpainting pipelines encode it. Qwen-Image-Edit's inpainting pipeline uses that same
        // sample as the reference tokens.
        let init_tokens = match &init {
            Some((rgb, w, h)) => {
                let mut eps = |lh: usize, lw: usize| Ok(chw_to_hwc(&normals_or_dump("posterior_eps.bin", &mut rng, z * lh * lw)?, z));
                let (lat, ih, iw) = self.vae.encode_sample(&to_unit(rgb), *h, *w, Some(&mut eps))?;
                debug_assert_eq!((ih, iw), (lh, lw));
                let t = pack(&lat, lh, lw, z);
                dump("init_latents.bin", &t);
                Some(t)
            }
            None => None,
        };
        let mut refs = Vec::new();
        for (i, (rgb, w, h)) in vae_rgb.iter().enumerate() {
            let tokens = match (&init_tokens, self.kind) {
                (Some(t), Kind::Edit) if (*w, *h) == (width, height) => t.clone(),
                _ => {
                    let (lat, rh, rw) = self.vae.encode(&to_unit(rgb), *h, *w)?;
                    pack(&lat, rh, rw, z)
                }
            };
            if i == 0 {
                dump("ref_latents0.bin", &tokens);
            }
            refs.push(RefImage {
                tokens,
                shape: ImageShape {
                    frames: 1,
                    rows: h / mult,
                    cols: w / mult,
                },
            });
        }
        // Control tokens: the packed latents (a posterior sample) of each control image; the
        // inpainting ControlNet takes the image with the repainted region blanked, plus the kept
        // region as a 17th latent channel.
        let mut controls: Vec<(Vec<f32>, f32)> = Vec::new();
        for (i, c) in p.control.iter().enumerate() {
            let (rgb, w, h) = resized(&c.image, (width, height));
            let mut px = to_unit(&rgb);
            if let (Some(m), true) = (&mask, inpaint_cn) {
                for (v, &mv) in px.as_chunks_mut::<3>().0.iter_mut().zip(m) {
                    if mv > 0.5 {
                        v.fill(-1.0);
                    }
                }
            }
            let name = if i == 0 {
                "control_eps.bin".to_string()
            } else {
                format!("control_eps{i}.bin")
            };
            let mut eps = |lh: usize, lw: usize| Ok(chw_to_hwc(&normals_or_dump(&name, &mut rng, z * lh * lw)?, z));
            let (lat, _, _) = self.vae.encode_sample(&px, h, w, Some(&mut eps))?;
            let tokens = match (&mask, inpaint_cn) {
                (Some(m), true) => {
                    let keep = latent_mask(m, width, height, lw, lh);
                    let mut l17 = vec![0f32; lh * lw * (z + 1)];
                    for (px, dst) in l17.chunks_exact_mut(z + 1).enumerate() {
                        dst[..z].copy_from_slice(&lat[px * z..(px + 1) * z]);
                        dst[z] = 1.0 - keep[px];
                    }
                    pack(&l17, lh, lw, z + 1)
                }
                _ => pack(&lat, lh, lw, z),
            };
            if i == 0 {
                dump("control0.bin", &tokens);
            }
            controls.push((tokens, c.scale));
        }
        tm.vae_encode = t1.elapsed().as_secs_f64();

        tm.load_transformer = self.load_transformer()?;
        if !controls.is_empty() {
            tm.load_transformer += self.load_controlnet()?;
        }
        let t2 = Instant::now();
        let tr = &self.tr.as_ref().unwrap().t;
        let cn = match controls.is_empty() {
            true => None,
            false => self.control.as_ref().map(|c| &c.c),
        };
        let target = ImageShape {
            frames: 1,
            rows: lh / 2,
            cols: lw / 2,
        };
        let n0 = target.tokens();
        let c4 = self.tcfg.in_channels;
        if let Some(cn) = cn {
            if cn.inpaint() != inpaint_cn {
                match cn.inpaint() {
                    true => bail!("an inpainting ControlNet needs a mask (Params::mask, --mask)"),
                    false => bail!("this ControlNet takes no mask (the inpainting ControlNet does)"),
                }
            }
            if controls.iter().any(|(t, _)| t.len() != n0 * cn.cond_channels()) {
                bail!("control tokens do not match the ControlNet's {} channels", cn.cond_channels());
            }
        }
        let mut shapes = vec![target];
        let mut ref_tokens = Vec::new();
        for r in &refs {
            shapes.push(r.shape);
            ref_tokens.extend_from_slice(&r.tokens);
        }
        drop(refs);
        let ja = self.tcfg.joint_attention_dim;
        let txt_pos = tr.embed_text(&pos, pos.len() / ja);
        let txt_neg = neg.as_ref().map(|n| tr.embed_text(n, n.len() / ja));
        let txt_cn = cn.map(|c| c.embed_text(&pos, pos.len() / ja));
        let sched = p.scheduler.clone().unwrap_or_else(|| self.sched.clone());
        let mut sigmas = sched.sigmas(p.steps, n0)?;
        // Image to image: the last `steps * strength` steps, from the image noised to their
        // first noise level (diffusers' `get_timesteps` / `scale_noise`).
        let mut x = noise.clone();
        if let Some(it) = &init_tokens {
            let strength = p
                .strength
                .unwrap_or(if self.kind == Kind::TextToImage { DEFAULT_STRENGTH } else { 1.0 });
            if !(0.0..=1.0).contains(&strength) {
                bail!("strength {strength} is outside 0..1");
            }
            let total = sigmas.len() - 1;
            let s = (strength as f64 * 1e6).round() / 1e6;
            let start = (total as f64 - (total as f64 * s).min(total as f64)).max(0.0) as usize;
            if start >= total {
                bail!("strength {strength} leaves no step of {total} to run");
            }
            sigmas.drain(..start);
            x = FlowMatch::scale_noise(it, &noise, sigmas[0]);
        }
        dump("x0.bin", &x);
        // Inpainting by latent blending: 1 where the image is repainted, per packed channel.
        let blend = match (&mask, &init_tokens, inpaint_cn) {
            (Some(m), Some(_), false) => {
                let lm = latent_mask(m, width, height, lw, lh);
                let rep: Vec<f32> = lm.iter().flat_map(|&v| std::iter::repeat_n(v, z)).collect();
                Some(pack(&rep, lh, lw, z))
            }
            (Some(_), None, false) => bail!("a mask needs the image to inpaint"),
            _ => None,
        };
        let steps = sigmas.len() - 1;
        // Attention logits run into the thousands: half-precision queries and keys (the matrix-unit
        // attention kernel) move the velocity by ~10%, so attention stays f32 (the matmuls do not).
        let _attn = AttnGuard(crate::gpu::set_exact_attention(true));
        let mut input = vec![0f32; (n0 * c4) + ref_tokens.len()];
        input[n0 * c4..].copy_from_slice(&ref_tokens);
        for i in 0..steps {
            self.check_cancel()?;
            let (s, s_next) = (sigmas[i], sigmas[i + 1]);
            input[..n0 * c4].copy_from_slice(&x);
            // ControlNet residuals (the same for the guided and unguided passes), while the step
            // lies inside [control_start, control_end] of the schedule.
            let keep = !((i as f32 / steps as f32) < p.control_start || ((i + 1) as f32 / steps as f32) > p.control_end);
            let mut residuals: Option<Vec<Vec<f32>>> = None;
            if let (Some(cn), Some(tc), true) = (cn, &txt_cn, keep) {
                for (tokens, scale) in &controls {
                    let r = cn.forward(&x, tokens, target, tc, s, *scale);
                    match &mut residuals {
                        None => residuals = Some(r),
                        Some(acc) => acc.iter_mut().zip(&r).for_each(|(a, b)| crate::kernels::add_inplace(a, b)),
                    }
                }
                if let (0, Some(r)) = (i, &residuals) {
                    dump("control_res.bin", &r.concat());
                }
            }
            let extra = Extra {
                control: residuals.as_deref(),
                ..Default::default()
            };
            let mut v = tr.forward_with(&input, &shapes, &txt_pos, s, &extra);
            tm.transformer_calls += 1;
            if let Some(tn) = &txt_neg {
                let vn = tr.forward_with(&input, &shapes, tn, s, &extra);
                tm.transformer_calls += 1;
                guide(&mut v, &vn, c4, p.true_cfg, p.cfg_normalize);
            }
            if i == 0 {
                dump("v0.bin", &v);
            }
            let dt = s_next - s;
            for (a, &b) in x.iter_mut().zip(&v) {
                *a += dt * b;
            }
            // Inpainting: the kept region returns to the image, noised to the next level.
            if let (Some(m), Some(it)) = (&blend, &init_tokens) {
                for (((a, &mv), &i0), &n) in x.iter_mut().zip(m).zip(it).zip(&noise) {
                    let keep = if i + 1 < steps { s_next * n + (1.0 - s_next) * i0 } else { i0 };
                    *a = (1.0 - mv) * keep + mv * *a;
                }
            }
            progress(i + 1, steps);
        }
        dump("latents_out.bin", &x);
        tm.denoise = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        let lat = unpack(&x, lh, lw, z);
        let (px, h, w) = self.vae.decode(&lat, lh, lw);
        dump("image.bin", &px);
        let rgb = to_u8(&px);
        tm.decode = t3.elapsed().as_secs_f64();
        Ok((Image { width: w, height: h, rgb }, tm))
    }

    /// A caption of an image (RGB8, `w x h`) by the text encoder, as diffusers'
    /// `get_image_caption` asks for it: at most 512 new tokens, greedy with the model's
    /// repetition penalty of 1.05 (its generation config: top-k 1).
    fn caption(&self, rgb: &[u8], w: usize, h: usize, en: bool) -> Result<String> {
        let vl = self.text.as_ref().context("text encoder not loaded")?;
        let was_exact = crate::gpu::exact();
        crate::gpu::set_exact(true);
        let ids = (|| {
            let img = vl.encode_image(rgb, w, h, 0)?;
            let prompt = if en { caption::PROMPT_EN } else { caption::PROMPT_CN };
            let text = prompt.replacen("<|vision_start|><|image_pad|><|vision_end|>", &img.placeholder(), 1);
            let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
                .iter()
                .filter_map(|t| vl.tokenizer.token_to_id(t))
                .collect();
            vl.generate(&text, &[img], 512, &eos, 1.05)
        })();
        crate::gpu::set_exact(was_exact);
        let text = vl
            .tokenizer
            .decode(&ids?, true)
            .map_err(|e| anyhow::anyhow!("decode the caption: {e}"))?;
        Ok(text.trim().to_string())
    }

    /// Qwen-Image-Layered: decomposes `image` into `p.layers` RGBA layers (back to front as the
    /// model orders them), following diffusers' `QwenImageLayeredPipeline`. `p.prompt` is a
    /// caption of the whole image (empty: the text encoder writes one, kept in
    /// [`Pipeline::caption`]); `p.resolution` (640 or 1024) sets the working size, with the
    /// image's aspect ratio.
    pub fn decompose(&mut self, image: &RgbImage, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Vec<RgbaImage>, Timings)> {
        if self.kind != Kind::Layered {
            bail!("{}: not a Qwen-Image-Layered pipeline", self.dir.display());
        }
        if image.data.len() != image.width * image.height * 3 || image.data.is_empty() {
            bail!("input image: {} bytes for {}x{} RGB", image.data.len(), image.width, image.height);
        }
        if p.layers == 0 {
            bail!("layers must be at least 1");
        }
        if !matches!(p.resolution, 640 | 1024) {
            bail!("resolution must be 640 or 1024, got {}", p.resolution);
        }
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut tm = Timings::default();
        let mult = self.vae.scale() * 2;
        let ic = self.vae.image_channels();
        let res = p.resolution as f64;
        let (cw, ch) = calculate_dimensions(res * res, image.width as f64 / image.height as f64);
        let (width, height) = (cw / mult * mult, ch / mult * mult);
        let (rgb, _, _) = resized(image, (cw, ch));
        // No prompt: the text encoder captions the image, and the caption is the prompt.
        let mut p = p.clone();
        if p.prompt.trim().is_empty() {
            tm.load_text = self.load_text_encoder()?;
            let t0 = Instant::now();
            p.prompt = self.caption(&rgb, cw, ch, p.caption_en)?;
            tm.text = t0.elapsed().as_secs_f64();
            self.caption = Some(p.prompt.clone());
        }
        let p = &p;
        let (load, text) = (tm.load_text, tm.text);
        let (pos, neg) = self.conditioning(&[], p, &mut tm)?;
        tm.load_text += load;
        tm.text += text;

        // The image as the condition: resized, with an opaque alpha channel for the RGBA VAE.
        let t1 = Instant::now();
        let z = self.vae.cfg.z_dim;
        let mut px = vec![1f32; cw * ch * ic];
        for (dst, src) in px.chunks_exact_mut(ic).zip(rgb.as_chunks::<3>().0) {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s as f32 / 255.0 * 2.0 - 1.0;
            }
        }
        let (lat, rh, rw) = self.vae.encode(&px, ch, cw)?;
        let cond = pack(&lat, rh, rw, z);
        dump("ref_latents0.bin", &cond);
        tm.vae_encode = t1.elapsed().as_secs_f64();

        tm.load_transformer = self.load_transformer()?;
        let t2 = Instant::now();
        let tr = &self.tr.as_ref().unwrap().t;
        let (lh, lw) = (2 * (height / mult), 2 * (width / mult));
        let frames = p.layers + 1;
        let layer = ImageShape {
            frames: 1,
            rows: lh / 2,
            cols: lw / 2,
        };
        let cond_shape = ImageShape {
            frames: 1,
            rows: rh / 2,
            cols: rw / 2,
        };
        let c4 = self.tcfg.in_channels;
        let nl = layer.tokens();
        let n0 = frames * nl;
        // Noise for the composite and every layer, `[frames, c, h, w]`, packed frame by frame.
        let mut rng = crate::sd::ops::Rng::new(p.seed);
        let noise = normals_or_dump("noise.bin", &mut rng, frames * z * lh * lw)?;
        let mut x = Vec::with_capacity(n0 * c4);
        for f in noise.chunks_exact(z * lh * lw) {
            x.extend(pack(&chw_to_hwc(f, z), lh, lw, z));
        }
        dump("x0.bin", &x);
        let mut shapes = vec![layer; frames];
        shapes.push(cond_shape);
        let ja = self.tcfg.joint_attention_dim;
        let txt_pos = tr.embed_text(&pos, pos.len() / ja);
        let txt_neg = neg.as_ref().map(|n| tr.embed_text(n, n.len() / ja));
        // Sigmas `linspace(1, 0, steps + 1)[:-1]`, shifted with `mu = sqrt(condition tokens / 256)`.
        let n = p.steps.max(1);
        let base: Vec<f32> = (0..n).map(|i| (1.0 + i as f64 * (-1.0 / n as f64)) as f32).collect();
        let mu = (cond_shape.tokens() as f64 / 256.0).sqrt();
        let sched = p.scheduler.clone().unwrap_or_else(|| self.sched.clone());
        let sigmas = sched.plan(&base, Some(mu))?.sigmas;
        let steps = sigmas.len() - 1;
        let _attn = AttnGuard(crate::gpu::set_exact_attention(true));
        let mut input = vec![0f32; (n0 + cond_shape.tokens()) * c4];
        input[n0 * c4..].copy_from_slice(&cond);
        let extra = Extra {
            targets: frames,
            t_cond: Some(0),
            ..Default::default()
        };
        for i in 0..steps {
            self.check_cancel()?;
            let (s, s_next) = (sigmas[i], sigmas[i + 1]);
            input[..n0 * c4].copy_from_slice(&x);
            let mut v = tr.forward_with(&input, &shapes, &txt_pos, s, &extra);
            tm.transformer_calls += 1;
            if let Some(tn) = &txt_neg {
                let vn = tr.forward_with(&input, &shapes, tn, s, &extra);
                tm.transformer_calls += 1;
                guide(&mut v, &vn, c4, p.true_cfg, p.cfg_normalize);
            }
            if i == 0 {
                dump("v0.bin", &v);
            }
            let dt = s_next - s;
            for (a, &b) in x.iter_mut().zip(&v) {
                *a += dt * b;
            }
            progress(i + 1, steps);
        }
        dump("latents_out.bin", &x);
        tm.denoise = t2.elapsed().as_secs_f64();

        // The first frame is the composite (the input again); the layers follow.
        let t3 = Instant::now();
        let mut out = Vec::with_capacity(p.layers);
        let mut all = Vec::new();
        for f in x.chunks_exact(nl * c4).skip(1) {
            let (px, h, w) = self.vae.decode(&unpack(f, lh, lw, z), lh, lw);
            if dump_dir().is_some() {
                all.extend_from_slice(&px);
            }
            out.push(RgbaImage {
                width: w,
                height: h,
                rgba: to_u8(&px),
            });
        }
        dump("image.bin", &all);
        tm.decode = t3.elapsed().as_secs_f64();
        Ok((out, tm))
    }

    /// The prompt (and negative prompt) conditioning with the condition images.
    fn encode_all(&self, cond_rgb: &[Rgb], p: &Params) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let vl = self.text.as_ref().context("text encoder not loaded")?;
        let cond: Vec<VlImage> = cond_rgb
            .iter()
            .map(|(rgb, w, h)| vl.encode_image(rgb, *w, *h, 0))
            .collect::<Result<_>>()?;
        if let Some(c) = cond.first() {
            dump("cond_image0.bin", &c.emb);
        }
        let pos = self.encode_prompt(&p.prompt, &cond)?;
        let guided = p.true_cfg > 1.0 && p.negative.is_some();
        let neg = match (&p.negative, guided) {
            (Some(n), true) => Some(self.encode_prompt(n, &cond)?),
            _ => None,
        };
        Ok((pos, neg))
    }

    /// The VAE (for component checks).
    pub fn vae(&self) -> &vae::Vae {
        &self.vae
    }

    /// The transformer, once loaded.
    pub fn transformer(&self) -> Option<&Transformer<'static>> {
        self.tr.as_ref().map(|t| &t.t)
    }

    /// The text encoder, while loaded.
    pub fn text_encoder(&self) -> Option<&QwenVl> {
        self.text.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_match_diffusers() {
        assert_eq!(calculate_dimensions(1024.0 * 1024.0, 1.0), (1024, 1024));
        assert_eq!(calculate_dimensions(1024.0 * 1024.0, 640.0 / 480.0), (1184, 896));
        assert_eq!(calculate_dimensions(384.0 * 384.0, 1.0), (384, 384));
        assert_eq!(calculate_dimensions(384.0 * 384.0, 16.0 / 9.0), (512, 288));
    }

    #[test]
    fn lightning_steps_from_name() {
        let o = |s: &str| Options {
            loras: vec![LoraSpec::parse(s).unwrap()],
            ..Default::default()
        };
        assert_eq!(
            o("x/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors").lightning_steps(),
            Some(4)
        );
        assert_eq!(o("Qwen-Image-Lightning-8steps-V1.1.safetensors:0.8").lightning_steps(), Some(8));
        assert_eq!(o("style.safetensors").lightning_steps(), None);
    }

    /// The mask reaches the latents as diffusers' mask processor and `F.interpolate` give it:
    /// binarized at 0.5, nearest neighbour at `floor(i * size / latent size)`.
    #[test]
    fn mask_to_latents() {
        let (w, h) = (32usize, 16usize);
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if x >= 16 {
                    255
                } else if y >= 8 {
                    127
                } else {
                    128
                };
                data[(y * w + x) * 3..(y * w + x + 1) * 3].fill(v);
            }
        }
        let m = mask_pixels(&RgbImage { width: w, height: h, data }, w, h);
        assert_eq!((m[0], m[8 * w], m[16]), (1.0, 0.0, 1.0));
        let l = latent_mask(&m, w, h, 4, 2);
        assert_eq!(l, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
        assert_eq!(chw_to_hwc(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn guidance_rescale() {
        let mut v = vec![3.0, 4.0, 1.0, 0.0];
        guide(&mut v, &[0.0, 0.0, 0.0, 1.0], 2, 2.0, true);
        // Token 0: 2 * (3, 4) rescaled to norm 5; token 1: (2, -1) rescaled to norm 1.
        assert!((v[0] - 3.0).abs() < 1e-6 && (v[1] - 4.0).abs() < 1e-6);
        assert!((v[2] - 2.0 / 5f32.sqrt()).abs() < 1e-6 && (v[3] + 1.0 / 5f32.sqrt()).abs() < 1e-6);
        let mut v = vec![3.0, 4.0];
        guide(&mut v, &[1.0, 1.0], 2, 2.0, false);
        assert_eq!(v, [5.0, 7.0]);
    }

    #[test]
    fn pack_roundtrip() {
        let (lh, lw, c) = (4, 6, 3);
        let lat: Vec<f32> = (0..lh * lw * c).map(|i| i as f32).collect();
        let p = pack(&lat, lh, lw, c);
        // Token 0, channel 0: pixel (0,0) c0; channel 1 (dx=1): pixel (0,1) c0; channel 4: (0,0) c1.
        assert_eq!((p[0], p[1], p[2], p[4]), (0.0, 3.0, 18.0, 1.0));
        assert_eq!(unpack(&p, lh, lw, c), lat);
    }
}
