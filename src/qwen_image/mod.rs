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
//! LoRA files (Lightning step distillation for 4 / 8 steps, style LoRAs) are merged into the
//! transformer at load; the transformer can come from a GGUF file (e.g. unsloth's
//! `Qwen-Image-Edit-2511-GGUF`) and stream from storage. Components load on first use, and the
//! 16 GB text encoder is dropped after encoding unless asked to stay.

pub mod sched;
pub mod transformer;
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
use transformer::{ImageShape, Transformer, TransformerConfig};
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
}

/// Whether `dir` is a Qwen-Image diffusers folder.
pub fn is_qwen_image(dir: &Path) -> bool {
    kind_of(dir).is_some()
}

fn kind_of(dir: &Path) -> Option<Kind> {
    let v: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("model_index.json")).ok()?).ok()?;
    match v["_class_name"].as_str()? {
        "QwenImagePipeline" | "QwenImageImg2ImgPipeline" => Some(Kind::TextToImage),
        "QwenImageEditPipeline" => Some(Kind::Edit),
        "QwenImageEditPlusPipeline" => Some(Kind::EditPlus),
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
    /// The transformer from elsewhere: a GGUF file or a diffusers `transformer/` folder.
    pub transformer: Option<PathBuf>,
    /// Stream the transformer's blocks from storage (weights mapped, read ahead per block).
    pub stream: Option<crate::stream::StreamOptions>,
    /// Keep the text encoder loaded between calls (otherwise dropped after each encoding).
    pub keep_text_encoder: bool,
    /// Accelerator choice, when the process has not initialized one yet.
    pub device: crate::platform::DevicePref,
}

impl Options {
    /// Sets an option by name (the C API's `key=value` options): `quant` and `text_quant` (bf16,
    /// q8, q4), `mmap`, `transformer` (a GGUF file or folder), `lora` (`FILE[:SCALE]`,
    /// repeatable), `keep_text_encoder`, `stream`.
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
            "lora" => self.loras.push(LoraSpec::parse(value)?),
            "keep_text_encoder" => self.keep_text_encoder = flag()?,
            "stream" => self.stream = flag()?.then(crate::stream::StreamOptions::default),
            "device" => {
                self.device = crate::platform::DevicePref::parse(value)
                    .with_context(|| format!("device: unknown device {value:?} (auto|cpu|cuda|metal|vulkan)"))?
            }
            _ => bail!("unknown option {key:?} (quant, text_quant, mmap, transformer, lora, keep_text_encoder, stream, device)"),
        }
        Ok(())
    }

    /// The steps of a Lightning LoRA among `loras`, from its file name (`...-4steps-...`).
    pub fn lightning_steps(&self) -> Option<usize> {
        self.loras.iter().find_map(|l| {
            let n = l.path.file_name()?.to_string_lossy().to_lowercase();
            if !n.contains("lightning") {
                return None;
            }
            let i = n.find("steps")?;
            let digits: String = n[..i].chars().rev().take_while(|c| c.is_ascii_digit()).collect();
            digits.chars().rev().collect::<String>().parse().ok()
        })
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
}

impl Default for Options {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            text_quant: Quant::Bf16,
            mmap: false,
            loras: Vec::new(),
            transformer: None,
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
        if !crate::gpu::initialized() {
            crate::gpu::init(&crate::platform::detect(), opts.device);
        }
        let kind = kind_of(&dir).with_context(|| format!("{}: not a Qwen-Image diffusers pipeline", dir.display()))?;
        let tdir = dir.join("transformer");
        let tcfg = if crate::fsx::exists(&tdir.join("config.json")) {
            TransformerConfig::load(&tdir)?
        } else {
            bail!("{}: no transformer/config.json", dir.display());
        };
        let sched = FlowMatch::load(&dir.join("scheduler").join("scheduler_config.json"))?;
        let vae = vae::Vae::load(&dir.join("vae"), kind != Kind::TextToImage)?;
        if vae.cfg.z_dim * 4 != tcfg.in_channels {
            bail!(
                "VAE latent channels {} do not match the transformer input {}",
                vae.cfg.z_dim,
                tcfg.in_channels
            );
        }
        Ok(Self {
            dir,
            kind,
            opts,
            sched,
            tcfg,
            text: None,
            tr: None,
            vae,
            lora_report: Vec::new(),
            cancel: Default::default(),
        })
    }

    /// The settings for `prompt` with this pipeline: Qwen's recommended ones ([`Params::new`]), or
    /// the Lightning ones when a Lightning LoRA is loaded (steps from its name).
    pub fn default_params(&self, prompt: &str) -> Params {
        match self.opts.lightning_steps() {
            Some(n) => Params::lightning(prompt, n),
            None => Params::new(prompt),
        }
    }

    /// [`Params`] from a [`Request`].
    pub fn params(&self, r: &Request) -> Params {
        let mut p = match (r.lightning, self.opts.lightning_steps()) {
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
        p
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
        if self.tr.is_some() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let src = self.opts.transformer.clone().unwrap_or_else(|| self.dir.join("transformer"));
        let in_memory = !(self.opts.mmap || self.opts.stream.is_some());
        let files = Box::new(ModelFiles::open(&src, in_memory).with_context(|| format!("transformer weights {}", src.display()))?);
        // SAFETY: the box's heap block does not move; `LoadedTransformer` drops the transformer
        // (every borrow of it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let prefix = ["", "model.diffusion_model.", "diffusion_model.", "transformer."]
            .into_iter()
            .find(|p| fs.has(&format!("{p}img_in.weight")))
            .with_context(|| format!("{}: no img_in.weight (not a Qwen-Image transformer)", src.display()))?
            .to_string();
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
    }

    /// The prompt template and the number of leading hidden states dropped.
    fn template(&self) -> (&'static str, usize) {
        match self.kind {
            Kind::TextToImage => (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                34,
            ),
            Kind::Edit => (
                "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n",
                64,
            ),
            Kind::EditPlus => (
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
        if self.kind == Kind::TextToImage {
            keep = keep.min(512);
        }
        Ok(h.last[drop * d..(drop + keep) * d].to_vec())
    }

    /// Text to image (any Qwen-Image folder; an edit model then generates without a reference).
    pub fn generate(&mut self, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.run(&[], p, progress)
    }

    /// Edits `images` (one for Qwen-Image-Edit; one or more for Edit-2509 / 2511) following the
    /// instruction in `p.prompt`.
    pub fn edit(&mut self, images: &[RgbImage], p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        if images.is_empty() {
            bail!("an edit needs at least one input image");
        }
        match self.kind {
            Kind::TextToImage => bail!("{}: a text-to-image model; use generate", self.dir.display()),
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

    fn run(&mut self, images: &[RgbImage], p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut tm = Timings::default();
        let sf = self.vae.scale();
        let mult = sf * 2;
        // Output size.
        let (cw, ch) = match (self.kind, images.last(), images.first()) {
            (Kind::EditPlus, Some(last), _) => calculate_dimensions(1024.0 * 1024.0, last.width as f64 / last.height as f64),
            (Kind::Edit, _, Some(first)) => calculate_dimensions(1024.0 * 1024.0, first.width as f64 / first.height as f64),
            _ => (1024, 1024),
        };
        let width = p.width.unwrap_or(cw) / mult * mult;
        let height = p.height.unwrap_or(ch) / mult * mult;
        if width < mult || height < mult {
            bail!("output size {width}x{height} is too small");
        }
        // Condition images (vision tower) and VAE images.
        let mut cond_rgb: Vec<(Vec<u8>, usize, usize)> = Vec::new();
        let mut vae_rgb: Vec<(Vec<u8>, usize, usize)> = Vec::new();
        for img in images {
            if img.data.len() != img.width * img.height * 3 {
                bail!("input image: {} bytes for {}x{} RGB", img.data.len(), img.width, img.height);
            }
            let ratio = img.width as f64 / img.height as f64;
            let resize = |(w, h): (usize, usize)| (crate::resample::resize_lanczos_rgb8(&img.data, img.width, img.height, w, h), w, h);
            match self.kind {
                Kind::EditPlus => {
                    cond_rgb.push(resize(calculate_dimensions(384.0 * 384.0, ratio)));
                    vae_rgb.push(resize(calculate_dimensions(1024.0 * 1024.0, ratio)));
                }
                _ => {
                    let r = resize((cw, ch));
                    cond_rgb.push(r.clone());
                    vae_rgb.push(r);
                }
            }
        }
        // Prompts. The text encoder's hidden states carry outlier channels (|x| ~ 200) that the
        // GPU's half-precision matrix units round visibly (1e-2 relative), so the encoding runs
        // with exact GPU matmuls; it is a small part of the work.
        tm.load_text = self.load_text_encoder()?;
        let t0 = Instant::now();
        let was_exact = crate::gpu::exact();
        crate::gpu::set_exact(true);
        let enc = self.encode_all(&cond_rgb, p);
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
        self.finish(&vae_rgb, p, pos, neg, width, height, tm, progress)
    }

    /// The prompt (and negative prompt) conditioning with the condition images.
    fn encode_all(&self, cond_rgb: &[(Vec<u8>, usize, usize)], p: &Params) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
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

    /// Reference latents, denoising and decoding.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &mut self,
        vae_rgb: &[(Vec<u8>, usize, usize)],
        p: &Params,
        pos: Vec<f32>,
        neg: Option<Vec<f32>>,
        width: usize,
        height: usize,
        mut tm: Timings,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<(Image, Timings)> {
        let mult = self.vae.scale() * 2;
        // Reference latents.
        let t1 = Instant::now();
        let z = self.vae.cfg.z_dim;
        let mut refs = Vec::new();
        for (i, (rgb, w, h)) in vae_rgb.iter().enumerate() {
            let (lat, lh, lw) = self.vae.encode(&to_unit(rgb), *h, *w)?;
            let tokens = pack(&lat, lh, lw, z);
            if i == 0 {
                dump("ref_latents0.bin", &tokens);
            }
            refs.push(RefImage {
                tokens,
                shape: ImageShape {
                    frames: 1,
                    rows: lh / 2,
                    cols: lw / 2,
                },
            });
        }
        tm.vae_encode = t1.elapsed().as_secs_f64();

        tm.load_transformer = self.load_transformer()?;
        let t2 = Instant::now();
        let tr = &self.tr.as_ref().unwrap().t;
        let (lh, lw) = (2 * (height / mult), 2 * (width / mult));
        let target = ImageShape {
            frames: 1,
            rows: lh / 2,
            cols: lw / 2,
        };
        let n0 = target.tokens();
        let c4 = self.tcfg.in_channels;
        // Initial noise, generated as [c, h, w] (or read from the dump folder), then packed.
        let noise_chw = match dump_dir().map(|d| d.join("noise.bin")).filter(|f| f.exists()) {
            Some(f) => {
                let v = read_f32(&f)?;
                if v.len() != z * lh * lw {
                    bail!("{} has {} values, expected {}", f.display(), v.len(), z * lh * lw);
                }
                v
            }
            None => crate::sd::ops::Rng::new(p.seed).normals(z * lh * lw),
        };
        let mut hwc = vec![0f32; noise_chw.len()];
        for c in 0..z {
            for px in 0..lh * lw {
                hwc[px * z + c] = noise_chw[c * lh * lw + px];
            }
        }
        let mut x = pack(&hwc, lh, lw, z);
        let mut shapes = vec![target];
        let mut ref_tokens = Vec::new();
        for r in &refs {
            shapes.push(r.shape);
            ref_tokens.extend_from_slice(&r.tokens);
        }
        drop(refs);
        let txt_pos = tr.embed_text(&pos, pos.len() / self.tcfg.joint_attention_dim);
        let txt_neg = neg.as_ref().map(|n| tr.embed_text(n, n.len() / self.tcfg.joint_attention_dim));
        let sched = p.scheduler.clone().unwrap_or_else(|| self.sched.clone());
        let sigmas = sched.sigmas(p.steps, n0)?;
        let steps = sigmas.len() - 1;
        // Attention logits run into the thousands: half-precision queries and keys (the matrix-unit
        // attention kernel) move the velocity by ~10%, so attention stays f32 (the matmuls do not).
        struct AttnGuard(bool);
        impl Drop for AttnGuard {
            fn drop(&mut self) {
                crate::gpu::set_exact_attention(self.0);
            }
        }
        let _attn = AttnGuard(crate::gpu::set_exact_attention(true));
        let mut input = vec![0f32; (n0 * c4) + ref_tokens.len()];
        input[n0 * c4..].copy_from_slice(&ref_tokens);
        for i in 0..steps {
            self.check_cancel()?;
            let (s, s_next) = (sigmas[i], sigmas[i + 1]);
            input[..n0 * c4].copy_from_slice(&x);
            let mut v = tr.forward(&input, &shapes, &txt_pos, s);
            tm.transformer_calls += 1;
            if let Some(tn) = &txt_neg {
                let vn = tr.forward(&input, &shapes, tn, s);
                tm.transformer_calls += 1;
                let g = p.true_cfg;
                crate::kernels::for_chunks(&mut v, c4, true, |t, row| {
                    let un = &vn[t * c4..(t + 1) * c4];
                    let cn = row.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
                    for (a, &b) in row.iter_mut().zip(un) {
                        *a = b + g * (*a - b);
                    }
                    let nn = row.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
                    let r = (cn / nn) as f32;
                    row.iter_mut().for_each(|a| *a *= r);
                });
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

        let t3 = Instant::now();
        let lat = unpack(&x, lh, lw, z);
        let (px, h, w) = self.vae.decode(&lat, lh, lw);
        dump("image.bin", &px);
        let rgb = px
            .iter()
            .map(|&v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        tm.decode = t3.elapsed().as_secs_f64();
        Ok((Image { width: w, height: h, rgb }, tm))
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
