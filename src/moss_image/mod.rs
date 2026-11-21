//! MOSS-Image text to image (OpenMOSS-Team/MOSS-Image-Base / -SFT and fine-tunes such as
//! `lodestones/Kromoss`), following the release's diffusers pipeline (`pipeline_moss_image.py`,
//! `MOSSImagePipeline`):
//!
//! - the prompt, trimmed, in Qwen3.5's chat template as a user turn with an open think block
//!   (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`), through Qwen3.5-9B
//!   (`text_encoder/`, a `Qwen3_5Model`: the language model alone; its final-normed hidden states
//!   condition the transformer). Kromoss' text encoder is `Qwen/Qwen3.5-9B` unchanged, so that
//!   checkpoint is used when the cache holds it and the repo's copy is not on disk;
//! - the ~5B single-stream DiT ([`transformer`]) predicts the flow velocity on 128-channel latents
//!   at 1/16 of the image size (the FLUX.2 VAE's 32 channels in 2 x 2 patches);
//! - classifier-free guidance (`uncond + g (cond - uncond)`) with the negative prompt (empty by
//!   default) run with blocks left out (block 22 by default, `uncond_skip_layers`), and optional
//!   skip-layer guidance on the positive branch (`slg_layers`, `slg_scale`, inside `slg_start..slg_stop`);
//! - Euler steps on the release's snapped schedule (shift 5), or ComfyUI's schedules and its
//!   `er_sde` sampler ([`sched`]);
//! - the latents de-normalized with the VAE's batch-norm statistics, unpatchified, and decoded by
//!   the FLUX.2 VAE's decoder (`AutoencoderKLFlux2`; Kromoss ships the small decoder, channels
//!   96 / 192 / 384 / 384) through [`crate::sd::vae::VaeDecoder`].
//!
//! LoRA files apply at load ([`lora`]: low-rank kohya / PEFT modules as a runtime side path, LoKr /
//! LoHa merged), including the release's 8-step DMD / DMD2 LoRAs (8 steps, guidance 1, no negative
//! branch). Presets ([`crate::advisor::PRESETS`]) bundle a base, LoRAs, a trigger word and the
//! sampling settings their authors recommend.

pub mod convrot;
pub mod lora;
pub mod sched;
pub mod transformer;

use crate::clock::Instant;
use crate::model::Quant;
use crate::safetensors::ModelFiles;
pub use crate::sd::Image;
use anyhow::{Context, Result, bail};
pub use sched::{Noise, Sampler, Schedule};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
pub use transformer::GpuPass;
use transformer::{Pass, StepMod, Transformer};

/// The release's default block left out of the negative branch.
pub const DEFAULT_UNCOND_SKIP: &[usize] = &[22];

/// Whether `dir` is a MOSS-Image diffusers folder (`model_index.json`'s `_class_name`, or a
/// `transformer/config.json` of class `MOSSImageTransformer`).
pub fn is_moss_image(dir: &Path) -> bool {
    let class = |p: &Path| {
        crate::fsx::read(p)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v["_class_name"].as_str().map(str::to_string))
    };
    class(&dir.join("model_index.json")).is_some_and(|c| c.starts_with("MOSSImage"))
        || class(&dir.join("transformer").join("config.json")).is_some_and(|c| c == "MOSSImageTransformer")
}

/// True for a GGUF file holding a MOSS-Image transformer (`general.architecture = moss_image`).
pub fn is_moss_gguf(path: &Path) -> bool {
    crate::gguf::is_gguf(path) && crate::gguf::header(path).is_ok_and(|h| h.arch() == "moss_image")
}

/// How to load a [`Pipeline`].
#[derive(Clone, Debug)]
pub struct Options {
    /// Precision of the transformer's block weights (a GGUF transformer keeps its own).
    pub quant: Quant,
    /// Precision of the Qwen3.5 text encoder.
    pub text_quant: Quant,
    /// The transformer from elsewhere: a diffusers folder, ComfyUI's single file, a GGUF file
    /// (`molbal/Kromoss-GGUF/kromoss-Q8_0.gguf`, `org/name:QUANT`).
    pub transformer: Option<String>,
    /// The Qwen3.5-9B text encoder from elsewhere (a folder or repo id).
    pub text_encoder: Option<String>,
    /// A diffusers `vae/` folder from elsewhere.
    pub vae: Option<String>,
    /// LoRA files (local paths or `org/name/file.safetensors`) and strengths.
    pub loras: Vec<lora::LoraSpec>,
    /// Merge low-rank LoRAs into the weights too (faster on a GPU; bf16 rounding of the update).
    pub lora_merge: bool,
    /// Keep the text encoder loaded between calls (otherwise mapped and dropped after encoding).
    pub keep_text_encoder: bool,
    /// Read the transformer into memory instead of mapping it.
    pub in_memory: bool,
    pub device: crate::platform::DevicePref,
    pub cache_dir: Option<PathBuf>,
    /// The preset applied ([`crate::advisor::PRESETS`]): its generation defaults.
    pub preset: Option<&'static crate::advisor::Preset>,
    /// How transformer passes use the device (whole passes on it, attention precision, the
    /// guidance branches batched).
    pub gpu: GpuPass,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            text_quant: Quant::Bf16,
            transformer: None,
            text_encoder: None,
            vae: None,
            loras: Vec::new(),
            lora_merge: false,
            keep_text_encoder: false,
            in_memory: false,
            device: crate::platform::DevicePref::Auto,
            cache_dir: None,
            preset: None,
            gpu: GpuPass::default(),
        }
    }
}

impl Options {
    /// Keys of [`Options::set`].
    pub const KEYS: &'static [&'static str] = &[
        "quant",
        "text_quant",
        "transformer",
        "text_encoder",
        "vae",
        "lora",
        "lora_merge",
        "keep_text_encoder",
        "in_memory",
        "device",
        "cache_dir",
        "preset",
        "gpu_pass",
        "exact_attention",
        "batch_guidance",
    ];

    /// Sets an option by name (the C API's and the server's `key=value` options): `quant`,
    /// `text_quant` (bf16, q8, q4), `transformer`, `text_encoder`, `vae`, `lora` (`FILE[:STRENGTH]`,
    /// repeatable), `lora_merge`, `keep_text_encoder`, `in_memory`, `device`, `cache_dir`, `preset`
    /// (a preset id: its LoRAs are added and its generation settings become the defaults),
    /// `gpu_pass` (whole transformer passes on the device, default on), `exact_attention` (their
    /// attention with float32 operands instead of half), `batch_guidance` (a step's guidance
    /// branches in one device batch, default on).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let flag = || match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(anyhow::anyhow!("{key}: expected a boolean, got {value:?}")),
        };
        let quant = || Quant::parse(value).with_context(|| format!("{key}: unknown quantization {value:?} (bf16, q8, q4)"));
        let some = || (!value.is_empty()).then(|| value.to_string());
        match key {
            "quant" => self.quant = quant()?,
            "text_quant" => self.text_quant = quant()?,
            "transformer" => self.transformer = some(),
            "text_encoder" => self.text_encoder = some(),
            "vae" => self.vae = some(),
            "lora" => self.loras.push(lora::LoraSpec::parse(value)?),
            "lora_merge" => self.lora_merge = flag()?,
            "keep_text_encoder" => self.keep_text_encoder = flag()?,
            "in_memory" | "preload" => self.in_memory = flag()?,
            "device" => {
                self.device = crate::platform::DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?}"))?
            }
            "cache_dir" => self.cache_dir = Some(PathBuf::from(value)),
            "preset" => self.apply_preset(value)?,
            "gpu_pass" => self.gpu.enabled = flag()?,
            "exact_attention" => self.gpu.exact_attention = flag()?,
            "batch_guidance" => self.gpu.batch = flag()?,
            _ => bail!("unknown MOSS-Image option {key:?} ({})", Self::KEYS.join(", ")),
        }
        Ok(())
    }

    /// Applies preset `id`: its load options (LoRAs) and, for [`Pipeline::params`], its
    /// generation defaults.
    pub fn apply_preset(&mut self, id: &str) -> Result<()> {
        let p = crate::advisor::preset(id).with_context(|| {
            format!(
                "unknown preset {id:?} (known: {})",
                crate::advisor::PRESETS.iter().map(|p| p.id).collect::<Vec<_>>().join(", ")
            )
        })?;
        if !is_moss_preset(p) {
            bail!("preset {id:?} is for {}, not a MOSS-Image model", p.base);
        }
        if self.preset.is_some_and(|q| q.id == p.id) {
            return Ok(());
        }
        for (k, v) in p.load {
            self.set(k, v)?;
        }
        self.preset = Some(p);
        Ok(())
    }
}

/// Whether a preset runs on this pipeline.
pub fn is_moss_preset(p: &crate::advisor::Preset) -> bool {
    p.pipeline == "moss-image"
}

/// One generation's settings (defaults: the release's pipeline, 1024 x 1024).
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub prompt: String,
    pub negative: String,
    pub steps: usize,
    /// Classifier-free guidance (1: no negative branch).
    pub guidance: f32,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    /// Blocks left out of the negative branch (zero-based; empty: plain CFG).
    pub uncond_skip_layers: Vec<usize>,
    /// Skip-layer guidance: blocks left out of an extra positive pass, and its scale (0: off).
    pub slg_layers: Vec<usize>,
    pub slg_scale: f32,
    /// Where (fractions `step / steps`) the skipping of both kinds applies.
    pub slg_start: f32,
    pub slg_stop: f32,
    /// The schedule's shift (default the scheduler config's, 5).
    pub shift: f32,
    pub schedule: Schedule,
    pub sampler: Sampler,
    /// A word put at the start of the prompt unless the prompt has it (a LoRA's trigger).
    pub trigger: String,
    /// The noise generator ([`Noise::Torch`]: PyTorch's CPU stream, the same initial noise as
    /// diffusers and ComfyUI for a seed).
    pub noise: Noise,
}

impl Params {
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            negative: String::new(),
            steps: 50,
            guidance: 3.0,
            seed: 42,
            width: 1024,
            height: 1024,
            uncond_skip_layers: DEFAULT_UNCOND_SKIP.to_vec(),
            slg_layers: Vec::new(),
            slg_scale: 0.0,
            slg_start: 0.0,
            slg_stop: 1.0,
            shift: 5.0,
            schedule: Schedule::Moss,
            sampler: Sampler::Euler,
            trigger: String::new(),
            noise: Noise::Oxinfer,
        }
    }

    /// Keys of [`Params::set`].
    pub const KEYS: &'static [&'static str] = &[
        "negative",
        "steps",
        "guidance",
        "seed",
        "width",
        "height",
        "size",
        "uncond_skip_layers",
        "slg_layers",
        "slg_scale",
        "slg_start",
        "slg_stop",
        "shift",
        "schedule",
        "scheduler",
        "sampler",
        "trigger",
        "noise",
    ];

    /// Sets a generation setting by name (preset defaults, `key=value` requests): see
    /// [`Params::KEYS`]. Layer lists are comma- or space-separated (`""` / `none`: empty).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let num = |v: &str| -> Result<f32> { v.trim().parse().with_context(|| format!("{key}: expected a number, got {v:?}")) };
        let int = |v: &str| -> Result<usize> { v.trim().parse().with_context(|| format!("{key}: expected an integer, got {v:?}")) };
        let list = |v: &str| -> Result<Vec<usize>> {
            if matches!(v.trim(), "" | "none") {
                return Ok(Vec::new());
            }
            v.split([',', ' ']).filter(|s| !s.is_empty()).map(int).collect()
        };
        match key {
            "negative" | "negative_prompt" => self.negative = value.to_string(),
            "steps" => self.steps = int(value)?,
            "guidance" | "cfg" | "guidance_scale" => self.guidance = num(value)?,
            "seed" => self.seed = value.trim().parse().with_context(|| format!("seed: {value:?}"))?,
            "width" => self.width = int(value)?,
            "height" => self.height = int(value)?,
            "size" => {
                let (w, h) = value
                    .split_once(['x', 'X'])
                    .with_context(|| format!("size {value:?}: expected WIDTHxHEIGHT"))?;
                (self.width, self.height) = (int(w)?, int(h)?);
            }
            "uncond_skip_layers" => self.uncond_skip_layers = list(value)?,
            "slg_layers" => self.slg_layers = list(value)?,
            "slg_scale" => self.slg_scale = num(value)?,
            "slg_start" => self.slg_start = num(value)?,
            "slg_stop" => self.slg_stop = num(value)?,
            "shift" => self.shift = num(value)?,
            "schedule" | "scheduler" => {
                self.schedule = Schedule::parse(value).with_context(|| format!("{key}: {value:?} (moss, simple, beta, beta57)"))?
            }
            "sampler" => self.sampler = Sampler::parse(value).with_context(|| format!("sampler: {value:?} (euler, er_sde)"))?,
            "trigger" => self.trigger = value.to_string(),
            "noise" => self.noise = Noise::parse(value).with_context(|| format!("noise: {value:?} (oxinfer, torch)"))?,
            _ => bail!("unknown MOSS-Image setting {key:?} ({})", Self::KEYS.join(", ")),
        }
        Ok(())
    }

    /// The prompt the text encoder sees: with the trigger word first when the prompt lacks it.
    pub fn full_prompt(&self) -> String {
        let t = self.trigger.trim();
        if t.is_empty() || self.prompt.to_lowercase().contains(&t.to_lowercase()) {
            self.prompt.clone()
        } else {
            format!("{t}, {}", self.prompt.trim())
        }
    }

    fn check(&self, layers: usize) -> Result<()> {
        if self.steps == 0 {
            bail!("steps must be at least 1");
        }
        if self.width == 0 || self.height == 0 || !self.width.is_multiple_of(16) || !self.height.is_multiple_of(16) {
            bail!(
                "width and height must be positive multiples of 16 (got {}x{})",
                self.width,
                self.height
            );
        }
        if !self.guidance.is_finite() || self.guidance < 0.0 || !self.slg_scale.is_finite() || self.slg_scale < 0.0 {
            bail!("guidance and slg_scale must be finite and nonnegative");
        }
        if !(0.0 <= self.slg_start && self.slg_start <= self.slg_stop && self.slg_stop <= 1.0) {
            bail!("need 0 <= slg_start <= slg_stop <= 1");
        }
        if let Some(l) = self.uncond_skip_layers.iter().chain(&self.slg_layers).find(|&&l| l >= layers) {
            bail!("skip layer {l} out of range (the transformer has {layers} blocks)");
        }
        Ok(())
    }
}

/// A generation request as JSON (the C API's `request_json`, the server's extra fields); every key
/// but `prompt` optional, defaults from the pipeline's preset, else [`Params::new`].
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub prompt: String,
    pub negative: Option<String>,
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub seed: Option<u64>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub uncond_skip_layers: Option<Vec<usize>>,
    pub slg_layers: Option<Vec<usize>>,
    pub slg_scale: Option<f32>,
    pub slg_start: Option<f32>,
    pub slg_stop: Option<f32>,
    pub shift: Option<f32>,
    pub schedule: Option<Schedule>,
    pub sampler: Option<Sampler>,
    pub trigger: Option<String>,
    pub noise: Option<Noise>,
}

impl Request {
    /// Over `base` (the pipeline's defaults).
    pub fn params_over(&self, mut p: Params) -> Params {
        p.prompt = self.prompt.clone();
        if let Some(v) = &self.negative {
            p.negative = v.clone();
        }
        p.steps = self.steps.unwrap_or(p.steps);
        p.guidance = self.guidance.unwrap_or(p.guidance);
        p.seed = self.seed.unwrap_or(p.seed);
        p.width = self.width.unwrap_or(p.width);
        p.height = self.height.unwrap_or(p.height);
        if let Some(v) = &self.uncond_skip_layers {
            p.uncond_skip_layers = v.clone();
        }
        if let Some(v) = &self.slg_layers {
            p.slg_layers = v.clone();
        }
        p.slg_scale = self.slg_scale.unwrap_or(p.slg_scale);
        p.slg_start = self.slg_start.unwrap_or(p.slg_start);
        p.slg_stop = self.slg_stop.unwrap_or(p.slg_stop);
        p.shift = self.shift.unwrap_or(p.shift);
        p.schedule = self.schedule.unwrap_or(p.schedule);
        p.sampler = self.sampler.unwrap_or(p.sampler);
        if let Some(t) = &self.trigger {
            p.trigger = t.clone();
        }
        p.noise = self.noise.unwrap_or(p.noise);
        p
    }
}

/// Timings of one call, in seconds.
#[derive(Default, Debug, Clone)]
pub struct Timings {
    pub load_text: f64,
    pub text: f64,
    pub denoise: f64,
    pub decode: f64,
    /// Transformer passes (one per guidance branch per step).
    pub transformer_calls: usize,
}

/// The loaded text encoder.
struct TextEncoder {
    // Drop order: the model (borrowing `_files`) first.
    model: crate::qwen::QwenModel<'static>,
    _files: Box<ModelFiles>,
}

/// The FLUX.2 VAE's decoder with its latent statistics.
pub struct Vae {
    pub decoder: crate::sd::vae::VaeDecoder,
    /// Per patchified channel (128): `sqrt(running_var + eps)` and `running_mean`.
    pub std: Vec<f32>,
    pub mean: Vec<f32>,
}

impl Vae {
    /// Loads a diffusers `AutoencoderKLFlux2` folder (decoder and batch-norm statistics).
    ///
    /// `path` may also be a single safetensors file: diffusers-named (`flux2-vae_small.safetensors`)
    /// or in Black Forest Labs' layout as ComfyUI ships the FLUX.2 VAE (`vae/flux2-vae.safetensors`:
    /// `decoder.up.N.block.M`, `decoder.mid.attn_1`, `decoder.post_quant_conv`, `bn.*`); the batch
    /// norm's epsilon is then read from a `config.json` beside it, else the FLUX.2 default 1e-4.
    pub fn load(path: &Path) -> Result<Self> {
        let (wp, cfg_path) = if path.is_file() {
            (path.to_path_buf(), path.with_file_name("config.json"))
        } else {
            let wp = ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors"]
                .iter()
                .map(|n| path.join(n))
                .find(|p| crate::fsx::exists(p))
                .with_context(|| format!("no VAE weights in {}", path.display()))?;
            (wp, path.join("config.json"))
        };
        let cfg: serde_json::Value = match crate::fsx::read(&cfg_path) {
            Ok(b) => serde_json::from_slice(&b)?,
            Err(_) if path.is_file() => serde_json::Value::Null,
            Err(e) => return Err(e).with_context(|| format!("{}", cfg_path.display())),
        };
        let eps = cfg["batch_norm_eps"].as_f64().unwrap_or(1e-4);
        let st = crate::safetensors::SafeTensors::open(&wp, false)?;
        let bfl = st.names().any(|n| n.starts_with("decoder.up.") && n.contains(".block."));
        let mut t = if bfl {
            crate::flux::vae::open_bfl(&wp)?.0
        } else {
            crate::sd::ops::Tensors::open(&wp)?
        };
        // The release decodes in float32: float32 weights stay exact (a bf16 pair each).
        t.set_split_f32(std::env::var_os("OXINFER_MOSS_VAE_BF16").is_none());
        let decoder = crate::sd::vae::VaeDecoder::from_tensors(&t, 1.0, 0.0)?;
        let mean = crate::flux::lora::tensor_f32(&st, "bn.running_mean").context("VAE: no bn.running_mean (not a FLUX.2 VAE)")?;
        let var = crate::flux::lora::tensor_f32(&st, "bn.running_var")?;
        let std = var.iter().map(|&v| (v as f64 + eps).sqrt() as f32).collect();
        Ok(Self { decoder, std, mean })
    }

    /// RGB in [-1, 1] (`[16 lh * 16 lw, 3]`) from transformer latents (`[lh * lw, 128]`).
    pub fn decode(&self, z: &[f32], lh: usize, lw: usize) -> Vec<f32> {
        let c4 = self.mean.len();
        let mut z = z.to_vec();
        for row in z.chunks_mut(c4) {
            for ((v, &s), &m) in row.iter_mut().zip(&self.std).zip(&self.mean) {
                *v = *v * s + m;
            }
        }
        let lat = crate::flux::vae::unpack(&z, 2 * lh, 2 * lw, c4 / 4);
        self.decoder.decode(&lat, 2 * lh, 2 * lw).0
    }
}

/// A MOSS-Image pipeline (see the module docs).
pub struct Pipeline {
    pub dir: PathBuf,
    pub opts: Options,
    text_dir: PathBuf,
    tok: crate::tokenizer::Tokenizer,
    text: Option<TextEncoder>,
    tr: Transformer,
    vae: Vae,
    /// The scheduler config's shift.
    shift: f32,
    cancel: Arc<AtomicBool>,
    pub load_secs: f64,
}

/// Dumps for reference checks: `OXINFER_MOSS_DUMP=DIR` writes the conditioning, the first
/// velocities and the final latents (f32 LE), and reads `noise.bin` (the initial noise,
/// `[128, lh, lw]` as the reference draws it) and `cond_text.bin` / `uncond_text.bin` (text
/// encoder outputs) from it when present.
fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("OXINFER_MOSS_DUMP").map(PathBuf::from)
}

fn dump(name: &str, v: &[f32]) {
    if let Some(d) = dump_dir() {
        let _ = std::fs::write(d.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    }
}

fn dumped(name: &str) -> Result<Option<Vec<f32>>> {
    match dump_dir().map(|d| d.join(name)).filter(|f| f.exists()) {
        Some(f) => {
            let b = std::fs::read(&f).with_context(|| format!("read {}", f.display()))?;
            Ok(Some(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()))
        }
        None => Ok(None),
    }
}

/// The repo id of Kromoss' (unchanged) text encoder.
pub const QWEN35_9B: &str = "Qwen/Qwen3.5-9B";

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

fn resolve_any(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    if spec.ends_with(".safetensors") || spec.ends_with(".gguf") {
        return crate::engine::resolve_file(spec, cache_dir);
    }
    crate::engine::resolve_model(spec, cache_dir)
}

/// Whether `dir` holds weights of a Qwen3.5 model.
fn has_weights(dir: &Path) -> bool {
    ["model.safetensors.index.json", "model.safetensors"]
        .iter()
        .any(|n| crate::fsx::exists(&dir.join(n)))
}

/// The cache folder of a repo, when it is there with its weights.
fn cached_repo(repo: &str, cache_dir: Option<&Path>) -> Option<PathBuf> {
    #[cfg(feature = "hf")]
    let root = cache_dir.map(Path::to_path_buf).unwrap_or_else(crate::hf::cache_root);
    #[cfg(not(feature = "hf"))]
    let root = cache_dir?.to_path_buf();
    let d = root.join(repo.replace('/', "--"));
    has_weights(&d).then_some(d)
}

impl Pipeline {
    /// Loads a pipeline: `model` is a MOSS-Image diffusers folder or repo id, or a preset id (see
    /// [`crate::advisor::PRESETS`]). The transformer and the VAE load now, the text encoder on
    /// first use.
    pub fn load(model: &str, mut opts: Options) -> Result<Self> {
        let t0 = Instant::now();
        let mut model = model.to_string();
        if !Path::new(&model).exists()
            && let Some(p) = crate::advisor::preset(&model)
        {
            opts.apply_preset(p.id)?;
            model = p.base.to_string();
        }
        let cache = opts.cache_dir.clone();
        let cache = cache.as_deref();
        // The text encoder: given, the repo's own, or Qwen/Qwen3.5-9B from the cache.
        let mut skip: Vec<&str> = Vec::new();
        let mut text_spec = opts.text_encoder.clone();
        if text_spec.is_none()
            && !Path::new(&model).exists()
            && !has_weights(&local_cache(&model, cache).join("text_encoder"))
            && let Some(d) = cached_repo(QWEN35_9B, cache)
        {
            text_spec = Some(d.to_string_lossy().into_owned());
        }
        if text_spec.is_some() {
            skip.push("text_encoder");
        }
        if opts.transformer.is_some() {
            skip.push("transformer");
        }
        if opts.vae.is_some() {
            skip.push("vae");
        }
        let dir = resolve_dir(&model, &skip, cache)?;
        if !is_moss_image(&dir) {
            bail!("{}: not a MOSS-Image diffusers folder", dir.display());
        }
        crate::gpu::ensure(opts.device)?;
        let text_dir = match &text_spec {
            Some(s) => resolve_any(s, cache)?,
            None if has_weights(&dir.join("text_encoder")) => dir.join("text_encoder"),
            None => resolve_any(QWEN35_9B, cache)?,
        };
        let tok_path = [dir.join("tokenizer").join("tokenizer.json"), text_dir.join("tokenizer.json")]
            .into_iter()
            .find(|p| crate::fsx::exists(p))
            .context("no tokenizer.json (tokenizer/ or the text encoder's folder)")?;
        let tok = crate::tokenizer::Tokenizer::from_bytes(crate::fsx::read(&tok_path)?).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let tr_path = match &opts.transformer {
            Some(s) => resolve_any(s, cache)?,
            None => dir.join("transformer"),
        };
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
        let set = if loras.is_empty() {
            lora::LoraSet::default()
        } else {
            let c = Transformer::peek_config(&tr_path)?;
            lora::LoraSet::load(&loras, c.dim, c.layers, opts.lora_merge)?
        };
        let tr =
            Transformer::load(&tr_path, opts.quant, &set, opts.in_memory).with_context(|| format!("transformer {}", tr_path.display()))?;
        let vae_dir = match &opts.vae {
            Some(s) => resolve_any(s, cache)?,
            None => dir.join("vae"),
        };
        let vae = Vae::load(&vae_dir).with_context(|| format!("VAE {}", vae_dir.display()))?;
        if vae.mean.len() != tr.cfg.latent_channels {
            bail!(
                "VAE latent statistics ({}) do not match the transformer's {} channels",
                vae.mean.len(),
                tr.cfg.latent_channels
            );
        }
        let shift = crate::fsx::read(&dir.join("scheduler").join("scheduler_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v["shift"].as_f64())
            .unwrap_or(5.0) as f32;
        Ok(Self {
            dir,
            opts,
            text_dir,
            tok,
            text: None,
            tr,
            vae,
            shift,
            cancel: Default::default(),
            load_secs: t0.elapsed().as_secs_f64(),
        })
    }

    /// The transformer.
    pub fn transformer(&self) -> &Transformer {
        &self.tr
    }

    /// The VAE.
    pub fn vae(&self) -> &Vae {
        &self.vae
    }

    /// Where the text encoder loads from.
    pub fn text_encoder_dir(&self) -> &Path {
        &self.text_dir
    }

    /// The LoRA report lines.
    pub fn lora_report(&self) -> &[String] {
        &self.tr.lora_report
    }

    /// A flag that stops the running (or next) call between denoising steps (any thread).
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// The default settings for `prompt`: the release's, with the scheduler's shift, then the
    /// preset's.
    pub fn params(&self, prompt: &str) -> Params {
        let mut p = Params::new(prompt);
        p.shift = self.shift;
        if let Some(pr) = self.opts.preset {
            for (k, v) in pr.defaults {
                // Preset tables are checked by the tests; a bad entry is skipped.
                let _ = p.set(k, v);
            }
            if !pr.trigger.is_empty() {
                p.trigger = pr.trigger.to_string();
            }
        }
        p
    }

    /// The parameters of a JSON request over this pipeline's defaults.
    pub fn request_params(&self, r: &Request) -> Params {
        r.params_over(self.params(&r.prompt))
    }

    /// Loads the text encoder (if not loaded); returns the seconds taken.
    pub fn load_text_encoder(&mut self) -> Result<f64> {
        if self.text.is_some() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let files = Box::new(ModelFiles::open(&self.text_dir, false).with_context(|| format!("text encoder {}", self.text_dir.display()))?);
        let cfg = crate::config::QwenConfig::load(&self.text_dir)?;
        if cfg.hidden_size != self.tr.cfg.text_dim {
            bail!(
                "text encoder width {} does not match the transformer's {}",
                cfg.hidden_size,
                self.tr.cfg.text_dim
            );
        }
        // SAFETY: the box's heap block does not move; `TextEncoder` drops the model first.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let model = crate::qwen::QwenModel::load(cfg, fs, self.opts.text_quant)?;
        self.text = Some(TextEncoder { model, _files: files });
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Drops the text encoder.
    pub fn unload_text_encoder(&mut self) {
        self.text = None;
    }

    /// The token ids of `prompt` in the chat template (truncated to 4096).
    pub fn prompt_ids(&self, prompt: &str) -> Result<Vec<u32>> {
        let text = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", prompt.trim());
        let mut ids = self
            .tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
            .ids
            .clone();
        ids.truncate(4096);
        Ok(ids)
    }

    /// The conditioning of `prompt`: Qwen3.5's final hidden states `[t, 4096]` (the text encoder
    /// must be loaded).
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        let te = self.text.as_ref().context("text encoder not loaded")?;
        let ids = self.prompt_ids(prompt)?;
        let mut sess = te.model.new_session();
        Ok(te.model.forward_hidden_mm(&mut sess, &ids, &[]))
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.swap(false, Ordering::Relaxed) {
            bail!("cancelled");
        }
        Ok(())
    }

    /// Text to image; `progress(step, steps)` after each step.
    pub fn generate(&mut self, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.cancel.store(false, Ordering::Relaxed);
        p.check(self.tr.cfg.layers)?;
        let mut tm = Timings::default();
        let cfg_on = p.guidance != 1.0;
        let prompt = p.full_prompt();
        // Conditioning (or the reference's, from the dump folder).
        let (mut cond, mut unc) = (dumped("cond_text.bin")?, if cfg_on { dumped("uncond_text.bin")? } else { None });
        if cond.is_none() || (cfg_on && unc.is_none()) {
            tm.load_text = self.load_text_encoder()?;
            let t0 = Instant::now();
            if cond.is_none() {
                cond = Some(self.encode_prompt(&prompt)?);
            }
            if cfg_on && unc.is_none() {
                unc = Some(self.encode_prompt(&p.negative)?);
            }
            tm.text = t0.elapsed().as_secs_f64();
            if !self.opts.keep_text_encoder {
                self.unload_text_encoder();
            }
        }
        let cond = cond.unwrap();
        dump("our_cond_text.bin", &cond);
        let cond = self.tr.embed_text(&cond)?;
        let unc = unc.map(|u| self.tr.embed_text(&u)).transpose()?;
        self.check_cancel()?;
        let (lh, lw) = (p.height / 16, p.width / 16);
        let c = self.tr.cfg.latent_channels;
        let t0 = Instant::now();
        let mut rng = crate::sd::ops::Rng::new(p.seed);
        // PyTorch's streams: the initial noise from the seed, ER-SDE's from seed + 1.
        let mut torch_step = (p.noise == Noise::Torch).then(|| crate::torch_rng::TorchGenerator::new(p.seed.wrapping_add(1)));
        let noise = match dumped("noise.bin")? {
            Some(n) if n.len() == c * lh * lw => n,
            Some(n) => bail!("noise.bin has {} values, expected {}", n.len(), c * lh * lw),
            None if p.noise == Noise::Torch => crate::torch_rng::TorchGenerator::new(p.seed).randn(c * lh * lw),
            None => rng.normals(c * lh * lw),
        };
        // [c, lh, lw] (the reference's layout) to tokens [lh * lw, c].
        let mut x = crate::flux::vae::chw_to_hwc(&noise, c, lh, lw);
        let sigmas = match p.schedule {
            Schedule::Moss => sched::moss_sigmas(p.steps, p.shift)?,
            s => sched::comfy_sigmas(s, p.steps, p.shift)?,
        };
        let mut run = sched::Run::new(p.sampler, sigmas, p.shift);
        let steps = run.steps();
        for i in 0..steps {
            self.check_cancel()?;
            let sigma = run.sigma(i);
            let m: StepMod = self.tr.step_mod(sigma * 1000.0);
            let active = (p.slg_start..p.slg_stop).contains(&(i as f32 / steps as f32));
            // The branches of the step, in one batch where the device takes them.
            let mut passes = vec![Pass { text: &cond, skip: &[] }];
            if let Some(u) = &unc {
                let skip: &[usize] = if active { &p.uncond_skip_layers } else { &[] };
                passes.push(Pass { text: u, skip });
            }
            let slg = !p.slg_layers.is_empty() && p.slg_scale > 0.0 && active;
            if slg {
                passes.push(Pass {
                    text: &cond,
                    skip: &p.slg_layers,
                });
            }
            let mut vs = self.tr.forward_batch(&x, lh, lw, &m, &passes, self.opts.gpu);
            tm.transformer_calls += passes.len();
            let vslg = if slg { vs.pop() } else { None };
            let vu = if unc.is_some() { vs.pop() } else { None };
            let vc = vs.pop().unwrap();
            let mut v = vc.clone();
            if let Some(vu) = &vu {
                for ((o, &a), &b) in v.iter_mut().zip(&vc).zip(vu) {
                    *o = b + p.guidance * (a - b);
                }
                if i == 0 {
                    dump("our_v0_uncond.bin", vu);
                }
            }
            if let Some(vs) = &vslg {
                if !cfg_on {
                    v.clone_from(vs);
                }
                for ((o, &a), &b) in v.iter_mut().zip(&vc).zip(vs) {
                    *o += p.slg_scale * (a - b);
                }
            }
            if i == 0 {
                dump("our_v0_cond.bin", &vc);
            }
            run.step(i, &mut x, &v, &mut |n| match &mut torch_step {
                // Drawn in the latent's [128, h, w] order, used as tokens.
                Some(g) => crate::flux::vae::chw_to_hwc(&g.randn(n), c, lh, lw),
                None => rng.normals(n),
            });
            progress(i + 1, steps);
        }
        tm.denoise = t0.elapsed().as_secs_f64();
        dump("our_latent.bin", &crate::flux::vae::hwc_to_chw(&x, c, lh, lw));
        let t0 = Instant::now();
        let img = self.vae.decode(&x, lh, lw);
        tm.decode = t0.elapsed().as_secs_f64();
        let rgb = crate::flux::vae::unit_to_rgb8(&img);
        Ok((
            Image {
                width: p.width,
                height: p.height,
                rgb,
            },
            tm,
        ))
    }
}

/// The cache folder `model` (a repo id) would use.
fn local_cache(model: &str, cache_dir: Option<&Path>) -> PathBuf {
    #[cfg(feature = "hf")]
    let root = cache_dir.map(Path::to_path_buf).unwrap_or_else(crate::hf::cache_root);
    #[cfg(not(feature = "hf"))]
    let root = cache_dir.map(Path::to_path_buf).unwrap_or_default();
    root.join(model.replace('/', "--"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_and_requests() {
        let mut p = Params::new("a cat");
        p.set("uncond_skip_layers", "none").unwrap();
        assert!(p.uncond_skip_layers.is_empty());
        p.set("slg_layers", "7, 9").unwrap();
        assert_eq!(p.slg_layers, vec![7, 9]);
        p.set("size", "768x1344").unwrap();
        assert_eq!((p.width, p.height), (768, 1344));
        p.set("sampler", "er_sde").unwrap();
        p.set("scheduler", "beta57").unwrap();
        assert_eq!((p.sampler, p.schedule), (Sampler::ErSde, Schedule::Beta57));
        assert!(p.set("bogus", "1").is_err());
        p.trigger = "famegrid".into();
        assert_eq!(p.full_prompt(), "famegrid, a cat");
        p.prompt = "FameGrid photo".into();
        assert_eq!(p.full_prompt(), "FameGrid photo");
        let r: Request =
            serde_json::from_str(r#"{"prompt":"x","steps":8,"sampler":"er_sde","schedule":"beta57","uncond_skip_layers":[]}"#).unwrap();
        let q = r.params_over(Params::new(""));
        assert_eq!((q.steps, q.sampler, q.schedule), (8, Sampler::ErSde, Schedule::Beta57));
        assert!(q.uncond_skip_layers.is_empty());
        assert!(serde_json::from_str::<Request>(r#"{"prompt":"x","bogus":1}"#).is_err());
    }

    #[test]
    fn presets_parse() {
        for pr in crate::advisor::PRESETS.iter().filter(|p| is_moss_preset(p)) {
            let mut p = Params::new("x");
            for (k, v) in pr.defaults {
                p.set(k, v).unwrap_or_else(|e| panic!("preset {}: {e}", pr.id));
            }
            let mut o = Options::default();
            for (k, v) in pr.load {
                o.set(k, v).unwrap_or_else(|e| panic!("preset {}: {e}", pr.id));
            }
        }
    }
}
