//! OmniGen2: instruction-based image editing, text to image and in-context generation (several
//! input images) from one model, following the official `OmniGen2Pipeline`
//! (`omnigen2/pipelines/omnigen2/pipeline_omnigen2.py`, github.com/VectorSpaceLab/OmniGen2):
//!
//! - the instruction, in the pipeline's chat template (a system turn, then the user turn), runs
//!   through Qwen2.5-VL-3B (`mllm/`, through [`crate::qwen_vl`]) as text only; its final hidden
//!   states condition the generator (the input images are not shown to it);
//! - the input images are resized (at most `max_pixels` pixels and `max_side` on the longer side,
//!   never upscaled, both sides floored to multiples of 16; Lanczos), VAE-encoded (the FLUX
//!   autoencoder, [`crate::flux::vae`]) and given to the transformer as their own token streams;
//! - the ~4B `OmniGen2Transformer2DModel` ([`transformer`]) predicts the velocity, integrated
//!   with Euler steps over `t` from 0 (noise) to 1 (image) on the schedule
//!   `t / (m - m t + t)`, `m = sqrt(latent pixels) / 40`;
//! - dual classifier-free guidance: `uncond + image_scale * (ref - uncond) + text_scale * (cond -
//!   ref)`, where `ref` sees the images with the negative prompt and `uncond` neither, inside
//!   `cfg_range` (fractions of the steps); text-only guidance when `image_scale <= 1`.
//!
//! Options that change the result, off by default: the official `--scheduler dpmsolver++`
//! ([`sched`]), the TeaCache and TaylorSeer step caches ([`crate::step_cache`], one state per
//! guidance branch as in the official pipeline), and sampling the VAE posterior of the input
//! images ([`crate::vae_posterior`]; the default is its mean).
//!
//! With one input image the output takes its (resized) size; otherwise the requested size (1024 x
//! 1024 by default) scaled into `max_pixels`, the decoded image then resized back to the request.
//! The understanding side of OmniGen2 is Qwen2.5-VL chat: load the `mllm/` folder with
//! [`crate::engine::Engine`] (the tokenizer and image processor come from `processor/`).

pub mod sched;
pub mod transformer;
pub mod weights;

use crate::clock::Instant;
use crate::model::Quant;
use crate::qwen_vl::{QwenVl, QwenVlOptions};
use crate::safetensors::ModelFiles;
pub use crate::sd::Image;
use crate::step_cache::{Calc, OMNIGEN2_TEACACHE_POLY, StepCache, TaylorSeer, TaylorSeerConfig, TeaCache};
use crate::video::RgbImage;
use anyhow::{Context, Result, bail};
pub use sched::Scheduler;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use transformer::{Grid, StepCond, Transformer, TransformerConfig};

/// The official default negative prompt (`inference.py`, `app.py`).
pub const DEFAULT_NEGATIVE: &str = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar";

/// The system turn of the pipeline's chat template.
pub const SYSTEM_PROMPT: &str = "You are a helpful assistant that generates high-quality images based on user instructions.";

/// Whether `dir` is an OmniGen2 diffusers folder (`model_index.json`'s `_class_name`).
pub fn is_omnigen2(dir: &Path) -> bool {
    crate::fsx::read(&dir.join("model_index.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["_class_name"].as_str().is_some_and(|c| c.starts_with("OmniGen2")))
}

/// How to load a [`Pipeline`].
#[derive(Clone, Debug)]
pub struct Options {
    /// Precision of the transformer's matmul weights (a GGUF transformer keeps its own).
    pub quant: Quant,
    /// Precision of the Qwen2.5-VL language model.
    pub text_quant: Quant,
    /// Map weights from disk instead of reading them into memory (float32 checkpoints, converted
    /// at load, are always mapped).
    pub mmap: bool,
    /// The transformer from elsewhere: a GGUF file or a diffusers `transformer/` folder.
    pub transformer: Option<PathBuf>,
    /// The Qwen2.5-VL-3B weights from elsewhere (a folder or repo id): OmniGen2's `mllm/` is
    /// `Qwen/Qwen2.5-VL-3B-Instruct` stored as float32, whose bf16 original is half the size.
    pub text_encoder: Option<String>,
    /// Keep the text encoder loaded between calls (otherwise dropped after each encoding).
    pub keep_text_encoder: bool,
    /// Accelerator choice, when the process has not initialized one yet.
    pub device: crate::platform::DevicePref,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            text_quant: Quant::Bf16,
            mmap: false,
            transformer: None,
            text_encoder: None,
            keep_text_encoder: false,
            device: crate::platform::DevicePref::Auto,
        }
    }
}

impl Options {
    /// Sets an option by name (the C API's `key=value` options): `quant`, `text_quant` (bf16, q8,
    /// q4), `mmap`, `transformer`, `text_encoder`, `keep_text_encoder`, `device`.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let flag = || match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(anyhow::anyhow!("{key}: expected a boolean, got {value:?}")),
        };
        let quant = || Quant::parse(value).with_context(|| format!("{key}: unknown quantization {value:?} (bf16, q8, q4)"));
        match key {
            "quant" => self.quant = quant()?,
            "text_quant" => self.text_quant = quant()?,
            "mmap" => self.mmap = flag()?,
            "transformer" => self.transformer = (!value.is_empty()).then(|| PathBuf::from(value)),
            "text_encoder" => self.text_encoder = (!value.is_empty()).then(|| value.to_string()),
            "keep_text_encoder" => self.keep_text_encoder = flag()?,
            "device" => {
                self.device = crate::platform::DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?}"))?
            }
            _ => bail!("unknown option {key:?} (quant, text_quant, mmap, transformer, text_encoder, keep_text_encoder, device)"),
        }
        Ok(())
    }
}

/// One generation's settings (defaults: the official `inference.py`).
#[derive(Clone, Debug)]
pub struct Params {
    pub prompt: String,
    pub negative: String,
    pub steps: usize,
    pub text_guidance: f32,
    /// Image guidance (input images only; 1 disables it).
    pub image_guidance: f32,
    /// Steps (as fractions `i / steps`) where guidance applies.
    pub cfg_range: (f32, f32),
    pub seed: u64,
    /// Output size. None: the input image's (resized) size for one input image, else 1024 x 1024.
    /// With one input image and a size, the image is resized to that size (so its token grid
    /// matches the output's).
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Pixel budget of each input image and of the generated latents.
    pub max_pixels: usize,
    /// Longest side of each input image.
    pub max_side: usize,
    /// The sampler (default Euler; `DpmSolver` is the official `--scheduler dpmsolver++`).
    pub scheduler: Scheduler,
    /// A step cache (default none: every step runs the transformer). TeaCache and TaylorSeer
    /// skip the joint blocks on some steps and so change the image.
    pub cache: StepCache,
    /// Sample the VAE posterior of the input images (`mean + std * noise`, as the official
    /// pipeline) instead of taking its mean. The noise comes from
    /// [`crate::vae_posterior::noise`], seeded by `seed`.
    pub sample_posterior: bool,
}

impl Params {
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            negative: DEFAULT_NEGATIVE.to_string(),
            steps: 50,
            text_guidance: 5.0,
            image_guidance: 2.0,
            cfg_range: (0.0, 1.0),
            seed: 0,
            width: None,
            height: None,
            max_pixels: 1024 * 1024,
            max_side: 1024,
            scheduler: Scheduler::Euler,
            cache: StepCache::None,
            sample_posterior: false,
        }
    }

    /// The official TeaCache (`--enable_teacache`) with this relative-L1 threshold (the
    /// official default: 0.05).
    pub fn with_teacache(mut self, rel_l1_thresh: f32) -> Self {
        self.cache = StepCache::TeaCache { rel_l1_thresh };
        self
    }

    /// The official TaylorSeer (`--enable_taylorseer`: order 4, the blocks every 3rd step after
    /// the first 5).
    pub fn with_taylorseer(mut self) -> Self {
        self.cache = StepCache::TaylorSeer(TaylorSeerConfig::OMNIGEN2);
        self
    }
}

/// A generation request as JSON (the C API's `request_json`; every key but `prompt` optional).
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub prompt: String,
    pub negative: Option<String>,
    pub steps: Option<usize>,
    pub text_guidance: Option<f32>,
    pub image_guidance: Option<f32>,
    pub cfg_range: Option<(f32, f32)>,
    pub seed: Option<u64>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub max_pixels: Option<usize>,
    pub max_side: Option<usize>,
    /// `"euler"` (default) or `"dpmsolver++"`.
    pub scheduler: Option<Scheduler>,
    /// TeaCache with this relative-L1 threshold (e.g. 0.05).
    pub teacache: Option<f32>,
    /// TaylorSeer (it takes precedence over `teacache`, as in the official script).
    pub taylorseer: Option<bool>,
    /// Sample the VAE posterior of the input images instead of taking its mean.
    pub sample_posterior: Option<bool>,
    /// Input image files (decoded with the `media` feature).
    pub images: Vec<String>,
}

impl Request {
    pub fn params(&self) -> Params {
        let mut p = Params::new(&self.prompt);
        if let Some(n) = &self.negative {
            p.negative = n.clone();
        }
        p.steps = self.steps.unwrap_or(p.steps);
        p.text_guidance = self.text_guidance.unwrap_or(p.text_guidance);
        p.image_guidance = self.image_guidance.unwrap_or(p.image_guidance);
        p.cfg_range = self.cfg_range.unwrap_or(p.cfg_range);
        p.seed = self.seed.unwrap_or(p.seed);
        p.width = self.width;
        p.height = self.height;
        p.max_pixels = self.max_pixels.unwrap_or(p.max_pixels);
        p.max_side = self.max_side.unwrap_or(p.max_side);
        p.scheduler = self.scheduler.unwrap_or(p.scheduler);
        if let Some(t) = self.teacache {
            p = p.with_teacache(t);
        }
        if self.taylorseer == Some(true) {
            p = p.with_taylorseer();
        }
        p.sample_posterior = self.sample_posterior.unwrap_or(false);
        p
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
    /// Joint-block passes (one per guidance branch per step).
    pub transformer_calls: usize,
    /// Passes a step cache answered without running the joint blocks.
    pub cached_passes: usize,
    /// With `sample_posterior`: per input image, the posterior's standard deviations relative
    /// to its means (L2 over the latent).
    pub posterior_std: Vec<f64>,
}

struct LoadedTransformer {
    // Drop order: the transformer (borrowing `_files`) first.
    t: Transformer<'static>,
    _files: Box<ModelFiles>,
}

/// An OmniGen2 pipeline (see the module docs).
pub struct Pipeline {
    pub dir: PathBuf,
    pub opts: Options,
    pub tcfg: TransformerConfig,
    text: Option<QwenVl>,
    tr: Option<LoadedTransformer>,
    vae: crate::flux::vae::Vae,
    cancel: Arc<AtomicBool>,
}

/// `OmniGen2ImageProcessor.get_new_height_width`: `(w, h)` scaled into `max_pixels` and
/// `max_side` (never up), each floored to a multiple of 16.
pub fn input_size(w: usize, h: usize, max_pixels: usize, max_side: usize) -> (usize, usize) {
    let side = max_side as f64 / w.max(h) as f64;
    let px = (max_pixels as f64 / (w * h) as f64).sqrt();
    let r = side.min(px).min(1.0);
    (((w as f64 * r) as usize) / 16 * 16, ((h as f64 * r) as usize) / 16 * 16)
}

/// The pipeline's output latent size for a requested `(w, h)`: scaled into `max_pixels` (never
/// up), floored to multiples of 16.
pub fn output_size(w: usize, h: usize, max_pixels: usize) -> (usize, usize) {
    let r = (max_pixels as f64 / (w * h) as f64).sqrt().min(1.0);
    (((w as f64 * r) as usize) / 16 * 16, ((h as f64 * r) as usize) / 16 * 16)
}

/// The Euler schedule: `steps` values of `t` from 0, shifted by `m = sqrt(tokens) / 40`, then 1.
pub fn timesteps(steps: usize, latent_pixels: usize) -> Vec<f32> {
    // numpy: a float32 linspace against the float64 `m` computes in float64.
    let m = (latent_pixels as f64).sqrt() / 40.0;
    let mut t: Vec<f32> = (0..steps)
        .map(|i| {
            let t = (i as f64 / steps as f64) as f32 as f64;
            (t / (m - m * t + t)) as f32
        })
        .collect();
    t.push(1.0);
    t
}

/// Bilinear resize of `[h * w, c]` floats (`F.interpolate(mode="bilinear")`, half-pixel
/// centres, no antialiasing).
pub fn resize_bilinear(x: &[f32], w: usize, h: usize, c: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0f32; tw * th * c];
    let (sx, sy) = (w as f32 / tw as f32, h as f32 / th as f32);
    let coord = |o: usize, s: f32, n: usize| {
        let f = ((o as f32 + 0.5) * s - 0.5).max(0.0);
        let i0 = (f as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        (i0, i1, f - i0 as f32)
    };
    crate::kernels::for_chunks(&mut out, tw * c, th > 16, |y, row| {
        let (y0, y1, fy) = coord(y, sy, h);
        for xo in 0..tw {
            let (x0, x1, fx) = coord(xo, sx, w);
            for ch in 0..c {
                let p = |yy: usize, xx: usize| x[(yy * w + xx) * c + ch];
                let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                row[xo * c + ch] = top * (1.0 - fy) + bot * fy;
            }
        }
    });
    out
}

fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("OXINFER_OG2_DUMP").map(PathBuf::from)
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

/// A file from the dump folder, when present (validation: the reference's tensors instead of ours).
fn dumped(name: &str) -> Result<Option<Vec<f32>>> {
    match dump_dir().map(|d| d.join(name)).filter(|f| f.exists()) {
        Some(f) => read_f32(&f).map(Some),
        None => Ok(None),
    }
}

/// An input image ready for the transformer: its patch tokens and grid.
struct RefLatent {
    tokens: Vec<f32>,
    grid: Grid,
}

/// The step-cache state of one guidance branch.
enum BranchCache {
    None,
    Tea(TeaCache),
    Taylor(TaylorSeer),
}

impl BranchCache {
    fn new(c: &StepCache) -> Self {
        match *c {
            StepCache::None => Self::None,
            StepCache::TeaCache { rel_l1_thresh } => Self::Tea(TeaCache::new(rel_l1_thresh, &OMNIGEN2_TEACACHE_POLY)),
            StepCache::TaylorSeer(cfg) => Self::Taylor(TaylorSeer::new(cfg)),
        }
    }
}

/// The image streams of one step, embedded when a pass first needs them (a step whose passes
/// are all TaylorSeer predictions needs none).
struct StepStreams<'a> {
    tr: &'a Transformer<'static>,
    x: &'a [f32],
    tg: Grid,
    refs: &'a [RefLatent],
    sc: &'a StepCond,
    noise: Option<Vec<f32>>,
    ref_embs: Option<Vec<Vec<f32>>>,
}

impl StepStreams<'_> {
    fn ensure(&mut self, with_refs: bool) -> Result<()> {
        if self.noise.is_none() {
            self.noise = Some(self.tr.embed_noise(self.x, self.tg, self.sc)?);
        }
        if with_refs && self.ref_embs.is_none() {
            let e = self
                .refs
                .iter()
                .enumerate()
                .map(|(j, r)| self.tr.embed_ref(&r.tokens, r.grid, j, self.sc))
                .collect::<Result<_>>()?;
            self.ref_embs = Some(e);
        }
        Ok(())
    }

    /// One guidance branch's velocity: `text` with or without the input images, through
    /// `cache`. Returns whether the joint blocks ran.
    fn pass(&mut self, cache: &mut BranchCache, text: &[f32], with_refs: bool, first_or_last: bool) -> Result<(Vec<f32>, bool)> {
        let (tr, tg, sc) = (self.tr, self.tg, self.sc);
        if let BranchCache::Taylor(ts) = cache {
            let calc = ts.begin();
            let v = match calc {
                Calc::Full => {
                    self.ensure(with_refs)?;
                    let h = tr.joint_hidden(text, &self.ref_list(with_refs), self.noise.as_ref().unwrap(), tg, sc)?;
                    let v = tr.output(&h, tg.tokens(), sc);
                    ts.update(h);
                    v
                }
                Calc::Taylor => tr.output(&ts.predict(), tg.tokens(), sc),
            };
            ts.end();
            return Ok((v, calc == Calc::Full));
        }
        self.ensure(with_refs)?;
        let refs = self.ref_list(with_refs);
        let noise = self.noise.as_ref().unwrap();
        match cache {
            BranchCache::Tea(tc) => tr.joint_teacache(text, &refs, noise, tg, sc, tc, first_or_last),
            _ => Ok((tr.joint(text, &refs, noise, tg, sc)?, true)),
        }
    }

    fn ref_list(&self, with_refs: bool) -> Vec<(&[f32], Grid)> {
        match (&self.ref_embs, with_refs) {
            (Some(e), true) => e.iter().zip(self.refs).map(|(e, r)| (&e[..], r.grid)).collect(),
            _ => Vec::new(),
        }
    }
}

impl Pipeline {
    /// Opens an OmniGen2 diffusers folder (or a Hugging Face repo id, fetched into the cache):
    /// configs and the VAE now, the text encoder and transformer on first use.
    pub fn load(path_or_repo: &str, opts: Options) -> Result<Self> {
        let dir = crate::engine::resolve_model(path_or_repo, None)?;
        if !is_omnigen2(&dir) {
            bail!("{}: not an OmniGen2 diffusers folder", dir.display());
        }
        crate::gpu::ensure(opts.device)?;
        let tdir = dir.join("transformer");
        let tcfg = TransformerConfig::load(&tdir).with_context(|| format!("{}", tdir.display()))?;
        let vae = crate::flux::vae::Vae::load(&dir.join("vae"), true)?;
        if vae.latent_channels != tcfg.in_channels {
            bail!(
                "VAE latent channels {} do not match the transformer input {}",
                vae.latent_channels,
                tcfg.in_channels
            );
        }
        Ok(Self {
            dir,
            opts,
            tcfg,
            text: None,
            tr: None,
            vae,
            cancel: Default::default(),
        })
    }

    /// A flag that stops the running (or next) call between denoising steps (any thread).
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.swap(false, Ordering::Relaxed) {
            bail!("cancelled");
        }
        Ok(())
    }

    /// Loads the Qwen2.5-VL language model (if not loaded); the vision tower is not needed for
    /// generation.
    pub fn load_text_encoder(&mut self) -> Result<f64> {
        if self.text.is_some() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let mdir = match &self.opts.text_encoder {
            Some(t) => crate::engine::resolve_model(t, None)?,
            None => self.dir.join("mllm"),
        };
        let pdir = self.dir.join("processor");
        let vo = QwenVlOptions {
            quant: self.opts.text_quant,
            // Mapped: it runs once per call (a few hundred tokens), float32 weights are converted
            // anyway, and a heap copy of the checkpoint was seen to outlive the model on Metal.
            mmap: true,
            tokenizer_dir: crate::fsx::exists(&pdir.join("tokenizer.json")).then(|| pdir.clone()),
            processor_dir: crate::fsx::exists(&pdir.join("preprocessor_config.json")).then(|| pdir.clone()),
            vision: false,
        };
        let vl = QwenVl::load(&mdir, &vo)?;
        if vl.hidden_size() != self.tcfg.text_feat_dim {
            bail!(
                "text encoder width {} does not match the transformer's {}",
                vl.hidden_size(),
                self.tcfg.text_feat_dim
            );
        }
        self.text = Some(vl);
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Drops the text encoder.
    pub fn unload_text_encoder(&mut self) {
        self.text = None;
    }

    /// Loads the transformer (if not loaded).
    pub fn load_transformer(&mut self) -> Result<f64> {
        if self.tr.is_some() {
            return Ok(0.0);
        }
        let t0 = Instant::now();
        let src = self.opts.transformer.clone().unwrap_or_else(|| self.dir.join("transformer"));
        let open = |in_memory: bool| ModelFiles::open(&src, in_memory).with_context(|| format!("transformer weights {}", src.display()));
        let mut files = Box::new(open(false)?);
        let prefix = ["", "model.diffusion_model.", "diffusion_model.", "transformer."]
            .into_iter()
            .find(|p| files.has(&format!("{p}x_embedder.weight")))
            .with_context(|| format!("{}: no x_embedder.weight (not an OmniGen2 transformer)", src.display()))?
            .to_string();
        // Weights converted at load (float32 ones, quantized ones, and on a GPU those padded to
        // its tiles: nearly all) are read through the mapping; the rest borrow the checkpoint,
        // read into memory unless `mmap`.
        let probe = format!("{prefix}layers.0.attn.to_q.weight");
        let gguf = files.ggml(&probe).is_some();
        let bf16 = files.info(&probe).is_some_and(|i| i.dtype == "BF16");
        let borrowed = gguf || (bf16 && self.opts.quant == Quant::Bf16 && crate::gpu::active().is_none());
        if borrowed && !self.opts.mmap {
            files = Box::new(open(true)?);
        }
        // SAFETY: the box's heap block does not move; `LoadedTransformer` drops the transformer
        // (every borrow of it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let ld = weights::Loader {
            files: fs,
            quant: self.opts.quant,
            prefix,
        };
        let t = Transformer::load(self.tcfg.clone(), &ld)?;
        self.tr = Some(LoadedTransformer { t, _files: files });
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Drops the transformer.
    pub fn unload_transformer(&mut self) {
        self.tr = None;
    }

    /// The transformer, once loaded.
    pub fn transformer(&self) -> Option<&Transformer<'static>> {
        self.tr.as_ref().map(|t| &t.t)
    }

    /// The text encoder, while loaded.
    pub fn text_encoder(&self) -> Option<&QwenVl> {
        self.text.as_ref()
    }

    /// The VAE.
    pub fn vae(&self) -> &crate::flux::vae::Vae {
        &self.vae
    }

    /// The conditioning of `prompt`: the chat template's final hidden states, `[t, 2048]`.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        let vl = self.text.as_ref().context("text encoder not loaded")?;
        let text = format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n");
        Ok(vl.hidden_states(&text, &[], &[])?.last)
    }

    /// Text to image.
    pub fn generate(&mut self, p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        self.run(&[], p, progress)
    }

    /// Edits `images` following the instruction in `p.prompt` (one image), or composes several
    /// (in-context generation: "the person in image 2 holding the toy from the first image").
    pub fn edit(&mut self, images: &[RgbImage], p: &Params, progress: &mut dyn FnMut(usize, usize)) -> Result<(Image, Timings)> {
        if images.is_empty() {
            bail!("an edit needs at least one input image");
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
        self.cancel.store(false, Ordering::Relaxed);
        if p.steps == 0 {
            bail!("steps must be at least 1");
        }
        let mut tm = Timings::default();
        let n_max = 5;
        if images.len() > n_max {
            bail!("OmniGen2 takes at most {n_max} input images");
        }
        // Input image sizes; with one image and an explicit size, the image takes that size.
        let fixed = match (images.len(), p.width, p.height) {
            (1, Some(w), Some(h)) => Some(output_size(w, h, p.max_pixels)),
            _ => None,
        };
        let mut vae_rgb: Vec<(Vec<u8>, usize, usize)> = Vec::new();
        for img in images {
            if img.data.len() != img.width * img.height * 3 {
                bail!("input image: {} bytes for {}x{} RGB", img.data.len(), img.width, img.height);
            }
            let (w, h) = fixed.unwrap_or_else(|| input_size(img.width, img.height, p.max_pixels, p.max_side));
            if w < 16 || h < 16 {
                bail!("input image {}x{} is too small", img.width, img.height);
            }
            let rgb = if (w, h) == (img.width, img.height) {
                img.data.clone()
            } else {
                crate::resample::resize_lanczos_rgb8(&img.data, img.width, img.height, w, h)
            };
            vae_rgb.push((rgb, w, h));
        }
        // Output size: the single image's, else the request (default 1024 x 1024) within
        // max_pixels, resized back to the request after decoding.
        let (ow, oh, lw_px, lh_px) = match (vae_rgb.len(), p.width, p.height) {
            (1, _, _) => (vae_rgb[0].1, vae_rgb[0].2, vae_rgb[0].1, vae_rgb[0].2),
            (_, w, h) => {
                let (w, h) = (w.unwrap_or(1024), h.unwrap_or(1024));
                let (gw, gh) = output_size(w, h, p.max_pixels);
                (w, h, gw, gh)
            }
        };
        if lw_px < 16 || lh_px < 16 {
            bail!("output size {ow}x{oh} is too small");
        }
        // Prompts. The Qwen2.5-VL hidden states carry outlier channels that the GPU's half
        // precision matrix units round visibly: the encoding runs with exact matmuls.
        let text_cfg = p.text_guidance > 1.0;
        let t0 = Instant::now();
        // Validation: the reference's conditioning instead of ours (dump folder).
        let (pre_pos, pre_neg) = (dumped("prompt_embeds_in.bin")?, dumped("negative_embeds_in.bin")?);
        let (pos, neg) = match (pre_pos, pre_neg) {
            (Some(pp), pn) if !text_cfg || pn.is_some() => (pp, pn.filter(|_| text_cfg)),
            _ => {
                tm.load_text = self.load_text_encoder()?;
                let was_exact = crate::gpu::exact();
                crate::gpu::set_exact(true);
                let enc = (|| -> Result<(Vec<f32>, Option<Vec<f32>>)> {
                    let pos = self.encode_prompt(&p.prompt)?;
                    let neg = if text_cfg { Some(self.encode_prompt(&p.negative)?) } else { None };
                    Ok((pos, neg))
                })();
                crate::gpu::set_exact(was_exact);
                enc?
            }
        };
        dump("prompt_embeds.bin", &pos);
        if let Some(n) = &neg {
            dump("negative_embeds.bin", n);
        }
        tm.text = t0.elapsed().as_secs_f64() - tm.load_text;
        if !self.opts.keep_text_encoder {
            self.unload_text_encoder();
        }
        self.check_cancel()?;

        // Reference latents.
        let t1 = Instant::now();
        let c = self.tcfg.in_channels;
        let mut refs = Vec::new();
        for (i, (rgb, w, h)) in vae_rgb.iter().enumerate() {
            let unit = crate::flux::vae::rgb8_to_unit(rgb);
            let (lat, lh, lw) = if p.sample_posterior {
                // Validation: the reference's posterior noise instead of ours (dump folder).
                let n = c * (h / 8) * (w / 8);
                let eps = match dumped(&format!("ref_noise{i}_in.bin"))? {
                    Some(v) => v,
                    None => crate::vae_posterior::noise(p.seed, i, n),
                };
                let s = crate::vae_posterior::encode_sampled(&self.vae, &unit, *h, *w, &eps, true)?;
                tm.posterior_std.push(s.relative_std);
                (s.latents, s.lh, s.lw)
            } else {
                self.vae.encode(&unit, *h, *w)?
            };
            let tokens = match dumped(&format!("ref_latents{i}_in.bin"))? {
                Some(v) => transformer::patchify(&crate::flux::vae::chw_to_hwc(&v, c, lh, lw), lh, lw, c),
                None => transformer::patchify(&lat, lh, lw, c),
            };
            dump(&format!("ref_tokens{i}.bin"), &tokens);
            refs.push(RefLatent {
                tokens,
                grid: Grid {
                    rows: lh / 2,
                    cols: lw / 2,
                },
            });
        }
        drop(vae_rgb);
        tm.vae_encode = t1.elapsed().as_secs_f64();

        tm.load_transformer = self.load_transformer()?;
        let t2 = Instant::now();
        let tr = &self.tr.as_ref().unwrap().t;
        let (lh, lw) = (lh_px / 8, lw_px / 8);
        let tg = Grid {
            rows: lh / 2,
            cols: lw / 2,
        };
        let noise = match dumped("noise.bin")? {
            Some(v) if v.len() == c * lh * lw => v,
            Some(v) => bail!("noise.bin has {} values, expected {}", v.len(), c * lh * lw),
            None => crate::sd::ops::Rng::new(p.seed).normals(c * lh * lw),
        };
        let mut x = transformer::patchify(&crate::flux::vae::chw_to_hwc(&noise, c, lh, lw), lh, lw, c);
        let txt_pos = tr.embed_text(&pos, pos.len() / self.tcfg.text_feat_dim)?;
        let txt_neg = neg
            .as_ref()
            .map(|n| tr.embed_text(n, n.len() / self.tcfg.text_feat_dim))
            .transpose()?;
        dump("txt_pos.bin", &txt_pos);
        let ts = timesteps(p.steps, lh * lw);
        let image_cfg = !refs.is_empty() && p.image_guidance > 1.0;
        let steps = p.steps;
        let mut dpm = (p.scheduler == Scheduler::DpmSolver).then(|| sched::DpmSolver::new(&ts[..steps]));
        // One cache state per guidance branch, as the official pipeline: conditional, reference
        // (images + negative prompt), unconditional.
        let mut caches = [BranchCache::new(&p.cache), BranchCache::new(&p.cache), BranchCache::new(&p.cache)];
        // Per pass "step branch ran", in the official pipeline's order of passes (dump folder).
        let mut decisions = String::new();
        for i in 0..steps {
            self.check_cancel()?;
            let (t, t_next) = (ts[i], ts[i + 1]);
            let sc = tr.step_cond(t);
            let mut streams = StepStreams {
                tr,
                x: &x,
                tg,
                refs: &refs,
                sc: &sc,
                noise: None,
                ref_embs: None,
            };
            let edge = i == 0 || i + 1 == steps;
            let frac = i as f32 / steps as f32;
            let in_range = p.cfg_range.0 <= frac && frac <= p.cfg_range.1;
            let tgs = if in_range { p.text_guidance } else { 1.0 };
            let igs = if in_range && image_cfg { p.image_guidance } else { 1.0 };
            let mut count = |ran: bool| {
                if ran {
                    tm.transformer_calls += 1;
                } else {
                    tm.cached_passes += 1;
                }
                ran as u8
            };
            let [c_cond, c_ref, c_uncond] = &mut caches;
            let (mut v, ran) = streams.pass(c_cond, &txt_pos, true, edge)?;
            let mut ran_all = vec![count(ran)];
            if i == 0 {
                dump("v0_cond.bin", &v);
            }
            if let (true, Some(tn)) = (tgs > 1.0, &txt_neg) {
                let (un, ran) = streams.pass(c_uncond, tn, false, edge)?;
                let ran_uncond = count(ran);
                if igs > 1.0 {
                    let (rf, ran) = streams.pass(c_ref, tn, true, edge)?;
                    ran_all.push(count(ran));
                    for ((a, &u), &r) in v.iter_mut().zip(&un).zip(&rf) {
                        *a = u + igs * (r - u) + tgs * (*a - r);
                    }
                } else {
                    for (a, &u) in v.iter_mut().zip(&un) {
                        *a = u + tgs * (*a - u);
                    }
                }
                ran_all.push(ran_uncond);
            }
            for (b, r) in ran_all.iter().enumerate() {
                decisions.push_str(&format!("{i} {b} {r}\n"));
            }
            if i == 0 {
                dump("v0.bin", &v);
            }
            drop(streams);
            match &mut dpm {
                Some(d) => d.step(i, &v, &mut x),
                None => {
                    let dt = t_next - t;
                    for (a, &b) in x.iter_mut().zip(&v) {
                        *a += dt * b;
                    }
                }
            }
            progress(i + 1, steps);
        }
        if let Some(d) = dump_dir() {
            let _ = std::fs::write(d.join("cache.txt"), &decisions);
        }
        tm.denoise = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        let lat = transformer::unpatchify(&x, lh, lw, c);
        dump("latents_out.bin", &crate::flux::vae::hwc_to_chw(&lat, c, lh, lw));
        let (mut px, mut h, mut w) = self.vae.decode(&lat, lh, lw);
        if (w, h) != (ow, oh) {
            px = resize_bilinear(&px, w, h, 3, ow, oh);
            (w, h) = (ow, oh);
        }
        let rgb = crate::flux::vae::unit_to_rgb8(&px);
        tm.decode = t3.elapsed().as_secs_f64();
        Ok((Image { width: w, height: h, rgb }, tm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_the_official_processor() {
        // 1024 side cap first, then the pixel budget; floors to 16.
        assert_eq!(input_size(1920, 1080, 1024 * 1024, 1024), (1024, 576));
        assert_eq!(input_size(640, 480, 1024 * 1024, 1024), (640, 480));
        assert_eq!(input_size(1000, 1000, 1024 * 1024, 1024), (992, 992));
        assert_eq!(input_size(2048, 2048, 1024 * 1024, 2048), (1024, 1024));
        assert_eq!(output_size(1024, 1024, 1024 * 1024), (1024, 1024));
        assert_eq!(output_size(1500, 1000, 1024 * 1024), (1248, 832));
    }

    #[test]
    fn schedule_shift() {
        // 1024 x 1024: m = 128 / 40 = 3.2.
        let t = timesteps(4, 128 * 128);
        assert_eq!(t.len(), 5);
        assert_eq!(t[0], 0.0);
        assert!((t[2] - 0.5 / (3.2 - 1.6 + 0.5)).abs() < 1e-6);
        assert_eq!(t[4], 1.0);
    }

    #[test]
    fn bilinear_identity_and_halves() {
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(resize_bilinear(&x, 4, 4, 1, 4, 4), x);
        let y = resize_bilinear(&x, 4, 4, 1, 2, 2);
        assert_eq!(y, vec![2.5, 4.5, 10.5, 12.5]);
    }
}
