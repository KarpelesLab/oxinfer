//! ByteDance BAGEL (`ByteDance-Seed/BAGEL-7B-MoT`): one unified model that reads images and
//! text and writes images and text, following the official `InterleaveInferencer`
//! (`github.com/ByteDance-Seed/Bagel`).
//!
//! * The language model ([`llm`]) is Qwen2.5-7B as a Mixture-of-Transformers: an understanding
//!   expert for text and SigLIP tokens, a generation expert for VAE latent tokens, sharing
//!   self-attention over one packed sequence.
//! * Images are read twice: as FLUX VAE latents (2x2 patches through `vae2llm`, plus a timestep
//!   embedding of 0 and a 2-D sin-cos position embedding) and as SigLIP so400m features
//!   ([`siglip`], NaViT-style at the image's own size). Each image takes one RoPE position.
//! * Images are written by rectified flow: noise latents at one position after the context are
//!   denoised in `steps - 1` Euler steps over shifted timesteps (`shift * t / (1 + (shift - 1) t)`),
//!   the generation expert predicting the velocity (noise minus data) through `llm2vae`.
//!   Guidance has two terms: text (the context without the last text) and image (the context
//!   without the images), applied within `cfg_interval`, with CFG-renorm (`global`, `channel` or
//!   `text_channel`) capping the guided velocity's norm.
//! * Text (answers, and the optional "thinking" plan written before an image) comes from the
//!   understanding expert's LM head, decoded after `<|im_start|>`.
//!
//! Every text segment is wrapped in `<|im_start|>` / `<|im_end|>` and every image in
//! `<|vision_start|>` / `<|vision_end|>`; there is no chat template beyond that.

pub mod llm;
pub mod siglip;

use crate::clock::Instant;
use crate::flux::vae::Vae;
use crate::generate::Flow;
use crate::model::Quant;
use crate::platform::DevicePref;
use crate::safetensors::ModelFiles;
use crate::sampler::Sampler;
use crate::video::RgbImage;
use anyhow::{Context, Result, anyhow, bail};
use llm::{Block, Ctx, Kv, Llm, LlmConfig, Proj, project};
use siglip::{Vit, VitConfig};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokenizers::Tokenizer;

/// A generated image (tightly packed RGB8; `to_video_frame()` with the `oxideav` feature).
pub use crate::sd::Image;

/// The system prompt of thinking before an answer (`VLM_THINK_SYSTEM_PROMPT`).
pub const VLM_THINK_SYSTEM_PROMPT: &str = "You should first think about the reasoning process in the mind and then provide the user with the answer. \nThe reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here";

/// The system prompt of thinking before an image (`GEN_THINK_SYSTEM_PROMPT`).
pub const GEN_THINK_SYSTEM_PROMPT: &str = "You should first think about the planning process in the mind and then generate the image. \nThe planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here";

/// True when `dir` holds a BAGEL checkpoint.
pub fn is_bagel(dir: &Path) -> bool {
    let cfg = crate::fsx::read(&dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
    cfg.is_some_and(|v| v["model_type"] == "bagel")
        || (crate::fsx::exists(&dir.join("llm_config.json"))
            && crate::fsx::exists(&dir.join("vit_config.json"))
            && crate::fsx::exists(&dir.join("ae.safetensors")))
}

/// Loading options.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Precision of the language model's and SigLIP's linear weights (default bf16).
    pub quant: Quant,
    /// Map the weights from disk instead of reading them into memory.
    pub mmap: bool,
    /// Compute device (process-wide; set by the first model loaded).
    pub device: DevicePref,
    /// Where repo ids are downloaded.
    pub cache_dir: Option<PathBuf>,
    /// Load the image generation parts: the generation expert, VAE, time / latent embeddings.
    pub generation: bool,
    /// Load SigLIP (images as understanding input: editing and questions).
    pub understanding: bool,
    /// Load the LM head (text output: answers, thinking).
    pub text: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            mmap: false,
            device: DevicePref::Auto,
            cache_dir: None,
            generation: true,
            understanding: true,
            text: true,
        }
    }
}

impl LoadOptions {
    /// Sets an option by name (the C API's `key=value` options): `quant` (bf16, q8, q4), `mmap`,
    /// `device`, `cache_dir`, `generation`, `understanding`, `text`.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let flag = || match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(anyhow!("{key}: expected a boolean, got {value:?}")),
        };
        match key {
            "quant" => self.quant = Quant::parse(value).with_context(|| format!("quant: unknown quantization {value:?} (bf16, q8, q4)"))?,
            "mmap" => self.mmap = flag()?,
            "device" => self.device = DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?}"))?,
            "cache_dir" => self.cache_dir = (!value.is_empty()).then(|| PathBuf::from(value)),
            "generation" => self.generation = flag()?,
            "understanding" => self.understanding = flag()?,
            "text" => self.text = flag()?,
            _ => bail!("unknown option {key:?} (quant, mmap, device, cache_dir, generation, understanding, text)"),
        }
        Ok(())
    }
}

/// CFG-renorm: how the guided velocity's norm is capped at the conditional one's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Renorm {
    /// Over the whole velocity (the default for text to image).
    Global,
    /// Per token.
    Channel,
    /// Per token, on the text term only, before image guidance (the default for editing).
    TextChannel,
}

impl Renorm {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "global" => Some(Self::Global),
            "channel" => Some(Self::Channel),
            "text_channel" => Some(Self::TextChannel),
            _ => None,
        }
    }
}

/// Image generation / editing options. `None` fields take the reference notebook's values for
/// the task: text to image `cfg_text_scale` 4, `cfg_img_scale` 1, `cfg_interval` [0.4, 1],
/// `global` renorm; editing 4, 2, [0, 1], `text_channel`. Deserializes from JSON with every key
/// optional.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageOptions {
    /// Timesteps of the schedule (default 50: 49 model evaluations per guidance branch).
    pub steps: usize,
    /// Timestep shift (default 3).
    pub timestep_shift: f32,
    pub cfg_text_scale: Option<f32>,
    pub cfg_img_scale: Option<f32>,
    /// Guidance is applied at timesteps t in (lo, hi].
    pub cfg_interval: Option<[f32; 2]>,
    /// Lower bound of the renorm scale (default 0; 1 disables renorm).
    pub cfg_renorm_min: f32,
    pub cfg_renorm_type: Option<Renorm>,
    /// Text to image: output size (default 1024 x 1024), rounded down to multiples of 16.
    /// Editing: the output takes the input's size after its resize.
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Input images are resized to fit these (default 1024 and 512, as the reference's
    /// `ImageTransform(1024, 512, 16)`; smaller values make editing much cheaper).
    pub max_image_size: usize,
    pub min_image_size: usize,
    pub seed: u64,
    /// Write a plan (`<think>...</think>`) before the image.
    pub think: bool,
    /// Thinking: most tokens (default 1000), sampling temperature (0, the default: greedy).
    pub max_think_tokens: usize,
    pub temperature: f32,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            steps: 50,
            timestep_shift: 3.0,
            cfg_text_scale: None,
            cfg_img_scale: None,
            cfg_interval: None,
            cfg_renorm_min: 0.0,
            cfg_renorm_type: None,
            width: None,
            height: None,
            max_image_size: 1024,
            min_image_size: 512,
            seed: 42,
            think: false,
            max_think_tokens: 1000,
            temperature: 0.0,
        }
    }
}

/// Text output options (image understanding). Greedy by default, as the reference.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextOptions {
    pub max_tokens: usize,
    /// 0 = greedy.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    /// Think first (the reasoning stays in the returned text, inside `<think>` tags).
    pub think: bool,
    /// Input image resizing (see [`ImageOptions::max_image_size`]).
    pub max_image_size: usize,
    pub min_image_size: usize,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            max_tokens: 1000,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
            think: false,
            max_image_size: 1024,
            min_image_size: 512,
        }
    }
}

/// Timings and details of one image generation, in seconds.
#[derive(Clone, Debug, Default)]
pub struct ImageStats {
    /// Context prefill (text, VAE and SigLIP passes of the inputs).
    pub context: f64,
    /// Thinking text generation.
    pub think: f64,
    pub denoise: f64,
    pub decode: f64,
    /// Language-model passes over the latent tokens (all guidance branches).
    pub passes: usize,
    /// Latent tokens per pass and the context lengths of the conditional branch.
    pub latent_tokens: usize,
    pub context_tokens: usize,
    /// The plan written in thinking mode.
    pub thinking: Option<String>,
}

/// Progress callback for denoising: `(step, steps)`; return false to cancel.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

/// Stops a running generation from another thread.
#[derive(Clone, Debug)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// The image generation parts.
struct GenParts<'a> {
    time0: Proj<'a>,
    time2: Proj<'a>,
    vae2llm: Proj<'a>,
    llm2vae: Proj<'a>,
    /// `[max_latent^2, hidden]`.
    latent_pos: &'a [u16],
    max_latent: usize,
}

struct Special {
    im_start: u32,
    im_end: u32,
    vision_start: u32,
    vision_end: u32,
}

struct Parts<'a> {
    llm: Llm<'a>,
    vit: Option<Vit<'a>>,
    gen_: Option<GenParts<'a>>,
}

/// A loaded BAGEL model.
pub struct Bagel {
    // Declared before `_files`, which it borrows: dropped first.
    parts: Parts<'static>,
    vae: Option<Vae>,
    tok: Tokenizer,
    special: Special,
    cancel: Arc<AtomicBool>,
    pub path: PathBuf,
    pub load_secs: f64,
    /// Patch size of the latent grid (2) and VAE downsampling (8).
    latent_patch: usize,
    latent_channels: usize,
    _files: Box<ModelFiles>,
}

/// A context: the KV cache of everything seen so far and the next RoPE position.
struct Seq {
    kv: Kv,
    rope: usize,
}

/// A guidance branch's view of a context: its first `len` positions, `rope` the next position.
#[derive(Clone, Copy)]
struct View<'c> {
    kv: &'c Kv,
    len: usize,
    rope: usize,
}

impl View<'_> {
    fn same(&self, o: &View) -> bool {
        std::ptr::eq(self.kv, o.kv) && self.len == o.len && self.rope == o.rope
    }
}

/// One input of an interleaved sequence.
pub enum Input<'a> {
    Text(&'a str),
    Image(&'a RgbImage),
}

/// Python's `round()` (half to even).
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// The reference's `MaxLongEdgeMinShortEdgeResize`: the size a `w` x `h` image is resized to
/// (sides multiples of `stride`, the longer at most `max_size`, the shorter at least `min_size`
/// where that allows, at most `max_pixels`).
pub fn resize_size(w: usize, h: usize, max_size: usize, min_size: usize, stride: usize, max_pixels: usize) -> (usize, usize) {
    let s = stride as i64;
    let div = |v: i64| -> usize { s.max(py_round(v as f64 / s as f64) * s) as usize };
    let apply = |w: usize, h: usize, scale: f64| (div(py_round(w as f64 * scale)), div(py_round(h as f64 * scale)));
    let mut scale = (max_size as f64 / w.max(h) as f64).min(1.0);
    scale = scale.max(min_size as f64 / w.min(h).max(1) as f64);
    let (mut nw, mut nh) = apply(w, h, scale);
    if (nw * nh) as f64 > max_pixels as f64 {
        let scale = max_pixels as f64 / (nw * nh) as f64;
        (nw, nh) = apply(nw, nh, scale);
    }
    if nw.max(nh) > max_size {
        let scale = max_size as f64 / nw.max(nh) as f64;
        (nw, nh) = apply(nw, nh, scale);
    }
    (nw, nh)
}

/// `max_pixels` of the reference's `ImageTransform` (14 * 14 * 9 * 1024).
const MAX_PIXELS: usize = 14 * 14 * 9 * 1024;

/// Resizes like torchvision's `F.resize` on a PIL image (Pillow bicubic, antialiased); the same
/// size is a copy, as Pillow returns.
fn resize_image(img: &RgbImage, w: usize, h: usize) -> RgbImage {
    if (w, h) == (img.width, img.height) {
        return img.clone();
    }
    RgbImage {
        width: w,
        height: h,
        data: crate::resample::resize_bicubic_rgb8(&img.data, img.width, img.height, w, h),
    }
}

/// RGB8 to channels-last floats normalized as `Normalize(0.5, 0.5)` after `ToTensor`.
fn normalize(rgb: &[u8]) -> Vec<f32> {
    rgb.iter().map(|&v| (v as f32 / 255.0 - 0.5) / 0.5).collect()
}

/// Dumps for reference checks: `OXINFER_BAGEL_DUMP=DIR` writes intermediate tensors (f32 LE) and
/// reads `noise_in.bin` (the initial latent tokens, `[n, 64]`) from it.
fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("OXINFER_BAGEL_DUMP").map(PathBuf::from)
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

impl Bagel {
    /// Loads a BAGEL checkpoint: a local folder or a Hugging Face repo id
    /// (`ByteDance-Seed/BAGEL-7B-MoT`), fetched into the cache on first use.
    pub fn load(path_or_repo: &str, opts: &LoadOptions) -> Result<Bagel> {
        let t0 = Instant::now();
        let dir = crate::engine::resolve_model(path_or_repo, opts.cache_dir.as_deref())?;
        if !is_bagel(&dir) {
            bail!("{}: not a BAGEL checkpoint", dir.display());
        }
        let read_json = |n: &str| -> Result<serde_json::Value> {
            Ok(serde_json::from_slice(
                &crate::fsx::read(&dir.join(n)).with_context(|| format!("{}: no {n}", dir.display()))?,
            )?)
        };
        let cfg = read_json("config.json").unwrap_or_default();
        let lcfg = match read_json("llm_config.json") {
            Ok(v) => LlmConfig::from_json(&v)?,
            Err(_) => LlmConfig::from_json(&cfg["llm_config"])?,
        };
        let vit_json = read_json("vit_config.json").unwrap_or_else(|_| cfg["vit_config"].clone());
        crate::gpu::ensure(opts.device)?;
        let files = Box::new(ModelFiles::open(&dir, !opts.mmap)?);
        // SAFETY: the box's heap block does not move, and `Bagel` drops `parts` (every borrow of
        // it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let tok = crate::engine::load_tokenizer(&dir)?;
        let id = |t: &str| tok.token_to_id(t).with_context(|| format!("tokenizer: no {t}"));
        let special = Special {
            im_start: id("<|im_start|>")?,
            im_end: id("<|im_end|>")?,
            vision_start: id("<|vision_start|>")?,
            vision_end: id("<|vision_end|>")?,
        };
        let h = lcfg.hidden_size;
        let llm = Llm::load(lcfg, fs, opts.quant, opts.generation, opts.text)?;
        let grid_side = |name: &str| -> Result<usize> {
            let rows = fs.info(name).with_context(|| format!("missing tensor {name}"))?.shape[0];
            let s = (rows as f64).sqrt().round() as usize;
            if s * s != rows {
                bail!("{name}: {rows} rows is not a square grid");
            }
            Ok(s)
        };
        let vit = if opts.understanding {
            let side = grid_side("vit_pos_embed.pos_embed")?;
            Some(Vit::load(VitConfig::from_json(&vit_json, side), fs, h, opts.quant)?)
        } else {
            None
        };
        let latent_patch = cfg["latent_patch_size"].as_u64().unwrap_or(2) as usize;
        let latent_channels = cfg["vae_config"]["z_channels"].as_u64().unwrap_or(16) as usize;
        let (gen_, vae) = if opts.generation {
            let pd = latent_patch * latent_patch * latent_channels;
            let max_latent = grid_side("latent_pos_embed.pos_embed")?;
            let latent_pos = fs.bf16("latent_pos_embed.pos_embed")?;
            let g = GenParts {
                time0: Proj::load(fs, "time_embedder.mlp.0", h, 256, Quant::Bf16, true)?,
                time2: Proj::load(fs, "time_embedder.mlp.2", h, h, Quant::Bf16, true)?,
                vae2llm: Proj::load(fs, "vae2llm", h, pd, Quant::Bf16, true)?,
                llm2vae: Proj::load(fs, "llm2vae", pd, h, Quant::Bf16, true)?,
                latent_pos,
                max_latent,
            };
            let vae = Vae::load_bfl(&dir.join("ae.safetensors"), true).context("VAE (ae.safetensors)")?;
            if vae.latent_channels != latent_channels {
                bail!("the VAE has {} latent channels, the config {latent_channels}", vae.latent_channels);
            }
            (Some(g), Some(vae))
        } else {
            (None, None)
        };
        Ok(Bagel {
            parts: Parts { llm, vit, gen_ },
            vae,
            tok,
            special,
            cancel: Arc::new(AtomicBool::new(false)),
            path: dir,
            load_secs: t0.elapsed().as_secs_f64(),
            latent_patch,
            latent_channels,
            _files: files,
        })
    }

    pub fn config(&self) -> &LlmConfig {
        &self.parts.llm.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// A handle that stops the running (or next) call; it then fails with "cancelled".
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancel.clone())
    }

    /// Bytes of the weights held in memory (0 for mapped bf16 weights is not distinguished).
    pub fn file_bytes(&self) -> u64 {
        self._files.total_bytes()
    }

    fn new_context(&self) -> Seq {
        Seq {
            kv: self.parts.llm.new_kv(),
            rope: 0,
        }
    }

    fn view(c: &Seq) -> View<'_> {
        View {
            kv: &c.kv,
            len: c.kv.len,
            rope: c.rope,
        }
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    /// `update_context_text`: `<|im_start|>` text `<|im_end|>`, causal, understanding expert.
    fn add_text(&self, c: &mut Seq, text: &str) -> Result<()> {
        let mut ids = vec![self.special.im_start];
        ids.extend(self.encode_text(text)?);
        ids.push(self.special.im_end);
        let llm = &self.parts.llm;
        let n = ids.len();
        llm.forward(
            Ctx::Append(&mut c.kv),
            Block {
                und: llm.embed(&ids),
                gen_: Vec::new(),
                pos: c.rope,
                text: true,
            },
        )?;
        c.rope += n;
        Ok(())
    }

    /// The resize every input image gets first (`vae_transform.resize_transform`).
    fn input_image(img: &RgbImage, max_size: usize, min_size: usize) -> Result<RgbImage> {
        if img.width == 0 || img.height == 0 || img.data.len() != img.width * img.height * 3 {
            bail!("image: {} bytes for {}x{} RGB", img.data.len(), img.width, img.height);
        }
        let (w, h) = resize_size(img.width, img.height, max_size.max(16), min_size, 16, MAX_PIXELS);
        Ok(resize_image(img, w, h))
    }

    fn gen_parts(&self) -> Result<&GenParts<'static>> {
        self.parts
            .gen_
            .as_ref()
            .context("image generation was not loaded (LoadOptions::generation)")
    }

    fn vae(&self) -> Result<&Vae> {
        self.vae.as_ref().context("the VAE was not loaded (LoadOptions::generation)")
    }

    /// `time_embedder(t)`: `[hidden]`.
    fn time_embed(&self, t: f32) -> Result<Vec<f32>> {
        let g = self.gen_parts()?;
        let h = self.parts.llm.cfg.hidden_size;
        // DiT's sinusoidal embedding (256 wide, cos first, max period 10000), then Linear,
        // SiLU, Linear. One row padded to 8: full-precision activations on any weight format.
        let half = 128;
        let mut f = vec![0f32; 8 * 256];
        for i in 0..half {
            let fr = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            let a = t * fr;
            f[i] = a.cos();
            f[half + i] = a.sin();
        }
        for r in 1..8 {
            f.copy_within(0..256, r * 256);
        }
        let mut y = project(&[&g.time0], 256, &f, 8).swap_remove(0);
        y.iter_mut().for_each(|v| *v = crate::sd::ops::silu(*v));
        let mut y = project(&[&g.time2], h, &y, 8).swap_remove(0);
        y.truncate(h);
        Ok(y)
    }

    /// The latent grid's position ids for an `w` x `h` image.
    fn latent_ids(&self, w: usize, h: usize) -> Result<Vec<usize>> {
        let g = self.gen_parts()?;
        let f = 8 * self.latent_patch;
        let (gh, gw) = (h / f, w / f);
        if gh > g.max_latent || gw > g.max_latent {
            bail!(
                "image {w}x{h} exceeds the {}x{} latent grid ({} pixels per side)",
                g.max_latent,
                g.max_latent,
                g.max_latent * f
            );
        }
        Ok((0..gh).flat_map(|r| (0..gw).map(move |c| r * g.max_latent + c)).collect())
    }

    /// Latent tokens `[n, 64]` (`hwpqc` patches) to the language model's input: `vae2llm` plus
    /// the timestep and position embeddings.
    fn latent_embed(&self, x: &[f32], ids: &[usize], t: f32) -> Result<Vec<f32>> {
        let g = self.gen_parts()?;
        let h = self.parts.llm.cfg.hidden_size;
        let pd = g.vae2llm.n_in;
        let n = ids.len();
        let mut e = project(&[&g.vae2llm], pd, x, n).swap_remove(0);
        let te = self.time_embed(t)?;
        let pos = &g.latent_pos;
        crate::kernels::for_chunks(&mut e, h, n > 64, |i, r| {
            let p = &pos[ids[i] * h..(ids[i] + 1) * h];
            for j in 0..h {
                r[j] += te[j] + crate::kernels::bf16_to_f32(p[j]);
            }
        });
        Ok(e)
    }

    /// `update_context_image` with `vae=True`: the image's clean latents through the generation
    /// expert (bidirectional, one position).
    fn add_vae_image(&self, c: &mut Seq, img: &RgbImage) -> Result<()> {
        let (w, h) = (img.width, img.height);
        let f = 8 * self.latent_patch;
        if !w.is_multiple_of(f) || !h.is_multiple_of(f) {
            bail!("VAE input {w}x{h} is not a multiple of {f}");
        }
        let ids = self.latent_ids(w, h)?;
        let z = match read_override("vae_latents_in.bin", (h / 8) * (w / 8) * self.latent_channels)? {
            Some(z) => z,
            None => self.vae()?.encode(&normalize(&img.data), h, w)?.0,
        };
        let tok = self.patchify(&z, h / 8, w / 8);
        dump("vae_tokens.bin", &tok);
        let gen_ = self.latent_embed(&tok, &ids, 0.0)?;
        let llm = &self.parts.llm;
        llm.forward(
            Ctx::Append(&mut c.kv),
            Block {
                und: llm.embed(&[self.special.vision_start, self.special.vision_end]),
                gen_,
                pos: c.rope,
                text: false,
            },
        )?;
        c.rope += 1;
        Ok(())
    }

    /// The SigLIP embeddings of an image (after the `vit_transform` resize), `[patches, hidden]`.
    pub fn vit_embeddings(&self, img: &RgbImage) -> Result<Vec<f32>> {
        let vit = self
            .parts
            .vit
            .as_ref()
            .context("SigLIP was not loaded (LoadOptions::understanding)")?;
        let p = vit.cfg.patch;
        let (w, h) = resize_size(
            img.width,
            img.height,
            vit.cfg.max_patches * p,
            224.min(vit.cfg.max_patches * p),
            p,
            MAX_PIXELS,
        );
        let im = resize_image(img, w, h);
        let mut feats = Vec::new();
        let e = vit.encode(&normalize(&im.data), h, w, dump_dir().is_some().then_some(&mut feats))?;
        if !feats.is_empty() {
            dump("vit_features.bin", &feats);
            dump("vit_embed.bin", &e);
        }
        Ok(e)
    }

    /// `update_context_image` with `vit=True`: SigLIP tokens through the understanding expert.
    fn add_vit_image(&self, c: &mut Seq, img: &RgbImage) -> Result<()> {
        let e = self.vit_embeddings(img)?;
        let llm = &self.parts.llm;
        let h = llm.cfg.hidden_size;
        let mut und = llm.embed(&[self.special.vision_start]);
        und.extend_from_slice(&e);
        und.extend(llm.embed(&[self.special.vision_end]));
        debug_assert_eq!(und.len() % h, 0);
        llm.forward(
            Ctx::Append(&mut c.kv),
            Block {
                und,
                gen_: Vec::new(),
                pos: c.rope,
                text: false,
            },
        )?;
        c.rope += 1;
        Ok(())
    }

    /// Channels-last latents `[lh * lw, c]` to tokens `[(lh/p) * (lw/p), p*p*c]` (`hwpqc`).
    fn patchify(&self, z: &[f32], lh: usize, lw: usize) -> Vec<f32> {
        let (p, c) = (self.latent_patch, self.latent_channels);
        let (gh, gw) = (lh / p, lw / p);
        let mut t = vec![0f32; gh * gw * p * p * c];
        for i in 0..gh {
            for j in 0..gw {
                for py in 0..p {
                    for px in 0..p {
                        let src = ((i * p + py) * lw + j * p + px) * c;
                        let dst = ((i * gw + j) * p * p + py * p + px) * c;
                        t[dst..dst + c].copy_from_slice(&z[src..src + c]);
                    }
                }
            }
        }
        t
    }

    fn unpatchify(&self, t: &[f32], gh: usize, gw: usize) -> Vec<f32> {
        let (p, c) = (self.latent_patch, self.latent_channels);
        let (lh, lw) = (gh * p, gw * p);
        let mut z = vec![0f32; lh * lw * c];
        for i in 0..gh {
            for j in 0..gw {
                for py in 0..p {
                    for px in 0..p {
                        let dst = ((i * p + py) * lw + j * p + px) * c;
                        let src = ((i * gw + j) * p * p + py * p + px) * c;
                        z[dst..dst + c].copy_from_slice(&t[src..src + c]);
                    }
                }
            }
        }
        z
    }

    /// Greedy or sampled text after the context (`gen_text`), up to `<|im_end|>`; the context is
    /// left as it was. `on_text` receives the text as it grows.
    fn gen_text(&self, c: &mut Seq, o: &TextOptions, on_text: &mut dyn FnMut(&str) -> Flow) -> Result<String> {
        let llm = &self.parts.llm;
        let (len0, rope0) = (c.kv.len, c.rope);
        let mut sampler = Sampler::new(o.temperature, o.top_k, o.top_p, o.seed);
        let mut cur = self.special.im_start;
        let mut out: Vec<u32> = Vec::new();
        let mut text = String::new();
        let mut pos = rope0;
        let res = (|| -> Result<()> {
            for step in 0..o.max_tokens {
                let (hid, _) = llm.forward(
                    Ctx::Append(&mut c.kv),
                    Block {
                        und: llm.embed(&[cur]),
                        gen_: Vec::new(),
                        pos,
                        text: true,
                    },
                )?;
                pos += 1;
                let logits = llm.logits(&hid, 1)?;
                if step == 0 {
                    dump("text_logits0.bin", &logits);
                }
                let t = sampler.sample(&logits);
                // As the reference: the end token stops, and the token predicted by the last of the
                // `max_tokens` passes is dropped.
                if t == self.special.im_end || step + 1 == o.max_tokens {
                    break;
                }
                out.push(t);
                cur = t;
                let full = self.tok.decode(&out, false).map_err(|e| anyhow!("decode: {e}"))?;
                if full.len() > text.len() && !full.ends_with('\u{FFFD}') && full.starts_with(&text) {
                    let delta = full[text.len()..].to_string();
                    text = full;
                    if on_text(&delta) == Flow::Stop {
                        break;
                    }
                }
                if self.cancel.load(Ordering::Relaxed) {
                    bail!("cancelled");
                }
            }
            Ok(())
        })();
        c.kv.truncate(len0, llm.cfg.head_dim);
        c.rope = rope0;
        res?;
        let full = self.tok.decode(&out, false).map_err(|e| anyhow!("decode: {e}"))?;
        let full = full.split("<|im_end|>").next().unwrap_or("");
        let full = full.split("<|im_start|>").next().unwrap_or("").to_string();
        if full.len() > text.len() && full.starts_with(&text) {
            on_text(&full[text.len()..]);
        }
        Ok(full)
    }

    /// Answers `question` about `image` (or without one), streaming the text to `on_text`.
    pub fn understand(
        &self,
        image: Option<&RgbImage>,
        question: &str,
        o: &TextOptions,
        on_text: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<String> {
        let mut inputs = Vec::new();
        if let Some(i) = image {
            inputs.push(Input::Image(i));
        }
        inputs.push(Input::Text(question));
        self.understand_interleaved(&inputs, o, on_text)
    }

    /// Text after an interleaved sequence of texts and images (`interleave_inference` with
    /// `understanding_output=True`).
    pub fn understand_interleaved(&self, inputs: &[Input], o: &TextOptions, on_text: &mut dyn FnMut(&str) -> Flow) -> Result<String> {
        self.cancel.store(false, Ordering::Relaxed);
        let mut c = self.new_context();
        if o.think {
            self.add_text(&mut c, VLM_THINK_SYSTEM_PROMPT)?;
        }
        for i in inputs {
            match i {
                Input::Text(t) => self.add_text(&mut c, t)?,
                Input::Image(img) => {
                    let img = Self::input_image(img, o.max_image_size, o.min_image_size)?;
                    self.add_vit_image(&mut c, &img)?;
                }
            }
        }
        self.gen_text(&mut c, o, on_text)
    }

    /// The next-token logits (`[vocab]`) of [`Bagel::understand`]'s first step, for checks.
    pub fn understanding_logits(&self, image: Option<&RgbImage>, question: &str, o: &TextOptions) -> Result<Vec<f32>> {
        let mut c = self.new_context();
        if let Some(i) = image {
            let img = Self::input_image(i, o.max_image_size, o.min_image_size)?;
            self.add_vit_image(&mut c, &img)?;
        }
        self.add_text(&mut c, question)?;
        let llm = &self.parts.llm;
        let (hid, _) = llm.forward(
            Ctx::Prefix(&c.kv, c.kv.len),
            Block {
                und: llm.embed(&[self.special.im_start]),
                gen_: Vec::new(),
                pos: c.rope,
                text: true,
            },
        )?;
        llm.logits(&hid, 1)
    }

    /// Text to image.
    pub fn generate(&self, prompt: &str, o: &ImageOptions, progress: Progress) -> Result<(Image, ImageStats)> {
        self.generate_interleaved(&[Input::Text(prompt)], o, progress)
    }

    /// Instruction-based editing: `image` changed as `instruction` says. The output has the
    /// input's size after the reference's resize (sides multiples of 16 within
    /// `max_image_size` / `min_image_size`).
    pub fn edit(&self, image: &RgbImage, instruction: &str, o: &ImageOptions, progress: Progress) -> Result<(Image, ImageStats)> {
        self.generate_interleaved(&[Input::Image(image), Input::Text(instruction)], o, progress)
    }

    #[cfg(feature = "oxideav")]
    /// [`Bagel::edit`] with an oxideav picture.
    pub fn edit_picture(
        &self,
        image: &crate::av::Picture,
        instruction: &str,
        o: &ImageOptions,
        progress: Progress,
    ) -> Result<(Image, ImageStats)> {
        self.edit(&image.to_rgb()?, instruction, o, progress)
    }

    #[cfg(feature = "oxideav")]
    /// [`Bagel::understand`] with an oxideav picture.
    pub fn understand_picture(
        &self,
        image: &crate::av::Picture,
        question: &str,
        o: &TextOptions,
        on_text: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<String> {
        self.understand(Some(&image.to_rgb()?), question, o, on_text)
    }

    /// An image after an interleaved sequence of texts and images (`interleave_inference`):
    /// guidance drops the last text (text term) and every image (image term).
    pub fn generate_interleaved(&self, inputs: &[Input], o: &ImageOptions, progress: Progress) -> Result<(Image, ImageStats)> {
        self.cancel.store(false, Ordering::Relaxed);
        self.gen_parts()?;
        let mut st = ImageStats::default();
        let has_image = inputs.iter().any(|i| matches!(i, Input::Image(_)));
        let cfg_text = o.cfg_text_scale.unwrap_or(4.0);
        let cfg_img = o.cfg_img_scale.unwrap_or(if has_image { 2.0 } else { 1.0 });
        let interval = o.cfg_interval.unwrap_or(if has_image { [0.0, 1.0] } else { [0.4, 1.0] });
        let renorm = o
            .cfg_renorm_type
            .unwrap_or(if has_image { Renorm::TextChannel } else { Renorm::Global });
        let t0 = Instant::now();
        // The conditional context, and the image-free one (as the reference, every text goes to
        // both; images only to the first). Text guidance views the conditional context as it was
        // before its last text.
        let mut gen_c = self.new_context();
        let mut img_c = self.new_context();
        let mut text_mark = (0usize, 0usize);
        if o.think {
            self.add_text(&mut gen_c, GEN_THINK_SYSTEM_PROMPT)?;
        }
        let mut shape = (o.width.unwrap_or(1024), o.height.unwrap_or(1024));
        let mut img_texts: Vec<&str> = Vec::new();
        for i in inputs {
            match i {
                Input::Text(t) => {
                    text_mark = (gen_c.kv.len, gen_c.rope);
                    self.add_text(&mut gen_c, t)?;
                    img_texts.push(t);
                }
                Input::Image(img) => {
                    let img = Self::input_image(img, o.max_image_size, o.min_image_size)?;
                    self.add_vae_image(&mut gen_c, &img)?;
                    self.add_vit_image(&mut gen_c, &img)?;
                    shape = (img.width, img.height);
                    text_mark = (gen_c.kv.len, gen_c.rope);
                }
            }
        }
        // The image-free context equals the conditional one's prefix up to the last text when no
        // image came; otherwise it is built separately (only when image guidance is on).
        let img_mark = if has_image && cfg_img > 1.0 {
            if o.think {
                self.add_text(&mut img_c, GEN_THINK_SYSTEM_PROMPT)?;
            }
            for t in &img_texts {
                self.add_text(&mut img_c, t)?;
            }
            (img_c.kv.len, img_c.rope)
        } else {
            (gen_c.kv.len, gen_c.rope)
        };
        st.context = t0.elapsed().as_secs_f64();
        if o.think {
            let t1 = Instant::now();
            let to = TextOptions {
                max_tokens: o.max_think_tokens,
                temperature: o.temperature,
                seed: o.seed,
                ..Default::default()
            };
            let plan = self.gen_text(&mut gen_c, &to, &mut |_| Flow::Continue)?;
            self.add_text(&mut gen_c, &plan)?;
            st.think = t1.elapsed().as_secs_f64();
            st.thinking = Some(plan);
        }
        let (w, h) = (shape.0 / 16 * 16, shape.1 / 16 * 16);
        if w < 16 || h < 16 {
            bail!("image size must be at least 16x16");
        }
        let cond = Self::view(&gen_c);
        let text_v = View {
            kv: &gen_c.kv,
            len: text_mark.0,
            rope: text_mark.1,
        };
        let img_v = View {
            kv: if has_image && cfg_img > 1.0 { &img_c.kv } else { &gen_c.kv },
            len: img_mark.0,
            rope: img_mark.1,
        };
        st.context_tokens = cond.len;
        let g = Guidance {
            text_scale: cfg_text,
            img_scale: cfg_img,
            interval,
            renorm,
            renorm_min: o.cfg_renorm_min,
        };
        let t2 = Instant::now();
        let lat = self.denoise(&[cond, text_v, img_v], w, h, o, &g, &mut st, progress)?;
        st.denoise = t2.elapsed().as_secs_f64();
        let t3 = Instant::now();
        let img = self.decode_latents(&lat, w, h)?;
        st.decode = t3.elapsed().as_secs_f64();
        Ok((img, st))
    }

    /// Latent tokens `[(h/16) * (w/16), 64]` to an image.
    pub fn decode_latents(&self, lat: &[f32], w: usize, h: usize) -> Result<Image> {
        let f = 8 * self.latent_patch;
        let z = self.unpatchify(lat, h / f, w / f);
        let (px, ph, pw) = self.vae()?.decode(&z, h / 8, w / 8);
        dump("image.bin", &px);
        // `(x * 0.5 + 0.5).clamp(0, 1) * 255` cast to uint8 (truncation), as the reference.
        let rgb = px.iter().map(|&v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8).collect();
        Ok(Image {
            width: pw,
            height: ph,
            rgb,
        })
    }

    /// The flow loop (`generate_image`): `views` are the conditional, text-free and image-free
    /// contexts.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        views: &[View; 3],
        w: usize,
        h: usize,
        o: &ImageOptions,
        g: &Guidance,
        st: &mut ImageStats,
        progress: Progress,
    ) -> Result<Vec<f32>> {
        let gp = self.gen_parts()?;
        let llm = &self.parts.llm;
        let ids = self.latent_ids(w, h)?;
        let n = ids.len();
        let pd = gp.llm2vae.n_out;
        st.latent_tokens = n;
        let mut x = match read_override("noise_in.bin", n * pd)? {
            Some(v) => v,
            None => crate::sd::ops::Rng::new(o.seed).normals(n * pd),
        };
        let steps = o.steps.max(2);
        let s = o.timestep_shift;
        let ts: Vec<f32> = (0..steps)
            .map(|i| {
                let t = 1.0 - i as f32 / (steps - 1) as f32;
                s * t / (1.0 + (s - 1.0) * t)
            })
            .collect();
        let markers = llm.embed(&[self.special.vision_start, self.special.vision_end]);
        // The velocities of the given branches (indexes into `views`), one batched pass; branches
        // whose view equals an earlier one reuse its result.
        let mut velocities = |which: &[usize], e: &[f32]| -> Result<Vec<Vec<f32>>> {
            let mut uniq: Vec<usize> = Vec::new();
            let map: Vec<usize> = which
                .iter()
                .map(|&w| match uniq.iter().position(|&u| views[u].same(&views[w])) {
                    Some(p) => p,
                    None => {
                        uniq.push(w);
                        uniq.len() - 1
                    }
                })
                .collect();
            let ctxs = uniq.iter().map(|&u| Ctx::Prefix(views[u].kv, views[u].len)).collect();
            let blocks = uniq
                .iter()
                .map(|&u| Block {
                    und: markers.clone(),
                    gen_: e.to_vec(),
                    pos: views[u].rope,
                    text: false,
                })
                .collect();
            let outs = llm.forward_multi(ctxs, blocks)?;
            st.passes += uniq.len();
            let hg: Vec<f32> = outs.into_iter().flat_map(|(_, g)| g).collect();
            let v = project(&[&gp.llm2vae], llm.cfg.hidden_size, &hg, n * uniq.len()).swap_remove(0);
            Ok(map.iter().map(|&m| v[m * n * pd..(m + 1) * n * pd].to_vec()).collect())
        };
        for i in 0..steps - 1 {
            if self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            let t = ts[i];
            let on = t > g.interval[0] && t <= g.interval[1];
            let (ts_, is_) = if on { (g.text_scale, g.img_scale) } else { (1.0, 1.0) };
            let e = self.latent_embed(&x, &ids, t)?;
            let which: &[usize] = match (ts_ > 1.0, is_ > 1.0) {
                (false, _) => &[0],
                (true, false) => &[0, 1],
                (true, true) => &[0, 1, 2],
            };
            let mut vs = velocities(which, &e)?;
            let v = if ts_ > 1.0 {
                let v_i = (vs.len() == 3).then(|| vs.pop().unwrap());
                let v_t = vs.pop().unwrap();
                let v_c = vs.pop().unwrap();
                if i == 0 {
                    dump("v0_cond.bin", &v_c);
                    dump("v0_text.bin", &v_t);
                    if let Some(vi) = &v_i {
                        dump("v0_img.bin", vi);
                    }
                }
                g.combine(&v_c, &v_t, v_i.as_deref(), ts_, is_, pd)
            } else {
                vs.swap_remove(0)
            };
            if i == 0 {
                dump("v0.bin", &v);
            }
            let dt = ts[i] - ts[i + 1];
            for (a, &b) in x.iter_mut().zip(&v) {
                *a -= b * dt;
            }
            if !progress(i + 1, steps - 1) {
                bail!("cancelled");
            }
        }
        dump("latents_out.bin", &x);
        Ok(x)
    }
}

struct Guidance {
    text_scale: f32,
    img_scale: f32,
    interval: [f32; 2],
    renorm: Renorm,
    renorm_min: f32,
}

impl Guidance {
    /// `_forward_flow`'s guidance: `v_c` conditional, `v_t` without the last text, `v_i`
    /// without images (when image guidance is on), `pd` channels per token.
    fn combine(&self, v_c: &[f32], v_t: &[f32], v_i: Option<&[f32]>, ts: f32, is: f32, pd: usize) -> Vec<f32> {
        let lerp = |base: &[f32], to: &[f32], s: f32| -> Vec<f32> { base.iter().zip(to).map(|(&b, &t)| b + s * (t - b)).collect() };
        let norm = |x: &[f32]| x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        let clamp = |a: f64, b: f64| ((a / (b + 1e-8)).clamp(self.renorm_min as f64, 1.0)) as f32;
        let mut text = lerp(v_t, v_c, ts);
        if self.renorm == Renorm::TextChannel {
            for (r, c) in text.chunks_mut(pd).zip(v_c.chunks(pd)) {
                let s = clamp(norm(c), norm(r));
                r.iter_mut().for_each(|v| *v *= s);
            }
            return match v_i {
                Some(vi) => lerp(vi, &text, is),
                None => text,
            };
        }
        let mut out = match v_i {
            Some(vi) => lerp(vi, &text, is),
            None => {
                text.shrink_to_fit();
                text
            }
        };
        match self.renorm {
            Renorm::Global => {
                let s = clamp(norm(v_c), norm(&out));
                out.iter_mut().for_each(|v| *v *= s);
            }
            _ => {
                for (r, c) in out.chunks_mut(pd).zip(v_c.chunks(pd)) {
                    let s = clamp(norm(c), norm(r));
                    r.iter_mut().for_each(|v| *v *= s);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_matches_the_reference() {
        // ImageTransform(1024, 512, 16) / (980, 224, 14) on common sizes (values from the
        // reference's MaxLongEdgeMinShortEdgeResize).
        assert_eq!(resize_size(1024, 1024, 1024, 512, 16, MAX_PIXELS), (1024, 1024));
        assert_eq!(resize_size(1024, 1024, 980, 224, 14, MAX_PIXELS), (980, 980));
        assert_eq!(resize_size(800, 600, 1024, 512, 16, MAX_PIXELS), (800, 608));
        assert_eq!(resize_size(400, 300, 1024, 512, 16, MAX_PIXELS), (688, 512));
        assert_eq!(resize_size(2000, 500, 1024, 512, 16, MAX_PIXELS), (1024, 256));
    }

    #[test]
    fn renorm_caps_the_norm() {
        let g = Guidance {
            text_scale: 4.0,
            img_scale: 2.0,
            interval: [0.0, 1.0],
            renorm: Renorm::Global,
            renorm_min: 0.0,
        };
        let (c, t) = (vec![1.0f32, 0.0], vec![0.0f32, 0.0]);
        let v = g.combine(&c, &t, None, 4.0, 1.0, 2);
        // text guidance gives [4, 0], renormed to the conditional norm 1.
        assert!((v[0] - 1.0).abs() < 1e-6 && v[1].abs() < 1e-6);
    }
}
