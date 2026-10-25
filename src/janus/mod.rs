//! DeepSeek Janus-Pro (`deepseek-ai/Janus-Pro-1B`, `-7B`): one autoregressive model that both
//! writes images and reads them.
//!
//! * **Text to image** ([`Janus::generate_image`]): the prompt in the SFT format plus
//!   `<begin_of_image>`, then 576 image tokens (a 24x24 grid) sampled from `gen_head` with
//!   classifier-free guidance. The conditional and unconditional sequences (the latter with the
//!   prompt replaced by padding) run as one batch, so each step reads the weights once; every
//!   sampled token is fed back through `gen_embed` + `gen_aligner`. The VQ decoder ([`vq`])
//!   turns the grid into a 384x384 RGB image.
//! * **Image understanding** ([`Janus::understand`]): a SigLIP-L/16 encoder and aligner
//!   ([`siglip`]) turn each image into 576 embeddings placed at `<image_placeholder>`, and the
//!   language model answers with its text head.
//!
//! The language model is a LLaMA decoder ([`llm`]); weights load from `pytorch_model.bin`
//! (sharded or not) or safetensors.

pub mod llm;
pub mod siglip;
pub mod vq;

use crate::clock::Instant;
use crate::generate::Flow;
use crate::model::Quant;
use crate::platform::DevicePref;
use crate::safetensors::ModelFiles;
use crate::sampler::Sampler;
use crate::video::RgbImage;
use anyhow::{Context, Result, anyhow, bail};
use llm::{LlamaConfig, Llm};
use siglip::{Dense, VisionEncoder, gelu_inplace};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokenizers::Tokenizer;
use vq::VqDecoder;

/// A generated image (tightly packed RGB8; `to_video_frame()` with the `oxideav` feature).
pub use crate::sd::Image;

/// The understanding system prompt (`VLChatProcessor.system_prompt`).
pub const SYSTEM_PROMPT: &str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.";

/// The image placeholder to put in a question where an image goes (one per image; without any,
/// the images are placed before the question).
pub const IMAGE_TAG: &str = "<image_placeholder>";

/// True when `dir` holds a Janus checkpoint (`model_type` `multi_modality` with a VQ
/// generation tower).
pub fn is_janus(dir: &Path) -> bool {
    crate::fsx::read(&dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["model_type"] == "multi_modality" && v.get("gen_vision_config").is_some())
}

/// Loading options.
#[derive(Clone, Debug)]
pub struct JanusOptions {
    /// Precision of the language model's linear weights and of `gen_head` (default bf16).
    pub quant: Quant,
    /// Map the weights from disk instead of reading them into memory.
    pub mmap: bool,
    /// Compute device (process-wide; set by the first model loaded).
    pub device: DevicePref,
    /// Where repo ids are downloaded (default [`crate::hf::cache_root`]).
    pub cache_dir: Option<PathBuf>,
    /// Load the image generation parts (`gen_head`, `gen_embed`, `gen_aligner`, VQ decoder).
    pub generation: bool,
    /// Load the understanding encoder (SigLIP + aligner).
    pub understanding: bool,
}

impl Default for JanusOptions {
    fn default() -> Self {
        Self {
            quant: Quant::Bf16,
            mmap: false,
            device: DevicePref::Auto,
            cache_dir: None,
            generation: true,
            understanding: true,
        }
    }
}

impl JanusOptions {
    /// Sets an option by name (the C API's `key=value` options): `quant` (bf16, q8, q4), `mmap`,
    /// `device` (auto, cpu, metal, cuda, vulkan), `cache_dir`, `generation`, `understanding`.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let flag = || match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(anyhow!("{key}: expected a boolean (1/0, true/false), got {value:?}")),
        };
        match key {
            "quant" => self.quant = Quant::parse(value).with_context(|| format!("quant: unknown quantization {value:?} (bf16, q8, q4)"))?,
            "mmap" => self.mmap = flag()?,
            "device" => {
                self.device =
                    DevicePref::parse(value).with_context(|| format!("device: unknown device {value:?} (auto|cpu|cuda|metal|vulkan)"))?
            }
            "cache_dir" => self.cache_dir = (!value.is_empty()).then(|| PathBuf::from(value)),
            "generation" => self.generation = flag()?,
            "understanding" => self.understanding = flag()?,
            _ => bail!("unknown option {key:?} (quant, mmap, device, cache_dir, generation, understanding)"),
        }
        Ok(())
    }
}

/// Text-to-image options. The defaults are the reference's (`generation_inference.py`) except
/// `parallel_size` (16 there). Deserializes from JSON with every key optional.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageOptions {
    /// Classifier-free guidance weight (default 5): `uncond + w * (cond - uncond)`.
    pub cfg_weight: f32,
    /// Sampling temperature (default 1); 0 picks the most likely token (greedy).
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens (0, the default: all).
    pub top_k: usize,
    /// Nucleus sampling threshold (1, the default: off).
    pub top_p: f32,
    /// Random seed.
    pub seed: u64,
    /// Images generated together from the prompt (one batch; default 1).
    pub parallel_size: usize,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            cfg_weight: 5.0,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
            parallel_size: 1,
        }
    }
}

/// Options for [`Janus::understand`]. Greedy by default, like the reference's `inference.py`.
/// Deserializes from JSON with every key optional.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatOptions {
    pub max_tokens: usize,
    /// 0 = greedy.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    /// The system prompt; None for [`SYSTEM_PROMPT`], an empty string for none.
    pub system_prompt: Option<String>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
            system_prompt: None,
        }
    }
}

/// Where image generation is, for the progress callback.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    /// Image tokens sampled so far (of every image of the batch).
    pub step: usize,
    /// Image tokens per image (576).
    pub steps: usize,
}

/// Timings of one text-to-image call, in seconds.
#[derive(Clone, Debug, Default)]
pub struct ImageStats {
    pub prompt_tokens: usize,
    /// The prompt through the language model (both CFG sequences).
    pub prefill: f64,
    /// The image-token loop.
    pub tokens: f64,
    /// VQ decoding of every image.
    pub decode: f64,
    pub steps: usize,
}

/// Stops a running generation from another thread.
#[derive(Clone, Debug)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// The generation parts: `gen_head` (hidden -> image-token logits), `gen_embed` (token ->
/// 8-dim code) and `gen_aligner` (code -> hidden).
struct GenParts<'a> {
    head0: Dense<'a>,
    head1: Dense<'a>,
    /// `[codes, code_dim]`.
    embed: Vec<f32>,
    code_dim: usize,
    /// The aligner's first layer (`code_dim` inputs), in f32.
    al0_w: Vec<f32>,
    al0_b: Vec<f32>,
    al1: Dense<'a>,
    vq: VqDecoder,
    tokens_per_image: usize,
    grid: usize,
}

impl GenParts<'_> {
    /// `gen_aligner(gen_embed(tokens))`: `[tokens.len(), hidden]`.
    fn embed_tokens(&self, toks: &[u32]) -> Vec<f32> {
        let (d, h) = (self.code_dim, self.al1.n_in);
        let mut x = vec![0f32; toks.len() * h];
        for (r, &t) in x.chunks_mut(h).zip(toks) {
            let e = &self.embed[t as usize * d..(t as usize + 1) * d];
            for (j, o) in r.iter_mut().enumerate() {
                let w = &self.al0_w[j * d..(j + 1) * d];
                *o = self.al0_b[j] + w.iter().zip(e).map(|(a, b)| a * b).sum::<f32>();
            }
        }
        gelu_inplace(&mut x);
        self.al1.forward(&x, toks.len())
    }

    /// Image-token logits `[rows, codes]` of final-normed hidden states.
    fn logits(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        let mut y = self.head0.forward(hidden, rows);
        gelu_inplace(&mut y);
        self.head1.forward(&y, rows)
    }
}

struct Special {
    image_start: u32,
    image_end: u32,
    image: u32,
    pad: u32,
    eos: u32,
}

/// The models, borrowing the checkpoint owned by [`Janus`].
struct Parts<'a> {
    llm: Llm<'a>,
    gen_parts: Option<GenParts<'a>>,
    vision: Option<VisionEncoder<'a>>,
}

/// A loaded Janus-Pro model.
pub struct Janus {
    // Declared before `files`, which it borrows: dropped first.
    parts: Parts<'static>,
    tok: Tokenizer,
    special: Special,
    cancel: Arc<AtomicBool>,
    pub path: PathBuf,
    pub load_secs: f64,
    _files: Box<ModelFiles>,
}

/// The checkpoint file holding tensor `name` (sharded safetensors or PyTorch files use their
/// index).
fn weight_file(dir: &Path, name: &str) -> Result<PathBuf> {
    for idx in ["model.safetensors.index.json", "pytorch_model.bin.index.json"] {
        let p = dir.join(idx);
        if crate::fsx::exists(&p) {
            let v: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&p)?)?;
            let f = v["weight_map"][name].as_str().with_context(|| format!("{idx}: no {name}"))?;
            return Ok(dir.join(f));
        }
    }
    for f in ["model.safetensors", "pytorch_model.bin"] {
        let p = dir.join(f);
        if crate::fsx::exists(&p) {
            return Ok(p);
        }
    }
    bail!("{}: no model.safetensors or pytorch_model.bin", dir.display())
}

impl Janus {
    /// Loads a Janus-Pro checkpoint: a local directory or a Hugging Face repo id
    /// (`deepseek-ai/Janus-Pro-1B`), fetched into the cache on first use.
    pub fn load(path_or_repo: &str, opts: &JanusOptions) -> Result<Janus> {
        let t0 = Instant::now();
        let dir = crate::engine::resolve_model(path_or_repo, opts.cache_dir.as_deref())?;
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        if cfg["model_type"] != "multi_modality" {
            bail!("{}: not a Janus checkpoint (model_type {})", dir.display(), cfg["model_type"]);
        }
        if !crate::gpu::initialized() {
            crate::gpu::init(&crate::platform::detect(), opts.device);
        }
        let lcfg = LlamaConfig::from_json(&cfg["language_config"])?;
        let h = lcfg.hidden_size;
        let files = Box::new(ModelFiles::open(&dir, !opts.mmap)?);
        // SAFETY: the box's heap block does not move, and `Janus` drops `parts` (every borrow of
        // it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let tok = crate::engine::load_tokenizer(&dir)?;
        let id = |t: &str| tok.token_to_id(t).with_context(|| format!("tokenizer: no {t}"));
        let special = Special {
            image_start: id("<begin_of_image>")?,
            image_end: id("<end_of_image>")?,
            image: id(IMAGE_TAG)?,
            pad: id("<｜▁pad▁｜>")?,
            eos: id("<｜end▁of▁sentence｜>")?,
        };
        let llm = Llm::load(lcfg, fs, "language_model.", opts.quant)?;
        let gen_parts = if opts.generation {
            let g = &cfg["gen_vision_config"]["params"];
            let codes = g["image_token_size"].as_u64().unwrap_or(16384) as usize;
            let code_dim = g["n_embed"].as_u64().unwrap_or(8) as usize;
            let hp = &cfg["gen_head_config"]["params"];
            let te = hp["image_token_embed"].as_u64().unwrap_or(h as u64) as usize;
            if cfg["gen_vision_config"]["cls"] != "VQ-16" {
                bail!(
                    "gen_vision_config: only VQ-16 is supported, got {}",
                    cfg["gen_vision_config"]["cls"]
                );
            }
            let embed = fs.f32("gen_embed.weight")?;
            if embed.len() != codes * code_dim {
                bail!("gen_embed.weight: {} values, expected {codes}x{code_dim}", embed.len());
            }
            let (al0_w, al0_b) = (fs.f32("gen_aligner.layers.0.weight")?, fs.f32("gen_aligner.layers.0.bias")?);
            if al0_w.len() != h * code_dim || al0_b.len() != h {
                bail!("gen_aligner.layers.0: expected {h}x{code_dim}");
            }
            let vq_file = weight_file(&dir, "gen_vision_model.decoder.conv_in.weight")?;
            let vq = VqDecoder::load(&crate::sd::ops::Tensors::open(&vq_file)?, "gen_vision_model")?;
            if vq.codes != codes || vq.dim != code_dim {
                bail!("VQ codebook {}x{} does not match gen_vision_config", vq.codes, vq.dim);
            }
            let grid = 384 / vq.scale();
            Some(GenParts {
                head0: Dense::load_quant(fs, "gen_head.output_mlp_projector", te, h, opts.quant)?,
                head1: Dense::load_quant(fs, "gen_head.vision_head", codes, te, opts.quant)?,
                embed,
                code_dim,
                al0_w,
                al0_b,
                al1: Dense::load(fs, "gen_aligner.layers.2", h, h)?,
                vq,
                tokens_per_image: grid * grid,
                grid,
            })
        } else {
            None
        };
        let vision = if opts.understanding {
            let vp = &cfg["vision_config"]["params"];
            let name = vp["model_name"].as_str().unwrap_or_default();
            if name != "siglip_large_patch16_384" {
                bail!("vision_config: only siglip_large_patch16_384 is supported, got {name}");
            }
            let size = vp["image_size"].as_u64().unwrap_or(384) as usize;
            let sel = vp["select_layer"].as_i64().unwrap_or(-1);
            let layers = if sel <= 0 {
                (24 + sel + 1).clamp(1, 24) as usize
            } else {
                (sel as usize).min(24)
            };
            let pre: serde_json::Value = crate::fsx::read(&dir.join("preprocessor_config.json"))
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or_default();
            let tri = |k: &str, d: f32| -> [f32; 3] {
                let a = pre[k].as_array();
                std::array::from_fn(|i| a.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).map_or(d, |v| v as f32))
            };
            let mean = tri("image_mean", 0.5);
            let bg = pre["background_color"].as_array();
            let background = std::array::from_fn(|i| {
                bg.and_then(|a| a.get(i))
                    .and_then(|v| v.as_u64())
                    .map_or((mean[i] * 255.0) as u8, |v| v as u8)
            });
            let n_embed = cfg["aligner_config"]["params"]["n_embed"].as_u64().unwrap_or(h as u64) as usize;
            if n_embed != h {
                bail!("aligner_config: n_embed {n_embed} is not the hidden size {h}");
            }
            Some(VisionEncoder::load(fs, size, layers, h, mean, tri("image_std", 0.5), background)?)
        } else {
            None
        };
        Ok(Janus {
            parts: Parts { llm, gen_parts, vision },
            tok,
            special,
            cancel: Arc::new(AtomicBool::new(false)),
            path: dir,
            load_secs: t0.elapsed().as_secs_f64(),
            _files: files,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.parts.llm.cfg
    }

    /// A handle that stops the running (or next) generation; it then fails with "cancelled".
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancel.clone())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, true).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    fn generator(&self) -> Result<&GenParts<'static>> {
        self.parts
            .gen_parts
            .as_ref()
            .context("image generation was not loaded (JanusOptions::generation)")
    }

    /// The text-to-image prompt: the "deepseek" SFT template with no system prompt, one user
    /// turn and an empty assistant turn, then `<begin_of_image>`.
    pub fn generation_prompt(prompt: &str) -> String {
        sft(&[("<|User|>", prompt), ("<|Assistant|>", "")], "") + "<begin_of_image>"
    }

    /// The understanding prompt for a question about `images` images: placeholders are added
    /// before the question when it has none.
    pub fn understanding_prompt(question: &str, images: usize, system: &str) -> String {
        let q = if images > 0 && !question.contains(IMAGE_TAG) {
            format!("{}{question}", format!("{IMAGE_TAG}\n").repeat(images))
        } else {
            question.to_string()
        };
        sft(&[("<|User|>", &q), ("<|Assistant|>", "")], system)
    }

    /// Generates one image (the first of the batch when `parallel_size` > 1; see
    /// [`Janus::generate_images`]).
    pub fn generate_image(&self, prompt: &str, opts: &ImageOptions) -> Result<Image> {
        let (mut imgs, _) = self.generate_images(prompt, opts, &mut |_| Flow::Continue)?;
        Ok(imgs.swap_remove(0))
    }

    /// Generates `parallel_size` images for `prompt`, calling `progress` after every image-token
    /// step (return [`Flow::Stop`] to cancel; the call then fails with "cancelled").
    pub fn generate_images(
        &self,
        prompt: &str,
        opts: &ImageOptions,
        progress: &mut dyn FnMut(Progress) -> Flow,
    ) -> Result<(Vec<Image>, ImageStats)> {
        let (grids, mut stats) = self.generate_tokens(prompt, opts, progress)?;
        let t0 = Instant::now();
        let imgs = grids.iter().map(|g| self.decode_tokens(g)).collect::<Result<Vec<_>>>()?;
        stats.decode = t0.elapsed().as_secs_f64();
        Ok((imgs, stats))
    }

    /// The image-token grids (`parallel_size` of them, 576 tokens each, row-major) for `prompt`.
    pub fn generate_tokens(
        &self,
        prompt: &str,
        opts: &ImageOptions,
        progress: &mut dyn FnMut(Progress) -> Flow,
    ) -> Result<(Vec<Vec<u32>>, ImageStats)> {
        self.run_image_tokens(prompt, opts, None, None, progress)
    }

    /// Teacher-forced guided logits: the CFG-combined image-token logits (before temperature)
    /// at each step of one image whose tokens are `tokens` (step i is predicted from the prompt
    /// and `tokens[..i]`). For checks against the reference.
    pub fn image_logits(&self, prompt: &str, cfg_weight: f32, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let opts = ImageOptions {
            cfg_weight,
            temperature: 0.0,
            ..Default::default()
        };
        let mut trace = Vec::new();
        self.run_image_tokens(prompt, &opts, Some(tokens), Some(&mut trace), &mut |_| Flow::Continue)?;
        Ok(trace)
    }

    fn run_image_tokens(
        &self,
        prompt: &str,
        opts: &ImageOptions,
        forced: Option<&[u32]>,
        mut trace: Option<&mut Vec<Vec<f32>>>,
        progress: &mut dyn FnMut(Progress) -> Flow,
    ) -> Result<(Vec<Vec<u32>>, ImageStats)> {
        self.cancel.store(false, Ordering::Relaxed);
        let g = self.generator()?;
        let llm = &self.parts.llm;
        let h = llm.cfg.hidden_size;
        let ids = self.encode(&Self::generation_prompt(prompt))?;
        let l = ids.len();
        if l < 2 {
            bail!("prompt too short");
        }
        let n = if forced.is_some() { 1 } else { opts.parallel_size.max(1) };
        let steps = forced.map_or(g.tokens_per_image, |f| f.len().min(g.tokens_per_image));
        // The unconditional sequence keeps BOS and <begin_of_image> and pads everything between.
        let mut uncond = ids.clone();
        uncond[1..l - 1].fill(self.special.pad);
        let (ce, ue) = (llm.embed(&ids), llm.embed(&uncond));
        let mut x = Vec::with_capacity(2 * n * l * h);
        for _ in 0..n {
            x.extend_from_slice(&ce);
            x.extend_from_slice(&ue);
        }
        let mut st = llm.new_state(2 * n);
        let t0 = Instant::now();
        let hid = llm.forward(&mut st, x, l);
        let mut last: Vec<f32> = (0..2 * n)
            .flat_map(|b| hid[(b * l + l - 1) * h..(b * l + l) * h].to_vec())
            .collect();
        let mut stats = ImageStats {
            prompt_tokens: l,
            prefill: t0.elapsed().as_secs_f64(),
            steps,
            ..Default::default()
        };
        let t1 = Instant::now();
        // Quantized weights: the int8 decode kernels up to 15 rows (several images' CFG pairs);
        // past the default 7 rows the matmul would dequantize into the prefill GEMM, ~3x slower
        // at these sizes. Process-wide, restored afterwards.
        struct Restore(usize);
        impl Drop for Restore {
            fn drop(&mut self) {
                crate::kernels::set_decode_max_tokens(self.0);
            }
        }
        let _restore = Restore(crate::kernels::decode_max_tokens());
        if (8..=15).contains(&(2 * n)) {
            crate::kernels::set_decode_max_tokens(2 * n);
        }
        let mut sampler = Sampler::new(opts.temperature, opts.top_k, opts.top_p, opts.seed);
        let mut grids: Vec<Vec<u32>> = vec![Vec::with_capacity(steps); n];
        let codes = g.vq.codes;
        for step in 0..steps {
            let logits = g.logits(&last, 2 * n);
            let mut next = Vec::with_capacity(n);
            for (i, grid) in grids.iter_mut().enumerate() {
                let (c, u) = (
                    &logits[2 * i * codes..(2 * i + 1) * codes],
                    &logits[(2 * i + 1) * codes..(2 * i + 2) * codes],
                );
                let guided: Vec<f32> = c.iter().zip(u).map(|(&c, &u)| u + opts.cfg_weight * (c - u)).collect();
                let tok = match forced {
                    Some(f) => f[step],
                    None => sample(&mut sampler, &guided),
                };
                if let Some(t) = trace.as_deref_mut() {
                    t.push(guided);
                }
                grid.push(tok);
                next.push(tok);
            }
            if progress(Progress { step: step + 1, steps }) == Flow::Stop || self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            if step + 1 == steps {
                break;
            }
            let e = g.embed_tokens(&next);
            let mut x = Vec::with_capacity(2 * n * h);
            for r in e.chunks(h) {
                x.extend_from_slice(r);
                x.extend_from_slice(r);
            }
            last = llm.forward(&mut st, x, 1);
        }
        stats.tokens = t1.elapsed().as_secs_f64();
        Ok((grids, stats))
    }

    /// Decodes a 24x24 grid of image tokens (row-major) to a 384x384 image.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<Image> {
        let (rgb, width, height) = self.decode_tokens_f32(tokens)?;
        Ok(Image {
            width,
            height,
            rgb: vq::to_rgb8(&rgb),
        })
    }

    /// [`Janus::decode_tokens`] before quantization to bytes: RGB in [-1, 1], `[h * w, 3]`.
    pub fn decode_tokens_f32(&self, tokens: &[u32]) -> Result<(Vec<f32>, usize, usize)> {
        let g = self.generator()?;
        g.vq.decode(tokens, g.grid, g.grid)
    }

    fn vision(&self) -> Result<&VisionEncoder<'static>> {
        self.parts
            .vision
            .as_ref()
            .context("image understanding was not loaded (JanusOptions::understanding)")
    }

    /// An image's language-model embeddings (`[576, hidden]`) and the SigLIP features before
    /// the aligner (`[576, 1024]`).
    pub fn encode_image(&self, img: &RgbImage) -> Result<(Vec<f32>, Vec<f32>)> {
        let v = self.vision()?;
        let px = v.preprocess(&img.data, img.width, img.height)?;
        let mut feats = Vec::new();
        let emb = v.encode(&px, Some(&mut feats));
        Ok((emb, feats))
    }

    /// The prompt token ids for an understanding question, with each image expanded to
    /// `<begin_of_image>`, 576 image tokens and `<end_of_image>`; and where the images go.
    fn understanding_ids(&self, question: &str, images: usize, system: &str) -> Result<(Vec<u32>, Vec<usize>)> {
        let ids = self.encode(&Self::understanding_prompt(question, images, system))?;
        let per = self.vision()?.tokens();
        let mut out = Vec::with_capacity(ids.len() + images * (per + 2));
        let mut starts = Vec::new();
        for &t in &ids {
            if t == self.special.image {
                out.push(self.special.image_start);
                starts.push(out.len());
                out.extend(std::iter::repeat_n(self.special.image, per));
                out.push(self.special.image_end);
            } else {
                out.push(t);
            }
        }
        if starts.len() != images {
            bail!("the question has {} image placeholders for {images} images", starts.len());
        }
        Ok((out, starts))
    }

    /// Text logits after the prompt of an understanding question (`[vocab]`), for checks.
    pub fn understanding_logits(&self, question: &str, images: &[RgbImage], system: Option<&str>) -> Result<Vec<f32>> {
        let hid = self.prefill_understanding(question, images, system.unwrap_or(SYSTEM_PROMPT))?.1;
        let h = self.parts.llm.cfg.hidden_size;
        Ok(self.parts.llm.logits(&hid[hid.len() - h..], 1))
    }

    fn prefill_understanding(&self, question: &str, images: &[RgbImage], system: &str) -> Result<(llm::BatchState, Vec<f32>, usize)> {
        let llm = &self.parts.llm;
        let h = llm.cfg.hidden_size;
        let (ids, starts) = self.understanding_ids(question, images.len(), system)?;
        let mut x = llm.embed(&ids);
        for (img, &s) in images.iter().zip(&starts) {
            let (emb, _) = self.encode_image(img)?;
            x[s * h..s * h + emb.len()].copy_from_slice(&emb);
        }
        let mut st = llm.new_state(1);
        let hid = llm.forward(&mut st, x, ids.len());
        Ok((st, hid, ids.len()))
    }

    /// Answers `question` about `images` (each placed at an `<image_placeholder>` of the
    /// question, or before it), streaming the text to `on_text`; returns the whole answer.
    pub fn understand(
        &self,
        question: &str,
        images: &[RgbImage],
        opts: &ChatOptions,
        on_text: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<String> {
        self.cancel.store(false, Ordering::Relaxed);
        let llm = &self.parts.llm;
        let h = llm.cfg.hidden_size;
        let system = opts.system_prompt.as_deref().unwrap_or(SYSTEM_PROMPT);
        let (mut st, hid, _) = self.prefill_understanding(question, images, system)?;
        let mut logits = llm.logits(&hid[hid.len() - h..], 1);
        let mut sampler = Sampler::new(opts.temperature, opts.top_k, opts.top_p, opts.seed);
        let mut out: Vec<u32> = Vec::new();
        let mut text = String::new();
        for _ in 0..opts.max_tokens {
            let t = sampler.sample(&logits);
            if t == self.special.eos {
                break;
            }
            out.push(t);
            let full = self.tok.decode(&out, true).map_err(|e| anyhow!("decode: {e}"))?;
            // Emit only complete characters (a partial UTF-8 sequence decodes to U+FFFD).
            if full.len() > text.len() && !full.ends_with('\u{FFFD}') && full.starts_with(&text) {
                let delta = full[text.len()..].to_string();
                text = full;
                if on_text(&delta) == Flow::Stop {
                    return Ok(text);
                }
            }
            if self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            let x = llm.embed(&[t]);
            logits = llm.logits(&llm.forward(&mut st, x, 1), 1);
        }
        let full = self.tok.decode(&out, true).map_err(|e| anyhow!("decode: {e}"))?;
        if full.len() > text.len() && full.starts_with(&text) {
            on_text(&full[text.len()..]);
        }
        Ok(full)
    }
}

/// Draws a token from `logits`: a single linear pass for plain temperature sampling (the 16384
/// image-token logits of every image at every step), `Sampler` when top-k / top-p cut them.
fn sample(s: &mut Sampler, logits: &[f32]) -> u32 {
    if s.greedy() || s.top_k != 0 || s.top_p < 1.0 {
        return s.sample(logits);
    }
    let inv_t = 1.0 / s.temperature;
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let p: Vec<f32> = logits.iter().map(|&v| ((v - max) * inv_t).exp()).collect();
    let mut r = s.uniform() * p.iter().sum::<f32>();
    for (i, &pi) in p.iter().enumerate() {
        r -= pi;
        if r <= 0.0 {
            return i as u32;
        }
    }
    p.iter().rposition(|&v| v > 0.0).unwrap_or(0) as u32
}

/// `Conversation.get_prompt()` of the "deepseek" template (sep "\n\n", sep2 the EOS text),
/// stripped as `apply_sft_template_for_multi_turn_prompts` does.
fn sft(turns: &[(&str, &str)], system: &str) -> String {
    let mut s = if system.is_empty() {
        String::new()
    } else {
        format!("{system}\n\n")
    };
    for (i, (role, msg)) in turns.iter().enumerate() {
        let msg = msg.trim();
        if msg.is_empty() {
            s.push_str(role);
            s.push(':');
        } else {
            let sep = if i % 2 == 0 { "\n\n" } else { "<｜end▁of▁sentence｜>" };
            s.push_str(&format!("{role}: {msg}{sep}"));
        }
    }
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_match_the_reference_template() {
        assert_eq!(
            Janus::generation_prompt(" a red fox "),
            "<|User|>: a red fox\n\n<|Assistant|>:<begin_of_image>"
        );
        assert_eq!(
            Janus::understanding_prompt("What is this?", 1, "SYS"),
            "SYS\n\n<|User|>: <image_placeholder>\nWhat is this?\n\n<|Assistant|>:"
        );
    }
}
