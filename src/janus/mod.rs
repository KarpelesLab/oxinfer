//! DeepSeek's Janus family (`deepseek-ai/Janus-Pro-1B`, `-7B`, `deepseek-ai/JanusFlow-1.3B`): one
//! language model that both writes images and reads them. The variant is read from the config
//! ([`Variant`]).
//!
//! * **Text to image, Janus-Pro** ([`Janus::generate_image`]): the prompt in the SFT format plus
//!   `<begin_of_image>`, then 576 image tokens (a 24x24 grid) sampled from `gen_head` with
//!   classifier-free guidance. The conditional and unconditional sequences (the latter with the
//!   prompt replaced by padding) run as one batch, so each step reads the weights once; every
//!   sampled token is fed back through `gen_embed` + `gen_aligner`. The VQ decoder ([`vq`])
//!   turns the grid into a 384x384 RGB image.
//! * **Text to image, JanusFlow**: rectified flow in the SDXL VAE's latent space with the
//!   language model predicting the velocity ([`flow`]), then the VAE decoder.
//! * **Image understanding** ([`Janus::understand`]): a SigLIP-L/16 encoder and aligner
//!   ([`siglip`]) turn each image into 576 embeddings placed at `<image_placeholder>`, and the
//!   language model answers with its text head. [`Janus::ask`] continues a [`Conversation`]
//!   over several turns.
//!
//! The language model is a LLaMA decoder ([`llm`]); weights load from `pytorch_model.bin`
//! (sharded or not) or safetensors.

pub mod flow;
pub mod llm;
pub mod siglip;
pub mod vq;

use crate::clock::Instant;
use crate::generate::Flow;
use crate::model::Quant;
use crate::platform::DevicePref;
use crate::safetensors::ModelFiles;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::video::RgbImage;
use anyhow::{Context, Result, anyhow, bail};
use llm::{LlamaConfig, Llm};
use siglip::{Aligner, Dense, Mlp, VisionEncoder, gelu_inplace};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use vq::VqDecoder;

/// A generated image (tightly packed RGB8; `to_video_frame()` with the `oxideav` feature).
pub use crate::sd::Image;

/// The understanding system prompt (`VLChatProcessor.system_prompt`).
pub const SYSTEM_PROMPT: &str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.";

/// The image placeholder to put in a question where an image goes (one per image; without any,
/// the images are placed before the question).
pub const IMAGE_TAG: &str = "<image_placeholder>";

/// The text of the end-of-sentence token, which closes an assistant turn in the SFT template.
const EOS_TEXT: &str = "<｜end▁of▁sentence｜>";

/// The model families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Janus-Pro: images as 576 VQ tokens sampled one by one.
    Pro,
    /// JanusFlow: images by rectified flow in the SDXL VAE's latent space.
    Flow,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::Pro => "Janus-Pro",
            Variant::Flow => "JanusFlow",
        }
    }
}

/// True when `dir` holds a Janus checkpoint (`model_type` `multi_modality` with a VQ
/// generation tower, or JanusFlow's generation encoder).
pub fn is_janus(dir: &Path) -> bool {
    crate::fsx::read(&dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| {
            v["model_type"] == "multi_modality" && (v.get("gen_vision_config").is_some() || v.get("vision_gen_enc_config").is_some())
        })
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
    /// Load the image generation parts (Janus-Pro: `gen_head`, `gen_embed`, `gen_aligner`, VQ
    /// decoder; JanusFlow: the generation encoder and decoder, and the VAE).
    pub generation: bool,
    /// Load the understanding encoder (SigLIP + aligner).
    pub understanding: bool,
    /// JanusFlow: the VAE that decodes the latents, a diffusers `AutoencoderKL` folder (or a
    /// pipeline folder with a `vae/` in it) or repo id. None: the checkpoint's own `vae/` folder
    /// if it has one, else [`flow::DEFAULT_VAE`] (the SDXL VAE, as in the reference).
    pub vae: Option<String>,
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
            vae: None,
        }
    }
}

impl JanusOptions {
    /// Sets an option by name (the C API's `key=value` options): `quant` (bf16, q8, q4), `mmap`,
    /// `device` (auto, cpu, metal, cuda, vulkan), `cache_dir`, `generation`, `understanding`,
    /// `vae`.
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
            "vae" => self.vae = (!value.is_empty()).then(|| value.to_string()),
            _ => bail!("unknown option {key:?} (quant, mmap, device, cache_dir, generation, understanding, vae)"),
        }
        Ok(())
    }
}

/// Text-to-image options. The defaults are the reference's (`generation_inference.py`,
/// `demo/app_janusflow.py`) except `parallel_size` (16 and 5 there). Deserializes from JSON with
/// every key optional.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageOptions {
    /// Classifier-free guidance weight, `uncond + w * (cond - uncond)`. None (the default): the
    /// model's, 5 for Janus-Pro and 2 for JanusFlow.
    pub cfg_weight: Option<f32>,
    /// Janus-Pro: sampling temperature (default 1); 0 picks the most likely token (greedy).
    pub temperature: f32,
    /// Janus-Pro: keep only the `top_k` most likely tokens (0, the default: all).
    pub top_k: usize,
    /// Janus-Pro: nucleus sampling threshold (1, the default: off).
    pub top_p: f32,
    /// Random seed (Janus-Pro: of the token sampling; JanusFlow: of the starting noise).
    pub seed: u64,
    /// Images generated together from the prompt (one batch; default 1).
    pub parallel_size: usize,
    /// JanusFlow: Euler steps of the flow (default 30).
    pub steps: usize,
    /// Output size in pixels; 0 and 0 (the default) for the model's own 384x384. Both models
    /// generate 384x384 only (see [`Janus::native_size`]); another size is that image
    /// resampled with a Lanczos filter, as the reference's demos do for display.
    pub width: usize,
    pub height: usize,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            cfg_weight: None,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
            parallel_size: 1,
            steps: flow::DEFAULT_STEPS,
            width: 0,
            height: 0,
        }
    }
}

/// Options for [`Janus::understand`] and [`Janus::ask`]. Greedy by default, like the reference's `inference.py`.
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
    /// The system prompt; None for [`SYSTEM_PROMPT`], an empty string for none. It opens the
    /// conversation: later turns of a [`Conversation`] ignore it.
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
    /// Janus-Pro: image tokens sampled so far (of every image of the batch); JanusFlow: Euler
    /// steps done.
    pub step: usize,
    /// Janus-Pro: image tokens per image (576); JanusFlow: Euler steps.
    pub steps: usize,
}

/// Timings of one text-to-image call, in seconds.
#[derive(Clone, Debug, Default)]
pub struct ImageStats {
    pub prompt_tokens: usize,
    /// The prompt through the language model (both CFG sequences).
    pub prefill: f64,
    /// The image-token loop (JanusFlow: the flow's steps).
    pub tokens: f64,
    /// VQ (JanusFlow: VAE) decoding of every image.
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

/// The SFT template's role names and what ends a text-to-image prompt.
struct Format {
    user: &'static str,
    assistant: &'static str,
    gen_tag: String,
}

/// A conversation with [`Janus::ask`]: the language model's state after the turns so far.
#[derive(Default)]
pub struct Conversation {
    st: Option<llm::BatchState>,
    ids: Vec<u32>,
    /// The last answer: where its tokens start in `ids`, and its text. The next question reads
    /// it again as text, as the reference's template does.
    answer: Option<(usize, String)>,
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    /// The token ids the model has read and written so far (images as their placeholders).
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// True before the first question.
    pub fn is_empty(&self) -> bool {
        self.st.is_none()
    }
}

/// The models, borrowing the checkpoint owned by [`Janus`].
struct Parts<'a> {
    llm: Llm<'a>,
    gen_parts: Option<GenParts<'a>>,
    flow: Option<flow::FlowParts<'a>>,
    vision: Option<VisionEncoder<'a>>,
    /// JanusFlow: the embedding at `<image_beg>` in front of an image's embeddings.
    beg_of_und: Option<Vec<f32>>,
}

/// A loaded Janus-Pro or JanusFlow model.
pub struct Janus {
    // Declared before `files`, which it borrows: dropped first.
    parts: Parts<'static>,
    tok: Tokenizer,
    special: Special,
    format: Format,
    pub variant: Variant,
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
    /// Loads a Janus-Pro or JanusFlow checkpoint: a local directory or a Hugging Face repo id
    /// (`deepseek-ai/Janus-Pro-1B`, `deepseek-ai/JanusFlow-1.3B`), fetched into the cache on
    /// first use. JanusFlow also loads its VAE ([`JanusOptions::vae`]) for generation.
    pub fn load(path_or_repo: &str, opts: &JanusOptions) -> Result<Janus> {
        let t0 = Instant::now();
        let dir = crate::engine::resolve_model(path_or_repo, opts.cache_dir.as_deref())?;
        let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("config.json"))?)?;
        if cfg["model_type"] != "multi_modality" {
            bail!("{}: not a Janus checkpoint (model_type {})", dir.display(), cfg["model_type"]);
        }
        crate::gpu::ensure(opts.device)?;
        let lcfg = LlamaConfig::from_json(&cfg["language_config"])?;
        let h = lcfg.hidden_size;
        let files = Box::new(ModelFiles::open(&dir, !opts.mmap)?);
        // SAFETY: the box's heap block does not move, and `Janus` drops `parts` (every borrow of
        // it) before `_files`.
        let fs: &'static ModelFiles = unsafe { &*(files.as_ref() as *const ModelFiles) };
        let tok = crate::engine::load_tokenizer(&dir)?;
        let variant = if cfg.get("vision_gen_enc_config").is_some() {
            Variant::Flow
        } else {
            Variant::Pro
        };
        let is_flow = variant == Variant::Flow;
        // The processor's tags and template (`VLChatProcessor`), with its defaults.
        let pc: serde_json::Value = crate::fsx::read(&dir.join("processor_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let tag = |k: &str, d: &str| pc[k].as_str().unwrap_or(d).to_string();
        let image_start = tag("image_start_tag", if is_flow { "<image_beg>" } else { "<begin_of_image>" });
        let image_end = tag("image_end_tag", if is_flow { "<image_end>" } else { "<end_of_image>" });
        let (user, assistant) = match pc["sft_format"]
            .as_str()
            .unwrap_or(if is_flow { "deepseek_old" } else { "deepseek" })
        {
            "deepseek" => ("<|User|>", "<|Assistant|>"),
            "deepseek_old" => ("User", "Assistant"),
            f => bail!("processor_config: unknown sft_format {f:?}"),
        };
        let format = Format {
            user,
            assistant,
            gen_tag: if is_flow {
                tag("image_gen_tag", "<｜begin▁of▁generation｜>")
            } else {
                image_start.clone()
            },
        };
        let id = |t: &str| tok.token_to_id(t).with_context(|| format!("tokenizer: no {t}"));
        let special = Special {
            image_start: id(&image_start)?,
            image_end: id(&image_end)?,
            image: id(&tag("image_tag", IMAGE_TAG))?,
            pad: id("<｜▁pad▁｜>")?,
            eos: id(EOS_TEXT)?,
        };
        id(&format.gen_tag)?;
        let llm = Llm::load(lcfg, fs, "language_model.", opts.quant)?;
        let flow = if opts.generation && is_flow {
            let vae = flow::load_vae(&dir, opts)?;
            Some(flow::FlowParts::load(fs, &cfg, h, llm.cfg.rms_norm_eps, vae)?)
        } else {
            None
        };
        let gen_parts = if opts.generation && !is_flow {
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
            let vp = &cfg[if is_flow { "vision_und_enc_config" } else { "vision_config" }]["params"];
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
            let (prefix, aligner) = if is_flow {
                (
                    "vision_und_enc_model.vision_tower",
                    Aligner::Linear(Box::new(Dense::load(fs, "vision_und_enc_aligner", h, VisionEncoder::WIDTH)?)),
                )
            } else {
                let n_embed = cfg["aligner_config"]["params"]["n_embed"].as_u64().unwrap_or(h as u64) as usize;
                if n_embed != h {
                    bail!("aligner_config: n_embed {n_embed} is not the hidden size {h}");
                }
                (
                    "vision_model.vision_tower",
                    Aligner::Mlp(Box::new(Mlp::load(fs, "aligner", VisionEncoder::WIDTH, h)?)),
                )
            };
            Some(VisionEncoder::load(
                fs,
                prefix,
                aligner,
                size,
                layers,
                mean,
                tri("image_std", 0.5),
                background,
            )?)
        } else {
            None
        };
        let beg_of_und = if opts.understanding && is_flow {
            let e = fs.f32("beg_of_und_embed")?;
            if e.len() != h {
                bail!("beg_of_und_embed: {} values, expected {h}", e.len());
            }
            Some(e)
        } else {
            None
        };
        Ok(Janus {
            parts: Parts {
                llm,
                gen_parts,
                flow,
                vision,
                beg_of_und,
            },
            tok,
            special,
            format,
            variant,
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
        self.encode_with(text, true)
    }

    /// Token ids of `text`, with BOS in front or not.
    fn encode_with(&self, text: &str, bos: bool) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, bos).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    fn generator(&self) -> Result<&GenParts<'static>> {
        self.parts.gen_parts.as_ref().context(if self.variant == Variant::Flow {
            "JanusFlow has no image tokens (it generates by rectified flow)"
        } else {
            "image generation was not loaded (JanusOptions::generation)"
        })
    }

    /// Janus-Pro's text-to-image prompt: the "deepseek" SFT template with no system prompt, one
    /// user turn and an empty assistant turn, then `<begin_of_image>`.
    pub fn generation_prompt(prompt: &str) -> String {
        sft(&[("<|User|>", prompt), ("<|Assistant|>", "")], "") + "<begin_of_image>"
    }

    /// Janus-Pro's understanding prompt for a question about `images` images: placeholders are
    /// added before the question when it has none.
    pub fn understanding_prompt(question: &str, images: usize, system: &str) -> String {
        sft(&[("<|User|>", &with_placeholders(question, images)), ("<|Assistant|>", "")], system)
    }

    /// This model's text-to-image prompt (JanusFlow: the "deepseek_old" roles `User` and
    /// `Assistant`, ending with `<｜begin▁of▁generation｜>`).
    pub fn generation_text(&self, prompt: &str) -> String {
        let f = &self.format;
        sft(&[(f.user, prompt), (f.assistant, "")], "") + &f.gen_tag
    }

    /// This model's prompt for one question of a conversation: the first turn (with the system
    /// prompt) or a later one, which follows the previous answer and the end-of-sentence token.
    pub fn turn_text(&self, question: &str, images: usize, system: &str, first: bool) -> String {
        let f = &self.format;
        sft(
            &[(f.user, &with_placeholders(question, images)), (f.assistant, "")],
            if first { system } else { "" },
        )
    }

    /// The side in pixels of the images the model generates: 384. It is fixed by training for
    /// both models: Janus-Pro writes a 24x24 grid of image tokens and JanusFlow a 48x48 latent
    /// (24x24 tokens), in raster order with one-dimensional positions, so another grid is not a
    /// larger or wider image to them.
    pub fn native_size(&self) -> usize {
        match &self.parts.flow {
            Some(f) => f.image_size(),
            None => self.parts.gen_parts.as_ref().map_or(384, |g| g.grid * g.vq.scale()),
        }
    }

    /// Generates one image (the first of the batch when `parallel_size` > 1; see
    /// [`Janus::generate_images`]).
    pub fn generate_image(&self, prompt: &str, opts: &ImageOptions) -> Result<Image> {
        let (mut imgs, _) = self.generate_images(prompt, opts, &mut |_| Flow::Continue)?;
        Ok(imgs.swap_remove(0))
    }

    /// Generates `parallel_size` images for `prompt`, calling `progress` after every step (an
    /// image token of Janus-Pro, an Euler step of JanusFlow; return [`Flow::Stop`] to cancel:
    /// the call then fails with "cancelled").
    pub fn generate_images(
        &self,
        prompt: &str,
        opts: &ImageOptions,
        progress: &mut dyn FnMut(Progress) -> Flow,
    ) -> Result<(Vec<Image>, ImageStats)> {
        if (opts.width == 0) != (opts.height == 0) {
            bail!("image size: give both width and height, or neither for the model's own size");
        }
        let (mut imgs, mut stats, t0);
        if self.variant == Variant::Flow {
            let (latents, st) = self.flow_latents(prompt, opts, None, None, progress)?;
            stats = st;
            t0 = Instant::now();
            imgs = latents.iter().map(|z| self.decode_latents(z)).collect::<Result<Vec<_>>>()?;
        } else {
            let (grids, st) = self.generate_tokens(prompt, opts, progress)?;
            stats = st;
            t0 = Instant::now();
            imgs = grids.iter().map(|g| self.decode_tokens(g)).collect::<Result<Vec<_>>>()?;
        }
        if opts.width != 0 {
            for img in &mut imgs {
                if (img.width, img.height) != (opts.width, opts.height) {
                    img.rgb = crate::resample::resize_lanczos_rgb8(&img.rgb, img.width, img.height, opts.width, opts.height);
                    (img.width, img.height) = (opts.width, opts.height);
                }
            }
        }
        stats.decode = t0.elapsed().as_secs_f64();
        Ok((imgs, stats))
    }

    /// JanusFlow: decodes a latent of [`Janus::flow_latents`] to a 384x384 image.
    pub fn decode_latents(&self, latents: &[f32]) -> Result<Image> {
        let (rgb, width, height) = self.decode_latents_f32(latents)?;
        Ok(Image {
            width,
            height,
            rgb: vq::to_rgb8(&rgb),
        })
    }

    /// Janus-Pro: the image-token grids (`parallel_size` of them, 576 tokens each, row-major) for `prompt`.
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
            cfg_weight: Some(cfg_weight),
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
        let ids = self.encode(&self.generation_text(prompt))?;
        let cfg_weight = opts.cfg_weight.unwrap_or(5.0);
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
                let guided: Vec<f32> = c.iter().zip(u).map(|(&c, &u)| u + cfg_weight * (c - u)).collect();
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

    /// Text logits after the prompt of an understanding question (`[vocab]`), for checks.
    pub fn understanding_logits(&self, question: &str, images: &[RgbImage], system: Option<&str>) -> Result<Vec<f32>> {
        self.ask_logits(&mut Conversation::new(), question, images, system)
    }

    /// The text logits (`[vocab]`) after the prompt of the conversation's next question, which
    /// is added to the conversation without an answer. For checks.
    pub fn ask_logits(&self, conv: &mut Conversation, question: &str, images: &[RgbImage], system: Option<&str>) -> Result<Vec<f32>> {
        let hid = self.prefill_turn(conv, question, images, system.unwrap_or(SYSTEM_PROMPT))?;
        Ok(self.parts.llm.logits(&hid, 1))
    }

    /// Feeds the next question of `conv` (the first with the system prompt) and its images to
    /// the language model; returns the last position's hidden state. Each image is expanded to
    /// the start tag, 576 image positions and the end tag.
    fn prefill_turn(&self, conv: &mut Conversation, question: &str, images: &[RgbImage], system: &str) -> Result<Vec<f32>> {
        let llm = &self.parts.llm;
        let h = llm.cfg.hidden_size;
        let first = conv.is_empty();
        let mut text = self.turn_text(question, images.len(), system, first);
        // The reference formats the whole conversation again at every turn, so the model reads
        // its previous answer as text (` ANSWER` after `Assistant:`, then the end-of-sentence
        // token), which is not always the tokens it wrote: those are replaced.
        if let Some((start, answer)) = conv.answer.take() {
            if let Some(st) = conv.st.as_mut() {
                st.rewind(&[start], start);
            }
            conv.ids.truncate(start);
            if !answer.trim().is_empty() {
                text = format!(" {}{EOS_TEXT}{text}", answer.trim());
            }
        }
        let per = if images.is_empty() { 0 } else { self.vision()?.tokens() };
        let mut ids = Vec::new();
        let mut starts = Vec::new();
        for t in self.encode_with(&text, first)? {
            if t == self.special.image {
                ids.push(self.special.image_start);
                starts.push(ids.len());
                ids.extend(std::iter::repeat_n(self.special.image, per));
                ids.push(self.special.image_end);
            } else {
                ids.push(t);
            }
        }
        if starts.len() != images.len() {
            bail!("the question has {} image placeholders for {} images", starts.len(), images.len());
        }
        let mut x = llm.embed(&ids);
        for (img, &s) in images.iter().zip(&starts) {
            let (emb, _) = self.encode_image(img)?;
            x[s * h..s * h + emb.len()].copy_from_slice(&emb);
            if let Some(beg) = &self.parts.beg_of_und {
                x[(s - 1) * h..s * h].copy_from_slice(beg);
            }
        }
        let st = conv.st.get_or_insert_with(|| llm.new_state(1));
        let hid = llm.forward(st, x, ids.len());
        conv.ids.extend_from_slice(&ids);
        Ok(hid[hid.len() - h..].to_vec())
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
        self.ask(&mut Conversation::new(), question, images, opts, on_text)
    }

    /// The next turn of a conversation: answers `question`, which may come with more `images`
    /// (each at an `<image_placeholder>`, or before the question), knowing the earlier turns and
    /// their images. The turns follow the reference's SFT template (the previous answer closed
    /// by the end-of-sentence token), and the model's state is kept, so a turn costs its own
    /// tokens and the previous answer's, read again as text. A failed or cancelled call leaves
    /// the conversation unusable.
    pub fn ask(
        &self,
        conv: &mut Conversation,
        question: &str,
        images: &[RgbImage],
        opts: &ChatOptions,
        on_text: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<String> {
        self.cancel.store(false, Ordering::Relaxed);
        let llm = &self.parts.llm;
        let system = opts.system_prompt.as_deref().unwrap_or(SYSTEM_PROMPT);
        let hid = self.prefill_turn(conv, question, images, system)?;
        let start = conv.ids.len();
        let mut logits = llm.logits(&hid, 1);
        let mut sampler = Sampler::new(opts.temperature, opts.top_k, opts.top_p, opts.seed);
        let mut out: Vec<u32> = Vec::new();
        let mut text = String::new();
        for _ in 0..opts.max_tokens {
            let t = sampler.sample(&logits);
            if t == self.special.eos {
                break;
            }
            // The token joins the conversation before its text is shown, so stopping there
            // leaves the state in step with the answer.
            out.push(t);
            conv.ids.push(t);
            let st = conv.st.as_mut().context("conversation state")?;
            logits = llm.logits(&llm.forward(st, llm.embed(&[t]), 1), 1);
            let full = self.tok.decode(&out, true).map_err(|e| anyhow!("decode: {e}"))?;
            // Emit only complete characters (a partial UTF-8 sequence decodes to U+FFFD).
            if full.len() > text.len() && !full.ends_with('\u{FFFD}') && full.starts_with(&text) {
                let delta = full[text.len()..].to_string();
                text = full;
                if on_text(&delta) == Flow::Stop {
                    conv.answer = Some((start, text.clone()));
                    return Ok(text);
                }
            }
            if self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
        }
        let full = self.tok.decode(&out, true).map_err(|e| anyhow!("decode: {e}"))?;
        if full.len() > text.len() && full.starts_with(&text) {
            on_text(&full[text.len()..]);
        }
        conv.answer = Some((start, full.clone()));
        Ok(full)
    }
}

/// `question` with one placeholder per image in front when it has none.
fn with_placeholders(question: &str, images: usize) -> String {
    if images > 0 && !question.contains(IMAGE_TAG) {
        format!("{}{question}", format!("{IMAGE_TAG}\n").repeat(images))
    } else {
        question.to_string()
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
            let sep = if i % 2 == 0 { "\n\n" } else { EOS_TEXT };
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
